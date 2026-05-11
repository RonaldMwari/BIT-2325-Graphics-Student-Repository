# =============================================================================
# MILESTONE 5 — OPTIMIZED VERSION (No BVH, lower resolution)
# Runs 5-10x faster on any PC
# =============================================================================

import numpy as np
import pygame
import sys
import time

# ── REDUCED RESOLUTION FOR SPEED ──────────────────────────────────────────────
WIDTH, HEIGHT = 300, 200   # instead of 500, 380
FOV_DEG = 60.0
FPS = 15                       # was 30
ANIMATION_DURATION = 8.0

# =============================================================================
# MATH UTILITIES
# =============================================================================

def normalize_rows(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-10)

def normalize_vec(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

# =============================================================================
# DYNAMIC SCENE (same as before)
# =============================================================================

class DynamicScene:
    def __init__(self):
        self.centers_static = np.array([
            [ 0.0, -100.5, -2.0],
            [ 0.0,    4.0, -2.0],
        ], dtype=float)
        self.radii_static   = np.array([100.0, 1.5])
        self.albedos_static = np.array([
            [0.55, 0.55, 0.55],
            [1.00, 1.00, 0.85],
        ])
        self.emissive_static = np.array([False, True])
        self.light_idx = 1

        self.num_dynamic = 2
        self.radii_dynamic   = np.array([0.5, 0.5])
        self.albedos_dynamic = np.array([
            [0.85, 0.10, 0.10],
            [0.10, 0.10, 0.85],
        ])
        self.emissive_dynamic = np.array([False, False])
        self.centers_dynamic = np.zeros((self.num_dynamic, 3))
        self.update_positions(0.0)

    def update_positions(self, t):
        """
        t in [0,1] : normalized time through the animation loop.
        Red sphere: figure‑eight in XZ plane (flat infinity).
        Blue sphere: circle in XY plane (vertical loop), closer to camera.
        """
        angle = t * 2.0 * np.pi

        # --- Red sphere: figure‑8 (infinity symbol) in XZ ---
        # Formula: x = sin(2*angle), z = sin(angle) * 0.8
        red_x = 1.2 * np.sin(2.0 * angle)  # left-right swing
        red_z = -2.0 + 0.5 * np.sin(angle)  # small forward-back
        red_y = 0.0

        # --- Blue sphere: circle in XY (vertical hoop) ---
        # Center at (0, 0, -1.5), radius 0.8
        blue_x = 0.8 * np.cos(angle)
        blue_y = 0.6 * np.sin(angle)  # up-down motion
        blue_z = -1.5  # fixed Z, in front of red

        self.centers_dynamic[0] = [red_x, red_y, red_z]
        self.centers_dynamic[1] = [blue_x, blue_y, blue_z]

    def get_all_centers(self):
        return np.vstack([self.centers_static, self.centers_dynamic])

    def get_all_radii(self):
        return np.concatenate([self.radii_static, self.radii_dynamic])

    def get_all_albedos(self):
        return np.vstack([self.albedos_static, self.albedos_dynamic])

    def get_all_emissive(self):
        return np.concatenate([self.emissive_static, self.emissive_dynamic])

# =============================================================================
# NAIVE INTERSECTION (FASTER THAN BVH FOR <10 OBJECTS)
# =============================================================================

def intersect_naive(ray_o, ray_d, scene):
    N = ray_o.shape[0]
    t_best = np.full(N, np.inf)
    idx_best = np.full(N, -1, dtype=int)
    centers = scene.get_all_centers()
    radii = scene.get_all_radii()

    for s in range(len(radii)):
        oc = ray_o - centers[s]
        b = np.einsum('ij,ij->i', oc, ray_d)
        c = np.einsum('ij,ij->i', oc, oc) - radii[s]**2
        disc = b*b - c
        valid = disc >= 0.0
        sq = np.where(valid, np.sqrt(disc), 0.0)
        t1 = -b - sq
        t2 = -b + sq
        t_cand = np.where(valid & (t1 > 1e-4), t1,
                 np.where(valid & (t2 > 1e-4), t2, np.inf))
        better = t_cand < t_best
        t_best = np.where(better, t_cand, t_best)
        idx_best = np.where(better, s, idx_best)
    return t_best, idx_best

# =============================================================================
# SHADING (same as before, but uses naive intersection for shadows)
# =============================================================================

def shade(ray_o, ray_d, t, idx, scene):
    N_rays = ray_d.shape[0]
    img = np.zeros((N_rays, 3))
    hit_mask = idx >= 0
    if not np.any(hit_mask):
        return img

    centers = scene.get_all_centers()
    radii = scene.get_all_radii()
    albedos = scene.get_all_albedos()
    emissive = scene.get_all_emissive()
    light_idx = scene.light_idx

    safe_idx = np.where(hit_mask, idx, 0)
    hit_pos = ray_o + ray_d * t[:, np.newaxis]
    hit_center = centers[safe_idx]
    hit_radius = radii[safe_idx]
    normals = (hit_pos - hit_center) / hit_radius[:, np.newaxis]
    albedo = albedos[safe_idx]
    emit = emissive[safe_idx]

    # Emissive (light)
    emi_mask = hit_mask & emit
    img[emi_mask] = albedo[emi_mask] * 5.0

    # Diffuse objects
    diff_mask = hit_mask & ~emit
    if not np.any(diff_mask):
        return img

    p = hit_pos[diff_mask]
    n = normals[diff_mask]
    rd = ray_d[diff_mask]
    a = albedo[diff_mask]

    light_pos = centers[light_idx]
    lv = light_pos - p
    ldist = np.linalg.norm(lv, axis=1)
    ln = lv / ldist[:, np.newaxis]

    # Shadow ray using naive intersection
    sh_t, sh_idx = intersect_naive(p + n * 1e-3, ln, scene)
    in_shadow = np.zeros(p.shape[0], dtype=bool)
    sh_hit = sh_idx >= 0
    if np.any(sh_hit):
        closer = sh_t[sh_hit] < ldist[sh_hit]
        not_light = ~emissive[sh_idx[sh_hit]]
        in_shadow[sh_hit] = closer & not_light

    ndotl = np.maximum(np.einsum('ij,ij->i', n, ln), 0.0)
    atten = 1.0 / (1.0 + 0.015 * ldist**2)
    refl = 2.0 * ndotl[:, np.newaxis] * n - ln
    view = normalize_rows(-rd)
    rdotv = np.maximum(np.einsum('ij,ij->i', normalize_rows(refl), view), 0.0)
    spec = rdotv ** 48
    shadow_f = np.where(in_shadow[:, np.newaxis], 0.0, 1.0)

    img[diff_mask] = (a * 0.07 + shadow_f * (a * ndotl[:, np.newaxis] * atten[:, np.newaxis] + spec[:, np.newaxis] * atten[:, np.newaxis] * 0.45))
    return img

# =============================================================================
# CAMERA RAY BUILDER (same fixed version)
# =============================================================================

def get_camera_rays(scene, t_norm, width, height, fov_deg):
    # Midpoint of the two spheres
    midpoint = (scene.centers_dynamic[0] + scene.centers_dynamic[1]) / 2.0

    # Orbit camera around the midpoint
    orbit_angle = t_norm * 2.0 * np.pi * 0.15
    radius = 4.5
    camera_x = midpoint[0] + radius * np.sin(orbit_angle)
    camera_z = midpoint[2] + radius * np.cos(orbit_angle)
    camera_y = midpoint[1] + 1.2
    eye = np.array([camera_x, camera_y, camera_z])

    # Standard look-at: forward from eye to target
    forward = normalize_vec(midpoint - eye)

    # Use a fixed world up (Y-up coordinate system)
    world_up = np.array([0.0, 1.0, 0.0])

    # Avoid gimbal lock when looking straight down/up
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])

    # Right = forward × world_up  (cross product order matters!)
    right = normalize_vec(np.cross(forward, world_up))
    # Up = right × forward
    up = np.cross(right, forward)

    # For debugging (optional): print once to see if up is (0,1,0) or (0,-1,0)
    # print(f"up vector: {up}")

    # Perspective projection parameters
    fov_rad = np.radians(fov_deg)
    aspect = width / height
    half_h = np.tan(fov_rad / 2.0)
    half_w = half_h * aspect

    # Pixel grid (NDC: -1 to 1)
    xs = (np.arange(width) + 0.5) / width * 2 - 1
    ys = -((np.arange(height) + 0.5) / height * 2 - 1)
    xx, yy = np.meshgrid(xs, ys)

    # Ray direction = forward + x*right*half_w + y*up*half_h
    # No negative signs – this keeps the image upright
    dirs = (forward[np.newaxis, np.newaxis, :] +
            right[np.newaxis, np.newaxis, :] * (xx[..., np.newaxis] * half_w) +
            up[np.newaxis, np.newaxis, :] * (yy[..., np.newaxis] * half_h))

    dirs_flat = dirs.reshape(-1, 3)
    dirs_flat = normalize_rows(dirs_flat)

    origins_flat = np.tile(eye, (width * height, 1))
    return origins_flat, dirs_flat, eye, midpoint

# =============================================================================
# MOTION ANALYSIS (same)
# =============================================================================

def motion_analysis(positions, times):
    positions = np.array(positions)
    times = np.array(times)
    dt = np.gradient(times)
    vel = np.gradient(positions, axis=0) / dt[:, np.newaxis]
    acc = np.gradient(vel, axis=0) / dt[:, np.newaxis]
    speed = np.linalg.norm(vel, axis=1)
    jerk = np.gradient(acc, axis=0) / dt[:, np.newaxis]
    smoothness = 1.0 / (1.0 + np.mean(np.linalg.norm(jerk, axis=1)))
    return {
        'avg_speed': np.mean(speed),
        'max_speed': np.max(speed),
        'avg_acceleration': np.mean(np.linalg.norm(acc, axis=1)),
        'path_smoothness': smoothness
    }

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Milestone 5 — Optimized (Faster)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 10)

    scene = DynamicScene()
    start_time = time.time()
    positions_red, positions_blue, timestamps = [], [], []
    frame_count = 0

    print("Running optimized ray caster at", WIDTH, "x", HEIGHT)
    print("Target FPS:", FPS)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        elapsed = (time.time() - start_time) % ANIMATION_DURATION
        t_norm = elapsed / ANIMATION_DURATION
        scene.update_positions(t_norm)

        if frame_count % 10 == 0:
            positions_red.append(scene.centers_dynamic[0].copy())
            positions_blue.append(scene.centers_dynamic[1].copy())
            timestamps.append(elapsed)

        origins, directions, eye, midpoint = get_camera_rays(scene, t_norm, WIDTH, HEIGHT, FOV_DEG)
        t_hit, idx = intersect_naive(origins, directions, scene)
        img = shade(origins, directions, t_hit, idx, scene)
        img = np.sqrt(np.clip(img, 0, 1))
        frame = (img.reshape(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surface, (0, 0))

        # HUD
        info = font.render(f"t={elapsed:.1f}s  Red: {scene.centers_dynamic[0][0]:.2f}", True, (255,255,255))
        screen.blit(info, (4, 4))
        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1

    pygame.quit()

    # Print motion analysis
    print("\nMotion Analysis:")
    red_an = motion_analysis(positions_red, timestamps)
    print(f"Red sphere: avg speed {red_an['avg_speed']:.2f}, smoothness {red_an['path_smoothness']:.2f}")
    blue_an = motion_analysis(positions_blue, timestamps)
    print(f"Blue sphere: avg speed {blue_an['avg_speed']:.2f}, smoothness {blue_an['path_smoothness']:.2f}")

if __name__ == "__main__":
    main()