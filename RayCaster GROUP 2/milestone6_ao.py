# =============================================================================
# MILESTONE 6 — Ambient Occlusion (Stable & Crash‑Free)
# =============================================================================

import numpy as np
import pygame
import sys
import time

# Suppress numpy warnings (cleaner output)
np.seterr(all='ignore')

# ── SETTINGS (lower for stability) ───────────────────────────────────────────
WIDTH, HEIGHT = 300, 200
FOV_DEG = 60.0
FPS = 15
ANIMATION_DURATION = 8.0

# Ambient Occlusion settings (start with low samples)
AO_ENABLED = True
AO_SAMPLES = 4          # was 8 → less load, no crash
AO_STRENGTH = 0.7
AO_MAX_DIST = 1.5

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
# DYNAMIC SCENE
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
        angle = t * 2.0 * np.pi
        # Red sphere: figure‑8 in XZ
        red_x = 1.2 * np.sin(2.0 * angle)
        red_z = -2.0 + 0.5 * np.sin(angle)
        red_y = 0.0
        # Blue sphere: circle in XY, fixed Z in front
        blue_x = 0.8 * np.cos(angle)
        blue_y = 0.6 * np.sin(angle)
        blue_z = -1.5
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
# STABLE INTERSECTION (no sqrt of negative numbers)
# =============================================================================

def intersect_naive(ray_o, ray_d, scene, max_dist=np.inf):
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
        # Safe sqrt: use max(disc, 0) to avoid tiny negatives
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = -b - sq
        t2 = -b + sq
        t_cand = np.where(valid & (t1 > 1e-4), t1,
                 np.where(valid & (t2 > 1e-4), t2, np.inf))
        better = t_cand < t_best
        t_best = np.where(better, t_cand, t_best)
        idx_best = np.where(better, s, idx_best)

    t_best = np.minimum(t_best, max_dist)
    return t_best, idx_best

# =============================================================================
# AMBIENT OCCLUSION (safe and fast enough)
# =============================================================================

def compute_ambient_occlusion(p, n, scene, num_samples=4, max_dist=1.5):
    """Return AO factor in [0,1]. Uses cosine-weighted hemisphere samples."""
    occlusion = 0.0
    for _ in range(num_samples):
        # Cosine-weighted hemisphere direction
        r1 = np.random.random()
        r2 = np.random.random()
        phi = 2.0 * np.pi * r1
        theta = np.arccos(np.sqrt(r2))
        local_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        # Build orthonormal basis from normal n
        if abs(n[0]) < 0.9:
            tangent = np.cross(n, [1, 0, 0])
        else:
            tangent = np.cross(n, [0, 1, 0])
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
        bitangent = np.cross(n, tangent)
        world_dir = (local_dir[0] * tangent +
                     local_dir[1] * bitangent +
                     local_dir[2] * n)
        world_dir = world_dir / (np.linalg.norm(world_dir) + 1e-8)

        # Fire occlusion ray
        oc_origin = p + n * 1e-3
        t_occ, idx_occ = intersect_naive(
            np.array([oc_origin]),
            np.array([world_dir]),
            scene,
            max_dist=max_dist
        )
        if idx_occ[0] >= 0 and t_occ[0] < max_dist:
            occlusion += 1.0

    ao_factor = 1.0 - (occlusion / num_samples) * AO_STRENGTH
    return ao_factor

# =============================================================================
# SHADING WITH AMBIENT OCCLUSION
# =============================================================================

def shade_with_ao(ray_o, ray_d, t, idx, scene):
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

    # Ambient Occlusion factor per hit point
    ao_factors = np.ones(p.shape[0])
    if AO_ENABLED:
        for i in range(p.shape[0]):
            ao_factors[i] = compute_ambient_occlusion(p[i], n[i], scene, AO_SAMPLES, AO_MAX_DIST)

    # Direct lighting
    light_pos = centers[light_idx]
    lv = light_pos - p
    ldist = np.linalg.norm(lv, axis=1)
    ln = lv / (ldist[:, np.newaxis] + 1e-10)

    # Shadow ray
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

    ambient = a * 0.07 * ao_factors[:, np.newaxis]
    diffuse = a * ndotl[:, np.newaxis] * atten[:, np.newaxis]
    specular = spec[:, np.newaxis] * atten[:, np.newaxis] * 0.45
    img[diff_mask] = ambient + shadow_f * (diffuse + specular)
    return img

# =============================================================================
# CAMERA RAY BUILDER (same as before)
# =============================================================================

def get_camera_rays(scene, t_norm, width, height, fov_deg):
    midpoint = (scene.centers_dynamic[0] + scene.centers_dynamic[1]) / 2.0
    orbit_angle = t_norm * 2.0 * np.pi * 0.15
    radius = 4.5
    camera_x = midpoint[0] + radius * np.sin(orbit_angle)
    camera_z = midpoint[2] + radius * np.cos(orbit_angle)
    camera_y = midpoint[1] + 1.2
    eye = np.array([camera_x, camera_y, camera_z])
    forward = normalize_vec(midpoint - eye)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])
    right = normalize_vec(np.cross(forward, world_up))
    up = np.cross(right, forward)
    fov_rad = np.radians(fov_deg)
    aspect = width / height
    half_h = np.tan(fov_rad / 2.0)
    half_w = half_h * aspect
    xs = (np.arange(width) + 0.5) / width * 2 - 1
    ys = (np.arange(height) + 0.5) / height * 2 - 1
    xx, yy = np.meshgrid(xs, ys)
    dirs = (forward[np.newaxis, np.newaxis, :] +
            right[np.newaxis, np.newaxis, :] * (xx[..., np.newaxis] * half_w) +
            up[np.newaxis, np.newaxis, :]    * (yy[..., np.newaxis] * half_h))
    dirs_flat = dirs.reshape(-1, 3)
    dirs_flat = normalize_rows(dirs_flat)
    # Remove any NaN/Inf
    dirs_flat = np.nan_to_num(dirs_flat)
    origins_flat = np.tile(eye, (width * height, 1))
    return origins_flat, dirs_flat, eye, midpoint

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    global AO_ENABLED   # must be first line in function

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Milestone 6 — AO {'ON' if AO_ENABLED else 'OFF'}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 10)
    scene = DynamicScene()
    start_time = time.time()
    running = True

    print("=" * 55)
    print("  MILESTONE 6 — Ambient Occlusion (Stable Version)")
    print(f"  AO Enabled = {AO_ENABLED}, Samples = {AO_SAMPLES}")
    print("  Press 'A' to toggle AO on/off")
    print("=" * 55)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                AO_ENABLED = not AO_ENABLED
                pygame.display.set_caption(f"Milestone 6 — AO {'ON' if AO_ENABLED else 'OFF'}")
                print(f"Ambient Occlusion toggled: {'ON' if AO_ENABLED else 'OFF'}")

        elapsed = (time.time() - start_time) % ANIMATION_DURATION
        t_norm = elapsed / ANIMATION_DURATION
        scene.update_positions(t_norm)

        origins, directions, eye, midpoint = get_camera_rays(scene, t_norm, WIDTH, HEIGHT, FOV_DEG)
        t_hit, idx = intersect_naive(origins, directions, scene)
        img = shade_with_ao(origins, directions, t_hit, idx, scene)
        img = np.sqrt(np.clip(img, 0, 1))
        frame = (img.reshape(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surface, (0, 0))

        info = font.render(f"AO={'ON' if AO_ENABLED else 'OFF'}  t={elapsed:.1f}s  Samples={AO_SAMPLES}", True, (255,255,255))
        screen.blit(info, (4, 4))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()