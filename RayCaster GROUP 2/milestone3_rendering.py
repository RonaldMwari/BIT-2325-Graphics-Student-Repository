# =============================================================================
# MILESTONE 3 — Rendering & Signal Processing
# Textures + Antialiasing: Artifact analysis & strategy comparison
# Fixed camera, 800x600, labelled output
# =============================================================================

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

WIDTH   = 800
HEIGHT  = 600
FOV_DEG = 55.0

# Camera — positioned behind and slightly above, looking at scene centre
EYE    = np.array([0.0, 0.6, 2.5])
TARGET = np.array([0.0, 0.0, -2.0])

# =============================================================================
# MATH
# =============================================================================

def normalize_rows(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-10)

def normalize_vec(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

def look_at_basis(eye, target):
    world_up = np.array([0.0, 1.0, 0.0])
    forward  = target - eye
    forward  = forward / np.linalg.norm(forward)
    if abs(np.dot(forward, world_up)) > 0.98:
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up    = np.cross(right, forward)
    return forward, right, up

# =============================================================================
# SCENE
# =============================================================================

class Scene:
    def __init__(self):
        self.centers  = np.array([
            [ 0.0, -100.5, -2.0],
            [-0.7,    0.0, -2.0],
            [ 0.7,    0.0, -2.0],
            [ 0.0,    4.0, -2.0],
        ], dtype=float)
        self.radii    = np.array([100.0, 0.5, 0.5, 1.5])
        self.emissive = np.array([False, False, False, True])
        self.light_idx = 3

BASE_ALBEDOS = np.array([
    [0.70, 0.70, 0.70],
    [0.85, 0.10, 0.10],
    [0.10, 0.10, 0.85],
    [1.00, 1.00, 0.85],
])

# =============================================================================
# TEXTURES
# =============================================================================

def checkerboard_texture(hit_pos, scale=1.2):
    """
    Procedural checkerboard on the ground plane.
    Math: pattern = (floor(x*s) + floor(z*s)) mod 2
    """
    u = int(np.floor(hit_pos[0] * scale)) % 2
    v = int(np.floor(hit_pos[2] * scale)) % 2
    if u == v:
        return np.array([0.82, 0.82, 0.82])
    else:
        return np.array([0.12, 0.12, 0.18])

def sphere_texture(normal, base_colour):
    """
    UV latitude banding on spheres.
    Math: v = acos(-Ny) / π  — maps normal to [0,1] latitude
    """
    ny   = float(normal[1])
    v    = np.arccos(np.clip(-ny, -1.0, 1.0)) / np.pi
    band = int(v * 8) % 2
    if band == 0:
        return base_colour
    return np.clip(base_colour * 0.6 + np.array([0.15, 0.15, 0.15]), 0, 1)

# =============================================================================
# INTERSECTION
# =============================================================================

def intersect_all(ray_o, ray_d, scene):
    N        = ray_o.shape[0]
    t_best   = np.full(N, np.inf)
    idx_best = np.full(N, -1, dtype=int)
    for s in range(len(scene.radii)):
        oc   = ray_o - scene.centers[s]
        b    = np.einsum('ij,ij->i', oc, ray_d)
        c    = np.einsum('ij,ij->i', oc, oc) - scene.radii[s]**2
        disc = b*b - c
        valid  = disc >= 0.0
        sq     = np.where(valid, np.sqrt(np.maximum(disc, 0.0)), 0.0)
        t1     = -b - sq
        t2     = -b + sq
        t_cand = np.where(valid & (t1 > 1e-4), t1,
                 np.where(valid & (t2 > 1e-4), t2, np.inf))
        better   = t_cand < t_best
        t_best   = np.where(better, t_cand,   t_best)
        idx_best = np.where(better, s,         idx_best)
    return t_best, idx_best

# =============================================================================
# SHADING WITH TEXTURES
# =============================================================================

def shade_textured(ray_o, ray_d, t, idx, scene):
    N_rays = ray_d.shape[0]
    img    = np.zeros((N_rays, 3))

    hit_mask = idx >= 0
    if not np.any(hit_mask): return img

    safe_idx   = np.where(hit_mask, idx, 0)
    hit_pos    = ray_o + ray_d * t[:, np.newaxis]
    hit_center = scene.centers[safe_idx]
    hit_radius = scene.radii[safe_idx]
    normals    = (hit_pos - hit_center) / hit_radius[:, np.newaxis]
    emissive   = scene.emissive[safe_idx]

    emi_mask = hit_mask & emissive
    img[emi_mask] = BASE_ALBEDOS[safe_idx[emi_mask]] * 5.0

    diff_mask = hit_mask & ~emissive
    if not np.any(diff_mask): return img

    p   = hit_pos[diff_mask]
    n   = normals[diff_mask]
    rd  = ray_d[diff_mask]
    obj = safe_idx[diff_mask]

    albedo = np.zeros((p.shape[0], 3))
    for i in range(p.shape[0]):
        o = obj[i]
        if o == 0:
            albedo[i] = checkerboard_texture(p[i], scale=1.2)
        elif o == 1:
            albedo[i] = sphere_texture(n[i], BASE_ALBEDOS[1])
        elif o == 2:
            albedo[i] = sphere_texture(n[i], BASE_ALBEDOS[2])

    lpos  = scene.centers[scene.light_idx]
    lv    = lpos - p
    ldist = np.linalg.norm(lv, axis=1)
    ln    = lv / ldist[:, np.newaxis]

    sh_o  = p + n * 1e-3
    sh_t, sh_idx = intersect_all(sh_o, ln, scene)
    in_shadow = np.zeros(p.shape[0], dtype=bool)
    sh_hit    = sh_idx >= 0
    if np.any(sh_hit):
        closer            = sh_t[sh_hit] < ldist[sh_hit]
        not_light         = ~scene.emissive[sh_idx[sh_hit]]
        in_shadow[sh_hit] = closer & not_light

    ndotl   = np.maximum(np.einsum('ij,ij->i', n, ln), 0.0)
    atten   = 1.0 / (1.0 + 0.015 * ldist**2)
    refl    = 2.0 * ndotl[:, np.newaxis] * n - ln
    view    = normalize_rows(-rd)
    rdotv   = np.maximum(np.einsum('ij,ij->i', normalize_rows(refl), view), 0.0)
    spec    = rdotv ** 48
    shadow_f = np.where(in_shadow[:, np.newaxis], 0.0, 1.0)

    img[diff_mask] = (albedo * 0.08
                    + shadow_f * (albedo  * ndotl[:, np.newaxis]
                                           * atten[:, np.newaxis]
                                + spec[:, np.newaxis]
                                           * atten[:, np.newaxis] * 0.45))
    return img

# =============================================================================
# RAY BUILDER
# =============================================================================

def build_rays(eye, target, width, height, fov_deg,
               offset_x=0.5, offset_y=0.5):
    fov     = np.radians(fov_deg)
    asp     = width / height
    ph      = np.tan(fov / 2.0)
    pw      = ph * asp
    forward, right, up = look_at_basis(eye, target)

    xs = (np.arange(width)  + offset_x) / width  * 2 - 1
    ys = -((np.arange(height) + offset_y) / height * 2 - 1)
    xx, yy = np.meshgrid(xs, ys)

    d = (forward[np.newaxis, np.newaxis, :]
       + right[np.newaxis, np.newaxis, :] * (xx[..., np.newaxis] * pw)
       + up[np.newaxis, np.newaxis, :]    * (yy[..., np.newaxis] * ph))

    d_flat = normalize_rows(d.reshape(-1, 3))
    o_flat = np.tile(eye, (width * height, 1))
    return o_flat, d_flat

# =============================================================================
# STRATEGY 1 — NO ANTIALIASING
# =============================================================================

def render_no_aa(scene, width, height):
    o, d   = build_rays(EYE, TARGET, width, height, FOV_DEG)
    t, idx = intersect_all(o, d, scene)
    img    = shade_textured(o, d, t, idx, scene)
    img    = np.sqrt(np.clip(img, 0, 1))
    return (img.reshape(height, width, 3) * 255).astype(np.uint8)

# =============================================================================
# STRATEGY 2 — JITTERED MSAA 4x
# =============================================================================

def render_msaa(scene, width, height, samples=4):
    accum = np.zeros((height * width, 3))
    for _ in range(samples):
        jx = np.random.random()
        jy = np.random.random()
        o, d   = build_rays(EYE, TARGET, width, height, FOV_DEG, jx, jy)
        t, idx = intersect_all(o, d, scene)
        accum += shade_textured(o, d, t, idx, scene)
    img = accum / samples
    img = np.sqrt(np.clip(img, 0, 1))
    return (img.reshape(height, width, 3) * 255).astype(np.uint8)

# =============================================================================
# ADD LABEL TO IMAGE
# =============================================================================

def add_label(img_array, title, subtitle=""):
    """Add a dark header bar with title text to the top of an image."""
    img    = Image.fromarray(img_array)
    W, H   = img.size
    bar_h  = 52
    canvas = Image.new("RGB", (W, H + bar_h), (20, 20, 35))
    canvas.paste(img, (0, bar_h))
    draw   = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sub   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_sub   = font_title

    draw.text((12, 8),  title,    font=font_title, fill=(240, 240, 255))
    draw.text((12, 32), subtitle, font=font_sub,   fill=(160, 160, 200))
    return np.array(canvas)

# =============================================================================
# MAIN
# =============================================================================

def main():
    scene = Scene()

    print("=" * 58)
    print("  MILESTONE 3 — Rendering & Signal Processing")
    print("=" * 58)

    # Strategy 1: No AA
    print("\n[1/2] Rendering: NO Antialiasing (1 sample/pixel)...")
    t0        = time.time()
    img_no_aa = render_no_aa(scene, WIDTH, HEIGHT)
    t_no_aa   = time.time() - t0
    print(f"      Done in {t_no_aa:.2f}s")

    # Strategy 2: MSAA 4x
    print("\n[2/2] Rendering: Jittered MSAA 4x (4 samples/pixel)...")
    t0       = time.time()
    img_msaa = render_msaa(scene, WIDTH, HEIGHT, samples=4)
    t_msaa   = time.time() - t0
    print(f"      Done in {t_msaa:.2f}s")

    # Artifact analysis
    diff      = np.abs(img_no_aa.astype(float) - img_msaa.astype(float))
    mean_diff = diff.mean()
    max_diff  = diff.max()
    print(f"\n── Artifact Analysis ──────────────────────────────")
    print(f"   Mean pixel difference : {mean_diff:.2f}")
    print(f"   Max  pixel difference : {max_diff:.2f}")
    print(f"   No AA render time     : {t_no_aa:.2f}s")
    print(f"   MSAA 4x render time   : {t_msaa:.2f}s")
    print(f"   AA time cost          : {t_msaa/t_no_aa:.1f}x slower")

    # Label and save individual images
    no_aa_labelled = add_label(
        img_no_aa,
        "Milestone 3 — Strategy 1: No Antialiasing (1 sample/pixel)",
        f"Render time: {t_no_aa:.2f}s  |  Artifacts: jagged sphere edges visible"
    )
    msaa_labelled = add_label(
        img_msaa,
        "Milestone 3 — Strategy 2: Jittered MSAA 4x (4 samples/pixel)",
        f"Render time: {t_msaa:.2f}s  |  Mean pixel diff vs No AA: {mean_diff:.2f}"
    )

    # Side-by-side comparison (stack vertically for readability at high res)
    combined = np.vstack([no_aa_labelled, msaa_labelled])
    combined_img = add_label(
        combined,
        "Milestone 3 — Rendering Strategy Comparison",
        "Top: No AA (fast, aliased)   |   Bottom: MSAA 4x (slower, smooth)"
    )

    Image.fromarray(no_aa_labelled).save("milestone3_no_aa.png")
    Image.fromarray(msaa_labelled).save("milestone3_msaa.png")
    Image.fromarray(combined_img).save("milestone3_comparison.png")

    print("\n   Saved: milestone3_no_aa.png")
    print("          milestone3_msaa.png")
    print("          milestone3_comparison.png")
    Image.fromarray(combined_img).show()

if __name__ == "__main__":
    main()