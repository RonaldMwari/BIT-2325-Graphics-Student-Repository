# =============================================================================
# MILESTONE 2 — Interactive Light Transport
# Slider-controlled camera orbit + moveable light source
# Uses matplotlib sliders (no pygame required)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
import time

WIDTH  = 420
HEIGHT = 320
FOV    = 55.0

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
    def __init__(self, light_pos=None):
        if light_pos is None:
            light_pos = [0.0, 4.0, -2.0]
        self.centers = np.array([
            [ 0.0, -100.5, -2.0],   # ground
            [-0.7,    0.0, -2.0],   # red sphere
            [ 0.7,    0.0, -2.0],   # blue sphere
            light_pos,              # moveable light
        ], dtype=float)
        self.radii    = np.array([100.0, 0.5, 0.5, 0.8])
        self.albedos  = np.array([
            [0.70, 0.70, 0.70],
            [0.85, 0.10, 0.10],
            [0.10, 0.10, 0.85],
            [1.00, 0.95, 0.70],
        ])
        self.emissive  = np.array([False, False, False, True])
        self.light_idx = 3

# =============================================================================
# INTERSECTION (vectorised)
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
# PHONG SHADING + SHADOWS
# =============================================================================

def shade(ray_o, ray_d, t, idx, scene):
    N_rays = ray_d.shape[0]
    img    = np.zeros((N_rays, 3))
    hit_mask = idx >= 0
    if not np.any(hit_mask): return img

    safe_idx   = np.where(hit_mask, idx, 0)
    hit_pos    = ray_o + ray_d * t[:, np.newaxis]
    hit_center = scene.centers[safe_idx]
    hit_radius = scene.radii[safe_idx]
    normals    = (hit_pos - hit_center) / hit_radius[:, np.newaxis]
    albedo     = scene.albedos[safe_idx]
    emissive   = scene.emissive[safe_idx]

    # Emissive (light sphere)
    emi_mask = hit_mask & emissive
    img[emi_mask] = albedo[emi_mask] * 6.0

    diff_mask = hit_mask & ~emissive
    if not np.any(diff_mask): return img

    p  = hit_pos[diff_mask]
    n  = normals[diff_mask]
    rd = ray_d[diff_mask]
    a  = albedo[diff_mask]

    lpos  = scene.centers[scene.light_idx]
    lv    = lpos - p
    ldist = np.linalg.norm(lv, axis=1)
    ln    = lv / ldist[:, np.newaxis]

    # Shadow rays
    sh_t, sh_idx = intersect_all(p + n * 1e-3, ln, scene)
    in_shadow    = np.zeros(p.shape[0], dtype=bool)
    sh_hit       = sh_idx >= 0
    if np.any(sh_hit):
        closer            = sh_t[sh_hit] < ldist[sh_hit]
        not_light         = ~scene.emissive[sh_idx[sh_hit]]
        in_shadow[sh_hit] = closer & not_light

    ndotl   = np.maximum(np.einsum('ij,ij->i', n, ln), 0.0)
    atten   = 1.0 / (1.0 + 0.012 * ldist**2)
    refl    = 2.0 * ndotl[:, np.newaxis] * n - ln
    view    = normalize_rows(-rd)
    rdotv   = np.maximum(np.einsum('ij,ij->i', normalize_rows(refl), view), 0.0)
    spec    = rdotv ** 48
    sf      = np.where(in_shadow[:, np.newaxis], 0.0, 1.0)

    img[diff_mask] = (a * 0.08
                    + sf * (a  * ndotl[:, np.newaxis] * atten[:, np.newaxis]
                          + spec[:, np.newaxis]        * atten[:, np.newaxis] * 0.5))
    return img

# =============================================================================
# RAY BUILDER
# =============================================================================

def build_rays(eye, target, width, height, fov_deg):
    fov = np.radians(fov_deg)
    asp = width / height
    ph  = np.tan(fov / 2.0)
    pw  = ph * asp
    forward, right, up = look_at_basis(eye, target)

    xs = (np.arange(width)  + 0.5) / width  * 2 - 1
    ys = (np.arange(height) + 0.5) / height * 2 - 1
    xx, yy = np.meshgrid(xs, ys)

    d = (forward[np.newaxis, np.newaxis, :]
       + right[np.newaxis, np.newaxis, :] * (xx[..., np.newaxis] * pw)
       + up[np.newaxis, np.newaxis, :]    * (yy[..., np.newaxis] * ph))

    d_flat = normalize_rows(d.reshape(-1, 3))
    o_flat = np.tile(eye, (width * height, 1))
    return o_flat, d_flat

# =============================================================================
# RENDER ONE FRAME
# =============================================================================

def render_frame(eye, target, light_pos, width, height):
    scene  = Scene(light_pos=light_pos)
    o, d   = build_rays(eye, target, width, height, FOV)
    t, idx = intersect_all(o, d, scene)
    img    = shade(o, d, t, idx, scene)
    img    = np.sqrt(np.clip(img, 0, 1))
    return img.reshape(height, width, 3)

# =============================================================================
# GET CAMERA EYE FROM SPHERICAL COORDS
# =============================================================================

def spherical_eye(phi_deg, theta_deg, radius,
                  target=np.array([0.0, 0.0, -2.0])):
    phi   = np.radians(phi_deg)
    theta = np.radians(np.clip(theta_deg, -80, 80))
    x = target[0] + radius * np.sin(phi)   * np.cos(theta)
    y = target[1] + radius * np.sin(theta)
    z = target[2] + radius * np.cos(phi)   * np.cos(theta)
    return np.array([x, y, z])

# =============================================================================
# MAIN — MATPLOTLIB SLIDER UI
# =============================================================================

def main():
    TARGET = np.array([0.0, 0.0, -2.0])

    # Initial values
    init = dict(
        phi    =  30.0,   # camera horizontal angle (deg)
        theta  =  15.0,   # camera vertical angle (deg)
        radius =   5.0,   # camera distance
        lx     =   0.0,   # light X
        ly     =   4.0,   # light Y
        lz     =  -2.0,   # light Z
    )

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8), facecolor='#12121e')
    fig.suptitle("Milestone 2 — Interactive Light Transport Viewer",
                 color='white', fontsize=13, y=0.98)

    # Main render area
    ax_img = fig.add_axes([0.01, 0.28, 0.65, 0.68])
    ax_img.set_facecolor('#000000')
    ax_img.axis('off')

    # Info panel (right side)
    ax_info = fig.add_axes([0.67, 0.28, 0.31, 0.68])
    ax_info.set_facecolor('#1a1a2e')
    ax_info.axis('off')

    # ── Slider axes (bottom strip) ────────────────────────────────────────────
    slider_color = '#2a2a4a'
    active_color = '#5588cc'

    def make_slider(rect, label, vmin, vmax, valinit, color=None):
        ax_s = fig.add_axes(rect, facecolor=slider_color)
        s    = Slider(ax_s, label, vmin, vmax, valinit=valinit,
                      color=active_color if color is None else color,
                      handle_style={'facecolor': 'white', 'size': 10})
        s.label.set_color('white')
        s.valtext.set_color('white')
        return s

    # Camera sliders (left column)
    s_phi    = make_slider([0.03, 0.20, 0.28, 0.03], 'Cam H°',  -180, 180, init['phi'])
    s_theta  = make_slider([0.03, 0.15, 0.28, 0.03], 'Cam V°',   -80,  80, init['theta'])
    s_radius = make_slider([0.03, 0.10, 0.28, 0.03], 'Zoom',      1.5, 12, init['radius'])

    # Light sliders (right column)
    s_lx = make_slider([0.38, 0.20, 0.28, 0.03], 'Light X', -6, 6, init['lx'],   '#cc5577')
    s_ly = make_slider([0.38, 0.15, 0.28, 0.03], 'Light Y', -1, 8, init['ly'],   '#55cc77')
    s_lz = make_slider([0.38, 0.10, 0.28, 0.03], 'Light Z', -8, 2, init['lz'],   '#cc9933')

    # Reset button
    ax_btn = fig.add_axes([0.03, 0.04, 0.10, 0.04])
    btn    = Button(ax_btn, 'Reset', color='#2a2a4a', hovercolor='#3a3a6a')
    btn.label.set_color('white')

    # Section labels
    fig.text(0.03, 0.245, '📷  CAMERA',   color='#88aaff', fontsize=9, weight='bold')
    fig.text(0.38, 0.245, '💡  LIGHT SOURCE', color='#ffaa55', fontsize=9, weight='bold')

    # ── State ─────────────────────────────────────────────────────────────────
    img_handle = [None]
    info_texts = [None]

    def update_info(eye, light_pos, render_ms):
        ax_info.cla()
        ax_info.set_facecolor('#1a1a2e')
        ax_info.axis('off')

        lines = [
            ("SCENE INFO", None, '#88aaff', 12),
            ("", None, 'white', 10),
            ("Camera eye:", None, '#aaaacc', 10),
            (f"  X: {eye[0]:+.2f}", None, 'white', 10),
            (f"  Y: {eye[1]:+.2f}", None, 'white', 10),
            (f"  Z: {eye[2]:+.2f}", None, 'white', 10),
            ("", None, 'white', 10),
            ("Light pos:", None, '#ffaa55', 10),
            (f"  X: {light_pos[0]:+.2f}", None, 'white', 10),
            (f"  Y: {light_pos[1]:+.2f}", None, 'white', 10),
            (f"  Z: {light_pos[2]:+.2f}", None, 'white', 10),
            ("", None, 'white', 10),
            ("Render time:", None, '#aaaacc', 10),
            (f"  {render_ms:.0f} ms", None, '#88ff88', 10),
            ("", None, 'white', 10),
            ("SHADING MODEL", None, '#88aaff', 11),
            ("Phong:", None, '#aaaacc', 10),
            ("  Ambient  + ", None, 'white', 10),
            ("  Diffuse  + ", None, 'white', 10),
            ("  Specular", None, 'white', 10),
            ("  + Shadows", None, 'white', 10),
        ]
        y = 0.97
        for text, _, col, sz in lines:
            ax_info.text(0.05, y, text, transform=ax_info.transAxes,
                         color=col, fontsize=sz, va='top',
                         fontfamily='monospace')
            y -= 0.048

    def redraw(val=None):
        eye       = spherical_eye(s_phi.val, s_theta.val, s_radius.val, TARGET)
        light_pos = [s_lx.val, s_ly.val, s_lz.val]

        t0    = time.time()
        frame = render_frame(eye, TARGET, light_pos, WIDTH, HEIGHT)
        ms    = (time.time() - t0) * 1000

        ax_img.cla()
        ax_img.imshow(frame, origin='upper', interpolation='bilinear')
        ax_img.set_title(
            f"Cam: ({eye[0]:.1f}, {eye[1]:.1f}, {eye[2]:.1f})   "
            f"Light: ({light_pos[0]:.1f}, {light_pos[1]:.1f}, {light_pos[2]:.1f})",
            color='#aaaadd', fontsize=8, pad=4)
        ax_img.axis('off')

        update_info(eye, light_pos, ms)
        fig.canvas.draw_idle()

    # Connect sliders
    for s in [s_phi, s_theta, s_radius, s_lx, s_ly, s_lz]:
        s.on_changed(redraw)

    def reset(event):
        s_phi.reset();  s_theta.reset(); s_radius.reset()
        s_lx.reset();   s_ly.reset();    s_lz.reset()

    btn.on_clicked(reset)

    # First render
    print("Rendering initial frame...")
    redraw()
    print("Done! Use the sliders to move the camera and light.")
    plt.show()

if __name__ == "__main__":
    main()