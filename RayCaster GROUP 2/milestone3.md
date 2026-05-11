# Milestone 3 — Rendering & Signal Processing
## Real-Time Global Illumination: Textures, Antialiasing & Artifact Analysis

---

## 1. System Overview

Milestone 3 extends the M2 interactive viewer with two core additions:

- **Procedural textures** applied to all surfaces (checkerboard ground,
  UV latitude-banded spheres)
- **Antialiasing comparison** between two rendering strategies —
  no AA (1 sample/pixel) vs jittered MSAA 4x (4 samples/pixel)

The milestone produces side-by-side rendered outputs for artifact
analysis and strategy comparison.

---

## 2. Texture & Surface Representation

### 2.1 Procedural Checkerboard (Ground Plane)

The ground plane uses a purely mathematical texture — no image file.

    Math:
        u = floor(Px * scale) mod 2
        v = floor(Pz * scale) mod 2

        colour = LIGHT  if u == v
                 DARK   otherwise

    Scale = 1.2 in our system
    LIGHT = (0.82, 0.82, 0.82)   grey tile
    DARK  = (0.12, 0.12, 0.18)   dark tile

This exploits integer parity to create an infinite repeating grid
with no texture memory or UV unwrapping required.

### 2.2 Spherical UV Mapping (Latitude Banding)

Spheres are textured using their surface normal vector as a UV
coordinate source. This is called spherical UV mapping:

    Longitude:  u = (atan2(Nz, Nx) + π) / 2π       range [0, 1]
    Latitude:   v = acos(clamp(-Ny, -1, 1)) / π     range [0, 1]

We use only the latitude (v) to create horizontal colour bands:

    band = floor(v * 8) mod 2

    band == 0 → base colour (red or blue)
    band == 1 → 60% base + 15% white (lighter variant)

This produces 8 alternating bands running around each sphere,
visible especially at the poles (top and bottom).

---

## 3. Signal Processing — Antialiasing

### 3.1 The Aliasing Problem

When one ray is fired through the exact centre of each pixel,
curved edges (like sphere silhouettes) show jagged staircase
artifacts called "jaggies." This is spatial aliasing — the
discrete pixel grid cannot represent continuous curves exactly.

The Nyquist theorem states that to reconstruct a signal correctly,
we must sample at least twice the highest frequency present.
A single sample per pixel undersamples curved edges.

### 3.2 Strategy 1 — No Antialiasing (1 sample/pixel)

    Ray direction: D = forward + right*(px*pw) + up*(py*ph)
    where (px, py) = exact pixel centre in [-1, 1]

    Samples per pixel: 1
    Render time: ~1.20s

Artifacts: visible staircase on sphere edges, hard shadow boundaries,
sharp texture transitions on checkerboard edges.

### 3.3 Strategy 2 — Jittered MSAA 4x (4 samples/pixel)

Multi-Sample Anti-Aliasing fires multiple rays per pixel at random
sub-pixel positions and averages the results:

    For sample i = 1..N:
        jx_i, jy_i ~ Uniform(0, 1)    random sub-pixel offset
        px_i = (pixel_x + jx_i) / width  * 2 - 1
        py_i = (pixel_y + jy_i) / height * 2 - 1
        colour_i = shade(ray_direction(px_i, py_i))

    final_colour = (1/N) * sum(colour_i)    N = 4

The averaging blurs the sharp aliasing boundary — nearby pixels
contribute slightly different values, creating a smooth gradient
instead of a hard step.

    Samples per pixel: 4
    Render time: ~4.49s   (3.8x slower — expected, linear cost)

---

## 4. Artifact Analysis

### 4.1 Measured Results

| Metric | Value |
|---|---|
| Mean pixel difference (MSAA vs No AA) | 0.83 |
| Max  pixel difference (MSAA vs No AA) | 153.00 |
| No AA render time | 1.20s |
| MSAA 4x render time | 4.49s |
| Time overhead of AA | 3.8x |

The max difference of 153/255 occurs at sphere silhouette edges —
exactly where aliasing is worst. The mean of 0.83 shows the overall
image is similar but edge regions differ significantly.

### 4.2 Artifact Types Observed

| Artifact | Cause | Present in No AA | Fixed by MSAA |
|---|---|---|---|
| Jagged sphere edges | Single sample per pixel | Yes | Partially |
| Hard shadow boundary | Binary shadow test | Yes | Partially |
| Texture aliasing | High-frequency checkerboard | Yes | Yes |
| Moire pattern | Undersampling repeating pattern | Yes | Yes |

### 4.3 Strategy Comparison

| Property | No AA | MSAA 4x |
|---|---|---|
| Samples per pixel | 1 | 4 |
| Render time | 1.20s | 4.49s |
| Edge quality | Aliased (jagged) | Smooth |
| Shadow edges | Hard | Slightly softer |
| Texture quality | Aliased | Anti-aliased |
| Memory cost | Minimal | 4x ray cost |

---

## 5. System Design

    Input  → Scene (spheres, ground, light) + rendering strategy
                 |
    Step 1 → For each pixel:
             Strategy 1: 1 ray at pixel centre
             Strategy 2: 4 rays at random sub-pixel positions
                 |
    Step 2 → Ray-sphere intersection (vectorised numpy)
                 |
    Step 3 → Shade with texture lookup:
             Ground → checkerboard(hit_pos)
             Spheres → sphere_uv(normal) → band index → colour
                 |
    Step 4 → Shadow ray + Phong shading
                 |
    Step 5 → Average samples (MSAA only)
                 |
    Step 6 → Gamma correct (sqrt) → save PNG
                 |
    Output → milestone3_no_aa.png, milestone3_msaa.png,
             milestone3_comparison.png

---

## 6. What Changed, Failed & Alternatives

| Issue | What Happened | Fix Applied |
|---|---|---|
| Checkerboard on "ceiling" | Camera angle too low, ground sphere curved overhead | Raised camera eye to (0, 0.6, 2.5) with look-at |
| UV seam on sphere | atan2 wraps at -π/π boundary | Acceptable artifact — documented |
| MSAA still shows some jaggies | 4 samples insufficient for perfect AA | 16+ samples would improve, not used for speed |
| Texture scale wrong | Too small produced noise-like pattern | Tuned scale to 1.2 empirically |

### Alternative Approaches Considered

- **Supersampling (SSAA)**: Render at 2x resolution then downsample —
  higher quality but 4x memory cost
- **Image-space AA (FXAA/SMAA)**: Post-process filter detects edges
  and blurs — very fast but less physically accurate
- **Stochastic sampling**: Use low-discrepancy sequences (Halton, Sobol)
  instead of random jitter — lower variance with same sample count