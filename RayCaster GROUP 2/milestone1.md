# Milestone 1 — Real-Time Global Illumination & Light Transport
## 2D Ray Traced Image: Two Spheres, Ground Plane & Overhead Light

---

## 1. System Overview

This milestone implements a **path tracer** — a program that simulates
how light physically travels through a scene to produce a photorealistic
2D image. The scene contains two coloured spheres (red and blue), a
large ground plane, and an overhead light source.

Rather than painting pixels manually, the system traces rays of light
backwards from the camera into the scene, computing how much light
each pixel receives through recursive bouncing.

---

## 2. Core Representation

### 2.1 Vector Mathematics

All positions, directions and colours are represented as 3D vectors:

    Vector3 = (x, y, z)   where x, y, z ∈ ℝ

Key vector operations used throughout the system:

    Dot product:    A · B = Ax·Bx + Ay·By + Az·Bz
                    → measures alignment between two directions

    Cross product:  A × B = (Ay·Bz - Az·By,  Az·Bx - Ax·Bz,  Ax·By - Ay·Bx)
                    → produces a vector perpendicular to both A and B

    Normalisation:  Â = A / |A|    where |A| = √(A · A)
                    → scales a vector to unit length (length = 1)

### 2.2 Ray Representation

A ray is a parametric line through 3D space:

    P(t) = O + t·D

    O = origin point (camera position)
    D = direction vector (normalised)
    t = scalar distance along the ray (t > 0 = forward)

Every pixel on screen corresponds to one unique ray fired from the camera.

---

## 3. Mathematical Foundation

### 3.1 Ray-Sphere Intersection

A sphere is defined as all points P at distance r from centre C:

    |P - C|² = r²

Substituting the ray equation P(t) = O + t·D:

    |O + t·D - C|² = r²

Let oc = O - C. Expanding:

    t²(D·D) + 2t(oc·D) + (oc·oc) - r² = 0

Since D is normalised, D·D = 1. Simplifying:

    t² + 2bt + c = 0

    where:
        b = oc · D
        c = oc · oc - r²

Solving with the quadratic formula, the discriminant is:

    h = b² - c

    If h < 0  → ray misses the sphere entirely
    If h ≥ 0  → two intersections:
                 t₁ = -b - √h   (entry point)
                 t₂ = -b + √h   (exit point)

We take the smallest positive t as the visible hit point.

### 3.2 Surface Normal

At a hit point P on a sphere with centre C and radius r:

    N = (P - C) / r

N is a unit vector pointing directly away from the sphere surface.
It is essential for all shading calculations.

### 3.3 Lambertian (Diffuse) Scattering

Matte surfaces scatter incoming light randomly across a hemisphere.
We sample a random direction using spherical coordinates:

    φ = 2π · r₁           (azimuth angle, r₁ ∈ [0,1))
    x = cos(φ) · √r₂
    y = sin(φ) · √r₂
    z = √(1 - r₂)         (r₂ ∈ [0,1))

This produces directions uniformly distributed on a unit hemisphere.

To align the hemisphere with the surface normal N, we construct a
local coordinate frame (u, v, w):

    w = N
    u = normalise((1,0,0) × w)    (or (0,1,0) if w ≈ (1,0,0))
    v = w × u

The scatter direction in world space is then:

    D_scatter = u·x + v·y + w·z

### 3.4 Recursive Light Transport (Path Tracing)

The rendering equation describes how light accumulates at a surface:

    L(P, D) = L_emit(P) + ∫ f_r · L_incoming · (N·D) dω

In practice we approximate this recursively (up to 4 bounces):

    trace(ray, depth):
        if depth > 4:     return black
        if hit emitter:   return emitter radiance
        if hit surface:   scatter ray
                          return albedo × trace(scattered_ray, depth+1)
        if no hit:        return sky colour

Each recursive call represents one additional light bounce.

### 3.5 Monte Carlo Sampling

A single ray per pixel would be extremely noisy.
We fire multiple samples per pixel and average them:

    pixel_colour = (1/N) · Σ trace(ray_i)    for i = 1..N

    N = 48 samples per pixel in our implementation

More samples → less noise, but longer render time.
This is the core principle of Monte Carlo integration.

### 3.6 Gamma Correction

Monitor displays are non-linear. Raw linear colour values
appear too dark. We apply a gamma of 2.0 (square root):

    output = √(linear_colour)

This remaps intensities so mid-tones display correctly.

---

## 4. System Design

```
Input  → Scene definition (spheres, materials, light)
            ↓
Step 1 → For each pixel: generate N sample rays from camera
            ↓
Step 2 → For each ray: find closest sphere intersection
            ↓
Step 3 → If emitter hit: return radiance directly
         If surface hit: scatter ray (Lambertian hemisphere sample)
                         recurse up to 4 bounces
            ↓
Step 4 → Average N samples per pixel
            ↓
Step 5 → Gamma correct (√) and clamp to [0, 1]
            ↓
Output → PNG image file (milestone1_render.png)
```

---

## 5. Scene Configuration

| Object       | Position        | Radius | Material          |
|--------------|-----------------|--------|-------------------|
| Ground plane | (0, -100.5, -2) | 100.0  | White Lambertian  |
| Red sphere   | (-0.7, 0, -2)   | 0.5    | Red Lambertian    |
| Blue sphere  | (0.7, 0, -2)    | 0.5    | Blue Lambertian   |
| Light source | (0, 4, -2)      | 1.5    | Emitter (15,15,15)|

Camera position : (0, 0.5, 0.5) looking toward (0, 0, -2)
Resolution      : 400 × 300 pixels @ 48 samples per pixel

---

## 6. What Changed, Failed & Alternatives

| Issue | What Happened | Fix Applied |
|---|---|---|
| Image too dark | Single sample per pixel was very noisy | Increased to 48 samples (Monte Carlo) |
| Self-intersection | Rays hitting the same surface they originated from | Added t > 1e-4 minimum distance threshold |
| Flat appearance | No bounced light, pure direct illumination only | Added recursive bouncing (max depth 4) |
| Overflow / NaN | Division by near-zero vector lengths | Added length > 1e-10 normalisation guard |

### Alternative Approaches Considered

- **Whitted ray tracing**: Deterministic reflections rather than random
  scattering — faster but less physically accurate
- **Bidirectional path tracing**: Trace from both camera AND light —
  better for caustics but far more complex to implement
- **Higher sample count**: 256+ samples would give near-noise-free
  results but render time increases linearly with N