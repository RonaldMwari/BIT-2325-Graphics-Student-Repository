# Milestone 2 — Real-Time Global Illumination & Light Transport
## Interactive 3D Viewer: Mouse-Controlled Orbit Camera & Phong Shading

---

## 1. System Overview

Milestone 2 extends the static path tracer from M1 into a fully
**interactive real-time viewer**. The same scene (two spheres, ground
plane, overhead light) is now rendered with:

- A perspective camera that orbits the scene via mouse drag
- Phong shading model (ambient + diffuse + specular)
- Hard shadow rays from every hit point to the light
- Vectorised numpy ray casting for performance
- A live pygame window with HUD readout

The key shift from M1 to M2 is moving from offline Monte Carlo sampling
to a deterministic, interactive rendering pipeline.

---

## 2. 3D Transformation Pipeline

### 2.1 Homogeneous Coordinates

All 3D points are represented in homogeneous form as 4D vectors:

    P_h = (x, y, z, 1)^T

This allows translation, rotation, and scaling to all be expressed
as 4x4 matrix multiplications — a unified transform pipeline.

### 2.2 Camera Basis — Look-At Construction

The camera is defined by three orthonormal basis vectors built from
an eye position and a target point:

    forward = normalise(target - eye)

    right   = normalise(forward x world_up)
              where world_up = (0, 1, 0)

    up      = right x forward

These three vectors define the camera's local coordinate frame.
Every ray direction is expressed as a combination of these vectors.

Gimbal lock guard: if |forward . world_up| > 0.98 (looking nearly
straight up or down), world_up is switched to (0, 0, 1) to
prevent degenerate cross products.

### 2.3 Perspective Projection

Each pixel (px, py) in the [-1, 1] normalised screen space maps to
a ray direction through the perspective projection:

    fov_h = tan(FOV / 2)           half-height of projection plane
    fov_w = fov_h x aspect_ratio   half-width

    D = normalise(forward
                + right   x (px x fov_w)
                + up      x (py x fov_h))

This models a pinhole camera. Objects further away appear smaller
because all rays converge at the single eye point.

Field of view used: 60 degrees

### 2.4 Orbit Camera — Spherical Coordinates

The camera orbits the target using spherical coordinates:

    x = r . sin(phi) . cos(theta)
    y = r . sin(theta)
    z = r . cos(phi) . cos(theta)

    phi   = horizontal orbit angle (changed by mouse X drag)
    theta = vertical   orbit angle (changed by mouse Y drag)
    r     = orbit radius            (changed by scroll wheel)

Mouse sensitivity : 0.008 radians per pixel
Vertical clamp    : theta in [-1.3, 1.3] radians (prevents flipping)
Zoom range        : r in [1.5, 20.0] units

---

## 3. Rendering Pipeline

### 3.1 Ray Generation (Vectorised)

Rather than a Python loop per pixel, all W x H rays are generated
simultaneously as numpy arrays:

    o_flat : (WxH, 3)  — all ray origins (camera eye, repeated)
    d_flat : (WxH, 3)  — all ray directions (one per pixel)

This is the primary performance optimisation over M1.

### 3.2 Vectorised Ray-Sphere Intersection

The same intersection formula from M1 is applied to all rays at once:

    oc   = o_flat - C          (Nx3)
    b    = einsum(oc, d_flat)  (N,)   dot product per ray
    c    = einsum(oc, oc) - r2 (N,)
    h    = b2 - c              (N,)   discriminant

    t1 = -b - sqrt(h)   (near hit)
    t2 = -b + sqrt(h)   (far hit)

We iterate over each sphere and track the closest hit per ray
using numpy where() — no Python loop over pixels.

### 3.3 Phong Shading Model

At each hit point we compute illumination using three components:

    I = I_ambient + (1 - shadow) x (I_diffuse + I_specular)

Ambient (base fill light, prevents pure black):

    I_ambient = k_a x albedo       k_a = 0.07

Diffuse (Lambert's cosine law — how directly surface faces light):

    I_diffuse = albedo x max(N . L, 0) x attenuation

    N = surface normal (unit vector away from sphere)
    L = normalise(light_pos - hit_pos)

Specular (shiny highlight — Phong reflection model):

    R = 2(N . L)N - L             reflection vector
    I_specular = max(R . V, 0)^s x attenuation x 0.45

    V = normalise(-ray_direction)  view vector (toward camera)
    s = 48                         shininess exponent

Light attenuation (intensity falls off with distance):

    attenuation = 1 / (1 + 0.015 x dist2)

### 3.4 Shadow Rays

For every hit point on a diffuse surface, a shadow ray is fired
toward the light source:

    shadow_origin    = hit_pos + N x 1e-3   (offset avoids self-hit)
    shadow_direction = normalise(light_pos - hit_pos)

If any opaque object intersects the shadow ray at distance less
than the distance to the light, the point is in shadow:

    in_shadow = (t_shadow < dist_to_light) AND (hit object is not emissive)

Shadow points receive only ambient light.

### 3.5 Gamma Correction

Same as M1 — square root applied to linearise display output:

    output = sqrt(clamp(colour, 0, 1))

---

## 4. System Design

    Input  -> Mouse events (drag = orbit, scroll = zoom)
                 |
    Step 1 -> Compute camera eye from spherical coords (phi, theta, r)
                 |
    Step 2 -> Build look-at basis (forward, right, up)
                 |
    Step 3 -> Generate WxH ray directions (perspective projection)
                 |
    Step 4 -> Vectorised ray-sphere intersection (all rays at once)
                 |
    Step 5 -> Phong shade each hit point
                 |
    Step 6 -> Fire shadow ray per hit point -> modulate shading
                 |
    Step 7 -> Gamma correct -> convert to uint8 -> blit to pygame
                 |
    Output -> Live interactive window with HUD overlay

---

## 5. Scene & Camera Configuration

| Object       | Position        | Radius | Material             |
|--------------|-----------------|--------|----------------------|
| Ground plane | (0, -100.5, -2) | 100.0  | Grey Phong surface   |
| Red sphere   | (-0.7, 0, -2)   | 0.5    | Red Phong surface    |
| Blue sphere  | (0.7, 0, -2)    | 0.5    | Blue Phong surface   |
| Light source | (0, 4, -2)      | 1.5    | Emissive (warm white)|

| Camera Parameter       | Value          |
|------------------------|----------------|
| Field of view          | 60 degrees     |
| Initial orbit radius   | 5.0 units      |
| Initial horiz angle    | 0.30 radians   |
| Initial vert angle     | 0.25 radians   |
| Target point           | (0, 0, -2)     |
| Render resolution      | 500 x 380 px   |

---

## 6. Numerical Stability Analysis

| Potential Issue              | Risk   | Mitigation                              |
|------------------------------|--------|-----------------------------------------|
| Division by zero in normalise| High   | Guard: divide only if length > 1e-10    |
| Self-intersection (acne)     | High   | Shadow ray origin offset: +N x 1e-3    |
| Gimbal lock at poles         | Medium | Swap world_up to (0,0,1) when needed   |
| Negative discriminant sqrt   | Medium | np.maximum(disc, 0) before sqrt        |
| Colour overflow              | Low    | np.clip(colour, 0, 1) before gamma     |

---

## 7. What Changed, Failed & Alternatives

| Issue                        | What Happened                         | Fix Applied                          |
|------------------------------|---------------------------------------|--------------------------------------|
| Slow per-pixel Python loop   | Frame took minutes to render          | Replaced with vectorised numpy arrays|
| Shadow ray self-hits         | Sphere shadowed itself                | Added 1e-3 normal-offset to origin   |
| Camera flipping at poles     | Up vector became invalid              | Added gimbal lock guard in look_at() |
| Emissive sphere shaded wrong | Light sphere had shadow applied to it | Added emissive mask bypassing Phong  |
| Dark scene                   | Ambient coefficient too low           | Raised k_a to 0.07                   |

### Alternative Approaches Considered

- **OpenGL / GPU rasterisation**: Far faster but requires graphics API
  knowledge and does not naturally produce shadows or global illumination
- **Soft shadows**: Fire multiple shadow rays with random offsets around
  the light for more realistic penumbra — multiplies render time by N
- **Reflection rays**: Add mirror-like surfaces by recursing on the
  reflection vector R = 2(N.L)N - L — planned for Milestone 3