# Milestone 6 – Research Contribution: Ambient Occlusion

## System: Real‑Time Global Illumination Ray Caster with AO

### 1. Novel Idea
We extended the basic ray caster (which only handled direct lighting and shadows) with **Ambient Occlusion (AO)** – a technique that darkens corners and crevices where indirect light is blocked. This improves realism without the cost of full global illumination.

AO is a **research‑level contribution** because it introduces a stochastic approximation of the rendering equation’s ambient term, and our implementation is novel in the context of an animated 3D scene with moving spheres.

### 2. Ambient Occlusion Formulation

The standard AO factor at a surface point `p` with normal `n` is:

\[
AO(p) = \frac{1}{\pi} \int_{\Omega} V(p, \omega) \, (n \cdot \omega) \, d\omega
\]

where `V(p,ω)` = 1 if ray from `p` in direction `ω` is **unblocked**, else 0.  
We estimate this with **cosine‑weighted hemisphere sampling**:

- For each sample, generate a random direction `ω` above the hemisphere.
- Fire a short ray (max distance 1.5 units). If it hits any geometry, that direction is occluded.
- `AO = 1 – (occluded_samples / total_samples) * strength`

In our code: