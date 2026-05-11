# =============================================================================
# MILESTONE 1 — Real-Time Global Illumination & Light Transport
# 2D Ray Traced Image: Two spheres, ground plane, overhead light
# =============================================================================

import numpy as np
from dataclasses import dataclass
from PIL import Image

# ── CORE MATH ─────────────────────────────────────────────────────────────────

class Vector3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = np.float64(x), np.float64(y), np.float64(z)

    def __add__(self, o): return Vector3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vector3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, s): return Vector3(self.x*s,   self.y*s,   self.z*s)

    def dot(self, o):    return np.float64(self.x*o.x + self.y*o.y + self.z*o.z)
    def cross(self, o):  return Vector3(self.y*o.z - self.z*o.y,
                                        self.z*o.x - self.x*o.z,
                                        self.x*o.y - self.y*o.x)
    def length(self):    return np.sqrt(self.dot(self))
    def normalize(self):
        l = self.length()
        return self * (1.0/l) if l > 1e-10 else Vector3(0, 0, 1)

@dataclass
class Ray:
    origin:    Vector3
    direction: Vector3
    def at(self, t): return self.origin + self.direction * t

@dataclass
class HitRecord:
    position: Vector3
    normal:   Vector3
    t:        float
    material: any

# ── MATERIALS & OBJECTS ───────────────────────────────────────────────────────

class Lambertian:
    """Diffuse (matte) material — scatters light randomly on hemisphere."""
    def __init__(self, albedo): self.albedo = albedo

    def scatter(self, hit):
        r1, r2 = np.random.random(), np.random.random()
        phi  = 2 * np.pi * r1
        x    = np.cos(phi) * np.sqrt(r2)
        y    = np.sin(phi) * np.sqrt(r2)
        z    = np.sqrt(1 - r2)
        w    = hit.normal
        u    = (Vector3(1,0,0) if abs(w.x)<0.9 else Vector3(0,1,0)).cross(w).normalize()
        v    = w.cross(u)
        direction = u*x + v*y + w*z
        return Ray(hit.position, direction), self.albedo

class Emitter:
    """Light source material — emits radiance directly."""
    def __init__(self, radiance): self.radiance = radiance

class Sphere:
    def __init__(self, center, radius, material):
        self.center, self.radius, self.material = center, radius, material

    def hit(self, ray):
        oc = ray.origin - self.center
        b  = oc.dot(ray.direction)
        c  = oc.dot(oc) - self.radius**2
        h  = b*b - c
        if h < 0: return None
        t = -b - np.sqrt(h)
        if not (1e-4 < t < 1e10):
            t = -b + np.sqrt(h)
            if not (1e-4 < t < 1e10): return None
        pos = ray.at(t)
        return HitRecord(pos, (pos - self.center)*(1.0/self.radius), t, self.material)

# ── PATH TRACER ───────────────────────────────────────────────────────────────

def trace(scene, ray, depth=0):
    """Recursively trace a ray through the scene (max 4 bounces)."""
    if depth > 4: return Vector3(0, 0, 0)
    hit, closest = None, 1e10
    for obj in scene:
        h = obj.hit(ray)
        if h and h.t < closest:
            closest, hit = h.t, h
    if not hit:            return Vector3(0.02, 0.02, 0.05)  # sky
    if isinstance(hit.material, Emitter): return hit.material.radiance
    scattered, attenuation = hit.material.scatter(hit)
    incoming = trace(scene, scattered, depth+1)
    return Vector3(attenuation.x*incoming.x,
                   attenuation.y*incoming.y,
                   attenuation.z*incoming.z)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    red   = Lambertian(Vector3(0.8, 0.1, 0.1))
    blue  = Lambertian(Vector3(0.1, 0.1, 0.8))
    white = Lambertian(Vector3(0.7, 0.7, 0.7))
    light = Emitter(Vector3(15, 15, 15))

    scene = [
        Sphere(Vector3( 0,  -100.5, -2), 100, white),   # ground plane
        Sphere(Vector3(-0.7,    0,  -2), 0.5, red),     # left sphere
        Sphere(Vector3( 0.7,    0,  -2), 0.5, blue),    # right sphere
        Sphere(Vector3( 0,      4,  -2), 1.5, light),   # overhead light
    ]

    W, H, SAMPLES = 400, 300, 48
    image = np.zeros((H, W, 3))

    print(f"Rendering {W}x{H} @ {SAMPLES} samples per pixel...")
    for y in range(H):
        for x in range(W):
            acc = Vector3(0, 0, 0)
            for _ in range(SAMPLES):
                u   = (x + np.random.random()) / W
                v   = (y + np.random.random()) / H
                dir = Vector3((u-0.5)*(W/H), (0.5-v), -1).normalize()
                acc = acc + trace(scene, Ray(Vector3(0, 0.5, 0.5), dir))
            avg = acc * (1.0/SAMPLES)
            image[y, x] = [np.sqrt(avg.x), np.sqrt(avg.y), np.sqrt(avg.z)]
        if y % 30 == 0:
            print(f"  {int(y/H*100)}% complete...")

    out = Image.fromarray((np.clip(image,0,1)*255).astype(np.uint8))
    out.save("milestone1_render.png")
    out.show()
    print("Done! Saved as milestone1_render.png")

if __name__ == "__main__":
    main()