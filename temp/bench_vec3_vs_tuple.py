#!/usr/bin/env python
"""Compare different approaches to point transformation."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')

from mathutils._vec3 import Mat4, Vec3, bilinear_interpolate
import time

# Create a transform matrix
mat = Mat4()
mat.set_trs(10, 20, 30, 0.5, 0.3, 0.1, 2, 2, 2)

# Quad points (like QuadInterpolator uses)
p0 = (0.0, 0.0, 0.0)
p1 = (1.0, 0.0, 0.0)  
p2 = (1.0, 1.0, 0.0)
p3 = (0.0, 1.0, 0.0)

# Test point (t, s values)
t, s = 0.5, 0.5

N = 100000

# Method 1: Current approach (bilinear + transform_point separate)
start = time.perf_counter()
for _ in range(N):
    # Bilinear interpolation
    result = bilinear_interpolate(t, s, p0, p1, p2, p3)
    interp = (result[0], result[1], 0.0 + result[2])  # This is what Python does
    # Transform
    v = mat.transform_point(interp)
    x, y, z = v[0], v[1], v[2]
end = time.perf_counter()
separate_time = (end-start)*1000
print(f"Separate (bilinear + transform): {separate_time:.2f}ms for {N} calls")

# Method 2: Combined bilinear_transform
start = time.perf_counter()
for _ in range(N):
    v = mat.bilinear_transform(t, s, 0.0, p0, p1, p2, p3)
    x, y, z = v[0], v[1], v[2]
end = time.perf_counter()
combined_time = (end-start)*1000
print(f"Combined (bilinear_transform): {combined_time:.2f}ms for {N} calls")

# Method 3: Just transform_point alone (baseline)
point = (0.5, 0.5, 0.0)
start = time.perf_counter()
for _ in range(N):
    v = mat.transform_point(point)
    x, y, z = v[0], v[1], v[2]
end = time.perf_counter()
print(f"Just transform_point: {(end-start)*1000:.2f}ms for {N} calls")

# Method 4: Just bilinear alone
start = time.perf_counter()
for _ in range(N):
    result = bilinear_interpolate(t, s, p0, p1, p2, p3)
    x, y, z = result[0], result[1], result[2]
end = time.perf_counter()
print(f"Just bilinear_interpolate: {(end-start)*1000:.2f}ms for {N} calls")

# Estimated savings for 15k transform_point calls per pipeline run
savings_per_call = (separate_time - combined_time) / N
calls_per_run = 15220
print(f"\nSavings: {savings_per_call*1000:.4f}ms per call")
print(f"Estimated savings per pipeline run: {savings_per_call * calls_per_run:.1f}ms")
