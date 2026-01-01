"""Benchmark C extension vs Python vs NumPy."""
import sys
sys.path.insert(0, 'src')

import time
import numpy as np

def benchmark(name, func, *args, iterations=100000):
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args)
    elapsed = time.perf_counter() - start
    return elapsed

# Import implementations
from gxml.mathutils.vec3 import _PythonVec3, _USE_C_EXTENSION
from gxml.mathutils._vec3 import Vec3 as CVec3, transform_point as c_transform_point, intersect_line_plane as c_intersect_line_plane
from gxml.mathutils.vec3 import _python_transform_point, _python_intersect_line_plane

print("=" * 70)
print("Microbenchmark: NumPy vs C Extension vs Pure Python (100k iterations)")
print("=" * 70)

# Test data
np_matrix = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.707, 0.707, 0.0],
    [0.0, -0.707, 0.707, 0.0],
    [5.0, 10.0, 15.0, 1.0]
])

# --- Vec3 Creation ---
print("\n--- Vector Creation ---")

t_np = benchmark("numpy", lambda: np.array([1.0, 2.0, 3.0]))
t_c = benchmark("C Vec3", CVec3, 1.0, 2.0, 3.0)
t_py = benchmark("Python Vec3", _PythonVec3, 1.0, 2.0, 3.0)
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

# From tuple
t_np = benchmark("numpy from tuple", np.array, (1.0, 2.0, 3.0))
t_c = benchmark("C from tuple", CVec3, (1.0, 2.0, 3.0))
t_py = benchmark("Python from tuple", _PythonVec3, (1.0, 2.0, 3.0))
print(f"\n  From tuple:")
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

# --- Vector Operations ---
print("\n--- Vector Addition ---")
np_v1, np_v2 = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
c_v1, c_v2 = CVec3(1, 2, 3), CVec3(4, 5, 6)
py_v1, py_v2 = _PythonVec3(1, 2, 3), _PythonVec3(4, 5, 6)

t_np = benchmark("numpy add", lambda: np_v1 + np_v2)
t_c = benchmark("C add", c_v1.__add__, c_v2)
t_py = benchmark("Python add", py_v1.__add__, py_v2)
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

print("\n--- Vector Subtraction ---")
t_np = benchmark("numpy sub", lambda: np_v1 - np_v2)
t_c = benchmark("C sub", c_v1.__sub__, c_v2)
t_py = benchmark("Python sub", py_v1.__sub__, py_v2)
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

print("\n--- Scalar Multiply ---")
t_np = benchmark("numpy mul", lambda: np_v1 * 2.0)
t_c = benchmark("C mul", c_v1.__mul__, 2.0)
t_py = benchmark("Python mul", py_v1.__mul__, 2.0)
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

print("\n--- Dot Product ---")
t_np = benchmark("numpy dot", lambda: np.dot(np_v1, np_v2))
t_c = benchmark("C dot", c_v1.dot, c_v2)
t_py = benchmark("Python dot", py_v1.dot, py_v2)
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

print("\n--- Cross Product ---")
t_np = benchmark("numpy cross", lambda: np.cross(np_v1, np_v2))
t_c = benchmark("C cross", c_v1.cross, c_v2)
t_py = benchmark("Python cross", py_v1.cross, py_v2)
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

print("\n--- Vector Length ---")
t_np = benchmark("numpy length", lambda: np.linalg.norm(np_v1))
t_c = benchmark("C length", c_v1.length)
t_py = benchmark("Python length", py_v1.length)
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

print("\n--- Vector Normalize ---")
t_np = benchmark("numpy normalize", lambda: np_v1 / np.linalg.norm(np_v1))
t_c = benchmark("C normalize", c_v1.normalized)
t_py = benchmark("Python normalize", py_v1.normalized)
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

# --- Transform Point ---
print("\n--- transform_point (4x4 matrix) ---")

def np_transform_point(point, matrix):
    x, y, z = point
    w = x * matrix[0, 3] + y * matrix[1, 3] + z * matrix[2, 3] + matrix[3, 3]
    if abs(w) > 1e-10:
        inv_w = 1.0 / w
        return np.array([
            (x * matrix[0, 0] + y * matrix[1, 0] + z * matrix[2, 0] + matrix[3, 0]) * inv_w,
            (x * matrix[0, 1] + y * matrix[1, 1] + z * matrix[2, 1] + matrix[3, 1]) * inv_w,
            (x * matrix[0, 2] + y * matrix[1, 2] + z * matrix[2, 2] + matrix[3, 2]) * inv_w
        ])
    return np.array([0.0, 0.0, 0.0])

point = (1.0, 2.0, 3.0)
t_np = benchmark("numpy", np_transform_point, point, np_matrix)
t_c = benchmark("C", c_transform_point, point, np_matrix)
t_py = benchmark("Python", _python_transform_point, point, np_matrix)
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

# --- Intersect Line Plane ---
print("\n--- intersect_line_plane ---")

def np_intersect_line_plane(lp, ld, pp, pn):
    lp, ld, pp, pn = np.array(lp), np.array(ld), np.array(pp), np.array(pn)
    denom = np.dot(ld, pn)
    if abs(denom) < 1e-10:
        return None
    t = np.dot(pp - lp, pn) / denom
    return lp + t * ld

line_point = (0, 0, 0)
line_dir = (1, 1, 1)
plane_point = (5, 5, 5)
plane_normal = (1, 0, 0)

t_np = benchmark("numpy", np_intersect_line_plane, line_point, line_dir, plane_point, plane_normal)
t_c = benchmark("C", c_intersect_line_plane, line_point, line_dir, plane_point, plane_normal)
t_py = benchmark("Python", _python_intersect_line_plane, line_point, line_dir, plane_point, plane_normal)
print(f"  NumPy:  {t_np*1000:>7.1f}ms")
print(f"  C:      {t_c*1000:>7.1f}ms  ({t_np/t_c:.1f}x vs numpy)")
print(f"  Python: {t_py*1000:>7.1f}ms  ({t_np/t_py:.1f}x vs numpy)")

# --- Summary ---
print("\n" + "=" * 70)
print("ANALYSIS:")
print("=" * 70)
print("""
NumPy is optimized for ARRAY operations (batch processing many vectors at once),
not single-vector operations. For individual 3D vectors:

- C extension is fastest due to zero overhead
- Pure Python is competitive for simple ops
- NumPy has significant per-call overhead for small arrays

NumPy shines when you can batch operations:
""")

# Batch operation comparison
print("--- Batch: Add 10,000 vectors at once ---")
np_batch1 = np.random.rand(10000, 3)
np_batch2 = np.random.rand(10000, 3)

def batch_c_add():
    return [CVec3(a[0], a[1], a[2]) + CVec3(b[0], b[1], b[2]) 
            for a, b in zip(np_batch1, np_batch2)]

t_np_batch = benchmark("numpy batch", lambda: np_batch1 + np_batch2, iterations=1000)
t_c_batch = benchmark("C batch", batch_c_add, iterations=1000)
print(f"  NumPy:  {t_np_batch*1000:>7.1f}ms (batch of 10k)")
print(f"  C loop: {t_c_batch*1000:>7.1f}ms (loop of 10k)")
print(f"  NumPy is {t_c_batch/t_np_batch:.1f}x faster for batched operations")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("""
For GXML's use case (processing individual vectors one at a time):
- C extension is the best choice
- NumPy's overhead makes it slower for single-vector operations

For batch processing (if you could restructure the algorithm):
- NumPy would be significantly faster
- But this would require major architecture changes
""")
