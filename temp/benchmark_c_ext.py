#!/usr/bin/env python
"""Benchmark C extension vs Python implementation."""
import sys
sys.path.insert(0, 'src')

import time

def benchmark(name, func, *args, iterations=100000):
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args)
    elapsed = time.perf_counter() - start
    return elapsed

# Import both implementations
from gxml.mathutils.vec3 import _PythonVec3, _USE_C_EXTENSION

if _USE_C_EXTENSION:
    from gxml.mathutils._vec3 import Vec3 as CVec3, transform_point as c_transform_point, intersect_line_plane as c_intersect_line_plane
else:
    print("C extension not available!")
    sys.exit(1)

import numpy as np

# Test data
np_matrix = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.707, 0.707, 0.0],
    [0.0, -0.707, 0.707, 0.0],
    [5.0, 10.0, 15.0, 1.0]
])

print("=" * 60)
print("Microbenchmark: C Extension vs Python (100k iterations)")
print("=" * 60)

# Vec3 creation
print("\n--- Vec3 Creation ---")
t_py = benchmark("Python Vec3(1,2,3)", _PythonVec3, 1.0, 2.0, 3.0)
t_c = benchmark("C Vec3(1,2,3)", CVec3, 1.0, 2.0, 3.0)
print(f"  Python: {t_py*1000:.1f}ms, C: {t_c*1000:.1f}ms, Speedup: {t_py/t_c:.1f}x")

# Vec3 from tuple
t_py = benchmark("Python Vec3(tuple)", _PythonVec3, (1.0, 2.0, 3.0))
t_c = benchmark("C Vec3(tuple)", CVec3, (1.0, 2.0, 3.0))
print(f"  From tuple - Python: {t_py*1000:.1f}ms, C: {t_c*1000:.1f}ms, Speedup: {t_py/t_c:.1f}x")

# Vec3 operations
print("\n--- Vec3 Operations ---")
py_v1, py_v2 = _PythonVec3(1, 2, 3), _PythonVec3(4, 5, 6)
c_v1, c_v2 = CVec3(1, 2, 3), CVec3(4, 5, 6)

t_py = benchmark("Python v1 + v2", py_v1.__add__, py_v2)
t_c = benchmark("C v1 + v2", c_v1.__add__, c_v2)
print(f"  Add: Python: {t_py*1000:.1f}ms, C: {t_c*1000:.1f}ms, Speedup: {t_py/t_c:.1f}x")

t_py = benchmark("Python v1 - v2", py_v1.__sub__, py_v2)
t_c = benchmark("C v1 - v2", c_v1.__sub__, c_v2)
print(f"  Sub: Python: {t_py*1000:.1f}ms, C: {t_c*1000:.1f}ms, Speedup: {t_py/t_c:.1f}x")

t_py = benchmark("Python v1 * 2", py_v1.__mul__, 2.0)
t_c = benchmark("C v1 * 2", c_v1.__mul__, 2.0)
print(f"  Mul: Python: {t_py*1000:.1f}ms, C: {t_c*1000:.1f}ms, Speedup: {t_py/t_c:.1f}x")

t_py = benchmark("Python v1.dot(v2)", py_v1.dot, py_v2)
t_c = benchmark("C v1.dot(v2)", c_v1.dot, c_v2)
print(f"  Dot: Python: {t_py*1000:.1f}ms, C: {t_c*1000:.1f}ms, Speedup: {t_py/t_c:.1f}x")

t_py = benchmark("Python v1.length()", py_v1.length)
t_c = benchmark("C v1.length()", c_v1.length)
print(f"  Length: Python: {t_py*1000:.1f}ms, C: {t_c*1000:.1f}ms, Speedup: {t_py/t_c:.1f}x")

# transform_point
print("\n--- transform_point ---")
from gxml.mathutils.vec3 import _python_transform_point
point = (1.0, 2.0, 3.0)
c_point = CVec3(1, 2, 3)

t_py = benchmark("Python transform_point", _python_transform_point, point, np_matrix)
t_c = benchmark("C transform_point", c_transform_point, point, np_matrix)
print(f"  Python: {t_py*1000:.1f}ms, C: {t_c*1000:.1f}ms, Speedup: {t_py/t_c:.1f}x")

# intersect_line_plane
print("\n--- intersect_line_plane ---")
from gxml.mathutils.vec3 import _python_intersect_line_plane
line_point = (0, 0, 0)
line_dir = (1, 1, 1)
plane_point = (5, 5, 5)
plane_normal = (1, 0, 0)

t_py = benchmark("Python intersect", _python_intersect_line_plane, line_point, line_dir, plane_point, plane_normal)
t_c = benchmark("C intersect", c_intersect_line_plane, line_point, line_dir, plane_point, plane_normal)
print(f"  Python: {t_py*1000:.1f}ms, C: {t_c*1000:.1f}ms, Speedup: {t_py/t_c:.1f}x")

print("\n" + "=" * 60)
