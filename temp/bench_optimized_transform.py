"""Benchmark optimized pure Python transform_point."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src')

import timeit

from gxml.mathutils._vec3 import Vec3, transform_point as c_transform_point
from gxml.mathutils.vec3 import _python_transform_point

# Test data
matrix = [
    [1, 0, 0, 10],
    [0, 1, 0, 20],
    [0, 0, 1, 30],
    [0, 0, 0, 1]
]
point = (0.5, 0.5, 0.5)

def optimized_transform_point(point, matrix):
    """Optimized pure Python - returns tuple, no try/except."""
    x, y, z = point[0], point[1], point[2]
    m = matrix  # local reference
    # For affine transforms, w is always 1 (matrix[3][3]=1, others=0)
    return (
        x * m[0][0] + y * m[0][1] + z * m[0][2] + m[0][3],
        x * m[1][0] + y * m[1][1] + z * m[1][2] + m[1][3],
        x * m[2][0] + y * m[2][1] + z * m[2][2] + m[2][3],
    )

n = 100000
print(f"Running {n} iterations each:\n")

# C extension
def bench_c():
    return c_transform_point(point, matrix)

t = timeit.timeit(bench_c, number=n)
print(f"  C transform_point (returns Vec3):      {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Python fallback (current)
def bench_python_current():
    return _python_transform_point(point, matrix)

t = timeit.timeit(bench_python_current, number=n)
print(f"  Python transform_point (Vec3):         {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Optimized pure Python
def bench_optimized():
    return optimized_transform_point(point, matrix)

t = timeit.timeit(bench_optimized, number=n)
print(f"  Optimized Python (returns tuple):      {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Verify correctness
c_result = c_transform_point(point, matrix)
opt_result = optimized_transform_point(point, matrix)
print(f"\nC result:   {c_result}")
print(f"Opt result: {opt_result}")
print(f"Match: {abs(c_result[0]-opt_result[0]) < 1e-9 and abs(c_result[1]-opt_result[1]) < 1e-9 and abs(c_result[2]-opt_result[2]) < 1e-9}")
