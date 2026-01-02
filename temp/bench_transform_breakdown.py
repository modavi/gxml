"""Benchmark transform_point breakdown."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src')

import timeit

from gxml.mathutils._vec3 import Vec3, transform_point, bilinear_interpolate

# Test data
matrix = [
    [1, 0, 0, 10],
    [0, 1, 0, 20],
    [0, 0, 1, 30],
    [0, 0, 0, 1]
]

# Pre-create tuple matrix vs nested list
matrix_tuple = tuple(tuple(row) for row in matrix)
point = (0.5, 0.5, 0.5)
point_vec3 = Vec3(0.5, 0.5, 0.5)

n = 100000
print(f"Running {n} iterations each:\n")

# Test 1: Transform with list matrix
def bench_list_matrix():
    return transform_point(point, matrix)

t = timeit.timeit(bench_list_matrix, number=n)
print(f"  transform_point(tuple, list_matrix):   {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Test 2: Transform with tuple matrix
def bench_tuple_matrix():
    return transform_point(point, matrix_tuple)

t = timeit.timeit(bench_tuple_matrix, number=n)
print(f"  transform_point(tuple, tuple_matrix):  {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Test 3: Transform with Vec3 input
def bench_vec3_input():
    return transform_point(point_vec3, matrix)

t = timeit.timeit(bench_vec3_input, number=n)
print(f"  transform_point(Vec3, list_matrix):    {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Compare with simpler operation
c1, c2, c3, c4 = (0,0,0), (1,0,0), (1,1,0), (0,1,0)
def bench_bilinear():
    return bilinear_interpolate(c1, c2, c3, c4, 0.5, 0.5)

t = timeit.timeit(bench_bilinear, number=n)
print(f"  bilinear_interpolate:                  {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Pure Python baseline
def bench_pure_python():
    x, y, z = 0.5, 0.5, 0.5
    m = matrix
    return (
        m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3],
        m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3],
        m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3],
    )

t = timeit.timeit(bench_pure_python, number=n)
print(f"  Pure Python matrix mul (returns tuple):{t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")
