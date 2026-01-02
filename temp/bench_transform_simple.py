"""Benchmark transform_point - simple."""
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
point = (0.5, 0.5, 0.5)
p0, p1, p2, p3 = (0,0,0), (1,0,0), (1,1,0), (0,1,0)

n = 100000
print(f"Running {n} iterations each:\n")

# Test C transform_point
def bench_c_transform():
    return transform_point(point, matrix)

t = timeit.timeit(bench_c_transform, number=n)
print(f"  C transform_point:        {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Test C bilinear_interpolate
def bench_c_bilinear():
    return bilinear_interpolate(0.5, 0.5, p0, p1, p2, p3)

t = timeit.timeit(bench_c_bilinear, number=n)
print(f"  C bilinear_interpolate:   {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Pure Python matrix multiply
def bench_pure_python():
    x, y, z = point
    m = matrix
    return (
        m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3],
        m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3],
        m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3],
    )

t = timeit.timeit(bench_pure_python, number=n)
print(f"  Pure Python matrix mul:   {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Pure Python bilinear
def bench_pure_bilinear():
    t_val, s_val = 0.5, 0.5
    inv_t = 1.0 - t_val
    inv_s = 1.0 - s_val
    w0 = inv_t * inv_s
    w1 = t_val * inv_s
    w2 = t_val * s_val
    w3 = inv_t * s_val
    return (
        p0[0]*w0 + p1[0]*w1 + p2[0]*w2 + p3[0]*w3,
        p0[1]*w0 + p1[1]*w1 + p2[1]*w2 + p3[1]*w3,
        p0[2]*w0 + p1[2]*w1 + p2[2]*w2 + p3[2]*w3,
    )

t = timeit.timeit(bench_pure_bilinear, number=n)
print(f"  Pure Python bilinear:     {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Vec3 creation
def bench_vec3_create():
    return Vec3(1.0, 2.0, 3.0)

def bench_tuple_create():
    return (1.0, 2.0, 3.0)

print()
t = timeit.timeit(bench_vec3_create, number=n)
print(f"  Vec3(1,2,3):              {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

t = timeit.timeit(bench_tuple_create, number=n)
print(f"  (1,2,3) tuple:            {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

print(f"\nVec3 overhead: ~{(172/13.7 - 1)*100:.0f}% slower than tuple (based on earlier)")
