"""Benchmark Vec3 vs tuple overhead."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src')

import timeit

# Import C extension
from gxml.mathutils._vec3 import Vec3, transform_point, dot

# Create test data
matrix = [
    [1, 0, 0, 10],
    [0, 1, 0, 20],
    [0, 0, 1, 30],
    [0, 0, 0, 1]
]
point = (0.5, 0.5, 0.5)

# Warm up
result = transform_point(point, matrix)
print(f"transform_point returns: {type(result).__name__} = {result}")

# Benchmark 1: Access Vec3 components via indexing
def bench_vec3_index():
    v = transform_point(point, matrix)
    x, y, z = v[0], v[1], v[2]
    return x + y + z

# Benchmark 2: Access Vec3 via .x .y .z (if available)
def bench_vec3_attr():
    v = transform_point(point, matrix)
    x, y, z = v.x, v.y, v.z
    return x + y + z

# Benchmark 3: What if we just returned tuple?
# (we'd need to modify C code - for now simulate)
def bench_tuple():
    v = transform_point(point, matrix)
    # Convert immediately 
    t = (v[0], v[1], v[2])
    x, y, z = t
    return x + y + z

# Benchmark 4: Pass Vec3 to dot (C can extract efficiently)
v1 = Vec3(1, 0, 0)
v2 = Vec3(0, 1, 0)
def bench_dot_vec3():
    return dot(v1, v2)

t1 = Vec3(1, 0, 0)
t2 = Vec3(0, 1, 0)
def bench_dot_tuple():
    return dot(t1, t2)

# Run benchmarks
n = 100000
print(f"\nRunning {n} iterations each:")

t = timeit.timeit(bench_vec3_index, number=n)
print(f"  Vec3 indexing [0],[1],[2]:  {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

t = timeit.timeit(bench_vec3_attr, number=n)
print(f"  Vec3 attr .x,.y,.z:         {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

t = timeit.timeit(bench_tuple, number=n)
print(f"  Vec3 to tuple then access:  {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

t = timeit.timeit(bench_dot_vec3, number=n)
print(f"  dot(Vec3, Vec3):            {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

t = timeit.timeit(bench_dot_tuple, number=n)
print(f"  dot(tuple, tuple):          {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

# Benchmark Vec3 creation overhead
def bench_create_vec3():
    return Vec3(1.0, 2.0, 3.0)

def bench_create_tuple():
    return (1.0, 2.0, 3.0)

t = timeit.timeit(bench_create_vec3, number=n)
print(f"\n  Vec3(1,2,3):                {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")

t = timeit.timeit(bench_create_tuple, number=n)
print(f"  (1,2,3) tuple:              {t*1000:.2f} ms ({t/n*1e6:.2f} µs/call)")
