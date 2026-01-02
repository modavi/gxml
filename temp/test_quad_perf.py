import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
from mathutils.gxml_math import create_transform_matrix_from_quad
import time

# Test with Vec3 objects
from mathutils.vec3 import Vec3
points = [Vec3(0,0,0), Vec3(1,0,0), Vec3(1,1,0), Vec3(0,1,0)]

start = time.perf_counter()
for _ in range(10000):
    create_transform_matrix_from_quad(points)
elapsed = time.perf_counter() - start
print(f'Vec3 points: {elapsed:.4f}s for 10k calls')

# Test with tuples
points_tuple = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
start = time.perf_counter()
for _ in range(10000):
    create_transform_matrix_from_quad(points_tuple)
elapsed = time.perf_counter() - start
print(f'Tuple points: {elapsed:.4f}s for 10k calls')

# Test with numpy arrays
import numpy as np
points_np = [np.array([0,0,0]), np.array([1,0,0]), np.array([1,1,0]), np.array([0,1,0])]
start = time.perf_counter()
for _ in range(10000):
    create_transform_matrix_from_quad(points_np)
elapsed = time.perf_counter() - start
print(f'NumPy points: {elapsed:.4f}s for 10k calls')
