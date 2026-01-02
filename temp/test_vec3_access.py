import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
from mathutils.vec3 import Vec3
import time

v = Vec3(1.0, 2.0, 3.0)

# Test __getitem__
start = time.perf_counter()
for _ in range(1000000):
    x = v[0]
    y = v[1]
    z = v[2]
elapsed = time.perf_counter() - start
print(f'Vec3 __getitem__: {elapsed:.4f}s for 1M iterations (3M accesses)')

# Test .x, .y, .z
start = time.perf_counter()
for _ in range(1000000):
    x = v.x
    y = v.y
    z = v.z
elapsed = time.perf_counter() - start
print(f'Vec3 .x/.y/.z: {elapsed:.4f}s for 1M iterations (3M accesses)')

# Test tuple
t = (1.0, 2.0, 3.0)
start = time.perf_counter()
for _ in range(1000000):
    x = t[0]
    y = t[1]
    z = t[2]
elapsed = time.perf_counter() - start
print(f'Tuple [0][1][2]: {elapsed:.4f}s for 1M iterations (3M accesses)')
