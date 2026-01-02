#!/usr/bin/env python
import sys
sys.path.insert(0, 'src/gxml')
from mathutils.vec3 import Vec3

v = Vec3(3.20711, 0, -0.707107)
print('hasattr x:', hasattr(v, 'x'))
print('v.x:', v.x)

def _allclose(actual, expected, atol=1e-6):
    # Handle Vec3 objects (have x, y, z attributes) 
    if hasattr(actual, 'x'):
        actual = (actual.x, actual.y, actual.z)
    if hasattr(expected, 'x'):
        expected = (expected.x, expected.y, expected.z)
    
    print('actual:', actual)
    print('expected:', expected)
    if len(actual) != len(expected):
        print('length mismatch')
        return False
    for i, (a, e) in enumerate(zip(actual, expected)):
        diff = abs(a - e)
        print(f'  [{i}] a={a}, e={e}, diff={diff}')
        if diff > atol:
            return False
    return True

expected = [3.20711, 0, -0.707107]
print('allclose result:', _allclose(v, expected))
