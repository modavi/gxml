#!/usr/bin/env python3
"""Test the shader backend."""

import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src')

from gxml.gpu import list_available_backends, get_shader_backend
import numpy as np

print('Available backends:')
for name, available in list_available_backends():
    status = '✅' if available else '❌'
    print(f'  {status} {name}')

print()
backend = get_shader_backend()
print(f'Selected backend: {backend.name}')

# Quick test
starts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
ends = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.float64)

i_idx, j_idx, t_i, t_j, positions = backend.find_intersections(starts, ends)
print(f'Found {len(i_idx)} intersections')
if len(positions) > 0:
    print(f'Position: {positions[0]}')
