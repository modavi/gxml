#!/usr/bin/env python3
"""Test Taichi solver imports."""

import sys
sys.path.insert(0, 'src/gxml')

print('Testing Taichi solver imports...')

try:
    from elements.solvers.taichi.taichi_intersection_solver import TaichiIntersectionSolver
    print('✓ TaichiIntersectionSolver imported')
except Exception as e:
    print(f'✗ TaichiIntersectionSolver: {e}')

try:
    from elements.solvers.taichi.taichi_face_solver import TaichiFaceSolver
    print('✓ TaichiFaceSolver imported')
except Exception as e:
    print(f'✗ TaichiFaceSolver: {e}')

try:
    from elements.solvers.taichi.taichi_geometry_builder import TaichiGeometryBuilder
    print('✓ TaichiGeometryBuilder imported')
except Exception as e:
    print(f'✗ TaichiGeometryBuilder: {e}')
