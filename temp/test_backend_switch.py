"""Test GPU backend integration."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')

from elements.solvers import (
    GeometryBuilder, 
    GPUGeometryBuilder, 
    set_geometry_backend, 
    get_geometry_backend,
    get_geometry_builder
)

print('Imports OK')
print(f'Default backend: {get_geometry_backend()}')
print(f'Default builder: {get_geometry_builder()}')

set_geometry_backend('gpu')
print(f'After set_geometry_backend("gpu"): {get_geometry_backend()}')
print(f'Builder: {get_geometry_builder()}')

set_geometry_backend('cpu')
print(f'After set_geometry_backend("cpu"): {get_geometry_backend()}')
print(f'Builder: {get_geometry_builder()}')
