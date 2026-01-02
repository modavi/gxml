# GPU acceleration module for GXML
# Provides GPU-accelerated geometry computation via Metal or Taichi

from .metal_geometry import MetalGeometryAccelerator, CPUGeometryAccelerator, is_metal_available, get_accelerator
from .taichi_geometry import TaichiGeometryAccelerator, is_taichi_available, get_taichi_accelerator

__all__ = [
    'MetalGeometryAccelerator',
    'CPUGeometryAccelerator', 
    'is_metal_available',
    'get_accelerator',
    'TaichiGeometryAccelerator',
    'is_taichi_available',
    'get_taichi_accelerator',
]
