# GPU acceleration module for GXML
# Provides GPU-accelerated geometry computation via Metal, WebGPU, or Taichi
# Uses lazy imports to avoid slow Taichi initialization on import

def __getattr__(name):
    """Lazy import attributes to avoid slow Taichi initialization."""
    
    # Metal geometry (fast import)
    if name in ('MetalGeometryAccelerator', 'CPUGeometryAccelerator', 
                'is_metal_available', 'get_accelerator'):
        from .metal_geometry import (
            MetalGeometryAccelerator, CPUGeometryAccelerator,
            is_metal_available, get_accelerator
        )
        return locals()[name]
    
    # Taichi geometry (slow import, only when needed)
    if name in ('TaichiGeometryAccelerator', 'is_taichi_available', 
                'get_taichi_accelerator'):
        from .taichi_geometry import (
            TaichiGeometryAccelerator, is_taichi_available,
            get_taichi_accelerator
        )
        return locals()[name]
    
    # Shader backend (fast import)
    if name in ('ShaderBackend', 'CPUBackend', 'MetalBackend', 'WGPUBackend',
                'CExtensionBackend', 'get_shader_backend', 'list_available_backends'):
        from .shader_backend import (
            ShaderBackend, CPUBackend, MetalBackend, WGPUBackend,
            CExtensionBackend, get_shader_backend, list_available_backends
        )
        return locals()[name]
    
    raise AttributeError(f"module 'gxml.gpu' has no attribute '{name}'")


__all__ = [
    # Legacy accelerators
    'MetalGeometryAccelerator',
    'CPUGeometryAccelerator', 
    'is_metal_available',
    'get_accelerator',
    'TaichiGeometryAccelerator',
    'is_taichi_available',
    'get_taichi_accelerator',
    # New shader backend
    'ShaderBackend',
    'CPUBackend',
    'MetalBackend',
    'WGPUBackend',
    'CExtensionBackend',
    'get_shader_backend',
    'list_available_backends',
]
