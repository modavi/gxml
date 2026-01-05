"""
Taichi-based GPU/SIMD acceleration for GXML geometry computation.

Taichi is a domain-specific language embedded in Python that compiles to
highly optimized backends: CUDA, Metal, Vulkan, OpenGL, or CPU SIMD.

This provides an alternative to the Metal-specific implementation that
works on any platform.

Requirements:
    pip install taichi

Usage:
    from elements.solvers.gpu.taichi_geometry import TaichiGeometryAccelerator, is_taichi_available
    
    if is_taichi_available():
        accel = TaichiGeometryAccelerator()
        world_points = accel.transform_points_batch(matrices, local_points)
"""

import numpy as np
from typing import Optional

_TAICHI_AVAILABLE = False
_TI = None

try:
    import taichi as ti
    _TI = ti
    _TAICHI_AVAILABLE = True
except ImportError:
    pass


def is_taichi_available() -> bool:
    """Check if Taichi is available."""
    return _TAICHI_AVAILABLE


class TaichiGeometryAccelerator:
    """
    GPU/SIMD-accelerated geometry computation using Taichi.
    
    Automatically selects the best available backend:
    - CUDA (NVIDIA GPU)
    - Metal (Apple GPU) 
    - Vulkan (Cross-platform GPU)
    - CPU (SIMD-optimized)
    """
    
    def __init__(self, backend: Optional[str] = None):
        """
        Initialize Taichi with specified or auto-detected backend.
        
        Args:
            backend: 'cuda', 'metal', 'vulkan', 'cpu', or None for auto
        """
        if not _TAICHI_AVAILABLE:
            raise RuntimeError("Taichi is not available. Install with: pip install taichi")
        
        # Initialize Taichi
        if backend:
            _TI.init(arch=getattr(_TI, backend))
        else:
            # Auto-select best available
            _TI.init()
        
        self._compiled = False
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile Taichi kernels."""
        if self._compiled:
            return
        
        # Define field types
        self.mat_field = _TI.Matrix.field(4, 4, dtype=_TI.f32, shape=())
        self.vec_field = _TI.Vector.field(4, dtype=_TI.f32, shape=())
        
        @_TI.kernel
        def transform_kernel(
            matrices: _TI.types.ndarray(dtype=_TI.f32, ndim=3),  # (N, 4, 4)
            points: _TI.types.ndarray(dtype=_TI.f32, ndim=3),    # (N, M, 4)
            output: _TI.types.ndarray(dtype=_TI.f32, ndim=3),    # (N, M, 4)
        ):
            for i, j in _TI.ndrange(matrices.shape[0], points.shape[1]):
                # Matrix-vector multiply
                for k in _TI.static(range(4)):
                    val = 0.0
                    for l in _TI.static(range(4)):
                        val += matrices[i, k, l] * points[i, j, l]
                    output[i, j, k] = val
        
        @_TI.kernel
        def face_points_kernel(
            matrices: _TI.types.ndarray(dtype=_TI.f32, ndim=3),      # (N, 4, 4)
            half_t: _TI.types.ndarray(dtype=_TI.f32, ndim=1),        # (N,)
            ts: _TI.types.ndarray(dtype=_TI.f32, ndim=2),            # (N, 2)
            sides: _TI.types.ndarray(dtype=_TI.i32, ndim=1),         # (N,)
            output: _TI.types.ndarray(dtype=_TI.f32, ndim=2),        # (N, 4)
        ):
            for i in range(matrices.shape[0]):
                t = ts[i, 0]
                s = ts[i, 1]
                ht = half_t[i]
                side = sides[i]
                
                # Compute local point based on face side
                local_x = 0.0
                local_y = 0.0
                local_z = 0.0
                
                if side == 0:  # FRONT
                    local_x = t
                    local_y = s
                    local_z = ht
                elif side == 1:  # BACK
                    local_x = t
                    local_y = s
                    local_z = -ht
                elif side == 2:  # TOP
                    local_x = t
                    local_y = 1.0
                    local_z = -ht + s * 2.0 * ht
                elif side == 3:  # BOTTOM
                    local_x = t
                    local_y = 0.0
                    local_z = -ht + s * 2.0 * ht
                elif side == 4:  # START
                    local_x = 0.0
                    local_y = s
                    local_z = -ht + t * 2.0 * ht
                elif side == 5:  # END
                    local_x = 1.0
                    local_y = s
                    local_z = -ht + t * 2.0 * ht
                
                # Transform: world = mat * local
                for k in _TI.static(range(4)):
                    local_w = 1.0 if k == 3 else 0.0
                    local_arr = [local_x, local_y, local_z, 1.0]
                    val = (matrices[i, k, 0] * local_x + 
                           matrices[i, k, 1] * local_y + 
                           matrices[i, k, 2] * local_z + 
                           matrices[i, k, 3])
                    output[i, k] = val
        
        @_TI.kernel
        def intersections_2d_kernel(
            lines_a: _TI.types.ndarray(dtype=_TI.f32, ndim=2),  # (N, 4)
            lines_b: _TI.types.ndarray(dtype=_TI.f32, ndim=2),  # (N, 4)
            output: _TI.types.ndarray(dtype=_TI.f32, ndim=2),   # (N, 4)
        ):
            for i in range(lines_a.shape[0]):
                ax1 = lines_a[i, 0]
                ay1 = lines_a[i, 1]
                ax2 = lines_a[i, 2]
                ay2 = lines_a[i, 3]
                
                bx1 = lines_b[i, 0]
                by1 = lines_b[i, 1]
                bx2 = lines_b[i, 2]
                by2 = lines_b[i, 3]
                
                # Direction vectors
                adx = ax2 - ax1
                ady = ay2 - ay1
                bdx = bx2 - bx1
                bdy = by2 - by1
                
                denom = adx * bdy - ady * bdx
                
                if _TI.abs(denom) < 1e-10:
                    output[i, 0] = -1.0
                    output[i, 1] = -1.0
                    output[i, 2] = 0.0
                    output[i, 3] = 0.0
                else:
                    dx = bx1 - ax1
                    dy = by1 - ay1
                    
                    t_a = (dx * bdy - dy * bdx) / denom
                    t_b = (dx * ady - dy * adx) / denom
                    
                    output[i, 0] = t_a
                    output[i, 1] = t_b
                    output[i, 2] = ax1 + t_a * adx
                    output[i, 3] = ay1 + t_a * ady
        
        self._transform_kernel = transform_kernel
        self._face_points_kernel = face_points_kernel
        self._intersections_kernel = intersections_2d_kernel
        self._compiled = True
    
    def transform_points_batch(
        self,
        matrices: np.ndarray,
        local_points: np.ndarray
    ) -> np.ndarray:
        """
        Transform batches of points by corresponding matrices.
        
        Args:
            matrices: N transformation matrices, shape (N, 4, 4) 
            local_points: M local points per matrix, shape (N, M, 4)
            
        Returns:
            World-space points, shape (N, M, 4)
        """
        matrices = np.ascontiguousarray(matrices, dtype=np.float32)
        local_points = np.ascontiguousarray(local_points, dtype=np.float32)
        output = np.zeros_like(local_points)
        
        self._transform_kernel(matrices, local_points, output)
        _TI.sync()
        
        return output
    
    def compute_face_points_batch(
        self,
        matrices: np.ndarray,
        half_thicknesses: np.ndarray,
        ts_coords: np.ndarray,
        face_sides: np.ndarray
    ) -> np.ndarray:
        """
        Compute world-space face points from (t, s) coordinates.
        """
        matrices = np.ascontiguousarray(matrices, dtype=np.float32)
        half_t = np.ascontiguousarray(half_thicknesses, dtype=np.float32)
        ts = np.ascontiguousarray(ts_coords, dtype=np.float32)
        sides = np.ascontiguousarray(face_sides, dtype=np.int32)
        output = np.zeros((len(ts), 4), dtype=np.float32)
        
        self._face_points_kernel(matrices, half_t, ts, sides, output)
        _TI.sync()
        
        return output
    
    def test_intersections_2d(
        self,
        lines_a: np.ndarray,
        lines_b: np.ndarray
    ) -> np.ndarray:
        """
        Test N pairs of 2D lines for intersection.
        """
        lines_a = np.ascontiguousarray(lines_a, dtype=np.float32)
        lines_b = np.ascontiguousarray(lines_b, dtype=np.float32)
        output = np.zeros((len(lines_a), 4), dtype=np.float32)
        
        self._intersections_kernel(lines_a, lines_b, output)
        _TI.sync()
        
        return output


def get_taichi_accelerator(backend: Optional[str] = None) -> TaichiGeometryAccelerator:
    """Get Taichi accelerator with specified or auto backend."""
    return TaichiGeometryAccelerator(backend)
