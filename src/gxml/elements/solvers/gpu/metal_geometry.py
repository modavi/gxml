"""
Metal GPU acceleration for GXML geometry computation.

This module provides GPU-accelerated matrix transforms for the geometry pipeline.
The key insight is that face segment corner transformations are embarrassingly parallel:
- Each panel has multiple face segments
- Each segment has 4 corners
- Each corner requires a Mat4 × Vec4 transform

For a 75-panel layout with ~1000 segments, that's ~4000 independent transforms.
GPU can process all of these in parallel, vs sequential on CPU.

Requirements:
- macOS 10.14+ 
- pyobjc-framework-Metal (pip install pyobjc-framework-Metal)

Usage:
    from elements.solvers.gpu import MetalGeometryAccelerator, is_metal_available
    
    if is_metal_available():
        accel = MetalGeometryAccelerator()
        world_points = accel.transform_points_batch(matrices, local_points)
"""

import numpy as np
from typing import List, Tuple, Optional
import struct

# Metal availability flag
_METAL_AVAILABLE = False
_METAL_DEVICE = None
_METAL_LIBRARY = None
_METAL_QUEUE = None

try:
    import Metal
    import objc
    _METAL_DEVICE = Metal.MTLCreateSystemDefaultDevice()
    _METAL_AVAILABLE = _METAL_DEVICE is not None
except ImportError:
    pass


def is_metal_available() -> bool:
    """Check if Metal GPU acceleration is available."""
    return _METAL_AVAILABLE


# Metal shader code for batch matrix transforms
METAL_SHADER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Transform a batch of points by a batch of matrices
// Input: matrices (4x4 floats per matrix), points (4 floats per point, w=1)
// Output: transformed points (4 floats per point)
kernel void transform_points_batch(
    device const float4x4* matrices [[buffer(0)]],    // N matrices
    device const float4* local_points [[buffer(1)]],  // M points per matrix
    device float4* world_points [[buffer(2)]],        // N*M output points
    device const uint* points_per_matrix [[buffer(3)]], // # points per matrix
    uint gid [[thread_position_in_grid]]
) {
    uint num_points = *points_per_matrix;
    uint matrix_idx = gid / num_points;
    uint point_idx = gid % num_points;
    
    float4x4 mat = matrices[matrix_idx];
    float4 local = local_points[matrix_idx * num_points + point_idx];
    
    // Matrix multiplication: world = mat * local
    float4 world;
    world.x = dot(mat[0], local);
    world.y = dot(mat[1], local);
    world.z = dot(mat[2], local);
    world.w = dot(mat[3], local);
    
    world_points[gid] = world;
}

// Batch compute face points from (t, s) coordinates
// This combines local point computation + transform in one kernel
kernel void compute_face_points(
    device const float4x4* matrices [[buffer(0)]],      // Panel transforms
    device const float* half_thicknesses [[buffer(1)]], // Panel half-thickness
    device const float2* ts_coords [[buffer(2)]],       // (t, s) coordinates
    device const uint* face_sides [[buffer(3)]],        // Face side enum values
    device float4* world_points [[buffer(4)]],          // Output world points
    uint gid [[thread_position_in_grid]]
) {
    // Face side enum values (must match Python PanelSide)
    // FRONT=0, BACK=1, TOP=2, BOTTOM=3, START=4, END=5
    
    float4x4 mat = matrices[gid];
    float half_t = half_thicknesses[gid];
    float2 ts = ts_coords[gid];
    uint side = face_sides[gid];
    
    float t = ts.x;
    float s = ts.y;
    float4 local;
    
    // Compute local point based on face side
    // Matches gxml_panel.py get_face_point()
    switch (side) {
        case 0:  // FRONT
            local = float4(t, s, half_t, 1.0);
            break;
        case 1:  // BACK
            local = float4(t, s, -half_t, 1.0);
            break;
        case 2:  // TOP
            local = float4(t, 1.0, -half_t + s * half_t * 2.0, 1.0);
            break;
        case 3:  // BOTTOM
            local = float4(t, 0.0, -half_t + s * half_t * 2.0, 1.0);
            break;
        case 4:  // START
            local = float4(0.0, s, -half_t + t * half_t * 2.0, 1.0);
            break;
        case 5:  // END
            local = float4(1.0, s, -half_t + t * half_t * 2.0, 1.0);
            break;
        default:
            local = float4(t, s, 0.0, 1.0);
            break;
    }
    
    // Transform to world space
    float4 world;
    world.x = dot(mat[0], local);
    world.y = dot(mat[1], local);
    world.z = dot(mat[2], local);
    world.w = dot(mat[3], local);
    
    world_points[gid] = world;
}

// Batch intersection test for panel centerlines
// Tests many line-line intersections in parallel
kernel void test_intersections_2d(
    device const float4* lines_a [[buffer(0)]],    // Line A: (x1, y1, x2, y2)
    device const float4* lines_b [[buffer(1)]],    // Line B: (x1, y1, x2, y2)
    device float4* results [[buffer(2)]],          // (t_a, t_b, x, y) or (-1,-1,0,0) if parallel
    uint gid [[thread_position_in_grid]]
) {
    float4 a = lines_a[gid];
    float4 b = lines_b[gid];
    
    // Line A: P = A1 + t*(A2-A1)
    // Line B: Q = B1 + s*(B2-B1)
    float ax = a.z - a.x;  // A direction
    float ay = a.w - a.y;
    float bx = b.z - b.x;  // B direction
    float by = b.w - b.y;
    
    float denom = ax * by - ay * bx;
    
    if (abs(denom) < 1e-10) {
        // Lines are parallel
        results[gid] = float4(-1.0, -1.0, 0.0, 0.0);
        return;
    }
    
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    
    float t = (dx * by - dy * bx) / denom;
    float s = (dx * ay - dy * ax) / denom;
    
    // Intersection point
    float ix = a.x + t * ax;
    float iy = a.y + t * ay;
    
    results[gid] = float4(t, s, ix, iy);
}
"""


class MetalGeometryAccelerator:
    """
    GPU-accelerated geometry computation using Metal.
    
    Provides batch operations for:
    - Matrix transforms (point transformations)
    - Face point computation (ts coords → world points)
    - Intersection testing (2D line-line)
    """
    
    def __init__(self):
        """Initialize Metal device and compile shaders."""
        if not _METAL_AVAILABLE:
            raise RuntimeError("Metal is not available. Install pyobjc-framework-Metal.")
        
        self.device = _METAL_DEVICE
        self.queue = self.device.newCommandQueue()
        
        # Compile shader library
        options = Metal.MTLCompileOptions.alloc().init()
        error = objc.nil
        self.library, error = self.device.newLibraryWithSource_options_error_(
            METAL_SHADER_SOURCE, options, None
        )
        if error is not None or self.library is None:
            raise RuntimeError(f"Failed to compile Metal shaders: {error}")
        
        # Get kernel functions
        self.transform_points_fn = self.library.newFunctionWithName_("transform_points_batch")
        self.compute_face_points_fn = self.library.newFunctionWithName_("compute_face_points")
        self.test_intersections_fn = self.library.newFunctionWithName_("test_intersections_2d")
        
        # Create compute pipelines
        self.transform_pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            self.transform_points_fn, None
        )
        self.face_points_pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            self.compute_face_points_fn, None
        )
        self.intersections_pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            self.test_intersections_fn, None
        )
        
        # Thread configuration
        self.max_threads = self.transform_pipeline.maxTotalThreadsPerThreadgroup()
    
    def _create_buffer(self, data: np.ndarray) -> 'Metal.MTLBuffer':
        """Create a Metal buffer from numpy array."""
        return self.device.newBufferWithBytes_length_options_(
            data.tobytes(),
            data.nbytes,
            Metal.MTLResourceStorageModeShared
        )
    
    def _create_empty_buffer(self, size: int) -> 'Metal.MTLBuffer':
        """Create an empty Metal buffer."""
        return self.device.newBufferWithLength_options_(
            size,
            Metal.MTLResourceStorageModeShared
        )
    
    def transform_points_batch(
        self,
        matrices: np.ndarray,  # Shape: (N, 4, 4) float32
        local_points: np.ndarray  # Shape: (N, M, 4) float32 (M points per matrix)
    ) -> np.ndarray:
        """
        Transform batches of points by corresponding matrices.
        
        Args:
            matrices: N transformation matrices, shape (N, 4, 4) 
            local_points: M local points per matrix, shape (N, M, 4)
            
        Returns:
            World-space points, shape (N, M, 4)
        """
        n_matrices = matrices.shape[0]
        n_points_per = local_points.shape[1]
        total_points = n_matrices * n_points_per
        
        # Flatten inputs for GPU
        matrices_flat = matrices.astype(np.float32).reshape(-1)
        points_flat = local_points.astype(np.float32).reshape(-1)
        points_per = np.array([n_points_per], dtype=np.uint32)
        
        # Create buffers
        matrices_buf = self._create_buffer(matrices_flat)
        points_buf = self._create_buffer(points_flat)
        points_per_buf = self._create_buffer(points_per)
        output_buf = self._create_empty_buffer(total_points * 4 * 4)  # 4 floats per point
        
        # Encode and execute
        cmd_buffer = self.queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        
        encoder.setComputePipelineState_(self.transform_pipeline)
        encoder.setBuffer_offset_atIndex_(matrices_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(points_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(points_per_buf, 0, 3)
        
        # Dispatch
        grid_size = Metal.MTLSizeMake(total_points, 1, 1)
        threadgroup_size = Metal.MTLSizeMake(min(total_points, self.max_threads), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        
        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()
        
        # Read results
        result_ptr = output_buf.contents()
        result_bytes = result_ptr.as_buffer(total_points * 4 * 4)
        result = np.frombuffer(result_bytes, dtype=np.float32).reshape(n_matrices, n_points_per, 4)
        
        return result
    
    def compute_face_points_batch(
        self,
        matrices: np.ndarray,        # Shape: (N,) of 4x4 matrices
        half_thicknesses: np.ndarray,  # Shape: (N,) float32
        ts_coords: np.ndarray,        # Shape: (N, 2) float32 (t, s)
        face_sides: np.ndarray        # Shape: (N,) uint32 (face side enum)
    ) -> np.ndarray:
        """
        Compute world-space face points from (t, s) coordinates.
        
        Combines local point computation + transform in one GPU pass.
        
        Args:
            matrices: Transform matrices for each point's panel
            half_thicknesses: Half-thickness of each panel
            ts_coords: (t, s) coordinates for each point
            face_sides: Face side enum (0=FRONT, 1=BACK, etc.)
            
        Returns:
            World-space points, shape (N, 4)
        """
        n_points = len(ts_coords)
        
        # Prepare inputs
        matrices_flat = matrices.astype(np.float32).reshape(-1)
        half_t = half_thicknesses.astype(np.float32)
        ts = ts_coords.astype(np.float32)
        sides = face_sides.astype(np.uint32)
        
        # Create buffers
        matrices_buf = self._create_buffer(matrices_flat)
        half_t_buf = self._create_buffer(half_t)
        ts_buf = self._create_buffer(ts.reshape(-1))
        sides_buf = self._create_buffer(sides)
        output_buf = self._create_empty_buffer(n_points * 4 * 4)
        
        # Execute kernel
        cmd_buffer = self.queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        
        encoder.setComputePipelineState_(self.face_points_pipeline)
        encoder.setBuffer_offset_atIndex_(matrices_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(half_t_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(ts_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(sides_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 4)
        
        grid_size = Metal.MTLSizeMake(n_points, 1, 1)
        threadgroup_size = Metal.MTLSizeMake(min(n_points, self.max_threads), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        
        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()
        
        # Read results
        result_ptr = output_buf.contents()
        result_bytes = result_ptr.as_buffer(n_points * 4 * 4)
        result = np.frombuffer(result_bytes, dtype=np.float32).reshape(n_points, 4)
        
        return result
    
    def test_intersections_2d(
        self,
        lines_a: np.ndarray,  # Shape: (N, 4) - (x1, y1, x2, y2)
        lines_b: np.ndarray   # Shape: (N, 4) - (x1, y1, x2, y2)
    ) -> np.ndarray:
        """
        Test N pairs of 2D lines for intersection.
        
        Returns (t_a, t_b, ix, iy) for each pair.
        t_a, t_b are parametric values (0-1 = on segment).
        Returns (-1, -1, 0, 0) for parallel lines.
        """
        n_tests = len(lines_a)
        
        lines_a_flat = lines_a.astype(np.float32).reshape(-1)
        lines_b_flat = lines_b.astype(np.float32).reshape(-1)
        
        a_buf = self._create_buffer(lines_a_flat)
        b_buf = self._create_buffer(lines_b_flat)
        output_buf = self._create_empty_buffer(n_tests * 4 * 4)
        
        cmd_buffer = self.queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        
        encoder.setComputePipelineState_(self.intersections_pipeline)
        encoder.setBuffer_offset_atIndex_(a_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(b_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 2)
        
        grid_size = Metal.MTLSizeMake(n_tests, 1, 1)
        threadgroup_size = Metal.MTLSizeMake(min(n_tests, self.max_threads), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        
        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()
        
        result_ptr = output_buf.contents()
        result_bytes = result_ptr.as_buffer(n_tests * 4 * 4)
        result = np.frombuffer(result_bytes, dtype=np.float32).reshape(n_tests, 4)
        
        return result


# Fallback CPU implementation for non-Metal systems
class CPUGeometryAccelerator:
    """CPU fallback using NumPy vectorized operations."""
    
    def transform_points_batch(
        self,
        matrices: np.ndarray,
        local_points: np.ndarray
    ) -> np.ndarray:
        """Batch transform using NumPy matmul."""
        # matrices: (N, 4, 4), local_points: (N, M, 4)
        # Result: (N, M, 4)
        result = np.einsum('nij,nmj->nmi', matrices, local_points)
        return result
    
    def compute_face_points_batch(
        self,
        matrices: np.ndarray,
        half_thicknesses: np.ndarray,
        ts_coords: np.ndarray,
        face_sides: np.ndarray
    ) -> np.ndarray:
        """Compute face points using vectorized NumPy."""
        n_points = len(ts_coords)
        local = np.zeros((n_points, 4), dtype=np.float32)
        local[:, 3] = 1.0  # w = 1
        
        t = ts_coords[:, 0]
        s = ts_coords[:, 1]
        ht = half_thicknesses
        
        # FRONT (0)
        mask = face_sides == 0
        local[mask, 0] = t[mask]
        local[mask, 1] = s[mask]
        local[mask, 2] = ht[mask]
        
        # BACK (1)
        mask = face_sides == 1
        local[mask, 0] = t[mask]
        local[mask, 1] = s[mask]
        local[mask, 2] = -ht[mask]
        
        # TOP (2)
        mask = face_sides == 2
        local[mask, 0] = t[mask]
        local[mask, 1] = 1.0
        local[mask, 2] = -ht[mask] + s[mask] * 2 * ht[mask]
        
        # BOTTOM (3)
        mask = face_sides == 3
        local[mask, 0] = t[mask]
        local[mask, 1] = 0.0
        local[mask, 2] = -ht[mask] + s[mask] * 2 * ht[mask]
        
        # START (4)
        mask = face_sides == 4
        local[mask, 0] = 0.0
        local[mask, 1] = s[mask]
        local[mask, 2] = -ht[mask] + t[mask] * 2 * ht[mask]
        
        # END (5)
        mask = face_sides == 5
        local[mask, 0] = 1.0
        local[mask, 1] = s[mask]
        local[mask, 2] = -ht[mask] + t[mask] * 2 * ht[mask]
        
        # Transform
        result = np.einsum('nij,nj->ni', matrices, local)
        return result
    
    def test_intersections_2d(
        self,
        lines_a: np.ndarray,
        lines_b: np.ndarray
    ) -> np.ndarray:
        """Vectorized 2D line intersection test."""
        # Extract line components
        ax1, ay1, ax2, ay2 = lines_a[:, 0], lines_a[:, 1], lines_a[:, 2], lines_a[:, 3]
        bx1, by1, bx2, by2 = lines_b[:, 0], lines_b[:, 1], lines_b[:, 2], lines_b[:, 3]
        
        # Direction vectors
        adx = ax2 - ax1
        ady = ay2 - ay1
        bdx = bx2 - bx1
        bdy = by2 - by1
        
        # Determinant
        denom = adx * bdy - ady * bdx
        
        # Delta from A start to B start
        dx = bx1 - ax1
        dy = by1 - ay1
        
        # Parametric values
        with np.errstate(divide='ignore', invalid='ignore'):
            t_a = (dx * bdy - dy * bdx) / denom
            t_b = (dx * ady - dy * adx) / denom
        
        # Intersection points
        ix = ax1 + t_a * adx
        iy = ay1 + t_a * ady
        
        # Mark parallel lines
        parallel = np.abs(denom) < 1e-10
        t_a[parallel] = -1.0
        t_b[parallel] = -1.0
        ix[parallel] = 0.0
        iy[parallel] = 0.0
        
        return np.column_stack([t_a, t_b, ix, iy]).astype(np.float32)


def get_accelerator() -> 'MetalGeometryAccelerator | CPUGeometryAccelerator':
    """Get the best available accelerator (Metal if available, else CPU)."""
    if is_metal_available():
        return MetalGeometryAccelerator()
    return CPUGeometryAccelerator()
