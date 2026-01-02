"""
GPU-accelerated geometry builder integration for GXML.

This module provides a drop-in replacement for GeometryBuilder.build_all()
that uses GPU acceleration for the transform-heavy operations.

Architecture:
    1. Collect all face segments across all panels
    2. Batch the (t, s) → world point transformations on GPU
    3. Distribute results back to face segments

This batching approach amortizes GPU overhead and maximizes parallelism.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src')

from gxml.gpu.metal_geometry import CPUGeometryAccelerator, get_accelerator


@dataclass
class BatchedTransformRequest:
    """A batch of transform requests ready for GPU processing."""
    # Input arrays
    matrices: np.ndarray       # (N, 4, 4) panel transform matrices
    half_thicknesses: np.ndarray  # (N,) panel half-thickness values
    ts_coords: np.ndarray      # (N, 2) (t, s) coordinates
    face_sides: np.ndarray     # (N,) face side enum values
    
    # Mapping back to segments
    segment_indices: List[int]  # Which segment each point belongs to
    corner_indices: List[int]   # Which corner (0-3) within segment


class GPUGeometryBuilder:
    """
    GPU-accelerated geometry builder.
    
    Collects all face segment corner computations and processes them
    in a single GPU batch, then distributes results back.
    """
    
    # Face side enum values (must match PanelSide in gxml_panel.py)
    FACE_FRONT = 0
    FACE_BACK = 1
    FACE_TOP = 2
    FACE_BOTTOM = 3
    FACE_START = 4
    FACE_END = 5
    
    def __init__(self):
        """Initialize with best available accelerator."""
        self.accel = get_accelerator()
        self._is_gpu = hasattr(self.accel, 'device')  # Metal has device attr
    
    def build_all_gpu(self, panel_faces: List, intersection_solution) -> None:
        """
        Build geometry for all panels using GPU acceleration.
        
        Args:
            panel_faces: List of SegmentedPanel from FaceSolver
            intersection_solution: IntersectionSolution for cap creation
        """
        # Step 1: Collect all corner computations
        requests = self._collect_transform_requests(panel_faces)
        
        if requests.matrices.shape[0] == 0:
            return
        
        # Step 2: Batch process on GPU
        world_points = self.accel.compute_face_points_batch(
            requests.matrices,
            requests.half_thicknesses,
            requests.ts_coords,
            requests.face_sides
        )
        
        # Step 3: Distribute results back to segments
        self._distribute_results(panel_faces, requests, world_points)
        
        # Step 4: Create polygons and caps (still on CPU for now)
        self._create_polygons(panel_faces)
    
    def _collect_transform_requests(self, panel_faces: List) -> BatchedTransformRequest:
        """
        Collect all corner transform requests from all segments.
        
        Returns a BatchedTransformRequest with arrays ready for GPU.
        """
        # Count total corners
        total_corners = 0
        for pf in panel_faces:
            for face_side, segments in pf.segments.items():
                total_corners += len(segments) * 4  # 4 corners per segment
        
        if total_corners == 0:
            return BatchedTransformRequest(
                matrices=np.zeros((0, 4, 4), dtype=np.float32),
                half_thicknesses=np.zeros(0, dtype=np.float32),
                ts_coords=np.zeros((0, 2), dtype=np.float32),
                face_sides=np.zeros(0, dtype=np.uint32),
                segment_indices=[],
                corner_indices=[]
            )
        
        # Pre-allocate arrays
        matrices = np.zeros((total_corners, 4, 4), dtype=np.float32)
        half_thicknesses = np.zeros(total_corners, dtype=np.float32)
        ts_coords = np.zeros((total_corners, 2), dtype=np.float32)
        face_sides = np.zeros(total_corners, dtype=np.uint32)
        segment_indices = []
        corner_indices = []
        
        # Fill arrays
        idx = 0
        seg_idx = 0
        
        for pf in panel_faces:
            panel = pf.panel
            
            # Get panel's world transform matrix
            world_mat = self._get_panel_matrix(panel)
            half_t = panel.thickness / 2.0
            
            for face_side, segments in pf.segments.items():
                face_enum = self._face_to_enum(face_side)
                
                for segment in segments:
                    for corner_idx, (t, s) in enumerate(segment.corners):
                        matrices[idx] = world_mat
                        half_thicknesses[idx] = half_t
                        ts_coords[idx] = [t, s]
                        face_sides[idx] = face_enum
                        segment_indices.append(seg_idx)
                        corner_indices.append(corner_idx)
                        idx += 1
                    
                    seg_idx += 1
        
        return BatchedTransformRequest(
            matrices=matrices,
            half_thicknesses=half_thicknesses,
            ts_coords=ts_coords,
            face_sides=face_sides,
            segment_indices=segment_indices,
            corner_indices=corner_indices
        )
    
    def _get_panel_matrix(self, panel) -> np.ndarray:
        """Extract 4x4 world transform matrix from panel."""
        # Get the panel's transform matrix
        if hasattr(panel, '_transform') and hasattr(panel._transform, '_world_matrix'):
            mat = panel._transform._world_matrix
            # Convert to numpy array if needed
            if hasattr(mat, 'data'):
                # Mat4 C type
                return np.array(mat.data, dtype=np.float32).reshape(4, 4)
            elif isinstance(mat, (list, tuple)):
                return np.array(mat, dtype=np.float32).reshape(4, 4)
            else:
                return np.array(mat, dtype=np.float32)
        
        # Fallback: identity
        return np.eye(4, dtype=np.float32)
    
    def _face_to_enum(self, face_side) -> int:
        """Convert PanelSide to integer enum."""
        name = face_side.name if hasattr(face_side, 'name') else str(face_side)
        mapping = {
            'FRONT': self.FACE_FRONT,
            'BACK': self.FACE_BACK,
            'TOP': self.FACE_TOP,
            'BOTTOM': self.FACE_BOTTOM,
            'START': self.FACE_START,
            'END': self.FACE_END,
        }
        return mapping.get(name, 0)
    
    def _distribute_results(self, panel_faces: List, requests: BatchedTransformRequest,
                           world_points: np.ndarray) -> None:
        """
        Distribute GPU results back to face segments.
        
        Updates each segment's cached world corners with the GPU-computed values.
        """
        # Build segment list in same order as collection
        segments = []
        for pf in panel_faces:
            for face_side, segs in pf.segments.items():
                segments.extend(segs)
        
        # Distribute results
        for i, (seg_idx, corner_idx) in enumerate(
            zip(requests.segment_indices, requests.corner_indices)
        ):
            segment = segments[seg_idx]
            
            # Ensure cached corners list exists
            if not segment._cached_world_corners:
                segment._cached_world_corners = [None, None, None, None]
            
            # Store world point (xyz, drop w)
            point = tuple(world_points[i, :3])
            segment._cached_world_corners[corner_idx] = point
    
    def _create_polygons(self, panel_faces: List) -> None:
        """Create polygon geometry from cached world corners."""
        for pf in panel_faces:
            panel = pf.panel
            
            for face_side, segments in pf.segments.items():
                for i, segment in enumerate(segments):
                    face_name = (face_side.name.lower() if len(segments) == 1 
                                else f"{face_side.name.lower()}-{i}")
                    
                    # Create polygon from cached world corners
                    if segment._cached_world_corners:
                        panel.create_panel_side(
                            face_name, 
                            face_side, 
                            corners=segment.corners
                        )


def demo_gpu_integration():
    """Demo showing GPU integration with GXML pipeline."""
    print("=" * 70)
    print("GPU Geometry Builder Integration Demo")
    print("=" * 70)
    print()
    
    # Check accelerator
    accel = get_accelerator()
    is_gpu = hasattr(accel, 'device')
    print(f"Using accelerator: {'Metal GPU' if is_gpu else 'CPU (NumPy vectorized)'}")
    print()
    
    # Simulate data for 75 panels
    n_panels = 75
    segments_per_panel = 15
    corners_per_segment = 4
    total_corners = n_panels * segments_per_panel * corners_per_segment
    
    print(f"Simulating {n_panels} panels:")
    print(f"  - {segments_per_panel} segments per panel")
    print(f"  - {corners_per_segment} corners per segment")
    print(f"  - {total_corners} total corner transforms")
    print()
    
    # Generate test data
    np.random.seed(42)
    matrices = np.random.randn(total_corners, 4, 4).astype(np.float32)
    # Make them valid orthogonal transforms
    for i in range(total_corners):
        matrices[i] = np.eye(4) + 0.1 * matrices[i]
    
    half_thicknesses = np.random.uniform(0.01, 0.1, total_corners).astype(np.float32)
    ts_coords = np.random.uniform(0, 1, (total_corners, 2)).astype(np.float32)
    face_sides = np.random.randint(0, 6, total_corners).astype(np.uint32)
    
    # Benchmark
    import time
    
    n_iters = 100
    times = []
    
    for _ in range(5):  # Warmup
        accel.compute_face_points_batch(matrices, half_thicknesses, ts_coords, face_sides)
    
    for _ in range(n_iters):
        start = time.perf_counter()
        result = accel.compute_face_points_batch(matrices, half_thicknesses, ts_coords, face_sides)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    
    print(f"Batch transform time: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"Throughput: {total_corners / avg_time * 1000:.0f} transforms/sec")
    print()
    
    # Compare to per-call overhead estimate
    print("Comparison to current per-call approach:")
    print(f"  - Current: ~{total_corners} individual transform_point() calls")
    print(f"  - GPU batch: 1 call processing all {total_corners} transforms")
    print()


if __name__ == "__main__":
    demo_gpu_integration()
