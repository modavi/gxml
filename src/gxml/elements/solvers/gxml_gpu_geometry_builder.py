"""
GPU-accelerated geometry builder for GXML.

This module provides a drop-in replacement for GeometryBuilder that uses
GPU acceleration for the transform-heavy operations while maintaining
full compatibility with the existing pipeline.

Usage:
    # Enable GPU acceleration globally
    from gxml.elements.solvers import set_geometry_backend
    set_geometry_backend('gpu')  # or 'cpu' to disable
    
    # Or use directly
    from gxml.elements.solvers.gxml_gpu_geometry_builder import GPUGeometryBuilder
    GPUGeometryBuilder.build_all(panel_faces, intersection_solution)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from elements.solvers.gxml_geometry_builder import GeometryBuilder
from elements.solvers.gxml_face_solver import FaceSegment, SegmentedPanel
from elements.solvers.gxml_intersection_solver import IntersectionSolution
from elements.gxml_panel import GXMLPanel, PanelSide

# Import GPU accelerator using shader_backend (cross-platform)
_SHADER_BACKEND = None

def _get_accelerator():
    """Get or create the GPU shader backend (lazy singleton)."""
    global _SHADER_BACKEND
    if _SHADER_BACKEND is None:
        try:
            from .gpu.shader_backend import get_shader_backend
            _SHADER_BACKEND = get_shader_backend()
        except ImportError:
            pass
    return _SHADER_BACKEND


def is_gpu_available():
    """Check if GPU acceleration is available."""
    backend = _get_accelerator()
    return backend is not None and backend.is_available


# Face side enum mapping (must match PanelSide)
FACE_ENUM = {
    PanelSide.FRONT: 0,
    PanelSide.BACK: 1,
    PanelSide.TOP: 2,
    PanelSide.BOTTOM: 3,
    PanelSide.START: 4,
    PanelSide.END: 5,
}


@dataclass
class BatchedCornerData:
    """Batched corner data for GPU processing."""
    # Input arrays for GPU
    matrices: np.ndarray           # (N, 4, 4) panel transforms
    half_thicknesses: np.ndarray   # (N,) half-thickness values
    ts_coords: np.ndarray          # (N, 2) (t, s) coordinates  
    face_sides: np.ndarray         # (N,) face side enum values
    
    # Mapping back to segments
    segment_refs: List[Tuple[FaceSegment, int]]  # (segment, corner_index) pairs
    
    @property
    def count(self) -> int:
        return len(self.ts_coords)


class GPUGeometryBuilder:
    """
    GPU-accelerated geometry builder.
    
    Provides the same API as GeometryBuilder but batches all corner
    transformations for GPU processing.
    """
    
    # -------------------------------------------------------------------------
    # Public API (matches GeometryBuilder)
    # -------------------------------------------------------------------------
    
    @staticmethod
    def build_all(panel_faces: List[SegmentedPanel],
                  intersection_solution: IntersectionSolution) -> None:
        """
        Build geometry for all panels using GPU acceleration.
        
        Args:
            panel_faces: List of SegmentedPanel from FaceSolver
            intersection_solution: The intersection solution
        """
        accel = _get_accelerator()
        if accel is None:
            # Fallback to CPU
            GeometryBuilder.build_all(panel_faces, intersection_solution)
            return
        
        # Step 1: Batch compute all world corners on GPU
        GPUGeometryBuilder._batch_compute_world_corners(panel_faces, accel)
        
        # Step 2: Create face polygons (now using cached corners)
        for pf in panel_faces:
            if pf.panel.is_valid(1e-4):
                GPUGeometryBuilder._create_panel_faces(pf.panel, pf)
        
        # Step 3: Create caps (reuses cached corners)
        GeometryBuilder._create_all_caps(panel_faces, intersection_solution)
    
    @staticmethod
    def build(panel: GXMLPanel, panel_faces: List[SegmentedPanel],
              intersection_solution: IntersectionSolution) -> None:
        """
        Build geometry for a single panel using GPU acceleration.
        
        For single-panel builds, the overhead of GPU batching may not be worth it,
        so we only GPU-accelerate if there are many segments.
        """
        pf = GPUGeometryBuilder._find_panel_faces(panel_faces, panel)
        if pf is None or not panel.is_valid(1e-4):
            return
        
        # Count corners for this panel
        corner_count = sum(
            len(segs) * 4 
            for segs in pf.segments.values()
        )
        
        accel = _get_accelerator()
        
        # Use GPU if we have enough work (threshold: 16 corners)
        if accel is not None and corner_count >= 16:
            GPUGeometryBuilder._batch_compute_world_corners([pf], accel)
            GPUGeometryBuilder._create_panel_faces(panel, pf)
        else:
            # Use standard CPU path for small panels
            GeometryBuilder._create_panel_faces(panel, pf)
        
        # Caps still use standard path
        GeometryBuilder._create_caps_for_panel(panel, panel_faces, intersection_solution)
    
    # -------------------------------------------------------------------------
    # GPU batching
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _batch_compute_world_corners(panel_faces: List[SegmentedPanel], accel) -> None:
        """
        Batch compute all world corners using GPU accelerator.
        
        Collects all (t, s) → world point computations and processes them
        in a single GPU call, then distributes results back to segments.
        """
        # Collect all corner data
        batch = GPUGeometryBuilder._collect_corner_data(panel_faces)
        
        if batch.count == 0:
            return
        
        # GPU compute - use shader_backend interface
        # Split ts_coords into separate t and s arrays
        t_coords = batch.ts_coords[:, 0]
        s_coords = batch.ts_coords[:, 1]
        
        world_points = accel.compute_face_points(
            batch.matrices,
            batch.half_thicknesses,
            t_coords,
            s_coords,
            batch.face_sides
        )
        
        # Distribute results back to segments
        GPUGeometryBuilder._distribute_world_corners(batch, world_points)
    
    @staticmethod
    def _collect_corner_data(panel_faces: List[SegmentedPanel]) -> BatchedCornerData:
        """
        Collect all corner data from all segments into GPU-ready arrays.
        """
        # First pass: count total corners
        total_corners = 0
        for pf in panel_faces:
            for segments in pf.segments.values():
                total_corners += len(segments) * 4
        
        if total_corners == 0:
            return BatchedCornerData(
                matrices=np.zeros((0, 4, 4), dtype=np.float32),
                half_thicknesses=np.zeros(0, dtype=np.float32),
                ts_coords=np.zeros((0, 2), dtype=np.float32),
                face_sides=np.zeros(0, dtype=np.uint32),
                segment_refs=[]
            )
        
        # Pre-allocate arrays
        matrices = np.zeros((total_corners, 4, 4), dtype=np.float32)
        half_thicknesses = np.zeros(total_corners, dtype=np.float32)
        ts_coords = np.zeros((total_corners, 2), dtype=np.float32)
        face_sides = np.zeros(total_corners, dtype=np.uint32)
        segment_refs = []
        
        # Second pass: fill arrays
        idx = 0
        for pf in panel_faces:
            panel = pf.panel
            
            # Get panel's world transform matrix
            world_mat = GPUGeometryBuilder._get_panel_matrix(panel)
            half_t = panel.thickness / 2.0
            
            for face_side, segments in pf.segments.items():
                face_enum = FACE_ENUM.get(face_side, 0)
                
                for segment in segments:
                    for corner_idx, (t, s) in enumerate(segment.corners):
                        matrices[idx] = world_mat
                        half_thicknesses[idx] = half_t
                        ts_coords[idx] = [t, s]
                        face_sides[idx] = face_enum
                        segment_refs.append((segment, corner_idx))
                        idx += 1
        
        return BatchedCornerData(
            matrices=matrices,
            half_thicknesses=half_thicknesses,
            ts_coords=ts_coords,
            face_sides=face_sides,
            segment_refs=segment_refs
        )
    
    @staticmethod
    def _get_panel_matrix(panel: GXMLPanel) -> np.ndarray:
        """Extract 4x4 world transform matrix from panel."""
        if hasattr(panel, '_transform') and hasattr(panel._transform, '_world_matrix'):
            mat = panel._transform._world_matrix
            # Handle Mat4 C type
            if hasattr(mat, 'data'):
                return np.array(mat.data, dtype=np.float32).reshape(4, 4)
            elif isinstance(mat, (list, tuple)):
                return np.array(mat, dtype=np.float32).reshape(4, 4)
            else:
                return np.array(mat, dtype=np.float32)
        
        # Fallback: identity
        return np.eye(4, dtype=np.float32)
    
    @staticmethod
    def _distribute_world_corners(batch: BatchedCornerData, world_points: np.ndarray) -> None:
        """
        Distribute GPU-computed world points back to face segments.
        """
        for i, (segment, corner_idx) in enumerate(batch.segment_refs):
            # Ensure cached corners list exists and is the right size
            if not segment._cached_world_corners or len(segment._cached_world_corners) != 4:
                segment._cached_world_corners = [None, None, None, None]
            
            # Store world point (xyz tuple, drop w component)
            segment._cached_world_corners[corner_idx] = tuple(world_points[i, :3])
    
    # -------------------------------------------------------------------------
    # Face creation (same as GeometryBuilder but uses cached corners)
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _create_panel_faces(panel: GXMLPanel, pf: SegmentedPanel) -> None:
        """
        Create face polygons from panel faces.
        
        Uses the GPU-computed cached world corners.
        """
        for face_side in PanelSide:
            segs = pf.segments.get(face_side, [])
            
            for i, segment in enumerate(segs):
                local_corners = segment.corners
                face_name = face_side.name.lower() if len(segs) == 1 else f"{face_side.name.lower()}-{i}"
                panel.create_panel_side(face_name, face_side, corners=local_corners)
    
    @staticmethod
    def _find_panel_faces(panel_faces: List[SegmentedPanel], panel: GXMLPanel) -> Optional[SegmentedPanel]:
        """Find the SegmentedPanel for a given panel."""
        return next((pf for pf in panel_faces if pf.panel is panel), None)


# -------------------------------------------------------------------------
# Backend switching
# -------------------------------------------------------------------------

# Current active backend - auto-detect best available
_GEOMETRY_BACKEND = None  # Will be set on first access


def _auto_detect_backend() -> str:
    """Auto-detect the best available geometry backend."""
    # Check if GPU is available and worthwhile
    backend = _get_accelerator()
    if backend is not None and backend.is_available:
        # GPU is available, but check if it's worth the overhead
        # For now, prefer CPU since C extension is very fast for transforms
        # GPU mainly helps for very large batches (1000+ corners)
        return 'cpu'  # Default to CPU, user can opt-in to GPU
    return 'cpu'


def set_geometry_backend(backend: str) -> None:
    """
    Set the geometry builder backend.
    
    Args:
        backend: 'gpu' for GPU acceleration, 'cpu' for standard CPU path, 'auto' for auto-detect
    """
    global _GEOMETRY_BACKEND
    if backend == 'auto':
        _GEOMETRY_BACKEND = _auto_detect_backend()
    elif backend not in ('gpu', 'cpu'):
        raise ValueError(f"Invalid backend: {backend}. Use 'gpu', 'cpu', or 'auto'.")
    else:
        _GEOMETRY_BACKEND = backend


def get_geometry_backend() -> str:
    """Get the current geometry builder backend."""
    global _GEOMETRY_BACKEND
    if _GEOMETRY_BACKEND is None:
        _GEOMETRY_BACKEND = _auto_detect_backend()
    return _GEOMETRY_BACKEND


def get_geometry_builder():
    """
    Get the appropriate geometry builder based on current backend setting.
    
    Returns:
        GeometryBuilder or GPUGeometryBuilder class
    """
    if _GEOMETRY_BACKEND == 'gpu':
        return GPUGeometryBuilder
    return GeometryBuilder
