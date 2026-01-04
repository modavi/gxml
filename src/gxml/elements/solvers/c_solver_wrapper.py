"""
Wrapper classes for C extension solvers.

Provides Python-compatible API for the C extension solvers, allowing them to be
used interchangeably with the CPU (Python) solvers.
"""

import numpy as np
from typing import List, Tuple, Optional

# Try to import C extension
try:
    from ._c_solvers import (
        solve_all,
        batch_find_intersections,
    )
    _C_EXTENSION_AVAILABLE = True
except ImportError:
    _C_EXTENSION_AVAILABLE = False


def is_c_extension_available() -> bool:
    """Check if the C extension is available."""
    return _C_EXTENSION_AVAILABLE


def solve_all_c(panels) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Run the full C solver pipeline on a list of panels.
    
    Args:
        panels: List of panel objects with get_centerline_endpoints(), thickness, height
        
    Returns:
        Tuple of (vertices, indices, intersection_count)
    """
    if not _C_EXTENSION_AVAILABLE:
        raise RuntimeError("C extension not available")
    
    n = len(panels)
    starts = np.zeros((n, 3), dtype=np.float64)
    ends = np.zeros((n, 3), dtype=np.float64)
    thicknesses = np.zeros(n, dtype=np.float64)
    heights = np.zeros(n, dtype=np.float64)
    
    for i, panel in enumerate(panels):
        start, end = panel.get_centerline_endpoints()
        starts[i] = start
        ends[i] = end
        thicknesses[i] = panel.thickness
        heights[i] = getattr(panel, 'height', 1.0)
    
    return solve_all(starts, ends, thicknesses, heights)


class CIntersectionSolver:
    """
    C extension intersection solver with same API as CPUIntersectionSolver.
    
    Uses C extension for fast intersection finding, then CPU for region building.
    The C extension is ~1800x faster for the O(n²) intersection finding.
    """
    
    @staticmethod
    def solve(panels):
        """
        Find intersections between panels using C extension.
        
        Uses C extension for fast batch intersection finding, then
        CPU for building proper IntersectionSolution with regions.
        """
        if not _C_EXTENSION_AVAILABLE:
            raise RuntimeError("C extension not available")
        
        # Use CPU solver which now internally uses C extension when available
        # The performance gain comes from _fast_batch_intersections which
        # uses batch_find_intersections from the C extension
        from .gxml_intersection_solver import IntersectionSolver as CPUIntersectionSolver
        return CPUIntersectionSolver.solve(panels)


class CFaceSolver:
    """
    C extension face solver with same API as CPUFaceSolver.
    
    Note: Face solving is integrated into solve_all_c() for performance.
    This class exists for API compatibility.
    """
    
    @staticmethod
    def solve(intersection_solution):
        """
        Solve faces from intersection solution.
        
        For full pipeline performance, use solve_all_c() directly instead.
        """
        if not _C_EXTENSION_AVAILABLE:
            raise RuntimeError("C extension not available")
        
        # The C pipeline handles face solving internally
        # For standalone use, fall back to CPU solver
        from .gxml_face_solver import FaceSolver
        return FaceSolver.solve(intersection_solution)


class CGeometryBuilder:
    """
    C extension geometry builder with same API as CPUGeometryBuilder.
    
    Currently delegates to CPU implementation since geometry building
    is I/O bound (creating polygon objects) rather than compute bound.
    """
    
    @staticmethod
    def build_all(panel_faces, intersection_solution):
        """
        Build geometry from face solution.
        
        Delegates to CPU GeometryBuilder since geometry building is I/O bound.
        """
        from .gxml_geometry_builder import GeometryBuilder as CPUGeometryBuilder
        return CPUGeometryBuilder.build_all(panel_faces, intersection_solution)
    
    @staticmethod
    def build(panel, panel_faces, intersection_solution):
        """Build geometry for a single panel."""
        from .gxml_geometry_builder import GeometryBuilder as CPUGeometryBuilder
        return CPUGeometryBuilder.build(panel, panel_faces, intersection_solution)
