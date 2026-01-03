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
    
    Note: This is a lightweight wrapper. The actual solving happens in solve_all_c()
    which runs the full pipeline. This class exists for API compatibility.
    """
    
    @staticmethod
    def solve(panels):
        """
        Find intersections between panels using C extension.
        
        For full pipeline performance, use solve_all_c() directly instead.
        """
        if not _C_EXTENSION_AVAILABLE:
            raise RuntimeError("C extension not available")
        
        # Import here to avoid circular imports
        from .gxml_intersection_solver import IntersectionSolution
        
        n = len(panels)
        starts = np.zeros((n, 3), dtype=np.float64)
        ends = np.zeros((n, 3), dtype=np.float64)
        
        for i, panel in enumerate(panels):
            start, end = panel.get_centerline_endpoints()
            starts[i] = start
            ends[i] = end
        
        # Use batch intersection finding
        i_arr, j_arr, t1_arr, t2_arr, pos_arr = batch_find_intersections(starts, ends)
        
        # Convert to IntersectionSolution format
        intersections = []
        for k in range(len(i_arr)):
            intersections.append({
                'panel_i': i_arr[k],
                'panel_j': j_arr[k],
                't1': t1_arr[k],
                't2': t2_arr[k],
                'position': pos_arr[k],
            })
        
        return IntersectionSolution(panels=panels, intersections=intersections)


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
    
    Note: Geometry building is integrated into solve_all_c() for performance.
    This class exists for API compatibility.
    """
    
    @staticmethod
    def build(face_solution):
        """
        Build geometry from face solution.
        
        For full pipeline performance, use solve_all_c() directly instead.
        """
        if not _C_EXTENSION_AVAILABLE:
            raise RuntimeError("C extension not available")
        
        # The C pipeline handles geometry building internally
        # For standalone use, fall back to CPU solver
        from .gxml_full_geometry_builder import FullGeometryBuilder
        return FullGeometryBuilder.build(face_solution)
