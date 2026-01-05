"""
Solvers for panel geometry and intersections.

Pipeline:
1. IntersectionSolver - finds where panel centerlines meet
2. FaceSolver - determines face segmentation AND computes trim/gap adjustments
3. GeometryBuilder - creates actual 3D geometry (polygons and caps)

Usage:
    from elements.solvers import IntersectionSolver, FaceSolver, GeometryBuilder
    
    # For backend selection, use the backend module:
    from backend import set_solver_backend
    set_solver_backend('c')  # Use C extension pipeline
"""

# Stage 1: Intersection Solving
from .gxml_intersection_solver import (
    IntersectionSolver as CPUIntersectionSolver,
    IntersectionSolution,
    IntersectionType,
    Intersection,
    PanelEndpoint,
    PanelAxis,
)

# Stage 2: Face Solving
from .gxml_face_solver import (
    FaceSolver as CPUFaceSolver,
    SegmentedPanel,
    FaceSegment,
    JointSide,
)

# Stage 3: Geometry Building
from .gxml_geometry_builder import (
    GeometryBuilder as CPUGeometryBuilder,
)

from .gxml_gpu_geometry_builder import GPUGeometryBuilder

# C extension solvers
from .c_solver_wrapper import (
    CIntersectionSolver,
    CFaceSolver,
    CGeometryBuilder,
    solve_all_c,
)

# Default class aliases (for backwards compatibility)
IntersectionSolver = CPUIntersectionSolver
FaceSolver = CPUFaceSolver
GeometryBuilder = CPUGeometryBuilder

__all__ = [
    # Stage 1: Intersection Solving
    'IntersectionSolver',
    'CPUIntersectionSolver',
    'CIntersectionSolver',
    'IntersectionSolution',
    'IntersectionType',
    'Intersection',
    'PanelEndpoint',
    'PanelAxis',
    # Stage 2: Face Solving
    'FaceSolver',
    'CPUFaceSolver',
    'CFaceSolver',
    'SegmentedPanel',
    'FaceSegment',
    # Stage 3: Geometry Building
    'GeometryBuilder',
    'CPUGeometryBuilder',
    'CGeometryBuilder',
    'GPUGeometryBuilder',
    # C extension utilities
    'solve_all_c',
]
