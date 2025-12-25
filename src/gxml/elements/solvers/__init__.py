"""
Solvers for panel geometry and intersections.

Pipeline:
1. IntersectionSolver - finds where panel centerlines meet
2. FaceSolver - determines face segmentation from intersection topology
3. BoundsSolver - computes trim/gap adjustments, provides coordinate lookup
4. GeometryBuilder - creates actual 3D geometry (polygons and caps)
"""

from .gxml_intersection_solver import (
    IntersectionSolver,
    IntersectionSolution,
    IntersectionType,
    Intersection,
    PanelEndpoint,
    Axis,
)

from .gxml_face_solver import (
    FaceSolver,
    FaceSolverResult,
    FaceSolution,
)

from .gxml_bounds_solver import (
    BoundsSolver,
    BoundsSolution,
)

from .gxml_geometry_builder import (
    GeometryBuilder,
)

__all__ = [
    # Stage 1: Intersection Solving
    'IntersectionSolver',
    'IntersectionSolution',
    'IntersectionType',
    'Intersection',
    'PanelEndpoint',
    'Axis',
    # Stage 2: Face Solving
    'FaceSolver',
    'FaceSolverResult',
    'FaceSolution',
    # Stage 3: Bounds Solving
    'BoundsSolver',
    'BoundsSolution',
    # Stage 4: Geometry Building
    'GeometryBuilder',
]
