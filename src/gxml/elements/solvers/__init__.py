"""
Solvers for panel geometry and intersections.

Pipeline:
1. IntersectionSolver - finds where panel centerlines meet
2. FaceSolver - determines face segmentation AND computes trim/gap adjustments
3. GeometryBuilder - creates actual 3D geometry (polygons and caps)
"""

from .gxml_intersection_solver import (
    IntersectionSolver,
    IntersectionSolution,
    IntersectionType,
    Intersection,
    PanelEndpoint,
    PanelAxis,
)

from .gxml_face_solver import (
    FaceSolver,
    SegmentedPanel,
    FaceSegment,
    JointSide,
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
    # Stage 2: Face Solving (combined face segmentation + bounds)
    'FaceSolver',
    'SegmentedPanel',
    'FaceSegment',
    # Stage 3: Geometry Building
    'GeometryBuilder',
]
