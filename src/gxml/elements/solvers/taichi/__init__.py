"""
Taichi GPU-accelerated solvers for GXML.

This module provides GPU-accelerated implementations of the core solvers:
- IntersectionSolver: Finds where panel centerlines meet
- FaceSolver: Determines face segmentation with gap-adjusted bounds
- GeometryBuilder: Creates actual 3D geometry (polygons and caps)

The Taichi backend can be selected via:
    from gxml.elements.solvers import set_solver_backend
    set_solver_backend('gpu')  # Use Taichi GPU solvers
    set_solver_backend('cpu')  # Use CPU solvers (default)

Requires Python 3.12 or earlier (Taichi doesn't support 3.14+).
"""

# Backend selection
_backend = 'cpu'

def set_solver_backend(backend: str) -> None:
    """
    Set the solver backend.
    
    Args:
        backend: 'gpu' for Taichi GPU solvers, 'cpu' for CPU solvers
    """
    global _backend
    if backend not in ('gpu', 'cpu'):
        raise ValueError(f"Unknown backend: {backend}. Use 'gpu' or 'cpu'.")
    _backend = backend

def get_solver_backend() -> str:
    """Get the current solver backend."""
    return _backend

# Lazy imports to avoid loading Taichi unless needed
def get_intersection_solver():
    """Get the IntersectionSolver for the current backend."""
    if _backend == 'gpu':
        from .taichi_intersection_solver import TaichiIntersectionSolver
        return TaichiIntersectionSolver
    else:
        from ..gxml_intersection_solver import IntersectionSolver
        return IntersectionSolver

def get_face_solver():
    """Get the FaceSolver for the current backend."""
    if _backend == 'gpu':
        from .taichi_face_solver import TaichiFaceSolver
        return TaichiFaceSolver
    else:
        from ..gxml_face_solver import FaceSolver
        return FaceSolver

def get_geometry_builder():
    """Get the GeometryBuilder for the current backend."""
    if _backend == 'gpu':
        from .taichi_geometry_builder import TaichiGeometryBuilder
        return TaichiGeometryBuilder
    else:
        from ..gxml_geometry_builder import GeometryBuilder
        return GeometryBuilder
