"""
Backend selection and availability for GXML.

This module centralizes all backend-related logic:
- Checking which backends are available (C solvers, C profiler, etc.)
- Setting and getting the current solver backend
- Returning the appropriate solver classes for the current backend

Usage:
    from backend import check_backends, set_solver_backend
    
    # Check what's available
    backends = check_backends()
    # {'cpu': True, 'c': True, 'c_profiler': True, 'taichi': False}
    
    # Set the backend
    set_solver_backend('c')
"""

from typing import Dict
import warnings


# =============================================================================
# Backend Availability
# =============================================================================

def check_backends() -> Dict[str, bool]:
    """Check which backends are available across all GXML subsystems.
    
    Returns:
        Dict with availability of each backend:
        - 'cpu': Always True (pure Python fallback)
        - 'c': True if C extensions are available (solvers, profiler, vec3)
        - 'taichi': Currently always False (disabled due to Vulkan issues)
    
    Example:
        >>> from gxml import check_backends
        >>> backends = check_backends()
        >>> if backends['c']:
        ...     print("C extensions available - using fast path")
    """
    availability = {
        'cpu': True,  # Always available
        'c': False,
        'taichi': False,  # Disabled - Vulkan overhead issues on Windows
    }
    
    # Check C extensions (all built together, so just check one)
    try:
        from elements.solvers.c_solver_wrapper import _C_EXTENSION_AVAILABLE
        availability['c'] = _C_EXTENSION_AVAILABLE
    except ImportError:
        pass
    
    return availability


# =============================================================================
# Solver Backend Selection
# =============================================================================

_solver_backend = 'cpu'


def set_solver_backend(backend: str) -> None:
    """
    Set the solver backend for all pipeline stages.
    
    Args:
        backend: 
            'cpu' - CPU solvers (default, works everywhere)
            'c' - C extension solvers (fast, requires compilation)
            'taichi' - Taichi GPU-accelerated solvers (requires Python 3.12, Taichi)
            'gpu' - Legacy GPU backend (only GeometryBuilder transforms)
    """
    global _solver_backend
    if backend not in ('cpu', 'c', 'taichi', 'gpu'):
        raise ValueError(f"Unknown backend: {backend}. Use 'cpu', 'c', 'taichi', or 'gpu'.")
    
    # Check C extension availability
    if backend == 'c' and not check_backends()['c']:
        warnings.warn("C extension not available, falling back to CPU")
        backend = 'cpu'
    
    _solver_backend = backend
    
    # Also set geometry backend for compatibility
    from elements.solvers.gxml_gpu_geometry_builder import set_geometry_backend
    if backend == 'gpu':
        set_geometry_backend('gpu')
    else:
        set_geometry_backend('cpu')


def get_solver_backend() -> str:
    """Get the current solver backend."""
    return _solver_backend


# =============================================================================
# Solver Class Getters
# =============================================================================

def get_intersection_solver():
    """Get the IntersectionSolver class for the current backend."""
    from elements.solvers.gxml_intersection_solver import IntersectionSolver as CPUIntersectionSolver
    
    if _solver_backend == 'c':
        if check_backends()['c']:
            from elements.solvers.c_solver_wrapper import CIntersectionSolver
            return CIntersectionSolver
        warnings.warn("C extension not available, falling back to CPU")
        return CPUIntersectionSolver
    
    if _solver_backend == 'taichi':
        try:
            from elements.solvers.taichi.taichi_intersection_solver import TaichiIntersectionSolver
            return TaichiIntersectionSolver
        except ImportError as e:
            warnings.warn(f"Taichi not available, falling back to CPU: {e}")
            return CPUIntersectionSolver
    
    return CPUIntersectionSolver


def get_face_solver():
    """Get the FaceSolver class for the current backend."""
    from elements.solvers.gxml_face_solver import FaceSolver as CPUFaceSolver
    
    if _solver_backend == 'c':
        if check_backends()['c']:
            from elements.solvers.c_solver_wrapper import CFaceSolver
            return CFaceSolver
        warnings.warn("C extension not available, falling back to CPU")
        return CPUFaceSolver
    
    if _solver_backend == 'taichi':
        try:
            from elements.solvers.taichi.taichi_face_solver import TaichiFaceSolver
            return TaichiFaceSolver
        except ImportError as e:
            warnings.warn(f"Taichi not available, falling back to CPU: {e}")
            return CPUFaceSolver
    
    return CPUFaceSolver


def get_full_geometry_builder():
    """Get the GeometryBuilder class for the current backend."""
    from elements.solvers.gxml_geometry_builder import GeometryBuilder as CPUGeometryBuilder
    
    if _solver_backend == 'c':
        if check_backends()['c']:
            from elements.solvers.c_solver_wrapper import CGeometryBuilder
            return CGeometryBuilder
        warnings.warn("C extension not available, falling back to CPU")
        return CPUGeometryBuilder
    
    if _solver_backend == 'taichi':
        try:
            from elements.solvers.taichi.taichi_geometry_builder import TaichiGeometryBuilder
            return TaichiGeometryBuilder
        except ImportError as e:
            warnings.warn(f"Taichi not available, falling back to CPU: {e}")
            return CPUGeometryBuilder
    
    return CPUGeometryBuilder
