"""
Solvers for panel geometry and intersections.

Pipeline:
1. IntersectionSolver - finds where panel centerlines meet
2. FaceSolver - determines face segmentation AND computes trim/gap adjustments
3. GeometryBuilder - creates actual 3D geometry (polygons and caps)

Backend Selection:
- 'cpu': Use CPU solvers (default, works everywhere)
- 'taichi': Use Taichi GPU-accelerated solvers (requires Python 3.12, Taichi)
- 'gpu': Legacy GPU backend (only GeometryBuilder transforms)

Usage:
    from gxml.elements.solvers import set_solver_backend
    set_solver_backend('taichi')  # Use full Taichi GPU pipeline
"""

from .gxml_intersection_solver import (
    IntersectionSolver as CPUIntersectionSolver,
    IntersectionSolution,
    IntersectionType,
    Intersection,
    PanelEndpoint,
    PanelAxis,
)

from .gxml_face_solver import (
    FaceSolver as CPUFaceSolver,
    SegmentedPanel,
    FaceSegment,
    JointSide,
)

from .gxml_geometry_builder import (
    GeometryBuilder as CPUGeometryBuilder,
)

from .gxml_gpu_geometry_builder import (
    GPUGeometryBuilder,
    set_geometry_backend,
    get_geometry_backend,
    get_geometry_builder,
)

from .c_solver_wrapper import (
    CIntersectionSolver,
    CFaceSolver,
    CGeometryBuilder,
    is_c_extension_available,
    solve_all_c,
)

# ==============================================================================
# Solver Backend Selection
# ==============================================================================

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
    if backend == 'c' and not is_c_extension_available():
        import warnings
        warnings.warn("C extension not available, falling back to CPU")
        backend = 'cpu'
    
    _solver_backend = backend
    
    # Also set geometry backend for compatibility
    if backend == 'gpu':
        set_geometry_backend('gpu')
    else:
        set_geometry_backend('cpu')

def get_solver_backend() -> str:
    """Get the current solver backend."""
    return _solver_backend

def get_intersection_solver():
    """Get the IntersectionSolver for the current backend."""
    if _solver_backend == 'c':
        if is_c_extension_available():
            return CIntersectionSolver
        import warnings
        warnings.warn("C extension not available, falling back to CPU")
        return CPUIntersectionSolver
    if _solver_backend == 'taichi':
        try:
            from .taichi.taichi_intersection_solver import TaichiIntersectionSolver
            return TaichiIntersectionSolver
        except ImportError as e:
            import warnings
            warnings.warn(f"Taichi not available, falling back to CPU: {e}")
            return CPUIntersectionSolver
    return CPUIntersectionSolver

def get_face_solver():
    """Get the FaceSolver for the current backend."""
    if _solver_backend == 'c':
        if is_c_extension_available():
            return CFaceSolver
        import warnings
        warnings.warn("C extension not available, falling back to CPU")
        return CPUFaceSolver
    if _solver_backend == 'taichi':
        try:
            from .taichi.taichi_face_solver import TaichiFaceSolver
            return TaichiFaceSolver
        except ImportError as e:
            import warnings
            warnings.warn(f"Taichi not available, falling back to CPU: {e}")
            return CPUFaceSolver
    return CPUFaceSolver

def get_full_geometry_builder():
    """Get the GeometryBuilder for the current backend."""
    if _solver_backend == 'c':
        if is_c_extension_available():
            return CGeometryBuilder
        import warnings
        warnings.warn("C extension not available, falling back to CPU")
        return CPUGeometryBuilder
    if _solver_backend == 'taichi':
        try:
            from .taichi.taichi_geometry_builder import TaichiGeometryBuilder
            return TaichiGeometryBuilder
        except ImportError as e:
            import warnings
            warnings.warn(f"Taichi not available, falling back to CPU: {e}")
            return CPUGeometryBuilder
    return CPUGeometryBuilder

# Expose the default classes (for backwards compatibility)
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
    # Stage 2: Face Solving (combined face segmentation + bounds)
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
    # Backend selection
    'set_solver_backend',
    'get_solver_backend',
    'get_intersection_solver',
    'get_face_solver',
    'get_full_geometry_builder',
    'is_c_extension_available',
    'solve_all_c',
    # Legacy geometry-only backend
    'set_geometry_backend',
    'get_geometry_backend',
    'get_geometry_builder',
]
