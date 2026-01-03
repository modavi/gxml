"""
Shader backend abstraction layer for GXML.

Provides a unified API for GPU compute across different platforms:
- macOS: Metal (via pyobjc)
- Windows: DirectX 12 (via wgpu with Dawn or native)
- Linux: Vulkan (via wgpu)
- Browser: WebGPU (via wgpu-py or JS interop)

The shader backends handle:
1. Intersection solving (parallel line-line tests)
2. Face point computation (batch matrix transforms)
3. Geometry building (vertex/index generation)

Usage:
    from gxml.gpu.shader_backend import get_shader_backend
    
    backend = get_shader_backend()
    intersections = backend.find_intersections(starts, ends)
    vertices = backend.transform_points(matrices, points)
"""

import platform
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from pathlib import Path

# Shader source directory
SHADER_DIR = Path(__file__).parent.parent / "shaders"


class ShaderBackend(ABC):
    """Abstract base class for GPU compute backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging/debugging."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        pass
    
    @abstractmethod
    def find_intersections(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        tolerance: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find all intersections between panel centerlines.
        
        Args:
            starts: (N, 3) array of start points
            ends: (N, 3) array of end points
            tolerance: Intersection tolerance
            
        Returns:
            Tuple of:
            - indices_i: Panel i indices for each intersection
            - indices_j: Panel j indices for each intersection  
            - t_values_i: T parameter on panel i
            - t_values_j: T parameter on panel j
            - positions: (M, 3) intersection positions
        """
        pass
    
    @abstractmethod
    def transform_points(
        self,
        matrices: np.ndarray,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Batch transform points by matrices.
        
        Args:
            matrices: (N, 4, 4) transformation matrices
            points: (N, M, 4) points in local space (w=1)
            
        Returns:
            (N, M, 4) points in world space
        """
        pass
    
    @abstractmethod
    def compute_face_points(
        self,
        matrices: np.ndarray,
        half_thicknesses: np.ndarray,
        t_coords: np.ndarray,
        s_coords: np.ndarray,
        face_sides: np.ndarray
    ) -> np.ndarray:
        """
        Compute face points from (t, s) coordinates.
        
        Args:
            matrices: (N, 4, 4) panel transforms
            half_thicknesses: (N,) half-thickness values
            t_coords: (N,) t coordinates [0, 1]
            s_coords: (N,) s coordinates [0, 1]
            face_sides: (N,) face side enum (0=FRONT, 1=BACK, etc.)
            
        Returns:
            (N, 4) world space points
        """
        pass


class CPUBackend(ShaderBackend):
    """
    Pure NumPy/CPU fallback backend.
    Works on any platform but without GPU acceleration.
    """
    
    @property
    def name(self) -> str:
        return "CPU (NumPy)"
    
    @property
    def is_available(self) -> bool:
        return True
    
    def find_intersections(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        tolerance: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """CPU implementation of intersection finding."""
        n = len(starts)
        results_i = []
        results_j = []
        results_t1 = []
        results_t2 = []
        results_pos = []
        
        tol_sq = tolerance * tolerance
        
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = starts[i], ends[i]
                p3, p4 = starts[j], ends[j]
                
                d1 = p2 - p1
                d2 = p4 - p3
                w = p3 - p1
                
                cross_d = np.cross(d1, d2)
                denom = np.dot(cross_d, cross_d)
                
                if denom < tol_sq:
                    continue
                
                wcd2 = np.cross(w, d2)
                t1 = np.dot(wcd2, cross_d) / denom
                if t1 < -tolerance or t1 > 1.0 + tolerance:
                    continue
                
                wcd1 = np.cross(w, d1)
                t2 = np.dot(wcd1, cross_d) / denom
                if t2 < -tolerance or t2 > 1.0 + tolerance:
                    continue
                
                i1 = p1 + t1 * d1
                i2 = p3 + t2 * d2
                diff = i1 - i2
                if np.dot(diff, diff) >= tol_sq:
                    continue
                
                results_i.append(i)
                results_j.append(j)
                results_t1.append(t1)
                results_t2.append(t2)
                results_pos.append(i1)
        
        return (
            np.array(results_i, dtype=np.int32),
            np.array(results_j, dtype=np.int32),
            np.array(results_t1, dtype=np.float32),
            np.array(results_t2, dtype=np.float32),
            np.array(results_pos, dtype=np.float32).reshape(-1, 3) if results_pos else np.zeros((0, 3), dtype=np.float32)
        )
    
    def transform_points(
        self,
        matrices: np.ndarray,
        points: np.ndarray
    ) -> np.ndarray:
        """CPU batch matrix transform."""
        n, m, _ = points.shape
        result = np.zeros_like(points)
        for i in range(n):
            result[i] = (matrices[i] @ points[i].T).T
        return result
    
    def compute_face_points(
        self,
        matrices: np.ndarray,
        half_thicknesses: np.ndarray,
        t_coords: np.ndarray,
        s_coords: np.ndarray,
        face_sides: np.ndarray
    ) -> np.ndarray:
        """CPU face point computation."""
        n = len(matrices)
        result = np.zeros((n, 4), dtype=np.float32)
        
        for i in range(n):
            half_t = half_thicknesses[i]
            t = t_coords[i]
            s = s_coords[i]
            side = face_sides[i]
            
            if side == 0:  # FRONT
                local = np.array([t, s, half_t, 1.0])
            elif side == 1:  # BACK
                local = np.array([t, s, -half_t, 1.0])
            elif side == 2:  # TOP
                local = np.array([t, 1.0, -half_t + s * half_t * 2.0, 1.0])
            elif side == 3:  # BOTTOM
                local = np.array([t, 0.0, -half_t + s * half_t * 2.0, 1.0])
            elif side == 4:  # START
                local = np.array([0.0, s, -half_t + t * half_t * 2.0, 1.0])
            elif side == 5:  # END
                local = np.array([1.0, s, -half_t + t * half_t * 2.0, 1.0])
            else:
                local = np.array([t, s, 0.0, 1.0])
            
            result[i] = matrices[i] @ local
        
        return result


class MetalBackend(ShaderBackend):
    """
    Metal compute shader backend for macOS.
    Uses pyobjc-framework-Metal for GPU acceleration.
    """
    
    def __init__(self):
        self._device = None
        self._queue = None
        self._library = None
        self._available = False
        self._init_metal()
    
    def _init_metal(self):
        """Initialize Metal device and compile shaders."""
        try:
            import Metal
            import objc
            
            self._device = Metal.MTLCreateSystemDefaultDevice()
            if self._device is None:
                return
            
            self._queue = self._device.newCommandQueue()
            
            # Load Metal shader from file or use embedded source
            shader_path = SHADER_DIR / "intersection_solver.metal"
            if shader_path.exists():
                with open(shader_path) as f:
                    source = f.read()
            else:
                # Use the source from metal_geometry.py
                from gxml.gpu.metal_geometry import METAL_SHADER_SOURCE
                source = METAL_SHADER_SOURCE
            
            options = Metal.MTLCompileOptions.alloc().init()
            self._library, error = self._device.newLibraryWithSource_options_error_(
                source, options, objc.nil
            )
            
            if error:
                print(f"Metal shader compilation error: {error}")
                return
            
            self._available = True
            
        except ImportError:
            pass
        except Exception as e:
            print(f"Metal initialization error: {e}")
    
    @property
    def name(self) -> str:
        return "Metal (macOS)"
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def find_intersections(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        tolerance: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Metal GPU intersection finding."""
        # For now, delegate to Metal implementation in metal_geometry.py
        # or fall back to CPU
        if not self._available:
            return CPUBackend().find_intersections(starts, ends, tolerance)
        
        # TODO: Implement Metal compute shader dispatch
        # This requires setting up compute pipelines and buffers
        return CPUBackend().find_intersections(starts, ends, tolerance)
    
    def transform_points(
        self,
        matrices: np.ndarray,
        points: np.ndarray
    ) -> np.ndarray:
        """Metal GPU batch transform."""
        if not self._available:
            return CPUBackend().transform_points(matrices, points)
        
        try:
            from gxml.gpu.metal_geometry import MetalGeometryAccelerator
            accel = MetalGeometryAccelerator()
            return accel.transform_points_batch(matrices, points)
        except Exception:
            return CPUBackend().transform_points(matrices, points)
    
    def compute_face_points(
        self,
        matrices: np.ndarray,
        half_thicknesses: np.ndarray,
        t_coords: np.ndarray,
        s_coords: np.ndarray,
        face_sides: np.ndarray
    ) -> np.ndarray:
        """Metal GPU face point computation."""
        if not self._available:
            return CPUBackend().compute_face_points(
                matrices, half_thicknesses, t_coords, s_coords, face_sides
            )
        
        try:
            from gxml.gpu.metal_geometry import MetalGeometryAccelerator
            accel = MetalGeometryAccelerator()
            return accel.compute_face_points(
                matrices, half_thicknesses, t_coords, s_coords, face_sides
            )
        except Exception:
            return CPUBackend().compute_face_points(
                matrices, half_thicknesses, t_coords, s_coords, face_sides
            )


class WGPUBackend(ShaderBackend):
    """
    WebGPU backend using wgpu-py.
    Works on Windows (via Dawn/DirectX), Linux (Vulkan), and macOS (Metal).
    Also works in browsers via wgpu/WebGPU.
    """
    
    def __init__(self):
        self._device = None
        self._available = False
        self._intersection_pipeline = None
        self._init_wgpu()
    
    def _init_wgpu(self):
        """Initialize wgpu device and compile shaders."""
        try:
            import wgpu
            
            # Request adapter and device
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter is None:
                return
            
            self._device = adapter.request_device_sync()
            
            # Load WGSL shader
            shader_path = SHADER_DIR / "intersection_solver.wgsl"
            if shader_path.exists():
                with open(shader_path) as f:
                    shader_source = f.read()
                
                # Create shader module
                self._shader_module = self._device.create_shader_module(code=shader_source)
                self._available = True
            
        except ImportError:
            pass
        except Exception as e:
            print(f"wgpu initialization error: {e}")
    
    @property
    def name(self) -> str:
        return "WebGPU (wgpu-py)"
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def find_intersections(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        tolerance: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """WebGPU intersection finding."""
        if not self._available:
            return CPUBackend().find_intersections(starts, ends, tolerance)
        
        # TODO: Implement full WebGPU compute dispatch
        return CPUBackend().find_intersections(starts, ends, tolerance)
    
    def transform_points(
        self,
        matrices: np.ndarray,
        points: np.ndarray
    ) -> np.ndarray:
        """WebGPU batch transform."""
        if not self._available:
            return CPUBackend().transform_points(matrices, points)
        
        # TODO: Implement WebGPU transform pipeline
        return CPUBackend().transform_points(matrices, points)
    
    def compute_face_points(
        self,
        matrices: np.ndarray,
        half_thicknesses: np.ndarray,
        t_coords: np.ndarray,
        s_coords: np.ndarray,
        face_sides: np.ndarray
    ) -> np.ndarray:
        """WebGPU face point computation."""
        if not self._available:
            return CPUBackend().compute_face_points(
                matrices, half_thicknesses, t_coords, s_coords, face_sides
            )
        
        # TODO: Implement WebGPU face point pipeline
        return CPUBackend().compute_face_points(
            matrices, half_thicknesses, t_coords, s_coords, face_sides
        )


class CExtensionBackend(ShaderBackend):
    """
    C extension backend using _c_solvers.
    Fastest CPU path, uses optimized C code with SIMD potential.
    """
    
    def __init__(self):
        self._available = False
        self._module = None
        self._init_c_extension()
    
    def _init_c_extension(self):
        """Try to load C extension directly (avoid package init issues)."""
        try:
            # Import directly to avoid solvers package __init__ issues
            import importlib.util
            import sys
            from pathlib import Path
            
            # Find the extension file
            solvers_dir = Path(__file__).parent.parent / "elements" / "solvers"
            
            # Look for the .so/.pyd file
            for ext_file in solvers_dir.glob("_c_solvers*.so"):
                spec = importlib.util.spec_from_file_location("_c_solvers", ext_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._module = module
                    self._available = True
                    return
            
            for ext_file in solvers_dir.glob("_c_solvers*.pyd"):
                spec = importlib.util.spec_from_file_location("_c_solvers", ext_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._module = module
                    self._available = True
                    return
                    
        except Exception as e:
            # Silently fail - will fall back to CPU
            pass
    
    @property
    def name(self) -> str:
        return "C Extension"
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def find_intersections(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        tolerance: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """C extension intersection finding."""
        if not self._available:
            return CPUBackend().find_intersections(starts, ends, tolerance)
        
        starts = np.ascontiguousarray(starts, dtype=np.float64)
        ends = np.ascontiguousarray(ends, dtype=np.float64)
        
        result = self._module.batch_find_intersections(starts, ends, tolerance)
        indices_i, indices_j, t_values_i, t_values_j, positions = result
        
        return (
            indices_i.astype(np.int32),
            indices_j.astype(np.int32),
            t_values_i.astype(np.float32),
            t_values_j.astype(np.float32),
            positions.astype(np.float32)
        )
    
    def transform_points(
        self,
        matrices: np.ndarray,
        points: np.ndarray
    ) -> np.ndarray:
        """C extension doesn't have transform_points, fall back to CPU."""
        return CPUBackend().transform_points(matrices, points)
    
    def compute_face_points(
        self,
        matrices: np.ndarray,
        half_thicknesses: np.ndarray,
        t_coords: np.ndarray,
        s_coords: np.ndarray,
        face_sides: np.ndarray
    ) -> np.ndarray:
        """C extension doesn't have face points, fall back to CPU."""
        return CPUBackend().compute_face_points(
            matrices, half_thicknesses, t_coords, s_coords, face_sides
        )


# ============================================================================
# Backend Selection
# ============================================================================

_cached_backend: Optional[ShaderBackend] = None


def get_shader_backend(preferred: Optional[str] = None) -> ShaderBackend:
    """
    Get the best available shader backend.
    
    Args:
        preferred: Preferred backend name ('metal', 'wgpu', 'c', 'cpu')
        
    Returns:
        ShaderBackend instance
    """
    global _cached_backend
    
    if _cached_backend is not None and preferred is None:
        return _cached_backend
    
    system = platform.system()
    
    # Try preferred backend first
    if preferred:
        preferred = preferred.lower()
        if preferred == 'metal':
            backend = MetalBackend()
            if backend.is_available:
                _cached_backend = backend
                return backend
        elif preferred == 'wgpu':
            backend = WGPUBackend()
            if backend.is_available:
                _cached_backend = backend
                return backend
        elif preferred == 'c':
            backend = CExtensionBackend()
            if backend.is_available:
                _cached_backend = backend
                return backend
        elif preferred == 'cpu':
            _cached_backend = CPUBackend()
            return _cached_backend
    
    # Auto-select best available backend
    
    # 1. Try C extension first (fastest CPU path)
    c_backend = CExtensionBackend()
    if c_backend.is_available:
        _cached_backend = c_backend
        return c_backend
    
    # 2. Try platform-specific GPU backend
    if system == 'Darwin':
        metal_backend = MetalBackend()
        if metal_backend.is_available:
            _cached_backend = metal_backend
            return metal_backend
    
    # 3. Try wgpu (cross-platform)
    wgpu_backend = WGPUBackend()
    if wgpu_backend.is_available:
        _cached_backend = wgpu_backend
        return wgpu_backend
    
    # 4. Fall back to CPU
    _cached_backend = CPUBackend()
    return _cached_backend


def list_available_backends() -> List[Tuple[str, bool]]:
    """
    List all backends and their availability.
    
    Returns:
        List of (name, is_available) tuples
    """
    backends = [
        CExtensionBackend(),
        MetalBackend(),
        WGPUBackend(),
        CPUBackend(),
    ]
    return [(b.name, b.is_available) for b in backends]
