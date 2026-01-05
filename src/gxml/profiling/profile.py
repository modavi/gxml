"""
Provides a way to measure performance of code blocks.

Usage:
    from gxml.profiling import profile, perf_marker, get_profile_results, reset_profile
    
    # Decorator for functions (recommended):
    @profile
    def my_function():
        ...
    
    # Or with custom name:
    @profile("custom_name")
    def my_function():
        ...
    
    # Context manager for code blocks:
    with perf_marker("my_section"):
        ...
    
    # Get results
    results = get_profile_results()
    # {'my_function': {'count': 1, 'total_ms': 5.2, 'min_ms': 5.2, 'max_ms': 5.2}}
    
    # Reset for next run
    reset_profile()

Zero-overhead mode:
    Profiling is completely compiled out (zero overhead) when:
    - Environment variable GXML_NO_PROFILING=1 is set, OR
    - Python is run with optimization (-O flag, which sets __debug__=False)
    
    This removes ALL overhead - no function calls, no checks.
    Requires process restart to take effect.
"""

import os
import time
import functools
from typing import Dict, Any, Optional, Callable, Union, List, Tuple

# =============================================================================
# Configuration
# =============================================================================

_PROFILING_COMPILED_OUT = (
    os.environ.get('GXML_NO_PROFILING', '').lower() in ('1', 'true', 'yes')
    or not __debug__
)

_FORCE_PYTHON = os.environ.get('GXML_FORCE_PYTHON_PROFILER', '').lower() in ('1', 'true', 'yes')

# Local reference for speed
_perf = time.perf_counter


# =============================================================================
# NoOp Backend (zero overhead when profiling compiled out)
# =============================================================================

class _NoOpMarker:
    """No-op context manager that can be reused."""
    __slots__ = ()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        return False


class _NoOpBackend:
    """Zero-overhead backend when profiling is compiled out."""
    
    def __init__(self):
        self._noop_marker = _NoOpMarker()
    
    def clear(self) -> None:
        pass
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        return {}
    
    def create_perf_marker(self, name: str):
        return self._noop_marker
    
    def create_profiled_function(self, func: Callable, name: str) -> Callable:
        return func


# =============================================================================
# Python Backend (fallback)
# =============================================================================

class _PythonPerfMarker:
    """Python context manager for performance marking."""
    __slots__ = ('marker_id', 'neg_marker', '_events')
    
    def __init__(self, marker_id: int, events: List):
        self.marker_id = marker_id
        self.neg_marker = -(marker_id + 1)
        self._events = events
    
    def __enter__(self):
        self._events.append((self.marker_id, _perf()))
        return self
    
    def __exit__(self, *args):
        self._events.append((self.neg_marker, _perf()))
        return False


class _PythonBackend:
    """Pure Python profiler backend."""
    
    def __init__(self):
        self._events: List[Tuple[int, float]] = []
        self._id_to_name: List[str] = []
        self._name_to_id: Dict[str, int] = {}
    
    def _register_marker(self, name: str) -> int:
        """Register a marker name and get its integer ID."""
        if name in self._name_to_id:
            return self._name_to_id[name]
        
        marker_id = len(self._id_to_name)
        self._id_to_name.append(name)
        self._name_to_id[name] = marker_id
        return marker_id
    
    def clear(self) -> None:
        self._events.clear()
    
    def _get_events(self):
        """Return the event stream. Override in subclasses."""
        return self._events
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Process events and return marker statistics."""
        markers: Dict[str, Dict[str, Any]] = {}
        stack: List[Tuple[int, float, int]] = []
        
        for marker_id, timestamp in self._get_events():
            if marker_id >= 0:
                parent_id = stack[-1][0] if stack else -1
                stack.append((marker_id, timestamp, parent_id))
            else:
                actual_id = -(marker_id + 1)
                if stack:
                    start_id, start_time, parent_id = stack.pop()
                    if start_id != actual_id:
                        continue
                    
                    elapsed_ms = (timestamp - start_time) * 1000
                    name = self._id_to_name[actual_id]
                    
                    if name not in markers:
                        markers[name] = {
                            'count': 0,
                            'total_ms': 0.0,
                            'min_ms': float('inf'),
                            'max_ms': 0.0,
                            'parents': {}
                        }
                    
                    m = markers[name]
                    m['count'] += 1
                    m['total_ms'] += elapsed_ms
                    m['min_ms'] = min(m['min_ms'], elapsed_ms)
                    m['max_ms'] = max(m['max_ms'], elapsed_ms)
                    
                    if parent_id >= 0:
                        parent_name = self._id_to_name[parent_id]
                        m['parents'][parent_name] = m['parents'].get(parent_name, 0) + 1
        
        # Finalize
        for name, m in markers.items():
            m['avg_ms'] = m['total_ms'] / m['count'] if m['count'] > 0 else 0.0
            if m['min_ms'] == float('inf'):
                m['min_ms'] = 0.0
            m['total_ms'] = round(m['total_ms'], 3)
            m['avg_ms'] = round(m['avg_ms'], 3)
            m['min_ms'] = round(m['min_ms'], 3)
            m['max_ms'] = round(m['max_ms'], 3)
        
        return markers
    
    def create_perf_marker(self, name: str):
        marker_id = self._register_marker(name)
        return _PythonPerfMarker(marker_id, self._events)
    
    def create_profiled_function(self, func: Callable, name: str) -> Callable:
        """Create a wrapped function that records entry/exit times."""
        marker_id = self._register_marker(name)
        neg_marker = -(marker_id + 1)
        events = self._events
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            events.append((marker_id, _perf()))
            try:
                return func(*args, **kwargs)
            finally:
                events.append((neg_marker, _perf()))
        
        return wrapper


# =============================================================================
# C Extension Backend (if available)
# =============================================================================

try:
    from . import _c_profile
    
    class _CBackend(_PythonBackend):
        """C extension profiler backend."""
        
        def clear(self) -> None:
            _c_profile.clear_events()
        
        def _get_events(self):
            return _c_profile.get_events()
        
        def create_perf_marker(self, name: str):
            marker_id = self._register_marker(name)
            return _c_profile.create_perf_marker(marker_id)
        
        def create_profiled_function(self, func: Callable, name: str) -> Callable:
            marker_id = self._register_marker(name)
            return _c_profile.create_profiled_function(func, marker_id)

except ImportError:
    _CBackend = None


# =============================================================================
# Backend Selection (at import time)
# =============================================================================

if _PROFILING_COMPILED_OUT:
    _USE_C_PROFILER = False
    _backend = _NoOpBackend()
elif _FORCE_PYTHON:
    _USE_C_PROFILER = False
    _backend = _PythonBackend()
elif _CBackend is not None:
    _USE_C_PROFILER = True
    _backend = _CBackend()
else:
    _USE_C_PROFILER = False
    _backend = _PythonBackend()


# =============================================================================
# Public API
# =============================================================================

def reset_profile():
    """Reset all collected profile data."""
    _backend.clear()


def get_profile_results() -> Dict[str, Dict[str, Any]]:
    """
    Get marker statistics from collected profiling data.
    
    Returns:
        Dict mapping marker names to their stats:
        {
            'marker_name': {
                'count': 10,
                'total_ms': 52.3,
                'avg_ms': 5.23,
                'min_ms': 4.1,
                'max_ms': 7.8,
                'parents': {'parent_marker': 10}
            }
        }
    """
    return _backend.get_results()


def perf_marker(name: Optional[str] = None):
    """
    Create a context manager for performance marking.
    
    Usage:
        with perf_marker("my_section"):
            # ... do work ...
    """
    return _backend.create_perf_marker(name or "unknown")


def profile(name_or_func: Union[str, Callable, None] = None) -> Callable:
    """
    Decorator for profiling functions.
    
    Usage:
        @profile
        def my_function():
            ...
        
        @profile("custom_name")
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        marker_name = name_or_func if isinstance(name_or_func, str) else func.__name__
        return _backend.create_profiled_function(func, marker_name)
    
    if callable(name_or_func):
        return decorator(name_or_func)
    else:
        return decorator
