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
from typing import Dict, Any, Optional, Callable, Union

from ._profiler_backend import NoOpBackend, PythonBackend


# =============================================================================
# Configuration & Backend Selection
# =============================================================================

_PROFILING_COMPILED_OUT = (
    os.environ.get('GXML_NO_PROFILING', '').lower() in ('1', 'true', 'yes')
    or not __debug__
)

_FORCE_PYTHON = os.environ.get('GXML_FORCE_PYTHON_PROFILER', '').lower() in ('1', 'true', 'yes')

# Select backend ONCE at import time
if _PROFILING_COMPILED_OUT:
    _USE_C_PROFILER = False
    _backend = NoOpBackend()
elif _FORCE_PYTHON:
    _USE_C_PROFILER = False
    _backend = PythonBackend()
else:
    try:
        from ._native_profiler_backend import CBackend
        _USE_C_PROFILER = True
        _backend = CBackend()
    except ImportError:
        _USE_C_PROFILER = False
        _backend = PythonBackend()


def is_c_profiler_available() -> bool:
    """Check if the C profiler extension is available."""
    return _USE_C_PROFILER


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
    
    Returns:
        A reusable context manager
    """
    return _backend.create_perf_marker(name or "unknown")


def get_marker(name: str):
    """
    Get a reusable cached marker for a given name.
    
    Usage:
        _my_marker = get_marker("my_section")  # Create once
        with _my_marker:  # Reuse
            ...
    
    Returns:
        A reusable context manager
    """
    return _backend.create_perf_marker(name)


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
