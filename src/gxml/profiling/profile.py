"""
Provides a way to measure performance of code blocks.

Usage:
    from gxml.profiling import profile, perf_marker, get_profile_results, reset_profile
    
    # Enable profiling
    enable_profiling(True)
    
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

Performance optimization:
    This profiler is ultra-optimized for minimal per-marker overhead:
    - Pre-registered integer marker IDs (no string operations in hot path)
    - Inlined push/pop in decorators (no function call overhead)
    - No if-checks in hot path (use -O flag to disable profiling)
    - Deferred processing - just stores (marker_id, timestamp) tuples
    - All statistics computed in get_profile_results()
"""

import os

# =============================================================================
# Check at import time - determines if profiling is compiled out entirely
# =============================================================================

# Profiling is compiled out if:
# 1. GXML_NO_PROFILING env var is set, OR
# 2. Running with python -O (optimized mode, __debug__ is False)
_PROFILING_COMPILED_OUT = (
    os.environ.get('GXML_NO_PROFILING', '').lower() in ('1', 'true', 'yes')
    or not __debug__
)

if _PROFILING_COMPILED_OUT:
    # =========================================================================
    # COMPILED-OUT MODE: Zero-overhead no-op implementations
    # =========================================================================
    from contextlib import contextmanager
    from typing import Dict, Any, Callable, Union
    
    # Backwards compat
    PROFILE_ENABLED = False
    
    def enable_profiling(enabled: bool = True, use_houdini: bool = False):
        """No-op when profiling is compiled out."""
        pass
    
    def is_profiling_enabled() -> bool:
        """Always False when profiling is compiled out."""
        return False
    
    def reset_profile():
        """No-op when profiling is compiled out."""
        pass
    
    def get_profile_results() -> Dict[str, Dict[str, Any]]:
        """Returns empty dict when profiling is compiled out."""
        return {}
    
    @contextmanager
    def perf_marker(name=None):
        """Zero-overhead context manager when profiling is compiled out."""
        yield
    
    # No-op reusable marker for compiled-out mode
    class _NoOpMarker:
        """No-op context manager that can be reused."""
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    
    _noop_marker = _NoOpMarker()
    
    def get_marker(name: str):
        """Returns a no-op reusable marker when profiling is compiled out."""
        return _noop_marker
    
    def profile(name_or_func: Union[str, Callable, None] = None) -> Callable:
        """Zero-overhead decorator when profiling is compiled out."""
        if callable(name_or_func):
            # Called as @profile without parentheses - return function unchanged
            return name_or_func
        else:
            # Called as @profile() or @profile("name") - return identity decorator
            return lambda f: f
    
    # Internal functions (no-ops)
    def _push_perf_marker(name=None):
        pass
    
    def _pop_perf_marker():
        pass
    
    def _update_compat():
        pass

else:
    # =========================================================================
    # FULL MODE: Ultra-optimized deferred-processing profiler
    # =========================================================================
    import time
    import functools
    from typing import Dict, Any, Optional, Callable, Union, List, Tuple

    # =========================================================================
    # Try to use C extension for hot path (2x faster)
    # Can be disabled with GXML_FORCE_PYTHON_PROFILER=1 for testing
    # =========================================================================
    _FORCE_PYTHON = os.environ.get('GXML_FORCE_PYTHON_PROFILER', '').lower() in ('1', 'true', 'yes')
    
    if _FORCE_PYTHON:
        _USE_C_PROFILER = False
        _c_create_profiled_function = None  # type: ignore
        _c_create_perf_marker = None  # type: ignore
    else:
        try:
            from gxml.profiling._c_profiler import (
                push_marker as _c_push,
                pop_marker as _c_pop,
                get_events as _c_get_events,
                clear_events as _c_clear_events,
                create_profiled_function as _c_create_profiled_function,
                create_perf_marker as _c_create_perf_marker,
            )
            _USE_C_PROFILER = True
        except ImportError:
            _USE_C_PROFILER = False
            _c_create_profiled_function = None  # type: ignore
            _c_create_perf_marker = None  # type: ignore

    # =========================================================================
    # Module-level state for maximum performance
    # =========================================================================
    
    # Event stream: (marker_id, timestamp)
    # Positive marker_id = push, negative (-(id+1)) = pop
    # All processing deferred to get_profile_results()
    # Only used if C extension not available
    _events: List[Tuple[int, float]] = []
    
    # Marker name <-> ID mapping
    _id_to_name: List[str] = []
    _name_to_id: Dict[str, int] = {}
    
    # Profiling state
    _enabled: bool = False
    _use_houdini: bool = False
    _houdini_stack: List = []
    
    # Local reference for speed (critical - saves attribute lookup)
    _perf = time.perf_counter
    
    # Backwards compat
    PROFILE_ENABLED = False

    # =========================================================================
    # Marker registration
    # =========================================================================
    
    def _register_marker(name: str) -> int:
        """
        Register a marker name and get its integer ID.
        Called at decoration time, not in hot path.
        """
        if name in _name_to_id:
            return _name_to_id[name]
        
        marker_id = len(_id_to_name)
        _id_to_name.append(name)
        _name_to_id[name] = marker_id
        return marker_id

    # =========================================================================
    # Public API
    # =========================================================================

    def enable_profiling(enabled: bool = True, use_houdini: bool = False):
        """
        Enable or disable profiling.
        
        Args:
            enabled: Whether to enable profiling
            use_houdini: If True, also use Houdini's perfMon (when available)
        """
        global _enabled, _use_houdini
        _enabled = enabled
        _use_houdini = use_houdini
        _update_compat()


    def is_profiling_enabled() -> bool:
        """Check if profiling is enabled."""
        return _enabled


    def reset_profile():
        """Reset all collected profile data."""
        if _USE_C_PROFILER:
            _c_clear_events()
        else:
            _events.clear()
        _houdini_stack.clear()


    def get_profile_results() -> Dict[str, Dict[str, Any]]:
        """
        Process all collected events and return marker statistics.
        
        All heavy processing is deferred to this function to minimize
        per-marker overhead during profiling.
        
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
        markers: Dict[str, Dict[str, Any]] = {}
        stack: List[Tuple[int, float, int]] = []  # (marker_id, start_time, parent_id)
        
        # Get events from C extension or Python list
        events = _c_get_events() if _USE_C_PROFILER else _events
        
        for marker_id, timestamp in events:
            if marker_id >= 0:
                # Push event - record start time and parent
                parent_id = stack[-1][0] if stack else -1
                stack.append((marker_id, timestamp, parent_id))
            else:
                # Pop event - compute duration and accumulate stats
                actual_id = -(marker_id + 1)
                if stack:
                    start_id, start_time, parent_id = stack.pop()
                    
                    # Sanity check
                    if start_id != actual_id:
                        continue
                    
                    elapsed_ms = (timestamp - start_time) * 1000
                    name = _id_to_name[actual_id]
                    
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
                        parent_name = _id_to_name[parent_id]
                        m['parents'][parent_name] = m['parents'].get(parent_name, 0) + 1
        
        # Calculate averages and finalize
        for name, m in markers.items():
            m['avg_ms'] = m['total_ms'] / m['count'] if m['count'] > 0 else 0.0
            if m['min_ms'] == float('inf'):
                m['min_ms'] = 0.0
            # Round values
            m['total_ms'] = round(m['total_ms'], 3)
            m['avg_ms'] = round(m['avg_ms'], 3)
            m['min_ms'] = round(m['min_ms'], 3)
            m['max_ms'] = round(m['max_ms'], 3)
        
        return markers


    def _push_perf_marker(name: Optional[str] = None):
        """
        Start a performance marker (internal implementation).
        
        Args:
            name: Marker name. Required for best performance.
        """
        if name is None:
            name = "unknown"
        
        marker_id = _name_to_id.get(name)
        if marker_id is None:
            marker_id = _register_marker(name)
        
        _events.append((marker_id, _perf()))


    def _pop_perf_marker():
        """
        End the current performance marker (internal implementation).
        
        Note: This requires knowing what marker we're popping. For best results,
        use the @profile decorator or perf_marker context manager instead.
        """
        # Can't reliably pop without knowing marker ID - this is a limitation
        # Use perf_marker context manager or @profile decorator instead
        pass


    def perf_marker(name: Optional[str] = None):
        """
        Create a context manager for performance marking. Exception-safe.
        
        Uses C extension if available for minimal overhead.
        
        Usage:
            with perf_marker("my_section"):
                # ... do work ...
        
        For maximum performance in hot loops, cache the marker:
            _my_marker = perf_marker("my_section")  # Create once
            with _my_marker:  # Reuse - no dict lookup or object creation
                ...
        
        Returns:
            A context manager (C PerfMarker if available, Python fallback otherwise)
        """
        if name is None:
            name = "unknown"
        
        marker_id = _name_to_id.get(name)
        if marker_id is None:
            marker_id = _register_marker(name)
        
        if _USE_C_PROFILER and _c_create_perf_marker is not None:
            # Use C context manager - faster than Python class
            return _c_create_perf_marker(marker_id)
        else:
            # Python fallback
            return _PythonPerfMarker(marker_id)
    
    
    # Cache for pre-created markers (avoids dict lookup + object creation)
    _marker_cache: Dict[str, Any] = {}
    
    def get_marker(name: str):
        """
        Get a reusable cached marker for a given name.
        
        This eliminates the per-call overhead of dict lookup and object creation.
        The returned marker can be reused as a context manager multiple times.
        
        Usage:
            # At module level or in __init__:
            _my_marker = get_marker("my_section")
            
            # In hot path (zero overhead beyond context manager protocol):
            with _my_marker:
                ...
        
        Returns:
            A reusable context manager
        """
        if name not in _marker_cache:
            marker_id = _name_to_id.get(name)
            if marker_id is None:
                marker_id = _register_marker(name)
            
            if _USE_C_PROFILER and _c_create_perf_marker is not None:
                _marker_cache[name] = _c_create_perf_marker(marker_id)
            else:
                _marker_cache[name] = _PythonPerfMarker(marker_id)
        
        return _marker_cache[name]
    

    class _PythonPerfMarker:
        """Python fallback context manager for when C extension is not available."""
        __slots__ = ('marker_id', 'neg_marker')
        
        def __init__(self, marker_id: int):
            self.marker_id = marker_id
            self.neg_marker = -(marker_id + 1)
        
        def __enter__(self):
            _events.append((self.marker_id, _perf()))
            return self
        
        def __exit__(self, *args):
            _events.append((self.neg_marker, _perf()))
            return False


    def profile(name_or_func: Union[str, Callable, None] = None) -> Callable:
        """
        Decorator for profiling functions. Exception-safe.
        
        The marker ID is pre-registered at decoration time for maximum
        performance during function calls. Uses C extension if available
        for fastest profiling - the C wrapper eliminates Python<->C boundary
        crossings for push/pop operations.
        
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
            # Pre-register marker ID at decoration time
            marker_id = _register_marker(marker_name)
            
            if _USE_C_PROFILER and _c_create_profiled_function is not None:
                # Use C-level wrapper - eliminates Python<->C boundary for push/pop
                return _c_create_profiled_function(func, marker_id)
            else:
                # Python fallback
                # Pre-compute negative marker ID for pop events
                neg_marker = -(marker_id + 1)
                
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    _events.append((marker_id, _perf()))
                    try:
                        return func(*args, **kwargs)
                    finally:
                        _events.append((neg_marker, _perf()))
                
                return wrapper
        
        # Handle @profile vs @profile() vs @profile("name")
        if callable(name_or_func):
            # Called as @profile without parentheses
            return decorator(name_or_func)
        else:
            # Called as @profile() or @profile("name")
            return decorator


    # =========================================================================
    # Backwards Compatibility
    # =========================================================================

    def _update_compat():
        """Update backwards-compat global."""
        global PROFILE_ENABLED
        PROFILE_ENABLED = _enabled
