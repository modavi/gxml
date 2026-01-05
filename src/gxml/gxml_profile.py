"""
Provides a way to measure performance of code blocks.

Usage:
    from gxml_profile import profile, perf_marker, get_profile_results, reset_profile
    
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
    # FULL MODE: Real profiling implementations
    # =========================================================================
    import inspect
    import time
    import functools
    from typing import Dict, Any, Optional, Callable, Union
    from dataclasses import dataclass, field
    from contextlib import contextmanager

    # =========================================================================
    # Profile State
    # =========================================================================

    @dataclass
    class MarkerStats:
        """Accumulated stats for a single marker."""
        count: int = 0
        total_ms: float = 0.0
        min_ms: float = float('inf')
        max_ms: float = 0.0
        # Track parent markers for hierarchy
        parents: Dict[str, int] = field(default_factory=dict)  # parent_name -> count
        
        def record(self, elapsed_ms: float, parent: Optional[str] = None):
            self.count += 1
            self.total_ms += elapsed_ms
            self.min_ms = min(self.min_ms, elapsed_ms)
            self.max_ms = max(self.max_ms, elapsed_ms)
            if parent:
                self.parents[parent] = self.parents.get(parent, 0) + 1
        
        @property
        def avg_ms(self) -> float:
            return self.total_ms / self.count if self.count > 0 else 0.0
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'count': self.count,
                'total_ms': round(self.total_ms, 3),
                'avg_ms': round(self.avg_ms, 3),
                'min_ms': round(self.min_ms, 3) if self.min_ms != float('inf') else 0.0,
                'max_ms': round(self.max_ms, 3),
                'parents': dict(self.parents) if self.parents else {},
            }


    @dataclass 
    class _ProfileState:
        """Global profiling state."""
        enabled: bool = False
        use_houdini: bool = False  # Use Houdini perfMon if available
        markers: Dict[str, MarkerStats] = field(default_factory=dict)
        stack: list = field(default_factory=list)  # Stack of (name, start_time)
        houdini_stack: list = field(default_factory=list)  # Houdini event stack


    _state = _ProfileState()


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
        _state.enabled = enabled
        _state.use_houdini = use_houdini


    def is_profiling_enabled() -> bool:
        """Check if profiling is enabled."""
        return _state.enabled


    def reset_profile():
        """Reset all collected profile data and clean up any orphaned markers."""
        # Pop any orphaned markers from the stack (records partial timing)
        _cleanup_orphaned_markers()
        _state.markers.clear()
        _state.stack.clear()
        _state.houdini_stack.clear()


    def _cleanup_orphaned_markers():
        """Clean up any markers left on the stack (e.g., from exceptions)."""
        end_time = time.perf_counter()
        while _state.stack:
            name, start_time = _state.stack.pop()
            elapsed_ms = (end_time - start_time) * 1000
            parent = _state.stack[-1][0] if _state.stack else None
            if name not in _state.markers:
                _state.markers[name] = MarkerStats()
            _state.markers[name].record(elapsed_ms, parent)
        
        # Clean up Houdini events too
        while _state.houdini_stack:
            event = _state.houdini_stack.pop()
            if event:
                try:
                    event.stop()
                except Exception:
                    pass


    def get_profile_results() -> Dict[str, Dict[str, Any]]:
        """
        Get collected profile results.
        
        Returns:
            Dict mapping marker names to their stats:
            {
                'marker_name': {
                    'count': 10,
                    'total_ms': 52.3,
                    'avg_ms': 5.23,
                    'min_ms': 4.1,
                    'max_ms': 7.8,
                }
            }
        """
        return {name: stats.to_dict() for name, stats in _state.markers.items()}


    def _push_perf_marker(name: Optional[str] = None):
        """
        Start a performance marker (internal implementation).
        
        Args:
            name: Marker name. If None, uses the calling function's name.
                  Note: Passing None incurs overhead from inspect.stack().
        """
        if not _state.enabled:
            return
        
        # Get name from caller if not provided (expensive!)
        if name is None:
            try:
                stack = inspect.stack()
                name = stack[1].function
            except Exception:
                name = "unknown"
        
        # Push to our timing stack
        _state.stack.append((name, time.perf_counter()))
        
        # Also push to Houdini if enabled
        if _state.use_houdini:
            try:
                event = __import__("hou").perfMon.startEvent(name)
                _state.houdini_stack.append(event)
            except Exception:
                _state.houdini_stack.append(None)


    def _pop_perf_marker():
        """
        End the current performance marker and record the elapsed time (internal implementation).
        """
        if not _state.enabled:
            return
        
        end_time = time.perf_counter()
        
        # Pop from our stack and record timing
        if _state.stack:
            name, start_time = _state.stack.pop()
            elapsed_ms = (end_time - start_time) * 1000
            
            # Determine parent (what's now on top of the stack)
            parent = _state.stack[-1][0] if _state.stack else None
            
            # Get or create marker stats
            if name not in _state.markers:
                _state.markers[name] = MarkerStats()
            _state.markers[name].record(elapsed_ms, parent)
        
        # Pop Houdini event if enabled
        if _state.use_houdini and _state.houdini_stack:
            event = _state.houdini_stack.pop()
            if event:
                try:
                    event.stop()
                except Exception:
                    pass


    @contextmanager
    def perf_marker(name: Optional[str] = None):
        """
        Context manager for performance marking. Exception-safe.
        
        Usage:
            with perf_marker("my_section"):
                # ... do work ...
        """
        if name is None:
            try:
                stack = inspect.stack()
                name = stack[2].function  # Caller of the 'with' statement
            except Exception:
                name = "unknown"
        
        _push_perf_marker(name)
        try:
            yield
        finally:
            _pop_perf_marker()


    def profile(name_or_func: Union[str, Callable, None] = None) -> Callable:
        """
        Decorator for profiling functions. Exception-safe.
        
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
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not _state.enabled:
                    return func(*args, **kwargs)
                
                _push_perf_marker(marker_name)
                try:
                    return func(*args, **kwargs)
                finally:
                    _pop_perf_marker()
            
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

    # Old global for code that checks it directly
    PROFILE_ENABLED = False

    def _update_compat():
        """Update backwards-compat global."""
        global PROFILE_ENABLED
        PROFILE_ENABLED = _state.enabled
