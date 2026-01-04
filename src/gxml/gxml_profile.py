"""
Provides a way to measure performance of code blocks.

Usage:
    from gxml_profile import push_perf_marker, pop_perf_marker, get_profile_results, reset_profile
    
    # Enable profiling
    enable_profiling(True)
    
    # In your code, wrap sections with markers:
    push_perf_marker("my_function")
    # ... do work ...
    pop_perf_marker()
    
    # Get results
    results = get_profile_results()
    # {'my_function': {'count': 1, 'total_ms': 5.2, 'min_ms': 5.2, 'max_ms': 5.2}}
    
    # Reset for next run
    reset_profile()
"""

import inspect
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

# =============================================================================
# Profile State
# =============================================================================

@dataclass
class MarkerStats:
    """Accumulated stats for a single marker."""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    
    def record(self, elapsed_ms: float):
        self.count += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)
    
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


# =============================================================================
# Public API
# =============================================================================

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
    """Reset all collected profile data."""
    _state.markers.clear()
    _state.stack.clear()
    _state.houdini_stack.clear()


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


def push_perf_marker(name: Optional[str] = None):
    """
    Start a performance marker.
    
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


def pop_perf_marker():
    """
    End the current performance marker and record the elapsed time.
    """
    if not _state.enabled:
        return
    
    end_time = time.perf_counter()
    
    # Pop from our stack and record timing
    if _state.stack:
        name, start_time = _state.stack.pop()
        elapsed_ms = (end_time - start_time) * 1000
        
        # Get or create marker stats
        if name not in _state.markers:
            _state.markers[name] = MarkerStats()
        _state.markers[name].record(elapsed_ms)
    
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
    Context manager for performance marking.
    
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
    
    push_perf_marker(name)
    try:
        yield
    finally:
        pop_perf_marker()


# =============================================================================
# Backwards Compatibility
# =============================================================================

# Old global for code that checks it directly
PROFILE_ENABLED = False

def _update_compat():
    """Update backwards-compat global."""
    global PROFILE_ENABLED
    PROFILE_ENABLED = _state.enabled
