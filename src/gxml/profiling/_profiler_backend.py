"""
Python profiler backends.

Contains the NoOp and pure Python profiler implementations.
"""

import time
import functools
from typing import List, Tuple, Callable, Dict, Any

# Local reference for speed
_perf = time.perf_counter


# =============================================================================
# NoOp Backend (Zero overhead when profiling compiled out)
# =============================================================================

class _NoOpMarker:
    """No-op context manager that can be reused."""
    __slots__ = ()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        return False


class NoOpBackend:
    """Zero-overhead backend when profiling is compiled out."""
    
    def __init__(self):
        self._noop_marker = _NoOpMarker()
    
    def register_marker(self, name: str) -> int:
        return 0
    
    def clear(self) -> None:
        pass
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        return {}
    
    def create_perf_marker(self, name: str):
        return self._noop_marker
    
    def create_profiled_function(self, func: Callable, name: str) -> Callable:
        return func  # Return unchanged


# =============================================================================
# Python Backend (Fallback)
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


class PythonBackend:
    """Pure Python profiler backend - fully self-contained."""
    
    def __init__(self):
        self._events: List[Tuple[int, float]] = []
        self._id_to_name: List[str] = []
        self._name_to_id: Dict[str, int] = {}
    
    def register_marker(self, name: str) -> int:
        """Register a marker name and get its integer ID."""
        if name in self._name_to_id:
            return self._name_to_id[name]
        
        marker_id = len(self._id_to_name)
        self._id_to_name.append(name)
        self._name_to_id[name] = marker_id
        return marker_id
    
    def clear(self) -> None:
        self._events.clear()
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Process events and return marker statistics."""
        markers: Dict[str, Dict[str, Any]] = {}
        stack: List[Tuple[int, float, int]] = []  # (marker_id, start_time, parent_id)
        
        for marker_id, timestamp in self._events:
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
        marker_id = self.register_marker(name)
        return _PythonPerfMarker(marker_id, self._events)
    
    def create_profiled_function(self, func: Callable, name: str) -> Callable:
        """Create a wrapped function that records entry/exit times."""
        marker_id = self.register_marker(name)
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
