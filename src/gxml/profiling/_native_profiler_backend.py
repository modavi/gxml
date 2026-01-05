"""
Native (C extension) profiler backend.

Provides high-performance profiling using the _c_profiler extension.
This module imports the C extension directly - if import fails, this module
should not be used (the selector in _backend.py handles the fallback).
"""

from typing import Dict, Any, Callable, List

# Import the C extension - this will raise ImportError if not available
from . import _c_profiler


class CBackend:
    """C extension profiler backend - faster hot path."""
    
    def __init__(self):
        self._c_clear_events = _c_profiler.clear_events
        self._c_get_events = _c_profiler.get_events
        self._c_create_perf_marker = _c_profiler.create_perf_marker
        self._c_create_profiled_function = _c_profiler.create_profiled_function
        # Still need Python-side marker registry for name lookups
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
        self._c_clear_events()
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Process events and return marker statistics."""
        markers: Dict[str, Dict[str, Any]] = {}
        stack: List[tuple] = []
        
        for marker_id, timestamp in self._c_get_events():
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
        return self._c_create_perf_marker(marker_id)
    
    def create_profiled_function(self, func: Callable, name: str) -> Callable:
        marker_id = self.register_marker(name)
        return self._c_create_profiled_function(func, marker_id)
