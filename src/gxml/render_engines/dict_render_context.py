"""
Dictionary Render Context for GXML.

This render context collects geometry data into a Python dictionary,
suitable for JSON serialization or direct consumption by Python code.
"""

from render_engines.base_render_context import BaseRenderContext


class DictRenderContext(BaseRenderContext):
    """
    Render context that collects geometry into a dictionary.
    
    Output format:
        {
            'panels': [
                {
                    'id': str,
                    'polygons': [[[x, y, z], ...], ...],
                    'lines': [[[x, y, z], ...], ...],
                }
            ],
            'stats': {
                'polygon_count': int,
                'line_count': int,
                'vertex_count': int,
            }
        }
    """
    
    def __init__(self):
        self._panels = {}  # id -> panel data
        self._current_id = None
        self._polygon_count = 0
        self._line_count = 0
        self._vertex_count = 0
    
    def pre_render(self, element):
        """Called before rendering an element."""
        self._current_id = getattr(element, 'id', None) or str(id(element))
        
        if self._current_id not in self._panels:
            self._panels[self._current_id] = {
                'id': self._current_id,
                'polygons': [],
                'lines': [],
            }
    
    def create_poly(self, id, points, geoKey=None):
        """Create a polygon from points."""
        point_list = []
        for p in points:
            # Handle various point formats
            try:
                x, y, z = float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0
            except (TypeError, KeyError):
                if hasattr(p, 'x'):
                    x, y, z = float(p.x), float(p.y), float(p.z)
                else:
                    x, y, z = float(p[0]), float(p[1]), 0.0
            point_list.append([x, y, z])
        
        if self._current_id and self._current_id in self._panels:
            self._panels[self._current_id]['polygons'].append(point_list)
        
        self._polygon_count += 1
        self._vertex_count += len(point_list)
    
    def create_line(self, id, points, geoKey=None):
        """Create a line from points."""
        point_list = []
        for p in points:
            try:
                x, y, z = float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0
            except (TypeError, KeyError):
                if hasattr(p, 'x'):
                    x, y, z = float(p.x), float(p.y), float(p.z)
                else:
                    x, y, z = float(p[0]), float(p[1]), 0.0
            point_list.append([x, y, z])
        
        if self._current_id and self._current_id in self._panels:
            self._panels[self._current_id]['lines'].append(point_list)
        
        self._line_count += 1
        self._vertex_count += len(point_list)
    
    def get_output(self) -> dict:
        """Get the collected geometry as a dictionary."""
        return {
            'panels': list(self._panels.values()),
            'stats': {
                'polygon_count': self._polygon_count,
                'line_count': self._line_count,
                'vertex_count': self._vertex_count,
            }
        }
