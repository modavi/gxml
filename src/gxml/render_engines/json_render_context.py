"""
JSON Render Context for GXML.

This render context collects geometry data and outputs it as a JSON-serializable
dictionary, suitable for web viewers and other JSON-based consumers.
"""

from render_engines.base_render_context import BaseRenderContext


class JSONRenderContext(BaseRenderContext):
    """
    Render context that collects geometry data as JSON-serializable dict.
    
    Output format (via get_output()):
        {
            'panels': [
                {
                    'id': str,
                    'points': [[x, y, z], ...],
                    'position': [cx, cy, cz],  # center
                    'size': [sx, sy, sz],
                    'color': '#hex',
                    'geoKey': str | None,
                    'startPoint': [x, y, z] | None,
                    'endPoint': [x, y, z] | None,
                    'rotation': [rx, ry, rz] | None,
                },
                ...
            ],
            'lines': [
                {'id': str, 'points': [[x, y, z], ...], 'geoKey': str | None},
                ...
            ],
        }
    """
    
    DEFAULT_COLOR_PALETTE = [
        '#e94560', '#0f3460', '#16213e', '#533483', 
        '#1a1a2e', '#4a4e69', '#9a8c98', '#c9ada7',
        '#22223b', '#f2e9e4', '#4361ee', '#7209b7',
    ]
    
    def __init__(self, color_palette=None):
        self.panels = []
        self.lines = []
        self.current_element = None
        self._panel_data = {}
        self._color_palette = color_palette or self.DEFAULT_COLOR_PALETTE
        self._color_index = 0
        
    def _get_next_color(self):
        """Get the next color from the palette."""
        color = self._color_palette[self._color_index % len(self._color_palette)]
        self._color_index += 1
        return color
        
    def pre_render(self, element):
        """Called before rendering an element."""
        self.current_element = element
        
        self._panel_data = {
            'id': getattr(element, 'id', None),
            'subId': getattr(element, 'subId', None),
        }
        
        # Get color from element or assign from palette
        if hasattr(element, 'color') and element.color:
            self._panel_data['color'] = element.color
        else:
            self._panel_data['color'] = self._get_next_color()
        
        # For panels, get endpoint info
        if hasattr(element, 'transform_point'):
            try:
                start_pt = element.transform_point((0, 0.5, 0))
                end_pt = element.transform_point((1, 0.5, 0))
                
                self._panel_data['startPoint'] = [
                    float(start_pt[0]), 
                    float(start_pt[1]), 
                    float(start_pt[2]) if len(start_pt) > 2 else 0.0
                ]
                self._panel_data['endPoint'] = [
                    float(end_pt[0]), 
                    float(end_pt[1]), 
                    float(end_pt[2]) if len(end_pt) > 2 else 0.0
                ]
                
                if hasattr(element, 'rotation'):
                    self._panel_data['rotation'] = [
                        float(element.rotation[0]), 
                        float(element.rotation[1]), 
                        float(element.rotation[2])
                    ]
            except Exception:
                pass
    
    def create_poly(self, id, points, geoKey=None):
        """Create a polygon from points."""
        point_list = []
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for p in points:
            try:
                x, y = float(p[0]), float(p[1])
                z = float(p[2]) if len(p) > 2 else 0.0
            except (TypeError, KeyError):
                if hasattr(p, 'x'):
                    x, y, z = float(p.x), float(p.y), float(p.z)
                else:
                    x, y, z = float(p[0]), float(p[1]), 0.0
            
            point_list.append([x, y, z])
            
            # Track bounding box
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
            if z < min_z: min_z = z
            if z > max_z: max_z = z
        
        # Calculate center and size
        if point_list:
            center = [
                (min_x + max_x) / 2,
                (min_y + max_y) / 2,
                (min_z + max_z) / 2,
            ]
            size = [max_x - min_x, max_y - min_y, max_z - min_z]
        else:
            center = [0, 0, 0]
            size = [0, 0, 0]
        
        self.panels.append({
            'id': id,
            'points': point_list,
            'position': center,
            'size': size,
            'color': self._panel_data.get('color'),
            'geoKey': geoKey,
            'startPoint': self._panel_data.get('startPoint'),
            'endPoint': self._panel_data.get('endPoint'),
            'rotation': self._panel_data.get('rotation'),
        })
    
    def create_line(self, id, points, geoKey=None):
        """Create a line from points."""
        point_list = []
        for p in points:
            try:
                x, y = float(p[0]), float(p[1])
                z = float(p[2]) if len(p) > 2 else 0.0
            except (TypeError, KeyError):
                if hasattr(p, 'x'):
                    x, y, z = float(p.x), float(p.y), float(p.z)
                else:
                    x, y, z = float(p[0]), float(p[1]), 0.0
            point_list.append([x, y, z])
        
        self.lines.append({
            'id': id,
            'points': point_list,
            'geoKey': geoKey,
        })
    
    def get_or_create_geo(self, key):
        """For JSON export, return self."""
        return self
    
    def combine_all_geo(self):
        """No-op for JSON export."""
        pass
    
    def get_output(self) -> dict:
        """Get collected geometry as a dictionary."""
        return {
            'panels': self.panels,
            'lines': self.lines,
        }
    
    # Alias for backwards compatibility
    def to_dict(self) -> dict:
        """Alias for get_output()."""
        return self.get_output()


# Backwards compatibility alias
JSONRenderEngine = JSONRenderContext
