"""
Binary Render Context for GXML.

This render context outputs geometry as packed binary data that can be
loaded directly into WebGL/Three.js Float32Arrays without JSON parsing.

Binary Format v2:
    Header (16 bytes):
        - Magic: 4 bytes "GXML"
        - Version: uint32 (2)
        - Panel count: uint32
        - Total vertex count: uint32
    
    For each panel:
        Panel header (20 bytes + optional 24 bytes for endpoints):
            - ID length: uint16
            - Vertex count: uint16
            - Color RGB: 3x float32 (12 bytes)
            - Has endpoints: uint8 (1 = yes, 0 = no)
            - Reserved: 3 bytes
            - Start point: 3x float32 (12 bytes) - only if has_endpoints
            - End point: 3x float32 (12 bytes) - only if has_endpoints
        Panel ID: variable (ID length bytes, UTF-8, padded to 4-byte alignment)
        Vertices: vertex_count * 3 * float32
"""

import struct
from render_engines.base_render_context import BaseRenderContext


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB floats (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)
    return (0.5, 0.5, 0.5)


class BinaryRenderContext(BaseRenderContext):
    """
    Render context that outputs geometry as packed binary data.
    
    Optimized for direct consumption by WebGL/Three.js without JSON parsing.
    """
    
    MAGIC = b'GXML'
    VERSION = 2
    
    DEFAULT_COLOR_PALETTE = [
        '#e94560', '#0f3460', '#16213e', '#533483', 
        '#1a1a2e', '#4a4e69', '#9a8c98', '#c9ada7',
        '#22223b', '#f2e9e4', '#4361ee', '#7209b7',
    ]
    
    def __init__(self, color_palette=None):
        self.panels = []  # List of (id, color_rgb, vertices_flat, start_point, end_point)
        self.current_element = None
        self._panel_data = {}
        self._color_palette = color_palette or self.DEFAULT_COLOR_PALETTE
        self._color_index = 0
        
    def _get_next_color(self) -> str:
        """Get the next color from the palette."""
        color = self._color_palette[self._color_index % len(self._color_palette)]
        self._color_index += 1
        return color
        
    def pre_render(self, element):
        """Called before rendering an element."""
        self.current_element = element
        
        self._panel_data = {
            'id': getattr(element, 'id', None) or '',
            'startPoint': None,
            'endPoint': None,
        }
        
        if hasattr(element, 'color') and element.color:
            self._panel_data['color'] = element.color
        else:
            self._panel_data['color'] = self._get_next_color()
        
        if hasattr(element, 'transform_point'):
            try:
                start_pt = element.transform_point((0, 0.5, 0))
                end_pt = element.transform_point((1, 0.5, 0))
                
                self._panel_data['startPoint'] = (
                    float(start_pt[0]), 
                    float(start_pt[1]), 
                    float(start_pt[2]) if len(start_pt) > 2 else 0.0
                )
                self._panel_data['endPoint'] = (
                    float(end_pt[0]), 
                    float(end_pt[1]), 
                    float(end_pt[2]) if len(end_pt) > 2 else 0.0
                )
            except Exception:
                pass
        
    def create_poly(self, id: str, points, geoKey=None):
        """Create a polygon from points."""
        vertices_flat = []
        for p in points:
            try:
                x, y = float(p[0]), float(p[1])
                z = float(p[2]) if len(p) > 2 else 0.0
            except (TypeError, KeyError):
                if hasattr(p, 'x'):
                    x, y, z = float(p.x), float(p.y), float(p.z)
                else:
                    x, y, z = float(p[0]), float(p[1]), 0.0
            vertices_flat.extend([x, y, z])
        
        color_rgb = hex_to_rgb(self._panel_data.get('color', '#888888'))
        panel_id = id or ''
        start_point = self._panel_data.get('startPoint')
        end_point = self._panel_data.get('endPoint')
        
        self.panels.append((panel_id, color_rgb, vertices_flat, start_point, end_point))
    
    def create_line(self, id, points, geoKey=None):
        """Lines not currently included in binary output."""
        pass
    
    def get_or_create_geo(self, key):
        """For binary export, return self."""
        return self
    
    def combine_all_geo(self):
        """No-op for binary export."""
        pass
    
    def get_output(self) -> bytes:
        """Get collected geometry as binary data."""
        return self.to_bytes()
    
    def to_bytes(self) -> bytes:
        """Convert collected geometry to binary format."""
        parts = []
        
        total_vertices = sum(len(v) // 3 for _, _, v, _, _ in self.panels)
        
        # Header
        header = struct.pack('<4sIII', 
            self.MAGIC,
            self.VERSION,
            len(self.panels),
            total_vertices
        )
        parts.append(header)
        
        # Each panel
        for panel_id, color_rgb, vertices_flat, start_point, end_point in self.panels:
            id_bytes = panel_id.encode('utf-8')
            vertex_count = len(vertices_flat) // 3
            has_endpoints = 1 if (start_point and end_point) else 0
            
            # Panel header
            panel_header = struct.pack('<HH3fB3x',
                len(id_bytes),
                vertex_count,
                color_rgb[0], color_rgb[1], color_rgb[2],
                has_endpoints
            )
            parts.append(panel_header)
            
            # Endpoint data
            if has_endpoints:
                endpoints = struct.pack('<6f',
                    start_point[0], start_point[1], start_point[2],
                    end_point[0], end_point[1], end_point[2]
                )
                parts.append(endpoints)
            
            # Panel ID
            parts.append(id_bytes)
            
            # Pad to 4-byte alignment
            padding_needed = (4 - (len(id_bytes) % 4)) % 4
            if padding_needed:
                parts.append(b'\x00' * padding_needed)
            
            # Vertices
            vertices_packed = struct.pack(f'<{len(vertices_flat)}f', *vertices_flat)
            parts.append(vertices_packed)
        
        return b''.join(parts)
    
    def to_dict(self) -> dict:
        """Also support dict output for compatibility."""
        panels = []
        for panel_id, color_rgb, vertices_flat, start_point, end_point in self.panels:
            points = []
            for i in range(0, len(vertices_flat), 3):
                points.append([vertices_flat[i], vertices_flat[i+1], vertices_flat[i+2]])
            
            r, g, b = color_rgb
            color_hex = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            
            panel_dict = {
                'id': panel_id,
                'points': points,
                'color': color_hex,
            }
            
            if start_point:
                panel_dict['startPoint'] = list(start_point)
            if end_point:
                panel_dict['endPoint'] = list(end_point)
            
            panels.append(panel_dict)
        
        return {'panels': panels, 'lines': []}


# Backwards compatibility alias
BinaryRenderEngine = BinaryRenderContext
