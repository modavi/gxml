"""
Binary Render Context for GXML.

This render context outputs geometry as packed binary data that can be
loaded directly into WebGL/Three.js Float32Arrays without JSON parsing.

Supports configurable features via flags:
- shared_vertices: Use shared/deduplicated vertices (GPU efficient)
- include_colors: Include per-polygon RGB colors
- include_endpoints: Include panel start/end points

Binary Format v4:
    Header (28 bytes):
        - Magic: 4 bytes "GXML"
        - Version: uint32 (4)
        - Flags: uint32 (bit 0 = shared_vertices, bit 1 = colors, bit 2 = endpoints)
        - Vertex count: uint32
        - Index count: uint32 (0 if not shared_vertices)
        - Polygon count: uint32
        - Reserved: uint32 (for future use)
    
    If shared_vertices mode (FLAG_SHARED_VERTICES):
        Vertices: vertex_count * 3 * float32
        Indices: index_count * uint32 (triangles)
        For each polygon:
            - ID length: uint16
            - Triangle count: uint16
            - Color RGB: 3x float32 (only if FLAG_COLORS)
            - Start point: 3x float32 (only if FLAG_ENDPOINTS)
            - End point: 3x float32 (only if FLAG_ENDPOINTS)
            - Panel ID: variable (UTF-8, padded to 4-byte alignment)
    
    If per-face mode (not shared_vertices):
        For each polygon:
            - ID length: uint16
            - Vertex count: uint16
            - Color RGB: 3x float32 (only if FLAG_COLORS)
            - Start point: 3x float32 (only if FLAG_ENDPOINTS)
            - End point: 3x float32 (only if FLAG_ENDPOINTS)
            - Panel ID: variable (UTF-8, padded to 4-byte alignment)
            - Vertices: vertex_count * 3 * float32
"""

import struct
import numpy as np
from render_engines.base_render_context import BaseRenderContext
from profiling import perf_marker


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
    
    Args:
        shared_vertices: If True, deduplicate/share vertices across faces (GPU efficient).
                         If False, each face gets its own unique vertices (default).
        include_colors: If True, include per-polygon RGB colors in output.
        include_endpoints: If True, include panel start/end points in output.
        color_palette: List of hex colors to cycle through for panels.
    
    Examples:
        # Simple mesh output (default)
        ctx = BinaryRenderContext()
        
        # GPU-efficient shared vertex mesh with colors
        ctx = BinaryRenderContext(shared_vertices=True, include_colors=True)
        
        # Full output with endpoints for visualization
        ctx = BinaryRenderContext(include_colors=True, include_endpoints=True)
    """
    
    MAGIC = b'GXML'
    VERSION = 4
    
    # Flag bits
    FLAG_SHARED_VERTICES = 0x01
    FLAG_COLORS = 0x02
    FLAG_ENDPOINTS = 0x04
    
    # Tolerance for vertex deduplication
    VERTEX_TOLERANCE = 1e-6
    
    DEFAULT_COLOR_PALETTE = [
        '#e94560', '#0f3460', '#16213e', '#533483', 
        '#1a1a2e', '#4a4e69', '#9a8c98', '#c9ada7',
        '#22223b', '#f2e9e4', '#4361ee', '#7209b7',
    ]
    
    def __init__(
        self, 
        shared_vertices: bool = False, 
        include_colors: bool = True,
        include_endpoints: bool = False,
        color_palette=None
    ):
        self.shared_vertices = shared_vertices
        self.include_colors = include_colors
        self.include_endpoints = include_endpoints
        self._color_palette = color_palette or self.DEFAULT_COLOR_PALETTE
        self._color_index = 0
        
        # Polygon storage
        # Each polygon: (id, color_rgb, vertices, start_point, end_point, triangle_count)
        self._polygons = []
        
        # Indexed mode storage
        self._vertices = []       # List of (x, y, z) tuples
        self._indices = []        # List of triangle indices
        self._vertex_map = {}     # For deduplication
        
        self.current_element = None
        self._panel_data = {}
        
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
        
        # Get color from element or palette
        if hasattr(element, 'color') and element.color:
            self._panel_data['color'] = element.color
        else:
            self._panel_data['color'] = self._get_next_color()
        
        # Extract endpoints if available
        if self.include_endpoints and hasattr(element, 'transform_point'):
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
    
    def _get_or_create_vertex(self, x: float, y: float, z: float) -> int:
        """Get index for a vertex, creating it if it doesn't exist (indexed mode only)."""
        # Use integer key for faster hashing
        scale = 1.0 / self.VERTEX_TOLERANCE
        key = (int(x * scale), int(y * scale), int(z * scale))
        
        if key in self._vertex_map:
            return self._vertex_map[key]
        
        idx = len(self._vertices)
        self._vertices.append((x, y, z))
        self._vertex_map[key] = idx
        return idx
        
    def create_poly(self, id: str, points, geoKey=None):
        """Create a polygon from points."""
        # Extract vertices
        verts = []
        for p in points:
            try:
                x, y = float(p[0]), float(p[1])
                z = float(p[2]) if len(p) > 2 else 0.0
            except (TypeError, KeyError):
                if hasattr(p, 'x'):
                    x, y, z = float(p.x), float(p.y), float(p.z)
                else:
                    x, y, z = float(p[0]), float(p[1]), 0.0
            verts.append((x, y, z))
        
        if len(verts) < 3:
            return
        
        panel_id = id or self._panel_data.get('id', '')
        color_rgb = hex_to_rgb(self._panel_data.get('color', '#888888'))
        start_point = self._panel_data.get('startPoint')
        end_point = self._panel_data.get('endPoint')
        
        if self.shared_vertices:
            # Shared vertices mode: deduplicate vertices and add triangles
            with perf_marker('shared_vertex_dedup'):
                indices = [self._get_or_create_vertex(*v) for v in verts]
            
            # Fan triangulation
            triangle_count = len(indices) - 2
            for i in range(1, len(indices) - 1):
                self._indices.extend([indices[0], indices[i], indices[i + 1]])
            
            self._polygons.append((panel_id, color_rgb, None, start_point, end_point, triangle_count))
        else:
            # Per-polygon mode: store vertices directly
            self._polygons.append((panel_id, color_rgb, verts, start_point, end_point, 0))
    
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
        with perf_marker('binary_to_bytes'):
            return self.to_bytes()
    
    def to_bytes(self) -> bytes:
        """Convert collected geometry to binary format."""
        parts = []
        
        # Build flags
        flags = 0
        if self.shared_vertices:
            flags |= self.FLAG_SHARED_VERTICES
        if self.include_colors:
            flags |= self.FLAG_COLORS
        if self.include_endpoints:
            flags |= self.FLAG_ENDPOINTS
        
        # Calculate counts
        if self.shared_vertices:
            vertex_count = len(self._vertices)
            index_count = len(self._indices)
        else:
            vertex_count = sum(len(verts) for _, _, verts, _, _, _ in self._polygons if verts)
            index_count = 0
        
        # Header (28 bytes)
        header = struct.pack('<4sIIIIII',
            self.MAGIC,
            self.VERSION,
            flags,
            vertex_count,
            index_count,
            len(self._polygons),
            0  # Reserved
        )
        parts.append(header)
        
        if self.shared_vertices:
            self._write_shared_vertex_data(parts)
        else:
            self._write_per_face_data(parts)
        
        return b''.join(parts)
    
    def _write_shared_vertex_data(self, parts: list):
        """Write shared vertices mode data."""
        # Vertices (float32 * 3 per vertex)
        if self._vertices:
            vertices_array = np.array(self._vertices, dtype=np.float32)
            parts.append(vertices_array.tobytes())
        
        # Indices (uint32)
        if self._indices:
            indices_array = np.array(self._indices, dtype=np.uint32)
            parts.append(indices_array.tobytes())
        
        # Polygon metadata
        for panel_id, color_rgb, _, start_point, end_point, triangle_count in self._polygons:
            id_bytes = panel_id.encode('utf-8')
            
            # Polygon header: id_len, triangle_count
            parts.append(struct.pack('<HH', len(id_bytes), triangle_count))
            
            # Optional color
            if self.include_colors:
                parts.append(struct.pack('<3f', *color_rgb))
            
            # Optional endpoints
            if self.include_endpoints:
                if start_point and end_point:
                    parts.append(struct.pack('<6f', 
                        start_point[0], start_point[1], start_point[2],
                        end_point[0], end_point[1], end_point[2]
                    ))
                else:
                    parts.append(struct.pack('<6f', 0, 0, 0, 0, 0, 0))
            
            # Panel ID
            parts.append(id_bytes)
            
            # Pad to 4-byte alignment
            header_size = 4 + (12 if self.include_colors else 0) + (24 if self.include_endpoints else 0)
            padding = (4 - ((header_size + len(id_bytes)) % 4)) % 4
            if padding:
                parts.append(b'\x00' * padding)
    
    def _write_per_face_data(self, parts: list):
        """Write per-face mode data."""
        for panel_id, color_rgb, verts, start_point, end_point, _ in self._polygons:
            id_bytes = panel_id.encode('utf-8')
            vertex_count = len(verts) if verts else 0
            
            # Polygon header: id_len, vertex_count
            parts.append(struct.pack('<HH', len(id_bytes), vertex_count))
            
            # Optional color
            if self.include_colors:
                parts.append(struct.pack('<3f', *color_rgb))
            
            # Optional endpoints
            if self.include_endpoints:
                if start_point and end_point:
                    parts.append(struct.pack('<6f',
                        start_point[0], start_point[1], start_point[2],
                        end_point[0], end_point[1], end_point[2]
                    ))
                else:
                    parts.append(struct.pack('<6f', 0, 0, 0, 0, 0, 0))
            
            # Panel ID
            parts.append(id_bytes)
            
            # Pad to 4-byte alignment
            header_size = 4 + (12 if self.include_colors else 0) + (24 if self.include_endpoints else 0)
            padding = (4 - ((header_size + len(id_bytes)) % 4)) % 4
            if padding:
                parts.append(b'\x00' * padding)
            
            # Vertices
            if verts:
                vertices_flat = []
                for x, y, z in verts:
                    vertices_flat.extend([x, y, z])
                vertices_packed = struct.pack(f'<{len(vertices_flat)}f', *vertices_flat)
                parts.append(vertices_packed)
    
    def get_stats(self) -> dict:
        """Get statistics about the mesh."""
        if self.shared_vertices:
            return {
                'shared_vertices': True,
                'vertex_count': len(self._vertices),
                'index_count': len(self._indices),
                'triangle_count': len(self._indices) // 3,
                'polygon_count': len(self._polygons),
            }
        else:
            total_verts = sum(len(verts) for _, _, verts, _, _, _ in self._polygons if verts)
            return {
                'shared_vertices': False,
                'vertex_count': total_verts,
                'polygon_count': len(self._polygons),
            }
    
    def to_dict(self) -> dict:
        """Support dict output for compatibility/debugging."""
        panels = []
        for panel_id, color_rgb, verts, start_point, end_point, _ in self._polygons:
            r, g, b = color_rgb
            color_hex = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            
            panel_dict = {
                'id': panel_id,
                'color': color_hex,
            }
            
            if verts:
                panel_dict['points'] = [[x, y, z] for x, y, z in verts]
            
            if start_point:
                panel_dict['startPoint'] = list(start_point)
            if end_point:
                panel_dict['endPoint'] = list(end_point)
            
            panels.append(panel_dict)
        
        return {'panels': panels, 'lines': []}
