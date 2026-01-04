"""
Indexed Render Context for GXML.

This render context outputs geometry as an indexed mesh with shared vertices
and triangle indices. This is more efficient for WebGL rendering than the
per-panel polygon format.

Binary Format (GXMF v1):
    Header (20 bytes):
        - Magic: 4 bytes "GXMF"
        - Version: uint32 (1)
        - Vertex count: uint32
        - Index count: uint32
        - Quad count: uint32
    
    Vertices: vertex_count * 3 * float32
    Indices: index_count * uint32 (triangles)
    
    For each quad:
        - ID length: uint16
        - Panel ID: variable (UTF-8, padded to 4-byte alignment)
"""

import struct
import numpy as np
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


class IndexedRenderContext(BaseRenderContext):
    """
    Render context that outputs geometry as indexed mesh data.
    
    Produces shared vertices with triangle indices, which is more efficient
    for GPU rendering than per-panel vertex lists.
    """
    
    MAGIC = b'GXMF'
    VERSION = 1
    
    # Tolerance for vertex deduplication (positions within this distance are merged)
    VERTEX_TOLERANCE = 1e-6
    
    def __init__(self):
        # Use lists during collection, convert to numpy at end
        self._vertices = []  # List of (x, y, z) tuples
        self._indices = []   # List of triangle indices
        self._panel_ids = [] # One per quad
        self._quad_count = 0
        
        # Vertex deduplication: map from rounded position tuple to index
        self._vertex_map = {}
        
        self.current_element = None
        self._panel_data = {}
        
    def pre_render(self, element):
        """Called before rendering an element."""
        self.current_element = element
        
        self._panel_data = {
            'id': getattr(element, 'id', None) or '',
        }
        
    def _get_or_create_vertex(self, x: float, y: float, z: float) -> int:
        """Get index for a vertex, creating it if it doesn't exist."""
        # Round to tolerance for deduplication
        key = (
            round(x / self.VERTEX_TOLERANCE) * self.VERTEX_TOLERANCE,
            round(y / self.VERTEX_TOLERANCE) * self.VERTEX_TOLERANCE,
            round(z / self.VERTEX_TOLERANCE) * self.VERTEX_TOLERANCE,
        )
        
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
        
        # Get or create vertex indices (with deduplication)
        indices = [self._get_or_create_vertex(*v) for v in verts]
        
        # Triangulate polygon (fan triangulation works for convex quads)
        for i in range(1, len(indices) - 1):
            self._indices.extend([indices[0], indices[i], indices[i + 1]])
        
        # Track panel ID for this quad
        panel_id = id or self._panel_data.get('id', '')
        self._panel_ids.append(panel_id)
        self._quad_count += 1
    
    def create_line(self, id, points, geoKey=None):
        """Lines not currently included in indexed output."""
        pass
    
    def get_or_create_geo(self, key):
        """For indexed export, return self."""
        return self
    
    def combine_all_geo(self):
        """No-op for indexed export."""
        pass
    
    def get_output(self) -> bytes:
        """Get collected geometry as binary data."""
        return self.to_bytes()
    
    def to_bytes(self) -> bytes:
        """Convert collected geometry to GXMF binary format."""
        parts = []
        
        vertex_count = len(self._vertices)
        index_count = len(self._indices)
        
        # Header
        header = struct.pack('<4sIIII',
            self.MAGIC,
            self.VERSION,
            vertex_count,
            index_count,
            self._quad_count
        )
        parts.append(header)
        
        # Vertices (float32 * 3 per vertex)
        if vertex_count > 0:
            vertices_array = np.array(self._vertices, dtype=np.float32)
            parts.append(vertices_array.tobytes())
        
        # Indices (uint32)
        if index_count > 0:
            indices_array = np.array(self._indices, dtype=np.uint32)
            parts.append(indices_array.tobytes())
        
        # Panel IDs for each quad
        for panel_id in self._panel_ids:
            id_bytes = panel_id.encode('utf-8')
            parts.append(struct.pack('<H', len(id_bytes)))
            parts.append(id_bytes)
            # Pad to 4-byte alignment
            padding = (4 - ((2 + len(id_bytes)) % 4)) % 4
            if padding:
                parts.append(b'\x00' * padding)
        
        return b''.join(parts)
    
    def get_stats(self) -> dict:
        """Get statistics about the indexed mesh."""
        return {
            'vertex_count': len(self._vertices),
            'index_count': len(self._indices),
            'triangle_count': len(self._indices) // 3,
            'quad_count': self._quad_count,
            'unique_panels': len(set(self._panel_ids)),
        }
