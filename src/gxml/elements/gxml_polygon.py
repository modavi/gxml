"""
    A polygon is an N-vertex face that supports parametric coordinate referencing.
    
    Unlike panels (which are always quads with specific axis semantics), polygons
    are arbitrary planar shapes defined by their vertices. They are used for:
    - Miter cap fills at panel intersections
    - Custom non-quad shapes
    - Any face that doesn't fit the panel paradigm
    
    Parametric Coordinates:
    - Vertices are stored in CCW order when viewed from the front (normal direction)
    - Edge references use (edge_index, t) where t is 0-1 along that edge
    - Interior points can use barycentric or other interpolation schemes
"""

from typing import List, Tuple, Optional
import math
from elements.gxml_base_element import GXMLLayoutElement
from gxml_types import *

# Try C extension, fall back to pure Python
try:
    from mathutils._vec3 import cross_product as _cross, normalize as _normalize, distance as _distance
except ImportError:
    # Pure Python fallbacks
    def _cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        )
    
    def _normalize(v):
        length = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        if length < 1e-10:
            return (0.0, 0.0, 0.0)
        inv_len = 1.0 / length
        return (v[0] * inv_len, v[1] * inv_len, v[2] * inv_len)
    
    def _distance(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)


def _vec_sub(a, b):
    """Subtract two 3D vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def _vec_add(a, b):
    """Add two 3D vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def _vec_scale(v, s):
    """Scale a 3D vector."""
    return (v[0] * s, v[1] * s, v[2] * s)

def _vec_length(v):
    """Length of a 3D vector."""
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


class GXMLPolygon(GXMLLayoutElement):
    """
    An N-vertex polygon face with parametric coordinate support.
    
    Vertices are stored in counter-clockwise order when viewed from the front.
    The polygon is assumed to be planar.
    
    Attributes:
        vertices: List of 3D points defining the polygon in CCW order
        normal: The face normal (computed from vertices or explicitly set)
    """
    
    def __init__(self, vertices: Optional[List[tuple]] = None):
        super().__init__()
        self._vertices: List[tuple] = []
        self._normal: Optional[tuple] = None
        
        if vertices:
            self.set_vertices(vertices)
    
    @property
    def vertices(self) -> List[tuple]:
        """Get the polygon vertices in CCW order."""
        return self._vertices
    
    @property
    def vertex_count(self) -> int:
        """Number of vertices in the polygon."""
        return len(self._vertices)
    
    @property
    def normal(self) -> Optional[tuple]:
        """Get the face normal, computed from first 3 vertices if not set."""
        if self._normal is not None:
            return self._normal
        if len(self._vertices) >= 3:
            return self._compute_normal()
        return None
    
    @normal.setter
    def normal(self, value):
        """Explicitly set the face normal."""
        v = (float(value[0]), float(value[1]), float(value[2]))
        length = _vec_length(v)
        if length > 1e-10:
            self._normal = _vec_scale(v, 1.0 / length)
        else:
            self._normal = v
    
    def set_vertices(self, vertices: List) -> 'GXMLPolygon':
        """
        Set the polygon vertices.
        
        Args:
            vertices: List of 3D points (as lists, tuples, or np.arrays)
            
        Returns:
            Self for chaining
        """
        self._vertices = [tuple(float(c) for c in v) for v in vertices]
        self._normal = None  # Reset cached normal
        return self
    
    def add_vertex(self, vertex) -> 'GXMLPolygon':
        """
        Add a vertex to the polygon.
        
        Args:
            vertex: 3D point to add
            
        Returns:
            Self for chaining
        """
        self._vertices.append(tuple(float(c) for c in vertex))
        self._normal = None
        return self
    
    def _compute_normal(self) -> tuple:
        """Compute the face normal from the first 3 vertices."""
        if len(self._vertices) < 3:
            return (0.0, 1.0, 0.0)  # Default up
        
        v0, v1, v2 = self._vertices[0], self._vertices[1], self._vertices[2]
        edge1 = _vec_sub(v1, v0)
        edge2 = _vec_sub(v2, v0)
        normal = _cross(edge1, edge2)
        length = _vec_length(normal)
        if length > 1e-10:
            return _vec_scale(normal, 1.0 / length)
        return (0.0, 1.0, 0.0)
    
    # -------------------------------------------------------------------------
    # Parametric coordinate access
    # -------------------------------------------------------------------------
    
    def get_vertex(self, index: int) -> tuple:
        """
        Get a vertex by index.
        
        Args:
            index: Vertex index (wraps around)
            
        Returns:
            The vertex position
        """
        return self._vertices[index % len(self._vertices)]
    
    def get_edge(self, edge_index: int) -> Tuple[tuple, tuple]:
        """
        Get the start and end points of an edge.
        
        Args:
            edge_index: Edge index (edge i connects vertex i to vertex i+1)
            
        Returns:
            Tuple of (start_point, end_point)
        """
        n = len(self._vertices)
        return (self._vertices[edge_index % n], 
                self._vertices[(edge_index + 1) % n])
    
    def point_on_edge(self, edge_index: int, t: float) -> tuple:
        """
        Get a point along an edge using parametric coordinate.
        
        Args:
            edge_index: Which edge (0 to vertex_count-1)
            t: Parameter from 0 (start vertex) to 1 (end vertex)
            
        Returns:
            The interpolated point
        """
        start, end = self.get_edge(edge_index)
        diff = _vec_sub(end, start)
        return _vec_add(start, _vec_scale(diff, t))
    
    def get_centroid(self) -> tuple:
        """
        Get the centroid (average) of all vertices.
        
        Returns:
            The centroid point
        """
        if not self._vertices:
            return (0.0, 0.0, 0.0)
        n = len(self._vertices)
        sx = sum(v[0] for v in self._vertices) / n
        sy = sum(v[1] for v in self._vertices) / n
        sz = sum(v[2] for v in self._vertices) / n
        return (sx, sy, sz)
    
    def point_from_barycentric(self, weights: List[float]) -> tuple:
        """
        Get a point using barycentric-style weights for each vertex.
        
        Args:
            weights: Weight for each vertex (will be normalized)
            
        Returns:
            The weighted average point
        """
        if len(weights) != len(self._vertices):
            raise ValueError(f"Expected {len(self._vertices)} weights, got {len(weights)}")
        
        total = sum(weights)
        if abs(total) < 1e-10:
            return self.get_centroid()
        
        rx, ry, rz = 0.0, 0.0, 0.0
        for w, v in zip(weights, self._vertices):
            factor = w / total
            rx += factor * v[0]
            ry += factor * v[1]
            rz += factor * v[2]
        return (rx, ry, rz)
    
    # -------------------------------------------------------------------------
    # Geometric queries
    # -------------------------------------------------------------------------
    
    def get_edge_length(self, edge_index: int) -> float:
        """Get the length of an edge."""
        start, end = self.get_edge(edge_index)
        return _distance(start, end)
    
    def get_perimeter(self) -> float:
        """Get the total perimeter length."""
        return sum(self.get_edge_length(i) for i in range(len(self._vertices)))

    
    def get_area(self) -> float:
        """
        Calculate the area of the polygon using the shoelace formula.
        Assumes the polygon is planar.
        """
        if len(self._vertices) < 3:
            return 0.0
        
        # Project onto the plane perpendicular to normal for 2D calculation
        normal = self.normal
        if normal is None:
            return 0.0
        
        # Find the dominant axis to project out
        abs_n = (abs(normal[0]), abs(normal[1]), abs(normal[2]))
        if abs_n[0] >= abs_n[1] and abs_n[0] >= abs_n[2]:
            # Project onto YZ plane
            coords = [(v[1], v[2]) for v in self._vertices]
        elif abs_n[1] >= abs_n[2]:
            # Project onto XZ plane
            coords = [(v[0], v[2]) for v in self._vertices]
        else:
            # Project onto XY plane
            coords = [(v[0], v[1]) for v in self._vertices]
        
        # Shoelace formula
        n = len(coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        
        return abs(area) / 2.0
    
    # -------------------------------------------------------------------------
    # Transformation
    # -------------------------------------------------------------------------
    
    def transform_point(self, point):
        """Transform a point from local to world space."""
        return super().transform_point(point)
    
    def get_world_vertices(self) -> List[tuple]:
        """Get all vertices transformed to world space."""
        return [self.transform_point(v) for v in self._vertices]
    
    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    
    def render(self, renderContext):
        """Render the polygon."""
        if len(self._vertices) < 3:
            return
        
        # Transform vertices to world space
        world_verts = self.get_world_vertices()
        
        # Create the polygon
        poly_id = f"{self.id}-{self.subId}" if self.subId else self.id
        renderContext.create_poly(poly_id, world_verts)
    
    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------
    
    @staticmethod
    def from_points(points: List, normal: Optional[tuple] = None) -> 'GXMLPolygon':
        """
        Create a polygon from a list of points.
        
        Args:
            points: List of 3D points in CCW order
            normal: Optional explicit normal
            
        Returns:
            New GXMLPolygon instance
        """
        poly = GXMLPolygon(points)
        if normal is not None:
            poly.normal = normal
        return poly
    
    @staticmethod
    def triangle(p0, p1, p2) -> 'GXMLPolygon':
        """Create a triangle from 3 points."""
        return GXMLPolygon([p0, p1, p2])
    
    @staticmethod
    def quad(p0, p1, p2, p3) -> 'GXMLPolygon':
        """Create a quad from 4 points in CCW order."""
        return GXMLPolygon([p0, p1, p2, p3])
