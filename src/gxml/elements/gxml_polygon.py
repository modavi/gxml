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
import numpy as np
from elements.gxml_base_element import GXMLLayoutElement
from gxml_types import *


class GXMLPolygon(GXMLLayoutElement):
    """
    An N-vertex polygon face with parametric coordinate support.
    
    Vertices are stored in counter-clockwise order when viewed from the front.
    The polygon is assumed to be planar.
    
    Attributes:
        vertices: List of 3D points defining the polygon in CCW order
        normal: The face normal (computed from vertices or explicitly set)
    """
    
    def __init__(self, vertices: Optional[List[np.ndarray]] = None):
        super().__init__()
        self._vertices: List[np.ndarray] = []
        self._normal: Optional[np.ndarray] = None
        
        if vertices:
            self.set_vertices(vertices)
    
    @property
    def vertices(self) -> List[np.ndarray]:
        """Get the polygon vertices in CCW order."""
        return self._vertices
    
    @property
    def vertex_count(self) -> int:
        """Number of vertices in the polygon."""
        return len(self._vertices)
    
    @property
    def normal(self) -> Optional[np.ndarray]:
        """Get the face normal, computed from first 3 vertices if not set."""
        if self._normal is not None:
            return self._normal
        if len(self._vertices) >= 3:
            return self._compute_normal()
        return None
    
    @normal.setter
    def normal(self, value: np.ndarray):
        """Explicitly set the face normal."""
        self._normal = np.array(value, dtype=float)
        norm = np.linalg.norm(self._normal)
        if norm > 1e-10:
            self._normal /= norm
    
    def set_vertices(self, vertices: List) -> 'GXMLPolygon':
        """
        Set the polygon vertices.
        
        Args:
            vertices: List of 3D points (as lists, tuples, or np.arrays)
            
        Returns:
            Self for chaining
        """
        self._vertices = [np.array(v, dtype=float) for v in vertices]
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
        self._vertices.append(np.array(vertex, dtype=float))
        self._normal = None
        return self
    
    def _compute_normal(self) -> np.ndarray:
        """Compute the face normal from the first 3 vertices."""
        if len(self._vertices) < 3:
            return np.array([0, 1, 0])  # Default up
        
        v0, v1, v2 = self._vertices[0], self._vertices[1], self._vertices[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            return normal / norm
        return np.array([0, 1, 0])
    
    # -------------------------------------------------------------------------
    # Parametric coordinate access
    # -------------------------------------------------------------------------
    
    def get_vertex(self, index: int) -> np.ndarray:
        """
        Get a vertex by index.
        
        Args:
            index: Vertex index (wraps around)
            
        Returns:
            The vertex position
        """
        return self._vertices[index % len(self._vertices)]
    
    def get_edge(self, edge_index: int) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def point_on_edge(self, edge_index: int, t: float) -> np.ndarray:
        """
        Get a point along an edge using parametric coordinate.
        
        Args:
            edge_index: Which edge (0 to vertex_count-1)
            t: Parameter from 0 (start vertex) to 1 (end vertex)
            
        Returns:
            The interpolated point
        """
        start, end = self.get_edge(edge_index)
        return start + t * (end - start)
    
    def get_centroid(self) -> np.ndarray:
        """
        Get the centroid (average) of all vertices.
        
        Returns:
            The centroid point
        """
        if not self._vertices:
            return np.array([0, 0, 0])
        return np.mean(self._vertices, axis=0)
    
    def point_from_barycentric(self, weights: List[float]) -> np.ndarray:
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
        
        result = np.zeros(3)
        for w, v in zip(weights, self._vertices):
            result += (w / total) * v
        return result
    
    # -------------------------------------------------------------------------
    # Geometric queries
    # -------------------------------------------------------------------------
    
    def get_edge_length(self, edge_index: int) -> float:
        """Get the length of an edge."""
        start, end = self.get_edge(edge_index)
        return float(np.linalg.norm(end - start))
    
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
        abs_normal = np.abs(normal)
        if abs_normal[0] >= abs_normal[1] and abs_normal[0] >= abs_normal[2]:
            # Project onto YZ plane
            coords = [(v[1], v[2]) for v in self._vertices]
        elif abs_normal[1] >= abs_normal[2]:
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
    
    def get_world_vertices(self) -> List[np.ndarray]:
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
    def from_points(points: List, normal: Optional[np.ndarray] = None) -> 'GXMLPolygon':
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
