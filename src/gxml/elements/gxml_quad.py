"""
GXMLQuad - A flat 4-vertex quadrilateral element.

A quad is defined by 4 corner points and supports bilinear interpolation
for parametric coordinate access. Quads are flat shapes with no volume.

For volumetric shapes, see GXMLPanel (extruded quad).
"""

from typing import List, Tuple, Optional
from elements.gxml_polygon import GXMLPolygon
from mathutils.quad_interpolator import QuadInterpolator
import mathutils.gxml_math as GXMLMath


class GXMLQuad(GXMLPolygon):
    """
    A flat 4-vertex quadrilateral with bilinear interpolation support.
    
    Unlike GXMLPolygon (which stores world vertices directly), GXMLQuad
    uses a QuadInterpolator for proper bilinear interpolation when 
    transforming points.
    """
    
    def __init__(self):
        super().__init__()
        self._interpolator: Optional[QuadInterpolator] = None
        self._cached_world_vertices: Optional[List] = None
    
    def transform_point(self, point):
        """
        Transform a point from parametric to world space.
        
        Uses bilinear interpolation through the quad interpolator if available,
        then applies the transformation matrix.
        
        Args:
            point: (t, s, z) or (x, y, z) local coordinates
            
        Returns:
            World space coordinates
        """
        if self._interpolator is not None:
            # Use bilinear interpolation for proper quad handling
            local_point = self._interpolator.get_interpolated_point(point)
            return GXMLMath.transform_point(local_point, self.transform.transformationMatrix)
        else:
            # Fall back to standard transform
            return super().transform_point(point)
    
    def get_world_vertices(self) -> List[tuple]:
        """
        Get the 4 corner vertices in world space (cached).
        
        Returns:
            List of 4 world-space vertex positions as (x, y, z) tuples
        """
        if self._cached_world_vertices is None:
            # Transform unit quad corners through our transform
            local_corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
            self._cached_world_vertices = [self.transform_point(c) for c in local_corners]
        return self._cached_world_vertices
    
    def render(self, renderContext):
        """Render the quad as a 4-vertex polygon."""
        world_verts = self.get_world_vertices()
        poly_id = f"{self.id}-{self.subId}" if self.subId else str(self.id)
        renderContext.create_poly(poly_id, world_verts)
