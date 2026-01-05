"""
GXMLRay - A ray with origin, direction, and length.
"""

import math
from dataclasses import dataclass
from typing import Optional
from .vec3 import Vec3

# Import C extension function if available
try:
    from . import _vec3
    _c_project_point_on_ray = _vec3.project_point_on_ray
except ImportError:
    _c_project_point_on_ray = None


@dataclass
class GXMLRay:
    """A ray with origin, direction, and length. Uses Vec3 for operator support."""
    origin: Vec3  # Position
    direction: Vec3  # Normalized direction
    length: float
    
    def point_at_t(self, t: float) -> Vec3:
        """Get a point along the ray at parameter t (0-1 maps to origin-end)."""
        dist = t * self.length
        return self.origin + self.direction * dist
    
    def project_point(self, point) -> float:
        """Project a point onto the ray and return the t-value (0-1 range)."""
        if _c_project_point_on_ray:
            return _c_project_point_on_ray(point, self.origin, self.direction, self.length)
        # Fallback to Python
        diff = Vec3(point) - self.origin
        return diff.dot(self.direction) / self.length
    
    @staticmethod
    def from_points(start, end, tolerance: float = 1e-6) -> Optional['GXMLRay']:
        """Create a GXMLRay from two points. Returns None if points are too close."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < tolerance:
            return None
        inv_length = 1.0 / length
        origin = Vec3(start[0], start[1], start[2])
        direction = Vec3(dx * inv_length, dy * inv_length, dz * inv_length)
        return GXMLRay(origin=origin, direction=direction, length=length)
