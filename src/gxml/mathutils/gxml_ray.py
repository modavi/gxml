"""
GXMLRay - A ray with origin, direction, and length.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GXMLRay:
    """A ray with origin, direction, and length."""
    origin: np.ndarray
    direction: np.ndarray  # Normalized
    length: float
    
    def point_at_t(self, t: float) -> np.ndarray:
        """Get a point along the ray at parameter t (0-1 maps to origin-end)."""
        return self.origin + self.direction * (t * self.length)
    
    def project_point(self, point: np.ndarray) -> float:
        """Project a point onto the ray and return the t-value (0-1 range)."""
        return np.dot(point - self.origin, self.direction) / self.length
    
    @staticmethod
    def from_points(start: np.ndarray, end: np.ndarray, tolerance: float = 1e-6) -> Optional['GXMLRay']:
        """Create a GXMLRay from two points. Returns None if points are too close."""
        direction = end - start
        length = np.linalg.norm(direction)
        if length < tolerance:
            return None
        return GXMLRay(origin=start, direction=direction / length, length=length)
