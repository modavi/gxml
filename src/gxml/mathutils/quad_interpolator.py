# Import C extension functions if available
try:
    from . import _vec3
    _c_bilinear_interpolate = _vec3.bilinear_interpolate
    _c_batch_bilinear_transform = _vec3.batch_bilinear_transform
except ImportError:
    _c_bilinear_interpolate = None
    _c_batch_bilinear_transform = None

from gxml_profile import profile


class QuadInterpolator(object):
    """
    A quad shape used for bilinear interpolation.
    
    This is a math utility for interpolating points within a quadrilateral,
    not a renderable element. For renderable quad elements, see elements.gxml_quad.GXMLQuad.
    """
    def __init__(self, p0, p1, p2, p3):
        # Store points as tuples for fast indexing (no numpy overhead)
        self.p0 = (float(p0[0]), float(p0[1]), float(p0[2]))
        self.p1 = (float(p1[0]), float(p1[1]), float(p1[2]))
        self.p2 = (float(p2[0]), float(p2[1]), float(p2[2]))
        self.p3 = (float(p3[0]), float(p3[1]), float(p3[2]))
    
    def get_quad_points(self):
        """Return quad points as tuple for batch operations."""
        return (self.p0, self.p1, self.p2, self.p3)
    
    def get_interpolated_point(self, point):
        return self.bilinear_interpolate_point(point)
    
    def bilinear_interpolate_point(self, point):
        t = point[0]
        s = point[1]
        
        # Use C extension if available
        if _c_bilinear_interpolate:
            result = _c_bilinear_interpolate(t, s, self.p0, self.p1, self.p2, self.p3)
            return (result[0], result[1], point[2] + result[2])
        
        # Fallback to Python
        p0, p1, p2, p3 = self.p0, self.p1, self.p2, self.p3
        
        # l1 = lerp(t, p0, p1) = p0 + t * (p1 - p0)
        # l2 = lerp(t, p3, p2) = p3 + t * (p2 - p3)
        # p = lerp(s, l1, l2) = l1 + s * (l2 - l1)
        l1_x = p0[0] + t * (p1[0] - p0[0])
        l1_y = p0[1] + t * (p1[1] - p0[1])
        l1_z = p0[2] + t * (p1[2] - p0[2])
        
        l2_x = p3[0] + t * (p2[0] - p3[0])
        l2_y = p3[1] + t * (p2[1] - p3[1])
        l2_z = p3[2] + t * (p2[2] - p3[2])
        
        px = l1_x + s * (l2_x - l1_x)
        py = l1_y + s * (l2_y - l1_y)
        pz = l1_z + s * (l2_z - l1_z)
        
        return (px, py, point[2] + pz)


@profile("batch_bilinear_transform")
def batch_bilinear_transform(points_with_offsets, quad_points, matrix):
    """
    Perform bilinear interpolation + matrix transform in a single C call for multiple points.
    
    Args:
        points_with_offsets: list of (t, s, z_offset) tuples
        quad_points: tuple of 4 quad corner points (p0, p1, p2, p3)
        matrix: 4x4 transformation matrix
        
    Returns:
        list of (x, y, z) tuples
    """
    if _c_batch_bilinear_transform:
        return _c_batch_bilinear_transform(points_with_offsets, quad_points, matrix)
    
    # Python fallback
    p0, p1, p2, p3 = quad_points
    result = []
    for t, s, z_offset in points_with_offsets:
        # Bilinear interpolation
        l1_x = p0[0] + t * (p1[0] - p0[0])
        l1_y = p0[1] + t * (p1[1] - p0[1])
        l1_z = p0[2] + t * (p1[2] - p0[2])
        
        l2_x = p3[0] + t * (p2[0] - p3[0])
        l2_y = p3[1] + t * (p2[1] - p3[1])
        l2_z = p3[2] + t * (p2[2] - p3[2])
        
        ix = l1_x + s * (l2_x - l1_x)
        iy = l1_y + s * (l2_y - l1_y)
        iz = l1_z + s * (l2_z - l1_z) + z_offset
        
        # Matrix transform
        m = matrix
        w = ix * m[0][3] + iy * m[1][3] + iz * m[2][3] + m[3][3]
        if abs(w) > 1e-10:
            inv_w = 1.0 / w
            rx = (ix * m[0][0] + iy * m[1][0] + iz * m[2][0] + m[3][0]) * inv_w
            ry = (ix * m[0][1] + iy * m[1][1] + iz * m[2][1] + m[3][1]) * inv_w
            rz = (ix * m[0][2] + iy * m[1][2] + iz * m[2][2] + m[3][2]) * inv_w
        else:
            rx, ry, rz = 0.0, 0.0, 0.0
        result.append((rx, ry, rz))
    return result
