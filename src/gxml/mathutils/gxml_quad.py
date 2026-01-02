import mathutils.gxml_math as GXMLMath


class QuadInterpolator(object):
    """
    A quad shape used for bilinear interpolation.
    
    This is a math utility for interpolating points within a quadrilateral,
    not a renderable element. For renderable quad elements, see elements.gxml_quad.GXMLQuad.
    """
    def __init__(self, p0, p1, p2, p3):
        # Store as tuples for consistent tuple-based math
        self.points = [
            (p0[0], p0[1], p0[2]),
            (p1[0], p1[1], p1[2]),
            (p2[0], p2[1], p2[2]),
            (p3[0], p3[1], p3[2])
        ]
    
    def get_interpolated_point(self, point):
        return self.bilinear_interpolate_point(point)
    
    def bilinear_interpolate_point(self, point):
        l1 = GXMLMath.lerp(point[0], self.points[0], self.points[1])
        l2 = GXMLMath.lerp(point[0], self.points[3], self.points[2])
        p = GXMLMath.lerp(point[1], l1, l2)
        return (p[0], p[1], point[2] + p[2])