import numpy as np
import mathutils.gxml_math as GXMLMath


class QuadInterpolator(object):
    """
    A quad shape used for bilinear interpolation.
    
    This is a math utility for interpolating points within a quadrilateral,
    not a renderable element. For renderable quad elements, see elements.gxml_quad.GXMLQuad.
    """
    def __init__(self, p0, p1, p2, p3):
        self.points = [
            np.array(p0),
            np.array(p1),
            np.array(p2),
            np.array(p3)
        ]
    
    def get_interpolated_point(self, point):
        return self.bilinear_interpolate_point(point)
    
    def bilinear_interpolate_point(self, point):
        l1 = GXMLMath.lerp(point[0], self.points[0], self.points[1])
        l2 = GXMLMath.lerp(point[0], self.points[3], self.points[2])
        p = GXMLMath.lerp(point[1], l1, l2)
        p = np.array([p[0], p[1], point[2] + p[2]])
        return p
