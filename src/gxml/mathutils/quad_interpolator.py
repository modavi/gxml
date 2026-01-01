import numpy as np


class QuadInterpolator(object):
    """
    A quad shape used for bilinear interpolation.
    
    This is a math utility for interpolating points within a quadrilateral,
    not a renderable element. For renderable quad elements, see elements.gxml_quad.GXMLQuad.
    """
    def __init__(self, p0, p1, p2, p3):
        # Store as flat arrays for faster indexing
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.p1 = np.asarray(p1, dtype=np.float64)
        self.p2 = np.asarray(p2, dtype=np.float64)
        self.p3 = np.asarray(p3, dtype=np.float64)
    
    def get_interpolated_point(self, point):
        return self.bilinear_interpolate_point(point)
    
    def bilinear_interpolate_point(self, point):
        # Inline lerp operations to avoid function call overhead
        t = point[0]
        s = point[1]
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
        
        return np.array([px, py, point[2] + pz])
