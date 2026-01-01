"""
Pure Python 3D vector math - no numpy dependency for hot paths.

This module provides Vec3, a lightweight 3D vector class optimized for performance.
For 3-element vectors, pure Python is 5-15x faster than numpy arrays
due to avoiding array creation overhead.

Vec3 supports arithmetic operators (+, -, *, /) and indexing.

If the _vec3 C extension is available, it will be used for maximum performance.
Otherwise, falls back to pure Python implementation.
"""
import math

# Try to import C extension for maximum performance
try:
    from gxml.mathutils._vec3 import (
        Vec3 as _CVec3,
        transform_point as _c_transform_point,
        intersect_line_plane as _c_intersect_line_plane,
        distance as _c_distance,
        length as _c_length,
        normalize as _c_normalize,
        dot as _c_dot,
        cross as _c_cross,
    )
    _USE_C_EXTENSION = True
except ImportError:
    _USE_C_EXTENSION = False


class _PythonVec3:
    """
    A lightweight 3D vector class that supports arithmetic operators.
    
    Stores components directly as attributes for fast access.
    Supports indexing like a tuple/list for compatibility.
    """
    __slots__ = ('x', 'y', 'z')
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        # Fast path: check if x is a simple number
        # Using try/except is faster than isinstance checks for the common case
        try:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
        except TypeError:
            # x is a sequence (tuple, list, array, Vec3)
            self.x = float(x[0])
            self.y = float(x[1])
            self.z = float(x[2])
    
    def __getitem__(self, i):
        if i == 0: return self.x
        if i == 1: return self.y
        if i == 2: return self.z
        raise IndexError(f"Vec3 index {i} out of range")
    
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
    
    def __len__(self):
        return 3
    
    def __repr__(self):
        return f"Vec3({self.x}, {self.y}, {self.z})"
    
    def __add__(self, other):
        # Try direct attribute access first (fast path for Vec3)
        try:
            return _PythonVec3(self.x + other.x, self.y + other.y, self.z + other.z)
        except AttributeError:
            return _PythonVec3(self.x + other[0], self.y + other[1], self.z + other[2])
    
    def __radd__(self, other):
        return _PythonVec3(self.x + other[0], self.y + other[1], self.z + other[2])
    
    def __sub__(self, other):
        # Try direct attribute access first (fast path for Vec3)
        try:
            return _PythonVec3(self.x - other.x, self.y - other.y, self.z - other.z)
        except AttributeError:
            return _PythonVec3(self.x - other[0], self.y - other[1], self.z - other[2])
    
    def __rsub__(self, other):
        return _PythonVec3(other[0] - self.x, other[1] - self.y, other[2] - self.z)
    
    def __mul__(self, scalar):
        return _PythonVec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        return _PythonVec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        inv = 1.0 / scalar
        return _PythonVec3(self.x * inv, self.y * inv, self.z * inv)
    
    def __neg__(self):
        return _PythonVec3(-self.x, -self.y, -self.z)
    
    def dot(self, other):
        """Dot product."""
        return self.x * other[0] + self.y * other[1] + self.z * other[2]
    
    def cross(self, other):
        """Cross product."""
        return _PythonVec3(
            self.y * other[2] - self.z * other[1],
            self.z * other[0] - self.x * other[2],
            self.x * other[1] - self.y * other[0]
        )
    
    def length_sq(self):
        """Squared length (avoids sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def length(self):
        """Vector length/magnitude."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalized(self):
        """Return normalized copy."""
        mag = self.length()
        if mag < 1e-10:
            return _PythonVec3(0.0, 0.0, 0.0)
        inv_mag = 1.0 / mag
        return _PythonVec3(self.x * inv_mag, self.y * inv_mag, self.z * inv_mag)
    
    def to_tuple(self):
        """Convert to tuple."""
        return (self.x, self.y, self.z)
    
    def to_list(self):
        """Convert to list."""
        return [self.x, self.y, self.z]


# Export the appropriate Vec3 class
Vec3 = _CVec3 if _USE_C_EXTENSION else _PythonVec3


# Standalone functions for tuple-based math (for places that don't use Vec3)

def vec3_add(a, b):
    """Add two vectors (works with any indexable)."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def vec3_sub(a, b):
    """Subtract two vectors (works with any indexable)."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def vec3_mul(v, s):
    """Multiply vector by scalar."""
    return (v[0] * s, v[1] * s, v[2] * s)

def vec3_dot(a, b):
    """Dot product."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def vec3_cross(a, b):
    """Cross product."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )

def vec3_length(v):
    """Vector length."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

def vec3_length_sq(v):
    """Squared length."""
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]

def vec3_normalize(v):
    """Normalize vector, returns tuple."""
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag < 1e-10:
        return (0.0, 0.0, 0.0)
    inv_mag = 1.0 / mag
    return (v[0] * inv_mag, v[1] * inv_mag, v[2] * inv_mag)

def vec3_distance(a, b):
    """Distance between two points."""
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def vec3_distance_sq(a, b):
    """Squared distance."""
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return dx * dx + dy * dy + dz * dz

def vec3_lerp(a, b, t):
    """Linear interpolation."""
    return (
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2])
    )


# ============================================================================
# Matrix Operations (4x4)
# ============================================================================

Mat4 = list   # 4x4 matrix as list of 4 lists

def mat4_identity():
    """Create 4x4 identity matrix."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]


def _python_transform_point(point, matrix):
    """Transform a 3D point by a 4x4 matrix. Returns Vec3."""
    x, y, z = point[0], point[1], point[2]
    
    # Check if it's a numpy array (use numpy indexing) or list (use list indexing)
    try:
        # Try numpy-style indexing first
        w = x * matrix[0, 3] + y * matrix[1, 3] + z * matrix[2, 3] + matrix[3, 3]
        if abs(w) > 1e-10:
            inv_w = 1.0 / w
            return Vec3(
                (x * matrix[0, 0] + y * matrix[1, 0] + z * matrix[2, 0] + matrix[3, 0]) * inv_w,
                (x * matrix[0, 1] + y * matrix[1, 1] + z * matrix[2, 1] + matrix[3, 1]) * inv_w,
                (x * matrix[0, 2] + y * matrix[1, 2] + z * matrix[2, 2] + matrix[3, 2]) * inv_w
            )
    except (TypeError, IndexError):
        # Fall back to list-of-lists indexing
        w = x * matrix[0][3] + y * matrix[1][3] + z * matrix[2][3] + matrix[3][3]
        if abs(w) > 1e-10:
            inv_w = 1.0 / w
            return Vec3(
                (x * matrix[0][0] + y * matrix[1][0] + z * matrix[2][0] + matrix[3][0]) * inv_w,
                (x * matrix[0][1] + y * matrix[1][1] + z * matrix[2][1] + matrix[3][1]) * inv_w,
                (x * matrix[0][2] + y * matrix[1][2] + z * matrix[2][2] + matrix[3][2]) * inv_w
            )
    return Vec3(0.0, 0.0, 0.0)


def _python_intersect_line_plane(line_point, line_direction, plane_point, plane_normal):
    """Intersect a line with a plane. Returns Vec3 or None."""
    lpx, lpy, lpz = line_point[0], line_point[1], line_point[2]
    ldx, ldy, ldz = line_direction[0], line_direction[1], line_direction[2]
    ppx, ppy, ppz = plane_point[0], plane_point[1], plane_point[2]
    pnx, pny, pnz = plane_normal[0], plane_normal[1], plane_normal[2]
    
    denom = ldx * pnx + ldy * pny + ldz * pnz
    
    if abs(denom) < 1e-10:
        return None
    
    t = ((ppx - lpx) * pnx + (ppy - lpy) * pny + (ppz - lpz) * pnz) / denom
    return Vec3(lpx + t * ldx, lpy + t * ldy, lpz + t * ldz)


# Use C version if available, otherwise Python fallback
transform_point = _c_transform_point if _USE_C_EXTENSION else _python_transform_point
intersect_line_plane = _c_intersect_line_plane if _USE_C_EXTENSION else _python_intersect_line_plane
