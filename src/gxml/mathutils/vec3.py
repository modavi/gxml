"""
Pure Python 3D vector math - no numpy dependency for hot paths.

This module provides Vec3, a lightweight 3D vector class optimized for performance.
For 3-element vectors, pure Python is 5-15x faster than numpy arrays
due to avoiding array creation overhead.

Vec3 supports arithmetic operators (+, -, *, /) and indexing.
"""
import math


class Vec3:
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
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        except AttributeError:
            return Vec3(self.x + other[0], self.y + other[1], self.z + other[2])
    
    def __radd__(self, other):
        return Vec3(self.x + other[0], self.y + other[1], self.z + other[2])
    
    def __sub__(self, other):
        # Try direct attribute access first (fast path for Vec3)
        try:
            return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        except AttributeError:
            return Vec3(self.x - other[0], self.y - other[1], self.z - other[2])
    
    def __rsub__(self, other):
        return Vec3(other[0] - self.x, other[1] - self.y, other[2] - self.z)
    
    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        inv = 1.0 / scalar
        return Vec3(self.x * inv, self.y * inv, self.z * inv)
    
    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)
    
    def dot(self, other):
        """Dot product."""
        return self.x * other[0] + self.y * other[1] + self.z * other[2]
    
    def cross(self, other):
        """Cross product."""
        return Vec3(
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
            return Vec3(0.0, 0.0, 0.0)
        inv_mag = 1.0 / mag
        return Vec3(self.x * inv_mag, self.y * inv_mag, self.z * inv_mag)
    
    def to_tuple(self):
        """Convert to tuple."""
        return (self.x, self.y, self.z)
    
    def to_list(self):
        """Convert to list."""
        return [self.x, self.y, self.z]


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
# Matrix Operations (4x4) - keeping for reference
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

def transform_point(point, matrix):
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
