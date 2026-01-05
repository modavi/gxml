import math
from pathlib import Path
from .vec3 import Vec3, transform_point as vec3_transform_point, intersect_line_plane as vec3_intersect_line_plane

# Try to import C extension functions, fall back to None if unavailable
_HAS_C_EXTENSION = False
_HAS_MAT4 = False
_c_lerp = _c_mat4_invert = _c_find_interpolated_point = None
_c_mat4_multiply = _c_cross_product = _c_is_point_on_line_segment = None
_c_batch_transform_points = _c_dot = _c_normalize = None
_c_distance = _c_length = _c_project_point_on_ray = None
_c_create_transform_matrix_from_quad = None
Mat4 = None

try:
    from gxml.native_loader import load_native_extension
    _vec3 = load_native_extension('_vec3', Path(__file__).parent / 'native')
    if _vec3 is not None:
        _c_lerp = _vec3.lerp
        _c_mat4_invert = _vec3.mat4_invert
        _c_find_interpolated_point = _vec3.find_interpolated_point
        _c_mat4_multiply = _vec3.mat4_multiply
        _c_cross_product = _vec3.cross_product
        _c_is_point_on_line_segment = _vec3.is_point_on_line_segment
        _c_batch_transform_points = _vec3.batch_transform_points
        _c_dot = _vec3.dot
        _c_normalize = _vec3.normalize
        _c_distance = _vec3.distance
        _c_length = _vec3.length
        _c_project_point_on_ray = _vec3.project_point_on_ray
        _c_create_transform_matrix_from_quad = _vec3.create_transform_matrix_from_quad
        Mat4 = _vec3.Mat4
        _HAS_C_EXTENSION = True
        _HAS_MAT4 = True
except Exception:
    pass
#from scipy.spatial.transform import Rotation as R

# Identity matrix as tuple-of-tuples (immutable)
_IDENTITY_4x4_TUPLE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0)
)

# Pre-create identity Mat4 if available (avoids repeated allocation)
_IDENTITY_MAT4 = Mat4(_IDENTITY_4x4_TUPLE) if _HAS_MAT4 else None


def unpack_args(*args):
    x = y = z = 0
    
    if len(args) == 3:
        x, y, z = args  # Assign the values to x, y, z
    elif len(args) == 1:
        if len(args[0]) == 3:
            x, y, z = args[0]
    else:
        raise ValueError("Invalid number of arguments. Expected either a tuple (x, y, z) or three individual values.")
    
    return x,y,z

def rot_matrix(*args, rotate_order="xyz"):
    """Compute combined rotation matrix as tuple-of-tuples."""
    rx, ry, rz = unpack_args(*args)
    
    # Convert to radians and compute sin/cos once
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)
    
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    
    # Build combined rotation matrix based on order
    order = rotate_order.lower()
    
    if order == "xyz":
        # Combined xyz rotation matrix: Rx @ Ry @ Rz
        return (
            (cy*cz, cy*sz, -sy, 0.0),
            (sx*sy*cz - cx*sz, sx*sy*sz + cx*cz, sx*cy, 0.0),
            (cx*sy*cz + sx*sz, cx*sy*sz - sx*cz, cx*cy, 0.0),
            (0.0, 0.0, 0.0, 1.0)
        )
    elif order == "zyx":
        # Combined zyx rotation matrix: Rz @ Ry @ Rx
        return (
            (cy*cz, cx*sz + sx*sy*cz, sx*sz - cx*sy*cz, 0.0),
            (-cy*sz, cx*cz - sx*sy*sz, sx*cz + cx*sy*sz, 0.0),
            (sy, -sx*cy, cx*cy, 0.0),
            (0.0, 0.0, 0.0, 1.0)
        )
    else:
        # Fallback for other orders - build individual matrices and multiply with C extension
        R_x = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, cx, sx, 0.0),
            (0.0, -sx, cx, 0.0),
            (0.0, 0.0, 0.0, 1.0)
        )
        
        R_y = (
            (cy, 0.0, -sy, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (sy, 0.0, cy, 0.0),
            (0.0, 0.0, 0.0, 1.0)
        )
        
        R_z = (
            (cz, sz, 0.0, 0.0),
            (-sz, cz, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0)
        )
        
        rotation_matrix = _IDENTITY_4x4_TUPLE
        for axis in reversed(order):
            if axis == 'x':
                rotation_matrix = _c_mat4_multiply(R_x, rotation_matrix)
            elif axis == 'y':
                rotation_matrix = _c_mat4_multiply(R_y, rotation_matrix)
            elif axis == 'z':
                rotation_matrix = _c_mat4_multiply(R_z, rotation_matrix)
        return rotation_matrix

def scale_matrix(*args):
    x,y,z = unpack_args(*args)
    return (
        (x, 0.0, 0.0, 0.0),
        (0.0, y, 0.0, 0.0),
        (0.0, 0.0, z, 0.0),
        (0.0, 0.0, 0.0, 1.0)
    )

def translate_matrix(*args):
    x,y,z = unpack_args(*args)
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (x, y, z, 1.0)
    )
    
def explode_matrix(matrix):
    """Decompose a 4x4 transformation matrix into translation, rotation, and scale.
    
    Returns: (translation, rotation_matrix, scale) where:
        - translation: tuple (x, y, z)
        - rotation_matrix: 3x3 tuple-of-tuples
        - scale: tuple (sx, sy, sz)
    """
    # Extract translation (from bottom row in row-major format)
    translation = (matrix[3][0], matrix[3][1], matrix[3][2])
    
    # Extract scale factors - length of each row of upper-left 3x3
    def row_length(row_idx):
        x, y, z = matrix[row_idx][0], matrix[row_idx][1], matrix[row_idx][2]
        return math.sqrt(x*x + y*y + z*z)
    
    sx, sy, sz = row_length(0), row_length(1), row_length(2)
    scale = (sx, sy, sz)
    
    # Remove scale from the rotation matrix, handling zero scales
    # Build as tuple-of-tuples for consistency with other matrix functions
    rot_rows = []
    for i in range(3):
        s = scale[i]
        if abs(s) < 1e-10:
            # If scale is zero, use identity row
            if i == 0:
                rot_rows.append((1.0, 0.0, 0.0))
            elif i == 1:
                rot_rows.append((0.0, 1.0, 0.0))
            else:
                rot_rows.append((0.0, 0.0, 1.0))
        else:
            inv_s = 1.0 / s
            rot_rows.append((
                matrix[i][0] * inv_s,
                matrix[i][1] * inv_s,
                matrix[i][2] * inv_s
            ))
    
    rotation_matrix = (rot_rows[0], rot_rows[1], rot_rows[2])
    return translation, rotation_matrix, scale
    
def extract_euler_rotation(matrix, degrees=True):
    # Extract the upper-left 3x3 rotation portion (row-major format)
    r00, r01, r02 = matrix[0][0], matrix[0][1], matrix[0][2]
    r10, r11, r12 = matrix[1][0], matrix[1][1], matrix[1][2]
    r20, r21, r22 = matrix[2][0], matrix[2][1], matrix[2][2]
    
    # Calculate sy = sqrt(r[0,0]^2 + r[0,1]^2)
    sy = math.sqrt(r00*r00 + r01*r01)

    # Check near-singularity to handle gimbal locks
    singular = sy < 1e-6

    if not singular:
        rx = math.atan2(r12, r22)
        ry = math.atan2(-r02, sy)    
        rz = math.atan2(r01, r00)
    else:
        # Fallback in near-singular scenario
        rx = math.atan2(-r21, r11)
        ry = math.atan2(-r02, sy)     
        rz = 0.0

    # Convert from radians to degrees
    if degrees:
        rx = math.degrees(rx)
        ry = math.degrees(ry)
        rz = math.degrees(rz)
    
    return (rx, ry, rz)

def build_transform_matrix(t,r,s, transform_order="srt", rotate_order="xyz"):
    t_matrix = translate_matrix(t)
    r_matrix = rot_matrix(r, rotate_order=rotate_order)
    s_matrix = scale_matrix(s)
    
    combined_matrix = identity()

    # Reverse the transform order to make it match the houdini API
    for op in reversed(transform_order.lower()):
        if op == 't':
            combined_matrix = mat_mul(t_matrix, combined_matrix)
        elif op == 'r':
            combined_matrix = mat_mul(r_matrix, combined_matrix)
        elif op == 's':
            combined_matrix = mat_mul(s_matrix, combined_matrix)
            
    return combined_matrix

def combine_transform_matrix(t,r,s, transform_order="srt"):
    combined_matrix = identity()

    # Reverse the transform order to make it match the houdini API
    for op in reversed(transform_order.lower()):
        if op == 't':
            combined_matrix = mat_mul(t, combined_matrix)
        elif op == 'r':
            combined_matrix = mat_mul(r, combined_matrix)
        elif op == 's':
            combined_matrix = mat_mul(s, combined_matrix)
            
    return combined_matrix
    
def identity():
    """Return the 4x4 identity matrix as tuple-of-tuples."""
    return _IDENTITY_4x4_TUPLE

# Use optimized transform_point from vec3 module (C extension if available)
transform_point = vec3_transform_point

def batch_transform_points(points, matrix):
    """Batch transform multiple points by a 4x4 matrix."""
    if _c_batch_transform_points is not None:
        return _c_batch_transform_points(points, matrix)
    # Pure Python fallback
    return [transform_point(p, matrix) for p in points]
    
# Rotate the vector by the input transformation matrix. The rotation component is extracted from the matrix and used to rotate the vector
# while ensuring its length stays unmodified.    
def transform_direction(vector, matrix):
    t,r,s = explode_matrix(matrix)
    # r is a 3x3 tuple-of-tuples rotation matrix, vector @ r
    vx, vy, vz = vector[0], vector[1], vector[2]
    return (
        vx * r[0][0] + vy * r[1][0] + vz * r[2][0],
        vx * r[0][1] + vy * r[1][1] + vz * r[2][1],
        vx * r[0][2] + vy * r[1][2] + vz * r[2][2]
    )

def distance(p1, p2):
    """Distance between two points. Uses C extension if available."""
    if _c_distance:
        return _c_distance(p1, p2)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def length(vector):
    """Length of a vector. Uses C extension if available."""
    if _c_length:
        return _c_length(vector)
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

def sub3(a, b):
    """Subtract two 3D vectors. Returns tuple."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def add3(a, b):
    """Add two 3D vectors. Returns tuple."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def mul3(v, s):
    """Multiply vector by scalar. Returns tuple."""
    return (v[0] * s, v[1] * s, v[2] * s)

def neg3(v):
    """Negate vector. Returns tuple."""
    return (-v[0], -v[1], -v[2])

def dot3(a, b):
    """Dot product of two 3D vectors. Uses C extension if available."""
    if _c_dot:
        return _c_dot(a, b)
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def normalize(vector):
    """Normalize a vector. Returns tuple. Uses C extension if available."""
    if _c_normalize:
        return _c_normalize(vector)
    mag = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if mag < 1e-10:
        raise ValueError("Cannot normalize a zero vector")
    inv_mag = 1.0 / mag
    return (vector[0] * inv_mag, vector[1] * inv_mag, vector[2] * inv_mag)

def safe_normalize(vector):
    """Normalize a vector, returning zero vector if input is zero. Returns tuple."""
    mag = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if mag < 1e-10:
        return (0.0, 0.0, 0.0)
    inv_mag = 1.0 / mag
    return (vector[0] * inv_mag, vector[1] * inv_mag, vector[2] * inv_mag)

def cross(vector1, vector2):
    """Cross product. Returns tuple."""
    return cross3(vector1, vector2)

def cross3(a, b):
    """Fast 3D cross product. Returns tuple."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )

def dot_product(vector1, vector2):
    """Dot product - alias for dot3."""
    return dot3(vector1, vector2)

def create_transform_matrix_from_quad(points):
    """
    Create transformation matrix from 4 corner points in world space forming a quad.
    Points should be in CCW order: [p0, p1, p2, p3] where:
      p0 = origin (bottom-left)
      p1 = bottom-right (defines X-axis)
      p2 = top-right
      p3 = top-left (defines Y-axis)
    Maps from unit square (0,0)-(1,1) to the world space quad.
    
    Returns a tuple-of-tuples (4x4 matrix) for efficient use with C transform_point.
    """
    # Use C extension if available
    if _c_create_transform_matrix_from_quad is not None:
        return _c_create_transform_matrix_from_quad(points)
    
    if len(points) != 4:
        raise ValueError("Must provide exactly 4 world points")
    
    p0, p1, p2, p3 = points
    
    # Calculate world space axes and origin from the quad corners
    # Use pure Python math to avoid numpy overhead
    origin = (p0[0], p0[1], p0[2])
    
    # xAxis = normalize(p1 - p0)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dz = p1[2] - p0[2]
    xScale = math.sqrt(dx*dx + dy*dy + dz*dz)
    if xScale > 1e-10:
        inv_xScale = 1.0 / xScale
        xAxis = (dx * inv_xScale, dy * inv_xScale, dz * inv_xScale)
    else:
        xAxis = (1.0, 0.0, 0.0)
        xScale = 0.0
    
    # yAxis = normalize(p3 - p0)
    dx = p3[0] - p0[0]
    dy = p3[1] - p0[1]
    dz = p3[2] - p0[2]
    yScale = math.sqrt(dx*dx + dy*dy + dz*dz)
    if yScale > 1e-10:
        inv_yScale = 1.0 / yScale
        yAxis = (dx * inv_yScale, dy * inv_yScale, dz * inv_yScale)
    else:
        yAxis = (0.0, 1.0, 0.0)
        yScale = 0.0
    
    # zAxis = cross(xAxis, yAxis)
    zAxis = (
        xAxis[1] * yAxis[2] - xAxis[2] * yAxis[1],
        xAxis[2] * yAxis[0] - xAxis[0] * yAxis[2],
        xAxis[0] * yAxis[1] - xAxis[1] * yAxis[0]
    )
    
    # Build transformation matrix in row-major format for row-vector multiplication (v @ M)
    # Basis vectors are in rows, translation in bottom row
    # Return tuple-of-tuples - compatible with C transform_point and avoids numpy allocation
    return (
        (xAxis[0] * xScale, xAxis[1] * xScale, xAxis[2] * xScale, 0.0),
        (yAxis[0] * yScale, yAxis[1] * yScale, yAxis[2] * yScale, 0.0), 
        (zAxis[0],          zAxis[1],          zAxis[2],          0.0),
        (origin[0],         origin[1],         origin[2],         1.0)
    )

def angle_between(vector1, vector2):
    """Angle between two vectors in degrees."""
    dot_product = dot3(vector1, vector2)
    dot_product = max(-1.0, min(1.0, dot_product))  # clip to [-1, 1]
    angle = math.acos(dot_product)
    return math.degrees(angle)
    
def rotate_vector(vector, axis, angleDegrees):
    """Rotate a vector around an axis using Rodrigues' rotation formula. Returns tuple."""
    angle_radians = math.radians(angleDegrees)
    
    # Normalize the axis
    ax, ay, az = axis[0], axis[1], axis[2]
    axis_len = math.sqrt(ax*ax + ay*ay + az*az)
    if axis_len < 1e-10:
        return vector
    inv_len = 1.0 / axis_len
    ax, ay, az = ax * inv_len, ay * inv_len, az * inv_len
    
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)
    
    # Rodrigues' rotation formula: v' = v*cos + (k x v)*sin + k*(k.v)*(1-cos)
    vx, vy, vz = vector[0], vector[1], vector[2]
    
    # k x v
    cross_x = ay * vz - az * vy
    cross_y = az * vx - ax * vz
    cross_z = ax * vy - ay * vx
    
    # k . v
    dot = ax * vx + ay * vy + az * vz
    
    one_minus_cos = 1.0 - cos_theta
    
    return (
        vx * cos_theta + cross_x * sin_theta + ax * dot * one_minus_cos,
        vy * cos_theta + cross_y * sin_theta + ay * dot * one_minus_cos,
        vz * cos_theta + cross_z * sin_theta + az * dot * one_minus_cos
    )

def get_plane_rotation_from_triangle(p0, p1, p2):
    xAxis = normalize(sub3(p2, p1))
    yAxis = normalize(sub3(p1, p0))
    zAxis = cross(xAxis, yAxis)
    return get_plane_rotation_from_axis(xAxis, yAxis, zAxis)

def get_plane_rotation_from_axis(xAxis, yAxis, zAxis):
    # Build 3x3 rotation as tuple-of-tuples then convert to 4x4
    rotation_3x3 = (
        (xAxis[0], xAxis[1], xAxis[2]),
        (yAxis[0], yAxis[1], yAxis[2]),
        (zAxis[0], zAxis[1], zAxis[2])
    )
    return mat3_to_4(rotation_3x3)
    
def mat_mul(matrix1, matrix2):
    """Multiply two 4x4 matrices."""
    if _c_mat4_multiply is not None:
        return _c_mat4_multiply(matrix1, matrix2)
    # Pure Python fallback
    result = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return tuple(tuple(row) for row in result)

def mat3_to_4(mat3):
    # Convert 3x3 to 4x4 - returns tuple-of-tuples
    return (
        (mat3[0][0], mat3[0][1], mat3[0][2], 0.0),
        (mat3[1][0], mat3[1][1], mat3[1][2], 0.0),
        (mat3[2][0], mat3[2][1], mat3[2][2], 0.0),
        (0.0, 0.0, 0.0, 1.0)
    )

def invert(matrix):
    """Invert a 4x4 matrix."""
    if _c_mat4_invert is not None:
        return _c_mat4_invert(matrix)
    # Pure Python 4x4 matrix inversion using adjugate method
    m = matrix
    
    def det3(a, b, c, d, e, f, g, h, i):
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    
    # Calculate cofactors
    cof = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            # Minor matrix (3x3) excluding row i and col j
            minor = []
            for r in range(4):
                if r == i:
                    continue
                row = []
                for c in range(4):
                    if c == j:
                        continue
                    row.append(m[r][c])
                minor.append(row)
            # Cofactor with sign
            sign = (-1) ** (i + j)
            cof[i][j] = sign * det3(minor[0][0], minor[0][1], minor[0][2],
                                    minor[1][0], minor[1][1], minor[1][2],
                                    minor[2][0], minor[2][1], minor[2][2])
    
    # Determinant from first row
    det = sum(m[0][j] * cof[0][j] for j in range(4))
    if abs(det) < 1e-10:
        return identity()  # Return identity if singular
    
    # Adjugate (transpose of cofactor matrix) divided by determinant
    inv_det = 1.0 / det
    result = [[cof[j][i] * inv_det for j in range(4)] for i in range(4)]
    return tuple(tuple(row) for row in result)

def transpose(matrix):
    # Transpose a matrix - returns tuple-of-tuples
    n = len(matrix)
    m = len(matrix[0])
    return tuple(tuple(matrix[j][i] for j in range(n)) for i in range(m))

def determinant(matrix):
    """Compute determinant of 4x4 matrix using pure Python."""
    # Using cofactor expansion along first row
    # For a 4x4 matrix, det = sum of a[0][j] * cofactor(0,j) for j=0..3
    m = matrix
    
    def det3(a, b, c, d, e, f, g, h, i):
        """3x3 determinant using rule of Sarrus"""
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    
    # Cofactors for first row
    cof0 = det3(m[1][1], m[1][2], m[1][3],
                m[2][1], m[2][2], m[2][3],
                m[3][1], m[3][2], m[3][3])
    cof1 = det3(m[1][0], m[1][2], m[1][3],
                m[2][0], m[2][2], m[2][3],
                m[3][0], m[3][2], m[3][3])
    cof2 = det3(m[1][0], m[1][1], m[1][3],
                m[2][0], m[2][1], m[2][3],
                m[3][0], m[3][1], m[3][3])
    cof3 = det3(m[1][0], m[1][1], m[1][2],
                m[2][0], m[2][1], m[2][2],
                m[3][0], m[3][1], m[3][2])
    
    return m[0][0] * cof0 - m[0][1] * cof1 + m[0][2] * cof2 - m[0][3] * cof3

class ScaleInheritance:
    # Simple inheritance: world = local * parent_world
    Default = 0

    # Child doesn’t scale with the parent local scales, but local translation is scaled: world = local_scale_rotates * parent_local_scales.inverted() * local_translates * parent_world
    OffsetOnly = 1

    # Local translation is scaled as before but parent local scaling is also reapplied by the child in local space: world = parent_local_scales * local_scale_rotates * parent_local_scales.inverted() * local_translates * parent_world
    OffsetAndScale = 2

    # Local translation is not scaled, but parent local scaling is reapplied by the child in local space: world = parent_local_scales * local * parent_local_scales.inverted() * parent_world
    ScaleOnly = 3

    # Child completely ignores any parent local scaling: world = local * parent_local_scales.inverted() * parent_world
    Ignore = 4
    
def combine_transform(localMatrix, parentMatrix, parentLocalMatrix, scaleInheritance):
    # Fast path for default scale inheritance - no matrix decomposition needed
    if scaleInheritance == ScaleInheritance.Default:
        return mat_mul(localMatrix, parentMatrix)
    
    # For Ignore with identity scale, simplify to Default behavior
    if scaleInheritance == ScaleInheritance.Ignore:
        # Quick check: if parentLocalMatrix is identity-like for scale (diagonal is 1s)
        # We can check the 3x3 upper-left portion using [row][col] tuple indexing
        if (abs(parentLocalMatrix[0][0] - 1.0) < 1e-10 and 
            abs(parentLocalMatrix[1][1] - 1.0) < 1e-10 and 
            abs(parentLocalMatrix[2][2] - 1.0) < 1e-10):
            # Parent has unit scale, so invert(parent_local_scales) is identity
            return mat_mul(localMatrix, parentMatrix)
    
    # Only decompose matrices for non-trivial scale inheritance
    tlp,rlp,slp = explode_matrix(parentLocalMatrix)
    tp,rp,sp = explode_matrix(parentMatrix)
    tl,rl,sl = explode_matrix(localMatrix)
    
    parentLocalScales = scale_matrix(slp)
    localScaleRotates = mat_mul(scale_matrix(sl), mat3_to_4(rl))        
        
    # Handle zero scales by using identity matrix for scaling
    det = determinant(parentLocalScales)
    if abs(det) < 1e-10:
        invertParentLocalScales = identity()
        parentMatrix = build_transform_matrix(tp, (0,0,0), (1,1,1))
    else:
        invertParentLocalScales = invert(parentLocalScales)
    
    match scaleInheritance:
        case ScaleInheritance.OffsetOnly:
            # world = local_scale_rotates * invert(parent_local_scales) * local_translates * parent_world
            return mat_mul(mat_mul(mat_mul(localScaleRotates, invertParentLocalScales), translate_matrix(tl)), parentMatrix)
        case ScaleInheritance.OffsetAndScale:
            # world = parent_local_scales * local_scale_rotates * invert(parent_local_scales) * T * parent_world
            return mat_mul(mat_mul(mat_mul(mat_mul(parentLocalScales, localScaleRotates), invertParentLocalScales), translate_matrix(tl)), parentMatrix)
        case ScaleInheritance.ScaleOnly:
            # world = parent_local_scales * local * invert(parent_local_scales) * parent_world
            return mat_mul(mat_mul(mat_mul(parentLocalScales, localMatrix), invertParentLocalScales), parentMatrix)
        case ScaleInheritance.Ignore:
            # world = local * invert(parent_local_scales) * parent_world
            return mat_mul(mat_mul(localMatrix, invertParentLocalScales), parentMatrix)
    
    # Fallback
    return mat_mul(localMatrix, parentMatrix)

# Finds the intersection from a ray starting at point, in direction, with the line segment created by p1 and p2 as the endpoints
def find_intersection_ray_to_segment(point, direction, p1, p2):
    """Find intersection between ray and line segment. Returns tuple or None."""
    px, py, pz = point[0], point[1], point[2]
    dx, dy, dz = direction[0], direction[1], direction[2]
    p1x, p1y, p1z = p1[0], p1[1], p1[2]
    p2x, p2y, p2z = p2[0], p2[1], p2[2]
    
    # Calculate the segment direction
    sx, sy, sz = p2x - p1x, p2y - p1y, p2z - p1z
    
    # Calculate the cross product of the ray direction and segment direction
    v_cross_x = dy * sz - dz * sy
    v_cross_y = dz * sx - dx * sz
    v_cross_z = dx * sy - dy * sx
    
    denom = v_cross_x*v_cross_x + v_cross_y*v_cross_y + v_cross_z*v_cross_z
    
    # If the denominator is zero, the lines are parallel
    if abs(denom) < 1e-10:
        return None
    
    # Calculate the vector from the segment start to the ray origin
    wx, wy, wz = px - p1x, py - p1y, pz - p1z
    
    # cross(segment_direction, w)
    sw_x = sy * wz - sz * wy
    sw_y = sz * wx - sx * wz
    sw_z = sx * wy - sy * wx
    
    # cross(direction, w)
    dw_x = dy * wz - dz * wy
    dw_y = dz * wx - dx * wz
    dw_z = dx * wy - dy * wx
    
    # Calculate the intersection parameters
    t = (sw_x*v_cross_x + sw_y*v_cross_y + sw_z*v_cross_z) / denom
    u = (dw_x*v_cross_x + dw_y*v_cross_y + dw_z*v_cross_z) / denom
    
    # Check if the intersection point is within the segment
    if t >= 0 and 0 <= u <= 1:
        return (px + t * dx, py + t * dy, pz + t * dz)
    
    return None

def find_intersection_between_rays(p1, dir1, p2, dir2):
    """Find intersection between two rays. Returns tuple or None."""
    p1x, p1y, p1z = p1[0], p1[1], p1[2]
    d1x, d1y, d1z = dir1[0], dir1[1], dir1[2]
    p2x, p2y, p2z = p2[0], p2[1], p2[2]
    d2x, d2y, d2z = dir2[0], dir2[1], dir2[2]
    
    # Calculate the cross product of the direction vectors
    cross_x = d1y * d2z - d1z * d2y
    cross_y = d1z * d2x - d1x * d2z
    cross_z = d1x * d2y - d1y * d2x
    denom = cross_x*cross_x + cross_y*cross_y + cross_z*cross_z
    
    # If the denominator is zero, the rays are parallel
    if denom < 1e-12:
        return None
    
    # Calculate the vector from the origin of the first ray to the origin of the second ray
    wx, wy, wz = p2x - p1x, p2y - p1y, p2z - p1z
    
    # Calculate the intersection parameters
    # cross(w, dir2)
    w_cross_d2_x = wy * d2z - wz * d2y
    w_cross_d2_y = wz * d2x - wx * d2z
    w_cross_d2_z = wx * d2y - wy * d2x
    
    # cross(w, dir1)
    w_cross_d1_x = wy * d1z - wz * d1y
    w_cross_d1_y = wz * d1x - wx * d1z
    w_cross_d1_z = wx * d1y - wy * d1x
    
    t1 = (w_cross_d2_x*cross_x + w_cross_d2_y*cross_y + w_cross_d2_z*cross_z) / denom
    t2 = (w_cross_d1_x*cross_x + w_cross_d1_y*cross_y + w_cross_d1_z*cross_z) / denom
    
    # Calculate the intersection points on each ray
    i1x = p1x + t1 * d1x
    i1y = p1y + t1 * d1y
    i1z = p1z + t1 * d1z
    i2x = p2x + t2 * d2x
    i2y = p2y + t2 * d2y
    i2z = p2z + t2 * d2z
    
    # Check if the intersection points are the same (within a tolerance)
    diff_x = i1x - i2x
    diff_y = i1y - i2y
    diff_z = i1z - i2z
    dist_sq = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z
    if dist_sq < 1e-12:  # tolerance squared
        return (i1x, i1y, i1z)
    
    return None

def find_intersection_between_segments(p1, p2, p3, p4, tol=1e-6):
    """
    Find the intersection point between two line segments in 3D space.
    
    Args:
        p1, p2: Endpoints of the first segment
        p3, p4: Endpoints of the second segment
        tol: Tolerance for floating point comparisons (default 1e-6)
        
    Returns:
        The intersection point as a numpy array, or None if segments don't intersect
    """
    # Extract coordinates directly (works for arrays, lists, tuples)
    p1x, p1y, p1z = p1[0], p1[1], p1[2]
    p2x, p2y, p2z = p2[0], p2[1], p2[2]
    p3x, p3y, p3z = p3[0], p3[1], p3[2]
    p4x, p4y, p4z = p4[0], p4[1], p4[2]
    
    # Calculate the segment vectors
    s1x, s1y, s1z = p2x - p1x, p2y - p1y, p2z - p1z
    s2x, s2y, s2z = p4x - p3x, p4y - p3y, p4z - p3z
    
    # Calculate cross product of segment vectors (inline)
    cx = s1y * s2z - s1z * s2y
    cy = s1z * s2x - s1x * s2z
    cz = s1x * s2y - s1y * s2x
    denom = cx * cx + cy * cy + cz * cz
    
    # If the denominator is zero, the segments are parallel
    tol_sq = tol * tol
    if denom < tol_sq:
        return None
    
    # Calculate the vector from p1 to p3
    wx, wy, wz = p3x - p1x, p3y - p1y, p3z - p1z
    
    # Calculate w × segment2 (for t1)
    wcs2_x = wy * s2z - wz * s2y
    wcs2_y = wz * s2x - wx * s2z
    wcs2_z = wx * s2y - wy * s2x
    t1 = (wcs2_x * cx + wcs2_y * cy + wcs2_z * cz) / denom
    
    # Check if t1 is within segment bounds early (avoid computing t2 if not needed)
    if t1 < -tol or t1 > 1 + tol:
        return None
    
    # Calculate w × segment1 (for t2)
    wcs1_x = wy * s1z - wz * s1y
    wcs1_y = wz * s1x - wx * s1z
    wcs1_z = wx * s1y - wy * s1x
    t2 = (wcs1_x * cx + wcs1_y * cy + wcs1_z * cz) / denom
    
    # Check if t2 is within segment bounds
    if t2 < -tol or t2 > 1 + tol:
        return None
    
    # Compute both intersection points
    ix1 = p1x + t1 * s1x
    iy1 = p1y + t1 * s1y
    iz1 = p1z + t1 * s1z
    
    ix2 = p3x + t2 * s2x
    iy2 = p3y + t2 * s2y
    iz2 = p3z + t2 * s2z
    
    # Check if the intersection points are the same (within tolerance)
    dx, dy, dz = ix1 - ix2, iy1 - iy2, iz1 - iz2
    if dx * dx + dy * dy + dz * dz < tol_sq:
        return (ix1, iy1, iz1)
    
    return None

def is_point_on_line_segment(point, p1, p2, tol=1e-6):
    if _c_is_point_on_line_segment is not None:
        return _c_is_point_on_line_segment(point, p1, p2, tol)
    # Pure Python fallback
    # Check if point is within tolerance of the line segment
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = (p2[2] - p1[2]) if len(p1) > 2 and len(p2) > 2 else 0.0
    
    len_sq = dx*dx + dy*dy + dz*dz
    if len_sq < 1e-12:
        # Degenerate segment - check distance to p1
        px = point[0] - p1[0]
        py = point[1] - p1[1]
        pz = (point[2] - p1[2]) if len(point) > 2 and len(p1) > 2 else 0.0
        return (px*px + py*py + pz*pz) <= tol*tol
    
    # Project point onto line
    px = point[0] - p1[0]
    py = point[1] - p1[1]
    pz = (point[2] - p1[2]) if len(point) > 2 and len(p1) > 2 else 0.0
    
    t = (px*dx + py*dy + pz*dz) / len_sq
    
    # Check if projection is within segment
    if t < -tol or t > 1.0 + tol:
        return False
    
    # Check distance from point to projection
    proj_x = p1[0] + t * dx
    proj_y = p1[1] + t * dy
    proj_z = (p1[2] + t * dz) if len(p1) > 2 else 0.0
    
    dist_sq = (point[0] - proj_x)**2 + (point[1] - proj_y)**2
    if len(point) > 2:
        dist_sq += (point[2] - proj_z)**2
    
    return dist_sq <= tol*tol

# Use optimized version from vec3 module (C extension if available)
intersect_line_with_plane = vec3_intersect_line_plane


def intersect_lines_2d(line1, line2, plane='xz', tol=1e-9):
    """
    Find intersection of two infinite lines projected onto an axis-aligned plane.
    
    The ignored axis coordinate of the result is taken from the first point of line1.
    
    Args:
        line1: Tuple of (p1, p2) defining the first line
        line2: Tuple of (p3, p4) defining the second line
        plane: Which plane to project onto ('xy', 'xz', or 'yz'). Default 'xz'.
        tol: Tolerance for parallel line detection (default 1e-9)
        
    Returns:
        3D intersection point as tuple, or None if lines are parallel
    """
    p1, p2 = line1
    p3, p4 = line2
    
    # Inline plane mapping and axis selection for xz (common case)
    if plane == 'xz':
        a1, b1 = p1[0], p1[2]
        a2, b2 = p2[0], p2[2]
        a3, b3 = p3[0], p3[2]
        a4, b4 = p4[0], p4[2]
        ignore_val = p1[1]
        
        denom = (a1 - a2) * (b3 - b4) - (b1 - b2) * (a3 - a4)
        if abs(denom) < tol:
            return None
        
        t = ((a1 - a3) * (b3 - b4) - (b1 - b3) * (a3 - a4)) / denom
        return (a1 + t * (a2 - a1), ignore_val, b1 + t * (b2 - b1))
    
    # General case for other planes
    plane_map = {'yz': 0, 'xy': 2}
    ignore_idx = plane_map.get(plane.lower(), 1)
    axes = [i for i in range(3) if i != ignore_idx]
    a, b = axes[0], axes[1]
    
    a1, b1 = p1[a], p1[b]
    a2, b2 = p2[a], p2[b]
    a3, b3 = p3[a], p3[b]
    a4, b4 = p4[a], p4[b]
    
    denom = (a1 - a2) * (b3 - b4) - (b1 - b2) * (a3 - a4)
    if abs(denom) < tol:
        return None
    
    t = ((a1 - a3) * (b3 - b4) - (b1 - b3) * (a3 - a4)) / denom
    
    # Build result as tuple (faster than numpy array)
    ra = a1 + t * (a2 - a1)
    rb = b1 + t * (b2 - b1)
    if ignore_idx == 0:
        return (p1[0], ra, rb)
    elif ignore_idx == 2:
        return (ra, rb, p1[2])
    else:
        return (ra, p1[1], rb)


def project_point_onto_plane(point, plane_point, plane_normal):
    """Project point onto plane. Uses pure Python math for performance."""
    px, py, pz = point[0], point[1], point[2]
    ppx, ppy, ppz = plane_point[0], plane_point[1], plane_point[2]
    
    # Normalize the plane normal (pure Python)
    nx, ny, nz = plane_normal[0], plane_normal[1], plane_normal[2]
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag < 1e-10:
        return Vec3(px, py, pz)
    inv_mag = 1.0 / mag
    nx, ny, nz = nx * inv_mag, ny * inv_mag, nz * inv_mag
    
    # Calculate the vector from the plane point to the point
    vx, vy, vz = px - ppx, py - ppy, pz - ppz
    
    # Calculate the distance from the point to the plane (dot product)
    distance = vx * nx + vy * ny + vz * nz
    
    # Project the point onto the plane
    return Vec3(px - distance * nx, py - distance * ny, pz - distance * nz)

def is_point_inside_polygon(point, polygon):
    """
    Check if a point is inside a polygon defined by 3D points on the XZ plane.
    
    Parameters:
    point: The point to check (tuple or list).
    polygon: The vertices of the polygon (list of tuples or lists).
    
    Returns:
    bool: True if the point is inside the polygon, False otherwise.
    """
    # Project the 3D points onto the XZ plane (returns tuple)
    def project_to_xz(p):
        return (p[0], p[2])
    
    point_xz = project_to_xz(point)
    polygon_xz = [project_to_xz(p) for p in polygon]
    
    # Use the winding number algorithm to check if the point is inside the polygon
    winding_number = 0
    n = len(polygon_xz)
        
    def is_left(p1, p2, point):
        return (p2[0] - p1[0]) * (point[1] - p1[1]) - (point[0] - p1[0]) * (p2[1] - p1[1])
    
    for i in range(n):
        p1 = polygon_xz[i]
        p2 = polygon_xz[(i + 1) % n]
        
        if p1[1] <= point_xz[1]:
            if p2[1] > point_xz[1] and is_left(p1, p2, point_xz) > 0:
                winding_number += 1
        else:
            if p2[1] <= point_xz[1] and is_left(p1, p2, point_xz) < 0:
                winding_number -= 1
    
    return winding_number != 0

''' Given a point 'p' that lies somewhere on the line segment created by p1 and p2, return
    the interpolated value (0-1) of the point between those two endpoints. If the point is past either endpoint
    it can go beyond 0-1. '''
def find_interpolated_point(point, p1, p2):
    if _c_find_interpolated_point is not None:
        return _c_find_interpolated_point(point, p1, p2)
    # Pure Python fallback: project point onto line p1->p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = (p2[2] - p1[2]) if len(p1) > 2 and len(p2) > 2 else 0.0
    
    len_sq = dx*dx + dy*dy + dz*dz
    if len_sq < 1e-12:
        return 0.0
    
    px = point[0] - p1[0]
    py = point[1] - p1[1]
    pz = (point[2] - p1[2]) if len(point) > 2 and len(p1) > 2 else 0.0
    
    return (px*dx + py*dy + pz*dz) / len_sq

def lerp(t, a, b):
    # For 3D vectors, use C extension - much faster than numpy
    # For scalars, use Python math
    try:
        return _c_lerp(t, a, b)
    except TypeError:
        # Fall back to scalar lerp
        return a + t * (b - a)