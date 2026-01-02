import numpy as np
import math
from .vec3 import Vec3, transform_point as vec3_transform_point, intersect_line_plane as vec3_intersect_line_plane
from ._vec3 import lerp as _c_lerp, mat4_invert as _c_mat4_invert, find_interpolated_point as _c_find_interpolated_point
#from scipy.spatial.transform import Rotation as R

# Pre-allocated identity matrix (read-only reference)
_IDENTITY_4x4 = np.eye(4, dtype=np.float64)
_IDENTITY_4x4.flags.writeable = False


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
    """Compute combined rotation matrix directly without intermediate matrices."""
    rx, ry, rz = unpack_args(*args)
    
    # Convert to radians and compute sin/cos once
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)
    
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    
    # Build combined rotation matrix based on order
    # For xyz order with left multiplication: Rx @ Ry @ Rz
    order = rotate_order.lower()
    
    if order == "xyz":
        # Combined xyz rotation matrix: Rx @ Ry @ Rz
        return np.array([
            [cy*cz, cy*sz, -sy, 0],
            [sx*sy*cz - cx*sz, sx*sy*sz + cx*cz, sx*cy, 0],
            [cx*sy*cz + sx*sz, cx*sy*sz - sx*cz, cx*cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
    elif order == "zyx":
        # Combined zyx rotation matrix: Rz @ Ry @ Rx
        return np.array([
            [cy*cz, cx*sz + sx*sy*cz, sx*sz - cx*sy*cz, 0],
            [-cy*sz, cx*cz - sx*sy*sz, sx*cz + cx*sy*sz, 0],
            [sy, -sx*cy, cx*cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
    else:
        # Fallback for other orders - build individual matrices
        R_x = np.array([
            [1, 0, 0, 0],
            [0, cx, sx, 0],
            [0, -sx, cx, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        R_y = np.array([
            [cy, 0, -sy, 0],
            [0, 1, 0, 0],
            [sy, 0, cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        R_z = np.array([
            [cz, sz, 0, 0],
            [-sz, cz, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        rotation_matrix = _IDENTITY_4x4.copy()
        for axis in reversed(order):
            if axis == 'x':
                rotation_matrix = R_x @ rotation_matrix
            elif axis == 'y':
                rotation_matrix = R_y @ rotation_matrix
            elif axis == 'z':
                rotation_matrix = R_z @ rotation_matrix
        return rotation_matrix

def scale_matrix(*args):
    x,y,z = unpack_args(*args)
    return np.array([
        [x, 0.0, 0.0, 0.0],
        [0.0, y, 0.0, 0.0],
        [0.0, 0.0, z, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def translate_matrix(*args):
    x,y,z = unpack_args(*args)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0]
    ])
    
def explode_matrix(matrix):
    matrix = np.array(matrix)
    
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix.")
    
    # Extract translation (from bottom row in row-major format)
    translation = matrix[3, :3]
    
    # Extract scale factors (from rows in row-major format)
    scale = np.linalg.norm(matrix[:3, :3], axis=1)
    
    # Remove scale from the rotation matrix, handling zero scales
    rotation_matrix = np.zeros((3, 3))
    for i in range(3):
        if abs(scale[i]) < 1e-10:
            # If scale is zero, use identity row
            rotation_matrix[i, i] = 1.0
        else:
            rotation_matrix[i] = matrix[i, :3] / scale[i]
    
    return translation, rotation_matrix, scale
    
def extract_euler_rotation(matrix, degrees=True):
    mat = np.array(matrix)
    if mat.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix.")

    # Extract the upper-left 3x3 rotation portion (row-major format)
    r = mat[:3, :3]
    
    # Calculate sy = sqrt(r[0,0]^2 + r[0,1]^2)
    sy = math.sqrt(r[0, 0]*r[0, 0] + r[0, 1]*r[0, 1])

    # Check near-singularity to handle gimbal locks
    singular = sy < 1e-6

    if not singular:
        rx = math.atan2(r[1, 2], r[2, 2])
        ry = math.atan2(-r[0, 2], sy)    
        rz = math.atan2(r[0, 1], r[0, 0])
    else:
        # Fallback in near-singular scenario
        rx = math.atan2(-r[2, 1], r[1, 1])
        ry = math.atan2(-r[0, 2], sy)     
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
    """Return a copy of the 4x4 identity matrix."""
    return _IDENTITY_4x4.copy()

# Use optimized transform_point from vec3 module (C extension if available)
transform_point = vec3_transform_point
    
# Rotate the vector by the input transformation matrix. The rotation component is extracted from the matrix and used to rotate the vector
# while ensuring its length stays unmodified.    
def transform_direction(vector, matrix):
    t,r,s = explode_matrix(matrix)
    return mat_mul(np.array(vector), r)

def distance(p1, p2):
    """Distance between two points. Works with tuples, lists, or arrays."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def length(vector):
    """Length of a vector. Works with tuples, lists, or arrays."""
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
    """Dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def normalize(vector):
    """Normalize a vector. Returns tuple."""
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
    return np.cross(vector1, vector2)

def cross3(a, b):
    """Fast 3D cross product. Returns tuple."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )

def dot_product(vector1, vector2):
    """Dot product - works with any indexable type."""
    return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]

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
    dot_product = np.dot(vector1, vector2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    return math.degrees(angle)
    
def rotate_vector(vector, axis, angleDegrees):
    angle_radians = np.radians(angleDegrees)
    vector = np.array(vector)
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    # Rodrigues' rotation formula
    rotated_vector = (vector * cos_theta +
                      np.cross(axis, vector) * sin_theta +
                      axis * np.dot(axis, vector) * (1 - cos_theta))
    
    return rotated_vector

def get_plane_rotation_from_triangle(p0, p1, p2):
    xAxis = normalize(np.array(p2) - np.array(p1))
    yAxis = normalize(np.array(p1) - np.array(p0))
    zAxis = cross(xAxis, yAxis)
    return get_plane_rotation_from_axis(xAxis, yAxis, zAxis)

def get_plane_rotation_from_axis(xAxis, yAxis, zAxis):
    rotation_3x3 = np.array([xAxis, yAxis, zAxis])
    return mat3_to_4(rotation_3x3)
    
def mat_mul(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

def mat3_to_4(mat3):
    mat4 = identity()
    mat4[:3, :3] = mat3
    return mat4

def invert(matrix):
    # Use C extension for 4x4 matrix inversion - avoids numpy.array allocation
    return _c_mat4_invert(matrix)

def transpose(matrix):
    return np.array(matrix).T

def determinant(matrix):
    return np.linalg.det(matrix)

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
        # We can check the 3x3 upper-left portion
        if (abs(parentLocalMatrix[0, 0] - 1.0) < 1e-10 and 
            abs(parentLocalMatrix[1, 1] - 1.0) < 1e-10 and 
            abs(parentLocalMatrix[2, 2] - 1.0) < 1e-10):
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
    point = np.array(point)
    direction = np.array(direction)
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Calculate the segment direction
    segment_direction = p2 - p1
    
    # Calculate the cross product of the ray direction and segment direction
    v_cross = np.cross(direction, segment_direction)
    denom = np.linalg.norm(v_cross) ** 2
    
    # If the denominator is zero, the lines are parallel
    if abs(denom) < 1e-10:
        return None
    
    # Calculate the vector from the segment start to the ray origin
    w = point - p1
    
    # Calculate the intersection parameters
    t = np.dot(np.cross(segment_direction, w), v_cross) / denom
    u = np.dot(np.cross(direction, w), v_cross) / denom
    
    # Check if the intersection point is within the segment
    if t >= 0 and 0 <= u <= 1:
        intersection_point = point + t * direction
        return np.array(intersection_point)
    
    return None

def find_intersection_between_rays(p1, dir1, p2, dir2):
    if not isinstance(p1, np.ndarray):
        p1 = np.asarray(p1)
    if not isinstance(dir1, np.ndarray):
        dir1 = np.asarray(dir1)
    if not isinstance(p2, np.ndarray):
        p2 = np.asarray(p2)
    if not isinstance(dir2, np.ndarray):
        dir2 = np.asarray(dir2)
    
    # Calculate the cross product of the direction vectors
    cross_dir = cross3(dir1, dir2)
    denom = cross_dir[0]*cross_dir[0] + cross_dir[1]*cross_dir[1] + cross_dir[2]*cross_dir[2]
    
    # If the denominator is zero, the rays are parallel
    if denom < 1e-12:
        return None
    
    # Calculate the vector from the origin of the first ray to the origin of the second ray
    w = p2 - p1
    
    # Calculate the intersection parameters (inline dot products)
    w_cross_dir2 = cross3(w, dir2)
    w_cross_dir1 = cross3(w, dir1)
    t1 = (w_cross_dir2[0]*cross_dir[0] + w_cross_dir2[1]*cross_dir[1] + w_cross_dir2[2]*cross_dir[2]) / denom
    t2 = (w_cross_dir1[0]*cross_dir[0] + w_cross_dir1[1]*cross_dir[1] + w_cross_dir1[2]*cross_dir[2]) / denom
    
    # Calculate the intersection points on each ray
    intersection1 = p1 + t1 * dir1
    intersection2 = p2 + t2 * dir2
    
    # Check if the intersection points are the same (within a tolerance)
    diff = intersection1 - intersection2
    dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
    if dist_sq < 1e-12:  # tolerance squared
        return intersection1
    
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
        return np.array([ix1, iy1, iz1])
    
    return None

def is_point_on_line_segment(point, p1, p2, tol=1e-6):
    point = np.array(point)
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Calculate the vectors
    segment_vector = p2 - p1
    point_vector = point - p1
    
    # Check if the point is on the line defined by the segment
    cross_product = np.cross(segment_vector, point_vector)
    if np.linalg.norm(cross_product) > tol:
        return False
    
    # Check if the point is within the bounds of the segment
    dot_product = np.dot(point_vector, segment_vector)
    if dot_product < 0 or dot_product > np.dot(segment_vector, segment_vector):
        return False
    
    return True

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
    point (np.array): The point to check.
    polygon (list of np.array): The vertices of the polygon.
    
    Returns:
    bool: True if the point is inside the polygon, False otherwise.
    """
    # Project the 3D points onto the XZ plane
    def project_to_xz(p):
        return np.array([p[0], p[2]])
    
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
    return _c_find_interpolated_point(point, p1, p2)

def lerp(t, a, b):
    # For 3D vectors, use C extension - much faster than numpy
    # For scalars, use Python math
    try:
        return _c_lerp(t, a, b)
    except TypeError:
        # Fall back to scalar lerp
        return a + t * (b - a)