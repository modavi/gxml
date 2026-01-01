import numpy as np
import math
#from scipy.spatial.transform import Rotation as R


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
    x,y,z = unpack_args(*args)
    x = np.radians(x)
    y = np.radians(y)
    z = np.radians(z)
    
    # Rotation around the X-axis
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(x), np.sin(x), 0],
        [0, -np.sin(x), np.cos(x), 0],
        [0, 0, 0, 1]
    ])
    
    # Rotation around the Y-axis
    R_y = np.array([
        [np.cos(y), 0, -np.sin(y), 0],
        [0, 1, 0, 0],
        [np.sin(y), 0, np.cos(y), 0],
        [0, 0, 0, 1]
    ])
    
    # Rotation around the Z-axis
    R_z = np.array([
        [np.cos(z), np.sin(z), 0, 0],
        [-np.sin(z), np.cos(z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    rotation_matrix = np.eye(4)
    for axis in reversed(rotate_order.lower()):
        if axis == 'x':
            rotation_matrix = np.dot(R_x, rotation_matrix)
        elif axis == 'y':
            rotation_matrix = np.dot(R_y, rotation_matrix)
        elif axis == 'z':
            rotation_matrix = np.dot(R_z, rotation_matrix)

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
        if np.isclose(scale[i], 0):
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
    return np.identity(4)

def transform_point(point, matrix):
    # Avoid creating new arrays when possible
    if isinstance(matrix, np.ndarray) and matrix.shape == (4, 4):
        # Fast path: matrix is already a 4x4 numpy array
        # Uses row-vector * matrix convention (same as np.dot(row_vec, matrix))
        x, y, z = point[0], point[1], point[2]
        w = x * matrix[0, 3] + y * matrix[1, 3] + z * matrix[2, 3] + matrix[3, 3]
        if w != 0:
            inv_w = 1.0 / w
            return np.array([
                (x * matrix[0, 0] + y * matrix[1, 0] + z * matrix[2, 0] + matrix[3, 0]) * inv_w,
                (x * matrix[0, 1] + y * matrix[1, 1] + z * matrix[2, 1] + matrix[3, 1]) * inv_w,
                (x * matrix[0, 2] + y * matrix[1, 2] + z * matrix[2, 2] + matrix[3, 2]) * inv_w
            ])
    
    # Fallback for non-numpy or wrong shape
    point_h = np.array([point[0], point[1], point[2], 1])
    matrix = np.asarray(matrix)
    
    if matrix.shape != (4, 4): 
        raise ValueError("Transformation matrix must be 4x4")
    
    transformed_point_h = mat_mul(point_h, matrix)
    return np.array(transformed_point_h[:3] / transformed_point_h[3])
    
# Rotate the vector by the input transformation matrix. The rotation component is extracted from the matrix and used to rotate the vector
# while ensuring its length stays unmodified.    
def transform_direction(vector, matrix):
    t,r,s = explode_matrix(matrix)
    return mat_mul(np.array(vector), r)

def distance(p1, p2):
    return np.linalg.norm(np.array(p2) - np.array(p1))

def length(vector):
    return np.linalg.norm(vector)

def normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector")
    return vector / magnitude

def safe_normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector * 0
    return vector / magnitude

def cross(vector1, vector2):
    return np.cross(vector1, vector2)

def cross3(a, b):
    """Fast 3D cross product without numpy overhead."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ])

def dot_product(vector1, vector2):
    return np.dot(vector1, vector2)

def create_transform_matrix_from_quad(points):
    """
    Create transformation matrix from 4 corner points in world space forming a quad.
    Points should be in CCW order: [p0, p1, p2, p3] where:
      p0 = origin (bottom-left)
      p1 = bottom-right (defines X-axis)
      p2 = top-right
      p3 = top-left (defines Y-axis)
    Maps from unit square (0,0)-(1,1) to the world space quad.
    """
    if len(points) != 4:
        raise ValueError("Must provide exactly 4 world points")
    
    corners = np.array(points)
    
    # Calculate world space axes and origin from the quad corners
    origin = corners[0]  # bottom-left corner
    xAxis = normalize(corners[1] - corners[0])  # bottom edge direction
    yAxis = normalize(corners[3] - corners[0])  # left edge direction  
    zAxis = cross(xAxis, yAxis)                 # normal direction
    
    # Calculate scales
    xScale = distance(corners[1], corners[0])
    yScale = distance(corners[3], corners[0])
    
    # Build transformation matrix in row-major format for row-vector multiplication (v @ M)
    # Basis vectors are in rows, translation in bottom row
    matrix = np.array([
        [xAxis[0] * xScale, xAxis[1] * xScale, xAxis[2] * xScale, 0],
        [yAxis[0] * yScale, yAxis[1] * yScale, yAxis[2] * yScale, 0], 
        [zAxis[0],          zAxis[1],          zAxis[2],          0],
        [origin[0],         origin[1],         origin[2],         1]
    ])
    
    return matrix

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
    return np.linalg.inv(np.array(matrix))

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
    tlp,rlp,slp = explode_matrix(parentLocalMatrix)
    tp,rp,sp = explode_matrix(parentMatrix)
    tl,rl,sl = explode_matrix(localMatrix)
    
    parentLocalScales = scale_matrix(slp)
    localScaleRotates = mat_mul(scale_matrix(sl), mat3_to_4(rl))        
        
    # Handle zero scales by using identity matrix for scaling
    if np.isclose(determinant(parentLocalScales), 0):
        invertParentLocalScales = identity()
        parentMatrix = build_transform_matrix(tp, (0,0,0), (1,1,1))
    else:
        invertParentLocalScales = invert(parentLocalScales)
    
    combined = identity()
    match scaleInheritance:
        case ScaleInheritance.Default:
            # world = local * parent_world
            combined = mat_mul(localMatrix, parentMatrix)
        case ScaleInheritance.OffsetOnly:
            # world = local_scale_rotates * invert(parent_local_scales) * local_translates * parent_world
            combined = mat_mul(mat_mul(mat_mul(localScaleRotates, invertParentLocalScales), translate_matrix(tl)), parentMatrix)
        case ScaleInheritance.OffsetAndScale:
            # world = parent_local_scales * local_scale_rotates * invert(parent_local_scales) * T * parent_world
            combined = mat_mul(mat_mul(mat_mul(mat_mul(parentLocalScales, localScaleRotates), invertParentLocalScales), translate_matrix(tl)), parentMatrix)
        case ScaleInheritance.ScaleOnly:
            # world = parent_local_scales * local * invert(parent_local_scales) * parent_world
            combined = mat_mul(mat_mul(mat_mul(parentLocalScales, localMatrix), invertParentLocalScales), parentMatrix)
        case ScaleInheritance.Ignore:
            # world = local * invert(parent_local_scales) * parent_world
            combined = mat_mul(mat_mul(localMatrix, invertParentLocalScales), parentMatrix)
    
    return combined

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
    if np.isclose(denom, 0):
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
    # Use asarray to avoid copies if already numpy arrays
    if not isinstance(p1, np.ndarray):
        p1 = np.asarray(p1)
    if not isinstance(p2, np.ndarray):
        p2 = np.asarray(p2)
    if not isinstance(p3, np.ndarray):
        p3 = np.asarray(p3)
    if not isinstance(p4, np.ndarray):
        p4 = np.asarray(p4)
    
    # Calculate the vectors
    segment1 = p2 - p1
    segment2 = p4 - p3
    
    # Calculate the cross product of the segment vectors (use inline cross for speed)
    cross_dir = cross3(segment1, segment2)
    denom = cross_dir[0]*cross_dir[0] + cross_dir[1]*cross_dir[1] + cross_dir[2]*cross_dir[2]
    
    # If the denominator is zero, the segments are parallel
    if denom < tol * tol:
        return None
    
    # Calculate the vector from the start of the first segment to the start of the second segment
    w = p3 - p1
    
    # Calculate the intersection parameters (inline cross products)
    w_cross_seg2 = cross3(w, segment2)
    w_cross_seg1 = cross3(w, segment1)
    t1 = (w_cross_seg2[0]*cross_dir[0] + w_cross_seg2[1]*cross_dir[1] + w_cross_seg2[2]*cross_dir[2]) / denom
    t2 = (w_cross_seg1[0]*cross_dir[0] + w_cross_seg1[1]*cross_dir[1] + w_cross_seg1[2]*cross_dir[2]) / denom
    
    # Check if the intersection points are within the segments (with tolerance for floating point)
    if -tol <= t1 <= 1 + tol and -tol <= t2 <= 1 + tol:
        intersection1 = p1 + t1 * segment1
        intersection2 = p3 + t2 * segment2
        
        # Check if the intersection points are the same (within tolerance)
        # Manual distance check instead of np.allclose
        diff = intersection1 - intersection2
        dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
        if dist_sq < tol * tol:
            return intersection1
    
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

def intersect_line_with_plane(line_point, line_direction, plane_point, plane_normal):
    # Convert to numpy arrays if needed
    if not isinstance(line_point, np.ndarray):
        line_point = np.asarray(line_point)
    if not isinstance(line_direction, np.ndarray):
        line_direction = np.asarray(line_direction)
    if not isinstance(plane_point, np.ndarray):
        plane_point = np.asarray(plane_point)
    if not isinstance(plane_normal, np.ndarray):
        plane_normal = np.asarray(plane_normal)
    
    # Calculate the denominator using inline dot product
    denom = line_direction[0]*plane_normal[0] + line_direction[1]*plane_normal[1] + line_direction[2]*plane_normal[2]
    
    # If the denominator is zero, the line is parallel to the plane
    if abs(denom) < 1e-10:
        return None
    
    # Calculate the distance from the line point to the plane
    diff = plane_point - line_point
    t = (diff[0]*plane_normal[0] + diff[1]*plane_normal[1] + diff[2]*plane_normal[2]) / denom
    
    # Calculate the intersection point
    return line_point + t * line_direction


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
        3D intersection point, or None if lines are parallel
    """
    p1, p2 = line1
    p3, p4 = line2
    
    # Map plane name to the axis to ignore
    plane_map = {'yz': 0, 'xz': 1, 'xy': 2}
    ignore_idx = plane_map.get(plane.lower(), 1)
    
    # Get the two axes we're using for 2D intersection
    axes = [i for i in range(3) if i != ignore_idx]
    a, b = axes[0], axes[1]
    
    a1, b1 = p1[a], p1[b]
    a2, b2 = p2[a], p2[b]
    a3, b3 = p3[a], p3[b]
    a4, b4 = p4[a], p4[b]
    
    denom = (a1 - a2) * (b3 - b4) - (b1 - b2) * (a3 - a4)
    if abs(denom) < tol:
        return None  # Lines are parallel
    
    t = ((a1 - a3) * (b3 - b4) - (b1 - b3) * (a3 - a4)) / denom
    
    # Build result point
    result = np.zeros(3)
    result[a] = a1 + t * (a2 - a1)
    result[b] = b1 + t * (b2 - b1)
    result[ignore_idx] = p1[ignore_idx]
    
    return result


def project_point_onto_plane(point, plane_point, plane_normal):
    point = np.array(point)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    
    # Normalize the plane normal
    plane_normal = normalize(plane_normal)
    
    # Calculate the vector from the plane point to the point
    point_vector = point - plane_point
    
    # Calculate the distance from the point to the plane
    distance = np.dot(point_vector, plane_normal)
    
    # Project the point onto the plane
    projected_point = point - distance * plane_normal
    
    return projected_point

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
    p1 = np.array(p1)
    p2 = np.array(p2)
    point = np.array(point)
    
    # Calculate the vectors
    segment_vector = p2 - p1
    point_vector = point - p1
    
    # Calculate the interpolation value
    dot_product = np.dot(point_vector, segment_vector)
    interpolation = dot_product / np.dot(segment_vector, segment_vector)
    
    return interpolation

def lerp(t, a, b):
    # Use numpy operations directly - a and b should already be numpy arrays
    return a + t * (b - a)