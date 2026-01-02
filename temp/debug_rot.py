import numpy as np
import math

# Original implementation - builds individual matrices and multiplies
def rot_matrix_old(args, rotate_order='xyz'):
    x, y, z = args
    x = np.radians(x)
    y = np.radians(y)
    z = np.radians(z)
    
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(x), np.sin(x), 0],
        [0, -np.sin(x), np.cos(x), 0],
        [0, 0, 0, 1]
    ])
    
    R_y = np.array([
        [np.cos(y), 0, -np.sin(y), 0],
        [0, 1, 0, 0],
        [np.sin(y), 0, np.cos(y), 0],
        [0, 0, 0, 1]
    ])
    
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

# Test
result = rot_matrix_old((55, 66, 77))
print('Original formula result:')
print(result)
print()

# Now derive the combined formula
# For xyz order with reversed loop: z, y, x  
# So we compute Rz @ (Ry @ (Rx @ I)) = Rz @ Ry @ Rx
# Let's verify
rx, ry, rz = map(math.radians, (55, 66, 77))
cx, sx = math.cos(rx), math.sin(rx)
cy, sy = math.cos(ry), math.sin(ry)
cz, sz = math.cos(rz), math.sin(rz)

R_x = np.array([
    [1, 0, 0, 0],
    [0, cx, sx, 0],
    [0, -sx, cx, 0],
    [0, 0, 0, 1]
])

R_y = np.array([
    [cy, 0, -sy, 0],
    [0, 1, 0, 0],
    [sy, 0, cy, 0],
    [0, 0, 0, 1]
])

R_z = np.array([
    [cz, sz, 0, 0],
    [-sz, cz, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Order: reversed(xyz) = zyx, so Rz @ Ry @ Rx
combined = R_z @ R_y @ R_x
print('Rz @ Ry @ Rx:')
print(combined)
print()

# Let me derive the formula by hand
# Rz @ Ry:
#   cz*cy,  sz, -cz*sy, 0
#  -sz*cy,  cz,  sz*sy, 0
#      sy,   0,     cy, 0
#       0,   0,      0, 1

# (Rz @ Ry) @ Rx:
# Row 0: [cz*cy, cz*cy*0 + sz*cx + (-cz*sy)*(-sx), cz*cy*0 + sz*sx + (-cz*sy)*cx]
# Actually let me just compute it directly

print('Manual derivation:')
# R = Rz @ Ry @ Rx
# 
# Rz @ Ry gives:
# | cy*cz   sz  -sy*cz  0 |
# | -cy*sz  cz  sy*sz   0 |
# | sy      0   cy      0 |
# | 0       0   0       1 |
#
# (Rz @ Ry) @ Rx gives:
rzy = R_z @ R_y
print('Rz @ Ry:')
print(rzy[:3,:3])
print()

# Now (Rz@Ry) @ Rx
# First column of result = (Rz@Ry) @ first column of Rx = (Rz@Ry) @ [1,0,0]^T
# = first column of (Rz@Ry) = [cy*cz, -cy*sz, sy]
# 
# Second column = (Rz@Ry) @ [0,cx,-sx]^T
# = 0*col0 + cx*col1 - sx*col2
# = cx*[sz, cz, 0] - sx*[-sy*cz, sy*sz, cy]
# = [cx*sz + sx*sy*cz, cx*cz - sx*sy*sz, -sx*cy]
#
# Third column = (Rz@Ry) @ [0,sx,cx]^T  
# = sx*col1 + cx*col2
# = sx*[sz, cz, 0] + cx*[-sy*cz, sy*sz, cy]
# = [sx*sz - cx*sy*cz, sx*cz + cx*sy*sz, cx*cy]

manual = np.array([
    [cy*cz, cx*sz + sx*sy*cz, sx*sz - cx*sy*cz, 0],
    [-cy*sz, cx*cz - sx*sy*sz, sx*cz + cx*sy*sz, 0],
    [sy, -sx*cy, cx*cy, 0],
    [0, 0, 0, 1]
])
print('Manual formula:')
print(manual[:3,:3])
print()

print('Matches combined?', np.allclose(manual, combined))
print('Matches old result?', np.allclose(manual, result))
