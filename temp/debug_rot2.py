import numpy as np
import math

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

# Original: reversed('xyz') = ['z', 'y', 'x']
# Loop does: Rz @ I, then Ry @ (Rz), then Rx @ (Ry @ Rz)
# So final result is Rx @ Ry @ Rz
combined = R_x @ R_y @ R_z
print('Rx @ Ry @ Rz:')
print(combined)
print()

# Original implementation result
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

result = rot_matrix_old((55, 66, 77))
print('Old result:')
print(result)
print()
print('Matches Rx @ Ry @ Rz?', np.allclose(combined, result))

# Now derive combined formula for Rx @ Ry @ Rz
# 
# Rx @ Ry:
# | cy    0   -sy  0 |
# | sx*sy cx  sx*cy 0 |
# | cx*sy -sx cx*cy 0 |
# | 0     0   0    1 |

rxy = R_x @ R_y
print('Rx @ Ry:')
print(rxy[:3,:3])

# Rx @ Ry @ Rz:
# First column: (Rx@Ry) @ [cz, -sz, 0]^T
# = cz*col0 - sz*col1
# = cz*[cy, sx*sy, cx*sy] - sz*[0, cx, -sx]
# = [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz]
#
# Second column: (Rx@Ry) @ [sz, cz, 0]^T
# = sz*col0 + cz*col1
# = sz*[cy, sx*sy, cx*sy] + cz*[0, cx, -sx]
# = [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz]
#
# Third column: (Rx@Ry) @ [0, 0, 1]^T = col2 = [-sy, sx*cy, cx*cy]

manual = np.array([
    [cy*cz, cy*sz, -sy, 0],
    [sx*sy*cz - cx*sz, sx*sy*sz + cx*cz, sx*cy, 0],
    [cx*sy*cz + sx*sz, cx*sy*sz - sx*cz, cx*cy, 0],
    [0, 0, 0, 1]
])
print()
print('Manual Rx @ Ry @ Rz:')
print(manual[:3,:3])
print()
print('Matches old result?', np.allclose(manual, result))
