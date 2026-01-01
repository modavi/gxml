"""Debug matrix multiplication."""
import numpy as np
import sys
sys.path.insert(0, 'src/gxml')

# Test matrix multiplication order
matrix = np.array([
    [1, 0, 0, 5],
    [0, 1, 0, 10],
    [0, 0, 1, 15],
    [0, 0, 0, 1]
], dtype=float)

point = np.array([1, 2, 3])

# Method 1: row vector * matrix (original - what mat_mul does)
point_h = np.array([point[0], point[1], point[2], 1])
result1 = np.dot(point_h, matrix)
print('Row vector * matrix (original):', result1[:3] / result1[3])

# My implementation indexes - assuming row-major for row vector
x, y, z = point
# For row vector * matrix: result[i] = sum_j(vec[j] * matrix[j, i])
# So for x result: point[0]*m[0,0] + point[1]*m[1,0] + point[2]*m[2,0] + 1*m[3,0]
w = x * matrix[0, 3] + y * matrix[1, 3] + z * matrix[2, 3] + matrix[3, 3]
rx = x * matrix[0, 0] + y * matrix[1, 0] + z * matrix[2, 0] + matrix[3, 0]
ry = x * matrix[0, 1] + y * matrix[1, 1] + z * matrix[2, 1] + matrix[3, 1]
rz = x * matrix[0, 2] + y * matrix[1, 2] + z * matrix[2, 2] + matrix[3, 2]
print('Fixed implementation:', np.array([rx, ry, rz]) / w)
