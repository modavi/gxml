"""
Benchmark: Compare numpy vs pure Python for 3D vector/point operations.

The hypothesis is that for small 3D vectors (3-4 elements), pure Python 
tuples with inline math will be faster than numpy due to:
1. No array creation overhead
2. No type dispatch overhead
3. Better for small fixed-size data
"""
import time
import math
import numpy as np

# Number of iterations
N = 100000

# ============================================================================
# NUMPY VERSIONS
# ============================================================================

def np_transform_point(point, matrix):
    """Transform point using numpy - current implementation."""
    x, y, z = point[0], point[1], point[2]
    w = x * matrix[0, 3] + y * matrix[1, 3] + z * matrix[2, 3] + matrix[3, 3]
    if w != 0:
        inv_w = 1.0 / w
        return np.array([
            (x * matrix[0, 0] + y * matrix[1, 0] + z * matrix[2, 0] + matrix[3, 0]) * inv_w,
            (x * matrix[0, 1] + y * matrix[1, 1] + z * matrix[2, 1] + matrix[3, 1]) * inv_w,
            (x * matrix[0, 2] + y * matrix[1, 2] + z * matrix[2, 2] + matrix[3, 2]) * inv_w
        ])
    return np.array([0, 0, 0])

def np_dot3(a, b):
    """Dot product using numpy."""
    return np.dot(a, b)

def np_cross3(a, b):
    """Cross product returning numpy array."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ])

def np_normalize(v):
    """Normalize using numpy."""
    mag = np.linalg.norm(v)
    if mag == 0:
        return v
    return v / mag

def np_length(v):
    """Length using numpy."""
    return np.linalg.norm(v)

# ============================================================================
# PURE PYTHON VERSIONS (using tuples)
# ============================================================================

def py_transform_point(point, matrix):
    """Transform point using pure Python - returns tuple."""
    x, y, z = point[0], point[1], point[2]
    w = x * matrix[0][3] + y * matrix[1][3] + z * matrix[2][3] + matrix[3][3]
    if w != 0:
        inv_w = 1.0 / w
        return (
            (x * matrix[0][0] + y * matrix[1][0] + z * matrix[2][0] + matrix[3][0]) * inv_w,
            (x * matrix[0][1] + y * matrix[1][1] + z * matrix[2][1] + matrix[3][1]) * inv_w,
            (x * matrix[0][2] + y * matrix[1][2] + z * matrix[2][2] + matrix[3][2]) * inv_w
        )
    return (0.0, 0.0, 0.0)

def py_dot3(a, b):
    """Dot product - pure Python."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def py_cross3(a, b):
    """Cross product - returns tuple."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )

def py_normalize(v):
    """Normalize - returns tuple."""
    mag = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if mag == 0:
        return v
    inv_mag = 1.0 / mag
    return (v[0] * inv_mag, v[1] * inv_mag, v[2] * inv_mag)

def py_length(v):
    """Length - pure Python."""
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def py_sub(a, b):
    """Vector subtraction - returns tuple."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def py_add(a, b):
    """Vector addition - returns tuple."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def py_scale(v, s):
    """Scale vector - returns tuple."""
    return (v[0] * s, v[1] * s, v[2] * s)

# ============================================================================
# BENCHMARKS
# ============================================================================

def benchmark(name, func, *args):
    """Run benchmark and return time."""
    # Warmup
    for _ in range(1000):
        func(*args)
    
    start = time.perf_counter()
    for _ in range(N):
        result = func(*args)
    elapsed = time.perf_counter() - start
    return elapsed

if __name__ == '__main__':
    # Test data
    np_point = np.array([1.5, 2.5, 3.5])
    np_matrix = np.eye(4)
    np_matrix[3, :3] = [10, 20, 30]  # Translation
    
    py_point = (1.5, 2.5, 3.5)
    py_matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [10, 20, 30, 1]]
    
    np_v1 = np.array([1.0, 2.0, 3.0])
    np_v2 = np.array([4.0, 5.0, 6.0])
    
    py_v1 = (1.0, 2.0, 3.0)
    py_v2 = (4.0, 5.0, 6.0)
    
    print(f"Benchmarking {N:,} iterations each\n")
    print(f"{'Operation':<25} {'NumPy':>12} {'Pure Python':>12} {'Speedup':>10}")
    print("=" * 60)
    
    # Transform point
    t_np = benchmark("np_transform", np_transform_point, np_point, np_matrix)
    t_py = benchmark("py_transform", py_transform_point, py_point, py_matrix)
    print(f"{'transform_point':<25} {t_np*1000:>10.2f}ms {t_py*1000:>10.2f}ms {t_np/t_py:>9.2f}x")
    
    # Dot product
    t_np = benchmark("np_dot", np_dot3, np_v1, np_v2)
    t_py = benchmark("py_dot", py_dot3, py_v1, py_v2)
    print(f"{'dot3':<25} {t_np*1000:>10.2f}ms {t_py*1000:>10.2f}ms {t_np/t_py:>9.2f}x")
    
    # Cross product
    t_np = benchmark("np_cross", np_cross3, np_v1, np_v2)
    t_py = benchmark("py_cross", py_cross3, py_v1, py_v2)
    print(f"{'cross3':<25} {t_np*1000:>10.2f}ms {t_py*1000:>10.2f}ms {t_np/t_py:>9.2f}x")
    
    # Normalize
    t_np = benchmark("np_normalize", np_normalize, np_v1)
    t_py = benchmark("py_normalize", py_normalize, py_v1)
    print(f"{'normalize':<25} {t_np*1000:>10.2f}ms {t_py*1000:>10.2f}ms {t_np/t_py:>9.2f}x")
    
    # Length
    t_np = benchmark("np_length", np_length, np_v1)
    t_py = benchmark("py_length", py_length, py_v1)
    print(f"{'length':<25} {t_np*1000:>10.2f}ms {t_py*1000:>10.2f}ms {t_np/t_py:>9.2f}x")
    
    # Verify results match
    print("\n" + "=" * 60)
    print("Verification:")
    
    np_result = np_transform_point(np_point, np_matrix)
    py_result = py_transform_point(py_point, py_matrix)
    print(f"  transform_point: np={np_result}, py={py_result}")
    
    np_result = np_cross3(np_v1, np_v2)
    py_result = py_cross3(py_v1, py_v2)
    print(f"  cross3: np={np_result}, py={py_result}")
