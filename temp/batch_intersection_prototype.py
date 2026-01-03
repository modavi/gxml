"""
Prototype: Batch Intersection Solver using NumPy

This tests the approach of doing ALL intersection computation in a single 
vectorized pass, which is what a C extension would do (but NumPy gives us
a quick prototype to validate the approach).

Key insight: The existing C extension (_vec3) already has SIMD support.
We could add a batch_find_intersections() function that:
1. Takes all panel endpoints as contiguous arrays
2. Does ALL pairwise checks in C with SIMD
3. Returns intersection results as arrays

Let's prototype this with NumPy first.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src')

from gxml.elements.solvers.gxml_intersection_solver import IntersectionSolver


def batch_centerline_intersections_numpy(starts: np.ndarray, ends: np.ndarray, 
                                          tol: float = 1e-6) -> tuple:
    """
    Find ALL pairwise centerline intersections using vectorized NumPy.
    
    This is what a C extension would do, but in NumPy for prototyping.
    
    Args:
        starts: (N, 3) array of segment start points
        ends: (N, 3) array of segment end points
        tol: tolerance for intersection checks
        
    Returns:
        Tuple of (panel_i, panel_j, t_i, t_j, positions) where each is an array
        of the intersecting pairs.
    """
    n = len(starts)
    
    # Direction vectors: (N, 3)
    dirs = ends - starts  
    
    # Build all pairwise indices (upper triangle only)
    idx = np.triu_indices(n, k=1)
    n_pairs = len(idx[0])
    
    # Get all pairs of start points and direction vectors
    # Shape: (n_pairs, 3)
    p1 = starts[idx[0]]  # Start points of first segments
    p3 = starts[idx[1]]  # Start points of second segments
    d1 = dirs[idx[0]]    # Directions of first segments
    d2 = dirs[idx[1]]    # Directions of second segments
    
    # Cross product of directions: d1 × d2
    # Shape: (n_pairs, 3)
    cross = np.cross(d1, d2)
    
    # Squared length of cross product (0 means parallel)
    denom = np.sum(cross * cross, axis=1)  # (n_pairs,)
    
    # Vector from p1 to p3
    w = p3 - p1  # (n_pairs, 3)
    
    # Compute t1 = (w × d2) · cross / denom
    w_cross_d2 = np.cross(w, d2)
    t1_num = np.sum(w_cross_d2 * cross, axis=1)
    
    # Compute t2 = (w × d1) · cross / denom
    w_cross_d1 = np.cross(w, d1)
    t2_num = np.sum(w_cross_d1 * cross, axis=1)
    
    # Avoid division by zero for parallel segments
    parallel_mask = denom < tol * tol
    denom_safe = np.where(parallel_mask, 1.0, denom)
    
    t1 = t1_num / denom_safe
    t2 = t2_num / denom_safe
    
    # Valid intersections: not parallel, t1 and t2 both in [0, 1]
    valid = (~parallel_mask & 
             (t1 >= -tol) & (t1 <= 1 + tol) &
             (t2 >= -tol) & (t2 <= 1 + tol))
    
    # Compute intersection points and check they match
    pos1 = p1 + t1[:, np.newaxis] * d1
    pos2 = p3 + t2[:, np.newaxis] * d2
    
    # Check that both intersection points are the same
    pos_diff = pos1 - pos2
    pos_dist_sq = np.sum(pos_diff * pos_diff, axis=1)
    valid &= pos_dist_sq < tol * tol
    
    # Extract valid results
    valid_idx = np.where(valid)[0]
    
    return (
        idx[0][valid_idx],  # panel indices i
        idx[1][valid_idx],  # panel indices j
        t1[valid_idx],       # t values for panel i
        t2[valid_idx],       # t values for panel j
        pos1[valid_idx],     # intersection positions
    )


def create_test_panels_numpy(n_horizontal: int, n_vertical: int):
    """Create a crossing grid pattern as numpy arrays in XZ plane."""
    n_total = n_horizontal + n_vertical
    starts = np.zeros((n_total, 3), dtype=np.float64)
    ends = np.zeros((n_total, 3), dtype=np.float64)
    
    # Horizontal panels (along X axis)
    for i in range(n_horizontal):
        z = i * 10.0
        starts[i] = [0, 0, z]
        ends[i] = [n_vertical * 10.0, 0, z]
    
    # Vertical panels (along Z axis)
    for i in range(n_vertical):
        x = i * 10.0
        starts[n_horizontal + i] = [x, 0, 0]
        ends[n_horizontal + i] = [x, 0, n_horizontal * 10.0]
    
    return starts, ends, n_total


def create_test_panels_gxml(n_horizontal: int, n_vertical: int):
    """Create the same grid pattern as GXML Panel objects in XZ plane."""
    import sys
    sys.path.insert(0, '/Users/morgan/Projects/gxml/tests')
    from test_fixtures.mocks import GXMLMockPanel
    
    panels = []
    
    # Horizontal panels (along X axis)
    for i in range(n_horizontal):
        z = i * 10.0
        p = GXMLMockPanel(
            panel_id=f"h{i}",
            start_pos=[0, 0, z],
            end_pos=[n_vertical * 10.0, 0, z],
            thickness=0.5,
            height=8.0
        )
        panels.append(p)
    
    # Vertical panels (along Z axis)
    for i in range(n_vertical):
        x = i * 10.0
        p = GXMLMockPanel(
            panel_id=f"v{i}",
            start_pos=[x, 0, 0],
            end_pos=[x, 0, n_horizontal * 10.0],
            thickness=0.5,
            height=8.0
        )
        panels.append(p)
    
    return panels


def benchmark_compare(n_horizontal: int, n_vertical: int):
    """Compare NumPy batch vs GXML IntersectionSolver."""
    expected_intersections = n_horizontal * n_vertical
    
    print(f"\n{'='*60}")
    print(f"Grid: {n_horizontal}×{n_vertical} = {n_horizontal + n_vertical} panels")
    print(f"Expected intersections: {expected_intersections}")
    print(f"{'='*60}")
    
    # NumPy batch approach
    starts, ends, n_total = create_test_panels_numpy(n_horizontal, n_vertical)
    
    t0 = time.perf_counter()
    result = batch_centerline_intersections_numpy(starts, ends)
    t_numpy = time.perf_counter() - t0
    
    n_found = len(result[0])
    print(f"\nNumPy batch intersection:")
    print(f"  Found: {n_found} intersections")
    print(f"  Time:  {t_numpy*1000:.2f} ms")
    
    # GXML IntersectionSolver
    panels = create_test_panels_gxml(n_horizontal, n_vertical)
    
    t0 = time.perf_counter()
    solution = IntersectionSolver.solve(panels)
    t_gxml = time.perf_counter() - t0
    
    print(f"\nGXML IntersectionSolver:")
    print(f"  Found: {len(solution.intersections)} intersections")
    print(f"  Time:  {t_gxml*1000:.2f} ms")
    
    print(f"\nSpeedup: {t_gxml/t_numpy:.1f}x faster with NumPy batch")
    
    return t_numpy, t_gxml


if __name__ == "__main__":
    print("="*60)
    print("Batch Intersection Solver Prototype")
    print("="*60)
    print("\nThis tests if a C extension would help by prototyping")
    print("the batch approach in NumPy first.\n")
    
    # Small test
    benchmark_compare(10, 10)
    
    # Medium test
    benchmark_compare(25, 25)
    
    # Large test
    benchmark_compare(50, 50)
    
    # Very large test  
    benchmark_compare(100, 100)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
The NumPy batch approach shows the potential speedup from:
1. Eliminating Python loops entirely
2. Doing all pairwise checks in one vectorized operation

A C extension would provide similar benefits, with potential for:
- SIMD vectorization (already in _vec3.c infrastructure)
- Even less overhead than NumPy
- Better memory locality

The key insight: We don't need to marshal data between stages if we
keep it in C memory. The C extension could return opaque handles that
get passed between solver stages without converting to Python objects.
""")
