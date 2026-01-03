"""
Benchmark the C extension vs NumPy vs Python solver.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml/tests')

from elements.solvers.gxml_intersection_solver import IntersectionSolver
from test_fixtures.mocks import GXMLMockPanel

# Import C extension
try:
    from elements.solvers._c_solvers import batch_find_intersections
    HAS_C_EXT = True
except ImportError as e:
    print(f"C extension not available: {e}")
    HAS_C_EXT = False


def batch_intersections_numpy(starts, ends, tol=1e-6):
    """NumPy vectorized intersection finding."""
    n = len(starts)
    dirs = ends - starts
    
    idx = np.triu_indices(n, k=1)
    n_pairs = len(idx[0])
    
    p1 = starts[idx[0]]
    p3 = starts[idx[1]]
    d1 = dirs[idx[0]]
    d2 = dirs[idx[1]]
    
    cross = np.cross(d1, d2)
    denom = np.sum(cross * cross, axis=1)
    w = p3 - p1
    
    w_cross_d2 = np.cross(w, d2)
    t1_num = np.sum(w_cross_d2 * cross, axis=1)
    
    w_cross_d1 = np.cross(w, d1)
    t2_num = np.sum(w_cross_d1 * cross, axis=1)
    
    parallel_mask = denom < tol * tol
    denom_safe = np.where(parallel_mask, 1.0, denom)
    
    t1 = t1_num / denom_safe
    t2 = t2_num / denom_safe
    
    valid = (~parallel_mask & 
             (t1 >= -tol) & (t1 <= 1 + tol) &
             (t2 >= -tol) & (t2 <= 1 + tol))
    
    pos1 = p1 + t1[:, np.newaxis] * d1
    pos2 = p3 + t2[:, np.newaxis] * d2
    
    pos_diff = pos1 - pos2
    pos_dist_sq = np.sum(pos_diff * pos_diff, axis=1)
    valid &= pos_dist_sq < tol * tol
    
    valid_idx = np.where(valid)[0]
    
    return (idx[0][valid_idx], idx[1][valid_idx], 
            t1[valid_idx], t2[valid_idx], pos1[valid_idx])


def create_test_data(n_horizontal: int, n_vertical: int):
    """Create test data as numpy arrays."""
    n_total = n_horizontal + n_vertical
    starts = np.zeros((n_total, 3), dtype=np.float64)
    ends = np.zeros((n_total, 3), dtype=np.float64)
    
    for i in range(n_horizontal):
        z = i * 10.0
        starts[i] = [0, 0, z]
        ends[i] = [n_vertical * 10.0, 0, z]
    
    for i in range(n_vertical):
        x = i * 10.0
        starts[n_horizontal + i] = [x, 0, 0]
        ends[n_horizontal + i] = [x, 0, n_horizontal * 10.0]
    
    return starts, ends


def create_gxml_panels(n_horizontal: int, n_vertical: int):
    """Create GXML panel objects."""
    panels = []
    
    for i in range(n_horizontal):
        z = i * 10.0
        p = GXMLMockPanel(f"h{i}", [0, 0, z], [n_vertical * 10.0, 0, z], 0.5, height=8.0)
        panels.append(p)
    
    for i in range(n_vertical):
        x = i * 10.0
        p = GXMLMockPanel(f"v{i}", [x, 0, 0], [x, 0, n_horizontal * 10.0], 0.5, height=8.0)
        panels.append(p)
    
    return panels


def benchmark(n_h, n_v, iterations=3):
    """Run benchmark comparison."""
    expected = n_h * n_v
    
    print(f"\n{'='*70}")
    print(f"Grid: {n_h}×{n_v} = {n_h + n_v} panels, {expected} expected intersections")
    print(f"{'='*70}")
    
    # Create test data
    starts, ends = create_test_data(n_h, n_v)
    
    # C extension
    if HAS_C_EXT:
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            result = batch_find_intersections(starts, ends)
            times.append(time.perf_counter() - t0)
        t_c = min(times)
        n_c = len(result[0])
        print(f"C extension:    {n_c:6d} intersections in {t_c*1000:8.2f} ms")
    else:
        t_c = None
        n_c = 0
    
    # NumPy
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = batch_intersections_numpy(starts, ends)
        times.append(time.perf_counter() - t0)
    t_numpy = min(times)
    n_numpy = len(result[0])
    print(f"NumPy batch:    {n_numpy:6d} intersections in {t_numpy*1000:8.2f} ms")
    
    # GXML solver
    panels = create_gxml_panels(n_h, n_v)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        solution = IntersectionSolver.solve(panels)
        times.append(time.perf_counter() - t0)
    t_gxml = min(times)
    n_gxml = len(solution.intersections)
    print(f"GXML solver:    {n_gxml:6d} intersections in {t_gxml*1000:8.2f} ms")
    
    # Speedups
    print(f"\nSpeedups vs GXML solver:")
    if t_c:
        print(f"  C extension: {t_gxml/t_c:6.1f}x faster")
    print(f"  NumPy batch: {t_gxml/t_numpy:6.1f}x faster")
    
    return t_c, t_numpy, t_gxml


if __name__ == "__main__":
    print("="*70)
    print("C Extension vs NumPy vs Python Solver Benchmark")
    print("="*70)
    
    results = []
    for n_h, n_v in [(10, 10), (25, 25), (50, 50), (100, 100)]:
        r = benchmark(n_h, n_v)
        results.append((n_h * n_v, r))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nIntersections | C Ext (ms) | NumPy (ms) | GXML (ms) | C/GXML | NumPy/GXML")
    print("-" * 75)
    for n_int, (t_c, t_numpy, t_gxml) in results:
        c_ms = f"{t_c*1000:.2f}" if t_c else "N/A"
        c_ratio = f"{t_gxml/t_c:.1f}x" if t_c else "N/A"
        print(f"{n_int:13d} | {c_ms:>10s} | {t_numpy*1000:>10.2f} | {t_gxml*1000:>9.2f} | {c_ratio:>6s} | {t_gxml/t_numpy:.1f}x")
