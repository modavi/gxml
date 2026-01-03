"""
Benchmark CPU vs Taichi (GPU) solvers on the 75-panel test.

This compares the full pipeline (IntersectionSolver, FaceSolver, GeometryBuilder)
between CPU and Taichi implementations.
"""

import sys
import os
from pathlib import Path
import time

# Add src/gxml to path (same as tests/conftest.py)
src_path = Path(__file__).parent.parent / "src" / "gxml"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add project root for test imports
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Initialize Taichi BEFORE importing anything else
import taichi as ti
ti.init(arch=ti.cpu)  # Use CPU arch - Metal has RHI issues on this system
print("[Using Taichi CPU backend]")

from tests.test_fixtures.mocks import GXMLMockPanel
from elements.solvers import set_solver_backend, get_intersection_solver, get_face_solver, get_full_geometry_builder

def create_75_panel_grid():
    """Create a grid of intersecting panels for benchmarking."""
    panels = []
    
    # Grid dimensions - create crossing grid pattern
    # 50 horizontal + 50 vertical = 100 panels with 2500 intersections
    grid_size = 50
    panel_length = 200
    panel_height = 5
    thickness = 0.5
    spacing = 4  # Distance between parallel panels
    
    # Create horizontal panels (along X axis)
    for row in range(grid_size):
        z = row * spacing
        panel = GXMLMockPanel(
            f"h_{row}",
            [0, 0, z],
            [panel_length, 0, z],
            thickness=thickness,
            height=panel_height
        )
        panels.append(panel)
    
    # Create vertical panels (along Z axis) that cross the horizontals
    for col in range(grid_size):
        x = col * spacing + spacing/2  # Offset so they intersect in middle
        panel = GXMLMockPanel(
            f"v_{col}",
            [x, 0, -spacing],
            [x, 0, panel_length],
            thickness=thickness,
            height=panel_height
        )
        panels.append(panel)
    
    return panels


def benchmark_pipeline(panels, backend, warmup=1, iterations=3):
    """Benchmark a solver pipeline with the given backend."""
    set_solver_backend(backend)
    
    IntersectionSolver = get_intersection_solver()
    FaceSolver = get_face_solver()
    GeometryBuilder = get_full_geometry_builder()
    
    # Warmup
    for _ in range(warmup):
        for p in panels:
            p.dynamicChildren.clear()  # Reset geometry
        solution = IntersectionSolver.solve(panels)
        panel_faces = FaceSolver.solve(solution)
        GeometryBuilder.build_all(panel_faces, solution)
    
    # Timed runs
    times = []
    for i in range(iterations):
        for p in panels:
            p.dynamicChildren.clear()
        
        start = time.perf_counter()
        
        t1 = time.perf_counter()
        solution = IntersectionSolver.solve(panels)
        t2 = time.perf_counter()
        panel_faces = FaceSolver.solve(solution)
        t3 = time.perf_counter()
        GeometryBuilder.build_all(panel_faces, solution)
        t4 = time.perf_counter()
        
        total = t4 - start
        times.append({
            'total': total,
            'intersection': t2 - t1,
            'face': t3 - t2,
            'geometry': t4 - t3,
        })
    
    # Calculate averages
    avg = {
        'total': sum(t['total'] for t in times) / len(times),
        'intersection': sum(t['intersection'] for t in times) / len(times),
        'face': sum(t['face'] for t in times) / len(times),
        'geometry': sum(t['geometry'] for t in times) / len(times),
    }
    
    return avg


def main():
    print("=" * 70)
    print("GXML Solver Benchmark: CPU vs Taichi")
    print("=" * 70)
    
    # Create test panels
    print("\nCreating 75-panel grid...")
    panels = create_75_panel_grid()
    print(f"Created {len(panels)} panels")
    
    # Benchmark CPU
    print("\n" + "-" * 70)
    print("Benchmarking CPU backend...")
    print("-" * 70)
    
    cpu_times = benchmark_pipeline(panels, 'cpu', warmup=1, iterations=5)
    
    print(f"  IntersectionSolver: {cpu_times['intersection']*1000:.2f} ms")
    print(f"  FaceSolver:         {cpu_times['face']*1000:.2f} ms")
    print(f"  GeometryBuilder:    {cpu_times['geometry']*1000:.2f} ms")
    print(f"  Total:              {cpu_times['total']*1000:.2f} ms")
    
    # Benchmark Taichi
    print("\n" + "-" * 70)
    print("Benchmarking Taichi backend (CPU arch)...")
    print("-" * 70)
    
    taichi_times = benchmark_pipeline(panels, 'taichi', warmup=1, iterations=5)
    
    print(f"  IntersectionSolver: {taichi_times['intersection']*1000:.2f} ms")
    print(f"  FaceSolver:         {taichi_times['face']*1000:.2f} ms")
    print(f"  GeometryBuilder:    {taichi_times['geometry']*1000:.2f} ms")
    print(f"  Total:              {taichi_times['total']*1000:.2f} ms")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    
    speedup = cpu_times['total'] / taichi_times['total']
    print(f"\nTotal speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    speedup_intersection = cpu_times['intersection'] / taichi_times['intersection'] if taichi_times['intersection'] > 0 else float('inf')
    speedup_face = cpu_times['face'] / taichi_times['face'] if taichi_times['face'] > 0 else float('inf')
    speedup_geometry = cpu_times['geometry'] / taichi_times['geometry'] if taichi_times['geometry'] > 0 else float('inf')
    
    print(f"  IntersectionSolver: {speedup_intersection:.2f}x")
    print(f"  FaceSolver:         {speedup_face:.2f}x")
    print(f"  GeometryBuilder:    {speedup_geometry:.2f}x")


if __name__ == "__main__":
    main()
