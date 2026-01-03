#!/usr/bin/env python3
"""
Benchmark comparing CPU vs Taichi GPU solver performance.

Run from venv312:
    source venv312/bin/activate
    python temp/bench_taichi_vs_cpu.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'gxml'))

def create_test_panels(num_panels: int):
    """Create a grid of intersecting panels."""
    from elements.gxml_panel import GXMLPanel
    from elements.gxml_base_element import GXMLBaseElement
    
    # Create a mock parent element
    class MockParent(GXMLBaseElement):
        tag = "Root"
        def __init__(self):
            super().__init__()
            self.computedX = 0
            self.computedY = 0
            self.computedWidth = 1000
            self.computedHeight = 1000
    
    parent = MockParent()
    panels = []
    
    # Create a grid of panels
    grid_size = int(num_panels ** 0.5) + 1
    spacing = 100
    panel_length = 150
    
    # Horizontal panels
    for i in range(grid_size):
        for j in range(grid_size // 2):
            panel = GXMLPanel()
            panel.parent = parent
            panel.id = f"h_{i}_{j}"
            panel.computedX = j * spacing
            panel.computedY = 0
            panel.computedWidth = panel_length
            panel.computedHeight = 50
            panel.thickness = 5
            # Position at different Z levels
            panel.computedZ = i * spacing
            panels.append(panel)
            
            if len(panels) >= num_panels:
                return panels
    
    # Vertical panels (crossing the horizontal ones)
    for i in range(grid_size):
        for j in range(grid_size // 2):
            panel = GXMLPanel()
            panel.parent = parent
            panel.id = f"v_{i}_{j}"
            panel.computedX = j * spacing + spacing // 2
            panel.computedY = 0
            panel.computedWidth = panel_length
            panel.computedHeight = 50
            panel.thickness = 5
            panel.computedZ = i * spacing
            # Rotate 90 degrees
            panel.computedRotation = 90
            panels.append(panel)
            
            if len(panels) >= num_panels:
                return panels
    
    return panels[:num_panels]


def benchmark_cpu(panels, iterations=5):
    """Benchmark CPU solvers."""
    from elements.solvers import (
        CPUIntersectionSolver, 
        CPUFaceSolver, 
        CPUGeometryBuilder
    )
    
    times = []
    
    for i in range(iterations):
        # Clear any cached geometry
        for p in panels:
            p.dynamicChildren = []
        
        start = time.perf_counter()
        
        # Run full pipeline
        solution = CPUIntersectionSolver.solve(panels)
        panel_faces = CPUFaceSolver.solve(solution)
        CPUGeometryBuilder.build_all(panel_faces, solution)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            print(f"  CPU warmup: {elapsed*1000:.1f}ms")
    
    return times[1:]  # Skip warmup


def benchmark_taichi(panels, iterations=5):
    """Benchmark Taichi GPU solvers."""
    try:
        from elements.solvers.taichi.taichi_intersection_solver import TaichiIntersectionSolver
        from elements.solvers.taichi.taichi_face_solver import TaichiFaceSolver
        from elements.solvers.taichi.taichi_geometry_builder import TaichiGeometryBuilder
    except ImportError as e:
        print(f"  Taichi import failed: {e}")
        return None
    
    times = []
    
    for i in range(iterations):
        # Clear any cached geometry
        for p in panels:
            p.dynamicChildren = []
        
        start = time.perf_counter()
        
        # Run full pipeline
        solution = TaichiIntersectionSolver.solve(panels)
        panel_faces = TaichiFaceSolver.solve(solution)
        TaichiGeometryBuilder.build_all(panel_faces, solution)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            print(f"  Taichi warmup: {elapsed*1000:.1f}ms (includes JIT compilation)")
    
    return times[1:]  # Skip warmup


def main():
    print("=" * 60)
    print("GXML Solver Benchmark: CPU vs Taichi GPU")
    print("=" * 60)
    
    # Test different panel counts
    panel_counts = [10, 25, 50, 75, 100]
    
    for num_panels in panel_counts:
        print(f"\n--- {num_panels} Panels ---")
        
        panels = create_test_panels(num_panels)
        print(f"Created {len(panels)} test panels")
        
        # CPU benchmark
        print("\nCPU Backend:")
        cpu_times = benchmark_cpu(panels, iterations=5)
        cpu_avg = sum(cpu_times) / len(cpu_times)
        cpu_min = min(cpu_times)
        print(f"  Average: {cpu_avg*1000:.1f}ms, Min: {cpu_min*1000:.1f}ms")
        
        # Taichi benchmark
        print("\nTaichi Backend:")
        taichi_times = benchmark_taichi(panels, iterations=5)
        
        if taichi_times:
            taichi_avg = sum(taichi_times) / len(taichi_times)
            taichi_min = min(taichi_times)
            print(f"  Average: {taichi_avg*1000:.1f}ms, Min: {taichi_min*1000:.1f}ms")
            
            # Speedup
            speedup = cpu_avg / taichi_avg
            print(f"\n  Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
