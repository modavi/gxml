"""Profile the TaichiIntersectionSolver to see where time is spent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'gxml'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import taichi as ti
ti.init(arch=ti.cpu)

from tests.test_fixtures.mocks import GXMLMockPanel

# Create test panels (50x50 grid = 100 panels, 2500 intersections)
panels = []
grid_size = 50
for row in range(grid_size):
    panels.append(GXMLMockPanel(f"h_{row}", [0, 0, row * 4], [200, 0, row * 4], thickness=0.5, height=5))
for col in range(grid_size):
    panels.append(GXMLMockPanel(f"v_{col}", [col * 4 + 2, 0, -4], [col * 4 + 2, 0, 200], thickness=0.5, height=5))

print(f"Created {len(panels)} panels")

# Import after ti.init
from elements.solvers.taichi.taichi_intersection_solver import (
    TaichiIntersectionSolver, panel_count, panel_starts, panel_ends,
    clear_intersection_buffers, find_all_intersections,
    intersection_found, intersection_t1, intersection_t2,
    intersection_x, intersection_y, intersection_z
)

n = len(panels)

# Time each phase
print("\n--- Profiling TaichiIntersectionSolver ---\n")

# Phase 1: Upload
t0 = time.perf_counter()
panel_count[None] = n
for i, panel in enumerate(panels):
    start, end = panel.get_centerline_endpoints()
    panel_starts[i] = ti.Vector([start[0], start[1], start[2]])
    panel_ends[i] = ti.Vector([end[0], end[1], end[2]])
t1 = time.perf_counter()
print(f"1. Upload panel data:     {(t1-t0)*1000:8.2f} ms  ({n} panels, {n*2} field writes)")

# Phase 2: Clear buffers
t2 = time.perf_counter()
clear_intersection_buffers()
ti.sync()
t3 = time.perf_counter()
print(f"2. Clear buffers kernel:  {(t3-t2)*1000:8.2f} ms  ({n*n} cells)")

# Phase 3: Find intersections kernel
t4 = time.perf_counter()
find_all_intersections()
ti.sync()
t5 = time.perf_counter()
print(f"3. Intersection kernel:   {(t5-t4)*1000:8.2f} ms  ({n*(n-1)//2} pairs tested)")

# Phase 4: Download results
t6 = time.perf_counter()
raw_intersections = []
reads = 0
for i in range(n):
    for j in range(i + 1, n):
        reads += 1
        if intersection_found[i, j]:
            raw_intersections.append({
                'panel1_idx': i,
                'panel2_idx': j,
                't1': float(intersection_t1[i, j]),
                't2': float(intersection_t2[i, j]),
                'position': (
                    float(intersection_x[i, j]),
                    float(intersection_y[i, j]),
                    float(intersection_z[i, j])
                )
            })
t7 = time.perf_counter()
print(f"4. Download results:      {(t7-t6)*1000:8.2f} ms  ({reads} field reads, {len(raw_intersections)} intersections found)")

# Phase 5: Build intersection objects
t8 = time.perf_counter()
intersections = TaichiIntersectionSolver._build_intersections(panels, raw_intersections)
t9 = time.perf_counter()
print(f"5. Build intersections:   {(t9-t8)*1000:8.2f} ms  ({len(intersections)} Intersection objects)")

# Phase 6: Build regions
t10 = time.perf_counter()
regions = TaichiIntersectionSolver._build_regions(panels, intersections)
t11 = time.perf_counter()
print(f"6. Build regions:         {(t11-t10)*1000:8.2f} ms  ({len(regions)} region trees)")

total = t11 - t0
print(f"\n   TOTAL:                 {total*1000:8.2f} ms")
print(f"\n   GPU work (kernel):     {(t5-t4)*1000:8.2f} ms ({(t5-t4)/total*100:.1f}%)")
print(f"   Python overhead:       {(total-(t5-t4))*1000:8.2f} ms ({(1-(t5-t4)/total)*100:.1f}%)")
