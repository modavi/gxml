"""Test if smaller field allocation helps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'gxml'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

from tests.test_fixtures.mocks import GXMLMockPanel

# Create test panels
panels = []
grid_size = 50
for row in range(grid_size):
    panels.append(GXMLMockPanel(f"h_{row}", [0, 0, row * 4], [200, 0, row * 4], thickness=0.5, height=5))
for col in range(grid_size):
    panels.append(GXMLMockPanel(f"v_{col}", [col * 4 + 2, 0, -4], [col * 4 + 2, 0, 200], thickness=0.5, height=5))

n = len(panels)
print(f"Created {n} panels")

# Allocate fields with EXACT size needed
small_starts = ti.Vector.field(3, dtype=ti.f32, shape=n)
small_ends = ti.Vector.field(3, dtype=ti.f32, shape=n)
small_found = ti.field(dtype=ti.i32, shape=(n, n))

@ti.kernel
def find_intersections_small(num: ti.i32):
    for i, j in ti.ndrange(num, num):
        if i < j:
            p1 = small_starts[i]
            p2 = small_ends[i]
            p3 = small_starts[j]
            p4 = small_ends[j]
            
            d1x = p2[0] - p1[0]
            d1z = p2[2] - p1[2]
            d2x = p4[0] - p3[0]
            d2z = p4[2] - p3[2]
            
            cross = d1x * d2z - d1z * d2x
            if ti.abs(cross) > 1e-4:
                dx = p3[0] - p1[0]
                dz = p3[2] - p1[2]
                t1 = (dx * d2z - dz * d2x) / cross
                t2 = (dx * d1z - dz * d1x) / cross
                if t1 >= 0 and t1 <= 1 and t2 >= 0 and t2 <= 1:
                    small_found[i, j] = 1

print("\n--- Using right-sized fields ---")

# Prepare data
starts_np = np.zeros((n, 3), dtype=np.float32)
ends_np = np.zeros((n, 3), dtype=np.float32)
for i, panel in enumerate(panels):
    start, end = panel.get_centerline_endpoints()
    starts_np[i] = [start[0], start[1], start[2]]
    ends_np[i] = [end[0], end[1], end[2]]

t0 = time.perf_counter()
small_starts.from_numpy(starts_np)
small_ends.from_numpy(ends_np)
t1 = time.perf_counter()
print(f"Upload ({n} panels): {(t1-t0)*1000:.2f} ms")

t2 = time.perf_counter()
find_intersections_small(n)
ti.sync()
t3 = time.perf_counter()
print(f"Kernel: {(t3-t2)*1000:.2f} ms")

t4 = time.perf_counter()
found_np = small_found.to_numpy()
t5 = time.perf_counter()
print(f"Download ({n}x{n} matrix): {(t5-t4)*1000:.2f} ms")

count = np.sum(found_np)
print(f"Found {count} intersections")

print(f"\nTotal: {(t5-t0)*1000:.2f} ms")

# Compare to pure CPU
print("\n--- Pure Python/NumPy (no Taichi) ---")
t0 = time.perf_counter()
found_py = 0
for i in range(n):
    for j in range(i+1, n):
        p1, p2 = starts_np[i], ends_np[i]
        p3, p4 = starts_np[j], ends_np[j]
        d1x, d1z = p2[0] - p1[0], p2[2] - p1[2]
        d2x, d2z = p4[0] - p3[0], p4[2] - p3[2]
        cross = d1x * d2z - d1z * d2x
        if abs(cross) > 1e-4:
            dx, dz = p3[0] - p1[0], p3[2] - p1[2]
            t1 = (dx * d2z - dz * d2x) / cross
            t2 = (dx * d1z - dz * d1x) / cross
            if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                found_py += 1
t1 = time.perf_counter()
print(f"Pure Python loop: {(t1-t0)*1000:.2f} ms ({found_py} intersections)")
