"""Test using numpy for bulk upload/download with Taichi."""
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

print(f"Created {len(panels)} panels")
n = len(panels)

# Import Taichi fields
from elements.solvers.taichi.taichi_intersection_solver import (
    panel_count, panel_starts, panel_ends,
    clear_intersection_buffers, find_all_intersections,
    intersection_found, intersection_t1, intersection_t2,
    intersection_x, intersection_y, intersection_z, MAX_PANELS
)

print("\n--- Method 1: Individual field writes (current) ---")
t0 = time.perf_counter()
panel_count[None] = n
for i, panel in enumerate(panels):
    start, end = panel.get_centerline_endpoints()
    panel_starts[i] = ti.Vector([start[0], start[1], start[2]])
    panel_ends[i] = ti.Vector([end[0], end[1], end[2]])
t1 = time.perf_counter()
print(f"Upload time: {(t1-t0)*1000:.2f} ms")

print("\n--- Method 2: NumPy bulk upload ---")
# Prepare numpy arrays
starts_np = np.zeros((MAX_PANELS, 3), dtype=np.float32)
ends_np = np.zeros((MAX_PANELS, 3), dtype=np.float32)

t0 = time.perf_counter()
for i, panel in enumerate(panels):
    start, end = panel.get_centerline_endpoints()
    starts_np[i] = [start[0], start[1], start[2]]
    ends_np[i] = [end[0], end[1], end[2]]
t1 = time.perf_counter()
print(f"Prepare numpy arrays: {(t1-t0)*1000:.2f} ms")

t2 = time.perf_counter()
panel_starts.from_numpy(starts_np)
panel_ends.from_numpy(ends_np)
panel_count[None] = n
t3 = time.perf_counter()
print(f"Bulk upload to Taichi: {(t3-t2)*1000:.2f} ms")
print(f"Total upload: {(t3-t0)*1000:.2f} ms")

# Run kernel
clear_intersection_buffers()
find_all_intersections()
ti.sync()

print("\n--- Method 1: Individual field reads (current) ---")
t0 = time.perf_counter()
raw1 = []
for i in range(n):
    for j in range(i + 1, n):
        if intersection_found[i, j]:
            raw1.append({
                't1': float(intersection_t1[i, j]),
                't2': float(intersection_t2[i, j]),
            })
t1 = time.perf_counter()
print(f"Download time: {(t1-t0)*1000:.2f} ms ({len(raw1)} intersections)")

print("\n--- Method 2: NumPy bulk download ---")
t0 = time.perf_counter()
found_np = intersection_found.to_numpy()
t1_np = intersection_t1.to_numpy()
t2_np = intersection_t2.to_numpy()
x_np = intersection_x.to_numpy()
y_np = intersection_y.to_numpy()
z_np = intersection_z.to_numpy()
t1 = time.perf_counter()
print(f"Bulk download to numpy: {(t1-t0)*1000:.2f} ms")

t2 = time.perf_counter()
raw2 = []
for i in range(n):
    for j in range(i + 1, n):
        if found_np[i, j]:
            raw2.append({
                't1': float(t1_np[i, j]),
                't2': float(t2_np[i, j]),
            })
t3 = time.perf_counter()
print(f"Process numpy arrays: {(t3-t2)*1000:.2f} ms")
print(f"Total download: {(t3-t0)*1000:.2f} ms ({len(raw2)} intersections)")

print(f"\nResults match: {len(raw1) == len(raw2)}")
