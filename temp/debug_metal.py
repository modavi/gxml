"""Debug the Metal RHI shader compilation issue."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'gxml'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import taichi as ti

print("Taichi version:", ti.__version__)
print("Attempting to init with Metal backend...")

try:
    ti.init(arch=ti.metal, debug=True)
    print("Metal init successful!")
except Exception as e:
    print(f"Metal init failed: {e}")
    sys.exit(1)

# Test a simple kernel first
print("\n--- Testing simple kernel ---")
test_field = ti.field(dtype=ti.f32, shape=10)

@ti.kernel
def simple_kernel():
    for i in range(10):
        test_field[i] = float(i) * 2.0

try:
    simple_kernel()
    ti.sync()
    print("Simple kernel: SUCCESS")
    print(f"Result: {[test_field[i] for i in range(10)]}")
except Exception as e:
    print(f"Simple kernel FAILED: {e}")

# Test vector field
print("\n--- Testing vector field ---")
vec_field = ti.Vector.field(3, dtype=ti.f32, shape=10)

@ti.kernel
def vec_kernel():
    for i in range(10):
        vec_field[i] = ti.Vector([float(i), float(i)*2, float(i)*3])

try:
    vec_kernel()
    ti.sync()
    print("Vector kernel: SUCCESS")
except Exception as e:
    print(f"Vector kernel FAILED: {e}")

# Test 2D field (like intersection_found)
print("\n--- Testing 2D field ---")
field_2d = ti.field(dtype=ti.i32, shape=(100, 100))

@ti.kernel
def kernel_2d():
    for i, j in ti.ndrange(100, 100):
        if i < j:
            field_2d[i, j] = 1

try:
    kernel_2d()
    ti.sync()
    print("2D field kernel: SUCCESS")
except Exception as e:
    print(f"2D field kernel FAILED: {e}")

# Test the actual intersection solver fields
print("\n--- Testing intersection solver pattern ---")
MAX_PANELS = 1024
panel_starts = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PANELS)
panel_ends = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PANELS)
panel_count = ti.field(dtype=ti.i32, shape=())
intersection_found = ti.field(dtype=ti.i32, shape=(MAX_PANELS, MAX_PANELS))

@ti.kernel
def find_intersections_test():
    n = panel_count[None]
    for i, j in ti.ndrange(n, n):
        if i < j:
            p1 = panel_starts[i]
            p2 = panel_ends[i]
            # Simple check
            if p1[0] < p2[0]:
                intersection_found[i, j] = 1

try:
    panel_count[None] = 10
    find_intersections_test()
    ti.sync()
    print("Intersection pattern kernel: SUCCESS")
except Exception as e:
    print(f"Intersection pattern kernel FAILED: {e}")
    import traceback
    traceback.print_exc()
