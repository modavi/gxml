"""
Test the Taichi GPU solvers to verify they work correctly.

This tests:
1. Import of Taichi solvers
2. Basic panel API (width, height, get_centerline_endpoints, get_world_transform_matrix)
3. Running the full Taichi pipeline
"""

import sys
import os
from pathlib import Path

# Add src/gxml to path (same as tests/conftest.py)
src_path = Path(__file__).parent.parent / "src" / "gxml"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add project root for test imports
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Initialize Taichi BEFORE importing gxml
import taichi as ti
ti.init(arch=ti.cpu)  # Use CPU arch for maximum compatibility

print("=" * 60)
print("Testing Taichi Solver Fixes")
print("=" * 60)

# Test 1: Import panel and check new APIs
print("\n1. Testing Panel API additions...")
from elements.gxml_panel import GXMLPanel, PanelSide
from tests.test_fixtures.mocks import GXMLMockPanel

# Create a simple panel (100 units long, height 50)
panel = GXMLMockPanel("test_panel", [0, 0, 0], [100, 0, 0], thickness=10, height=50)

# Test width property
print(f"   panel.width = {panel.width}")
assert abs(panel.width - 100) < 1e-4, f"Expected width ~100, got {panel.width}"

# Test height property
print(f"   panel.height = {panel.height}")
assert abs(panel.height - 50) < 1e-4, f"Expected height ~50, got {panel.height}"

# Test get_centerline_endpoints
start, end = panel.get_centerline_endpoints()
print(f"   get_centerline_endpoints(): start={start}, end={end}")
assert abs(start[0] - 0) < 1e-4, f"Expected start x ~0, got {start[0]}"
assert abs(end[0] - 100) < 1e-4, f"Expected end x ~100, got {end[0]}"

# Test get_world_transform_matrix
matrix = panel.get_world_transform_matrix()
print(f"   get_world_transform_matrix(): {type(matrix)}, shape={matrix.shape if hasattr(matrix, 'shape') else 'N/A'}")
print("   ✓ Panel API tests passed!")

# Test 2: Import Taichi solvers
print("\n2. Testing Taichi solver imports...")
try:
    from elements.solvers.taichi.taichi_intersection_solver import TaichiIntersectionSolver
    print("   ✓ TaichiIntersectionSolver imported")
except ImportError as e:
    print(f"   ✗ TaichiIntersectionSolver import failed: {e}")
    sys.exit(1)

try:
    from elements.solvers.taichi.taichi_face_solver import TaichiFaceSolver
    print("   ✓ TaichiFaceSolver imported")
except ImportError as e:
    print(f"   ✗ TaichiFaceSolver import failed: {e}")
    sys.exit(1)

try:
    from elements.solvers.taichi.taichi_geometry_builder import TaichiGeometryBuilder
    print("   ✓ TaichiGeometryBuilder imported")
except ImportError as e:
    print(f"   ✗ TaichiGeometryBuilder import failed: {e}")
    sys.exit(1)

# Test 3: Run the IntersectionSolver on simple panels
print("\n3. Testing TaichiIntersectionSolver...")

# Create two intersecting panels using GXMLMockPanel
# Panel 1: horizontal along X axis from (0,0,0) to (100,0,0)
panel1 = GXMLMockPanel("p1", [0, 0, 0], [100, 0, 0], thickness=10, height=50)

# Panel 2: perpendicular, from (50,0,-50) to (50,0,50)
panel2 = GXMLMockPanel("p2", [50, 0, -50], [50, 0, 50], thickness=10, height=50)

panels = [panel1, panel2]

try:
    solution = TaichiIntersectionSolver.solve(panels)
    print(f"   Found {len(solution.intersections)} intersections")
    for inter in solution.intersections:
        print(f"   - Type: {inter.type}, Position: {inter.position}")
    print("   ✓ TaichiIntersectionSolver test passed!")
except Exception as e:
    import traceback
    print(f"   ✗ TaichiIntersectionSolver test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Run the FaceSolver
print("\n4. Testing TaichiFaceSolver...")

try:
    panel_faces = TaichiFaceSolver.solve(solution)
    print(f"   Generated face segments for {len(panel_faces)} panels")
    for pf in panel_faces:
        seg_count = sum(len(segs) for segs in pf.segments.values())
        print(f"   - Panel has {seg_count} total segments")
    print("   ✓ TaichiFaceSolver test passed!")
except Exception as e:
    import traceback
    print(f"   ✗ TaichiFaceSolver test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Run the GeometryBuilder
print("\n5. Testing TaichiGeometryBuilder...")

try:
    TaichiGeometryBuilder.build_all(panel_faces, solution)
    
    # Check that polygons were created
    total_polygons = 0
    for panel in panels:
        total_polygons += len(panel.dynamicChildren)
    
    print(f"   Generated {total_polygons} total polygons")
    print("   ✓ TaichiGeometryBuilder test passed!")
except Exception as e:
    import traceback
    print(f"   ✗ TaichiGeometryBuilder test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
