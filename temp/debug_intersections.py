"""Debug why GXML IntersectionSolver finds 0 intersections."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml/tests')

from test_fixtures.mocks import GXMLMockPanel
from elements.solvers.gxml_intersection_solver import IntersectionSolver

# Create a simple 2x2 grid in XZ plane (horizontal plane)
# Horizontal = along X, Vertical = along Z
panels = []

# Two horizontal panels (along X)
p1 = GXMLMockPanel("h0", start_pos=[0, 0, 0], end_pos=[100, 0, 0], thickness=0.5, height=8.0)
p2 = GXMLMockPanel("h1", start_pos=[0, 0, 20], end_pos=[100, 0, 20], thickness=0.5, height=8.0)

# Two vertical panels (along Z)
p3 = GXMLMockPanel("v0", start_pos=[30, 0, -10], end_pos=[30, 0, 40], thickness=0.5, height=8.0)
p4 = GXMLMockPanel("v1", start_pos=[70, 0, -10], end_pos=[70, 0, 40], thickness=0.5, height=8.0)

panels = [p1, p2, p3, p4]

print("Panel centerlines:")
for p in panels:
    start, end = p.get_centerline_endpoints()
    print(f"  {p.id}: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f}) -> ({end[0]:.1f}, {end[1]:.1f}, {end[2]:.1f})")

print("\nSolving intersections...")
solution = IntersectionSolver.solve(panels)

print(f"\nFound {len(solution.intersections)} intersections:")
for inter in solution.intersections:
    print(f"  Type: {inter.type}, Position: ({inter.position[0]:.1f}, {inter.position[1]:.1f}, {inter.position[2]:.1f})")
    for entry in inter.panels:
        print(f"    - {entry.panel.id} at t={entry.t:.3f}")
