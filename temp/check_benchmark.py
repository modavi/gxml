"""Check if benchmark is actually doing real work."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'gxml'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_fixtures.mocks import GXMLMockPanel
from elements.solvers import set_solver_backend, get_intersection_solver, get_face_solver, get_full_geometry_builder

# Create a small test grid - all horizontal, no intersections!
panels = []
rows, cols = 5, 5
for row in range(rows):
    for col in range(cols):
        x = col * 10
        z = row * 10
        panel = GXMLMockPanel(f'h_{row}_{col}', [x, 0, z], [x + 10, 0, z], thickness=0.5, height=5)
        panels.append(panel)

print(f'Created {len(panels)} panels')

set_solver_backend('cpu')
IntersectionSolver = get_intersection_solver()
FaceSolver = get_face_solver()
GeometryBuilder = get_full_geometry_builder()

solution = IntersectionSolver.solve(panels)
print(f'Found {len(solution.intersections)} intersections')

panel_faces = FaceSolver.solve(solution)
print(f'Generated faces for {len(panel_faces)} panels')

GeometryBuilder.build_all(panel_faces, solution)
total_polys = sum(len(p.dynamicChildren) for p in panels)
print(f'Generated {total_polys} total polygons')
