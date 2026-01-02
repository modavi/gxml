"""Debug spatial grid for failing test."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout

xml = '''<root>
    <panel thickness="0.1"/>
    <panel thickness="0.1" rotate="90"/>
    <panel thickness="0.1" rotate="90"/>
    <panel thickness="0.1" rotate="90"/>
    <panel thickness="0.1" span-id="1" span-point="1.0"/>
    <panel thickness="0.1" attach-id="0" span-id="2" span-point="1.0"/>
</root>'''

root = GXMLParser.parse(xml)
GXMLLayout.layout(root)

panels = [e for e in root.children if hasattr(e, 'thickness')]
print(f'{len(panels)} panels')

# Show panel positions
for i, p in enumerate(panels):
    start = p.transform_point([0,0,0])
    end = p.transform_point([1,0,0])
    print(f'Panel {i}: ({start[0]:.2f},{start[1]:.2f},{start[2]:.2f}) -> ({end[0]:.2f},{end[1]:.2f},{end[2]:.2f})')

# Test grid building
from elements.solvers.gxml_intersection_solver import _build_spatial_grid, _get_candidate_pairs_from_grid
grid = _build_spatial_grid(panels, cell_size=20.0)
print(f'\nGrid cells: {len(grid)}')
for cell, items in grid.items():
    print(f'  Cell {cell}: {len(items)} panels')

pairs = _get_candidate_pairs_from_grid(grid)
print(f'\nCandidate pairs: {len(pairs)}')
print(f'Expected pairs (all): {len(panels) * (len(panels)-1) // 2}')

# Show which cells each panel is in
print('\nPanel cell assignments:')
cell_size = 20.0
for i, p in enumerate(panels):
    start = p.transform_point([0,0,0])
    end = p.transform_point([1,0,0])
    min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
    min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
    min_z, max_z = min(start[2], end[2]), max(start[2], end[2])
    
    min_cx = int(min_x // cell_size)
    max_cx = int(max_x // cell_size)
    min_cy = int(min_y // cell_size)
    max_cy = int(max_y // cell_size)
    min_cz = int(min_z // cell_size)
    max_cz = int(max_z // cell_size)
    
    cells = []
    for cx in range(min_cx, max_cx + 1):
        for cy in range(min_cy, max_cy + 1):
            for cz in range(min_cz, max_cz + 1):
                cells.append((cx, cy, cz))
    print(f'  Panel {i}: x=[{min_x:.6f}, {max_x:.6f}] cx=[{min_cx}, {max_cx}] z=[{min_z:.15f}, {max_z:.15f}] cz=[{min_cz}, {max_cz}] -> {cells}')

# Show which pairs we get
pair_ids = set()
for p1, p2, _, _, _, _ in pairs:
    i1 = panels.index(p1)
    i2 = panels.index(p2)
    pair_ids.add((min(i1,i2), max(i1,i2)))

print('\nMissing pairs:', set((i,j) for i in range(len(panels)) for j in range(i+1, len(panels))) - pair_ids)
