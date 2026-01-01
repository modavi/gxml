"""Debug thickness calculation."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout

xml = '<root><panel thickness="0.25"/></root>'
panel = GXMLParser.parse(xml)
GXMLLayout.layout(panel)

# Get the first panel element
def find_panels(elem):
    panels = []
    tag = getattr(elem, 'tag', elem.__class__.__name__.lower().replace('gxml', ''))
    if tag == 'panel':
        panels.append(elem)
    for child in getattr(elem, 'children', []):
        panels.extend(find_panels(child))
    return panels

panels = find_panels(panel)
p = panels[0]

print(f"Panel thickness: {p.thickness}")
print(f"Panel transform:\n{p.transform}")

# Check the local points for each side
from elements.gxml_panel import PanelSide
for side in PanelSide:
    local_pts = p.get_local_points_from_side(side)
    print(f"\n{side.name}:")
    print(f"  Local points: {local_pts}")
    
    # Check what world points we'd get
    for lp in local_pts:
        if side in (PanelSide.FRONT, PanelSide.BACK):
            local_for_transform = (lp[0], lp[1], (lp[2] - 0.5) * p.thickness)
        elif side in (PanelSide.TOP, PanelSide.BOTTOM):
            local_for_transform = (lp[0], lp[1], (lp[2] - 0.5) * p.thickness)
        else:  # START, END
            local_for_transform = (lp[0], lp[1], (lp[2] - 0.5) * p.thickness)
        
        world = p.transform_point(local_for_transform)
        print(f"  {lp} -> {local_for_transform} -> world {world}")
