"""Debug full render pipeline."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml-web/src')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from gxml_web.json_render_engine import JSONRenderEngine
from elements.gxml_panel import PanelSide
import numpy as np

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

p = find_panels(panel)[0]
print(f"Panel thickness: {p.thickness}")

# Create the FRONT face quad and check its transform
front_quad = p.create_panel_side("test", PanelSide.FRONT)
print(f"\nFRONT quad transform matrix:")
print(front_quad.transform.transformationMatrix)

print(f"\nFRONT quad get_world_vertices():")
for v in front_quad.get_world_vertices():
    print(f"  {v}")

print(f"\nFRONT quad transform_point tests:")
test_pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
for pt in test_pts:
    world = front_quad.transform_point(pt)
    print(f"  {pt} -> {world}")

