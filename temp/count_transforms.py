"""Quick transform_point count for 75 panels."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src')

from gxml.gxml_parser import GXMLParser
from gxml.gxml_layout import GXMLLayout
from gxml.gxml_render import GXMLRender
from gxml.render_engines.json_render_engine import JSONRenderEngine
# Count transform_point calls
original_transform = panel_module.GXMLPanel.transform_point
call_count = 0

def counting_transform(self, point):
    global call_count
    call_count += 1
    return original_transform(self, point)

panel_module.GXMLPanel.transform_point = counting_transform

# Run once
root = GXMLParser.parse(xml_75_panels)
GXMLLayout.layout(root)
render_engine = JSONRenderEngine()
GXMLRender.render(root, render_engine)

print(f'transform_point calls for 75 panels: {call_count}')
print(f'Per panel: {call_count / 75:.1f}')
