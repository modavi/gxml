"""Quick transform_point count for 75 panels."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from render_engines.json_render_engine import JSONRenderEngine
import elements.gxml_panel as panel_module

xml_75_panels = '''<root>
    <panel width="57.979" thickness="0.25"/>
    <panel width="7.016" thickness="0.25" rotate="-90" attach="0:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="1:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="2:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="3:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="4:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="5:1"/>
    <panel width="27.586" thickness="0.25" rotate="-90" attach="6:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="7:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="8:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="9:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="10:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="11:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="12:1"/>
    <panel width="27.586" thickness="0.25" rotate="-90" attach="13:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="14:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="15:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="16:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="17:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="18:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="19:1"/>
    <panel width="27.586" thickness="0.25" rotate="-90" attach="20:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="21:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="22:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="23:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="24:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="25:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="26:1"/>
    <panel width="27.586" thickness="0.25" rotate="-90" attach="27:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="28:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="29:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="30:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="31:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="32:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="33:1"/>
    <panel width="27.586" thickness="0.25" rotate="-90" attach="34:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="35:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="36:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="37:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="38:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="39:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="40:1"/>
    <panel width="27.586" thickness="0.25" rotate="-90" attach="41:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="42:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="43:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="44:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="45:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="46:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="47:1"/>
    <panel width="27.586" thickness="0.25" rotate="-90" attach="48:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="49:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="50:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="51:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="52:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="53:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="54:1"/>
    <panel width="27.586" thickness="0.25" rotate="-90" attach="55:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="56:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="57:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="58:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="59:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="60:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="61:1"/>
    <panel width="27.586" thickness="0.25" rotate="-90" attach="62:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="63:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="64:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="65:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="66:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="67:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="68:1"/>
    <panel width="27.586" thickness="0.25" rotate="-90" attach="69:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="70:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="71:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="72:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="73:1"/>
</root>'''

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
