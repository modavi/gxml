#!/usr/bin/env python
"""Profile Vec3 operations to understand allocation patterns."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml')

import cProfile
import pstats
from io import StringIO

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from tests.test_fixtures.mocks import GXMLTestRenderContext

xml = '''<root>
    <panel thickness="0.25"/>
    <panel width="2.55" thickness="0.25" rotate="90" attach="0:1"/>
    <panel width="2.76" thickness="0.25" rotate="-135" attach="1:1"/>
    <panel width="2.873" thickness="0.25" rotate="-45" attach="2:1"/>
    <panel width="2.726" thickness="0.25" rotate="-90" attach="3:1"/>
    <panel width="4.716" thickness="0.25" rotate="315" attach="4:1"/>
    <panel width="6.608" thickness="0.25" rotate="-45" attach="5:1"/>
    <panel width="4.568" thickness="0.25" rotate="-90" attach="6:1"/>
    <panel width="2.627" thickness="0.25" rotate="-45" attach="7:1"/>
    <panel width="8.179" thickness="0.25" rotate="-45" attach="8:1"/>
    <panel width="2.338" thickness="0.25" attach="9:1"/>
    <panel width="3.419" thickness="0.25" rotate="-90" attach="10:1"/>
    <panel width="12.747" thickness="0.25" rotate="315" attach="11:1"/>
    <panel width="15.002" thickness="0.25" rotate="-45" attach="12:1"/>
    <panel width="11.687" thickness="0.25" rotate="-135" attach="13:1"/>
    <panel width="4.46" thickness="0.25" rotate="45" attach="14:1"/>
    <panel width="12.011" thickness="0.25" rotate="-90" attach="15:1"/>
    <panel width="2.839" thickness="0.25" rotate="45" attach="16:1"/>
    <panel width="1.719" thickness="0.25" rotate="-90" attach="17:1"/>
    <panel width="2.481" thickness="0.25" rotate="45" attach="18:1"/>
    <panel width="3.649" thickness="0.25" rotate="-90" attach="19:1"/>
    <panel width="10.153" thickness="0.25" rotate="315" attach="20:1"/>
    <panel width="9.466" thickness="0.25" rotate="45" attach="21:1"/>
    <panel width="48.329" thickness="0.25" rotate="-45" attach="22:1"/>
    <panel width="33.266" thickness="0.25" rotate="-90" attach="23:1"/>
    <panel width="39.774" thickness="0.25" rotate="-90" attach="24:1"/>
    <panel width="9.288" thickness="0.25" rotate="-90" attach="25:1"/>
    <panel width="6.634" thickness="0.25" rotate="270" attach="26:1"/>
    <panel width="6.199" thickness="0.25" rotate="-315" attach="27:1"/>
    <panel width="7.363" thickness="0.25" rotate="90" attach="28:1"/>
    <panel width="19.744" thickness="0.25" rotate="90" attach="29:1"/>
    <panel width="18.081" thickness="0.25" rotate="-90" attach="30:1"/>
    <panel width="4.109" thickness="0.25" rotate="45" attach="31:1"/>
    <panel width="3.253" thickness="0.25" rotate="-45" attach="32:1"/>
    <panel width="2.065" thickness="0.25" attach="33:1"/>
    <panel width="8.026" thickness="0.25" attach="34:1"/>
    <panel width="9.161" thickness="0.25" rotate="270" attach="35:1"/>
    <panel width="4.546" thickness="0.25" rotate="-90" attach="36:1"/>
    <panel width="13.424" thickness="0.25" rotate="-45" attach="37:1"/>
    <panel width="2.398" thickness="0.25" rotate="315" attach="38:1"/>
    <panel width="2.76" thickness="0.25" rotate="315" attach="39:1"/>
    <panel width="2.879" thickness="0.25" rotate="-315" attach="40:1"/>
    <panel width="2.726" thickness="0.25" rotate="270" attach="41:1"/>
    <panel width="3.419" thickness="0.25" rotate="-90" attach="42:1"/>
    <panel width="4.185" thickness="0.25" rotate="-45" attach="43:1"/>
    <panel width="1.897" thickness="0.25" rotate="315" attach="44:1"/>
    <panel width="5.113" thickness="0.25" rotate="315" attach="45:1"/>
    <panel width="9.003" thickness="0.25" rotate="315" attach="46:1"/>
    <panel width="4.668" thickness="0.25" rotate="-90" attach="47:1"/>
    <panel width="5.135" thickness="0.25" rotate="-45" attach="48:1"/>
    <panel width="5.278" thickness="0.25" rotate="-90" attach="49:1"/>
    <panel width="2.782" thickness="0.25" rotate="45" attach="50:1"/>
    <panel width="1.746" thickness="0.25" rotate="-90" attach="51:1"/>
    <panel width="3.605" thickness="0.25" rotate="45" attach="52:1"/>
    <panel width="5.113" thickness="0.25" rotate="315" attach="53:1"/>
    <panel width="9.003" thickness="0.25" rotate="315" attach="54:1"/>
    <panel width="5.893" thickness="0.25" rotate="-90" attach="55:1"/>
    <panel width="8.546" thickness="0.25" rotate="-90" attach="56:1"/>
    <panel width="10.153" thickness="0.25" rotate="315" attach="57:1"/>
    <panel width="9.466" thickness="0.25" rotate="45" attach="58:1"/>
    <panel width="4.06" thickness="0.25" rotate="-45" attach="59:1"/>
    <panel width="1.953" thickness="0.25" rotate="-90" attach="60:1"/>
    <panel width="4.849" thickness="0.25" rotate="315" attach="61:1"/>
    <panel width="6.608" thickness="0.25" rotate="315" attach="62:1"/>
    <panel width="4.362" thickness="0.25" rotate="-45" attach="63:1"/>
    <panel width="2.879" thickness="0.25" rotate="-315" attach="64:1"/>
    <panel width="2.726" thickness="0.25" rotate="270" attach="65:1"/>
    <panel width="3.419" thickness="0.25" rotate="-90" attach="66:1"/>
    <panel width="4.185" thickness="0.25" rotate="-45" attach="67:1"/>
    <panel width="1.897" thickness="0.25" rotate="315" attach="68:1"/>
    <panel width="5.113" thickness="0.25" rotate="315" attach="69:1"/>
    <panel width="9.003" thickness="0.25" rotate="315" attach="70:1"/>
    <panel width="4.668" thickness="0.25" rotate="-90" attach="71:1"/>
    <panel width="5.135" thickness="0.25" rotate="-45" attach="72:1"/>
    <panel width="5.278" thickness="0.25" rotate="-90" attach="73:1"/>
</root>'''

def run_pipeline():
    root = GXMLParser.parse(xml)
    GXMLLayout.layout(root)
    ctx = GXMLTestRenderContext()
    GXMLRender.render(root, ctx)

# Profile with cProfile
pr = cProfile.Profile()
pr.enable()
for _ in range(5):
    run_pipeline()
pr.disable()

# Print stats sorted by cumulative time
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(50)
print(s.getvalue())

# Also show call counts for Vec3-related functions
print("\n=== Vec3-related function calls ===")
s2 = StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats('calls')
ps2.print_stats('vec3|Vec3|transform_point|transform_direction', 30)
print(s2.getvalue())
