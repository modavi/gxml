"""Detailed profiler for pre-layout and post-layout passes."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml-web/src')

import time
import cProfile
import pstats
from io import StringIO

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from gxml_web.json_render_engine import JSONRenderEngine

# Test with 16 panels
xml_16 = '''<root>
    <panel id="p1" thickness="0.1"/>
    <panel id="p2" thickness="0.1" x="1.2"/>
    <panel id="p3" thickness="0.1" x="2.4"/>
    <panel id="p4" thickness="0.1" x="3.6"/>
    <panel id="p5" thickness="0.1" y="1.2"/>
    <panel id="p6" thickness="0.1" x="1.2" y="1.2"/>
    <panel id="p7" thickness="0.1" x="2.4" y="1.2"/>
    <panel id="p8" thickness="0.1" x="3.6" y="1.2"/>
    <panel id="c1" thickness="0.1" rotation="0 0 90"/>
    <panel id="c2" thickness="0.1" rotation="0 0 90" x="1.2"/>
    <panel id="c3" thickness="0.1" rotation="0 0 90" x="2.4"/>
    <panel id="c4" thickness="0.1" rotation="0 0 90" x="3.6"/>
    <panel id="c5" thickness="0.1" rotation="0 0 90" y="1.2"/>
    <panel id="c6" thickness="0.1" rotation="0 0 90" x="1.2" y="1.2"/>
    <panel id="c7" thickness="0.1" rotation="0 0 90" x="2.4" y="1.2"/>
    <panel id="c8" thickness="0.1" rotation="0 0 90" x="3.6" y="1.2"/>
</root>'''

def run_layout():
    for _ in range(10):
        root = GXMLParser.parse(xml_16)
        GXMLLayout.layout(root)
        render_engine = JSONRenderEngine()
        GXMLRender.render(root, render_engine)

# Profile
profiler = cProfile.Profile()
profiler.enable()
run_layout()
profiler.disable()

# Print results
s = StringIO()
stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
stats.print_stats(50)
print(s.getvalue())

print("\n" + "=" * 80)
print("Top functions by total time:")
print("=" * 80)
s2 = StringIO()
stats2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
stats2.print_stats(30)
print(s2.getvalue())
