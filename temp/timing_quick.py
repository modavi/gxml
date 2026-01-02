"""Quick timing test for 16 panels."""
import time
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml-web/src')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from gxml_web.json_render_engine import JSONRenderEngine

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

# Warmup
for _ in range(5):
    root = GXMLParser.parse(xml_16)
    GXMLLayout.layout(root)
    render_engine = JSONRenderEngine()
    GXMLRender.render(root, render_engine)

# Measure multiple runs, take minimum
times = []
for run in range(5):
    start = time.perf_counter()
    for _ in range(10):
        root = GXMLParser.parse(xml_16)
        GXMLLayout.layout(root)
        render_engine = JSONRenderEngine()
        GXMLRender.render(root, render_engine)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    print(f'Run {run+1}: {elapsed:.3f}s ({elapsed*100:.1f}ms avg)')

print(f'\nBest: {min(times)*100:.1f}ms per layout')
print(f'Average: {sum(times)/len(times)*100:.1f}ms per layout')
