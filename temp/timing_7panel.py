#!/usr/bin/env python
"""Time the 7-panel layout without profiler overhead."""
import sys
import time
sys.path.insert(0, 'src/gxml')
sys.path.insert(0, '.')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from tests.test_fixtures.mocks import GXMLTestRenderContext

xml = """<root>
    <panel width="1"/>
    <panel width="1" rotate="-90" attach="0:1"/>
    <panel width="1" rotate="-180" attach="1:1"/>
    <panel width="1" rotate="-90" attach="2:1"/>
    <panel width="1" rotate="-90" attach="3:1"/>
    <panel width="1" rotate="-90" attach="4:1"/>
    <panel width="1" rotate="-90" attach="5:1"/>
</root>"""

def run_pipeline():
    renderContext = GXMLTestRenderContext()
    root = GXMLParser.parse(xml)
    GXMLLayout.layout(root)
    GXMLRender.render(root, renderContext)
    return root

if __name__ == '__main__':
    # Warmup
    for _ in range(3):
        run_pipeline()
    
    times = []
    for i in range(10):
        start = time.perf_counter()
        run_pipeline()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    times = times[2:]  # Skip first 2
    print(f'7 panels - Average: {sum(times)/len(times):.0f}ms (min: {min(times):.0f}ms, max: {max(times):.0f}ms)')
