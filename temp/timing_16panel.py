"""Timing benchmark for 16 panels."""
import sys
import time
import statistics
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml')
from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from tests.test_fixtures.mocks import GXMLTestRenderContext

xml = '''<root>
<panel width="100" height="100" depth="100" />
<panel width="100" height="100" depth="100" t="50 50 0" />
<panel width="100" height="100" depth="100" t="100 100 0" />
<panel width="100" height="100" depth="100" t="150 150 0" />
<panel width="100" height="100" depth="100" t="200 200 0" />
<panel width="100" height="100" depth="100" t="250 250 0" />
<panel width="100" height="100" depth="100" t="300 300 0" />
<panel width="100" height="100" depth="100" t="350 350 0" />
<panel width="100" height="100" depth="100" t="400 400 0" />
<panel width="100" height="100" depth="100" t="450 450 0" />
<panel width="100" height="100" depth="100" t="500 500 0" />
<panel width="100" height="100" depth="100" t="550 550 0" />
<panel width="100" height="100" depth="100" t="600 600 0" />
<panel width="100" height="100" depth="100" t="650 650 0" />
<panel width="100" height="100" depth="100" t="700 700 0" />
<panel width="100" height="100" depth="100" t="750 750 0" />
</root>'''

def run_pipeline():
    renderContext = GXMLTestRenderContext()
    root = GXMLParser.parse(xml)
    GXMLLayout.layout(root)
    GXMLRender.render(root, renderContext)
    return root

# Warmup
for _ in range(3):
    run_pipeline()

# Time it
times = []
for _ in range(20):
    start = time.perf_counter()
    run_pipeline()
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f'16 panels - Avg: {statistics.mean(times):.1f}ms, Min: {min(times):.1f}ms, Max: {max(times):.1f}ms')
