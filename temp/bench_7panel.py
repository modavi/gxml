"""Quick benchmark to test 7 panels CPU vs GPU."""
import sys
import time
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml/tests')

import taichi as ti
ti.init(arch=ti.cpu)

from gxml_layout import GXMLLayout
from gxml_parser import GXMLParser
from gxml_render import GXMLRender
from test_fixtures.mocks import GXMLTestRenderContext
from elements.solvers.taichi import set_solver_backend

xml = """<root>
    <panel thickness="0.25"/>
    <panel width="2.55" thickness="0.25" rotate="90" attach="0:1"/>
    <panel width="2.76" thickness="0.25" rotate="-135" attach="1:1"/>
    <panel width="2.873" thickness="0.25" rotate="-45" attach="2:1"/>
    <panel width="2.726" thickness="0.25" rotate="-90" attach="3:1"/>
    <panel width="4.716" thickness="0.25" rotate="315" attach="4:1"/>
    <panel width="6.608" thickness="0.25" rotate="-45" attach="5:1"/>
</root>"""

def run(xml):
    panel = GXMLParser.parse(xml)
    GXMLLayout.layout(panel)
    ctx = GXMLTestRenderContext()
    GXMLRender.render(panel, ctx)
    return panel

print('7 panel test (smaller, faster):')
print()

# CPU
set_solver_backend('cpu')
for _ in range(3): run(xml)  # warmup
times = []
for _ in range(10):
    start = time.perf_counter()
    run(xml)
    times.append((time.perf_counter() - start) * 1000)
print(f'CPU: {sum(times)/len(times):.2f}ms avg (min={min(times):.2f}, max={max(times):.2f})')

# GPU
set_solver_backend('gpu')
for _ in range(3): run(xml)  # warmup
times = []
for _ in range(10):
    start = time.perf_counter()
    run(xml)
    times.append((time.perf_counter() - start) * 1000)
print(f'GPU: {sum(times)/len(times):.2f}ms avg (min={min(times):.2f}, max={max(times):.2f})')
