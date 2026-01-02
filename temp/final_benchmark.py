"""Final performance benchmark comparing before/after today's optimizations."""
import time
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml-web/src')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from gxml_web.json_render_engine import JSONRenderEngine

# Test cases
xml_7_panels = '''<root>
    <panel id="p1" thickness="0.1"/>
    <panel id="p2" thickness="0.1" x="1.2"/>
    <panel id="p3" thickness="0.1" x="2.4"/>
    <panel id="c1" thickness="0.1" rotation="0 0 90"/>
    <panel id="c2" thickness="0.1" rotation="0 0 90" x="1.2"/>
    <panel id="c3" thickness="0.1" rotation="0 0 90" x="2.4"/>
    <panel id="c4" thickness="0.1" rotation="0 0 90" x="3.6"/>
</root>'''

xml_16_panels = '''<root>
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

def benchmark(xml, name, iterations=50):
    """Run benchmark and return best/avg times."""
    # Warmup
    for _ in range(5):
        root = GXMLParser.parse(xml)
        GXMLLayout.layout(root)
        render_engine = JSONRenderEngine()
        GXMLRender.render(root, render_engine)
    
    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        root = GXMLParser.parse(xml)
        GXMLLayout.layout(root)
        render_engine = JSONRenderEngine()
        GXMLRender.render(root, render_engine)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)
    
    best = min(times)
    avg = sum(times) / len(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    
    return best, avg, p95

print("=" * 70)
print("GXML Performance Benchmark")
print("=" * 70)
print()

# 7 panels
best, avg, p95 = benchmark(xml_7_panels, "7 panels")
print(f"7 panels:  best={best:.2f}ms  avg={avg:.2f}ms  p95={p95:.2f}ms")

# 16 panels  
best, avg, p95 = benchmark(xml_16_panels, "16 panels")
print(f"16 panels: best={best:.2f}ms  avg={avg:.2f}ms  p95={p95:.2f}ms")

print()
print("=" * 70)
print("Comparison with baseline (before today's optimizations):")
print("=" * 70)
print()
print("Metric         | Before | After  | Speedup")
print("-" * 45)
print(f"16 panels best | 115ms  | {best:.0f}ms   | {115/best:.1f}x")
print(f"16 panels avg  | ~200ms | {avg:.0f}ms   | {200/avg:.1f}x")
print()
print("Original baseline (before caching): 326ms → current: {:.0f}ms = {:.1f}x faster".format(best, 326/best))
