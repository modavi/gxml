"""
Benchmark comparing CPU vs Taichi GPU solvers.
"""
import sys
import time
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml/tests')

# Initialize Taichi before importing solvers
import taichi as ti
ti.init(arch=ti.cpu)  # Start with CPU arch for testing

from gxml_layout import GXMLLayout
from gxml_parser import GXMLParser
from gxml_render import GXMLRender
from test_fixtures.mocks import GXMLTestRenderContext

# Use proper backend API from main solvers module
from elements.solvers import set_solver_backend, get_solver_backend

# 75-panel test XML from integration tests
XML_75_PANELS = """<root>
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
    <panel width="10.949" thickness="0.25" rotate="90" attach="37:1"/>
    <panel width="98.513" thickness="0.25" rotate="-45" attach="38:1"/>
    <panel width="62.132" thickness="0.25" rotate="-45" attach="39:1"/>
    <panel width="93.063" thickness="0.25" rotate="-135" attach="40:1"/>
    <panel width="36.601" thickness="0.25" attach="41:1"/>
    <panel width="29.459" thickness="0.25" rotate="-45" attach="42:1"/>
    <panel width="6.113" thickness="0.25" rotate="45" attach="43:1"/>
    <panel width="7.991" thickness="0.25" rotate="-45" attach="44:1"/>
    <panel width="3.706" thickness="0.25" rotate="225" attach="45:1"/>
    <panel width="4.254" thickness="0.25" rotate="-225" attach="46:1"/>
    <panel width="2.842" thickness="0.25" rotate="-45" attach="47:1"/>
    <panel width="3.318" thickness="0.25" attach="48:1"/>
    <panel width="4.059" thickness="0.25" rotate="-45" attach="49:1"/>
    <panel width="5.011" thickness="0.25" rotate="270" attach="50:1"/>
    <panel width="34.35" thickness="0.25" rotate="45" attach="51:1"/>
    <panel width="220.31" thickness="0.25" attach="52:1"/>
    <panel width="117.746" thickness="0.25" rotate="-135" attach="53:1"/>
    <panel width="39.222" thickness="0.25" rotate="-45" attach="54:1"/>
    <panel width="129.095" thickness="0.25" rotate="-45" attach="55:1"/>
    <panel width="44.185" thickness="0.25" rotate="90" attach="56:1"/>
    <panel width="17.985" thickness="0.25" rotate="-90" attach="57:1"/>
    <panel width="19.16" thickness="0.25" rotate="-90" attach="58:1"/>
    <panel width="79.035" thickness="0.25" rotate="-90" attach="59:1"/>
    <panel width="61.367" thickness="0.25" rotate="90" attach="60:1"/>
    <panel width="47.785" thickness="0.25" rotate="-270" attach="61:1"/>
    <panel width="84.833" thickness="0.25" rotate="90" attach="62:1"/>
    <panel width="35.99" thickness="0.25" rotate="-90" attach="63:1"/>
    <panel width="22.698" thickness="0.25" rotate="270" attach="64:1"/>
    <panel width="25.056" thickness="0.25" rotate="-45" attach="65:1"/>
    <panel width="48.026" thickness="0.25" rotate="-90" attach="66:1"/>
    <panel width="55.272" thickness="0.25" rotate="-135" attach="67:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="68:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="69:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="70:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="71:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="72:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="73:1"/>
    <panel width="0.586" thickness="0.25" rotate="-90" attach="74:1"/>
</root>"""

def run_pipeline(xml):
    """Run the full GXML pipeline."""
    panel = GXMLParser.parse(xml)
    GXMLLayout.layout(panel)
    ctx = GXMLTestRenderContext()
    GXMLRender.render(panel, ctx)
    return panel

def benchmark_backend(backend: str, iterations: int = 5):
    """Benchmark a specific backend."""
    set_solver_backend(backend)
    print(f"\n{'='*50}")
    print(f"Backend: {get_solver_backend().upper()}")
    print(f"{'='*50}")
    
    # Warm-up run
    _ = run_pipeline(XML_75_PANELS)
    
    # Timed runs
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        result = run_pipeline(XML_75_PANELS)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed * 1000)  # Convert to ms
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms")
    
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    
    print(f"\n  Average: {avg:.2f}ms")
    print(f"  Min: {min_t:.2f}ms, Max: {max_t:.2f}ms")
    
    return avg

def main():
    print("GXML Solver Backend Benchmark")
    print("=" * 50)
    print("Test: 75 panels with rotations/attachments")
    
    # Benchmark CPU (full CPU pipeline)
    cpu_avg = benchmark_backend('cpu', iterations=5)
    
    # Benchmark GPU (CPU solvers + GPU geometry builder via Metal)
    gpu_avg = benchmark_backend('gpu', iterations=5)
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"CPU Average: {cpu_avg:.2f}ms")
    print(f"GPU (Metal GeometryBuilder) Average: {gpu_avg:.2f}ms")
    
    if gpu_avg < cpu_avg:
        speedup = cpu_avg / gpu_avg
        print(f"\nGPU is {speedup:.2f}x FASTER")
    else:
        slowdown = gpu_avg / cpu_avg
        print(f"\nGPU is {slowdown:.2f}x SLOWER (expected for small workloads)")

if __name__ == '__main__':
    main()
