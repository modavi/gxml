"""
Benchmark CPU vs GPU backend for the full GXML pipeline.

This uses the actual GXML test infrastructure.
"""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml')

import time

from tests.test_fixtures.base_integration_test import BaseIntegrationTest
from elements.solvers import set_geometry_backend, get_geometry_backend

# 75-panel test XML (from performance test)
XML_75_PANEL = """<root>
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


class BenchmarkRunner(BaseIntegrationTest):
    """Helper to run benchmarks using test infrastructure."""
    
    def run_benchmark(self, backend: str, n_iters: int = 5):
        """Run benchmark with specified backend."""
        set_geometry_backend(backend)
        print(f"    (backend now: {get_geometry_backend()})")
        
        # Warmup
        for _ in range(2):
            self.setUp()
            self.parsePanel(XML_75_PANEL)
        
        # Timed runs
        times = []
        for i in range(n_iters):
            self.setUp()
            start = time.perf_counter()
            self.parsePanel(XML_75_PANEL)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"    Iteration {i+1}: {elapsed*1000:.1f} ms")
        
        return sum(times) / len(times)


def main():
    print("=" * 60)
    print("GXML Backend Comparison: CPU vs GPU")
    print("=" * 60)
    print()
    
    runner = BenchmarkRunner()
    n_iters = 10
    
    # CPU backend
    print(f"Testing CPU backend ({n_iters} iterations)...")
    cpu_time = runner.run_benchmark('cpu', n_iters)
    print(f"  Average time: {cpu_time*1000:.1f} ms")
    
    # GPU backend  
    print(f"\nTesting GPU backend ({n_iters} iterations)...")
    gpu_time = runner.run_benchmark('gpu', n_iters)
    print(f"  Average time: {gpu_time*1000:.1f} ms")
    
    # Comparison
    print()
    print("-" * 60)
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"GPU backend is {speedup:.2f}x faster than CPU")
    else:
        slowdown = gpu_time / cpu_time
        print(f"GPU backend is {slowdown:.2f}x slower (batch overhead > transform savings)")
    
    print()
    print("Note: GPU backend uses vectorized NumPy on this system")
    print("      (Metal GPU not available)")


if __name__ == "__main__":
    main()
