"""
Performance integration tests for GXML processing pipeline.

These tests measure execution time for various XML configurations
to track performance regressions and establish baselines.
"""
import time
import statistics
from tests.test_fixtures.base_integration_test import BaseIntegrationTest


class TestPerformance(BaseIntegrationTest):
    """Performance benchmarks for GXML processing."""

    def test_basic_7_panel_layout(self):
        """Benchmark: 7 panels with various rotations and attachments."""
        xml = """<root>
    <panel thickness="0.25"/>
    <panel width="2.347" thickness="0.25" rotate="-90" attach="0:1"/>
    <panel width="1.522" thickness="0.25" rotate="90" attach="1:1"/>
    <panel width="1.637" thickness="0.25" rotate="90" attach="2:1"/>
    <panel width="4.296" thickness="0.25" rotate="90" attach="3:1"/>
    <panel width="4.854" thickness="0.25" rotate="-225" attach="4:1"/>
    <panel width="6.332" thickness="0.25" rotate="135" attach="5:1"/>
</root>"""

        # Warmup run
        self.parsePanel(xml)

        # Timed runs
        iterations = 10
        times = []
        
        for _ in range(iterations):
            # Need fresh renderContext each iteration
            self.setUp()
            
            start = time.perf_counter()
            result = self.parsePanel(xml)
            end = time.perf_counter()
            times.append(end - start)
            
            # Verify we got valid output
            assert result is not None

        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)

        print(f"\n{'='*60}")
        print(f"Performance: 7 panels with rotations/attachments")
        print(f"{'='*60}")
        print(f"  Iterations: {iterations}")
        print(f"  Average:    {avg_time*1000:.2f} ms")
        print(f"  Std Dev:    {std_dev*1000:.2f} ms")
        print(f"  Min:        {min_time*1000:.2f} ms")
        print(f"  Max:        {max_time*1000:.2f} ms")
        print(f"{'='*60}")

        # Optimized baseline is ~380ms (was ~1.2s before optimizations)
        # Fail if it regresses beyond 2x baseline
        assert avg_time < 1.0, f"Performance regression: {avg_time:.3f}s (baseline ~0.38s)"

    def test_complex_16_panel_layout(self):
        """Benchmark: 16 panels with spans and complex angles."""
        xml = """<root>
    <panel thickness="0.25"/>
    <panel width="2.347" thickness="0.25" rotate="-90" attach="0:1"/>
    <panel width="1.522" thickness="0.25" rotate="90" attach="1:1"/>
    <panel width="1.637" thickness="0.25" rotate="90" attach="2:1"/>
    <panel width="4.296" thickness="0.25" rotate="90" attach="3:1"/>
    <panel width="4.854" thickness="0.25" rotate="-225" attach="4:1"/>
    <panel width="6.332" thickness="0.25" rotate="135" attach="5:1"/>
    <panel width="6.439" thickness="0.25" rotate="90" attach="6:1"/>
    <panel width="2.943" thickness="0.25" rotate="-270" attach="7:1"/>
    <panel width="3.721" thickness="0.25" rotate="45" attach="8:1"/>
    <panel width="5.148" thickness="0.25" rotate="45" attach="9:1"/>
    <panel width="7.019" thickness="0.25" rotate="90" attach="10:1"/>
    <panel width="9.576" thickness="0.25" rotate="45" attach="11:1"/>
    <panel width="15.138" thickness="0.25" rotate="-225" attach="12:1"/>
    <panel width="2.123" thickness="0.25" rotate="135" span="9:0.866" attach="13:1"/>
    <panel width="2.433" thickness="0.25" rotate="36.771" span="4:1" attach="14:1"/>
</root>"""

        # Warmup run
        self.parsePanel(xml)

        # Timed runs
        iterations = 10
        times = []
        
        for _ in range(iterations):
            self.setUp()
            
            start = time.perf_counter()
            result = self.parsePanel(xml)
            end = time.perf_counter()
            times.append(end - start)
            
            assert result is not None

        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)

        print(f"\n{'='*60}")
        print(f"Performance: 16 panels with spans and complex angles")
        print(f"{'='*60}")
        print(f"  Iterations: {iterations}")
        print(f"  Average:    {avg_time*1000:.2f} ms")
        print(f"  Std Dev:    {std_dev*1000:.2f} ms")
        print(f"  Min:        {min_time*1000:.2f} ms")
        print(f"  Max:        {max_time*1000:.2f} ms")
        print(f"{'='*60}")

        # Optimized baseline is ~1.4s
        # Fail if it regresses beyond 2x baseline
        assert avg_time < 3.0, f"Performance regression: {avg_time:.3f}s (baseline ~1.4s)"

    def test_large_75_panel_layout(self):
        """Benchmark: 75 panels - stress test with many rotations and attachments."""
        xml = """<root>
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

        # Warmup run
        self.parsePanel(xml)

        # Timed runs
        iterations = 5
        times = []
        
        for _ in range(iterations):
            self.setUp()
            
            start = time.perf_counter()
            result = self.parsePanel(xml)
            end = time.perf_counter()
            times.append(end - start)
            
            assert result is not None

        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)

        print(f"\n{'='*60}")
        print(f"Performance: 75 panels stress test")
        print(f"{'='*60}")
        print(f"  Iterations: {iterations}")
        print(f"  Average:    {avg_time*1000:.2f} ms")
        print(f"  Std Dev:    {std_dev*1000:.2f} ms")
        print(f"  Min:        {min_time*1000:.2f} ms")
        print(f"  Max:        {max_time*1000:.2f} ms")
        print(f"{'='*60}")

        # Optimized baseline is ~280ms (after SIMD optimizations)
        # Fail if it regresses beyond 2x baseline
        assert avg_time < 0.6, f"Performance regression: {avg_time:.3f}s (baseline ~0.28s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
