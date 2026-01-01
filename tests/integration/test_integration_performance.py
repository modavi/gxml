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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
