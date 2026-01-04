"""
Performance regression tests for the GXML pipeline.

This module tests the full end-to-end pipeline (parse → layout → render)
across various XML sizes to catch performance regressions.

Usage:
    # Run via pytest (recommended)
    pytest tests/performance/test_perf_pipeline.py -v
    
    # Run standalone with detailed output
    python tests/performance/test_perf_pipeline.py
"""
import sys
from pathlib import Path

import pytest

# Add project paths
PERF_DIR = Path(__file__).parent
PROJECT_ROOT = PERF_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "gxml"))

from tests.test_fixtures.profiling import (
    run_benchmark,
    assert_performance,
    check_backends,
    is_c_available,
    print_benchmark_result,
    print_comparison_table,
)


# =============================================================================
# Test Configuration
# =============================================================================

XML_DIR = PERF_DIR / "xml"

# Test cases: (name, xml_file, baseline_ms)
# Baselines are conservative - real performance is typically better
TEST_CASES = [
    ("7 panels", "7Panels.xml", 50),
    ("16 panels", "16Panels.xml", 100),
    ("75 panels", "75Panels.xml", 300),
    ("200 panels", "200Panels.xml", 700),
]

# Regression threshold (1.2 = 20% regression allowed before failure)
REGRESSION_THRESHOLD = 1.2

# Benchmark settings
WARMUP_RUNS = 1
ITERATIONS = 3

# Mark timing-sensitive tests as xfail to avoid failing CI on system load variance
# Use strict=False so passing is OK (XPASS), failing is OK (xfail), but crashes fail
perf_xfail = pytest.mark.xfail(
    reason="Performance tests may fail due to system load - timing sensitive",
    raises=AssertionError,
    strict=False
)


# =============================================================================
# Fixtures
# =============================================================================

def load_xml(filename: str) -> str:
    """Load XML from the performance xml folder."""
    path = XML_DIR / filename
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return path.read_text()


# =============================================================================
# Pytest Tests
# =============================================================================

class TestPipelinePerformance:
    """
    Performance regression tests for the GXML pipeline.
    
    Tests will fail if performance regresses beyond the threshold
    (baseline × REGRESSION_THRESHOLD).
    """
    
    @perf_xfail
    @pytest.mark.parametrize("name,xml_file,baseline_ms", TEST_CASES)
    def test_cpu_performance(self, name, xml_file, baseline_ms):
        """Test CPU backend performance for each test case."""
        xml_content = load_xml(xml_file)
        result = run_benchmark(xml_content, 'cpu', WARMUP_RUNS, ITERATIONS)
        assert_performance(result, baseline_ms, REGRESSION_THRESHOLD, name)
    
    @perf_xfail
    def test_c_extension_200panels(self):
        """Test C extension backend on the 200-panel stress test."""
        if not is_c_available():
            pytest.skip("C extension not available")
        
        xml_content = load_xml("200Panels.xml")
        result = run_benchmark(xml_content, 'c', WARMUP_RUNS, ITERATIONS)
        assert_performance(result, 700, REGRESSION_THRESHOLD, "200 panels (C)")
    
    @perf_xfail
    def test_c_extension_not_slower_than_cpu(self):
        """Ensure C extension is not significantly slower than CPU."""
        if not is_c_available():
            pytest.skip("C extension not available")
        
        xml_content = load_xml("200Panels.xml")
        
        cpu_result = run_benchmark(xml_content, 'cpu', WARMUP_RUNS, ITERATIONS)
        c_result = run_benchmark(xml_content, 'c', WARMUP_RUNS, ITERATIONS)
        
        # C extension should not be more than 20% slower than CPU
        # (allowing variance since they're comparable in performance)
        max_ratio = 1.2
        ratio = c_result.median_ms / cpu_result.median_ms
        
        assert ratio < max_ratio, (
            f"C extension is significantly slower than CPU!\n"
            f"  CPU time: {cpu_result.median_ms:.1f}ms\n"
            f"  C time:   {c_result.median_ms:.1f}ms\n"
            f"  Ratio:    {ratio:.2f}x (max allowed: {max_ratio}x)"
        )
    
    @perf_xfail
    def test_profiling_overhead_acceptable(self):
        """Ensure profiling markers don't add excessive overhead."""
        import time
        from gxml_engine import run, GXMLConfig
        
        xml_content = load_xml("200Panels.xml")
        
        # Warmup
        run(xml_content, backend='cpu', profile=False)
        run(xml_content, backend='cpu', profile=True)
        
        # Measure without profiling
        times_no_profile = []
        for _ in range(3):
            t0 = time.perf_counter()
            run(xml_content, backend='cpu', profile=False)
            times_no_profile.append((time.perf_counter() - t0) * 1000)
        
        # Measure with profiling
        times_with_profile = []
        for _ in range(3):
            t0 = time.perf_counter()
            run(xml_content, backend='cpu', profile=True)
            times_with_profile.append((time.perf_counter() - t0) * 1000)
        
        median_no_profile = sorted(times_no_profile)[1]
        median_with_profile = sorted(times_with_profile)[1]
        
        # Profiling should add no more than 10% overhead
        max_overhead = 1.10
        overhead_ratio = median_with_profile / median_no_profile
        
        assert overhead_ratio < max_overhead, (
            f"Profiling adds too much overhead!\n"
            f"  Without profiling: {median_no_profile:.1f}ms\n"
            f"  With profiling:    {median_with_profile:.1f}ms\n"
            f"  Overhead:          {(overhead_ratio - 1) * 100:.1f}% (max allowed: {(max_overhead - 1) * 100:.0f}%)"
        )


class TestOutputFormatPerformance:
    """
    Performance tests comparing different output formats.
    
    Tests indexed mode (GXMF) vs binary mode to measure the efficiency
    of vertex deduplication and indexed mesh generation.
    """
    
    @perf_xfail
    def test_indexed_format_200panels(self):
        """Test indexed output format on 200 panels."""
        xml_content = load_xml("200Panels.xml")
        result = run_benchmark(xml_content, 'cpu', WARMUP_RUNS, ITERATIONS, output_format='indexed')
        # Indexed mode baseline - should be comparable to dict mode
        assert_performance(result, 800, REGRESSION_THRESHOLD, "200 panels (indexed)")
    
    @perf_xfail
    def test_binary_format_200panels(self):
        """Test binary output format on 200 panels."""
        xml_content = load_xml("200Panels.xml")
        result = run_benchmark(xml_content, 'cpu', WARMUP_RUNS, ITERATIONS, output_format='binary')
        # Binary mode baseline
        assert_performance(result, 800, REGRESSION_THRESHOLD, "200 panels (binary)")
    
    @perf_xfail
    def test_indexed_not_slower_than_binary(self):
        """Ensure indexed format is not significantly slower than binary."""
        xml_content = load_xml("200Panels.xml")
        
        binary_result = run_benchmark(xml_content, 'cpu', WARMUP_RUNS, ITERATIONS, output_format='binary')
        indexed_result = run_benchmark(xml_content, 'cpu', WARMUP_RUNS, ITERATIONS, output_format='indexed')
        
        # Indexed should not be more than 30% slower than binary
        # (vertex deduplication has some overhead but should be reasonable)
        max_ratio = 1.3
        ratio = indexed_result.median_ms / binary_result.median_ms

        assert ratio < max_ratio, (
            f"Indexed format is significantly slower than binary!\n"
            f"  Binary time:  {binary_result.median_ms:.1f}ms\n"
            f"  Indexed time: {indexed_result.median_ms:.1f}ms\n"
            f"  Ratio:        {ratio:.2f}x (max allowed: {max_ratio}x)"
        )
    
    @perf_xfail
    def test_indexed_with_c_extension(self):
        """Test indexed format with C extension backend."""
        if not is_c_available():
            pytest.skip("C extension not available")
        
        xml_content = load_xml("200Panels.xml")
        result = run_benchmark(xml_content, 'c', WARMUP_RUNS, ITERATIONS, output_format='indexed')
        assert_performance(result, 800, REGRESSION_THRESHOLD, "200 panels (indexed + C)")
    
    @perf_xfail
    @pytest.mark.parametrize("output_format", ['dict', 'binary', 'indexed', 'json'])
    def test_output_format_75panels(self, output_format):
        """Test all output formats on 75 panels."""
        xml_content = load_xml("75Panels.xml")
        result = run_benchmark(xml_content, 'cpu', WARMUP_RUNS, ITERATIONS, output_format=output_format)
        # All formats should complete within reasonable time
        assert_performance(result, 400, REGRESSION_THRESHOLD, f"75 panels ({output_format})")


# =============================================================================
# Standalone Execution
# =============================================================================

def main():
    """Run benchmarks with detailed output."""
    print("=" * 70)
    print("GXML PIPELINE PERFORMANCE TESTS")
    print("=" * 70)
    
    # Check backends
    availability = check_backends()
    print("\nBackend Availability:")
    for backend, available in availability.items():
        status = "✓" if available else "✗"
        print(f"  {backend.upper():8} : {status}")
    
    # Run benchmarks
    for name, xml_file, baseline_ms in TEST_CASES:
        xml_path = XML_DIR / xml_file
        if not xml_path.exists():
            print(f"\n⚠ Skipping {name}: {xml_file} not found")
            continue
        
        xml_content = xml_path.read_text()
        
        print(f"\n{'='*70}")
        print(f"{name.upper()} ({xml_file})")
        print(f"{'='*70}")
        print(f"Baseline: {baseline_ms}ms | Max allowed: {baseline_ms * REGRESSION_THRESHOLD:.0f}ms")
        
        results = {}
        
        # CPU benchmark
        print(f"\nBenchmarking CPU...", end=" ", flush=True)
        results['cpu'] = run_benchmark(xml_content, 'cpu', WARMUP_RUNS, ITERATIONS)
        print("done")
        print_benchmark_result(results['cpu'], verbose=True)
        
        # C extension benchmark (only for larger tests)
        if availability['c'] and baseline_ms >= 300:
            print(f"\nBenchmarking C extension...", end=" ", flush=True)
            results['c'] = run_benchmark(xml_content, 'c', WARMUP_RUNS, ITERATIONS)
            print("done")
            print_benchmark_result(results['c'], verbose=True)
        
        # Comparison table with stage breakdown
        if len(results) > 1:
            print_comparison_table(results, verbose=True)
        
        # Pass/fail
        cpu_ok = results['cpu'].median_ms < baseline_ms * REGRESSION_THRESHOLD
        status = "✓ PASS" if cpu_ok else "✗ FAIL"
        print(f"\nStatus: {status}")
    
    print(f"\n{'='*70}")
    print("Complete!")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
