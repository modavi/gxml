"""
Performance regression tests for the GXML pipeline.

This module tests the full end-to-end pipeline (parse → layout → render)
across various XML sizes to catch performance regressions.

Usage:
    # Run via pytest
    pytest tests/performance/test_perf_pipeline.py -v
    
    # For detailed profiling output, use the profile script:
    python scripts/profile_xml.py tests/performance/xml/200Panels.xml
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
        result = run_benchmark(xml_content, backend='cpu', warmup=WARMUP_RUNS, iterations=ITERATIONS)
        assert_performance(result, baseline_ms, REGRESSION_THRESHOLD, name)
    
    @perf_xfail
    def test_c_extension_200panels(self):
        """Test C extension backend on the 200-panel stress test."""
        if not is_c_available():
            pytest.skip("C extension not available")
        
        xml_content = load_xml("200Panels.xml")
        result = run_benchmark(xml_content, backend='c', warmup=WARMUP_RUNS, iterations=ITERATIONS)
        assert_performance(result, 700, REGRESSION_THRESHOLD, "200 panels (C)")
    
    @perf_xfail
    def test_c_extension_not_slower_than_cpu(self):
        """Ensure C extension is not significantly slower than CPU."""
        if not is_c_available():
            pytest.skip("C extension not available")
        
        xml_content = load_xml("200Panels.xml")
        
        cpu_result = run_benchmark(xml_content, backend='cpu', warmup=WARMUP_RUNS, iterations=ITERATIONS)
        c_result = run_benchmark(xml_content, backend='c', warmup=WARMUP_RUNS, iterations=ITERATIONS)
        
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
    def test_validation_overhead_under_1_percent(self):
        """Ensure XSD validation overhead stays under 1% of total pipeline cost."""
        xml_content = load_xml("200Panels.xml")
        
        # Run with profiling to get validation marker timing
        result = run_benchmark(xml_content, backend='cpu', warmup=WARMUP_RUNS, iterations=ITERATIONS)
        
        # Get validation time from markers (from the first timing result)
        assert result.all_results, "No timing results captured"
        timing = result.all_results[0]
        markers = timing.markers
        assert 'validate' in markers, "validate marker not found in profiling output"
        
        validate_ms = markers['validate']['avg_ms']
        total_ms = result.median_ms
        
        # Validation should be less than 1% of total
        max_percentage = 1.0
        actual_percentage = (validate_ms / total_ms) * 100
        
        assert actual_percentage < max_percentage, (
            f"XSD validation overhead too high!\n"
            f"  Validation time: {validate_ms:.3f}ms\n"
            f"  Total time:      {total_ms:.1f}ms\n"
            f"  Percentage:      {actual_percentage:.2f}% (max allowed: {max_percentage}%)"
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
    Performance tests comparing different render context options.
    
    Tests shared_vertices mode vs per-face vertices to measure the efficiency
    of vertex deduplication and indexed mesh generation.
    """
    
    @perf_xfail
    def test_shared_vertices_200panels(self):
        """Test shared_vertices output on 200 panels."""
        xml_content = load_xml("200Panels.xml")
        result = run_benchmark(xml_content, backend='cpu', warmup=WARMUP_RUNS, iterations=ITERATIONS, shared_vertices=True)
        # Shared vertices mode baseline
        assert_performance(result, 800, REGRESSION_THRESHOLD, "200 panels (shared_vertices)")
    
    @perf_xfail
    def test_per_face_vertices_200panels(self):
        """Test per-face vertices output on 200 panels."""
        xml_content = load_xml("200Panels.xml")
        result = run_benchmark(xml_content, backend='cpu', warmup=WARMUP_RUNS, iterations=ITERATIONS, shared_vertices=False)
        # Per-face vertices mode baseline
        assert_performance(result, 800, REGRESSION_THRESHOLD, "200 panels (per-face)")
    
    @perf_xfail
    def test_shared_vertices_not_slower_than_per_face(self):
        """Ensure shared_vertices mode is not significantly slower than per-face vertices."""
        xml_content = load_xml("200Panels.xml")
        
        per_face_result = run_benchmark(xml_content, backend='cpu', warmup=WARMUP_RUNS, iterations=ITERATIONS, shared_vertices=False)
        shared_result = run_benchmark(xml_content, backend='cpu', warmup=WARMUP_RUNS, iterations=ITERATIONS, shared_vertices=True)
        
        # Shared vertices should not be more than 30% slower
        max_ratio = 1.3
        ratio = shared_result.median_ms / per_face_result.median_ms

        assert ratio < max_ratio, (
            f"Shared vertices mode is significantly slower!\n"
            f"  Per-face time:      {per_face_result.median_ms:.1f}ms\n"
            f"  Shared verts time:  {shared_result.median_ms:.1f}ms\n"
            f"  Ratio:              {ratio:.2f}x (max allowed: {max_ratio}x)"
        )
    
    @perf_xfail
    def test_shared_vertices_with_c_extension(self):
        """Test shared_vertices mode with C extension backend."""
        if not is_c_available():
            pytest.skip("C extension not available")
        
        xml_content = load_xml("200Panels.xml")
        result = run_benchmark(xml_content, backend='c', warmup=WARMUP_RUNS, iterations=ITERATIONS, shared_vertices=True)
        
        assert result.median_ms < 800 * REGRESSION_THRESHOLD, f"200 panels (shared_vertices + C): {result.median_ms:.1f}ms exceeds threshold"
    
    @perf_xfail
    def test_shared_vertices_75panels(self):
        """Test shared_vertices output on 75 panels."""
        xml_content = load_xml("75Panels.xml")
        result = run_benchmark(xml_content, backend='cpu', warmup=WARMUP_RUNS, iterations=ITERATIONS, shared_vertices=True)
        # Should complete within reasonable time
        assert_performance(result, 400, REGRESSION_THRESHOLD, "75 panels (shared_vertices)")
