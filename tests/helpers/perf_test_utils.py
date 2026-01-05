"""
Profiling utilities for GXML performance tests.

This module provides simple in-process benchmarking for pytest performance tests.
For CLI benchmarking with subprocess isolation, use scripts/benchmark.py instead.
"""
import sys
import time
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any

# Setup paths for imports
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SRC_PATH = _PROJECT_ROOT / "src"
_GXML_PATH = _PROJECT_ROOT / "src" / "gxml"

if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
if str(_GXML_PATH) not in sys.path:
    sys.path.insert(0, str(_GXML_PATH))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TimingResult:
    """Stores timing results for a single pipeline run."""
    backend: str
    total_ms: float = 0.0
    markers: Dict[str, Any] = field(default_factory=dict)
    panel_count: int = 0
    intersection_count: int = 0
    polygon_count: int = 0
    vertex_count: int = 0
    
    def get_ms(self, name: str) -> float:
        """Get total_ms for a marker by name."""
        if name in self.markers:
            return self.markers[name].get('total_ms', 0.0)
        return 0.0


@dataclass
class PerfResult:
    """Aggregated results from multiple timed runs."""
    backend: str
    iterations: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    all_results: List[TimingResult] = field(default_factory=list)
    
    @property
    def panel_count(self) -> int:
        return self.all_results[0].panel_count if self.all_results else 0
    
    @property
    def intersection_count(self) -> int:
        return self.all_results[0].intersection_count if self.all_results else 0


# =============================================================================
# Backend Utilities
# =============================================================================

def check_backends() -> Dict[str, bool]:
    """Check which solver backends are available."""
    availability = {
        'cpu': True,
        'c': False,
        'taichi': False,
    }
    
    try:
        from elements.solvers import is_c_extension_available
        availability['c'] = is_c_extension_available()
    except Exception:
        pass
    
    return availability


def is_c_available() -> bool:
    """Quick check if C extension is available."""
    return check_backends()['c']


# =============================================================================
# Pipeline Runner
# =============================================================================

def _run_pipeline(xml_content: str, backend: str = 'cpu', shared_vertices: bool = False) -> TimingResult:
    """
    Run the full GXML pipeline end-to-end.
    
    Args:
        xml_content: The XML string to process
        backend: Solver backend ('cpu' or 'c')
        shared_vertices: Whether to use shared vertices mode
    
    Returns:
        TimingResult with timing data
    """
    from gxml_engine import run, GXMLConfig
    from render_engines.binary_render_context import BinaryRenderContext
    
    render_ctx = BinaryRenderContext(shared_vertices=shared_vertices)
    config = GXMLConfig(
        backend=backend,
        mesh_render_context=render_ctx,
        profile=True,
    )
    
    t_start = time.perf_counter()
    gxml_result = run(xml_content, config=config)
    total_ms = (time.perf_counter() - t_start) * 1000
    
    result = TimingResult(
        backend=backend,
        total_ms=total_ms,
        markers=gxml_result.timings or {},
    )
    
    if gxml_result.stats:
        result.panel_count = gxml_result.stats.get('panel_count', 0)
        result.intersection_count = gxml_result.stats.get('intersection_count', 0)
        result.polygon_count = gxml_result.stats.get('polygon_count', 0)
    
    return result


def run_perf_test(
    xml_content: str,
    backend: str = 'cpu',
    warmup: int = 1,
    iterations: int = 3,
    shared_vertices: bool = False,
) -> PerfResult:
    """
    Run a performance test with warmup and multiple iterations.
    
    Args:
        xml_content: XML to test
        backend: Solver backend
        warmup: Number of warmup runs
        iterations: Number of timed runs
        shared_vertices: Whether to use shared vertices mode
    
    Returns:
        PerfResult with aggregated statistics
    """
    # Warmup
    for _ in range(warmup):
        _run_pipeline(xml_content, backend, shared_vertices)
    
    # Timed runs
    results = []
    for _ in range(iterations):
        results.append(_run_pipeline(xml_content, backend, shared_vertices))
    
    times = [r.total_ms for r in results]
    
    return PerfResult(
        backend=backend,
        iterations=iterations,
        median_ms=statistics.median(times),
        mean_ms=statistics.mean(times),
        min_ms=min(times),
        max_ms=max(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0,
        all_results=results,
    )


# =============================================================================
# Assertion Helpers
# =============================================================================

def assert_performance(
    result: PerfResult,
    baseline_ms: float,
    threshold: float = 1.2,
    label: str = ""
) -> None:
    """
    Assert that benchmark performance is within acceptable limits.
    
    Args:
        result: Benchmark result to check
        baseline_ms: Expected baseline in milliseconds
        threshold: Multiplier for regression threshold (default 1.2 = 20%)
        label: Optional label for error messages
    
    Raises:
        AssertionError if performance exceeds baseline * threshold
    """
    max_allowed_ms = baseline_ms * threshold
    
    assert result.median_ms < max_allowed_ms, (
        f"Performance regression detected{' for ' + label if label else ''}!\n"
        f"  Backend:     {result.backend}\n"
        f"  Median time: {result.median_ms:.1f}ms\n"
        f"  Baseline:    {baseline_ms}ms\n"
        f"  Max allowed: {max_allowed_ms:.1f}ms (baseline × {threshold})\n"
        f"  Regression:  {result.median_ms / baseline_ms:.2f}x baseline"
    )
