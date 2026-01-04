"""
Profiling utilities for GXML performance tests.

This module provides shared timing, benchmarking, and reporting utilities
for performance regression testing.
"""
import sys
import time
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

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
    # Top-level timings
    parse_ms: float = 0.0
    layout_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0
    # Layout pass breakdown
    measure_ms: float = 0.0
    pre_layout_ms: float = 0.0
    layout_pass_ms: float = 0.0
    post_layout_ms: float = 0.0
    # Solver breakdown
    intersection_ms: float = 0.0
    face_solver_ms: float = 0.0
    geometry_ms: float = 0.0
    # Stats
    panel_count: int = 0
    intersection_count: int = 0
    polygon_count: int = 0
    vertex_count: int = 0
    # Raw marker data
    raw_markers: Dict[str, Any] = field(default_factory=dict)
    
    def breakdown_str(self) -> str:
        """Return a formatted string showing the timing breakdown."""
        lines = [
            f"  Parse:              {self.parse_ms:>7.1f}ms",
            f"  Layout:             {self.layout_ms:>7.1f}ms",
            f"    ├─ Measure:       {self.measure_ms:>7.1f}ms",
            f"    ├─ Pre-layout:    {self.pre_layout_ms:>7.1f}ms",
            f"    ├─ Layout pass:   {self.layout_pass_ms:>7.1f}ms",
            f"    └─ Post-layout:   {self.post_layout_ms:>7.1f}ms",
            f"       ├─ Intersect:  {self.intersection_ms:>7.1f}ms",
            f"       ├─ Face solve: {self.face_solver_ms:>7.1f}ms",
            f"       └─ Geometry:   {self.geometry_ms:>7.1f}ms",
            f"  Render:             {self.render_ms:>7.1f}ms",
            f"  Total:              {self.total_ms:>7.1f}ms",
        ]
        return "\n".join(lines)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results from multiple runs."""
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

# Taichi is currently broken on Windows (Vulkan has massive overhead)
ENABLE_TAICHI = False


def check_backends() -> Dict[str, bool]:
    """Check which solver backends are available."""
    availability = {
        'cpu': True,  # Always available
        'c': False,
        'taichi': False,
    }
    
    # Check C extension
    try:
        from elements.solvers import is_c_extension_available
        availability['c'] = is_c_extension_available()
    except Exception:
        pass
    
    # Check Taichi (only if enabled)
    if ENABLE_TAICHI:
        try:
            import taichi as ti
            gpu_backends = [ti.cuda, ti.vulkan, ti.metal]
            for gpu_arch in gpu_backends:
                try:
                    ti.init(arch=gpu_arch, offline_cache=True, print_ir=False)
                    availability['taichi'] = True
                    break
                except Exception:
                    continue
            
            if not availability['taichi']:
                ti.init(arch=ti.cpu, offline_cache=True, print_ir=False)
                availability['taichi'] = True
                
        except ImportError:
            pass
        except Exception:
            pass
    
    return availability


def is_c_available() -> bool:
    """Quick check if C extension is available."""
    return check_backends()['c']


# =============================================================================
# Render Collector
# =============================================================================

class RenderCollector:
    """Simple render collector for timing the render pass."""
    
    def __init__(self):
        self.poly_count = 0
        self.vertex_count = 0
    
    def conditional_render(self, element):
        if element.isVisible:
            if element.isVisibleSelf:
                element.render(self)
            for child in element.children:
                self.conditional_render(child)
    
    def create_poly(self, id, points, geoKey=None):
        self.poly_count += 1
        self.vertex_count += len(points)
    
    def create_line(self, id, points, geoKey=None):
        pass


# =============================================================================
# Pipeline Runner
# =============================================================================

def collect_panels(element) -> list:
    """Recursively collect all panel elements from the tree."""
    panels = []
    if type(element).__name__ == 'GXMLPanel':
        panels.append(element)
    for child in element.children:
        panels.extend(collect_panels(child))
    return panels


def run_pipeline(xml_content: str, backend: str = 'cpu') -> TimingResult:
    """
    Run the full GXML pipeline end-to-end using gxml_engine.
    
    This uses the gxml_engine.run() function with profiling enabled
    to get detailed stage timings from actual instrumented code sections.
    
    Args:
        xml_content: The XML string to process
        backend: Solver backend ('cpu', 'c', or 'taichi')
    
    Returns:
        TimingResult with detailed timing breakdown including:
        - Top-level: parse, layout, render, total
        - Layout passes: measure, pre_layout, layout_pass, post_layout
        - Solvers: intersection, face_solver, geometry
    """
    from gxml_engine import run, GXMLConfig
    
    config = GXMLConfig(
        backend=backend,
        output_format='dict',
        profile=True,
    )
    
    t_start = time.perf_counter()
    gxml_result = run(xml_content, config=config)
    total_ms = (time.perf_counter() - t_start) * 1000
    
    # Helper to extract timing from markers
    def get_ms(name: str) -> float:
        if gxml_result.timings and name in gxml_result.timings:
            return gxml_result.timings[name].get('total_ms', 0.0)
        return 0.0
    
    # Build TimingResult from marker data
    result = TimingResult(backend=backend)
    
    # Top-level timings
    result.parse_ms = get_ms('parse')
    result.layout_ms = get_ms('layout')
    result.render_ms = get_ms('render')
    result.total_ms = total_ms
    
    # Layout pass breakdown
    result.measure_ms = get_ms('measure_pass')
    result.pre_layout_ms = get_ms('pre_layout_pass')
    result.layout_pass_ms = get_ms('layout_pass')
    result.post_layout_ms = get_ms('post_layout_pass')
    
    # Solver breakdown
    result.intersection_ms = get_ms('intersection_solver')
    result.face_solver_ms = get_ms('face_solver')
    result.geometry_ms = get_ms('geometry_builder')
    
    # Store raw markers for detailed analysis
    result.raw_markers = gxml_result.timings or {}
    
    # Stats
    if gxml_result.stats:
        result.panel_count = gxml_result.stats.get('panel_count', 0)
        result.intersection_count = gxml_result.stats.get('intersection_count', 0)
        result.polygon_count = gxml_result.stats.get('polygon_count', 0)
    
    # Count polygons from output if available
    if gxml_result.output and 'polygons' in gxml_result.output:
        result.polygon_count = len(gxml_result.output['polygons'])
        result.vertex_count = sum(
            len(poly.get('points', [])) 
            for poly in gxml_result.output['polygons']
        )
    
    return result


def run_warmup(xml_content: str, backend: str = 'cpu', runs: int = 2):
    """Run warmup iterations to stabilize timings."""
    for _ in range(runs):
        run_pipeline(xml_content, backend)


def run_benchmark(
    xml_content: str,
    backend: str = 'cpu',
    warmup: int = 1,
    iterations: int = 3
) -> BenchmarkResult:
    """
    Run a full benchmark with warmup and multiple iterations.
    
    Args:
        xml_content: XML to benchmark
        backend: Solver backend
        warmup: Number of warmup runs
        iterations: Number of timed runs
    
    Returns:
        BenchmarkResult with aggregated statistics
    """
    # Warmup
    run_warmup(xml_content, backend, warmup)
    
    # Timed runs
    results = []
    for _ in range(iterations):
        results.append(run_pipeline(xml_content, backend))
    
    times = [r.total_ms for r in results]
    
    return BenchmarkResult(
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
    result: BenchmarkResult,
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


# =============================================================================
# Printing Utilities
# =============================================================================

def print_benchmark_result(result: BenchmarkResult, verbose: bool = False):
    """Print a single benchmark result."""
    print(f"\n--- {result.backend.upper()} Backend ---")
    print(f"  Median: {result.median_ms:.1f}ms")
    print(f"  Mean:   {result.mean_ms:.1f}ms (±{result.std_ms:.1f}ms)")
    print(f"  Range:  {result.min_ms:.1f}ms - {result.max_ms:.1f}ms")
    
    if verbose and result.all_results:
        r = result.all_results[0]
        print(f"\n  Stage Breakdown (first run):")
        print(r.breakdown_str())
        print(f"\n  Stats:")
        print(f"    Panels: {r.panel_count}")
        print(f"    Intersections: {r.intersection_count}")
        print(f"    Polygons: {r.polygon_count}")


def print_timing_breakdown(results: List[TimingResult]):
    """Print detailed timing breakdown averaged across runs."""
    if not results:
        return
        
    n = len(results)
    avg = lambda attr: sum(getattr(r, attr) for r in results) / n
    
    print(f"\n  Average Breakdown ({n} runs):")
    print(f"    Parse:              {avg('parse_ms'):>7.1f}ms")
    print(f"    Layout:             {avg('layout_ms'):>7.1f}ms")
    print(f"      ├─ Measure:       {avg('measure_ms'):>7.1f}ms")
    print(f"      ├─ Pre-layout:    {avg('pre_layout_ms'):>7.1f}ms")
    print(f"      ├─ Layout pass:   {avg('layout_pass_ms'):>7.1f}ms")
    print(f"      └─ Post-layout:   {avg('post_layout_ms'):>7.1f}ms")
    print(f"         ├─ Intersect:  {avg('intersection_ms'):>7.1f}ms")
    print(f"         ├─ Face solve: {avg('face_solver_ms'):>7.1f}ms")
    print(f"         └─ Geometry:   {avg('geometry_ms'):>7.1f}ms")
    print(f"    Render:             {avg('render_ms'):>7.1f}ms")
    print(f"    Total:              {avg('total_ms'):>7.1f}ms")


def print_comparison_table(results: Dict[str, BenchmarkResult], verbose: bool = False):
    """Print a comparison table of multiple backend results."""
    if not results:
        return
    
    backends = list(results.keys())
    first = results[backends[0]]
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPARISON ({first.iterations} iterations)")
    print(f"{'='*60}")
    
    # Header
    header = f"{'Metric':<15}"
    for b in backends:
        header += f"{b.upper():>12}"
    if len(backends) >= 2:
        header += f"{'Speedup':>12}"
    print(header)
    print("-" * 60)
    
    # Median row
    row = f"{'Median':<15}"
    values = []
    for b in backends:
        val = results[b].median_ms
        values.append(val)
        row += f"{val:>10.1f}ms"
    if len(values) >= 2 and values[-1] > 0:
        speedup = values[0] / values[-1]
        row += f"{speedup:>11.2f}x"
    print(row)
    
    # Stats
    if first.panel_count:
        print(f"\n  Panels: {first.panel_count}")
        print(f"  Intersections: {first.intersection_count}")
    
    # Detailed breakdown if verbose
    if verbose:
        for backend, bench_result in results.items():
            if bench_result.all_results:
                print(f"\n--- {backend.upper()} Stage Breakdown ---")
                print_timing_breakdown(bench_result.all_results)
