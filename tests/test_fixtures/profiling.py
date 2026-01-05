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
    total_ms: float = 0.0
    # Raw marker data - all profiling markers captured during the run
    markers: Dict[str, Any] = field(default_factory=dict)
    # Stats
    panel_count: int = 0
    intersection_count: int = 0
    polygon_count: int = 0
    vertex_count: int = 0
    
    def get_ms(self, name: str) -> float:
        """Get total_ms for a marker by name."""
        if name in self.markers:
            return self.markers[name].get('total_ms', 0.0)
        return 0.0
    
    def breakdown_str(self, detailed: bool = False) -> str:
        """
        Return a formatted string showing the timing breakdown.
        
        Args:
            detailed: If True, show all markers. If False, show simplified summary.
        """
        if detailed:
            return self._detailed_breakdown()
        return self._simplified_breakdown()
    
    def _simplified_breakdown(self) -> str:
        """Show top-level timing summary."""
        lines = []
        # Show top-level markers sorted by time
        top_level = ['parse', 'layout', 'render']
        for name in top_level:
            ms = self.get_ms(name)
            if ms > 0:
                lines.append(f"  {name:<20} {ms:>7.2f}ms")
        lines.append(f"  {'total':<20} {self.total_ms:>7.2f}ms")
        return "\n".join(lines)
    
    def _detailed_breakdown(self) -> str:
        """Show all captured markers."""
        if not self.markers:
            return "  (no markers captured)"
        
        lines = []
        # Sort markers by total_ms descending
        sorted_markers = sorted(
            self.markers.items(),
            key=lambda x: x[1].get('total_ms', 0),
            reverse=True
        )
        
        # Find the longest marker name for alignment
        max_name_len = max(len(name) for name, _ in sorted_markers) if sorted_markers else 10
        max_name_len = max(max_name_len, 10)  # Minimum width
        
        for name, stats in sorted_markers:
            total = stats.get('total_ms', 0)
            count = stats.get('count', 1)
            avg = stats.get('avg_ms', total)
            
            if count > 1:
                lines.append(f"  {name:<{max_name_len}} {total:>8.2f}ms  ({count}× avg {avg:.2f}ms)")
            else:
                lines.append(f"  {name:<{max_name_len}} {total:>8.2f}ms")
        
        lines.append(f"  {'─' * (max_name_len + 20)}")
        lines.append(f"  {'total':<{max_name_len}} {self.total_ms:>8.2f}ms")
        return "\n".join(lines)
    
    def hierarchical_breakdown(self, use_color: bool = True) -> str:
        """Show markers in a hierarchical tree based on parent relationships."""
        if not self.markers:
            return "  (no markers captured)"
        
        # ANSI color codes
        if use_color:
            RESET = "\033[0m"
            DIM = "\033[2m"
            BOLD = "\033[1m"
            YELLOW = "\033[33m"
            GREEN = "\033[32m"
            RED = "\033[91m"
            ORANGE = "\033[38;5;208m"
            GRAY = "\033[90m"
            CYAN = "\033[36m"
        else:
            RESET = DIM = BOLD = YELLOW = GREEN = RED = ORANGE = GRAY = CYAN = ""
        
        def time_color(ms: float) -> str:
            """Get color based on absolute time thresholds."""
            if ms > 500:
                return RED
            elif ms > 20:
                return ORANGE
            elif ms > 5:
                return YELLOW
            else:
                return GREEN
        
        # Build parent->children mapping from the parents data
        children_of = {}  # parent_name -> list of child names
        root_markers = []  # markers with no parent
        
        for name, stats in self.markers.items():
            parents = stats.get('parents', {})
            if not parents:
                root_markers.append(name)
            else:
                # Use the most common parent
                parent = max(parents.keys(), key=lambda p: parents[p])
                if parent not in children_of:
                    children_of[parent] = []
                children_of[parent].append(name)
        
        lines = []
        NAME_COL_WIDTH = 40  # Fixed width for name+tree column
        
        def format_marker(name: str, depth: int = 0, prefix_chars: str = "") -> None:
            """Recursively format a marker and its children."""
            stats = self.markers.get(name, {})
            total = stats.get('total_ms', 0)
            count = stats.get('count', 1)
            tc = time_color(total)
            
            # Calculate the visible length of prefix + name
            visible_name = f"{prefix_chars}{name}"
            visible_len = len(visible_name)
            
            # Apply colors to the line
            colored_prefix = f"{DIM}{prefix_chars}{RESET}" if prefix_chars else ""
            colored_name = f"{CYAN}{name}{RESET}"
            
            # Pad to fixed width for alignment
            padding = " " * max(1, NAME_COL_WIDTH - visible_len)
            
            # Always show count
            count_str = f"  {GRAY}({count}x){RESET}"
            
            line = f"  {colored_prefix}{colored_name}{padding}{tc}{BOLD}{total:>8.2f}ms{RESET}{count_str}"
            lines.append(line)
            
            # Process children sorted by total_ms descending
            if name in children_of:
                child_list = sorted(
                    children_of[name],
                    key=lambda c: self.markers.get(c, {}).get('total_ms', 0),
                    reverse=True
                )
                for i, child in enumerate(child_list):
                    is_last = (i == len(child_list) - 1)
                    # Build prefix for children
                    if depth == 0:
                        child_prefix = "└──" if is_last else "├──"
                    else:
                        # Replace the last connector with continuation or space
                        base = prefix_chars[:-3]  # Remove last connector
                        if prefix_chars.endswith("└──"):
                            base += "   "
                        else:
                            base += "│  "
                        child_prefix = base + ("└──" if is_last else "├──")
                    
                    format_marker(child, depth + 1, child_prefix)
        
        # Sort root markers by total_ms descending
        root_markers.sort(key=lambda n: self.markers.get(n, {}).get('total_ms', 0), reverse=True)
        
        # Add header
        header = f"  {DIM}{'Marker':<{NAME_COL_WIDTH}}{'Time':>10}  {'Calls':>8}{RESET}"
        lines.append(header)
        sep_line = f"  {DIM}{'─' * NAME_COL_WIDTH}{'─' * 10}  {'─' * 8}{RESET}"
        lines.append(sep_line)
        
        for root in root_markers:
            format_marker(root)
        
        lines.append(sep_line)
        lines.append(f"  {BOLD}{'total':<{NAME_COL_WIDTH}}{self.total_ms:>8.2f}ms{RESET}")
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


def run_pipeline(xml_content: str, backend: str = 'cpu', shared_vertices: bool = False) -> TimingResult:
    """
    Run the full GXML pipeline end-to-end using gxml_engine.
    
    This uses the gxml_engine.run() function with profiling enabled
    to get detailed stage timings from actual instrumented code sections.
    
    Args:
        xml_content: The XML string to process
        backend: Solver backend ('cpu', 'c', or 'taichi')
        shared_vertices: Whether to use shared vertices mode in render context
    
    Returns:
        TimingResult with detailed timing breakdown including:
        - Top-level: parse, layout, render, total
        - Layout passes: measure, pre_layout, layout_pass, post_layout
        - Solvers: intersection, face_solver, geometry
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
    
    # Build TimingResult from marker data
    result = TimingResult(
        backend=backend,
        total_ms=total_ms,
        markers=gxml_result.timings or {},
    )
    
    # Stats
    if gxml_result.stats:
        result.panel_count = gxml_result.stats.get('panel_count', 0)
        result.intersection_count = gxml_result.stats.get('intersection_count', 0)
        result.polygon_count = gxml_result.stats.get('polygon_count', 0)
    
    # Count polygons from output if available (only for dict format)
    if gxml_result.output and isinstance(gxml_result.output, dict) and 'polygons' in gxml_result.output:
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
    iterations: int = 3,
    shared_vertices: bool = False,
) -> BenchmarkResult:
    """
    Run a full benchmark with warmup and multiple iterations.
    
    Args:
        xml_content: XML to benchmark
        backend: Solver backend
        warmup: Number of warmup runs
        iterations: Number of timed runs
        shared_vertices: Whether to use shared vertices mode
    
    Returns:
        BenchmarkResult with aggregated statistics
    """
    # Warmup
    for _ in range(warmup):
        run_pipeline(xml_content, backend, shared_vertices)
    
    # Timed runs
    results = []
    for _ in range(iterations):
        results.append(run_pipeline(xml_content, backend, shared_vertices))
    
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

def print_benchmark_result(result: BenchmarkResult, detailed: bool = False, hierarchical: bool = False):
    """
    Print a single benchmark result.
    
    Args:
        result: Benchmark result to print
        detailed: If True, show all markers flat. If False, show summary only.
        hierarchical: If True, show markers in tree view (overrides detailed).
    """
    print(f"\n--- {result.backend.upper()} Backend ---")
    print(f"  Median: {result.median_ms:.1f}ms")
    print(f"  Mean:   {result.mean_ms:.1f}ms (±{result.std_ms:.1f}ms)")
    print(f"  Range:  {result.min_ms:.1f}ms - {result.max_ms:.1f}ms")
    
    if result.all_results:
        r = result.all_results[0]
        if hierarchical:
            print(f"\n  Timing Hierarchy (first run):")
            print(r.hierarchical_breakdown())
        else:
            print(f"\n  Timing Breakdown (first run):")
            print(r.breakdown_str(detailed=detailed))
        print(f"\n  Stats:")
        print(f"    Panels: {r.panel_count}")
        print(f"    Intersections: {r.intersection_count}")
        print(f"    Polygons: {r.polygon_count}")


def print_timing_breakdown(results: List[TimingResult], detailed: bool = False):
    """
    Print timing breakdown averaged across runs.
    
    Args:
        results: List of TimingResult from multiple runs
        detailed: If True, show all markers. If False, show summary only.
    """
    if not results:
        return
    
    n = len(results)
    
    # Collect all marker names across all runs
    all_markers = set()
    for r in results:
        all_markers.update(r.markers.keys())
    
    if not all_markers:
        print(f"\n  (no markers captured)")
        return
    
    # Calculate averages for each marker
    marker_avgs = {}
    for name in all_markers:
        totals = [r.get_ms(name) for r in results]
        marker_avgs[name] = sum(totals) / n
    
    # Sort by average time descending
    sorted_markers = sorted(marker_avgs.items(), key=lambda x: x[1], reverse=True)
    
    # Filter for display
    if not detailed:
        # Show only top-level markers in simplified view
        top_level = ['parse', 'layout', 'render']
        sorted_markers = [(n, v) for n, v in sorted_markers if n in top_level]
    
    print(f"\n  Average Breakdown ({n} runs):")
    
    # Find max name length for alignment
    max_name_len = max(len(name) for name, _ in sorted_markers) if sorted_markers else 10
    max_name_len = max(max_name_len, 10)
    
    for name, avg_ms in sorted_markers:
        if avg_ms > 0.001:  # Skip near-zero markers
            print(f"    {name:<{max_name_len}} {avg_ms:>8.2f}ms")
    
    # Total
    avg_total = sum(r.total_ms for r in results) / n
    print(f"    {'─' * (max_name_len + 12)}")
    print(f"    {'total':<{max_name_len}} {avg_total:>8.2f}ms")


def print_comparison_table(results: Dict[str, BenchmarkResult], detailed: bool = False):
    """
    Print a side-by-side comparison table of backend results.
    
    Args:
        results: Dict mapping backend name to BenchmarkResult
        detailed: If True, show all markers with full stats. If False, show summary only.
    """
    if not results:
        return
    
    backends = list(results.keys())
    first = results[backends[0]]
    
    # ANSI colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[91m"
    YELLOW = "\033[33m"
    ORANGE = "\033[38;5;208m"
    
    print(f"\n{DIM}{'='*60}{RESET}")
    print(f"{BOLD}BACKEND COMPARISON{RESET} ({first.panel_count} panels, {first.intersection_count} intersections)")
    print(f"{DIM}{'='*60}{RESET}")
    
    # Key markers to compare (in display order)
    key_markers = [
        ('total', 'Total'),
        ('validate', '  Validate'),
        ('parse', '  Parse'),
        ('layout', '  Layout'),
        ('render', '  Render'),
        ('intersection_solver', 'Intersection Solver'),
        ('face_solver', 'Face Solver'),
        ('geometry_builder', 'Geometry Builder'),
    ]
    
    # Get average times for each marker per backend
    def get_avg_ms(bench_result: BenchmarkResult, marker: str) -> float:
        if marker == 'total':
            return bench_result.mean_ms
        if not bench_result.all_results:
            return 0.0
        totals = [r.get_ms(marker) for r in bench_result.all_results]
        return sum(totals) / len(totals) if totals else 0.0
    
    # Header row
    header = f"  {'Stage':<20}"
    for b in backends:
        header += f"{BOLD}{b.upper():>12}{RESET}"
    if len(backends) >= 2:
        header += f"     {'Delta':>8}"
    print(header)
    print(f"  {'-'*56}")
    
    # Data rows
    for marker, label in key_markers:
        values = [get_avg_ms(results[b], marker) for b in backends]
        
        # Skip if all zeros
        if all(v == 0 for v in values):
            continue
        
        row = f"  {label:<20}"
        for val in values:
            # Color based on time
            if val > 100:
                color = RED
            elif val > 10:
                color = YELLOW
            elif val > 0:
                color = GREEN
            else:
                color = DIM
            row += f"{color}{val:>10.1f}ms{RESET}"
        
        # Delta column
        if len(values) >= 2 and values[0] > 0:
            diff_pct = ((values[-1] - values[0]) / values[0]) * 100
            if abs(diff_pct) < 5:
                delta_color = DIM
            elif diff_pct < 0:
                delta_color = GREEN  # C is faster
            else:
                delta_color = RED    # C is slower
            row += f"     {delta_color}{diff_pct:>+7.1f}%{RESET}"
        
        print(row)
    
    # Detailed stats section - show all markers with full statistics
    if detailed:
        print(f"\n{DIM}{'─'*60}{RESET}")
        print(f"{BOLD}DETAILED STATS{RESET}")
        print(f"{DIM}{'─'*60}{RESET}")
        
        for backend in backends:
            bench = results[backend]
            print(f"\n{BOLD}{backend.upper()}{RESET} ({bench.iterations} iterations, median={bench.median_ms:.1f}ms, std={bench.std_ms:.1f}ms)")
            
            # Get representative result (first one has the marker data)
            if bench.all_results:
                print(bench.all_results[0].hierarchical_breakdown(use_color=True))
    
    print()
