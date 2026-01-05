"""
Benchmarking utilities for GXML performance testing.

This module provides shared timing, benchmarking, and reporting utilities
for performance regression testing and profiling.

Usage:
    from gxml.profiling import run_benchmark, TimingResult, BenchmarkResult
    
    # Run a benchmark
    result = run_benchmark(xml_content, backend='cpu', iterations=3)
    print(f"Median: {result.median_ms}ms")
    
    # Get detailed breakdown
    timing = result.all_results[0]
    print(timing.hierarchical_breakdown())
    
    # Measure profiling overhead
    result = run_benchmark(xml_content, measure_overhead=True)
"""

import json
import os
import subprocess
import sys
import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


# =============================================================================
# Terminal Color Support
# =============================================================================

def supports_color() -> bool:
    """Check if terminal supports color output."""
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True


# ANSI color codes - set at import time
if supports_color():
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RED = "\033[91m"
    ORANGE = "\033[38;5;208m"
    GRAY = "\033[90m"
    CYAN = "\033[36m"
    WHITE = "\033[97m"
else:
    RESET = DIM = BOLD = YELLOW = GREEN = RED = ORANGE = GRAY = CYAN = WHITE = ""


def time_color(ms: float) -> str:
    """Get ANSI color based on time thresholds."""
    if ms >= 500:
        return RED
    elif ms >= 100:
        return ORANGE
    elif ms >= 10:
        return YELLOW
    else:
        return GREEN


def format_time(ms: float) -> str:
    """Format milliseconds for human-readable display."""
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    elif ms >= 1:
        return f"{ms:.2f}ms"
    else:
        return f"{ms*1000:.1f}µs"


def _average_markers(results: List['TimingResult']) -> Dict[str, Any]:
    """Average marker timings across multiple results."""
    if not results:
        return {}
    if len(results) == 1:
        return results[0].markers
    
    # Collect all marker names
    all_markers = set()
    for r in results:
        all_markers.update(r.markers.keys())
    
    averaged = {}
    for name in all_markers:
        totals = []
        counts = []
        parents_merged = {}
        
        for r in results:
            if name in r.markers:
                m = r.markers[name]
                totals.append(m.get('total_ms', 0.0))
                counts.append(m.get('count', 0))
                # Merge parent counts
                for parent, cnt in m.get('parents', {}).items():
                    parents_merged[parent] = parents_merged.get(parent, 0) + cnt
        
        if totals:
            avg_total = sum(totals) / len(totals)
            avg_count = sum(counts) / len(counts)
            averaged[name] = {
                'total_ms': round(avg_total, 3),
                'count': int(round(avg_count)),
                'avg_ms': round(avg_total / avg_count, 3) if avg_count > 0 else 0.0,
                'min_ms': round(min(totals), 3),
                'max_ms': round(max(totals), 3),
                'parents': {k: v // len(results) for k, v in parents_merged.items()},
            }
    
    return averaged


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
        
        # Use module-level colors or empty strings
        if use_color:
            _RESET, _DIM, _BOLD = RESET, DIM, BOLD
            _YELLOW, _GREEN, _RED, _ORANGE = YELLOW, GREEN, RED, ORANGE
            _GRAY, _CYAN = GRAY, CYAN
        else:
            _RESET = _DIM = _BOLD = _YELLOW = _GREEN = _RED = _ORANGE = _GRAY = _CYAN = ""
        
        def _time_color(ms: float) -> str:
            """Get color based on absolute time thresholds."""
            if ms > 500:
                return _RED
            elif ms > 20:
                return _ORANGE
            elif ms > 5:
                return _YELLOW
            else:
                return _GREEN
        
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
    overhead: Optional[Any] = None  # OverheadResult, set if measure_overhead=True
    
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
# Pipeline Runner
# =============================================================================

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
    
    gxml_result = run(xml_content, config=config)
    
    # Use the 'run' marker time as total (excludes get_profile_results overhead)
    timings = gxml_result.timings or {}
    total_ms = timings.get('run', {}).get('total_ms', 0.0)
    
    # Build TimingResult from marker data
    result = TimingResult(
        backend=backend,
        total_ms=total_ms,
        markers=timings,
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


def _run_subprocess_iteration(
    xml_content: str,
    backend: str = 'cpu',
    shared_vertices: bool = False,
    profile_enabled: bool = True,
    use_optimized: bool = False,
) -> Dict[str, Any]:
    """
    Run a SINGLE benchmark iteration in a subprocess.
    
    This ensures each iteration has fresh process state (no accumulated
    memory pressure, fresh JIT state, etc). No warmup is needed since
    we run each iteration in a fresh subprocess and interleave modes.
    
    Args:
        xml_content: XML to benchmark
        backend: Solver backend
        shared_vertices: Whether to use shared vertices mode
        profile_enabled: If True, run with profile=True in config
        use_optimized: If True, run with python -O (compiles out profiling markers)
    
    Returns:
        Dict with timing_ms, markers (if profile_enabled), panel_count, intersection_count, etc.
    """
    import tempfile
    project_root = _get_project_root()
    
    profile_str = "True" if profile_enabled else "False"
    shared_verts_str = "True" if shared_vertices else "False"
    
    # Write XML to a temp file to avoid command line length limits
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as xml_file:
        xml_file.write(xml_content)
        xml_path = xml_file.name
    
    # Script does: imports, then ONE timed run with full data collection
    # No warmup needed - we run each iteration in a fresh subprocess and interleave modes
    measure_script = f'''
import time
t_script_start = time.perf_counter()

import sys
sys.path.insert(0, r"{project_root}")
sys.path.insert(0, r"{project_root}/src")
sys.path.insert(0, r"{project_root}/src/gxml")

import json

with open(r"{xml_path}", "r", encoding="utf-8") as f:
    xml_content = f.read()

from gxml_engine import run as gxml_run, GXMLConfig
from render_engines.binary_render_context import BinaryRenderContext

t_imports_done = time.perf_counter()

# Single timed run with full data collection
ctx = BinaryRenderContext(shared_vertices={shared_verts_str})
t0 = time.perf_counter()
result = gxml_run(xml_content, config=GXMLConfig(backend="{backend}", mesh_render_context=ctx, profile={profile_str}))
wall_clock_ms = (time.perf_counter() - t0) * 1000

# Use the 'run' marker time if profiling, otherwise fall back to wall-clock
timings = result.timings if result.timings else {{}}
timing_ms = timings.get('run', {{}}).get('total_ms', 0.0) or wall_clock_ms

# Collect output
output = {{
    "timing_ms": timing_ms,
    "markers": timings,
    "panel_count": result.stats.get("panel_count", 0) if result.stats else 0,
    "intersection_count": result.stats.get("intersection_count", 0) if result.stats else 0,
    "polygon_count": result.stats.get("polygon_count", 0) if result.stats else 0,
    "import_time_ms": (t_imports_done - t_script_start) * 1000,
}}

print(json.dumps(output))
'''
    
    # Write script to temp file too (avoid command line limits)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as script_file:
        script_file.write(measure_script)
        script_path = script_file.name
    
    try:
        cmd = [sys.executable]
        if use_optimized:
            cmd.append("-O")
        cmd.append(script_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess benchmark failed: {result.stderr}")
        
        return json.loads(result.stdout.strip())
    finally:
        # Clean up temp files
        import os
        try:
            os.unlink(xml_path)
            os.unlink(script_path)
        except Exception:
            pass


def _run_timed_iterations(
    xml_content: str,
    backend: str = 'cpu',
    iterations: int = 3,
    shared_vertices: bool = False,
    profile_enabled: bool = True,
    use_optimized: bool = False,
    verbose_callback=None,
    discard_outliers: bool = False,
    outlier_threshold: float = 0.10,
    max_retries: int = 3,
) -> BenchmarkResult:
    """
    Run a full benchmark with multiple iterations, each in a separate subprocess.
    
    Each iteration runs in its own subprocess for isolation:
    - Fresh process state (no memory accumulation)
    - Fresh JIT compilation state  
    - No warmup needed (subprocess isolation handles variance)
    
    Args:
        xml_content: XML to benchmark
        backend: Solver backend
        iterations: Number of subprocess iterations
        shared_vertices: Whether to use shared vertices mode
        profile_enabled: If True, run with profile=True
        use_optimized: If True, run with python -O
        verbose_callback: Optional callback(iteration, total, retried) for progress
        discard_outliers: If True, detect and rerun outlier iterations
        outlier_threshold: Threshold for outlier detection (default 0.10 = ±10%)
        max_retries: Maximum retries per outlier iteration
    
    Returns:
        BenchmarkResult with aggregated statistics
    """
    results = []
    
    for i in range(iterations):
        if verbose_callback:
            verbose_callback(i + 1, iterations, False)
        
        data = _run_subprocess_iteration(
            xml_content, backend,
            shared_vertices=shared_vertices,
            profile_enabled=profile_enabled,
            use_optimized=use_optimized,
        )
        
        # Build TimingResult from subprocess data
        timing_result = TimingResult(
            backend=backend,
            total_ms=data["timing_ms"],
            markers=data.get("markers", {}),
        )
        timing_result.panel_count = data.get("panel_count", 0)
        timing_result.intersection_count = data.get("intersection_count", 0)
        timing_result.polygon_count = data.get("polygon_count", 0)
        
        results.append(timing_result)
    
    # Outlier detection and rerun (using median for robustness)
    if discard_outliers and len(results) >= 3:
        retries_made = 0
        while retries_made < max_retries * iterations:
            times = [r.total_ms for r in results]
            median = statistics.median(times)
            
            # Find outliers (outside ±threshold of median)
            outlier_indices = []
            for idx, t in enumerate(times):
                deviation = abs(t - median) / median
                if deviation > outlier_threshold:
                    outlier_indices.append((idx, deviation))
            
            if not outlier_indices:
                break  # No more outliers
            
            # Sort by deviation (worst first) and rerun the worst outlier
            outlier_indices.sort(key=lambda x: x[1], reverse=True)
            idx = outlier_indices[0][0]
            if verbose_callback:
                verbose_callback(idx + 1, iterations, True)
            
            data = _run_subprocess_iteration(
                xml_content, backend,
                shared_vertices=shared_vertices,
                profile_enabled=profile_enabled,
                use_optimized=use_optimized,
            )
            
            timing_result = TimingResult(
                backend=backend,
                total_ms=data["timing_ms"],
                markers=data.get("markers", {}),
            )
            timing_result.panel_count = data.get("panel_count", 0)
            timing_result.intersection_count = data.get("intersection_count", 0)
            timing_result.polygon_count = data.get("polygon_count", 0)
            
            results[idx] = timing_result
            retries_made += 1
    
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
# Overhead Measurement
# =============================================================================

@dataclass
class OverheadResult:
    """Results from profiling overhead measurement.
    
    Compares two modes:
    1. Unoptimized - normal Python with profiling enabled (profile=True)
    2. Optimized - python -O which compiles out profiling markers entirely
    
    The overhead represents the cost of having profiling markers in the code.
    """
    # Subprocess timings
    unoptimized_ms: float    # Normal Python with profile=True
    optimized_ms: float      # python -O - markers compiled out
    
    # Subprocess phase timings
    import_time_ms: float = 0.0    # Time to import gxml modules
    
    # Averaged markers from all unoptimized runs
    markers: Dict[str, Any] = field(default_factory=dict)
    
    # Stats from runs
    panel_count: int = 0
    intersection_count: int = 0
    polygon_count: int = 0
    
    @property
    def overhead_ms(self) -> float:
        """Profiling overhead in milliseconds."""
        return self.unoptimized_ms - self.optimized_ms
    
    @property
    def overhead_pct(self) -> float:
        """Profiling overhead as percentage of optimized time."""
        if self.optimized_ms > 0:
            return (self.overhead_ms / self.optimized_ms) * 100
        return 0.0
    
    # Aliases for backward compatibility
    @property
    def profile_enabled_ms(self) -> float:
        return self.unoptimized_ms
    
    @property
    def compiled_out_ms(self) -> float:
        return self.optimized_ms
    
    @property
    def total_overhead_ms(self) -> float:
        return self.overhead_ms
    
    @property
    def total_overhead_pct(self) -> float:
        return self.overhead_pct
    
    @property
    def is_minimal(self) -> bool:
        """Overhead < 5%"""
        return self.overhead_pct < 5
    
    @property
    def is_acceptable(self) -> bool:
        """Overhead < 15%"""
        return self.overhead_pct < 15


def _get_project_root() -> str:
    """Get the project root directory."""
    # This file is at src/gxml/profiling/runner.py
    return str(Path(__file__).parent.parent.parent.parent)


def _run_subprocess_benchmark(
    xml_content: str,
    backend: str,
    iterations: int,
    warmup: int,
    shared_vertices: bool,
    profile_enabled: bool = True,
    use_optimized: bool = False,
) -> Dict[str, float]:
    """
    Run benchmark in a subprocess.
    
    Args:
        xml_content: XML to benchmark
        backend: Solver backend
        iterations: Number of timed runs
        warmup: Number of warmup runs
        shared_vertices: Whether to use shared vertices mode
        profile_enabled: If True, run with profile=True in config
        use_optimized: If True, run with python -O (compiles out profiling markers)
    
    Returns:
        Dict with median_ms, mean_ms, min_ms, max_ms
    """
    import base64
    project_root = _get_project_root()
    
    xml_b64 = base64.b64encode(xml_content.encode('utf-8')).decode('ascii')
    profile_str = "True" if profile_enabled else "False"
    shared_verts_str = "True" if shared_vertices else "False"
    
    measure_script = f'''
import sys
import base64
sys.path.insert(0, r"{project_root}")
sys.path.insert(0, r"{project_root}/src")
sys.path.insert(0, r"{project_root}/src/gxml")

import json
import statistics
import time

xml_content = base64.b64decode("{xml_b64}").decode('utf-8')

from gxml_engine import run as gxml_run, GXMLConfig
from render_engines.binary_render_context import BinaryRenderContext

# Warmup
for _ in range({warmup}):
    ctx = BinaryRenderContext(shared_vertices={shared_verts_str})
    gxml_run(xml_content, config=GXMLConfig(backend="{backend}", mesh_render_context=ctx, profile={profile_str}))

# Timed runs
times = []
for _ in range({iterations}):
    ctx = BinaryRenderContext(shared_vertices={shared_verts_str})
    t0 = time.perf_counter()
    gxml_run(xml_content, config=GXMLConfig(backend="{backend}", mesh_render_context=ctx, profile={profile_str}))
    times.append((time.perf_counter() - t0) * 1000)

print(json.dumps({{
    "median_ms": statistics.median(times),
    "mean_ms": statistics.mean(times),
    "min_ms": min(times),
    "max_ms": max(times),
}}))
'''
    
    # Use -O flag only for optimized mode (compiles out profiling)
    cmd = [sys.executable]
    if use_optimized:
        cmd.append("-O")
    cmd.extend(["-c", measure_script])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess benchmark failed: {result.stderr}")
    
    return json.loads(result.stdout.strip())


def measure_profiling_overhead(
    xml_content: str,
    backend: str = 'cpu',
    iterations: int = 5,
    shared_vertices: bool = False,
    verbose_callback=None,
) -> OverheadResult:
    """
    Measure profiling overhead by comparing two modes:
    1. Unoptimized - normal Python with profiling enabled
    2. Optimized - python -O which compiles out profiling markers
    
    Each iteration runs in a fresh subprocess with no warmup needed since
    we interleave modes and each process starts fresh.
    
    Args:
        xml_content: XML to benchmark
        backend: Solver backend
        iterations: Number of iterations for each mode
        shared_vertices: Whether to use shared vertices mode
        verbose_callback: Optional callback(mode, iteration, total, timing_ms) for progress
    
    Returns:
        OverheadResult with timing comparison
    """
    # Collect times for both modes
    unoptimized_times = []
    optimized_times = []
    
    # Collect markers from all unoptimized runs for averaging
    all_markers = []
    
    # Also collect timing metadata and stats
    import_times = []
    last_panel_count = 0
    last_intersection_count = 0
    last_polygon_count = 0
    
    # Run iterations INTERLEAVED: U-O, U-O, U-O...
    # This ensures each mode experiences similar system conditions
    for i in range(iterations):
        # Unoptimized (normal Python with profiling)
        data = _run_subprocess_iteration(
            xml_content, backend,
            shared_vertices=shared_vertices,
            profile_enabled=True,
            use_optimized=False,
        )
        unoptimized_times.append(data["timing_ms"])
        import_times.append(data.get("import_time_ms", 0))
        all_markers.append(data.get("markers", {}))
        last_panel_count = data.get("panel_count", 0)
        last_intersection_count = data.get("intersection_count", 0)
        last_polygon_count = data.get("polygon_count", 0)
        if verbose_callback:
            verbose_callback("unoptimized", i + 1, iterations, data["timing_ms"])
        
        # Optimized (python -O, markers compiled out)
        data = _run_subprocess_iteration(
            xml_content, backend,
            shared_vertices=shared_vertices,
            profile_enabled=False,
            use_optimized=True,
        )
        optimized_times.append(data["timing_ms"])
        if verbose_callback:
            verbose_callback("optimized", i + 1, iterations, data["timing_ms"])
    
    # Compute medians
    unoptimized_ms = statistics.median(unoptimized_times)
    optimized_ms = statistics.median(optimized_times)
    
    # Average markers across all unoptimized runs
    averaged_markers = {}
    if all_markers:
        # Collect all marker names
        all_marker_names = set()
        for markers in all_markers:
            all_marker_names.update(markers.keys())
        
        # Average each marker, preserving hierarchy via parents
        for name in all_marker_names:
            totals = []
            counts = []
            parents_merged = {}
            
            for markers in all_markers:
                if name in markers:
                    m = markers[name]
                    totals.append(m.get("total_ms", 0))
                    counts.append(m.get("count", 0))
                    # Merge parents dict
                    for parent, cnt in m.get("parents", {}).items():
                        if parent not in parents_merged:
                            parents_merged[parent] = []
                        parents_merged[parent].append(cnt)
            
            if totals:
                # Average the parent counts too
                parents_avg = {}
                for parent, cnt_list in parents_merged.items():
                    parents_avg[parent] = round(statistics.mean(cnt_list))
                
                averaged_markers[name] = {
                    "total_ms": statistics.mean(totals),
                    "count": round(statistics.mean(counts)),
                    "parents": parents_avg,
                }
    
    return OverheadResult(
        unoptimized_ms=unoptimized_ms,
        optimized_ms=optimized_ms,
        import_time_ms=statistics.median(import_times) if import_times else 0.0,
        markers=averaged_markers,
        panel_count=last_panel_count,
        intersection_count=last_intersection_count,
        polygon_count=last_polygon_count,
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
# High-Level Profiling API
# =============================================================================

def run_benchmark(
    xml_content: str,
    backend: str = 'cpu',
    iterations: int = 1,
    warmup: bool = True,
    shared_vertices: bool = False,
    show_hierarchy: bool = True,
    label: str = "",
    verbose: bool = True,
    measure_overhead: bool = False,
    discard_outliers: bool = False,
    outlier_threshold: float = 0.10,
) -> BenchmarkResult:
    """
    Run a benchmark on XML content and print a detailed report.
    
    Each iteration runs in a separate subprocess for isolation (fresh process
    state). Modes are interleaved to reduce ordering bias.
    
    This is the main entry point for profiling - runs the benchmark and 
    prints formatted results.
    
    Args:
        xml_content: The XML string to profile
        backend: Solver backend ('cpu', 'c', or 'taichi')
        iterations: Number of timed iterations (each in separate subprocess)
        warmup: If True (default), run one untimed iteration first to warm OS caches
        shared_vertices: Whether to use shared vertices mode in render context
        show_hierarchy: If True, show hierarchical tree. If False, show flat list.
        label: Optional label for the report header (e.g. filename)
        verbose: If True, print progress messages and report
        measure_overhead: If True, also run optimized mode to measure profiling overhead
        discard_outliers: If True, detect and rerun iterations that are ±threshold from mean
        outlier_threshold: Threshold for outlier detection (default 0.10 = ±10%)
    
    Returns:
        BenchmarkResult with all timing data (overhead_result attached if measure_overhead=True)
    
    Raises:
        ValueError: If requested backend is not available
    """
    # Check backend availability
    availability = check_backends()
    if backend != 'cpu' and not availability.get(backend, False):
        raise ValueError(f"{backend.upper()} backend not available")
    
    if verbose:
        print(f"Running benchmark ({iterations} iteration{'s' if iterations > 1 else ''})...")
    
    # Warmup: run one untimed subprocess to warm OS caches (file system, memory pages, DLLs)
    if warmup:
        if verbose:
            print("  Warmup...", end="", flush=True)
        _run_subprocess_iteration(
            xml_content, backend,
            shared_vertices=shared_vertices,
            profile_enabled=True,
            use_optimized=False,
        )
        if verbose:
            print(" done")
    
    overhead_result = None
    
    if measure_overhead:
        # When measuring overhead, run BOTH modes interleaved: U-O, U-O, U-O
        # This ensures fair comparison by giving each mode similar system conditions
        if verbose:
            print("  Unoptimized=profile enabled  Optimized=python -O (markers compiled out)")
        
        iteration_times = {}  # {iteration: {mode: timing_ms}}
        
        # ANSI color codes
        GREEN = "\033[32m"
        RED = "\033[31m"
        RESET = "\033[0m"
        
        def interleaved_callback(mode, iteration, total, timing_ms):
            if verbose:
                # Collect times for this iteration
                if iteration not in iteration_times:
                    iteration_times[iteration] = {}
                iteration_times[iteration][mode] = timing_ms
                
                # When we have both times for an iteration, print summary with colors
                if len(iteration_times[iteration]) == 2:
                    times = iteration_times[iteration]
                    
                    # Color: green for faster, red for slower
                    if times['unoptimized'] <= times['optimized']:
                        u_col, o_col = GREEN, RED
                    else:
                        u_col, o_col = RED, GREEN
                    
                    print(f"  Iter {iteration}/{total}: Unoptimized={u_col}{times['unoptimized']:.0f}ms{RESET}  Optimized={o_col}{times['optimized']:.0f}ms{RESET}")
        
        overhead_result = measure_profiling_overhead(
            xml_content, backend, 
            iterations=iterations,
            shared_vertices=shared_vertices,
            verbose_callback=interleaved_callback if verbose else None,
        )
        
        # Build a BenchmarkResult from the overhead measurement
        # Markers are already averaged across all unoptimized runs
        timing_result = TimingResult(
            backend=backend,
            total_ms=overhead_result.profile_enabled_ms,
            markers=overhead_result.markers,
        )
        timing_result.panel_count = overhead_result.panel_count
        timing_result.intersection_count = overhead_result.intersection_count
        timing_result.polygon_count = overhead_result.polygon_count
        
        result = BenchmarkResult(
            backend=backend,
            iterations=iterations,
            median_ms=overhead_result.profile_enabled_ms,
            mean_ms=overhead_result.profile_enabled_ms,  # We only have median
            min_ms=overhead_result.profile_enabled_ms,
            max_ms=overhead_result.profile_enabled_ms,
            std_ms=0.0,
            all_results=[timing_result],
        )
        result.overhead = overhead_result
    else:
        # Normal benchmark - just run Enabled mode
        def verbose_callback(iteration, total, retried=False):
            if verbose:
                status = "  (retry)" if retried else ""
                print(f"\r  Enabled (profiling):     iteration {iteration}/{total}{status}    ", end="", flush=True)
        
        result = _run_timed_iterations(
            xml_content, backend, 
            iterations=iterations, 
            shared_vertices=shared_vertices,
            profile_enabled=True,
            use_optimized=False,
            verbose_callback=verbose_callback if verbose else None,
            discard_outliers=discard_outliers,
            outlier_threshold=outlier_threshold,
        )
        
        if verbose:
            print()
    
    if verbose:
        print_profile_report(result, label=label, show_hierarchy=show_hierarchy, overhead=overhead_result)
    
    return result


# =============================================================================
# Printing Utilities
# =============================================================================

def print_profile_report(
    result: BenchmarkResult,
    label: str = "",
    show_hierarchy: bool = True,
    overhead: Optional[OverheadResult] = None,
) -> None:
    """
    Print a complete profile report for a benchmark result.
    
    This is the main output function for profiling - prints a nicely formatted
    report with stats and timing breakdown.
    
    Args:
        result: BenchmarkResult from run_benchmark()
        label: Optional label (e.g. filename) for the header
        show_hierarchy: If True, show hierarchical tree. If False, show flat list.
        overhead: Optional overhead measurement results
    """
    # Average markers across all results for display
    averaged_result = None
    if result.all_results:
        averaged_markers = _average_markers(result.all_results)
        # Use averaged 'run' marker time as total (consistent with averaged markers)
        avg_total = averaged_markers.get('run', {}).get('total_ms', result.mean_ms)
        averaged_result = TimingResult(
            backend=result.backend,
            total_ms=avg_total,
            markers=averaged_markers,
        )
    
    print()
    print("=" * 70)
    print(f"PROFILE RESULTS{': ' + label if label else ''}")
    print("=" * 70)
    print()
    print(f"  Backend:       {result.backend.upper()}")
    print(f"  Panels:        {result.panel_count}")
    print(f"  Intersections: {result.intersection_count}")
    print(f"  Iterations:    {result.iterations}")
    print()
    
    if result.iterations > 1:
        variance_pct = (result.std_ms / result.mean_ms * 100) if result.mean_ms > 0 else 0
        print(f"  Mean: {format_time(result.mean_ms)} ±{variance_pct:.1f}%")
        print(f"  Min: {format_time(result.min_ms)}, Max: {format_time(result.max_ms)}")
        print()
    
    print("─" * 70)
    print("TIMING BREAKDOWN")
    print("─" * 70)
    print()
    
    if averaged_result:
        if show_hierarchy:
            print(averaged_result.hierarchical_breakdown())
        else:
            print(averaged_result.breakdown_str(detailed=True))
    else:
        print("  (no results)")
    
    # Print overhead section if available
    if overhead:
        print()
        print("─" * 70)
        print("PROFILING OVERHEAD")
        print("─" * 70)
        print()
        
        # Show both modes
        print(f"  {'Mode':<24} {'Median':>10}  {'Overhead':>18}")
        print(f"  {'-'*24} {'-'*10}  {'-'*18}")
        print(f"  {'Optimized (python -O)':<24} {overhead.optimized_ms:>8.2f}ms  {'(baseline)':>18}")
        print(f"  {'Unoptimized (profiling)':<24} {overhead.unoptimized_ms:>8.2f}ms  {overhead.overhead_ms:>+8.2f}ms ({overhead.overhead_pct:>+5.1f}%)")
        print()
        
        # Summary
        if overhead.is_minimal:
            print(f"  {GREEN}✓ Profiling overhead is minimal (<5%){RESET}")
        elif overhead.is_acceptable:
            print(f"  {YELLOW}⚠ Profiling overhead is acceptable (<15%){RESET}")
        else:
            print(f"  {RED}✗ Profiling overhead is significant (>15%){RESET}")
    
    print()


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
