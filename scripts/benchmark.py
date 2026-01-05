#!/usr/bin/env python
"""
Profile a GXML document and display detailed timing breakdown.

Usage:
    python scripts/profile_xml.py <path_to_xml>
    python scripts/profile_xml.py <path_to_xml> --iterations 5
    python scripts/profile_xml.py <path_to_xml> --backend c
    python scripts/profile_xml.py <path_to_xml> --overhead
    python scripts/profile_xml.py <path_to_xml> --sampling

Examples:
    python scripts/profile_xml.py tests/performance/xml/7Panels.xml
    python scripts/profile_xml.py my_layout.gxml --iterations 10 --backend cpu
    python scripts/profile_xml.py tests/performance/xml/200Panels.xml --overhead
    python scripts/profile_xml.py tests/performance/xml/200Panels.xml --sampling
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Add project paths for development
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "gxml"))

from gxml.profiling import run_benchmark


# ANSI colors for output
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
GRAY = "\033[90m"


def find_class_for_line(file_path: str, line_number: int) -> str:
    """Find the class name that contains the given line number."""
    import re
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Walk backwards from line_number to find the containing class
        class_pattern = re.compile(r'^class\s+(\w+)')
        for i in range(min(line_number - 1, len(lines) - 1), -1, -1):
            match = class_pattern.match(lines[i])
            if match:
                return match.group(1)
    except (IOError, OSError):
        pass
    return None


def parse_speedscope_json(json_path: Path, sampling_rate: int = 1000) -> Tuple[List[Tuple[str, str, float]], int, float]:
    """
    Parse py-spy speedscope JSON output and compute self-time percentages.
    
    Returns:
        results: list of (function_name, file_path, percentage) tuples sorted by percentage
        total_samples: total number of samples collected
        total_time_ms: estimated total time in milliseconds (samples / rate * 1000)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['shared']['frames']
    profile = data['profiles'][0]
    samples = profile['samples']
    weights = profile.get('weights', [1] * len(samples))
    
    # Count samples where each frame is at the "leaf" (executing)
    # Key by (name, file, line) to distinguish methods from different classes
    leaf_counts = defaultdict(float)
    total_weight = 0.0
    
    for sample, weight in zip(samples, weights):
        total_weight += weight
        if sample:
            leaf_frame = frames[sample[-1]]
            name = leaf_frame['name']
            file = leaf_frame.get('file', '')
            line = leaf_frame.get('line', 0)
            key = (name, file, line)
            leaf_counts[key] += weight
    
    # Convert to list of (name, file, percentage)
    # Look up class names from source files
    results = []
    class_cache = {}  # Cache (file, line) -> class_name lookups
    
    for (name, file, line), count in leaf_counts.items():
        pct = (count / total_weight * 100) if total_weight > 0 else 0
        
        # Try to find the class name for this function
        if line and file:
            cache_key = (file, line)
            if cache_key not in class_cache:
                class_cache[cache_key] = find_class_for_line(file, line)
            class_name = class_cache[cache_key]
            display_name = f"{class_name}.{name}" if class_name else name
        else:
            display_name = name
            
        results.append((display_name, file, pct))
    
    # Aggregate results by (display_name, file) to combine same methods
    aggregated = defaultdict(float)
    for display_name, file, pct in results:
        aggregated[(display_name, file)] += pct
    
    results = [(name, file, pct) for (name, file), pct in aggregated.items()]
    
    # Sort by percentage descending
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Calculate total time from samples and rate
    total_samples = len(samples)
    total_time_ms = (total_samples / sampling_rate) * 1000
    
    return results, total_samples, total_time_ms


def print_sampling_results(results: List[Tuple[str, str, float]], total_samples: int, total_time_ms: float, iterations: int, xml_name: str = "", top_n: int = 30):
    """Print sampling profiler results in a formatted table."""
    time_per_iter = total_time_ms / iterations if iterations > 0 else total_time_ms
    
    # Header matching instrumented mode style
    print()
    print("=" * 70)
    print(f"SAMPLING RESULTS{': ' + xml_name if xml_name else ''}")
    print("=" * 70)
    print()
    print(f"  Profiler:      py-spy (sampling)")
    print(f"  Iterations:    {iterations}")
    print(f"  Samples:       {total_samples:,}")
    print()
    print(f"  Time per iteration: {time_per_iter:,.1f}ms")
    print()
    
    print("─" * 70)
    print("HOTSPOTS (leaf functions)")
    print("─" * 70)
    print()
    
    # Filter to gxml functions only for cleaner output
    gxml_items = []
    other_total = 0.0
    
    for func_name, file_path, pct in results:
        if 'gxml' in file_path.lower() and 'site-packages' not in file_path.lower():
            gxml_items.append((func_name, pct, file_path))
        else:
            other_total += pct
    
    # Print header
    print(f"  {'Function':<40} {'Time':>14}  {'File'}")
    print(f"  {'─' * 40} {'─' * 14}  {'─' * 30}")
    
    gxml_total = 0.0
    for func_name, pct, file_path in gxml_items[:top_n]:
        gxml_total += pct
        # Simplify file path to show just the relevant part
        if 'gxml' in file_path:
            parts = file_path.replace('\\', '/').split('/')
            if 'src' in parts:
                idx = parts.index('src')
                short_path = '/'.join(parts[idx+1:])
            elif 'gxml' in parts:
                idx = len(parts) - 1 - parts[::-1].index('gxml')
                short_path = '/'.join(parts[idx:])
            else:
                short_path = Path(file_path).name
        else:
            short_path = Path(file_path).name
        
        # Calculate time for this function (per iteration)
        func_time_ms = (pct / 100) * time_per_iter
        
        # Color coding based on percentage
        if pct >= 5.0:
            color = YELLOW
        elif pct >= 1.0:
            color = CYAN
        else:
            color = GRAY
        
        # Format combined time and percentage
        if func_time_ms >= 1.0:
            time_pct_str = f"{func_time_ms:.1f}ms ({pct:.1f}%)"
        else:
            time_pct_str = f"{func_time_ms * 1000:.0f}µs ({pct:.1f}%)"
        
        print(f"  {color}{func_name:<40}{RESET} {time_pct_str:>14}  {DIM}{short_path}{RESET}")
    
    # Summary section
    print(f"  {'─' * 40} {'─' * 14}  {'─' * 30}")
    other_time = (other_total / 100) * time_per_iter
    gxml_time = (gxml_total / 100) * time_per_iter
    print(f"  {DIM}{'other (stdlib, numpy, imports)':<40}{RESET} {DIM}{other_time:.1f}ms ({other_total:.1f}%){RESET:>14}")
    print(f"  {BOLD}{'gxml total':<40}{RESET} {BOLD}{gxml_time:.1f}ms ({gxml_total:.1f}%){RESET:>14}")
    print()


def run_sampling_profiler(xml_path: Path, iterations: int, backend: str, rate: int = 1000) -> int:
    """
    Run py-spy sampling profiler and display formatted results.
    
    Returns exit code.
    """
    import shutil
    import os
    
    if not shutil.which("py-spy"):
        print("Error: py-spy not found. Install with: pip install py-spy")
        return 1
    
    # Use temp file for speedscope JSON
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        json_output = Path(tmp.name)
    
    # Create a simple runner script that directly runs the pipeline
    runner_code = f'''
import sys
sys.path.insert(0, r"{PROJECT_ROOT}")
sys.path.insert(0, r"{PROJECT_ROOT / 'src'}")

from gxml import run

xml_content = open(r"{xml_path.absolute()}", encoding="utf-8").read()

# Run multiple iterations for better sampling
for i in range({iterations}):
    result = run(xml_content, backend="{backend}")
    
# Print summary
print(f"Completed {{result.stats.get('panel_count', 0)}} panels, {{result.stats.get('intersection_count', 0)}} intersections")
'''
    
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as runner_file:
        runner_file.write(runner_code)
        runner_path = Path(runner_file.name)
    
    try:
        cmd = [
            "py-spy", "record",
            "-o", str(json_output),
            "-f", "speedscope",  # JSON format we can parse
            "--rate", str(rate),
            "--", sys.executable,  # No -O so we profile actual code
            str(runner_path),
        ]
        
        print(f"{BOLD}SAMPLING PROFILER{RESET}")
        print(f"{DIM}Using py-spy at {rate}Hz sampling rate{RESET}")
        print(f"{DIM}Running {iterations} iterations of actual pipeline code...{RESET}")
        print()
        
        # Run py-spy - use encoding='utf-8' and errors='replace' for Windows compatibility
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        # Show runner output (panel counts etc)
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        
        if result.stderr:
            # py-spy writes progress to stderr - show relevant lines
            for line in result.stderr.split('\n'):
                if any(kw in line for kw in ['Samples:', 'Errors:']):
                    print(f"{DIM}  {line.strip()}{RESET}")
        
        print()
        
        # Parse and display results even if subprocess had errors
        if json_output.exists() and json_output.stat().st_size > 0:
            results, total_samples, total_time_ms = parse_speedscope_json(json_output, rate)
            print_sampling_results(results, total_samples, total_time_ms, iterations, xml_path.name)
            return 0
        else:
            print("Warning: No sampling data collected")
            if result.stderr:
                print(result.stderr)
            return result.returncode if result.returncode != 0 else 1
        
    finally:
        # Clean up temp files
        if json_output.exists():
            json_output.unlink()
        if runner_path.exists():
            runner_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Profile a GXML document and display detailed timing breakdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/profile_xml.py tests/performance/xml/7Panels.xml
    python scripts/profile_xml.py my_layout.gxml --iterations 10
    python scripts/profile_xml.py layout.xml --backend c --no-hierarchy
    python scripts/profile_xml.py tests/performance/xml/200Panels.xml --overhead
    python scripts/profile_xml.py tests/performance/xml/200Panels.xml --sampling
        """
    )
    
    parser.add_argument("xml_path", type=Path, help="Path to the GXML/XML file to profile")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="Number of iterations (default: 1)")
    parser.add_argument("-b", "--backend", choices=["cpu", "c", "taichi"], default="cpu", help="Backend (default: cpu)")
    parser.add_argument("--no-hierarchy", action="store_true", help="Show flat list instead of tree")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup iteration")
    parser.add_argument("--overhead", action="store_true", help="Measure profiling overhead (runs with and without -O)")
    parser.add_argument("--sampling", action="store_true", help="Use py-spy sampling profiler (zero overhead)")
    parser.add_argument("--sampling-rate", type=int, default=1000, help="Sampling rate in Hz (default: 1000)")
    parser.add_argument("--discard-outliers", action="store_true", help="Rerun iterations that are ±10%% from mean")
    parser.add_argument("--outlier-threshold", type=float, default=0.10, help="Outlier threshold (default: 0.10 = ±10%%)")
    
    args = parser.parse_args()
    
    if not args.xml_path.exists():
        print(f"Error: File not found: {args.xml_path}")
        return 1
    
    # Sampling mode: use py-spy
    if args.sampling:
        return run_sampling_profiler(
            args.xml_path,
            args.iterations,
            args.backend,
            args.sampling_rate
        )
    
    try:
        run_benchmark(
            xml_content=args.xml_path.read_text(encoding='utf-8'),
            backend=args.backend,
            iterations=args.iterations,
            warmup=not args.no_warmup,
            show_hierarchy=not args.no_hierarchy,
            label=args.xml_path.name,
            measure_overhead=args.overhead,
            discard_outliers=args.discard_outliers,
            outlier_threshold=args.outlier_threshold,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
