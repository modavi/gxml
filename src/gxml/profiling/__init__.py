"""
GXML Profiling Package

This package provides profiling and benchmarking utilities for GXML:

- profile: Lightweight profiling markers (@profile decorator, perf_marker context manager)
- benchmark: Full benchmarking utilities (run_benchmark, TimingResult, BenchmarkResult)

Quick usage:
    from gxml.profiling import profile, perf_marker, enable_profiling
    
    @profile
    def my_function():
        with perf_marker("my_section"):
            ...
    
    # For benchmarking:
    from gxml.profiling import run_benchmark
    result = run_benchmark(xml_content, backend='cpu', iterations=3)
"""

# Re-export from profile module
from .profile import (
    enable_profiling,
    is_profiling_enabled,
    reset_profile,
    get_profile_results,
    perf_marker,
    get_marker,
    profile,
    _PROFILING_COMPILED_OUT,
)

# Re-export from runner module
from .runner import (
    TimingResult,
    BenchmarkResult,
    OverheadResult,
    check_backends,
    is_c_available,
    run_pipeline,
    run_warmup,
    run_benchmark,
    assert_performance,
    print_profile_report,
    print_benchmark_result,
    print_timing_breakdown,
    print_comparison_table,
    format_time,
)

__all__ = [
    # Profile markers
    'enable_profiling',
    'is_profiling_enabled', 
    'reset_profile',
    'get_profile_results',
    'perf_marker',
    'get_marker',
    'profile',
    '_PROFILING_COMPILED_OUT',
    # Benchmark utilities
    'TimingResult',
    'BenchmarkResult',
    'OverheadResult',
    'check_backends',
    'is_c_available',
    'run_pipeline',
    'run_warmup',
    'run_benchmark',
    'assert_performance',
    'print_profile_report',
    'print_benchmark_result',
    'print_timing_breakdown',
    'print_comparison_table',
    'format_time',
]
