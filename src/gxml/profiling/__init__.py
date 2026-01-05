"""
GXML Profiling Package

This package provides lightweight profiling markers for GXML:

- perf_marker: Context manager for timing code sections
- @profile: Decorator for timing functions
- get_profile_results: Retrieve timing data after a run

Quick usage:
    from gxml.profiling import profile, perf_marker
    
    @profile
    def my_function():
        with perf_marker("my_section"):
            ...

For benchmarking utilities, use scripts/benchmark.py or tests/test_fixtures/profiling.py
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
]
