"""
GXML Profiling Package

Lightweight profiling markers for GXML performance measurement.

Usage:
    from gxml.profiling import profile, perf_marker, get_profile_results, reset_profile
    
    @profile
    def my_function():
        with perf_marker("my_section"):
            ...
    
    results = get_profile_results()
    reset_profile()

For benchmarking utilities, use scripts/benchmark.py
"""

from .profile import (
    # Public API
    profile,
    perf_marker,
    get_profile_results,
    reset_profile,
    # Internal (used by tests)
    _PROFILING_COMPILED_OUT,
)

__all__ = [
    'profile',
    'perf_marker',
    'get_profile_results',
    'reset_profile',
]
