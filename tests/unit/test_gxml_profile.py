"""
Tests for gxml_profile module.
"""
import subprocess
import sys
import pytest


class TestProfilingCompiledOut:
    """Tests for the zero-overhead compile-out feature."""
    
    def test_normal_mode_not_compiled_out(self):
        """In normal mode (no -O flag), profiling should NOT be compiled out."""
        from gxml.profiling import _PROFILING_COMPILED_OUT
        assert _PROFILING_COMPILED_OUT is False
    
    def test_optimized_mode_compiled_out(self):
        """When running with python -O, profiling should be compiled out."""
        # Run a subprocess with -O flag to test
        result = subprocess.run(
            [sys.executable, "-O", "-c", 
             "from gxml.profiling import _PROFILING_COMPILED_OUT; "
             "print(_PROFILING_COMPILED_OUT)"],
            capture_output=True,
            text=True,
            cwd=str(pytest.importorskip("gxml").__file__).rsplit("\\", 2)[0].rsplit("/", 2)[0]
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "True" in result.stdout
    
    def test_env_var_compiles_out(self):
        """When GXML_NO_PROFILING=1, profiling should be compiled out."""
        import os
        env = os.environ.copy()
        env["GXML_NO_PROFILING"] = "1"
        
        result = subprocess.run(
            [sys.executable, "-c",
             "from gxml.profiling import _PROFILING_COMPILED_OUT; "
             "print(_PROFILING_COMPILED_OUT)"],
            capture_output=True,
            text=True,
            env=env
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "True" in result.stdout
    
    def test_compiled_out_mode_has_noop_functions(self):
        """When compiled out, all functions should be no-ops that don't error."""
        import os
        env = os.environ.copy()
        env["GXML_NO_PROFILING"] = "1"
        
        # Test that all public functions work without error in compiled-out mode
        code = """
from gxml.profiling import (
    reset_profile, get_profile_results, perf_marker, profile
)

# All these should work without error
reset_profile()
assert get_profile_results() == {}

# Context manager should work
with perf_marker("test"):
    pass

# Decorator should work
@profile
def my_func():
    return 42

@profile("custom_name")
def my_func2():
    return 43

assert my_func() == 42
assert my_func2() == 43

print("OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "OK" in result.stdout


class TestProfilingFunctionality:
    """Tests for actual profiling functionality."""
    
    def test_perf_marker_records_timing(self):
        """Test that perf_marker records timing data."""
        from gxml.profiling import reset_profile, perf_marker, get_profile_results
        import time
        
        reset_profile()
        
        with perf_marker("test_marker"):
            time.sleep(0.01)  # Sleep 10ms
        
        results = get_profile_results()
        
        assert "test_marker" in results
        assert results["test_marker"]["count"] == 1
        assert results["test_marker"]["total_ms"] >= 5  # At least 5ms (allowing some slack)
    
    def test_profile_decorator_records_timing(self):
        """Test that @profile decorator records timing data."""
        from gxml.profiling import reset_profile, profile, get_profile_results
        import time
        
        reset_profile()
        
        @profile
        def slow_function():
            time.sleep(0.01)
            return 42
        
        result = slow_function()
        
        results = get_profile_results()
        
        assert result == 42
        assert "slow_function" in results
        assert results["slow_function"]["count"] == 1
        assert results["slow_function"]["total_ms"] >= 5
    
    def test_profile_decorator_with_custom_name(self):
        """Test @profile decorator with custom name."""
        from gxml.profiling import reset_profile, profile, get_profile_results
        
        reset_profile()
        
        @profile("my_custom_name")
        def some_function():
            return 123
        
        some_function()
        
        results = get_profile_results()
        
        assert "my_custom_name" in results
        assert "some_function" not in results
    
    def test_nested_markers_track_hierarchy(self):
        """Test that nested markers track parent-child relationships."""
        from gxml.profiling import reset_profile, perf_marker, get_profile_results
        
        reset_profile()
        
        with perf_marker("outer"):
            with perf_marker("inner"):
                pass
        
        results = get_profile_results()
        
        assert "outer" in results
        assert "inner" in results
        # Inner should have outer as parent
        assert results["inner"]["parents"].get("outer", 0) == 1
    
    def test_multiple_calls_accumulate(self):
        """Test that multiple calls to same marker accumulate stats."""
        from gxml.profiling import reset_profile, perf_marker, get_profile_results
        
        reset_profile()
        
        for _ in range(5):
            with perf_marker("repeated"):
                pass
        
        results = get_profile_results()
        
        assert results["repeated"]["count"] == 5
    
    def test_reset_clears_data(self):
        """Test that reset_profile clears all data."""
        from gxml.profiling import reset_profile, perf_marker, get_profile_results
        
        with perf_marker("before_reset"):
            pass
        
        assert "before_reset" in get_profile_results()
        
        reset_profile()
        
        assert get_profile_results() == {}
