"""GXML - Geometric XML layout library."""
import sys
from pathlib import Path
from typing import Dict

__version__ = "0.1.0"

# Add src/gxml to path for flat imports used by the codebase
# This allows imports like `from gxml_engine import run` to work
_src_path = Path(__file__).parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Main API - use flat import (after path setup)
from gxml_engine import run, GXMLConfig, GXMLResult


# =============================================================================
# Backend Utilities
# =============================================================================

def check_backends() -> Dict[str, bool]:
    """Check which solver backends are available.
    
    Returns:
        Dict with keys 'cpu', 'c', 'taichi' and boolean availability values.
        'cpu' is always True.
    """
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
    
    # Taichi check disabled - currently broken on Windows (Vulkan overhead)
    # Can be re-enabled when Taichi issues are resolved
    
    return availability


def is_c_available() -> bool:
    """Quick check if C extension is available."""
    return check_backends()['c']


__all__ = ['run', 'GXMLConfig', 'GXMLResult', 'check_backends', 'is_c_available']