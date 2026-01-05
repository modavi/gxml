"""GXML - Geometric XML layout library."""
import sys
from pathlib import Path

__version__ = "0.1.0"

# Add src/gxml to path for flat imports used by the codebase
# This allows imports like `from gxml_engine import run` to work
_src_path = Path(__file__).parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Main API - use flat import (after path setup)
from gxml_engine import run, GXMLConfig, GXMLResult, check_backends

__all__ = ['run', 'GXMLConfig', 'GXMLResult', 'check_backends']