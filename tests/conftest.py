"""
Pytest configuration for gxml tests.
Adds the src/gxml directory to sys.path so that test imports work correctly.
"""
import sys
from pathlib import Path

# Add src/gxml to the path so imports like 'from gxml_layout import ...' work
src_path = Path(__file__).parent.parent / "src" / "gxml"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
