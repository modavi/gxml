"""
Native extension loader for GXML.

Loads compiled .pyd/.so files from native/build/ folders,
keeping build artifacts separate from source code.

Structure:
    feature/
        native/
            c/        # C source files
            build/    # Compiled .pyd/.so files (gitignored)
"""

import importlib.util
import sys
from pathlib import Path


def load_native_extension(name: str, native_dir: Path):
    """
    Load a native extension from a native/build/ directory.
    
    Args:
        name: Extension module name (e.g., '_c_solvers')
        native_dir: Path to the native/ folder (build/ is inside it)
        
    Returns:
        The loaded module, or None if not found
        
    Example:
        native_dir = Path(__file__).parent / 'native'
        _c_solvers = load_native_extension('_c_solvers', native_dir)
    """
    build_dir = native_dir / 'build'
    if not build_dir.exists():
        return None
    
    # Find the extension file (.pyd on Windows, .so on Unix)
    patterns = [f'{name}*.pyd', f'{name}*.so']
    ext_file = None
    
    for pattern in patterns:
        matches = list(build_dir.glob(pattern))
        if matches:
            ext_file = matches[0]
            break
    
    if ext_file is None:
        return None
    
    # Load the module using importlib
    spec = importlib.util.spec_from_file_location(name, ext_file)
    if spec is None or spec.loader is None:
        return None
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    
    return module
