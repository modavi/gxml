"""
Build script for GXML native C extensions.

Compiles C extensions into their feature-local native/build/ folders:
    - src/gxml/elements/solvers/native/build/_c_solvers.*.pyd
    - src/gxml/profiling/native/build/_c_profiler.*.pyd
    - src/gxml/mathutils/native/build/_vec3.*.pyd

Usage:
    python build_native.py

For development, extensions are loaded via native_loader.py from these locations.
"""

import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np


# Extension definitions: (name, source_path, output_dir, needs_numpy)
EXTENSIONS = [
    ('_c_solvers', 'src/gxml/elements/solvers/native/c/solvers.c', 
     'src/gxml/elements/solvers/native/build', True),
    ('_c_profiler', 'src/gxml/profiling/native/c/profiler.c',
     'src/gxml/profiling/native/build', False),
    ('_vec3', 'src/gxml/mathutils/native/c/vec3.c',
     'src/gxml/mathutils/native/build', False),
]


def get_compiler_flags():
    """Get platform-specific compiler flags."""
    import platform
    
    if sys.platform == 'win32':
        return ['/O2', '/fp:fast', '/GL'], ['/LTCG']
    elif platform.system() == 'Darwin':
        return ['-O3', '-ffast-math', '-flto'], ['-flto']
    else:
        return ['-O3', '-march=native', '-ffast-math', '-flto'], ['-flto']


def build_extension(name, source, output_dir, needs_numpy=False):
    """Build a single extension using setuptools."""
    from setuptools import Extension, Distribution
    from setuptools.command.build_ext import build_ext
    
    compile_args, link_args = get_compiler_flags()
    
    include_dirs = [np.get_include()] if needs_numpy else []
    
    ext = Extension(
        name,
        sources=[source],
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
    
    dist = Distribution({'ext_modules': [ext]})
    dist.parse_config_files()
    
    cmd = build_ext(dist)
    cmd.ensure_finalized()
    cmd.inplace = False
    cmd.build_lib = output_dir
    cmd.build_temp = str(Path(output_dir) / 'temp')
    cmd.run()
    
    # Clean up temp directory
    temp_dir = Path(output_dir) / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print(f"  ✓ {name} -> {output_dir}")


def main():
    print("Building GXML native extensions...\n")
    
    for name, source, output_dir, needs_numpy in EXTENSIONS:
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        build_extension(name, source, output_dir, needs_numpy)
    
    print("\nDone! Extensions built to native/build/ folders.")


if __name__ == '__main__':
    main()
