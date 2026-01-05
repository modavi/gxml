"""
Setup script for GXML C extensions.

Usage:
    pip install -e .           # Editable install (recommended for development)
    pip install -e ".[dev]"    # With dev dependencies

This builds native extensions:
    - _c_solvers  (intersection, face, geometry solvers)
    - _c_profiler (high-performance profiling)  
    - _vec3       (vector/matrix math)

For editable installs, .pyd/.so files are placed in src/gxml/... 
These are excluded from git via .gitignore.
"""

from setuptools import setup, Extension
import numpy as np
import sys
import platform


def get_extensions():
    """Build extension definitions with platform-specific flags."""
    
    # Platform-specific compiler flags
    if sys.platform == 'win32':
        extra_compile_args = ['/O2', '/fp:fast', '/GL']
        extra_link_args = ['/LTCG']
    elif platform.system() == 'Darwin':
        extra_compile_args = ['-O3', '-ffast-math', '-flto']
        extra_link_args = ['-flto']
    else:
        extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-flto']
        extra_link_args = ['-flto']

    return [
        # C Solvers (intersection, face, geometry)
        Extension(
            'gxml.elements.solvers._c_solvers',
            sources=['src/gxml/elements/solvers/native/c/solvers.c'],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
        # C Profiler
        Extension(
            'gxml.profiling._c_profiler',
            sources=['src/gxml/profiling/native/c/profiler.c'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
        # Vec3 math
        Extension(
            'gxml.mathutils._vec3',
            sources=['src/gxml/mathutils/native/c/vec3.c'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]


if __name__ == '__main__':
    setup(
        ext_modules=get_extensions(),
        package_dir={'': 'src'},
    )
