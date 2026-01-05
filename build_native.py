"""
Unified build script for all GXML native C extensions.

Usage:
    python build_native.py build_ext --inplace

This compiles all C extensions:
    - _c_solvers -> gxml.elements.solvers._c_solvers
    - _c_profiler -> gxml.profiling._c_profiler
    - _vec3 -> gxml.mathutils._vec3

The source files are located in native/c/ subdirectories:
    - src/gxml/elements/solvers/native/c/solvers.c
    - src/gxml/profiling/native/c/profiler.c
    - src/gxml/mathutils/native/c/vec3.c
"""

from setuptools import setup, Extension
import numpy as np
import sys
import platform

# Platform-specific compiler flags
if sys.platform == 'win32':
    # MSVC compiler flags
    extra_compile_args = ['/O2', '/fp:fast', '/GL']
    extra_link_args = ['/LTCG']
elif platform.system() == 'Darwin':
    # macOS clang flags
    extra_compile_args = ['-O3', '-ffast-math', '-flto']
    extra_link_args = ['-flto']
else:
    # Linux GCC flags
    extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-flto']
    extra_link_args = ['-flto']


# C Solvers extension (intersection solver, face solver, geometry builder)
c_solvers_ext = Extension(
    'gxml.elements.solvers._c_solvers',
    sources=['src/gxml/elements/solvers/native/c/solvers.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

# C Profiler extension (high-performance profiling)
c_profiler_ext = Extension(
    'gxml.profiling._c_profiler',
    sources=['src/gxml/profiling/native/c/profiler.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

# Vec3 extension (3D vector/matrix math)
vec3_ext = Extension(
    'gxml.mathutils._vec3',
    sources=['src/gxml/mathutils/native/c/vec3.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)


if __name__ == '__main__':
    setup(
        name='gxml-native',
        version='1.0.0',
        description='Native C extensions for GXML',
        ext_modules=[c_solvers_ext, c_profiler_ext, vec3_ext],
        package_dir={'': 'src'},
    )
