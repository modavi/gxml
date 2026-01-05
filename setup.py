"""
Setup script for GXML with C extensions.

Usage:
    pip install -e .           # Editable install with C extensions
    pip install -e ".[dev]"    # With dev dependencies

C extensions are built next to their Python source (standard approach):
    - src/gxml/elements/solvers/_c_solvers.*.pyd
    - src/gxml/profiling/_c_profile.*.pyd
    - src/gxml/mathutils/_vec3.*.pyd

These .pyd/.so files are gitignored.
"""

from setuptools import setup, Extension
import numpy as np
import sys
import platform


def get_compiler_flags():
    """Get platform-specific compiler flags."""
    if sys.platform == 'win32':
        return ['/O2', '/fp:fast', '/GL'], ['/LTCG']
    elif platform.system() == 'Darwin':
        return ['-O3', '-ffast-math', '-flto'], ['-flto']
    else:
        return ['-O3', '-march=native', '-ffast-math', '-flto'], ['-flto']


def get_extensions():
    """Build Extension objects."""
    compile_args, link_args = get_compiler_flags()
    
    return [
        # C Solvers (intersection, face, geometry)
        Extension(
            'gxml.elements.solvers._c_solvers',
            sources=['src/gxml/elements/solvers/_c_solvers.c'],
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        # C Profiler
        Extension(
            'gxml.profiling._c_profile',
            sources=['src/gxml/profiling/_c_profile.c'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        # Vec3 math
        Extension(
            'gxml.mathutils._vec3',
            sources=['src/gxml/mathutils/_vec3.c'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
    ]


setup(
    ext_modules=get_extensions(),
    package_dir={'': 'src'},
)
