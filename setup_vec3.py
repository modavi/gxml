"""
Build script for the _vec3 C extension.

Usage:
    python setup_vec3.py build_ext --inplace

This will compile the C extension and place _vec3.cpython-*.so 
in the src/gxml/mathutils/ directory.
"""

from setuptools import setup, Extension
import sys
import os

# Optimization flags
extra_compile_args = []
extra_link_args = []

if sys.platform == 'darwin':
    # macOS optimizations
    extra_compile_args = [
        '-O3',           # Maximum optimization
        '-ffast-math',   # Fast floating point (may change IEEE compliance)
        '-march=native', # Optimize for current CPU
        '-flto',         # Link-time optimization
    ]
    extra_link_args = ['-flto']
elif sys.platform == 'linux':
    extra_compile_args = [
        '-O3',
        '-ffast-math',
        '-march=native',
        '-flto',
    ]
    extra_link_args = ['-flto']
elif sys.platform == 'win32':
    extra_compile_args = [
        '/O2',           # Maximum optimization
        '/fp:fast',      # Fast floating point
    ]

vec3_extension = Extension(
    'gxml.mathutils._vec3',
    sources=['src/gxml/mathutils/_vec3.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='gxml-vec3',
    version='1.0.0',
    description='High-performance 3D vector math C extension for GXML',
    ext_modules=[vec3_extension],
    # This ensures the .so file ends up in the right place
    package_dir={'': 'src'},
)
