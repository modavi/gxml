"""
Setup script for building the C extension solvers.

Run with: python setup_c_solvers.py build_ext --inplace
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

c_solvers_ext = Extension(
    'gxml.elements.solvers._c_solvers',
    sources=['src/gxml/elements/solvers/_c_solvers.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

if __name__ == '__main__':
    setup(
        name='gxml-c-solvers',
        ext_modules=[c_solvers_ext],
    )
