"""
Setup script for building the C extension solvers.

Run with: python setup_c_solvers.py build_ext --inplace
"""

from setuptools import setup, Extension
import numpy as np

c_solvers_ext = Extension(
    'gxml.elements.solvers._c_solvers',
    sources=['src/gxml/elements/solvers/_c_solvers.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3', '-march=native', '-ffast-math'],
)

if __name__ == '__main__':
    setup(
        name='gxml-c-solvers',
        ext_modules=[c_solvers_ext],
    )
