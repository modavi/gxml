"""
Setup script for building the C profiler extension.

Run with: python setup_c_profiler.py build_ext --inplace
"""

from setuptools import setup, Extension
import sys
import platform

# Platform-specific compiler flags
if sys.platform == 'win32':
    # MSVC compiler flags
    extra_compile_args = ['/O2', '/GL']
    extra_link_args = ['/LTCG']
elif platform.system() == 'Darwin':
    # macOS clang flags
    extra_compile_args = ['-O3', '-flto']
    extra_link_args = ['-flto']
else:
    # Linux GCC flags
    extra_compile_args = ['-O3', '-flto']
    extra_link_args = ['-flto']

c_profiler_ext = Extension(
    'gxml.profiling._c_profiler',
    sources=['src/gxml/profiling/_c_profiler.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

if __name__ == '__main__':
    setup(
        name='gxml-c-profiler',
        ext_modules=[c_profiler_ext],
    )
