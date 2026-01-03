# CMake build system for GXML cross-platform C extensions
# Supports: macOS, Windows, Linux, and WASM (via Emscripten)
#
# Usage:
#   mkdir build && cd build
#   cmake .. && cmake --build .
#
# For WASM:
#   emcmake cmake .. && emmake make

cmake_minimum_required(VERSION 3.15)
project(gxml_solvers VERSION 1.0.0 LANGUAGES C)

# ============================================================================
# Platform Detection
# ============================================================================

set(BUILD_WASM OFF)
set(BUILD_PYTHON ON)

if(DEFINED ENV{EMSDK} OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(BUILD_WASM ON)
    set(BUILD_PYTHON OFF)
    message(STATUS "Building for WebAssembly (Emscripten)")
elseif(WIN32)
    message(STATUS "Building for Windows")
elseif(APPLE)
    message(STATUS "Building for macOS")
else()
    message(STATUS "Building for Linux/Unix")
endif()

# ============================================================================
# Compiler Flags
# ============================================================================

if(BUILD_WASM)
    # Emscripten flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s WASM=1")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap']")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s ALLOW_MEMORY_GROWTH=1")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s MODULARIZE=1")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s EXPORT_NAME='GXMLSolvers'")
elseif(MSVC)
    # Windows MSVC
    add_compile_options(/O2 /fp:fast /W3)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang (macOS, Linux)
    add_compile_options(-O3 -ffast-math -Wall -Wextra)
    if(NOT APPLE)
        add_compile_options(-march=native)
    endif()
endif()

# ============================================================================
# Source Files
# ============================================================================

set(SOLVER_SOURCES
    src/gxml/elements/solvers/_c_solvers.c
)

set(VEC3_SOURCES
    src/gxml/mathutils/_vec3.c
)

# ============================================================================
# WASM Build (standalone, no Python)
# ============================================================================

if(BUILD_WASM)
    # Create WASM-specific wrapper
    add_executable(gxml_solvers_wasm
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gxml/wasm/wasm_solvers.c
    )
    
    set_target_properties(gxml_solvers_wasm PROPERTIES
        OUTPUT_NAME "gxml_solvers"
        SUFFIX ".js"
    )
    
    # Also output standalone wasm
    target_link_options(gxml_solvers_wasm PRIVATE
        -s EXPORTED_FUNCTIONS=['_solve_intersections','_build_geometry','_malloc','_free']
        -s STANDALONE_WASM=0
    )
    
    # Install WASM files
    install(FILES
        ${CMAKE_BINARY_DIR}/gxml_solvers.js
        ${CMAKE_BINARY_DIR}/gxml_solvers.wasm
        DESTINATION wasm
    )

# ============================================================================
# Python Extension Build
# ============================================================================

elseif(BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
    
    # _c_solvers extension
    Python3_add_library(_c_solvers MODULE ${SOLVER_SOURCES})
    target_include_directories(_c_solvers PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
    target_link_libraries(_c_solvers PRIVATE Python3::Python Python3::NumPy)
    
    # _vec3 extension
    Python3_add_library(_vec3 MODULE ${VEC3_SOURCES})
    target_link_libraries(_vec3 PRIVATE Python3::Python)
    
    # Set output directories
    set_target_properties(_c_solvers PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/gxml/elements/solvers
    )
    set_target_properties(_vec3 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/gxml/mathutils
    )
    
    # Install Python extensions
    install(TARGETS _c_solvers _vec3
        LIBRARY DESTINATION lib
    )
endif()

# ============================================================================
# Testing
# ============================================================================

enable_testing()

if(NOT BUILD_WASM)
    add_test(NAME python_import_test
        COMMAND ${Python3_EXECUTABLE} -c "import sys; sys.path.insert(0, '${CMAKE_SOURCE_DIR}/src'); from gxml.elements.solvers import _c_solvers; print('C solvers loaded successfully')"
    )
endif()
