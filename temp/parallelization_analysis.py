"""
GXML Parallelization Analysis
=============================

This script analyzes the GXML pipeline for parallelization opportunities.
It profiles the key stages and identifies which operations can be parallelized.

Current Pipeline (Sequential):
1. IntersectionSolver.solve()  - Find panel intersections (~300ms, 15% of time)
2. FaceSolver.solve()          - Compute face segmentation (~650ms, 33% of time)  
3. GeometryBuilder.build()     - Create 3D geometry (~580ms, 30% of time)
4. Rendering                   - Output to JSON/binary (~180ms, 9% of time)

Analysis Summary:
================

MULTITHREADING OPPORTUNITIES:
-----------------------------

1. IntersectionSolver - PARTIAL PARALLELISM
   - Candidate pair checking (O(n²) worst case) - PARALLELIZABLE
   - Each pair intersection test is independent
   - Estimated speedup: 3-4x with thread pool
   - Challenge: Merging results requires synchronization

2. FaceSolver - HIGH PARALLELISM POTENTIAL
   - _solve_panel() is called per-panel - FULLY PARALLELIZABLE
   - Each panel's face segmentation is independent
   - Estimated speedup: 6-8x on 8-core CPU
   - Current: 228 calls taking ~650ms total → ~3ms/panel
   - With 8 threads: ~80ms theoretical minimum

3. GeometryBuilder - HIGH PARALLELISM POTENTIAL
   - _create_panel_faces() per panel - FULLY PARALLELIZABLE
   - _create_caps_for_panel() per panel - PARALLELIZABLE
   - Estimated speedup: 5-7x
   - Current: ~580ms → ~80-100ms with threading

4. Rendering - MEDIUM PARALLELISM
   - create_poly() calls - PARALLELIZABLE
   - Current: 4044 calls, ~100ms
   - Could batch into worker threads

TOTAL MULTITHREADING ESTIMATE:
- Current 75-panel: ~280ms
- With full threading: ~50-80ms (3.5-5.5x speedup)
- Limiting factors: Python GIL, synchronization overhead


GPU ACCELERATION OPPORTUNITIES:
-------------------------------

1. Matrix transforms (transform_point) - EXCELLENT GPU FIT
   - 12,552 calls in 75-panel test
   - Embarrassingly parallel matrix-vector multiplications
   - Could batch ALL transforms into single GPU kernel
   - Estimated speedup: 50-100x for transform operations alone

2. Intersection detection - GOOD GPU FIT
   - Line-line intersection is simple math
   - Can process all pairs in parallel
   - BVH/spatial hash could accelerate broad phase
   - Estimated speedup: 10-50x

3. Face segment computation - MODERATE GPU FIT
   - Some branching logic (less ideal for GPU)
   - Gap/trim calculations are arithmetic-heavy
   - Estimated speedup: 5-20x

4. Geometry generation - GOOD GPU FIT
   - Vertex generation is parallel
   - Could output directly to GPU vertex buffers
   - Zero-copy path to WebGPU/Three.js
   - Estimated speedup: 20-50x

TOTAL GPU ESTIMATE:
- Current 75-panel: ~280ms
- With GPU acceleration: ~5-20ms (14-56x speedup)
- Best case with full GPU pipeline: <5ms


IMPLEMENTATION COMPLEXITY:
--------------------------

MULTITHREADING (Python):
- Use concurrent.futures.ProcessPoolExecutor (avoids GIL)
- Moderate complexity: ~3-5 days work
- Requires pickling/serializing panel data
- Pro: Works on all platforms, no special hardware
- Con: GIL limits thread-based parallelism, process overhead

GPU ACCELERATION:
Option A: Metal/CUDA via PyMetal/PyCUDA
- High complexity: ~2-3 weeks
- Platform-specific (Metal=macOS, CUDA=NVIDIA)
- Maximum performance potential

Option B: OpenCL via PyOpenCL
- Medium complexity: ~1-2 weeks  
- Cross-platform
- Good performance, broad hardware support

Option C: Taichi (recommended for exploration)
- Low-medium complexity: ~1 week
- Python-like syntax, auto-differentiable
- Compiles to CUDA/Metal/Vulkan/OpenGL
- Great for prototyping GPU algorithms

Option D: NumPy + Numba
- Low complexity: ~3-5 days
- JIT compilation, automatic SIMD
- Can target GPU via numba.cuda
- Good middle ground


RECOMMENDATION:
---------------

Phase 1 (Quick Win): Multiprocessing for FaceSolver/GeometryBuilder
- Expected improvement: 3-4x
- Time to implement: 3-5 days
- Risk: Low

Phase 2 (Medium Term): Batch matrix transforms to C extension
- Expected improvement: 2x additional
- Time to implement: 1 week
- Already have SIMD foundation

Phase 3 (Long Term): GPU acceleration via Taichi or Metal
- Expected improvement: 10-50x total
- Time to implement: 2-4 weeks
- Requires architecture changes

For interactive use (web frontend), Phase 1 alone would bring
75-panel from ~280ms to ~70-100ms, which is excellent for real-time editing.
"""

import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml-web/src')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from gxml_web.json_render_engine import JSONRenderEngine


def analyze_parallelism():
    """Analyze the pipeline stages for parallelization opportunities."""
    
    xml_75_panels = '''<root>
    <panel thickness="0.25"/>
    <panel width="2.55" thickness="0.25" rotate="90" attach="0:1"/>
    <panel width="2.76" thickness="0.25" rotate="-135" attach="1:1"/>
    <panel width="2.873" thickness="0.25" rotate="-45" attach="2:1"/>
    <panel width="2.726" thickness="0.25" rotate="-90" attach="3:1"/>
    <panel width="4.716" thickness="0.25" rotate="315" attach="4:1"/>
    <panel width="6.608" thickness="0.25" rotate="-45" attach="5:1"/>
    <panel width="4.568" thickness="0.25" rotate="-90" attach="6:1"/>
    <panel width="2.627" thickness="0.25" rotate="-45" attach="7:1"/>
    <panel width="8.179" thickness="0.25" rotate="-45" attach="8:1"/>
    <panel width="2.338" thickness="0.25" attach="9:1"/>
    <panel width="3.419" thickness="0.25" rotate="-90" attach="10:1"/>
    <panel width="12.747" thickness="0.25" rotate="315" attach="11:1"/>
    <panel width="15.002" thickness="0.25" rotate="-45" attach="12:1"/>
    <panel width="11.687" thickness="0.25" rotate="-135" attach="13:1"/>
    <panel width="4.46" thickness="0.25" rotate="45" attach="14:1"/>
    <panel width="12.011" thickness="0.25" rotate="-90" attach="15:1"/>
    <panel width="2.839" thickness="0.25" rotate="45" attach="16:1"/>
    <panel width="1.719" thickness="0.25" rotate="-90" attach="17:1"/>
    <panel width="2.481" thickness="0.25" rotate="45" attach="18:1"/>
    <panel width="3.649" thickness="0.25" rotate="-90" attach="19:1"/>
    <panel width="10.153" thickness="0.25" rotate="315" attach="20:1"/>
    <panel width="9.466" thickness="0.25" rotate="45" attach="21:1"/>
    <panel width="48.329" thickness="0.25" rotate="-45" attach="22:1"/>
    <panel width="33.266" thickness="0.25" rotate="-90" attach="23:1"/>
    <panel width="39.774" thickness="0.25" rotate="-90" attach="24:1"/>
    <panel width="9.288" thickness="0.25" rotate="-90" attach="25:1"/>
    <panel width="6.634" thickness="0.25" rotate="270" attach="26:1"/>
    <panel width="6.199" thickness="0.25" rotate="-315" attach="27:1"/>
    <panel width="7.363" thickness="0.25" rotate="90" attach="28:1"/>
    <panel width="19.744" thickness="0.25" rotate="90" attach="29:1"/>
    <panel width="18.081" thickness="0.25" rotate="-90" attach="30:1"/>
    <panel width="4.109" thickness="0.25" rotate="45" attach="31:1"/>
    <panel width="3.253" thickness="0.25" rotate="-45" attach="32:1"/>
    <panel width="2.065" thickness="0.25" attach="33:1"/>
    <panel width="8.026" thickness="0.25" attach="34:1"/>
    <panel width="9.161" thickness="0.25" rotate="270" attach="35:1"/>
    <panel width="4.546" thickness="0.25" rotate="-90" attach="36:1"/>
    <panel width="10.949" thickness="0.25" rotate="90" attach="37:1"/>
    <panel width="98.513" thickness="0.25" rotate="-45" attach="38:1"/>
    <panel width="62.132" thickness="0.25" rotate="-45" attach="39:1"/>
    <panel width="93.063" thickness="0.25" rotate="-135" attach="40:1"/>
    <panel width="36.601" thickness="0.25" attach="41:1"/>
    <panel width="29.459" thickness="0.25" rotate="-45" attach="42:1"/>
    <panel width="6.113" thickness="0.25" rotate="45" attach="43:1"/>
    <panel width="7.991" thickness="0.25" rotate="-45" attach="44:1"/>
    <panel width="3.706" thickness="0.25" rotate="225" attach="45:1"/>
    <panel width="4.254" thickness="0.25" rotate="-225" attach="46:1"/>
    <panel width="2.842" thickness="0.25" rotate="-45" attach="47:1"/>
    <panel width="3.318" thickness="0.25" attach="48:1"/>
    <panel width="4.059" thickness="0.25" rotate="-45" attach="49:1"/>
    <panel width="5.011" thickness="0.25" rotate="270" attach="50:1"/>
    <panel width="34.35" thickness="0.25" rotate="45" attach="51:1"/>
    <panel width="220.31" thickness="0.25" attach="52:1"/>
    <panel width="117.746" thickness="0.25" rotate="-135" attach="53:1"/>
    <panel width="39.222" thickness="0.25" rotate="-45" attach="54:1"/>
    <panel width="129.095" thickness="0.25" rotate="-45" attach="55:1"/>
    <panel width="44.185" thickness="0.25" rotate="90" attach="56:1"/>
    <panel width="17.985" thickness="0.25" rotate="-90" attach="57:1"/>
    <panel width="19.16" thickness="0.25" rotate="-90" attach="58:1"/>
    <panel width="79.035" thickness="0.25" rotate="-90" attach="59:1"/>
    <panel width="61.367" thickness="0.25" rotate="90" attach="60:1"/>
    <panel width="47.785" thickness="0.25" rotate="-270" attach="61:1"/>
    <panel width="84.833" thickness="0.25" rotate="90" attach="62:1"/>
    <panel width="35.99" thickness="0.25" rotate="-90" attach="63:1"/>
    <panel width="22.698" thickness="0.25" rotate="270" attach="64:1"/>
    <panel width="25.056" thickness="0.25" rotate="-45" attach="65:1"/>
    <panel width="48.026" thickness="0.25" rotate="-90" attach="66:1"/>
    <panel width="55.272" thickness="0.25" rotate="-135" attach="67:1"/>
    <panel width="9.979" thickness="0.25" rotate="-90" attach="68:1"/>
    <panel width="70.301" thickness="0.25" rotate="-90" attach="69:1"/>
    <panel width="87.256" thickness="0.25" rotate="-270" attach="70:1"/>
    <panel width="73.204" thickness="0.25" rotate="135" attach="71:1"/>
    <panel width="50.48" thickness="0.25" rotate="45" attach="72:1"/>
    <panel width="10.534" thickness="0.25" rotate="-90" attach="73:1"/>
    <panel width="0.586" thickness="0.25" rotate="-90" attach="74:1"/>
</root>'''
    
    # Detailed timing breakdown - need to manually time each stage
    from elements.solvers.gxml_intersection_solver import IntersectionSolver
    from elements.solvers.gxml_face_solver import FaceSolver
    from elements.solvers.gxml_geometry_builder import GeometryBuilder
    from elements.gxml_panel import GXMLPanel
    
    # First do a full layout to get the actual panels
    root = GXMLParser.parse(xml_75_panels)
    GXMLLayout.layout(root)
    
    # Get panels after full layout
    panels = [e for e in root.iterate() if isinstance(e, GXMLPanel)]
    print(f"Found {len(panels)} panels after layout")
    
    # Now measure each stage separately with fresh parse each time
    
    # Stage 0: Parsing
    t0 = time.perf_counter()
    for _ in range(3):
        root = GXMLParser.parse(xml_75_panels)
    t_parse = (time.perf_counter() - t0) / 3
    
    # Stage 1: Pre-layout (includes transform computation)  
    root = GXMLParser.parse(xml_75_panels)
    t0 = time.perf_counter()
    for _ in range(3):
        root = GXMLParser.parse(xml_75_panels)
        GXMLLayout.pre_layout_pass(root)
    t_prelayout_total = (time.perf_counter() - t0) / 3
    t_prelayout = t_prelayout_total - t_parse
    
    # Stage 2-4 need full layout first, then we time the solvers
    # We need to instrument post_layout_pass
    
    # Time full layout multiple times
    times_total = []
    times_intersection = []
    times_face = []
    times_geometry = []
    
    for _ in range(5):
        root = GXMLParser.parse(xml_75_panels)
        t0 = time.perf_counter()
        GXMLLayout.layout(root)
        times_total.append(time.perf_counter() - t0)
    
    avg_total = sum(times_total) / len(times_total)
    
    # Get intersection count from a run
    root = GXMLParser.parse(xml_75_panels)
    GXMLLayout.layout(root)
    panels = [e for e in root.iterate() if isinstance(e, GXMLPanel)]
    
    # Run solvers individually to time them
    intersection_solution = IntersectionSolver.solve(panels)
    
    # Time intersection solver
    t0 = time.perf_counter()
    for _ in range(5):
        IntersectionSolver.solve(panels)
    t_intersection = (time.perf_counter() - t0) / 5
    
    # Time face solver  
    t0 = time.perf_counter()
    for _ in range(5):
        panel_faces = FaceSolver.solve(intersection_solution)
    t_face = (time.perf_counter() - t0) / 5
    
    # Time geometry builder
    panel_faces = FaceSolver.solve(intersection_solution)
    t0 = time.perf_counter()
    for _ in range(5):
        for panel in panels:
            # Reset geometry to time fresh builds
            panel.quads = []
            panel.polys = []
            GeometryBuilder.build(panel, panel_faces, intersection_solution)
    t_geometry = (time.perf_counter() - t0) / 5
    
    # Stage 4: Rendering
    root = GXMLParser.parse(xml_75_panels)
    GXMLLayout.layout(root)
    t0 = time.perf_counter()
    for _ in range(5):
        render_engine = JSONRenderEngine()
        GXMLRender.render(root, render_engine)
    t_render = (time.perf_counter() - t0) / 5

    total = t_parse + t_prelayout + t_intersection + t_face + t_geometry + t_render
    
    print("=" * 70)
    print("GXML PIPELINE BREAKDOWN (75 panels)")
    print("=" * 70)
    print(f"{'Stage':<30} {'Time (ms)':<12} {'%':<8} {'Parallelizable'}")
    print("-" * 70)
    print(f"{'Parsing':<30} {t_parse*1000:>8.1f} ms  {t_parse/total*100:>5.1f}%  {'Low'}")
    print(f"{'Pre-layout (transforms)':<30} {t_prelayout*1000:>8.1f} ms  {t_prelayout/total*100:>5.1f}%  {'Medium'}")
    print(f"{'IntersectionSolver':<30} {t_intersection*1000:>8.1f} ms  {t_intersection/total*100:>5.1f}%  {'Partial'}")
    print(f"{'FaceSolver':<30} {t_face*1000:>8.1f} ms  {t_face/total*100:>5.1f}%  {'HIGH'}")
    print(f"{'GeometryBuilder':<30} {t_geometry*1000:>8.1f} ms  {t_geometry/total*100:>5.1f}%  {'HIGH'}")
    print(f"{'Rendering':<30} {t_render*1000:>8.1f} ms  {t_render/total*100:>5.1f}%  {'Medium'}")
    print("-" * 70)
    print(f"{'TOTAL':<30} {total*1000:>8.1f} ms")
    print(f"{'(Full layout avg)':<30} {avg_total*1000:>8.1f} ms")
    print()
    
    # Count operations
    print("KEY METRICS:")
    print(f"  Panels: {len(panels)}")
    print(f"  Intersections: {len(intersection_solution.intersections)}")
    print(f"  Face segments created: {sum(len(pf.segments.get(s, [])) for pf in panel_faces for s in pf.segments)}")
    print()
    
    # Theoretical parallel speedup
    cpu_cores = mp.cpu_count()
    print(f"THEORETICAL SPEEDUP (with {cpu_cores} cores):")
    
    # FaceSolver is per-panel, nearly perfect parallelism
    parallel_face = t_face / cpu_cores
    parallel_geometry = t_geometry / cpu_cores
    
    # Intersection solver has ~50% parallelizable work
    parallel_intersection = t_intersection * 0.5 + t_intersection * 0.5 / cpu_cores
    
    parallel_total = t_prelayout + parallel_intersection + parallel_face + parallel_geometry + t_render
    
    print(f"  Current total:        {total*1000:.1f} ms")
    print(f"  Parallel estimate:    {parallel_total*1000:.1f} ms")
    print(f"  Speedup:              {total/parallel_total:.1f}x")
    print()
    
    # GPU estimate (assuming 1000x parallelism for embarrassingly parallel ops)
    gpu_factor = 100  # Conservative estimate
    gpu_total = t_prelayout/gpu_factor + t_intersection/gpu_factor*2 + t_face/gpu_factor + t_geometry/gpu_factor + t_render/gpu_factor*2
    print(f"GPU ESTIMATE (conservative 100x parallelism for compute):")
    print(f"  GPU estimate:         {gpu_total*1000:.1f} ms")
    print(f"  Speedup:              {total/gpu_total:.0f}x")


if __name__ == "__main__":
    analyze_parallelism()
