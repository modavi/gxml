"""
Profile individual solver times for CPU vs GPU (Taichi).
"""
import sys
import time
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml/tests')

import taichi as ti
# Use CPU backend for now - Metal has RHI shader compilation issues with complex kernels
ti.init(arch=ti.cpu)

from gxml_layout import GXMLLayout
from gxml_parser import GXMLParser
from gxml_render import GXMLRender
from test_fixtures.mocks import GXMLTestRenderContext
from elements.solvers.taichi import set_solver_backend

# CPU solvers
from elements.solvers.gxml_intersection_solver import IntersectionSolver
from elements.solvers.gxml_face_solver import FaceSolver
from elements.solvers.gxml_geometry_builder import GeometryBuilder

# Taichi solvers
from elements.solvers.taichi.taichi_intersection_solver import TaichiIntersectionSolver
from elements.solvers.taichi.taichi_face_solver import TaichiFaceSolver
from elements.solvers.taichi.taichi_geometry_builder import TaichiGeometryBuilder

# 75-panel test XML
XML_75_PANELS = """<root>
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
</root>"""

def get_panels():
    """Parse XML and layout to get panels."""
    panel = GXMLParser.parse(XML_75_PANELS)
    GXMLLayout.layout(panel)
    
    # Collect all panels
    from elements.gxml_panel import GXMLPanel
    panels = [e for e in panel.iterate() if isinstance(e, GXMLPanel)]
    return panels

def profile_solver(name, solver_cls, input_data, iterations=10):
    """Profile a single solver."""
    # Warmup
    for _ in range(3):
        solver_cls.solve(input_data)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = solver_cls.solve(input_data)
        times.append((time.perf_counter() - start) * 1000)
    
    avg = sum(times) / len(times)
    print(f"  {name}: {avg:.2f}ms avg (min={min(times):.2f}, max={max(times):.2f})")
    return avg, result

def main():
    print("GXML Solver Profile (CPU vs Taichi Metal)")
    print("=" * 60)
    print("Test: 75 panels")
    print()
    
    panels = get_panels()
    print(f"Parsed {len(panels)} panels")
    print()
    
    # Profile CPU solvers
    print("CPU Solvers:")
    print("-" * 40)
    cpu_int_time, int_solution = profile_solver("IntersectionSolver", IntersectionSolver, panels)
    cpu_face_time, face_solution = profile_solver("FaceSolver", FaceSolver, int_solution)
    # GeometryBuilder needs segmented panels - skip for now
    cpu_total = cpu_int_time + cpu_face_time
    print(f"  Total (Int + Face): {cpu_total:.2f}ms")
    print()
    
    # Profile Taichi solvers
    print("Taichi GPU Solvers (Metal):")
    print("-" * 40)
    gpu_int_time, int_solution = profile_solver("TaichiIntersectionSolver", TaichiIntersectionSolver, panels)
    gpu_face_time, face_solution = profile_solver("TaichiFaceSolver", TaichiFaceSolver, int_solution)
    gpu_total = gpu_int_time + gpu_face_time
    print(f"  Total (Int + Face): {gpu_total:.2f}ms")
    print()
    
    # Summary
    print("=" * 60)
    print("SPEEDUP SUMMARY")
    print("=" * 60)
    print(f"IntersectionSolver: CPU={cpu_int_time:.2f}ms, GPU={gpu_int_time:.2f}ms -> {cpu_int_time/gpu_int_time:.2f}x")
    print(f"FaceSolver:         CPU={cpu_face_time:.2f}ms, GPU={gpu_face_time:.2f}ms -> {cpu_face_time/gpu_face_time:.2f}x")
    print(f"Total:              CPU={cpu_total:.2f}ms, GPU={gpu_total:.2f}ms -> {cpu_total/gpu_total:.2f}x")

if __name__ == '__main__':
    main()
