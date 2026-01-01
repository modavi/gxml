"""Profile GXML pipeline stages in detail."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml-web/src')

import time
from contextlib import contextmanager
from collections import defaultdict

# Timing infrastructure
timings = {}
call_counts = defaultdict(int)

@contextmanager
def timed(name):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if name not in timings:
        timings[name] = []
    timings[name].append(elapsed)
    call_counts[name] += 1

# Patch the key functions to add timing
from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender

# Import solvers to patch them
from elements.solvers.gxml_intersection_solver import IntersectionSolver
from elements.solvers.gxml_face_solver import FaceSolver
from elements.solvers.gxml_geometry_builder import GeometryBuilder
from layouts.gxml_construct_layout import GXMLConstructLayout

# Store original methods
_original_parse = GXMLParser.parse
_original_layout = GXMLLayout.layout
_original_render = GXMLRender.render
_original_intersection_solve = IntersectionSolver.solve
_original_face_solve = FaceSolver.solve
_original_geometry_build_all = GeometryBuilder.build_all
_original_pre_layout = GXMLConstructLayout.pre_layout_element
_original_build_local = GXMLConstructLayout.build_local_transform
_original_build_world = GXMLConstructLayout.build_world_transform

# Patch with timing
@staticmethod
def timed_parse(xml_str):
    with timed("1. Parse"):
        return _original_parse(xml_str)
GXMLParser.parse = timed_parse

@staticmethod  
def timed_layout(root):
    with timed("2. Layout (total)"):
        # Measure pass
        with timed("2.1 Measure pass"):
            GXMLLayout.measure_pass(root)
        # Pre-layout pass
        with timed("2.2 Pre-layout pass"):
            GXMLLayout.pre_layout_pass(root)
        # Layout pass
        with timed("2.3 Layout pass"):
            GXMLLayout.layout_pass(root)
        # Post-layout pass
        with timed("2.4 Post-layout pass"):
            GXMLLayout.post_layout_pass(root)
GXMLLayout.layout = timed_layout

@staticmethod
def timed_render(root, ctx):
    with timed("3. Render"):
        return _original_render(root, ctx)
GXMLRender.render = timed_render

@staticmethod
def timed_intersection_solve(panels):
    with timed(f"2a. IntersectionSolver"):
        return _original_intersection_solve(panels)
IntersectionSolver.solve = timed_intersection_solve

@staticmethod
def timed_face_solve(solution):
    with timed("2b. FaceSolver"):
        return _original_face_solve(solution)
FaceSolver.solve = timed_face_solve

@staticmethod
def timed_geometry_build_all(panel_faces, intersection_solution):
    with timed("2c. GeometryBuilder"):
        return _original_geometry_build_all(panel_faces, intersection_solution)
GeometryBuilder.build_all = timed_geometry_build_all

def timed_pre_layout(self, element):
    with timed("  pre_layout_element"):
        return _original_pre_layout(self, element)
GXMLConstructLayout.pre_layout_element = timed_pre_layout

def timed_build_local(self, element, size0, size1):
    with timed("    build_local_transform"):
        return _original_build_local(self, element, size0, size1)
GXMLConstructLayout.build_local_transform = timed_build_local

def timed_build_world(self, element, height):
    with timed("    build_world_transform"):
        return _original_build_world(self, element, height)
GXMLConstructLayout.build_world_transform = timed_build_world

# Now import render engine
from gxml_web.json_render_engine import JSONRenderEngine

# Test XML with increasing complexity
test_cases = [
    ("1 panel", '<root><panel thickness="0.25"/></root>'),
    ("4 panels (grid)", '''<root>
        <panel thickness="0.1"/>
        <panel thickness="0.1" y="1"/>
        <panel thickness="0.1" rotation="0 0 90"/>
        <panel thickness="0.1" rotation="0 0 90" x="1"/>
    </root>'''),
    ("7 panels", '''<root>
        <panel id="floor" width="4" height="4" rotation="90 0 0" y="-0.05" thickness="0.1"/>
        <panel id="left" width="4" height="2" x="-0.05" rotation="0 -90 0" thickness="0.1"/>
        <panel id="right" width="4" height="2" x="4.05" rotation="0 90 0" thickness="0.1"/>
        <panel id="back" width="4" height="2" y="0" z="-0.05" thickness="0.1"/>
        <panel id="front" width="4" height="2" y="0" z="4.05" rotation="0 180 0" thickness="0.1"/>
        <panel id="shelf1" width="4" height="1" rotation="90 0 0" y="0.75" z="0" thickness="0.1"/>
        <panel id="shelf2" width="4" height="1" rotation="90 0 0" y="1.5" z="0" thickness="0.1"/>
    </root>'''),
    ("16 panels", '''<root>
        <panel id="p1" thickness="0.1"/>
        <panel id="p2" thickness="0.1" x="1.2"/>
        <panel id="p3" thickness="0.1" x="2.4"/>
        <panel id="p4" thickness="0.1" x="3.6"/>
        <panel id="p5" thickness="0.1" y="1.2"/>
        <panel id="p6" thickness="0.1" x="1.2" y="1.2"/>
        <panel id="p7" thickness="0.1" x="2.4" y="1.2"/>
        <panel id="p8" thickness="0.1" x="3.6" y="1.2"/>
        <panel id="c1" thickness="0.1" rotation="0 0 90"/>
        <panel id="c2" thickness="0.1" rotation="0 0 90" x="1.2"/>
        <panel id="c3" thickness="0.1" rotation="0 0 90" x="2.4"/>
        <panel id="c4" thickness="0.1" rotation="0 0 90" x="3.6"/>
        <panel id="c5" thickness="0.1" rotation="0 0 90" y="1.2"/>
        <panel id="c6" thickness="0.1" rotation="0 0 90" x="1.2" y="1.2"/>
        <panel id="c7" thickness="0.1" rotation="0 0 90" x="2.4" y="1.2"/>
        <panel id="c8" thickness="0.1" rotation="0 0 90" x="3.6" y="1.2"/>
    </root>'''),
]

print("=" * 80)
print("GXML Pipeline Profiling")
print("=" * 80)

for name, xml in test_cases:
    timings.clear()
    call_counts.clear()
    
    # Run multiple times for more stable measurements
    iterations = 5
    for _ in range(iterations):
        root = GXMLParser.parse(xml)
        GXMLLayout.layout(root)
        
        render_engine = JSONRenderEngine()
        GXMLRender.render(root, render_engine)
    
    print(f"\n{name} ({iterations} iterations, averaged):")
    print("-" * 70)
    
    # Sort by key to get consistent ordering
    for key in sorted(timings.keys()):
        avg_ms = sum(timings[key]) / len(timings[key]) * 1000
        total_ms = sum(timings[key]) * 1000
        calls = call_counts[key]
        print(f"  {key:35} {avg_ms:8.2f}ms avg  ({calls:4d} calls)")
    
    # Calculate total time  
    total_time = sum(sum(v) for v in timings.values()) / iterations * 1000
    print(f"  {'TOTAL':35} {total_time:8.2f}ms")

print("\n" + "=" * 80)
print("Analysis")
print("=" * 80)
