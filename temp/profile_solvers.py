#!/usr/bin/env python
"""Profile the solver pipeline to identify hotspots."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml')

import cProfile
import pstats
from io import StringIO

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from tests.test_fixtures.mocks import GXMLTestRenderContext

# Complex 75-panel scene with lots of intersections
def generate_complex_xml():
    """Generate a complex scene with many panels and intersections."""
    panels = []
    # Create a grid-like structure with intersecting panels
    for i in range(15):
        # Horizontal panels
        panels.append(f'<panel width="10" thickness="0.25" x="{i*2}" z="0"/>')
        panels.append(f'<panel width="10" thickness="0.25" x="{i*2}" z="5"/>')
        panels.append(f'<panel width="10" thickness="0.25" x="{i*2}" z="10"/>')
        # Vertical connecting panels  
        panels.append(f'<panel width="5" thickness="0.25" rotate="90" x="{i*2}" z="0"/>')
        panels.append(f'<panel width="5" thickness="0.25" rotate="90" x="{i*2}" z="5"/>')
    
    return f'<root>\n{"".join(panels)}\n</root>'

# Simpler chain of attached panels (like the 7-panel benchmark)
def generate_chain_xml(n=75):
    """Generate a chain of attached panels."""
    panels = ['<panel thickness="0.25"/>']
    for i in range(1, n):
        angle = (i * 37) % 360 - 180  # Varied angles
        panels.append(f'<panel width="{2 + (i % 3)}" thickness="0.25" rotate="{angle}" attach="{i-1}:1"/>')
    return f'<root>\n{"".join(panels)}\n</root>'

def run_pipeline(xml):
    """Run the full pipeline: parse -> layout -> render."""
    root = GXMLParser.parse(xml)
    GXMLLayout.layout(root)
    ctx = GXMLTestRenderContext()
    GXMLRender.render(root, ctx)
    return len(ctx.polys)

def profile_pipeline(xml, label):
    """Profile the pipeline and print results."""
    print(f"\n{'='*60}")
    print(f"Profiling: {label}")
    print(f"{'='*60}")
    
    # Warm up
    run_pipeline(xml)
    
    # Profile
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(5):  # Run 5 times for better stats
        polys = run_pipeline(xml)
    
    pr.disable()
    
    # Print results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(40)  # Top 40 functions
    print(s.getvalue())
    
    print(f"\nGenerated {polys} polygons per run")
    
    # Also print by internal time (where time is actually spent)
    print(f"\n--- Top 20 by internal time (tottime) ---")
    s2 = StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(20)
    print(s2.getvalue())

if __name__ == '__main__':
    # Test 1: Chain of 75 attached panels
    chain_xml = generate_chain_xml(75)
    profile_pipeline(chain_xml, "75-panel chain (attached panels)")
    
    # Test 2: Grid layout with intersections
    # grid_xml = generate_complex_xml()
    # profile_pipeline(grid_xml, "Grid layout with intersections")
