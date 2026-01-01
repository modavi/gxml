#!/usr/bin/env python
"""Detailed profile of the 16-panel layout."""
import sys
sys.path.insert(0, 'src/gxml')
sys.path.insert(0, '.')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from tests.test_fixtures.mocks import GXMLTestRenderContext
import cProfile
import pstats
from io import StringIO

xml = """<root>
    <panel thickness="0.25"/>
    <panel width="2.347" thickness="0.25" rotate="-90" attach="0:1"/>
    <panel width="1.522" thickness="0.25" rotate="90" attach="1:1"/>
    <panel width="1.637" thickness="0.25" rotate="90" attach="2:1"/>
    <panel width="4.296" thickness="0.25" rotate="90" attach="3:1"/>
    <panel width="4.854" thickness="0.25" rotate="-225" attach="4:1"/>
    <panel width="6.332" thickness="0.25" rotate="135" attach="5:1"/>
    <panel width="6.439" thickness="0.25" rotate="90" attach="6:1"/>
    <panel width="2.943" thickness="0.25" rotate="-270" attach="7:1"/>
    <panel width="3.721" thickness="0.25" rotate="45" attach="8:1"/>
    <panel width="5.148" thickness="0.25" rotate="45" attach="9:1"/>
    <panel width="7.019" thickness="0.25" rotate="90" attach="10:1"/>
    <panel width="9.576" thickness="0.25" rotate="45" attach="11:1"/>
    <panel width="15.138" thickness="0.25" rotate="-225" attach="12:1"/>
    <panel width="2.123" thickness="0.25" rotate="135" span="9:0.866" attach="13:1"/>
    <panel width="2.433" thickness="0.25" rotate="36.771" span="4:1" attach="14:1"/>
</root>"""

def run_pipeline():
    renderContext = GXMLTestRenderContext()
    root = GXMLParser.parse(xml)
    GXMLLayout.layout(root)
    GXMLRender.render(root, renderContext)
    return root

if __name__ == '__main__':
    # Profile
    pr = cProfile.Profile()
    pr.enable()
    run_pipeline()
    pr.disable()
    
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(40)
    print(s.getvalue())
