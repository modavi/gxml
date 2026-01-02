#!/usr/bin/env python
"""Count how many times numpy.array is called during a layout operation."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml-web/src')

import numpy as np

count = 0
orig_array = np.array

def counting_array(*args, **kwargs):
    global count
    count += 1
    return orig_array(*args, **kwargs)

np.array = counting_array

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from tests.test_fixtures.mocks import GXMLTestRenderContext
from gxml_render import GXMLRender

xml = '''<root>
    <panel thickness="0.25"/>
    <panel width="2.55" thickness="0.25" rotate="90" attach="0:1"/>
    <panel width="2.76" thickness="0.25" rotate="-135" attach="1:1"/>
    <panel width="2.873" thickness="0.25" rotate="-45" attach="2:1"/>
    <panel width="2.726" thickness="0.25" rotate="-90" attach="3:1"/>
    <panel width="4.716" thickness="0.25" rotate="315" attach="4:1"/>
    <panel width="6.608" thickness="0.25" rotate="-45" attach="5:1"/>
</root>'''

panel = GXMLParser.parse(xml)
GXMLLayout.layout(panel)

render = GXMLTestRenderContext()
GXMLRender.render(panel, render)
print(f'numpy.array called {count} times for 7 panels')
