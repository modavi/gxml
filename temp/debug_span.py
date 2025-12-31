#!/usr/bin/env python
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from layouts.gxml_construct_layout import GXMLConstructLayout
import numpy as np

GXMLLayout.bind_layout('construct', GXMLConstructLayout())

# Test case from integration tests that passes
print("=== WORKING TEST CASE ===")
xml_working = '''<root>
    <panel/>
    <panel rotate="90" size="2"/>
    <panel rotate="90"/>
    <panel attach-id="0" span-id="2" span-point="auto" rotate="90" attach-point="0.5"/>
</root>'''

root = GXMLParser.parse(xml_working)
GXMLLayout.layout(root)

for i, p in enumerate(root.children):
    corners = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
    world_corners = [p.transform.transform_point(c) for c in corners]
    print(f'Panel {i}: {[f"({wc[0]:.2f},{wc[1]:.2f},{wc[2]:.2f})" for wc in world_corners]}')

print()
print("=== USER'S XML ===")
xml_user = '''<root>
    <panel thickness="0.25" width="3"/>
    <panel thickness="0.25" rotate="90" width="2"/>
    <panel thickness="0.25" rotate="90" width="1.5"/>
    <panel thickness="0.25" rotate="90" span-id="0" span-point="auto"/>
</root>'''

root = GXMLParser.parse(xml_user)
GXMLLayout.layout(root)

for i, p in enumerate(root.children):
    corners = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
    world_corners = [p.transform.transform_point(c) for c in corners]
    print(f'Panel {i}: {[f"({wc[0]:.2f},{wc[1]:.2f},{wc[2]:.2f})" for wc in world_corners]}')
