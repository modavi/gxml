"""Debug actual panel endpoint values."""
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout

xml = '''<root>
    <panel thickness="0.1"/>
    <panel thickness="0.1" rotate="90"/>
    <panel thickness="0.1" rotate="90"/>
    <panel thickness="0.1" rotate="90"/>
    <panel thickness="0.1" span-id="1" span-point="1.0"/>
    <panel thickness="0.1" attach-id="0" span-id="2" span-point="1.0"/>
</root>'''

root = GXMLParser.parse(xml)
GXMLLayout.layout(root)

panels = [e for e in root.children if hasattr(e, 'thickness')]
print('Actual panel endpoints:')
for i, p in enumerate(panels):
    start = p.transform_point([0,0,0])
    end = p.transform_point([1,0,0])
    print(f'Panel {i}: start={start}, end={end}')
    print(f'  types: start={type(start).__name__}, end={type(end).__name__}')
    
    min_z = min(start[2], end[2])
    max_z = max(start[2], end[2])
    cell_size = 20.0
    min_cz = int(min_z // cell_size)
    max_cz = int(max_z // cell_size)
    print(f'  min_z={min_z}, max_z={max_z}')
    print(f'  min_z // 20 = {min_z // cell_size}, max_z // 20 = {max_z // cell_size}')
    print(f'  min_cz={min_cz}, max_cz={max_cz}')
    print()
