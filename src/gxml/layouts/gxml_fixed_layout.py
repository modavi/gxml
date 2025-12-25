"""
    A layout that is used to layout each child element connected to the previous element, allowing for rotating them, and attaching
    elements to other elements to construct a larger shape of individual panels.
"""

from gxml_types import *
from layouts.gxml_base_layout import GXMLBaseLayout

class GXMLFixedLayout(GXMLBaseLayout):
    layoutScheme = "fixed"
    
    def parse_layout_attributes(self, element, ctx):
        super().parse_layout_attributes(element, ctx)
        
        element.position = (0,0,0)
        element.size = (1,1,1)
        
    def measure_element(self, element):
        super().measure_element(element)
        
    def layout_element(self, element):
        super().layout_element(element)
        
        element.transform.inheritScale = False
        element.transform.apply_local_transformations((0.0,0.0,0), (0,0,0), (0.5,0.5,1))
        element.transform.recalculate(element.parent.transform)