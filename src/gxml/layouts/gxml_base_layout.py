"""
    Provides a base class for GXML layout processors. This will perform a specific layout scheme on
    the direct children of a GXML element, such as a panel.
"""

class GXMLBaseLayout(object):
    def __init__(self):
        pass
    
    def apply_default_layout_properties(self, element):
        element.appliedDefaultLayoutProperties = True
        
    def conditional_apply_defaults(self, element):
        if not hasattr(element, "appliedDefaultLayoutProperties"):
            self.apply_default_layout_properties(element)
    
    def parse_layout_attributes(self, element, ctx):
        self.conditional_apply_defaults(element)
    
    def measure_element(self, element):
        pass
    
    def pre_layout_element(self, element):
        pass
    
    def layout_element(self, element):
        pass
    
    def post_layout_element(self, element):
        pass