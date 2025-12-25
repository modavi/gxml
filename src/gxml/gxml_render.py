"""
    Renders a gxml structure, generating the appropriate geometry for the entire hierarchy
"""

from gxml_types import *

class GXMLRender(object):
    """
    Renders a GXML structure.
    """
    
    def render(element, renderContext):
        renderContext.render_hierarchy(element)
        renderContext.combine_all_geo()