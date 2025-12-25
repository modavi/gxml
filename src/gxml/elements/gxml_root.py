"""
A custom element type to hold the root element for a document
"""

from elements.gxml_base_element import GXMLLayoutElement
from gxml_types import GXMLLayoutScheme

class GXMLRoot(GXMLLayoutElement):
    def __init__(self):
        super().__init__()
        self.childLayoutScheme = GXMLLayoutScheme.Construct