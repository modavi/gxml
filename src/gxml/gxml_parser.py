"""
    Parses an xml document containing gxml into a hierarchy of gxml objects
"""

import xml.etree.ElementTree as ET
from elements.gxml_root import GXMLRoot
from elements.gxml_panel import GXMLPanel
from gxml_types import *
from gxml_layout import GXMLLayout

class GXMLParsingContext:
    """
        Provides context for parsing a gxml element, including information about its position in the hierarchy
        and its relationship to other elements.
    """
    def __init__(self):
        self.childIndex = -1
        self.desc = None
        self.prevDesc = None
        self.prevElem = None
        self.vars = None
        self.elementMap = {}
        self.elements = []
        
    def getAttribute(self, key, defaultVal):
        str = self.desc.get(key) or defaultVal
        if(str == "~") and self.prevDesc != None:
            str = self.prevDesc.get(key) or defaultVal
            self.desc.set(key, str)
        return str
    
    def pushVars(self, vars):
        self.vars = vars
        
    def eval(self, str):
        # Fast path for simple numeric values (most common case)
        # This avoids the overhead of Python's eval() for simple numbers
        try:
            # Try int first (cheaper check)
            if str.isdigit() or (str[0] == '-' and str[1:].isdigit()):
                return int(str)
            # Then try float
            return float(str)
        except (ValueError, IndexError):
            # Fall back to full eval for expressions
            return eval(str, None, self.vars)
    
class GXMLParser(object):
    """
        Parses a gxml element and its children into a hierarchy of gxml objects.
    """

    boundElementTypes = {
        "root": GXMLRoot,
        "panel": GXMLPanel  # Use new intersection solver
    }
    
    def parse(gxml):
        root = ET.fromstring(gxml)
        
        if root:
            ctx = GXMLParsingContext()
            ctx.layoutScheme = GXMLLayoutScheme.Construct
            rootElement = GXMLParser.parse_element(root, ctx)
            rootElement.isVisibleSelf = False
            return rootElement
        else:
            raise Exception("Could not parse gxml.")
        
    def parse_element(desc, ctx):
        # We use the previous description to allow for ~ syntax to refer to attributes in the previous element.
        # Often we want to just re-use the same attribute again and again for multiple elements, and this allows
        # us to refer to those elements.
        ctx.prevDesc = ctx.desc
        ctx.desc = desc
        
        if desc.tag == "vars":
            ctx.pushVars(GXMLParser.parse_vars(desc, ctx))
            return None
        else:
            boundElementType = GXMLParser.get_bound_element_type(desc.tag)
            
            if boundElementType is None:
                raise Exception(f"Could not find bound element type of type '{desc.tag}'")
            
            element = boundElementType()
            
            if element is None:
                raise Exception(f"Could not create element of type '{desc.tag}'")
            
            element.parse(ctx)
            
            if element.id in ctx.elementMap:
                raise ValueError(f'Duplicate panel id found {element.id}')
            
            layoutEngine = GXMLLayout.get_bound_layout_processor(ctx.layoutScheme)
            
            if layoutEngine is None:
                raise ValueError(f'Could not find bound layout processor for layout scheme {ctx.layoutScheme}')
            
            layoutEngine.parse_layout_attributes(element, ctx)
            
            # Just validate the child layout scheme here if it's set. If we don't do this here, it will only raise an exception if
            # this element has children which is kind of weird.
            if element.childLayoutScheme is not None:
                validatedLayoutEngine = GXMLLayout.get_bound_layout_processor(element.childLayoutScheme)
                
                if validatedLayoutEngine is None:
                    raise ValueError(f'Could not find bound layout processor for child layout scheme {element.childLayoutScheme}')
            
            ctx.elementMap[element.id] = element
            ctx.elements.append(element)
            
            childCtx = GXMLParsingContext()
            
            idx = 0
            for childDesc in desc:
                childCtx.childIndex = idx
                childCtx.prevElem = childCtx.elements[idx-1] if idx > 0 else None
                childCtx.layoutScheme = element.childLayoutScheme or ctx.layoutScheme
                
                childElement = GXMLParser.parse_element(childDesc, childCtx)
                
                if childElement:
                    idx += 1
                    element.add_child(childElement)
            
            return element
    
    def get_bound_element_type(elementType):
        if elementType in GXMLParser.boundElementTypes:
            return GXMLParser.boundElementTypes[elementType]
        
        return None
        
    def parse_vars(varsRoot, ctx):
        variables = {}
        if varsRoot:
            for elem in varsRoot:
                variables[elem.tag] = ctx.eval(elem.text)
        return variables