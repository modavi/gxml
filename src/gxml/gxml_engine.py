"""Supplies the main entry point into rendering GXML content into a Houdini node."""

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from render_engines.houdini_render_context import HoudiniRenderContext
            
def run(node, gxml_str):
    """Main entry point for running GXML rendering.
    This function parses the provided GXML string, applies layout processing,
    and renders the result using the render context.

    Args:
        node (_type_): The Houdini node that this GXML will be rendered into.
        gxml_str (str): The GXML string to be rendered.
    """
    
    panel = GXMLParser.parse(gxml_str)
    GXMLLayout.layout(panel)
    
    houRenderCtx = HoudiniRenderContext(node)
    GXMLRender.render(panel, houRenderCtx)