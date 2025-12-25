"""
    GXMLLayout is responsible for managing the layout of GXML elements.
    It processes the layout in multiple passes: measure, pre-layout,
    layout, and post-layout.
    Each pass allows for different aspects of the layout to be handled, such as measuring sizes,
    applying transformations, and finalizing the layout of child elements.
"""

from gxml_types import *
from elements.gxml_base_element import *
from layouts.gxml_construct_layout import GXMLConstructLayout
from layouts.gxml_stack_layout import GXMLStackLayout
from layouts.gxml_fixed_layout import GXMLFixedLayout

class GXMLLayout(object):
    """
        Handles the layout of GXML elements. Runs the mult-pass layout algorithm
        to measure, pre-layout, layout, and post-layout elements in the hierarchy.
    """
    
    boundLayouts = {
        GXMLLayoutScheme.Construct: GXMLConstructLayout(),
        GXMLLayoutScheme.Stack: GXMLStackLayout(),
        GXMLLayoutScheme.Fixed: GXMLFixedLayout()
    }
    
    def layout(rootElement):
        """
            Layouts the given element and its children using the multi pass
            layout algorithm. This function will call the measure, pre-layout,
            layout, and post-layout passes to layout all the elements in the hierarchy.

            Args:
                element (GXMLElement): The element to layout.
        """
        
        GXMLLayout.measure_pass(rootElement)
        GXMLLayout.pre_layout_pass(rootElement)
        GXMLLayout.layout_pass(rootElement)
        GXMLLayout.post_layout_pass(rootElement)
        
    def measure_pass(rootElement):
        """
            Measures the children of the given element, allowing each of them to calculate their ideal size.
            Though this may not be the actual size they end up at. It's just a chance for them each to calculate
            their ideal size based on their content, and any other factors that may affect their size.
            
            This lays out elements in bottom up order, meaning children are measured first before their parent.
            This is important for elements that may depend on their children's sizes to determine their own size.

            Args:
                rootElement (GXMLElement): The element to layout.
        """
        
        for element in (e for e in rootElement.iterate(TraversalDirection.BottomUp) if e.parent):
            layoutProcessor = GXMLLayout.get_bound_layout_processor(element.parent.childLayoutScheme)
            
            # This is to support creating panels programmatically (without actually being parsed from XML)
            # This is the earliest point in the layout process, so calling this here will ensure that this
            # layout engine applies all its defaults to the child element before it's laid out.
            layoutProcessor.conditional_apply_defaults(element)
            layoutProcessor.measure_element(element)

    def pre_layout_pass(rootElement):
        """
            Runs a pre-layout pass on the given element and its children. It's up to the layout processor to decide what this means.
            This is typically used to prepare the element for layout, such as calculating independent sizes, positions, and other properties
            that don't rely on the results of other layout passes.

            Args:
                rootElement (GXMLElement): The element to layout.
        """
        
        for element in (e for e in rootElement.iterate(TraversalDirection.TopDownBreadthFirst) if e.parent):
            layoutProcessor = GXMLLayout.get_bound_layout_processor(element.parent.childLayoutScheme)
            layoutProcessor.pre_layout_element(element)
            element.on_pre_layout()
    
    def layout_pass(rootElement):
        """
            Runs the layout pass on the given element and its children. This is where the actual layout calculations are performed.
            This is typically used to calculate the final positions and sizes of the elements based on the results of the pre-layout and
            measure passes. In here we can resolve layout dependencies between elements that should have already had their independent
            elements calculated in the pre-layout pass.

            Args:
                rootElement (GXMLElement): The element to layout.
        """
        
        for element in (e for e in rootElement.iterate(TraversalDirection.TopDownBreadthFirst) if e.parent):
            layoutProcessor = GXMLLayout.get_bound_layout_processor(element.parent.childLayoutScheme)
            layoutProcessor.layout_element(element)
            element.on_layout()

    def post_layout_pass(rootElement):
        """
            Runs the post-layout pass on the given element and its children. This is where any final adjustments to the layout are made.
            Can be used to run any final adjustments to the layouts that a processor may want to do after an element has been laid out such as
            maybe applying a post process to the elements position or scale.
        """
        
        for element in (e for e in rootElement.iterate(TraversalDirection.TopDownBreadthFirst) if e.parent):
            layoutProcessor = GXMLLayout.get_bound_layout_processor(element.parent.childLayoutScheme)
            layoutProcessor.post_layout_element(element)
            element.on_post_layout()
            
    @staticmethod
    def bind_layout(layoutScheme, layoutProcessor):
        GXMLLayout.boundLayouts[layoutScheme] = layoutProcessor
        
    @staticmethod
    def get_bound_layout_processor(layoutSchema):
        if layoutSchema in GXMLLayout.boundLayouts:
            return GXMLLayout.boundLayouts[layoutSchema]
        
        return None