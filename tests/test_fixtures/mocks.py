"""Mock classes for GXML testing."""

from enum import Enum
from elements.gxml_panel import GXMLPanel
from mathutils.quad_interpolator import QuadInterpolator
import mathutils.gxml_math as GXMLMath
from render_engines.base_render_context import BaseRenderContext
from layouts.gxml_base_layout import GXMLBaseLayout


class GXMLMockParsingContext:
    """Mock parsing context for unit testing layout attribute parsing.
    
    Simulates the parsing context that would be passed to layout.parse_layout_attributes()
    during real XML parsing. Allows tests to set up element maps and attributes without
    needing actual XML parsing.
    
    Attributes:
        elementMap: Dict mapping element IDs to element instances
        prevElem: The previous sibling element (for implicit attach-id)
        _attributes: Internal dict storing attributes set via setAttribute()
    
    Examples:
        Test attribute parsing:
            >>> ctx = GXMLMockParsingContext()
            >>> ctx.elementMap = {"panel1": p1}
            >>> ctx.setAttribute("attach-id", "panel1")
            >>> layout.parse_layout_attributes(element, ctx)
    """
    
    def __init__(self):
        self.elementMap = {}
        self.prevElem = None
        self._attributes = {}
        self._variables = {}
    
    def setAttribute(self, name: str, value):
        """Set an attribute value for testing.
        
        Args:
            name: Attribute name (e.g., "attach-id", "attach-point")
            value: Attribute value as string
        """
        self._attributes[name] = value
    
    def getAttribute(self, name: str, default=None):
        """Get an attribute value, returning default if not set.
        
        Args:
            name: Attribute name to retrieve
            default: Value to return if attribute not set
            
        Returns:
            The attribute value or default
        """
        return self._attributes.get(name, default)
    
    def hasAttribute(self, name: str) -> bool:
        """Check if an attribute is set.
        
        Args:
            name: Attribute name to check
            
        Returns:
            True if the attribute is set, False otherwise
        """
        return name in self._attributes
    
    def clearAttributes(self):
        """Clear all attributes. Useful for resetting between test cases."""
        self._attributes.clear()
    
    def eval(self, expr):
        """Evaluate a simple expression (mock implementation).
        
        For testing purposes, this just tries to convert to float/int,
        or returns the string if it can't be parsed.
        
        Args:
            expr: Expression string to evaluate
            
        Returns:
            Evaluated result (number or string)
        """
        if expr in self._variables:
            return self._variables[expr]
        try:
            if '.' in str(expr):
                return float(expr)
            return int(expr)
        except (ValueError, TypeError):
            return expr
    
    def setVariable(self, name: str, value):
        """Set a variable for expression evaluation.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self._variables[name] = value


class GXMLMockPanel(GXMLPanel):
    """Simplified panel mock for unit and integration testing.
    
    A real GXMLPanel subclass that simplifies panel creation for tests. Provides a
    streamlined constructor for defining panels by position without requiring XML parsing
    or the full layout system. Includes layout tracking properties for verifying layout
    pass execution order in integration tests.
    
    Args:
        panel_id: Unique identifier for the panel
        start_pos: [x, y, z] coordinates of the panel start (bottom corner). If None,
            no geometry is created (useful for layout tracking tests)
        end_pos: [x, y, z] coordinates of the panel end (bottom corner). If None,
            no geometry is created
        thickness: Panel thickness in world units (default 0.1)
        sub_id: Optional sub-identifier for the panel
        height: Panel height in Y direction (default 1.0)
    
    Examples:
        Create a horizontal panel for intersection tests:
            >>> panel = GXMLMockPanel("p1", [-1, 0, 0], [1, 0, 0], thickness=0.1)
        
        Create a panel without geometry for layout tests:
            >>> panel = GXMLMockPanel(panel_id="p1")
        
        Apply rotation using the transformation API:
            >>> panel = GXMLMockPanel("p1", [-1, 0, 0], [1, 0, 0], thickness=0.1)
            >>> panel.setRotation([0, 45, 0])
    
    Attributes:
        measureOrder: Execution order in measure pass (set by GXMLMockLayout)
        preLayoutOrder: Execution order in pre-layout pass
        layoutOrder: Execution order in layout pass
        postLayoutOrder: Execution order in post-layout pass
        ops: List of LayoutPass values tracking which passes executed
        childLayoutScheme: Layout scheme identifier for testing
    """
    
    def __init__(self, panel_id: str = "", start_pos: list = None, end_pos: list = None, thickness: float = 0.1, sub_id: str = "", height: float = 1.0):
        super().__init__()
        self.id = panel_id
        self.subId = sub_id
        self.thickness = thickness
        
        # Layout tracking properties
        self.measureOrder = 0
        self.preLayoutOrder = 0
        self.layoutOrder = 0
        self.postLayoutOrder = 0
        self.ops = []
        self.childLayoutScheme = "test"
        
        # Create geometry if positions provided
        if start_pos is not None and end_pos is not None:
            # Create unit square in local space (architecture: quad is local, transform positions it)
            p0 = [0, 0, 0]  # bottom-left
            p1 = [1, 0, 0]  # bottom-right
            p2 = [1, 1, 0]  # top-right
            p3 = [0, 1, 0]  # top-left
            self.quad_interpolator = QuadInterpolator(p0, p1, p2, p3)
            
            # Calculate world space transformation
            import numpy as np
            direction = np.array(end_pos) - np.array(start_pos)
            length = np.linalg.norm(direction)
            
            if length > 1e-10:  # Avoid division by zero
                direction = direction / length
                
                # Calculate rotation: align local X-axis with direction vector
                # In Y-up right-handed system, Y-rotation transforms [1,0,0] -> [cos(theta),0,-sin(theta)]
                # To align with direction [dx, 0, dz], we need: cos(theta)=dx, -sin(theta)=dz
                # Therefore: theta = atan2(-dz, dx)
                angle_y = np.degrees(np.arctan2(-direction[2], direction[0]))
                
                # For now, only support horizontal panels (Y component of direction = 0)
                # Full 3D rotation would require calculating pitch and roll as well
                rotation = [0, angle_y, 0]
            else:
                length = 1.0
                rotation = [0, 0, 0]
            
            # Position at start point, scale to desired dimensions
            translation = list(start_pos)
            scale = [length, height, 1.0]
            
            # Apply transformation directly
            self.transform.apply_local_transformations(tuple(translation), tuple(rotation), tuple(scale))
            self.recalculate_transform()


class GXMLTestRenderContext(BaseRenderContext):
    """Mock render context for tracking render operations in tests.
    
    Extends BaseRenderContext to capture rendering activity without creating actual
    geometry. Tracks the number of times each element is rendered and stores polygon
    creation calls for verification in integration tests.
    
    Attributes:
        polys: List of (id, points) tuples for each polygon created
    
    Examples:
        Verify rendering in an integration test:
            >>> context = GXMLTestRenderContext()
            >>> GXMLRender.render(root, context)
            >>> assert len(context.polys) == 2  # Two polygons created
    """
    
    def __init__(self):
        super().__init__()
        self.polys = []
    
    def render_hierarchy(self, element):
        for child in element.iterate():
            if not hasattr(child, 'renderCount'):
                child.renderCount = 0

        super().render_hierarchy(element)

    def pre_render(self, element):
        super().pre_render(element)
        
        if not hasattr(element, 'renderCount'):
            element.renderCount = 0
            
        element.renderCount += 1
        
    def create_poly(self, id, points, geoKey=None):
        self.polys.append((id, points))


class LayoutPass(Enum):
    """Layout pass identifiers for tracking execution order.
    
    Used by GXMLMockLayout to record which layout passes execute and in what order.
    These values are appended to element.ops lists during layout operations.
    
    Values:
        Measure: Measurement pass (size calculation)
        PreLayout: Pre-layout pass (preparation before positioning)
        Layout: Main layout pass (element positioning)
        PostLayout: Post-layout pass (cleanup and finalization)
    """
    Measure = 1
    PreLayout = 2
    Layout = 3
    PostLayout = 4


class GXMLMockLayout(GXMLBaseLayout):
    """Mock layout that tracks layout pass execution using runtime class injection.
    
    Uses the spy pattern to intercept layout method calls by dynamically injecting
    itself as a base class of the target layout. Tracks execution order and pass
    sequence for each element without modifying production code.
    
    Integration tests can use this to verify:
    - Layout pass execution order (measure -> pre-layout -> layout -> post-layout)
    - Which elements were processed during layout
    - Relative execution order across the element hierarchy
    
    Warning:
        Uses runtime class modification. Always call restore_all() in test tearDown
        to clean up modified classes.
    
    Examples:
        Track layout operations in an integration test:
            >>> layout = GXMLMockLayout.create(my_layout)
            >>> my_layout.layoutElement(root)
            >>> assert root.ops == [LayoutPass.Measure, LayoutPass.Layout]
            >>> GXMLMockLayout.restore_all()  # Cleanup
    
    Attributes:
        layoutOps: List of LayoutPass values for this layout instance
        measureOrder: Counter for measure pass execution order
        preLayoutOrder: Counter for pre-layout pass execution order
        layoutOrder: Counter for layout pass execution order
        postLayoutOrder: Counter for post-layout pass execution order
    """
    
    _modifiedClasses = {}  # Track classes we've modified so we can restore them
    
    @classmethod
    def create(cls, proxyLayout):
        """Inject tracking into a layout instance via runtime class modification.
        
        Replaces the layouts base class with GXMLMockLayout, enabling method
        interception. Initializes tracking properties on the layout instance.
        
        Args:
            proxyLayout: Layout instance to instrument with tracking
            
        Returns:
            The same layout instance, now with tracking enabled
        """
        proxyLayout.layoutOps = []
        
        proxyLayout.measureOrder = 0
        proxyLayout.preLayoutOrder = 0
        proxyLayout.layoutOrder = 0
        proxyLayout.postLayoutOrder = 0
        
        # Save original base classes before modification
        layoutClass = proxyLayout.__class__
        if layoutClass not in cls._modifiedClasses:
            cls._modifiedClasses[layoutClass] = layoutClass.__bases__
        
        # Inject ourselves as the base class for this layout, so that we can add
        # tracking to each of the calls for test purposes
        proxyLayout.__class__.__bases__ = (cls,)
        
        return proxyLayout
    
    @classmethod
    def restore_all(cls):
        """Restore all modified layout classes to their original state.
        
        Reverts runtime base class modifications made by create(). Should always
        be called in test tearDown to prevent state leakage between tests.
        """
        for layoutClass, originalBases in cls._modifiedClasses.items():
            layoutClass.__bases__ = originalBases
        cls._modifiedClasses.clear()
    
    def apply_default_layout_properties(self, element):
        super().apply_default_layout_properties(element)
        
        element.ops = []
        element.layoutEngineUsed = None
    
    def measure_element(self, element):
        super().measure_element(element)
        
        element.ops.append(LayoutPass.Measure)
        element.measureOrder = self.measureOrder
        self.measureOrder += 1
        self.layoutOps.append(LayoutPass.Measure)
        
    def pre_layout_element(self, element):
        super().pre_layout_element(element)
        
        element.ops.append(LayoutPass.PreLayout)
        element.preLayoutOrder = self.preLayoutOrder
        self.preLayoutOrder += 1
        self.layoutOps.append(LayoutPass.PreLayout)
        
    def layout_element(self, element):
        super().layout_element(element)
        
        element.layoutEngineUsed = self
        element.ops.append(LayoutPass.Layout)
        element.layoutOrder = self.layoutOrder
        self.layoutOrder += 1
        self.layoutOps.append(LayoutPass.Layout)
        
    def post_layout_element(self, element):
        super().post_layout_element(element)

        element.ops.append(LayoutPass.PostLayout)
        element.postLayoutOrder = self.postLayoutOrder
        self.postLayoutOrder += 1
        self.layoutOps.append(LayoutPass.PostLayout)
