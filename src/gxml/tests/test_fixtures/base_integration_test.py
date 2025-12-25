"""Base test class for integration tests that parse XML and use the full layout system."""

import unittest
from gxml_layout import GXMLLayout
from gxml_parser import GXMLParser
from gxml_render import GXMLRender
from .mocks import GXMLTestRenderContext, GXMLMockLayout
from .xml_integration_test_tools import XMLIntegrationTestValidator


class BaseIntegrationTest(unittest.TestCase):
    """Base class for XML-based integration tests.
    
    Provides infrastructure for integration tests that parse XML markup and
    exercise the full layout and render pipeline. Automatically sets up render
    context and cleans up any layout spy modifications.
    
    Use this base class when testing:
    - Complete XML parsing, layout, and rendering flow
    - Element behavior in the full layout pipeline
    - Layout pass execution order with GXMLMockLayout tracking
    - Integration between parser, layout system, and renderer
    
    For unit tests of individual components (solvers, geometry builders, etc.),
    inherit directly from unittest.TestCase instead.
    
    Attributes:
        renderContext: GXMLTestRenderContext instance for tracking render calls
    
    Examples:
        Create an integration test for panel layout:
            >>> class PanelLayoutTests(BaseIntegrationTest):
            ...     def testHorizontalLayout(self):
            ...         root = self.parsePanel('<panel id="p1" />')
            ...         assert root.id == "p1"
    """
    
    def setUp(self):
        self.renderContext = GXMLTestRenderContext()
        
    def tearDown(self):
        self.renderContext = None
        # Automatically restore any modified layout classes
        GXMLMockLayout.restore_all()
    
    def parsePanel(self, xml):
        """Parse, layout, and render XML markup through the full pipeline.
        
        Convenience helper that runs XML through all processing stages: parsing,
        layout calculation, and rendering. The render context captures polygon
        creation and other render operations for verification.
        
        Args:
            xml: XML markup string to process
            
        Returns:
            Root element after full processing with layout and render complete
        
        Examples:
            >>> root = self.parsePanel('<panel id="p1" />')
            >>> assert root.id == "p1"
            >>> assert len(self.renderContext.polys) > 0
        """
        root = GXMLParser.parse(xml)
        GXMLLayout.layout(root)
        GXMLRender.render(root, self.renderContext)
        return root
    
    def assertXMLOutput(self, xml_input, expected_xml):
        """Parse input XML and validate output matches expected XML structure.
        
        Convenience helper that combines parsePanel and XMLIntegrationTestValidator
        for cleaner test code. Parses, layouts, renders, and validates in one call.
        
        Args:
            xml_input: XML markup string to process
            expected_xml: Expected output XML with corner points and structure
            
        Raises:
            AssertionError: If validation fails with detailed error message
        
        Examples:
            >>> self.assertXMLOutput(
            ...     '<root><panel/></root>',
            ...     '<root><r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0"/></root>'
            ... )
        """
        root = self.parsePanel(xml_input)
        validator = XMLIntegrationTestValidator()
        validator.validate(expected_xml, root.children)
