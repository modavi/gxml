"""
Integration tests for panel geometry attributes.

Tests width, height, rotation, translation, and visibility attributes
and their interactions through the complete parsing and layout pipeline.
"""

import unittest
import numpy as np
from gxml_layout import GXMLLayout
from layouts.gxml_construct_layout import GXMLConstructLayout
from layouts.gxml_stack_layout import GXMLStackLayout
from layouts.gxml_fixed_layout import GXMLFixedLayout
from tests.helpers.base_integration_test import BaseIntegrationTest


class PanelRotationXMLTests(BaseIntegrationTest):
    """Integration tests for panel rotation behavior.
    
    Tests rotation around primary axis and how rotation interacts with
    different primary axes and attachment points.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLConstructLayout())
        GXMLLayout.bind_layout("stack", GXMLStackLayout())
        GXMLLayout.bind_layout("fixed", GXMLFixedLayout())
    
    def testRotateAroundPrimaryAxis(self):
        """Test rotation around primary axis (X by default)."""
        self.assertXMLOutput(
            '<root><panel rotate="90"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|0,0,-1|0,1,-1|0,1,0"/>
            </root>'''
        )
    
    def testRotateAroundAlternateAxis(self):
        """Test rotation when primary axis is Y."""
        self.assertXMLOutput(
            '''<root>
                <panel id="test" attach-point="top"/>
                <panel rotate="90"/>
            </root>''',
            '''<root>
                <r id="test" pts="0,0,0|1,0,0|1,1,0|0,1,0"/>
                <r id="1" pts="0,1,0|1,1,0|1,1,1|0,1,1"/>
            </root>'''
        )


class PanelWidthXMLTests(BaseIntegrationTest):
    """Integration tests for panel width attributes.
    
    Tests width specification via size attribute and dedicated width attribute,
    including priority rules when both are specified.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLConstructLayout())
        GXMLLayout.bind_layout("stack", GXMLStackLayout())
        GXMLLayout.bind_layout("fixed", GXMLFixedLayout())
    
    def testWidthAttr(self):
        """Test width attribute sets panel width."""
        self.assertXMLOutput(
            '<root><panel width="2"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|2,0,0|2,1,0|0,1,0"/>
            </root>'''
        )
    
    def testWidthSpecifiedInSizeAttr(self):
        """Test width can be specified via size attribute."""
        self.assertXMLOutput(
            '<root><panel size="2"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|2,0,0|2,1,0|0,1,0"/>
            </root>'''
        )

    def testWidthAttrTakesPriorityOverSize(self):
        """Test width attribute takes priority over size attribute."""
        self.assertXMLOutput(
            '<root><panel size="1" width="2"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|2,0,0|2,1,0|0,1,0"/>
            </root>'''
        )


class PanelHeightXMLTests(BaseIntegrationTest):
    """Integration tests for panel height attributes.
    
    Tests height specification, variable heights at endpoints, height expansion
    with * syntax, and interaction with thickness and attachments.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLConstructLayout())
        GXMLLayout.bind_layout("stack", GXMLStackLayout())
        GXMLLayout.bind_layout("fixed", GXMLFixedLayout())
    
    def testHeightAttr(self):
        """Test height attribute sets panel height."""
        self.assertXMLOutput(
            '<root><panel height="2"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,2,0|0,2,0"/>
            </root>'''
        )

    def testHeightSpecifiedInSizeAttr(self):
        """Test height can be specified via size attribute."""
        self.assertXMLOutput(
            '<root><panel size="1,2"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,2,0|0,2,0"/>
            </root>'''
        )

    def testHeightAttrTakesPriorityOverSize(self):
        """Test height attribute takes priority over size attribute."""
        self.assertXMLOutput(
            '<root><panel size="1,1" height="2"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,2,0|0,2,0"/>
            </root>'''
        )

    def testDifferingHeightsAtEachEndpoint(self):
        """Test height can vary along primary axis using colon syntax."""
        self.assertXMLOutput(
            '<root><panel height="2:1"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,2,0"/>
            </root>'''
        )

    def testDifferingHeightsWithThickness(self):
        """Test variable height with thickness creates proper side faces."""
        self.assertXMLOutput(
            '<root><panel height="0.5:1.5" thickness="0.5"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1.5,0|0,0.5,0">
                    <r id="front" pts="0,0,0.25|1,0,0.25|1,1.5,0.25|0,0.5,0.25"/>
                    <r id="back" pts="1,0,-0.25|0,0,-0.25|0,0.5,-0.25|1,1.5,-0.25"/>
                    <r id="top" pts="0,0.5,-0.25|0,0.5,0.25|1,1.5,0.25|1,1.5,-0.25"/>
                    <r id="bottom" pts="0,0,-0.25|1,0,-0.25|1,0,0.25|0,0,0.25"/>
                    <r id="start" pts="0,0,-0.25|0,0,0.25|0,0.5,0.25|0,0.5,-0.25"/>
                    <r id="end" pts="1,0,0.25|1,0,-0.25|1,1.5,-0.25|1,1.5,0.25"/>
                </r>
            </root>'''
        )
    
    def testWidthAndVariableHeight(self):
        """Test width attribute with variable height."""
        self.assertXMLOutput(
            '<root><panel width="2" height="2:1"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|2,0,0|2,1,0|0,2,0"/>
            </root>'''
        )
    
    def testHeightExpand(self):
        """Test height='*' expands to match adjacent panels."""
        self.assertXMLOutput(
            '''<root>
                <panel height="2"/>
                <panel height="*"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,2,0|0,2,0"/>
                <r id="1" pts="1,0,0|2,0,0|2,2,0|1,2,0"/>
            </root>'''
        )
    
    def testHeightExpandWithAttachmentAndAnchors(self):
        """Test height expansion with explicit attachment and anchoring."""
        self.assertXMLOutput(
            '''<root>
                <panel height="2"/>
                <panel height="4" rotate="90"/>
                <panel height="*:*" attach-id="0" span-id="1" attach-point="0" span-point="1"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,2,0|0,2,0">
                    <r id="back" pts="1,0,0|0,0,0|0,2,0|1,2,0"/>
                </r>
                <r id="1" pts="1,0,0|1,0,-1|1,4,-1|1,4,0">
                    <r id="back" pts="1,0,-1|1,0,0|1,4,0|1,4,-1"/>
                </r>
                <r id="2" pts="0,0,0|1,0,-1|1,4,-1|0,2,0">
                    <r id="front" pts="0,0,0|1,0,-1|1,4,-1|0,2,0"/>
                </r>
            </root>'''
        )
        
        self.assertXMLOutput(
            '''<root>
                <panel height="2"/>
                <panel height="4" rotate="90"/>
                <panel height="*:*" attach-id="1" span-id="0"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,2,0|0,2,0">
                    <r id="back" pts="1,0,0|0,0,0|0,2,0|1,2,0"/>
                </r>
                <r id="1" pts="1,0,0|1,0,-1|1,4,-1|1,4,0">
                    <r id="back" pts="1,0,-1|1,0,0|1,4,0|1,4,-1"/>
                </r>
                <r id="2" pts="1,0,-1|0,0,0|0,2,0|1,4,-1">
                    <r id="back" pts="0,0,0|1,0,-1|1,4,-1|0,2,0"/>
                </r>
            </root>'''
        )
    
    def testHeightExpandToDownstreamPanel(self):
        """Test height expands to match panel downstream in sequence."""
        self.assertXMLOutput(
            '''<root>
                <panel height="2"/>
                <panel height="*:*"/>
                <panel height="4"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,2,0|0,2,0"/>
                <r id="1" pts="1,0,0|2,0,0|2,4,0|1,2,0"/>
                <r id="2" pts="2,0,0|3,0,0|3,4,0|2,4,0"/>
            </root>'''
        )
    
    def testHeightExpandShorthand(self):
        """Test height='*' is shorthand for '*:*'."""
        self.assertXMLOutput(
            '''<root>
                <panel height="2"/>
                <panel height="*"/>
                <panel height="4"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,2,0|0,2,0"/>
                <r id="1" pts="1,0,0|2,0,0|2,4,0|1,2,0"/>
                <r id="2" pts="2,0,0|3,0,0|3,4,0|2,4,0"/>
            </root>'''
        )
    
    def testPartialHeightExpand(self):
        """Test one endpoint can be fixed while other expands."""
        self.assertXMLOutput(
            '''<root>
                <panel height="2"/>
                <panel height="*:3"/>
                <panel height="4"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,2,0|0,2,0"/>
                <r id="1" pts="1,0,0|2,0,0|2,3,0|1,2,0"/>
                <r id="2" pts="2,0,0|3,0,0|3,4,0|2,4,0"/>
            </root>'''
        )
    
    def testHeightExpandWithThickness(self):
        """Test height expansion with thickness creates proper geometry."""
        self.assertXMLOutput(
            '''<root>
                <panel width="2" height="2" thickness="1"/>
                <panel height="*:*" thickness="1"/>
                <panel width="2" height="4:2" thickness="1"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|2,0,0|2,2,0|0,2,0">
                    <r id="front" pts="0,0,0.5|2,0,0.5|2,2,0.5|0,2,0.5"/>
                    <r id="back" pts="2,0,-0.5|0,0,-0.5|0,2,-0.5|2,2,-0.5"/>
                    <r id="top" pts="0,2,-0.5|0,2,0.5|2,2,0.5|2,2,-0.5"/>
                    <r id="bottom" pts="0,0,-0.5|2,0,-0.5|2,0,0.5|0,0,0.5"/>
                    <r id="start" pts="0,0,-0.5|0,0,0.5|0,2,0.5|0,2,-0.5"/>
                    <r id="end" pts="2,0,0.5|2,0,-0.5|2,2,-0.5|2,2,0.5"/>
                </r>
                <r id="1" pts="2,0,0|3,0,0|3,4,0|2,2,0">
                    <r id="front" pts="2,0,0.5|3,0,0.5|3,4,0.5|2,2,0.5"/>
                    <r id="back" pts="3,0,-0.5|2,0,-0.5|2,2,-0.5|3,4,-0.5"/>
                    <r id="top" pts="2,2,-0.5|2,2,0.5|3,4,0.5|3,4,-0.5"/>
                    <r id="bottom" pts="2,0,-0.5|3,0,-0.5|3,0,0.5|2,0,0.5"/>
                    <r id="start" pts="2,0,-0.5|2,0,0.5|2,2,0.5|2,2,-0.5"/>
                    <r id="end" pts="3,0,0.5|3,0,-0.5|3,4,-0.5|3,4,0.5"/>
                </r>
                <r id="2" pts="3,0,0|5,0,0|5,2,0|3,4,0">
                    <r id="front" pts="3,0,0.5|5,0,0.5|5,2,0.5|3,4,0.5"/>
                    <r id="back" pts="5,0,-0.5|3,0,-0.5|3,4,-0.5|5,2,-0.5"/>
                    <r id="top" pts="3,4,-0.5|3,4,0.5|5,2,0.5|5,2,-0.5"/>
                    <r id="bottom" pts="3,0,-0.5|5,0,-0.5|5,0,0.5|3,0,0.5"/>
                    <r id="start" pts="3,0,-0.5|3,0,0.5|3,4,0.5|3,4,-0.5"/>
                    <r id="end" pts="5,0,0.5|5,0,-0.5|5,2,-0.5|5,2,0.5"/>
                </r>
            </root>'''
        )


class PanelTranslationXMLTests(BaseIntegrationTest):
    """Integration tests for panel translation/offset attributes.
    
    Tests x, y, z offset attributes and variable offsets at endpoints.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLConstructLayout())
        GXMLLayout.bind_layout("stack", GXMLStackLayout())
        GXMLLayout.bind_layout("fixed", GXMLFixedLayout())
    
    def testOffsetAttributes(self):
        """Test x, y, z attributes offset panel in world space."""
        self.assertXMLOutput(
            '<root><panel x="2" y="3" z="0.5"/></root>',
            '''<root>
                <r id="0" pts="2,3,0.5|3,3,0.5|3,4,0.5|2,4,0.5"/>
            </root>'''
        )
    
    def testDifferingYOffsetsAtEachEndpoint(self):
        """Test y offset can vary along primary axis."""
        self.assertXMLOutput(
            '<root><panel y="3:4"/></root>',
            '''<root>
                <r id="0" pts="0,3,0|1,4,0|1,5,0|0,4,0"/>
            </root>'''
        )
    
    def testDifferingXOffsetsAtEachEndpoint(self):
        """Test x offset can vary along primary axis."""
        self.assertXMLOutput(
            '<root><panel x="3:4"/></root>',
            '''<root>
                <r id="0" pts="3,0,0|4,0,0|5,1,0|4,1,0"/>
            </root>'''
        )
    
    def testDifferingZOffsetsAtEachEndpoint(self):
        """Test z offset can vary along primary axis."""
        self.assertXMLOutput(
            '<root><panel z="3:4"/></root>',
            '''<root>
                <r id="0" pts="0,0,3|1,0,3|1,1,4|0,1,4"/>
            </root>'''
        )
    
    def testDifferingOffsetsAllAxesCombined(self):
        """Test variable offsets on all three axes simultaneously."""
        self.assertXMLOutput(
            '<root><panel x="3:4" y="3:4" z="3:4"/></root>',
            '''<root>
                <r id="0" pts="3,3,3|4,4,3|5,5,4|4,4,4"/>
            </root>'''
        )


class PanelVisibilityXMLTests(BaseIntegrationTest):
    """Integration tests for panel visibility attributes.
    
    Tests visible and visible-self attributes and their interaction in hierarchies.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLConstructLayout())
        GXMLLayout.bind_layout("stack", GXMLStackLayout())
        GXMLLayout.bind_layout("fixed", GXMLFixedLayout())
    
    def testVisibleAttribute(self):
        """Test visible attribute controls rendering of panel and children."""
        root = self.parsePanel('''<root>
            <panel/>
            <panel visible="False"/>
            <panel visible="True"/>
        </root>''')
        
        # Root never gets rendered
        self.assertEqual(root.renderCount, 0)
        
        # Panels visible by default
        self.assertEqual(root.children[0].renderCount, 1)
        
        # visible="False" suppresses rendering
        self.assertEqual(root.children[1].renderCount, 0)
        
        # visible="True" explicitly enables rendering
        self.assertEqual(root.children[2].renderCount, 1)
    
    def testVisibleSelfAttribute(self):
        """Test visible-self controls only self, not children."""
        root = self.parsePanel('''<root>
            <panel visible-self="False">
                <panel visible-self="True"/>
            </panel>
        </root>''')
        
        self.assertEqual(root.renderCount, 0)
        
        # Parent not rendered but child is
        self.assertEqual(root.children[0].renderCount, 0)
        self.assertEqual(root.children[0].children[0].renderCount, 1)
    
    def testVisibleAndVisibleSelfInteraction(self):
        """Test visible and visible-self interact correctly in hierarchy."""
        root = self.parsePanel('''<root>
            <panel visible="False">
                <panel visible-self="True"/>
            </panel>
            <panel visible="True">
                <panel visible-self="False"/>
            </panel>
        </root>''')
        
        self.assertEqual(root.renderCount, 0)
        self.assertEqual(root.children[0].renderCount, 0)
        # visible-self="True" overridden by parent visible="False"
        self.assertEqual(root.children[0].children[0].renderCount, 0)
        
        self.assertEqual(root.children[1].renderCount, 1)
        # visible-self="False" prevents rendering even though parent visible
        self.assertEqual(root.children[1].children[0].renderCount, 0)


class PanelThicknessXMLTests(BaseIntegrationTest):
    """Integration tests for panel thickness block generation.
    
    Tests creation of 3D panel blocks with front, back, top, bottom,
    start, and end faces when thickness is applied.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLConstructLayout())
        GXMLLayout.bind_layout("stack", GXMLStackLayout())
        GXMLLayout.bind_layout("fixed", GXMLFixedLayout())
    
    def testSinglePanelWithThickness(self):
        """Test panel with thickness creates six side faces.
        
        A panel with thickness=1.0 creates a 3D block with:
        - FRONT face at z=0.5
        - BACK face at z=-0.5
        - TOP face at y=1
        - BOTTOM face at y=0
        - START face at x=0
        - END face at x=1
        """
        self.assertXMLOutput(
            '<root><panel thickness="1.0"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front" pts="0,0,0.5|1,0,0.5|1,1,0.5|0,1,0.5"/>
                    <r id="back" pts="1,0,-0.5|0,0,-0.5|0,1,-0.5|1,1,-0.5"/>
                    <r id="top" pts="0,1,-0.5|0,1,0.5|1,1,0.5|1,1,-0.5"/>
                    <r id="bottom" pts="0,0,-0.5|1,0,-0.5|1,0,0.5|0,0,0.5"/>
                    <r id="start" pts="0,0,-0.5|0,0,0.5|0,1,0.5|0,1,-0.5"/>
                    <r id="end" pts="1,0,0.5|1,0,-0.5|1,1,-0.5|1,1,0.5"/>
                </r>
            </root>'''
        )
    
    def testMultiplePanelsWithThickness(self):
        """Test multiple panels in sequence each generate proper thickness faces.
        
        Two sequential panels (default size=1) create:
        - Panel 0: from x=0 to x=1 with all 6 faces
        - Panel 1: from x=1 to x=2 with all 6 faces
        Each panel has its own complete block geometry.
        """
        self.assertXMLOutput(
            '<root><panel thickness="1.0"/><panel thickness="1.0"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front" pts="0,0,0.5|1,0,0.5|1,1,0.5|0,1,0.5"/>
                    <r id="back" pts="1,0,-0.5|0,0,-0.5|0,1,-0.5|1,1,-0.5"/>
                    <r id="top" pts="0,1,-0.5|0,1,0.5|1,1,0.5|1,1,-0.5"/>
                    <r id="bottom" pts="0,0,-0.5|1,0,-0.5|1,0,0.5|0,0,0.5"/>
                    <r id="start" pts="0,0,-0.5|0,0,0.5|0,1,0.5|0,1,-0.5"/>
                    <r id="end" pts="1,0,0.5|1,0,-0.5|1,1,-0.5|1,1,0.5"/>
                </r>
                <r id="1" pts="1,0,0|2,0,0|2,1,0|1,1,0">
                    <r id="front" pts="1,0,0.5|2,0,0.5|2,1,0.5|1,1,0.5"/>
                    <r id="back" pts="2,0,-0.5|1,0,-0.5|1,1,-0.5|2,1,-0.5"/>
                    <r id="top" pts="1,1,-0.5|1,1,0.5|2,1,0.5|2,1,-0.5"/>
                    <r id="bottom" pts="1,0,-0.5|2,0,-0.5|2,0,0.5|1,0,0.5"/>
                    <r id="start" pts="1,0,-0.5|1,0,0.5|1,1,0.5|1,1,-0.5"/>
                    <r id="end" pts="2,0,0.5|2,0,-0.5|2,1,-0.5|2,1,0.5"/>
                </r>
            </root>'''
        )
    
    def testPanelWithThicknessAndRotation(self):
        """Test rotated panels generate thickness faces in correct orientations.
        
        Panel 0 is horizontal with thickness=0.1 (extends z=-0.05 to z=0.05).
        Panel 1 is rotated 90 degrees and attached at x=0.5 via T-junction.
        
        The T-junction causes:
        - Panel 0's BACK face is split into back-0 and back-1
        - Panel 1's START face is omitted (interior geometry)
        - Panel 1's thickness is rotated: x-axis becomes the thickness axis
        """
        self.assertXMLOutput(
            '<root><panel thickness="0.1"/><panel thickness="0.1" rotate="90" attach-id="0" attach-point="0.5"/></root>',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front" pts="0,0,0.05|1,0,0.05|1,1,0.05|0,1,0.05"/>
                    <r id="back-0" pts="0.45,0,-0.05|0,0,-0.05|0,1,-0.05|0.45,1,-0.05"/>
                    <r id="back-1" pts="1,0,-0.05|0.55,0,-0.05|0.55,1,-0.05|1,1,-0.05"/>
                    <r id="top" pts="0,1,-0.05|0,1,0.05|1,1,0.05|1,1,-0.05"/>
                    <r id="bottom" pts="0,0,-0.05|1,0,-0.05|1,0,0.05|0,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                    <r id="end" pts="1,0,0.05|1,0,-0.05|1,1,-0.05|1,1,0.05"/>
                </r>
                <r id="1" pts="0.5,0,0|0.5,0,-1|0.5,1,-1|0.5,1,0">
                    <r id="front" pts="0.55,0,-0.05|0.55,0,-1|0.55,1,-1|0.55,1,-0.05"/>
                    <r id="back" pts="0.45,0,-1|0.45,0,-0.05|0.45,1,-0.05|0.45,1,-1"/>
                    <r id="top" pts="0.45,1,-0.05|0.55,1,-0.05|0.55,1,-1|0.45,1,-1"/>
                    <r id="bottom" pts="0.45,0,-0.05|0.45,0,-1|0.55,0,-1|0.55,0,-0.05"/>
                    <r id="end" pts="0.55,0,-1|0.45,0,-1|0.45,1,-1|0.55,1,-1"/>
                </r>
            </root>'''
        )


if __name__ == '__main__':
    unittest.main()
