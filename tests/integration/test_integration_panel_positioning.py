"""
Integration tests for panel positioning and attachment system.

Tests the complete attachment system including implicit/explicit attachment,
attach-to/attach-point positioning, anchor elements, intersection-based
positioning, and primary axis inference.
"""

import unittest
import numpy as np
from gxml_layout import GXMLLayout
from layouts.gxml_construct_layout import GXMLConstructLayout
from layouts.gxml_stack_layout import GXMLStackLayout
from layouts.gxml_fixed_layout import GXMLFixedLayout
from tests.test_fixtures.base_integration_test import BaseIntegrationTest
from tests.test_fixtures.assertions import assert_corner_points


class PanelAttachmentXMLTests(BaseIntegrationTest):
    """Integration tests for panel attachment and span system.
    
    Tests the complete attachment system including implicit/explicit attachment,
    attach-point positioning, span elements, and span-point positions.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLConstructLayout())
        GXMLLayout.bind_layout("stack", GXMLStackLayout())
        GXMLLayout.bind_layout("fixed", GXMLFixedLayout())
    
    def testImplicitAttachment(self):
        """Test panels implicitly attach to previous sibling."""
        panels = self.parsePanel('''
            <root>
                <panel/>
                <panel/>
                <panel/>
            </root>''').children
        
        self.assertIsNone(panels[0].attachElement)
        self.assertEqual(panels[1].attachElement, panels[0])
        self.assertEqual(panels[2].attachElement, panels[1])
    
    def testExplicitAttachmentById(self):
        """Test explicit attach-id overrides implicit attachment."""
        panels = self.parsePanel('''
            <root>
                <panel/>
                <panel/>
                <panel/>
                <panel attach-id="1"/>
                <panel/>
            </root>''').children
        
        self.assertIsNone(panels[0].attachElement)
        self.assertEqual(panels[1].attachElement, panels[0])
        self.assertEqual(panels[2].attachElement, panels[1])
        self.assertEqual(panels[3].attachElement, panels[1])  # Attaches to panel 1, not panel 2
        self.assertEqual(panels[4].attachElement, panels[3])
    
    def testAttachToRotatedAndScaledPanel(self):
        """Test attachment respects parent transform."""
        root = self.parsePanel('''
            <root>
                <panel rotate="0,90,0" size="2,1,1" pivot="0.5,0.5"/>
                <panel/>
            </root>''')
        assert_corner_points(self, root.children[1], [0,-.5,-1], [0,-.5,-2], [0,.5,-2], [0,.5,-1])
    
    def testAttachToZeroScalePanel(self):
        """Test attachment to zero-scale parent still works."""
        root = self.parsePanel('''
            <root>
                <panel size="0,0,0"/>
                <panel/>
            </root>''')
        assert_corner_points(self, root.children[1], [0,0,0], [1,0,0], [1,1,0], [0,1,0])
        
        root = self.parsePanel('''
            <root>
                <panel rotate="90" size="0"/>
                <panel/>
            </root>''')
        assert_corner_points(self, root.children[1], [0,0,0], [0,0,-1], [0,1,-1], [0,1,0])
    
    def testAttachToPointAlongPrimaryAxis(self):
        """Test attach-point specifies position along parent's primary axis."""
        root = self.parsePanel('''<root>
            <panel/>
            <panel attach-point="0.5"/>
        </root>''')
        assert_corner_points(self, root.children[1], [0.5,0,0], [1.5,0,0], [1.5,1.0,0], [0.5,1.0,0])
    
    def testAttachPointOnParent(self):
        """Test attach-point on parent affects where next sibling attaches."""
        root = self.parsePanel('''<root>
            <panel attach-point="0.5"/>
            <panel/>
        </root>''')
        assert_corner_points(self, root.children[1], [0.5,0,0], [1.5,0,0], [1.5,1.0,0], [0.5,1.0,0])
    
    def testAttachedOrientation(self):
        """Test attached panel starts at attachment point with correct orientation."""
        root = self.parsePanel('''<root>
            <panel/>
            <panel/>
        </root>''')

        # Attached panel should start where parent ends
        block1_end = root.children[0].transform.transform_point((1, 0, 0))
        block2_start = root.children[1].transform.transform_point((0, 0, 0))
        self.assertTrue(np.allclose(block1_end, block2_start))
    
    def testSpanToElement(self):
        """Test span-id causes panel to orient toward span element."""
        root = self.parsePanel('''<root>
            <panel/>
            <panel rotate="90"/>
            <panel span-id="0"/>
        </root>''')
        assert_corner_points(self, root.children[2], [1,0,-1], [0,0,0], [0,1,0], [1,1,-1])
    
    def testSpanToSpecificPoint(self):
        """Test span-point specifies position on span element."""
        root = self.parsePanel('''<root>
            <panel/>
            <panel rotate="90"/>
            <panel attach-id="0" attach-point="0.5" span-id="1" span-point="0.5"/>
        </root>''')
        assert_corner_points(self, root.children[2], [0.5,0,0], [1.0,0,-0.5], [1.0,1.0,-0.5], [0.5,1.0,0])
        
        root = self.parsePanel('''<root>
            <panel/>
            <panel rotate="90" span-point-self="0.5"/>
            <panel attach-id="0" attach-point="0.5" span-id="1"/>
        </root>''')
        assert_corner_points(self, root.children[2], [0.5,0,0], [1.0,0,-0.5], [1.0,1.0,-0.5], [0.5,1.0,0])
    
    def testSpanToSamePointAsAttach(self):
        """Test spanning and attaching to same point creates degenerate quad."""
        root = self.parsePanel('''<root>
            <panel rotate="-90"/>
            <panel rotate="90"/>
            <panel attach-id="0" span-id="1"/>
        </root>''')
        assert_corner_points(self, root.children[2], [0,0,1], [0,0,1], [0,1,1], [0,1,1])
    
    def testSpanElementThatDoesntExist(self):
        """Test non-existent span-id falls back to default positioning."""
        root = self.parsePanel('''<root>
            <panel size="5"/>
            <panel rotate="90" size="2"/>
            <panel rotate="90" size="5"/>
            <panel attach-id="0" span-id="NOTAREALELEMENT" rotate="45" attach-point="0.5"/>    
        </root>''')
        assert_corner_points(self, root.children[3], [2.5,0,0], [3.20711,0,-0.707107], 
                           [3.20711,1.0,-0.707107], [2.5,1,0])


class PanelIntersectionXMLTests(BaseIntegrationTest):
    """Integration tests for intersection-based panel positioning.
    
    Tests the auto-intersection system (span-point="auto") which positions panels
    based on geometric intersections with span elements.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLConstructLayout())
        GXMLLayout.bind_layout("stack", GXMLStackLayout())
        GXMLLayout.bind_layout("fixed", GXMLFixedLayout())
    
    def testSpanWithAutoIntersect(self):
        """Test span-point='auto' finds intersection with span element."""
        root = self.parsePanel('''<root>
            <panel/>
            <panel rotate="90" size="2"/>
            <panel rotate="90"/>
            <panel attach-id="0" span-id="2" span-point="auto" rotate="90" attach-point="0.5"/>
        </root>''')
        assert_corner_points(self, root.children[3], [0.5,0,0], [0.5,0,-2], [0.5,1,-2], [0.5,1,0])
    
    def testAutoSpanWithNoIntersection(self):
        """Test auto span falls back gracefully when no intersection found."""
        root = self.parsePanel('''<root>
            <panel size="5"/>
            <panel rotate="90" size="2"/>
            <panel rotate="90"/>
            <panel attach-id="0" span-id="2" span-point="auto" rotate="90" attach-point="0.5"/>
        </root>''')
        assert_corner_points(self, root.children[3], [2.5,0,0], [4.0,0,-2], [4.0,1,-2], [2.5,1,0])
    
    def testAutoSpanWithComplexGeometry(self):
        """Test auto span with multiple panels in scene."""
        root = self.parsePanel('''<root>
            <panel size="5"/>
            <panel rotate="90" size="2"/>
            <panel rotate="90"/>
            <panel size="2" attach-id="0" attach-point="0" rotate="90"/>
            <panel rotate="-90"/>
            <panel attach-id="0" span-id="4" span-point="auto" rotate="90" attach-point="0.5"/>
        </root>''')
        assert_corner_points(self, root.children[5], [2.5,0,0], [1.0,0,-2], [1.0,1.0,-2], [2.5,1,0])
    
    def testAutoSpanWithRotation(self):
        """Test auto span combined with rotation on attaching panel."""
        root = self.parsePanel('''<root>
            <panel size="5"/>
            <panel rotate="90" size="2"/>
            <panel rotate="90" size="5"/>
            <panel attach-id="0" span-id="2" rotate="45" attach-point="0.5"/>    
        </root>''')
        assert_corner_points(self, root.children[3], [2.5,0,0], [4.5,0,-2], [4.5,1.0,-2], [2.5,1,0])


class PanelPrimaryAxisXMLTests(BaseIntegrationTest):
    """Integration tests for primary axis inference and positioning.
    
    Tests how panels determine their primary axis based on attach-point,
    and how primary axis affects panel orientation.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLConstructLayout())
        GXMLLayout.bind_layout("stack", GXMLStackLayout())
        GXMLLayout.bind_layout("fixed", GXMLFixedLayout())
    
    def testPrimaryAxisDefault(self):
        """Test default primary axis is X and scales along X."""
        panels = self.parsePanel('''
            <root>
                <panel size="2"/>
            </root>''').children
        assert_corner_points(self, panels[0], [0,0,0], [2,0,0], [2,1,0], [0,1,0])
        
        panels = self.parsePanel('''
            <root>
                <panel attach-point="1"/>
                <panel size="2"/>
            </root>''').children
        assert_corner_points(self, panels[1], [1,0,0], [3,0,0], [3,1,0], [1,1,0])
    
    def testPrimaryAxisInferenceWithAttachPoint(self):
        """Test attach-point side changes primary axis."""
        panels = self.parsePanel('''
            <root>
                <panel/>
                <panel size="2" attach-point="top"/>
            </root>''').children
        assert_corner_points(self, panels[1], [0,1,0], [1,1,0], [1,3,0], [0,3,0])
    
    def testPrimaryAxisInferenceWithImplicitAttachPoint(self):
        """Test implicit attachment uses parent's attach-point for primary axis."""
        panels = self.parsePanel('''
            <root>
                <panel attach-point="top"/>
                <panel size="2"/>
            </root>''').children
        assert_corner_points(self, panels[1], [0,1,0], [1,1,0], [1,3,0], [0,3,0])
    
    def testPrimaryAxisInferenceWithExplicitAttachPoint(self):
        """Test explicit attachment uses specified element's attach-point."""
        panels = self.parsePanel('''
            <root>
                <panel attach-point="top"/>
                <panel/>
                <panel attach-id="0" size="2" attach-point="top"/>
            </root>''').children
        assert_corner_points(self, panels[2], [0,1,0], [1,1,0], [1,3,0], [0,3,0])
    
    def testPrimaryAxisWithPivot(self):
        """Test pivot affects positioning along primary axis."""
        panels = self.parsePanel('''
            <root>
                <panel pivot="0.5"/>
            </root>''').children
        assert_corner_points(self, panels[0], [-0.5,0.0,0], [0.5,0.0,0.0], [0.5,1.0,0], [-0.5,1.0,0])
        
        panels = self.parsePanel('''
            <root>
                <panel size="2" pivot="0.5,0.5"/>
            </root>''').children
        assert_corner_points(self, panels[0], [-1,-0.5,0], [1,-0.5,0], [1,0.5,0], [-1,0.5,0])


if __name__ == '__main__':
    unittest.main()
