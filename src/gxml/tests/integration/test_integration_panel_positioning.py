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
    """Integration tests for panel attachment and anchoring system.
    
    Tests the complete attachment system including implicit/explicit attachment,
    attach-to/attach-point positioning, anchor elements, and anchor-to points.
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
        """Test attach-to specifies position along parent's primary axis."""
        root = self.parsePanel('''<root>
            <panel/>
            <panel attach-to="0.5"/>
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
    
    def testAnchorToElement(self):
        """Test anchor-id causes panel to orient toward anchor element."""
        root = self.parsePanel('''<root>
            <panel/>
            <panel rotate="90"/>
            <panel anchor-id="0"/>
        </root>''')
        assert_corner_points(self, root.children[2], [1,0,-1], [0,0,0], [0,1,0], [1,1,-1])
    
    def testAnchorToSpecificPoint(self):
        """Test anchor-to specifies position on anchor element."""
        root = self.parsePanel('''<root>
            <panel/>
            <panel rotate="90"/>
            <panel attach-id="0" attach-to="0.5" anchor-id="1" anchor-to="0.5"/>
        </root>''')
        assert_corner_points(self, root.children[2], [0.5,0,0], [1.0,0,-0.5], [1.0,1.0,-0.5], [0.5,1.0,0])
        
        root = self.parsePanel('''<root>
            <panel/>
            <panel rotate="90" anchor-point="0.5"/>
            <panel attach-id="0" attach-to="0.5" anchor-id="1"/>
        </root>''')
        assert_corner_points(self, root.children[2], [0.5,0,0], [1.0,0,-0.5], [1.0,1.0,-0.5], [0.5,1.0,0])
    
    def testAnchorToSamePointAsAttach(self):
        """Test anchoring and attaching to same point creates degenerate quad."""
        root = self.parsePanel('''<root>
            <panel rotate="-90"/>
            <panel rotate="90"/>
            <panel attach-id="0" anchor-id="1"/>
        </root>''')
        assert_corner_points(self, root.children[2], [0,0,1], [0,0,1], [0,1,1], [0,1,1])
    
    def testAnchorElementThatDoesntExist(self):
        """Test non-existent anchor-id falls back to default positioning."""
        root = self.parsePanel('''<root>
            <panel size="5"/>
            <panel rotate="90" size="2"/>
            <panel rotate="90" size="5"/>
            <panel attach-id="0" anchor-id="NOTAREALELEMENT" rotate="45" attach-to="0.5"/>    
        </root>''')
        assert_corner_points(self, root.children[3], [2.5,0,0], [3.20711,0,-0.707107], 
                           [3.20711,1.0,-0.707107], [2.5,1,0])


class PanelIntersectionXMLTests(BaseIntegrationTest):
    """Integration tests for intersection-based panel positioning.
    
    Tests the auto-intersection system (anchor-to="auto") which positions panels
    based on geometric intersections with anchor elements.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLConstructLayout())
        GXMLLayout.bind_layout("stack", GXMLStackLayout())
        GXMLLayout.bind_layout("fixed", GXMLFixedLayout())
    
    def testAnchorWithAutoIntersect(self):
        """Test anchor-to='auto' finds intersection with anchor element."""
        root = self.parsePanel('''<root>
            <panel/>
            <panel rotate="90" size="2"/>
            <panel rotate="90"/>
            <panel attach-id="0" anchor-id="2" anchor-to="auto" rotate="90" attach-to="0.5"/>
        </root>''')
        assert_corner_points(self, root.children[3], [0.5,0,0], [0.5,0,-2], [0.5,1,-2], [0.5,1,0])
    
    def testAutoAnchorWithNoIntersection(self):
        """Test auto anchor falls back gracefully when no intersection found."""
        root = self.parsePanel('''<root>
            <panel size="5"/>
            <panel rotate="90" size="2"/>
            <panel rotate="90"/>
            <panel attach-id="0" anchor-id="2" anchor-to="auto" rotate="90" attach-to="0.5"/>
        </root>''')
        assert_corner_points(self, root.children[3], [2.5,0,0], [4.0,0,-2], [4.0,1,-2], [2.5,1,0])
    
    def testAutoAnchorWithComplexGeometry(self):
        """Test auto anchor with multiple panels in scene."""
        root = self.parsePanel('''<root>
            <panel size="5"/>
            <panel rotate="90" size="2"/>
            <panel rotate="90"/>
            <panel size="2" attach-id="0" attach-to="0" rotate="90"/>
            <panel rotate="-90"/>
            <panel attach-id="0" anchor-id="4" anchor-to="auto" rotate="90" attach-to="0.5"/>
        </root>''')
        assert_corner_points(self, root.children[5], [2.5,0,0], [1.0,0,-2], [1.0,1.0,-2], [2.5,1,0])
    
    def testAutoAnchorWithRotation(self):
        """Test auto anchor combined with rotation on attaching panel."""
        root = self.parsePanel('''<root>
            <panel size="5"/>
            <panel rotate="90" size="2"/>
            <panel rotate="90" size="5"/>
            <panel attach-id="0" anchor-id="2" rotate="45" attach-to="0.5"/>    
        </root>''')
        assert_corner_points(self, root.children[3], [2.5,0,0], [4.5,0,-2], [4.5,1.0,-2], [2.5,1,0])


class PanelPrimaryAxisXMLTests(BaseIntegrationTest):
    """Integration tests for primary axis inference and positioning.
    
    Tests how panels determine their primary axis based on attach-to,
    attach-point, and how primary axis affects panel orientation.
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
    
    def testPrimaryAxisInferenceWithAttachTo(self):
        """Test attach-to side changes primary axis."""
        panels = self.parsePanel('''
            <root>
                <panel/>
                <panel size="2" attach-to="top"/>
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
                <panel attach-id="0" size="2" attach-to="top"/>
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
