"""
Integration tests for panel layout engine selection and application.

Tests that layout engines are correctly selected, bound, and applied
through the panel hierarchy based on layout attributes.
"""

import unittest
from gxml_layout import GXMLLayout
from layouts.gxml_construct_layout import GXMLConstructLayout
from layouts.gxml_stack_layout import GXMLStackLayout
from layouts.gxml_fixed_layout import GXMLFixedLayout
from tests.test_fixtures.base_integration_test import BaseIntegrationTest
from tests.test_fixtures.mocks import GXMLMockLayout


class PanelLayoutEngineXMLTests(BaseIntegrationTest):
    """Integration tests for layout engine selection and application.
    
    Tests that layout engines are correctly selected based on layout attribute
    and applied through the hierarchy.
    """
    
    def setUp(self):
        super().setUp()
        # Wrap layouts with mock to enable layoutEngineUsed tracking
        GXMLLayout.bind_layout("construct", GXMLMockLayout.create(GXMLConstructLayout()))
        GXMLLayout.bind_layout("stack", GXMLMockLayout.create(GXMLStackLayout()))
        GXMLLayout.bind_layout("fixed", GXMLMockLayout.create(GXMLFixedLayout()))
    
    def testDefaultLayoutEngines(self):
        """Test default layout engines for root and panels."""
        root = self.parsePanel('''<root>
            <panel>
                <panel/>
            </panel>
        </root>''')
        
        # The root node never gets laid out - layoutEngineUsed should be None
        self.assertIsNone(root.layoutEngineUsed)
        
        # The root by default lays its children out using the construct layout engine
        self.assertEqual(root.children[0].layoutEngineUsed, 
                        GXMLLayout.get_bound_layout_processor(GXMLConstructLayout.layoutScheme))
        
        # Panels by default lay their children out using the stack layout engine
        self.assertEqual(root.children[0].children[0].layoutEngineUsed, 
                        GXMLLayout.get_bound_layout_processor(GXMLStackLayout.layoutScheme))
    
    def testComplexLayoutEngineHierarchy(self):
        """Test layout engines propagate correctly through complex hierarchy."""
        root = self.parsePanel('''<root>
            <panel layout="construct">
                <panel layout="stack">
                    <panel layout="fixed"/>
                    <panel layout="fixed"/> 
                </panel>
                <panel layout="fixed">
                    <panel layout="stack"/>
                    <panel layout="stack"/>
                </panel>
            </panel>
            <panel layout="stack">
                <panel layout="fixed">
                    <panel layout="construct"/>
                    <panel layout="stack"/>
                </panel>
                <panel layout="construct">
                    <panel layout="stack"/>
                    <panel layout="construct"/>
                </panel>
            </panel>
            <panel layout="fixed">
                <panel layout="construct">
                    <panel layout="stack"/>
                    <panel layout="construct"/>
                </panel>
                <panel layout="stack">
                    <panel layout="fixed"/>
                    <panel layout="stack"/>
                </panel>
            </panel>
        </root>''')
        
        construct = GXMLLayout.get_bound_layout_processor(GXMLConstructLayout.layoutScheme)
        stack = GXMLLayout.get_bound_layout_processor(GXMLStackLayout.layoutScheme)
        fixed = GXMLLayout.get_bound_layout_processor(GXMLFixedLayout.layoutScheme)
        
        # Root never gets laid out - layoutEngineUsed should be None
        self.assertIsNone(root.layoutEngineUsed)
        self.assertEqual(root.children[0].layoutEngineUsed, construct)
        
        self.assertEqual(root.children[0].children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[0].children[0].children[0].layoutEngineUsed, stack)
        self.assertEqual(root.children[0].children[0].children[1].layoutEngineUsed, stack)
        
        self.assertEqual(root.children[0].children[1].layoutEngineUsed, construct)
        self.assertEqual(root.children[0].children[1].children[0].layoutEngineUsed, fixed)
        self.assertEqual(root.children[0].children[1].children[1].layoutEngineUsed, fixed)
        
        self.assertEqual(root.children[1].layoutEngineUsed, construct)
        
        self.assertEqual(root.children[1].children[0].layoutEngineUsed, stack)
        self.assertEqual(root.children[1].children[0].children[0].layoutEngineUsed, fixed)
        self.assertEqual(root.children[1].children[0].children[1].layoutEngineUsed, fixed)
        
        self.assertEqual(root.children[1].children[1].layoutEngineUsed, stack)
        self.assertEqual(root.children[1].children[1].children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[1].children[1].children[1].layoutEngineUsed, construct)
        
        self.assertEqual(root.children[2].layoutEngineUsed, construct)
        
        self.assertEqual(root.children[2].children[0].layoutEngineUsed, fixed)
        self.assertEqual(root.children[2].children[0].children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[2].children[0].children[1].layoutEngineUsed, construct)
        
        self.assertEqual(root.children[2].children[1].layoutEngineUsed, fixed)
        self.assertEqual(root.children[2].children[1].children[0].layoutEngineUsed, stack)
        self.assertEqual(root.children[2].children[1].children[1].layoutEngineUsed, stack)
    
    def testInvalidLayoutEngineRaisesError(self):
        """Test invalid layout attribute value raises ValueError."""
        with self.assertRaises(ValueError):
            self.parsePanel('''<root>
                <panel layout="invalid"/>
            </root>''')
    
    def testMixedLayoutTypesInSiblings(self):
        """Test sibling panels with different layout engines don't interfere."""
        root = self.parsePanel('''<root>
            <panel layout="construct">
                <panel/>
                <panel/>
            </panel>
            <panel layout="fixed">
                <panel/>
                <panel/>
            </panel>
            <panel layout="stack">
                <panel/>
                <panel/>
            </panel>
        </root>''')
        
        construct = GXMLLayout.get_bound_layout_processor(GXMLConstructLayout.layoutScheme)
        stack = GXMLLayout.get_bound_layout_processor(GXMLStackLayout.layoutScheme)
        fixed = GXMLLayout.get_bound_layout_processor(GXMLFixedLayout.layoutScheme)
        
        # First panel and its children use construct
        self.assertEqual(root.children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[0].children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[0].children[1].layoutEngineUsed, construct)
        
        # Second panel and its children use fixed
        self.assertEqual(root.children[1].layoutEngineUsed, construct)  # Laid out by root's construct
        self.assertEqual(root.children[1].children[0].layoutEngineUsed, fixed)
        self.assertEqual(root.children[1].children[1].layoutEngineUsed, fixed)
        
        # Third panel and its children use stack
        self.assertEqual(root.children[2].layoutEngineUsed, construct)  # Laid out by root's construct
        self.assertEqual(root.children[2].children[0].layoutEngineUsed, stack)
        self.assertEqual(root.children[2].children[1].layoutEngineUsed, stack)
    
    def testLayoutSwitchingMidHierarchy(self):
        """Test changing layout mid-hierarchy affects only descendants."""
        root = self.parsePanel('''<root>
            <panel layout="construct">
                <panel/>
                <panel layout="fixed">
                    <panel/>
                    <panel layout="stack">
                        <panel/>
                    </panel>
                </panel>
                <panel/>
            </panel>
        </root>''')
        
        construct = GXMLLayout.get_bound_layout_processor(GXMLConstructLayout.layoutScheme)
        stack = GXMLLayout.get_bound_layout_processor(GXMLStackLayout.layoutScheme)
        fixed = GXMLLayout.get_bound_layout_processor(GXMLFixedLayout.layoutScheme)
        
        # Root children use construct
        self.assertEqual(root.children[0].layoutEngineUsed, construct)
        
        # First level children use parent's construct layout
        self.assertEqual(root.children[0].children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[0].children[1].layoutEngineUsed, construct)
        self.assertEqual(root.children[0].children[2].layoutEngineUsed, construct)
        
        # Middle panel switched to fixed, so its children use fixed
        self.assertEqual(root.children[0].children[1].children[0].layoutEngineUsed, fixed)
        self.assertEqual(root.children[0].children[1].children[1].layoutEngineUsed, fixed)
        
        # Nested panel switched back to stack, so its child uses stack
        self.assertEqual(root.children[0].children[1].children[1].children[0].layoutEngineUsed, stack)


class PanelLayoutEdgeCasesXMLTests(BaseIntegrationTest):
    """Integration tests for layout engine edge cases.
    
    Tests edge cases including empty panels, single children, and zero-scale
    parents with different layout engines.
    """
    
    def setUp(self):
        super().setUp()
        GXMLLayout.bind_layout("construct", GXMLMockLayout.create(GXMLConstructLayout()))
        GXMLLayout.bind_layout("stack", GXMLMockLayout.create(GXMLStackLayout()))
        GXMLLayout.bind_layout("fixed", GXMLMockLayout.create(GXMLFixedLayout()))
    
    def testEmptyPanelsWithDifferentLayouts(self):
        """Test empty panels (no children) with different layout attributes."""
        root = self.parsePanel('''<root>
            <panel layout="construct"/>
            <panel layout="fixed"/>
            <panel layout="stack"/>
        </root>''')
        
        construct = GXMLLayout.get_bound_layout_processor(GXMLConstructLayout.layoutScheme)
        
        # All panels are laid out by root's construct layout
        self.assertEqual(root.children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[1].layoutEngineUsed, construct)
        self.assertEqual(root.children[2].layoutEngineUsed, construct)
        
        # Verify they have no children
        self.assertEqual(len(root.children[0].children), 0)
        self.assertEqual(len(root.children[1].children), 0)
        self.assertEqual(len(root.children[2].children), 0)
    
    def testSingleChildWithDifferentLayoutEngines(self):
        """Test panels with single child using different layout engines."""
        root = self.parsePanel('''<root>
            <panel layout="construct">
                <panel/>
            </panel>
            <panel layout="fixed">
                <panel/>
            </panel>
            <panel layout="stack">
                <panel/>
            </panel>
        </root>''')
        
        construct = GXMLLayout.get_bound_layout_processor(GXMLConstructLayout.layoutScheme)
        stack = GXMLLayout.get_bound_layout_processor(GXMLStackLayout.layoutScheme)
        fixed = GXMLLayout.get_bound_layout_processor(GXMLFixedLayout.layoutScheme)
        
        # Parent panels laid out by root's construct
        self.assertEqual(root.children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[1].layoutEngineUsed, construct)
        self.assertEqual(root.children[2].layoutEngineUsed, construct)
        
        # Single children laid out by their parent's layout engine
        self.assertEqual(root.children[0].children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[1].children[0].layoutEngineUsed, fixed)
        self.assertEqual(root.children[2].children[0].layoutEngineUsed, stack)
    
    def testLayoutWithZeroScaleParent(self):
        """Test layout engines work correctly with zero-scale parent panels."""
        root = self.parsePanel('''<root>
            <panel size="0,0,0" layout="construct">
                <panel/>
                <panel/>
            </panel>
            <panel size="0,0,0" layout="fixed">
                <panel/>
            </panel>
            <panel size="0,0,0" layout="stack">
                <panel/>
            </panel>
        </root>''')
        
        construct = GXMLLayout.get_bound_layout_processor(GXMLConstructLayout.layoutScheme)
        stack = GXMLLayout.get_bound_layout_processor(GXMLStackLayout.layoutScheme)
        fixed = GXMLLayout.get_bound_layout_processor(GXMLFixedLayout.layoutScheme)
        
        # Zero-scale parents are still laid out normally
        self.assertEqual(root.children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[1].layoutEngineUsed, construct)
        self.assertEqual(root.children[2].layoutEngineUsed, construct)
        
        # Their children use the parent's layout engine
        self.assertEqual(root.children[0].children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[0].children[1].layoutEngineUsed, construct)
        self.assertEqual(root.children[1].children[0].layoutEngineUsed, fixed)
        self.assertEqual(root.children[2].children[0].layoutEngineUsed, stack)
    
    def testNestedZeroScaleWithLayoutChanges(self):
        """Test nested zero-scale panels with layout engine changes."""
        root = self.parsePanel('''<root>
            <panel size="0,0,0" layout="construct">
                <panel size="0,0,0" layout="fixed">
                    <panel layout="stack">
                        <panel/>
                    </panel>
                </panel>
            </panel>
        </root>''')
        
        construct = GXMLLayout.get_bound_layout_processor(GXMLConstructLayout.layoutScheme)
        stack = GXMLLayout.get_bound_layout_processor(GXMLStackLayout.layoutScheme)
        fixed = GXMLLayout.get_bound_layout_processor(GXMLFixedLayout.layoutScheme)
        
        # Verify layout engines propagate correctly through zero-scale hierarchy
        self.assertEqual(root.children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[0].children[0].layoutEngineUsed, construct)
        self.assertEqual(root.children[0].children[0].children[0].layoutEngineUsed, fixed)
        self.assertEqual(root.children[0].children[0].children[0].children[0].layoutEngineUsed, stack)


if __name__ == '__main__':
    unittest.main()
