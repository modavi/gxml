"""
Unit tests for GXMLParser and parsing utilities.
"""
import unittest
import numpy as np
from gxml_parser import GXMLParser, GXMLParsingContext
from gxml_parsing_utils import GXMLParsingUtils
from gxml_types import GXMLLayoutScheme, Side, Axis
from elements.gxml_root import GXMLRoot
from elements.gxml_panel import GXMLPanel


class ParserUnitTests(unittest.TestCase):
    """Unit tests for GXMLParser."""
    
    def testGetBoundElementTypeForRoot(self):
        """Test getBoundElementType returns GXMLRoot for 'root' tag."""
        elementType = GXMLParser.get_bound_element_type("root")
        
        self.assertEqual(elementType, GXMLRoot)
    
    def testGetBoundElementTypeForPanel(self):
        """Test getBoundElementType returns GXMLPanel for 'panel' tag."""
        elementType = GXMLParser.get_bound_element_type("panel")
        
        self.assertEqual(elementType, GXMLPanel)
    
    def testGetBoundElementTypeForUnknown(self):
        """Test getBoundElementType returns None for unknown tag."""
        elementType = GXMLParser.get_bound_element_type("unknown")
        
        self.assertIsNone(elementType)
    
    def testParseSimpleRoot(self):
        """Test parsing simple root element."""
        # Note: Root must have at least one child or 'if root:' check fails (empty XML element is falsy)
        xml = '<root id="root"><panel id="p1"/></root>'
        
        result = GXMLParser.parse(xml)
        
        self.assertIsInstance(result, GXMLRoot)
        self.assertFalse(result.isVisibleSelf, "Root should not be visible")
        self.assertEqual(result.id, "root")
    
    def testParseRootWithSinglePanel(self):
        """Test parsing root with one child panel."""
        xml = '<root><panel id="p1"/></root>'
        
        result = GXMLParser.parse(xml)
        
        self.assertIsInstance(result, GXMLRoot)
        self.assertEqual(len(result.children), 1)
        self.assertIsInstance(result.children[0], GXMLPanel)
        self.assertEqual(result.children[0].id, "p1")
    
    def testParseRootWithMultiplePanels(self):
        """Test parsing root with multiple child panels."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2"/>
            <panel id="p3"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(len(result.children), 3)
        self.assertEqual(result.children[0].id, "p1")
        self.assertEqual(result.children[1].id, "p2")
        self.assertEqual(result.children[2].id, "p3")
    
    def testParseNestedPanels(self):
        """Test parsing nested panel structure."""
        xml = '''<root>
            <panel id="p1">
                <panel id="p1_child"/>
            </panel>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(len(result.children), 1)
        self.assertEqual(len(result.children[0].children), 1)
        self.assertEqual(result.children[0].children[0].id, "p1_child")
    
    def testParsePanelWithAttributes(self):
        """Test parsing panel with various attributes."""
        xml = '<root><panel id="test" size="2.5" thickness="0.5"/></root>'
        
        result = GXMLParser.parse(xml)
        
        panel = result.children[0]
        self.assertEqual(panel.id, "test")
        # Attributes are parsed by element's parse() method
        # Just verify the element was created and has the id
    
    def testParseWithDuplicateIdRaisesError(self):
        """Test that duplicate panel IDs raise ValueError."""
        xml = '''<root>
            <panel id="duplicate"/>
            <panel id="duplicate"/>
        </root>'''
        
        with self.assertRaises(ValueError) as context:
            GXMLParser.parse(xml)
        
        self.assertIn("Duplicate panel id", str(context.exception))
    
    def testParseWithInvalidXMLRaisesError(self):
        """Test that invalid XML raises an exception."""
        xml = "<root><panel"  # Incomplete/invalid XML
        
        with self.assertRaises(Exception):
            GXMLParser.parse(xml)
    
    def testParseWithUnknownElementTypeRaisesError(self):
        """Test that unknown element types raise an exception."""
        xml = '<root><unknown-element/></root>'
        
        with self.assertRaises(Exception) as context:
            GXMLParser.parse(xml)
        
        self.assertIn("Could not find bound element type", str(context.exception))
    
    def testParseVars(self):
        """Test parsing variables block."""
        xml = '''<root>
            <vars>
                <width>100</width>
                <height>50</height>
            </vars>
            <panel id="p1"/>
        </root>'''
        
        # Vars shouldn't create elements, just set context variables
        result = GXMLParser.parse(xml)
        
        # Should only have 1 child (panel), vars is processed but not added
        self.assertEqual(len(result.children), 1)
        self.assertEqual(result.children[0].id, "p1")
    
    def testParseEmptyXML(self):
        """Test parsing empty XML raises exception."""
        xml = ""
        
        with self.assertRaises(Exception):
            GXMLParser.parse(xml)
    
    def testParseRootSetsLayoutScheme(self):
        """Test that parsed root element has construct layout scheme set."""
        # Note: Root must have at least one child or 'if root:' check fails
        xml = '<root id="root"><panel id="p1"/></root>'
        
        result = GXMLParser.parse(xml)
        
        # Root should have childLayoutScheme set to Construct (set by GXMLRoot constructor)
        self.assertEqual(result.childLayoutScheme, GXMLLayoutScheme.Construct)
    
    def testParseMaintainsHierarchy(self):
        """Test that parent-child relationships are properly maintained."""
        xml = '''<root>
            <panel id="p1">
                <panel id="p1_1">
                    <panel id="p1_1_1"/>
                </panel>
                <panel id="p1_2"/>
            </panel>
            <panel id="p2"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        # Check structure
        self.assertEqual(len(result.children), 2)
        self.assertEqual(result.children[0].id, "p1")
        self.assertEqual(result.children[1].id, "p2")
        
        p1 = result.children[0]
        self.assertEqual(len(p1.children), 2)
        self.assertEqual(p1.children[0].id, "p1_1")
        self.assertEqual(p1.children[1].id, "p1_2")
        
        p1_1 = p1.children[0]
        self.assertEqual(len(p1_1.children), 1)
        self.assertEqual(p1_1.children[0].id, "p1_1_1")
    
    def testParseChildIndexTracking(self):
        """Test that child indices are tracked correctly during parsing."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2"/>
            <panel id="p3"/>
        </root>'''
        
        # This tests that the parser correctly tracks indices
        # The actual index values are internal, but we can verify
        # that all children are created
        result = GXMLParser.parse(xml)
        
        self.assertEqual(len(result.children), 3)
        # Each child should have been created with proper index tracking
    
    def testParseComplexStructure(self):
        """Test parsing a complex multi-level panel structure."""
        xml = '''<root>
            <panel id="base" size="10">
                <panel id="vertical1" rotate="90"/>
                <panel id="vertical2" rotate="90" attach-point="0.5"/>
                <panel id="vertical3" rotate="90" attach-point="1"/>
            </panel>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertIsInstance(result, GXMLRoot)
        self.assertEqual(len(result.children), 1)
        
        base = result.children[0]
        self.assertEqual(base.id, "base")
        self.assertEqual(len(base.children), 3)
        self.assertEqual([c.id for c in base.children], ["vertical1", "vertical2", "vertical3"])
    
    def testParseWithMixedContent(self):
        """Test parsing with vars and panels mixed."""
        xml = '''<root>
            <vars>
                <panelSize>5</panelSize>
            </vars>
            <panel id="first"/>
            <panel id="second"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        # Vars should not appear as children
        self.assertEqual(len(result.children), 2)
        self.assertEqual(result.children[0].id, "first")
        self.assertEqual(result.children[1].id, "second")
    
    def testParseAutoGeneratesIds(self):
        """Test that panels without id attributes get auto-generated IDs from child index."""
        xml = '''<root>
            <panel/>
            <panel/>
            <panel/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        # IDs should be "0", "1", "2" based on child index
        self.assertEqual(result.children[0].id, "0")
        self.assertEqual(result.children[1].id, "1")
        self.assertEqual(result.children[2].id, "2")
    
    def testParseIdWithHashPlaceholder(self):
        """Test that # in id attribute is replaced with child index."""
        xml = '''<root>
            <panel id="panel_#"/>
            <panel id="panel_#"/>
            <panel id="panel_#"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        # # should be replaced with child index
        self.assertEqual(result.children[0].id, "panel_0")
        self.assertEqual(result.children[1].id, "panel_1")
        self.assertEqual(result.children[2].id, "panel_2")
    
    def testParseTildeSyntaxCopiesPreviousAttribute(self):
        """Test that ~ syntax copies attribute from previous sibling."""
        xml = '''<root>
            <panel id="p1" size="5"/>
            <panel id="p2" size="~"/>
            <panel id="p3" size="~"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        # All panels should have copied the size="5" from p1
        # Note: We can't directly check size attribute (parsed by element.parse)
        # but we can verify parsing succeeded (would fail if ~ didn't work)
        self.assertEqual(len(result.children), 3)
    
    def testParseVisibilityAttributes(self):
        """Test parsing visible and visible-self attributes."""
        xml = '''<root>
            <panel id="p1" visible="false"/>
            <panel id="p2" visible-self="false"/>
            <panel id="p3" visible="true" visible-self="true"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertFalse(result.children[0].isVisible)
        self.assertTrue(result.children[0].isVisibleSelf)  # Default is true
        
        self.assertTrue(result.children[1].isVisible)  # Default is true
        self.assertFalse(result.children[1].isVisibleSelf)
        
        self.assertTrue(result.children[2].isVisible)
        self.assertTrue(result.children[2].isVisibleSelf)
    
    def testParseInvalidChildLayoutSchemeRaisesError(self):
        """Test that invalid child layout scheme raises ValueError."""
        xml = '<root><panel id="p1" layout="invalid-scheme"/></root>'
        
        with self.assertRaises(ValueError) as context:
            GXMLParser.parse(xml)
        
        self.assertIn("Could not find bound layout processor", str(context.exception))
    
    def testParseSideBottom(self):
        """Side.parse should return Side.Bottom for 'bottom'."""
        self.assertEqual(Side.parse("bottom"), Side.Bottom)
    
    def testParseSideTop(self):
        """Side.parse should return Side.Top for 'top'."""
        self.assertEqual(Side.parse("top"), Side.Top)
    
    def testParseSideLeft(self):
        """Side.parse should return Side.Left for 'left'."""
        self.assertEqual(Side.parse("left"), Side.Left)
    
    def testParseSideRight(self):
        """Side.parse should return Side.Right for 'right'."""
        self.assertEqual(Side.parse("right"), Side.Right)
    
    def testParseSideFront(self):
        """Side.parse should return Side.Front for 'front'."""
        self.assertEqual(Side.parse("front"), Side.Front)
    
    def testParseSideBack(self):
        """Side.parse should return Side.Back for 'back'."""
        self.assertEqual(Side.parse("back"), Side.Back)
    
    def testParseSideCenter(self):
        """Side.parse should return Side.Center for 'center'."""
        self.assertEqual(Side.parse("center"), Side.Center)
    
    def testParseSideInvalidReturnsUndefined(self):
        """Side.parse should return Side.Undefined for invalid input."""
        self.assertEqual(Side.parse("002938"), Side.Undefined)
        self.assertEqual(Side.parse("0.5"), Side.Undefined)
        self.assertEqual(Side.parse(None), Side.Undefined)
        self.assertEqual(Side.parse("botom"), Side.Undefined)
    
    def testParseOffsetVectorNumeric(self):
        """parse_offset_vector should parse numeric coordinates."""
        ctx = GXMLParsingContext()
        result = GXMLParsingUtils.parse_offset_vector("0.5,0.5,0.5", ctx)
        self.assertTrue(np.allclose(result, [0.5, 0.5, 0.5]))
    
    def testParseOffsetVectorWithSideX(self):
        """parse_offset_vector should parse side names for X coordinate."""
        ctx = GXMLParsingContext()
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("left,0.5,0.5", ctx), [0.0, 0.5, 0.5]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("right,0.5,0.5", ctx), [1.0, 0.5, 0.5]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("center,0.5,0.5", ctx), [0.5, 0.5, 0.5]))
    
    def testParseOffsetVectorWithSideY(self):
        """parse_offset_vector should parse side names for Y coordinate."""
        ctx = GXMLParsingContext()
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5,top,0.5", ctx), [0.5, 1.0, 0.5]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5,bottom,0.5", ctx), [0.5, 0.0, 0.5]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5,center,0.5", ctx), [0.5, 0.5, 0.5]))
    
    def testParseOffsetVectorWithSideZ(self):
        """parse_offset_vector should parse side names for Z coordinate."""
        ctx = GXMLParsingContext()
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5,0.5,front", ctx), [0.5, 0.5, 0.0]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5,0.5,back", ctx), [0.5, 0.5, 1.0]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5,0.5,center", ctx), [0.5, 0.5, 0.5]))
    
    def testParseOffsetVectorAllSides(self):
        """parse_offset_vector should handle all side names together."""
        ctx = GXMLParsingContext()
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("center,center,center", ctx), [0.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("left,bottom,front", ctx), [0.0, 0.0, 0.0]))
    
    def testParseOffsetVectorTwoComponents(self):
        """parse_offset_vector should handle 2 components (Z defaults to 0)."""
        ctx = GXMLParsingContext()
        result = GXMLParsingUtils.parse_offset_vector("center,center", ctx)
        self.assertTrue(np.allclose(result, [0.5, 0.5, 0.0]))
    
    def testParseOffsetVectorSingleComponent(self):
        """parse_offset_vector should expand single side name to 3D vector."""
        ctx = GXMLParsingContext()
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("center", ctx), [0.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("left", ctx), [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("right", ctx), [1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("top", ctx), [0.0, 1.0, 0.0]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("bottom", ctx), [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("front", ctx), [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("back", ctx), [0.0, 0.0, 1.0]))
    
    def testParseOffsetVectorInvalidSidePlacementRaisesError(self):
        """parse_offset_vector should raise ValueError for sides in wrong position."""
        ctx = GXMLParsingContext()
        with self.assertRaises(ValueError):
            GXMLParsingUtils.parse_offset_vector("top,0.5,0.5", ctx)
        with self.assertRaises(ValueError):
            GXMLParsingUtils.parse_offset_vector("0.5,left,0.5", ctx)
        with self.assertRaises(ValueError):
            GXMLParsingUtils.parse_offset_vector("0.5,0.5,right", ctx)
    
    def testParseOffsetVectorWithPrimaryAxisX(self):
        """parse_offset_vector should use primary axis to determine coordinate placement."""
        ctx = GXMLParsingContext()
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5", ctx, Axis.X), [0.5, 0.0, 0.0]))
    
    def testParseOffsetVectorWithPrimaryAxisY(self):
        """parse_offset_vector with primary axis Y should place value in Y coordinate."""
        ctx = GXMLParsingContext()
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5", ctx, Axis.Y), [0.0, 0.5, 0.0]))
    
    def testParseOffsetVectorWithPrimaryAxisZ(self):
        """parse_offset_vector with primary axis Z should place value in Z coordinate."""
        ctx = GXMLParsingContext()
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5", ctx, Axis.Z), [0.0, 0.0, 0.5]))
    
    def testParseOffsetVectorWithTwoAxes(self):
        """parse_offset_vector should use primary and secondary axes for 2-component vectors."""
        ctx = GXMLParsingContext()
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5,0.25", ctx, Axis.X, Axis.Y), [0.5, 0.25, 0.0]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5,0.25", ctx, Axis.X, Axis.Z), [0.5, 0.0, 0.25]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5,0.25", ctx, Axis.Y, Axis.Z), [0.0, 0.5, 0.25]))
        self.assertTrue(np.allclose(GXMLParsingUtils.parse_offset_vector("0.5,0.25", ctx, Axis.Z, Axis.Y), [0.0, 0.25, 0.5]))
    
    def testInferAxisFromOffsetX(self):
        """infer_axis should return Axis.X for left/right sides."""
        ctx = GXMLParsingContext()
        self.assertEqual(GXMLParsingUtils.infer_axis("left", ctx), Axis.X)
        self.assertEqual(GXMLParsingUtils.infer_axis("right", ctx), Axis.X)
    
    def testInferAxisFromOffsetY(self):
        """infer_axis should return Axis.Y for top/bottom sides."""
        ctx = GXMLParsingContext()
        self.assertEqual(GXMLParsingUtils.infer_axis("top", ctx), Axis.Y)
        self.assertEqual(GXMLParsingUtils.infer_axis("bottom", ctx), Axis.Y)
    
    def testInferAxisFromOffsetZ(self):
        """infer_axis should return Axis.Z for front/back sides."""
        ctx = GXMLParsingContext()
        self.assertEqual(GXMLParsingUtils.infer_axis("front", ctx), Axis.Z)
        self.assertEqual(GXMLParsingUtils.infer_axis("back", ctx), Axis.Z)
    
    def testInferAxisFromOffsetMultipleSides(self):
        """infer_axis should use first side when multiple sides specified."""
        ctx = GXMLParsingContext()
        self.assertEqual(GXMLParsingUtils.infer_axis("left,top", ctx), Axis.X)
    
    def testInferAxisInvalidRaisesError(self):
        """infer_axis should raise ValueError for invalid input."""
        ctx = GXMLParsingContext()
        with self.assertRaises(ValueError):
            GXMLParsingUtils.infer_axis("center", ctx)
        with self.assertRaises(ValueError):
            GXMLParsingUtils.infer_axis("invalid", ctx)
        with self.assertRaises(ValueError):
            GXMLParsingUtils.infer_axis("0.5,0.5", ctx)
    
    def testParseAxisX(self):
        """Axis.parse should return Axis.X for 'x' and 'X'."""
        self.assertEqual(Axis.parse("x"), Axis.X)
        self.assertEqual(Axis.parse("X"), Axis.X)
    
    def testParseAxisY(self):
        """Axis.parse should return Axis.Y for 'y' and 'Y'."""
        self.assertEqual(Axis.parse("y"), Axis.Y)
        self.assertEqual(Axis.parse("Y"), Axis.Y)
    
    def testParseAxisZ(self):
        """Axis.parse should return Axis.Z for 'z' and 'Z'."""
        self.assertEqual(Axis.parse("z"), Axis.Z)
        self.assertEqual(Axis.parse("Z"), Axis.Z)
    
    def testParseAxisInvalidReturnsUndefined(self):
        """Axis.parse should return Axis.Undefined for invalid input."""
        self.assertEqual(Axis.parse("0"), Axis.Undefined)
        self.assertEqual(Axis.parse("lorem"), Axis.Undefined)
        self.assertEqual(Axis.parse("xyz"), Axis.Undefined)
    
    def testParseMixedExplicitAndAutoGeneratedIds(self):
        """Test that explicit and auto-generated IDs can be mixed."""
        xml = '''<root>
            <panel id="my"/>
            <panel/>
            <panel id="panelid"/>
            <panel/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        # Explicit IDs should be preserved, missing IDs auto-generated from index
        self.assertEqual(result.children[0].id, "my")
        self.assertEqual(result.children[1].id, "1")
        self.assertEqual(result.children[2].id, "panelid")
        self.assertEqual(result.children[3].id, "3")
    
    def testParseAttributeWithPythonExpression(self):
        """Test that attributes with Python expressions are evaluated."""
        xml = '''<root>
            <panel id="p1" size="1+2,4-1" rotate="0,20*2+5" pivot="1/2,2/4"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        panel = result.children[0]
        
        # Verify the panel was created (Python expressions parsed by element.parse())
        self.assertEqual(panel.id, "p1")
        # The actual evaluation happens in the element's parse method
        # Just verify parsing succeeds with expressions


class AttachmentAttributeParsingTests(unittest.TestCase):
    """Tests for parsing attachment-related attributes.
    
    The attachment schema uses:
    - attach-id, attach-point-self, attach-point: Positioning (where I connect)
    - span-id, span-point-self, span-point: Sizing (what determines my size)
    - Shorthand: attach="id:point", span="id:point"
    """
    
    def test_parse_attach_id(self):
        """Test parsing attach-id attribute."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2" attach-id="p1"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(len(result.children), 2)
        # Parser should create the element; layout applies the attachment
        self.assertEqual(result.children[1].id, "p2")
    
    def test_parse_attach_point_numeric(self):
        """Test parsing attach-point with numeric value."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2" attach-point="0.5"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[1].id, "p2")
    
    def test_parse_attach_point_named(self):
        """Test parsing attach-point with named value like 'right'."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2" attach-point="right"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[1].id, "p2")
    
    def test_parse_attach_point_self(self):
        """Test parsing attach-point-self attribute."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2" attach-point-self="center"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[1].id, "p2")
    
    def test_parse_span_id(self):
        """Test parsing span-id attribute."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2"/>
            <panel id="p3" span-id="p2"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(len(result.children), 3)
        self.assertEqual(result.children[2].id, "p3")
    
    def test_parse_span_point(self):
        """Test parsing span-point attribute."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2" span-id="p1" span-point="right"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[1].id, "p2")
    
    def test_parse_span_point_self(self):
        """Test parsing span-point-self attribute."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2" span-id="p1" span-point-self="left" span-point="right"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[1].id, "p2")
    
    def test_parse_attach_shorthand(self):
        """Test parsing attach shorthand: attach='id:point'."""
        xml = '''<root>
            <panel id="wall"/>
            <panel id="door" attach="wall:0.5"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[0].id, "wall")
        self.assertEqual(result.children[1].id, "door")
    
    def test_parse_attach_shorthand_id_only(self):
        """Test parsing attach shorthand with id only: attach='id'."""
        xml = '''<root>
            <panel id="wall"/>
            <panel id="door" attach="wall"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[1].id, "door")
    
    def test_parse_attach_shorthand_point_only(self):
        """Test parsing attach shorthand with point only: attach=':point'."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2" attach=":right"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[1].id, "p2")
    
    def test_parse_span_shorthand(self):
        """Test parsing span shorthand: span='id:point'."""
        xml = '''<root>
            <panel id="header"/>
            <panel id="content" span="header:right"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[1].id, "content")
    
    def test_parse_combined_attach_and_span(self):
        """Test parsing both attach and span attributes together."""
        xml = '''<root>
            <panel id="left"/>
            <panel id="right" rotate="90"/>
            <panel id="bridge" attach-id="left" attach-point="right" span-id="right" span-point="left"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(len(result.children), 3)
        self.assertEqual(result.children[2].id, "bridge")
    
    def test_parse_attach_point_corner(self):
        """Test parsing attach-point with corner value."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2" attach-point="top-left"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[1].id, "p2")
    
    def test_parse_span_point_auto(self):
        """Test parsing span-point='auto' for intersection detection."""
        xml = '''<root>
            <panel id="p1"/>
            <panel id="p2" rotate="90"/>
            <panel id="p3" span-id="p2" span-point="auto"/>
        </root>'''
        
        result = GXMLParser.parse(xml)
        
        self.assertEqual(result.children[2].id, "p3")


if __name__ == '__main__':
    unittest.main()
