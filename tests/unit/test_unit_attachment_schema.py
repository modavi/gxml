"""
Unit tests for the panel attachment schema.

This file tests the attribute naming convention for panel attachments and spanning:
- attach-id, attach-self, attach-point: Positioning relationship (where I connect)
- span-id, span-self, span-point: Sizing relationship (what determines my size)
- Shorthand syntax: attach="parent:right", span="root:left"

Schema principles:
- "attach" relates to positioning (where this panel connects to another)
- "span" relates to sizing (what panel determines this panel's dimensions)
- "-id" suffix: which panel (target reference)
- "-self" suffix: where on THIS panel (local point)
- "-point" suffix: where on THAT panel (target point)
"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from elements.gxml_root import GXMLRoot
from elements.gxml_panel import GXMLPanel
from gxml_layout import GXMLLayout
from layouts.gxml_construct_layout import GXMLConstructLayout
from gxml_types import Offset, Axis
from tests.test_fixtures.mocks import GXMLMockPanel, GXMLMockParsingContext


class AttachIdTests(unittest.TestCase):
    """Tests for attach-id attribute - which panel to connect to."""
    
    def setUp(self):
        self.layout = GXMLConstructLayout()
        self.root = GXMLRoot()
        self.root.childLayoutScheme = "stack"
    
    def _setup_panels(self, *panels):
        for panel in panels:
            self.layout.apply_default_layout_properties(panel)
    
    def test_attach_id_resolves_to_named_element(self):
        """attach-id should resolve to the referenced panel element."""
        p1 = GXMLMockPanel("panel1")
        p2 = GXMLMockPanel("panel2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"panel1": p1, "panel2": p2}
        ctx.setAttribute("attach-id", "panel1")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertEqual(p2.attachElement, p1)
    
    def test_attach_id_tilde_uses_previous_element(self):
        """attach-id='~' should use the previous sibling element."""
        p1 = GXMLMockPanel("panel1")
        p2 = GXMLMockPanel("panel2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"panel1": p1, "panel2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("attach-id", "~")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertEqual(p2.attachElement, p1)
    
    def test_missing_attach_id_defaults_to_previous(self):
        """Missing attach-id should default to the previous sibling element."""
        p1 = GXMLMockPanel("panel1")
        p2 = GXMLMockPanel("panel2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"panel1": p1, "panel2": p2}
        ctx.prevElem = p1
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertEqual(p2.attachElement, p1)
    
    def test_attach_id_numeric(self):
        """attach-id should work with numeric string IDs."""
        p1 = GXMLMockPanel("0")
        p2 = GXMLMockPanel("1")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"0": p1, "1": p2}
        ctx.setAttribute("attach-id", "0")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertEqual(p2.attachElement, p1)


class AttachPointTests(unittest.TestCase):
    """Tests for attach-point attribute - where on the target panel to connect."""
    
    def setUp(self):
        self.layout = GXMLConstructLayout()
    
    def _setup_panels(self, *panels):
        for panel in panels:
            self.layout.apply_default_layout_properties(panel)
    
    def test_attach_point_numeric(self):
        """attach-point='0.5' should position at middle of panel."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("attach-point", "0.5")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertAlmostEqual(p2.attachOffset[0], 0.5, places=5)
    
    def test_attach_point_left(self):
        """attach-point='left' should position at (0, 0.5, 0)."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("attach-point", "left")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertAlmostEqual(p2.attachOffset[0], 0.0, places=5)
    
    def test_attach_point_right(self):
        """attach-point='right' should position at (1, 0.5, 0)."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("attach-point", "right")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertAlmostEqual(p2.attachOffset[0], 1.0, places=5)
    
    def test_attach_point_center(self):
        """attach-point='center' should position at (0.5, 0.5, 0)."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("attach-point", "center")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertAlmostEqual(p2.attachOffset[0], 0.5, places=5)
    
    def test_attach_point_top_left(self):
        """attach-point='top-left' should position at (0, 1, 0)."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("attach-point", "top-left")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertAlmostEqual(p2.attachOffset[0], 0.0, places=5)
        self.assertAlmostEqual(p2.attachOffset[1], 1.0, places=5)
    
    def test_attach_point_bottom_right(self):
        """attach-point='bottom-right' should position at (1, 0, 0)."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("attach-point", "bottom-right")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertAlmostEqual(p2.attachOffset[0], 1.0, places=5)
        self.assertAlmostEqual(p2.attachOffset[1], 0.0, places=5)


class AttachSelfTests(unittest.TestCase):
    """Tests for attach-self attribute - where on THIS panel the connection originates."""
    
    def setUp(self):
        self.layout = GXMLConstructLayout()
    
    def _setup_panels(self, *panels):
        for panel in panels:
            self.layout.apply_default_layout_properties(panel)
    
    def test_attach_self_right(self):
        """attach-self='right' should set pivot to (1, 0, 0)."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("attach-self", "right")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertAlmostEqual(p2.transform.pivot[0], 1.0, places=5)
    
    def test_attach_self_center(self):
        """attach-self='center' should set pivot to (0.5, 0.5, 0)."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("attach-self", "center")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertAlmostEqual(p2.transform.pivot[0], 0.5, places=5)


class SpanIdTests(unittest.TestCase):
    """Tests for span-id attribute - which panel determines size."""
    
    def setUp(self):
        self.layout = GXMLConstructLayout()
    
    def _setup_panels(self, *panels):
        for panel in panels:
            self.layout.apply_default_layout_properties(panel)
    
    def test_span_id_resolves_to_element(self):
        """span-id should resolve to the referenced panel for sizing."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        p3 = GXMLMockPanel("p3")
        self._setup_panels(p1, p2, p3)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2, "p3": p3}
        ctx.prevElem = p1
        ctx.setAttribute("span-id", "p2")
        
        self.layout.parse_layout_attributes(p3, ctx)
        
        self.assertEqual(p3.spanElement, p2)


class SpanPointTests(unittest.TestCase):
    """Tests for span-point attribute - where on span element to size to."""
    
    def setUp(self):
        self.layout = GXMLConstructLayout()
    
    def _setup_panels(self, *panels):
        for panel in panels:
            self.layout.apply_default_layout_properties(panel)
    
    def test_span_point_numeric(self):
        """span-point='1.0' should set spanOffset to (1, 0, 0)."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("span-id", "p1")
        ctx.setAttribute("span-point", "1.0")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertAlmostEqual(p2.spanOffset[0], 1.0, places=5)
    
    def test_span_point_right(self):
        """span-point='right' should set spanOffset to (1, 0.5, 0)."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("span-id", "p1")
        ctx.setAttribute("span-point", "right")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertAlmostEqual(p2.spanOffset[0], 1.0, places=5)
    
    def test_span_point_auto(self):
        """span-point='auto' should enable auto-detection for sizing target."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("span-id", "p1")
        ctx.setAttribute("span-point", "auto")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertTrue(p2.spanOffset.auto)


class SpanSelfTests(unittest.TestCase):
    """Tests for span-self attribute - where on THIS panel for sizing."""
    
    def setUp(self):
        self.layout = GXMLConstructLayout()
    
    def _setup_panels(self, *panels):
        for panel in panels:
            self.layout.apply_default_layout_properties(panel)
    
    def test_span_self_sets_local_sizing_point(self):
        """span-self='left' should set spanSelfOffset to (0, 0, 0)."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("span-id", "p1")
        ctx.setAttribute("span-self", "left")
        ctx.setAttribute("span-point", "right")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertTrue(hasattr(p2, 'spanSelfOffset'))
        self.assertAlmostEqual(p2.spanSelfOffset[0], 0.0, places=5)


class AttachShorthandTests(unittest.TestCase):
    """Tests for attach shorthand syntax: attach='id:point'."""
    
    def setUp(self):
        self.layout = GXMLConstructLayout()
    
    def _setup_panels(self, *panels):
        for panel in panels:
            self.layout.apply_default_layout_properties(panel)
    
    def test_attach_shorthand_id_and_point(self):
        """attach='parent:right' should expand to attach-id and attach-point."""
        p1 = GXMLMockPanel("parent")
        p2 = GXMLMockPanel("child")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"parent": p1, "child": p2}
        ctx.setAttribute("attach", "parent:right")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertEqual(p2.attachElement, p1)
        self.assertAlmostEqual(p2.attachOffset[0], 1.0, places=5)
    
    def test_attach_shorthand_id_only(self):
        """attach='header' should use id with default point (right)."""
        p1 = GXMLMockPanel("header")
        p2 = GXMLMockPanel("content")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"header": p1, "content": p2}
        ctx.setAttribute("attach", "header")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertEqual(p2.attachElement, p1)
        self.assertAlmostEqual(p2.attachOffset[0], 1.0, places=5)
    
    def test_attach_shorthand_point_only(self):
        """attach=':right' should use contextual id with explicit point."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1, "p2": p2}
        ctx.prevElem = p1
        ctx.setAttribute("attach", ":right")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertEqual(p2.attachElement, p1)
        self.assertAlmostEqual(p2.attachOffset[0], 1.0, places=5)
    
    def test_attach_shorthand_numeric_point(self):
        """attach='wall:0.5' should parse numeric point correctly."""
        p1 = GXMLMockPanel("wall")
        p2 = GXMLMockPanel("door")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"wall": p1, "door": p2}
        ctx.setAttribute("attach", "wall:0.5")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertEqual(p2.attachElement, p1)
        self.assertAlmostEqual(p2.attachOffset[0], 0.5, places=5)
    
    def test_attach_shorthand_numeric_id_and_point(self):
        """attach='0:0.5' should handle both numeric id and numeric point."""
        p1 = GXMLMockPanel("0")
        p2 = GXMLMockPanel("1")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"0": p1, "1": p2}
        ctx.setAttribute("attach", "0:0.5")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertEqual(p2.attachElement, p1)
        self.assertAlmostEqual(p2.attachOffset[0], 0.5, places=5)


class SpanShorthandTests(unittest.TestCase):
    """Tests for span shorthand syntax: span='id:point'."""
    
    def setUp(self):
        self.layout = GXMLConstructLayout()
    
    def _setup_panels(self, *panels):
        for panel in panels:
            self.layout.apply_default_layout_properties(panel)
    
    def test_span_shorthand_id_and_point(self):
        """span='root:left' should expand to span-id and span-point."""
        p_root = GXMLMockPanel("root")
        p_child = GXMLMockPanel("child")
        self._setup_panels(p_root, p_child)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"root": p_root, "child": p_child}
        ctx.prevElem = p_root
        ctx.setAttribute("span", "root:left")
        
        self.layout.parse_layout_attributes(p_child, ctx)
        
        self.assertEqual(p_child.spanElement, p_root)
        self.assertAlmostEqual(p_child.spanOffset[0], 0.0, places=5)
    
    def test_span_shorthand_id_only(self):
        """span='footer' should use id with default point."""
        p1 = GXMLMockPanel("footer")
        p2 = GXMLMockPanel("content")
        self._setup_panels(p1, p2)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"footer": p1, "content": p2}
        ctx.setAttribute("span", "footer")
        
        self.layout.parse_layout_attributes(p2, ctx)
        
        self.assertEqual(p2.spanElement, p1)


class EdgeCaseTests(unittest.TestCase):
    """Tests for edge cases and error handling."""
    
    def setUp(self):
        self.layout = GXMLConstructLayout()
    
    def _setup_panels(self, *panels):
        for panel in panels:
            self.layout.apply_default_layout_properties(panel)
    
    def test_attach_to_nonexistent_panel(self):
        """Attaching to a nonexistent panel should handle gracefully."""
        p1 = GXMLMockPanel("p1")
        self._setup_panels(p1)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1}
        ctx.setAttribute("attach-id", "nonexistent")
        
        self.layout.parse_layout_attributes(p1, ctx)
        
        self.assertIsNone(p1.attachElement)
    
    def test_self_referential_attach(self):
        """Panel attaching to itself should handle gracefully."""
        p1 = GXMLMockPanel("p1")
        self._setup_panels(p1)
        
        ctx = GXMLMockParsingContext()
        ctx.elementMap = {"p1": p1}
        ctx.setAttribute("attach-id", "p1")
        
        self.layout.parse_layout_attributes(p1, ctx)
        
        # Should be None or self, but not crash
        self.assertTrue(p1.attachElement is None or p1.attachElement is p1)
    
    def test_circular_attachment_chain(self):
        """Circular attachment references should not cause infinite loops."""
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        
        p1.parent = p2.parent = GXMLMockPanel("root")
        p1.parent.children = [p1, p2]
        
        self._setup_panels(p1, p2)
        
        p1.attachElement = p2
        p2.attachElement = p1
        p2.attachedElements.append(p1)
        p1.attachedElements.append(p2)
        
        try:
            result = self.layout.find_attached_elements(p1)
            self.assertTrue(True)  # Didn't crash
        except RecursionError:
            self.fail("Circular attachment should not cause infinite recursion")


if __name__ == '__main__':
    unittest.main()
