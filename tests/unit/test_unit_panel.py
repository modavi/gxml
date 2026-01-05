"""
Unit tests for GXMLPanel orientation and direction methods.

Tests panel axis directions, face normals, and face detection in world space,
including behavior under rotation transformations.
"""
import unittest
import numpy as np
from elements.gxml_panel import GXMLPanel, PanelSide
from gxml_types import Side, GXMLTransform
from tests.helpers import assert_corner_points


class PanelSideGeometryTests(unittest.TestCase):
    """Unit tests for panel side geometry methods."""
    
    def test_get_local_points_from_side(self):
        """get_local_points_from_side should return correct CCW-ordered points for all sides."""
        panel = GXMLPanel()
        
        self.assertEqual(panel.get_local_points_from_side(PanelSide.START), [(0,0,0), (0,0,1), (0,1,1), (0,1,0)])
        self.assertEqual(panel.get_local_points_from_side(PanelSide.END), [(1,0,1), (1,0,0), (1,1,0), (1,1,1)])
        self.assertEqual(panel.get_local_points_from_side(PanelSide.BOTTOM), [(0,0,0), (1,0,0), (1,0,1), (0,0,1)])
        self.assertEqual(panel.get_local_points_from_side(PanelSide.TOP), [(0,1,0), (0,1,1), (1,1,1), (1,1,0)])
        self.assertEqual(panel.get_local_points_from_side(PanelSide.BACK), [(1,0,0), (0,0,0), (0,1,0), (1,1,0)])
        self.assertEqual(panel.get_local_points_from_side(PanelSide.FRONT), [(0,0,1), (1,0,1), (1,1,1), (0,1,1)])
    
    def testCreatePanelSideFullBounds(self):
        """createPanelSide should create correct geometry with full bounds."""
        panel = GXMLPanel()
        panel.id = "parent_panel"
        panel.thickness = 1.0
        
        # Test START side
        side_panel = panel.create_panel_side("start", PanelSide.START)
        self.assertIn(side_panel, panel.dynamicChildren)
        self.assertEqual(side_panel.id, "parent_panel")
        self.assertEqual(side_panel.subId, "start")
        assert_corner_points(self, side_panel, [0, 0, -0.5], [0, 0, 0.5], [0, 1, 0.5], [0, 1, -0.5])
        
        # Test FRONT side
        side_panel = panel.create_panel_side("front", PanelSide.FRONT)
        self.assertIn(side_panel, panel.dynamicChildren)
        self.assertEqual(side_panel.id, "parent_panel")
        self.assertEqual(side_panel.subId, "front")
        assert_corner_points(self, side_panel, [0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5])
    
    def test_create_panel_side_with_partial_bounds(self):
        """create_panel_side should respect corners for partial segments."""
        panel = GXMLPanel()
        panel.thickness = 1.0
        
        # corners = [(t0,s0), (t1,s1), (t2,s2), (t3,s3)] 
        # = [start-bottom, end-bottom, end-top, start-top]
        
        # Test FRONT side - t varies with x, s varies with y
        side_panel = panel.create_panel_side("front-partial", PanelSide.FRONT,
                                          corners=[(0.25, 0.2), (0.75, 0.2), (0.75, 0.8), (0.25, 0.8)])
        assert_corner_points(self, side_panel,
                           [0.25, 0.2, 0.5], [0.75, 0.2, 0.5], [0.75, 0.8, 0.5], [0.25, 0.8, 0.5])
        
        # Test BACK side - same as FRONT but z=-0.5
        side_panel = panel.create_panel_side("back-partial", PanelSide.BACK,
                                          corners=[(0.25, 0.2), (0.75, 0.2), (0.75, 0.8), (0.25, 0.8)])
        # BACK local points are in different order: (1,0,0), (0,0,0), (0,1,0), (1,1,0)
        assert_corner_points(self, side_panel,
                           [0.75, 0.2, -0.5], [0.25, 0.2, -0.5], [0.25, 0.8, -0.5], [0.75, 0.8, -0.5])
        
        # Test TOP side - t varies with x, s represents thickness (0=back, 1=front)
        # For TOP, y is fixed at the 's' value (0.8 here - 80% up the panel)
        # corners format: [(t_start, s_back), (t_end, s_back), (t_end, s_front), (t_start, s_front)]
        side_panel = panel.create_panel_side("top-partial", PanelSide.TOP,
                                          corners=[(0.25, 0), (0.75, 0), (0.75, 1), (0.25, 1)])
        # Expected: s=0 maps to z=-0.5 (back), s=1 maps to z=0.5 (front), y fixed at 1
        # But we're testing partial bounds, so we need y=0.8
        # Actually for TOP face, y is fixed at 1.0 (the face is at y=1)
        # The 's' values in corners determine z (thickness)
        assert_corner_points(self, side_panel,
                           [0.25, 1, -0.5], [0.25, 1, 0.5], [0.75, 1, 0.5], [0.75, 1, -0.5])
        
        # Test BOTTOM side - same pattern, y fixed at 0
        side_panel = panel.create_panel_side("bottom-partial", PanelSide.BOTTOM,
                                          corners=[(0.25, 0), (0.75, 0), (0.75, 1), (0.25, 1)])
        assert_corner_points(self, side_panel,
                           [0.25, 0, -0.5], [0.75, 0, -0.5], [0.75, 0, 0.5], [0.25, 0, 0.5])
        
        # Test START side - x is fixed at 0, s varies with y
        # For START, t values are ignored (x=0 always), s controls y position
        side_panel = panel.create_panel_side("start-partial", PanelSide.START,
                                          corners=[(0, 0.2), (0, 0.2), (0, 0.8), (0, 0.8)])
        assert_corner_points(self, side_panel,
                           [0, 0.2, -0.5], [0, 0.2, 0.5], [0, 0.8, 0.5], [0, 0.8, -0.5])
        
        # Test END side - x is fixed at 1, s varies with y
        side_panel = panel.create_panel_side("end-partial", PanelSide.END,
                                          corners=[(1, 0.2), (1, 0.2), (1, 0.8), (1, 0.8)])
        assert_corner_points(self, side_panel,
                           [1, 0.2, 0.5], [1, 0.2, -0.5], [1, 0.8, -0.5], [1, 0.8, 0.5])
    
    def test_create_panel_side_id_inheritance(self):
        """create_panel_side should inherit parent id and set unique subId."""
        panel = GXMLPanel()
        panel.id = "wall_panel_5"
        panel.thickness = 1.0
        
        # Create multiple side panels
        start_side = panel.create_panel_side("start_cap", PanelSide.START)
        end_side = panel.create_panel_side("end_cap", PanelSide.END)
        top_side = panel.create_panel_side("top_trim", PanelSide.TOP)
        
        # All should inherit parent ID
        self.assertEqual(start_side.id, "wall_panel_5")
        self.assertEqual(end_side.id, "wall_panel_5")
        self.assertEqual(top_side.id, "wall_panel_5")
        
        # Each should have unique subId
        self.assertEqual(start_side.subId, "start_cap")
        self.assertEqual(end_side.subId, "end_cap")
        self.assertEqual(top_side.subId, "top_trim")
        
        # Verify all are children of parent
        self.assertIn(start_side, panel.dynamicChildren)
        self.assertIn(end_side, panel.dynamicChildren)
        self.assertIn(top_side, panel.dynamicChildren)


class PanelThicknessTests(unittest.TestCase):
    """Unit tests for panel thickness property and its effects."""
    
    def testDefaultThickness(self):
        """Default thickness should be 0.0."""
        panel = GXMLPanel()
        self.assertEqual(panel.thickness, 0.0)
    
    def testSetThickness(self):
        """Thickness should be settable and retrievable."""
        panel = GXMLPanel()
        
        panel.thickness = 1.5
        self.assertEqual(panel.thickness, 1.5)
        
        panel.thickness = 0.0
        self.assertEqual(panel.thickness, 0.0)
        
        panel.thickness = 10.0
        self.assertEqual(panel.thickness, 10.0)
    
    def testThicknessAffectsSideGeometry(self):
        """Thickness should affect the Z-offset of panel sides."""
        panel = GXMLPanel()
        panel.thickness = 2.0
        
        # FRONT side should be at +thickness/2 = +1.0
        front_side = panel.create_panel_side("front", PanelSide.FRONT)
        assert_corner_points(self, front_side, [0, 0, 1.0], [1, 0, 1.0], [1, 1, 1.0], [0, 1, 1.0])
        
        # BACK side should be at -thickness/2 = -1.0
        back_side = panel.create_panel_side("back", PanelSide.BACK)
        assert_corner_points(self, back_side, [1, 0, -1.0], [0, 0, -1.0], [0, 1, -1.0], [1, 1, -1.0])
    
    def testThicknessWithScale(self):
        """Thickness is applied in local space, then scaled by transform."""
        panel = GXMLPanel()
        panel.thickness = 2.0
        
        # Scale by 3x in all directions
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (3, 3, 3))
        panel.recalculate_transform()
        
        # FRONT side: local Z=+1.0, then scaled by 3 = world Z=+3.0
        front_side = panel.create_panel_side("front", PanelSide.FRONT)
        assert_corner_points(self, front_side, [0, 0, 3.0], [3, 0, 3.0], [3, 3, 3.0], [0, 3, 3.0])
        
        # BACK side: local Z=-1.0, then scaled by 3 = world Z=-3.0
        back_side = panel.create_panel_side("back", PanelSide.BACK)
        assert_corner_points(self, back_side, [3, 0, -3.0], [0, 0, -3.0], [0, 3, -3.0], [3, 3, -3.0])
    
    def testThicknessWithNonUniformScale(self):
        """Thickness scales with Z-axis scale specifically."""
        panel = GXMLPanel()
        panel.thickness = 1.0
        
        # Scale: X=2, Y=3, Z=5
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (2, 3, 5))
        panel.recalculate_transform()
        
        # FRONT side: local Z=+0.5, scaled by 5 = world Z=+2.5
        # X scaled by 2, Y scaled by 3
        front_side = panel.create_panel_side("front", PanelSide.FRONT)
        assert_corner_points(self, front_side, [0, 0, 2.5], [2, 0, 2.5], [2, 3, 2.5], [0, 3, 2.5])
    
    def testThicknessWithRotation(self):
        """Thickness direction rotates with panel."""
        panel = GXMLPanel()
        panel.thickness = 2.0
        
        # Rotate 90° around Y-axis: Z becomes X
        panel.transform.apply_local_transformations((0, 0, 0), (0, 90, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        # FRONT side: local Z=+1.0 rotates to world X=+1.0
        front_side = panel.create_panel_side("front", PanelSide.FRONT)
        assert_corner_points(self, front_side, [1, 0, 0], [1, 0, -1], [1, 1, -1], [1, 1, 0])
        
        # BACK side: local Z=-1.0 rotates to world X=-1.0
        back_side = panel.create_panel_side("back", PanelSide.BACK)
        assert_corner_points(self, back_side, [-1, 0, -1], [-1, 0, 0], [-1, 1, 0], [-1, 1, -1])
    
    def testZeroThickness(self):
        """Zero thickness should place front and back at same location."""
        panel = GXMLPanel()
        panel.thickness = 0.0
        
        # Both FRONT and BACK should be at Z=0
        front_side = panel.create_panel_side("front", PanelSide.FRONT)
        assert_corner_points(self, front_side, [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0])
        
        back_side = panel.create_panel_side("back", PanelSide.BACK)
        assert_corner_points(self, back_side, [1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0])
    
    def testThicknessIndependentOfTransform(self):
        """Thickness property is independent of transform property."""
        panel = GXMLPanel()
        panel.thickness = 5.0
        
        # Change transform
        panel.transform.apply_local_transformations((10, 20, 30), (45, 90, 135), (2, 3, 4))
        panel.recalculate_transform()
        
        # Thickness should remain unchanged
        self.assertEqual(panel.thickness, 5.0)
        
        # Change thickness
        panel.thickness = 0.5
        
        # Transform should remain unchanged
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [10, 20, 30]))
        self.assertTrue(np.allclose(r, [45, 90, 135]))
        self.assertTrue(np.allclose(s, [2, 3, 4]))


class PanelValidityTests(unittest.TestCase):
    """Unit tests for panel validity checking."""
    
    def testValidPanelReturnsTrue(self):
        """Test that a normal panel with non-zero dimensions is valid."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        self.assertTrue(panel.is_valid())
    
    def testZeroPrimaryAxisReturnsFalse(self):
        """Test that a panel with zero-length primary axis (X scale=0) is invalid."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (0, 1, 1))
        panel.recalculate_transform()
        
        self.assertFalse(panel.is_valid())
    
    def testZeroSecondaryAxisReturnsFalse(self):
        """Test that a panel with zero-length secondary axis (Y scale=0) is invalid."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (1, 0, 1))
        panel.recalculate_transform()
        
        self.assertFalse(panel.is_valid())
    
    def testCustomToleranceRespected(self):
        """Test that custom tolerance is respected in validity check."""
        panel = GXMLPanel()
        # Very small scale on primary axis
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (0.001, 1, 1))
        panel.recalculate_transform()
        
        # With default tolerance (1e-4), should be valid
        self.assertTrue(panel.is_valid())
        
        # With larger tolerance, should be invalid
        self.assertFalse(panel.is_valid(tolerance=0.01))


class PanelOrientationTests(unittest.TestCase):
    """Unitan tests for panel orientation methods."""
    
    def assertVectorEqual(self, v1, v2, msg=None):
        """Assert that two vectors are approximately equal."""
        np.testing.assert_allclose(v1, v2, rtol=1e-5, atol=1e-8, err_msg=msg)
    
    def testPanelAxesUnrotated(self):
        """Test all three panel axes for unrotated panel."""
        panel = GXMLPanel()
        
        # Primary axis (local X) should point in +X direction
        self.assertVectorEqual(panel.get_primary_axis(), [1, 0, 0], "Primary axis should be +X")
        
        # Secondary axis (local Y) should point in +Y direction
        self.assertVectorEqual(panel.get_secondary_axis(), [0, 1, 0], "Secondary axis should be +Y")
        
        # Normal axis (local Z) should point in +Z direction
        self.assertVectorEqual(panel.get_normal_axis(), [0, 0, 1], "Normal axis should be +Z")
    
    def testPanelAxesRotated90(self):
        """Test panel axes after 90° Y-axis rotation."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((0, 0, 0), (0, 90, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        # After 90° Y rotation: X->-Z, Y->Y, Z->X
        self.assertVectorEqual(panel.get_primary_axis(), [0, 0, -1], "Primary axis should be -Z after 90° rotation")
        self.assertVectorEqual(panel.get_secondary_axis(), [0, 1, 0], "Secondary axis remains +Y")
        self.assertVectorEqual(panel.get_normal_axis(), [1, 0, 0], "Normal axis should be +X after 90° rotation")
    
    def testPanelAxesRotated180(self):
        """Test panel axes after 180° Y-axis rotation."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((0, 0, 0), (0, 180, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        # After 180° Y rotation: X->-X, Y->Y, Z->-Z
        self.assertVectorEqual(panel.get_primary_axis(), [-1, 0, 0], "Primary axis should be -X after 180° rotation")
        self.assertVectorEqual(panel.get_secondary_axis(), [0, 1, 0], "Secondary axis remains +Y")
        self.assertVectorEqual(panel.get_normal_axis(), [0, 0, -1], "Normal axis should be -Z after 180° rotation")
    
    def testFaceNormalsUnrotated(self):
        """Test all face normals for unrotated panel."""
        panel = GXMLPanel()
        panel.thickness = 1.0
        
        # Front/Back normals (along normal axis +/-Z)
        self.assertVectorEqual(panel.get_face_normal(PanelSide.FRONT), [0, 0, 1], "FRONT normal should be +Z")
        self.assertVectorEqual(panel.get_face_normal(PanelSide.BACK), [0, 0, -1], "BACK normal should be -Z")
        
        # Start/End normals (along primary axis +/-X)
        self.assertVectorEqual(panel.get_face_normal(PanelSide.START), [-1, 0, 0], "START normal should be -X")
        self.assertVectorEqual(panel.get_face_normal(PanelSide.END), [1, 0, 0], "END normal should be +X")
        
        # Top/Bottom normals (along secondary axis +/-Y)
        self.assertVectorEqual(panel.get_face_normal(PanelSide.TOP), [0, 1, 0], "TOP normal should be +Y")
        self.assertVectorEqual(panel.get_face_normal(PanelSide.BOTTOM), [0, -1, 0], "BOTTOM normal should be -Y")
    
    def testFaceNormalsRotated(self):
        """Test face normals under various rotations."""
        # 90° rotation: Z->X, so FRONT normal goes from +Z to +X
        panel = GXMLPanel()
        panel.thickness = 1.0
        panel.transform.apply_local_transformations((0, 0, 0), (0, 90, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        self.assertVectorEqual(panel.get_face_normal(PanelSide.FRONT), [1, 0, 0], "FRONT normal should be +X after 90°")
        self.assertVectorEqual(panel.get_face_normal(PanelSide.BACK), [-1, 0, 0], "BACK normal should be -X after 90°")
        
        # 180° rotation: Z->-Z, so FRONT normal goes from +Z to -Z
        panel = GXMLPanel()
        panel.thickness = 1.0
        panel.transform.apply_local_transformations((0, 0, 0), (0, 180, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        self.assertVectorEqual(panel.get_face_normal(PanelSide.FRONT), [0, 0, -1], "FRONT normal should be -Z after 180°")
        self.assertVectorEqual(panel.get_face_normal(PanelSide.BACK), [0, 0, 1], "BACK normal should be +Z after 180°")
    
    def testFaceClosestToDirection(self):
        """Test face selection based on direction for various panel rotations."""
        # Unrotated panel
        panel = GXMLPanel()
        panel.thickness = 1.0
        
        self.assertEqual(panel.get_face_closest_to_direction([0, 0, 1]), PanelSide.FRONT, "Direction +Z should select FRONT")
        self.assertEqual(panel.get_face_closest_to_direction([0, 0, -1]), PanelSide.BACK, "Direction -Z should select BACK")
        self.assertEqual(panel.get_face_closest_to_direction([0, 1, 0]), PanelSide.TOP, "Direction +Y should select TOP")
        
        # 90° rotated panel: FRONT face normal is now +X
        panel = GXMLPanel()
        panel.thickness = 1.0
        panel.transform.apply_local_transformations((0, 0, 0), (0, 90, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        self.assertEqual(panel.get_face_closest_to_direction([1, 0, 0]), PanelSide.FRONT, "Direction +X should select FRONT after 90°")
        self.assertEqual(panel.get_face_closest_to_direction([-1, 0, 0]), PanelSide.BACK, "Direction -X should select BACK after 90°")
        
        # 180° rotated panel: FRONT face normal is now -Z
        panel = GXMLPanel()
        panel.thickness = 1.0
        panel.transform.apply_local_transformations((0, 0, 0), (0, 180, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        self.assertEqual(panel.get_face_closest_to_direction([0, 0, -1]), PanelSide.FRONT, "Direction -Z should select FRONT after 180°")
        self.assertEqual(panel.get_face_closest_to_direction([0, 0, 1]), PanelSide.BACK, "Direction +Z should select BACK after 180°")
    
    def testFaceClosestToDirectionZeroDirection(self):
        """Test that get_face_closest_to_direction handles zero-length direction vectors."""
        panel = GXMLPanel()
        panel.id = "test"
        panel.transform = GXMLTransform()
        
        # Zero direction should normalize to zero, not crash with division by zero
        result = panel.get_face_closest_to_direction([0, 0, 0])
        
        # Should return a valid PanelSide even with zero direction
        self.assertIsInstance(result, PanelSide)

    def testFaceClosestToDirectionNormalization(self):
        """Test that direction vectors are normalized before comparison."""
        panel = GXMLPanel()
        panel.id = "test"
        panel.transform = GXMLTransform()
        
        # Test with unnormalized direction (should give same result as normalized)
        unnormalized = panel.get_face_closest_to_direction([10, 0, 0])
        normalized = panel.get_face_closest_to_direction([1, 0, 0])
        
        self.assertEqual(unnormalized, normalized)


class PanelTransformTests(unittest.TestCase):
    """Unit tests for panel transformation operations."""
    
    # Basic setter tests
    def testSetTranslation(self):
        """transform.apply_local_transformations should update translation and recalculate transform matrix."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations((10, 20, 30), panel.transform.rotation, panel.transform.scale)
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [10, 20, 30]), f"Expected [10, 20, 30], got {t}")
        self.assertTrue(np.allclose(r, [0, 0, 0]), "Rotation should remain at [0, 0, 0]")
        self.assertTrue(np.allclose(s, [1, 1, 1]), "Scale should remain at [1, 1, 1]")
        
        # Verify transformPoint reflects the translation
        transformed = panel.transform_point([0, 0, 0])
        self.assertTrue(np.allclose(transformed, [10, 20, 30], atol=1e-6),
                       f"Origin should transform to [10, 20, 30], got {transformed}")
        
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [11, 20, 30], atol=1e-6),
                       f"Point [1,0,0] should transform to [11, 20, 30], got {transformed}")
    
    def testSetRotation(self):
        """transform.apply_local_transformations should update rotation and recalculate transform matrix."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations(panel.transform.translation, (0, 90, 0), panel.transform.scale)
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [0, 0, 0]), "Translation should remain at [0, 0, 0]")
        self.assertTrue(np.allclose(r, [0, 90, 0]), f"Expected rotation [0, 90, 0], got {r}")
        self.assertTrue(np.allclose(s, [1, 1, 1]), "Scale should remain at [1, 1, 1]")
        
        # Verify 90° Y rotation: +X becomes -Z, +Z becomes +X
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [0, 0, -1], atol=1e-6),
                       f"+X should rotate to -Z, got {transformed}")
        
        transformed = panel.transform_point([0, 0, 1])
        self.assertTrue(np.allclose(transformed, [1, 0, 0], atol=1e-6),
                       f"+Z should rotate to +X, got {transformed}")
    
    def testSetScaleUniform(self):
        """transform.apply_local_transformations with uniform scale should apply uniform scaling."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations(panel.transform.translation, panel.transform.rotation, (2.5, 2.5, 2.5))
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [0, 0, 0]), "Translation should remain at [0, 0, 0]")
        self.assertTrue(np.allclose(r, [0, 0, 0]), "Rotation should remain at [0, 0, 0]")
        self.assertTrue(np.allclose(s, [2.5, 2.5, 2.5]), f"Expected uniform scale [2.5, 2.5, 2.5], got {s}")
        
        # Verify transformPoint reflects the scaling
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [2.5, 0, 0], atol=1e-6),
                       f"Point [1,0,0] should scale to [2.5, 0, 0], got {transformed}")
        
        transformed = panel.transform_point([0, 1, 0])
        self.assertTrue(np.allclose(transformed, [0, 2.5, 0], atol=1e-6),
                       f"Point [0,1,0] should scale to [0, 2.5, 0], got {transformed}")
    
    def testSetScaleNonUniform(self):
        """transform.apply_local_transformations with non-uniform scale should apply non-uniform scaling."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations(panel.transform.translation, panel.transform.rotation, (2, 3, 4))
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [0, 0, 0]), "Translation should remain at [0, 0, 0]")
        self.assertTrue(np.allclose(r, [0, 0, 0]), "Rotation should remain at [0, 0, 0]")
        self.assertTrue(np.allclose(s, [2, 3, 4]), f"Expected scale [2, 3, 4], got {s}")
        
        # Verify each axis scales independently
        self.assertTrue(np.allclose(panel.transform_point([1, 0, 0]), [2, 0, 0], atol=1e-6))
        self.assertTrue(np.allclose(panel.transform_point([0, 1, 0]), [0, 3, 0], atol=1e-6))
        self.assertTrue(np.allclose(panel.transform_point([0, 0, 1]), [0, 0, 4], atol=1e-6))
    
    def testSetTransformAllParameters(self):
        """transform.apply_local_transformations with all parameters should update translation, rotation, and scale."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations((5, 10, 15), (0, 45, 0), (2.0, 2.0, 2.0))
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [5, 10, 15]), f"Expected translation [5, 10, 15], got {t}")
        self.assertTrue(np.allclose(r, [0, 45, 0]), f"Expected rotation [0, 45, 0], got {r}")
        self.assertTrue(np.allclose(s, [2, 2, 2]), f"Expected scale [2, 2, 2], got {s}")
    
    def testSetTransformPartialUpdate(self):
        """transform.apply_local_transformations with selective updates should work with stored values."""
        panel = GXMLPanel()
        
        # Set initial values
        panel.transform.apply_local_transformations((10, 20, 30), (0, 0, 90), (3.0, 3.0, 3.0))
        panel.recalculate_transform()
        
        # Update only rotation
        panel.transform.apply_local_transformations(panel.transform.translation, (45, 0, 0), panel.transform.scale)
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [10, 20, 30]), "Translation should be preserved")
        self.assertTrue(np.allclose(r, [45, 0, 0]), "Rotation should be updated")
        self.assertTrue(np.allclose(s, [3, 3, 3]), "Scale should be preserved")
        
        # Update only translation and scale
        panel.transform.apply_local_transformations((0, 0, 0), panel.transform.rotation, (1, 2, 1))
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [0, 0, 0]), "Translation should be updated")
        self.assertTrue(np.allclose(r, [45, 0, 0]), "Rotation should be preserved")
        self.assertTrue(np.allclose(s, [1, 2, 1]), "Scale should be updated")
    
    def testGetTransform(self):
        """get_transform should return current translation, rotation, and scale as tuple."""
        panel = GXMLPanel()
        
        # Default values
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [0, 0, 0]), "Default translation should be [0, 0, 0]")
        self.assertTrue(np.allclose(r, [0, 0, 0]), "Default rotation should be [0, 0, 0]")
        self.assertTrue(np.allclose(s, [1, 1, 1]), "Default scale should be [1, 1, 1]")
        
        # After setting values
        panel.transform.apply_local_transformations((7, 8, 9), (30, 60, 90), (0.5, 1.5, 2.5))
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [7, 8, 9]), f"Should return set translation, got {t}")
        self.assertTrue(np.allclose(r, [30, 60, 90]), f"Should return set rotation, got {r}")
        self.assertTrue(np.allclose(s, [0.5, 1.5, 2.5]), f"Should return set scale, got {s}")
    
    # Combined transformation tests
    def testTranslationAndRotation(self):
        """Combined translation and rotation should apply correctly."""
        panel = GXMLPanel()
        
        # Translate then rotate 90° around Y
        panel.transform.apply_local_transformations((10, 0, 0), (0, 90, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        # Local [1, 0, 0] rotates to world [0, 0, -1], then translates to [10, 0, -1]
        # Actually, transformation order is: rotate first (in local space), then translate
        # So [1, 0, 0] → rotate 90° Y → [0, 0, -1] → translate [10, 0, 0] → [10, 0, -1]
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [10, 0, -1], atol=1e-6),
                       f"Expected [10, 0, -1], got {transformed}")
    
    def testTranslationAndScale(self):
        """Combined translation and scale should apply correctly."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations((5, 10, 15), (0, 0, 0), (2.0, 2.0, 2.0))
        panel.recalculate_transform()
        
        # Scale applies first, then translation
        # [1, 0, 0] → scale 2x → [2, 0, 0] → translate [5, 10, 15] → [7, 10, 15]
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [7, 10, 15], atol=1e-6),
                       f"Expected [7, 10, 15], got {transformed}")
        
        transformed = panel.transform_point([0, 1, 0])
        self.assertTrue(np.allclose(transformed, [5, 12, 15], atol=1e-6),
                       f"Expected [5, 12, 15], got {transformed}")
    
    def testRotationAndScale(self):
        """Combined rotation and scale should apply correctly."""
        panel = GXMLPanel()
        
        # Scale by 2, rotate 90° around Y
        panel.transform.apply_local_transformations((0, 0, 0), (0, 90, 0), (2.0, 2.0, 2.0))
        panel.recalculate_transform()
        
        # [1, 0, 0] → scale 2x → [2, 0, 0] → rotate 90° Y → [0, 0, -2]
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [0, 0, -2], atol=1e-6),
                       f"Expected [0, 0, -2], got {transformed}")
    
    def testTranslationRotationScale(self):
        """All three transformations combined should apply in correct order."""
        panel = GXMLPanel()
        
        # Scale, then rotate, then translate
        panel.transform.apply_local_transformations((10, 0, 0), (0, 90, 0), (3.0, 3.0, 3.0))
        panel.recalculate_transform()
        
        # [1, 0, 0] → scale 3x → [3, 0, 0] → rotate 90° Y → [0, 0, -3] → translate → [10, 0, -3]
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [10, 0, -3], atol=1e-6),
                       f"Expected [10, 0, -3], got {transformed}")
    
    # Sequential transformation tests
    def testMultipleTranslations(self):
        """Multiple applyLocalTransformations calls should replace, not accumulate."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations((10, 0, 0), panel.transform.rotation, panel.transform.scale)
        panel.recalculate_transform()
        panel.transform.apply_local_transformations((5, 0, 0), panel.transform.rotation, panel.transform.scale)
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [5, 0, 0]), f"Translation should be [5, 0, 0], got {t}")
        
        transformed = panel.transform_point([0, 0, 0])
        self.assertTrue(np.allclose(transformed, [5, 0, 0], atol=1e-6),
                       "Origin should be at [5, 0, 0]")
    
    def testMultipleRotations(self):
        """Multiple applyLocalTransformations calls should replace, not accumulate."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations(panel.transform.translation, (0, 90, 0), panel.transform.scale)
        panel.recalculate_transform()
        panel.transform.apply_local_transformations(panel.transform.translation, (0, 45, 0), panel.transform.scale)
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(r, [0, 45, 0]), f"Rotation should be [0, 45, 0], got {r}")
    
    def testMultipleScales(self):
        """Multiple applyLocalTransformations calls should replace, not accumulate."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations(panel.transform.translation, panel.transform.rotation, (2.0, 2.0, 2.0))
        panel.recalculate_transform()
        panel.transform.apply_local_transformations(panel.transform.translation, panel.transform.rotation, (3.0, 3.0, 3.0))
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(s, [3, 3, 3]), f"Scale should be [3, 3, 3], got {s}")
        
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [3, 0, 0], atol=1e-6),
                       "Point should be scaled by 3, not 6")
    
    def testMixedSequentialUpdates(self):
        """Sequential applyLocalTransformations calls should preserve other components."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations((10, 20, 30), panel.transform.rotation, panel.transform.scale)
        panel.recalculate_transform()
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(r, [0, 0, 0]), "Rotation should still be default after setting translation")
        self.assertTrue(np.allclose(s, [1, 1, 1]), "Scale should still be default after setting translation")
        
        panel.transform.apply_local_transformations(panel.transform.translation, (0, 90, 0), panel.transform.scale)
        panel.recalculate_transform()
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [10, 20, 30]), "Translation should be preserved after setting rotation")
        self.assertTrue(np.allclose(s, [1, 1, 1]), "Scale should still be default after setting rotation")
        
        panel.transform.apply_local_transformations(panel.transform.translation, panel.transform.rotation, (2.0, 2.0, 2.0))
        panel.recalculate_transform()
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [10, 20, 30]), "Translation should be preserved after setScale")
        self.assertTrue(np.allclose(r, [0, 90, 0]), "Rotation should be preserved after setScale")
    
    # Edge case tests
    def testZeroScale(self):
        """Zero scale in any axis should be handled gracefully."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations(panel.transform.translation, panel.transform.rotation, (0, 1, 1))
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(s, [0, 1, 1]), "Zero scale should be stored")
        
        # Points along X should collapse to origin's X coordinate
        transformed = panel.transform_point([5, 0, 0])
        self.assertAlmostEqual(transformed[0], 0, places=6, msg="X should be scaled to 0")
    
    def testNegativeScale(self):
        """Negative scale should flip/mirror transformation."""
        panel = GXMLPanel()
        
        panel.transform.apply_local_transformations(panel.transform.translation, panel.transform.rotation, (-1, 1, 1))
        panel.recalculate_transform()
        
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [-1, 0, 0], atol=1e-6),
                       "Negative X scale should mirror point")
    
    def testLargeRotationAngles(self):
        """Rotation angles > 360° should normalize correctly."""
        panel = GXMLPanel()
        
        # 450° = 90° (one full rotation plus 90°)
        panel.transform.apply_local_transformations(panel.transform.translation, (0, 450, 0), panel.transform.scale)
        panel.recalculate_transform()
        
        # Should behave same as 90° rotation
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [0, 0, -1], atol=1e-6),
                       "450° rotation should be equivalent to 90°")
    
    def testNegativeRotationAngles(self):
        """Negative rotation angles should work correctly."""
        panel = GXMLPanel()
        
        # -90° around Y should be opposite of +90°
        panel.transform.apply_local_transformations(panel.transform.translation, (0, -90, 0), panel.transform.scale)
        panel.recalculate_transform()
        
        # +X should rotate to +Z (opposite of 90° which goes to -Z)
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [0, 0, 1], atol=1e-6),
                       "-90° Y rotation should rotate +X to +Z")
    
    def testIdentityTransform(self):
        """Default transform should be identity (no change)."""
        panel = GXMLPanel()
        
        # Without any setter calls, transform should be identity
        transformed = panel.transform_point([1, 2, 3])
        self.assertTrue(np.allclose(transformed, [1, 2, 3], atol=1e-6),
                       "Identity transform should not change points")
    
    def testResetToIdentity(self):
        """Setting to default values should restore identity transform."""
        panel = GXMLPanel()
        
        # Apply various transformations
        panel.transform.apply_local_transformations((10, 20, 30), (45, 90, 135), (5.0, 5.0, 5.0))
        panel.recalculate_transform()
        
        # Reset to identity
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (1.0, 1.0, 1.0))
        panel.recalculate_transform()
        
        t, r, s = panel.get_transform()
        self.assertTrue(np.allclose(t, [0, 0, 0]), "Translation should be reset")
        self.assertTrue(np.allclose(r, [0, 0, 0]), "Rotation should be reset")
        self.assertTrue(np.allclose(s, [1, 1, 1]), "Scale should be reset")
        
        transformed = panel.transform_point([1, 2, 3])
        self.assertTrue(np.allclose(transformed, [1, 2, 3], atol=1e-6),
                       "Reset transform should be identity")


class PanelPivotTests(unittest.TestCase):
    """Unit tests for panel transformation pivot point behavior."""
    
    def testDefaultPivotAtOrigin(self):
        """Default pivot should be at local origin (0, 0, 0)."""
        panel = GXMLPanel()
        
        self.assertTrue(np.allclose(panel.transform.pivot, [0, 0, 0]),
                       "Default pivot should be at origin")
    
    def testPivotWithRotation(self):
        """Rotation should occur around the pivot point."""
        panel = GXMLPanel()
        
        # Set pivot to center of unit square
        panel.transform.pivot = (0.5, 0.5, 0)
        
        # Rotate 90 degrees around Y axis
        panel.transform.apply_local_transformations((0, 0, 0), (0, 90, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        # Pivot transformation order: translate by -pivot, then rotate
        # Point [0,0,0] -> translate by [-0.5,-0.5,0] = [-0.5,-0.5,0]
        # -> rotate 90 deg around Y: [0, -0.5, 0.5]
        transformed = panel.transform_point([0, 0, 0])
        self.assertTrue(np.allclose(transformed, [0, -0.5, 0.5], atol=1e-6),
                       f"Bottom-left corner should be at [0, -0.5, 0.5], got {transformed}")
        
        # Point [1,0,0] -> translate by [-0.5,-0.5,0] = [0.5,-0.5,0]
        # -> rotate 90 deg around Y: [0, -0.5, -0.5]
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [0, -0.5, -0.5], atol=1e-6),
                       f"Bottom-right corner should be at [0, -0.5, -0.5], got {transformed}")
    
    def testPivotWithScale(self):
        """Scaling should occur around the pivot point."""
        panel = GXMLPanel()
        
        # Set pivot to center of unit square
        panel.transform.pivot = (0.5, 0.5, 0)
        
        # Scale by 2x
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (2, 2, 2))
        panel.recalculate_transform()
        
        # Pivot transformation: translate by -pivot, then scale
        # Point [0,0,0] -> translate by [-0.5,-0.5,0] = [-0.5,-0.5,0]
        # -> scale by 2: [-1,-1,0]
        transformed = panel.transform_point([0, 0, 0])
        self.assertTrue(np.allclose(transformed, [-1, -1, 0], atol=1e-6),
                       f"Scaled bottom-left should be at [-1, -1, 0], got {transformed}")
        
        # Point [1,0,0] -> translate by [-0.5,-0.5,0] = [0.5,-0.5,0]
        # -> scale by 2: [1,-1,0]
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [1, -1, 0], atol=1e-6),
                       f"Scaled bottom-right should be at [1, -1, 0], got {transformed}")
        
        # Point [1,1,0] -> translate by [-0.5,-0.5,0] = [0.5,0.5,0]
        # -> scale by 2: [1,1,0]
        transformed = panel.transform_point([1, 1, 0])
        self.assertTrue(np.allclose(transformed, [1, 1, 0], atol=1e-6),
                       f"Scaled top-right should be at [1, 1, 0], got {transformed}")
    
    def testPivotWithTranslation(self):
        """Pivot affects all transformations including translation."""
        panel = GXMLPanel()
        
        # Set pivot to center
        panel.transform.pivot = (0.5, 0.5, 0)
        
        # Apply translation only
        panel.transform.apply_local_transformations((10, 20, 30), (0, 0, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        # Pivot transformation: translate by -pivot, then apply local translation
        # Point [0,0,0] -> translate by [-0.5,-0.5,0] = [-0.5,-0.5,0]
        # -> translate by [10,20,30]: [9.5, 19.5, 30]
        transformed = panel.transform_point([0, 0, 0])
        self.assertTrue(np.allclose(transformed, [9.5, 19.5, 30], atol=1e-6),
                       f"Translated point should be at [9.5, 19.5, 30], got {transformed}")
        
        # Point [1,0,0] -> translate by [-0.5,-0.5,0] = [0.5,-0.5,0]
        # -> translate by [10,20,30]: [10.5, 19.5, 30]
        transformed = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(transformed, [10.5, 19.5, 30], atol=1e-6),
                       f"Translated point should be at [10.5, 19.5, 30], got {transformed}")
    
    def testPivotWithCombinedTransforms(self):
        """Pivot affects all transformations: scale, rotation, and translation."""
        panel = GXMLPanel()
        
        # Set pivot to center
        panel.transform.pivot = (0.5, 0.5, 0)
        
        # Combine scale, rotation, and translation
        panel.transform.apply_local_transformations((10, 0, 0), (0, 90, 0), (2, 2, 2))
        panel.recalculate_transform()
        
        # Pivot transformation: translate by -pivot, scale, rotate, then translate
        # Point [0,0,0] -> translate by [-0.5,-0.5,0] = [-0.5,-0.5,0]
        #               -> scale by 2: [-1,-1,0]
        #               -> rotate 90 deg Y: [0,-1,1]
        #               -> translate by [10,0,0]: [10,-1,1]
        transformed = panel.transform_point([0, 0, 0])
        self.assertTrue(np.allclose(transformed, [10, -1, 1], atol=1e-6),
                       f"Combined transform point should be at [10, -1, 1], got {transformed}")
    
    def testChangingPivot(self):
        """Changing pivot should affect subsequent transformations."""
        panel = GXMLPanel()
        
        # First: scale with default pivot (origin)
        panel.transform.pivot = (0, 0, 0)
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (2, 2, 2))
        panel.recalculate_transform()
        
        point = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(point, [2, 0, 0], atol=1e-6),
                       "With pivot at origin, [1,0,0] should scale to [2,0,0]")
        
        # Now change pivot and apply same scale
        panel.transform.pivot = (0.5, 0.5, 0)
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (2, 2, 2))
        panel.recalculate_transform()
        
        # Point [1,0,0] -> translate by [-0.5,-0.5,0] = [0.5,-0.5,0]
        # -> scale by 2: [1,-1,0]
        point = panel.transform_point([1, 0, 0])
        self.assertTrue(np.allclose(point, [1, -1, 0], atol=1e-6),
                       f"With pivot at center, [1,0,0] should scale to [1,-1,0], got {point}")


class SceneToLocalTransformTests(unittest.TestCase):
    """Unit tests for sceneToLocal coordinate transformation."""
    
    def testSceneToLocalIdentityTransform(self):
        """sceneToLocal with identity transform should return the same point."""
        panel = GXMLPanel()
        panel.recalculate_transform()
        
        localPoint = panel.transform.sceneToLocal([2, 3, 4])
        self.assertTrue(np.allclose(localPoint, [2, 3, 4]))
    
    def testSceneToLocalWithTranslation(self):
        """sceneToLocal should inverse translate points back to local space."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((10, 20, 30), (0, 0, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        # A point at [15, 25, 35] in world space should be [5, 5, 5] in local space
        localPoint = panel.transform.sceneToLocal([15, 25, 35])
        self.assertTrue(np.allclose(localPoint, [5, 5, 5]))
        
        # The world origin should be at [-10, -20, -30] in local space
        localPoint = panel.transform.sceneToLocal([0, 0, 0])
        self.assertTrue(np.allclose(localPoint, [-10, -20, -30]))
    
    def testSceneToLocalWithScale(self):
        """sceneToLocal should inverse scale points back to local space."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (5, 5, 1))
        panel.recalculate_transform()
        
        # A point at [4, 2.5, 0] in world space should be [0.8, 0.5, 0] in local space
        localPoint = panel.transform.sceneToLocal([4, 2.5, 0])
        self.assertTrue(np.allclose(localPoint, [0.8, 0.5, 0]))
        
        # Test another point
        localPoint = panel.transform.sceneToLocal([10, 5, 0])
        self.assertTrue(np.allclose(localPoint, [2, 1, 0]))
    
    def testSceneToLocalWithRotation(self):
        """sceneToLocal should inverse rotate points back to local space."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((0, 0, 0), (0, 90, 0), (1, 1, 1))
        panel.recalculate_transform()
        
        # After 90° Y rotation, +X in local goes to -Z in world and +Z in local goes to +X in world
        # So inverse: -Z in world came from +X in local, and +X in world came from +Z in local
        
        # A point at [0, 0, -1] in world space should be [1, 0, 0] in local space
        localPoint = panel.transform.sceneToLocal([0, 0, -1])
        self.assertTrue(np.allclose(localPoint, [1, 0, 0], atol=1e-6))
        
        # A point at [1, 0, 0] in world space should be [0, 0, 1] in local space
        localPoint = panel.transform.sceneToLocal([1, 0, 0])
        self.assertTrue(np.allclose(localPoint, [0, 0, 1], atol=1e-6))
    
    def testSceneToLocalWithCombinedTransforms(self):
        """sceneToLocal should handle combined translation, rotation, and scale."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((5, 0, 0), (0, 90, 0), (2, 2, 2))
        panel.recalculate_transform()
        
        # The panel's origin (0,0,0) in local space is at (5,0,0) in world space
        localPoint = panel.transform.sceneToLocal([5, 0, 0])
        self.assertTrue(np.allclose(localPoint, [0, 0, 0], atol=1e-6))
        
        # Point [1,0,0] in local space -> scale to [2,0,0] -> rotate to [0,0,-2] -> translate to [5,0,-2]
        # So [5,0,-2] in world space should be [1,0,0] in local space
        localPoint = panel.transform.sceneToLocal([5, 0, -2])
        self.assertTrue(np.allclose(localPoint, [1, 0, 0], atol=1e-6))
    
    def testSceneToLocalZeroScale(self):
        """sceneToLocal with zero scale should transform all points to origin."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((0, 0, 0), (0, 0, 0), (0, 0, 0))
        panel.recalculate_transform()
        
        # Any world point should map to local origin when scale is zero
        localPoint = panel.transform.sceneToLocal([100, 200, 300])
        self.assertTrue(np.allclose(localPoint, [0, 0, 0]))
        
        localPoint = panel.transform.sceneToLocal([0, 0, 0])
        self.assertTrue(np.allclose(localPoint, [0, 0, 0]))
    
    def testSceneToLocalInverseOfTransformPoint(self):
        """sceneToLocal should be the inverse of transformPoint."""
        panel = GXMLPanel()
        panel.transform.apply_local_transformations((3, 5, 7), (15, 30, 45), (2, 3, 1.5))
        panel.recalculate_transform()
        
        # Test several local points
        for local_pt in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5], [1, 1, 1]]:
            # Transform to world space
            world_pt = panel.transform_point(local_pt)
            
            # Transform back to local space
            recovered_local = panel.transform.sceneToLocal(world_pt)
            
            # Should recover the original point
            self.assertTrue(np.allclose(recovered_local, local_pt, atol=1e-6),
                          f"Local {local_pt} -> World {world_pt} -> Local {recovered_local}")


class GetFaceCenterLocalTests(unittest.TestCase):
    """Unit tests for get_face_center_local method."""
    
    def test_all_face_centers(self):
        """Each face center should return correct (t, s, z) offset."""
        panel = GXMLPanel()
        panel.thickness = 2.0
        half = 1.0  # half_thickness
        
        self.assertEqual(panel.get_face_center_local(PanelSide.FRONT),  (0.5, 0.5, half))
        self.assertEqual(panel.get_face_center_local(PanelSide.BACK),   (0.5, 0.5, -half))
        self.assertEqual(panel.get_face_center_local(PanelSide.TOP),    (0.5, 1.0, 0.0))
        self.assertEqual(panel.get_face_center_local(PanelSide.BOTTOM), (0.5, 0.0, 0.0))
        self.assertEqual(panel.get_face_center_local(PanelSide.START),  (0.0, 0.5, 0.0))
        self.assertEqual(panel.get_face_center_local(PanelSide.END),    (1.0, 0.5, 0.0))
    
    def test_zero_thickness_panel(self):
        """Zero thickness panel should have FRONT/BACK at z=0."""
        panel = GXMLPanel()
        panel.thickness = 0.0
        
        _, _, z_front = panel.get_face_center_local(PanelSide.FRONT)
        _, _, z_back = panel.get_face_center_local(PanelSide.BACK)
        
        self.assertEqual(z_front, 0.0)
        self.assertEqual(z_back, 0.0)
    
    def test_center_transforms_to_face_plane(self):
        """Face center should transform to a point on the actual face plane."""
        panel = GXMLPanel()
        panel.thickness = 1.0
        
        # FRONT face center should be at z=0.5 in world space (half thickness)
        center = panel.get_face_center_local(PanelSide.FRONT)
        world_point = panel.transform_point(center)
        self.assertAlmostEqual(world_point[2], 0.5)
        
        # BACK face center should be at z=-0.5
        center = panel.get_face_center_local(PanelSide.BACK)
        world_point = panel.transform_point(center)
        self.assertAlmostEqual(world_point[2], -0.5)


if __name__ == '__main__':
    unittest.main()

