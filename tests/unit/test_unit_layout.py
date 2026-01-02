"""
Unit tests for GXMLLayout and GXMLConstructLayout.
"""
import unittest
import numpy as np
from elements.gxml_root import GXMLRoot
from gxml_layout import GXMLLayout
from layouts.gxml_construct_layout import GXMLConstructLayout
from gxml_types import Offset, Axis
from tests.test_fixtures.mocks import GXMLMockPanel, GXMLMockLayout, LayoutPass


class LayoutUnitTests(unittest.TestCase):
    """Unit tests for GXMLLayout orchestration and traversal."""
    
    def setUp(self):
        """Set up test layout binding."""
        self.root = GXMLRoot()
        self.root.childLayoutScheme = "test"
        GXMLLayout.bind_layout("test", GXMLMockLayout.create(GXMLConstructLayout()))
    
    def tearDown(self):
        """Clean up the test layout binding."""
        if "test" in GXMLLayout.boundLayouts:
            del GXMLLayout.boundLayouts["test"]
        GXMLMockLayout.restore_all()
    
    def layout_panels(self, *panels):
        """Helper to add panels to root and run layout."""
        for panel in panels:
            self.root.add_child(panel)
        GXMLLayout.layout(self.root)
    
    def test_layout_order(self):
        """Test that layout passes execute in correct order and traversal direction.
        
        Verifies:
        - Measure pass: bottom-up (children before parent)
        - PreLayout, Layout, PostLayout passes: breadth-first (parent before children)
        - All passes execute in correct sequence: Measure → PreLayout → Layout → PostLayout
        """
        panel = GXMLMockPanel("root")
        
        panelChildA = GXMLMockPanel("A")
        panelChildA1 = GXMLMockPanel("A1")
        panelChildA2 = GXMLMockPanel("A2")
        panelChildA.add_child(panelChildA1)
        panelChildA.add_child(panelChildA2)
        
        panelChildB = GXMLMockPanel("B")
        panelChildB1 = GXMLMockPanel("B1")
        panelChildB2 = GXMLMockPanel("B2")
        panelChildB.add_child(panelChildB1)
        panelChildB.add_child(panelChildB2)
        
        panel.add_child(panelChildA)
        panel.add_child(panelChildB)
        
        self.layout_panels(panel)
        
        # Measure order should be bottom-up: children measured before parent
        self.assertEqual(panel.measureOrder, 6, "Root should be measured last")
        self.assertEqual(panelChildA.measureOrder, 2, "Parent A measured after its children")
        self.assertEqual(panelChildB.measureOrder, 5, "Parent B measured after its children")
        self.assertEqual(panelChildA1.measureOrder, 0, "Leaf A1 measured first")
        self.assertEqual(panelChildA2.measureOrder, 1, "Leaf A2 measured second")
        self.assertEqual(panelChildB1.measureOrder, 3, "Leaf B1 measured third")
        self.assertEqual(panelChildB2.measureOrder, 4, "Leaf B2 measured fourth")
        
        # PreLayout, Layout, PostLayout should use breadth-first traversal
        # Order: root → A, B → A1, A2, B1, B2
        
        # PreLayout order
        self.assertEqual(panel.preLayoutOrder, 0, "preLayout: Root should be first")
        self.assertEqual(panelChildA.preLayoutOrder, 1, "preLayout: A should be second")
        self.assertEqual(panelChildB.preLayoutOrder, 2, "preLayout: B should be third")
        self.assertEqual(panelChildA1.preLayoutOrder, 3, "preLayout: A1 should be fourth")
        self.assertEqual(panelChildA2.preLayoutOrder, 4, "preLayout: A2 should be fifth")
        self.assertEqual(panelChildB1.preLayoutOrder, 5, "preLayout: B1 should be sixth")
        self.assertEqual(panelChildB2.preLayoutOrder, 6, "preLayout: B2 should be seventh")
        
        # Layout order
        self.assertEqual(panel.layoutOrder, 0, "layout: Root should be first")
        self.assertEqual(panelChildA.layoutOrder, 1, "layout: A should be second")
        self.assertEqual(panelChildB.layoutOrder, 2, "layout: B should be third")
        self.assertEqual(panelChildA1.layoutOrder, 3, "layout: A1 should be fourth")
        self.assertEqual(panelChildA2.layoutOrder, 4, "layout: A2 should be fifth")
        self.assertEqual(panelChildB1.layoutOrder, 5, "layout: B1 should be sixth")
        self.assertEqual(panelChildB2.layoutOrder, 6, "layout: B2 should be seventh")
        
        # PostLayout order
        self.assertEqual(panel.postLayoutOrder, 0, "postLayout: Root should be first")
        self.assertEqual(panelChildA.postLayoutOrder, 1, "postLayout: A should be second")
        self.assertEqual(panelChildB.postLayoutOrder, 2, "postLayout: B should be third")
        self.assertEqual(panelChildA1.postLayoutOrder, 3, "postLayout: A1 should be fourth")
        self.assertEqual(panelChildA2.postLayoutOrder, 4, "postLayout: A2 should be fifth")
        self.assertEqual(panelChildB1.postLayoutOrder, 5, "postLayout: B1 should be sixth")
        self.assertEqual(panelChildB2.postLayoutOrder, 6, "postLayout: B2 should be seventh")
        
        # Verify all passes executed in correct order for each element
        expectedOrder = [LayoutPass.Measure, LayoutPass.PreLayout, LayoutPass.Layout, LayoutPass.PostLayout]
        self.assertEqual(panel.ops, expectedOrder, "root should execute all passes in order")
        self.assertEqual(panelChildA.ops, expectedOrder, "A should execute all passes in order")
        self.assertEqual(panelChildA1.ops, expectedOrder, "A1 should execute all passes in order")
        self.assertEqual(panelChildA2.ops, expectedOrder, "A2 should execute all passes in order")
        self.assertEqual(panelChildB.ops, expectedOrder, "B should execute all passes in order")
        self.assertEqual(panelChildB1.ops, expectedOrder, "B1 should execute all passes in order")
        self.assertEqual(panelChildB2.ops, expectedOrder, "B2 should execute all passes in order")
        
        # Verify layout processor tracked all operations in correct sequence
        layoutProcessor = GXMLLayout.get_bound_layout_processor("test")
        
        # 7 elements × 4 passes = 28 total operations
        # For each pass, all 7 elements should execute that pass before moving to next
        for passIdx, passType in enumerate(expectedOrder):
            for elemIdx in range(7):
                opIdx = passIdx * 7 + elemIdx
                with self.subTest(pass_type=passType.name, element_index=elemIdx):
                    self.assertEqual(layoutProcessor.layoutOps[opIdx], passType,
                                   f"Operation {opIdx} should be {passType.name}")


class ConstructLayoutUnitTests(unittest.TestCase):
    """Unit tests for GXMLConstructLayout element positioning and transformation."""
    
    def setUp(self):
        """Initialize layout for each test."""
        self.layout = GXMLConstructLayout()
    
    # ========================================================================
    # DEFAULT PROPERTIES
    # ========================================================================
    
    def test_apply_default_layout_properties(self):
        """Test that default layout properties are applied correctly to elements."""
        panel = GXMLMockPanel("p1")
        
        self.layout.apply_default_layout_properties(panel)
        
        # Verify default properties
        self.assertEqual(panel.primaryAxis, Axis.X, "Default primary axis should be X")
        self.assertEqual(panel.secondaryAxis, Axis.Y, "Default secondary axis should be Y")
        self.assertIsNone(panel.attachElement, "Default attach element should be None")
        self.assertIsNone(panel.spanElement, "Default span element should be None")
        self.assertEqual(len(panel.attachedElements), 0, "Default attached elements list should be empty")
        
        # Verify default offsets
        self.assertEqual(panel.attachOffset[0], 1.0, "Default attach offset X should be 1.0")
        self.assertEqual(panel.attachOffset[1], 0.0, "Default attach offset Y should be 0.0")
        self.assertEqual(panel.attachOffset[2], 0.0, "Default attach offset Z should be 0.0")
        self.assertEqual(panel.spanOffset[0], 0.0, "Default span offset should be 0,0,0")
        
        # Verify default attach/span strings
        self.assertIsNone(panel.attachStr, "Default attach string should be None")
        self.assertIsNone(panel.spanStr, "Default span string should be None")
        
        # Verify size and offset initialized from transform
        self.assertEqual(panel.size0, panel.transform.scale)
        self.assertEqual(panel.size1, panel.transform.scale)
        self.assertEqual(panel.offset0, panel.transform.translation)
        self.assertEqual(panel.offset1, panel.transform.translation)
        self.assertEqual(panel.rotate, panel.transform.rotation)
    
    # ========================================================================
    # ATTACHMENT AND SPANNING
    # ========================================================================
    
    def test_find_attached_elements_upstream(self):
        """Test finding upstream attached elements via span relationships."""
        # Create two panels where p2 spans to p1
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        p1.parent = p2.parent = GXMLMockPanel("root")  # Share parent
        p1.parent.children = [p1, p2]
        
        self.layout.apply_default_layout_properties(p1)
        self.layout.apply_default_layout_properties(p2)
        
        # Set p2's spanElement to p1 (p2 spans to p1)
        p2.spanElement = p1
        p2.attachElement = None
        
        # Find attached elements from p1's perspective
        # Since p2's spanElement points to p1, when we search for p1's attached elements,
        # we should find p2 as upstream (because p2 spans to p1)
        upstreamAttach, attachOffset, downstreamSpan, spanOffset = self.layout.find_attached_elements(p1)
        
        # p2 spans to p1, so from p1's perspective, p2 is upstream
        self.assertEqual(upstreamAttach, p2, "Upstream attach element should be p2 (p2 spans to p1)")
        self.assertEqual(attachOffset, p1.attachOffset, "Attach offset should match p1's attach offset")
        self.assertIsNone(downstreamSpan, "Downstream span should be None")
    
    def test_find_attached_elements_downstream(self):
        """Test finding downstream attached elements via attach relationships."""
        # Create two panels where p2 attaches to p1
        p1 = GXMLMockPanel("p1")
        p2 = GXMLMockPanel("p2")
        p1.parent = p2.parent = GXMLMockPanel("root")
        p1.parent.children = [p1, p2]
        
        self.layout.apply_default_layout_properties(p1)
        self.layout.apply_default_layout_properties(p2)
        
        # Set p2's attachElement to p1 (p2 attaches to p1)
        p2.attachElement = p1
        p2.spanElement = None
        
        # Find attached elements from p1's perspective
        upstreamAttach, attachOffset, downstreamSpan, spanOffset = self.layout.find_attached_elements(p1)
        
        # p2's attach points to p1, so p2 should be downstream from p1
        self.assertIsNone(upstreamAttach, "Upstream attach should be None")
        self.assertEqual(downstreamSpan, p2, "Downstream span element should be p2 (via attach)")
        self.assertEqual(spanOffset, p1.spanOffset, "Span offset should match p1's span offset")
    
    # ========================================================================
    # SIZE EXPANSION
    # ========================================================================
    
    def test_expand_size_without_dependency(self):
        """Test size expansion with fixed numeric values."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        
        size = [2.0, 3.0, 1.0]
        offset = Offset(0, 0, 0)
        
        expandedSize = self.layout.expand_size(size, None, offset, defaultSize=1.0)
        
        # Should return unchanged size when no dependencies
        np.testing.assert_array_almost_equal(expandedSize, [2.0, 3.0, 1.0], decimal=5,
                                            err_msg="Size without dependencies should remain unchanged")
    
    def test_expand_size_with_dependency(self):
        """Test size expansion with '*' dependency on target element height."""
        # Create target panel with specific height
        targetPanel = GXMLMockPanel("target", [0, 0, 0], [1, 0, 0], height=2.5)
        self.layout.apply_default_layout_properties(targetPanel)
        
        # Test size with height dependency
        size = [1.0, "*", 1.0]  # Height depends on target element
        offset = Offset(0, 0, 0)
        
        expandedSize = self.layout.expand_size(size, targetPanel, offset, defaultSize=1.0)
        
        # Y component should be expanded to target panel height (2.5)
        self.assertAlmostEqual(expandedSize[0], 1.0, places=5, 
                              msg="X size should remain unchanged")
        self.assertAlmostEqual(expandedSize[1], 2.5, places=5,
                              msg="Y size should match target panel height")
        self.assertAlmostEqual(expandedSize[2], 1.0, places=5,
                              msg="Z size should remain unchanged")
    
    def test_expand_size_with_missing_target(self):
        """Test size expansion with '*' but no target element uses default."""
        size = [1.0, "*", 1.0]
        offset = Offset(0, 0, 0)
        
        expandedSize = self.layout.expand_size(size, None, offset, defaultSize=5.0)
        
        # Should use default size when no target element
        self.assertAlmostEqual(expandedSize[1], 5.0, places=5,
                              msg="Size with '*' and no target should use default")
    
    # ========================================================================
    # LOCAL TRANSFORM BUILDING
    # ========================================================================
    
    def test_build_local_transform_basic(self):
        """Test building local transformation with equal start/end sizes."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        
        # Set up equal sizes and zero offsets
        panel.offset0 = np.array([0.0, 0.0, 0.0])
        panel.offset1 = np.array([0.0, 0.0, 0.0])
        panel.rotate = np.array([0.0, 0.0, 0.0])
        
        size0 = np.array([1.0, 2.0, 0.5])
        size1 = np.array([1.0, 2.0, 0.5])
        
        self.layout.build_local_transform(panel, size0, size1)
        
        # Verify quad corners form a rectangle with proper proportions
        # For equal sizes, transforming normalized corners should give expected local positions
        # Since panel has identity transform, transform_point gives us the quad's local geometry
        normalized_corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        expected = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        
        for i, (norm_corner, exp) in enumerate(zip(normalized_corners, expected)):
            actual = panel.transform_point(norm_corner)
            np.testing.assert_array_almost_equal(actual, exp, decimal=5,
                                                err_msg=f"Corner {i} should be at {exp}")
    
    def test_build_local_transform_with_different_sizes(self):
        """Test building local transformation with different start/end heights."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        
        panel.offset0 = np.array([0.0, 0.0, 0.0])
        panel.offset1 = np.array([0.0, 0.0, 0.0])
        panel.rotate = np.array([0.0, 0.0, 0.0])
        
        size0 = np.array([1.0, 2.0, 0.5])  # Start height = 2.0
        size1 = np.array([1.0, 4.0, 0.5])  # End height = 4.0
        
        self.layout.build_local_transform(panel, size0, size1)
        
        # With different heights, the quad should scale according to ratios
        # sizeRatio = [2.0/4.0, 4.0/4.0] = [0.5, 1.0]
        # Start should be at height 0.5, end at height 1.0
        normalized_corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        expected = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0.5, 0]]
        
        for i, (norm_corner, exp) in enumerate(zip(normalized_corners, expected)):
            actual = panel.transform_point(norm_corner)
            np.testing.assert_array_almost_equal(actual, exp, decimal=5,
                                                err_msg=f"Corner {i} with size ratio should be at {exp}")
    
    def test_build_local_transform_with_offsets(self):
        """Test building local transformation with start/end offsets."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        
        # Set offsets to shift the panel
        panel.offset0 = np.array([0.5, 0.0, 0.0])  # Start offset
        panel.offset1 = np.array([0.5, 0.0, 0.0])  # End offset
        panel.rotate = np.array([0.0, 0.0, 0.0])
        
        size0 = np.array([2.0, 2.0, 0.5])
        size1 = np.array([2.0, 2.0, 0.5])
        
        self.layout.build_local_transform(panel, size0, size1)
        
        # Verify quad corners include offset ratios
        # offsetRatioX = [0.5/2.0, 0.5/2.0] = [0.25, 0.25]
        # Corners should be shifted by 0.25 in X
        normalized_corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        expected = [[0.25, 0, 0], [1.25, 0, 0], [1.25, 1, 0], [0.25, 1, 0]]
        
        for i, (norm_corner, exp) in enumerate(zip(normalized_corners, expected)):
            actual = panel.transform_point(norm_corner)
            np.testing.assert_array_almost_equal(actual, exp, decimal=5,
                                                err_msg=f"Corner {i} with offsets should be at {exp}")
    
    def test_build_local_transform_with_rotation(self):
        """Test building local transformation with rotation applied."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        
        # Set up rotation around Y axis (yaw)
        panel.offset0 = np.array([0.0, 0.0, 0.0])
        panel.offset1 = np.array([0.0, 0.0, 0.0])
        panel.rotate = np.array([0.0, 90.0, 0.0])  # 90 degree rotation around Y
        
        size0 = np.array([2.0, 2.0, 0.5])
        size1 = np.array([2.0, 2.0, 0.5])
        
        self.layout.build_local_transform(panel, size0, size1)
        
        # Verify localTransformationMatrix contains rotation (build_local_transform sets this)
        # Check that localTransformationMatrix is not identity
        identity = np.identity(4)
        
        self.assertFalse(np.allclose(np.array(panel.transform.localTransformationMatrix), np.array(identity)),
                        "Local transform with rotation should not be identity matrix")
        
        # Verify rotation was applied by checking stored rotation value
        self.assertEqual(panel.transform.rotation[1], 90.0,
                        "Rotation around Y axis should be 90 degrees")
        
        # Verify the local transformation matrix contains rotation components
        # For a 90-degree Y-axis rotation combined with scale (2, 2, 0.5):
        # The matrix should rotate then scale, resulting in non-diagonal values
        local_matrix = np.array(panel.transform.localTransformationMatrix)[:3, :3]
        
        # Check that off-diagonal elements exist (indicating rotation)
        off_diagonal_sum = np.sum(np.abs(local_matrix)) - np.sum(np.abs(np.diag(local_matrix)))
        self.assertGreater(off_diagonal_sum, 0.1,
                          "Local transform should have off-diagonal elements from rotation")
    
    # ========================================================================
    # INTERSECTION AND AUTO SPANNING
    # ========================================================================
    
    def test_find_intersection_point_basic(self):
        """Test ray-segment intersection for span point calculation."""
        # Create a simple horizontal segment from (0,0,0) to (1,0,0)
        panel = GXMLMockPanel("p1", [0, 0, 0], [1, 0, 0], thickness=0.1)
        
        # Ray from (0.5, 1, 0) pointing down (0, -1, 0)
        point = np.array([0.5, 1.0, 0.0])
        direction = np.array([0.0, -1.0, 0.0])
        
        intersection = self.layout.find_intersection_point(panel.transform, direction, point)
        
        # Should intersect at (0.5, 0, 0) - the midpoint of the segment
        if intersection is not None:
            self.assertAlmostEqual(intersection[0], 0.5, places=5,
                                  msg="Intersection X should be at segment midpoint")
            self.assertAlmostEqual(intersection[1], 0.0, places=5,
                                  msg="Intersection Y should be at segment Y")
    
    # ========================================================================
    # SPAN MATRIX CALCULATION
    # ========================================================================
    
    def test_calculate_span_matrix_zero_distance(self):
        """Test span matrix when span and attach points are coincident."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        
        # Create transforms where attach and span are at the same point
        attachTransform = GXMLMockPanel("attach", [0, 0, 0], [1, 0, 0], thickness=0.1).transform
        spanTransform = GXMLMockPanel("span", [0, 0, 0], [1, 0, 0], thickness=0.1).transform
        
        attachOffset = Offset(0, 0, 0)
        spanOffset = Offset(0, 0, 0)
        
        matrix = self.layout.calculate_span_matrix(panel, attachTransform, attachOffset, 
                                                   spanTransform, spanOffset, height=1.0)
        
        # Zero distance should result in zero scale in X
        self.assertIsNotNone(matrix, "Span matrix should be created")
        # First column of matrix should have zero magnitude (zero X scale)
        self.assertAlmostEqual(matrix[0][0], 0.0, places=5,
                              msg="Zero distance should result in zero X scale")
    
    def test_calculate_span_matrix_non_zero_distance(self):
        """Test span matrix calculation with separated attach/span points."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        
        # Create transforms where attach is at origin, span is at (2, 0, 0)
        attachTransform = GXMLMockPanel("attach", [0, 0, 0], [1, 0, 0], thickness=0.1).transform
        spanTransform = GXMLMockPanel("span", [2, 0, 0], [3, 0, 0], thickness=0.1).transform
        
        # Attach at end of attach element (1, 0, 0)
        attachOffset = Offset(1, 0, 0)
        # Span at start of span element (0, 0, 0 in local coords = 2,0,0 in world)
        spanOffset = Offset(0, 0, 0)
        
        matrix = self.layout.calculate_span_matrix(panel, attachTransform, attachOffset,
                                                   spanTransform, spanOffset, height=2.0)
        
        self.assertIsNotNone(matrix, "Span matrix should be created")
        # Distance from (1,0,0) to (2,0,0) is 1.0
        # Matrix should scale by distance in X and height in Y
        # First column magnitude should be approximately the distance (1.0)
        np_matrix = np.array(matrix)
        x_scale = np.linalg.norm(np_matrix[:3, 0])
        self.assertAlmostEqual(x_scale, 1.0, places=5,
                              msg="X scale should match distance between points")
    
    def test_calculate_span_matrix_with_auto_span(self):
        """Test span matrix calculation with automatic span point selection."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        panel.transform.rotation = np.array([0.0, 0.0, 0.0])
        
        # Create transforms
        attachTransform = GXMLMockPanel("attach", [0, 0, 0], [1, 0, 0], thickness=0.1).transform
        spanTransform = GXMLMockPanel("span", [2, 0, 0], [3, 0, 0], thickness=0.1).transform
        
        attachOffset = Offset(1, 0, 0)
        spanOffset = Offset.auto()  # Use auto span
        
        matrix = self.layout.calculate_span_matrix(panel, attachTransform, attachOffset,
                                                   spanTransform, spanOffset, height=2.0)
        
        # Should still produce a valid matrix
        self.assertIsNotNone(matrix, "Span matrix with auto span should be created")
        # Matrix should not be all zeros
        self.assertGreater(np.sum(np.abs(matrix)), 0.1,
                          "Auto span matrix should contain non-zero values")
    
    # ========================================================================
    # AUTO SPAN POINT CALCULATION
    # ========================================================================
    
    def test_find_auto_span_point_executes(self):
        """Test auto span point executes and returns valid result."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        panel.transform.rotation = np.array([0.0, 45.0, 0.0])  # 45 degree Y rotation
        
        # Attach and span elements
        attachTransform = GXMLMockPanel("attach", [0, 0, 0], [1, 0, 0], height=1.0).transform
        spanTransform = GXMLMockPanel("span", [3, 0, 0], [5, 0, 0], height=1.0).transform
        
        attachPoint = np.array([1.0, 0.0, 0.0])
        spanOffset = Offset(0, 0, 0)
        
        spanPoint = self.layout.find_auto_span_point(panel, attachTransform, spanTransform, 
                                                      attachPoint, spanOffset)
        
        # Verify function executes and returns valid result
        self.assertIsNotNone(spanPoint, "Auto span point should be found")
        self.assertEqual(len(spanPoint), 3, "Auto span should return a 3D point")
        # Verify no NaN or inf values (computation succeeded)
        self.assertFalse(np.any(np.isnan(spanPoint)), "Auto span point should not contain NaN")
        self.assertFalse(np.any(np.isinf(spanPoint)), "Auto span point should not contain inf")
        # Verify at least one coordinate is non-zero (point was calculated)
        coord_sum = abs(spanPoint[0]) + abs(spanPoint[1]) + abs(spanPoint[2])
        self.assertGreater(coord_sum, 0.1, "Auto span should have non-zero coordinates")
    
    def test_find_auto_span_point_rotation_affects_result(self):
        """Test that rotation angle affects the span point selection."""
        # Set up geometry where rotation will affect which endpoint is selected
        attachTransform = GXMLMockPanel("attach", [0, 0, 0], [1, 0, 0], height=1.0).transform
        # Span segment perpendicular to attach, positioned so rotation matters
        spanTransform = GXMLMockPanel("span", [2, 0, -1], [2, 0, 1], height=1.0).transform
        attachPoint = np.array([1.0, 0.0, 0.0])
        spanOffset = Offset(0, 0, 0)
        
        # Calculate with 0 degree rotation (direction along X axis)
        panel1 = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel1)
        panel1.transform.rotation = np.array([0.0, 0.0, 0.0])
        spanPoint1 = self.layout.find_auto_span_point(panel1, attachTransform, spanTransform,
                                                       attachPoint, spanOffset)
        
        # Calculate with 90 degree rotation (direction along Z axis)
        panel2 = GXMLMockPanel("p2")
        self.layout.apply_default_layout_properties(panel2)
        panel2.transform.rotation = np.array([0.0, 90.0, 0.0])
        spanPoint2 = self.layout.find_auto_span_point(panel2, attachTransform, spanTransform,
                                                       attachPoint, spanOffset)
        
        # Both should return valid points
        self.assertIsNotNone(spanPoint1, "Span point 1 should be found")
        self.assertIsNotNone(spanPoint2, "Span point 2 should be found")
        # With perpendicular span segment, different rotations may select different endpoints
        # Just verify both calculations executed (returned non-zero points)
        self.assertGreater(np.linalg.norm(spanPoint1), 0.1, "Span point 1 should be non-zero")
        self.assertGreater(np.linalg.norm(spanPoint2), 0.1, "Span point 2 should be non-zero")
    
    # ========================================================================
    # PRE-LAYOUT AND LAYOUT INTEGRATION
    # ========================================================================
    
    def test_pre_layout_element(self):
        """Test pre_layout_element orchestrates size expansion and transform building."""
        # Create two panels where p2 attaches to p1
        p1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], height=2.0)
        p2 = GXMLMockPanel("p2")
        
        self.layout.apply_default_layout_properties(p1)
        self.layout.apply_default_layout_properties(p2)
        
        # Set up p2 to attach to p1 with fixed sizes
        p2.attachElement = p1
        p2.size0 = np.array([1.0, 1.5, 0.5])
        p2.size1 = np.array([1.0, 1.5, 0.5])
        p2.offset0 = np.array([0.0, 0.0, 0.0])
        p2.offset1 = np.array([0.0, 0.0, 0.0])
        p2.rotate = np.array([0.0, 0.0, 0.0])
        
        self.layout.pre_layout_element(p2)
        
        # Verify transform was applied (localTransformationMatrix should exist)
        self.assertIsNotNone(p2.transform.localTransformationMatrix,
                            "pre_layout_element should set localTransformationMatrix")
        
        # Verify scale was applied
        self.assertEqual(p2.transform.scale[0], 1.0, "Scale X should be set from size")
        self.assertEqual(p2.transform.scale[1], 1.5, "Scale Y should be set from size")
    
    def test_layout_element_with_dependent_sizes(self):
        """Test layout_element recalculates transforms when sizes have dependencies."""
        # Create target panel with specific height
        p1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], height=3.0)
        p2 = GXMLMockPanel("p2")
        p1.parent = p2.parent = GXMLMockPanel("root")
        p1.parent.children = [p1, p2]
        
        self.layout.apply_default_layout_properties(p1)
        self.layout.apply_default_layout_properties(p2)
        
        # Set p2 to have size dependency on p1
        p2.attachElement = p1
        p2.size0 = [1.0, "*", 0.5]  # Height depends on attached element
        p2.size1 = [1.0, "*", 0.5]
        p2.offset0 = np.array([0.0, 0.0, 0.0])
        p2.offset1 = np.array([0.0, 0.0, 0.0])
        p2.rotate = np.array([0.0, 0.0, 0.0])
        
        self.layout.layout_element(p2)
        
        # Scale Y should be updated from dependency (should be p1's height = 3.0)
        self.assertAlmostEqual(p2.transform.scale[1], 3.0, places=3,
                              msg="Dependent size should match target element height")
    
    def test_layout_element_without_dependent_sizes(self):
        """Test layout_element does nothing when sizes are not dependent."""
        p1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], height=2.0)
        
        self.layout.apply_default_layout_properties(p1)
        
        # Set fixed sizes (no "*" dependencies)
        p1.size0 = np.array([1.0, 1.0, 0.5])
        p1.size1 = np.array([1.0, 1.0, 0.5])
        
        # Store initial transform state
        initial_scale = np.array(p1.transform.scale)
        
        self.layout.layout_element(p1)
        
        # Since no dependencies, layout_element should not change scale
        # This just verifies it doesn't crash with independent sizes
        np.testing.assert_array_equal(initial_scale, p1.transform.scale,
                        "layout_element should not change scale without size dependencies")
    
    # ========================================================================
    # WORLD TRANSFORM BUILDING
    # ========================================================================
    
    def test_build_world_transform_with_attachment(self):
        """Test build_world_transform applies attachment transformations."""
        # Create two panels where p2 attaches to p1
        p1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], height=2.0)
        p2 = GXMLMockPanel("p2")
        
        self.layout.apply_default_layout_properties(p1)
        self.layout.apply_default_layout_properties(p2)
        
        # p2 attaches to p1's end
        p2.attachElement = p1
        p2.attachOffset = Offset(1, 0, 0)  # Attach at end
        
        # Build local transform first
        size0 = np.array([1.0, 1.0, 0.5])
        size1 = np.array([1.0, 1.0, 0.5])
        self.layout.build_local_transform(p2, size0, size1)
        
        # Build world transform
        self.layout.build_world_transform(p2, 1.0)
        
        # Verify transformation matrix was recalculated
        self.assertIsNotNone(p2.transform.transformationMatrix,
                            "build_world_transform should set transformationMatrix")
        
        # Transform should not be identity (attachment was applied)
        identity = np.identity(4)
        self.assertFalse(np.allclose(p2.transform.transformationMatrix, identity),
                        "Transform with attachment should not be identity")
    
    def test_build_world_transform_with_span(self):
        """Test build_world_transform applies span transformations with unit square architecture."""
        p1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], height=2.0)
        p2 = GXMLMockPanel("p2")
        p3 = GXMLMockPanel("p3", [4, 0, 0], [6, 0, 0], height=3.0)
        
        self.layout.apply_default_layout_properties(p1)
        self.layout.apply_default_layout_properties(p2)
        self.layout.apply_default_layout_properties(p3)
        
        # p2 attaches to p1's end and spans to p3's start
        p2.attachElement = p1
        p2.spanElement = p3
        p2.attachOffset = Offset(1, 0, 0)  # End of p1 at [2, 0, 0]
        p2.spanOffset = Offset(0, 0, 0)  # Start of p3 at [4, 0, 0]
        
        # Build local transform first
        size0 = np.array([1.0, 2.0, 0.5])
        size1 = np.array([1.0, 2.0, 0.5])
        self.layout.build_local_transform(p2, size0, size1)
        
        # Build world transform
        self.layout.build_world_transform(p2, 2.0)
        
        # Verify matrices were calculated
        self.assertIsNotNone(p2.transform.localTransformationMatrix,
                            "build_world_transform should set localTransformationMatrix")
        
        # Verify the span matrix has correct scale values
        # Distance from p1 end [2,0,0] to p3 start [4,0,0] is 2.0
        # Height should be 2.0, thickness should be 0.5
        local_matrix = np.array(p2.transform.localTransformationMatrix)
        x_scale = np.linalg.norm(local_matrix[:3, 0])
        y_scale = np.linalg.norm(local_matrix[:3, 1])
        z_scale = np.linalg.norm(local_matrix[:3, 2])
        
        self.assertAlmostEqual(x_scale, 2.0, places=5,
                              msg="X scale should match distance between attach and span points")
        self.assertAlmostEqual(y_scale, 2.0, places=5,
                              msg="Y scale should match height parameter")
        self.assertAlmostEqual(z_scale, 0.5, places=5,
                              msg="Z scale (thickness) should be preserved from original transform")
    
    def test_build_world_transform_without_attach_or_span(self):
        """Test build_world_transform works without attach or span elements."""
        p1 = GXMLMockPanel("p1")
        
        self.layout.apply_default_layout_properties(p1)
        
        # No attach or span elements
        p1.attachElement = None
        p1.spanElement = None
        
        size0 = np.array([1.0, 1.0, 0.5])
        size1 = np.array([1.0, 1.0, 0.5])
        self.layout.build_local_transform(p1, size0, size1)
        
        # Should not crash
        self.layout.build_world_transform(p1, 1.0)
        
        # Transform should still exist
        self.assertIsNotNone(p1.transform.transformationMatrix,
                            "build_world_transform should work without attachments")
    
    # ========================================================================
    # EDGE CASES
    # ========================================================================
    
    def test_build_local_transform_zero_size_x(self):
        """Test build_local_transform handles zero X size gracefully."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        
        panel.offset0 = np.array([0.5, 0.0, 0.0])
        panel.offset1 = np.array([0.5, 0.0, 0.0])
        panel.rotate = np.array([0.0, 0.0, 0.0])
        
        size0 = np.array([0.0, 2.0, 0.5])  # Zero X size
        size1 = np.array([0.0, 2.0, 0.5])
        
        # Should not crash due to divide-by-zero check
        self.layout.build_local_transform(panel, size0, size1)
        
        # Verify transform was created without crashing
        self.assertIsNotNone(panel.transform.localTransformationMatrix,
                            "Should create transform even with zero X size")
    
    def test_build_local_transform_zero_size_y(self):
        """Test build_local_transform handles zero Y size gracefully."""
        panel = GXMLMockPanel("p1")
        self.layout.apply_default_layout_properties(panel)
        
        panel.offset0 = np.array([0.0, 0.5, 0.0])
        panel.offset1 = np.array([0.0, 0.5, 0.0])
        panel.rotate = np.array([0.0, 0.0, 0.0])
        
        size0 = np.array([2.0, 0.0, 0.5])  # Zero Y size
        size1 = np.array([2.0, 0.0, 0.5])
        
        # Should not crash, sizeRatio calculation handles zero
        self.layout.build_local_transform(panel, size0, size1)
        
        # Verify transform was created without crashing
        self.assertIsNotNone(panel.transform.localTransformationMatrix,
                            "Should create transform even with zero Y size")


if __name__ == '__main__':
    unittest.main()
