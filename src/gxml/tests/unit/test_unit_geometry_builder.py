"""
Unit tests for GeometryBuilder.
"""
import unittest
import numpy as np
from elements.solvers import GeometryBuilder, IntersectionSolver, FaceSolver
from tests.test_fixtures.mocks import GXMLMockPanel
from tests.test_fixtures.assertions import assert_face_corners


class GeometryBuilderUnitTests(unittest.TestCase):
    """Unit tests for GeometryBuilder face splitting and geometry generation."""
    
    # ========================================================================
    # BASIC THICKNESS GEOMETRY
    # ========================================================================
    
    def testPanelThicknessFaceOrdering(self):
        """Test that panel with thickness generates 6 faces in correct order with proper subId values."""
        # Create a simple panel with thickness
        p1 = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=1.0)
        
        solution = IntersectionSolver.solve([p1])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Verify panel has exactly 6 face children (front, back, top, bottom, start, end)
        self.assertEqual(len(p1.dynamicChildren), 6, 
                        "Panel with thickness should generate exactly 6 faces")
        
        # Verify face ordering and subId values
        self.assertEqual(p1.dynamicChildren[0].subId, "front", 
                        "First face should be 'front'")
        self.assertEqual(p1.dynamicChildren[1].subId, "back", 
                        "Second face should be 'back'")
        self.assertEqual(p1.dynamicChildren[2].subId, "top", 
                        "Third face should be 'top'")
        self.assertEqual(p1.dynamicChildren[3].subId, "bottom", 
                        "Fourth face should be 'bottom'")
        self.assertEqual(p1.dynamicChildren[4].subId, "start", 
                        "Fifth face should be 'start'")
        self.assertEqual(p1.dynamicChildren[5].subId, "end", 
                        "Sixth face should be 'end'")
        
        # Verify corner points for each face (panel from [0,0,0] to [1,0,0], thickness=1.0)
        assert_face_corners(self, p1.dynamicChildren[0], [[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]],
                           msg="Front face should have correct corner points")
        assert_face_corners(self, p1.dynamicChildren[1], [[1, 0, -0.5], [0, 0, -0.5], [0, 1, -0.5], [1, 1, -0.5]],
                           msg="Back face should have correct corner points (flipped from front)")
        assert_face_corners(self, p1.dynamicChildren[2], [[0, 1, -0.5], [0, 1, 0.5], [1, 1, 0.5], [1, 1, -0.5]],
                           msg="Top face should have correct corner points")
        assert_face_corners(self, p1.dynamicChildren[3], [[0, 0, -0.5], [1, 0, -0.5], [1, 0, 0.5], [0, 0, 0.5]],
                           msg="Bottom face should have correct corner points")
        assert_face_corners(self, p1.dynamicChildren[4], [[0, 0, -0.5], [0, 0, 0.5], [0, 1, 0.5], [0, 1, -0.5]],
                           msg="Start face should have correct corner points")
        assert_face_corners(self, p1.dynamicChildren[5], [[1, 0, 0.5], [1, 0, -0.5], [1, 1, -0.5], [1, 1, 0.5]],
                           msg="End face should have correct corner points (flipped from start)")
    
    def testZeroScalePanelSkipsGeometry(self):
        """Test that zero-scale panels don't crash and produce no face children.
        
        Zero-scale panels have degenerate geometry (zero-length primary or secondary axis)
        which would cause normalization errors. GeometryBuilder should skip them.
        """
        # Test 1: Zero primary axis scale (zero width)
        p1 = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=1.0)
        p1.transform.apply_local_transformations((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 1.0))
        p1.recalculate_transform()
        
        solution1 = IntersectionSolver.solve([p1])
        GeometryBuilder.build_all(FaceSolver.solve(solution1))
        
        self.assertEqual(len(p1.dynamicChildren), 0,
                        "Zero primary axis scale panel should have no face children")
        
        # Test 2: Zero secondary axis scale (zero height)
        p2 = GXMLMockPanel("p2", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=1.0)
        p2.transform.apply_local_transformations((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 1.0))
        p2.recalculate_transform()
        
        solution2 = IntersectionSolver.solve([p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution2))
        
        self.assertEqual(len(p2.dynamicChildren), 0,
                        "Zero secondary axis scale panel should have no face children")
    
    # ========================================================================
    # T-JUNCTION SPLITTING
    # ========================================================================
    
    def testTJunctionSplitting(self):
        """Test that a panel is split when a T-junction occurs at its midspan."""
        # Create panels and solve intersections
        p1 = GXMLMockPanel("p1", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        p2 = GXMLMockPanel("p2", [0.0, 0.0, 0.0], [0.0, 0.0, 2.0])
        
        solution = IntersectionSolver.solve([p1, p2])
        
        # Build geometry for all panels
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Verify p1 has face children
        # Front face should be split (2 pieces) + back/top/bottom/start/end continuous = 7 faces
        # But with current implementation, we should have at least the 6 basic faces
        self.assertGreater(len(p1.dynamicChildren), 0, "Panel p1 should have face children")
        
        # Check that front face exists and is split
        front_faces = [c for c in p1.dynamicChildren if "front" in c.subId.lower()]
        self.assertGreater(len(front_faces), 1, "Panel p1 front face should be split into multiple pieces")
        
        # Verify p2 was not split (T-junction at endpoint)
        # Should have 5 face children: front, back, top, bottom, end
        # The START face is omitted because it's interior geometry (closed by p1's back face)
        self.assertEqual(len(p2.dynamicChildren), 5, "Panel p2 should have 5 face children (no start face)")
        
        # Verify p2 faces are not split (each face appears once)
        face_names = [c.subId for c in p2.dynamicChildren]
        # Should have: front, back, top, bottom, end (no start - it's interior)
        self.assertIn("front", face_names, "Should have front face")
        self.assertIn("back", face_names, "Should have back face")
        self.assertIn("end", face_names, "Should have end face")
        self.assertNotIn("start", face_names, "Should NOT have start face (interior geometry)")
    
    def testZeroThicknessTJunctionSplitting(self):
        """Test that a zero-thickness panel is correctly split at a T-junction.
        
        A zero-thickness panel should still have its front face split into regions
        when another panel creates a T-junction at its midspan.
        """
        # Create two zero-thickness panels forming a T-junction
        p1 = GXMLMockPanel("p1", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.0)
        p2 = GXMLMockPanel("p2", [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], thickness=0.0)
        
        solution = IntersectionSolver.solve([p1, p2])
        
        # Verify intersection was detected
        self.assertEqual(len(solution.intersections), 1, "Should detect one T-junction")
        
        # Build geometry
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # p1 should have its front face split into 2 regions (at t=0.5)
        # Even with 0 thickness, the front face should be created and split
        self.assertEqual(len(p1.dynamicChildren), 2, 
                        "Zero-thickness panel p1 should have 2 face children (front-0, front-1)")
        
        # Verify the faces are front-0 and front-1 in correct order (no sorting needed)
        face_ids = [c.subId for c in p1.dynamicChildren]
        self.assertEqual(face_ids, ["front-0", "front-1"], 
                        "Split faces should be front-0 and front-1 in order")
        
        # p2 should have no face children (T-junction at endpoint, 0 thickness)
        # or just 1 unsplit front face
        self.assertLessEqual(len(p2.dynamicChildren), 1,
                            "Zero-thickness panel p2 at endpoint should have at most 1 face")
    
    # ========================================================================
    # DIRECTIONAL FACE SPLITTING
    # ========================================================================
    
    def testRotatedPanelFaceSplitting(self):
        """Test that face splitting works correctly when panel approaches from different directions.
        
        For zero-thickness panels, only the face aligned with the intersection direction
        should be created. This tests that _determine_split_faces correctly identifies
        which face is affected by the approach direction.
        """
        # Test 1: Panel p2 approaches p1 from +Z direction (normal case)
        # With zero thickness, only the front face (normal +Z) should be created
        p1_case1 = GXMLMockPanel("p1", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.0)
        p2_case1 = GXMLMockPanel("p2", [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], thickness=0.0)
        solution1 = IntersectionSolver.solve([p1_case1, p2_case1])
        
        # Test 2: Panel p2 approaches p1 from -Z direction (rotated 180°)
        # With zero thickness, only the back face (normal -Z) should be created
        p1_case2 = GXMLMockPanel("p1", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.0)
        p2_case2 = GXMLMockPanel("p2", [0.0, 0.0, 0.0], [0.0, 0.0, -2.0], thickness=0.0)
        solution2 = IntersectionSolver.solve([p1_case2, p2_case2])
        
        # Build geometry for both cases
        GeometryBuilder.build_all(FaceSolver.solve(solution1))
        GeometryBuilder.build_all(FaceSolver.solve(solution2))
        
        p1_case1 = solution1.panels[0]
        p1_case2 = solution2.panels[0]
        
        # Get split face for each case
        def get_split_face(panel):
            split_faces = [c.subId.split('-')[0] for c in panel.dynamicChildren if '-' in c.subId]
            if split_faces:
                return split_faces[0]
            return None
        
        face1 = get_split_face(p1_case1)
        face2 = get_split_face(p1_case2)
        
        # Both should have splits
        self.assertIsNotNone(face1, "Case 1 (+Z direction) should split a face")
        self.assertIsNotNone(face2, "Case 2 (-Z direction) should split a face")
        
        # The split faces should be OPPOSITE faces (front vs back)
        # because the approach direction is opposite
        self.assertIn(face1, ['front', 'back'], f"Case 1 split unexpected face: {face1}")
    # ========================================================================
    # CROSSING INTERSECTIONS
    # ========================================================================
    
        self.assertIn(face2, ['front', 'back'], f"Case 2 split unexpected face: {face2}")
        self.assertNotEqual(face1, face2, 
                          f"Opposite approach directions should split opposite faces, but both split {face1}")
    
    def testCrossingBothPanelsSplit(self):
        """Test that both panels get split in a crossing intersection."""
        # Create a crossing: p1 horizontal along X, p2 vertical along Z
        # Both intersect at their midpoints (0.5 t-value)
        p1 = GXMLMockPanel("p1", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        p2 = GXMLMockPanel("p2", [0.0, 0.0, -1.0], [0.0, 0.0, 1.0])
        
        solution = IntersectionSolver.solve([p1, p2])
        
        # Build geometry for both panels
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Helper to check if a panel has splits
        def has_splits(panel):
            split_faces = [c for c in panel.dynamicChildren if '-' in c.subId]
            return len(split_faces) > 0
        
        # Helper to get split face types (can be multiple for crossings)
        def get_split_faces(panel):
            split_faces = set()
            for child in panel.dynamicChildren:
                if '-' in child.subId:
                    face_type = child.subId.split('-')[0]
                    split_faces.add(face_type)
            return split_faces
        
        # Both panels should have splits (this is a CROSSING, not a T-junction)
        self.assertTrue(has_splits(p1), "Panel p1 should be split at crossing")
        self.assertTrue(has_splits(p2), "Panel p2 should be split at crossing")
        
        # Get which faces were split
        p1_split_faces = get_split_faces(p1)
        p2_split_faces = get_split_faces(p2)
        
        self.assertTrue(len(p1_split_faces) > 0, "Panel p1 should have split faces")
        self.assertTrue(len(p2_split_faces) > 0, "Panel p2 should have split faces")
        
        # p1 is horizontal (along X), p2 approaches along Z axis from both directions (crossing)
        # So p1 should split BOTH front AND back faces
        self.assertIn('front', p1_split_faces, "Panel p1 should split front face")
        self.assertIn('back', p1_split_faces, "Panel p1 should split back face")
        
        # p2 is vertical (along Z), p1 approaches along X axis from both directions (crossing)
        # So p2 should split BOTH start AND end faces
        self.assertIn('front', p2_split_faces, "Panel p2 should split start face")
        self.assertIn('back', p2_split_faces, "Panel p2 should split end face")
    
    def test_create_panel_side_transformations(self):
        """Test that create_panel_side (called by GeometryBuilder) creates correct transformation matrix.
        
        Verifies that the START side (x=0 plane) of a panel with thickness 1.0 correctly
        transforms unit square coordinates to expected world positions accounting for
        thickness offset. This is an internal method called by GeometryBuilder.
        """
        from elements.gxml_panel import GXMLPanel, PanelSide
        
        panel = GXMLPanel()
        panel.thickness = 1.0
        
        # Create a sub-panel for the START side (x=0 plane)
        # This is what GeometryBuilder calls internally
        subPanel = panel.create_panel_side("start", PanelSide.START)
        
        # For START side with default panel and thickness 1.0:
        # Unit square in local space should map to world coordinates
        # with thickness offset applied: z_world = z_local * thickness - thickness/2
        
        # Test that unit square corners transform to expected world positions
        p1Actual = subPanel.transform_point([0, 0, 0])
        p2Actual = subPanel.transform_point([1, 0, 0])
        p3Actual = subPanel.transform_point([1, 1, 0])
        p4Actual = subPanel.transform_point([0, 1, 0])
        
        # Expected world coordinates for START side (x=0 plane):
        # Corner order: (x=0, y=0, z-range), (x=0, y=0, z+range), 
        #               (x=0, y=1, z+range), (x=0, y=1, z-range)
        # where z-range = -0.5 and z+range = 0.5 (thickness centered at 0)
        self.assertTrue(np.allclose(p1Actual, [0, 0, -0.5], rtol=1e-10),
                       f"P1 mismatch: {p1Actual} vs [0, 0, -0.5]")
        self.assertTrue(np.allclose(p2Actual, [0, 0, 0.5], rtol=1e-10),
                       f"P2 mismatch: {p2Actual} vs [0, 0, 0.5]")
        self.assertTrue(np.allclose(p3Actual, [0, 1, 0.5], rtol=1e-10),
                       f"P3 mismatch: {p3Actual} vs [0, 1, 0.5]")
        self.assertTrue(np.allclose(p4Actual, [0, 1, -0.5], rtol=1e-10),
                       f"P4 mismatch: {p4Actual} vs [0, 1, -0.5]")


class GapCalculationTests(unittest.TestCase):
    """Unit tests for gap calculation at panel intersections."""
    
    def testGapSymmetryAtPerpendicularAngle(self):
        """Test gap calculation and application for perpendicular intersection (90 degrees).
        
        When panels intersect at 90 degrees, the gap should be symmetric
        around the intersection point, both in raw calculation and in applied geometry.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.gxml_panel import PanelSide
        calculate_gap_t_values = BoundsSolver._calculate_gap_t_values
        
        # Panel 1: horizontal along X axis (0,0,0) to (2,0,0) with thickness 0.1
        p1 = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], thickness=0.1)
        # Panel 2: perpendicular, starts at (1,0,0), goes into -Z with thickness 0.1
        p2 = GXMLMockPanel("p2", [1.0, 0.0, 0.0], [1.0, 0.0, -1.0], thickness=0.1)
        
        # Test raw gap calculation
        gap_start, gap_end = calculate_gap_t_values(p1, PanelSide.BACK, p2, 0.5)
        
        # At 90 degrees, gap width = thickness / panel_length = 0.1 / 2.0 = 0.05
        expected_half_gap = 0.1 / 2.0 / 2.0  # thickness / 2 / panel_length
        self.assertAlmostEqual(gap_start, 0.5 - expected_half_gap, places=5,
                               msg=f"Gap start should be ~{0.5 - expected_half_gap}")
        self.assertAlmostEqual(gap_end, 0.5 + expected_half_gap, places=5,
                               msg=f"Gap end should be ~{0.5 + expected_half_gap}")
        
        # Verify gap is symmetric (centered at intersection point)
        gap_center = (gap_start + gap_end) / 2
        self.assertAlmostEqual(gap_center, 0.5, places=5,
                               msg="Gap should be centered at intersection point")
        
        # Test applied geometry - crossing should create symmetric gap in faces
        p1_cross = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], thickness=0.1)
        p2_cross = GXMLMockPanel("p2", [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1_cross, p2_cross])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        front_segments = sorted(
            [c for c in p1_cross.dynamicChildren if c.subId.startswith('front')],
            key=lambda c: min(v[0] for v in c.get_world_vertices())  # Sort by min X position
        )
        
        self.assertEqual(len(front_segments), 2, "Should have 2 segments")
        
        # Get the X bounds of each segment to verify gap is centered
        seg0_max_x = max(v[0] for v in front_segments[0].get_world_vertices())
        seg1_min_x = min(v[0] for v in front_segments[1].get_world_vertices())
        
        # Verify applied gap is also centered at x=1.0 (t=0.5 on a length-2 panel)
        applied_gap_center = (seg0_max_x + seg1_min_x) / 2
        self.assertAlmostEqual(applied_gap_center, 1.0, places=4,
                              msg="Applied gap should be centered at intersection point (x=1.0)")
    
    def testGapWidensWithShallowerAngles(self):
        """Test that shallower intersection angles create progressively wider gaps.
        
        Tests 30°, 45°, 60°, and 90° to verify gap width increases as angle decreases.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.gxml_panel import PanelSide
        import math
        calculate_gap_t_values = BoundsSolver._calculate_gap_t_values
        
        p1 = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], thickness=0.1)
        
        gaps = {}
        for angle in [30, 45, 60, 90]:
            cos_a = math.cos(math.radians(angle))
            sin_a = math.sin(math.radians(angle))
            p2 = GXMLMockPanel(f"p2_{angle}", [1.0, 0.0, 0.0], [1.0 + cos_a, 0.0, -sin_a], thickness=0.1)
            gap_start, gap_end = calculate_gap_t_values(p1, PanelSide.BACK, p2, 0.5)
            gaps[angle] = gap_end - gap_start
        
        # Verify gap ordering: 30° > 45° > 60° > 90°
        self.assertGreater(gaps[30], gaps[45], "30° gap should be wider than 45° gap")
        self.assertGreater(gaps[45], gaps[60], "45° gap should be wider than 60° gap")
        self.assertGreater(gaps[60], gaps[90], "60° gap should be wider than 90° gap")
        
        # All gaps should include the intersection point
        for angle in [30, 45, 60]:
            cos_a = math.cos(math.radians(angle))
            sin_a = math.sin(math.radians(angle))
            p2 = GXMLMockPanel(f"p2_{angle}", [1.0, 0.0, 0.0], [1.0 + cos_a, 0.0, -sin_a], thickness=0.1)
            gap_start, gap_end = calculate_gap_t_values(p1, PanelSide.BACK, p2, 0.5)
            self.assertLess(gap_start, 0.5, f"{angle}° gap start should be before intersection")
            self.assertGreater(gap_end, 0.5, f"{angle}° gap end should be after intersection")
    
    def testZeroThicknessNoGap(self):
        """Test that zero thickness intersecting panel creates no gap."""
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.gxml_panel import PanelSide
        calculate_gap_t_values = BoundsSolver._calculate_gap_t_values
        
        p1 = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], thickness=0.1)
        p2 = GXMLMockPanel("p2", [1.0, 0.0, 0.0], [1.0, 0.0, -1.0], thickness=0.0)
        
        gap_start, gap_end = calculate_gap_t_values(p1, PanelSide.BACK, p2, 0.5)
        
        # With zero thickness, gap should be zero (start == end == intersection_t)
        self.assertAlmostEqual(gap_start, 0.5, places=5)
        self.assertAlmostEqual(gap_end, 0.5, places=5)


class PartitionedFaceGapApplicationTests(unittest.TestCase):
    """Tests that verify gap calculations are actually applied to partitioned faces.
    
    These tests ensure that when faces are split at crossings, the t_start/t_end
    values on the created face segments properly account for the gap created by
    the intersecting panel's thickness.
    """
    
    def testCrossingFaceSegmentBoundsIncludeGap(self):
        """Test that face segments at a crossing have correct t bounds with gap.
        
        When two panels cross, the intersected panel's face should be split into
        two segments with a gap between them for the intersecting panel's thickness.
        """
        from elements.solvers.gxml_geometry_builder import GeometryBuilder
        from elements.solvers.gxml_intersection_solver import IntersectionSolver
        
        # Panel 1: horizontal along X axis, length 2
        p1 = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], thickness=0.1)
        # Panel 2: perpendicular, crosses p1 at t=0.5 (x=1.0)
        p2 = GXMLMockPanel("p2", [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Get the front face segments for p1
        front_segments = [c for c in p1.dynamicChildren if c.subId.startswith('front')]
        
        # Should have 2 segments (before and after the gap)
        self.assertEqual(len(front_segments), 2, 
                        "Front face should be split into 2 segments at crossing")
        
        # Sort by min X position to get them in order
        front_segments.sort(key=lambda c: min(v[0] for v in c.get_world_vertices()))
        
        first_segment = front_segments[0]
        second_segment = front_segments[1]
        
        # Get world X bounds (panel goes from x=0 to x=2, so t maps to x/2)
        first_min_x = min(v[0] for v in first_segment.get_world_vertices())
        first_max_x = max(v[0] for v in first_segment.get_world_vertices())
        second_min_x = min(v[0] for v in second_segment.get_world_vertices())
        second_max_x = max(v[0] for v in second_segment.get_world_vertices())
        
        # First segment should start at x=0 and end BEFORE x=1.0
        self.assertAlmostEqual(first_min_x, 0.0, places=5,
                              msg="First segment should start at x=0")
        self.assertLess(first_max_x, 1.0,
                       msg="First segment should end before intersection point (x=1.0)")
        
        # Second segment should start AFTER x=1.0 and end at x=2.0
        self.assertGreater(second_min_x, 1.0,
                          msg="Second segment should start after intersection point")
        self.assertAlmostEqual(second_max_x, 2.0, places=5,
                              msg="Second segment should end at x=2")
        
        # The gap between segments should equal the intersecting panel's thickness
        gap_size = second_min_x - first_max_x
        expected_gap = 0.1  # thickness at 90 degrees
        self.assertAlmostEqual(gap_size, expected_gap, places=4,
                              msg=f"Gap should be ~{expected_gap} (thickness)")
    
    def testWithoutGapCalculationSegmentsWouldOverlap(self):
        """Test that validates the gap calculation is necessary.
        
        This test verifies that the raw region bounds (without gap calculation)
        would cause segments to overlap or touch at the intersection point,
        proving that the gap calculation code is doing meaningful work.
        """
        from elements.solvers.gxml_geometry_builder import GeometryBuilder
        from elements.solvers.gxml_intersection_solver import IntersectionSolver
        
        p1 = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], thickness=0.1)
        p2 = GXMLMockPanel("p2", [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1, p2])
        
        # Get the region tree without building geometry
        region_tree = solution.get_region_tree_for_panel(p1)
        leaf_regions = region_tree.get_leaves()
        
        # With a crossing, should have 2 regions
        self.assertEqual(len(leaf_regions), 2, "Should have 2 regions at crossing")
        
        # Raw region bounds should be 0.0 to 0.5 and 0.5 to 1.0 (no gap)
        # These are the t-values BEFORE gap calculation
        first_region = leaf_regions[0]
        second_region = leaf_regions[1]
        
        # Verify raw bounds touch at t=0.5 (no gap in raw data)
        self.assertAlmostEqual(first_region.tStart, 0.0, places=5)
        self.assertAlmostEqual(second_region.tStart, 0.5, places=5)
        
        # Now build geometry - the gap calculation should create proper spacing
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        front_segments = sorted(
            [c for c in p1.dynamicChildren if c.subId.startswith('front')],
            key=lambda c: min(v[0] for v in c.get_world_vertices())  # Sort by min X position
        )
        
        # Get X bounds of segments
        seg0_max_x = max(v[0] for v in front_segments[0].get_world_vertices())
        seg1_min_x = min(v[0] for v in front_segments[1].get_world_vertices())
        
        # After gap calculation, segments should NOT touch at x=1.0
        self.assertLess(seg0_max_x, 1.0,
                       "First segment end should be BEFORE x=1.0 after gap calculation")
        self.assertGreater(seg1_min_x, 1.0,
                          "Second segment start should be AFTER x=1.0 after gap calculation")


class AngledCrossingPerEdgeTValueTests(unittest.TestCase):
    """Tests for per-edge t-value calculation on TOP/BOTTOM faces at angled crossings.
    
    When two panels cross at an angle (not 90°), the crossing panel intersects
    the FRONT face at a different x position than the BACK face. This means
    TOP/BOTTOM faces need different t-values for their front vs back edges,
    creating a trapezoidal shape rather than rectangular.
    """
    
    def _get_bounds_with_gaps(self, panel, face_side, segment):
        """Helper: get nominal bounds and apply gaps to them."""
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        bounds = segment.get_nominal_bounds()
        BoundsSolver._apply_gaps(panel, face_side, segment, bounds)
        return bounds
    
    def testPerpendicularCrossingNoPerEdgeDifference(self):
        """At 90° crossing, TOP/BOTTOM faces should have same t-values for front and back edges.
        
        When panels cross perpendicularly, the gap is the same width at both edges,
        so t_start_front should equal t_start (or be None since no difference needed).
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver, FaceBounds
        from elements.solvers.gxml_intersection_solver import IntersectionSolver
        from elements.solvers.gxml_face_solver import FaceSolver
        from elements.gxml_panel import PanelSide
        
        # Panel 1: horizontal along X axis
        p1 = GXMLMockPanel("p1", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        # Panel 2: perpendicular crossing at origin
        p2 = GXMLMockPanel("p2", [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1, p2])
        face_result = FaceSolver.solve(solution)
        face_solution = face_result.get(p1)
        
        # Get segments for TOP face
        segments = face_solution.get_segments(PanelSide.TOP)
        self.assertGreater(len(segments), 0, "Should have segments for TOP face")
        
        # Get bounds for TOP face on first segment
        bounds = self._get_bounds_with_gaps(p1, PanelSide.TOP, segments[0])
        
        # At perpendicular crossing, either t_start_front is None or equals t_start
        if bounds.t_start_front is not None:
            self.assertAlmostEqual(bounds.t_start, bounds.t_start_front, places=5,
                msg="At 90° crossing, front and back edge t-values should be equal")
    
    def testAngledCrossingHasDifferentPerEdgeTValues(self):
        """At angled crossing, TOP/BOTTOM faces should have different t-values for front vs back.
        
        When the crossing panel approaches at an angle, it creates a wider gap
        on one edge than the other. For a crossing panel approaching from -Z
        at 45°, the FRONT edge gap will be at a different position than the BACK edge gap.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.solvers.gxml_intersection_solver import IntersectionSolver
        from elements.solvers.gxml_face_solver import FaceSolver
        from elements.gxml_panel import PanelSide
        import math
        
        # Panel 1: horizontal along X axis, 2 units long
        p1 = GXMLMockPanel("p1", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Panel 2: crosses at 45 degrees - from (-1, 0, -1) to (1, 0, 1)
        # This crosses p1 at the origin
        p2 = GXMLMockPanel("p2", [-1.0, 0.0, -1.0], [1.0, 0.0, 1.0], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1, p2])
        face_result = FaceSolver.solve(solution)
        face_solution = face_result.get(p1)
        
        # Get segments for TOP face
        segments = face_solution.get_segments(PanelSide.TOP)
        self.assertGreater(len(segments), 0, "Should have segments for TOP face")
        
        # Get bounds for TOP face on first segment (segment before crossing)
        bounds = self._get_bounds_with_gaps(p1, PanelSide.TOP, segments[0])
        
        # At 45° crossing, t_start_front should be different from t_start
        self.assertIsNotNone(bounds.t_end_front, 
            "Angled crossing should produce per-edge t values for TOP face")
        
        # The difference indicates the trapezoidal shape
        self.assertNotAlmostEqual(bounds.t_end, bounds.t_end_front, places=3,
            msg="At 45° crossing, front and back edge t-values should differ")
    
    def testAngledCrossingGapDirection(self):
        """Verify that the gap offset direction is correct for angled crossings.
        
        For a crossing panel angled from bottom-left to top-right (in XZ plane),
        the FRONT edge (positive Z) will have its gap offset in one direction,
        while the BACK edge (negative Z) will have it offset the other way.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.solvers.gxml_intersection_solver import IntersectionSolver
        from elements.solvers.gxml_face_solver import FaceSolver
        from elements.gxml_panel import PanelSide
        
        # Panel 1: horizontal along X axis
        p1 = GXMLMockPanel("p1", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Panel 2: crosses at angle from (-1, 0, -1) to (1, 0, 1)
        # At negative X, the crossing is at negative Z (BACK side of p1)
        # At positive X, the crossing is at positive Z (FRONT side of p1)
        p2 = GXMLMockPanel("p2", [-1.0, 0.0, -1.0], [1.0, 0.0, 1.0], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1, p2])
        face_result = FaceSolver.solve(solution)
        face_solution = face_result.get(p1)
        
        # Get segments for TOP face
        segments = face_solution.get_segments(PanelSide.TOP)
        self.assertGreater(len(segments), 0, "Should have segments for TOP face")
        
        # Get bounds for TOP face - first segment (before crossing)
        bounds_first = self._get_bounds_with_gaps(p1, PanelSide.TOP, segments[0])
        
        # The front edge gap should end at a different x than the back edge gap
        # Due to the 45° angle, one should be ahead of the other
        if bounds_first.t_end_front is not None:
            # For this geometry, front edge is at higher t (further along X)
            # because the crossing panel goes from -X,-Z to +X,+Z
            gap_difference = bounds_first.t_end_front - bounds_first.t_end
            self.assertNotEqual(gap_difference, 0,
                "Front and back edge gaps should be at different positions")
    
    def testFrontBackFacesNoPerEdgeValues(self):
        """FRONT and BACK faces should not have per-edge t values, even at angled crossings.
        
        Per-edge t values only apply to TOP/BOTTOM faces which span the full
        thickness of the panel. FRONT/BACK faces are single-depth surfaces.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.solvers.gxml_intersection_solver import IntersectionSolver
        from elements.solvers.gxml_face_solver import FaceSolver
        from elements.gxml_panel import PanelSide
        
        # Panel 1: horizontal along X axis
        p1 = GXMLMockPanel("p1", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Panel 2: angled crossing
        p2 = GXMLMockPanel("p2", [-1.0, 0.0, -1.0], [1.0, 0.0, 1.0], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1, p2])
        face_result = FaceSolver.solve(solution)
        face_solution = face_result.get(p1)
        
        # Get segments for FRONT and BACK faces
        front_segments = face_solution.get_segments(PanelSide.FRONT)
        back_segments = face_solution.get_segments(PanelSide.BACK)
        self.assertGreater(len(front_segments), 0, "Should have segments for FRONT face")
        self.assertGreater(len(back_segments), 0, "Should have segments for BACK face")
        
        # Get bounds for FRONT face
        bounds_front = self._get_bounds_with_gaps(p1, PanelSide.FRONT, front_segments[0])
        
        # Get bounds for BACK face  
        bounds_back = self._get_bounds_with_gaps(p1, PanelSide.BACK, back_segments[0])
        
        # FRONT/BACK should not have per-edge values
        self.assertIsNone(bounds_front.t_start_front,
            "FRONT face should not have per-edge t_start_front")
        self.assertIsNone(bounds_front.t_end_front,
            "FRONT face should not have per-edge t_end_front")
        self.assertIsNone(bounds_back.t_start_front,
            "BACK face should not have per-edge t_start_front")
        self.assertIsNone(bounds_back.t_end_front,
            "BACK face should not have per-edge t_end_front")
    
    def testTopBottomFacesHaveConsistentPerEdgeValues(self):
        """TOP and BOTTOM faces should both get per-edge values at angled crossings.
        
        Both edge faces span the panel thickness and should have the same
        per-edge gap calculation applied.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.solvers.gxml_intersection_solver import IntersectionSolver
        from elements.solvers.gxml_face_solver import FaceSolver
        from elements.gxml_panel import PanelSide
        
        # Panel 1: horizontal along X axis
        p1 = GXMLMockPanel("p1", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Panel 2: angled crossing
        p2 = GXMLMockPanel("p2", [-1.0, 0.0, -1.0], [1.0, 0.0, 1.0], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1, p2])
        face_result = FaceSolver.solve(solution)
        face_solution = face_result.get(p1)
        
        # Get segments for TOP and BOTTOM faces
        top_segments = face_solution.get_segments(PanelSide.TOP)
        bottom_segments = face_solution.get_segments(PanelSide.BOTTOM)
        self.assertGreater(len(top_segments), 0, "Should have segments for TOP face")
        self.assertGreater(len(bottom_segments), 0, "Should have segments for BOTTOM face")
        
        # Get bounds for TOP face
        bounds_top = self._get_bounds_with_gaps(p1, PanelSide.TOP, top_segments[0])
        
        # Get bounds for BOTTOM face
        bounds_bottom = self._get_bounds_with_gaps(p1, PanelSide.BOTTOM, bottom_segments[0])
        
        # Both should have per-edge values at angled crossing
        self.assertIsNotNone(bounds_top.t_end_front,
            "TOP face should have per-edge t_end_front at angled crossing")
        self.assertIsNotNone(bounds_bottom.t_end_front,
            "BOTTOM face should have per-edge t_end_front at angled crossing")
        
        # TOP and BOTTOM should have the same per-edge values
        # (they share the same front/back edges)
        self.assertAlmostEqual(bounds_top.t_end, bounds_bottom.t_end, places=5,
            msg="TOP and BOTTOM should have same back edge t_end")
        self.assertAlmostEqual(bounds_top.t_end_front, bounds_bottom.t_end_front, places=5,
            msg="TOP and BOTTOM should have same front edge t_end_front")


class EndpointTrimTests(unittest.TestCase):
    """Unit tests for calculate_face_endpoint_trim function."""
    
    def testPerpendicularTrimAllFacesEqual(self):
        """Test that at 90 degrees, all faces have the same trim value.
        
        When a panel meets another panel at a right angle, FRONT, BACK, 
        TOP, and BOTTOM faces should all trim equally.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.gxml_panel import PanelSide
        calculate_face_endpoint_trim = BoundsSolver._compute_trim_to_panel_face
        
        # Wall panel: horizontal along X
        wall = GXMLMockPanel("wall", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        # Divider: perpendicular, starting at wall's midpoint, going into -Z
        divider = GXMLMockPanel("divider", [0.5, 0.0, 0.0], [0.5, 0.0, -1.0], thickness=0.1)
        
        front_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.FRONT)
        back_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.BACK)
        top_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.TOP)
        bottom_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.BOTTOM)
        
        # At 90 degrees, all trims should be equal
        self.assertAlmostEqual(front_trim, back_trim, places=5,
                               msg="Front and back trims should be equal at 90°")
        self.assertAlmostEqual(front_trim, top_trim, places=5,
                               msg="Front and top trims should be equal at 90°")
        self.assertAlmostEqual(front_trim, bottom_trim, places=5,
                               msg="Front and bottom trims should be equal at 90°")
        
        # Trim should be approximately half_thickness / panel_length = 0.05 / 1.0 = 0.05
        expected_trim = 0.05  # half of wall thickness
        self.assertAlmostEqual(front_trim, expected_trim, places=2,
                               msg=f"Trim should be approximately {expected_trim}")
    
    def testAngledTrimFrontBackDifferent(self):
        """Test that at angles other than 90°, FRONT and BACK have different trims.
        
        At 45 degrees, the FRONT face (facing the acute angle) trims more,
        and the BACK face (facing the obtuse angle) trims less.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.gxml_panel import PanelSide
        import math
        calculate_face_endpoint_trim = BoundsSolver._compute_trim_to_panel_face
        
        wall = GXMLMockPanel("wall", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Divider at 45 degrees
        angle = 45
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        divider = GXMLMockPanel("divider", [0.5, 0.0, 0.0], [0.5 + cos_a, 0.0, -sin_a], thickness=0.1)
        
        front_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.FRONT)
        back_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.BACK)
        
        # At 45 degrees, front and back should be different
        self.assertNotAlmostEqual(front_trim, back_trim, places=3,
                                  msg="Front and back trims should differ at 45°")
        
        # Front trim should be larger (acute angle requires more trim to reach surface)
        self.assertGreater(front_trim, back_trim,
                          "Front face should trim more than back at 45°")
    
    def testTrimValuesBoundedByOne(self):
        """Test that trim values are clamped to max 1.0.
        
        At very shallow angles, the computed trim could exceed 1.0 (entire panel),
        but should be clamped to prevent invalid geometry.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.gxml_panel import PanelSide
        import math
        calculate_face_endpoint_trim = BoundsSolver._compute_trim_to_panel_face
        
        wall = GXMLMockPanel("wall", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Divider at very shallow angle (10 degrees)
        angle = 10
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        # Short panel to make trim more likely to exceed 1.0
        divider = GXMLMockPanel("divider", [0.5, 0.0, 0.0], [0.5 + cos_a * 0.1, 0.0, -sin_a * 0.1], thickness=0.1)
        
        front_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.FRONT)
        back_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.BACK)
        
        self.assertLessEqual(front_trim, 1.0, "Front trim should not exceed 1.0")
        self.assertLessEqual(back_trim, 1.0, "Back trim should not exceed 1.0")
    
    def testZeroThicknessWallNoTrim(self):
        """Test that zero thickness wall produces no trim."""
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.gxml_panel import PanelSide
        calculate_face_endpoint_trim = BoundsSolver._compute_trim_to_panel_face
        
        wall = GXMLMockPanel("wall", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.0)
        divider = GXMLMockPanel("divider", [0.5, 0.0, 0.0], [0.5, 0.0, -1.0], thickness=0.1)
        
        front_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.FRONT)
        
        self.assertAlmostEqual(front_trim, 0.0, places=5,
                               msg="Zero thickness wall should produce no trim")
    
    def testObtuseAngleTrim(self):
        """Test trim calculation for obtuse angles (> 90°).
        
        At 135 degrees, the panel extends in the opposite direction.
        The trim calculation should still work correctly.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.gxml_panel import PanelSide
        import math
        calculate_face_endpoint_trim = BoundsSolver._compute_trim_to_panel_face
        
        wall = GXMLMockPanel("wall", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Divider at 135 degrees (extends in -X direction while going into -Z)
        angle = 135
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        divider = GXMLMockPanel("divider", [0.5, 0.0, 0.0], [0.5 + cos_a, 0.0, -sin_a], thickness=0.1)
        
        front_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.FRONT)
        back_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.BACK)
        
        # Both should be valid non-negative values
        self.assertGreaterEqual(front_trim, 0.0, "Front trim should be non-negative")
        self.assertGreaterEqual(back_trim, 0.0, "Back trim should be non-negative")
        
        # At obtuse angle, back face should trim more (it's now on the acute side)
        self.assertGreater(back_trim, front_trim,
                          "At 135°, back face should trim more than front")
    
    def testExtremeAngle44Degrees(self):
        """Test the specific 44-degree case that was failing.
        
        At 44 degrees, the approach direction has a larger dot product with
        the END face than the BACK face. The function should still correctly
        target the BACK face using candidate_faces restriction.
        """
        from elements.solvers.gxml_bounds_solver import BoundsSolver
        from elements.gxml_panel import PanelSide
        import math
        calculate_face_endpoint_trim = BoundsSolver._compute_trim_to_panel_face
        
        wall = GXMLMockPanel("wall", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Divider at exactly 44 degrees
        angle = 44
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        divider = GXMLMockPanel("divider", [0.5, 0.0, 0.0], [0.5 + cos_a, 0.0, -sin_a], thickness=0.1)
        
        front_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.FRONT)
        back_trim = calculate_face_endpoint_trim(divider, wall, PanelSide.START, PanelSide.BACK)
        
        # Values should be reasonable (not 1.0 from failed calculation)
        self.assertLess(front_trim, 0.5, "Front trim should be reasonable at 44°")
        self.assertLess(back_trim, 0.5, "Back trim should be reasonable at 44°")
        
        # Front should trim more than back (acute angle)
        self.assertGreater(front_trim, back_trim,
                          "At 44°, front face should trim more than back")


class FaceDirectionTests(unittest.TestCase):
    """Unit tests for get_face_closest_to_direction with candidate_faces parameter."""
    
    def testAllFacesCandidateSelection(self):
        """Test face selection when all faces are candidates (default behavior)."""
        from elements.gxml_panel import PanelSide
        
        panel = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Direction pointing in +Z should select FRONT
        face = panel.get_face_closest_to_direction([0, 0, 1])
        self.assertEqual(face, PanelSide.FRONT, "Direction +Z should select FRONT")
        
        # Direction pointing in -Z should select BACK
        face = panel.get_face_closest_to_direction([0, 0, -1])
        self.assertEqual(face, PanelSide.BACK, "Direction -Z should select BACK")
        
        # Direction pointing in +X should select END
        face = panel.get_face_closest_to_direction([1, 0, 0])
        self.assertEqual(face, PanelSide.END, "Direction +X should select END")
        
        # Direction pointing in -X should select START
        face = panel.get_face_closest_to_direction([-1, 0, 0])
        self.assertEqual(face, PanelSide.START, "Direction -X should select START")
    
    def testRestrictedCandidateFaces(self):
        """Test face selection with restricted candidate faces.
        
        When candidate_faces is specified, only those faces should be considered,
        even if another face would have a higher dot product.
        """
        from elements.gxml_panel import PanelSide
        
        panel = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Direction at 44 degrees (more aligned with END than BACK)
        import math
        angle = 44
        direction = [math.cos(math.radians(angle)), 0, -math.sin(math.radians(angle))]
        
        # Without restriction, should select END (higher dot product)
        face_unrestricted = panel.get_face_closest_to_direction(direction)
        self.assertEqual(face_unrestricted, PanelSide.END,
                        "Without restriction, 44° direction should select END")
        
        # With FRONT/BACK restriction, should select BACK
        face_restricted = panel.get_face_closest_to_direction(
            direction,
            candidate_faces=[PanelSide.FRONT, PanelSide.BACK]
        )
        self.assertEqual(face_restricted, PanelSide.BACK,
                        "With FRONT/BACK restriction, 44° direction should select BACK")
    
    def testObtuseAngleWithRestriction(self):
        """Test face selection at obtuse angles with restriction."""
        from elements.gxml_panel import PanelSide
        import math
        
        panel = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Direction at 136 degrees (more aligned with START than BACK)
        angle = 136
        direction = [math.cos(math.radians(angle)), 0, -math.sin(math.radians(angle))]
        
        # Without restriction
        face_unrestricted = panel.get_face_closest_to_direction(direction)
        # This could be START or BACK depending on exact angle
        
        # With FRONT/BACK restriction, should select BACK
        face_restricted = panel.get_face_closest_to_direction(
            direction,
            candidate_faces=[PanelSide.FRONT, PanelSide.BACK]
        )
        self.assertEqual(face_restricted, PanelSide.BACK,
                        "With FRONT/BACK restriction, 136° direction should select BACK")
    
    def testSingleCandidateFace(self):
        """Test that single candidate face is always returned."""
        from elements.gxml_panel import PanelSide
        
        panel = GXMLMockPanel("p1", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], thickness=0.1)
        
        # Direction pointing completely away from FRONT
        direction = [0, 0, -1]  # Points to BACK
        
        # With only FRONT as candidate, should still return FRONT
        face = panel.get_face_closest_to_direction(
            direction,
            candidate_faces=[PanelSide.FRONT]
        )
        self.assertEqual(face, PanelSide.FRONT,
                        "Single candidate should always be returned")


class JointCapTests(unittest.TestCase):
    """Unit tests for joint cap polygon generation at 3+ panel joints."""
    
    def _get_cap_polygons(self, children):
        """Get cap polygons from children, excluding GXMLQuad face segments.
        
        Caps are GXMLPolygon instances that are NOT GXMLQuad (face segments).
        """
        from elements.gxml_polygon import GXMLPolygon
        from elements.gxml_quad import GXMLQuad
        return [c for c in children if isinstance(c, GXMLPolygon) and not isinstance(c, GXMLQuad)]
    
    def testThreePanelJointCapVertices(self):
        """Test that 3-panel joint generates cap polygons with correct vertices.
        
        Three panels meeting at [1,0,0] with angles matching the XML:
            <panel thickness="0.1"/>                     (along +X, origin to [1,0,0])
            <panel thickness="0.1" rotate="80"/>         (80° from +X, starts at [1,0,0])
            <panel thickness="0.1" rotate="-130" attach-to="0.0"/>  (-130° from +X, starts at [1,0,0])
        
        The cap polygons should fill the triangular gap between the mitered panel faces.
        Expected cap vertices at TOP (y=1):
            [0.976685, 1.0, 0.05]
            [0.958045, 1.0, -0.05]
            [1.05329, 1.0, -0.0142788]
        """
        from elements.gxml_polygon import GXMLPolygon
        
        # Create panels matching the XML layout - all meet at [1,0,0]
        # Panel positions match the real layout system output:
        #   0: start=[0,0,0], end=[1,0,0]
        #   1: start=[1,0,0], end=[1.17364818, 0, -0.98480775]
        #   2: start=[1,0,0], end=[1.64278761, 0, 0.76604444]
        
        p0 = GXMLMockPanel('p0', [0, 0, 0], [1, 0, 0], 0.1)
        p1 = GXMLMockPanel('p1', [1, 0, 0], [1.17364818, 0, -0.98480775], 0.1)
        p2 = GXMLMockPanel('p2', [1, 0, 0], [1.64278761, 0, 0.76604444], 0.1)
        
        solution = IntersectionSolver.solve([p0, p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Find the cap polygons - they should be on one of the panels
        caps = []
        for panel in [p0, p1, p2]:
            caps.extend(self._get_cap_polygons(panel.dynamicChildren))
        
        # Should have 2 caps: top and bottom
        self.assertEqual(len(caps), 2, "Should generate exactly 2 cap polygons (top and bottom)")
        
        top_cap = next((c for c in caps if c.subId == 'cap-top'), None)
        bottom_cap = next((c for c in caps if c.subId == 'cap-bottom'), None)
        
        self.assertIsNotNone(top_cap, "Should have a top cap")
        self.assertIsNotNone(bottom_cap, "Should have a bottom cap")
        
        # Expected vertices for TOP cap
        expected_top = [
            np.array([0.976685, 1.0, 0.05]),
            np.array([0.958045, 1.0, -0.05]),
            np.array([1.05329, 1.0, -0.0142788]),
        ]
        
        # Verify top cap has 3 vertices
        self.assertEqual(top_cap.vertex_count, 3, "Top cap should have 3 vertices")
        
        # Check each expected vertex is present (order may vary due to winding)
        for expected in expected_top:
            found = any(np.allclose(v, expected, atol=1e-4) for v in top_cap.vertices)
            self.assertTrue(found, f"Top cap should contain vertex near {expected}")
        
        # Expected vertices for BOTTOM cap (same x,z but y=0)
        expected_bottom = [
            np.array([0.976685, 0.0, 0.05]),
            np.array([0.958045, 0.0, -0.05]),
            np.array([1.05329, 0.0, -0.0142788]),
        ]
        
        # Verify bottom cap has 3 vertices
        self.assertEqual(bottom_cap.vertex_count, 3, "Bottom cap should have 3 vertices")
        
        # Check each expected vertex is present
        for expected in expected_bottom:
            found = any(np.allclose(v, expected, atol=1e-4) for v in bottom_cap.vertices)
            self.assertTrue(found, f"Bottom cap should contain vertex near {expected}")

    def testTwoPanelJointNoCap(self):
        """Test that 2-panel joints do NOT generate cap polygons (edges meet cleanly)."""
        # Create an L-joint with two panels
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], 0.2)
        panel2 = GXMLMockPanel("p2", [2, 0, 0], [2, 0, 2], 0.2)
        
        solution = IntersectionSolver.solve([panel1, panel2])
        
        # Build all geometry including caps
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Check that NO cap polygons were created for 2-panel joint
        all_children = panel1.dynamicChildren + panel2.dynamicChildren
        cap_polygons = self._get_cap_polygons(all_children)
        
        # 2-panel joints don't need caps - mitered edges meet cleanly
        self.assertEqual(len(cap_polygons), 0, "2-panel joints should not have cap polygons")

    def testCapWindingOrder(self):
        """Test that cap normals face correctly: top cap up (+Y), bottom cap down (-Y).
        
        Top cap should have CCW winding when viewed from above (+Y).
        Bottom cap should have CW winding when viewed from above (CCW from below).
        """
        # Create a simple 3-panel joint in XZ plane
        p0 = GXMLMockPanel('p0', [0, 0, 0], [1, 0, 0], 0.1)
        p1 = GXMLMockPanel('p1', [0, 0, 0], [0, 0, 1], 0.1)
        p2 = GXMLMockPanel('p2', [0, 0, 0], [-1, 0, -1], 0.1)
        
        solution = IntersectionSolver.solve([p0, p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Collect cap polygons
        all_children = p0.dynamicChildren + p1.dynamicChildren + p2.dynamicChildren
        caps = self._get_cap_polygons(all_children)
        
        top_cap = next((c for c in caps if c.subId == 'cap-top'), None)
        bottom_cap = next((c for c in caps if c.subId == 'cap-bottom'), None)
        
        self.assertIsNotNone(top_cap, "Should have top cap")
        self.assertIsNotNone(bottom_cap, "Should have bottom cap")
        
        # Calculate normal for top cap using cross product of first two edges
        v0, v1, v2 = top_cap.vertices[0], top_cap.vertices[1], top_cap.vertices[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        top_normal = np.cross(edge1, edge2)
        top_normal = top_normal / np.linalg.norm(top_normal)
        
        # Top cap normal should point up (+Y direction)
        self.assertGreater(top_normal[1], 0.9, 
                          f"Top cap normal should point up (+Y), got {top_normal}")
        
        # Calculate normal for bottom cap
        v0, v1, v2 = bottom_cap.vertices[0], bottom_cap.vertices[1], bottom_cap.vertices[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        bottom_normal = np.cross(edge1, edge2)
        bottom_normal = bottom_normal / np.linalg.norm(bottom_normal)
        
        # Bottom cap normal should point down (-Y direction)
        self.assertLess(bottom_normal[1], -0.9, 
                       f"Bottom cap normal should point down (-Y), got {bottom_normal}")

    def testCapAssignedToFirstPanel(self):
        """Test that cap polygons are assigned to the first panel in CCW order."""
        # Create 3-panel joint - panels are sorted CCW by their direction angle
        p0 = GXMLMockPanel('p0', [0, 0, 0], [1, 0, 0], 0.1)    # atan2(1,0) = pi/2 = 90°
        p1 = GXMLMockPanel('p1', [0, 0, 0], [0, 0, 1], 0.1)    # atan2(0,1) = 0°
        p2 = GXMLMockPanel('p2', [0, 0, 0], [-1, 0, 0], 0.1)   # atan2(-1,0) = -pi/2 = -90°
        
        solution = IntersectionSolver.solve([p0, p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Find which panel is first in CCW order
        joint = solution.intersections[0]
        first_panel = joint.panels[0].panel
        
        # All caps should be assigned to the first panel in CCW order
        all_panels = [p0, p1, p2]
        for panel in all_panels:
            caps = self._get_cap_polygons(panel.dynamicChildren)
            if panel == first_panel:
                self.assertEqual(len(caps), 2, f"First panel {panel.id} should have both caps")
                # Verify the parent references
                for cap in caps:
                    self.assertEqual(cap.parent, first_panel, "Cap parent should be the first panel")
                    self.assertEqual(cap.id, first_panel.id, "Cap id should match first panel's id")
            else:
                self.assertEqual(len(caps), 0, f"Non-first panel {panel.id} should have no caps")

    def testFourPanelJointQuadrilateralCap(self):
        """Test that 4-panel joint generates quadrilateral (4-vertex) caps."""
        from elements.solvers import IntersectionType
        
        # Create 4 panels meeting at origin like a plus sign
        p0 = GXMLMockPanel('p0', [0, 0, 0], [1, 0, 0], 0.1)    # +X
        p1 = GXMLMockPanel('p1', [0, 0, 0], [0, 0, 1], 0.1)    # +Z
        p2 = GXMLMockPanel('p2', [0, 0, 0], [-1, 0, 0], 0.1)   # -X
        p3 = GXMLMockPanel('p3', [0, 0, 0], [0, 0, -1], 0.1)   # -Z
        
        solution = IntersectionSolver.solve([p0, p1, p2, p3])
        
        # Verify it's a 4-panel joint
        self.assertEqual(len(solution.intersections), 1)
        self.assertEqual(solution.intersections[0].type, IntersectionType.JOINT)
        self.assertEqual(len(solution.intersections[0].panels), 4, "Should have 4 panels")
        
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Collect caps
        all_children = (p0.dynamicChildren + p1.dynamicChildren + 
                       p2.dynamicChildren + p3.dynamicChildren)
        caps = self._get_cap_polygons(all_children)
        
        # Should have 2 caps: top and bottom
        self.assertEqual(len(caps), 2, "Should have 2 cap polygons")
        
        # Each cap should have 4 vertices (one per panel)
        for cap in caps:
            self.assertEqual(cap.vertex_count, 4, 
                           f"4-panel joint cap {cap.subId} should have 4 vertices")

    def testFivePanelJointPentagonCap(self):
        """Test that 5-panel joint generates pentagonal (5-vertex) caps."""
        # Create 5 panels at 72° intervals
        import math
        panels = []
        for i in range(5):
            angle = i * (2 * math.pi / 5)  # 72° intervals
            end_x = math.cos(angle)
            end_z = math.sin(angle)
            panels.append(GXMLMockPanel(f'p{i}', [0, 0, 0], [end_x, 0, end_z], 0.1))
        
        solution = IntersectionSolver.solve(panels)
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Collect all caps
        all_children = []
        for p in panels:
            all_children.extend(p.dynamicChildren)
        caps = self._get_cap_polygons(all_children)
        
        self.assertEqual(len(caps), 2, "Should have 2 cap polygons")
        
        for cap in caps:
            self.assertEqual(cap.vertex_count, 5, 
                           f"5-panel joint cap {cap.subId} should have 5 vertices")

    def testJointCapWithVaryingThickness(self):
        """Test that caps correctly handle panels with different thicknesses."""
        # Create 3 panels with different thicknesses
        p0 = GXMLMockPanel('p0', [0, 0, 0], [1, 0, 0], 0.1)    # thin
        p1 = GXMLMockPanel('p1', [0, 0, 0], [0, 0, 1], 0.2)    # medium
        p2 = GXMLMockPanel('p2', [0, 0, 0], [-1, 0, 0], 0.3)   # thick
        
        solution = IntersectionSolver.solve([p0, p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Collect caps
        all_children = p0.dynamicChildren + p1.dynamicChildren + p2.dynamicChildren
        caps = self._get_cap_polygons(all_children)
        
        self.assertEqual(len(caps), 2, "Should have 2 cap polygons")
        
        top_cap = next((c for c in caps if c.subId == 'cap-top'), None)
        bottom_cap = next((c for c in caps if c.subId == 'cap-bottom'), None)
        
        self.assertIsNotNone(top_cap)
        self.assertIsNotNone(bottom_cap)
        
        # All top cap vertices should be at y=1
        for v in top_cap.vertices:
            self.assertAlmostEqual(v[1], 1.0, places=4,
                                   msg=f"Top cap vertex y should be 1.0, got {v[1]}")
        
        # All bottom cap vertices should be at y=0
        for v in bottom_cap.vertices:
            self.assertAlmostEqual(v[1], 0.0, places=4,
                                   msg=f"Bottom cap vertex y should be 0.0, got {v[1]}")
        
        # Cap should still have 3 vertices
        self.assertEqual(top_cap.vertex_count, 3)
        self.assertEqual(bottom_cap.vertex_count, 3)

    def testJointAtEndpointGeneratesCaps(self):
        """Test that joints at panel endpoints (not origins) still generate correct caps."""
        # Create 3 panels meeting at [2,0,0], not at origin
        p0 = GXMLMockPanel('p0', [0, 0, 0], [2, 0, 0], 0.1)     # ends at joint
        p1 = GXMLMockPanel('p1', [2, 0, 0], [3, 0, 1], 0.1)     # starts at joint
        p2 = GXMLMockPanel('p2', [2, 0, 0], [3, 0, -1], 0.1)    # starts at joint
        
        solution = IntersectionSolver.solve([p0, p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Collect caps
        all_children = p0.dynamicChildren + p1.dynamicChildren + p2.dynamicChildren
        caps = self._get_cap_polygons(all_children)
        
        self.assertEqual(len(caps), 2, "Should have 2 cap polygons")
        
        top_cap = next((c for c in caps if c.subId == 'cap-top'), None)
        self.assertIsNotNone(top_cap)
        
        # All vertices should be near the joint at x~2
        for v in top_cap.vertices:
            self.assertGreater(v[0], 1.5, f"Cap vertex X should be near joint (>1.5), got {v[0]}")
            self.assertLess(v[0], 2.5, f"Cap vertex X should be near joint (<2.5), got {v[0]}")


class CrossingCapTests(unittest.TestCase):
    """Tests for crossing cap polygon generation at crossing intersections."""

    def _get_cap_polygons(self, children):
        """Get cap polygons from children, excluding face segments (GXMLQuad)."""
        from elements.gxml_polygon import GXMLPolygon
        from elements.gxml_quad import GXMLQuad
        return [c for c in children if isinstance(c, GXMLPolygon) and not isinstance(c, GXMLQuad)]

    def testCrossingGeneratesTopAndBottomCaps(self):
        """Test that a crossing intersection generates TOP and BOTTOM cap polygons."""
        # Create a perpendicular crossing
        p1 = GXMLMockPanel("p1", [-1, 0, 0], [1, 0, 0], thickness=0.1)  # horizontal along X
        p2 = GXMLMockPanel("p2", [0, 0, -1], [0, 0, 1], thickness=0.1)  # perpendicular along Z
        
        solution = IntersectionSolver.solve([p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Collect cap polygons from both panels (caps assigned to first panel in intersection)
        all_children = p1.dynamicChildren + p2.dynamicChildren
        caps = self._get_cap_polygons(all_children)
        crossing_caps = [c for c in caps if c.subId.startswith('crossing-cap')]
        
        self.assertEqual(len(crossing_caps), 2, "Should have 2 crossing cap polygons")
        
        top_cap = next((c for c in crossing_caps if c.subId == 'crossing-cap-top'), None)
        bottom_cap = next((c for c in crossing_caps if c.subId == 'crossing-cap-bottom'), None)
        
        self.assertIsNotNone(top_cap, "Should have a top crossing cap")
        self.assertIsNotNone(bottom_cap, "Should have a bottom crossing cap")

    def testCrossingCapVerticesAreAtCorrectHeight(self):
        """Test that crossing cap vertices are at the correct Y positions."""
        p1 = GXMLMockPanel("p1", [-1, 0, 0], [1, 0, 0], thickness=0.1)
        p2 = GXMLMockPanel("p2", [0, 0, -1], [0, 0, 1], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        all_children = p1.dynamicChildren + p2.dynamicChildren
        caps = self._get_cap_polygons(all_children)
        
        top_cap = next((c for c in caps if c.subId == 'crossing-cap-top'), None)
        bottom_cap = next((c for c in caps if c.subId == 'crossing-cap-bottom'), None)
        
        # Top cap vertices should be at y=1 (panel height)
        for v in top_cap.vertices:
            self.assertAlmostEqual(v[1], 1.0, places=4,
                                  msg=f"Top cap vertex should be at y=1.0, got {v[1]}")
        
        # Bottom cap vertices should be at y=0
        for v in bottom_cap.vertices:
            self.assertAlmostEqual(v[1], 0.0, places=4,
                                  msg=f"Bottom cap vertex should be at y=0.0, got {v[1]}")

    def testCrossingCapHasFourVertices(self):
        """Test that crossing caps are quadrilaterals with 4 vertices."""
        p1 = GXMLMockPanel("p1", [-1, 0, 0], [1, 0, 0], thickness=0.1)
        p2 = GXMLMockPanel("p2", [0, 0, -1], [0, 0, 1], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        all_children = p1.dynamicChildren + p2.dynamicChildren
        caps = self._get_cap_polygons(all_children)
        top_cap = next((c for c in caps if c.subId == 'crossing-cap-top'), None)
        
        self.assertEqual(len(top_cap.vertices), 4, "Crossing cap should have 4 vertices")

    def testCrossingCapCenteredAtIntersection(self):
        """Test that crossing cap is centered at the intersection point."""
        import numpy as np
        
        p1 = GXMLMockPanel("p1", [-2, 0, 0], [2, 0, 0], thickness=0.1)  # crosses at x=0
        p2 = GXMLMockPanel("p2", [0, 0, -2], [0, 0, 2], thickness=0.1)  # crosses at z=0
        
        solution = IntersectionSolver.solve([p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        all_children = p1.dynamicChildren + p2.dynamicChildren
        caps = self._get_cap_polygons(all_children)
        top_cap = next((c for c in caps if c.subId == 'crossing-cap-top'), None)
        
        # Cap center should be at origin (intersection point)
        center = np.mean(top_cap.vertices, axis=0)
        self.assertAlmostEqual(center[0], 0.0, places=3,
                              msg=f"Cap center X should be ~0, got {center[0]}")
        self.assertAlmostEqual(center[2], 0.0, places=3,
                              msg=f"Cap center Z should be ~0, got {center[2]}")

    def testTopBottomFacesSplitAtCrossing(self):
        """Test that TOP and BOTTOM faces are split into segments at crossings."""
        p1 = GXMLMockPanel("p1", [-1, 0, 0], [1, 0, 0], thickness=0.1)
        p2 = GXMLMockPanel("p2", [0, 0, -1], [0, 0, 1], thickness=0.1)
        
        solution = IntersectionSolver.solve([p1, p2])
        GeometryBuilder.build_all(FaceSolver.solve(solution))
        
        # Count TOP face segments
        top_faces = [c for c in p1.dynamicChildren 
                     if hasattr(c, 'subId') and c.subId.startswith('top')]
        
        # Should have 2 TOP segments (split at crossing)
        self.assertEqual(len(top_faces), 2, 
                        f"Should have 2 top segments, got {len(top_faces)}: {[f.subId for f in top_faces]}")

    def testCrossingCapSizeScalesWithThickness(self):
        """Test that crossing cap size is determined by panel thickness, not height."""
        import numpy as np
        
        # Create two crossings with different thicknesses
        # Both have same length (4 units) and height (1 unit)
        
        # Thin panels (thickness 0.2)
        p1_thin = GXMLMockPanel("p1", [-2, 0, 0], [2, 0, 0], thickness=0.2)
        p2_thin = GXMLMockPanel("p2", [0, 0, -2], [0, 0, 2], thickness=0.2)
        
        solution_thin = IntersectionSolver.solve([p1_thin, p2_thin])
        GeometryBuilder.build_all(FaceSolver.solve(solution_thin))
        
        all_children_thin = p1_thin.dynamicChildren + p2_thin.dynamicChildren
        caps_thin = self._get_cap_polygons(all_children_thin)
        top_cap_thin = next((c for c in caps_thin if c.subId == 'crossing-cap-top'), None)
        
        # Thick panels (thickness 0.4 - double the thin ones)
        p1_thick = GXMLMockPanel("p1", [-2, 0, 0], [2, 0, 0], thickness=0.4)
        p2_thick = GXMLMockPanel("p2", [0, 0, -2], [0, 0, 2], thickness=0.4)
        
        solution_thick = IntersectionSolver.solve([p1_thick, p2_thick])
        GeometryBuilder.build_all(FaceSolver.solve(solution_thick))
        
        all_children_thick = p1_thick.dynamicChildren + p2_thick.dynamicChildren
        caps_thick = self._get_cap_polygons(all_children_thick)
        top_cap_thick = next((c for c in caps_thick if c.subId == 'crossing-cap-top'), None)
        
        self.assertIsNotNone(top_cap_thin, "Thin panels should have crossing cap")
        self.assertIsNotNone(top_cap_thick, "Thick panels should have crossing cap")
        
        # Calculate cap areas (cross product of diagonals / 2 for quadrilateral)
        def quad_area(vertices):
            # Using shoelace formula for quadrilateral in XZ plane
            v = vertices
            area = 0.5 * abs(
                (v[0][0] - v[2][0]) * (v[1][2] - v[3][2]) -
                (v[0][2] - v[2][2]) * (v[1][0] - v[3][0])
            )
            return area
        
        area_thin = quad_area(top_cap_thin.vertices)
        area_thick = quad_area(top_cap_thick.vertices)
        
        # Thick cap should be ~4x larger (2x thickness in each direction)
        # For perpendicular crossing: cap area ≈ thickness1 * thickness2
        # Thin: 0.2 * 0.2 = 0.04
        # Thick: 0.4 * 0.4 = 0.16 (4x larger)
        area_ratio = area_thick / area_thin
        self.assertAlmostEqual(area_ratio, 4.0, places=1,
                              msg=f"Thick cap should be ~4x larger than thin cap, got ratio {area_ratio}")


if __name__ == '__main__':
    unittest.main()

