"""
Unit tests for the intersection solver.
"""

import unittest
import numpy as np
from elements.solvers import (
    IntersectionSolver, 
    IntersectionType,
    PanelAxis
)
from tests.test_fixtures.mocks import GXMLMockPanel

class IntersectionSolverUnitTests(unittest.TestCase):
    """Unit tests for IntersectionSolver and IntersectionSolution"""
   
    # ========================================================================
    # BASIC INTERSECTION TYPES (2-panel)
    # ========================================================================
    
    def testSolveJoints(self):
        """Test multiple joints (endpoint-to-endpoint intersections)"""
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0, 0, 0], [0, 0, 1], 0.1)
        panel3 = GXMLMockPanel("p3", [2, 0, 0], [2, 0, 1], 0.1)
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3])
        
        self.assertEqual(len(solution.intersections), 2, "Should find exactly 2 intersections")
        
        # Both should be joints
        self.assertEqual(solution.intersections[0].type, IntersectionType.JOINT, "First intersection should be joint")
        self.assertEqual(solution.intersections[1].type, IntersectionType.JOINT, "Second intersection should be joint")
        
        # Verify first joint at (0,0,0) - panel1 start, panel2 start
        joint0 = solution.intersections[0]
        self.assertTrue(np.allclose(joint0.position, [0, 0, 0], atol=1e-6), "First joint should be at (0, 0, 0)")
        self.assertEqual(len(joint0.panels), 2, "First joint should involve 2 panels")
        # Panels are sorted CCW when viewed from +Y, using atan2(x, z) angle calculation
        # This makes 0° point along +Z axis, so: panel2 (+Z, 0°) comes before panel1 (+X, 90°)
        self.assertEqual(joint0.panels[0].panel, panel2, "First panel should be p2")
        self.assertEqual(joint0.panels[1].panel, panel1, "Second panel should be p1")
        self.assertAlmostEqual(joint0.panels[0].t, 0.0, places=6, msg="Panel p2 should be at t=0.0 (start)")
        self.assertAlmostEqual(joint0.panels[1].t, 0.0, places=6, msg="Panel p1 should be at t=0.0 (start)")
        
        # Verify second joint at (2,0,0) - panel1 end, panel3 start
        joint1 = solution.intersections[1]
        self.assertTrue(np.allclose(joint1.position, [2, 0, 0], atol=1e-6), "Second joint should be at (2, 0, 0)")
        self.assertEqual(len(joint1.panels), 2, "Second joint should involve 2 panels")
        self.assertEqual(joint1.panels[0].panel, panel1, "First panel should be p1")
        self.assertEqual(joint1.panels[1].panel, panel3, "Second panel should be p3")
        self.assertAlmostEqual(joint1.panels[0].t, 1.0, places=6, msg="Panel p1 should be at t=1.0 (end)")
        self.assertAlmostEqual(joint1.panels[1].t, 0.0, places=6, msg="Panel p3 should be at t=0.0 (start)")
        
        # Joints have no panel regions (all panels at endpoints, no splits needed)
    
    def testSolveTJunctions(self):
        """Test multiple T-junctions on a single panel"""
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [1, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0.75, 0, 0], [0.75, 0, 1], 0.1)
        panel3 = GXMLMockPanel("p3", [1, 0, -0.5], [1, 0, 0.5], 0.1)
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3])
        
        self.assertEqual(len(solution.intersections), 2, "Should find exactly 2 intersections")
        
        tj0 = solution.intersections[0]
        tj1 = solution.intersections[1]
        
        # Both should be T-junctions
        self.assertEqual(tj0.type, IntersectionType.T_JUNCTION, "First intersection should be T-junction")
        self.assertEqual(tj1.type, IntersectionType.T_JUNCTION, "Second intersection should be T-junction")
        
        # Verify first T-junction at x=0.75 (panel1 at t=0.75, panel2 at t=0)
        self.assertTrue(np.allclose(tj0.position, [0.75, 0, 0], atol=1e-6), "First intersection should be at (0.75, 0, 0)")
        self.assertEqual(len(tj0.panels), 2, "First T-junction should involve 2 panels")
        self.assertEqual(tj0.panels[0].panel, panel1, "First panel should be p1")
        self.assertEqual(tj0.panels[1].panel, panel2, "Second panel should be p2")
        self.assertAlmostEqual(tj0.panels[0].t, 0.75, places=6, msg="Panel p1 should be at t=0.75")
        self.assertAlmostEqual(tj0.panels[1].t, 0.0, places=6, msg="Panel p2 should be at t=0.0 (start)")
        
        # Verify second T-junction at x=1 (panel1 at t=1.0, panel3 at t=0.5)
        self.assertTrue(np.allclose(tj1.position, [1, 0, 0], atol=1e-6), "Second intersection should be at (1, 0, 0)")
        self.assertEqual(len(tj1.panels), 2, "Second T-junction should involve 2 panels")
        self.assertEqual(tj1.panels[0].panel, panel1, "First panel should be p1")
        self.assertEqual(tj1.panels[1].panel, panel3, "Second panel should be p3")
        self.assertAlmostEqual(tj1.panels[0].t, 1.0, places=6, msg="Panel p1 should be at t=1.0 (end)")
        self.assertAlmostEqual(tj1.panels[1].t, 0.5, places=6, msg="Panel p3 should be at t=0.5 (midpoint)")
        
        # Verify unified regions are generated correctly via solution.regions_per_panel
        # panel1 at t=0.75 (midspan - should have partitions)
        self.assertIsNotNone(solution.regions_per_panel.get(panel1), "Panel p1 at midspan should have partitions")
        p1_partitions = solution.regions_per_panel[panel1]
        self.assertEqual(len(p1_partitions.children), 2, "Panel p1 should be split into 2 regions")
        self.assertEqual(p1_partitions.children[0].tStart, 0.0, "First region should start at 0.0")
        self.assertAlmostEqual(p1_partitions.children[1].tStart, 0.75, places=6, msg="Second region should start at 0.75")
        
        # panel3 at t=0.5 (midspan - has partitions)
        self.assertIsNotNone(solution.regions_per_panel.get(panel3), "Panel p3 at midspan should have partitions")
        p3_partitions = solution.regions_per_panel[panel3]
        self.assertEqual(len(p3_partitions.children), 2, "Panel p3 should be split into 2 regions")
        self.assertEqual(p3_partitions.children[0].tStart, 0.0, "First region should start at 0.0")
        self.assertAlmostEqual(p3_partitions.children[1].tStart, 0.5, places=6, msg="Second region should start at 0.5")
    
    def testSolveTJunctionWithFloatingPointNoise(self):
        """Test T-junction detection is robust to floating point noise in coordinates.
        
        This tests the case where panel coordinates have tiny floating point errors
        (e.g., -3e-17 instead of 0.0) that can occur from rotation/transformation
        calculations. The intersection solver should still detect the T-junction.
        
        Real-world case: A panel rotated 90 degrees may have start coordinates like
        [0.5, 0.0, -3.061617e-17] instead of exact [0.5, 0.0, 0.0] due to trig functions.
        """
        # Panel 1: horizontal along X axis
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [1, 0, 0], 0.1)
        
        # Panel 2: starts at panel 1's midpoint with tiny floating point noise in Z
        # This simulates what happens when a panel is rotated and transformed
        panel2 = GXMLMockPanel("p2", [0.5, 0, -3.061617e-17], [0.5, 0, -1], 0.1)
        
        solution = IntersectionSolver.solve([panel1, panel2])
        
        self.assertEqual(len(solution.intersections), 1, "Should find exactly 1 intersection despite floating point noise")
        
        tj = solution.intersections[0]
        self.assertEqual(tj.type, IntersectionType.T_JUNCTION, "Should be a T-junction")
        self.assertTrue(np.allclose(tj.position, [0.5, 0, 0], atol=1e-6), "T-junction should be at (0.5, 0, 0)")
    
    def testSolveCrossings(self):
        """Test multiple crossings (midspan-to-midspan intersections)"""
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0.5, 0, -0.5], [0.5, 0, 0.5], 0.1)
        panel3 = GXMLMockPanel("p3", [1.5, 0, -0.5], [1.5, 0, 0.5], 0.1)
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3])
        
        self.assertEqual(len(solution.intersections), 2, "Should find exactly 2 intersections")
        
        # Both should be crossings
        self.assertEqual(solution.intersections[0].type, IntersectionType.CROSSING, "First intersection should be crossing")
        self.assertEqual(solution.intersections[1].type, IntersectionType.CROSSING, "Second intersection should be crossing")
        
        # Verify first crossing at (0.5,0,0) - panel1 at t=0.25, panel2 at t=0.5
        crossing0 = solution.intersections[0]
        self.assertTrue(np.allclose(crossing0.position, [0.5, 0, 0], atol=1e-6), "First crossing should be at (0.5, 0, 0)")
        self.assertEqual(len(crossing0.panels), 2, "First crossing should involve 2 panels")
        
        self.assertEqual(crossing0.panels[0].panel, panel2, "First panel should be p2")
        self.assertEqual(crossing0.panels[1].panel, panel1, "Second panel should be p1")
        self.assertAlmostEqual(crossing0.panels[0].t, 0.5, places=6, msg="Panel p2 should be at t=0.5 (midspan)")
        self.assertAlmostEqual(crossing0.panels[1].t, 0.25, places=6, msg="Panel p1 should be at t=0.25 (midspan)")
        
        # Verify second crossing at (1.5,0,0) - panel1 at t=0.75, panel3 at t=0.5
        crossing1 = solution.intersections[1]
        self.assertTrue(np.allclose(crossing1.position, [1.5, 0, 0], atol=1e-6), "Second crossing should be at (1.5, 0, 0)")
        self.assertEqual(len(crossing1.panels), 2, "Second crossing should involve 2 panels")
        
        
        self.assertEqual(crossing1.panels[0].panel, panel1, "First panel should be p1")
        self.assertEqual(crossing1.panels[1].panel, panel3, "Second panel should be p3")
        self.assertAlmostEqual(crossing1.panels[0].t, 0.75, places=6, msg="Panel p1 should be at t=0.75 (midspan)")
        self.assertAlmostEqual(crossing1.panels[1].t, 0.5, places=6, msg="Panel p3 should be at t=0.5 (midspan)")
        
        # Verify unified partition structure via solution.regions_per_panel
        # Panel1 has two crossings, so its unified region should include both split points
        self.assertIsNotNone(solution.regions_per_panel.get(panel1), "Panel p1 should have partitions")
        p1_partitions = solution.regions_per_panel[panel1]
        self.assertEqual(len(p1_partitions.children), 3, "Panel p1 should be split into 3 regions (two crossings)")
        self.assertEqual(p1_partitions.children[0].tStart, 0.0, "First region should start at 0.0")
        self.assertAlmostEqual(p1_partitions.children[1].tStart, 0.25, places=6, msg="Second region should start at 0.25")
        self.assertAlmostEqual(p1_partitions.children[2].tStart, 0.75, places=6, msg="Third region should start at 0.75")
    
    # ========================================================================
    # MULTI-PANEL VARIATIONS
    # ========================================================================
    
    def testThreePanelJoint(self):
        """Test 3 panels converging at a single joint"""
        # Three panels meeting at origin, forming a Y shape
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [1, 0, 0], 0.1)      # +X direction
        panel2 = GXMLMockPanel("p2", [0, 0, 0], [0, 0, 1], 0.1)      # +Z direction
        panel3 = GXMLMockPanel("p3", [0, 0, 0], [-1, 0, -1], 0.1)    # -X,-Z direction
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3])
        
        # Should find exactly 1 intersection (joint at origin)
        self.assertEqual(len(solution.intersections), 1, "Should find exactly 1 joint")
        
        joint = solution.intersections[0]
        self.assertEqual(joint.type, IntersectionType.JOINT, "Should be a joint")
        self.assertTrue(np.allclose(joint.position, [0, 0, 0], atol=1e-6), "Joint should be at origin")
        self.assertEqual(len(joint.panels), 3, "Joint should involve exactly 3 panels")
        
        # All panels should be at their start (t=0)
        for panel_at_intersection in joint.panels:
            self.assertAlmostEqual(panel_at_intersection.t, 0.0, places=6, 
                                 msg=f"Panel {panel_at_intersection.panel.id} should be at t=0.0 (start)")
        
        # Panels are sorted CCW when viewed from +Y, using atan2(x, z) where 0° = +Z axis
        # p3: atan2(-1, -1) ≈ -135° → p2: atan2(0, 1) = 0° → p1: atan2(1, 0) = 90°
        # Expected CCW order: p3 (-X,-Z), p2 (+Z), p1 (+X)
        self.assertEqual(joint.panels[0].panel, panel3, "First panel (CCW) should be p3")
        self.assertEqual(joint.panels[1].panel, panel2, "Second panel (CCW) should be p2")
        self.assertEqual(joint.panels[2].panel, panel1, "Third panel (CCW) should be p1")
        
        # Joints have no panel regions (all panels at endpoints, no splits needed)
    
    def testThreePanelJointMixedDirections(self):
        """Test 3 panels at a joint with mixed travel directions (some toward, some away).
        
        This tests the scenario where:
        - Panel 0 travels TOWARD the joint (END at joint) along +X
        - Panel 1 travels AWAY from the joint (START at joint) along -Z  
        - Panel 2 travels AWAY from the joint (START at joint) along +X+Z diagonal
        
        The CCW sorting should use the OUTWARD direction from the joint for each panel,
        regardless of whether the panel is traveling toward or away from the joint.
        """
        # Panel 0: END at (1,0,0), traveling +X toward the joint
        panel0 = GXMLMockPanel("p0", [0, 0, 0], [1, 0, 0], 0.1)
        # Panel 1: START at (1,0,0), traveling -Z away from the joint  
        panel1 = GXMLMockPanel("p1", [1, 0, 0], [1, 0, -1], 0.1)
        # Panel 2: START at (1,0,0), traveling +X+Z diagonal (45°) away from the joint
        panel2 = GXMLMockPanel("p2", [1, 0, 0], [2, 0, 1], 0.1)
        
        solution = IntersectionSolver.solve([panel0, panel1, panel2])
        
        self.assertEqual(len(solution.intersections), 1, "Should find exactly 1 joint")
        joint = solution.intersections[0]
        self.assertEqual(joint.type, IntersectionType.JOINT, "Should be a joint")
        self.assertTrue(np.allclose(joint.position, [1, 0, 0], atol=1e-6), "Joint should be at (1, 0, 0)")
        self.assertEqual(len(joint.panels), 3, "Joint should involve 3 panels")
        
        # Verify travel directions (which endpoint is at the joint)
        panel_dict = {p.panel.id: p for p in joint.panels}
        self.assertAlmostEqual(panel_dict["p0"].t, 1.0, places=6, 
                             msg="Panel p0 should be at t=1.0 (END at joint, traveling toward)")
        self.assertAlmostEqual(panel_dict["p1"].t, 0.0, places=6, 
                             msg="Panel p1 should be at t=0.0 (START at joint, traveling away)")
        self.assertAlmostEqual(panel_dict["p2"].t, 0.0, places=6, 
                             msg="Panel p2 should be at t=0.0 (START at joint, traveling away)")
        
        # CCW sorting uses OUTWARD direction from joint:
        # - Panel 0: END at joint, so outward = start - end = (-1, 0, 0), angle = atan2(-1, 0) = -90°
        # - Panel 1: START at joint, so outward = end - start = (0, 0, -1), angle = atan2(0, -1) = 180°
        # - Panel 2: START at joint, so outward = end - start = (1, 0, 1), angle = atan2(1, 1) = 45°
        #
        # Sorted ascending: -90° (p0) → 45° (p2) → 180° (p1)
        self.assertEqual(joint.panels[0].panel.id, "p0", "First panel (CCW) should be p0 at -90°")
        self.assertEqual(joint.panels[1].panel.id, "p2", "Second panel (CCW) should be p2 at 45°")
        self.assertEqual(joint.panels[2].panel.id, "p1", "Third panel (CCW) should be p1 at 180°")

    def testTravelDirectionAtJoint(self):
        """Test that travel direction (toward/away) is correctly identified by t-value.
        
        A panel's t-value at the joint determines its travel direction:
        - t ≈ 0.0 (START at joint): Panel travels AWAY from joint
        - t ≈ 1.0 (END at joint): Panel travels TOWARD joint
        
        This affects which face (FRONT/BACK) is on which side of the wedge.
        """
        from elements.solvers.gxml_face_solver import FaceSolver, JointSide
        
        # Create a 3-panel joint with mixed directions at (1, 0, 0)
        # P0: END at joint (traveling toward) - goes from (0,0,0) to (1,0,0)
        panel0 = GXMLMockPanel("p0", [0, 0, 0], [1, 0, 0], 0.1)
        # P1: START at joint (traveling away) - goes from (1,0,0) to (1,0,-1)
        panel1 = GXMLMockPanel("p1", [1, 0, 0], [1, 0, -1], 0.1)
        # P2: START at joint (traveling away) - goes from (1,0,0) to (2,0,1)
        panel2 = GXMLMockPanel("p2", [1, 0, 0], [2, 0, 1], 0.1)
        
        solution = IntersectionSolver.solve([panel0, panel1, panel2])
        joint = solution.intersections[0]
        
        # Create lookup by panel id
        entries = {e.panel.id: e for e in joint.panels}
        
        # Verify t-values indicate correct travel direction
        self.assertGreater(entries["p0"].t, 0.5, 
            "P0 should have t > 0.5 (END at joint, traveling TOWARD)")
        self.assertLess(entries["p1"].t, 0.5, 
            "P1 should have t < 0.5 (START at joint, traveling AWAY)")
        self.assertLess(entries["p2"].t, 0.5, 
            "P2 should have t < 0.5 (START at joint, traveling AWAY)")
        
        # Verify the helper correctly identifies which face is on which side
        # For END at joint: FRONT is CCW side, BACK is CW side
        # For START at joint: BACK is CCW side, FRONT is CW side
        from elements.gxml_panel import PanelSide
        
        p0_ccw = FaceSolver._get_outward_face(entries["p0"], JointSide.CCW)
        p0_cw = FaceSolver._get_outward_face(entries["p0"], JointSide.CW)
        self.assertEqual(p0_ccw, PanelSide.FRONT, "P0 (END at joint): CCW side should be FRONT")
        self.assertEqual(p0_cw, PanelSide.BACK, "P0 (END at joint): CW side should be BACK")
        
        p1_ccw = FaceSolver._get_outward_face(entries["p1"], JointSide.CCW)
        p1_cw = FaceSolver._get_outward_face(entries["p1"], JointSide.CW)
        self.assertEqual(p1_ccw, PanelSide.BACK, "P1 (START at joint): CCW side should be BACK")
        self.assertEqual(p1_cw, PanelSide.FRONT, "P1 (START at joint): CW side should be FRONT")
        
        p2_ccw = FaceSolver._get_outward_face(entries["p2"], JointSide.CCW)
        p2_cw = FaceSolver._get_outward_face(entries["p2"], JointSide.CW)
        self.assertEqual(p2_ccw, PanelSide.BACK, "P2 (START at joint): CCW side should be BACK")
        self.assertEqual(p2_cw, PanelSide.FRONT, "P2 (START at joint): CW side should be FRONT")
        self.assertEqual(p2_cw, PanelSide.FRONT, "P2 (START at joint): CW side should be FRONT")

    def testAdjacentFacesWithMixedTravelDirections(self):
        """Test face adjacency for a 3-panel joint with mixed travel directions.
        
        This test mimics a configuration similar to:
            <panel thickness="0.1"/>
            <panel thickness="0.1" rotate="80"/>
            <panel thickness="0.1" rotate="-135" attach-to="0.0"/>
        
        Where:
        - P0's END is at the joint (traveling toward)
        - P1's START is at the joint (traveling away)
        - P2's START is at the joint (traveling away)
        
        CCW order (by angle): P2 (-135°) → P0 (-90°) → P1 (80°)
        
        Travel direction determines which face is on which side:
        - P0 (END at joint): FRONT on CCW side, BACK on CW side
        - P1 (START at joint): BACK on CCW side, FRONT on CW side
        - P2 (START at joint): BACK on CCW side, FRONT on CW side
        
        Expected adjacencies (CCW side meets next panel's CW side):
        - P2.BACK (CCW) ↔ P0.BACK (CW)
        - P0.FRONT (CCW) ↔ P1.FRONT (CW)  
        - P1.BACK (CCW) ↔ P2.FRONT (CW)
        """
        from elements.solvers.gxml_face_solver import FaceSolver
        from elements.gxml_panel import PanelSide
        import math
        
        # Joint at origin, panels radiating outward with correct travel directions
        # P0: END at origin (traveling toward from -X direction), angle -90°
        panel0 = GXMLMockPanel("p0", [-1, 0, 0], [0, 0, 0], 0.1)
        
        # P1: START at origin, angle 80°
        angle1 = math.radians(80)
        p1_end = [math.sin(angle1), 0, math.cos(angle1)]
        panel1 = GXMLMockPanel("p1", [0, 0, 0], p1_end, 0.1)
        
        # P2: START at origin, angle -135°
        angle2 = math.radians(-135)
        p2_end = [math.sin(angle2), 0, math.cos(angle2)]
        panel2 = GXMLMockPanel("p2", [0, 0, 0], p2_end, 0.1)
        
        solution = IntersectionSolver.solve([panel0, panel1, panel2])
        
        self.assertEqual(len(solution.intersections), 1, "Should find exactly 1 joint")
        joint = solution.intersections[0]
        self.assertEqual(joint.type, IntersectionType.JOINT)
        
        # Verify CCW order: P2 → P0 → P1
        self.assertEqual([p.panel.id for p in joint.panels], ["p2", "p0", "p1"],
            "CCW order should be P2 (-135°) → P0 (-90°) → P1 (80°)")
        
        # Verify travel directions
        entries = {e.panel.id: e for e in joint.panels}
        self.assertGreater(entries["p0"].t, 0.5, "P0 END should be at joint")
        self.assertLess(entries["p1"].t, 0.5, "P1 START should be at joint")
        self.assertLess(entries["p2"].t, 0.5, "P2 START should be at joint")
        
        # Test expected adjacencies based on CCW order and travel directions
        
        # P2.BACK (CCW side) should be adjacent to P0.BACK (CW side)
        adj = FaceSolver._get_adjacent_face(joint, panel2, PanelSide.BACK)
        self.assertIsNotNone(adj, "P2.BACK should have an adjacent face")
        self.assertEqual(adj[0].id, "p0", "P2.BACK should be adjacent to P0")
        self.assertEqual(adj[1], PanelSide.BACK, "P2.BACK should be adjacent to P0.BACK")
        
        # P0.BACK (CW side) should be adjacent to P2.BACK (CCW side) - reverse
        adj = FaceSolver._get_adjacent_face(joint, panel0, PanelSide.BACK)
        self.assertIsNotNone(adj, "P0.BACK should have an adjacent face")
        self.assertEqual(adj[0].id, "p2", "P0.BACK should be adjacent to P2")
        self.assertEqual(adj[1], PanelSide.BACK, "P0.BACK should be adjacent to P2.BACK")
        
        # P0.FRONT (CCW side) should be adjacent to P1.FRONT (CW side)
        adj = FaceSolver._get_adjacent_face(joint, panel0, PanelSide.FRONT)
        self.assertIsNotNone(adj, "P0.FRONT should have an adjacent face")
        self.assertEqual(adj[0].id, "p1", "P0.FRONT should be adjacent to P1")
        self.assertEqual(adj[1], PanelSide.FRONT, "P0.FRONT should be adjacent to P1.FRONT")
        
        # P1.FRONT (CW side) should be adjacent to P0.FRONT (CCW side) - reverse
        adj = FaceSolver._get_adjacent_face(joint, panel1, PanelSide.FRONT)
        self.assertIsNotNone(adj, "P1.FRONT should have an adjacent face")
        self.assertEqual(adj[0].id, "p0", "P1.FRONT should be adjacent to P0")
        self.assertEqual(adj[1], PanelSide.FRONT, "P1.FRONT should be adjacent to P0.FRONT")
        
        # P1.BACK (CCW side) should be adjacent to P2.FRONT (CW side)
        adj = FaceSolver._get_adjacent_face(joint, panel1, PanelSide.BACK)
        self.assertIsNotNone(adj, "P1.BACK should have an adjacent face")
        self.assertEqual(adj[0].id, "p2", "P1.BACK should be adjacent to P2")
        self.assertEqual(adj[1], PanelSide.FRONT, "P1.BACK should be adjacent to P2.FRONT")
        
        # P2.FRONT (CW side) should be adjacent to P1.BACK (CCW side) - reverse
        adj = FaceSolver._get_adjacent_face(joint, panel2, PanelSide.FRONT)
        self.assertIsNotNone(adj, "P2.FRONT should have an adjacent face")
        self.assertEqual(adj[0].id, "p1", "P2.FRONT should be adjacent to P1")
        self.assertEqual(adj[1], PanelSide.BACK, "P2.FRONT should be adjacent to P1.BACK")

    def testThreePanelTJunction(self):
        """Test 3 panels converging at a single T-junction (2 panels end at 1 panel's midspan)"""
        # panel1 is horizontal, panel2 and panel3 both end at its midpoint
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], 0.1)      # Horizontal base
        panel2 = GXMLMockPanel("p2", [1, 0, 0], [1, 0, 1], 0.1)      # Vertical, starts at p1's midpoint
        panel3 = GXMLMockPanel("p3", [1, 0, 0], [1, 0, -1], 0.1)     # Vertical, starts at p1's midpoint
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3])
        
        # Should find exactly 1 intersection (T-junction at (1,0,0))
        self.assertEqual(len(solution.intersections), 1, "Should find exactly 1 T-junction")
        
        tj = solution.intersections[0]
        self.assertEqual(tj.type, IntersectionType.T_JUNCTION, "Should be a T-junction")
        self.assertTrue(np.allclose(tj.position, [1, 0, 0], atol=1e-6), "T-junction should be at (1, 0, 0)")
        self.assertEqual(len(tj.panels), 3, "T-junction should involve exactly 3 panels")
        
        # panel1 at midspan (t=0.5), panel2 and panel3 at start (t=0.0)
        panel_dict = {p.panel.id: p for p in tj.panels}
        self.assertAlmostEqual(panel_dict["p1"].t, 0.5, places=6, msg="Panel p1 should be at t=0.5 (midspan)")
        self.assertAlmostEqual(panel_dict["p2"].t, 0.0, places=6, msg="Panel p2 should be at t=0.0 (start)")
        self.assertAlmostEqual(panel_dict["p3"].t, 0.0, places=6, msg="Panel p3 should be at t=0.0 (start)")
        
        # Verify panels are ordered CCW when viewed from +Y
        # The actual CCW ordering depends on the angle calculation at the intersection
        # Just verify we have 3 panels and they're all present
        panel_ids = {p.panel.id for p in tj.panels}
        self.assertEqual(panel_ids, {"p1", "p2", "p3"}, "T-junction should contain all 3 panels")
        
        # Verify unified partition structure for panel1 (split at t=0.5) via solution.regions_per_panel
        self.assertIsNotNone(solution.regions_per_panel.get(panel1), 
                           "Panel p1 at midspan should have partitions")
        partitions = solution.regions_per_panel[panel1]
        self.assertEqual(len(partitions.children), 2, "Panel p1 should be split into 2 regions")
        self.assertEqual(partitions.children[0].tStart, 0.0, "First region starts at 0.0")
        self.assertAlmostEqual(partitions.children[1].tStart, 0.5, places=6, msg="Second region starts at 0.5")
    
    # ========================================================================
    # COMPLEX SCENARIOS
    # ========================================================================
    
    def testSolveMixedIntersectionTypes(self):
        """Test that all three intersection types can be detected together"""
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [3, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0, 0, 0], [0, 0, 1], 0.1)
        panel3 = GXMLMockPanel("p3", [1, 0, -0.5], [1, 0, 0.5], 0.1)
        panel4 = GXMLMockPanel("p4", [2, 0, 0], [2, 0, 1], 0.1)
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3, panel4])
        
        self.assertEqual(len(solution.intersections), 3, "Should find exactly 3 intersections")
        
        # Verify intersection types and ordering
        joint = solution.intersections[0]
        crossing = solution.intersections[1]
        t_junction = solution.intersections[2]
        
        self.assertEqual(joint.type, IntersectionType.JOINT, "First intersection should be joint")
        self.assertEqual(crossing.type, IntersectionType.CROSSING, "Second intersection should be crossing")
        self.assertEqual(t_junction.type, IntersectionType.T_JUNCTION, "Third intersection should be T-junction")
        
        # Verify joint at (0,0,0)
        self.assertTrue(np.allclose(joint.position, [0, 0, 0], atol=1e-6), "Joint should be at (0, 0, 0)")
        self.assertEqual(len(joint.panels), 2, "Joint should involve 2 panels")
        self.assertEqual(joint.panels[0].panel, panel2, "First panel should be p2")
        self.assertEqual(joint.panels[1].panel, panel1, "Second panel should be p1")
        
        # Verify crossing at (1,0,0)
        self.assertTrue(np.allclose(crossing.position, [1, 0, 0], atol=1e-6), "Crossing should be at (1, 0, 0)")
        self.assertEqual(len(crossing.panels), 2, "Crossing should involve 2 panels")
        self.assertEqual(crossing.panels[0].panel, panel1, "First panel should be p1")
        self.assertEqual(crossing.panels[1].panel, panel3, "Second panel should be p3")
        
        # Verify T-junction at (2,0,0)
        self.assertTrue(np.allclose(t_junction.position, [2, 0, 0], atol=1e-6), "T-junction should be at (2, 0, 0)")
        self.assertEqual(len(t_junction.panels), 2, "T-junction should involve 2 panels")
        self.assertEqual(t_junction.panels[0].panel, panel1, "First panel should be p1")
        self.assertEqual(t_junction.panels[1].panel, panel4, "Second panel should be p4")
        
        # Verify unified partitions via solution.regions_per_panel
        # panel1 has crossing at t=1/3 and T-junction at t=2/3 - should have 3 regions
        self.assertIsNotNone(solution.regions_per_panel.get(panel1), "Panel p1 should have partitions")
        # panel3 has crossing at midspan - should have 2 regions
        self.assertIsNotNone(solution.regions_per_panel.get(panel3), "Panel p3 should have partitions")
    
    def testMultipleSplitsOnSamePanel(self):
        """Test that a panel with multiple crossings gets correct partition structures"""
        # panel1 runs horizontally and gets crossed twice
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [4, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [1, 0, -0.5], [1, 0, 0.5], 0.1)  # Crosses at x=1 (t=0.25)
        panel3 = GXMLMockPanel("p3", [3, 0, -0.5], [3, 0, 0.5], 0.1)  # Crosses at x=3 (t=0.75)
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3])
        
        # Should find 2 crossings
        self.assertEqual(len(solution.intersections), 2, "Should find exactly 2 crossings")
        
        crossing0 = solution.intersections[0]
        crossing1 = solution.intersections[1]
        
        # Both should be crossings
        self.assertEqual(crossing0.type, IntersectionType.CROSSING, "First should be crossing")
        self.assertEqual(crossing1.type, IntersectionType.CROSSING, "Second should be crossing")
        
        # Verify positions
        self.assertTrue(np.allclose(crossing0.position, [1, 0, 0], atol=1e-6), "First crossing at x=1")
        self.assertTrue(np.allclose(crossing1.position, [3, 0, 0], atol=1e-6), "Second crossing at x=3")
        
        # Verify panel1 appears in both intersections at different t-values
        self.assertAlmostEqual(crossing0.panels[0].t, 0.25, places=6, msg="Panel p1 at t=0.25 in first crossing")
        self.assertAlmostEqual(crossing1.panels[0].t, 0.75, places=6, msg="Panel p1 at t=0.75 in second crossing")
        
        # Verify unified partitions merge both splits into single BSP tree
        unified_partitions = solution.regions_per_panel[panel1]
        self.assertIsNotNone(unified_partitions, "Panel p1 should have unified partitions")
        self.assertEqual(unified_partitions.childSubdivisionAxis, PanelAxis.PRIMARY, "Unified partitions should be PRIMARY axis")
        self.assertEqual(len(unified_partitions.children), 3, "Unified should have 3 regions: [0.0-0.25], [0.25-0.75], [0.75-1.0]")
        self.assertAlmostEqual(unified_partitions.children[0].tStart, 0.0, places=6, msg="First region at t=0.0")
        self.assertAlmostEqual(unified_partitions.children[1].tStart, 0.25, places=6, msg="Second region at t=0.25")
        self.assertAlmostEqual(unified_partitions.children[2].tStart, 0.75, places=6, msg="Third region at t=0.75")
    
    def testTCrossingDifferentHeights(self):
        """Test T-junction where panels have different heights.
        
        Note: While _generate_regions creates secondary axis splits for height mismatches,
        the _merge_regions function currently only merges PRIMARY axis splits. This test
        verifies the current unified region behavior (primary axis only).
        """
        panel1 = GXMLMockPanel("p1", [0, 0, -0.5], [4, 0, -0.5], 0.1, height=1.0)
        panel2 = GXMLMockPanel("p2", [2, 0, -0.5], [2, 0, 0.5], 0.1, height=0.6)
        
        panels = [panel1, panel2]
        solution = IntersectionSolver.solve(panels)
        
        # Verify we found the T-junction
        self.assertEqual(len(solution.intersections), 1, "Should detect one intersection")
        intersection = solution.intersections[0]
        self.assertEqual(intersection.type, IntersectionType.T_JUNCTION, "Should be T_JUNCTION")
        
        # Verify panel1 gets partitions (primary axis split at t=0.5 where panel2 crosses)
        self.assertIsNotNone(solution.regions_per_panel.get(panel1), 
                           "Panel p1 should have partitions")
        
        partitions = solution.regions_per_panel[panel1]
        
        # Unified regions use PRIMARY axis (merge logic doesn't preserve SECONDARY splits)
        self.assertEqual(partitions.childSubdivisionAxis, PanelAxis.PRIMARY, "Unified regions use PRIMARY axis")
        self.assertEqual(len(partitions.children), 2, "Should have 2 primary regions")
        self.assertAlmostEqual(partitions.children[0].tStart, 0.0, places=6, msg="First region starts at t=0.0")
        self.assertAlmostEqual(partitions.children[1].tStart, 0.5, places=6, msg="Second region starts at t=0.5")
    
    def testEndpointTolerance(self):
        """Test intersection detection near endpoint tolerance boundaries"""
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [1, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0.001, 0, -0.5], [0.001, 0, 0.5], 0.1)
        panel3 = GXMLMockPanel("p3", [0.049, 0, -0.5], [0.049, 0, 0.5], 0.1)
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3])
        
        self.assertEqual(len(solution.intersections), 2, "Should find exactly 2 intersections")
        
        # Both should be classified as T-junctions (panel2 at near-endpoint, panel1 at midspan)
        # The classification depends on whether t-values are considered "at endpoint" (< ENDPOINT_TOLERANCE)
        intersection0 = solution.intersections[0]
        intersection1 = solution.intersections[1]
        
        # Verify both intersections exist and have correct positions
        self.assertTrue(np.allclose(intersection0.position, [0.001, 0, 0], atol=1e-6), 
                       "First intersection should be at (0.001, 0, 0)")
        self.assertTrue(np.allclose(intersection1.position, [0.049, 0, 0], atol=1e-6),
                       "Second intersection should be at (0.049, 0, 0)")
        
        # Verify t-values
        panel_dict0 = {p.panel.id: p for p in intersection0.panels}
        panel_dict1 = {p.panel.id: p for p in intersection1.panels}
        
        self.assertAlmostEqual(panel_dict0["p1"].t, 0.001, places=6, msg="Panel p1 should be at t=0.001")
        self.assertAlmostEqual(panel_dict1["p1"].t, 0.049, places=6, msg="Panel p1 should be at t=0.049")
        
        # Verify unified partitions via solution.regions_per_panel
        # panel2 at midspan should have partitions, panel3 at midspan should have partitions
        # panel1 may or may not have partitions depending on endpoint tolerance
        self.assertIsNotNone(solution.regions_per_panel.get(panel2), "Panel p2 at midspan should have partitions")
        self.assertIsNotNone(solution.regions_per_panel.get(panel3), "Panel p3 at midspan should have partitions")
    
    def testEndpointBoundaryCondition(self):
        """Test that t-value exactly at ENDPOINT_THRESHOLD boundary is treated correctly"""
        # Test _generate_regions directly with exact t-values to avoid floating point issues
        from elements.solvers.gxml_intersection_solver import ENDPOINT_THRESHOLD, Intersection, IntersectionType
        
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [1, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0, 0, 0], [0, 0, 1], 0.1)
        
        # Test with t exactly at threshold (0.01)
        panel_t_values = {
            panel1: 0.01,  # Exactly at threshold
            panel2: 0.5    # Midspan
        }
        
        # Create a mock intersection for the test
        mock_intersection = Intersection(
            type=IntersectionType.T_JUNCTION,
            position=np.array([0, 0, 0]),
            panels=[]
        )
        
        regions = IntersectionSolver._generate_regions(
            panel_t_values,
            [panel1, panel2],
            np.array([0, 0, 0]),
            mock_intersection
        )
        
        # With t < ENDPOINT_THRESHOLD: 0.01 < 0.01 is False, so should have regions
        # With mutation t <= ENDPOINT_THRESHOLD: 0.01 <= 0.01 is True, so would have None
        self.assertIsNotNone(regions[panel1], 
                           "Panel at t=0.01 (exactly at threshold) should have regions (not treated as endpoint)")
        
        # Test with t just below threshold
        panel_t_values_below = {
            panel1: 0.009,  # Below threshold
            panel2: 0.5
        }
        
        regions_below = IntersectionSolver._generate_regions(
            panel_t_values_below,
            [panel1, panel2],
            np.array([0, 0, 0]),
            mock_intersection
        )
        
        # 0.009 < 0.01 is True, so should NOT have regions
        self.assertIsNone(regions_below[panel1], 
                        "Panel at t=0.009 (below threshold) should not have regions (treated as endpoint)")
    
    # ========================================================================
    # GEOMETRIC PROPERTIES
    # ========================================================================
    
    def testSolutionIncludesNonIntersectingPanels(self):
        """Test that IntersectionSolution includes all panels, even those without intersections"""
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [1, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0, 0, 0], [0, 0, 1], 0.1)
        panel3 = GXMLMockPanel("p3", [10, 0, 10], [11, 0, 10], 0.1)  # Far away, no intersections
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3])
        
        # Verify all panels are in the solution
        self.assertEqual(len(solution.panels), 3, "Solution should include all 3 panels")
        self.assertIn(panel1, solution.panels, "Panel p1 should be in solution")
        self.assertIn(panel2, solution.panels, "Panel p2 should be in solution")
        self.assertIn(panel3, solution.panels, "Panel p3 should be in solution")
        
        # Verify only one intersection (p1 and p2 at origin)
        self.assertEqual(len(solution.intersections), 1, "Should find exactly 1 intersection")
        
        # Verify panel3 has no intersections
        panel3_intersections = solution.get_intersections_for_panel(panel3)
        self.assertEqual(len(panel3_intersections), 0, "Panel p3 should have no intersections")
        
        # Verify panel3 has no split regions
        panel3_region = solution.regions_per_panel.get(panel3)
        self.assertIsNone(panel3_region, "Panel p3 should have no regions")
        
        # Verify panel1 and panel2 do have the intersection
        panel1_intersections = solution.get_intersections_for_panel(panel1)
        panel2_intersections = solution.get_intersections_for_panel(panel2)
        self.assertEqual(len(panel1_intersections), 1, "Panel p1 should have 1 intersection")
        self.assertEqual(len(panel2_intersections), 1, "Panel p2 should have 1 intersection")
        
        # Intersection is a joint (both at endpoints), so panels have no split regions
    
    # ========================================================================
    # EDGE CASES
    # ========================================================================
    
    def testEdgeCaseEmptyPanelList(self):
        """Test that empty panel list returns empty solution"""
        solution = IntersectionSolver.solve([])
        
        self.assertEqual(len(solution.panels), 0, "Solution should have no panels")
        self.assertEqual(len(solution.intersections), 0, "Solution should have no intersections")
    
    def testEdgeCaseSinglePanel(self):
        """Test that single panel returns solution with no intersections"""
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [1, 0, 0], 0.1)
        
        solution = IntersectionSolver.solve([panel1])
        
        self.assertEqual(len(solution.panels), 1, "Solution should have 1 panel")
        self.assertEqual(solution.panels[0], panel1, "Solution should contain panel1")
        self.assertEqual(len(solution.intersections), 0, "Solution should have no intersections")
        
        # Verify no regions for single panel
        panel1_region = solution.regions_per_panel.get(panel1)
        self.assertIsNone(panel1_region, "Single panel should have no regions")
    
    def testNoIntersection(self):
        """Test that non-intersecting panels produce no intersections"""
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [1, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0, 0, 2], [1, 0, 2], 0.1)
        panel3 = GXMLMockPanel("p3", [2, 0, 0], [2, 0, 1], 0.1)
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3])
        
        self.assertEqual(len(solution.intersections), 0, "Should find no intersections")
    
    def testCollinearPanelsNoIntersection(self):
        """Test that collinear panels (same line, non-overlapping) produce no intersections"""
        # Two panels on same line but separated
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [1, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [2, 0, 0], [3, 0, 0], 0.1)  # Same line, gap from x=1 to x=2
        
        solution = IntersectionSolver.solve([panel1, panel2])
        
        self.assertEqual(len(solution.intersections), 0, 
                        "Collinear non-overlapping panels should have no intersections")
        self.assertEqual(len(solution.panels), 2, "Solution should include both panels")
    
    def testParallelOffsetPanelsNoIntersection(self):
        """Test that parallel but offset panels produce no intersections"""
        # Two horizontal panels at different Z coordinates
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0, 0, 1], [2, 0, 1], 0.1)  # Parallel, offset by 1 unit in Z
        
        solution = IntersectionSolver.solve([panel1, panel2])
        
        self.assertEqual(len(solution.intersections), 0, 
                        "Parallel offset panels should have no intersections")
        self.assertEqual(len(solution.panels), 2, "Solution should include both panels")
    
    def testOverlappingCollinearPanels(self):
        """Test that overlapping collinear panels are not detected as intersections"""
        # Two panels on same line, overlapping
        # Current implementation only detects perpendicular intersections, not collinear overlaps
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [2, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [1, 0, 0], [3, 0, 0], 0.1)  # Overlaps from x=1 to x=2
        
        solution = IntersectionSolver.solve([panel1, panel2])
        
        # Collinear panels are not currently detected as intersections
        # This is expected behavior - the system is designed for perpendicular intersections
        self.assertEqual(len(solution.intersections), 0, 
                        "Collinear overlapping panels should not be detected as intersections")
        self.assertEqual(len(solution.panels), 2, "Solution should include both panels")
    
    def testGetRegionTreeForPanel_Crossing(self):
        """Test get_region_tree_for_panel returns correct regions for a crossing intersection."""
        # Create crossing: both panels are split at midspan
        panel1 = GXMLMockPanel("p1", [-2, 0, 0], [2, 0, 0], 0.1)  # Split at t=0.5 (x=0)
        panel2 = GXMLMockPanel("p2", [0, 0, -1], [0, 0, 1], 0.1)  # Split at t=0.5 (z=0)
        
        solution = IntersectionSolver.solve([panel1, panel2])
        
        # Both panels should have regions (crossing at midspan)
        panel1_region_tree = solution.get_region_tree_for_panel(panel1)
        panel2_region_tree = solution.get_region_tree_for_panel(panel2)
        self.assertIsNotNone(panel1_region_tree, "Panel1 should have regions")
        self.assertIsNotNone(panel2_region_tree, "Panel2 should have regions")
        
        # Verify panel1 has correct regions (split at t=0.5)
        panel1_leaves = panel1_region_tree.get_leaves()
        self.assertEqual(len(panel1_leaves), 2, "Panel1 should have 2 leaf regions")
        self.assertAlmostEqual(panel1_leaves[0].tStart, 0.0, places=6)
        self.assertAlmostEqual(panel1_leaves[1].tStart, 0.5, places=6)
        
        # Verify panel2 has correct regions (also split at t=0.5)
        panel2_leaves = panel2_region_tree.get_leaves()
        self.assertEqual(len(panel2_leaves), 2, "Panel2 should have 2 leaf regions")
        self.assertAlmostEqual(panel2_leaves[0].tStart, 0.0, places=6)
        self.assertAlmostEqual(panel2_leaves[1].tStart, 0.5, places=6)
        
        # Verify we're getting distinct region objects (not the same regions)
        self.assertIsNot(panel1_region_tree, panel2_region_tree, "Should return different region trees")
        self.assertIsNot(panel1_leaves[0], panel2_leaves[0], "Should return different leaf regions")

    def testGetRegionTreeForPanel_TJunction(self):
        """Test get_region_tree_for_panel returns correct regions for a T-junction."""
        panel1 = GXMLMockPanel("p1", [-1, 0, 0], [1, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0, 0, 0], [0, 0, 2], 0.1)  # Starts at panel1's midspan
        
        solution = IntersectionSolver.solve([panel1, panel2])
        
        panel1_region_tree = solution.get_region_tree_for_panel(panel1)
        panel2_region_tree = solution.get_region_tree_for_panel(panel2)
        
        # Panel1 should have 2 regions (split at midspan)
        panel1_leaves = panel1_region_tree.get_leaves()
        self.assertEqual(len(panel1_leaves), 2, "Panel1 should have 2 leaf regions")
        self.assertAlmostEqual(panel1_leaves[1].tStart, 0.5, places=6, msg="Split at midspan")
        
        # Panel2 should have 1 region (endpoint, no split - just default full region)
        panel2_leaves = panel2_region_tree.get_leaves()
        self.assertEqual(len(panel2_leaves), 1, "Panel2 should have 1 leaf region (no split)")
        self.assertAlmostEqual(panel2_leaves[0].tStart, 0.0, places=6)

    def testGetRegionTreeForPanel_InvalidPanel(self):
        """Test get_region_tree_for_panel raises ValueError for a panel not in the solution."""
        panel1 = GXMLMockPanel("p1", [-2, 0, 0], [2, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0, 0, -1], [0, 0, 1], 0.1)
        
        solution = IntersectionSolver.solve([panel1, panel2])
        
        # Panel not in original solve should raise ValueError
        panel3 = GXMLMockPanel("p3", [10, 10, 10], [20, 20, 20], 0.1)
        with self.assertRaises(ValueError):
            solution.get_region_tree_for_panel(panel3)

    def testRegionTreePreservesIntersectionReference(self):
        """Test that unified region trees preserve references to the causing intersection.
        
        When regions are merged from multiple intersections, each region's `intersection`
        attribute should reference the Intersection object that caused that split.
        This allows downstream code to determine which panel(s) caused a face split.
        """
        # Create a T-junction: panel1 is split at midspan by panel2
        panel1 = GXMLMockPanel("p1", [-1, 0, 0], [1, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [0, 0, 0], [0, 0, 1], 0.1)  # Starts at panel1's midspan
        
        solution = IntersectionSolver.solve([panel1, panel2])
        
        # Should have exactly 1 T-junction
        self.assertEqual(len(solution.intersections), 1)
        intersection = solution.intersections[0]
        self.assertEqual(intersection.type, IntersectionType.T_JUNCTION)
        
        # Get the unified region tree for panel1 (the one that gets split)
        region_tree = solution.get_region_tree_for_panel(panel1)
        self.assertIsNotNone(region_tree)
        
        # Should have 2 leaf regions
        leaves = region_tree.get_leaves()
        self.assertEqual(len(leaves), 2, "Panel1 should have 2 leaf regions")
        
        # First region (t=0) doesn't have an intersection (it's the original start)
        # Second region (t=0.5) should have the intersection that caused it
        self.assertIsNone(leaves[0].intersection, "First region at t=0 should not have intersection")
        self.assertIsNotNone(leaves[1].intersection, "Second region should have intersection reference")
        self.assertIs(leaves[1].intersection, intersection, "Region should reference the causing intersection")
        
        # Verify we can get the causing panel from the intersection
        causing_panels = [p.panel for p in leaves[1].intersection.panels if p.panel != panel1]
        self.assertEqual(len(causing_panels), 1)
        self.assertIs(causing_panels[0], panel2, "Should be able to identify panel2 as the cause")

    def testRegionTreePreservesIntersectionReferenceMultipleSplits(self):
        """Test intersection references are preserved when a panel has multiple splits.
        
        When a panel is split by multiple other panels, each split region should
        reference its specific causing intersection.
        """
        # Panel1 is a long panel, crossed by panel2 and panel3 at different points
        panel1 = GXMLMockPanel("p1", [0, 0, 0], [3, 0, 0], 0.1)
        panel2 = GXMLMockPanel("p2", [1, 0, -0.5], [1, 0, 0.5], 0.1)  # Crosses at t=1/3
        panel3 = GXMLMockPanel("p3", [2, 0, -0.5], [2, 0, 0.5], 0.1)  # Crosses at t=2/3
        
        solution = IntersectionSolver.solve([panel1, panel2, panel3])
        
        # Should have 2 crossings
        self.assertEqual(len(solution.intersections), 2)
        
        # Find the intersections by position
        intersection_at_1 = next(i for i in solution.intersections if np.allclose(i.position, [1, 0, 0], atol=1e-6))
        intersection_at_2 = next(i for i in solution.intersections if np.allclose(i.position, [2, 0, 0], atol=1e-6))
        
        # Get the unified region tree for panel1
        region_tree = solution.get_region_tree_for_panel(panel1)
        self.assertIsNotNone(region_tree)
        
        # Should have 3 leaf regions (split at t≈0.333 and t≈0.667)
        leaves = region_tree.get_leaves()
        self.assertEqual(len(leaves), 3, "Panel1 should have 3 leaf regions")
        
        # First region (t=0) - no causing intersection
        self.assertIsNone(leaves[0].intersection, "First region at t=0 should not have intersection")
        
        # Second region (t≈0.333) - caused by intersection_at_1
        self.assertIsNotNone(leaves[1].intersection, "Second region should have intersection")
        self.assertIs(leaves[1].intersection, intersection_at_1, "Second region should reference intersection at x=1")
        
        # Third region (t≈0.667) - caused by intersection_at_2  
        self.assertIsNotNone(leaves[2].intersection, "Third region should have intersection")
        self.assertIs(leaves[2].intersection, intersection_at_2, "Third region should reference intersection at x=2")
        
        # Verify we can trace back to the causing panels
        panel2_from_region = [p.panel for p in leaves[1].intersection.panels if p.panel != panel1]
        panel3_from_region = [p.panel for p in leaves[2].intersection.panels if p.panel != panel1]
        self.assertIn(panel2, panel2_from_region, "Should identify panel2 from region 1")
        self.assertIn(panel3, panel3_from_region, "Should identify panel3 from region 2")


if __name__ == '__main__':
    unittest.main()
