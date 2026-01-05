"""Test 2D s-split support in FaceSolver."""

import unittest
import numpy as np

from elements.gxml_panel import GXMLPanel, PanelSide
from gxml.elements.solvers.gxml_intersection_solver import (
    Region, Intersection, IntersectionType, PanelAxis
)
from gxml.elements.solvers.gxml_face_solver import FaceSolver, SegmentedPanel
from tests.helpers.mocks import GXMLMockPanel


class TestFaceSolverSSplits(unittest.TestCase):
    """Test FaceSolver's handling of 2D (t, s) splits."""

    def test_get_leaf_bounds_with_s_splits(self):
        """Verify get_leaf_bounds correctly extracts 2D bounds from s-split regions."""
        # Create mock panels
        panel_long = GXMLMockPanel("long", [0, 0, 0], [4, 0, 0], thickness=0.1, height=1.0)
        panel_short = GXMLMockPanel("short", [2, 0, -0.5], [2, 0, 0.5], thickness=0.1, height=0.6)

        # Create a fake intersection
        intersection = Intersection(
            type=IntersectionType.CROSSING,
            position=np.array([2.0, 0.0, 0.0]),
            panels=[
                Intersection.PanelEntry(panel=panel_long, t=0.5),
                Intersection.PanelEntry(panel=panel_short, t=0.5),
            ]
        )

        # Manually create a Region with s-splits
        # Structure: SECONDARY first, then PRIMARY within regions
        region_with_s_splits = Region(
            tStart=0.0,
            childSubdivisionAxis=PanelAxis.SECONDARY,
            children=[
                # Region s=0 to s=0.6 (where short panel exists) - needs PRIMARY split
                Region(
                    tStart=0.0,  # s=0
                    childSubdivisionAxis=PanelAxis.PRIMARY,
                    children=[
                        Region(tStart=0.0),  # t=0 to 0.5
                        Region(tStart=0.5, intersection=intersection),  # t=0.5 to 1.0
                    ]
                ),
                # Region s=0.6 to s=1.0 (above short panel) - no split needed
                Region(
                    tStart=0.6,  # s=0.6
                    intersection=intersection  # Mark where the s-split came from
                ),
            ]
        )

        leaf_bounds = region_with_s_splits.get_leaf_bounds()
        
        # Should have 3 leaves
        self.assertEqual(len(leaf_bounds), 3)
        
        # Leaf 0: t=[0.0, 0.5], s=[0.0, 0.6] - no intersection
        self.assertAlmostEqual(leaf_bounds[0].t_start, 0.0)
        self.assertAlmostEqual(leaf_bounds[0].t_end, 0.5)
        self.assertAlmostEqual(leaf_bounds[0].s_start, 0.0)
        self.assertAlmostEqual(leaf_bounds[0].s_end, 0.6)
        self.assertIsNone(leaf_bounds[0].intersection)
        
        # Leaf 1: t=[0.5, 1.0], s=[0.0, 0.6] - with intersection
        self.assertAlmostEqual(leaf_bounds[1].t_start, 0.5)
        self.assertAlmostEqual(leaf_bounds[1].t_end, 1.0)
        self.assertAlmostEqual(leaf_bounds[1].s_start, 0.0)
        self.assertAlmostEqual(leaf_bounds[1].s_end, 0.6)
        self.assertIsNotNone(leaf_bounds[1].intersection)
        
        # Leaf 2: t=[0.0, 1.0], s=[0.6, 1.0] - with intersection (s-boundary)
        self.assertAlmostEqual(leaf_bounds[2].t_start, 0.0)
        self.assertAlmostEqual(leaf_bounds[2].t_end, 1.0)
        self.assertAlmostEqual(leaf_bounds[2].s_start, 0.6)
        self.assertAlmostEqual(leaf_bounds[2].s_end, 1.0)
        self.assertIsNotNone(leaf_bounds[2].intersection)

    def test_build_segments_from_leaves_with_s_splits(self):
        """Verify _build_segments_from_leaves creates 2D grid of segments."""
        # Create mock panels
        panel_long = GXMLMockPanel("long", [0, 0, 0], [4, 0, 0], thickness=0.1, height=1.0)
        panel_short = GXMLMockPanel("short", [2, 0, -0.5], [2, 0, 0.5], thickness=0.1, height=0.6)

        # Create a fake intersection
        intersection = Intersection(
            type=IntersectionType.CROSSING,
            position=np.array([2.0, 0.0, 0.0]),
            panels=[
                Intersection.PanelEntry(panel=panel_long, t=0.5),
                Intersection.PanelEntry(panel=panel_short, t=0.5),
            ]
        )

        # Manually create leaf bounds that simulate s-splits
        # Note: In real usage, these come from Region.get_leaf_bounds()
        leaf_bounds = [
            Region.LeafBounds(t_start=0.0, t_end=0.5, s_start=0.0, s_end=0.6, intersection=None),
            Region.LeafBounds(t_start=0.5, t_end=1.0, s_start=0.0, s_end=0.6, intersection=intersection),
            Region.LeafBounds(t_start=0.0, t_end=1.0, s_start=0.6, s_end=1.0, intersection=intersection),
        ]

        panel_faces = SegmentedPanel(panel=panel_long)
        FaceSolver._build_segments_from_leaves(panel_faces, PanelSide.FRONT, leaf_bounds)
        
        segments = panel_faces.segments.get(PanelSide.FRONT, [])
        
        # With proper 2D grid support, we should get a grid based on unique boundaries:
        # t-boundaries: 0.0, 0.5, 1.0 -> 2 t-ranges
        # s-boundaries: 0.0, 0.6, 1.0 -> 2 s-ranges
        # Total: 2 * 2 = 4 segments in a grid
        self.assertEqual(len(segments), 4, 
            f"Expected 4 segments in 2D grid, got {len(segments)}")
        
        # Verify the s-ranges are correctly used (not all s=0 to s=1)
        s_ranges = set()
        for seg in segments:
            s_start = seg.corners[0][1]  # bottom-left s
            s_end = seg.corners[2][1]    # top-right s
            s_ranges.add((round(s_start, 2), round(s_end, 2)))
        
        # Should have at least 2 different s-ranges
        self.assertGreater(len(s_ranges), 1, 
            f"Expected multiple s-ranges, got only: {s_ranges}")
        
        # Verify specific s-ranges exist
        self.assertIn((0.0, 0.6), s_ranges, "Should have s=[0, 0.6] segments")
        self.assertIn((0.6, 1.0), s_ranges, "Should have s=[0.6, 1.0] segments")

    def test_build_segments_no_splits(self):
        """Verify empty leaf_bounds produces single full-face segment."""
        panel = GXMLMockPanel("p1", [0, 0, 0], [4, 0, 0], thickness=0.1, height=1.0)
        panel_faces = SegmentedPanel(panel=panel)
        
        FaceSolver._build_segments_from_leaves(panel_faces, PanelSide.FRONT, [])
        
        segments = panel_faces.segments.get(PanelSide.FRONT, [])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].corners, [
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
        ])


if __name__ == '__main__':
    unittest.main()
