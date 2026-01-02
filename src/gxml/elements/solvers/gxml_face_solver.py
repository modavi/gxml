"""
Face and bounds solving logic - determines face segmentation and computes trim/gap adjustments.

This module is Stage 2 of the geometry pipeline:
1. IntersectionSolver - finds where panel centerlines meet
2. FaceSolver - determines face segmentation with gap-adjusted bounds
3. GeometryBuilder - creates actual 3D geometry (polygons and caps)

The FaceSolver.solve() method returns List[SegmentedPanel], one per panel in the input.
Each SegmentedPanel contains a reference to its panel and the face segments with gap-adjusted bounds.

Key data structures:
- FaceSegment: Represents a face segment with gap-adjusted bounds (corners in (t, s) coords)
- SegmentedPanel: Per-panel container with segments for each face side, plus helper methods
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Tuple, Optional, List, Dict

import numpy as np

import mathutils.gxml_math as GXMLMath
from .gxml_intersection_solver import IntersectionSolution, Intersection, IntersectionType, Region
from elements.gxml_panel import GXMLPanel, PanelSide
from mathutils.gxml_math import intersect_line_with_plane


# ============================================================================
# CONSTANTS
# ============================================================================

# Tolerance for t-value comparisons (same as used in intersection solver)
ENDPOINT_TOLERANCE = 0.01

# Tolerance for geometry calculations (thickness checks)
TOLERANCE = 1e-4


@dataclass
class FaceSegment:
    """A single segment of a face between two boundaries.
    
    Stores 4 corners in (t, s) coordinates for geometry generation.
    
    Coordinate system:
    - t: Along primary axis
    - s: Across secondary axis
    """
    
    parent: 'SegmentedPanel'
    face_side: PanelSide
    corners: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ])
    _cached_world_corners: List = field(default_factory=list, repr=False)
    
    def get_world_corners(self) -> List[np.ndarray]:
        """
        Get world-space corners for this face segment (cached).
        
        Returns:
            List of 4 world-space corner positions
        """
        if not self._cached_world_corners:
            panel = self.parent.panel
            self._cached_world_corners = [panel.get_face_point(self.face_side, t, s) for t, s in self.corners]
        return self._cached_world_corners
    
    def invalidate_cache(self):
        """Clear cached world corners (call if corners change)."""
        self._cached_world_corners = []

@dataclass 
class SegmentedPanel:
    """Face segments for a single panel.
    
    For each face side, stores the list of segments (in t-order).
    Unsplit faces have a single segment spanning [0, 1].
    """
    panel: GXMLPanel
    segments: Dict[PanelSide, List[FaceSegment]] = field(default_factory=dict)
    
    def is_split(self, face: PanelSide) -> bool:
        """Whether this face has multiple segments."""
        return face in self.segments and len(self.segments[face]) > 1


class JointSide(Enum):
    """Side of a panel at a joint for adjacency lookup."""
    CCW = auto()  # Counter-clockwise side (toward next panel)
    CW = auto()   # Clockwise side (toward previous panel)


class FaceSolver:
    """
    Stage 2: Face segmentation and bounds solving.
    
    Takes the intersection solution and:
    1. Determines face segmentation (which faces are split, boundary types)
    2. Computes trim/gap adjustments (ray-plane intersections)
    
    The solve() method returns a List[SegmentedPanel].
    """
    
    @staticmethod
    def solve(intersection_solution: IntersectionSolution) -> List[SegmentedPanel]:
        """
        Compute face segmentation and bounds for all panels.
        
        Args:
            intersection_solution: Output from IntersectionSolver
            
        Returns:
            List of SegmentedPanel, one per panel in the input
        """
        
        panel_faces = []
        for panel in intersection_solution.panels:
            pf, start_inter, end_inter = FaceSolver._solve_panel(panel, intersection_solution)
            
            if pf.panel.is_valid(TOLERANCE):
                for face_side in PanelSide:
                    segments = pf.segments.get(face_side, [])
                    if segments:
                        FaceSolver._apply_endpoint_trims(pf.panel, segments, face_side, pf, start_inter, end_inter)
            
            panel_faces.append(pf)
        
        return panel_faces
    
    @staticmethod
    def _solve_panel(panel: GXMLPanel, solution: IntersectionSolution) -> Tuple[SegmentedPanel, Optional[Intersection], Optional[Intersection]]:
        """
        Compute face segmentation for a single panel.
        
        Uses the pre-computed Region tree from IntersectionSolver for split points.
        Creates FaceSegments from leaf regions:
        - FRONT/BACK: Each leaf region becomes a FaceSegment with (t, s) bounds
        - TOP/BOTTOM: Grouped by t-boundaries (s-axis is thickness, not height)
        
        Args:
            panel: The panel to analyze
            solution: The intersection solution
            
        Returns:
            Tuple of (SegmentedPanel, start_intersection, end_intersection)
        """
        panel_faces = SegmentedPanel(panel=panel)
        start_intersection: Optional[Intersection] = None
        end_intersection: Optional[Intersection] = None
        
        # Track endpoint intersections (needed for START/END face suppression and trims)
        for intersection in solution.get_intersections_for_panel(panel):
            entry = intersection.get_entry(panel)
            if entry.t < ENDPOINT_TOLERANCE:
                start_intersection = intersection
            if entry.t > (1.0 - ENDPOINT_TOLERANCE):
                end_intersection = intersection
        
        # Get pre-computed leaf regions from Region tree
        region = solution.regions_per_panel.get(panel)
        leaf_bounds = region.get_leaf_bounds() if region else []
        
        # Build FRONT/BACK segments as a 2D grid from leaf bounds.
        # These faces use height (s) as the secondary axis, and gaps are uniform
        # across all heights since the blocking panel's thickness is constant.
        for face_side in [PanelSide.FRONT, PanelSide.BACK]:
            FaceSolver._build_segments_from_leaves(panel_faces, face_side, leaf_bounds)
        
        # Build TOP/BOTTOM segments from t-boundaries only.
        # These faces use depth/thickness (s) as the secondary axis and need
        # different t-values at BACK (s=0) vs FRONT (s=1) edges for mitered corners.
        # This requires per-corner gap calculation, which _build_segments_for_face handles.
        for face_side in [PanelSide.TOP, PanelSide.BOTTOM]:
            t_intersections = FaceSolver._extract_t_intersections_for_face(panel, face_side, leaf_bounds)
            FaceSolver._build_segments_for_face(panel_faces, face_side, t_intersections)
        
        # Handle START and END faces
        if start_intersection is not None or panel.thickness <= TOLERANCE:
            panel_faces.segments[PanelSide.START] = []
        else:
            panel_faces.segments[PanelSide.START] = [
                FaceSegment(panel_faces, PanelSide.START, [(0.0, 0.0), (0.0, 0.0), (0.0, 1.0), (0.0, 1.0)])
            ]
        
        if end_intersection is not None or panel.thickness <= TOLERANCE:
            panel_faces.segments[PanelSide.END] = []
        else:
            panel_faces.segments[PanelSide.END] = [
                FaceSegment(panel_faces, PanelSide.END, [(1.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 1.0)])
            ]
        
        # Apply zero-thickness suppression if needed
        if panel.thickness <= TOLERANCE:
            # Suppress edge faces (they collapse to lines)
            for face in PanelSide.edge_faces():
                panel_faces.segments[face] = []
            
            # For thickness faces, determine visibility
            has_joint_connections = (start_intersection is not None or 
                                     end_intersection is not None)
            
            for face in PanelSide.thickness_faces():
                is_split = panel_faces.is_split(face)
                
                if is_split:
                    # Split face - show it (the split makes it unique)
                    pass
                elif has_joint_connections:
                    # Unsplit face at a joint: pick one face based on winding
                    visible_face = panel.get_visible_thickness_face()
                    if face != visible_face:
                        panel_faces.segments[face] = []
                else:
                    # No split and no joint - suppress this face
                    panel_faces.segments[face] = []
        
        return (panel_faces, start_intersection, end_intersection)
    
    @staticmethod
    def _extract_t_intersections_for_face(panel: GXMLPanel, face: PanelSide,
                                           leaf_bounds: List[Region.LeafBounds]) -> List[tuple]:
        """
        Extract t-boundary intersections that affect a specific face.
        
        Returns list of (t, intersection, [other_panels]) for midspan splits
        where the intersection actually affects the given face.
        Used for TOP/BOTTOM faces which only care about t-axis divisions.
        """
        # Collect unique t-splits with their intersections, filtered by face
        t_splits: Dict[float, Tuple[Intersection, List[GXMLPanel]]] = {}
        
        for leaf in leaf_bounds:
            if leaf.intersection is None:
                continue
            if leaf.t_start <= ENDPOINT_TOLERANCE:
                continue
            
            t = leaf.t_start
            if t in t_splits:
                continue  # Already processed this t-value
            
            intersection = leaf.intersection
            
            # Filter: only include if this intersection affects the given face
            for other_panel in intersection.get_other_panels(panel):
                affected_faces = intersection.get_affected_faces(panel, other_panel)
                if face in affected_faces:
                    if t not in t_splits:
                        t_splits[t] = (intersection, [])
                    t_splits[t][1].append(other_panel)
        
        # Convert to list format expected by _build_segments_for_face
        return [(t, inter, list(panels)) for t, (inter, panels) in sorted(t_splits.items())]
    
    @staticmethod
    def _build_segments_from_leaves(panel_faces: SegmentedPanel, face: PanelSide,
                                     leaf_bounds: List[Region.LeafBounds]) -> None:
        """
        Build FaceSegments for FRONT/BACK faces from leaf regions (2D grid).
        
        Creates one FaceSegment per leaf region, using both t and s bounds.
        Gap adjustments are applied at boundaries where intersections affect this face.
        
        Args:
            panel_faces: The SegmentedPanel to add segments to
            face: Which face (FRONT or BACK)
            leaf_bounds: List of leaf region bounds from Region tree
        """
        panel = panel_faces.panel
        
        if not leaf_bounds:
            # No splits - create single full-face segment
            panel_faces.segments[face] = [
                FaceSegment(panel_faces, face, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
            ]
            return
        
        # Collect all boundaries with their intersections (if they affect this face)
        # t_boundary -> (intersection, other_panel) or None
        t_boundaries: Dict[float, Optional[Tuple[Intersection, GXMLPanel]]] = {0.0: None, 1.0: None}
        s_boundaries: Dict[float, Optional[Tuple[Intersection, GXMLPanel]]] = {0.0: None, 1.0: None}
        
        for leaf in leaf_bounds:
            if leaf.intersection is None:
                continue
            
            intersection = leaf.intersection
            
            # Check if this intersection affects this face
            for other_panel in intersection.get_other_panels(panel):
                affected_faces = intersection.get_affected_faces(panel, other_panel)
                if face in affected_faces:
                    # Record boundaries with this intersection
                    if leaf.t_start > ENDPOINT_TOLERANCE:
                        t_boundaries[leaf.t_start] = (intersection, other_panel)
                    if leaf.s_start > ENDPOINT_TOLERANCE:
                        s_boundaries[leaf.s_start] = (intersection, other_panel)
                    break
        
        # Sort boundary values
        sorted_t = sorted(t_boundaries.keys())
        sorted_s = sorted(s_boundaries.keys())
        
        # If only default boundaries (0 and 1), no splits affecting this face
        if len(sorted_t) == 2 and len(sorted_s) == 2:
            panel_faces.segments[face] = [
                FaceSegment(panel_faces, face, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
            ]
            return
        
        # Build 2D grid of segments
        segments: List[FaceSegment] = []
        
        for i in range(len(sorted_t) - 1):
            for j in range(len(sorted_s) - 1):
                t_start_raw = sorted_t[i]
                t_end_raw = sorted_t[i + 1]
                s_start_raw = sorted_s[j]
                s_end_raw = sorted_s[j + 1]
                
                # Apply gap adjustments at t boundaries
                t_start_adj = t_start_raw
                t_end_adj = t_end_raw
                
                # Gap at t_start (if there's an intersection there)
                t_start_info = t_boundaries.get(t_start_raw)
                if t_start_info is not None:
                    intersection, other_panel = t_start_info
                    if intersection.type in (IntersectionType.CROSSING, IntersectionType.T_JUNCTION):
                        _, gap_end = FaceSolver._calculate_gap_t_values(
                            panel, face, other_panel, t_start_raw
                        )
                        t_start_adj = gap_end  # Start after the gap
                
                # Gap at t_end (if there's an intersection there)
                t_end_info = t_boundaries.get(t_end_raw)
                if t_end_info is not None:
                    intersection, other_panel = t_end_info
                    if intersection.type in (IntersectionType.CROSSING, IntersectionType.T_JUNCTION):
                        gap_start, _ = FaceSolver._calculate_gap_t_values(
                            panel, face, other_panel, t_end_raw
                        )
                        t_end_adj = gap_start  # End before the gap
                
                # Note: s-gaps would be similar but need _calculate_gap_s_values
                # For now, use raw s values (s-splits are typically for height mismatches,
                # which create separate regions but don't need gaps)
                s_start_adj = s_start_raw
                s_end_adj = s_end_raw
                
                # Create FaceSegment with adjusted corners
                # Corner ordering: [start-bottom, end-bottom, end-top, start-top]
                segments.append(FaceSegment(panel_faces, face, [
                    (t_start_adj, s_start_adj),
                    (t_end_adj, s_start_adj),
                    (t_end_adj, s_end_adj),
                    (t_start_adj, s_end_adj),
                ]))
        
        panel_faces.segments[face] = segments
    
    @staticmethod
    def _build_segments_for_face(panel_faces: SegmentedPanel, face: PanelSide,
                                  intersections: List[tuple]) -> None:
        """
        Build segments for a face from its intersection list.
        
        Computes gap-adjusted bounds directly. For CROSSING and T_JUNCTION
        intersections, gaps are calculated based on the intersecting panel's
        thickness. For JOINT intersections, no gap is needed (trims are
        applied separately).
        
        Args:
            panel_faces: The SegmentedPanel to add segments to
            face: Which face we're building segments for
            intersections: List of (t, intersection, [panels]) tuples, sorted by t
        """
        panel = panel_faces.panel
        is_edge_face = face in PanelSide.edge_faces()
        
        # Helper to compute gap-adjusted t value at a boundary
        def compute_gap_adjusted_t(nominal_t: float, intersection: Optional[Intersection], 
                                   intersecting_panel: Optional[GXMLPanel], 
                                   is_segment_end: bool) -> Tuple[float, Optional[float]]:
            """
            Compute gap-adjusted t value(s) for a boundary.
            
            Returns:
                Tuple of (t_back, t_front) where t_front is None for FRONT/BACK faces
            """
            if intersection is None or intersecting_panel is None:
                return (nominal_t, None)
            
            # CROSSING and T_JUNCTION create gaps; JOINT does not (both panels terminate)
            if intersection.type not in (IntersectionType.CROSSING, IntersectionType.T_JUNCTION):
                return (nominal_t, None)
            
            if is_edge_face:
                # For TOP/BOTTOM: compute gap at BACK and FRONT edges
                gap_start_back, gap_end_back = FaceSolver._calculate_gap_t_values(
                    panel, PanelSide.BACK, intersecting_panel, nominal_t
                )
                gap_start_front, gap_end_front = FaceSolver._calculate_gap_t_values(
                    panel, PanelSide.FRONT, intersecting_panel, nominal_t
                )
                if is_segment_end:
                    return (gap_start_back, gap_start_front)
                else:
                    return (gap_end_back, gap_end_front)
            else:
                # For FRONT/BACK: single gap value
                gap_start, gap_end = FaceSolver._calculate_gap_t_values(
                    panel, face, intersecting_panel, nominal_t
                )
                if is_segment_end:
                    return (gap_start, None)
                else:
                    return (gap_end, None)
        
        segments: List[FaceSegment] = []
        
        # Filter to only midspan intersections (not at endpoints)
        midspan = [
            (t, inter, panels) for t, inter, panels in intersections
            if ENDPOINT_TOLERANCE <= t <= (1.0 - ENDPOINT_TOLERANCE)
        ]
        
        # Start boundary tracking
        current_start_t = 0.0
        current_start_t_front: Optional[float] = None
        
        # Check if there's an intersection at t≈0
        start_intersections = [
            (t, inter, panels) for t, inter, panels in intersections
            if t < ENDPOINT_TOLERANCE
        ]
        if start_intersections:
            t, inter, panels = start_intersections[0]
            intersecting_panel = panels[0] if panels else None
            current_start_t, current_start_t_front = compute_gap_adjusted_t(
                t, inter, intersecting_panel, is_segment_end=False
            )
        
        # Build segments from midspan intersections
        for t, inter, panels in midspan:
            intersecting_panel = panels[0] if panels else None
            
            # Compute gap-adjusted end t for this segment
            end_t, end_t_front = compute_gap_adjusted_t(
                t, inter, intersecting_panel, is_segment_end=True
            )
            
            segments.append(FaceSegment(panel_faces, face, [
                (current_start_t, 0.0),
                (end_t, 0.0),
                (end_t_front if end_t_front is not None else end_t, 1.0),
                (current_start_t_front if current_start_t_front is not None else current_start_t, 1.0),
            ]))
            
            # Next segment starts at this intersection (with gap on other side)
            current_start_t, current_start_t_front = compute_gap_adjusted_t(
                t, inter, intersecting_panel, is_segment_end=False
            )
        
        # Final segment to t=1
        end_t = 1.0
        end_t_front: Optional[float] = None
        
        # Check if there's an intersection at t≈1
        end_intersections = [
            (t, inter, panels) for t, inter, panels in intersections
            if t > (1.0 - ENDPOINT_TOLERANCE)
        ]
        if end_intersections:
            t, inter, panels = end_intersections[0]
            intersecting_panel = panels[0] if panels else None
            end_t, end_t_front = compute_gap_adjusted_t(
                t, inter, intersecting_panel, is_segment_end=True
            )
        
        segments.append(FaceSegment(panel_faces, face, [
            (current_start_t, 0.0),
            (end_t, 0.0),
            (end_t_front if end_t_front is not None else end_t, 1.0),
            (current_start_t_front if current_start_t_front is not None else current_start_t, 1.0),
        ]))
        
        panel_faces.segments[face] = segments
    
    # -------------------------------------------------------------------------
    # Endpoint trim calculation and application
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _compute_face_trim(panel: GXMLPanel, face: PanelSide, endpoint: PanelSide,
                           intersection: Intersection) -> float:
        """
        Compute the trim value for a specific face at an endpoint intersection.
        
        Uses ray-plane intersection to find where the face edge meets the target.
        For angled intersections, each face needs a different trim amount.
        
        Args:
            panel: The panel being trimmed
            face: Which face to compute trim for
            endpoint: Which endpoint (START or END)
            intersection: The intersection at the endpoint
            
        Returns:
            Trim value (positive = trim inward, negative = overshoot)
        """
        other_panels = intersection.get_other_panels(panel)
        if not other_panels:
            return 0.0
        
        # Use adjacency lookup for thickness faces (works for both joints and T-junctions)
        if face in PanelSide.thickness_faces():
            adjacent = FaceSolver._get_adjacent_face(intersection, panel, face)
            if adjacent is None:
                return 0.0
            target_panel, target_face = adjacent
        else:
            # Non-thickness faces: use approached face logic
            target_panel, target_face = other_panels[0], None
        
        # Compute the trim using ray-plane intersection
        if target_panel.thickness < TOLERANCE:
            return 0.0
        
        ray = panel.get_primary_axis_ray()
        if ray is None:
            return 0.0
        
        # Get endpoint info from endpoint (START -> t=0, END -> t=1)
        t_value, _, _ = panel.get_face_center_local(endpoint)
        
        # Determine target face if not specified (approached face logic for T-junctions)
        if target_face is None:
            approach_dir = ray.direction if t_value < 0.5 else -ray.direction
            target_face = target_panel.get_face_closest_to_direction(
                approach_dir, candidate_faces=PanelSide.thickness_faces())
        
        # Get point on the face being trimmed (corner at the endpoint)
        # s: use face's s_center if it's an edge (0 or 1), otherwise use bottom edge
        _, s_center, z_offset = panel.get_face_center_local(face)
        s_value = s_center if s_center in (0.0, 1.0) else 0.0
        face_corner = panel.transform_point((t_value, s_value, z_offset))
        
        intersection_point = FaceSolver._intersect_line_with_panel_face(
            face_corner, ray.direction, target_panel, target_face)
        
        if intersection_point is None:
            return 0.0
        
        # Calculate offset: positive = trim inward, negative = overshoot
        if t_value < 0.5:  # START endpoint
            offset_vector = GXMLMath.sub3(intersection_point, face_corner)
        else:  # END endpoint
            offset_vector = GXMLMath.sub3(face_corner, intersection_point)
        
        offset_distance = GXMLMath.dot3(offset_vector, ray.direction)
        t_offset = min(offset_distance / ray.length, 1.0)
        
        return t_offset
    
    @staticmethod
    def _apply_endpoint_trims(panel: GXMLPanel, segments: List[FaceSegment], face_side: PanelSide,
                               panel_faces: SegmentedPanel,
                               start_intersection: Optional[Intersection],
                               end_intersection: Optional[Intersection]) -> None:
        """
        Apply endpoint trims to segment bounds in-place, computing trims inline.
        
        For unsplit TOP/BOTTOM faces, applies mitered corner trims where each
        corner gets independent t-values derived from FRONT/BACK trims.
        
        For split faces and FRONT/BACK faces, applies uniform trims at the
        first and last segments only.
        
        Args:
            panel: The panel
            segments: List of FaceSegment to modify in-place
            face_side: Which face these segments are for
            panel_faces: Panel faces for checking if face is split
            start_intersection: Intersection at panel start, if any
            end_intersection: Intersection at panel end, if any
        """
        if not segments:
            return
        
        # START/END faces don't get endpoint trims (they ARE the endpoints)
        if face_side in (PanelSide.START, PanelSide.END):
            return
        
        is_split = panel_faces.is_split(face_side)
        is_top_or_bottom = face_side in (PanelSide.TOP, PanelSide.BOTTOM)
        
        # Helper to compute trim and convert to t-value
        def get_t_start(face: PanelSide) -> float:
            if start_intersection is None:
                return 0.0
            trim = FaceSolver._compute_face_trim(panel, face, PanelSide.START, start_intersection)
            return trim if abs(trim) > TOLERANCE else 0.0
        
        def get_t_end(face: PanelSide) -> float:
            if end_intersection is None:
                return 1.0
            trim = FaceSolver._compute_face_trim(panel, face, PanelSide.END, end_intersection)
            return 1.0 - (trim if abs(trim) > TOLERANCE else 0.0)
        
        if is_top_or_bottom:
            # TOP/BOTTOM faces get mitered corner trims derived from FRONT/BACK
            t_start_back = get_t_start(PanelSide.BACK)
            t_end_back = get_t_end(PanelSide.BACK)
            t_start_front = get_t_start(PanelSide.FRONT)
            t_end_front = get_t_end(PanelSide.FRONT)
            
            if not is_split:
                # Unsplit: all corners get trims (should only be one segment)
                for seg in segments:
                    seg.corners[0] = (t_start_back, 0.0)
                    seg.corners[1] = (t_end_back, 0.0)
                    seg.corners[2] = (t_end_front, 1.0)
                    seg.corners[3] = (t_start_front, 1.0)
            else:
                # Split: first segment gets start trims, last segment gets end trims
                first_seg = segments[0]
                first_seg.corners[0] = (t_start_back, 0.0)
                first_seg.corners[3] = (t_start_front, 1.0)
                
                last_seg = segments[-1]
                last_seg.corners[1] = (t_end_back, 0.0)
                last_seg.corners[2] = (t_end_front, 1.0)
        else:
            # FRONT/BACK: apply uniform trims at first/last segment only
            t_start = get_t_start(face_side)
            t_end = get_t_end(face_side)
            
            # First segment gets start trim (both back and front edges)
            first_seg = segments[0]
            first_seg.corners[0] = (t_start, 0.0)
            first_seg.corners[3] = (t_start, 1.0)
            
            # Last segment gets end trim (both back and front edges)
            last_seg = segments[-1]
            last_seg.corners[1] = (t_end, 0.0)
            last_seg.corners[2] = (t_end, 1.0)

    # -------------------------------------------------------------------------
    # Gap calculation helper (used during segment creation)
    # -------------------------------------------------------------------------

    @staticmethod
    def _calculate_gap_t_values(intersected_panel: GXMLPanel, face_side: PanelSide, 
                                 intersecting_panel: GXMLPanel, 
                                 intersection_t: float) -> Tuple[float, float]:
        """
        Calculate t-values where gap edges should be at an intersection point.
        
        The gap accounts for the intersecting panel's dimension that blocks
        the face. For FRONT/BACK faces, this is the intersecting panel's thickness.
        For TOP/BOTTOM faces, this is the intersecting panel's height.
        
        At angled intersections, the gap is asymmetric.
        
        Returns:
            Tuple of (gap_start_t, gap_end_t)
        """
        # The gap in any lengthwise face (FRONT, BACK, TOP, BOTTOM) is always
        # determined by the intersecting panel's thickness - the distance between
        # its FRONT and BACK planes. This is because all these faces run along
        # the panel's primary axis, and the obstruction width in that direction
        # is the other panel's thickness.
        blocking_faces = (PanelSide.FRONT, PanelSide.BACK)
        blocking_dimension = intersecting_panel.thickness
        
        if blocking_dimension < TOLERANCE:
            return (intersection_t, intersection_t)
        
        # Get the ray along the intersected face (in world space)
        _, _, face_z_offset = intersected_panel.get_face_center_local(face_side)
        ray = intersected_panel.get_primary_axis_ray(face_z_offset)
        
        if ray is None:
            return (intersection_t, intersection_t)
        
        # Intersect with the blocking faces of the intersecting panel
        face1_intersection = FaceSolver._intersect_line_with_panel_face(
            ray.origin, ray.direction, intersecting_panel, blocking_faces[0])
        face2_intersection = FaceSolver._intersect_line_with_panel_face(
            ray.origin, ray.direction, intersecting_panel, blocking_faces[1])
        
        if face1_intersection is None or face2_intersection is None:
            simple_offset = (blocking_dimension / 2) / ray.length
            return (intersection_t - simple_offset, intersection_t + simple_offset)
        
        face1_t = ray.project_point(face1_intersection)
        face2_t = ray.project_point(face2_intersection)
        
        return (min(face1_t, face2_t), max(face1_t, face2_t))

    @staticmethod
    def _get_outward_face(entry: Intersection.PanelEntry, side: JointSide) -> PanelSide:
        """
        Get which face (FRONT or BACK) is on the CCW or CW side of a panel at a joint.
        
        The mapping depends on travel direction:
        - END at joint (traveling toward): FRONT is on CCW side, BACK is on CW side
        - START at joint (traveling away): BACK is on CCW side, FRONT is on CW side
        
        Args:
            entry: The panel entry from the joint
            side: JointSide.CCW for face toward next panel, JointSide.CW for face toward previous
            
        Returns:
            PanelSide.FRONT or PanelSide.BACK
        """
        is_end_at_joint = entry.t > 0.5  # END at joint means traveling toward
        
        if side == JointSide.CCW:
            return PanelSide.FRONT if is_end_at_joint else PanelSide.BACK
        else:  # JointSide.CW
            return PanelSide.BACK if is_end_at_joint else PanelSide.FRONT

    @staticmethod
    def _get_adjacent_face(intersection: Intersection, panel: GXMLPanel, face: PanelSide) -> Optional[Tuple[GXMLPanel, PanelSide]]:
        """
        Get the adjacent face of the neighboring panel at an intersection.
        
        For JOINT intersections (CCW-sorted panels):
        Each panel has FRONT and BACK faces, but which face is on which side of 
        the "wedge" between panels depends on the panel's travel direction:
        - END at joint (traveling toward): FRONT is on CCW side, BACK is on CW side
        - START at joint (traveling away): BACK is on CCW side, FRONT is on CW side
        Adjacency: current panel's CCW-side face meets next panel's CW-side face.
        
        For T_JUNCTION intersections:
        Both FRONT and BACK of the approaching panel map to the same "approached face"
        of the target panel (whichever of FRONT/BACK faces the approach direction).
        
        Args:
            intersection: The intersection (JOINT or T_JUNCTION)
            panel: The panel whose face we're starting from
            face: Which face of the panel (FRONT or BACK)
            
        Returns:
            Tuple of (neighbor_panel, neighbor_face) or None if not found
        """
        if face not in PanelSide.thickness_faces():
            return None
        
        panel_entry = intersection.get_entry(panel)
        if panel_entry is None:
            return None
        
        # T-junction: both faces map to the approached face of the target
        if intersection.type == IntersectionType.T_JUNCTION:
            other_panels = intersection.get_other_panels(panel)
            if not other_panels:
                return None
            target_panel = other_panels[0]
            
            # Compute approach direction from this panel toward the target
            # START at intersection (t≈0): approach is along positive primary axis
            # END at intersection (t≈1): approach is along negative primary axis
            direction = panel.get_primary_axis()
            if panel_entry.t >= 0.5:  # END at intersection
                direction = -direction
            
            # Find which face of target this direction approaches
            approached_face = target_panel.get_face_closest_to_direction(
                direction, candidate_faces=PanelSide.thickness_faces())
            return (target_panel, approached_face)
        
        # Joint: use CCW adjacency
        if intersection.type != IntersectionType.JOINT:
            return None
        
        panel_index = intersection.panels.index(panel_entry)
        num_panels = len(intersection.panels)
        
        # Determine which side (CCW or CW) this face is on for this panel
        ccw_face = FaceSolver._get_outward_face(panel_entry, JointSide.CCW)
        cw_face = FaceSolver._get_outward_face(panel_entry, JointSide.CW)
        
        if face == ccw_face:
            # This face is on the CCW side, so it meets the next panel's CW-side face
            next_index = (panel_index + 1) % num_panels
            neighbor_entry = intersection.panels[next_index]
            neighbor_face = FaceSolver._get_outward_face(neighbor_entry, JointSide.CW)
            return (neighbor_entry.panel, neighbor_face)
        elif face == cw_face:
            # This face is on the CW side, so it meets the previous panel's CCW-side face
            prev_index = (panel_index - 1) % num_panels
            neighbor_entry = intersection.panels[prev_index]
            neighbor_face = FaceSolver._get_outward_face(neighbor_entry, JointSide.CCW)
            return (neighbor_entry.panel, neighbor_face)
        
        return None

    @staticmethod
    def _intersect_line_with_panel_face(line_start: np.ndarray, line_direction: np.ndarray,
                                         panel: GXMLPanel, face: PanelSide) -> Optional[np.ndarray]:
        """
        Intersect a line with a panel face plane.
        
        Args:
            line_start: Starting point of the line
            line_direction: Normalized direction of the line
            panel: The panel whose face to intersect with
            face: Which face
            
        Returns:
            Intersection point, or None if parallel
        """
        face_normal = panel.get_face_normal(face)
        face_point = panel.get_face_center_world(face)
        return intersect_line_with_plane(line_start, line_direction, face_point, face_normal)