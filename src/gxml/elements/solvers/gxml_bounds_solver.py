"""
Bounds solving logic - computes trim/gap adjustments for panel faces.

This module is Stage 3 of the geometry pipeline:
1. IntersectionSolver - finds where panel centerlines meet
2. FaceSolver - determines face segmentation from intersection topology
3. BoundsSolver - computes trim/gap adjustments, provides coordinate lookup
4. GeometryBuilder - creates actual 3D geometry (polygons and caps)

Key responsibilities:
- Compute endpoint trim values for mitered joints (ray-plane intersection)
- Compute gap t-values at crossing intersections (ray-plane intersection)
- Provide world-space coordinate lookup for face segments

Does NOT create (that's GeometryBuilder's job):
- Polygon geometry
- Joint caps and crossing caps
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, Optional, List, Dict

import numpy as np

from .gxml_intersection_solver import IntersectionSolution, IntersectionType, Intersection, PanelEndpoint
from .gxml_face_solver import FaceSolverResult, FaceSolution, FaceSegment, BoundaryType
from elements.gxml_panel import GXMLPanel, PanelSide
from mathutils.gxml_math import intersect_line_with_plane

# Tolerance for geometry calculations
TOLERANCE = 1e-4


class JointSide(Enum):
    """Side of a panel at a joint for adjacency lookup."""
    CCW = auto()  # Counter-clockwise side (toward next panel)
    CW = auto()   # Clockwise side (toward previous panel)


@dataclass
class FaceBounds:
    """Bounds for a face segment along the panel's primary and secondary axes.
    
    For TOP/BOTTOM faces at angled crossings, the gap edges are at different
    t-values for the front edge vs the back edge. In this case:
    - t_start/t_end are the values at the back edge (s=0)
    - t_start_front/t_end_front are the values at the front edge (s=1)
    
    For FRONT/BACK faces, all four values are the same (rectangular region).
    """
    t_start: float = 0.0
    t_end: float = 1.0
    s_start: float = 0.0
    s_end: float = 1.0
    # Per-edge t values for angled crossings on TOP/BOTTOM faces
    t_start_front: Optional[float] = None  # If None, use t_start
    t_end_front: Optional[float] = None    # If None, use t_end
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """
        Get the corner coordinates for this face segment.
        
        Returns:
            List of 4 (t, s) corner coordinates in order:
            start-back, end-back, end-front, start-front
        """
        t_start_front = self.t_start_front if self.t_start_front is not None else self.t_start
        t_end_front = self.t_end_front if self.t_end_front is not None else self.t_end
        return [
            (self.t_start, self.s_start),
            (self.t_end, self.s_start),
            (t_end_front, self.s_end),
            (t_start_front, self.s_end),
        ]


@dataclass
class PanelEndpointTrims:
    """Trim values for both panel endpoints (START and END)."""
    start_trims: Dict[PanelSide, float] = field(default_factory=dict)
    end_trims: Dict[PanelSide, float] = field(default_factory=dict)
    start_intersected_panel: Optional[GXMLPanel] = None
    end_intersected_panel: Optional[GXMLPanel] = None
    
    def get_start(self, face: PanelSide, default: float = 0.0) -> float:
        """Get trim value for a face at START endpoint."""
        return self.start_trims.get(face, default)
    
    def get_end(self, face: PanelSide, default: float = 0.0) -> float:
        """Get trim value for a face at END endpoint."""
        return self.end_trims.get(face, default)
    
    def has_start_intersection(self) -> bool:
        """Whether START endpoint has an intersection."""
        return self.start_intersected_panel is not None
    
    def has_end_intersection(self) -> bool:
        """Whether END endpoint has an intersection."""
        return self.end_intersected_panel is not None
    
    @staticmethod
    def _apply_trim_to_bounds(start_trim: float, end_trim: float) -> Tuple[float, float]:
        """
        Convert trim values to t_start/t_end bounds, handling overshoot.
        
        Args:
            start_trim: Trim value for start endpoint (positive = trim, negative = overshoot)
            end_trim: Trim value for end endpoint (positive = trim, negative = overshoot)
            
        Returns:
            Tuple of (t_start, t_end) with trims applied
        """
        t_start = start_trim if abs(start_trim) > TOLERANCE else 0.0
        t_end = 1.0 - end_trim if abs(end_trim) > TOLERANCE else 1.0
        return (t_start, t_end)
    
    def get_t_bounds(self, face: PanelSide) -> Tuple[float, float]:
        """
        Get the t_start/t_end bounds for a face after applying trims.
        
        Args:
            face: Which face to get bounds for
            
        Returns:
            Tuple of (t_start, t_end) with this face's trims applied
        """
        return self._apply_trim_to_bounds(self.get_start(face), self.get_end(face))
    
    def get_face_t_bounds(self, face: PanelSide) -> Tuple[float, float, float, float]:
        """
        Get the t-bounds for all four corners of a face.
        
        For TOP/BOTTOM faces, returns per-edge values derived from FRONT/BACK trims
        to create proper mitered corners. For START/END faces, returns fixed values
        at t=0 or t=1. For other faces, all four values are the same.
        
        Returns:
            Tuple of (t_start_back, t_end_back, t_start_front, t_end_front)
            - t_start_back/t_end_back: t-values at s=0 (BACK edge)
            - t_start_front/t_end_front: t-values at s=1 (FRONT edge)
        """
        if face == PanelSide.START:
            return (0.0, 0.0, 0.0, 0.0)
        elif face == PanelSide.END:
            return (1.0, 1.0, 1.0, 1.0)
        elif face in (PanelSide.TOP, PanelSide.BOTTOM):
            # Corner trims derived from FRONT/BACK
            t_start_back, t_end_back = self._apply_trim_to_bounds(
                self.get_start(PanelSide.BACK), self.get_end(PanelSide.BACK))
            t_start_front, t_end_front = self._apply_trim_to_bounds(
                self.get_start(PanelSide.FRONT), self.get_end(PanelSide.FRONT))
            return (t_start_back, t_end_back, t_start_front, t_end_front)
        else:
            # Same value for all corners
            t_start, t_end = self._apply_trim_to_bounds(
                self.get_start(face), self.get_end(face))
            return (t_start, t_end, t_start, t_end)


@dataclass
class BoundsSolution:
    """Result of bounds solving - adjustments and coordinate lookup.
    
    Contains the face segmentation result, computed trims, and computed gap
    adjustments. Provides methods to get world-space coordinates for face
    segments.
    """
    face_result: FaceSolverResult
    trims: Dict[GXMLPanel, PanelEndpointTrims] = field(default_factory=dict)
    # Face segments with computed bounds per panel per face
    _face_segments: Dict[GXMLPanel, Dict[PanelSide, List[FaceSegment]]] = field(default_factory=dict)
    
    @property
    def intersection_solution(self) -> IntersectionSolution:
        """Get the underlying intersection solution."""
        return self.face_result.intersection_solution
    
    def get_segments(self, panel: GXMLPanel, face_side: PanelSide) -> List[FaceSegment]:
        """Get face segments from underlying face result."""
        face_solution = self.face_result.get(panel)
        if face_solution is None:
            return []
        return face_solution.get_segments(face_side)
    
    def get_endpoint_trims(self, panel: GXMLPanel) -> Optional[PanelEndpointTrims]:
        """Get endpoint trims for a panel."""
        return self.trims.get(panel)
    
    def get_segment_bounds(self, panel: GXMLPanel, face_side: PanelSide, 
                           segment_index: int = 0) -> Optional[FaceBounds]:
        """Get the computed bounds for a face segment."""
        segments = self._face_segments.get(panel, {}).get(face_side)
        if segments is None or segment_index >= len(segments):
            return None
        return segments[segment_index].computed_bounds
    
    def get_face_corners(self, panel: GXMLPanel, face_side: PanelSide,
                         segment_index: int = 0) -> Optional[List[np.ndarray]]:
        """
        Get world-space corners for a face segment.
        
        Args:
            panel: The panel
            face_side: Which face
            segment_index: Which segment (0 for unsplit faces)
            
        Returns:
            List of 4 world-space corner positions, or None if not found
        """
        seg_bounds = self.get_segment_bounds(panel, face_side, segment_index)
        if seg_bounds is None:
            return None
        return BoundsSolver._bounds_to_world_corners(panel, face_side, seg_bounds.bounds)
    
    def get_all_face_corners(self, panel: GXMLPanel, face_side: PanelSide) -> List[List[np.ndarray]]:
        """
        Get world-space corners for all segments of a face.
        
        Returns:
            List of corner lists (one per segment)
        """
        segments = self._face_segments.get(panel, {}).get(face_side)
        if segments is None:
            return []
        return [
            BoundsSolver._bounds_to_world_corners(panel, face_side, seg.computed_bounds)
            for seg in segments if seg.computed_bounds is not None
        ]
    
    def get_segment_count(self, panel: GXMLPanel, face_side: PanelSide) -> int:
        """Get the number of segments for a face."""
        segments = self._face_segments.get(panel, {}).get(face_side)
        return len(segments) if segments else 0
    
    def get_crossing_gap_edge(self, panel: GXMLPanel, face_side: PanelSide, 
                               is_top: bool) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the gap edge for a FRONT/BACK face at a crossing.
        
        At a crossing, FRONT/BACK faces are split. The gap is between segment[0]'s
        end and segment[1]'s start. We return the edge line for cap creation.
        
        Args:
            panel: The panel
            face_side: Which face (FRONT or BACK)
            is_top: True for top edge (s=1), False for bottom (s=0)
            
        Returns:
            Tuple of (point1, point2) for the gap edge, or None
        """
        if self.get_segment_count(panel, face_side) < 2:
            return None  # No crossing gap if not split
        
        # Get corners for both segments
        corners0 = self.get_face_corners(panel, face_side, 0)
        corners1 = self.get_face_corners(panel, face_side, 1)
        
        if corners0 is None or corners1 is None:
            return None
        
        # Corner ordering: 0=start-back, 1=end-back, 2=end-front, 3=start-front
        # For FRONT/BACK faces: back=bottom (s=0), front=top (s=1)
        if is_top:
            # Top edge of gap: segment[0]'s end-top to segment[1]'s start-top
            return (corners0[2], corners1[3])
        else:
            # Bottom edge of gap
            return (corners0[1], corners1[0])


class BoundsSolver:
    """
    Stage 3: Computes trim/gap adjustments for face segments.
    
    Takes FaceSolverResult and computes:
    - Endpoint trims (ray-plane intersection for miters)
    - Gap t-values (ray-plane intersection for crossing gaps)
    
    Results are stored in BoundsSolution which provides coordinate lookup.
    """
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    @staticmethod
    def solve(face_result: FaceSolverResult) -> BoundsSolution:
        """
        Compute trim/gap adjustments for all panels.
        
        Args:
            face_result: The face solver result with segmentation
            
        Returns:
            BoundsSolution with computed adjustments and coordinate lookup
        """
        solution = face_result.intersection_solution
        result = BoundsSolution(face_result=face_result)
        
        # Compute trims for all panels
        for panel in solution.panels:
            if not panel.is_valid(TOLERANCE):
                continue
            trims = BoundsSolver._calculate_endpoint_trims(panel, solution)
            result.trims[panel] = trims
        
        # Compute gap-adjusted bounds for all panels
        for panel in solution.panels:
            if not panel.is_valid(TOLERANCE):
                continue
            
            face_solution = face_result.get(panel)
            if face_solution is None:
                continue
            
            trims = result.trims.get(panel, PanelEndpointTrims())
            panel_segments = BoundsSolver._compute_panel_bounds(panel, face_solution, trims)
            result._face_segments[panel] = panel_segments
        
        return result
    
    # -------------------------------------------------------------------------
    # Panel bounds computation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _compute_panel_bounds(panel: GXMLPanel, face_solution: FaceSolution,
                               trims: PanelEndpointTrims) -> Dict[PanelSide, List[FaceSegment]]:
        """
        Compute gap-adjusted bounds for all faces of a panel.
        
        Args:
            panel: The panel
            face_solution: Face segmentation for this panel
            trims: Endpoint trims for this panel
            
        Returns:
            Dict mapping face side to list of FaceSegment (with computed_bounds populated)
        """
        result: Dict[PanelSide, List[FaceSegment]] = {}
        
        for face_side in PanelSide:
            segments = face_solution.get_segments(face_side)
            if not segments:
                continue
            
            # Convert segments to bounds with gaps
            bounds_list = []
            for segment in segments:
                bounds = segment.get_nominal_bounds()
                BoundsSolver._apply_gaps(panel, face_side, segment, bounds)
                bounds_list.append(bounds)
            
            # Apply endpoint trims
            BoundsSolver._apply_endpoint_trims(bounds_list, face_side, trims, face_solution)
            
            # Populate computed_bounds on each segment
            for segment, bounds in zip(segments, bounds_list):
                segment.computed_bounds = bounds
            
            result[face_side] = segments
        
        return result
        
        return result
    
    # -------------------------------------------------------------------------
    # Coordinate transformation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _bounds_to_world_corners(panel: GXMLPanel, face_side: PanelSide,
                                  bounds: FaceBounds) -> List[np.ndarray]:
        """
        Convert FaceBounds to world-space corners.
        
        The mapping from (t, s) bounds to 3D coordinates depends on the face:
        - FRONT/BACK: t->x, s->y, z is fixed at ±half_thickness
        - TOP/BOTTOM: t->x, s->z (thickness), y is fixed at 0 or 1
        - START/END: t->z (thickness), s->y, x is fixed at 0 or 1
        
        Args:
            panel: The panel
            face_side: Which face
            bounds: The bounds with t/s values
            
        Returns:
            List of 4 world-space corner positions
        """
        local_corners = bounds.get_corners()
        
        def local_to_face_3d(t: float, s: float) -> Tuple[float, float, float]:
            """Convert (t, s) bounds to local 3D coordinates for this face."""
            half_thickness = panel.thickness / 2
            if face_side == PanelSide.FRONT:
                return (t, s, half_thickness)
            elif face_side == PanelSide.BACK:
                return (t, s, -half_thickness)
            elif face_side == PanelSide.TOP:
                # s maps to z (thickness direction), t maps to length
                z = -half_thickness + s * panel.thickness
                return (t, 1.0, z)
            elif face_side == PanelSide.BOTTOM:
                z = -half_thickness + s * panel.thickness
                return (t, 0.0, z)
            elif face_side == PanelSide.START:
                # t maps to z (thickness), s maps to y (height)
                z = -half_thickness + t * panel.thickness
                return (0.0, s, z)
            elif face_side == PanelSide.END:
                z = -half_thickness + t * panel.thickness
                return (1.0, s, z)
            return (t, s, 0.0)
        
        world_corners = []
        for t, s in local_corners:
            local_3d = local_to_face_3d(t, s)
            world_pos = panel.transform_point(local_3d)
            world_corners.append(world_pos)
        
        return world_corners
    
    # -------------------------------------------------------------------------
    # Endpoint trim calculation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _calculate_endpoint_trims(panel: GXMLPanel, solution: IntersectionSolution) -> PanelEndpointTrims:
        """
        Calculate per-face trims for START and END based on intersections at endpoints.
        
        When a panel meets another panel at its start or end (T-junction or joint),
        each face needs to be trimmed individually because angled intersections cause
        different faces to meet the target at different points.
        
        For T-junctions: trim to the approached face (FRONT or BACK) of the target panel.
        For joints: use CCW adjacency to find correct neighbor for each face.
        
        Args:
            panel: The panel to calculate trims for
            solution: The intersection solution
            
        Returns:
            PanelEndpointTrims containing trim data for both endpoints
        """
        result = PanelEndpointTrims()
        
        for intersection in solution.intersections:
            entry = intersection.get_entry(panel)
            if entry is None:
                continue
            
            endpoint = entry.get_endpoint()
            if endpoint is None:
                continue
            
            face_side = PanelSide.START if endpoint == PanelEndpoint.START else PanelSide.END
            is_start = (endpoint == PanelEndpoint.START)
            trims_dict = result.start_trims if is_start else result.end_trims
            
            other_panels = intersection.get_other_panels(panel)
            if not other_panels:
                continue
            
            for face in PanelSide:
                # Use adjacency lookup for thickness faces (works for both joints and T-junctions)
                if face in PanelSide.thickness_faces():
                    adjacent = BoundsSolver.get_adjacent_face(intersection, panel, face)
                    if adjacent is None:
                        continue
                    target_panel, target_face = adjacent
                else:
                    # Non-thickness faces: use approached face logic
                    target_panel, target_face = other_panels[0], None
                
                trim = BoundsSolver._compute_trim_to_panel_face(
                    panel, target_panel, face_side, face, target_face=target_face)
                
                # Set the intersected panel for face suppression
                if is_start and result.start_intersected_panel is None:
                    result.start_intersected_panel = other_panels[0]
                elif not is_start and result.end_intersected_panel is None:
                    result.end_intersected_panel = other_panels[0]
                
                # Keep the maximum trim (most conservative)
                current = trims_dict.get(face, float('-inf'))
                trims_dict[face] = max(current, trim)
        
        return result
    
    @staticmethod
    def _compute_trim_to_panel_face(panel: GXMLPanel, intersected_panel: GXMLPanel, 
                                     face_side: PanelSide, face_to_trim: PanelSide,
                                     target_face: Optional[PanelSide] = None,
                                     allow_overshoot: bool = True) -> float:
        """
        Compute the trim offset for a face to meet another panel's face.
        
        Uses ray-plane intersection to find where the face edge meets the target.
        For angled intersections, each face needs a different trim amount.
        
        Args:
            panel: The panel being trimmed
            intersected_panel: The panel we're intersecting with
            face_side: Which endpoint (START or END) is being trimmed
            face_to_trim: Which face (FRONT, BACK, etc.) to calculate trim for
            target_face: Which face of intersected_panel to intersect with.
                         If None, uses "approached face" logic (for T-junctions).
            allow_overshoot: If True, allow negative trim values (face extends past intersection)
        
        Returns:
            The t-value offset to apply. Negative values indicate overshoot.
        """
        if intersected_panel.thickness < TOLERANCE:
            return 0.0
        
        ray = panel.get_primary_axis_ray()
        if ray is None:
            return 0.0
        
        # Get endpoint info from face_side (START -> t=0, END -> t=1)
        t_value, _, _ = panel.get_face_center_local(face_side)
        
        # Determine target face if not specified (approached face logic for T-junctions)
        if target_face is None:
            approach_dir = ray.direction if t_value < 0.5 else -ray.direction
            target_face = intersected_panel.get_face_closest_to_direction(
                approach_dir, candidate_faces=PanelSide.thickness_faces())
        
        # Get point on the face being trimmed (corner at the endpoint)
        # s: use face's s_center if it's an edge (0 or 1), otherwise use bottom edge
        _, s_center, z_offset = panel.get_face_center_local(face_to_trim)
        s_value = s_center if s_center in (0.0, 1.0) else 0.0
        face_corner = panel.transform_point((t_value, s_value, z_offset))
        
        intersection_point = BoundsSolver._intersect_line_with_panel_face(
            face_corner, ray.direction, intersected_panel, target_face)
        
        if intersection_point is None:
            return 0.0
        
        # Calculate offset: positive = trim inward, negative = overshoot
        if t_value < 0.5:  # START endpoint
            offset_vector = intersection_point - face_corner
        else:  # END endpoint
            offset_vector = face_corner - intersection_point
        
        offset_distance = np.dot(offset_vector, ray.direction)
        t_offset = min(offset_distance / ray.length, 1.0)
        
        return t_offset if allow_overshoot else max(0.0, t_offset)
    
    @staticmethod
    def _apply_endpoint_trims(bounds_list: List[FaceBounds], face_side: PanelSide,
                               trims: PanelEndpointTrims,
                               face_solution: FaceSolution) -> None:
        """
        Apply endpoint trims to face bounds in-place.
        
        For unsplit TOP/BOTTOM faces, applies mitered corner trims where each
        corner gets independent t-values derived from FRONT/BACK trims.
        
        For split faces and FRONT/BACK faces, applies uniform trims at the
        first and last segments only.
        
        Args:
            bounds_list: List of FaceBounds to modify in-place
            face_side: Which face these bounds are for
            trims: Pre-computed endpoint trims
            face_solution: Face solution for checking if face is split
        """
        if not bounds_list:
            return
        
        # START/END faces don't get endpoint trims (they ARE the endpoints)
        if face_side in (PanelSide.START, PanelSide.END):
            return
        
        is_split = face_solution.is_split(face_side)
        is_top_or_bottom = face_side in (PanelSide.TOP, PanelSide.BOTTOM)
        
        if is_top_or_bottom:
            # TOP/BOTTOM faces always get mitered corner trims derived from FRONT/BACK
            t_start_back, t_end_back, t_start_front, t_end_front = trims.get_face_t_bounds(face_side)
            
            if not is_split:
                # Unsplit: all corners get trims (should only be one segment)
                for bounds in bounds_list:
                    bounds.t_start = t_start_back
                    bounds.t_end = t_end_back
                    bounds.t_start_front = t_start_front
                    bounds.t_end_front = t_end_front
            else:
                # Split: first segment gets start trims, last segment gets end trims
                first_bounds = bounds_list[0]
                first_bounds.t_start = t_start_back
                first_bounds.t_start_front = t_start_front
                
                last_bounds = bounds_list[-1]
                last_bounds.t_end = t_end_back
                last_bounds.t_end_front = t_end_front
        else:
            # Split faces or FRONT/BACK: apply uniform trims at first/last segment only
            panel_t_start, panel_t_end = trims.get_t_bounds(face_side)
            
            # First segment gets start trim
            first_bounds = bounds_list[0]
            first_bounds.t_start = panel_t_start
            first_bounds.t_start_front = panel_t_start
            
            # Last segment gets end trim
            last_bounds = bounds_list[-1]
            last_bounds.t_end = panel_t_end
            last_bounds.t_end_front = panel_t_end
    
    # -------------------------------------------------------------------------
    # Gap calculation for crossings and T-junctions
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _apply_gaps(panel: GXMLPanel, face_side: PanelSide,
                     segment: FaceSegment, bounds: FaceBounds) -> None:
        """
        Apply gap adjustments to face bounds in-place.
        
        Modifies the bounds' t-values to account for the thickness of
        intersecting panels at crossing/T-junction boundaries.
        
        For TOP/BOTTOM faces at angled crossings, the gap edges are at different
        t-values for the front edge vs the back edge.
        
        Args:
            panel: The panel being processed
            face_side: Which face is being created
            segment: The face segment with boundary info
            bounds: FaceBounds to modify in-place
        """
        is_edge_face = face_side in PanelSide.edge_faces()
        
        # Apply gaps at start boundary
        # CROSSING and T_JUNCTION create gaps; JOINT does not (both panels terminate)
        if segment.start_type in (BoundaryType.CROSSING, BoundaryType.T_JUNCTION) and segment.start_intersecting_panels:
            intersecting_panel = segment.start_intersecting_panels[0]
            if is_edge_face:
                # For TOP/BOTTOM: compute gap at BACK and FRONT edges
                _, gap_end_back = BoundsSolver._calculate_gap_t_values(
                    panel, PanelSide.BACK, intersecting_panel, bounds.t_start
                )
                _, gap_end_front = BoundsSolver._calculate_gap_t_values(
                    panel, PanelSide.FRONT, intersecting_panel, bounds.t_start
                )
                bounds.t_start = gap_end_back
                bounds.t_start_front = gap_end_front
            else:
                _, gap_end = BoundsSolver._calculate_gap_t_values(
                    panel, face_side, intersecting_panel, bounds.t_start
                )
                bounds.t_start = gap_end
        
        # Apply gaps at end boundary
        if segment.end_type in (BoundaryType.CROSSING, BoundaryType.T_JUNCTION) and segment.end_intersecting_panels:
            intersecting_panel = segment.end_intersecting_panels[0]
            if is_edge_face:
                gap_start_back, _ = BoundsSolver._calculate_gap_t_values(
                    panel, PanelSide.BACK, intersecting_panel, bounds.t_end
                )
                gap_start_front, _ = BoundsSolver._calculate_gap_t_values(
                    panel, PanelSide.FRONT, intersecting_panel, bounds.t_end
                )
                bounds.t_end = gap_start_back
                bounds.t_end_front = gap_start_front
            else:
                gap_start, _ = BoundsSolver._calculate_gap_t_values(
                    panel, face_side, intersecting_panel, bounds.t_end
                )
                bounds.t_end = gap_start

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
        face1_intersection = BoundsSolver._intersect_line_with_panel_face(
            ray.origin, ray.direction, intersecting_panel, blocking_faces[0])
        face2_intersection = BoundsSolver._intersect_line_with_panel_face(
            ray.origin, ray.direction, intersecting_panel, blocking_faces[1])
        
        if face1_intersection is None or face2_intersection is None:
            simple_offset = (blocking_dimension / 2) / ray.length
            return (intersection_t - simple_offset, intersection_t + simple_offset)
        
        face1_t = ray.project_point(face1_intersection)
        face2_t = ray.project_point(face2_intersection)
        
        return (min(face1_t, face2_t), max(face1_t, face2_t))
    
    # -------------------------------------------------------------------------
    # Adjacent face lookup (for mitered joints)
    # -------------------------------------------------------------------------
    
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
    def get_adjacent_face(intersection: Intersection, panel: GXMLPanel, face: PanelSide) -> Optional[Tuple[GXMLPanel, PanelSide]]:
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
        ccw_face = BoundsSolver._get_outward_face(panel_entry, JointSide.CCW)
        cw_face = BoundsSolver._get_outward_face(panel_entry, JointSide.CW)
        
        if face == ccw_face:
            # This face is on the CCW side, so it meets the next panel's CW-side face
            next_index = (panel_index + 1) % num_panels
            neighbor_entry = intersection.panels[next_index]
            neighbor_face = BoundsSolver._get_outward_face(neighbor_entry, JointSide.CW)
            return (neighbor_entry.panel, neighbor_face)
        elif face == cw_face:
            # This face is on the CW side, so it meets the previous panel's CCW-side face
            prev_index = (panel_index - 1) % num_panels
            neighbor_entry = intersection.panels[prev_index]
            neighbor_face = BoundsSolver._get_outward_face(neighbor_entry, JointSide.CCW)
            return (neighbor_entry.panel, neighbor_face)
        
        return None

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
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
        face_offset = panel.get_face_center_local(face)
        face_point = panel.transform_point(face_offset)
        return intersect_line_with_plane(line_start, line_direction, face_point, face_normal)
