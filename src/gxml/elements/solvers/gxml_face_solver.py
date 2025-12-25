"""
Face segmentation logic - determines how panel faces are divided by intersections.

This module is Stage 2 of the geometry pipeline:
1. IntersectionSolver - finds where panel centerlines meet
2. FaceSolver - determines face segmentation from intersection topology
3. GeometryBuilder - creates actual 3D geometry

Key responsibilities:
- Determine which faces are split (and into how many segments)
- Identify what boundaries exist between segments
- Track which panels affect each boundary

Does NOT compute (that's GeometryBuilder's job):
- Actual gap t-values (requires ray-plane intersection)
- Endpoint trim values (requires ray-plane intersection)
- Polygon vertex coordinates
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict

from .gxml_intersection_solver import IntersectionSolution, Intersection, IntersectionType
from elements.gxml_panel import GXMLPanel, PanelSide


class BoundaryType(Enum):
    """What kind of boundary terminates a face segment."""
    PANEL_START = auto()   # t=0 (no intersection)
    PANEL_END = auto()     # t=1 (no intersection)
    CROSSING = auto()      # Midspan crossing - creates gap
    T_JUNCTION = auto()    # Endpoint meets midspan - creates trim
    JOINT = auto()         # Endpoint meets endpoint - creates trim


@dataclass
class FaceSegment:
    """A single segment of a face between two boundaries.
    
    Describes WHAT is at each boundary (topology), not the exact geometry.
    BoundsSolver computes actual gap-adjusted bounds and populates computed_bounds.
    """
    # Start boundary
    start_type: BoundaryType
    start_t: float  # Nominal t-value (intersection t, or 0.0)
    
    # End boundary
    end_type: BoundaryType
    end_t: float  # Nominal t-value (intersection t, or 1.0)
    
    # Optional fields with defaults
    start_intersection: Optional[Intersection] = None
    start_intersecting_panels: List[GXMLPanel] = field(default_factory=list)
    end_intersection: Optional[Intersection] = None
    end_intersecting_panels: List[GXMLPanel] = field(default_factory=list)
    
    # Gap-adjusted bounds (populated by BoundsSolver)
    computed_bounds: Optional['FaceBounds'] = None
    
    @property
    def nominal_t_start(self) -> float:
        return self.start_t
    
    @property
    def nominal_t_end(self) -> float:
        return self.end_t
    
    def get_nominal_bounds(self) -> 'FaceBounds':
        """
        Get nominal face bounds for this segment (without gap adjustments).
        
        Returns a FaceBounds with t_start/t_end from the boundary t-values.
        BoundsSolver should apply gap adjustments afterward.
        
        Returns:
            FaceBounds with nominal t-range (s_start=0, s_end=1)
        """
        # Import here to avoid circular dependency
        from .gxml_bounds_solver import FaceBounds
        return FaceBounds(
            t_start=self.start_t,
            t_end=self.end_t,
            s_start=0.0,
            s_end=1.0
        )


@dataclass 
class FaceSolution:
    """Face segmentation result for a single panel.
    
    For each face side, stores the list of segments (in t-order).
    Unsplit faces have a single segment spanning [0, 1].
    """
    panel: GXMLPanel
    # face -> ordered list of segments
    segments: Dict[PanelSide, List[FaceSegment]] = field(default_factory=dict)
    
    # Endpoint intersections (for START/END face suppression decision)
    start_intersection: Optional[Intersection] = None
    end_intersection: Optional[Intersection] = None
    
    def is_split(self, face: PanelSide) -> bool:
        """Whether this face has multiple segments."""
        return face in self.segments and len(self.segments[face]) > 1
    
    def get_segments(self, face: PanelSide) -> List[FaceSegment]:
        """Get segments for a face (empty list if face not present)."""
        return self.segments.get(face, [])
    
    def has_start_intersection(self) -> bool:
        """Whether there's an intersection at the START endpoint."""
        return self.start_intersection is not None
    
    def has_end_intersection(self) -> bool:
        """Whether there's an intersection at the END endpoint."""
        return self.end_intersection is not None


@dataclass
class FaceSolverResult:
    """Complete face segmentation for all panels."""
    # panel -> FaceSolution
    solutions: Dict[GXMLPanel, FaceSolution] = field(default_factory=dict)
    
    # Original intersection solution (for reference by downstream stages)
    intersection_solution: Optional[IntersectionSolution] = None
    
    def get(self, panel: GXMLPanel) -> Optional[FaceSolution]:
        """Get face solution for a panel."""
        return self.solutions.get(panel)


# Tolerance for t-value comparisons (same as used in intersection solver)
ENDPOINT_TOLERANCE = 0.01

# Tolerance for geometry calculations (thickness checks)
TOLERANCE = 1e-4


class FaceSolver:
    """
    Stage 2: Determines face segmentation from intersection topology.
    
    Takes the intersection solution and determines:
    - Which faces are split (and into how many segments)
    - What boundaries exist between segments
    - Which panels affect each boundary
    
    Does NOT compute:
    - Actual gap t-values (that's geometry)
    - Endpoint trim values (that's geometry)
    - Polygon coordinates (that's geometry)
    """
    
    @staticmethod
    def solve(intersection_solution: IntersectionSolution) -> FaceSolverResult:
        """
        Compute face segmentation for all panels.
        
        Args:
            intersection_solution: Output from IntersectionSolver
            
        Returns:
            FaceSolverResult with segment structure for all panels
        """
        result = FaceSolverResult(intersection_solution=intersection_solution)
        
        for panel in intersection_solution.panels:
            result.solutions[panel] = FaceSolver._solve_panel(panel, intersection_solution)
        
        return result
    
    @staticmethod
    def _solve_panel(panel: GXMLPanel, solution: IntersectionSolution) -> FaceSolution:
        """
        Compute face segmentation for a single panel.
        
        Args:
            panel: The panel to analyze
            solution: The intersection solution
            
        Returns:
            FaceSolution with segments for each face
        """
        face_solution = FaceSolution(panel=panel)
        
        # Collect intersection info per face
        # face -> list of (t, intersection, intersecting_panels)
        face_intersections: Dict[PanelSide, List[tuple]] = {}
        
        for intersection in solution.get_intersections_for_panel(panel):
            entry = intersection.get_entry(panel)
            t = entry.t
            is_at_start = t < ENDPOINT_TOLERANCE
            is_at_end = t > (1.0 - ENDPOINT_TOLERANCE)
            
            # Track endpoint intersections for START/END face suppression
            if is_at_start:
                face_solution.start_intersection = intersection
            if is_at_end:
                face_solution.end_intersection = intersection
            
            # For each other panel, determine which faces it affects
            for other_panel in intersection.get_other_panels(panel):
                affected_faces = FaceSolver._get_affected_faces(
                    panel, other_panel, intersection)
                
                for face in affected_faces:
                    if face not in face_intersections:
                        face_intersections[face] = []
                    
                    # Check if we already have this intersection recorded for this face
                    existing = next(
                        (item for item in face_intersections[face] if item[1] is intersection),
                        None
                    )
                    
                    if existing:
                        # Add to existing intersection's panel list
                        existing[2].append(other_panel)
                    else:
                        face_intersections[face].append((t, intersection, [other_panel]))
        
        # Sort each face's intersections by t
        for face in face_intersections:
            face_intersections[face].sort(key=lambda x: x[0])
        
        # Build segments for FRONT, BACK, TOP, BOTTOM faces
        for face_side in [PanelSide.FRONT, PanelSide.BACK, PanelSide.TOP, PanelSide.BOTTOM]:
            intersections = face_intersections.get(face_side, [])
            face_solution.segments[face_side] = FaceSolver._build_segments_for_face(
                panel, face_side, intersections)
        
        # Handle START and END faces
        FaceSolver._add_endpoint_faces(face_solution)
        
        # Apply zero-thickness suppression if needed
        if panel.thickness <= TOLERANCE:
            FaceSolver._apply_zero_thickness_suppression(face_solution)
        
        return face_solution
    
    @staticmethod
    def _get_affected_faces(panel: GXMLPanel, other_panel: GXMLPanel, 
                            intersection: Intersection) -> List[PanelSide]:
        """
        Determine which faces of panel are affected by other_panel at intersection.
        
        - Crossings affect all four lengthwise faces (FRONT, BACK, TOP, BOTTOM)
        - T-junctions/joints only affect the approached face
        
        Args:
            panel: The panel whose faces we're checking
            other_panel: The panel creating the intersection
            intersection: The intersection object
            
        Returns:
            List of affected PanelSide values
        """
        if intersection.type == IntersectionType.CROSSING:
            # Crossings affect all four lengthwise faces
            return list(PanelSide.thickness_faces()) + list(PanelSide.edge_faces())
        else:
            # T-junctions and joints only affect the approached face
            other_entry = intersection.get_entry(other_panel)
            direction = other_panel.get_primary_axis()
            if other_entry.t >= 0.5:
                direction = -direction
            approached = panel.get_face_closest_to_direction(
                direction, candidate_faces=PanelSide.thickness_faces())
            return [approached]
    
    @staticmethod
    def _build_segments_for_face(panel: GXMLPanel, face: PanelSide,
                                  intersections: List[tuple]) -> List[FaceSegment]:
        """
        Build segments for a face from its intersection list.
        
        Args:
            panel: The panel
            face: Which face we're building segments for
            intersections: List of (t, intersection, [panels]) tuples, sorted by t
            
        Returns:
            List of FaceSegment objects spanning [0, 1]
        """
        segments: List[FaceSegment] = []
        
        # Filter to only midspan intersections (not at endpoints)
        midspan = [
            (t, inter, panels) for t, inter, panels in intersections
            if ENDPOINT_TOLERANCE <= t <= (1.0 - ENDPOINT_TOLERANCE)
        ]
        
        # Start boundary - will be used as the first segment's start
        current_start_type = BoundaryType.PANEL_START
        current_start_t = 0.0
        current_start_intersection = None
        current_start_intersecting_panels = []
        
        # Check if there's an intersection at t≈0
        start_intersections = [
            (t, inter, panels) for t, inter, panels in intersections
            if t < ENDPOINT_TOLERANCE
        ]
        if start_intersections:
            t, inter, panels = start_intersections[0]
            current_start_type = FaceSolver._intersection_to_boundary_type(inter)
            current_start_t = t
            current_start_intersection = inter
            current_start_intersecting_panels = panels.copy()
        
        # Build segments from midspan intersections
        for t, inter, panels in midspan:
            boundary_type = FaceSolver._intersection_to_boundary_type(inter)
            
            segments.append(FaceSegment(
                start_type=current_start_type,
                start_t=current_start_t,
                start_intersection=current_start_intersection,
                start_intersecting_panels=current_start_intersecting_panels,
                end_type=boundary_type,
                end_t=t,
                end_intersection=inter,
                end_intersecting_panels=panels.copy()
            ))
            
            # Next segment starts at this intersection
            current_start_type = boundary_type
            current_start_t = t
            current_start_intersection = inter
            current_start_intersecting_panels = panels.copy()
        
        # Final segment to t=1
        end_type = BoundaryType.PANEL_END
        end_t = 1.0
        end_intersection = None
        end_intersecting_panels = []
        
        # Check if there's an intersection at t≈1
        end_intersections = [
            (t, inter, panels) for t, inter, panels in intersections
            if t > (1.0 - ENDPOINT_TOLERANCE)
        ]
        if end_intersections:
            t, inter, panels = end_intersections[0]
            end_type = FaceSolver._intersection_to_boundary_type(inter)
            end_t = t
            end_intersection = inter
            end_intersecting_panels = panels.copy()
        
        segments.append(FaceSegment(
            start_type=current_start_type,
            start_t=current_start_t,
            start_intersection=current_start_intersection,
            start_intersecting_panels=current_start_intersecting_panels,
            end_type=end_type,
            end_t=end_t,
            end_intersection=end_intersection,
            end_intersecting_panels=end_intersecting_panels
        ))
        
        return segments
    
    @staticmethod
    def _intersection_to_boundary_type(intersection: Intersection) -> BoundaryType:
        """Convert intersection type to boundary type."""
        if intersection.type == IntersectionType.CROSSING:
            return BoundaryType.CROSSING
        elif intersection.type == IntersectionType.T_JUNCTION:
            return BoundaryType.T_JUNCTION
        else:  # JOINT
            return BoundaryType.JOINT
    
    @staticmethod
    def _add_endpoint_faces(face_solution: FaceSolution) -> None:
        """
        Create segments for START and END faces.
        
        Args:
            face_solution: The face solution to add segments to (modified in-place)
        """
        panel = face_solution.panel
        
        # START face
        if face_solution.has_start_intersection():
            # Suppressed - no segments
            face_solution.segments[PanelSide.START] = []
        elif panel.thickness <= TOLERANCE:
            # Zero thickness - no endpoint faces
            face_solution.segments[PanelSide.START] = []
        else:
            # Create synthetic segment for START face
            face_solution.segments[PanelSide.START] = [
                FaceSegment(
                    start_type=BoundaryType.PANEL_START,
                    start_t=0.0,
                    end_type=BoundaryType.PANEL_START,
                    end_t=0.0
                )
            ]
        
        # END face
        if face_solution.has_end_intersection():
            # Suppressed - no segments
            face_solution.segments[PanelSide.END] = []
        elif panel.thickness <= TOLERANCE:
            # Zero thickness - no endpoint faces
            face_solution.segments[PanelSide.END] = []
        else:
            # Create synthetic segment for END face
            face_solution.segments[PanelSide.END] = [
                FaceSegment(
                    start_type=BoundaryType.PANEL_END,
                    start_t=1.0,
                    end_type=BoundaryType.PANEL_END,
                    end_t=1.0
                )
            ]
    
    @staticmethod
    def _apply_zero_thickness_suppression(face_solution: FaceSolution) -> None:
        """
        Suppress faces for zero-thickness panels.
        
        Zero-thickness panels need special handling:
        - Edge faces (TOP/BOTTOM) collapse to lines - suppress them
        - Thickness faces (FRONT/BACK) are coplanar - only show one
        
        For thickness faces, the rules are:
        - If the face is split by a crossing, show it (the split makes it unique)
        - If at a joint, show only the "visible" face based on winding
        - If standalone (no joints, no splits), suppress all faces
        
        Args:
            face_solution: The face solution to modify in-place
        """
        panel = face_solution.panel
        
        # Suppress edge faces (they collapse to lines)
        for face in PanelSide.edge_faces():
            face_solution.segments[face] = []
        
        # For thickness faces, determine visibility
        has_joint_connections = (face_solution.has_start_intersection() or 
                                 face_solution.has_end_intersection())
        
        for face in PanelSide.thickness_faces():
            is_split = face_solution.is_split(face)
            
            if is_split:
                # Split face - show it (the split makes it unique)
                pass
            elif has_joint_connections:
                # Unsplit face at a joint: pick one face based on winding
                visible_face = panel.get_visible_thickness_face()
                if face != visible_face:
                    face_solution.segments[face] = []
            else:
                # No split and no joint - suppress this face
                face_solution.segments[face] = []
