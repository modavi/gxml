"""
Geometry building logic - creates 3D mesh geometry for GXMLPanel instances.

This module is Stage 4 of the geometry pipeline:
1. IntersectionSolver - finds where panel centerlines meet
2. FaceSolver - determines face segmentation from intersection topology
3. BoundsSolver - computes trim/gap adjustments, provides coordinate lookup
4. GeometryBuilder - creates actual 3D geometry (polygons and caps)

Key responsibilities:
- Create polygon geometry for each face segment
- Create joint caps and crossing caps

Uses BoundsSolution to get world-space coordinates for faces.
"""

from typing import Optional, List, Dict, Union

import numpy as np

from .gxml_intersection_solver import IntersectionSolution, IntersectionType, Intersection
from .gxml_bounds_solver import BoundsSolution, BoundsSolver, PanelEndpointTrims, JointSide
from .gxml_face_solver import FaceSolverResult
from elements.gxml_panel import GXMLPanel, PanelSide
from elements.gxml_polygon import GXMLPolygon
from mathutils.gxml_math import intersect_lines_2d

# Tolerance for geometry calculations
TOLERANCE = 1e-4


class GeometryBuilder:
    """
    Stage 4: Creates 3D mesh geometry from bounds solution.
    
    Takes BoundsSolution and creates actual geometry:
    - Polygon vertices for each face segment
    - Joint caps for 3+ panel joints
    - Crossing caps for crossing intersections
    
    Can also accept FaceSolverResult directly for convenience - will internally
    call BoundsSolver.solve() first.
    """
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    @staticmethod
    def build_all(input_result: Union[BoundsSolution, FaceSolverResult]) -> None:
        """
        Build geometry for all panels.
        
        Args:
            input_result: Either a BoundsSolution or FaceSolverResult.
                         If FaceSolverResult, BoundsSolver.solve() is called first.
        """
        # Handle convenience case: FaceSolverResult passed directly
        if isinstance(input_result, FaceSolverResult):
            bounds_result = BoundsSolver.solve(input_result)
        else:
            bounds_result = input_result
        
        solution = bounds_result.intersection_solution
        
        # Create face polygons for all panels
        for panel in solution.panels:
            if not panel.is_valid(TOLERANCE):
                continue
            GeometryBuilder._create_panel_faces(panel, bounds_result)
        
        # Create caps
        GeometryBuilder._create_all_caps(bounds_result)
    
    @staticmethod
    def build(panel: GXMLPanel, input_result: Union[BoundsSolution, FaceSolverResult]) -> None:
        """
        Build geometry for a single panel.
        
        Args:
            panel: The panel to build geometry for
            input_result: Either a BoundsSolution or FaceSolverResult.
                         If FaceSolverResult, BoundsSolver.solve() is called first.
        """
        # Handle convenience case: FaceSolverResult passed directly
        if isinstance(input_result, FaceSolverResult):
            bounds_result = BoundsSolver.solve(input_result)
        else:
            bounds_result = input_result
        
        if not panel.is_valid(TOLERANCE):
            return
        
        # Create faces for specified panel
        GeometryBuilder._create_panel_faces(panel, bounds_result)
        
        # Create caps (only for intersections involving this panel as first panel)
        GeometryBuilder._create_caps_for_panel(panel, bounds_result)
    
    # -------------------------------------------------------------------------
    # Face creation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _create_panel_faces(panel: GXMLPanel, bounds_result: BoundsSolution) -> None:
        """
        Create face polygons from bounds solution.
        
        Args:
            panel: The panel to create faces for
            bounds_result: The bounds solution with coordinates
        """
        for face_side in PanelSide:
            segment_count = bounds_result.get_segment_count(panel, face_side)
            
            for i in range(segment_count):
                seg_bounds = bounds_result.get_segment_bounds(panel, face_side, i)
                if seg_bounds is None:
                    continue
                
                local_corners = seg_bounds.get_corners()
                face_name = face_side.name.lower() if segment_count == 1 else f"{face_side.name.lower()}-{i}"
                panel.create_panel_side(face_name, face_side, corners=local_corners)
    
    # -------------------------------------------------------------------------
    # Cap creation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _create_all_caps(bounds_result: BoundsSolution) -> None:
        """
        Create all joint and crossing caps.
        
        Args:
            bounds_result: The bounds solution with coordinates and trims
        """
        solution = bounds_result.intersection_solution
        
        for intersection in solution.intersections:
            if intersection.type == IntersectionType.JOINT and len(intersection.panels) >= 3:
                GeometryBuilder._create_joint_cap(
                    intersection, bounds_result, is_top=True)
                GeometryBuilder._create_joint_cap(
                    intersection, bounds_result, is_top=False)
            
            elif intersection.type == IntersectionType.CROSSING:
                GeometryBuilder._create_crossing_caps(
                    intersection, bounds_result)
    
    @staticmethod
    def _create_caps_for_panel(panel: GXMLPanel, bounds_result: BoundsSolution) -> None:
        """
        Create caps for intersections where this panel is the first panel.
        
        Args:
            panel: The panel being built
            bounds_result: The bounds solution
        """
        solution = bounds_result.intersection_solution
        
        for intersection in solution.intersections:
            if intersection.type == IntersectionType.JOINT and len(intersection.panels) >= 3:
                if intersection.panels[0].panel == panel:
                    GeometryBuilder._create_joint_cap(
                        intersection, bounds_result, is_top=True)
                    GeometryBuilder._create_joint_cap(
                        intersection, bounds_result, is_top=False)
            
            elif intersection.type == IntersectionType.CROSSING:
                if intersection.panels[0].panel == panel:
                    GeometryBuilder._create_crossing_caps(
                        intersection, bounds_result)
    
    @staticmethod
    def _create_joint_cap(joint: Intersection,
                           bounds_result: BoundsSolution,
                           is_top: bool) -> None:
        """
        Create a miter cap polygon for a joint intersection with 3+ panels.
        
        The cap fills the polygon-shaped gap at the TOP or BOTTOM of the joint
        where mitered panel faces meet. Each cap vertex is computed by averaging
        two adjacent panels' corners.
        
        Args:
            joint: The joint intersection (must have 3+ panels)
            bounds_result: The bounds solution
            is_top: True for TOP cap, False for BOTTOM cap
        """
        if len(joint.panels) < 3:
            return
        
        # Assign cap to first panel in CCW-sorted joint panel list
        cap_owner = joint.panels[0].panel
        
        face_side = PanelSide.TOP if is_top else PanelSide.BOTTOM
                # First pass: determine which panels have split faces and should be skipped
        # A panel is skipped if its endpoint face segment doesn't extend to the true endpoint
        panels_to_skip = set()
        for entry in joint.panels:
            panel = entry.panel
            segment_count = bounds_result.get_segment_count(panel, face_side)
            if segment_count > 1:
                # Split face - panel should be skipped from cap contribution
                # because its corner is at the gap edge, not at the joint
                panels_to_skip.add(panel)
        
        # Second pass: build cap vertices
        # For each panel, add its CCW corner (pointing toward next panel)
        # When a panel is skipped, add the NEXT panel's CW corner to fill the gap
        cap_vertices = []
        num_panels = len(joint.panels)
        
        for i, entry in enumerate(joint.panels):
            panel = entry.panel
            
            if panel in panels_to_skip:
                # This panel is skipped - add the NEXT panel's CW corner to fill the gap
                # The next panel's CW face points back toward this skipped panel
                next_entry = joint.panels[(i + 1) % num_panels]
                if next_entry.panel not in panels_to_skip:
                    next_is_end = next_entry.t > 0.5
                    next_segment = GeometryBuilder._find_endpoint_segment(
                        bounds_result, next_entry.panel, face_side, next_is_end)
                    next_corners = bounds_result.get_face_corners(next_entry.panel, face_side, next_segment)
                    if next_corners is not None:
                        cw_face = BoundsSolver._get_outward_face(next_entry, JointSide.CW)
                        if next_is_end:
                            corner = next_corners[1] if cw_face == PanelSide.BACK else next_corners[2]
                        else:
                            corner = next_corners[0] if cw_face == PanelSide.BACK else next_corners[3]
                        cap_vertices.append(corner)
                continue
            
            trims = bounds_result.get_endpoint_trims(panel)
            if trims is None:
                continue
            
            is_end_at_joint = entry.t > 0.5
            segment_index = GeometryBuilder._find_endpoint_segment(
                bounds_result, panel, face_side, is_end_at_joint)
            
            corners = bounds_result.get_face_corners(panel, face_side, segment_index)
            if corners is None:
                continue
            
            # Corner ordering: 0=start-back, 1=end-back, 2=end-front, 3=start-front
            if is_end_at_joint:
                back_corner = corners[1]   # end-back
                front_corner = corners[2]  # end-front
            else:
                back_corner = corners[0]   # start-back
                front_corner = corners[3]  # start-front
            
            # Determine which face is CCW (toward next panel)
            ccw_face = BoundsSolver._get_outward_face(entry, JointSide.CCW)
            
            # Add the CCW corner
            if ccw_face == PanelSide.FRONT:
                cap_vertices.append(front_corner)
            else:
                cap_vertices.append(back_corner)
        
        if len(cap_vertices) < 3:
            return
        
        # Panels are already sorted in CCW order, so vertices are in CCW order
        # For bottom cap, reverse to maintain outward-facing normals
        if not is_top:
            cap_vertices = cap_vertices[::-1]
        
        cap_name = "cap-top" if is_top else "cap-bottom"
        
        cap = GXMLPolygon(cap_vertices)
        cap.id = cap_owner.id
        cap.subId = cap_name
        cap.parent = cap_owner
        cap_owner.dynamicChildren.append(cap)
    
    @staticmethod
    def _find_endpoint_segment(bounds_result: BoundsSolution, panel: GXMLPanel,
                                face_side: PanelSide, is_end: bool) -> int:
        """
        Find the segment index that contains the panel's START or END.
        
        For split faces, the endpoint we need might not be in segment 0.
        
        Args:
            bounds_result: The bounds solution
            panel: The panel
            face_side: Which face to check
            is_end: True to find END, False to find START
            
        Returns:
            Segment index (0 if unsplit or START, last segment if END on split face)
        """
        segment_count = bounds_result.get_segment_count(panel, face_side)
        if segment_count <= 1:
            return 0
        
        # For split faces, START is in first segment, END is in last segment
        return segment_count - 1 if is_end else 0
    
    @staticmethod
    def _create_crossing_caps(crossing: Intersection,
                               bounds_result: BoundsSolution) -> None:
        """
        Create cap polygons for a crossing intersection.
        
        At a crossing, FRONT/BACK faces are split and we create a cap to fill
        the quadrilateral gap at TOP and BOTTOM.
        
        Args:
            crossing: The crossing intersection
            bounds_result: The bounds solution
        """
        if len(crossing.panels) != 2:
            return
        
        panel1 = crossing.panels[0].panel
        panel2 = crossing.panels[1].panel
        
        for is_top in [True, False]:
            cap_vertices = GeometryBuilder._compute_crossing_cap_vertices(
                panel1, panel2, bounds_result, is_top
            )
            
            if cap_vertices is None or len(cap_vertices) < 3:
                continue
            
            if not is_top:
                cap_vertices = cap_vertices[::-1]
            
            cap_name = f"crossing-cap-{'top' if is_top else 'bottom'}"
            
            cap = GXMLPolygon(cap_vertices)
            cap.id = panel1.id
            cap.subId = cap_name
            cap.parent = panel1
            panel1.dynamicChildren.append(cap)
    
    @staticmethod
    def _compute_crossing_cap_vertices(panel1: GXMLPanel, panel2: GXMLPanel,
                                        bounds_result: BoundsSolution,
                                        is_top: bool) -> Optional[List[np.ndarray]]:
        """
        Compute crossing cap vertices from bounds solution.
        
        For FRONT/BACK faces that were split at a crossing, we get the gap edges
        from consecutive segments and intersect them to find cap vertices.
        
        Args:
            panel1: First panel
            panel2: Second panel
            bounds_result: The bounds solution
            is_top: True for TOP cap, False for BOTTOM cap
            
        Returns:
            List of 4 corner vertices in CCW winding order, or None
        """
        # Get gap edges for both panels' FRONT and BACK faces
        p1_front = bounds_result.get_crossing_gap_edge(panel1, PanelSide.FRONT, is_top)
        p1_back = bounds_result.get_crossing_gap_edge(panel1, PanelSide.BACK, is_top)
        p2_front = bounds_result.get_crossing_gap_edge(panel2, PanelSide.FRONT, is_top)
        p2_back = bounds_result.get_crossing_gap_edge(panel2, PanelSide.BACK, is_top)
        
        if any(e is None for e in [p1_front, p1_back, p2_front, p2_back]):
            return None
        
        # Find the 4 cap vertices as intersections of the gap edges
        v1 = intersect_lines_2d(p1_front, p2_front)
        v2 = intersect_lines_2d(p2_front, p1_back)
        v3 = intersect_lines_2d(p1_back, p2_back)
        v4 = intersect_lines_2d(p2_back, p1_front)
        
        if any(v is None for v in [v1, v2, v3, v4]):
            return None
        
        # Sort vertices in CCW order around their centroid
        cap_vertices = [v1, v2, v3, v4]
        centroid = np.mean(cap_vertices, axis=0)
        
        def angle_from_centroid(v: np.ndarray) -> float:
            dx = v[0] - centroid[0]
            dz = v[2] - centroid[2]
            return -np.arctan2(dz, dx)
        
        cap_vertices.sort(key=angle_from_centroid)
        
        return cap_vertices
