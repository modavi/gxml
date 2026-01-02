"""
Geometry building logic - creates 3D mesh geometry for GXMLPanel instances.

This module is Stage 3 of the geometry pipeline:
1. IntersectionSolver - finds where panel centerlines meet
2. FaceSolver - determines face segmentation with gap-adjusted bounds
3. GeometryBuilder - creates actual 3D geometry (polygons and caps)

Key responsibilities:
- Create polygon geometry for each face segment
- Create joint caps and crossing caps
"""

from typing import Optional, List, Sequence

from elements.solvers.gxml_intersection_solver import IntersectionSolution, IntersectionType, Intersection
from elements.solvers.gxml_face_solver import FaceSolver, JointSide, SegmentedPanel
from elements.gxml_panel import GXMLPanel, PanelSide
from elements.gxml_polygon import GXMLPolygon
from mathutils.gxml_math import intersect_lines_2d

# Tolerance for geometry calculations
TOLERANCE = 1e-4


def _get_crossing_gap_edge(pf: SegmentedPanel, face_side: PanelSide, 
                           is_top: bool) -> Optional[tuple]:
    """
    Get the gap edge for a FRONT/BACK face at a crossing.
    
    At a crossing, FRONT/BACK faces are split. The gap is between segment[0]'s
    end and segment[1]'s start. We return the edge line for cap creation.
    
    Args:
        pf: The SegmentedPanel
        face_side: Which face (FRONT or BACK)
        is_top: True for top edge (s=1), False for bottom (s=0)
        
    Returns:
        Tuple of (point1, point2) for the gap edge, or None
    """
    segs = pf.segments.get(face_side, [])
    if len(segs) < 2:
        return None  # No crossing gap if not split
    
    # Get corners for both segments
    corners0 = segs[0].get_world_corners()
    corners1 = segs[1].get_world_corners()
    
    # Corner ordering: 0=start-back, 1=end-back, 2=end-front, 3=start-front
    # For FRONT/BACK faces: back=bottom (s=0), front=top (s=1)
    if is_top:
        # Top edge of gap: segment[0]'s end-top to segment[1]'s start-top
        return (corners0[2], corners1[3])
    else:
        # Bottom edge of gap
        return (corners0[1], corners1[0])


class GeometryBuilder:
    """
    Stage 3: Creates 3D mesh geometry from face solver result.
    
    Takes the output of FaceSolver.solve() and creates actual geometry:
    - Polygon vertices for each face segment
    - Joint caps for 3+ panel joints
    - Crossing caps for crossing intersections
    """
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    @staticmethod
    def build_all(panel_faces: List[SegmentedPanel],
                  intersection_solution: IntersectionSolution) -> None:
        """
        Build geometry for all panels.
        
        Args:
            panel_faces: List of SegmentedPanel
            intersection_solution: The intersection solution
        """
        # Create face polygons for all panels
        for pf in panel_faces:
            if not pf.panel.is_valid(TOLERANCE):
                continue
            GeometryBuilder._create_panel_faces(pf.panel, pf)
        
        # Create caps
        GeometryBuilder._create_all_caps(panel_faces, intersection_solution)
    
    @staticmethod
    def build(panel: GXMLPanel, panel_faces: List[SegmentedPanel],
              intersection_solution: IntersectionSolution) -> None:
        """
        Build geometry for a single panel.
        
        Args:
            panel: The panel to build geometry for
            panel_faces: List of SegmentedPanel
            intersection_solution: The intersection solution
        """
        if not panel.is_valid(TOLERANCE):
            return
        
        pf = GeometryBuilder._find_panel_faces(panel_faces, panel)
        if pf:
            GeometryBuilder._create_panel_faces(panel, pf)
        
        # Create caps (only for intersections involving this panel as first panel)
        GeometryBuilder._create_caps_for_panel(panel, panel_faces, intersection_solution)
    
    # -------------------------------------------------------------------------
    # Face creation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _create_panel_faces(panel: GXMLPanel, pf: SegmentedPanel) -> None:
        """
        Create face polygons from panel faces.
        
        Args:
            panel: The panel to create faces for
            pf: The panel's face data
        """
        for face_side in PanelSide:
            segs = pf.segments.get(face_side, [])
            
            for i, segment in enumerate(segs):
                local_corners = segment.corners
                face_name = face_side.name.lower() if len(segs) == 1 else f"{face_side.name.lower()}-{i}"
                panel.create_panel_side(face_name, face_side, corners=local_corners)
    
    # -------------------------------------------------------------------------
    # Cap creation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _create_all_caps(panel_faces: List[SegmentedPanel],
                         intersection_solution: IntersectionSolution) -> None:
        """
        Create all joint and crossing caps.
        
        Args:
            panel_faces: List of SegmentedPanel
            intersection_solution: The intersection solution
        """
        for intersection in intersection_solution.intersections:
            if intersection.type == IntersectionType.JOINT and len(intersection.panels) >= 3:
                GeometryBuilder._create_joint_cap(
                    intersection, panel_faces, is_top=True)
                GeometryBuilder._create_joint_cap(
                    intersection, panel_faces, is_top=False)
            
            elif intersection.type == IntersectionType.CROSSING:
                GeometryBuilder._create_crossing_caps(
                    intersection, panel_faces)
    
    @staticmethod
    def _create_caps_for_panel(panel: GXMLPanel, panel_faces: List[SegmentedPanel],
                               intersection_solution: IntersectionSolution) -> None:
        """
        Create caps for intersections where this panel is the first panel.
        
        Args:
            panel: The panel being built
            panel_faces: List of SegmentedPanel
            intersection_solution: The intersection solution
        """
        for intersection in intersection_solution.intersections:
            if intersection.type == IntersectionType.JOINT and len(intersection.panels) >= 3:
                if intersection.panels[0].panel == panel:
                    GeometryBuilder._create_joint_cap(
                        intersection, panel_faces, is_top=True)
                    GeometryBuilder._create_joint_cap(
                        intersection, panel_faces, is_top=False)
            
            elif intersection.type == IntersectionType.CROSSING:
                if intersection.panels[0].panel == panel:
                    GeometryBuilder._create_crossing_caps(
                        intersection, panel_faces)
    
    @staticmethod
    def _create_joint_cap(joint: Intersection,
                           panel_faces: List[SegmentedPanel],
                           is_top: bool) -> None:
        """
        Create a miter cap polygon for a joint intersection with 3+ panels.
        
        The cap fills the polygon-shaped gap at the TOP or BOTTOM of the joint
        where mitered panel faces meet. Each cap vertex is computed by averaging
        two adjacent panels' corners.
        
        Args:
            joint: The joint intersection (must have 3+ panels)
            panel_faces: List of SegmentedPanel
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
            pf = GeometryBuilder._find_panel_faces(panel_faces, panel)
            if pf and len(pf.segments.get(face_side, [])) > 1:
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
            pf = GeometryBuilder._find_panel_faces(panel_faces, panel)
            
            if panel in panels_to_skip:
                # This panel is skipped - add the NEXT panel's CW corner to fill the gap
                # The next panel's CW face points back toward this skipped panel
                next_entry = joint.panels[(i + 1) % num_panels]
                next_pf = GeometryBuilder._find_panel_faces(panel_faces, next_entry.panel)
                if next_entry.panel not in panels_to_skip and next_pf:
                    next_is_end = next_entry.t > 0.5
                    next_segment_idx = GeometryBuilder._find_endpoint_segment(
                        next_pf, face_side, next_is_end)
                    next_segs = next_pf.segments.get(face_side, [])
                    if next_segment_idx < len(next_segs):
                        next_corners = next_segs[next_segment_idx].get_world_corners()
                        cw_face = FaceSolver._get_outward_face(next_entry, JointSide.CW)
                        if next_is_end:
                            corner = next_corners[1] if cw_face == PanelSide.BACK else next_corners[2]
                        else:
                            corner = next_corners[0] if cw_face == PanelSide.BACK else next_corners[3]
                        cap_vertices.append(corner)
                continue
            
            if pf is None:
                continue
            
            is_end_at_joint = entry.t > 0.5
            segment_index = GeometryBuilder._find_endpoint_segment(
                pf, face_side, is_end_at_joint)
            
            segs = pf.segments.get(face_side, [])
            if segment_index >= len(segs):
                continue
            corners = segs[segment_index].get_world_corners()
            
            # Corner ordering: 0=start-back, 1=end-back, 2=end-front, 3=start-front
            if is_end_at_joint:
                back_corner = corners[1]   # end-back
                front_corner = corners[2]  # end-front
            else:
                back_corner = corners[0]   # start-back
                front_corner = corners[3]  # start-front
            
            # Determine which face is CCW (toward next panel)
            ccw_face = FaceSolver._get_outward_face(entry, JointSide.CCW)
            
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
    def _find_endpoint_segment(pf: SegmentedPanel, face_side: PanelSide, is_end: bool) -> int:
        """
        Find the segment index that contains the panel's START or END.
        
        For split faces, the endpoint we need might not be in segment 0.
        
        Args:
            pf: The panel faces
            face_side: Which face to check
            is_end: True to find END, False to find START
            
        Returns:
            Segment index (0 if unsplit or START, last segment if END on split face)
        """
        segs = pf.segments.get(face_side, [])
        if len(segs) <= 1:
            return 0
        
        # For split faces, START is in first segment, END is in last segment
        return len(segs) - 1 if is_end else 0
    
    @staticmethod
    def _find_panel_faces(panel_faces: List[SegmentedPanel], panel: GXMLPanel) -> Optional[SegmentedPanel]:
        """
        Find the SegmentedPanel for a given panel.
        
        Args:
            panel_faces: List of SegmentedPanel to search
            panel: The panel to find
            
        Returns:
            The SegmentedPanel for the panel, or None if not found
        """
        return next((pf for pf in panel_faces if pf.panel is panel), None)
    
    @staticmethod
    def _create_crossing_caps(crossing: Intersection,
                               panel_faces: List[SegmentedPanel]) -> None:
        """
        Create cap polygons for a crossing intersection.
        
        At a crossing, FRONT/BACK faces are split and we create a cap to fill
        the quadrilateral gap at TOP and BOTTOM.
        
        Args:
            crossing: The crossing intersection
            panel_faces: List of SegmentedPanel
        """
        if len(crossing.panels) != 2:
            return
        
        panel1 = crossing.panels[0].panel
        panel2 = crossing.panels[1].panel
        pf1 = GeometryBuilder._find_panel_faces(panel_faces, panel1)
        pf2 = GeometryBuilder._find_panel_faces(panel_faces, panel2)
        
        if pf1 is None or pf2 is None:
            return
        
        for is_top in [True, False]:
            cap_vertices = GeometryBuilder._compute_crossing_cap_vertices(
                pf1, pf2, is_top
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
    def _compute_crossing_cap_vertices(pf1: SegmentedPanel, pf2: SegmentedPanel,
                                        is_top: bool) -> Optional[List[tuple]]:
        """
        Compute crossing cap vertices from panel faces.
        
        For FRONT/BACK faces that were split at a crossing, we get the gap edges
        from consecutive segments and intersect them to find cap vertices.
        
        Args:
            pf1: First panel's faces
            pf2: Second panel's faces
            is_top: True for TOP cap, False for BOTTOM cap
            
        Returns:
            List of 4 corner vertices as (x,y,z) tuples in CCW winding order, or None
        """
        # Get gap edges for both panels' FRONT and BACK faces
        p1_front = _get_crossing_gap_edge(pf1, PanelSide.FRONT, is_top)
        p1_back = _get_crossing_gap_edge(pf1, PanelSide.BACK, is_top)
        p2_front = _get_crossing_gap_edge(pf2, PanelSide.FRONT, is_top)
        p2_back = _get_crossing_gap_edge(pf2, PanelSide.BACK, is_top)
        
        if any(e is None for e in [p1_front, p1_back, p2_front, p2_back]):
            return None
        
        # Find the 4 cap vertices as intersections of the gap edges
        v1 = intersect_lines_2d(p1_front, p2_front)
        v2 = intersect_lines_2d(p2_front, p1_back)
        v3 = intersect_lines_2d(p1_back, p2_back)
        v4 = intersect_lines_2d(p2_back, p1_front)
        
        if any(v is None for v in [v1, v2, v3, v4]):
            return None
        
        # Sort vertices in CCW order around their centroid (pure Python, no numpy)
        cap_vertices = [v1, v2, v3, v4]
        cx = (v1[0] + v2[0] + v3[0] + v4[0]) * 0.25
        cz = (v1[2] + v2[2] + v3[2] + v4[2]) * 0.25
        
        import math
        def angle_from_centroid(v):
            return -math.atan2(v[2] - cz, v[0] - cx)
        
        cap_vertices.sort(key=angle_from_centroid)
        
        return cap_vertices
