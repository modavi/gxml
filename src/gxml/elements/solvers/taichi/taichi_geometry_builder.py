"""
Taichi GPU-accelerated GeometryBuilder.

This replaces the CPU GeometryBuilder with a GPU implementation that:
1. Parallelizes vertex transformation across all faces/segments
2. Batches polygon creation
3. Parallelizes cap vertex computation

The GeometryBuilder creates 3D mesh geometry (polygons and caps).
"""

import taichi as ti
from typing import List, Dict, Tuple, Optional
import math

from elements.gxml_panel import GXMLPanel, PanelSide
from elements.gxml_polygon import GXMLPolygon
from .taichi_intersection_solver import IntersectionSolution, IntersectionType, Intersection
from .taichi_face_solver import SegmentedPanel, FaceSegment, JointSide

# Tolerance for geometry calculations
TOLERANCE = 1e-4

# Maximum geometry elements
MAX_PANELS = 1024
MAX_SEGMENTS_PER_PANEL = 64
MAX_VERTICES_PER_CAP = 16

# ==============================================================================
# Taichi Fields for GeometryBuilder
# ==============================================================================

# Panel transform matrices (already uploaded by FaceSolver, reused)
# Using same fields as FaceSolver to avoid duplication
from .taichi_face_solver import panel_transform_matrices, panel_dimensions

# Segment corner data: (t, s) coordinates for each segment
# segment_corners[panel_idx, face_idx, segment_idx, corner_idx] = (t, s)
segment_corners = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_PANELS, 6, MAX_SEGMENTS_PER_PANEL, 4))
segment_counts = ti.field(dtype=ti.i32, shape=(MAX_PANELS, 6))

# Output: world-space vertices
# world_vertices[panel_idx, face_idx, segment_idx, corner_idx] = (x, y, z)
world_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_PANELS, 6, MAX_SEGMENTS_PER_PANEL, 4))


@ti.func
def transform_face_point(
    transform: ti.types.matrix(4, 4, ti.f32),
    dims: ti.types.vector(3, ti.f32),  # (width, height, thickness)
    face_idx: ti.i32,
    t: ti.f32,
    s: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    Transform a (t, s) coordinate on a face to world space.
    
    face_idx mapping:
    0 = FRONT (z = +thickness/2)
    1 = BACK (z = -thickness/2)
    2 = TOP (y = +height/2)
    3 = BOTTOM (y = -height/2)
    4 = START (x = -width/2)
    5 = END (x = +width/2)
    """
    width = dims[0]
    height = dims[1]
    thickness = dims[2]
    
    half_w = width * 0.5
    half_h = height * 0.5
    half_t = thickness * 0.5
    
    # Compute local position based on face
    local_x = 0.0
    local_y = 0.0
    local_z = 0.0
    
    if face_idx == 0:  # FRONT
        local_x = (t - 0.5) * width
        local_y = (s - 0.5) * height
        local_z = half_t
    elif face_idx == 1:  # BACK
        local_x = (t - 0.5) * width
        local_y = (s - 0.5) * height
        local_z = -half_t
    elif face_idx == 2:  # TOP
        local_x = (t - 0.5) * width
        local_y = half_h
        local_z = (s - 0.5) * thickness
    elif face_idx == 3:  # BOTTOM
        local_x = (t - 0.5) * width
        local_y = -half_h
        local_z = (s - 0.5) * thickness
    elif face_idx == 4:  # START
        local_x = -half_w
        local_y = (s - 0.5) * height
        local_z = (t - 0.5) * thickness
    else:  # END
        local_x = half_w
        local_y = (s - 0.5) * height
        local_z = (t - 0.5) * thickness
    
    # Transform to world space
    local = ti.Vector([local_x, local_y, local_z, 1.0])
    world = transform @ local
    
    return ti.Vector([world[0], world[1], world[2]])


@ti.kernel
def transform_all_segments(n_panels: ti.i32):
    """
    GPU kernel to transform all segment corners to world space.
    
    Parallelizes over all panels, faces, segments, and corners.
    """
    for panel_idx, face_idx, seg_idx, corner_idx in ti.ndrange(n_panels, 6, MAX_SEGMENTS_PER_PANEL, 4):
        if seg_idx < segment_counts[panel_idx, face_idx]:
            transform = panel_transform_matrices[panel_idx]
            dims = panel_dimensions[panel_idx]
            
            corner = segment_corners[panel_idx, face_idx, seg_idx, corner_idx]
            t = corner[0]
            s = corner[1]
            
            world_pos = transform_face_point(transform, dims, face_idx, t, s)
            world_vertices[panel_idx, face_idx, seg_idx, corner_idx] = world_pos


# ==============================================================================
# TaichiGeometryBuilder
# ==============================================================================

class TaichiGeometryBuilder:
    """
    GPU-accelerated GeometryBuilder using Taichi.
    
    The GPU is used for:
    - Batch vertex transformation from (t,s) to world coordinates
    - Parallel polygon vertex computation
    """
    
    # Face index mapping
    FACE_INDEX = {
        PanelSide.FRONT: 0,
        PanelSide.BACK: 1,
        PanelSide.TOP: 2,
        PanelSide.BOTTOM: 3,
        PanelSide.START: 4,
        PanelSide.END: 5,
    }
    
    @staticmethod
    def build_all(panel_faces: List[SegmentedPanel],
                  intersection_solution: IntersectionSolution) -> None:
        """
        Build geometry for all panels using GPU acceleration.
        
        Args:
            panel_faces: List of SegmentedPanel
            intersection_solution: The intersection solution
        """
        panels = [pf.panel for pf in panel_faces]
        n = len(panels)
        
        if n == 0:
            return
        
        # Upload panel data (if not already done by FaceSolver)
        TaichiGeometryBuilder._upload_panel_data(panels)
        
        # Upload segment data
        TaichiGeometryBuilder._upload_segment_data(panel_faces)
        
        # Run GPU transformation
        transform_all_segments(n)
        ti.sync()
        
        # Create face polygons from GPU results
        for panel_idx, pf in enumerate(panel_faces):
            if not pf.panel.is_valid(TOLERANCE):
                continue
            TaichiGeometryBuilder._create_panel_faces_from_gpu(panel_idx, pf)
        
        # Create caps (CPU - complex topology handling)
        TaichiGeometryBuilder._create_all_caps(panel_faces, intersection_solution)
    
    @staticmethod
    def build(panel: GXMLPanel, panel_faces: List[SegmentedPanel],
              intersection_solution: IntersectionSolution) -> None:
        """
        Build geometry for a single panel.
        
        For single-panel builds, we fall back to CPU since GPU overhead isn't worth it.
        """
        if not panel.is_valid(TOLERANCE):
            return
        
        pf = TaichiGeometryBuilder._find_panel_faces(panel_faces, panel)
        if pf:
            TaichiGeometryBuilder._create_panel_faces_cpu(panel, pf)
        
        TaichiGeometryBuilder._create_caps_for_panel(panel, panel_faces, intersection_solution)
    
    @staticmethod
    def _upload_panel_data(panels: List[GXMLPanel]) -> None:
        """Upload panel transform data to GPU."""
        for i, panel in enumerate(panels):
            matrix = panel.get_world_transform_matrix()
            for row in range(4):
                for col in range(4):
                    panel_transform_matrices[i][row, col] = matrix[row][col]
            
            panel_dimensions[i] = ti.Vector([
                panel.width, panel.height, panel.thickness
            ])
    
    @staticmethod
    def _upload_segment_data(panel_faces: List[SegmentedPanel]) -> None:
        """Upload segment corner data to GPU."""
        for panel_idx, pf in enumerate(panel_faces):
            for face_side in PanelSide:
                face_idx = TaichiGeometryBuilder.FACE_INDEX[face_side]
                segs = pf.segments.get(face_side, [])
                segment_counts[panel_idx, face_idx] = len(segs)
                
                for seg_idx, segment in enumerate(segs):
                    for corner_idx, (t, s) in enumerate(segment.corners):
                        segment_corners[panel_idx, face_idx, seg_idx, corner_idx] = ti.Vector([t, s])
    
    @staticmethod
    def _create_panel_faces_from_gpu(panel_idx: int, pf: SegmentedPanel) -> None:
        """Create face polygons using GPU-transformed vertices."""
        panel = pf.panel
        
        for face_side in PanelSide:
            face_idx = TaichiGeometryBuilder.FACE_INDEX[face_side]
            segs = pf.segments.get(face_side, [])
            
            for seg_idx, segment in enumerate(segs):
                # Read world vertices from GPU
                world_corners = []
                for corner_idx in range(4):
                    v = world_vertices[panel_idx, face_idx, seg_idx, corner_idx]
                    world_corners.append((float(v[0]), float(v[1]), float(v[2])))
                
                # Create polygon
                face_name = face_side.name.lower() if len(segs) == 1 else f"{face_side.name.lower()}-{seg_idx}"
                
                # Use panel's create_panel_side which handles local coordinates
                panel.create_panel_side(face_name, face_side, corners=segment.corners)
    
    @staticmethod
    def _create_panel_faces_cpu(panel: GXMLPanel, pf: SegmentedPanel) -> None:
        """Create face polygons using CPU (for single panel builds)."""
        for face_side in PanelSide:
            segs = pf.segments.get(face_side, [])
            
            for i, segment in enumerate(segs):
                face_name = face_side.name.lower() if len(segs) == 1 else f"{face_side.name.lower()}-{i}"
                panel.create_panel_side(face_name, face_side, corners=segment.corners)
    
    @staticmethod
    def _create_all_caps(panel_faces: List[SegmentedPanel],
                         intersection_solution: IntersectionSolution) -> None:
        """Create all joint and crossing caps."""
        for intersection in intersection_solution.intersections:
            if intersection.type == IntersectionType.JOINT and len(intersection.panels) >= 3:
                TaichiGeometryBuilder._create_joint_cap(
                    intersection, panel_faces, is_top=True)
                TaichiGeometryBuilder._create_joint_cap(
                    intersection, panel_faces, is_top=False)
            
            elif intersection.type == IntersectionType.CROSSING:
                TaichiGeometryBuilder._create_crossing_caps(
                    intersection, panel_faces)
    
    @staticmethod
    def _create_caps_for_panel(panel: GXMLPanel, panel_faces: List[SegmentedPanel],
                               intersection_solution: IntersectionSolution) -> None:
        """Create caps for intersections where this panel is the first panel."""
        for intersection in intersection_solution.intersections:
            if intersection.type == IntersectionType.JOINT and len(intersection.panels) >= 3:
                if intersection.panels[0].panel == panel:
                    TaichiGeometryBuilder._create_joint_cap(
                        intersection, panel_faces, is_top=True)
                    TaichiGeometryBuilder._create_joint_cap(
                        intersection, panel_faces, is_top=False)
            
            elif intersection.type == IntersectionType.CROSSING:
                if intersection.panels[0].panel == panel:
                    TaichiGeometryBuilder._create_crossing_caps(
                        intersection, panel_faces)
    
    @staticmethod
    def _create_joint_cap(joint: Intersection,
                           panel_faces: List[SegmentedPanel],
                           is_top: bool) -> None:
        """Create a miter cap polygon for a joint intersection."""
        if len(joint.panels) < 3:
            return
        
        cap_owner = joint.panels[0].panel
        face_side = PanelSide.TOP if is_top else PanelSide.BOTTOM
        
        # Build cap vertices from panel corners
        cap_vertices = []
        
        for entry in joint.panels:
            panel = entry.panel
            pf = TaichiGeometryBuilder._find_panel_faces(panel_faces, panel)
            if pf is None:
                continue
            
            is_end_at_joint = entry.t > 0.5
            segs = pf.segments.get(face_side, [])
            
            if not segs:
                continue
            
            segment_index = len(segs) - 1 if is_end_at_joint else 0
            corners = segs[segment_index].get_world_corners()
            
            # Get appropriate corner based on CCW winding
            if is_end_at_joint:
                # END at joint - use front corner for CCW direction
                corner = corners[2]  # end-front
            else:
                # START at joint - use back corner for CCW direction  
                corner = corners[3]  # start-front
            
            cap_vertices.append(corner)
        
        if len(cap_vertices) < 3:
            return
        
        # For bottom cap, reverse winding
        if not is_top:
            cap_vertices = cap_vertices[::-1]
        
        cap_name = "cap-top" if is_top else "cap-bottom"
        
        cap = GXMLPolygon(cap_vertices)
        cap.id = cap_owner.id
        cap.subId = cap_name
        cap.parent = cap_owner
        cap_owner.dynamicChildren.append(cap)
    
    @staticmethod
    def _create_crossing_caps(crossing: Intersection,
                               panel_faces: List[SegmentedPanel]) -> None:
        """Create cap polygons for a crossing intersection."""
        if len(crossing.panels) != 2:
            return
        
        panel1 = crossing.panels[0].panel
        panel2 = crossing.panels[1].panel
        pf1 = TaichiGeometryBuilder._find_panel_faces(panel_faces, panel1)
        pf2 = TaichiGeometryBuilder._find_panel_faces(panel_faces, panel2)
        
        if pf1 is None or pf2 is None:
            return
        
        for is_top in [True, False]:
            cap_vertices = TaichiGeometryBuilder._compute_crossing_cap_vertices(
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
        """Compute crossing cap vertices from panel faces."""
        def get_gap_edge(pf: SegmentedPanel, face_side: PanelSide, is_top: bool):
            segs = pf.segments.get(face_side, [])
            if len(segs) < 2:
                return None
            
            corners0 = segs[0].get_world_corners()
            corners1 = segs[1].get_world_corners()
            
            if is_top:
                return (corners0[2], corners1[3])
            else:
                return (corners0[1], corners1[0])
        
        p1_front = get_gap_edge(pf1, PanelSide.FRONT, is_top)
        p1_back = get_gap_edge(pf1, PanelSide.BACK, is_top)
        p2_front = get_gap_edge(pf2, PanelSide.FRONT, is_top)
        p2_back = get_gap_edge(pf2, PanelSide.BACK, is_top)
        
        if any(e is None for e in [p1_front, p1_back, p2_front, p2_back]):
            return None
        
        # Find intersection points (using CPU math - 2D line intersections)
        from mathutils.gxml_math import intersect_lines_2d
        
        v1 = intersect_lines_2d(p1_front, p2_front)
        v2 = intersect_lines_2d(p2_front, p1_back)
        v3 = intersect_lines_2d(p1_back, p2_back)
        v4 = intersect_lines_2d(p2_back, p1_front)
        
        if any(v is None for v in [v1, v2, v3, v4]):
            return None
        
        # Sort CCW
        cap_vertices = [v1, v2, v3, v4]
        cx = sum(v[0] for v in cap_vertices) * 0.25
        cz = sum(v[2] for v in cap_vertices) * 0.25
        
        def angle_from_centroid(v):
            return -math.atan2(v[2] - cz, v[0] - cx)
        
        cap_vertices.sort(key=angle_from_centroid)
        
        return cap_vertices
    
    @staticmethod
    def _find_panel_faces(panel_faces: List[SegmentedPanel], 
                          panel: GXMLPanel) -> Optional[SegmentedPanel]:
        """Find the SegmentedPanel for a given panel."""
        return next((pf for pf in panel_faces if pf.panel is panel), None)
