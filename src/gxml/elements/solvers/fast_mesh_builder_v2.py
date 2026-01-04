"""
Fast mesh builder v2 - uses FaceSolver for corner computation, skips GeometryBuilder.

This version takes a hybrid approach:
1. Use FaceSolver to compute face segments (it's already optimized)
2. Skip GXMLQuad creation in GeometryBuilder
3. Write directly to vertex buffers from FaceSegments

This avoids duplicating FaceSolver's complex logic while eliminating object creation overhead.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

from elements.gxml_panel import GXMLPanel, PanelSide
from .gxml_intersection_solver import IntersectionSolution, Intersection, IntersectionType
from .gxml_face_solver import FaceSolver, SegmentedPanel, FaceSegment, JointSide
from mathutils.quad_interpolator import batch_bilinear_transform
from mathutils.gxml_math import intersect_lines_2d


TOLERANCE = 1e-4


@dataclass
class MeshData:
    """Raw mesh data for direct rendering."""
    vertices: np.ndarray  # Shape: (N, 3) float32 - world positions
    indices: np.ndarray   # Shape: (M,) uint32 - triangle indices
    panel_ids: List[str]  # One per quad (for hit testing)
    face_sides: List[int] # One per quad (PanelSide enum value)
    
    @property
    def quad_count(self) -> int:
        return len(self.panel_ids)
    
    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3
    
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)


class FastMeshBuilderV2:
    """
    Fast mesh builder that reuses FaceSolver but skips GXMLQuad creation.
    
    This is a hybrid approach:
    - FaceSolver computes face segments with correct gaps/trims (already fast)
    - We skip GeometryBuilder's GXMLQuad creation overhead
    - Write directly to numpy arrays
    """
    
    @staticmethod
    def build(intersection_solution: IntersectionSolution, profile: bool = False) -> MeshData:
        """
        Build mesh data using FaceSolver + direct buffer writes.
        """
        import time
        
        t0 = time.perf_counter()
        # Use FaceSolver for corner computation
        panel_faces = FaceSolver.solve(intersection_solution)
        t_face = time.perf_counter() - t0
        
        panels = intersection_solution.panels
        
        # Pre-allocate buffers
        estimated_quads = len(panels) * 6 * 2 + len(intersection_solution.intersections) * 2
        estimated_verts = estimated_quads * 4
        
        vertices = np.empty((estimated_verts, 3), dtype=np.float32)
        indices = []
        panel_ids = []
        face_sides = []
        
        vert_idx = 0
        t_transform = 0.0
        t_write = 0.0
        
        # Convert FaceSegments directly to vertices (skip GXMLQuad creation)
        for pf in panel_faces:
            panel = pf.panel
            if not panel.is_valid(TOLERANCE):
                continue
            
            # Pre-fetch panel transform data once
            quad_points = panel.quad_interpolator.get_quad_points()
            matrix = panel.transform.transformationMatrix
            half_thickness = panel.thickness / 2
            thickness = panel.thickness
            
            for face_side in PanelSide:
                segments = pf.segments.get(face_side, [])
                
                for segment in segments:
                    corners = segment.corners
                    
                    t0 = time.perf_counter()
                    # Build transform input from corners
                    points_for_transform = FastMeshBuilderV2._corners_to_transform_input(
                        corners, face_side, half_thickness, thickness
                    )
                    
                    # Batch transform
                    world_points = batch_bilinear_transform(
                        points_for_transform, quad_points, matrix
                    )
                    t_transform += time.perf_counter() - t0
                    
                    t0 = time.perf_counter()
                    # Ensure buffer space
                    if vert_idx + 4 > len(vertices):
                        new_vertices = np.empty((len(vertices) * 2, 3), dtype=np.float32)
                        new_vertices[:vert_idx] = vertices[:vert_idx]
                        vertices = new_vertices
                    
                    # Write to buffer
                    for i, pt in enumerate(world_points):
                        vertices[vert_idx + i] = pt
                    
                    # Add indices
                    indices.extend([
                        vert_idx, vert_idx+1, vert_idx+2,
                        vert_idx, vert_idx+2, vert_idx+3
                    ])
                    
                    panel_ids.append(panel.id)
                    face_sides.append(int(face_side))
                    vert_idx += 4
                    t_write += time.perf_counter() - t0
        
        # Generate caps for 3+ panel joints and crossings
        t0 = time.perf_counter()
        pf_lookup = {pf.panel: pf for pf in panel_faces}
        
        for intersection in intersection_solution.intersections:
            if intersection.type == IntersectionType.JOINT and len(intersection.panels) >= 3:
                # Generate top and bottom joint caps
                for is_top in [True, False]:
                    cap_verts = FastMeshBuilderV2._compute_joint_cap_vertices(
                        intersection, pf_lookup, is_top
                    )
                    if cap_verts and len(cap_verts) >= 3:
                        # Ensure buffer space
                        n_verts = len(cap_verts)
                        if vert_idx + n_verts > len(vertices):
                            new_vertices = np.empty((len(vertices) * 2, 3), dtype=np.float32)
                            new_vertices[:vert_idx] = vertices[:vert_idx]
                            vertices = new_vertices
                        
                        # Write vertices
                        for i, v in enumerate(cap_verts):
                            vertices[vert_idx + i] = v
                        
                        # Fan triangulation for n-gon caps
                        for i in range(1, n_verts - 1):
                            indices.extend([vert_idx, vert_idx + i, vert_idx + i + 1])
                        
                        panel_ids.append(intersection.panels[0].panel.id)
                        face_sides.append(int(PanelSide.TOP if is_top else PanelSide.BOTTOM))
                        vert_idx += n_verts
            
            elif intersection.type == IntersectionType.CROSSING and len(intersection.panels) == 2:
                # Generate top and bottom crossing caps
                pf1 = pf_lookup.get(intersection.panels[0].panel)
                pf2 = pf_lookup.get(intersection.panels[1].panel)
                
                if pf1 and pf2:
                    for is_top in [True, False]:
                        cap_verts = FastMeshBuilderV2._compute_crossing_cap_vertices(
                            pf1, pf2, is_top
                        )
                        if cap_verts and len(cap_verts) >= 3:
                            # Ensure buffer space
                            n_verts = len(cap_verts)
                            if vert_idx + n_verts > len(vertices):
                                new_vertices = np.empty((len(vertices) * 2, 3), dtype=np.float32)
                                new_vertices[:vert_idx] = vertices[:vert_idx]
                                vertices = new_vertices
                            
                            # Write vertices
                            for i, v in enumerate(cap_verts):
                                vertices[vert_idx + i] = v
                            
                            # Quad indices (2 triangles)
                            indices.extend([vert_idx, vert_idx + 1, vert_idx + 2,
                                          vert_idx, vert_idx + 2, vert_idx + 3])
                            
                            panel_ids.append(intersection.panels[0].panel.id)
                            face_sides.append(int(PanelSide.TOP if is_top else PanelSide.BOTTOM))
                            vert_idx += n_verts
        
        t_caps = time.perf_counter() - t0
        
        if profile:
            total = t_face + t_transform + t_write + t_caps
            print(f"\n  FastMeshBuilderV2 breakdown:")
            print(f"    FaceSolver.solve:      {t_face*1000:.1f}ms ({t_face/total*100:.0f}%)")
            print(f"    batch_transforms:      {t_transform*1000:.1f}ms ({t_transform/total*100:.0f}%)")
            print(f"    buffer writes:         {t_write*1000:.1f}ms ({t_write/total*100:.0f}%)")
            print(f"    cap generation:        {t_caps*1000:.1f}ms ({t_caps/total*100:.0f}%)")
        
        return MeshData(
            vertices=vertices[:vert_idx].copy(),
            indices=np.array(indices, dtype=np.uint32),
            panel_ids=panel_ids,
            face_sides=face_sides
        )
    
    @staticmethod
    def _corners_to_transform_input(corners: List[Tuple[float, float]], 
                                     face_side: PanelSide,
                                     half_thickness: float,
                                     thickness: float) -> List[Tuple[float, float, float]]:
        """
        Convert (t, s) corners to (t, s, z_offset) for batch transform.
        """
        result = []
        
        if face_side == PanelSide.FRONT:
            for t, s in corners:
                result.append((t, s, half_thickness))
        elif face_side == PanelSide.BACK:
            for t, s in corners:
                result.append((t, s, -half_thickness))
        elif face_side == PanelSide.TOP:
            for t, s in corners:
                z = -half_thickness + s * thickness
                result.append((t, 1.0, z))
        elif face_side == PanelSide.BOTTOM:
            for t, s in corners:
                z = -half_thickness + s * thickness
                result.append((t, 0.0, z))
        elif face_side == PanelSide.START:
            for t, s in corners:
                z = -half_thickness + t * thickness
                result.append((0.0, s, z))
        elif face_side == PanelSide.END:
            for t, s in corners:
                z = -half_thickness + t * thickness
                result.append((1.0, s, z))
        else:
            for t, s in corners:
                result.append((t, s, 0.0))
        
        return result

    @staticmethod
    def _compute_joint_cap_vertices(joint: Intersection, 
                                     pf_lookup: Dict[GXMLPanel, SegmentedPanel],
                                     is_top: bool) -> Optional[List[Tuple[float, float, float]]]:
        """
        Compute vertices for a joint cap (3+ panel intersection).
        
        Returns list of world-space vertices in CCW order (for top) or CW (for bottom).
        """
        if len(joint.panels) < 3:
            return None
        
        face_side = PanelSide.TOP if is_top else PanelSide.BOTTOM
        
        # First pass: determine which panels have split faces
        panels_to_skip = set()
        for entry in joint.panels:
            pf = pf_lookup.get(entry.panel)
            if pf and len(pf.segments.get(face_side, [])) > 1:
                panels_to_skip.add(entry.panel)
        
        # Second pass: build cap vertices
        cap_vertices = []
        num_panels = len(joint.panels)
        
        for i, entry in enumerate(joint.panels):
            panel = entry.panel
            pf = pf_lookup.get(panel)
            
            if panel in panels_to_skip:
                # Add NEXT panel's CW corner
                next_entry = joint.panels[(i + 1) % num_panels]
                next_pf = pf_lookup.get(next_entry.panel)
                if next_entry.panel not in panels_to_skip and next_pf:
                    next_is_end = next_entry.t > 0.5
                    next_segs = next_pf.segments.get(face_side, [])
                    next_seg_idx = len(next_segs) - 1 if next_is_end else 0
                    if next_seg_idx < len(next_segs):
                        corners = next_segs[next_seg_idx].get_world_corners()
                        cw_face = FaceSolver._get_outward_face(next_entry, JointSide.CW)
                        if next_is_end:
                            corner = corners[1] if cw_face == PanelSide.BACK else corners[2]
                        else:
                            corner = corners[0] if cw_face == PanelSide.BACK else corners[3]
                        cap_vertices.append(corner)
                continue
            
            if pf is None:
                continue
            
            is_end_at_joint = entry.t > 0.5
            segs = pf.segments.get(face_side, [])
            seg_idx = len(segs) - 1 if is_end_at_joint else 0
            
            if seg_idx >= len(segs):
                continue
            
            corners = segs[seg_idx].get_world_corners()
            
            # Corner ordering: 0=start-back, 1=end-back, 2=end-front, 3=start-front
            if is_end_at_joint:
                back_corner = corners[1]
                front_corner = corners[2]
            else:
                back_corner = corners[0]
                front_corner = corners[3]
            
            ccw_face = FaceSolver._get_outward_face(entry, JointSide.CCW)
            cap_vertices.append(front_corner if ccw_face == PanelSide.FRONT else back_corner)
        
        if len(cap_vertices) < 3:
            return None
        
        # Reverse for bottom cap to maintain outward normal
        if not is_top:
            cap_vertices = cap_vertices[::-1]
        
        return cap_vertices

    @staticmethod
    def _compute_crossing_cap_vertices(pf1: SegmentedPanel, pf2: SegmentedPanel,
                                        is_top: bool) -> Optional[List[Tuple[float, float, float]]]:
        """
        Compute vertices for a crossing cap (2-panel intersection).
        
        Returns list of 4 world-space vertices forming a quad.
        """
        def get_gap_edge(pf: SegmentedPanel, face_side: PanelSide, is_top: bool):
            """Get gap edge between split segments."""
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
        
        # Find 4 cap vertices as intersections of gap edges
        v1 = intersect_lines_2d(p1_front, p2_front)
        v2 = intersect_lines_2d(p2_front, p1_back)
        v3 = intersect_lines_2d(p1_back, p2_back)
        v4 = intersect_lines_2d(p2_back, p1_front)
        
        if any(v is None for v in [v1, v2, v3, v4]):
            return None
        
        # Sort vertices in CCW order around centroid
        cap_vertices = [v1, v2, v3, v4]
        cx = (v1[0] + v2[0] + v3[0] + v4[0]) * 0.25
        cz = (v1[2] + v2[2] + v3[2] + v4[2]) * 0.25
        
        def angle_from_centroid(v):
            return -math.atan2(v[2] - cz, v[0] - cx)
        
        cap_vertices.sort(key=angle_from_centroid)
        
        # Reverse for bottom cap
        if not is_top:
            cap_vertices = cap_vertices[::-1]
        
        return cap_vertices
