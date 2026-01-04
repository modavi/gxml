"""
Fast mesh builder - combines face solving and geometry building into a single pass.

This module provides a high-performance alternative to the FaceSolver + GeometryBuilder
pipeline by:
1. Eliminating intermediate FaceSegment objects
2. Writing directly to numpy vertex buffers
3. Computing world coordinates inline

The output is a MeshData structure containing raw vertex/index buffers suitable
for direct consumption by Three.js or other 3D renderers.

Performance target: ~5-10x faster than FaceSolver + GeometryBuilder for 200 panels.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Sequence
import numpy as np

from elements.gxml_panel import GXMLPanel, PanelSide
from .gxml_intersection_solver import (
    IntersectionSolution, Intersection, IntersectionType, Region
)
from mathutils.gxml_math import intersect_line_with_plane
from mathutils.quad_interpolator import batch_bilinear_transform

# Tolerance for t-value comparisons
ENDPOINT_TOLERANCE = 0.01
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


class FastMeshBuilder:
    """
    High-performance mesh builder that combines face solving and geometry generation.
    
    Instead of creating intermediate FaceSegment objects, this writes directly to
    numpy arrays. The key insight is that FaceSegment stores (t, s) corners which
    are immediately converted to world coordinates - we skip the intermediate step.
    """
    
    @staticmethod
    def build(intersection_solution: IntersectionSolution, profile: bool = False) -> MeshData:
        """
        Build mesh data for all panels in one pass.
        
        Args:
            intersection_solution: Output from IntersectionSolver
            profile: If True, print timing breakdown
            
        Returns:
            MeshData with vertices, indices, and panel IDs
        """
        import time
        panels = intersection_solution.panels
        
        # Pre-allocate buffers with conservative estimate
        # ~6 faces × 4 verts per panel, ×2 for splits, ×1.5 for caps
        estimated_quads = len(panels) * 6 * 2 + len(intersection_solution.intersections) * 2
        estimated_verts = estimated_quads * 4
        
        vertices = np.empty((estimated_verts, 3), dtype=np.float32)
        indices = []
        panel_ids = []
        face_sides = []
        
        vert_idx = 0
        
        # Timing accumulators
        t_corners = 0.0
        t_transform = 0.0
        t_write = 0.0
        
        # Process each panel
        for panel in panels:
            if not panel.is_valid(TOLERANCE):
                continue
                
            # Get intersection data for this panel
            region = intersection_solution.regions_per_panel.get(panel)
            leaf_bounds = region.get_leaf_bounds() if region else []
            
            # Track endpoint intersections
            start_inter = None
            end_inter = None
            for intersection in intersection_solution.get_intersections_for_panel(panel):
                entry = intersection.get_entry(panel)
                if entry.t < ENDPOINT_TOLERANCE:
                    start_inter = intersection
                if entry.t > (1.0 - ENDPOINT_TOLERANCE):
                    end_inter = intersection
            
            # Build faces for this panel
            for face_side in PanelSide:
                t0 = time.perf_counter()
                corners_list = FastMeshBuilder._compute_face_corners(
                    panel, face_side, leaf_bounds, start_inter, end_inter
                )
                t_corners += time.perf_counter() - t0
                
                for corners in corners_list:
                    if not corners:
                        continue
                    
                    t0 = time.perf_counter()
                    # Convert (t, s) corners directly to world vertices
                    world_verts = FastMeshBuilder._corners_to_world(panel, face_side, corners)
                    t_transform += time.perf_counter() - t0
                    
                    t0 = time.perf_counter()
                    # Ensure we have space in the buffer
                    if vert_idx + 4 > len(vertices):
                        # Expand buffer
                        new_vertices = np.empty((len(vertices) * 2, 3), dtype=np.float32)
                        new_vertices[:vert_idx] = vertices[:vert_idx]
                        vertices = new_vertices
                    
                    # Write to buffer
                    vertices[vert_idx:vert_idx+4] = world_verts
                    
                    # Add triangle indices (2 triangles per quad)
                    indices.extend([
                        vert_idx, vert_idx+1, vert_idx+2,
                        vert_idx, vert_idx+2, vert_idx+3
                    ])
                    
                    panel_ids.append(panel.id)
                    face_sides.append(int(face_side))
                    vert_idx += 4
                    t_write += time.perf_counter() - t0
        
        if profile:
            total = t_corners + t_transform + t_write
            print(f"\n  FastMeshBuilder breakdown:")
            print(f"    _compute_face_corners: {t_corners*1000:.1f}ms ({t_corners/total*100:.0f}%)")
            print(f"    _corners_to_world:     {t_transform*1000:.1f}ms ({t_transform/total*100:.0f}%)")
            print(f"    buffer writes:         {t_write*1000:.1f}ms ({t_write/total*100:.0f}%)")
        
        # TODO: Add cap generation for 3+ panel joints and crossings
        # For now we skip caps to focus on face performance
        
        return MeshData(
            vertices=vertices[:vert_idx].copy(),
            indices=np.array(indices, dtype=np.uint32),
            panel_ids=panel_ids,
            face_sides=face_sides
        )
    
    @staticmethod
    def _corners_to_world(panel: GXMLPanel, face_side: PanelSide, 
                          corners: List[Tuple[float, float]]) -> np.ndarray:
        """
        Convert (t, s) corners to world-space vertices using batched C transform.
        
        Args:
            panel: The panel
            face_side: Which face
            corners: List of 4 (t, s) tuples
            
        Returns:
            (4, 3) numpy array of world positions
        """
        half_thickness = panel.thickness / 2
        
        # Build points_for_transform list: (t, s, z_offset) tuples
        points_for_transform = []
        for t, s in corners:
            # Map (t, s) to local coordinates based on face type
            if face_side == PanelSide.FRONT:
                points_for_transform.append((t, s, half_thickness))
            elif face_side == PanelSide.BACK:
                points_for_transform.append((t, s, -half_thickness))
            elif face_side == PanelSide.TOP:
                z = -half_thickness + s * panel.thickness
                points_for_transform.append((t, 1.0, z))
            elif face_side == PanelSide.BOTTOM:
                z = -half_thickness + s * panel.thickness
                points_for_transform.append((t, 0.0, z))
            elif face_side == PanelSide.START:
                z = -half_thickness + t * panel.thickness
                points_for_transform.append((0.0, s, z))
            elif face_side == PanelSide.END:
                z = -half_thickness + t * panel.thickness
                points_for_transform.append((1.0, s, z))
            else:
                points_for_transform.append((t, s, 0.0))
        
        # Batch transform using C extension
        world_points = batch_bilinear_transform(
            points_for_transform,
            panel.quad_interpolator.get_quad_points(),
            panel.transform.transformationMatrix
        )
        
        return np.array(world_points, dtype=np.float32)
    
    @staticmethod
    def _compute_face_corners(panel: GXMLPanel, face_side: PanelSide,
                               leaf_bounds: List[Region.LeafBounds],
                               start_inter: Optional[Intersection],
                               end_inter: Optional[Intersection]) -> List[List[Tuple[float, float]]]:
        """
        Compute all corner sets for a face (may be split into multiple segments).
        
        This combines the logic from FaceSolver's segment building with endpoint
        trim application, producing corner lists directly without creating FaceSegment objects.
        
        Args:
            panel: The panel
            face_side: Which face
            leaf_bounds: Leaf regions from intersection solver
            start_inter: Intersection at panel start, if any
            end_inter: Intersection at panel end, if any
            
        Returns:
            List of corner lists, each with 4 (t, s) tuples
        """
        # Handle START/END faces (caps at panel endpoints)
        if face_side == PanelSide.START:
            if start_inter is not None or panel.thickness <= TOLERANCE:
                return []  # Suppressed
            return [[(0.0, 0.0), (0.0, 0.0), (0.0, 1.0), (0.0, 1.0)]]
        
        if face_side == PanelSide.END:
            if end_inter is not None or panel.thickness <= TOLERANCE:
                return []  # Suppressed
            return [[(1.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 1.0)]]
        
        # Handle zero-thickness panels
        if panel.thickness <= TOLERANCE:
            # Suppress edge faces (TOP/BOTTOM)
            if face_side in PanelSide.edge_faces():
                return []
            # For thickness faces, check visibility
            has_joint = start_inter is not None or end_inter is not None
            if has_joint:
                visible_face = panel.get_visible_thickness_face()
                if face_side != visible_face:
                    return []
            else:
                return []  # No joints, suppress thickness faces
        
        # Build segments for FRONT/BACK/TOP/BOTTOM
        if face_side in (PanelSide.FRONT, PanelSide.BACK):
            corners_list = FastMeshBuilder._build_thickness_face_corners(
                panel, face_side, leaf_bounds
            )
        else:  # TOP/BOTTOM
            corners_list = FastMeshBuilder._build_edge_face_corners(
                panel, face_side, leaf_bounds
            )
        
        # Apply endpoint trims
        if corners_list:
            FastMeshBuilder._apply_endpoint_trims(
                panel, face_side, corners_list, start_inter, end_inter
            )
        
        return corners_list
    
    @staticmethod
    def _build_thickness_face_corners(panel: GXMLPanel, face_side: PanelSide,
                                       leaf_bounds: List[Region.LeafBounds]) -> List[List[Tuple[float, float]]]:
        """
        Build corners for FRONT/BACK faces from leaf regions.
        
        Creates one set of corners per leaf region, with gap adjustments at boundaries.
        """
        if not leaf_bounds:
            # No splits - single full face
            return [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]
        
        # Collect boundaries with their intersections (if they affect this face)
        t_boundaries: Dict[float, Optional[Tuple[Intersection, GXMLPanel]]] = {0.0: None, 1.0: None}
        s_boundaries: Dict[float, Optional[Tuple[Intersection, GXMLPanel]]] = {0.0: None, 1.0: None}
        
        for leaf in leaf_bounds:
            if leaf.intersection is None:
                continue
            
            intersection = leaf.intersection
            
            for other_panel in intersection.get_other_panels(panel):
                affected_faces = intersection.get_affected_faces(panel, other_panel)
                if face_side in affected_faces:
                    if leaf.t_start > ENDPOINT_TOLERANCE:
                        t_boundaries[leaf.t_start] = (intersection, other_panel)
                    if leaf.s_start > ENDPOINT_TOLERANCE:
                        s_boundaries[leaf.s_start] = (intersection, other_panel)
                    break
        
        sorted_t = sorted(t_boundaries.keys())
        sorted_s = sorted(s_boundaries.keys())
        
        # If only default boundaries, no splits affecting this face
        if len(sorted_t) == 2 and len(sorted_s) == 2:
            return [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]
        
        # Build 2D grid of corners
        corners_list = []
        
        for i in range(len(sorted_t) - 1):
            for j in range(len(sorted_s) - 1):
                t_start_raw = sorted_t[i]
                t_end_raw = sorted_t[i + 1]
                s_start_raw = sorted_s[j]
                s_end_raw = sorted_s[j + 1]
                
                # Apply gap adjustments at t boundaries
                t_start_adj = t_start_raw
                t_end_adj = t_end_raw
                
                t_start_info = t_boundaries.get(t_start_raw)
                if t_start_info is not None:
                    intersection, other_panel = t_start_info
                    if intersection.type in (IntersectionType.CROSSING, IntersectionType.T_JUNCTION):
                        _, gap_end = FastMeshBuilder._calculate_gap_t_values(
                            panel, face_side, other_panel, t_start_raw
                        )
                        t_start_adj = gap_end
                
                t_end_info = t_boundaries.get(t_end_raw)
                if t_end_info is not None:
                    intersection, other_panel = t_end_info
                    if intersection.type in (IntersectionType.CROSSING, IntersectionType.T_JUNCTION):
                        gap_start, _ = FastMeshBuilder._calculate_gap_t_values(
                            panel, face_side, other_panel, t_end_raw
                        )
                        t_end_adj = gap_start
                
                s_start_adj = s_start_raw
                s_end_adj = s_end_raw
                
                corners_list.append([
                    (t_start_adj, s_start_adj),
                    (t_end_adj, s_start_adj),
                    (t_end_adj, s_end_adj),
                    (t_start_adj, s_end_adj),
                ])
        
        return corners_list
    
    @staticmethod
    def _build_edge_face_corners(panel: GXMLPanel, face_side: PanelSide,
                                  leaf_bounds: List[Region.LeafBounds]) -> List[List[Tuple[float, float]]]:
        """
        Build corners for TOP/BOTTOM faces from leaf regions.
        
        TOP/BOTTOM faces only split on t-axis (not s-axis, which maps to thickness).
        """
        # Extract t-intersections that affect this face
        t_splits: Dict[float, Tuple[Intersection, List[GXMLPanel]]] = {}
        
        for leaf in leaf_bounds:
            if leaf.intersection is None:
                continue
            if leaf.t_start <= ENDPOINT_TOLERANCE:
                continue
            
            t = leaf.t_start
            if t in t_splits:
                continue
            
            intersection = leaf.intersection
            
            for other_panel in intersection.get_other_panels(panel):
                affected_faces = intersection.get_affected_faces(panel, other_panel)
                if face_side in affected_faces:
                    if t not in t_splits:
                        t_splits[t] = (intersection, [])
                    t_splits[t][1].append(other_panel)
        
        # Convert to sorted list
        intersections = [(t, inter, panels) for t, (inter, panels) in sorted(t_splits.items())]
        
        # Filter to midspan only
        midspan = [
            (t, inter, panels) for t, inter, panels in intersections
            if ENDPOINT_TOLERANCE <= t <= (1.0 - ENDPOINT_TOLERANCE)
        ]
        
        corners_list = []
        current_start_t = 0.0
        current_start_t_front = None
        
        # Check for intersection at t≈0
        start_inters = [(t, inter, panels) for t, inter, panels in intersections if t < ENDPOINT_TOLERANCE]
        if start_inters:
            t, inter, panels = start_inters[0]
            intersecting_panel = panels[0] if panels else None
            if intersecting_panel and inter.type in (IntersectionType.CROSSING, IntersectionType.T_JUNCTION):
                _, current_start_t = FastMeshBuilder._calculate_gap_t_values(
                    panel, PanelSide.BACK, intersecting_panel, t)
                _, current_start_t_front = FastMeshBuilder._calculate_gap_t_values(
                    panel, PanelSide.FRONT, intersecting_panel, t)
        
        for t, inter, panels in midspan:
            intersecting_panel = panels[0] if panels else None
            
            end_t = t
            end_t_front = None
            if intersecting_panel and inter.type in (IntersectionType.CROSSING, IntersectionType.T_JUNCTION):
                end_t, _ = FastMeshBuilder._calculate_gap_t_values(
                    panel, PanelSide.BACK, intersecting_panel, t)
                end_t_front, _ = FastMeshBuilder._calculate_gap_t_values(
                    panel, PanelSide.FRONT, intersecting_panel, t)
            
            corners_list.append([
                (current_start_t, 0.0),
                (end_t, 0.0),
                (end_t_front if end_t_front is not None else end_t, 1.0),
                (current_start_t_front if current_start_t_front is not None else current_start_t, 1.0),
            ])
            
            # Next segment starts after this intersection
            if intersecting_panel and inter.type in (IntersectionType.CROSSING, IntersectionType.T_JUNCTION):
                _, current_start_t = FastMeshBuilder._calculate_gap_t_values(
                    panel, PanelSide.BACK, intersecting_panel, t)
                _, current_start_t_front = FastMeshBuilder._calculate_gap_t_values(
                    panel, PanelSide.FRONT, intersecting_panel, t)
            else:
                current_start_t = t
                current_start_t_front = None
        
        # Final segment to t=1
        end_t = 1.0
        end_t_front = None
        
        end_inters = [(t, inter, panels) for t, inter, panels in intersections if t > (1.0 - ENDPOINT_TOLERANCE)]
        if end_inters:
            t, inter, panels = end_inters[0]
            intersecting_panel = panels[0] if panels else None
            if intersecting_panel and inter.type in (IntersectionType.CROSSING, IntersectionType.T_JUNCTION):
                end_t, _ = FastMeshBuilder._calculate_gap_t_values(
                    panel, PanelSide.BACK, intersecting_panel, t)
                end_t_front, _ = FastMeshBuilder._calculate_gap_t_values(
                    panel, PanelSide.FRONT, intersecting_panel, t)
        
        corners_list.append([
            (current_start_t, 0.0),
            (end_t, 0.0),
            (end_t_front if end_t_front is not None else end_t, 1.0),
            (current_start_t_front if current_start_t_front is not None else current_start_t, 1.0),
        ])
        
        return corners_list
    
    @staticmethod
    def _apply_endpoint_trims(panel: GXMLPanel, face_side: PanelSide,
                               corners_list: List[List[Tuple[float, float]]],
                               start_inter: Optional[Intersection],
                               end_inter: Optional[Intersection]) -> None:
        """
        Apply endpoint trims to corner lists in-place.
        """
        if not corners_list:
            return
        
        is_top_or_bottom = face_side in (PanelSide.TOP, PanelSide.BOTTOM)
        
        def get_t_start(face: PanelSide) -> float:
            if start_inter is None:
                return 0.0
            trim = FastMeshBuilder._compute_face_trim(panel, face, PanelSide.START, start_inter)
            return trim if abs(trim) > TOLERANCE else 0.0
        
        def get_t_end(face: PanelSide) -> float:
            if end_inter is None:
                return 1.0
            trim = FastMeshBuilder._compute_face_trim(panel, face, PanelSide.END, end_inter)
            return 1.0 - (trim if abs(trim) > TOLERANCE else 0.0)
        
        if is_top_or_bottom:
            t_start_back = get_t_start(PanelSide.BACK)
            t_end_back = get_t_end(PanelSide.BACK)
            t_start_front = get_t_start(PanelSide.FRONT)
            t_end_front = get_t_end(PanelSide.FRONT)
            
            if len(corners_list) == 1:
                # Unsplit: all corners get trims
                corners_list[0][0] = (t_start_back, 0.0)
                corners_list[0][1] = (t_end_back, 0.0)
                corners_list[0][2] = (t_end_front, 1.0)
                corners_list[0][3] = (t_start_front, 1.0)
            else:
                # Split: first/last segments
                corners_list[0][0] = (t_start_back, 0.0)
                corners_list[0][3] = (t_start_front, 1.0)
                corners_list[-1][1] = (t_end_back, 0.0)
                corners_list[-1][2] = (t_end_front, 1.0)
        else:
            # FRONT/BACK
            t_start = get_t_start(face_side)
            t_end = get_t_end(face_side)
            
            corners_list[0][0] = (t_start, 0.0)
            corners_list[0][3] = (t_start, 1.0)
            corners_list[-1][1] = (t_end, 0.0)
            corners_list[-1][2] = (t_end, 1.0)
    
    @staticmethod
    def _calculate_gap_t_values(panel: GXMLPanel, face_side: PanelSide,
                                 other_panel: GXMLPanel, 
                                 intersection_t: float) -> Tuple[float, float]:
        """Calculate t-values where gap edges should be at an intersection."""
        blocking_dimension = other_panel.thickness
        
        if blocking_dimension < TOLERANCE:
            return (intersection_t, intersection_t)
        
        _, _, face_z_offset = panel.get_face_center_local(face_side)
        ray = panel.get_primary_axis_ray(face_z_offset)
        
        if ray is None:
            return (intersection_t, intersection_t)
        
        face1_inter = FastMeshBuilder._intersect_line_with_panel_face(
            ray.origin, ray.direction, other_panel, PanelSide.FRONT)
        face2_inter = FastMeshBuilder._intersect_line_with_panel_face(
            ray.origin, ray.direction, other_panel, PanelSide.BACK)
        
        if face1_inter is None or face2_inter is None:
            simple_offset = (blocking_dimension / 2) / ray.length
            return (intersection_t - simple_offset, intersection_t + simple_offset)
        
        face1_t = ray.project_point(face1_inter)
        face2_t = ray.project_point(face2_inter)
        
        return (min(face1_t, face2_t), max(face1_t, face2_t))
    
    @staticmethod
    def _compute_face_trim(panel: GXMLPanel, face: PanelSide, endpoint: PanelSide,
                           intersection: Intersection) -> float:
        """Compute trim value for a face at an endpoint intersection."""
        from .gxml_face_solver import FaceSolver
        # Delegate to FaceSolver's implementation for now
        # This can be inlined later for more performance
        return FaceSolver._compute_face_trim(panel, face, endpoint, intersection)
    
    @staticmethod
    def _intersect_line_with_panel_face(line_start, line_direction,
                                         panel: GXMLPanel, face: PanelSide):
        """Intersect a line with a panel face plane."""
        face_normal = panel.get_face_normal(face)
        face_point = panel.get_face_center_world(face)
        return intersect_line_with_plane(line_start, line_direction, face_point, face_normal)
