"""
    A panel is basically a quad with directionality to it.
"""

import math
from enum import IntEnum
from typing import Optional
from elements.gxml_base_element import GXMLLayoutElement
from gxml_types import *
from gxml_profile import *
from mathutils.quad_interpolator import QuadInterpolator, batch_bilinear_transform
from mathutils.gxml_ray import GXMLRay
import mathutils.gxml_math as GXMLMath
from elements.gxml_quad import GXMLQuad


class PanelSide(IntEnum):
    FRONT = 0
    BACK = 1
    TOP = 2
    BOTTOM = 3
    START = 4
    END = 5
    
    @classmethod
    def thickness_faces(cls) -> list['PanelSide']:
        """Faces that define panel thickness (FRONT/BACK)."""
        return [cls.FRONT, cls.BACK]
    
    @classmethod
    def edge_faces(cls) -> list['PanelSide']:
        """Faces along the top/bottom edges of the panel."""
        return [cls.TOP, cls.BOTTOM]

class GXMLPanel(GXMLLayoutElement):
    '''
        A panel is a quad that can be oriented in 3D space. Panels can be attached to one another
        to form larger structures, such as walls, floors, and ceilings, and they have automatic handling
        for generating joints and intersections between them, as well as thickness.
        
        They have several main defining characteristics:
        
        - Local Coordinate Space: Panels have a local coordinate space that is used to determine their orientation and position
                                in 3D space. The local coordinate space is defined by the primary axis, normal axis, 
                                and secondary axis. This allows us to position other elements relative to the coordinate space
                                of the panel.

        - Normalized Coordinate Space: The normalized coordinate space of the panel allows us to refer to positions along the panel
                            in normalized terms. 0 is the startpoint of the panel, and 1 is the endpoint of the panel along the primary axis,
                            and then along the secondary axis, 0 is the bottom of the panel and 1 is the top of the panel. Then finally
                            along the normal axis, 0 is the back of the panel and 1 is the front of the panel.
        
        - A primary axis: Panels have directionality to them which determines which way they flow. This determines the startpoint and
                        endpoint of the panel, which can be used to determine how they connect to other panels. The startpoint of
                        one panel is automatically attached to the endpoint of the previous panel. The primary axis by default
                        is the X axis in the local coordinate space of the panel, but can be overridden by the user.
                            
        - A normal axis: The normal axis is the direction the panel faces relative to the primary axis. It always points outward
                        from the 'front' face of the panel or the +Z direction (relative to the panels local coordinate space).
    
        - An secondary axis: The secondary axis of a panel is the direction that the panel extends upwards in the +Y direction (relative
                            to the panels local coordinate space). It is always perpendicular to the primary axis and the normal axis
                            and can be derived from the cross product of the two.

        - Sides: Panels with 0 thickness have 2 sides, front and back. Panels with thickness have 6 sides, 
                front, back, start, end, top, and bottom.

        - Thickness: The thickness of the panel, which determines how far it extends along the normal axis.

        - Normalized Vertices: The vertices that defines the shape of the panel. They define how the 4 points of the panel relate to each other
            along the primary and secondary axes. It allows for creating skewed panels that are not rectangular, but still exhibit
            the properties of a panel. They are defined by 4 normalized coordinates specified relative to the local coordinate space
            of the panel. Start / Bottom, End / Bottom, End / Top, Start / Top. These 4 points can be adjusted to shift in the 4
            vertices of the panel, and then requests for positions along the normalized coordinate space of the panel will be
            bilinear interpolated between these 4 points. 

        ##################################################################################  
        # Local Coordinate System of a Panel:
        # The bottom left corner of the panel at the start/bottom/back face intersection is the origin (0,0,0)
        #
        # Returns the coordinates in clockwise order, facing directly towards the panel.
        #
        #             Top 
        #           +------+ 
        #          /|     /|          +Y Secondary (Bottom/Top)
        #         / |    / |           |
        #        +------+  |           |  
        #  Back  |  |   |  |  Front    |
        #        |  +---|--+           +-----> +Z  Normal (Back/Front)
        #        | /    | /           /
        #        |/     |/           /
        #        +------+           -X Primary (Start/End)
        #         Start      
    '''
        
    def __init__(self):
        super().__init__()
        self.childLayoutScheme = "stack"
        self.thickness = 0.0
        
        self.quad_interpolator = QuadInterpolator([0,0,0], [1,0,0], [1,1,0], [0,1,0])
        
        # Caches for expensive computations (cleared on transform changes)
        self._face_normal_cache = {}
        self._face_point_cache = {}
        self._primary_axis_ray_cache = {}
        
    def parse(self, ctx):
        super().parse(ctx)
        self.thickness = float(ctx.getAttribute("thickness", self.thickness))
        
    def transform_point(self, point):
        # Use combined bilinear + transform when available
        t, s, z = point[0], point[1], point[2] if len(point) > 2 else 0.0
        quad_points = self.quad_interpolator.get_quad_points()
        return self.transform.bilinear_transform_point(t, s, z, quad_points)
    
    def invalidate_caches(self):
        """Clear cached computations. Call when transform changes."""
        self._face_normal_cache.clear()
        self._face_point_cache.clear()
        self._primary_axis_ray_cache.clear()
    
    @staticmethod
    def _get_sibling_panels(panel):
        """Get all panel siblings including the panel itself."""
        panels = []
        for child in panel.parent.children:
            if isinstance(child, GXMLPanel):
                panels.append(child)
        return panels
    
    @staticmethod  
    def _get_or_compute_solution(panel):
        """Get cached solution from parent, or compute it if not available.
        
        The solution is cached on the parent element so all sibling panels
        share the same computation. This is cleared at the start of each
        post-layout pass.
        """
        from elements.solvers import IntersectionSolver, FaceSolver
        
        cache_key = '_panel_solution_cache'
        parent = panel.parent
        
        # Check if solution is already cached on parent
        if hasattr(parent, cache_key):
            return getattr(parent, cache_key)
        
        # Compute solution for all sibling panels
        all_panels = GXMLPanel._get_sibling_panels(panel)
        intersection_solution = IntersectionSolver.solve(all_panels)
        panel_faces = FaceSolver.solve(intersection_solution)
        
        # Cache on parent for sibling panels to reuse
        solution = (intersection_solution, panel_faces, False)  # False = not yet built
        setattr(parent, cache_key, solution)
        
        return solution
        
    def on_post_layout(self):
        push_perf_marker(None)
        
        from elements.solvers import get_geometry_builder, get_geometry_backend
        
        # Get or compute the shared solution (cached on parent)
        cache_key = '_panel_solution_cache'
        parent = self.parent
        solution = GXMLPanel._get_or_compute_solution(self)
        intersection_solution, panel_faces, already_built = solution
        
        builder = get_geometry_builder()
        
        # GPU backend: build_all on first panel, skip subsequent
        # CPU backend: build per-panel as before
        if get_geometry_backend() == 'gpu' and not already_built:
            # Build all panels at once (GPU batching benefit)
            builder.build_all(panel_faces, intersection_solution)
            # Mark as built so subsequent panels skip
            setattr(parent, cache_key, (intersection_solution, panel_faces, True))
        elif get_geometry_backend() == 'cpu':
            # CPU path: build per-panel (original behavior)
            builder.build(self, panel_faces, intersection_solution)
            
        pop_perf_marker()
    
    def render(self, renderContext):
        if not self.dynamicChildren:
            points = [
                self.transform_point((corner[0], corner[1], 0))
                for corner in [(0,0), (1,0), (1,1), (0,1)]
            ]
            renderContext.create_poly(f"{self.id}-{self.subId}", points)
        else:
            for child in self.dynamicChildren:
                child.render(renderContext)
    
    def create_panel_side(self, subId, side, corners=None):
        """
        Create a panel side (face) as a dynamic child GXMLQuad.
        
        Args:
            subId: Identifier for this face
            side: Which face (PanelSide enum)
            corners: Optional list of 4 (t, s) tuples defining the quad corners in CCW order.
                     Default is [(0,0), (1,0), (1,1), (0,1)] (full unit quad).
                     Corner order: start-bottom, end-bottom, end-top, start-top
                     
        Returns:
            GXMLQuad representing the face
        """
        push_perf_marker()
        
        if corners is None:
            corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        quad = GXMLQuad()
        quad.id = self.id
        quad.subId = subId
        
        self.add_dynamic_child(quad)
        
        localPoints = self.get_local_points_from_side(side)
        
        # Build list of (t, s, z_offset) for batch transform
        # corners are (t, s) values in parent panel's normalized space
        # Local points have (x, y, z) where the face's constant axis determines which
        # local coordinate maps to t and which maps to s:
        #   FRONT/BACK (z constant): x->t, y->s, z->thickness
        #   TOP/BOTTOM (y constant): x->t, s->thickness, y is fixed
        #   START/END (x constant): z->thickness, y->s, x is fixed
        points_for_transform = []
        for lp in localPoints:
            # Map local (x, y, z) to corner (t, s) based on face type
            # Each face type has one constant axis and two varying axes
            # We map the varying axes to (u, v) for corner lookup
            if side in (PanelSide.FRONT, PanelSide.BACK):
                # z is constant; x varies with t, y varies with s
                u, v = lp[0], lp[1]
            elif side in (PanelSide.TOP, PanelSide.BOTTOM):
                # y is constant; x varies with t, z (back/front) varies with s
                u, v = lp[0], lp[2]
            else:  # START, END
                # x is constant; z varies with t (thickness direction), y varies with s
                u, v = lp[2], lp[1]
            
            # Map (u, v) in [0,1] range to corner index
            # corners: [0]=(0,0), [1]=(1,0), [2]=(1,1), [3]=(0,1)
            u_int, v_int = int(u), int(v)
            if v_int == 0:
                corner_idx = u_int  # bottom row: 0 or 1
            else:
                corner_idx = 3 - u_int  # top row: 3 or 2
            
            t, s = corners[corner_idx]
            
            # Build local point for transform based on face type
            # For FRONT/BACK: (t, s, thickness_from_face)
            # For TOP/BOTTOM: (t, fixed_y, thickness_from_s)
            # For START/END: (fixed_x, s, thickness_from_z)
            if side in (PanelSide.FRONT, PanelSide.BACK):
                points_for_transform.append((t, s, (lp[2] - 0.5) * self.thickness))
            elif side in (PanelSide.TOP, PanelSide.BOTTOM):
                # y is fixed at the face position, s determines thickness offset
                points_for_transform.append((t, lp[1], (s - 0.5) * self.thickness))
            else:  # START, END
                # x is fixed at the face position (0 for START, 1 for END)
                points_for_transform.append((lp[0], s, (lp[2] - 0.5) * self.thickness))
        
        # Batch transform all 4 points at once using C extension
        worldPoints = batch_bilinear_transform(
            points_for_transform,
            self.quad_interpolator.get_quad_points(),
            self.transform.transformationMatrix
        )
        
        # Build affine transformation matrix from 3 corners (p0, p1, p3)
        # This matrix maps the unit square to the parallelogram defined by those corners
        matrix = GXMLMath.create_transform_matrix_from_quad(worldPoints)
        
        # For non-rectangular faces (trapezoids), the 4th corner (p2) deviates from
        # the parallelogram. We encode this deviation in the quad by transforming
        # all world points back to local space - this gives us the "ratio" quad.
        inv_matrix = GXMLMath.invert(matrix)
        localQuadPoints = GXMLMath.batch_transform_points(worldPoints, inv_matrix)
        
        quad._interpolator = QuadInterpolator(localQuadPoints[0], localQuadPoints[1], 
                              localQuadPoints[2], localQuadPoints[3])
        quad.transform.transformationMatrix = matrix
        # Cache the world points we already computed to avoid recomputing in render()
        quad._cached_world_vertices = worldPoints
        
        pop_perf_marker()
        return quad
    
    def get_local_points_from_side(self, side):
        """
        Returns the 4 corner points for a given side of the panel in CCW order
        when viewed from outside. Each side is a plane where one coordinate is constant.
        
        The ordering ensures proper winding for the outward-facing normal:
        - For sides facing positive axis direction (+X, +Y, +Z): standard CCW
        - For sides facing negative axis direction (-X, -Y, -Z): reversed to maintain outward normal
        """
        
        # Define corners systematically: each side has one fixed coordinate and two varying
        # Points are returned in CCW order when viewed from outside (outward normal)
        
        if side == PanelSide.START:     # x=0 plane (facing -X)
            return [(0,0,0), (0,0,1), (0,1,1), (0,1,0)]
        elif side == PanelSide.END:       # x=1 plane (facing +X)
            return [(1,0,1), (1,0,0), (1,1,0), (1,1,1)]
        elif side == PanelSide.BOTTOM:    # y=0 plane (facing -Y)
            return [(0,0,0), (1,0,0), (1,0,1), (0,0,1)]
        elif side == PanelSide.TOP:       # y=1 plane (facing +Y)
            return [(0,1,0), (0,1,1), (1,1,1), (1,1,0)]
        elif side == PanelSide.BACK:      # z=0 plane (facing -Z)
            return [(1,0,0), (0,0,0), (0,1,0), (1,1,0)]
        elif side == PanelSide.FRONT:     # z=1 plane (facing +Z)
            return [(0,0,1), (1,0,1), (1,1,1), (0,1,1)]
        else:
            raise ValueError(f"Invalid panel side: {side}")
    
    def get_face_point(self, face_side: 'PanelSide', t: float, s: float) -> tuple:
        """
        Map (t, s) coordinates to a world-space point on a face.
        
        Args:
            face_side: Which face of the panel
            t: Position along primary axis (0 to 1)
            s: Position along secondary axis (0 to 1)
            
        Returns:
            World-space point on the face as (x, y, z) tuple
        """
        half_thickness = self.thickness / 2
        
        if face_side == PanelSide.FRONT:
            local = (t, s, half_thickness)
        elif face_side == PanelSide.BACK:
            local = (t, s, -half_thickness)
        elif face_side == PanelSide.TOP:
            z = -half_thickness + s * self.thickness
            local = (t, 1.0, z)
        elif face_side == PanelSide.BOTTOM:
            z = -half_thickness + s * self.thickness
            local = (t, 0.0, z)
        elif face_side == PanelSide.START:
            z = -half_thickness + t * self.thickness
            local = (0.0, s, z)
        elif face_side == PanelSide.END:
            z = -half_thickness + t * self.thickness
            local = (1.0, s, z)
        else:
            local = (t, s, 0.0)
        
        return self.transform_point(local)
    
    def get_primary_axis(self):
        """
        Returns the primary axis direction (start→end, +X in local space) in world coordinates.
        This is the panel's flow direction.
        """
        ray = self.get_primary_axis_ray()
        if ray is None:
            return (1.0, 0.0, 0.0)
        return ray.direction
    
    def get_secondary_axis(self):
        """
        Returns the secondary axis direction (bottom→top, +Y in local space) in world coordinates.
        This is the panel's "up" direction.
        """
        bottom_point = self.transform_point((0, 0, 0))
        top_point = self.transform_point((0, 1, 0))
        ax = top_point[0] - bottom_point[0]
        ay = top_point[1] - bottom_point[1]
        az = top_point[2] - bottom_point[2]
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm > 0:
            inv = 1.0 / norm
            return (ax * inv, ay * inv, az * inv)
        return (ax, ay, az)
    
    def get_normal_axis(self):
        """
        Returns the normal axis direction (back→front, +Z in local space) in world coordinates.
        This is the panel's outward-facing direction.
        """
        back_point = self.transform_point((0, 0, 0))
        front_point = self.transform_point((0, 0, 1))
        ax = front_point[0] - back_point[0]
        ay = front_point[1] - back_point[1]
        az = front_point[2] - back_point[2]
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm > 0:
            inv = 1.0 / norm
            return (ax * inv, ay * inv, az * inv)
        return (ax, ay, az)
    
    def is_valid(self, tolerance: float = 1e-4) -> bool:
        """
        Check if this panel has valid geometry for building.
        
        A panel is invalid if it has degenerate geometry that would cause
        errors during geometry building (e.g., zero-scale axes that can't be normalized).
        
        Args:
            tolerance: Minimum axis length to be considered valid
            
        Returns:
            True if the panel has valid geometry, False otherwise
        """
        # Check primary axis (width/length)
        start_point = self.transform_point((0, 0, 0))
        end_point = self.transform_point((1, 0, 0))
        primary_length = GXMLMath.distance(start_point, end_point)
        if primary_length < tolerance:
            return False
        
        # Check secondary axis (height)
        top_point = self.transform_point((0, 1, 0))
        secondary_length = GXMLMath.distance(start_point, top_point)
        if secondary_length < tolerance:
            return False
        
        return True
    
    def get_face_normal(self, side):
        """
        Returns the outward-facing normal vector for a given face in world coordinates.
        
        Args:
            side: PanelSide enum value
            
        Returns:
            tuple: normalized normal vector in world space
        """
        # Check cache first
        if side in self._face_normal_cache:
            return self._face_normal_cache[side]
        
        # Get three points on the face to compute the normal (batched for speed)
        local_points = self.get_local_points_from_side(side)
        points_for_transform = [
            (lp[0], lp[1], (lp[2] - 0.5) * self.thickness)
            for lp in local_points[:3]
        ]
        world_points = batch_bilinear_transform(
            points_for_transform,
            self.quad_interpolator.get_quad_points(),
            self.transform.transformationMatrix
        )
        
        # Compute normal using cross product of two edge vectors
        edge1 = (world_points[1][0] - world_points[0][0], 
                 world_points[1][1] - world_points[0][1],
                 world_points[1][2] - world_points[0][2])
        edge2 = (world_points[2][0] - world_points[0][0],
                 world_points[2][1] - world_points[0][1],
                 world_points[2][2] - world_points[0][2])
        normal = GXMLMath.cross3(edge1, edge2)
        
        norm = math.sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
        if norm > 0:
            inv = 1.0 / norm
            result = (normal[0] * inv, normal[1] * inv, normal[2] * inv)
        else:
            result = normal
        
        # Cache and return
        self._face_normal_cache[side] = result
        return result
    
    def get_visible_thickness_face(self) -> 'PanelSide':
        """
        Determine which thickness face (FRONT or BACK) is "visible" for zero-thickness panels.
        
        For zero-thickness panels, FRONT and BACK are coplanar - they're the same geometric
        plane with opposite normals. Since they're geometrically identical, we need a
        convention for which face to display.
        
        Convention:
        - Rectangular panels (standard panels): always show BACK
        - Non-rectangular panels (twisted/warped by attachments): 
          - Show FRONT if the quad's normal has positive Z component
          - Show BACK if the quad's normal has negative Z component
        
        Returns:
            PanelSide based on the conventions above.
        """
        # Get the panel's four corners in world space
        corners = [
            self.transform_point((0, 0, 0)),  # start-bottom (p0)
            self.transform_point((1, 0, 0)),  # end-bottom (p1)
            self.transform_point((1, 1, 0)),  # end-top (p2)
            self.transform_point((0, 1, 0)),  # start-top (p3)
        ]
        
        # Check if the quad is rectangular by comparing opposite edge lengths
        def edge_len(p1, p2):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]
            return math.sqrt(dx*dx + dy*dy + dz*dz)
        
        edge1_len = edge_len(corners[0], corners[1])  # bottom edge
        edge2_len = edge_len(corners[1], corners[2])  # right edge
        edge3_len = edge_len(corners[2], corners[3])  # top edge
        edge4_len = edge_len(corners[3], corners[0])  # left edge
        
        tolerance = 1e-6
        is_rectangular = (abs(edge1_len - edge3_len) < tolerance and 
                          abs(edge2_len - edge4_len) < tolerance)
        
        if is_rectangular:
            # Rectangular panels always show BACK by convention
            return PanelSide.BACK
        else:
            # For non-rectangular panels, compute the quad normal
            edge1 = (corners[1][0] - corners[0][0], 
                     corners[1][1] - corners[0][1],
                     corners[1][2] - corners[0][2])
            edge2 = (corners[3][0] - corners[0][0],
                     corners[3][1] - corners[0][1],
                     corners[3][2] - corners[0][2])
            normal = GXMLMath.cross3(edge1, edge2)
            
            # Show FRONT if normal has positive Z, BACK if negative
            if normal[2] > 0:
                return PanelSide.FRONT
            else:
                return PanelSide.BACK
    
    # Class-level lookup table for face center local coordinates
    # Structure: (t_offset, s_offset, z_factor) where z_factor is multiplied by half_thickness
    _FACE_CENTER_LOCAL_FACTORS = {
        PanelSide.FRONT: (0.5, 0.5, 1.0),
        PanelSide.BACK: (0.5, 0.5, -1.0),
        PanelSide.TOP: (0.5, 1.0, 0.0),
        PanelSide.BOTTOM: (0.5, 0.0, 0.0),
        PanelSide.START: (0.0, 0.5, 0.0),
        PanelSide.END: (1.0, 0.5, 0.0),
    }
    
    def get_face_center_local(self, face: PanelSide) -> tuple:
        """
        Get the local-space coordinates of a face's center point.
        
        For FRONT/BACK faces, the center is at (0.5, 0.5, ±half_thickness).
        For edge faces (TOP/BOTTOM/START/END), the center spans the thickness
        at z=0, positioned at the appropriate t/s edge.
        
        Args:
            face: Which face (PanelSide enum)
            
        Returns:
            (t, s, z) tuple in panel local space
        """
        factors = self._FACE_CENTER_LOCAL_FACTORS.get(face)
        if factors is None:
            return (0.5, 0.5, 0.0)
        return (factors[0], factors[1], factors[2] * self.thickness / 2)
    
    def get_face_center_world(self, face: PanelSide) -> tuple:
        """
        Get the world-space center point of a face (cached).
        
        Args:
            face: Which face (PanelSide enum)
            
        Returns:
            tuple: (x, y, z) world-space position of face center
        """
        if face in self._face_point_cache:
            return self._face_point_cache[face]
        
        face_offset = self.get_face_center_local(face)
        point = self.transform_point(face_offset)
        self._face_point_cache[face] = point
        return point
    
    def get_primary_axis_ray(self, z_offset: float = 0.0) -> Optional[GXMLRay]:
        """
        Get a ray along the panel's primary (t) axis in world space.
        
        Args:
            z_offset: Z offset in panel local space (e.g., half_thickness for front face)
            
        Returns:
            Ray from t=0 to t=1, or None if panel is degenerate
        """
        if z_offset in self._primary_axis_ray_cache:
            return self._primary_axis_ray_cache[z_offset]
        
        start = self.transform_point((0, 0, z_offset))
        end = self.transform_point((1, 0, z_offset))
        ray = GXMLRay.from_points(start, end)
        self._primary_axis_ray_cache[z_offset] = ray
        return ray
    
    def get_face_closest_to_direction(self, direction, candidate_faces=None):
        """
        Determines which face is most aligned with a given direction vector.
        Useful for determining which face is being intersected based on approach direction.
        
        Args:
            direction: tuple, list, or array representing a direction in world space
            candidate_faces: Optional list of PanelSide values to consider.
                             If None, all faces are considered.
            
        Returns:
            PanelSide: the face whose normal is most aligned with the given direction
        """
        dx, dy, dz = direction[0], direction[1], direction[2]
        direction_norm = math.sqrt(dx*dx + dy*dy + dz*dz)
        if direction_norm > 0:
            inv = 1.0 / direction_norm
            dx, dy, dz = dx * inv, dy * inv, dz * inv
        
        best_side = None
        best_dot = -float('inf')
        
        faces_to_check = candidate_faces if candidate_faces is not None else PanelSide
        for side in faces_to_check:
            face_normal = self.get_face_normal(side)
            dot = face_normal[0]*dx + face_normal[1]*dy + face_normal[2]*dz
            if dot > best_dot:
                best_dot = dot
                best_side = side
        
        return best_side
