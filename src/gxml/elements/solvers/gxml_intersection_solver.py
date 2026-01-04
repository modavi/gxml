"""
Panel intersection solver - discovers panel intersections for thick panels.

This module provides pure topological discovery of panel intersections using a unified
architecture where all intersection types (joints, T-junctions, crossings) are handled
consistently through the Intersection class.

Core Concepts:
    Thick Panels:
        Each panel is a volumetric shape with thickness. At endpoints, there are 4 corners:
        front-face and back-face at each end. These corners need proper miter cuts when
        panels meet (solved separately by GeometryBuilder).
    
    Intersection Types (IntersectionType enum):
        - JOINT: Multiple panels meeting endpoint-to-endpoint
        - T_JUNCTION: One panel's endpoint meets another's midspan
        - CROSSING: Two panels intersecting midspan-to-midspan
    
    Unified Representation:
        All intersections use Intersection with:
        - type: Which kind of intersection
        - position: 3D centerline intersection point
        - panels: List of Intersection.PanelEntry objects (sorted CCW)
        
        Each Intersection.PanelEntry contains:
        - panel: Which panel
        - t: Position along panel centerline (0-1)
    
    Key Insight:
        T-junctions need BOTH mitering (at the endpoint) AND splitting (at the midspan).
        This dual requirement is why panels can appear at both endpoints and midspans.

Main API:
    # Solve all intersections
    solution = IntersectionSolver.solve(panels)
    
    # Access unified intersection list
    solution.intersections -> List[Intersection]
    
    # Query by type
    solution.get_intersections_of_type(IntersectionType.JOINT) -> List[Intersection]
    
    # Per-panel queries
    solution.get_intersections_for_panel(panel) -> List[Intersection]
    solution.get_intersection_for_panel_endpoint(panel, endpoint) -> Optional[Intersection]
    
    # Get BSP partition structure for geometry generation
    solution.regions_per_panel -> Dict[GXMLPanel, Optional[Region]]

Data Structures:
    Intersection.PanelEntry:
        How one panel participates in an intersection (panel + t).
        Groups all per-panel data at an intersection into one cohesive object.
    
    Intersection:
        Unified representation of any intersection type.
        Contains position (centerline point) and list of Intersection.PanelEntry objects.
    
    IntersectionSolution:
        Complete solution containing all intersections and query methods.

For geometry building and miter corner solving, see GXMLGeometryBuilder.py.
For integration with the panel system, see GXMLPanel.py.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import mathutils.gxml_math as GXMLMath

if TYPE_CHECKING:
    from elements.gxml_panel import GXMLPanel, PanelSide


# ============================================================================
# CONSTANTS
# ============================================================================

# Threshold for determining if a panel is at an endpoint vs midspan
# A normalized position within this distance of 0 or 1 is considered an endpoint
ENDPOINT_THRESHOLD = 0.01

# Tolerance for distance comparisons in world space
DISTANCE_TOLERANCE = 1e-6

# Precision for spatial hashing (decimal places for rounding coordinates)
SPATIAL_HASH_PRECISION = 6


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class PanelAxis(Enum):
    """Which axis a partition operates on"""
    PRIMARY = 1    # Along panel length (t-parameter, x-axis in local space)
    SECONDARY = 2  # Along panel height (s-parameter, y-axis in local space)

class PanelEndpoint(Enum):
    """Which endpoint of a panel we're referring to"""
    START = 0  # x=0 in panel local space
    END = 1    # x=1 in panel local space

class IntersectionType(Enum):
    """Type of intersection between panels"""
    JOINT = 1        # Multiple panels meeting at endpoints
    T_JUNCTION = 2   # One endpoint meets another's midspan
    CROSSING = 3     # Two midspans intersecting

@dataclass
class Region:
    """
    A rectangular region within a face, possibly further subdivided.
    Forms a node in a BSP (Binary Space Partitioning) tree.
    
    Leaf regions (children=None) represent actual face segments.
    Interior regions have childSubdivisionAxis and children defining how they subdivide.
    
    Attributes:
        tStart: Starting position (0-1) in parent's coordinate space (0.0 for root)
        childSubdivisionAxis: How this region subdivides its children (None for leaf nodes)
        children: Child regions created by subdivision (None for leaf nodes)
        intersection: The intersection that created this region's start boundary (None for t=0)
    """
    tStart: float = 0.0
    childSubdivisionAxis: Optional[PanelAxis] = None
    children: Optional[List['Region']] = None
    intersection: Optional['Intersection'] = None
    
    def is_leaf(self) -> bool:
        """True if this is a leaf region (actual face segment)"""
        return self.children is None
    
    def get_leaves(self) -> List['Region']:
        """Get all leaf regions recursively"""
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves
    
    @dataclass
    class LeafBounds:
        """Bounds for a leaf region in the BSP tree."""
        t_start: float
        t_end: float
        s_start: float
        s_end: float
        intersection: Optional['Intersection']
    
    def get_leaf_bounds(self) -> List['Region.LeafBounds']:
        """
        Get bounds for all leaf regions in this BSP tree.
        
        Each leaf represents a rectangular region with (t, s) bounds.
        The intersection is the one that created the split leading to this leaf
        (None for regions at t=0 or s=0 boundaries).
        
        Returns:
            List of LeafBounds for each leaf region
        """
        results: List['Region.LeafBounds'] = []
        
        def collect(region: 'Region', t_start: float, t_end: float, 
                   s_start: float, s_end: float) -> None:
            if region.is_leaf():
                results.append(Region.LeafBounds(
                    t_start=t_start,
                    t_end=t_end,
                    s_start=s_start,
                    s_end=s_end,
                    intersection=region.intersection
                ))
                return
            
            children = region.children
            for i, child in enumerate(children):
                # Compute end boundary (next child's start, or 1.0 for last child)
                if region.childSubdivisionAxis == PanelAxis.PRIMARY:
                    child_t_start = child.tStart
                    child_t_end = children[i + 1].tStart if i + 1 < len(children) else 1.0
                    collect(child, child_t_start, child_t_end, s_start, s_end)
                else:  # SECONDARY
                    child_s_start = child.tStart  # tStart is overloaded for s-axis in secondary splits
                    child_s_end = children[i + 1].tStart if i + 1 < len(children) else 1.0
                    collect(child, t_start, t_end, child_s_start, child_s_end)
        
        collect(self, 0.0, 1.0, 0.0, 1.0)
        return results


@dataclass
class Intersection:
    """
    Unified representation of any type of panel intersection.
    
    Attributes:
        type: What kind of intersection this is
        position: 3D centerline intersection point
        panels: All panels at this intersection (sorted CCW for joints/T-junctions)
    """
    
    @dataclass
    class PanelEntry:
        """
        Describes how one panel participates in an intersection.
        
        Attributes:
            panel: The panel
            t: Normalized position (0-1) along panel's centerline where intersection occurs
        """
        panel: GXMLPanel
        t: float
        
        def is_at_endpoint(self) -> bool:
            """True if this panel meets the intersection at an endpoint (not midspan)"""
            return self.t < ENDPOINT_THRESHOLD or self.t > (1.0 - ENDPOINT_THRESHOLD)
        
        def get_endpoint(self) -> Optional[PanelEndpoint]:
            """Get which endpoint this is at, or None if midspan"""
            if self.t < ENDPOINT_THRESHOLD:
                return PanelEndpoint.START
            elif self.t > (1.0 - ENDPOINT_THRESHOLD):
                return PanelEndpoint.END
            return None
    
    type: IntersectionType
    position: tuple  # (x, y, z) world position
    panels: List[PanelEntry]
    _panel_lookup: dict = None  # Cached lookup dict, built on first access
    
    def get_entry(self, panel: 'GXMLPanel') -> Optional[PanelEntry]:
        """Get the PanelEntry for a specific panel, or None if not in this intersection."""
        # Build lookup cache on first access
        if self._panel_lookup is None:
            object.__setattr__(self, '_panel_lookup', {e.panel: e for e in self.panels})
        return self._panel_lookup.get(panel)
    
    def get_other_panels(self, panel: 'GXMLPanel') -> List['GXMLPanel']:
        """Get all panels at this intersection except the specified one."""
        return [e.panel for e in self.panels if e.panel != panel]

    def get_panels(self) -> List[GXMLPanel]:
        """Get list of GXMLPanel objects at this intersection"""
        return [p.panel for p in self.panels]

    def get_affected_faces(self, panel: 'GXMLPanel', other_panel: 'GXMLPanel') -> List['PanelSide']:
        """
        Determine which faces of panel are affected by other_panel at this intersection.
        
        - Crossings affect all four lengthwise faces (FRONT, BACK, TOP, BOTTOM)
        - T-junctions/joints only affect the approached face
        
        Args:
            panel: The panel whose faces we're checking
            other_panel: The panel creating the intersection
            
        Returns:
            List of affected PanelSide values
        """
        from elements.gxml_panel import PanelSide
        
        if self.type == IntersectionType.CROSSING:
            # Crossings affect all four lengthwise faces
            return list(PanelSide.thickness_faces()) + list(PanelSide.edge_faces())
        else:
            # T-junctions and joints only affect the approached face
            other_entry = self.get_entry(other_panel)
            direction = other_panel.get_primary_axis()
            if other_entry.t >= 0.5:
                direction = -direction
            approached = panel.get_face_closest_to_direction(
                direction, candidate_faces=PanelSide.thickness_faces())
            return [approached]

# ============================================================================
# INTERSECTION DISCOVERY AND SOLVING
# ============================================================================

def _spatial_hash(position: Sequence) -> Tuple[float, float, float]:
    """Create a spatial hash key for a 3D position for grouping nearby points"""
    return (round(position[0], SPATIAL_HASH_PRECISION),
            round(position[1], SPATIAL_HASH_PRECISION),
            round(position[2], SPATIAL_HASH_PRECISION))


def _get_panel_centerline_bounds(panel: 'GXMLPanel') -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Get axis-aligned bounding box for a panel's centerline."""
    p_start = panel.transform_point([0, 0, 0])
    p_end = panel.transform_point([1, 0, 0])
    
    min_x = min(p_start[0], p_end[0])
    min_y = min(p_start[1], p_end[1])
    min_z = min(p_start[2], p_end[2])
    max_x = max(p_start[0], p_end[0])
    max_y = max(p_start[1], p_end[1])
    max_z = max(p_start[2], p_end[2])
    
    return ((min_x, min_y, min_z), (max_x, max_y, max_z)), (p_start, p_end)


def _build_spatial_grid(panels: List['GXMLPanel'], cell_size: float = 10.0) -> Dict[Tuple[int, int, int], List[Tuple['GXMLPanel', Tuple, Tuple]]]:
    """
    Build a spatial hash grid for panels.
    
    Each panel is inserted into all grid cells that its bounding box overlaps.
    Returns a dict mapping cell coordinates to lists of (panel, start_point, end_point).
    """
    grid: Dict[Tuple[int, int, int], List] = {}
    
    # Small epsilon to handle floating-point edge cases at cell boundaries
    # e.g., a coordinate of -2.22e-16 should be treated as 0, not -1 cell
    epsilon = 1e-9
    
    for panel in panels:
        (min_pt, max_pt), (p_start, p_end) = _get_panel_centerline_bounds(panel)
        
        # Calculate which grid cells this panel overlaps
        # Add epsilon to avoid floating-point boundary issues
        min_cell_x = int((min_pt[0] - epsilon) // cell_size)
        min_cell_y = int((min_pt[1] - epsilon) // cell_size)
        min_cell_z = int((min_pt[2] - epsilon) // cell_size)
        max_cell_x = int((max_pt[0] + epsilon) // cell_size)
        max_cell_y = int((max_pt[1] + epsilon) // cell_size)
        max_cell_z = int((max_pt[2] + epsilon) // cell_size)
        
        # Insert panel into all overlapping cells
        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                for cz in range(min_cell_z, max_cell_z + 1):
                    cell_key = (cx, cy, cz)
                    if cell_key not in grid:
                        grid[cell_key] = []
                    grid[cell_key].append((panel, p_start, p_end))
    
    return grid


def _get_candidate_pairs_from_grid(grid: Dict[Tuple[int, int, int], List]) -> set:
    """
    Get candidate panel pairs from spatial grid.
    
    Only returns pairs that are in the same grid cell (they might intersect).
    Returns set of (panel1, panel2, p1_start, p1_end, p2_start, p2_end) tuples.
    """
    checked_pairs = set()
    candidate_pairs = []
    
    for cell_key, cell_panels in grid.items():
        # Check all pairs within this cell
        for i, (panel1, p1_start, p1_end) in enumerate(cell_panels):
            for panel2, p2_start, p2_end in cell_panels[i+1:]:
                # Use id() to create a unique pair key
                pair_key = (id(panel1), id(panel2)) if id(panel1) < id(panel2) else (id(panel2), id(panel1))
                if pair_key not in checked_pairs:
                    checked_pairs.add(pair_key)
                    candidate_pairs.append((panel1, panel2, p1_start, p1_end, p2_start, p2_end))
    
    return candidate_pairs


class IntersectionSolver:
    """
    Discovers panel relationships and calculates intersection geometry.
    
    Main API:
        IntersectionSolver.solve(panels) -> IntersectionSolution
        
    The solve() method is the primary entry point. Helper methods (_compute_secondary_splits,
    _generate_regions, _merge_regions, _collect_splits) are internal implementation details.
    """
    
    @staticmethod
    def _compute_secondary_splits(panel: GXMLPanel, 
                                  t_value: float,
                                  intersection_position: Sequence,
                                  intersecting_panels: List[GXMLPanel]) -> Optional[Region]:
        """
        Compute secondary axis (s-axis/height) splits for a panel at an intersection.
        
        When other panels cross this panel but don't span its full height, we need
        to subdivide along the secondary axis to create proper attachment surfaces.
        
        Args:
            panel: Panel to compute splits for
            t_value: Position along panel's primary axis where intersection occurs
            intersection_position: 3D position of intersection
            intersecting_panels: Other panels at this intersection
            
        Returns:
            Partitions for secondary axis, or None if no splits needed
        """
        # Get the panel's local coordinate frame at the intersection point
        # We need to determine what s-values (0-1 along height) the intersecting panels span
        s_boundaries = set([0.0])  # Start of panel - last region extends to 1.0 implicitly
        
        for other_panel in intersecting_panels:
            if other_panel == panel:
                continue
              
            # Get the quad corners to determine height extent
            # Bottom corners: [t, 0, 0] and top corners: [t, 1, 0]
            other_bottom_start = other_panel.transform_point([0, 0, 0])
            other_top_start = other_panel.transform_point([0, 1, 0])
            other_bottom_end = other_panel.transform_point([1, 0, 0])
            other_top_end = other_panel.transform_point([1, 1, 0])
            
            # Check if this other panel actually crosses near our intersection point
            # We need to find where the other panel's height boundaries map to our panel's s-axis
            # This is complex - for now, we'll use a simpler heuristic:
            # Check if the other panel's start/end heights differ from our panel's at this intersection
            
            # Transform intersection point to panel's local coords to get our s-value
            # For simplicity, we'll check the Y-extent of the other panel vs our panel
            other_y_min = min(other_bottom_start[1], other_bottom_end[1], 
                             other_top_start[1], other_top_end[1])
            other_y_max = max(other_bottom_start[1], other_bottom_end[1],
                             other_top_start[1], other_top_end[1])
            
            panel_bottom = panel.transform_point([t_value, 0, 0])
            panel_top = panel.transform_point([t_value, 1, 0])
            panel_y_min = min(panel_bottom[1], panel_top[1])
            panel_y_max = max(panel_bottom[1], panel_top[1])
            panel_height = panel_y_max - panel_y_min
            
            # If the other panel doesn't span the full height, compute s-values
            if panel_height > 1e-6:  # Avoid division by zero
                if abs(other_y_min - panel_y_min) > 1e-6:
                    s_min = (other_y_min - panel_y_min) / panel_height
                    s_boundaries.add(max(0.0, min(1.0, s_min)))
                    
                if abs(other_y_max - panel_y_max) > 1e-6:
                    s_max = (other_y_max - panel_y_min) / panel_height
                    s_boundaries.add(max(0.0, min(1.0, s_max)))
        
        # If we only have the start boundary, no splits needed
        if len(s_boundaries) <= 1:
            return None
            
        # Create regions for each s-boundary (each extends to next boundary or 1.0)
        sorted_boundaries = sorted(s_boundaries)
            
        regions = [Region(tStart=s) for s in sorted_boundaries]
        return Region(tStart=0.0, childSubdivisionAxis=PanelAxis.SECONDARY, children=regions)

    @staticmethod
    def _generate_regions(panel_t_values: Dict[GXMLPanel, float],
                         sorted_panels: List[GXMLPanel],
                         intersection_position: Sequence,
                         intersection: 'Intersection') -> Dict[GXMLPanel, Optional[Region]]:
        """
        Generate BSP region structures for panels that need splitting.
        
        Strategy: Split SECONDARY axis first (by height), then PRIMARY axis (by length)
        within regions that actually need it. This minimizes face count.
        
        Args:
            panel_t_values: Dict mapping panels to their t-values at intersection
            sorted_panels: Panels sorted CCW around intersection
            intersection_position: 3D world position where panels intersect
            intersection: The intersection that causes these splits
            
        Returns:
            Dict mapping each panel to its Region (or None if not split)
        """
        regions_per_panel = {}
        
        for panel in sorted_panels:
            t = panel_t_values[panel]
            is_at_endpoint = t < ENDPOINT_THRESHOLD or t > (1.0 - ENDPOINT_THRESHOLD)
            
            if is_at_endpoint:
                # Panel at endpoint doesn't get split
                regions_per_panel[panel] = None
            else:
                # Panel at midspan - check for height mismatches first
                secondary_split_info = IntersectionSolver._compute_secondary_splits(
                    panel, t, intersection_position, sorted_panels
                )
                
                if secondary_split_info is None:
                    # No height mismatch - simple primary axis split
                    regions_per_panel[panel] = Region(
                        tStart=0.0,
                        childSubdivisionAxis=PanelAxis.PRIMARY,
                        children=[
                            Region(tStart=0.0),
                            Region(tStart=t, intersection=intersection)
                        ]
                    )
                else:
                    # Height mismatch exists - split SECONDARY first, then PRIMARY within affected regions
                    # Only the bottom region (where intersecting panel exists) needs primary split
                    secondary_regions = []
                    for i, s_boundary in enumerate(secondary_split_info.children):
                        if i == 0:
                            # First region (bottom) - add primary split at crossing point
                            secondary_regions.append(Region(
                                tStart=s_boundary.tStart,
                                childSubdivisionAxis=PanelAxis.PRIMARY,
                                children=[
                                    Region(tStart=0.0),
                                    Region(tStart=t, intersection=intersection)
                                ]
                            ))
                        else:
                            # Other regions don't need primary split
                            secondary_regions.append(Region(tStart=s_boundary.tStart, intersection=intersection))
                    
                    regions_per_panel[panel] = Region(
                        tStart=0.0,
                        childSubdivisionAxis=PanelAxis.SECONDARY,
                        children=secondary_regions
                    )
        
        return regions_per_panel
    
    @staticmethod
    def _merge_regions(intersection_regions: List[Tuple[Intersection, Dict[GXMLPanel, Optional[Region]]]]) -> Dict[GXMLPanel, Optional[Region]]:
        """
        Merge per-intersection region structures into unified BSP trees per panel.
        
        Each intersection generates regions for panels at midspan. When a panel
        participates in multiple intersections, we need to merge all split points
        into a single BSP tree for geometry generation.
        
        Args:
            intersection_regions: List of (intersection, regions_per_panel) tuples
            
        Returns:
            Dict mapping each panel to its unified region structure
        """
        # Collect all split points per panel per axis, with their causing intersections
        # Dict[panel, Dict[axis, Dict[t_value, intersection]]]
        panel_splits: Dict[GXMLPanel, Dict[PanelAxis, Dict[float, 'Intersection']]] = {}
        
        for intersection, regions_per_panel in intersection_regions:
            for panel, region in regions_per_panel.items():
                if region is None:
                    continue
                    
                if panel not in panel_splits:
                    panel_splits[panel] = {PanelAxis.PRIMARY: {}, PanelAxis.SECONDARY: {}}
                
                # Extract split points from this region structure
                IntersectionSolver._collect_splits(region, panel_splits[panel], intersection)
        
        # Build unified partition structures
        unified: Dict[GXMLPanel, Optional[Region]] = {}
        for panel, splits_by_axis in panel_splits.items():
            # Sort and build unified structure
            # For now, we only support PRIMARY axis (SECONDARY merging is complex)
            primary_splits = splits_by_axis[PanelAxis.PRIMARY]
            sorted_t_values = sorted(primary_splits.keys())
            if len(sorted_t_values) > 1:  # Need at least start + one split
                regions = [
                    Region(tStart=t, intersection=primary_splits.get(t))
                    for t in sorted_t_values
                ]
                unified[panel] = Region(tStart=0.0, childSubdivisionAxis=PanelAxis.PRIMARY, children=regions)
            else:
                unified[panel] = None
        
        return unified
    
    @staticmethod
    def _collect_splits(region: Region, splits_by_axis: Dict[PanelAxis, Dict[float, 'Intersection']], fallback_intersection: 'Intersection') -> None:
        """
        Recursively collect all split points from a region structure.
        
        Args:
            region: Region structure to extract splits from
            splits_by_axis: Dict to accumulate splits into, keyed by Axis, then by t-value to Intersection
            fallback_intersection: The intersection to use if a region doesn't have one set (except at t=0)
        """
        # Add all boundaries for this axis if not a leaf
        if not region.is_leaf():
            for child in region.children:
                # For t=0, this is the original panel start, not a split - no intersection
                # For t>0, use the child's intersection if set, otherwise use fallback
                if child.tStart == 0.0:
                    intersection = None
                else:
                    intersection = child.intersection if child.intersection else fallback_intersection
                splits_by_axis[region.childSubdivisionAxis][child.tStart] = intersection
                
                # Recursively collect from nested regions
                if not child.is_leaf():
                    IntersectionSolver._collect_splits(child, splits_by_axis, fallback_intersection)
    
    @staticmethod
    def _fast_batch_intersections(panels: List[GXMLPanel]):
        """
        Use C extension for batch intersection finding (much faster).
        
        Returns:
            Dict mapping spatial hash keys to intersection data with panel t-values.
        """
        try:
            from .c_solver_wrapper import is_c_extension_available
            if not is_c_extension_available():
                return None
            from ._c_solvers import batch_find_intersections
        except ImportError:
            return None
        
        import numpy as np
        n = len(panels)
        
        # Extract centerline endpoints
        starts = np.zeros((n, 3), dtype=np.float64)
        ends = np.zeros((n, 3), dtype=np.float64)
        
        for i, panel in enumerate(panels):
            p_start = panel.transform_point([0, 0, 0])
            p_end = panel.transform_point([1, 0, 0])
            starts[i] = p_start
            ends[i] = p_end
        
        # Batch find all intersections
        i_arr, j_arr, t1_arr, t2_arr, pos_arr = batch_find_intersections(starts, ends, 1e-6)
        
        # Convert to intersection_data format
        intersection_data: Dict[Tuple[float, float, float], Dict] = {}
        
        for k in range(len(i_arr)):
            i_idx, j_idx = i_arr[k], j_arr[k]
            t1, t2 = t1_arr[k], t2_arr[k]
            pos = pos_arr[k]
            
            panel1 = panels[i_idx]
            panel2 = panels[j_idx]
            
            pos_key = _spatial_hash(pos)
            if pos_key not in intersection_data:
                intersection_data[pos_key] = {
                    'panel_t_values': {},
                    'position': pos
                }
            
            data = intersection_data[pos_key]
            
            if panel1 not in data['panel_t_values']:
                data['panel_t_values'][panel1] = []
            data['panel_t_values'][panel1].append(t1)
            
            if panel2 not in data['panel_t_values']:
                data['panel_t_values'][panel2] = []
            data['panel_t_values'][panel2].append(t2)
        
        return intersection_data

    @staticmethod
    def solve(panels: List[GXMLPanel]) -> IntersectionSolution:
        """
        Find all panel centerline intersections and classify them.
        
        Discovers joints (endpoint-to-endpoint), T-junctions (endpoint-to-midspan),
        and crossings (midspan-to-midspan) by checking all pairwise centerline
        intersections and classifying based on t-values.
        
        Args:
            panels: List of raw panels to process
            
        Returns:
            IntersectionSolution containing all discovered intersections
        """
        # Handle edge cases
        if not panels or len(panels) == 1:
            return IntersectionSolution(
                panels=panels if panels else [],
                intersections=[],
                regions_per_panel={}
            )
        
        # Try fast C extension path first
        intersection_data = IntersectionSolver._fast_batch_intersections(panels)
        
        # Fall back to Python path if C extension not available
        if intersection_data is None:
            # Track intersection data as we discover pairs
            # Key: spatial hash, Value: {panel_t_values: dict, position: ndarray}
            intersection_data: Dict[Tuple[float, float, float], Dict] = {}
            
            # Build spatial grid and get candidate pairs (O(n) instead of O(n²) for sparse layouts)
            # Choose cell size based on typical panel dimensions - larger cells for fewer false negatives
            grid = _build_spatial_grid(panels, cell_size=20.0)
            candidate_pairs = _get_candidate_pairs_from_grid(grid)
            
            # Check only candidate pairs from spatial grid
            for panel1, panel2, p1_start, p1_end, p2_start, p2_end in candidate_pairs:
                # Find where centerlines intersect
                intersection_pos = GXMLMath.find_intersection_between_segments(
                    p1_start, p1_end, p2_start, p2_end
                )
                
                if intersection_pos is None:
                    continue
                
                # Calculate t-values for both panels
                t1 = GXMLMath.find_interpolated_point(intersection_pos, p1_start, p1_end)
                t2 = GXMLMath.find_interpolated_point(intersection_pos, p2_start, p2_end)
                
                if t1 is None or t2 is None:
                    continue
                
                # Add to intersection data at this spatial location
                pos_key = _spatial_hash(intersection_pos)
                if pos_key not in intersection_data:
                    intersection_data[pos_key] = {
                        'panel_t_values': {},
                        'position': intersection_pos
                    }
                
                data = intersection_data[pos_key]
                
                # Accumulate t-values for deduplication/averaging
                if panel1 not in data['panel_t_values']:
                    data['panel_t_values'][panel1] = []
                data['panel_t_values'][panel1].append(t1)
                
                if panel2 not in data['panel_t_values']:
                    data['panel_t_values'][panel2] = []
                data['panel_t_values'][panel2].append(t2)
        
        # Finalize intersections from collected data
        intersections = []
        intersection_regions: List[Tuple[Intersection, Dict[GXMLPanel, Optional[Region]]]] = []
        for pos_key, data in intersection_data.items():
            panel_t_values = data['panel_t_values']
            
            if len(panel_t_values) < 2:
                continue
            
            # Average t-values (they may differ slightly due to floating point)
            unique_panels = {panel: sum(ts) / len(ts) for panel, ts in panel_t_values.items()}
            position = data['position']
            
            if len(unique_panels) < 2:
                continue
            
            # Determine intersection type based on how many panels are at endpoints
            num_at_endpoints = sum(
                1 for t in unique_panels.values()
                if t < ENDPOINT_THRESHOLD or t > (1.0 - ENDPOINT_THRESHOLD)
            )
            
            if num_at_endpoints == len(unique_panels):
                intersection_type = IntersectionType.JOINT
            elif num_at_endpoints > 0:
                intersection_type = IntersectionType.T_JUNCTION
            else:
                intersection_type = IntersectionType.CROSSING
            
            # Sort panels counter-clockwise around intersection
            # Pre-compute angles once instead of in sort key (avoids repeated transform_point calls)
            panel_angles = {}
            for panel in unique_panels:
                # Use cached primary axis ray for start/end
                ray = panel.get_primary_axis_ray()
                start = ray.origin
                end = ray.origin + ray.direction
                
                # Determine which direction to use based on which endpoint is at center
                diff = start - position
                dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
                if dist_sq < DISTANCE_TOLERANCE * DISTANCE_TOLERANCE:
                    direction = end - start
                else:
                    direction = start - end
                
                # Snap tiny direction components to zero to avoid floating point noise
                # affecting atan2 results (e.g., 1e-16 vs 0 can flip -180° to +180°)
                dx = direction[0] if abs(direction[0]) > DISTANCE_TOLERANCE else 0.0
                dz = direction[2] if abs(direction[2]) > DISTANCE_TOLERANCE else 0.0
                
                # Project to XZ plane and get angle
                panel_angles[panel] = math.atan2(dx, dz)
            
            sorted_panels = sorted(unique_panels.keys(), key=lambda p: panel_angles[p])
            
            # Create intersection with PanelEntry objects in sorted order
            participating = [Intersection.PanelEntry(panel, unique_panels[panel]) for panel in sorted_panels]
            
            # Create intersection
            intersection = Intersection(
                type=intersection_type,
                position=position,
                panels=participating
            )
            
            # Generate regions for panels that get split (now with intersection reference)
            regions_per_panel = IntersectionSolver._generate_regions(
                unique_panels, sorted_panels, position, intersection
            )
            
            intersections.append(intersection)
            intersection_regions.append((intersection, regions_per_panel))
        
        # Merge per-intersection regions into unified BSP trees
        unified_regions = IntersectionSolver._merge_regions(intersection_regions)
        
        return IntersectionSolution(
            panels=panels,
            intersections=intersections,
            regions_per_panel=unified_regions
        )
    
@dataclass
class IntersectionSolution:
    """
    Complete solution for panel intersections.
    Contains all panels and their discovered intersections (joints, T-junctions, crossings).
    Provides methods to query intersection data for panel generation.
    
    Key Fields:
        panels: All panels in the solution
        intersections: All discovered intersections
        regions_per_panel: Unified BSP tree per panel (merges all intersection splits)
    
    Each Intersection also has regions_per_panel showing its local contribution.
    """
    panels: List[GXMLPanel]
    intersections: List[Intersection]
    regions_per_panel: Dict[GXMLPanel, Optional[Region]]
    
    def get_intersections_of_type(self, intersection_type: IntersectionType) -> List[Intersection]:
        """Get all intersections of a specific type"""
        return [i for i in self.intersections if i.type == intersection_type]
    
    def get_intersections_for_panel(self, panel: GXMLPanel) -> List[Intersection]:
        """Get all intersections that this panel participates in"""
        return [i for i in self.intersections if panel in i.get_panels()]
    
    def get_region_tree_for_panel(self, panel: GXMLPanel) -> Region:
        """Get the unified BSP region tree for a panel.
        
        Always returns a Region for panels that were part of the solve - either
        the actual split regions if this panel has intersections, or a single
        full-face region (tStart=0) if not.
        
        Raises:
            ValueError: If the panel was not part of the original solve
        """
        if panel not in self.panels:
            raise ValueError(f"Panel '{panel.id}' was not part of the original solve")
        
        region = self.regions_per_panel.get(panel)
        if region is not None:
            return region
        # Return a default full-face region (null object pattern)
        return Region(tStart=0.0)