"""
Taichi GPU-accelerated IntersectionSolver.

This replaces the CPU IntersectionSolver with a GPU implementation that:
1. Parallelizes all N×N line-line intersection tests on GPU
2. Parallelizes t-value computation
3. Uses host code for classification and region building (inherently sequential)

The GPU is primarily used for the expensive O(n²) intersection detection phase.
"""

import taichi as ti
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum, auto

from elements.gxml_panel import GXMLPanel, PanelSide

# NOTE: ti.init() should be called by the application before importing this module.
# This allows the application to choose the backend (metal, cuda, vulkan, cpu).

# Constants
TOLERANCE = 1e-4
ENDPOINT_TOLERANCE = 0.01
CELL_SIZE = 20.0


# ==============================================================================
# Taichi Fields and Kernels
# ==============================================================================

# Maximum panels we support (can be increased)
MAX_PANELS = 1024
MAX_INTERSECTIONS = MAX_PANELS * MAX_PANELS // 2

# Panel centerline data (start x, y, z, end x, y, z)
panel_starts = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PANELS)
panel_ends = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PANELS)
panel_count = ti.field(dtype=ti.i32, shape=())

# Intersection output buffers
# intersection_results[i, j] = (found, t1, t2, x, y, z) for panels i and j
intersection_found = ti.field(dtype=ti.i32, shape=(MAX_PANELS, MAX_PANELS))
intersection_t1 = ti.field(dtype=ti.f32, shape=(MAX_PANELS, MAX_PANELS))
intersection_t2 = ti.field(dtype=ti.f32, shape=(MAX_PANELS, MAX_PANELS))
intersection_x = ti.field(dtype=ti.f32, shape=(MAX_PANELS, MAX_PANELS))
intersection_y = ti.field(dtype=ti.f32, shape=(MAX_PANELS, MAX_PANELS))
intersection_z = ti.field(dtype=ti.f32, shape=(MAX_PANELS, MAX_PANELS))


@ti.func
def line_line_intersection_2d(
    p1: ti.types.vector(3, ti.f32),
    p2: ti.types.vector(3, ti.f32),
    p3: ti.types.vector(3, ti.f32),
    p4: ti.types.vector(3, ti.f32)
) -> ti.types.vector(5, ti.f32):  # (found, t1, t2, x, z)
    """
    Find intersection of two line segments in XZ plane.
    Returns (found, t1, t2, x, z) where:
    - found: 1 if intersection exists, 0 otherwise
    - t1: parameter along first segment [0, 1]
    - t2: parameter along second segment [0, 1]
    - x, z: intersection point coordinates
    """
    result = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Direction vectors
    d1x = p2[0] - p1[0]
    d1z = p2[2] - p1[2]
    d2x = p4[0] - p3[0]
    d2z = p4[2] - p3[2]
    
    # Cross product (determinant)
    cross = d1x * d2z - d1z * d2x
    
    if ti.abs(cross) > TOLERANCE:
        # Not parallel
        dx = p3[0] - p1[0]
        dz = p3[2] - p1[2]
        
        t1 = (dx * d2z - dz * d2x) / cross
        t2 = (dx * d1z - dz * d1x) / cross
        
        # Check if intersection is within both segments
        if t1 >= -TOLERANCE and t1 <= 1.0 + TOLERANCE and t2 >= -TOLERANCE and t2 <= 1.0 + TOLERANCE:
            # Clamp t values
            t1 = ti.max(0.0, ti.min(1.0, t1))
            t2 = ti.max(0.0, ti.min(1.0, t2))
            
            # Compute intersection point
            ix = p1[0] + t1 * d1x
            iz = p1[2] + t1 * d1z
            
            result = ti.Vector([1.0, t1, t2, ix, iz])
    
    return result


@ti.kernel
def find_all_intersections():
    """
    GPU kernel to find all pairwise line intersections.
    
    Parallelizes over all pairs (i, j) where i < j.
    Results are stored in intersection_* fields.
    """
    n = panel_count[None]
    
    for i, j in ti.ndrange(n, n):
        if i < j:
            # Get panel centerlines
            p1 = panel_starts[i]
            p2 = panel_ends[i]
            p3 = panel_starts[j]
            p4 = panel_ends[j]
            
            # Find intersection
            result = line_line_intersection_2d(p1, p2, p3, p4)
            
            if result[0] > 0.5:  # Found
                intersection_found[i, j] = 1
                intersection_t1[i, j] = result[1]
                intersection_t2[i, j] = result[2]
                intersection_x[i, j] = result[3]
                # Y is interpolated from panel heights
                y1 = p1[1] + result[1] * (p2[1] - p1[1])
                y2 = p3[1] + result[2] * (p4[1] - p3[1])
                intersection_y[i, j] = (y1 + y2) * 0.5
                intersection_z[i, j] = result[4]
            else:
                intersection_found[i, j] = 0


@ti.kernel
def clear_intersection_buffers():
    """Clear all intersection result buffers."""
    n = panel_count[None]
    for i, j in ti.ndrange(n, n):
        intersection_found[i, j] = 0
        intersection_t1[i, j] = 0.0
        intersection_t2[i, j] = 0.0
        intersection_x[i, j] = 0.0
        intersection_y[i, j] = 0.0
        intersection_z[i, j] = 0.0


# ==============================================================================
# Data structures (same as CPU version for compatibility)
# ==============================================================================

class IntersectionType(Enum):
    """Type of intersection between panels."""
    JOINT = auto()       # All panels have endpoints at intersection
    T_JUNCTION = auto()  # Some panels have endpoints, others pass through
    CROSSING = auto()    # All panels pass through (midspan intersection)


@dataclass
class Region:
    """BSP tree node for face segmentation (same as CPU version)."""
    t_start: float = 0.0
    t_end: float = 1.0
    s_start: float = 0.0
    s_end: float = 1.0
    intersection: 'Intersection' = None
    children: List['Region'] = field(default_factory=list)
    
    @dataclass
    class LeafBounds:
        """Bounds from a leaf region."""
        t_start: float
        t_end: float
        s_start: float
        s_end: float
        intersection: Optional['Intersection']
    
    def get_leaf_bounds(self) -> List['Region.LeafBounds']:
        """Get bounds from all leaf regions."""
        if not self.children:
            return [Region.LeafBounds(
                self.t_start, self.t_end,
                self.s_start, self.s_end,
                self.intersection
            )]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaf_bounds())
        return leaves
    
    def split_at_t(self, t: float, intersection: 'Intersection') -> None:
        """Split this region at t value."""
        if not self.children:
            if self.t_start < t < self.t_end:
                self.children = [
                    Region(self.t_start, t, self.s_start, self.s_end, self.intersection),
                    Region(t, self.t_end, self.s_start, self.s_end, intersection),
                ]
        else:
            for child in self.children:
                child.split_at_t(t, intersection)


@dataclass
class Intersection:
    """Unified intersection representation (same as CPU version)."""
    
    @dataclass
    class PanelEntry:
        """Entry for a panel in an intersection."""
        panel: GXMLPanel
        t: float
        
        def is_endpoint(self) -> bool:
            return self.t < ENDPOINT_TOLERANCE or self.t > (1.0 - ENDPOINT_TOLERANCE)
    
    position: Tuple[float, float, float]
    panels: List[PanelEntry] = field(default_factory=list)
    type: IntersectionType = IntersectionType.CROSSING
    
    def get_entry(self, panel: GXMLPanel) -> Optional[PanelEntry]:
        """Get the entry for a specific panel."""
        for entry in self.panels:
            if entry.panel is panel:
                return entry
        return None
    
    def get_other_panels(self, panel: GXMLPanel) -> List[GXMLPanel]:
        """Get all panels except the given one."""
        return [e.panel for e in self.panels if e.panel is not panel]
    
    def get_affected_faces(self, panel: GXMLPanel, other_panel: GXMLPanel) -> Set[PanelSide]:
        """Get faces affected by this intersection."""
        # Simplified version - full implementation in CPU solver
        return {PanelSide.FRONT, PanelSide.BACK, PanelSide.TOP, PanelSide.BOTTOM}


@dataclass
class IntersectionSolution:
    """Complete solution from IntersectionSolver (same as CPU version)."""
    panels: List[GXMLPanel]
    intersections: List[Intersection] = field(default_factory=list)
    regions_per_panel: Dict[GXMLPanel, Region] = field(default_factory=dict)
    
    def get_intersections_for_panel(self, panel: GXMLPanel) -> List[Intersection]:
        """Get all intersections involving a panel."""
        return [i for i in self.intersections if any(e.panel is panel for e in i.panels)]


# ==============================================================================
# TaichiIntersectionSolver
# ==============================================================================

class TaichiIntersectionSolver:
    """
    GPU-accelerated IntersectionSolver using Taichi.
    
    The GPU is used for the O(n²) intersection detection phase.
    Classification and region building remain on CPU (inherently sequential).
    """
    
    @staticmethod
    def solve(panels: List[GXMLPanel]) -> IntersectionSolution:
        """
        Find all intersections between panel centerlines.
        
        Args:
            panels: List of GXMLPanel instances
            
        Returns:
            IntersectionSolution with all intersections and regions
        """
        n = len(panels)
        if n == 0:
            return IntersectionSolution(panels=panels)
        
        if n > MAX_PANELS:
            raise ValueError(f"Too many panels ({n}). Maximum supported: {MAX_PANELS}")
        
        # Upload panel data to GPU
        panel_count[None] = n
        for i, panel in enumerate(panels):
            # Get centerline endpoints
            start, end = panel.get_centerline_endpoints()
            panel_starts[i] = ti.Vector([start[0], start[1], start[2]])
            panel_ends[i] = ti.Vector([end[0], end[1], end[2]])
        
        # Clear buffers and run GPU intersection detection
        clear_intersection_buffers()
        find_all_intersections()
        ti.sync()  # Wait for GPU completion
        
        # Read results back from GPU and build intersections
        raw_intersections = []
        for i in range(n):
            for j in range(i + 1, n):
                if intersection_found[i, j]:
                    raw_intersections.append({
                        'panel1_idx': i,
                        'panel2_idx': j,
                        't1': float(intersection_t1[i, j]),
                        't2': float(intersection_t2[i, j]),
                        'position': (
                            float(intersection_x[i, j]),
                            float(intersection_y[i, j]),
                            float(intersection_z[i, j])
                        )
                    })
        
        # Build Intersection objects (CPU)
        intersections = TaichiIntersectionSolver._build_intersections(panels, raw_intersections)
        
        # Build region trees (CPU)
        regions = TaichiIntersectionSolver._build_regions(panels, intersections)
        
        return IntersectionSolution(
            panels=panels,
            intersections=intersections,
            regions_per_panel=regions
        )
    
    @staticmethod
    def _build_intersections(panels: List[GXMLPanel], 
                              raw: List[dict]) -> List[Intersection]:
        """Build Intersection objects from raw GPU results."""
        # Group by spatial proximity to merge co-located intersections
        merged: Dict[Tuple[int, int, int], List[dict]] = {}
        
        for item in raw:
            # Spatial hash key
            pos = item['position']
            key = (int(pos[0] / CELL_SIZE), int(pos[1] / CELL_SIZE), int(pos[2] / CELL_SIZE))
            if key not in merged:
                merged[key] = []
            merged[key].append(item)
        
        intersections = []
        for key, items in merged.items():
            # Average position
            n = len(items)
            avg_pos = (
                sum(i['position'][0] for i in items) / n,
                sum(i['position'][1] for i in items) / n,
                sum(i['position'][2] for i in items) / n,
            )
            
            # Collect panel entries
            panel_entries: Dict[int, Intersection.PanelEntry] = {}
            for item in items:
                idx1, idx2 = item['panel1_idx'], item['panel2_idx']
                if idx1 not in panel_entries:
                    panel_entries[idx1] = Intersection.PanelEntry(panels[idx1], item['t1'])
                if idx2 not in panel_entries:
                    panel_entries[idx2] = Intersection.PanelEntry(panels[idx2], item['t2'])
            
            # Classify intersection type
            entries = list(panel_entries.values())
            all_endpoints = all(e.is_endpoint() for e in entries)
            any_endpoints = any(e.is_endpoint() for e in entries)
            
            if all_endpoints:
                int_type = IntersectionType.JOINT
            elif any_endpoints:
                int_type = IntersectionType.T_JUNCTION
            else:
                int_type = IntersectionType.CROSSING
            
            # Sort panels CCW around intersection (for joints)
            if int_type == IntersectionType.JOINT:
                entries = TaichiIntersectionSolver._sort_panels_ccw(entries, avg_pos)
            
            intersection = Intersection(
                position=avg_pos,
                panels=entries,
                type=int_type
            )
            intersections.append(intersection)
        
        return intersections
    
    @staticmethod
    def _sort_panels_ccw(entries: List[Intersection.PanelEntry], 
                          center: Tuple[float, float, float]) -> List[Intersection.PanelEntry]:
        """Sort panel entries counter-clockwise around intersection point."""
        import math
        
        def get_angle(entry: Intersection.PanelEntry) -> float:
            # Get centerline endpoints
            start, end = entry.panel.get_centerline_endpoints()
            # Use the direction pointing away from the intersection
            if entry.t < 0.5:  # START at intersection
                dx = end[0] - start[0]
                dz = end[2] - start[2]
            else:  # END at intersection
                dx = start[0] - end[0]
                dz = start[2] - end[2]
            return math.atan2(dz, dx)
        
        return sorted(entries, key=get_angle)
    
    @staticmethod
    def _build_regions(panels: List[GXMLPanel],
                        intersections: List[Intersection]) -> Dict[GXMLPanel, Region]:
        """Build BSP region trees for each panel."""
        regions: Dict[GXMLPanel, Region] = {}
        
        for panel in panels:
            region = Region()
            
            # Find intersections for this panel and split
            for intersection in intersections:
                entry = intersection.get_entry(panel)
                if entry is not None and not entry.is_endpoint():
                    region.split_at_t(entry.t, intersection)
            
            regions[panel] = region
        
        return regions
