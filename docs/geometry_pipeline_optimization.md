# Geometry Pipeline Optimization Proposal

## Current Pipeline (3-stage)

```
Panels → IntersectionSolver → FaceSolver → GeometryBuilder → Render
              (8%)              (39%)           (53%)
```

### Stage 2: FaceSolver
**Output:** `List[SegmentedPanel]` containing:
- `FaceSegment` objects with `corners: List[(t, s)]` (4 tuples per segment)
- ~10,856 face segments for 200 panels

### Stage 3: GeometryBuilder  
**Input:** `List[SegmentedPanel]`
**Operations:**
1. Iterate all `SegmentedPanel` and their `FaceSegment`s
2. For each segment, call `panel.create_panel_side(name, face_side, corners)`
3. `create_panel_side()` does:
   - Create `GXMLQuad` object
   - Map `(t, s)` corners to local 3D coords
   - `batch_bilinear_transform()` to world coords (C extension)
   - Compute affine matrix from quad
   - Store in `quad._cached_world_vertices`

### The Overhead Problem
- **Object creation:** ~10,000 `FaceSegment` objects, ~10,000 `GXMLQuad` objects
- **Python dict lookups:** `segments.get(face_side)` per panel × 6 faces
- **Intermediate (t, s) storage:** Only used to pass to `get_face_point()` / `create_panel_side()`
- **Affine matrix computation:** Done per quad, but only needed for CSS selectors

---

## Proposed: Two-Path Architecture

### Path A: Fast Mesh Builder (GPU-compatible)
Direct vertex buffer generation, no intermediate objects.

```python
class FastMeshBuilder:
    """Writes directly to vertex buffer, no intermediate objects."""
    
    @staticmethod
    def build_mesh(panels: List[GXMLPanel], 
                   intersection_solution: IntersectionSolution) -> MeshData:
        """
        Single-pass mesh generation.
        
        Returns:
            MeshData with vertex_buffer, index_buffer, panel_ids
        """
        # Pre-allocate buffers based on estimated face count
        # ~6 faces × 4 verts per panel, plus extra for splits
        estimated_verts = len(panels) * 6 * 4 * 2  # 2x safety margin
        
        vertex_buffer = np.empty((estimated_verts, 3), dtype=np.float32)
        index_buffer = []
        panel_ids = []  # For hit testing
        
        vert_idx = 0
        for panel in panels:
            region = intersection_solution.regions_per_panel.get(panel)
            leaf_bounds = region.get_leaf_bounds() if region else []
            
            # Directly compute vertices for each face
            for face_side in PanelSide:
                corners_list = _compute_face_corners(panel, face_side, leaf_bounds)
                
                for corners in corners_list:
                    # Write 4 vertices directly to buffer
                    world_pts = panel.batch_get_face_points(face_side, corners)
                    vertex_buffer[vert_idx:vert_idx+4] = world_pts
                    
                    # Record quad indices
                    index_buffer.extend([vert_idx, vert_idx+1, vert_idx+2,
                                        vert_idx, vert_idx+2, vert_idx+3])
                    panel_ids.append(panel.id)
                    vert_idx += 4
        
        return MeshData(
            vertices=vertex_buffer[:vert_idx],
            indices=np.array(index_buffer, dtype=np.uint32),
            panel_ids=panel_ids
        )
```

**Key optimizations:**
- No `FaceSegment` or `GXMLQuad` object creation
- No intermediate `(t, s)` storage—compute world coords inline
- Single numpy buffer allocation
- Could be ported to GPU (Taichi/CUDA)

### Path B: Rich Object Model (for CSS queries)
Keeps current architecture but lazily evaluated.

```python
class RichGeometryBuilder:
    """Creates queryable object model for CSS-style selectors."""
    
    @staticmethod  
    def build(panels, intersection_solution) -> GeometryModel:
        """
        Creates full object hierarchy for queries like:
        - panel[id="wall-1"] > face[side="front"]
        - panel:intersects(panel[id="wall-2"])
        """
        # Current FaceSolver + GeometryBuilder logic
        # But with lazy world-coord computation
        pass
```

---

## Implementation Plan

### Phase 1: FastMeshBuilder (immediate win)
1. Add `batch_get_face_points(face_side, corners_list)` to GXMLPanel
2. Create `FastMeshBuilder` that combines face/bounds computation with vertex output
3. Skip `GXMLQuad` creation entirely for fast path
4. Return `MeshData` struct for direct Three.js consumption

### Phase 2: GPU Acceleration  
1. Port `FastMeshBuilder` core loop to Taichi
2. Upload panel transforms + intersection data to GPU
3. Compute all vertices in parallel
4. Download vertex buffer (or use shared memory)

### Phase 3: Lazy Rich Model
1. Make `FaceSegment.get_world_corners()` lazy (already is)
2. Add CSS-like selector API
3. Only compute what's queried

---

## Expected Performance

| Stage | Current | Fast Path | Speedup |
|-------|---------|-----------|---------|
| FaceSolver | 200ms | 0ms (merged) | ∞ |
| GeometryBuilder | 268ms | ~50ms | 5x |
| **Total Solver** | 468ms | ~50ms | **9x** |

With GPU acceleration, could potentially hit <10ms for 200 panels.

---

## Data Flow Comparison

### Current:
```
panels → IntersectionSolution → Dict[Panel, Region]
                                       ↓
                               List[SegmentedPanel]  ← FaceSolver creates these
                                       ↓
                               for segment in segments:
                                   panel.create_panel_side(corners)  ← GeometryBuilder
                                       ↓
                                   GXMLQuad objects stored in panel.dynamicChildren
```

### Proposed Fast Path:
```
panels → IntersectionSolution → Dict[Panel, Region]
                                       ↓
                               FastMeshBuilder.build_mesh()
                                       ↓
                               MeshData(vertices, indices, panel_ids)
                                       ↓
                               Direct to Three.js BufferGeometry
```

---

## API Design

```python
from gxml.elements.solvers import solve_geometry, SolverMode

# Fast mode - returns raw mesh
mesh = solve_geometry(panels, mode=SolverMode.FAST_MESH)
# mesh.vertices: np.ndarray (N, 3)
# mesh.indices: np.ndarray (M,)
# mesh.panel_ids: List[str]

# Rich mode - returns queryable model  
model = solve_geometry(panels, mode=SolverMode.RICH_MODEL)
# model.query("panel[id='wall-1'] > face[side='front']")
# model.get_panel_faces(panel_id)
```
