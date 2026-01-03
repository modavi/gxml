# C Extension Solver Architecture for GXML

## Summary

A C extension can provide **2,000-10,000x speedup** for the intersection solver phase by eliminating Python overhead.

### Benchmark Results (200 panels, 10,000 intersections)

| Approach | Time | Speedup vs Python |
|----------|------|-------------------|
| **C Extension** | **0.60 ms** | **2,722x** |
| NumPy Batch | 20.22 ms | 80x |
| Python Solver | 1,626 ms | 1x (baseline) |

## Why C is So Much Faster

The Python solver spends almost all its time in overhead:
1. **Object creation**: Each intersection creates Python objects (Intersection, PanelEntry)
2. **Method calls**: Python method dispatch overhead for every operation  
3. **Memory allocation**: Python's memory allocator is general-purpose, not optimized
4. **GIL**: Even single-threaded code pays GIL overhead

The C extension:
1. Uses contiguous C arrays - no Python object overhead
2. Direct memory access - no method dispatch
3. Stack allocation for temporaries - no heap overhead
4. Can use SIMD (SSE/AVX/NEON) for parallel math

## Proposed Architecture

### Data Flow Without Marshaling

```
Python                          C Extension
────────                        ───────────
panels[] ──────────────────────► c_solver_context*
  (just extract endpoints)        │
                                  ├─► batch_find_intersections()
                                  │     (stays in C memory)
                                  │
                                  ├─► batch_solve_faces()
                                  │     (uses intersection results directly)
                                  │
                                  ├─► batch_build_geometry()
                                  │     (generates vertices/indices)
                                  │
                                  ▼
                             numpy arrays ◄── Only marshal at the END
```

### Key Insight: Opaque Handles

The C extension can return an opaque handle (pointer wrapped in PyCapsule) that:
- Stays in C memory between solver stages
- Contains all intermediate results
- Only converts to Python objects at the very end

```python
# Python code
context = c_solvers.create_context(starts, ends, transforms)
c_solvers.solve_intersections(context)  # Modifies context in-place
c_solvers.solve_faces(context)          # Uses intersection data in context
vertices, indices = c_solvers.build_geometry(context)  # Final marshaling
c_solvers.free_context(context)
```

### C Extension Structure

```
src/gxml/elements/solvers/
├── _c_solvers.c           # Main C extension (this file exists!)
│   ├── batch_find_intersections()  ✓ Implemented
│   ├── batch_solve_faces()         TODO
│   ├── batch_build_geometry()      TODO
│   └── Context management          TODO
└── c_intersection_solver.py        # Python wrapper (TODO)
```

## Implementation Plan

### Phase 1: Intersection Solver (Done!)
- [x] `batch_find_intersections(starts, ends)` → indices, t-values, positions
- [x] Benchmark showing 2,000x+ speedup

### Phase 2: Full Pipeline Integration
- [ ] Create `CIntersectionSolver` class that wraps the C extension
- [ ] Match the `IntersectionSolution` API for drop-in replacement
- [ ] Build Python objects only when needed (lazy conversion)

### Phase 3: Face Solver in C
- [ ] Port face segmentation logic to C
- [ ] Keep face data in C memory between stages
- [ ] Return corner coordinates as numpy arrays

### Phase 4: Geometry Builder in C
- [ ] Port vertex generation to C
- [ ] Return vertices/indices as numpy arrays
- [ ] This is the final stage - perfect for direct GPU upload

### Phase 5: SIMD Optimization (Optional)
- [ ] Use SSE/AVX for x86 batch operations
- [ ] Use NEON for ARM (Apple Silicon)
- [ ] Infrastructure already exists in `_vec3.c`

## Comparison: C Extension vs Taichi

| Aspect | C Extension | Taichi |
|--------|-------------|--------|
| Setup overhead | ~0.01 ms | ~500 ms (field creation) |
| Per-call overhead | Minimal | ~100ms (upload/download) |
| GPU support | Not directly | Yes (when it works) |
| Metal support | Via Accelerate | Broken on macOS 14+ |
| Code complexity | Higher | Lower |
| Maintainability | Needs C expertise | Pure Python |

**Recommendation**: Use C extension for CPU path. Taichi is only useful when GPU actually works.

## Files Created

- `src/gxml/elements/solvers/_c_solvers.c` - C extension implementation
- `setup_c_solvers.py` - Build script
- `temp/benchmark_c_extension.py` - Benchmark script
- `temp/batch_intersection_prototype.py` - NumPy prototype
