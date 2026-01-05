# GXML Copilot Instructions

## ⚠️ CRITICAL REMINDERS

### Temp Scripts

**Always place temporary/debugging scripts in a `temp/` folder** - do NOT create them in the project root or src directories. This keeps the workspace clean and makes cleanup easy.

### XML in Command Line

**NEVER pass Python code with embedded XML strings to the command line** (e.g., `python -c "xml = '<Panel>...'"`). This always fails due to shell escaping issues. Instead:

- Write the code to a temp file and run it
- Use the test fixtures
- Load XML from a file

---

## Project Overview

GXML is a Python library for geometric XML layout, panel intersection solving, and geometry generation.

### Key Concepts

- **Panel** - A rectangular element with width, height, depth; has two faces (front/back) and a centerline
- **Centerline** - The line running through the center of a panel's thickness
- **Face** - One side of a panel (front or back); faces get segmented by intersections
- **Segment** - A portion of a panel face between intersection points
- **Intersection** - Where panel centerlines meet; can be joints, T-junctions, or crossings
- **Trim/Gap** - Adjustments at panel ends and intersections for proper fit---

## Development

### Running Tests

```bash
pytest                           # All tests
pytest tests/unit                # Unit tests only
pytest tests/integration         # Integration tests only
pytest -v                        # Verbose output
pytest --tb=short                # Shorter tracebacks
```

### Test Organization

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for full pipeline
- `tests/performance/` - Performance benchmarks (XML files in `xml/` subfolder)
- `tests/test_fixtures/` - Shared test utilities, mocks, assertions, and profiling
- `tests/conftest.py` - Pytest fixtures

### Performance Testing

**Always run the official performance test after making changes that could affect performance:**

```bash
# Quick regression check via pytest
pytest tests/performance/ -v

# Detailed profiling with breakdown (PREFERRED - shows timing table)
python tests/performance/test_perf_pipeline.py
```

**When running performance tests, prefer running the script directly** (`python tests/performance/test_perf_pipeline.py`) rather than via pytest. The script outputs a detailed comparison table showing per-stage timing (Validate, Parse, Layout, Render) and sub-stage breakdown (Intersection Solver, Face Solver, Geometry Builder) for each backend.

This test runs the full end-to-end pipeline (parse → layout → render) on multiple test files (7, 16, 75, 200 panels) and compares CPU vs C extension backends.

**Pytest tests will fail if performance regresses >20% from baseline** (e.g., 700ms baseline × 1.2 = 840ms max for 200 panels).

Key metrics:
- **7 panels**: ~3-5ms baseline
- **16 panels**: ~8-15ms baseline  
- **75 panels**: ~70-100ms baseline
- **200 panels**: ~650-700ms baseline (C extension similar or slightly faster)
- **200 panels, ~1200 intersections, ~10,800 polygons**

### Generating Integration Tests

Use `tests/generate_test_xml.py` to generate expected results for integration tests. It accepts XML input via stdin and outputs an expected result string that can be validated in tests. This is **more accurate than manually inferring results** because it uses the actual pipeline logic.

```bash
# Pipe XML directly to generate expected output
echo '<Root>...</Root>' | python tests/generate_test_xml.py
```

**When generating integration tests, add them directly to the relevant test file** (e.g., `tests/integration/test_integration_panel_intersections.py`) without showing the code in chat first.

```---

## Related Projects

### gxml-houdini

The `gxml-houdini` package provides Houdini integration for GXML. It lives in a sibling folder and contains:

- Custom render engines for Houdini
- HDA (Houdini Digital Asset) files in `otls/`
- Overrides for `gxml_engine.py` and `gxml_render.py`
- Includes gxml as a submodule which can be ignored entirely unless we're doing houdini work.

---

## Code Conventions

- Solvers follow the pattern: `SolverName.solve(input) -> solution`
- Elements inherit from `GXMLBaseElement`
- Use numpy for vector/matrix math
- Type hints are encouraged (package includes `py.typed`)
