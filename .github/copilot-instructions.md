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

### Core Pipeline (4 stages, in order)

1. **IntersectionSolver** - Finds where panel centerlines meet (joints, T-junctions, crossings)
2. **FaceSolver** - Determines face segmentation from intersection topology
3. **BoundsSolver** - Computes trim/gap adjustments for precise geometry
4. **GeometryBuilder** - Creates actual 3D geometry with proper mitering

### Key Concepts

- **Panel** - A rectangular element with width, height, depth; has two faces (front/back) and a centerline
- **Centerline** - The line running through the center of a panel's thickness
- **Face** - One side of a panel (front or back); faces get segmented by intersections
- **Segment** - A portion of a panel face between intersection points
- **Intersection** - Where panel centerlines meet; can be joints, T-junctions, or crossings
- **Trim/Gap** - Adjustments at panel ends and intersections for proper fit

---

## Project Structure

```
src/gxml/
├── elements/          # Core element types (Panel, Point, Polygon, etc.)
│   └── solvers/       # IntersectionSolver, FaceSolver, BoundsSolver, GeometryBuilder
├── layouts/           # Layout computation
├── mathutils/         # Math utilities (vectors, intersections, etc.)
├── render_engines/    # Different rendering backends
├── gxml_parser.py     # XML parsing
├── gxml_engine.py     # Main engine
└── gxml_types.py      # Type definitions
```

---

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
- `tests/test_fixtures/` - Shared test utilities, mocks, and assertions
- `tests/conftest.py` - Pytest fixtures

### Generating Integration Tests

Use `tests/generate_test_xml.py` to generate expected results for integration tests. It accepts XML input via stdin and outputs an expected result string that can be validated in tests. This is **more accurate than manually inferring results** because it uses the actual pipeline logic.

```bash
# Pipe XML directly to generate expected output
echo '<Root>...</Root>' | python tests/generate_test_xml.py
```

**When generating integration tests, add them directly to the relevant test file** (e.g., `tests/integration/test_integration_panel_intersections.py`) without showing the code in chat first.

### Installing for Development

```bash
pip install -e ".[dev]"
```

---

## Related Projects

### gxml-houdini

The `gxml-houdini` package provides Houdini integration for GXML. It lives in a sibling folder and contains:
- Custom render engines for Houdini
- HDA (Houdini Digital Asset) files in `otls/`
- Overrides for `gxml_engine.py` and `gxml_render.py`

---

## Code Conventions

- Solvers follow the pattern: `SolverName.solve(input) -> solution`
- Elements inherit from `GXMLBaseElement`
- Use numpy for vector/matrix math
- Type hints are encouraged (package includes `py.typed`)
