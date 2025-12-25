# GXML

A Python library for geometric XML layout, panel intersection solving, and geometry generation.

## Features

- **XML-based layout definition** - Define complex panel layouts using declarative XML
- **Intersection solving** - Compute where panel centerlines meet (joints, T-junctions, crossings)
- **Face segmentation** - Determine how panel faces are subdivided by intersections
- **Bounds computation** - Calculate trim/gap adjustments for precise geometry
- **Geometry generation** - Create 3D panel geometry with proper mitering and gaps

## Installation

```bash
pip install gxml
```

## Quick Start

```python
from gxml import GXMLParser
from gxml.elements.solvers import IntersectionSolver, FaceSolver, BoundsSolver

# Parse XML layout
parser = GXMLParser()
root = parser.parse_file("layout.xml")

# Extract panels
panels = root.get_panels()

# Solve geometry
intersection_solution = IntersectionSolver.solve(panels)
face_result = FaceSolver.solve(intersection_solution)
bounds_solution = BoundsSolver.solve(face_result)

# Access computed geometry
for panel in panels:
    segments = bounds_solution.get_segments(panel, face_side)
    corners = bounds_solution.get_face_corners(panel, face_side)
```

## Pipeline Overview

The geometry pipeline has 4 stages:

1. **IntersectionSolver** - Finds where panel centerlines meet
2. **FaceSolver** - Determines face segmentation from intersection topology
3. **BoundsSolver** - Computes trim/gap adjustments
4. **GeometryBuilder** - Creates actual 3D geometry

## Documentation

Full documentation available at: [https://github.com/yourusername/gxml/docs](https://github.com/yourusername/gxml/docs)

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/gxml.git
cd gxml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.
