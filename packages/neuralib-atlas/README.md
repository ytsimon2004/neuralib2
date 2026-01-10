# neuralib-atlas

Tools for hierarchical representations of the mouse brain anatomy, providing Python interfaces to brain atlases, 3D
visualization, and anatomical data manipulation.

## Overview

`neuralib-atlas` is a modular Python package for working with mouse brain atlases in systems neuroscience research. It
provides:

- **Atlas Data Access**: Load and query multiple mouse brain atlases (Allen CCF, Kim, Perens, Princeton) via
  BrainGlobeAtlas API
- **3D Visualization**: Interactive brain region rendering using BrainRender
- **CCF Tools**: DataFrame-based manipulation of Allen Common Coordinate Framework (CCF) annotations
- **Anatomical Mapping**: Coordinate transformations and region hierarchy navigation
- **CLI Tools**: Command-line interfaces for quick visualization tasks

## Features

### Atlas Support

Access to multiple mouse brain atlases at various resolutions:

- Allen Mouse Brain Atlas (10um, 25um, 50um, 100um)
- Kim Mouse Atlas (10um, 25um, 50um, 100um)
- Perens LSFM Mouse (20um)
- Perens Stereotaxic Mouse MRI (25um)
- Princeton Mouse (20um)

### 3D Brain Rendering

- Render specific brain regions with customizable colors and transparency
- Visualize region-of-interest (ROI) coordinates from experimental data
- Display probe placements with depth specifications
- Export high-quality images and interactive scenes

### CCF Dataframe Operations

- Load and manipulate Allen CCF structure hierarchies as Polars DataFrames
- Navigate parent-child relationships in the anatomical tree
- Filter regions by criteria (cortex, subcortex, specific hierarchies)
- Map between annotation IDs and anatomical structures

### Cell Atlas Integration

Query and analyze cell type distributions across brain regions using integrated cell atlas data.

## Installation

### From PyPI

```bash
pip install neuralib-atlas
```

### From Source (uv venv recommended)

```bash
git clone https://github.com/ytsimon2004/neuralib.git
cd neuralib/packages/neuralib-atlas
uv pip install -e . # if using uv
pip install -e .  # if using conda env
```

### Requirements

- Python 3.11 or 3.12
- Dependencies: `brainrender`, `pynrrd`, `openpyxl`, `anytree`, `plotly`, `fastexcel`
- See `pyproject.toml` for full dependency list

## Quick Start

### Loading Atlas Data

```python
from neuralib.atlas.data import load_bg_structure_tree, load_bg_volumes

# Load Allen CCF structure hierarchy
structures = load_bg_structure_tree('allen_mouse_10um')

# Load annotation and reference volumes
annotation, reference = load_bg_volumes('allen_mouse_10um')
```

### 3D Brain Visualization

```python
from neuralib.atlas.brainrender import BrainRenderScene

# Create a scene and add brain regions
scene = BrainRenderScene()
scene.add_brain_region(['VISp', 'VISl', 'VISal'], alpha=0.8)
scene.render()
```

### CCF Dataframe Manipulation

```python
from neuralib.atlas.ccf import CCFDataFrame

# Load CCF structure tree as DataFrame
ccf = CCFDataFrame.from_brainglobe('allen_mouse_10um')

# Query specific regions
visual_cortex = ccf.filter_by_acronym(['VISp', 'VISl', 'VISal'])

# Get all children of a region
vis_children = ccf.get_children('VIS')
```

## CLI Tools

### Render Brain Regions

```bash
# Render visual cortex areas
neuralib_brainrender area -R VISp,VISl,VISal,VISam

# Render with custom colors
neuralib_brainrender area -R SSp,SSs --color red,blue
```

### Render ROI from File

```bash
# Visualize coordinates from experimental data
neuralib_brainrender roi -F roi_coordinates.csv
```

### Render Probe Placement

```bash
# Display probe track with specified depth
neuralib_brainrender probe -F probe_coords.csv --depth 3000
```

## Documentation

- **Main Documentation**: [https://neuralib2.readthedocs.io/](https://neuralib.readthedocs.io/en/latest/index.html)
- **GitHub Repository**: [https://github.com/ytsimon2004/neuralib2](https://github.com/ytsimon2004/neuralib2)
- **Issue Tracker**: [https://github.com/ytsimon2004/neuralib2/issues](https://github.com/ytsimon2004/neuralib/issues)

## Related Packages

`neuralib-atlas` is part of the neuralib ecosystem:

- `neuralib-utils`: Core utilities and I/O functions (required dependency)
- Other neuralib packages: imaging, segmentation, tracking, ephys

## Development

### Running Tests

```bash
pytest test/
```

### Code Style

This package follows the neuralib coding conventions:

- Ruff linter with 120-character line length
- Type hints for public APIs
- NumPy-style docstrings

## License

BSD 3-Clause License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

For bug reports and feature requests, use the [GitHub issue tracker](https://github.com/ytsimon2004/neuralib2/issues).

## Acknowledgments

This package builds on:

- [BrainRender](https://github.com/brainglobe/brainrender) - 3D visualization of brain data
- [BrainGlobe Atlas API](https://github.com/brainglobe/brainglobe-atlasapi) - Unified atlas access
- [Allen Brain Atlas](https://atlas.brain-map.org/) - Mouse brain reference atlases