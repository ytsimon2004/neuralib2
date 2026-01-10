# neuralib-parser

Data parsers for neuroscience research tools and file formats, providing unified interfaces for segmentation,
morphology, imaging, tracking, and confocal microscopy data.

## Overview

`neuralib-parser` provides Python parsers and data structures for output from open-source packages and specific imaging
devices. It includes:

- **Segmentation**: Cellpose, StarDist cell segmentation
- **Morphology**: SWC neuron morphology files
- **Imaging**: Suite2p, Scanbox data formats
- **Tracking**: DeepLabCut, Facemap behavioral tracking
- **Confocal Scans**: Zeiss CZI, LSM file formats


## Installation

### From PyPI

```bash
pip install neuralib-parser
```

### With Optional Dependencies

```bash
# StarDist support (requires numpy<2.0)
pip install neuralib-parser[stardist]

# Suite2p integration
pip install neuralib-parser[suite2p]

# Zeiss CZI format support
pip install neuralib-parser[czi]

# Scanbox reader
pip install neuralib-parser[sbx]

# All features (excludes stardist due to numpy<2.0 constraint)
pip install neuralib-parser[all]
```

### From Source (uv recommended)

```bash
git clone https://github.com/ytsimon2004/neuralib2.git
cd neuralib2/packages/neuralib-parser
uv pip install -e .        # Basic installation
uv pip install -e ".[all]" # With optional features
```

### Requirements

- Python 3.11 or 3.12
- Core dependencies: `neuralib-utils`, `numpy`
- See `pyproject.toml` for optional dependencies

## Documentation

- **Main Documentation**: [https://neuralib2.readthedocs.io/](https://neuralib2.readthedocs.io/en/latest/index.html)
- **GitHub Repository**: [https://github.com/ytsimon2004/neuralib2](https://github.com/ytsimon2004/neuralib2)
- **Issue Tracker**: [https://github.com/ytsimon2004/neuralib2/issues](https://github.com/ytsimon2004/neuralib2/issues)

## Important Notes

### StarDist and NumPy 2.0

StarDist requires `numpy<2.0` due to upstream compatibility. The `stardist` extra explicitly pins this constraint:

## License

BSD 3-Clause License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

For bug reports and feature requests, use the [GitHub issue tracker](https://github.com/ytsimon2004/neuralib2/issues).

## Acknowledgments

This package provides interfaces to:

- [Suite2p](https://github.com/MouseLand/suite2p) - Calcium imaging data analysis
- [Scanbox](http://scanbox.org/) - Two-photon imaging system
- [StarDist](https://github.com/stardist/stardist) - Cell segmentation
- [Cellpose](https://github.com/MouseLand/cellpose) - Cell segmentation
- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) - Pose estimation
- [Facemap](https://github.com/MouseLand/facemap) - Behavioral video analysis