# neuralib-imaging

Tools for cellular imaging and widefield imaging analysis, providing spike inference, image registration, and widefield
data processing for neuroscience research.

## Overview

`neuralib-imaging` provides Python tools for analyzing calcium imaging and widefield imaging data. It includes:

- **Spike Inference**: CASCADE (deep learning), OASIS deconvolution
- **Registration**: Coordinate transformation and atlas alignment
- **Widefield Analysis**: FFT processing, SVD decomposition, hemodynamic correction
- **Visualization**: Interactive napari-based viewers and plotting tools

## Installation

### From PyPI

```bash
pip install neuralib-imaging
```

### With Optional Dependencies

```bash
# CASCADE spike inference (requires TensorFlow)
pip install neuralib-imaging[cascade]

# Widefield analysis (requires napari, bokeh, scikit-learn)
pip install neuralib-imaging[widefield]

# All features
pip install neuralib-imaging[all]
```

### From Source (uv recommended)

```bash
git clone https://github.com/ytsimon2004/neuralib2.git
cd neuralib2/packages/neuralib-imaging
uv pip install -e .        # Basic installation
uv pip install -e ".[all]" # With optional features
```

### Requirements

- Python 3.11 or 3.12
- Core dependencies: `neuralib-utils`, `numpy`, `numba`, `matplotlib`
- See `pyproject.toml` for optional dependencies

## Documentation

- **Main Documentation**: [https://neuralib2.readthedocs.io/](https://neuralib2.readthedocs.io/en/latest/index.html)
- **GitHub Repository**: [https://github.com/ytsimon2004/neuralib2](https://github.com/ytsimon2004/neuralib2)
- **Issue Tracker**: [https://github.com/ytsimon2004/neuralib2/issues](https://github.com/ytsimon2004/neuralib2/issues)

## License

BSD 3-Clause License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

For bug reports and feature requests, use the [GitHub issue tracker](https://github.com/ytsimon2004/neuralib2/issues).

## Acknowledgments

This package integrates with:

- [CASCADE](https://github.com/HelmchenLabSoftware/Cascade) - Deep learning spike inference
- [napari](https://napari.org/) - Multi-dimensional image viewer