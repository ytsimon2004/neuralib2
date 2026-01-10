# neuralib-metric

Analysis metrics and computational modeling tools for neuroscience research, providing dimensionality reduction,
Bayesian decoding, and locomotion analysis.

## Overview

`neuralib-metric` provides Python tools for analyzing neural activity and behavior. It includes:

- **Rastermap**: Dimensionality reduction and visualization of neural recordings
- **Bayesian Decoding**: Position decoding from neural activity
- **Locomotion Analysis**: Movement epoch detection and position tracking

## Installation

### From PyPI

```bash
pip install neuralib-metric
```

### With Optional Dependencies

```bash
# Rastermap support
pip install neuralib-metric[rastermap]

# All features
pip install neuralib-metric[all]
```

### From Source (uv recommended)

```bash
git clone https://github.com/ytsimon2004/neuralib2.git
cd neuralib2/packages/neuralib-metric
uv pip install -e .        # Basic installation
uv pip install -e ".[all]" # With optional features
```

### Requirements

- Python 3.11 or 3.12
- Core dependencies: `neuralib-utils`, `numpy`, `numba`
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

- [Rastermap](https://github.com/MouseLand/rastermap) - Dimensionality reduction for neural data