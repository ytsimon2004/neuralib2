# neuralib-utils

Core utilities and foundational tools for the neuralib ecosystem, providing I/O operations, data manipulation, type
definitions, visualization helpers, and common infrastructure for neuroscience research workflows.

## Overview

`neuralib-utils` is the foundational package for neuralib, providing essential utilities that other neuralib packages
depend on. It includes:

- **File I/O**: Efficient HDF5 lazy loading, JSON/CSV handling, dataset management
- **Type System**: Unified type aliases supporting both pandas and polars DataFrames
- **Utilities**: Logging, colored output, DataFrame operations, GPU monitoring, profiling
- **Plotting**: Publication-quality plotting tools (PSTH, Venn diagrams, colormaps)
- **Dashboard**: Bokeh-based web UI framework for interactive data visualization
- **Image Utilities**: Array manipulation, labeling, color management, normalization
- **External Tools**: Google Sheets and Slack integrations

This package serves as the core dependency for all other neuralib packages (`neuralib-atlas`, `neuralib-imaging`, etc.).

## Installation

### From PyPI

```bash
pip install neuralib-utils
```

### With Optional Dependencies

```bash
# Dashboard support (Bokeh)
pip install neuralib-utils[dashboard]

# Plotting support (matplotlib, seaborn)
pip install neuralib-utils[plot]

# Optional tools (Google Sheets, Slack, GPU monitoring)
pip install neuralib-utils[opt]

# All features
pip install neuralib-utils[all]
```

### From Source (uv recommended)

```bash
git clone https://github.com/ytsimon2004/neuralib2.git
cd neuralib2/packages/neuralib-utils
uv pip install -e .        # Basic installation
uv pip install -e ".[all]" # With all features
```

### Requirements

- Python 3.11 or 3.12
- Core dependencies: `numpy`, `scipy`, `pandas`, `polars`, `h5py`, `rich`, `tqdm`
- See `pyproject.toml` for full dependency list

## Documentation

- **Main Documentation**: [https://neuralib2.readthedocs.io/](https://neuralib2.readthedocs.io/en/latest/index.html)
- **GitHub Repository**: [https://github.com/ytsimon2004/neuralib2](https://github.com/ytsimon2004/neuralib2)
- **Issue Tracker**: [https://github.com/ytsimon2004/neuralib2/issues](https://github.com/ytsimon2004/neuralib2/issues)

## Related Packages

`neuralib-utils` is the foundation for the neuralib ecosystem:

- `neuralib-atlas`: Brain atlas tools and hierarchical representations of mouse brain anatomy
- `neuralib-imaging`: Cellular imaging and widefield imaging tools
- `neuralib-parser`: Data parsers for segmentation, morphology, Suite2p, Scanbox, CZI, StarDist
- `neuralib-metric`: Analysis metrics and computational tools

## License

BSD 3-Clause License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

For bug reports and feature requests, use the [GitHub issue tracker](https://github.com/ytsimon2004/neuralib2/issues).
