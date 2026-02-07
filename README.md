# neuralib2

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-brightgreen.svg)](https://neuralib2.readthedocs.io/)

**Package Versions:**

[![neuralib-atlas](https://img.shields.io/pypi/v/neuralib-atlas.svg?label=neuralib-atlas)](https://pypi.org/project/neuralib-atlas/)
[![neuralib-imaging](https://img.shields.io/pypi/v/neuralib-imaging.svg?label=neuralib-imaging)](https://pypi.org/project/neuralib-imaging/)
[![neuralib-parser](https://img.shields.io/pypi/v/neuralib-parser.svg?label=neuralib-parser)](https://pypi.org/project/neuralib-parser/)
[![neuralib-metric](https://img.shields.io/pypi/v/neuralib-metric.svg?label=neuralib-metric)](https://pypi.org/project/neuralib-metric/)
[![neuralib-utils](https://img.shields.io/pypi/v/neuralib-utils.svg?label=neuralib-utils)](https://pypi.org/project/neuralib-utils/)


---

Utility toolkit for rodent systems neuroscience research

This module is the migration of the original [neuralib](https://github.com/ytsimon2004/neuralib), aiming to split the
all-in-one package into several modular packages to minimize dependencies and enable case-specific installation.


---

## 📖 Documentation

> **[Read the Full Documentation](https://neuralib2.readthedocs.io/)** | *
*[API Reference](https://neuralib2.readthedocs.io/en/latest/api/neuralib.html)**

---

## Packages & Installation

### Quick Install

Install individual packages based on your needs:

```bash
# Atlas tools
pip install neuralib-atlas

# Imaging analysis
pip install neuralib-imaging

# Data parsers
pip install neuralib-parser

# Analysis metrics
pip install neuralib-metric

# Core utilities (dependency for all packages)
pip install neuralib-utils
```

### Install from Source

```bash
git clone https://github.com/ytsimon2004/neuralib2.git
cd neuralib2/packages/<package-name>
uv pip install -e .        # Basic installation
uv pip install -e ".[all]" # With optional features
```

---

### Available Packages

#### **[neuralib-atlas](packages/neuralib-atlas/README.md)** | [PyPI](https://pypi.org/project/neuralib-atlas/)

Brain atlas tools and hierarchical representations of mouse brain anatomy. Provides Python interfaces to multiple mouse
brain atlases, 3D visualization, and anatomical data manipulation.

**Key modules:**

- `neuralib.atlas` - Atlas data access and manipulation (Allen CCF, Kim, Perens, Princeton atlases)
- `neuralib.atlas.ccf` - Allen Common Coordinate Framework (CCF) operations
- `neuralib.atlas.cellatlas` - Cell type distributions across brain regions
- `neuralib.atlas.brainrender` - 3D brain region visualization

```bash
pip install neuralib-atlas
```

#### **[neuralib-imaging](packages/neuralib-imaging/README.md)** | [PyPI](https://pypi.org/project/neuralib-imaging/)

Cellular imaging and widefield imaging analysis tools. Provides spike inference, image registration, and widefield data
processing for calcium imaging experiments.

**Key modules:**

- `neuralib.spikes` - Spike inference (CASCADE deep learning, OASIS deconvolution)
- `neuralib.registration` - Coordinate transformation and atlas alignment
- `neuralib.widefield` - Widefield imaging analysis (FFT, SVD, hemodynamic correction)

```bash
pip install neuralib-imaging            # Basic
pip install neuralib-imaging[cascade]   # With spike inference
pip install neuralib-imaging[widefield] # With widefield analysis
pip install neuralib-imaging[all]       # All features
```

#### **[neuralib-parser](packages/neuralib-parser/README.md)** | [PyPI](https://pypi.org/project/neuralib-parser/)

Data parsers for neuroscience tools and file formats. Provides unified interfaces for segmentation, morphology, imaging,
tracking, and confocal microscopy data.

**Supports:**

- Cell segmentation (Cellpose, StarDist)
- Neuron morphology (SWC files)
- Imaging platforms (Suite2p, Scanbox)
- Behavioral tracking (DeepLabCut, Facemap)
- Confocal microscopy (Zeiss CZI, LSM)

```bash
pip install neuralib-parser           # Basic
pip install neuralib-parser[suite2p]  # Suite2p support
pip install neuralib-parser[czi]      # CZI format support
pip install neuralib-parser[all]      # All features
```

#### **[neuralib-metric](packages/neuralib-metric/README.md)** | [PyPI](https://pypi.org/project/neuralib-metric/)

Analysis metrics and computational modeling tools. Provides dimensionality reduction, Bayesian decoding, and locomotion
analysis for neural activity and behavior.

**Features:**

- Rastermap dimensionality reduction
- Bayesian position decoding
- Locomotion epoch detection

```bash
pip install neuralib-metric              # Basic
pip install neuralib-metric[rastermap]   # With rastermap
pip install neuralib-metric[all]         # All features
```

#### **[neuralib-utils](packages/neuralib-utils/README.md)** | [PyPI](https://pypi.org/project/neuralib-utils/)

Core utilities and foundational tools for the neuralib ecosystem. Required dependency for all other neuralib packages.

**Key modules:**

- `neuralib.io` - File I/O (HDF5, JSON, CSV)
- `neuralib.typing` - Unified type system (pandas/polars DataFrames)
- `neuralib.util` - Logging, DataFrame operations, GPU monitoring
- `neuralib.plot` - Publication-quality plotting (PSTH, Venn diagrams)
- `neuralib.dashboard` - Bokeh-based interactive visualization
- `neuralib.imglib` - Image processing utilities
- `neuralib.tools` - External integrations (Google Sheets, Slack)

```bash
pip install neuralib-utils              # Basic
pip install neuralib-utils[dashboard]   # With Bokeh dashboard
pip install neuralib-utils[plot]        # With plotting tools
pip install neuralib-utils[all]         # All features
```
