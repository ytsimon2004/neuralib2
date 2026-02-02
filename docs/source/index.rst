Welcome to NeuraLib2's documentation!
=====================================
**NeuraLib2** is a utility toolkit for rodent systems neuroscience research.

This is the migration of the original `neuralib <https://github.com/ytsimon2004/neuralib>`_,
aiming to split the all-in-one package into several modular packages to minimize dependencies
and enable case-specific installation.

Key Features
------------

- **Utility tools for rodent neuroscience experiments**
- **Open-source parsers and wrappers**
- **Lightweight and modular design for easy integration**
- **Clean documentation and comprehensive API reference**

Resources
---------

- `GitHub Repository <https://github.com/ytsimon2004/neuralib2>`_
- `PyPI Packages <https://pypi.org/search/?q=neuralib>`_

Installation
------------

NeuraLib2 is split into multiple packages. Install only what you need:

**Atlas Tools**

.. code-block:: bash

    $ pip install neuralib-atlas

**Imaging Analysis**

.. code-block:: bash

    $ pip install neuralib-imaging            # Basic
    $ pip install neuralib-imaging[cascade]   # With spike inference
    $ pip install neuralib-imaging[widefield] # With widefield analysis
    $ pip install neuralib-imaging[all]       # All features

**Data Parsers**

.. code-block:: bash

    $ pip install neuralib-parser           # Basic
    $ pip install neuralib-parser[suite2p]  # Suite2p support
    $ pip install neuralib-parser[czi]      # CZI format support
    $ pip install neuralib-parser[all]      # All features

**Analysis Metrics**

.. code-block:: bash

    $ pip install neuralib-metric              # Basic
    $ pip install neuralib-metric[rastermap]   # With rastermap
    $ pip install neuralib-metric[all]         # All features

**Core Utilities**

.. code-block:: bash

    $ pip install neuralib-utils              # Basic (required by all packages)
    $ pip install neuralib-utils[dashboard]   # With Bokeh dashboard
    $ pip install neuralib-utils[plot]        # With plotting tools
    $ pip install neuralib-utils[all]         # All features

**From Source**

.. code-block:: bash

    $ git clone https://github.com/ytsimon2004/neuralib2.git
    $ cd neuralib2/packages/<package-name>
    $ uv pip install -e .        # Basic installation
    $ uv pip install -e ".[all]" # With optional features

For more detailed instructions, see :doc:`installation`.

Available Packages
------------------

**neuralib-atlas** (`PyPI <https://pypi.org/project/neuralib-atlas/>`_ | `Changelog <https://github.com/ytsimon2004/neuralib2/blob/main/packages/neuralib-atlas/CHANGELOG.md>`_)
    Brain atlas tools and hierarchical representations of mouse brain anatomy.
    Provides Python interfaces to multiple mouse brain atlases, 3D visualization, and anatomical data manipulation.

    Key modules:

    - ``neuralib.atlas`` - Atlas data access and manipulation (Allen CCF, Kim, Perens, Princeton atlases)
    - ``neuralib.atlas.ccf`` - Allen Common Coordinate Framework (CCF) operations
    - ``neuralib.atlas.cellatlas`` - Cell type distributions across brain regions
    - ``neuralib.atlas.brainrender`` - 3D brain region visualization

**neuralib-imaging** (`PyPI <https://pypi.org/project/neuralib-imaging/>`_ | `Changelog <https://github.com/ytsimon2004/neuralib2/blob/main/packages/neuralib-imaging/CHANGELOG.md>`_)
    Cellular imaging and widefield imaging analysis tools.
    Provides spike inference, image registration, and widefield data processing for calcium imaging experiments.

    Key modules:

    - ``neuralib.spikes`` - Spike inference (CASCADE deep learning, OASIS deconvolution)
    - ``neuralib.registration`` - Coordinate transformation and atlas alignment
    - ``neuralib.widefield`` - Widefield imaging analysis (FFT, SVD, hemodynamic correction)

**neuralib-parser** (`PyPI <https://pypi.org/project/neuralib-parser/>`_ | `Changelog <https://github.com/ytsimon2004/neuralib2/blob/main/packages/neuralib-parser/CHANGELOG.md>`_)
    Data parsers for neuroscience tools and file formats.
    Provides unified interfaces for segmentation, morphology, imaging, tracking, and confocal microscopy data.

    Supports:

    - Cell segmentation (Cellpose, StarDist)
    - Neuron morphology (SWC files)
    - Imaging platforms (Suite2p, Scanbox)
    - Behavioral tracking (DeepLabCut, Facemap)
    - Confocal microscopy (Zeiss CZI, LSM)

**neuralib-metric** (`PyPI <https://pypi.org/project/neuralib-metric/>`_ | `Changelog <https://github.com/ytsimon2004/neuralib2/blob/main/packages/neuralib-metric/CHANGELOG.md>`_)
    Analysis metrics and computational modeling tools.
    Provides dimensionality reduction, Bayesian decoding, and locomotion analysis for neural activity and behavior.

    Features:

    - Rastermap dimensionality reduction
    - Bayesian position decoding
    - Locomotion epoch detection

**neuralib-utils** (`PyPI <https://pypi.org/project/neuralib-utils/>`_ | `Changelog <https://github.com/ytsimon2004/neuralib2/blob/main/packages/neuralib-utils/CHANGELOG.md>`_)
    Core utilities and foundational tools for the neuralib ecosystem.
    Required dependency for all other neuralib packages.

    Key modules:

    - ``neuralib.io`` - File I/O (HDF5, JSON, CSV)
    - ``neuralib.typing`` - Unified type system (pandas/polars DataFrames)
    - ``neuralib.util`` - Logging, DataFrame operations, GPU monitoring
    - ``neuralib.plot`` - Publication-quality plotting (PSTH, Venn diagrams)
    - ``neuralib.dashboard`` - Bokeh-based interactive visualization
    - ``neuralib.imglib`` - Image processing utilities
    - ``neuralib.tools`` - External integrations (Google Sheets, Slack)

Getting Started
---------------

.. toctree::
   :maxdepth: 3
   :caption: Atlas

   atlas/index

.. toctree::
   :maxdepth: 3
   :caption: Imaging

   imaging/index

.. toctree::
   :maxdepth: 3
   :caption: Other

   other/index

.. toctree::
   :maxdepth: 3
   :caption: Utility

   util/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/neuralib.rst

Command-Line Tools
------------------

neuralib_brainrender
^^^^^^^^^^^^^^^^^^^^

- Visualize brain region data with built-in rendering support
- See examples in the :doc:`atlas/brainrender`

.. code-block:: bash

    $ neuralib_brainrender -h

neuralib_widefield
^^^^^^^^^^^^^^^^^^

- Widefield imaging CLI analysis
- See examples in the :doc:`imaging/widefield`

.. code-block:: bash

    $ neuralib_widefield -h

Array Annotation Syntax
-----------------------

Used in documentation to describe array-shaped data structures:

- ``Array[DType, [*Shape]]`` where:
  - ``DType`` = data type (e.g., `int`, `float`, `bool`)
  - ``Shape`` = array shape (e.g., `[N, T]`)
  - ``|`` = denotes a union of shapes or types

**Examples:**

- Boolean or integer array with shape `(N, 3)`:

  ``Array[int|bool, [N, 3]]``

- Float array with shape `(N, 2)` or `(N, T, 2)`:

  ``Array[float, [N, 2] | [N, T, 2]]``


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
