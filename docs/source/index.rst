NeuraLib2
=========

| |atlas| |imaging| |parser| |metric| |utils| |license|

.. |atlas| image:: https://img.shields.io/pypi/v/neuralib-atlas.svg?label=atlas
   :target: https://pypi.org/project/neuralib-atlas/

.. |imaging| image:: https://img.shields.io/pypi/v/neuralib-imaging.svg?label=imaging
   :target: https://pypi.org/project/neuralib-imaging/

.. |parser| image:: https://img.shields.io/pypi/v/neuralib-parser.svg?label=parser
   :target: https://pypi.org/project/neuralib-parser/

.. |metric| image:: https://img.shields.io/pypi/v/neuralib-metric.svg?label=metric
   :target: https://pypi.org/project/neuralib-metric/

.. |utils| image:: https://img.shields.io/pypi/v/neuralib-utils.svg?label=utils
   :target: https://pypi.org/project/neuralib-utils/

.. |license| image:: https://img.shields.io/github/license/ytsimon2004/neuralib2
   :target: https://github.com/ytsimon2004/neuralib2/blob/main/LICENSE

**A modular Python toolkit for rodent systems neuroscience research.**

NeuraLib2 is a collection of lightweight, modular packages designed for neuroscience data analysis.
It is the successor to `neuralib <https://github.com/ytsimon2004/neuralib>`_, split into focused packages
to minimize dependencies and enable case-specific installation.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Atlas Tools
      :link: atlas/index
      :link-type: doc

      Brain atlas interfaces, 3D visualization, and anatomical data manipulation.

   .. grid-item-card:: Imaging Analysis
      :link: imaging/index
      :link-type: doc

      Spike inference, image registration, and widefield imaging analysis.

   .. grid-item-card:: Data Parsers
      :link: parser/index
      :link-type: doc

      Unified interfaces for segmentation, tracking, and microscopy data.

   .. grid-item-card:: Analysis Metrics
      :link: metric/index
      :link-type: doc

      Dimensionality reduction, Bayesian decoding, and locomotion analysis.

----

Installation
------------

Install only the packages you need:

.. tab-set::

   .. tab-item:: Atlas

      .. code-block:: bash

         pip install neuralib-atlas

   .. tab-item:: Imaging

      .. code-block:: bash

         pip install neuralib-imaging            # Basic
         pip install neuralib-imaging[cascade]   # With spike inference
         pip install neuralib-imaging[widefield] # With widefield analysis
         pip install neuralib-imaging[all]       # All features

   .. tab-item:: Parser

      .. code-block:: bash

         pip install neuralib-parser           # Basic
         pip install neuralib-parser[suite2p]  # Suite2p support
         pip install neuralib-parser[czi]      # CZI format support
         pip install neuralib-parser[all]      # All features

   .. tab-item:: Metric

      .. code-block:: bash

         pip install neuralib-metric              # Basic
         pip install neuralib-metric[rastermap]   # With rastermap
         pip install neuralib-metric[all]         # All features

   .. tab-item:: Utils

      .. code-block:: bash

         pip install neuralib-utils              # Basic (required by all)
         pip install neuralib-utils[dashboard]   # With Bokeh dashboard
         pip install neuralib-utils[plot]        # With plotting tools
         pip install neuralib-utils[all]         # All features

   .. tab-item:: From Source

      .. code-block:: bash

         git clone https://github.com/ytsimon2004/neuralib2.git
         cd neuralib2/packages/<package-name>
         uv pip install -e .        # Basic
         uv pip install -e ".[all]" # With all features

----

Packages
--------

neuralib-atlas
^^^^^^^^^^^^^^

`PyPI <https://pypi.org/project/neuralib-atlas/>`_ ·
`Changelog <https://github.com/ytsimon2004/neuralib2/blob/main/packages/neuralib-atlas/CHANGELOG.md>`_

Brain atlas tools and hierarchical representations of mouse brain anatomy.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - ``neuralib.atlas``
     - Atlas data access (Allen CCF, Kim, Perens, Princeton)
   * - ``neuralib.atlas.ccf``
     - Allen Common Coordinate Framework operations
   * - ``neuralib.atlas.cellatlas``
     - Cell type distributions across brain regions
   * - ``neuralib.atlas.brainrender``
     - 3D brain region visualization

neuralib-imaging
^^^^^^^^^^^^^^^^

`PyPI <https://pypi.org/project/neuralib-imaging/>`_ ·
`Changelog <https://github.com/ytsimon2004/neuralib2/blob/main/packages/neuralib-imaging/CHANGELOG.md>`_

Cellular imaging and widefield imaging analysis tools.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - ``neuralib.spikes``
     - Spike inference (CASCADE, OASIS deconvolution)
   * - ``neuralib.registration``
     - Coordinate transformation and atlas alignment
   * - ``neuralib.widefield``
     - Widefield imaging (FFT, SVD, hemodynamic correction)

neuralib-parser
^^^^^^^^^^^^^^^

`PyPI <https://pypi.org/project/neuralib-parser/>`_ ·
`Changelog <https://github.com/ytsimon2004/neuralib2/blob/main/packages/neuralib-parser/CHANGELOG.md>`_

Data parsers for neuroscience tools and file formats.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - ``neuralib.cellpose``
     - Cellpose segmentation result parser
   * - ``neuralib.stardist``
     - StarDist segmentation parser and CLI
   * - ``neuralib.suite2p``
     - Suite2p imaging data parser
   * - ``neuralib.scanbox``
     - Scanbox imaging data parser
   * - ``neuralib.deeplabcut``
     - DeepLabCut tracking parser
   * - ``neuralib.facemap``
     - Facemap behavioral tracking parser
   * - ``neuralib.morpho``
     - Neuron morphology (SWC files)
   * - ``neuralib.scan``
     - Confocal microscopy (Zeiss CZI, LSM)

neuralib-metric
^^^^^^^^^^^^^^^

`PyPI <https://pypi.org/project/neuralib-metric/>`_ ·
`Changelog <https://github.com/ytsimon2004/neuralib2/blob/main/packages/neuralib-metric/CHANGELOG.md>`_

Analysis metrics and computational modeling tools.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - Rastermap
     - Dimensionality reduction for neural data
   * - Bayesian decoding
     - Position decoding from neural activity
   * - Locomotion
     - Locomotion epoch detection

neuralib-utils
^^^^^^^^^^^^^^

`PyPI <https://pypi.org/project/neuralib-utils/>`_ ·
`Changelog <https://github.com/ytsimon2004/neuralib2/blob/main/packages/neuralib-utils/CHANGELOG.md>`_

Core utilities and foundational tools. **Required by all other packages.**

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - ``neuralib.io``
     - File I/O (HDF5, JSON, CSV)
   * - ``neuralib.typing``
     - Unified type system (pandas/polars DataFrames)
   * - ``neuralib.util``
     - Logging, DataFrame ops, GPU monitoring
   * - ``neuralib.persistence``
     - Persistence decorator for structured data caching
   * - ``neuralib.plot``
     - Publication-quality plotting (PSTH, Venn)
   * - ``neuralib.dashboard``
     - Bokeh-based interactive visualization
   * - ``neuralib.imglib``
     - Image processing utilities
   * - ``neuralib.tools``
     - External integrations (Google Sheets, Slack)

----

Command-Line Tools
------------------

.. grid:: 2
   :gutter: 3

   .. grid-item::

      **nl_brainrender**

      Visualize brain region data with built-in rendering support.
      See :doc:`atlas/brainrender` for examples.

      .. code-block:: bash

         nl_brainrender -h

   .. grid-item::

      **nl_widefield**

      Widefield imaging CLI analysis.
      See :doc:`imaging/widefield` for examples.

      .. code-block:: bash

         nl_widefield -h

----

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Packages

   atlas/index
   imaging/index
   parser/index
   metric/index
   utils/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/neuralib.rst

----

Array Annotation Syntax
-----------------------

Used in documentation to describe array-shaped data structures:

``Array[DType, [*Shape]]``

- ``DType`` = data type (e.g., ``int``, ``float``, ``bool``)
- ``Shape`` = array shape (e.g., ``[N, T]``)
- ``|`` = union of shapes or types

**Examples:**

- ``Array[int|bool, [N, 3]]`` — Boolean or integer array with shape ``(N, 3)``
- ``Array[float, [N, 2] | [N, T, 2]]`` — Float array with shape ``(N, 2)`` or ``(N, T, 2)``

----

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`