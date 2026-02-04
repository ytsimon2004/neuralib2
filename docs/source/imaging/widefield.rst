WideField
=========

This module provides tools for processing widefield video sequence data

Sequence Preprocessing
-------------------------

A preprocessing pipeline for widefield calcium imaging data. Computes ΔF/F with time-varying baseline using rolling window percentile calculation.

**Features:**

- Motion correction using ECC (Enhanced Correlation Coefficient) algorithm
- Rolling window baseline (F0) calculation with configurable percentile
- ΔF/F computation with interpolated baseline
- GPU acceleration support (CuPy) and Numba JIT optimization
- Memory-efficient chunked processing for large datasets

**CLI Usage:**

.. code-block:: bash

    $ nl_widefield preproc --file <TIF_FILE> [OPTIONS]
    $ nl_widefield preproc --directory <TIF_DIR> [OPTIONS]

**Options:**

.. code-block:: text

    Data I/O Options:
      --file FILE           Single input TIF file
      --directory DIR       Directory containing TIF files
      --suffix SUFFIX       File suffix pattern (default: .tif)
      --output_dir DIR      Output directory for results

    Processing Options:
      --motion_corr         Enable motion correction
      --max_shift N         Max shift in pixels for motion correction (default: 20)
      --rotate DEGREES      Rotate all sequences by specified degrees
      --chunk_size N        Frames per processing chunk (default: 3000)
      --window_size N       Rolling baseline window size in frames (default: 100)
      --percentile N        Percentile for baseline calculation (default: 10)
      --n_jobs N            Parallel jobs (-1 = all CPUs, default: -1)
      --force_compute       Force recomputation even if outputs exist
      --save_f0             Save F0 baseline to disk

    Acceleration Options:
      --use_gpu             Use GPU acceleration with CuPy

**Example:**

.. code-block:: bash

    # Basic preprocessing
    $ nl_widefield preproc --file recording.tif

    # With motion correction and GPU acceleration
    $ nl_widefield preproc --directory ./tifs --motion_corr --use_gpu

    # Custom parameters
    $ nl_widefield preproc --file data.tif --window_size 200 --percentile 5 --rotate 90

**Output Files:**

- ``dff.npy``: ΔF/F array (T, H, W)
- ``f0.h5``: F0 baseline array (if ``--save_f0`` enabled)
- ``reference_frame.tif``: Mean reference frame
- ``motion_transforms.h5``: Motion correction transforms (if ``--motion_corr`` enabled)
- ``metadata.json``: Processing parameters and data info



FFT View with Bokeh
---------------------

A Bokeh-based visualization of the HSV colormap representation of the Fourier-transformed video (e.g., for visual retinotopy).

Usage:

.. code-block:: python

    nl_widefield fft <FILE>

|fft_view|



Align with Napari
--------------------------

A Napari-based tool for aligning video data to a dorsal cortex view.

Usage:

.. code-block:: python

    nl_widefield align <FILE> [-R REFERENCE_FILE] [-M MAP_FILE]

|align_view|



.. |fft_view| image:: ../_static/example_fft.jpg
.. |align_view| image:: ../_static/example_align.jpg