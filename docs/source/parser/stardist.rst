StarDist
===================

Parser for `StarDist <https://github.com/stardist/stardist>`_ segmentation results.

- **Refer to API**: :class:`~neuralib.stardist.StarDistSegmentation`

**Supported Models**

- ``2D_versatile_fluo``: Versatile fluorescence model
- ``2D_versatile_he``: Versatile H&E staining model
- ``2D_paper_dsb2018``: DSB 2018 paper model
- ``2D_demo``: Demo model

**Example of loading stardist segmentation results**

.. code-block:: python

    from neuralib.stardist import read_stardist

    # Load stardist segmentation result (.npz file)
    seg = read_stardist("/path/to/stardist_seg.npz")

    # Get number of segmented cells
    print(f"Number of cells: {seg.n_segmentation}")

    # Get image dimensions
    print(f"Image size: {seg.width} x {seg.height}")

    # Access labeled image (NaN = background)
    labels = seg.labels  # Array[float, [H, W]]

    # Access polygon coordinates for each cell
    cords = seg.cords  # Array[float, [N, 2, E]]

    # Get detection probabilities
    probs = seg.prob  # Array[float, N]

    # Get cell center coordinates
    centers = seg.points  # Array[float, [N, 2]]

    # Get model used for segmentation
    print(f"Model: {seg.model}")

**Filter by probability threshold**

.. code-block:: python

    # Mask out low-confidence detections
    seg.mask_probability(threshold=0.5)

**Convert to ImageJ ROI**

.. code-block:: python

    from neuralib.stardist import stardist_point_roi_helper

    # Convert segmentation to ImageJ point ROI
    stardist_point_roi_helper(
        "/path/to/stardist_seg.npz",
        "/path/to/output.roi"
    )

**Save as compressed NPZ**

.. code-block:: python

    # Save segmentation results
    seg.to_npz("/path/to/output.npz")