Cellpose
===================

Parser for `Cellpose <https://github.com/MouseLand/cellpose>`_ segmentation results.

- **Refer to API**: :class:`~neuralib.cellpose.CellposeSegmentation`

**Example of loading cellpose segmentation results**

.. code-block:: python

    from neuralib.cellpose import read_cellpose

    # Load cellpose segmentation result (.npy file)
    seg = read_cellpose("/path/to/cellpose_seg.npy")

    # Get number of segmented cells
    print(f"Number of cells: {seg.n_segmentation}")

    # Get image dimensions
    print(f"Image size: {seg.width} x {seg.height}")

    # Access segmentation masks (0 = background, 1,2,... = cell labels)
    masks = seg.masks  # Array[uint16, [H, W]]

    # Access cell outlines
    outlines = seg.outlines  # Array[uint16, [H, W]]

    # Get cell center coordinates
    centers = seg.points  # Array[int, [N, 2]]

    # Get cell diameter used in segmentation
    print(f"Cell diameter: {seg.diameter}")

**Convert to ImageJ ROI**

.. code-block:: python

    from neuralib.cellpose import cellpose_point_roi_helper

    # Convert segmentation to ImageJ point ROI
    cellpose_point_roi_helper(
        "/path/to/cellpose_seg.npy",
        "/path/to/output.roi"
    )

.. seealso::

    `Cellpose Output Documentation <https://cellpose.readthedocs.io/en/latest/outputs.html#seg-npy-output>`_
