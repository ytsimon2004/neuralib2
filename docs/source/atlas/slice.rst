Slice View
=======================

This module provides tools for visualizing 2D brain slice views with optional region annotations and angle offsets.

**Refer to API**: :doc:`../api/neuralib.atlas.view`

Basic Usage
-----------

Use :func:`~neuralib.atlas.view.get_slice_view` to create a slice view and select a plane at a specific index:

.. code-block:: python

    from neuralib.atlas.view import get_slice_view

    # Get a reference view in coronal plane at 10µm resolution
    view = get_slice_view('reference', plane_type='coronal', resolution=10)

    # Select a plane at slice index 700
    plane = view.plane_at(700)

    # Plot with region boundaries
    plane.plot(boundaries=True)

**View types**:

- ``reference``: Grayscale reference image
- ``annotation``: Color-coded region annotation

**Plane types**:

- ``coronal``: Front-to-back sections
- ``sagittal``: Left-to-right sections
- ``transverse``: Top-to-bottom sections (horizontal)


Angle Offset
------------

Apply angular offsets to simulate angled slice cuts using ``with_angle_offset()``:

**ML offset in coronal slice**

Use ``deg_x`` for medial-lateral axis rotation:

.. code-block:: python

    import matplotlib.pyplot as plt
    from neuralib.atlas.view import get_slice_view

    slice_index = 700
    plane = get_slice_view('reference', plane_type='coronal', resolution=10).plane_at(slice_index)

    _, ax = plt.subplots(ncols=2, figsize=(20, 10))

    plane.plot(ax=ax[0], boundaries=True)
    ax[0].set_title('without offset')

    plane.with_angle_offset(deg_x=10).plot(ax=ax[1], boundaries=True)
    ax[1].set_title('+10 degree ML offset')

**DV offset in sagittal slice**

Use ``deg_y`` for dorsal-ventral axis rotation:

.. code-block:: python

    slice_index = 500
    plane = get_slice_view('annotation', plane_type='sagittal', resolution=10).plane_at(slice_index)

    _, ax = plt.subplots(ncols=2, figsize=(20, 10))

    plane.plot(ax=ax[0], boundaries=True)
    ax[0].set_title('without offset')

    plane.with_angle_offset(deg_y=20).plot(ax=ax[1], boundaries=True)
    ax[1].set_title('+20 degree DV offset')


Region Annotation
-----------------

Highlight specific brain regions using the ``annotation_region`` parameter:

.. code-block:: python

    from neuralib.atlas.data import get_children
    from neuralib.atlas.view import get_slice_view

    slice_index = 900
    plane = get_slice_view('reference', plane_type='coronal', resolution=10).plane_at(slice_index)

    _, ax = plt.subplots(ncols=2, figsize=(20, 10))

    # Annotate CA1 and Primary Visual Cortex
    plane.plot(ax=ax[0], boundaries=True, annotation_region=['CA1', 'VISp'])

    # Get all VISp subregions (layers)
    primary_visual_layers = get_children('VISp', dataframe=False)
    print(primary_visual_layers)  # ['VISp1', 'VISp2/3', 'VISp4', 'VISp5', 'VISp6a', 'VISp6b']

    # Annotate with offset view and detailed tree regions
    plane.with_angle_offset(deg_x=10).plot(ax=ax[1], boundaries=True, annotation_region=['CA1'] + primary_visual_layers)


**Annotation in transverse view**

.. code-block:: python

    slice_index = 300
    plane = get_slice_view('reference', plane_type='transverse', resolution=10).plane_at(slice_index)

    _, ax = plt.subplots(ncols=2, figsize=(20, 10))

    # Default colormap
    plane.plot(ax=ax[0], annotation_region=['ACA', 'LP'])

    # Custom colormap with boundaries
    plane.plot(ax=ax[1], annotation_region=['SS', 'MO'], annotation_cmap='PiYG', boundaries=True)


Max Projection
--------------

Generate a maximum intensity projection across all slices for specified regions:

.. code-block:: python

    from neuralib.atlas.data import get_children
    from neuralib.atlas.view import get_slice_view

    view = get_slice_view('reference', plane_type='transverse', resolution=10)

    _, ax = plt.subplots()

    # Get all visual area subregions
    regions = get_children('VIS')
    print(regions)  # ['VISal', 'VISam', 'VISl', 'VISli', 'VISp', 'VISpl', 'VISpm', 'VISpor']

    # Plot max projection for all visual areas
    view.plot_max_projection(ax, annotation_regions=regions)


.. seealso::

    - :func:`~neuralib.atlas.data.get_children` for retrieving subregions
    - :doc:`structure_tree` for brain region hierarchy