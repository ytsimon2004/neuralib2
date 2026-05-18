from pathlib import Path

import pytest
from matplotlib import pyplot as plt

from neuralib.atlas.cellatlas.core import load_cellatlas
from neuralib.atlas.data import build_annotation_leaf_map, get_children, get_leaf_in_annotation, load_bg_volumes
from neuralib.atlas.view import get_slice_view

DATA_EXISTS = (Path().home() / '.brainglobe' / 'allen_mouse_100um_v1.2').exists()


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_get_child_id():
    ret = get_children(385, dataframe=False, atlas_name='allen_mouse_100um')  # VISp
    assert ret == [593, 821, 721, 778, 33, 305]


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_get_child_acronym():
    ret = get_children('VISp', dataframe=False, atlas_name='allen_mouse_100um')
    assert ret == ['VISp1', 'VISp2/3', 'VISp4', 'VISp5', 'VISp6a', 'VISp6b']


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_get_leaf_in_annotation():
    assert len(get_leaf_in_annotation('VISp', name=True, atlas_name='allen_mouse_100um')) > 0


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_build_all_leaf_map():
    x = set(build_annotation_leaf_map(atlas_name='allen_mouse_100um')[385])
    y = set(get_leaf_in_annotation('VISp', atlas_name='allen_mouse_100um'))
    assert x == y


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_slice_view_reference(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)

    slice_index = 30
    plane = get_slice_view('reference', plane_type='coronal', resolution=100).plane_at(slice_index)

    _, ax = plt.subplots(ncols=3, figsize=(20, 10))
    plane.plot(ax=ax[0], boundaries=True)
    plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax[1], boundaries=True)
    plane.with_angle_offset(deg_x=0, deg_y=20).plot(ax=ax[2], boundaries=True)
    plt.show()


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_slice_view_annotation(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)

    slice_index = 500
    plane = get_slice_view('annotation', plane_type='sagittal', resolution=100).plane_at(slice_index)

    _, ax = plt.subplots(ncols=3, figsize=(20, 10))
    plane.plot(ax=ax[0])
    plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax[1])
    plane.with_angle_offset(deg_x=0, deg_y=20).plot(ax=ax[2])
    plt.show()


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_annotation_region(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)

    slice_index = 800
    plane = get_slice_view('reference', plane_type='coronal', resolution=100).plane_at(slice_index)

    _, ax = plt.subplots()
    plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax, annotation_region=['RSP', 'VISp'])
    plt.show()


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_affine_transform(monkeypatch):
    import matplotlib.transforms as mtransforms

    monkeypatch.setattr(plt, 'show', lambda: None)

    slice_index = 800
    plane = get_slice_view('reference', plane_type='coronal', resolution=100).plane_at(slice_index)

    _, ax = plt.subplots()
    aff = mtransforms.Affine2D().skew_deg(-20, 0)
    t = aff + ax.transData

    plane.with_angle_offset().plot(ax=ax, transform=t)
    plt.show()


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_max_projection(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)

    view = get_slice_view('reference', plane_type='transverse', resolution=100)

    _, ax = plt.subplots()
    regions = get_children('VIS', atlas_name='allen_mouse_100um')
    view.plot_max_projection(ax, annotation_regions=regions)
    plt.show()


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_volume_different_source_data(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)

    x = load_bg_volumes('allen_mouse_100um')
    y = load_cellatlas().select('acronym', 'Volumes [mm^3]')

    z = x.join(y, on='acronym')
    cols = z['volume_mm3', 'Volumes [mm^3]'].to_numpy()
    plt.plot(cols[:, 0], cols[:, 1], 'k.')
    plt.show()
