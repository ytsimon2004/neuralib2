from unittest.mock import patch

import numpy as np
import pytest
from matplotlib import pyplot as plt

from neuralib.io import NEUROLIB_DATASET_DIRECTORY
from neuralib.io.dataset import load_example_czi, load_example_lsm
from neuralib.scan.core import AbstractScanner
from neuralib.scan.czi import czi_file
from neuralib.scan.lsm import lsm_file

CZI_DATA = NEUROLIB_DATASET_DIRECTORY / 'test.czi'
LSM_DATA = NEUROLIB_DATASET_DIRECTORY / 'test.lsm'
DOWNLOAD_CACHE = False

if not CZI_DATA.exists() and DOWNLOAD_CACHE:
    load_example_czi(cached=True)

if not LSM_DATA.exists() and DOWNLOAD_CACHE:
    load_example_lsm(cached=True)


@pytest.fixture(scope='module')
def czi():
    with czi_file(CZI_DATA) as czi:
        yield czi


@pytest.fixture(scope='module')
def lsm():
    with lsm_file(LSM_DATA) as lsm:
        yield lsm


def test_z_projection_respects_axis():
    stacks = np.arange(24).reshape(2, 3, 4)

    np.testing.assert_array_equal(AbstractScanner.z_projection(stacks, 'avg', axis=1), np.mean(stacks, axis=1))
    np.testing.assert_array_equal(AbstractScanner.z_projection(stacks, 'max', axis=1), np.max(stacks, axis=1))
    np.testing.assert_array_equal(AbstractScanner.z_projection(stacks, 'min', axis=1), np.min(stacks, axis=1))
    np.testing.assert_array_equal(AbstractScanner.z_projection(stacks, 'std', axis=1), np.std(stacks, axis=1))


@pytest.mark.skipif(not CZI_DATA.exists(), reason='no cached data')
def test_czi_config(czi):
    assert czi.consistent_config
    assert czi.is_mosaic


@pytest.mark.skipif(not CZI_DATA.exists(), reason='no cached data')
def test_czi_dimcode(czi):
    assert czi.dimcode == 'HSTCZMYX'
    assert czi.get_code(0, 'C') == 3
    assert czi.get_code(0, 'M') == 54
    assert czi.get_code(0, 'H') == 1
    assert czi.get_code(0, 'X') == 512
    assert czi.get_code(0, 'Y') == 512


@pytest.mark.skipif(not CZI_DATA.exists(), reason='no cached data')
def test_czi_nscenes(czi):
    assert czi.n_scenes == 2


@pytest.mark.skipif(not CZI_DATA.exists(), reason='no cached data')
def test_czi_channel_names(czi):
    assert czi.get_channel_names(0) == ['AF488-T1', 'AF405-T2', 'AF555-T2']


@pytest.mark.skipif(not CZI_DATA.exists(), reason='no cached data')
@patch('matplotlib.pyplot.show')
def test_czi_view(mock_show, czi):
    arr = czi.view()
    plt.imshow(arr, cmap='gray')
    plt.show()


@pytest.mark.skipif(not LSM_DATA.exists(), reason='no cached data')
def test_lsm_config(lsm):
    assert lsm.file_type == '.lsm'
    assert lsm.get_channel_names() == ['Ch1-T1', 'Ch2-T1', 'Ch1-T2']
    assert lsm.dimcode == 'ZCYX'
    assert lsm.n_scenes == 1
