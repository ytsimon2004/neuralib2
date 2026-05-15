from pathlib import Path

import numpy as np
import pytest

import neuralib.widefield.ccf as ccf_module
from neuralib.widefield.ccf import DorsalCCF, RegionLabel, validate_ccf_array


@pytest.fixture()
def dorsal_ccf_json(monkeypatch: pytest.MonkeyPatch) -> Path:
    labels_file = Path(__file__).parents[3] / 'res' / 'atlas' / 'dorsal_cortex_ccf_labels.json'
    monkeypatch.setattr(ccf_module, '_CCF_CACHE', labels_file)
    return ccf_module.get_dorsal_ccf_file()


def test_dorsal_ccf_from_json_and_boundaries(dorsal_ccf_json: Path):
    label = RegionLabel.from_json(dorsal_ccf_json, 'MOp')
    assert label.acronym == 'MOp'
    assert label.name == 'Primary motor area'
    assert label.label == 3

    ccf = DorsalCCF.from_json(dorsal_ccf_json, image_shape=(32, 32))
    assert 'MOp' in ccf.region_list
    assert ccf.region_dict['MOp'] == 3

    motor = ccf.select_region('MOp')
    left = motor.select_hemisphere('left').to_numpy()
    right = motor.select_hemisphere('right').to_numpy()
    assert left.shape == (32, 32)
    assert left.dtype == np.uint16
    assert np.count_nonzero(left == 3) > 0
    assert np.count_nonzero(right == 3) > 0
    assert np.count_nonzero(left & right) == 0

    boundary = motor.select_hemisphere('left').to_boundary(outlines_only=False)
    assert boundary.shape == (32, 32)
    assert boundary.dtype == np.bool_
    assert np.count_nonzero(boundary) > 0


def test_dorsal_ccf_validation_and_selection_errors(dorsal_ccf_json: Path):
    ccf = DorsalCCF.from_json(dorsal_ccf_json, image_shape=(16, 16))

    with pytest.raises(ValueError, match='Unknown region'):
        ccf.select_region('UNKNOWN')

    with pytest.raises(ValueError, match='invalid hemisphere'):
        ccf.select_hemisphere('middle')

    validate_ccf_array(np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match='2D numpy array'):
        validate_ccf_array(np.zeros((2, 2, 2), dtype=bool))
