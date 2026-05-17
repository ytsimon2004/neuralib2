from pathlib import Path

import numpy as np
import pytest
from argclz.core import parse_args
from neuralib.io import NEUROLIB_DATASET_DIRECTORY
from neuralib.io.dataset import load_example_rois_dir, load_example_rois_image
from neuralib.stardist.run_2d import StarDist2DOptions


@pytest.fixture(scope='module')
def rois_file() -> Path:
    load_example_rois_image(cached=True, rename_file='rois.png')
    return NEUROLIB_DATASET_DIRECTORY / 'rois.png'


@pytest.fixture(scope='module')
def rois_dir() -> Path:
    return load_example_rois_dir(cached=True, rename_folder='rois')


def test_empty_option():
    opt = parse_args(StarDist2DOptions(), [])
    with pytest.raises(RuntimeError):
        opt.run()


def test_file_run(rois_file: Path):
    opt = parse_args(StarDist2DOptions(), ['--file', str(rois_file)])

    assert np.issubdtype(opt.process_image().dtype, np.floating)
    assert opt.file_mode

    opt.run()


def test_dir_mode(rois_dir: Path):
    opt = parse_args(StarDist2DOptions(), ['--dir', str(rois_dir), '--invalid', '--save_rois'])

    assert opt.batch_mode

    opt.run()
