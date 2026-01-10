from pathlib import Path

import numpy as np
import unittest

from argclz.core import parse_args
from neuralib.io import NEUROLIB_DATASET_DIRECTORY
from neuralib.io.dataset import load_example_rois_image, load_example_rois_dir
from neuralib.stardist.run_2d import StarDist2DOptions
from importlib.metadata import version
from packaging.version import parse as parse_version

_NUMPY_VERSION = parse_version(version("numpy"))
_STARDIST_UNSUPPORTED = _NUMPY_VERSION >= parse_version("2.0")


@unittest.skipIf(_STARDIST_UNSUPPORTED, "stardist requires numpy<2.0")
class TestStarDist(unittest.TestCase):
    arr: np.ndarray
    dirpath: Path
    filepath: Path

    @classmethod
    def setUpClass(cls):
        load_example_rois_image(cached=True, rename_file='rois.png')
        cls.filepath = NEUROLIB_DATASET_DIRECTORY / 'rois.png'
        cls.dirpath = load_example_rois_dir(cached=True, rename_folder='rois')

    def test_empty_option(self):
        opt = parse_args(StarDist2DOptions(), [])
        with self.assertRaises(RuntimeError):
            opt.run()

    def test_file_run(self, test_napari: bool = False):
        args = ['--file', str(self.filepath)]
        if test_napari:
            args.append('--napari')

        opt = parse_args(StarDist2DOptions(), args)
        self.assertTrue(np.issubdtype(opt.process_image().dtype, np.floating))
        self.assertTrue(opt.file_mode)
        opt.run()

    def test_dir_mode(self):
        opt = parse_args(StarDist2DOptions(), ['--dir', str(self.dirpath), '--invalid', '--save_roi'])
        self.assertTrue(opt.batch_mode)
        opt.run()

    # @classmethod
    # def tearDownClass(cls):
    #     clean_all_cache_dataset()
