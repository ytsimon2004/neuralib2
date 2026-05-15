from pathlib import Path

import h5py
import numpy as np
import tifffile

from neuralib.widefield.preproc import PreprocessOptions, load_preprocess_meta


def test_preprocess_options_run_on_temporary_tif(tmp_path: Path):
    tif_path = tmp_path / 'synthetic.tif'
    output_dir = tmp_path / 'preprocessed'

    rng = np.random.default_rng(42)
    frames = (rng.random((6, 8, 7), dtype=np.float32) * 20 + 100).astype(np.float32)
    tifffile.imwrite(tif_path, frames)

    opts = PreprocessOptions()
    opts.file = tif_path
    opts.directory = None
    opts.suffix_pattern = '.tif'
    opts._output_dir = output_dir
    opts.motion_correction = False
    opts.max_shift = 2
    opts.rotate = None
    opts.chunk_size = 3
    opts.window_size = 3
    opts.percentile = 20
    opts.n_jobs = 1
    opts.force_compute = True
    opts.save_f0 = True
    opts.use_gpu = False

    opts.run()

    dff_path = output_dir / 'dff.npy'
    f0_path = output_dir / 'f0.h5'
    reference_path = output_dir / 'reference_frame.tif'
    metadata_path = output_dir / 'metadata.json'

    assert dff_path.exists()
    assert f0_path.exists()
    assert reference_path.exists()
    assert metadata_path.exists()
    assert not (output_dir / 'temp_frames.npy').exists()

    dff = np.load(dff_path, mmap_mode='r')
    assert dff.shape == frames.shape
    assert dff.dtype == np.float32
    assert np.isfinite(dff[:]).all()

    with h5py.File(f0_path, 'r') as f:
        assert f['f0'].shape == frames.shape
        assert f['f0'].dtype == np.float32

    reference = tifffile.imread(reference_path)
    assert reference.shape == frames.shape[1:]

    metadata = load_preprocess_meta(metadata_path)
    assert metadata['input_arguments']['input_source'] == str(tif_path)
    assert metadata['input_arguments']['motion_correction'] is False
    assert metadata['input_arguments']['save_f0'] is True
    assert metadata['data_info']['total_frames'] == frames.shape[0]
    assert metadata['data_info']['frame_shape'] == [frames.shape[1], frames.shape[2]]
    assert metadata['processing']['has_numba'] is True
    assert metadata['f0_baseline'] is not None
    assert metadata['f0_baseline']['n_keyframes'] >= 2
    assert metadata['motion_correction'] is None
