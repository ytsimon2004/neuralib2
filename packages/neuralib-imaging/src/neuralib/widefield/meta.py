from typing import TypedDict
from neuralib.io.json import load_json

__all__ = [
    'load_preprocess_meta',
    'PreprocessMeta',
    'InputArguments',
    'DataInfo',
    'ProcessedInfo',
    'BaselineInfo',
    'MotionCorrInfo'
]

from neuralib.typing import PathLike


class PreprocessMeta(TypedDict):
    timestamp: str
    input_arguments: 'InputArguments'
    data_info: 'DataInfo'
    processing: 'ProcessedInfo'
    f0_baseline: 'BaselineInfo'
    motion_correction: 'MotionCorrInfo'


class InputArguments(TypedDict):
    input_source: str
    suffix_pattern: str
    output_dir: str
    motion_correction: bool
    rotate: float
    chunk_size: int
    window_size: int
    percentile: int
    n_jobs: int
    max_shift: int
    force_compute: bool
    save_f0: bool
    use_gpu: bool


class DataInfo(TypedDict):
    n_tif_files: int
    tif_files: list[str]
    total_frames: int
    frame_shape: tuple[int, int]
    image_height: int
    image_width: int


class ProcessedInfo(TypedDict):
    rotation_applied: int
    rotation_degrees: int
    has_numba: bool
    has_cupy: bool
    gpu_used: bool


class BaselineInfo(TypedDict):
    percentile: int
    stride: int
    n_keyframes: int


class MotionCorrInfo(TypedDict):
    pass


def load_preprocess_meta(filepath: PathLike) -> PreprocessMeta:
    return load_json(filepath, verbose=False)
