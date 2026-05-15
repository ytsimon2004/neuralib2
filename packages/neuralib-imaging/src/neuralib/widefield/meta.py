from typing import TypedDict, cast

from neuralib.io.json import load_json
from neuralib.typing import PathLike

__all__ = [
    'load_preprocess_meta',
    'PreprocessMeta',
    'InputArguments',
    'DataInfo',
    'ProcessedInfo',
    'BaselineInfo',
    'MotionCorrInfo'
]


class PreprocessMeta(TypedDict, total=False):
    timestamp: str
    input_arguments: 'InputArguments'
    data_info: 'DataInfo'
    processing: 'ProcessedInfo'
    f0_baseline: 'BaselineInfo | None'
    motion_correction: 'MotionCorrInfo | None'


class InputArguments(TypedDict):
    input_source: str
    suffix_pattern: str
    output_dir: str
    motion_correction: bool
    rotate: float | None
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
    frame_shape: list[int] | None
    image_height: int | None
    image_width: int | None


class ProcessedInfo(TypedDict):
    rotation_applied: bool
    rotation_degrees: float | None
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
    return cast(PreprocessMeta, load_json(filepath, verbose=False))
