from __future__ import annotations

import abc
from pathlib import Path
from typing import Self

import numpy as np

from neuralib.typing import PathLike

__all__ = ['lazy_load_widefield']


def lazy_load_widefield(file: PathLike) -> LazyWideFieldNumpy:
    """
    Convenience loader for LazyWideFieldData. Use in a 'with' block:

    :param file: Path to the `.npy` file.
    :return: WideFieldData instance (use in a context manager).
    """
    file = Path(file)

    ext = file.suffix
    if ext == '.npy':
        return LazyWideFieldNumpy(file)
    else:
        raise ValueError(f'Unsupported file extension: {ext}')


class _BaseWideField(metaclass=abc.ABCMeta):
    """Base class for lazy-access wide field imaging data wrappers."""

    @abc.abstractmethod
    def __enter__(self) -> Self:
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    @abc.abstractmethod
    def dff(self) -> np.ndarray:
        """Load entire dFF matrix into memory. `Array[float32, [F, H, W]]`"""
        pass

    @property
    @abc.abstractmethod
    def image_height(self) -> int:
        """Image height in pixels"""
        pass

    @property
    @abc.abstractmethod
    def image_width(self) -> int:
        """Image width in pixels"""
        pass

    @property
    @abc.abstractmethod
    def num_frames(self) -> int:
        """Total number of frames"""
        pass

    @abc.abstractmethod
    def get_frame(self, idx: int) -> np.ndarray:
        """Lazy-access a single frame, `Array[float32, [H, W]]`"""
        pass

    @abc.abstractmethod
    def get_frames(self, start: int, stop: int) -> np.ndarray:
        """Lazy-access frames [start:stop]. ``Array[float32, [F, H, W]]`"""
        pass


class LazyWideFieldNumpy(_BaseWideField):
    """Lazy-access wrapper for wide field imaging data from a `.npy` file.
    The file should contain a 3D array with shape (num_frames, image_height, image_width).
    """

    def __init__(self, filepath: Path):
        self._filepath = filepath
        self._file: np.ndarray | None = None

    def __enter__(self) -> LazyWideFieldNumpy:
        self._file = np.load(self._filepath, mmap_mode='r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            # if hasattr(self._file, '_mmap'):
            #     self._file._mmap.close()

            del self._file
            self._file = None

    def __repr__(self) -> str:
        if self._file is not None:
            shape_str = f"shape={self._file.shape}"
        else:
            shape_str = "[file not open]"
        return f"<LazyWideFieldNumpy file='{self._filepath.name}' {shape_str}>"

    @property
    def dff(self) -> np.ndarray:
        if self._file is None:
            raise RuntimeError("File is not open. Use a 'with' context manager.")
        return self._file[:]

    @property
    def image_height(self) -> int:
        if self._file is None:
            raise RuntimeError("File is not open. Use a 'with' context manager.")
        return self._file.shape[1]

    @property
    def image_width(self) -> int:
        if self._file is None:
            raise RuntimeError("File is not open. Use a 'with' context manager.")
        return self._file.shape[2]

    @property
    def num_frames(self) -> int:
        if self._file is None:
            raise RuntimeError("File is not open. Use a 'with' context manager.")
        return self._file.shape[0]

    def get_frame(self, idx: int) -> np.ndarray:
        if self._file is None:
            raise RuntimeError("File is not open. Use a 'with' context manager.")
        return self._file[idx]

    def get_frames(self, start: int, stop: int) -> np.ndarray:
        if self._file is None:
            raise RuntimeError("File is not open. Use a 'with' context manager.")
        return self._file[start:stop]
