from __future__ import annotations

from os import PathLike as OsPathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

__all__ = [
    'ArrayLike',
    'ArrayLikeStr',
    'AxesArray',
    #
    'PathLike',
    'PathLikeType',
    #
    'Series',
    'DataFrame'
]

# ===== #
# Array #
# ===== #

ArrayLike: TypeAlias = Union[NDArray[Any], Sequence[Any], 'pd.Series', 'pl.Series']
"""Alias for array-like objects, including numpy arrays, lists, tuples, and series"""

ArrayLikeStr: TypeAlias = Union[NDArray[np.str_], list[str], tuple[str, ...], 'pd.Series', 'pl.Series']
"""Alias for array-like objects of strings, including numpy arrays, lists, tuples, and series"""

AxesArray: TypeAlias = Union[np.ndarray, list[Any]]
"""Alias for matplotlib Axes numpy array"""

# ==== #
# Path #
# ==== #

PathLike: TypeAlias = Union[str, Path, OsPathLike[str]]
"""Alias for filesystem path-like objects."""

PathLikeType = (str, Path, OsPathLike)
"""Runtime-checkable types accepted by :data:`PathLike`."""

# ================== #
# Series / DataFrame #
# ================== #

Series: TypeAlias = Union['pd.Series', 'pl.Series']
"""Alias for series objects from pandas or polars"""

DataFrame: TypeAlias = Union['pd.DataFrame', 'pl.DataFrame']
"""Alias for dataframe objects from pandas or polars"""
