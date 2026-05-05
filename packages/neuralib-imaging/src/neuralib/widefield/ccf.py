import json
import urllib.request
from pathlib import Path
from typing import Self, NamedTuple, Literal, get_args, Sequence

import numpy as np
from neuralib.typing import PathLike

__all__ = [
    'DorsalRegion',
    'HEMISPHERE_TYPE',
    #
    'RegionLabel',
    'DorsalCCF',
    'validate_ccf_array'
]

DorsalRegion = str

HEMISPHERE_TYPE = Literal['left', 'right', 'both']

_CCF_FILENAME = 'dorsal_cortex_ccf_labels.json'
_CCF_CACHE = Path.home() / '.neuralib' / _CCF_FILENAME
_CCF_URL = 'https://raw.githubusercontent.com/ytsimon2004/neuralib2/main/res/atlas/dorsal_cortex_ccf_labels.json'


def get_dorsal_ccf_file() -> Path:
    """Return CCF JSON path from ~/.neuralib cache, downloading from GitHub on first use."""
    if not _CCF_CACHE.exists():
        _CCF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading CCF labels to {_CCF_CACHE} ...')
        urllib.request.urlretrieve(_CCF_URL, _CCF_CACHE)
    return _CCF_CACHE


class RegionLabel(NamedTuple):
    """
    :param acronym: Short uppercase code for the region (e.g. 'VISp').
    :param name: Full name of the region.
    :param reference: Depth reference range in microns (start, end) relative to brain surface.
    :param resolution: Spatial sampling in microns per pixel.
    :param label: Unique integer label for this region in a label map.
    :param allen_id: Allen Brain Atlas identifier for the region.
    :param allen_rgb: RGB color defined by the Allen Atlas for visualization.
    :param left_center: (x, y) coordinates in mm of the centroid on the left hemisphere.
    :param right_center: (x, y) coordinates in mm of the centroid on the right hemisphere.
    :param left_x: 1D array of x-coordinates (mm) defining the left polygon boundary.
    :param left_y: 1D array of y-coordinates (mm) defining the left polygon boundary.
    :param right_x: 1D array of x-coordinates (mm) defining the right polygon boundary.
    :param right_y: 1D array of y-coordinates (mm) defining the right polygon boundary.
    """

    acronym: DorsalRegion
    name: str
    reference: tuple[int, int]
    resolution: int
    label: int
    allen_id: int
    allen_rgb: tuple[int, int, int]
    left_center: tuple[int, int]
    right_center: tuple[int, int]
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray

    # noinspection PyTypeChecker
    @classmethod
    def from_json(cls, file: PathLike, acronym: DorsalRegion) -> Self:
        data = json.loads(file.read_text())
        for r in data:
            if r['acronym'] == acronym:
                return cls(
                    acronym=r['acronym'],
                    name=r['name'],
                    reference=tuple(r['reference']),
                    resolution=r['resolution'],
                    label=r['label'],
                    allen_id=r['allen_id'],
                    allen_rgb=tuple(r['allen_rgb']),
                    left_center=tuple(r['left_center']),
                    right_center=tuple(r['right_center']),
                    left_x=np.array(r['left_x']),
                    left_y=np.array(r['left_y']),
                    right_x=np.array(r['right_x']),
                    right_y=np.array(r['right_y']),
                )
        raise ValueError(f"Region acronym '{acronym}' not found in {file}")


class DorsalCCF:

    def __init__(self, region_labels: list[RegionLabel],
                 image_shape: tuple[int, int],
                 hemisphere: HEMISPHERE_TYPE = 'both'):
        self.region_labels = region_labels

        if hemisphere not in get_args(HEMISPHERE_TYPE):
            raise ValueError(f'invalid hemisphere: {hemisphere}, should be one of {get_args(HEMISPHERE_TYPE)}')
        self._hemisphere = hemisphere

        self._image_shape = image_shape
        self._fov_mm = (11, 11)

        self.__array = None

    @classmethod
    def from_json(cls, file: PathLike | None = None,
                  image_shape: tuple[int, int] = (512, 512)) -> Self:

        if file is None:
            file = get_dorsal_ccf_file()

        labels = []
        data = json.loads(file.read_text())
        for r in data:
            labels.append(RegionLabel.from_json(file, r['acronym']))

        return cls(labels, image_shape, hemisphere='both')

    @classmethod
    def from_array(cls, array: np.ndarray, hemisphere: HEMISPHERE_TYPE = 'both'):
        validate_ccf_array(array)

        h, w = array.shape
        present_labels = sorted(set(np.unique(array)) - {0})

        # full
        ccf = cls.from_json(image_shape=(h, w))
        reverse_map = {v: k for k, v in ccf.region_dict.items()}

        labels = [RegionLabel.from_json(get_dorsal_ccf_file(), reverse_map[val]) for val in present_labels]

        instance = cls(labels, (h, w), hemisphere=hemisphere)
        instance._mask_array = array

        return instance

    @property
    def image_shape(self) -> tuple[int, int]:
        return self._image_shape

    @property
    def fov_mm(self) -> tuple[float, float]:
        return self._fov_mm

    @property
    def hemisphere(self) -> HEMISPHERE_TYPE:
        return self._hemisphere

    @property
    def region_list(self) -> list[DorsalRegion]:
        return [label.acronym for label in self.region_labels]

    @property
    def region_dict(self) -> dict[DorsalRegion, int]:
        """region int dict"""
        return {
            label.acronym: label.label
            for label in self.region_labels
        }

    def _reset_fov(self, x: int | None = None,
                   y: int | None = None):

        cur_x, cur_y = self._fov_mm
        new_x = x if x is not None else cur_x
        new_y = y if y is not None else cur_y
        self._fov_mm = (new_x, new_y)

    def select_region(self, acronym: DorsalRegion | Sequence[DorsalRegion]) -> Self:
        """select a specific region or list of region"""
        acronym_list = [acronym] if isinstance(acronym, str) else acronym

        unknown = [ac for ac in acronym_list if ac not in self.region_list]
        if unknown:
            raise ValueError(f'Unknown region(s): {unknown}. Available: {self.region_list}')

        selected = [r for r in self.region_labels if r.acronym in acronym_list]
        return DorsalCCF(selected, self.image_shape, self._hemisphere)

    def select_hemisphere(self, hemisphere: HEMISPHERE_TYPE) -> Self:
        return DorsalCCF(
            self.region_labels,
            self.image_shape,
            hemisphere=hemisphere
        )

    def to_numpy(self) -> np.ndarray:
        if self.__array is not None:
            return self.__array
        else:
            from matplotlib.path import Path

            h, w = self.image_shape
            label_map = np.zeros((h, w), dtype=np.uint16)
            for region in self.region_labels:

                match self.hemisphere:
                    case 'left':
                        x_mm = region.left_x
                        y_mm = region.left_y
                    case 'right':
                        x_mm = region.right_x
                        y_mm = region.right_y
                    case 'both':
                        x_mm = np.concatenate([region.left_x, region.right_x])
                        y_mm = np.concatenate([region.left_y, region.right_y])
                    case _:
                        raise ValueError('')

                x_pix = ((x_mm / self.fov_mm[0]) + 0.5) * w
                y_pix = ((y_mm / self.fov_mm[1]) + 0.5) * h
                poly = np.stack([x_pix, y_pix], axis=1)

                path = Path(poly)
                coords = np.vstack(np.meshgrid(np.arange(w), np.arange(h))).reshape(2, -1).T
                mask = path.contains_points(coords).reshape(h, w)
                label_map[mask] = region.label

            self.__array = label_map

        return label_map

    def to_boundary(self, outlines_only: bool = True) -> np.ndarray:
        from skimage.segmentation import find_boundaries
        boundary = find_boundaries(self.to_numpy(), mode='outer')
        if outlines_only:
            boundary = np.ma.masked_where(~boundary, boundary)
        return boundary


def validate_ccf_array(array: np.ndarray) -> None:
    """
    check if is a ccf array

    :param array: ccf array. `Array[int, [H, W]]`
    """
    if not isinstance(array, np.ndarray):
        raise TypeError('mask must be a numpy array')

    if array.ndim != 2:
        raise ValueError('mask must be a 2D numpy array')

    # value
    if array.dtype == np.bool_:
        return
    else:
        unique_vals = set(np.unique(array)) - {0}
        valid_vals = set(DorsalCCF.from_json().region_dict.values())

        if not unique_vals.issubset(valid_vals):
            raise ValueError('mask values are not matched with dorsal ccf arrays')
