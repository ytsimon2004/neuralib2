from typing import NamedTuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from typing_extensions import Self

__all__ = [
    'SpatialInfoResult',
    'spatial_info',
    'PlaceFieldResult',
    'place_field',
]


class SpatialInfoResult(NamedTuple):
    """Result of spatial information calculation.

    `Dimension parameters`:

        B = number of spatial bins
    """

    position: np.ndarray
    """Bin edges. `Array[float, B+1]`"""
    occupancy: np.ndarray
    """Time spent per bin (smoothed). `Array[float, B]`"""
    activity: np.ndarray
    """Mean activity per bin (smoothed). `Array[float, B]`"""
    spatial_info: float
    """Spatial information score in bits/event (Skaggs et al., 1996)"""


def spatial_info(t: np.ndarray,
                 x: np.ndarray,
                 ta: np.ndarray,
                 a: np.ndarray,
                 x_bins: int = 100,
                 epoch: np.ndarray | None = None) -> SpatialInfoResult:
    """Calculate spatial information score for a single neuron on a 1D linear track.

    Note: the absolute value is sensitive to the df/f normalization method.

    `Dimension parameters`:

        P = number of position samples

        F = number of activity samples

        B = number of spatial bins

    :param t: Position time array. `Array[float, P]`
    :param x: Normalized position array in [0, 1]. `Array[float, P]`
    :param ta: Activity time array. `Array[float, F]`
    :param a: Activity value array. `Array[float, F]`
    :param x_bins: Number of spatial bins
    :param epoch: Boolean mask for restricting to a specific epoch (e.g., running).
        Must have the same length as ``t`` and ``x``. `Array[bool, P]`
    :return: ``SpatialInfoResult``
    """
    bins = np.linspace(0, 1, num=x_bins + 1, endpoint=True)
    at = interp1d(ta, a, bounds_error=False, fill_value=0)(t)
    dt = np.diff(t, prepend=t[0])

    if epoch is not None:
        x = x[epoch]
        at = at[epoch]
        dt = dt[epoch]

    count = gaussian_filter1d(np.histogram(x, bins)[0], 3, mode='wrap')
    occupancy = gaussian_filter1d(np.histogram(x, bins, weights=dt)[0], 3, mode='wrap')
    activity = gaussian_filter1d(np.histogram(x, bins, weights=at)[0], 3, mode='wrap') / count

    si = _spatial_info(occupancy, activity)

    return SpatialInfoResult(bins, occupancy, activity, si)


def _spatial_info(occupancy: np.ndarray, activity: np.ndarray) -> float:
    r"""Compute spatial information score (bits/event).

    .. math:: \sum_{i=1}^n P_i \frac{\lambda_i}{\lambda} \log_2\!\left(\frac{\lambda_i}{\lambda}\right)

    Skaggs et al., 1996.

    :param occupancy: Time per bin. `Array[float, B]`
    :param activity: Mean activity per bin. `Array[float, B]`
    :return: Spatial information in bits/event
    """
    occp = occupancy / np.sum(occupancy)
    mean_act = np.sum(occp * activity)
    ratio = activity / mean_act
    occp = occp[ratio > 0]
    ratio = ratio[ratio > 0]
    return float(np.sum(occp * ratio * np.log2(ratio)))


class PlaceFieldResult(NamedTuple):
    """Result of place field detection.

    `Dimension parameters`:

        B = number of spatial bins
    """

    start: int
    """First valid bin index (always 0)"""
    end: int
    """Last valid bin index (= window - 1)"""
    bin_size: float
    """Spatial bin size in cm"""
    pf: list[tuple[int, int]]
    """Detected place fields as (start_bin, end_bin) index pairs"""
    act: np.ndarray
    """Trial-averaged transient activity. `Array[float, B]`"""
    baseline: float
    """Trial-averaged baseline activity (scalar)"""
    threshold: float
    """Amplitude threshold used for field detection"""

    @property
    def n_pf(self) -> int:
        """Number of detected place fields"""
        return len(self.pf)

    @property
    def pf_width(self) -> list[float]:
        """Width of each place field in cm"""
        return [self.bin_size * float(it[1] - it[0]) for it in self.pf]

    @property
    def pf_peak(self) -> list[float]:
        """Peak location of each place field in cm"""
        peaks = []
        for p in self.pf:
            bins = np.arange(p[0], p[1]) % (self.end + 1)  # modulo handles PP wrap-around
            peak_bin = bins[np.argmax(self.act[bins])]
            peaks.append(self.bin_size * float(peak_bin))
        return peaks

    def with_width_filter(self, place_field_range: tuple[int, int]) -> Self:
        """Return a copy keeping only fields within the given width range.

        :param place_field_range: ``(min_cm, max_cm)`` width bounds (exclusive)
        """
        lo, hi = place_field_range
        ret = [it for it in self.pf if lo < self.bin_size * (it[1] - it[0]) < hi]
        return self._replace(pf=ret)

    def with_reliability_filter(self, reliability: list[float], at_least: float = 0.33) -> Self:
        """Return a copy keeping only fields active in at least ``at_least`` fraction of trials.

        :param reliability: Per-field fraction of trials in which the field was active
        :param at_least: Minimum reliability threshold
        """
        if len(self.pf) != len(reliability):
            raise ValueError('length of reliability does not match number of place fields')
        pf_arr = np.array(self.pf)
        mask = [r > at_least for r in reliability]
        return self._replace(pf=list(pf_arr[mask]))

    def with_peak_filter(self, greater_than: float | None, less_than: float | None) -> Self:
        """Return a copy keeping only fields whose peak location passes the given bounds.

        :param greater_than: Exclude fields with peak < this value (cm)
        :param less_than: Exclude fields with peak > this value (cm)
        """
        ret = []
        for pf, peak in zip(self.pf, self.pf_peak):
            if greater_than is not None and peak < greater_than:
                continue
            if less_than is not None and peak > less_than:
                continue
            ret.append(pf)
        return self._replace(pf=ret)


def place_field(signal: np.ndarray,
                baseline: np.ndarray,
                threshold: float,
                window: int = 100,
                track_length: int = 150) -> PlaceFieldResult:
    """Detect place fields from position-binned calcium activity.

    Bins where the trial-averaged activity exceeds ``threshold`` fraction of
    the (peak − baseline) range are labelled as place field regions.
    Four boundary cases (NN / NP / PN / PP) handle fields that start or end
    at the track edges.

    .. seealso:: Mao et al., 2017. Nature Communications

    `Dimension parameters`:

        L = number of laps (trials)

        B = number of spatial bins

    :param signal: Position-binned transient activity. `Array[float, [L, B]]`
    :param baseline: Position-binned baseline activity. `Array[float, [L, B]]`
    :param threshold: Fraction of (peak − baseline) used as the detection threshold
    :param window: Number of spatial bins
    :param track_length: Track length in cm
    :return: ``PlaceFieldResult``
    """
    act = np.nanmean(signal, axis=0)
    bas = float(np.mean(np.nanmean(baseline, axis=0)))

    act_threshold = (np.max(act) - bas) * threshold + bas
    bin_size = track_length / window

    crossings = np.nonzero(np.diff((act > act_threshold).astype(int)) != 0)[0]
    left = float((act - act_threshold)[0])
    right = float((act - act_threshold)[-1])

    start = 0
    end = window - 1

    if left < 0 and right < 0:      # NN: fields entirely within track
        pf_idx = [(crossings[i], crossings[i + 1]) for i in range(0, len(crossings), 2)]
    elif left < 0 < right:           # NP: last field extends past right edge
        pf_idx = [
            (crossings[i], end) if i + 1 == len(crossings) else (crossings[i], crossings[i + 1])
            for i in range(0, len(crossings), 2)
        ]
    elif right < 0 < left:           # PN: first field starts at left edge
        pf_idx = [(start, crossings[0])] + [
            (crossings[i], crossings[i + 1]) for i in range(1, len(crossings), 2)
        ]
    elif left > 0 and right > 0:     # PP: activity wraps around both edges
        pf_idx = [
            (crossings[i], end + crossings[0]) if i + 1 == len(crossings) else (crossings[i], crossings[i + 1])
            for i in range(1, len(crossings), 2)
        ]
    else:
        pf_idx = []

    return PlaceFieldResult(start, end, bin_size, pf_idx, act, bas, act_threshold)