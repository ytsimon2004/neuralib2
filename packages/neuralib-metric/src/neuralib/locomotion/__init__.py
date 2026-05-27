"""
Locomotion
==========

Animal locomotion utilities for epoch detection, position tracking, and spatial metrics.

**Epoch detection** (``neuralib.locomotion.epoch``):

- Identify running vs. stationary epochs on a 1D linear track (:func:`~neuralib.locomotion.epoch.running_mask1d`)
- Remove motion-artefact frames from 2D tracking (:func:`~neuralib.locomotion.epoch.jump_mask2d`)

**Position** (``neuralib.locomotion.position``):

- :class:`~neuralib.locomotion.position.CircularPosition` — lap-structured position container (time, position, displacement, velocity, trial indices)
- Linear interpolation and gap-filling for 1D and 2D trajectories
- Speed and direction computation for 2D open-field tracking

**Spatial metrics** (``neuralib.locomotion.spatial``):

- Spatial information score (bits/event) from occupancy-weighted activity maps (:func:`~neuralib.locomotion.spatial.spatial_info`)
- Place field detection with width, peak location, and trial-reliability filters (:func:`~neuralib.locomotion.spatial.place_field`)
"""

from .epoch import *
from .position import *
from .spatial import *
