Position Decoding
=================

Bayesian decoding of spatial position from population neural activity.

**Refer to API**: :doc:`../api/neuralib.decoding`

place_bayes
-----------

Decode animal position using Bayesian inference on population firing rates.

This function implements a naive Bayesian decoder that estimates spatial position
from the firing rates of a population of neurons, given their spatial tuning curves (rate maps).

**Algorithm:**

Given firing rates ``fr`` and rate maps ``rate_map``, computes the posterior probability
of the animal being at each spatial bin using Bayes' theorem with a uniform prior.

**Dimension Parameters:**

- ``N``: number of neurons
- ``T``: number of temporal bins
- ``X``: number of spatial bins

**Usage:**

.. code-block:: python

    import numpy as np
    from neuralib.decoding.position import place_bayes

    # Example: 50 neurons, 100 time bins, 40 spatial bins
    n_neurons = 50
    n_time_bins = 100
    n_spatial_bins = 40
    spatial_bin_size = 2.5  # cm

    # Firing rates: (T, N) array
    fr = np.random.rand(n_time_bins, n_neurons)

    # Rate maps (spatial tuning curves): (X, N) array
    rate_map = np.random.rand(n_spatial_bins, n_neurons)

    # Decode position
    posterior = place_bayes(fr, rate_map, spatial_bin_size)
    # posterior shape: (T, X) - probability of each position at each time

    # Get decoded position (maximum a posteriori)
    decoded_position = np.argmax(posterior, axis=1)  # (T,)

**Parameters:**

- ``fr``: Firing rate array of shape ``(T, N)``
- ``rate_map``: Spatial tuning curves of shape ``(X, N)``
- ``spatial_bin_size``: Spatial bin size in cm

**Returns:**

- Posterior probability matrix of shape ``(T, X)``

.. seealso::

    `buzcode placeBayes.m <https://github.com/buzsakilab/buzcode/blob/master/analysis/positionDecoding/placeBayes.m>`_