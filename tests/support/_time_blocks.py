"""Simple folds for the held-out tests: contiguous blocks of time.

Not a design the runners use (they hold out folds of whole trials,
``tetrode_holdout.trial_folds``); any folds array is a valid input to the
held-out functions, and these are the easiest to reason about in a test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import N_FOLDS

TIME_BLOCK_MARGIN_BINS = 2
"""The margin used with :func:`time_blocks`, bins: the library pads each
training run by one bin width, and two bins leave no sliver of the block in
the fit."""


def time_blocks(time: NDArray, n_folds: int = N_FOLDS) -> NDArray[np.int64]:
    """Each bin's block: ``n_folds`` contiguous blocks of equal duration.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        Time, s, increasing.
    n_folds : int

    Returns
    -------
    ndarray of int, shape (n_time,)
        0 .. n_folds - 1.
    """
    time = np.asarray(time, dtype=float)
    span = time[-1] - time[0]
    block = np.floor((time - time[0]) / span * n_folds).astype(np.int64)
    return np.minimum(block, n_folds - 1)
