"""An epoch's run-speed baseline, shared by the decoders and the read path.

``run_speed_baseline`` is a robust, threshold-free per-epoch speed normalizer:
a percentile of head speed over the epoch. The trajectory rollups divide an
event's replay speed by it (``trajectory``'s ``compression_factor``).

Both decoders' aggregators (``data_aggregation.get_data_over_time`` and the
ContFrag aggregator) write it into the decode frame over every bin. The read
through the SWR source (``swr_source.with_swr_source``) computes it again over
the bins outside the artifact mask, because it is an input: inputs use the
mask itself, as the ripple-power covariate does. The theta-power z-score, the
frame's other epoch normalizer, is re-z-scored there too.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["RUN_SPEED_BASELINE_PERCENTILE", "run_speed_baseline"]

RUN_SPEED_BASELINE_PERCENTILE = 90
"""The percentile of head speed that is an epoch's run-speed baseline."""


def run_speed_baseline(head_speed: ArrayLike) -> float:
    """The :data:`RUN_SPEED_BASELINE_PERCENTILE` of head speed, NaN ignored.

    Parameters
    ----------
    head_speed : array-like, shape (n_time,)
        Head speed (cm/s) of the bins the baseline is taken over.

    Returns
    -------
    float
        cm/s.

    Raises
    ------
    ValueError
        If the baseline is not finite (every sample NaN).
    """
    baseline = float(
        np.nanpercentile(np.asarray(head_speed), RUN_SPEED_BASELINE_PERCENTILE)
    )
    if not np.isfinite(baseline):
        raise ValueError("run_speed_baseline is not finite (head_speed all-NaN?)")
    return baseline
