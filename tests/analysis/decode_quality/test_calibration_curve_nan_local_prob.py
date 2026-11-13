"""Regression test for ``compute_calibration_curve`` NaN handling.

``np.digitize`` returns ``len(bin_edges)`` for NaN inputs, which
after the function's ``.clip(0, n_bins - 1)`` mapped every NaN
``Local_prob`` row into the **highest** probability bin. Decoder
errors at NaN-Local_prob timepoints (often crashes / data drops)
then dominated the highest bin's median error.

Fix: drop NaN ``Local_prob`` rows before binning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.decode_quality.quality_metrics import (
    compute_calibration_curve,
)


class TestCalibrationCurveNaNLocalProb:
    def test_nan_local_prob_does_not_pollute_high_bin(self):
        # Three valid rows in distinct bins + one NaN row whose
        # ``ahead_behind_distance`` is huge. The NaN row used to land
        # in prob_bin_center≈0.95 and dominate that bin's median.
        data = pd.DataFrame(
            {
                "animal": ["m"] * 4,
                "day": [1] * 4,
                "epoch": ["e"] * 4,
                "Local_prob": [0.05, 0.15, 0.95, np.nan],
                "ahead_behind_distance": [3.0, 2.0, 1.0, 500.0],
                "in_excluded_time": False,
            },
            index=[0.0, 0.1, 0.2, 0.3],
        )
        out = compute_calibration_curve(data, n_bins=10)
        # Highest bin (≈0.95) should reflect only the valid 0.95 row.
        high = out.loc[out["prob_bin_center"].round(2) == 0.95]
        assert len(high) == 1
        assert high["median_error_cm"].iloc[0] == 1.0, (
            f"highest-bin median is {high['median_error_cm'].iloc[0]}; "
            "NaN row contaminated the high bin (expected median = 1.0 "
            "from the lone valid row)."
        )
