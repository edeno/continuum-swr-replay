"""Regression test: ``check_transition_matrix_flags`` must flag rows
with any NaN transition entry.

The structural row-sum check used to call ``out[present].sum(axis=1)``
with pandas' default ``skipna=True``, so a row with 3 of 4 transitions
summing to 1.0 and the fourth NaN passed the check despite being an
incomplete transition matrix (the user reproduced this with
``T_local_nlc = NaN``). The fix uses ``min_count=4`` plus an explicit
NaN-presence check so any incomplete row is flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.decode_quality.hpc_raw_stats import (
    check_transition_matrix_flags,
)


def _summary_row(**transitions: float) -> pd.DataFrame:
    """Build a one-row summary covering all 16 ``T_<src>_<dst>`` columns.

    Defaults to a realistic, no-error matrix (diagonals safely below
    the upper-bound thresholds and above the lower-bound errors;
    Local off-diagonals < 0.02; NL cross-types below their
    self-transitions). Caller overrides specific entries via kwargs.
    """
    base: dict[str, float] = {
        # Local row
        "T_local_local": 0.99,
        "T_local_nospike": 0.005,
        "T_local_nlc": 0.003,
        "T_local_nlf": 0.002,
        # No-Spike row
        "T_nospike_local": 0.04,
        "T_nospike_nospike": 0.95,
        "T_nospike_nlc": 0.005,
        "T_nospike_nlf": 0.005,
        # NLC row
        "T_nlc_local": 0.04,
        "T_nlc_nospike": 0.005,
        "T_nlc_nlc": 0.94,
        "T_nlc_nlf": 0.015,
        # NLF row (cross-type < self-transition)
        "T_nlf_local": 0.04,
        "T_nlf_nospike": 0.005,
        "T_nlf_nlc": 0.025,
        "T_nlf_nlf": 0.93,
    }
    base.update(transitions)
    return pd.DataFrame([base])


class TestTransitionMatrixFlagNaNRow:
    def test_nan_in_local_row_with_other_three_summing_to_one_is_flagged(self):
        # User repro: T_local_nlc = NaN, the other three Local
        # transitions sum to 1.0 cleanly. Pre-fix structural check
        # used ``skipna=True`` and let this through.
        df = _summary_row(
            T_local_local=0.99,
            T_local_nospike=0.005,
            T_local_nlc=np.nan,
            T_local_nlf=0.005,
        )
        out = check_transition_matrix_flags(df)
        assert bool(out["flag_transition_error"].iloc[0]) is True, (
            "row with one NaN transition slipped past structural check"
        )

    def test_complete_valid_row_is_not_flagged(self):
        df = _summary_row()
        out = check_transition_matrix_flags(df)
        assert bool(out["flag_transition_error"].iloc[0]) is False, (
            "valid transition matrix was falsely flagged"
        )

    def test_nan_in_off_diagonal_row_is_flagged(self):
        # NoSpike row: replace T_nospike_nlc with NaN.
        df = _summary_row(T_nospike_nlc=np.nan)
        out = check_transition_matrix_flags(df)
        assert bool(out["flag_transition_error"].iloc[0]) is True
