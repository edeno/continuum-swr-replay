"""Regression test for the drift check in HPC + ContFrag get_data_over_time.

The aggregations build ``data_over_time`` on ``position_info.index``
but read column values from ``results.time``-indexed arrays. A
length-only check passed when the two axes had the same row count
but were time-shifted; that silently attached event labels /
probabilities / MAP stats to the wrong position rows.

This test pins the stricter ``np.abs(results.time - position.index).max() > 1e-6``
guard that now matches ``analysis.replay._helpers._align_posterior_to_dot``.
We do not import the full functions (they would drag in spyglass /
non_local_detector); the guard logic is identical between the two
aggregation paths, so we replay it inline and assert behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _drift_guard(results_time: np.ndarray, position_index: pd.Index) -> None:
    """The same guard now used in both ``get_data_over_time`` paths."""
    if len(results_time) != len(position_index):
        raise ValueError(
            f"results.time ({len(results_time)}) and position_info "
            f"({len(position_index)}) have different lengths; "
            "they must be aligned 1-to-1."
        )
    drift = float(
        np.abs(
            np.asarray(results_time, dtype=float)
            - position_index.to_numpy(dtype=float)
        ).max()
    )
    if drift > 1e-6:
        raise ValueError(
            f"results.time drifts from position_info.index by up to "
            f"{drift:.3g}s — same-length axes do not imply same time "
            "origin; refusing to attach decoder outputs to position "
            "rows that don't share their timebase."
        )


class TestDriftGuard:
    def test_aligned_axes_pass(self):
        t = np.linspace(100.0, 110.0, 1000)
        idx = pd.Index(t, name="time")
        _drift_guard(t, idx)  # no raise

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="different lengths"):
            _drift_guard(np.arange(100), pd.Index(np.arange(99)))

    def test_same_length_time_shifted_axes_raise(self):
        t = np.linspace(100.0, 110.0, 1000)
        idx = pd.Index(t + 0.5, name="time")  # shifted by 0.5s
        with pytest.raises(ValueError, match="drifts from position_info.index"):
            _drift_guard(t, idx)

    def test_within_tolerance_passes(self):
        t = np.linspace(100.0, 110.0, 1000)
        idx = pd.Index(t + 1e-9, name="time")  # well below 1e-6
        _drift_guard(t, idx)  # no raise


def test_aggregation_modules_use_the_drift_guard():
    """Pin both ``get_data_over_time`` definitions through the same
    drift-check pattern. We grep the source rather than execute the
    function (which needs xarray + non_local_detector + spyglass).
    """
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    hpc_path = (
        repo_root
        / "src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py"
    )
    contfrag_path = (
        repo_root
        / "src/continuum_swr_replay/analysis/contfrag/data_aggregation.py"
    )
    for p in (hpc_path, contfrag_path):
        src = p.read_text()
        assert "drifts from position_info.index" in src, (
            f"{p} no longer carries the strict drift-check error string"
        )
        assert "1e-6" in src, f"{p} missing the 1e-6 drift tolerance constant"
