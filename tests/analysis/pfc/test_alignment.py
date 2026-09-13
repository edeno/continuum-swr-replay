"""Value-level tests for the shared-timestamp alignment on the PFC load path.

HPC decodes saved before ``get_position_info`` was trimmed carry a few more
time bins than ``load_data`` now returns for the same epoch, so
:func:`validate_results_position_time_alignment` — which refuses any length
difference — rejects the pair at load.
:func:`~continuum_swr_replay.analysis.hippocampus.data_aggregation.align_results_to_position`
restricts both objects to the timestamps they share, but only when the
mismatch is a small head/tail trim: an interleaved mismatch, an oversized
trim, or drift inside the shared run must still raise rather than silently
re-align data that does not share a timebase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    align_results_to_position,
)

SAMPLING_FREQUENCY_HZ = 500.0
DT = 1.0 / SAMPLING_FREQUENCY_HZ


@pytest.fixture
def time_axis():
    """10-second epoch at 500 Hz."""
    return np.arange(0.0, 10.0, DT)


def _make_results(time: np.ndarray) -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "acausal_state_probabilities": (
                ("time", "states"),
                rng.random((len(time), 2)),
            )
        },
        coords={"time": time, "states": ["Local", "Non-Local Continuous"]},
    )


def _make_position(time: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"head_speed": np.linspace(0.0, 5.0, len(time))},
        index=pd.Index(time, name="time"),
    )


class TestIdentityOnMatchedAxes:
    def test_matched_axes_are_returned_untouched(self, time_axis):
        results = _make_results(time_axis)
        position_info = _make_position(time_axis)

        aligned_results, aligned_position, counts = align_results_to_position(
            results, position_info
        )

        assert counts == {
            "n_results_bins_dropped": 0,
            "n_position_rows_dropped": 0,
            "dropped_seconds_results": 0.0,
            "dropped_seconds_position": 0.0,
            "dropped_at": "none",
        }
        # Bit-for-bit: a session with no mismatch must see no change at all.
        xr.testing.assert_identical(aligned_results, results)
        pd.testing.assert_frame_equal(aligned_position, position_info, check_exact=True)
        assert np.array_equal(
            aligned_results.acausal_state_probabilities.to_numpy(),
            results.acausal_state_probabilities.to_numpy(),
        )


class TestHeadAndTailTrims:
    def test_position_trimmed_at_tail_drops_results_bins(self, time_axis):
        """The real failure mode: position is 5 samples shorter at the end."""
        results = _make_results(time_axis)
        position_info = _make_position(time_axis[:-5])

        aligned_results, aligned_position, counts = align_results_to_position(
            results, position_info
        )

        assert counts["n_results_bins_dropped"] == 5
        assert counts["n_position_rows_dropped"] == 0
        assert counts["dropped_at"] == "tail"
        assert counts["dropped_seconds_results"] == pytest.approx(5 * DT)
        assert counts["dropped_seconds_position"] == 0.0
        assert len(aligned_results.time) == len(aligned_position) == len(time_axis) - 5
        np.testing.assert_allclose(
            aligned_results.time.to_numpy(),
            aligned_position.index.to_numpy(),
            atol=1e-6,
        )
        # The kept bins are the original leading bins, values untouched.
        assert np.array_equal(
            aligned_results.acausal_state_probabilities.to_numpy(),
            results.acausal_state_probabilities.to_numpy()[:-5],
        )

    def test_position_trimmed_at_head_drops_leading_results_bins(self, time_axis):
        results = _make_results(time_axis)
        position_info = _make_position(time_axis[3:])

        aligned_results, aligned_position, counts = align_results_to_position(
            results, position_info
        )

        assert counts["n_results_bins_dropped"] == 3
        assert counts["n_position_rows_dropped"] == 0
        assert counts["dropped_at"] == "head"
        assert counts["dropped_seconds_results"] == pytest.approx(3 * DT)
        assert aligned_results.time.to_numpy()[0] == pytest.approx(time_axis[3])
        assert np.array_equal(
            aligned_results.acausal_state_probabilities.to_numpy(),
            results.acausal_state_probabilities.to_numpy()[3:],
        )

    def test_trims_at_both_ends_report_both(self, time_axis):
        results = _make_results(time_axis)
        position_info = _make_position(time_axis[3:-4])

        _, _, counts = align_results_to_position(results, position_info)

        assert counts["n_results_bins_dropped"] == 7
        assert counts["n_position_rows_dropped"] == 0
        assert counts["dropped_at"] == "both"
        assert counts["dropped_seconds_results"] == pytest.approx(7 * DT)

    def test_each_axis_can_lose_a_different_end(self, time_axis):
        """Partial overlap: results starts later, position ends later."""
        results = _make_results(time_axis[4:])
        position_info = _make_position(time_axis[:-6])

        aligned_results, aligned_position, counts = align_results_to_position(
            results, position_info
        )

        assert counts["n_results_bins_dropped"] == 6
        assert counts["n_position_rows_dropped"] == 4
        assert counts["dropped_at"] == "both"
        assert counts["dropped_seconds_results"] == pytest.approx(6 * DT)
        assert counts["dropped_seconds_position"] == pytest.approx(4 * DT)
        np.testing.assert_allclose(
            aligned_results.time.to_numpy(), time_axis[4:-6], atol=1e-6
        )
        np.testing.assert_allclose(
            aligned_position.index.to_numpy(), time_axis[4:-6], atol=1e-6
        )


class TestRefusals:
    def test_interleaved_mismatch_raises(self, time_axis):
        """An interior row missing is not a head/tail trim."""
        results = _make_results(time_axis)
        keep = np.ones(len(time_axis), dtype=bool)
        keep[1000] = False
        position_info = _make_position(time_axis[keep])

        with pytest.raises(ValueError, match="interleaved") as excinfo:
            align_results_to_position(results, position_info)

        message = str(excinfo.value)
        assert str(len(time_axis)) in message
        assert str(len(time_axis) - 1) in message
        assert f"{time_axis[1000]:.6f}" in message

    def test_oversized_trim_raises(self, time_axis):
        """A trim bigger than the budget is a different timebase, not a trim."""
        results = _make_results(time_axis)
        position_info = _make_position(time_axis[: -int(2.0 * SAMPLING_FREQUENCY_HZ)])

        with pytest.raises(ValueError, match="over the 1.0s budget"):
            align_results_to_position(results, position_info)

        # The same mismatch is accepted once the budget covers it.
        _, _, counts = align_results_to_position(
            results, position_info, max_dropped_seconds=3.0
        )
        assert counts["n_results_bins_dropped"] == int(2.0 * SAMPLING_FREQUENCY_HZ)

    def test_no_shared_timestamps_raises(self, time_axis):
        results = _make_results(time_axis + 100.0)
        position_info = _make_position(time_axis)

        with pytest.raises(ValueError, match="share no timestamps"):
            align_results_to_position(results, position_info)

    def test_drift_inside_shared_run_raises_via_validator(self, time_axis):
        """A loose ``atol`` must not buy a silently time-shifted pair."""
        results = _make_results(time_axis + 1e-4)
        position_info = _make_position(time_axis)

        with pytest.raises(ValueError, match="drifts from position_info"):
            align_results_to_position(results, position_info, atol=1e-3)

    def test_unsorted_axis_raises(self, time_axis):
        results = _make_results(time_axis)
        shuffled = time_axis.copy()
        shuffled[[10, 11]] = shuffled[[11, 10]]
        position_info = _make_position(shuffled)

        with pytest.raises(ValueError, match="not strictly increasing"):
            align_results_to_position(results, position_info)
