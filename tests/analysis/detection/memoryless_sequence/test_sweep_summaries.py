"""The memoryless sweeps' long-format summaries, on tables small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    sweep_summaries,
)

DETECTORS = (
    "sliding_accepted",
    "sliding_published",
    "state_space_canonical",
    "state_space_continuous",
)
"""The controls sweep's four scored outputs."""


def replay_cells() -> pd.DataFrame:
    """Four injected replays. Cells 0 and 3 are in the published definition's
    domain (path >= 45 cm, duration >= 0.12 s); cell 1's path and cell 2's
    duration are too short. Cells 0, 1 and 3 are matched by both detectors
    before the shuffle, which cell 1 then fails: cells 0 and 3 are the replays
    both detectors report. A timing error is against the replay's own first
    or last spike: the sliding window starts cell 0 3 ms early and cell 3 10 ms
    late; the state-space detector starts cell 0 4 ms early; both end cell 0
    20 ms late."""
    n = 4
    return pd.DataFrame(
        {
            "path_cm": [45.0, 30.0, 60.0, 90.0],
            "duration_s": [0.12, 0.12, 0.08, 0.12],
            "sliding_found": [True, False, False, True],
            "sliding_matches": [True, False, False, True],
            "sliding_matches_before_shuffle": [True, True, False, True],
            "sliding_matches_spread_only": [True, True, False, True],
            "sliding_any_candidate": np.ones(n, dtype=bool),
            "sliding_p_spread": [0.01, 0.5, 0.01, 0.01],
            "sliding_p_jump": [0.01, 0.01, 0.01, 0.01],
            "sliding_n_accepted": np.ones(n, dtype=np.int64),
            "sliding_n_candidates": np.ones(n, dtype=np.int64),
            "sliding_best_dispersion_cm": np.full(n, 20.0),
            "sliding_spread_ok": np.full(n, 0.5),
            "state_space_found": np.ones(n, dtype=bool),
            "state_space_matches": np.ones(n, dtype=bool),
            "state_space_replay_matches": [True, True, True, False],
            "state_space_n_intervals": np.ones(n, dtype=np.int64),
            "sliding_start_error_s": [-0.003, 0.002, 0.0, 0.010],
            "sliding_end_error_s": [0.020, 0.0, 0.0, 0.0],
            "state_space_start_error_s": [-0.004, 0.0, 0.0, 0.0],
            "state_space_end_error_s": [0.020, 0.0, 0.0, 0.0],
        }
    )


class TestReplaySweepSummary:
    def values(self, block: str) -> pd.Series:
        summary = sweep_summaries.replay_sweep_summary(replay_cells())
        return summary[summary.summary == block]

    def test_the_domain_block_reads_only_the_cells_the_definition_can_accept(self):
        overall = self.values("overall").set_index("measure").value
        domain = self.values("overall_in_domain").set_index("measure").value

        assert (overall["n_cells"], domain["n_cells"]) == (4, 2)
        assert overall["state_space_replay_matches"] == pytest.approx(0.75)
        assert domain["state_space_replay_matches"] == pytest.approx(0.5)

    def test_pooled_timing_errors_are_of_the_replays_both_report(self):
        pooled = self.values("timing_pooled").set_index("measure").value

        # cells 0 and 3: sliding start errors -3 and 10 ms
        assert pooled["n_replays"] == 2
        assert pooled["sliding_start_error_ms_min"] == pytest.approx(-3.0)
        assert pooled["sliding_start_error_ms_median"] == pytest.approx(3.5)
        assert pooled["sliding_start_error_ms_max"] == pytest.approx(10.0)
        # the state-space detector starts cell 0 4 ms before its first spike
        # and ends it 20 ms after its last
        assert pooled["state_space_start_error_ms_min"] == pytest.approx(-4.0)
        assert pooled["state_space_end_error_ms_max"] == pytest.approx(20.0)
        # the sliding window's start error is the larger on cell 3 only
        # (3 < 4, 10 > 0), its end error on neither (20 = 20, 0 = 0)
        assert pooled["sliding_larger_start_share"] == 0.5
        assert pooled["sliding_larger_end_share"] == 0.0

    def test_a_far_edge_is_one_beyond_the_threshold(self):
        cells = replay_cells()
        cells.loc[3, "sliding_end_error_s"] = 0.150
        pooled = sweep_summaries.pooled_timing_errors(
            cells[cells.sliding_matches & cells.state_space_matches]
        )

        assert sweep_summaries.LARGE_TIMING_ERROR_MS == 100.0
        assert pooled["sliding_end_error_ms_far_share"] == 0.5
        assert pooled["state_space_end_error_ms_far_share"] == 0.0

    def test_exceedance_is_the_share_of_replays_beyond_each_error_size(self):
        shares = self.values("timing_exceedance").set_index("measure").value

        # the sliding window's start errors against the first spike: 3 and 10 ms
        assert shares["sliding_start_error_beyond_0ms"] == 1.0
        assert shares["sliding_start_error_beyond_2ms"] == 1.0
        assert shares["sliding_start_error_beyond_3ms"] == 0.5
        assert shares["sliding_start_error_beyond_10ms"] == 0.0
        # the state-space detector's end errors: 20 and 0 ms
        assert shares["state_space_end_error_beyond_0ms"] == 0.5
        assert shares["state_space_end_error_beyond_20ms"] == 0.0
        assert len(shares) == 4 * len(sweep_summaries.EXCEEDANCE_THRESHOLDS_MS)

    def test_the_far_share_is_the_exceedance_at_its_threshold(self):
        cells = replay_cells()
        cells.loc[3, "sliding_end_error_s"] = 0.150
        found = cells[cells.sliding_matches & cells.state_space_matches]
        far_ms = int(sweep_summaries.LARGE_TIMING_ERROR_MS)

        assert (
            sweep_summaries.timing_error_exceedance(found)[
                f"sliding_end_error_beyond_{far_ms}ms"
            ]
            == sweep_summaries.pooled_timing_errors(found)[
                "sliding_end_error_ms_far_share"
            ]
            == 0.5
        )

    def test_the_misses_are_the_state_space_matches_the_window_lost(self):
        overall = self.values("overall").set_index("measure").value

        # cells 1 and 2: cell 1 was matched before the shuffle, cell 2 was not
        assert overall["missed_failed_shuffle"] == pytest.approx(0.5)
        assert overall["found_by_state_space_only"] == pytest.approx(0.5)
        assert overall["sliding_accepted_but_not_the_injection"] == 0.0

    def test_the_shuffle_pass_rates_are_among_the_replays_reported_before_it(self):
        overall = self.values("overall").set_index("measure").value

        assert overall["sliding_shuffle_spread_pass_rate"] == pytest.approx(2 / 3)
        assert overall["sliding_shuffle_jump_pass_rate"] == pytest.approx(1.0)


def test_a_share_of_nothing_is_undefined_not_zero():
    assert np.isnan(sweep_summaries.share(0, 0))
    assert sweep_summaries.share(1, 4) == pytest.approx(0.25)


class TestControlsSummary:
    def test_immobile_time_is_counted_once_per_stretch(self):
        no_replay = pd.DataFrame(
            [
                {
                    "recording": 0,
                    "stretch": stretch,
                    "exposure_s": 30.0,
                    "detector": detector,
                    "n_events": 1 if detector == "state_space_canonical" else 0,
                    "event_time_s": 0.3 if detector == "state_space_canonical" else 0.0,
                }
                for stretch in range(2)
                for detector in DETECTORS
            ]
        )
        trials = pd.DataFrame(
            columns=["condition", "level", "detector", "called", "peak_rate_hz"]
        )

        summary = sweep_summaries.controls_summary(trials, no_replay)
        rows = summary.set_index(["measure", "detector"]).value

        assert rows["immobile_minutes", ""] == pytest.approx(1.0)
        assert rows["events_per_minute", "state_space_canonical"] == pytest.approx(2.0)
        assert rows["time_share", "state_space_canonical"] == pytest.approx(0.01)

    def test_losses_to_the_context_are_counted_even_when_pooled_recovery_rises(self):
        no_replay = pd.DataFrame(
            columns=["recording", "stretch", "exposure_s", "detector", "n_events"]
        )
        trials = pd.DataFrame(
            {
                "condition": "chain_elsewhere",
                "level": 0.1,
                "detector": "sliding_published",
                "n_parts": 2,
                "n_recovered": [1, 2, 2],
                "n_recovered_alone": [2, 0, 2],
                "segmentation": "other",
            }
        )

        summary = sweep_summaries.controls_summary(trials, no_replay)
        rows = summary[summary.level.isna()].set_index("measure").value

        assert rows["n_trials"] == 3
        assert rows["n_trials_fewer_in_context"] == 1
        assert rows["n_trials_more_in_context"] == 1
