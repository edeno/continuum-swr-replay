"""The decoder-validation statistics over sessions, on tables checked by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.decode_quality.quality_metrics import (
    STATE_PROBABILITY_COLUMNS,
    calibration_curve_statistics,
    local_decoding_error_statistics,
    session_decoder_quality,
    session_state_fractions,
    state_fraction_statistics,
)


def test_decoding_error_is_each_animals_median_session_then_the_mean():
    # a: sessions 1, 4 and 2 (median 2) and one with no finite error, which is
    # not one of its sessions; b: one session, 14. The mean over animals is 8
    # (the median over the four pooled sessions would be 3).
    error_df = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "a", "b"],
            "day": [1, 1, 2, 2, 1],
            "epoch": ["02_r1", "04_r2", "02_r1", "04_r2", "02_r1"],
            "median_error_cm": [1.0, 4.0, np.nan, 2.0, 14.0],
        }
    )

    statistics = local_decoding_error_statistics(error_df)

    assert statistics == {"n_sessions": 4, "median_error_cm": 8.0}


def test_calibration_curve_is_median_and_quartiles_over_sessions_per_bin():
    # Bin 0.15: three sessions 1, 2, 9 (median 2, not the mean 4). Bin 0.95:
    # four sessions 0, 0, 2, 4 (median 1; linear quartiles 0 and 2.5).
    calibration = pd.DataFrame(
        {
            "prob_bin_center": [0.95, 0.15, 0.95, 0.15, 0.95, 0.15, 0.95],
            "median_error_cm": [0.0, 1.0, 4.0, 9.0, 2.0, 2.0, 0.0],
        }
    )

    curve = calibration_curve_statistics(calibration)

    assert curve.index.tolist() == [0.15, 0.95]
    np.testing.assert_allclose(curve["median"], [2.0, 1.0])
    np.testing.assert_allclose(curve["q25"], [1.5, 0.0])
    np.testing.assert_allclose(curve["q75"], [5.5, 2.5])


def _session(animal: str, speed: list[float], local: list[float]) -> pd.DataFrame:
    """One session's bins: Local or Non-Local Continuous, all analyzed."""
    local = np.asarray(local)
    return pd.DataFrame(
        {
            "animal": animal,
            "day": 1,
            "epoch": "02_r1",
            "head_speed": speed,
            "Local_prob": local,
            "No-Spike_prob": 0.0,
            "Non-Local_Continuous_prob": 1.0 - local,
            "Non-Local_Fragmented_prob": 0.0,
            "in_excluded_time": False,
        },
        index=np.arange(len(local)) / 500.0,
    )


def _three_sessions() -> pd.DataFrame:
    frame = pd.concat(
        [
            # a: moving local 1, 1; immobile local 1, 1, 0 (mean 2/3).
            _session("a", [10, 10, 1, 1, 1], [1, 1, 1, 1, 0]),
            # b: moving local 0; immobile local 1, 0 (mean 1/2). A bin at the
            # threshold (4 cm/s) is in neither bar, nor is a bin without a
            # speed: counted as immobile, either would lower b's mean.
            _session("b", [10, 2, 2, 4, np.nan], [0, 1, 0, 0, 0]),
            # c: moving local 1, 0 (mean 0.5); immobile local 0, and one more
            # immobile bin (local 1) that lies in excluded time.
            _session("c", [10, 10, 1, 1], [1, 0, 0, 1]),
        ]
    )
    excluded = (
        (frame["animal"] == "c")
        & (frame["Local_prob"] == 1)
        & (frame["head_speed"] == 1)
    )
    frame.loc[excluded, "in_excluded_time"] = True
    return frame


def test_session_state_fractions_are_each_session_s_mean_probabilities():
    fractions = session_state_fractions(_three_sessions())

    assert fractions["animal"].tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(fractions["moving_Local_prob"], [1.0, 0.0, 0.5])
    np.testing.assert_allclose(fractions["immobile_Local_prob"], [2 / 3, 0.5, 0.0])
    np.testing.assert_allclose(
        fractions["immobile_Non-Local_Continuous_prob"], [1 / 3, 0.5, 1.0]
    )


def test_the_immobile_fractions_take_only_the_bins_inside_a_well_visit():
    """Given the well-visit mask, the immobile bars count only the immobile
    bins inside a visit; the moving bars do not change."""
    frame = _three_sessions()
    # a's last immobile bin (local 0) and b's first immobile bin (local 1)
    # are outside every visit
    outside = np.zeros(len(frame), dtype=bool)
    outside[4] = True
    outside[6] = True
    fractions = session_state_fractions(frame, in_visit=~outside)
    every = session_state_fractions(frame)
    np.testing.assert_allclose(fractions["immobile_Local_prob"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        fractions["moving_Local_prob"], every["moving_Local_prob"]
    )


def test_a_session_with_no_moving_bin_has_no_moving_fraction():
    frame = pd.concat(
        [
            _session("a", [10, 1], [1, 0]),
            # every bin immobile: no moving fraction, which the median skips
            _session("b", [1, 1], [1, 1]),
            _session("c", [10, 10, 1], [0, 0, 1]),
        ]
    )

    fractions = session_state_fractions(frame)
    statistics = state_fraction_statistics(fractions)

    assert np.isnan(fractions.set_index("animal").loc["b", "moving_Local_prob"])
    # moving: a 1 and c 0 -> 0.5 (b is left out); immobile: a 0, b 1, c 1 -> 2/3
    np.testing.assert_allclose(
        statistics["Local_prob"].to_numpy(), [0.5, 2 / 3], err_msg="moving, immobile"
    )


def test_session_decoder_quality_joins_the_error_and_the_fractions_per_session():
    frame = _three_sessions()
    frame["ahead_behind_distance"] = np.where(frame["animal"] == "b", -6.0, 2.0)

    quality = session_decoder_quality(frame)

    assert len(quality) == 3
    # b's one bin with Local above 0.5 is 6 cm off; a's and c's Local bins
    # outside excluded time are 2 cm off
    np.testing.assert_allclose(quality["median_error_cm"], [2.0, 6.0, 2.0])
    np.testing.assert_allclose(quality["moving_Local_prob"], [1.0, 0.0, 0.5])


def test_state_fractions_are_session_means_then_the_mean_over_animals():
    fractions = state_fraction_statistics(session_state_fractions(_three_sessions()))

    assert fractions.index.tolist() == ["moving", "immobile"]
    assert fractions.columns.tolist() == list(STATE_PROBABILITY_COLUMNS)
    # One session per animal. Moving: 1, 0, 0.5 -> 0.5.
    np.testing.assert_allclose(
        fractions.loc["moving", ["Local_prob", "Non-Local_Continuous_prob"]],
        [0.5, 0.5],
    )
    # Immobile: a 2/3, b 1/2 and c 0 -> 7/18 (the median over the sessions
    # would be 1/2).
    np.testing.assert_allclose(
        fractions.loc["immobile", ["Local_prob", "Non-Local_Continuous_prob"]],
        [7 / 18, 11 / 18],
    )
    np.testing.assert_allclose(
        fractions.loc[:, ["No-Spike_prob", "Non-Local_Fragmented_prob"]], 0.0
    )


def test_state_fractions_weigh_an_animals_sessions_then_the_animals():
    """a: three sessions with moving Local 1, 1 and 0.4 (mean 0.8) and one
    with no moving bin, which is not one of its sessions; b: one session, 0.2.
    The mean over animals is 0.5; the median over the four pooled sessions
    would be 0.7."""
    rows = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "a", "b"],
            "day": [1, 1, 2, 2, 1],
            "epoch": ["02_r1", "04_r2", "02_r1", "04_r2", "02_r1"],
        }
    )
    for movement in ("moving", "immobile"):
        for column in STATE_PROBABILITY_COLUMNS:
            rows[f"{movement}_{column}"] = 0.0
    rows["moving_Local_prob"] = [1.0, 1.0, 0.4, np.nan, 0.2]

    statistics = state_fraction_statistics(rows)

    assert statistics.loc["moving", "Local_prob"] == pytest.approx(0.5)
