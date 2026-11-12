"""The decoder-validation statistics over sessions, on tables checked by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.quality_metrics import (
    STATE_PROBABILITY_COLUMNS,
    calibration_curve_statistics,
    local_decoding_error_statistics,
    session_decoder_quality,
    session_state_fractions,
    state_fraction_statistics,
)


def test_decoding_error_median_is_over_sessions_with_a_finite_error():
    # Four finite sessions: an even count, so the median is the mean of the
    # middle two (2 and 4), not the mean of all four (5.25). The NaN session
    # is not counted.
    error_df = pd.DataFrame({"median_error_cm": [1.0, 4.0, np.nan, 2.0, 14.0]})

    statistics = local_decoding_error_statistics(error_df)

    assert statistics == {"n_sessions": 4, "median_error_cm": 3.0}


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
    """One session's bins: Local or Non-Local Continuous, all analysed."""
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


def test_a_session_with_no_moving_bin_has_no_moving_fraction():
    frame = pd.concat(
        [
            _session("a", [10, 1], [1, 0]),
            # every bin immobile: no moving fraction, which the median skips
            _session("b", [1, 1], [1, 1]),
            _session("c", [10, 10, 1], [0, 0, 1]),
        ]
    )

    fractions = session_state_fractions(frame).set_index("animal")
    statistics = state_fraction_statistics(fractions)

    assert np.isnan(fractions.loc["b", "moving_Local_prob"])
    # moving: a 1 and c 0 -> 0.5; immobile: a 0, b 1, c 1 -> 1
    np.testing.assert_allclose(
        statistics["Local_prob"].to_numpy(), [0.5, 1.0], err_msg="moving, immobile"
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


def test_state_fractions_are_session_means_then_the_median_over_sessions():
    fractions = state_fraction_statistics(session_state_fractions(_three_sessions()))

    assert fractions.index.tolist() == ["moving", "immobile"]
    assert fractions.columns.tolist() == list(STATE_PROBABILITY_COLUMNS)
    # Moving: sessions 1, 0, 0.5 -> median 0.5.
    np.testing.assert_allclose(
        fractions.loc["moving", ["Local_prob", "Non-Local_Continuous_prob"]],
        [0.5, 0.5],
    )
    # Immobile: sessions a 2/3, b 1/2 and c 0 -> median 1/2 (the mean is
    # 7/18; with b's threshold bin counted the median would be 1/3).
    np.testing.assert_allclose(
        fractions.loc["immobile", ["Local_prob", "Non-Local_Continuous_prob"]],
        [0.5, 0.5],
    )
    np.testing.assert_allclose(
        fractions.loc[:, ["No-Spike_prob", "Non-Local_Fragmented_prob"]], 0.0
    )
