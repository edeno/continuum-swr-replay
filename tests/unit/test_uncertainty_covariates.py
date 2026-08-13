"""Tests for the model-free behavioral-uncertainty covariates.

Hand-computed truth on tiny synthetic per-session stop frames; checks the
omission streak, the no-leakage rolling switch rate, the win-stay/lose-shift
truth table, the leaky-Q pre-update margin recurrence, and that the helper does
not mutate its input.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import add_uncertainty_covariates

_SPEC = importlib.util.spec_from_file_location(
    "fit_uncertainty_rate",
    Path(__file__).resolve().parents[2] / "scripts" / "rate" / "fit_uncertainty_rate.py",
)
fur = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fur)


def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {"animal": "a", "day": 1, "epoch": "02_r1"}
    out = []
    for i, r in enumerate(rows):
        rec = {**base, "stop_idx": i}
        rec.update(r)
        out.append(rec)
    df = pd.DataFrame(out)
    # default the columns the helper reads if a test omits them
    for col in (
        "is_stay",
        "is_after_reward",
        "is_after_omission",
        "reward_outcome",
        "current_arm",
    ):
        if col not in df:
            df[col] = 0.0
    for j in range(1, 4):
        if f"reward_lag{j}" not in df:
            df[f"reward_lag{j}"] = np.nan
    # Real stop-table frames carry the origin arm and the outcome of the most
    # recently completed trip.  With one stop per visit, these are exactly the
    # lagged counterparts of ``is_stay`` used by the historical definition.
    if "prev_arm" not in df:
        df["prev_arm"] = np.arange(len(df), dtype=float)
    if "is_after_switch" not in df:
        df["is_after_switch"] = (1.0 - df["is_stay"]).shift(1)
    return df


def test_omission_streak():
    df = _frame(
        [
            {"reward_lag1": 0.0, "reward_lag2": 0.0, "reward_lag3": 1.0},  # 2
            {"reward_lag1": 1.0, "reward_lag2": 0.0, "reward_lag3": 0.0},  # 0
            {"reward_lag1": 0.0, "reward_lag2": np.nan, "reward_lag3": np.nan},  # 1
            {"reward_lag1": 0.0, "reward_lag2": 0.0, "reward_lag3": 0.0},  # 3
        ]
    )
    out = add_uncertainty_covariates(df)
    assert list(out["omission_streak"]) == [2, 0, 1, 3]


def test_recent_switch_rate_no_leakage():
    # With one stop per completed choice, retain the historical result exactly:
    # is_stay -> is_switch = 1 - is_stay, lagged by one visit.
    df = _frame(
        [
            {"is_stay": 1.0},  # switch 0
            {"is_stay": 0.0},  # switch 1
            {"is_stay": 0.0},  # switch 1
            {"is_stay": 1.0},  # switch 0
        ]
    )
    out = add_uncertainty_covariates(df, switch_window=5).sort_values("stop_idx")
    rates = out["recent_switch_rate"].to_numpy()
    assert np.isnan(rates[0])  # no prior stop
    assert rates[1] == pytest.approx(0.0)  # mean of [switch@0]=0
    assert rates[2] == pytest.approx(0.5)  # mean of [0,1]
    assert rates[3] == pytest.approx(2 / 3)  # mean of [0,1,1]


def test_recent_switch_rate_is_computed_once_per_completed_visit():
    """Repeated bouts cannot expose or over-count a visit's future choice."""
    df = _frame(
        [
            # Six rows shaped like real stop-table bouts: ``prev_arm`` and
            # ``is_after_switch`` stay fixed until another trip completes.
            # ``is_stay`` is the eventual *outbound* choice and is future
            # information throughout each two-bout visit.
            {"prev_arm": 3.0, "is_after_switch": 0.0, "is_stay": 0.0},
            {"prev_arm": 3.0, "is_after_switch": 0.0, "is_stay": 0.0},
            {"prev_arm": 2.0, "is_after_switch": 1.0, "is_stay": 1.0},
            {"prev_arm": 2.0, "is_after_switch": 1.0, "is_stay": 1.0},
            {"prev_arm": 0.0, "is_after_switch": 0.0, "is_stay": 0.0},
            {"prev_arm": 0.0, "is_after_switch": 0.0, "is_stay": 0.0},
        ]
    )

    out = add_uncertainty_covariates(df, switch_window=5)

    np.testing.assert_allclose(
        out["recent_switch_rate"].to_numpy(),
        np.array([np.nan, np.nan, 1.0, 1.0, 0.5, 0.5]),
        equal_nan=True,
    )


def test_recent_switch_rate_ignores_eventual_choice_within_visit():
    """Changing a not-yet-completed choice cannot change same-visit history."""
    df = _frame(
        [
            {"prev_arm": 3.0, "is_after_switch": 0.0, "is_stay": 0.0},
            {"prev_arm": 3.0, "is_after_switch": 0.0, "is_stay": 0.0},
            {"prev_arm": 2.0, "is_after_switch": 1.0, "is_stay": 1.0},
            {"prev_arm": 2.0, "is_after_switch": 1.0, "is_stay": 1.0},
        ]
    )
    changed_future = df.copy()
    changed_future.loc[changed_future["prev_arm"] == 3.0, "is_stay"] = 1.0

    original = add_uncertainty_covariates(df)["recent_switch_rate"]
    changed = add_uncertainty_covariates(changed_future)["recent_switch_rate"]

    np.testing.assert_allclose(original, changed, equal_nan=True)


def test_wsls_deviation_truth_table():
    df = _frame(
        [
            {"is_after_reward": 1.0, "is_after_omission": 0.0, "is_stay": 1.0},  # win-stay -> 0
            {"is_after_reward": 0.0, "is_after_omission": 1.0, "is_stay": 0.0},  # lose-shift -> 0
            {"is_after_reward": 1.0, "is_after_omission": 0.0, "is_stay": 0.0},  # win-shift -> 1
            {"is_after_reward": 0.0, "is_after_omission": 1.0, "is_stay": 1.0},  # lose-stay -> 1
        ]
    )
    out = add_uncertainty_covariates(df)
    assert list(out["wsls_deviation"]) == [0.0, 0.0, 1.0, 1.0]


def test_wsls_deviation_no_prior_is_nan():
    # Neither is_after_reward nor is_after_omission set (e.g. first stop of a
    # session) -> NaN, not 1.0, so the row drops at fit time.
    df = _frame(
        [
            {"is_after_reward": 0.0, "is_after_omission": 0.0, "is_stay": 1.0},
            {"is_after_reward": 1.0, "is_after_omission": 0.0, "is_stay": 1.0},  # win-stay -> 0
        ]
    )
    out = add_uncertainty_covariates(df).sort_values("stop_idx")
    vals = out["wsls_deviation"].to_numpy()
    assert np.isnan(vals[0])
    assert vals[1] == 0.0


def test_leaky_q_value_diff_nan_arm():
    # A NaN current_arm yields a real margin (the current Q-spread) and does NOT
    # update any arm's Q, so the margin on the stop after it is unchanged.
    df = _frame(
        [
            {"current_arm": 0.0, "reward_outcome": 1.0},  # Q0 -> 0.75
            {"current_arm": np.nan, "reward_outcome": 1.0},  # margin .25; no update
            {"current_arm": 1.0, "reward_outcome": 0.0},  # margin still .25
        ]
    )
    out = add_uncertainty_covariates(df, q_alpha=0.5).sort_values("stop_idx")
    margins = out["leaky_q_value_diff"].to_numpy()
    assert margins[0] == pytest.approx(0.0)
    assert np.isfinite(margins[1])  # NaN arm -> real margin, not NaN
    assert margins[1] == pytest.approx(0.25)
    # Q unchanged across the NaN-arm stop, so stop 2 sees the same .25 margin.
    assert margins[2] == pytest.approx(0.25)


def test_leaky_q_value_diff_recurrence():
    # Two arms only (0 and 1); all start at 0.5. q_alpha = 0.5.
    # Pre-update margin = sort(Q)[-1] - sort(Q)[-2].
    df = _frame(
        [
            {"current_arm": 0.0, "reward_outcome": 1.0},
            {"current_arm": 0.0, "reward_outcome": 1.0},
            {"current_arm": 1.0, "reward_outcome": 0.0},
        ]
    )
    out = add_uncertainty_covariates(df, q_alpha=0.5).sort_values("stop_idx")
    margins = out["leaky_q_value_diff"].to_numpy()
    # stop0: all Q=0.5 -> margin 0; then Q0 = .5*.5 + .5*1 = .75
    # stop1: Q=[.75,.5,.5,...] -> margin .75-.5 = .25; then Q0 = .5*.75+.5 = .875
    # stop2: Q=[.875,.5,.5,...] -> margin .875-.5 = .375
    assert margins[0] == pytest.approx(0.0)
    assert margins[1] == pytest.approx(0.25)
    assert margins[2] == pytest.approx(0.375)


def test_no_mutation():
    df = _frame(
        [
            {"current_arm": 0.0, "reward_outcome": 1.0, "is_stay": 1.0},
            {"current_arm": 1.0, "reward_outcome": 0.0, "is_stay": 0.0},
        ]
    )
    before = df.copy(deep=True)
    out = add_uncertainty_covariates(df)
    pd.testing.assert_frame_equal(df, before)  # input unchanged
    assert out.index.equals(df.index)
    assert len(out) == len(df)
    for col in (
        "omission_streak",
        "recent_switch_rate",
        "wsls_deviation",
        "leaky_q_value_diff",
    ):
        assert col in out.columns


def test_drop_first_session_per_animal():
    st = pd.DataFrame(
        {
            "animal": ["a", "a", "b"],
            "day": [1, 2, 1],
            "epoch": ["02_r1", "02_r1", "02_r1"],
            "stop_id": [0, 1, 2],
        }
    )
    out = fur._drop_first_session_per_animal(st)
    # a's first session (day 1) and b's only session (day 1) dropped.
    assert set(out["stop_id"]) == {1}


def test_post_outcome_bin_gate_uses_physical_left_edge_and_keeps_equality():
    design = pd.DataFrame(
        {
            "stop_id": [1, 1, 1],
            "stop_start": [10.0, 10.0, 10.0],
            "binj": [0, 1, 2],
            "outcome_known_time": [10.0 + fur.BIN_S] * 3,
        }
    )

    gated, audit = fur._gate_bins_after_outcome(design)

    assert gated["binj"].tolist() == [1, 2]
    assert gated["response_time"].tolist() == pytest.approx(
        [10.0 + fur.BIN_S, 10.0 + 2 * fur.BIN_S]
    )
    assert audit == {
        "n_bins_before_post_outcome_gate": 3,
        "n_bins_after_post_outcome_gate": 2,
        "n_bins_excluded_missing_timing": 0,
        "n_bins_excluded_pre_outcome": 1,
    }


def test_post_outcome_bin_gate_fails_loud_without_physical_timing():
    design = pd.DataFrame({"stop_id": [1], "binj": [0]})
    with pytest.raises(KeyError, match="outcome_known_time"):
        fur._gate_bins_after_outcome(design)


def test_post_outcome_bin_gate_rejects_inconsistent_stop_boundary():
    design = pd.DataFrame(
        {
            "stop_id": [1, 1],
            "stop_start": [10.0, 10.0],
            "binj": [0, 1],
            "outcome_known_time": [10.0, 10.1],
        }
    )
    with pytest.raises(ValueError, match="constant within stop_id"):
        fur._gate_bins_after_outcome(design)


@pytest.mark.slow
def test_runner_smoke():
    """End-to-end runner on one post-exposure session per animal.

    Requires ``qvals_table.pkl`` alongside the canonical tables. Writes
    ``uncertainty_rate_smoke.pkl`` and checks the tidy schema carries every
    uncertainty/value coefficient across the three populations.
    """
    from continuum_swr_replay.analysis.replay import (
        N_REWARD_LAGS,
        attach_is_swr_to_onsets,
    )
    from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

    rc = PROCESSED_DATA_DIR / "replay_content"
    qvals_path = rc / "models_per_bin_rate" / "qvals_table.pkl"
    if not qvals_path.exists():
        pytest.skip(
            "qvals_table.pkl not present "
            "(run continuum_swr_replay.analysis.behavior.qvals_table --write first)"
        )

    stop = pd.read_pickle(rc / "stop_table.pkl")
    onsets = attach_is_swr_to_onsets(
        pd.read_pickle(rc / "event_onset_table.pkl"),
        pd.read_pickle(rc / "event_table.pkl"),
    )

    stop = add_uncertainty_covariates(stop)
    # Nominal schedule + real model covariates, all from qvals_table -- one
    # exactly poke_out-aligned source, mirroring the runner's main().
    qvals = pd.read_pickle(qvals_path)
    if "outcome_known_time" not in qvals:
        pytest.skip(
            "qvals_table.pkl predates physical post-outcome timing; regenerate it"
        )
    qcols = qvals[
        ["stop_id", "outcome_known_time", *fur.VALUE_COVARIATES, "rpe", *fur.RL_COVARIATES]
    ].rename(
        columns={"rpe": "rpe_model"}
    )
    stop = stop.merge(
        qcols, on="stop_id", how="inner", validate="many_to_one"
    )
    stop = fur._drop_first_session_per_animal(stop)
    stop = fur._subset_max_per_animal(stop, 1)

    models = fur.fit_uncertainty_rate(stop, onsets)
    assert len(models) > 0
    assert set(models["outcome"]).issubset(set(fur.POPULATIONS_TO_FIT))
    # every curated uncertainty/value covariate the model fits has a
    # coefficient/RR column (CURATED_COVARIATES is the set actually fit; it
    # supersets VALUE_COVARIATES).
    for cov in fur.CURATED_COVARIATES:
        assert f"coef_{cov}" in models.columns
        assert f"rr_{cov}" in models.columns
    # reward kernel is also present (the controlled-for block).
    assert f"coef_reward_lag{N_REWARD_LAGS}" in models.columns
