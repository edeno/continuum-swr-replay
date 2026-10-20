"""Tests for the model-free behavioral-uncertainty covariates.

Hand-computed truth on tiny synthetic per-session stop frames; checks the
omission streak, the no-leakage rolling switch rate, the win-stay/lose-shift
truth table, the leaky-Q pre-update margin recurrence, and that the helper does
not mutate its input.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import add_uncertainty_covariates
from continuum_swr_replay.analysis.replay.covariates import (
    CANONICAL_SWITCH_WINDOW,
    canonical_port_value_history,
    canonical_trial_switch_history,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MU_UNCERTAINTY_CURATED_COVARIATES,
)
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, table_path
from continuum_swr_replay.tables.io import read_table, table_exists
from continuum_swr_replay.tables.session_keys import subset_max_per_animal

_SPEC = importlib.util.spec_from_file_location(
    "fit_uncertainty_rate",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "rate"
    / "fit_uncertainty_rate.py",
)
fur = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fur)


def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {"animal": "a", "session_date": 1, "epoch": "02_r1"}
    out = []
    for i, r in enumerate(rows):
        rec = {**base, "stop_idx": i}
        rec.update(r)
        out.append(rec)
    df = pd.DataFrame(out)
    # default the columns the helper reads if a test omits them
    for col in (
        "is_stay",
        "is_rewarded_outcome",
        "completed_trial_reward",
        "occupancy_well_arm",
    ):
        if col not in df:
            df[col] = 0.0
    if "completed_trial_id" not in df:
        df["completed_trial_id"] = pd.array(range(len(df)), dtype="Int64")
    else:
        df["completed_trial_id"] = pd.array(df["completed_trial_id"], dtype="Int64")
    if "completed_trial_arm" not in df:
        df["completed_trial_arm"] = df["occupancy_well_arm"]
    has_completed = df["completed_trial_id"].notna()
    if "patch_switch_fraction_last5_completed_trips" not in df:
        df["patch_switch_fraction_last5_completed_trips"] = np.where(
            has_completed, 0.0, np.nan
        )
    if "leaky_q_value_diff" not in df:
        df["leaky_q_value_diff"] = np.where(has_completed, 0.0, np.nan)
    for j in range(1, 11):
        if f"known_reward_outcome_lag_{j}" not in df:
            df[f"known_reward_outcome_lag_{j}"] = np.nan
    # Real stop-table frames carry an explicit regenerated completed-trial ID.
    # Most tests do not exercise switch history, so use a valid neutral default.
    if "prev_arm" not in df:
        df["prev_arm"] = np.arange(len(df), dtype=float)
    if "did_last_completed_trip_switch_patch" not in df:
        df["did_last_completed_trip_switch_patch"] = 0.0
    return df


def test_omission_streak():
    df = _frame(
        [
            {
                "known_reward_outcome_lag_1": 0.0,
                "known_reward_outcome_lag_2": 0.0,
                "known_reward_outcome_lag_3": 1.0,
            },  # 2
            {
                "known_reward_outcome_lag_1": 1.0,
                "known_reward_outcome_lag_2": 0.0,
                "known_reward_outcome_lag_3": 0.0,
            },  # 0
            {
                "known_reward_outcome_lag_1": 0.0,
                "known_reward_outcome_lag_2": np.nan,
                "known_reward_outcome_lag_3": np.nan,
            },  # 1
            {
                "known_reward_outcome_lag_1": 0.0,
                "known_reward_outcome_lag_2": 0.0,
                "known_reward_outcome_lag_3": 0.0,
            },  # 3
        ]
    )
    out = add_uncertainty_covariates(df)
    assert list(out["omission_streak"]) == [2, 0, 1, 3]


def test_omission_streak_requires_exact_contiguous_binary_lag_schema():
    missing = _frame([{}]).drop(columns=["known_reward_outcome_lag_4"])
    with pytest.raises(KeyError, match="exact contiguous"):
        add_uncertainty_covariates(missing)

    unexpected = _frame([{}]).assign(known_reward_outcome_lag_11=0.0)
    with pytest.raises(KeyError, match="unexpected"):
        add_uncertainty_covariates(unexpected)

    nonbinary = _frame([{"known_reward_outcome_lag_4": 2.0}])
    with pytest.raises(ValueError, match="binary"):
        add_uncertainty_covariates(nonbinary)


def test_omission_streak_schema_does_not_depend_on_dataframe_column_order():
    frame = _frame(
        [{"known_reward_outcome_lag_1": 0.0, "known_reward_outcome_lag_2": 1.0}]
    )
    reordered = frame[list(reversed(frame.columns))]

    out = add_uncertainty_covariates(reordered)

    assert out["omission_streak"].iloc[0] == 1.0


def test_recent_switch_rate_no_leakage():
    # The completed trip's switch status is already past context at each stop.
    df = _frame(
        [
            {
                "did_last_completed_trip_switch_patch": 0.0,
                "patch_switch_fraction_last5_completed_trips": 0.0,
                "is_stay": 1.0,
            },
            {
                "did_last_completed_trip_switch_patch": 1.0,
                "patch_switch_fraction_last5_completed_trips": 0.5,
                "is_stay": 0.0,
            },
            {
                "did_last_completed_trip_switch_patch": 1.0,
                "patch_switch_fraction_last5_completed_trips": 2 / 3,
                "is_stay": 0.0,
            },
            {
                "did_last_completed_trip_switch_patch": 0.0,
                "patch_switch_fraction_last5_completed_trips": 0.5,
                "is_stay": 1.0,
            },
        ]
    )
    out = add_uncertainty_covariates(df, switch_window=5).sort_values("stop_idx")
    rates = out["patch_switch_fraction_last5_completed_trips"].to_numpy()
    assert rates[0] == pytest.approx(0.0)
    assert rates[1] == pytest.approx(0.5)
    assert rates[2] == pytest.approx(2 / 3)
    assert rates[3] == pytest.approx(0.5)


def test_recent_switch_rate_is_computed_once_per_completed_visit():
    """Repeated bouts cannot expose or over-count a visit's future choice."""
    df = _frame(
        [
            # Six rows shaped like real stop-table bouts: ``prev_arm`` and
            # ``is_after_switch`` stay fixed until another trip completes.
            # ``is_stay`` is the eventual *outbound* choice and is future
            # information throughout each two-bout visit.
            {
                "completed_trial_id": 0,
                "prev_arm": 3.0,
                "did_last_completed_trip_switch_patch": 0.0,
                "patch_switch_fraction_last5_completed_trips": 0.0,
                "is_stay": 0.0,
            },
            {
                "completed_trial_id": 0,
                "prev_arm": 3.0,
                "did_last_completed_trip_switch_patch": 0.0,
                "patch_switch_fraction_last5_completed_trips": 0.0,
                "is_stay": 0.0,
            },
            {
                "completed_trial_id": 1,
                "prev_arm": 2.0,
                "did_last_completed_trip_switch_patch": 1.0,
                "patch_switch_fraction_last5_completed_trips": 0.5,
                "is_stay": 1.0,
            },
            {
                "completed_trial_id": 1,
                "prev_arm": 2.0,
                "did_last_completed_trip_switch_patch": 1.0,
                "patch_switch_fraction_last5_completed_trips": 0.5,
                "is_stay": 1.0,
            },
            {
                "completed_trial_id": 2,
                "prev_arm": 0.0,
                "did_last_completed_trip_switch_patch": 0.0,
                "patch_switch_fraction_last5_completed_trips": 1 / 3,
                "is_stay": 0.0,
            },
            {
                "completed_trial_id": 2,
                "prev_arm": 0.0,
                "did_last_completed_trip_switch_patch": 0.0,
                "patch_switch_fraction_last5_completed_trips": 1 / 3,
                "is_stay": 0.0,
            },
        ]
    )

    out = add_uncertainty_covariates(df, switch_window=5)

    np.testing.assert_allclose(
        out["patch_switch_fraction_last5_completed_trips"].to_numpy(),
        np.array([0.0, 0.0, 0.5, 0.5, 1 / 3, 1 / 3]),
        equal_nan=True,
    )


def test_recent_switch_rate_ignores_eventual_choice_within_visit():
    """Changing a not-yet-completed choice cannot change same-visit history."""
    df = _frame(
        [
            {
                "completed_trial_id": 0,
                "prev_arm": 3.0,
                "did_last_completed_trip_switch_patch": 0.0,
                "is_stay": 0.0,
            },
            {
                "completed_trial_id": 0,
                "prev_arm": 3.0,
                "did_last_completed_trip_switch_patch": 0.0,
                "is_stay": 0.0,
            },
            {
                "completed_trial_id": 1,
                "prev_arm": 2.0,
                "did_last_completed_trip_switch_patch": 1.0,
                "is_stay": 1.0,
            },
            {
                "completed_trial_id": 1,
                "prev_arm": 2.0,
                "did_last_completed_trip_switch_patch": 1.0,
                "is_stay": 1.0,
            },
        ]
    )
    changed_future = df.copy()
    changed_future.loc[changed_future["prev_arm"] == 3.0, "is_stay"] = 1.0

    original = add_uncertainty_covariates(df)[
        "patch_switch_fraction_last5_completed_trips"
    ]
    changed = add_uncertainty_covariates(changed_future)[
        "patch_switch_fraction_last5_completed_trips"
    ]

    np.testing.assert_allclose(original, changed, equal_nan=True)


def test_recent_switch_rate_leaves_only_true_carry_in_rows_unknown():
    df = _frame(
        [
            {
                "completed_trial_id": pd.NA,
                "completed_trial_arm": np.nan,
                "is_rewarded_outcome": np.nan,
                "completed_trial_reward": np.nan,
                "did_last_completed_trip_switch_patch": np.nan,
            },
            {
                "completed_trial_id": 0,
                "completed_trial_arm": 2.0,
                "is_rewarded_outcome": 1.0,
                "completed_trial_reward": 1.0,
                "did_last_completed_trip_switch_patch": 1.0,
                "patch_switch_fraction_last5_completed_trips": 1.0,
            },
            {
                "completed_trial_id": 0,
                "completed_trial_arm": 2.0,
                "is_rewarded_outcome": 1.0,
                "completed_trial_reward": 1.0,
                "did_last_completed_trip_switch_patch": 1.0,
                "patch_switch_fraction_last5_completed_trips": 1.0,
            },
        ]
    )

    rates = add_uncertainty_covariates(df)[
        "patch_switch_fraction_last5_completed_trips"
    ].to_numpy()

    np.testing.assert_allclose(rates, [np.nan, 1.0, 1.0], equal_nan=True)


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("completed_trial_arm", 2.0),
        ("is_rewarded_outcome", 0.0),
        ("did_last_completed_trip_switch_patch", 1.0),
        ("patch_switch_fraction_last5_completed_trips", 1.0),
        ("leaky_q_value_diff", 1.0),
    ],
)
def test_completed_trial_context_must_be_constant_within_id(column, replacement):
    rows = [
        {
            "completed_trial_id": 0,
            "completed_trial_arm": 1.0,
            "is_rewarded_outcome": 1.0,
            "did_last_completed_trip_switch_patch": 0.0,
        },
        {
            "completed_trial_id": 0,
            "completed_trial_arm": 1.0,
            "is_rewarded_outcome": 1.0,
            "did_last_completed_trip_switch_patch": 0.0,
        },
    ]
    rows[1][column] = replacement

    with pytest.raises(ValueError, match=column):
        add_uncertainty_covariates(_frame(rows))


def test_completed_trial_outcome_can_transition_once_from_unknown_to_known():
    frame = _frame(
        [
            {"completed_trial_id": 0, "is_rewarded_outcome": np.nan},
            {"completed_trial_id": 0, "is_rewarded_outcome": 1.0},
        ]
    )

    out = add_uncertainty_covariates(frame)

    assert len(out) == 2


def test_completed_trial_outcome_availability_cannot_regress():
    frame = _frame(
        [
            {"completed_trial_id": 0, "is_rewarded_outcome": 1.0},
            {"completed_trial_id": 0, "is_rewarded_outcome": np.nan},
        ]
    )
    with pytest.raises(ValueError, match="is_rewarded_outcome"):
        add_uncertainty_covariates(frame)


def test_completed_trial_id_gaps_are_valid_but_order_must_be_nondecreasing():
    gap = _frame(
        [
            {
                "completed_trial_id": 0,
                "patch_switch_fraction_last5_completed_trips": 0.0,
            },
            {
                "completed_trial_id": 2,
                "patch_switch_fraction_last5_completed_trips": 2 / 3,
            },
        ]
    )
    out = add_uncertainty_covariates(gap)
    np.testing.assert_allclose(
        out["patch_switch_fraction_last5_completed_trips"], [0.0, 2 / 3]
    )

    reversed_ids = _frame(
        [
            {"completed_trial_id": 1},
            {"completed_trial_id": 0},
        ]
    )
    with pytest.raises(ValueError, match="nondecreasing"):
        add_uncertainty_covariates(reversed_ids)


def test_carry_in_rows_must_precede_observed_completed_trial_ids():
    df = _frame(
        [
            {"completed_trial_id": 0},
            {"completed_trial_id": pd.NA},
        ]
    )
    with pytest.raises(ValueError, match="carry-in"):
        add_uncertainty_covariates(df)


def test_completed_trial_ids_must_be_nonnegative():
    with pytest.raises(ValueError, match="nonnegative"):
        add_uncertainty_covariates(_frame([{"completed_trial_id": -1}]))


@pytest.mark.parametrize(
    "column", ["patch_switch_fraction_last5_completed_trips", "leaky_q_value_diff"]
)
@pytest.mark.parametrize("bad_value", [-0.1, 1.1, np.nan, np.inf])
def test_canonical_history_covariates_must_be_finite_probabilities(column, bad_value):
    df = _frame([{column: bad_value}])
    with pytest.raises(ValueError, match=column):
        add_uncertainty_covariates(df)


def test_wsls_deviation_truth_table():
    df = _frame(
        [
            {"completed_trial_reward": 1.0, "is_stay": 1.0},  # win-stay   -> 0
            {"completed_trial_reward": 0.0, "is_stay": 0.0},  # lose-shift -> 0
            {"completed_trial_reward": 1.0, "is_stay": 0.0},  # win-shift  -> 1
            {"completed_trial_reward": 0.0, "is_stay": 1.0},  # lose-stay  -> 1
        ]
    )
    out = add_uncertainty_covariates(df)
    assert list(out["wsls_deviation"]) == [0.0, 0.0, 1.0, 1.0]


def test_wsls_deviation_survives_the_bout_start_outcome_gate():
    """The covariate must be defined where win-stay/lose-shift means something.

    An at-well bout begins at arrival, before the trip's outcome exists, so
    the gated ``is_rewarded_outcome`` is missing on every at-well stop.
    Reading the realized outcome instead keeps the covariate
    defined; reading the gated flags would make it NaN throughout.
    """
    df = _frame(
        [
            {"completed_trial_reward": 1.0, "is_stay": 1.0},
            {"completed_trial_reward": 0.0, "is_stay": 1.0},
        ]
    )
    # Exactly the state of a regenerated at-well stop: outcome not yet known
    # at the frozen context time.
    df["is_rewarded_outcome"] = np.nan

    out = add_uncertainty_covariates(df)

    assert out["wsls_deviation"].notna().all()
    assert list(out["wsls_deviation"]) == [0.0, 1.0]


def test_wsls_deviation_no_prior_is_nan():
    # No completed trip yet (e.g. first stop of a session) -> NaN, not 1.0,
    # so the row drops at fit time instead of counting as a deviation.
    df = _frame(
        [
            {
                "completed_trial_id": pd.NA,
                "completed_trial_arm": np.nan,
                "completed_trial_reward": np.nan,
                "is_rewarded_outcome": np.nan,
                "did_last_completed_trip_switch_patch": np.nan,
                "is_stay": 1.0,
            },
            {
                "completed_trial_id": 0,
                "completed_trial_arm": 2.0,
                "is_rewarded_outcome": 1.0,
                "completed_trial_reward": 1.0,
                "did_last_completed_trip_switch_patch": 0.0,
                "is_stay": 1.0,
            },  # win-stay -> 0
        ]
    )
    out = add_uncertainty_covariates(df).sort_values("stop_idx")
    vals = out["wsls_deviation"].to_numpy()
    assert np.isnan(vals[0])
    assert vals[1] == 0.0


def test_missing_final_completed_reward_is_rejected():
    df = _frame(
        [
            {
                "completed_trial_id": 0,
                "completed_trial_arm": 1.0,
                "completed_trial_reward": 1.0,
                "is_stay": 1.0,
            },
            {
                "completed_trial_id": 1,
                "completed_trial_arm": 2.0,
                "completed_trial_reward": np.nan,
                "is_rewarded_outcome": np.nan,
                "is_stay": 0.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="completed_trial_reward must be finite"):
        add_uncertainty_covariates(df)


@pytest.mark.parametrize(
    "column,bad",
    [
        ("completed_trial_reward", 2.0),
        ("completed_trial_reward", -1.0),
        ("is_stay", 2.0),
        ("is_stay", 0.5),
    ],
)
def test_wsls_consumed_columns_must_be_binary(column, bad):
    df = _frame([{column: bad}])
    with pytest.raises(ValueError, match=column):
        add_uncertainty_covariates(df)


def test_add_uncertainty_preserves_canonical_leaky_q_margin():
    df = _frame(
        [
            {
                "completed_trial_id": 0,
                "completed_trial_arm": 0.0,
                "is_rewarded_outcome": 1.0,
                "leaky_q_value_diff": 0.0,
            },
            {
                "completed_trial_id": 1,
                "completed_trial_arm": 0.0,
                "is_rewarded_outcome": 1.0,
                "leaky_q_value_diff": 0.15,
            },
            {
                "completed_trial_id": 2,
                "completed_trial_arm": 1.0,
                "is_rewarded_outcome": 0.0,
                "leaky_q_value_diff": 0.255,
            },
        ]
    )
    out = add_uncertainty_covariates(df).sort_values("stop_idx")
    np.testing.assert_allclose(out["leaky_q_value_diff"], [0.0, 0.15, 0.255])


def test_leaky_q_value_diff_recurrence():
    experiences = pd.DataFrame(
        {
            "experience_id": [0, 1, 2],
            "movement_trial_number": pd.array([pd.NA, 1, 2], dtype="Int64"),
            "is_initial_experience": [True, False, False],
            "is_boundary_derived": [False, False, False],
            "outcome_observed": [True, True, True],
            "port": [0, 0, 1],
            "patch": [1, 1, 1],
            "arrival_time": [0.0, 2.0, 4.0],
            "departure_time": [1.5, 3.5, 5.5],
            "reward_on_time": [1.0, 3.0, np.nan],
            "outcome_known_time": [1.0, 3.0, 5.0],
            "is_reward": [1, 1, 0],
        }
    )
    # alpha=.3: Q0 .5→.65→.755, with every margin evaluated pre-update.
    np.testing.assert_allclose(
        canonical_port_value_history(experiences), [0.0, 0.15, 0.255]
    )


def test_switch_rate_window_truncates_at_the_canonical_length():
    """Pin ``CANONICAL_SWITCH_WINDOW`` itself, not just the rolling mean.

    With ``min_periods=1`` a three-trial sequence gives the same answer
    for every window >= 3, so it cannot distinguish 5 from 4 or 10.
    Seven switch outcomes can: the trailing mean is 3/5 only at width 5
    (4/6 at 6, 2/4 at 4).
    """
    switch_rate = canonical_trial_switch_history(
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    )
    assert CANONICAL_SWITCH_WINDOW == 5
    assert switch_rate[-1] == pytest.approx(3 / 5)


def test_canonical_leaky_q_is_broadcast_once_per_explicit_completed_trial():
    df = _frame(
        [
            {
                "completed_trial_id": 0,
                "completed_trial_arm": 0.0,
                "occupancy_well_arm": np.nan,
                "is_rewarded_outcome": 1.0,
                "leaky_q_value_diff": 0.0,
            },
            {
                "completed_trial_id": 0,
                "completed_trial_arm": 0.0,
                "occupancy_well_arm": 4.0,
                "is_rewarded_outcome": 1.0,
                "leaky_q_value_diff": 0.0,
            },
            {
                "completed_trial_id": 1,
                "completed_trial_arm": 0.0,
                "occupancy_well_arm": 3.0,
                "is_rewarded_outcome": 1.0,
                "leaky_q_value_diff": 0.15,
            },
            {
                "completed_trial_id": 1,
                "completed_trial_arm": 0.0,
                "occupancy_well_arm": np.nan,
                "is_rewarded_outcome": 1.0,
                "leaky_q_value_diff": 0.15,
            },
            {
                "completed_trial_id": 2,
                "completed_trial_arm": 1.0,
                "occupancy_well_arm": 5.0,
                "is_rewarded_outcome": 0.0,
                "leaky_q_value_diff": 0.255,
            },
        ]
    )

    margins = add_uncertainty_covariates(df)["leaky_q_value_diff"].to_numpy()

    np.testing.assert_allclose(margins, [0.0, 0.0, 0.15, 0.15, 0.255])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"switch_window": 3}, "full trials table"),
        ({"q_alpha": 0.5}, "full trials table"),
    ],
)
def test_noncanonical_history_parameters_require_full_trials(kwargs, match):
    with pytest.raises(ValueError, match=match):
        add_uncertainty_covariates(_frame([{}]), **kwargs)


def test_missing_canonical_history_columns_requires_regeneration():
    df = _frame([{}]).drop(columns=["patch_switch_fraction_last5_completed_trips"])
    with pytest.raises(KeyError, match="regenerate stop_table"):
        add_uncertainty_covariates(df)


def test_empty_legacy_table_still_requires_canonical_schema():
    legacy = pd.DataFrame(columns=["animal", "session_date", "epoch", "stop_idx"])
    with pytest.raises(KeyError, match="regenerate stop_table"):
        add_uncertainty_covariates(legacy)


def test_no_mutation():
    df = _frame(
        [
            {"occupancy_well_arm": 0.0, "is_rewarded_outcome": 1.0, "is_stay": 1.0},
            {"occupancy_well_arm": 1.0, "is_rewarded_outcome": 0.0, "is_stay": 0.0},
        ]
    )
    before = df.copy(deep=True)
    out = add_uncertainty_covariates(df)
    pd.testing.assert_frame_equal(df, before)  # input unchanged
    assert out.index.equals(df.index)
    assert len(out) == len(df)
    for col in (
        "omission_streak",
        "patch_switch_fraction_last5_completed_trips",
        "wsls_deviation",
        "leaky_q_value_diff",
    ):
        assert col in out.columns


def test_visit_context_is_validated_per_session_not_pooled():
    """Each session's completed-trial sequence restarts independently.

    Two epochs of the same animal/session_date each number their completed trials
    from zero. Validating the pooled frame would see the second epoch's
    IDs step backwards and reject an ordinary stop table, so the session
    grouping is load-bearing rather than incidental.
    """
    rows = [
        {
            "occupancy_well_arm": 0.0,
            "completed_trial_id": 0,
            "patch_switch_fraction_last5_completed_trips": 0.0,
        },
        {
            "occupancy_well_arm": 1.0,
            "completed_trial_id": 1,
            "patch_switch_fraction_last5_completed_trips": 0.5,
        },
    ]
    first = _frame(rows)
    second = _frame(rows)
    second["epoch"] = "04_r2"
    pooled = pd.concat([first, second], ignore_index=True)

    out = add_uncertainty_covariates(pooled)

    assert len(out) == len(pooled)
    # Both epochs keep their own history, unshifted by the other's.
    for epoch in ("02_r1", "04_r2"):
        got = out.loc[
            out["epoch"] == epoch, "patch_switch_fraction_last5_completed_trips"
        ].tolist()
        assert got == [0.0, 0.5], epoch


def test_drop_first_session_per_animal():
    st = pd.DataFrame(
        {
            "animal": ["a", "a", "b"],
            "session_date": [1, 2, 1],
            "epoch": ["02_r1", "02_r1", "02_r1"],
            "stop_id": [0, 1, 2],
        }
    )
    out = fur._drop_first_session_per_animal(st)
    # a's first session (session_date 1) and b's only session (session_date 1) dropped.
    assert set(out["stop_id"]) == {1}


def _with_mu_fit_columns(data: dict) -> pd.DataFrame:
    """Complete a tiny physical-gate fixture with the exact MU design schema."""
    design = pd.DataFrame(data)
    # ``tc`` is the within-stop bin centre, not a filler: the gate audit
    # reports the retained time-course support, so a constant here would
    # make that report vacuous.
    if "tc" not in design and "binj" in design:
        design["tc"] = (design["binj"].to_numpy(dtype=float) + 0.5) * fur.BIN_S
    for column in fur._design_columns(fur.MU_UNCERTAINTY):
        if column not in design:
            design[column] = 0.0
    if "exposure" not in design:
        design["exposure"] = fur.BIN_S
    return design


def test_post_outcome_bin_gate_uses_physical_left_edge_and_keeps_equality():
    design = _with_mu_fit_columns(
        {
            "stop_id": [1, 1, 1, 2, 2, 2],
            "stop_start": [10.0] * 6,
            "binj": [0, 1, 2, 0, 1, 2],
            "outcome_known_time": [10.0 + fur.BIN_S] * 6,
            "trial_reward": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        }
    )

    gated, audit = fur._gate_bins_after_outcome(design)

    assert gated["binj"].tolist() == [1, 2, 1, 2]
    assert gated["response_time"].tolist() == pytest.approx(
        [
            10.0 + fur.BIN_S,
            10.0 + 2 * fur.BIN_S,
            10.0 + fur.BIN_S,
            10.0 + 2 * fur.BIN_S,
        ]
    )
    assert audit == {
        "n_bins_before_post_outcome_gate": 6,
        "n_bins_after_post_outcome_gate": 4,
        "n_rewarded_bins_before_post_outcome_gate": 3,
        "n_rewarded_bins_after_post_outcome_gate": 2,
        "n_omission_bins_before_post_outcome_gate": 3,
        "n_omission_bins_after_post_outcome_gate": 2,
        "n_bins_before_complete_case": 4,
        "n_rewarded_bins_before_complete_case": 2,
        "n_omission_bins_before_complete_case": 2,
        "n_bins_after_complete_case": 4,
        "n_rewarded_bins_after_complete_case": 2,
        "n_omission_bins_after_complete_case": 2,
        "n_bins_excluded_missing_timing": 0,
        "n_bins_excluded_pre_outcome": 2,
        "n_bins_excluded_incomplete_fit": 0,
        "n_bins_excluded_nonpositive_exposure": 0,
        "outcome_balance_valid": True,
        "outcome_balance_status": "both_outcomes_retained",
        "n_bins_after_outcome_balance_guard": 4,
        # Symmetric here by construction: both classes share a boundary.
        "min_retained_tc_rewarded": pytest.approx(1.5 * fur.BIN_S),
        "median_retained_tc_rewarded": pytest.approx(2.0 * fur.BIN_S),
        "min_retained_tc_omission": pytest.approx(1.5 * fur.BIN_S),
        "median_retained_tc_omission": pytest.approx(2.0 * fur.BIN_S),
        "min_retained_tc_omission_minus_rewarded": pytest.approx(0.0),
    }


def test_gate_audit_exposes_asymmetric_time_course_truncation():
    """An omission's later boundary shortens its retained time course.

    The spline's boundary knots are fixed, so a class whose early bins are
    all gated away contributes nothing to the low-``tc`` region and the
    basis stops being orthogonal to outcome. Counts alone cannot show this;
    the retained support can.
    """
    design = _with_mu_fit_columns(
        {
            "stop_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "stop_start": [10.0] * 8,
            "binj": [0, 1, 2, 3, 0, 1, 2, 3],
            # Rewarded stop 1 clears at once; omission stop 2 only at bin 3.
            "outcome_known_time": [10.0] * 4 + [10.0 + 3 * fur.BIN_S] * 4,
            "trial_reward": [1.0] * 4 + [0.0] * 4,
        }
    )

    _, audit = fur._gate_bins_after_outcome(design)

    assert audit["outcome_balance_valid"]
    assert audit["min_retained_tc_rewarded"] == pytest.approx(0.5 * fur.BIN_S)
    assert audit["min_retained_tc_omission"] == pytest.approx(3.5 * fur.BIN_S)
    assert audit["min_retained_tc_omission_minus_rewarded"] == pytest.approx(
        3.0 * fur.BIN_S
    )


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
            "trial_reward": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="constant within stop_id"):
        fur._gate_bins_after_outcome(design)


def test_post_outcome_bin_gate_suppresses_one_outcome_survivor_set():
    design = _with_mu_fit_columns(
        {
            "stop_id": [1, 1, 2, 2],
            "stop_start": [10.0] * 4,
            "binj": [0, 1, 0, 1],
            "outcome_known_time": [10.0, 10.0, 11.0, 11.0],
            "trial_reward": [1.0, 1.0, 0.0, 0.0],
        }
    )

    gated, audit = fur._gate_bins_after_outcome(design)

    assert gated.empty
    assert audit["n_rewarded_bins_after_post_outcome_gate"] == 2
    assert audit["n_omission_bins_after_post_outcome_gate"] == 0
    assert audit["outcome_balance_valid"] is False
    assert audit["n_bins_after_outcome_balance_guard"] == 0


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_complete_case_gate_blocks_rewarded_only_uncertainty_fit(
    monkeypatch, invalid_value
):
    """The sole omission's non-finite predictor is audited before any GLM call."""
    stop = pd.DataFrame(
        {
            "animal": ["a", "a"],
            "session_date": [1, 1],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [1, 2],
            "immobility_duration_s": [0.06, 0.06],
            "stop_start": [10.0, 20.0],
            "outcome_known_time": [10.0, 20.0],
            "trial_reward": [1.0, 0.0],
            "reward_fraction_last10_known_trials": [0.5, 0.5],
            "did_last_completed_trip_switch_patch": [0.0, 0.0],
            **{
                f"reward_history_rcos_{k}": [0.4 + 0.1 * k, 0.5 - 0.1 * k]
                for k in range(3)
            },
            "reward_history_incomplete": [0.0, 0.0],
            **{column: [0.1, 0.2] for column in fur.FITTED_COVARIATES},
        }
    )
    stop.loc[stop["trial_reward"] == 0.0, "task_reward_probability_chosen_arm"] = (
        invalid_value
    )
    onsets = pd.DataFrame(
        {
            "animal": ["a", "a", "a"],
            "session_date": [1, 1, 1],
            "epoch": ["02_r1", "02_r1", "02_r1"],
            "stop_id": [1, 1, 1],
            "t_since_stop_onset": [0.01, 0.02, 0.03],
            "event_type": ["SWR", "non_local", "non_local"],
            "is_swr": [True, True, False],
        }
    )
    calls = []

    def _record_fit(design, spec):
        calls.append((design, spec))
        return {"n_stops": design["stop_id"].nunique()}

    monkeypatch.setattr(fur, "fit_per_bin_rate_model", _record_fit)

    models = fur.fit_uncertainty_rate(stop, onsets)

    assert models.empty
    assert calls == []
    audits = models.attrs["post_outcome_gate_audit"]
    assert len(audits) == len(fur.POPULATIONS_TO_FIT)
    for audit in audits:
        assert audit["n_omission_bins_before_complete_case"] > 0
        assert audit["n_omission_bins_after_complete_case"] == 0
        assert audit["n_rewarded_bins_after_complete_case"] > 0
        assert audit["n_bins_excluded_incomplete_fit"] > 0
        assert audit["outcome_balance_status"] == "missing_after_complete_case:omission"


@pytest.mark.slow
@pytest.mark.real_data
def test_runner_smoke():
    """End-to-end runner on one post-exposure session per animal.

    Requires the canonical q-values table alongside the canonical tables. Writes
    ``uncertainty_rate_smoke.pkl`` and checks the tidy schema carries every
    uncertainty/value coefficient across the three populations.
    """
    from continuum_swr_replay.analysis.replay import (
        REWARD_HISTORY_COLUMNS,
        attach_is_swr_to_onsets,
    )

    rc = REPLAY_CONTENT_DIR
    qvals_path = table_path("qvals_table", rc, fmt="pkl")
    if not table_exists(qvals_path):
        pytest.fail(
            "qvals_table not present (run "
            "`uv run python scripts/behavior/build_qvals_table.py --write` first)"
        )

    stop = read_table(table_path("stop_table", rc, fmt="pkl"))
    canonical_columns = {
        "completed_trial_id",
        "completed_trial_arm",
        "patch_switch_fraction_last5_completed_trips",
        "leaky_q_value_diff",
        "completed_trial_reward",
    }
    if not canonical_columns.issubset(stop.columns):
        pytest.fail("canonical stop_table.pkl has not been regenerated")
    onset_table = read_table(table_path("event_onset_table", rc, fmt="pkl"))
    event_table = read_table(table_path("event_table", rc, fmt="pkl"))
    # The helper freezes the shared natural-session cohort and warns about any
    # independently regenerated onset-only/event-only sessions. Unit tests pin
    # that warning/audit; this slow smoke should continue across the shared
    # sessions instead of reinstating the old whole-corpus count blocker.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        onsets = attach_is_swr_to_onsets(onset_table, event_table)

    stop = add_uncertainty_covariates(stop)
    # Nominal schedule + real model covariates, all from qvals_table -- one
    # stop-start-aligned source with a separate physical outcome boundary,
    # mirroring the runner's main().
    qvals = read_table(qvals_path)
    if "outcome_known_time" not in qvals:
        pytest.fail(
            "qvals_table.pkl predates physical post-outcome timing; regenerate it"
        )
    qcols = qvals[
        [
            "stop_id",
            "outcome_known_time",
            "trial_reward",
            *fur.VALUE_COVARIATES,
            "rpe_postoutcome",
            *fur.RL_COVARIATES,
        ]
    ]
    stop = stop.merge(qcols, on="stop_id", how="inner", validate="many_to_one")
    stop = fur._drop_first_session_per_animal(stop)
    stop = subset_max_per_animal(stop, 1, sort_sessions=True)

    models = fur.fit_uncertainty_rate(stop, onsets)
    assert len(models) > 0
    assert set(models["outcome"]).issubset(set(fur.POPULATIONS_TO_FIT))
    # every curated uncertainty/value covariate the model fits has a
    # coefficient/RR column (MU_UNCERTAINTY_CURATED_COVARIATES is the set
    # actually fit; it supersets VALUE_COVARIATES).
    for cov in MU_UNCERTAINTY_CURATED_COVARIATES:
        assert f"coef_{cov}" in models.columns
        assert f"rr_{cov}" in models.columns
    # reward-history basis is also present (the controlled-for block).
    for column in REWARD_HISTORY_COLUMNS:
        assert f"coef_{column}" in models.columns
