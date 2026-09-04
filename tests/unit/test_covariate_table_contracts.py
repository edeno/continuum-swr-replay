"""Scientific row-contract tests for canonical covariate tables.

These tests exercise the real pandas validation code.  They intentionally use
small, fully specified frames so failures identify the scientific invariant
that was violated rather than a downstream groupby or model-fitting error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.table_contracts import (
    validate_event_arm_table,
    validate_event_patch_table,
)


def _event_table(event_ids: tuple[int, ...] = (10, 11)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": "rat",
            "session_date": 20200101,
            "epoch": "02_r1",
            "event_id": event_ids,
        }
    )


def _event_arm_table(event_ids: tuple[int, ...] = (10, 11)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event_id in event_ids:
        for arm in range(9):
            rows.append(
                {
                    "animal": "rat",
                    "session_date": 20200101,
                    "epoch": "02_r1",
                    "event_id": event_id,
                    "candidate_arm": arm,
                    "arm_count_hard": int(arm == event_id % 9),
                    "arm_fraction": float(arm == event_id % 9),
                    "arm_fraction_posterior": 1.0 / 9.0,
                    "n_nonlocal_bins": 1,
                    "n_undecoded_bins": 0,
                    "visit_recency_trials": float(arm + 1) if arm < 6 else 0.0,
                    "is_candidate_arm_never_visited": 0.0,
                    "reward_fraction_candidate_arm_last10_visits": (
                        float(arm) / 10 if arm < 6 else 0.0
                    ),
                    "task_reward_probability_candidate_arm": (
                        float(arm) / 10 if arm < 6 else 0.0
                    ),
                    "reward_belief_mean_candidate_arm_preoutcome": (
                        float(arm) / 10 if arm < 6 else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_event_arm_contract_accepts_exact_candidate_rows() -> None:
    validate_event_arm_table(_event_arm_table(), event_table=_event_table())


@pytest.mark.parametrize("bad_arm", [-1, 9, 1.5, np.nan])
def test_event_arm_contract_rejects_invalid_candidate_arm(bad_arm: float) -> None:
    table = _event_arm_table()
    table["candidate_arm"] = table["candidate_arm"].astype(float)
    table.loc[table.index[0], "candidate_arm"] = bad_arm
    with pytest.raises(ValueError, match="candidate-arm set"):
        validate_event_arm_table(table, event_table=_event_table())


def test_event_arm_contract_rejects_duplicate_that_replaces_a_candidate() -> None:
    table = _event_arm_table()
    event_rows = table.index[table["event_id"].eq(10)]
    table.loc[event_rows[-1], "candidate_arm"] = 7
    with pytest.raises(ValueError, match="candidate-arm set"):
        validate_event_arm_table(table, event_table=_event_table())


def test_event_arm_contract_rejects_event_set_mismatch() -> None:
    table = _event_arm_table(event_ids=(10,))
    with pytest.raises(ValueError, match="event-ID set"):
        validate_event_arm_table(table, event_table=_event_table())


def test_event_arm_contract_rejects_response_nonconservation() -> None:
    table = _event_arm_table()
    table.loc[
        (table["event_id"] == 10) & (table["candidate_arm"] == 0), "arm_count_hard"
    ] = 2
    with pytest.raises(ValueError, match="response counts do not conserve"):
        validate_event_arm_table(table, event_table=_event_table())


def test_event_arm_contract_rejects_partial_leaf_support() -> None:
    table = _event_arm_table()
    table.loc[
        (table["event_id"] == 10) & (table["candidate_arm"] == 0),
        "task_reward_probability_candidate_arm",
    ] = np.nan
    with pytest.raises(ValueError, match="partial leaf support"):
        validate_event_arm_table(table, event_table=_event_table())


def test_event_arm_contract_rejects_partial_or_unnormalized_fraction() -> None:
    partial = _event_arm_table()
    partial.loc[
        (partial["event_id"] == 10) & (partial["candidate_arm"] == 0),
        "arm_fraction_posterior",
    ] = np.nan
    with pytest.raises(ValueError, match="partial candidate support"):
        validate_event_arm_table(partial, event_table=_event_table())

    unnormalized = _event_arm_table()
    unnormalized.loc[unnormalized["event_id"] == 10, "arm_fraction_posterior"] = 0.1
    with pytest.raises(ValueError, match="sum to one"):
        validate_event_arm_table(unnormalized, event_table=_event_table())


def _event_patch_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": "rat",
            "session_date": 20200101,
            "epoch": "02_r1",
            "event_id": np.repeat([10, 11], 3),
            "patch": np.tile([1, 2, 3], 2),
            "patch_count_hard": [1, 0, 0, 0, 1, 0],
        }
    )


def test_event_patch_contract_accepts_exact_patch_rows() -> None:
    validate_event_patch_table(_event_patch_table(), event_table=_event_table())


def test_event_patch_contract_rejects_duplicate_patch_replacement() -> None:
    table = _event_patch_table()
    table.loc[(table["event_id"] == 10) & (table["patch"] == 3), "patch"] = 2
    with pytest.raises(ValueError, match="patch set"):
        validate_event_patch_table(table, event_table=_event_table())
