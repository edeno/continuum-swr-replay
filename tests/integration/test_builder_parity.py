"""Phase-6 parity: the folded standardization reproduces the materialized ``_z``.

Proves the builder fold is inert — ``attach_standardized`` over the full table gives the
same ``_z`` columns that currently ship (originally stamped by the retired
``_add_standardized_cols`` one-off, same formula). Also exercises the frozen-reference
round-trip (F5): a reference computed from the full table, re-applied to a one-session
slice, must match the materialized ``_z`` for that slice — a per-session *recompute*
would not (that is why the parity gate must apply the frozen reference).

Marked slow: reads the full materialized tables. Skips if they are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.qvals_table import QVALS_STANDARDIZED_COLS
from continuum_swr_replay.analysis.replay.standardization import attach_standardized
from continuum_swr_replay.analysis.replay.tables import EVENT_ARM_STANDARDIZED_COLS
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

pytestmark = pytest.mark.slow

_EVENT_ARM = PROCESSED_DATA_DIR / "replay_content" / "event_arm_table.pkl"
_QVALS = PROCESSED_DATA_DIR / "replay_content" / "models_per_bin_rate" / "qvals_table.pkl"


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"materialized table not present: {path}")
    return pd.read_pickle(path)


@pytest.mark.parametrize(
    "path, cols",
    [(_EVENT_ARM, EVENT_ARM_STANDARDIZED_COLS), (_QVALS, QVALS_STANDARDIZED_COLS)],
)
def test_folded_standardization_reproduces_materialized_z(path, cols):
    df = _require(path)
    cols = [c for c in cols if c in df.columns and f"{c}_z" in df.columns]
    assert cols, f"no standardized columns to check in {path.name}"
    recomputed, _ = attach_standardized(df[cols].copy(), cols)
    for c in cols:
        np.testing.assert_allclose(
            recomputed[f"{c}_z"].to_numpy(),
            df[f"{c}_z"].to_numpy(),
            atol=1e-10, equal_nan=True,
            err_msg=f"folded _z drifted from materialized for {c} in {path.name}",
        )


def test_frozen_reference_matches_on_one_session_slice():
    """F5: apply a full-table reference to one session's raw columns and match the
    materialized _z for those rows — a per-session recompute would NOT match."""
    df = _require(_EVENT_ARM)
    cols = [
        c for c in EVENT_ARM_STANDARDIZED_COLS
        if c in df.columns and f"{c}_z" in df.columns
    ]
    _, reference = attach_standardized(df[cols].copy(), cols)  # full-table reference

    session_cols = ["animal", "day", "epoch"]
    first_key = df[session_cols].drop_duplicates().iloc[0]
    mask = (df[session_cols] == first_key).all(axis=1).to_numpy()
    slice_raw = df.loc[mask, cols].copy()

    applied, _ = attach_standardized(slice_raw, cols, reference=reference)
    for c in cols:
        np.testing.assert_allclose(
            applied[f"{c}_z"].to_numpy(),
            df.loc[mask, f"{c}_z"].to_numpy(),
            atol=1e-10, equal_nan=True,
            err_msg=f"frozen-reference _z mismatch for {c}",
        )
