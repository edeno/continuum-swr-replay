"""Phase-6 parity: the folded standardization reproduces the materialized ``_z``.

Proves the builder fold is inert — ``attach_standardized`` over the full table gives the
same ``_z`` columns that currently ship (originally stamped by the retired
``_add_standardized_cols`` one-off, same formula). Also exercises the frozen-reference
round-trip (F5): a reference computed from the full table, re-applied to a one-session
slice, must match the materialized ``_z`` for that slice — a per-session *recompute*
would not (that is why the parity gate must apply the frozen reference).

Marked slow: reads the full materialized tables. Skips if they are absent.

Scope note — the **raw (non-``_z``) column** one-session rebuild parity (rebuild one
session's ``event_arm_table`` and diff its non-``_z`` columns against the materialized
file) is NOT run here: it needs that session's decoder outputs, which are unavailable on
this host while Spyglass/NWB access is down (see ``docs/cached_trials_for_testing.md``).
It is the operational belt-and-suspenders check, run after the full rebuild. The
``_z`` work leaves the raw columns unchanged *by construction* — ``build_event_arm_table`` only
appends all-NaN ``_z`` placeholders before its existing ``long[column_order]`` return
(verified by ``test_replay_tables.py::...test_standardized_z_are_per_session_nan_placeholders``),
and the global ``_z`` fill happens post-concat in ``prepare_data``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from continuum_swr_replay.analysis.behavior.qvals_table import QVALS_STANDARDIZED_COLS
from continuum_swr_replay.analysis.replay.standardization import (
    attach_standardized,
    load_reference,
)
from continuum_swr_replay.analysis.replay.tables import EVENT_ARM_STANDARDIZED_COLS
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import read_table, table_exists

pytestmark = pytest.mark.slow

_RC = PROCESSED_DATA_DIR / "replay_content"
_EVENT_ARM = _RC / "event_arm_table.pkl"
_EVENT_ARM_REF = _RC / "standardization_reference.json"
_QVALS = _RC / "models_per_bin_rate" / "qvals_table.pkl"
_QVALS_REF = _RC / "models_per_bin_rate" / "qvals_standardization_reference.json"


def _require(path: Path):
    if not table_exists(path):
        pytest.skip(f"materialized table not present: {path}")
    return read_table(path)


@pytest.mark.parametrize(
    "path, cols",
    [(_EVENT_ARM, EVENT_ARM_STANDARDIZED_COLS), (_QVALS, QVALS_STANDARDIZED_COLS)],
)
def test_folded_standardization_reproduces_materialized_z(path, cols):
    """Inertness of the FORMULA: recomputing the fold gives the materialized _z. (Not the
    frozen-reference gate — that is the JSON test below.)"""
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


@pytest.mark.parametrize(
    "table_path, ref_path, cols",
    [
        (_EVENT_ARM, _EVENT_ARM_REF, EVENT_ARM_STANDARDIZED_COLS),
        (_QVALS, _QVALS_REF, QVALS_STANDARDIZED_COLS),
    ],
)
def test_persisted_reference_reproduces_materialized_z(table_path, ref_path, cols):
    """F5 frozen-reference gate: apply the PERSISTED reference JSON to the materialized
    raw columns and match the materialized _z within atol=1e-10. Skips (does not pass)
    until the operational rebuild has written the reference — so an absent/stale
    reference cannot silently pass."""
    df = _require(table_path)
    if not ref_path.exists():
        pytest.skip(
            f"reference not written yet: {ref_path} (run the full builder rebuild)"
        )
    reference = load_reference(ref_path)
    cols = [c for c in cols if c in df.columns and f"{c}_z" in df.columns]
    applied, _ = attach_standardized(df[cols].copy(), cols, reference=reference)
    for c in cols:
        np.testing.assert_allclose(
            applied[f"{c}_z"].to_numpy(),
            df[f"{c}_z"].to_numpy(),
            atol=1e-10, equal_nan=True,
            err_msg=f"persisted-reference _z mismatch for {c} in {table_path.name}",
        )


def test_frozen_reference_matches_on_one_session_slice():
    """F5: a full-table reference applied to ONE session's raw rows matches the
    materialized _z for those rows — a per-session recompute would NOT (different
    mean/SD). Uses the persisted reference when available, else the table-derived one."""
    df = _require(_EVENT_ARM)
    cols = [
        c for c in EVENT_ARM_STANDARDIZED_COLS
        if c in df.columns and f"{c}_z" in df.columns
    ]
    if _EVENT_ARM_REF.exists():
        reference = load_reference(_EVENT_ARM_REF)
    else:
        _, reference = attach_standardized(df[cols].copy(), cols)

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
