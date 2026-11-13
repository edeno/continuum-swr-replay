"""Persisted standardization reproduces the materialized ``_z``.

Proves the builder fold is reproducible — ``attach_standardized`` over the declared
scientific support gives the same ``_z`` columns that ship after regeneration.
Also exercises the frozen-reference
round-trip (F5): a reference computed from the full table, re-applied to a one-session
slice, must match the materialized ``_z`` for that slice — a per-session *recompute*
would not (that is why the parity gate must apply the frozen reference).

Marked slow and real_data: reads the full materialized tables, and fails if they
are absent or stale.

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

from continuum_swr_replay.analysis.behavior.covariates.qvals_table import (
    QVALS_STANDARDIZED_COLS,
)
from continuum_swr_replay.analysis.event_tables._common import (
    EVENT_ARM_STANDARDIZED_COLS,
)
from continuum_swr_replay.paths import replay_content_dir, table_path
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table, table_exists
from continuum_swr_replay.tables.standardization import (
    attach_standardized,
    load_reference,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.real_data]


# The live tables, resolved in the test (a real_data test reads the live root;
# this module's import sees the suite's sentinel root).
def _event_arm() -> Path:
    return table_path("event_arm_table")


def _event_arm_ref() -> Path:
    return replay_content_dir() / "standardization_reference.json"


def _qvals() -> Path:
    return table_path("qvals_table")


def _qvals_ref() -> Path:
    return replay_content_dir() / "qvals_standardization_reference.json"


def _require(path: Path, *, table: str):
    if not table_exists(path):
        pytest.fail(f"materialized table not present: {path}")
    frame = read_table(path)
    try:
        require_covariate_schema(frame, table=table)
    except ValueError as error:
        pytest.fail(f"materialized table requires canonical regeneration: {error}")
    return frame


@pytest.mark.parametrize(
    "table_file, cols",
    [(_event_arm, EVENT_ARM_STANDARDIZED_COLS), (_qvals, QVALS_STANDARDIZED_COLS)],
)
def test_folded_standardization_reproduces_materialized_z(table_file, cols):
    """Recomputing the scientific reference gives the materialized transform."""
    path = table_file()
    table = "event_arm" if table_file is _event_arm else "qvals"
    df = _require(path, table=table)
    cols = [c for c in cols if c in df.columns and f"{c}_z" in df.columns]
    assert cols, f"no standardized columns to check in {path.name}"
    if table_file is _event_arm:
        if not _event_arm_ref().exists():
            pytest.fail(
                "leaf-support reference not written; regenerate event-arm table"
            )
        current_reference = load_reference(_event_arm_ref())
        if any(
            current_reference.get(column, {}).get("support") != "leaf_arms"
            for column in cols
        ):
            pytest.fail(
                "materialized event-arm z columns predate leaf-support scaling; "
                "regenerate canonical tables"
            )
        recomputed, _ = attach_standardized(
            df[cols].copy(),
            cols,
            support_mask=df["candidate_arm"].isin(range(6)),
            support_label="leaf_arms",
        )
    else:
        recomputed, _ = attach_standardized(df[cols].copy(), cols)
    for c in cols:
        np.testing.assert_allclose(
            recomputed[f"{c}_z"].to_numpy(),
            df[f"{c}_z"].to_numpy(),
            atol=1e-10,
            equal_nan=True,
            err_msg=f"folded _z drifted from materialized for {c} in {path.name}",
        )


@pytest.mark.parametrize(
    "table_file, ref_file, cols",
    [
        (_event_arm, _event_arm_ref, EVENT_ARM_STANDARDIZED_COLS),
        (_qvals, _qvals_ref, QVALS_STANDARDIZED_COLS),
    ],
)
def test_persisted_reference_reproduces_materialized_z(table_file, ref_file, cols):
    """F5 frozen-reference gate: apply the PERSISTED reference JSON to the materialized
    raw columns and match the materialized _z within atol=1e-10. Fails until the
    operational rebuild has written the reference — so an absent/stale reference
    cannot silently pass."""
    table_path, ref_path = table_file(), ref_file()
    table = "event_arm" if table_file is _event_arm else "qvals"
    df = _require(table_path, table=table)
    if not ref_path.exists():
        pytest.fail(
            f"reference not written yet: {ref_path} (run the full builder rebuild)"
        )
    reference = load_reference(ref_path)
    if any(
        not {"support", "n_reference_rows"}.issubset(reference.get(column, {}))
        for column in cols
    ):
        pytest.fail(
            f"materialized {ref_path.name} predates support-aware reference "
            "metadata; regenerate its owning table"
        )
    if table_file is _event_arm and any(
        reference.get(column, {}).get("support") != "leaf_arms" for column in cols
    ):
        pytest.fail(
            "materialized event-arm reference predates leaf-support scaling; "
            "regenerate canonical tables"
        )
    cols = [c for c in cols if c in df.columns and f"{c}_z" in df.columns]
    applied, _ = attach_standardized(df[cols].copy(), cols, reference=reference)
    for c in cols:
        np.testing.assert_allclose(
            applied[f"{c}_z"].to_numpy(),
            df[f"{c}_z"].to_numpy(),
            atol=1e-10,
            equal_nan=True,
            err_msg=f"persisted-reference _z mismatch for {c} in {table_path.name}",
        )


def test_frozen_reference_matches_on_one_session_slice():
    """F5: a full-table reference applied to ONE session's raw rows matches the
    materialized _z for those rows — a per-session recompute would NOT (different
    mean/SD). Uses the persisted reference when available, else the table-derived one."""
    df = _require(_event_arm(), table="event_arm")
    cols = [
        c
        for c in EVENT_ARM_STANDARDIZED_COLS
        if c in df.columns and f"{c}_z" in df.columns
    ]
    if _event_arm_ref().exists():
        reference = load_reference(_event_arm_ref())
        if any(
            reference.get(column, {}).get("support") != "leaf_arms" for column in cols
        ):
            pytest.fail(
                "materialized event-arm reference predates leaf-support scaling; "
                "regenerate canonical tables"
            )
    else:
        _, reference = attach_standardized(
            df[cols].copy(),
            cols,
            support_mask=df["candidate_arm"].isin(range(6)),
            support_label="leaf_arms",
        )

    session_cols = ["animal", "session_date", "epoch"]
    first_key = df[session_cols].drop_duplicates().iloc[0]
    mask = (df[session_cols] == first_key).all(axis=1).to_numpy()
    slice_raw = df.loc[mask, cols].copy()

    applied, _ = attach_standardized(slice_raw, cols, reference=reference)
    for c in cols:
        np.testing.assert_allclose(
            applied[f"{c}_z"].to_numpy(),
            df.loc[mask, f"{c}_z"].to_numpy(),
            atol=1e-10,
            equal_nan=True,
            err_msg=f"frozen-reference _z mismatch for {c}",
        )
