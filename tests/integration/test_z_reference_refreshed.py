"""Phase-4: the standardization reference is refreshed for the encoded table.

The mechanism-A encode (unvisited-leaf ``reward_rate_arm_k10`` -> 0.0) shifts that
column's distribution, so its pooled-global mean/SD must be recomputed over the encoded
table. A stale reference (means/SDs left over from the pre-encode table) would make
``reward_rate_arm_k10_z`` a wrong transform of the raw column — the exact regression the
plan's ``test_z_reference_refreshed`` guards.

Durable formulation: rather than diff against transient pre-encode backup values, assert
the persisted reference **equals the current leaf-support table's** ``np.nanmean`` /
``np.nanstd`` — structural-zero trunks do not identify the scientific unit — and that
``<col>_z`` is exactly ``(raw - ref_mean) / ref_std``. Marked slow; skips if the
materialized tables are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from continuum_swr_replay.analysis.replay._tables_common import (
    EVENT_ARM_STANDARDIZED_COLS,
)
from continuum_swr_replay.analysis.replay.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.analysis.replay.standardization import load_reference
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import read_table, table_exists

pytestmark = pytest.mark.slow

_RC = PROCESSED_DATA_DIR / "replay_content"
_EVENT_ARM = _RC / "event_arm_table.pkl"
_EVENT_ARM_REF = _RC / "standardization_reference.json"

_ENCODED_COL = "reward_fraction_candidate_arm_last10_visits"


def _require(path: Path):
    """Guard a plain file (the JSON reference), which has no columnar sibling."""
    if not path.exists():
        pytest.skip(f"materialized file not present: {path}")
    return path


def _require_table(path: Path):
    """Guard a canonical table, which may be on disk in either format."""
    if not table_exists(path):
        pytest.skip(f"materialized table not present: {path}")
    return path


def _load():
    ea = read_table(_require_table(_EVENT_ARM))
    try:
        require_covariate_schema(ea, table="event_arm")
    except ValueError as error:
        pytest.skip(f"materialized table requires canonical regeneration: {error}")
    ref = load_reference(_require(_EVENT_ARM_REF))
    return ea, ref


def test_reference_matches_encoded_table_moments():
    """Persisted mean/SD == nanmean/nanstd of the *current* (encoded) raw column.

    A reference frozen before the encode would carry the old (higher) mean and a
    different SD; matching the encoded table's own moments proves it was refreshed.
    """
    ea, ref = _load()
    assert _ENCODED_COL in ref, f"{_ENCODED_COL} missing from reference"
    if ref[_ENCODED_COL].get("support") != "leaf_arms":
        pytest.skip(
            "materialized reference predates leaf-support scaling; regenerate tables"
        )
    leaf = ea["candidate_arm"].isin(range(6))
    x = ea.loc[leaf, _ENCODED_COL].to_numpy(dtype=float)
    assert ref[_ENCODED_COL]["support"] == "leaf_arms"
    assert ref[_ENCODED_COL]["n_reference_rows"] == int(np.isfinite(x).sum())
    assert np.nanmean(x) == pytest.approx(ref[_ENCODED_COL]["mean"], rel=1e-9, abs=1e-9)
    assert np.nanstd(x) == pytest.approx(ref[_ENCODED_COL]["std"], rel=1e-9, abs=1e-9)


def test_z_equals_reference_transform():
    """``<col>_z`` is exactly ``(raw - ref_mean) / ref_std`` for every finite raw row."""
    ea, ref = _load()
    present = [c for c in EVENT_ARM_STANDARDIZED_COLS if c in ea.columns]
    # Fail loud rather than pass vacuously if the standardized cols vanished.
    assert present, "no EVENT_ARM_STANDARDIZED_COLS present in materialized table"
    for col in present:
        zc = f"{col}_z"
        assert zc in ea.columns, f"missing {zc}"
        mu, sd = ref[col]["mean"], ref[col]["std"]
        if not sd > 0:
            continue
        raw = ea[col].to_numpy(dtype=float)
        z = ea[zc].to_numpy(dtype=float)
        finite = np.isfinite(raw)
        np.testing.assert_allclose((raw[finite] - mu) / sd, z[finite], rtol=1e-9, atol=1e-9)
        # NaN raw -> NaN _z (the encode leaves no-trial-history rows NaN)
        assert np.isnan(z[~finite]).all()


def test_encode_signature_present():
    """The encoded column carries the 0.0 fill: many exact zeros, few residual NaN.

    Sanity that this test is reading a *post-encode* table (not a pre-encode artifact):
    the unvisited-leaf/trunk encode makes exact-0.0 the dominant value and leaves only
    the session-opening no-history rows NaN.
    """
    ea, _ = _load()
    x = ea[_ENCODED_COL].to_numpy(dtype=float)
    n_zero = int((x == 0.0).sum())
    n_nan = int(np.isnan(x).sum())
    assert n_zero > n_nan, f"expected encode zeros to dominate NaN (zeros={n_zero}, nan={n_nan})"
