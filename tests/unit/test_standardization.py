"""Pooled-global standardization: formula, zero-variance branch, reference roundtrip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.standardization import (
    assert_standardized_present,
    attach_standardized,
    load_reference,
    save_reference,
)


def test_z_formula():
    """(x - nanmean) / nanstd, with NaNs ignored in the moments and propagated."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan, 5.0]})
    out, ref = attach_standardized(df, ["a"])
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    expected = (x - np.nanmean(x)) / np.nanstd(x)
    np.testing.assert_allclose(out["a_z"].to_numpy(), expected, equal_nan=True)
    assert ref["a"]["mean"] == np.nanmean(x)
    assert ref["a"]["std"] == np.nanstd(x)


def test_zero_variance_is_nan():
    """A constant column has std==0 -> all-NaN _z (coefficient unidentified)."""
    df = pd.DataFrame({"c": [4.0, 4.0, 4.0]})
    out, ref = attach_standardized(df, ["c"])
    assert out["c_z"].isna().all()
    assert ref["c"]["std"] == 0.0


def test_absent_column_skipped():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    out, ref = attach_standardized(df, ["a", "missing"])
    assert "a_z" in out.columns
    assert "missing_z" not in out.columns
    assert "missing" not in ref


def test_reference_roundtrip(tmp_path):
    """Applying a persisted reference to raw columns reproduces the _z exactly."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    _, ref = attach_standardized(df.copy(), ["a", "b"])

    path = tmp_path / "reference.json"
    save_reference(ref, path)
    loaded = load_reference(path)

    # A fresh frame with the SAME raw values, standardized via the frozen reference.
    fresh = pd.DataFrame({"a": df["a"], "b": df["b"]})
    applied, used = attach_standardized(fresh, ["a", "b"], reference=loaded)
    direct, _ = attach_standardized(df.copy(), ["a", "b"])
    for c in ("a_z", "b_z"):
        np.testing.assert_allclose(applied[c].to_numpy(), direct[c].to_numpy(), atol=1e-12)
    assert used == loaded


def test_assert_present_passes_after_attach():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    attach_standardized(df, ["a", "b"])
    assert_standardized_present(df, ["a", "b"])  # no raise


def test_assert_present_fires_when_z_dropped():
    """The rebuild guard must fail if a _z column is missing (not a tautology)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    attach_standardized(df, ["a"])
    df = df.drop(columns=["a_z"])  # simulate a refactor silently dropping it
    with pytest.raises(RuntimeError, match="missing"):
        assert_standardized_present(df, ["a"])


def test_assert_present_fires_when_all_nan():
    """If every standardized column is all-NaN (e.g. a total wipeout), the guard fires."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    df["a_z"] = np.nan
    with pytest.raises(RuntimeError, match="all-NaN"):
        assert_standardized_present(df, ["a"])


def test_reference_applied_differs_from_recompute_on_subset():
    """The point of a frozen reference: a subset re-standardized against the FULL
    reference differs from a naive per-subset recompute (the F5 hazard)."""
    rng = np.random.default_rng(1)
    full = pd.DataFrame({"a": rng.normal(10.0, 3.0, size=200)})
    _, ref = attach_standardized(full.copy(), ["a"])
    subset = full.iloc[:20].copy()
    via_ref, _ = attach_standardized(subset.copy(), ["a"], reference=ref)
    per_subset, _ = attach_standardized(subset.copy(), ["a"])
    # Same raw values, but different means/SDs -> different _z (this is why the parity
    # gate must apply the frozen reference, not recompute per session).
    assert not np.allclose(via_ref["a_z"].to_numpy(), per_subset["a_z"].to_numpy())
