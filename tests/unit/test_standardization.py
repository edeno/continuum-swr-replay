"""Pooled-global standardization: formula, zero-variance branch, reference roundtrip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.tables.standardization import (
    assert_standardized_matches_reference,
    assert_standardized_present,
    attach_standardized,
    load_reference,
    save_reference,
    z_reference_path,
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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
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
        np.testing.assert_allclose(
            applied[c].to_numpy(), direct[c].to_numpy(), atol=1e-12
        )
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


def test_assert_present_fires_when_z_all_nan_but_source_varies():
    """A wiped _z whose SOURCE has variance is a failure (per-column check)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    df["a_z"] = np.nan
    with pytest.raises(RuntimeError, match="varying source"):
        assert_standardized_present(df, ["a"])


def test_assert_present_fires_on_partial_failure():
    """One valid _z must NOT mask a second column whose _z is wrongly all-NaN."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    attach_standardized(df, ["a", "b"])
    df["b_z"] = np.nan  # b varies but its _z was wiped
    with pytest.raises(RuntimeError, match="b_z"):
        assert_standardized_present(df, ["a", "b"])


def test_assert_present_allows_zero_variance_all_nan():
    """A genuinely zero-variance (finite constant) source yields an all-NaN _z; allowed."""
    df = pd.DataFrame({"c": [4.0, 4.0, 4.0]})  # constant -> std 0 -> _z all NaN
    attach_standardized(df, ["c"])
    assert df["c_z"].isna().all()
    assert_standardized_present(df, ["c"])  # no raise


def test_assert_present_fires_on_all_nan_source():
    """An entirely-NaN source (a failed upstream join) must NOT be waved through as if
    it were a legitimate zero-variance column."""
    df = pd.DataFrame({"q": [np.nan, np.nan, np.nan]})
    with pytest.raises(ValueError, match="no finite values"):
        attach_standardized(df, ["q"])


def test_apply_missing_reference_column_raises():
    """A partial/stale reference (missing a requested column) must fail loud, not
    silently recompute that column's mean/SD."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    partial = {
        "a": {
            "mean": 2.0,
            "std": 1.0,
            "support": "all_rows",
            "n_reference_rows": 3,
        }
    }  # 'b' absent
    with pytest.raises(KeyError, match="'b'"):
        attach_standardized(df, ["a", "b"], reference=partial)


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


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_nonfinite_source_fails_loudly(bad):
    """Infinity is corrupted input, not ignorable missingness."""
    df = pd.DataFrame({"a": [1.0, 2.0, bad]})
    with pytest.raises(ValueError, match="non-finite.*a"):
        attach_standardized(df, ["a"])


@pytest.mark.parametrize(
    "reference",
    [
        {"a": {"mean": np.inf, "std": 1.0}},
        {"a": {"mean": 0.0, "std": np.inf}},
        {"a": {"mean": 0.0, "std": np.nan}},
    ],
)
def test_nonfinite_reference_fails_loudly(reference):
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="non-finite standardization reference"):
        attach_standardized(df, ["a"], reference=reference)


def test_support_mask_sets_reference_on_scientific_support():
    """Structural-zero trunk rows must not set a leaf-identified coefficient unit."""
    df = pd.DataFrame(
        {
            "candidate_arm": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "uncertainty": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0],
        }
    )
    leaf = df["candidate_arm"].lt(6).to_numpy()
    out, ref = attach_standardized(
        df,
        ["uncertainty"],
        support_mask=leaf,
        support_label="leaf_arms",
    )

    np.testing.assert_allclose(
        np.nanstd(out.loc[leaf, "uncertainty_z"].to_numpy(dtype=float)), 1.0
    )
    assert ref["uncertainty"]["support"] == "leaf_arms"
    assert ref["uncertainty"]["n_reference_rows"] == 6


def test_apply_rejects_legacy_reference_missing_support_metadata():
    frame = pd.DataFrame({"q": [0.1, 0.2, 0.3]})
    legacy = {"q": {"mean": 0.2, "std": 0.1}}

    with pytest.raises(ValueError, match="missing required metadata"):
        attach_standardized(
            frame,
            ["q"],
            reference=legacy,
            support_label="leaf_arms",
        )


def test_empty_support_fails_without_emitting_nan_reference():
    frame = pd.DataFrame({"q": [0.1, 0.2, 0.3]})

    with pytest.raises(ValueError, match="at least one row"):
        attach_standardized(frame, ["q"], support_mask=[False, False, False])


def test_completeness_guard_uses_same_scientific_support():
    frame = pd.DataFrame(
        {"candidate_arm": list(range(9)), "q": [0.5] * 6 + [0.0, 0.0, 0.0]}
    )
    leaf = frame["candidate_arm"].lt(6)
    attach_standardized(frame, ["q"], support_mask=leaf)

    assert frame["q_z"].isna().all()
    assert_standardized_present(frame, ["q"], support_mask=leaf)


def test_materialized_z_must_match_frozen_scientific_support():
    """Correct names cannot hide moments computed on the wrong candidate rows."""
    raw = pd.DataFrame(
        {
            "candidate_arm": list(range(9)),
            "uncertainty": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0],
        }
    )
    leaf = raw["candidate_arm"].lt(6)
    current, reference = attach_standardized(
        raw.copy(),
        ["uncertainty"],
        support_mask=leaf,
        support_label="leaf_arms",
    )
    assert_standardized_matches_reference(
        current,
        ["uncertainty"],
        reference,
        support_label="leaf_arms",
    )

    stale, _ = attach_standardized(raw.copy(), ["uncertainty"])
    with pytest.raises(ValueError, match="do not match.*leaf_arms"):
        assert_standardized_matches_reference(
            stale,
            ["uncertainty"],
            reference,
            support_label="leaf_arms",
        )


def test_materialized_z_guard_rejects_missing_raw_or_z_column():
    reference = {
        "uncertainty": {
            "mean": 0.0,
            "std": 1.0,
            "support": "leaf_arms",
            "n_reference_rows": 2,
        }
    }
    with pytest.raises(ValueError, match="missing raw/standardized"):
        assert_standardized_matches_reference(
            pd.DataFrame({"uncertainty_z": [0.0, 1.0]}),
            ["uncertainty"],
            reference,
            support_label="leaf_arms",
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_zero_variance_can_be_centered_instead():
    """``zero_std="center"`` leaves a constant column as zeros, not NaN.

    A per-session design matrix pools one fit over several predictors: NaN-ing a
    within-session constant predictor would discard that session's other
    coefficients too, so its column is centered to all-zero (unidentified,
    coefficient 0) instead.
    """
    df = pd.DataFrame({"c": [4.0, 4.0, 4.0], "v": [0.0, 1.0, 2.0]})
    out, ref = attach_standardized(df.copy(), ["c", "v"], zero_std="center")
    np.testing.assert_array_equal(out["c_z"].to_numpy(), np.zeros(3))
    assert ref["c"]["std"] == 0.0
    np.testing.assert_allclose(
        out["v_z"].to_numpy(), (df["v"] - 1.0) / np.nanstd(df["v"])
    )
    with pytest.raises(ValueError, match="zero_std"):
        attach_standardized(df.copy(), ["c"], zero_std="drop")


def test_ddof_selects_the_sample_sd():
    """``ddof=1`` reports per sample SD; the default stays the population SD."""
    rng = np.random.default_rng(3)
    values = rng.normal(size=40)
    df = pd.DataFrame({"a": values})
    _, population = attach_standardized(df.copy(), ["a"])
    _, sample = attach_standardized(df.copy(), ["a"], ddof=1)
    assert population["a"]["std"] == float(np.nanstd(values))
    assert sample["a"]["std"] == float(np.nanstd(values, ddof=1))
    assert sample["a"]["std"] > population["a"]["std"]


def test_runner_references_written(tmp_path):
    """Both rate runners write their z reference beside the model they fit.

    Exercises each runner's real output-writing block, so deleting its
    ``save_reference`` call fails here rather than passing silently.
    """
    from scripts.coupling import fit_swr_power_model
    from scripts.rate import fit_uncertainty_rate

    frame = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0]})
    _, flat_reference = attach_standardized(
        frame.copy(), ["a"], support_label=fit_uncertainty_rate.FIT_SUPPORT_LABEL
    )
    empty_audit = pd.DataFrame({"animal": ["chimi"]})

    uncertainty_out = fit_uncertainty_rate._write_outputs(
        tmp_path / "uncertainty",
        models=frame.copy(),
        z_reference=flat_reference,
        gate_audit=empty_audit,
        cohort_records=[{"model": "MU_uncertainty", "n_in": 1}],
        fit_inputs={},
        argv=["fit_uncertainty_rate.py", "--max-per-animal", "1"],
        smoke=True,
        stop_cohort="at_well",
    )
    assert uncertainty_out.name == "uncertainty_rate_smoke.pkl"
    sidecar = z_reference_path(uncertainty_out)
    assert sidecar.name == "uncertainty_rate_smoke.z_reference.json"
    for moments in load_reference(sidecar).values():
        assert moments["support"] == "fit_cohort_all_rows"

    # The power model standardizes inside a per-session loop, so its payload is
    # one reference per session under the by-session filename.
    _, session_reference = attach_standardized(
        frame.copy(), ["a"], support_label=fit_swr_power_model.FIT_SUPPORT_LABEL
    )
    power_out = fit_swr_power_model._write_outputs(
        tmp_path / "power",
        coefs=frame.copy(),
        z_reference={"chimi|20200212|02_r1": session_reference},
        gate_audit=empty_audit,
        cohort_records=[
            {
                "model": "swr_power",
                "n_in": 1,
                "n_events_excluded_pre_outcome": 0,
                "n_events_excluded_outcome_imbalance": 0,
            }
        ],
        smoke=True,
    )
    assert power_out.name == "swr_power_model_smoke.pkl"
    session_sidecar = z_reference_path(power_out, by_session=True)
    assert session_sidecar.name == "swr_power_model_smoke.z_reference_by_session.json"
    by_session = load_reference(session_sidecar)
    assert set(by_session) == {"chimi|20200212|02_r1"}
    for moments in by_session["chimi|20200212|02_r1"].values():
        assert moments["support"] == "fit_cohort_all_rows"
