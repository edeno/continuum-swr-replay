"""Aggregation rules of the marked-family population summary."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "summarize_pfc_marked_family",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "pfc"
    / "summarize_pfc_marked_family.py",
)
summarize = importlib.util.module_from_spec(_SPEC)
sys.modules["summarize_pfc_marked_family"] = summarize
_SPEC.loader.exec_module(summarize)


def _units() -> pd.DataFrame:
    """Two animals; animal a has two sessions, animal b one, with a blow-up."""
    rows = []
    for animal, session, deltas in (
        ("a", "a1", [1.0, -1.0, 2.0, np.nan]),  # 2/3 positive over finite
        ("a", "a2", [-1.0, -2.0, 3.0, -4.0]),  # 1/4 positive
        ("b", "b1", [5.0, 5000.0, -1.0, 1.0]),  # blow-up excluded -> 2/3
    ):
        for delta in deltas:
            rows.append(
                {
                    "animal": animal,
                    "session": session,
                    "model": "m",
                    "event_class": "non_local_no_SWR",
                    "target": "t",
                    "delta": delta,
                    "curve_finite": True,
                }
            )
    return pd.DataFrame(rows)


def test_fraction_positive_is_animal_balanced_and_excludes_nan_and_blowups():
    out = summarize.animal_balanced_fraction_positive(_units(), blowup=1000.0)
    row = out.iloc[0]
    assert row.n_units == 12
    assert row.n_usable == 10
    assert row.n_excluded_blowup == 1
    assert row.n_sessions == 3 and row.n_animals == 2
    # animal a: median(2/3, 1/4) = 11/24; animal b: 2/3; animal-balanced median
    expected_a = np.median([2 / 3, 1 / 4])
    assert row.frac_pos == np.median([expected_a, 2 / 3])
    assert row.frac_pos_min == expected_a and row.frac_pos_max == 2 / 3


def test_population_curve_peak_and_sign_consistency():
    lag = np.linspace(-0.2, 0.2, 5)
    bump = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    curves = {
        ("m", "non_local_no_SWR", "t"): {
            "a1": 2 * bump,
            "a2": 0.5 * bump,
            "b1": -0.5 * bump,
        }
    }
    out = summarize.population_curve_summary(
        curves, lag, {"a1": "a", "a2": "a", "b1": "b"}
    )
    row = out.iloc[0]
    assert row.peak_lag_ms == 0.0
    # unit -> session -> animal: animal a's sessions average first.
    assert row.peak_value == np.mean([np.mean([2, 0.5]), -0.5])
    assert row.sem_at_peak == pytest.approx(np.std([1.25, -0.5], ddof=1) / np.sqrt(2))
    assert row.sessions_same_sign == 2 / 3
    assert row.animals_same_sign == 1 and row.curve_n_animals == 2
    # Lags 0.0 and 0.1 s fall in the 0 to +150 ms window: animal a's mean there
    # is +0.625, b's -0.25, the population's +0.1875.
    assert row.post_onset_animals_same_sign == 1


def test_post_onset_sign_count_reads_the_fixed_window_not_the_peak():
    """Animals agree at the peak lag but not over 0 to +150 ms: the post-onset
    count must not inherit the peak's agreement."""
    lag = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    curves = {
        ("m", "non_local_no_SWR", "t"): {
            # Both animals peak at +0.2 s (outside the window) with one sign...
            "a1": np.array([0.0, 0.0, 0.3, 0.3, 2.0]),
            # ...but b is negative inside the window.
            "b1": np.array([0.0, 0.0, -0.2, -0.2, 1.0]),
        }
    }
    row = summarize.population_curve_summary(curves, lag, {"a1": "a", "b1": "b"}).iloc[
        0
    ]
    assert row.peak_lag_ms == pytest.approx(200.0)
    assert row.animals_same_sign == 2
    assert row.post_onset_mean > 0
    assert row.post_onset_animals_same_sign == 1


def test_summary_band_and_peak_call():
    lag = np.linspace(-0.2, 0.2, 5)
    bump = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    key = ("m", "non_local_no_SWR", "t")
    # Draws r = 0..99 put r/100 - 0.495 at every lag: the band is known.
    draws = (np.arange(100) / 100 - 0.495)[:, None] * np.ones(5)
    curves = {key: {"a1": 0.9 * bump, "a2": 0.7 * bump, "b1": 0.8 * bump}}
    nulls = {key: {"a1": draws, "a2": draws, "b1": draws}}
    animal_of = {"a1": "a", "a2": "a", "b1": "b"}

    out = summarize.null_band_summary(curves, nulls, lag, animal_of).iloc[0]

    expected_low, expected_high = np.percentile(draws[:, 0], [2.5, 97.5])
    assert out.band_low_at_peak == pytest.approx(expected_low)
    assert out.band_high_at_peak == pytest.approx(expected_high)
    assert out.peak_outside_band
    assert out.animals_peak_outside_band == 2 and out.n_null_animals == 2
    # animal a: mean(0.9, 0.7) = 0.8; animal b: 0.8 -> population 0.8 at the
    # peak lag, 0.16 over the window.
    assert out.window_mean == pytest.approx(0.16)
    # Max statistic: every draw's largest |curve| is |r/100 - 0.495|, at most
    # 0.505, so the observed 0.8 exceeds all 100 draws.
    assert out.max_abs_value == pytest.approx(0.8)
    assert out.max_abs_null_p95 == pytest.approx(np.percentile(np.abs(draws[:, 0]), 95))
    assert out.max_outside_reference
    assert out.max_p_value == pytest.approx(1 / 101)
    assert not out.window_outside_band
    assert out.n_draws == 100

    inside = summarize.null_band_summary(
        {key: dict.fromkeys(animal_of, 0.3 * bump)}, nulls, lag, animal_of
    ).iloc[0]
    assert not inside.peak_outside_band


def test_summary_pairs_joint_and_marginal_for_each_target():
    merged = pd.DataFrame(
        {
            "model": ["joint_model", "joint_model", "marginal_t"],
            "tier": ["joint", "joint", "marginal"],
            "event_class": ["non_local_no_SWR", "non_local_SWR", "non_local_no_SWR"],
            "target": ["t", "t", "t"],
            "post_onset_mean": [0.1, 0.2, 0.3],
            "frac_pos": [0.5, 0.6, 0.7],
        }
    )

    pairs = summarize.joint_versus_marginal(merged)

    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row.event_class == "non_local_no_SWR"
    assert (row.joint_post_onset_mean, row.marginal_post_onset_mean) == (0.1, 0.3)
    assert (row.joint_frac_pos, row.marginal_frac_pos) == (0.5, 0.7)


def test_summary_table_is_one_row_per_model_tier_class_target_statistic():
    keys = {"model": ["m"], "event_class": ["pooled"], "target": ["t"]}
    fractions = pd.DataFrame({**keys, "frac_pos": [0.6], "n_units": [10]})
    peaks = pd.DataFrame({**keys, "peak_value": [0.2]})
    bands = pd.DataFrame({**keys, "peak_outside_band": [True]})
    tiers = pd.DataFrame({"model": ["m"], "tier": ["joint"]})

    table = summarize.summary_table(fractions, peaks, bands, tiers)

    assert list(table.columns) == [
        "model",
        "tier",
        "event_class",
        "target",
        "statistic",
        "value",
    ]
    assert not table.duplicated(
        ["model", "tier", "event_class", "target", "statistic"]
    ).any()
    values = table.set_index("statistic")["value"]
    assert values["frac_pos"] == 0.6
    assert values["peak_outside_band"] == 1.0


def _write_bundle(directory, session, *, version, event_set="canonical"):
    """One complete fake bundle: sidecar, statistics, lag axis and curves."""
    base = directory / f"{session}_mPFC_m_marked_family"
    lag = np.linspace(-0.2, 0.2, 5)
    pd.DataFrame({"non_local_no_SWR:t: heldout_delta_deviance": [1.0, -1.0]}).to_pickle(
        base.with_suffix(".pkl")
    )
    np.save(base.with_suffix(".lag_time.npy"), lag)
    np.save(base.with_suffix(".non_local_no_SWR.t.slope.npy"), np.ones((5, 2)))
    np.save(
        base.with_suffix(".non_local_no_SWR.t.slope_null.npy"),
        np.zeros((3, 5, 2), np.float32),
    )
    nwb, epoch = session.split("_", 1)
    record = {
        "bundle_status": "complete",
        "bundle_contract_version": version,
        "event_set": event_set,
        "model_name": "m",
        "tier": "joint",
        "target_classes": ["non_local_no_SWR"],
        "nwb_file_name": f"{nwb}_.nwb",
        "epoch_name": epoch,
        "marked_family_diagnostics": {"scientific_marks": ["t"]},
    }
    (directory / f"{base.name}.provenance.json").write_text(json.dumps(record))


def test_summary_refuses_mixed_contracts(tmp_path):
    _write_bundle(tmp_path, "j1620210710_08_r4", version=2)
    _write_bundle(tmp_path, "j1620210710_12_r6", version=2)

    units, curves, nulls, _ = summarize.load_bundles(tmp_path)
    assert set(units.session) == {"j1620210710_08_r4", "j1620210710_12_r6"}
    assert nulls[("m", "non_local_no_SWR", "t")]["j1620210710_08_r4"].shape == (3, 5)

    _write_bundle(tmp_path, "j1620210711_02_r1", version=1)
    with pytest.raises(summarize.MixedContractError, match="mixes bundle contracts"):
        summarize.load_bundles(tmp_path)
    (tmp_path / "j1620210711_02_r1_mPFC_m_marked_family.provenance.json").unlink()
    _write_bundle(
        tmp_path, "j1620210711_02_r1", version=2, event_set="with_swr_without_replay"
    )
    with pytest.raises(summarize.MixedContractError, match="event sets"):
        summarize.load_bundles(tmp_path)


def test_observed_curve_and_reference_share_the_animal_weighting():
    """Two sessions of animal a and one of b: every quantity weights animals equally."""
    lag = np.linspace(-0.2, 0.2, 5)
    bump = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    key = ("m", "non_local_no_SWR", "t")
    animal_of = {"a1": "a", "a2": "a", "b1": "b"}
    curves = {key: {"a1": 0.6 * bump, "a2": 0.6 * bump, "b1": 0.0 * bump}}
    # Each session's draws are constant; the population draw is 0.3 when the
    # animals are weighted equally and 0.4 when the sessions are.
    nulls = {
        key: {
            "a1": np.full((20, 5), 0.6),
            "a2": np.full((20, 5), 0.6),
            "b1": np.zeros((20, 5)),
        }
    }

    peaks = summarize.population_curve_summary(curves, lag, animal_of).iloc[0]
    band = summarize.null_band_summary(curves, nulls, lag, animal_of).iloc[0]

    assert peaks.peak_value == pytest.approx(0.3)
    assert band.band_low_at_peak == pytest.approx(0.3)
    assert band.band_high_at_peak == pytest.approx(0.3)
    assert band.max_abs_value == pytest.approx(peaks.peak_value)
    assert band.max_abs_null_p95 == pytest.approx(0.3)
    marginal = {("marginal_t", "non_local_no_SWR", "t"): curves[key]}
    marginal_peaks = summarize.population_curve_summary(marginal, lag, animal_of)
    assert marginal_peaks.iloc[0].post_onset_mean == pytest.approx(
        peaks.post_onset_mean
    )


def test_summary_refuses_mixed_model_sets(tmp_path):
    _write_bundle(tmp_path, "j1620210710_08_r4", version=2)
    sidecar = tmp_path / "j1620210710_08_r4_mPFC_m_marked_family.provenance.json"
    record = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps({**record, "model_set": "followup"}))
    _write_bundle(tmp_path, "j1620210710_12_r6", version=2)
    other = tmp_path / "j1620210710_12_r6_mPFC_m_marked_family.provenance.json"
    other.write_text(
        json.dumps({**json.loads(other.read_text()), "model_set": "recommended"})
    )

    with pytest.raises(summarize.MixedContractError, match="model sets"):
        summarize.load_bundles(tmp_path)


def test_max_statistic_call_is_the_plus_one_p_not_the_interpolated_percentile():
    lag = np.linspace(-0.2, 0.2, 5)
    key = ("m", "non_local_no_SWR", "t")
    animal_of = {"a1": "a"}
    # 20 draws whose largest |curve| is 0.05, 0.10, ..., 1.00.
    draws = (np.arange(1, 21) / 20)[:, None] * np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    nulls = {key: {"a1": draws}}

    def call(observed_peak):
        curves = {key: {"a1": observed_peak * np.array([0.0, 0.0, 1.0, 0.0, 0.0])}}
        return summarize.null_band_summary(curves, nulls, lag, animal_of).iloc[0]

    # 0.97 is above the interpolated 95th percentile (0.9525) but one draw
    # (1.00) is at least as large: p = 2/21, inside.
    near = call(0.97)
    assert near.max_abs_value > near.max_abs_null_p95
    assert near.max_p_value == pytest.approx(2 / 21)
    assert not near.max_outside_reference
    # Above every draw: p = 1/21 <= 0.05, outside.
    above = call(1.2)
    assert above.max_p_value == pytest.approx(1 / 21)
    assert above.max_outside_reference


def test_mark_correlations_are_animal_balanced(tmp_path):
    """Two chimi sessions at r = 0.2 and 0.4 and one wilbur session at 0.9
    give the mean of the animal means, 0.6, not the session mean."""

    def geometry(r):
        return {
            "non_local_no_SWR": {
                "marks": ["a", "b"],
                "correlation": [[1.0, r], [r, 1.0]],
            }
        }

    for session, r in (
        ("chimi20200212_02_r1", 0.2),
        ("chimi20200212_04_r2", 0.4),
        ("wilbur20210329_02_r1", 0.9),
    ):
        _write_bundle(tmp_path, session, version=2)
        path = tmp_path / f"{session}_mPFC_m_marked_family.provenance.json"
        record = json.loads(path.read_text())
        record["marked_family_diagnostics"]["event_mark_geometry_by_class"] = geometry(
            r
        )
        path.write_text(json.dumps(record))

    table = summarize.mark_correlations(tmp_path)

    assert table["target"].tolist() == ["a & b"]
    assert table["statistic"].tolist() == ["mark_correlation"]
    assert table["value"].iloc[0] == pytest.approx(0.6)
    assert table["n_sessions"].iloc[0] == 3


def test_a_class_dropped_from_a_session_is_skipped(tmp_path):
    """A target class the session dropped (too few events) has no columns."""
    _write_bundle(tmp_path, "chimi20200212_02_r1", version=2)
    path = tmp_path / "chimi20200212_02_r1_mPFC_m_marked_family.provenance.json"
    record = json.loads(path.read_text())
    record["target_classes"] = ["non_local_no_SWR", "SWR_no_non_local"]
    record["fitted_event_classes"] = ["non_local_no_SWR"]
    path.write_text(json.dumps(record))

    units, _, _, _ = summarize.load_bundles(tmp_path)

    assert set(units["event_class"]) == {"non_local_no_SWR"}


def test_a_pooled_target_class_is_kept(tmp_path):
    """The pooled model's target class is not an event class; it stays."""
    _write_bundle(tmp_path, "chimi20200212_02_r1", version=2)
    base = tmp_path / "chimi20200212_02_r1_mPFC_m_marked_family"
    pd.DataFrame({"pooled:t: heldout_delta_deviance": [1.0, -1.0]}).to_pickle(
        base.with_suffix(".pkl")
    )
    np.save(base.with_suffix(".pooled.t.slope.npy"), np.ones((5, 2)))
    path = base.with_name(base.name + ".provenance.json")
    record = json.loads(path.read_text())
    record["target_classes"] = ["pooled"]
    record["fitted_event_classes"] = ["non_local_SWR", "non_local_no_SWR"]
    path.write_text(json.dumps(record))

    units, _, _, _ = summarize.load_bundles(tmp_path)

    assert set(units["event_class"]) == {"pooled"}
