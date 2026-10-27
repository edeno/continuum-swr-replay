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
    assert row.peak_value == np.mean([2, 0.5, -0.5])
    assert row.sessions_same_sign == 2 / 3
    assert row.animals_same_sign == 1 and row.curve_n_animals == 2


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

    # animal a: mean(0.9, 0.7) = 0.8; animal b: 0.8 -> population 0.8
    assert out.animal_balanced_peak_value == pytest.approx(0.8)
    assert out.animal_balanced_peak_lag_ms == 0.0
    expected_low, expected_high = np.percentile(draws[:, 0], [2.5, 97.5])
    assert out.band_low_at_peak == pytest.approx(expected_low)
    assert out.band_high_at_peak == pytest.approx(expected_high)
    assert out.peak_outside_band
    assert out.animals_peak_outside_band == 2 and out.n_null_animals == 2
    assert out.animal_balanced_window_mean == pytest.approx(0.16)
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
