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
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "pfc"
    / "summarize_pfc_marked_family.py",
)
summarize = importlib.util.module_from_spec(_SPEC)
sys.modules["summarize_pfc_marked_family"] = summarize
_SPEC.loader.exec_module(summarize)


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


def test_the_figure_draws_the_shuffled_band_the_summary_reads(tmp_path, monkeypatch):
    lag = np.linspace(-0.2, 0.2, 5)
    key = ("m", "non_local_no_SWR", "t")
    bump = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    curves = {key: {"a1": bump, "b1": 0.5 * bump}}
    # a1 holds three draws (0, 1, 2 at every lag), b1 two (zeros): the band is
    # over the first two draws, whose population curves are 0 and 0.5.
    nulls = {
        key: {
            "a1": np.repeat(np.arange(3.0)[:, None], 5, axis=1),
            "b1": np.zeros((2, 5)),
        }
    }
    animal_of = {"a1": "a", "b1": "b"}
    close = summarize.plt.close
    figures = []
    monkeypatch.setattr(summarize.plt, "close", figures.append)

    summarize.plot_model("m", curves, lag, animal_of, tmp_path, nulls)

    band = figures[0].axes[0].collections[0].get_paths()[0].vertices[:, 1]
    assert band.min() == pytest.approx(np.percentile([0.0, 0.5], 2.5))
    assert band.max() == pytest.approx(np.percentile([0.0, 0.5], 97.5))
    close(figures[0])
