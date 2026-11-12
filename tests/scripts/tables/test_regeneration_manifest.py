"""Promotion-manifest tests for staged model-validation bundles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.tables.write_regeneration_manifest import _validate_model_bundle


def _fit_rows() -> pd.DataFrame:
    rows = []
    for animal, date, epoch, n_events in (
        ("a", 20200101, "02_r1", 12),
        ("b", 20200102, "02_r1", 15),
    ):
        for model, offset in (("M0", 0.0), ("M1", 1.0)):
            rows.append(
                {
                    "animal": animal,
                    "session_date": date,
                    "epoch": epoch,
                    "model": model,
                    "n_events": n_events,
                    "aic": 10.0 + offset,
                    "bic": 11.0 + offset,
                    "llf": -4.0 - offset,
                    "pseudo_r2": 0.1 + offset / 10,
                    "converged": True,
                }
            )
    return pd.DataFrame(rows)


def _write_bundle(directory: Path, results: pd.DataFrame) -> None:
    directory.mkdir()
    results.to_pickle(directory / "all_models.pkl")
    for model, frame in results.groupby("model", observed=True):
        frame.to_pickle(directory / f"{model}.pkl")
    pd.DataFrame(
        {
            "model": ["M0", "M1"],
            "n_sessions_modeled": [2, 2],
            "n_modeled": [27, 27],
        }
    ).to_csv(directory / "all_models_cohort_audit.csv", index=False)


def test_validate_model_bundle_requires_one_exact_common_session_block(
    tmp_path: Path,
) -> None:
    results = _fit_rows()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, results)
    keys = {("a", 20200101, "02_r1"), ("b", 20200102, "02_r1")}

    audit = _validate_model_bundle(bundle, expected_session_keys=keys)

    assert audit["sessions_modeled"] == 2
    assert audit["sessions_excluded"] == []
    assert audit["events_modeled_per_model"] == {"M0": 27, "M1": 27}


def test_a_bundle_holding_an_old_model_comparison_file_still_validates(
    tmp_path: Path,
) -> None:
    results = _fit_rows()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, results)
    pd.DataFrame({"parent": ["M0"], "child": ["M1"]}).to_pickle(
        bundle / "model_comparison.pkl"
    )
    keys = {("a", 20200101, "02_r1"), ("b", 20200102, "02_r1")}

    audit = _validate_model_bundle(bundle, expected_session_keys=keys)

    assert audit["models"] == ["M0", "M1"]


def test_validate_model_bundle_rejects_rung_specific_session_loss(
    tmp_path: Path,
) -> None:
    results = _fit_rows()
    results = results.loc[~(results["model"].eq("M1") & results["animal"].eq("b"))]
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, results)
    pd.DataFrame(
        {
            "model": ["M0", "M1"],
            "n_sessions_modeled": [2, 1],
            "n_modeled": [27, 12],
        }
    ).to_csv(bundle / "all_models_cohort_audit.csv", index=False)

    with pytest.raises(ValueError, match="exact session block"):
        _validate_model_bundle(
            bundle,
            expected_session_keys={
                ("a", 20200101, "02_r1"),
                ("b", 20200102, "02_r1"),
            },
        )


@pytest.mark.parametrize("flag", [False, np.nan])
def test_validate_model_bundle_rejects_a_fit_that_did_not_converge_or_has_no_flag(
    tmp_path: Path, flag: object
) -> None:
    results = _fit_rows()
    results["converged"] = results["converged"].astype(object)
    results.loc[0, "converged"] = flag
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, results)

    with pytest.raises(ValueError, match="745204e8"):
        _validate_model_bundle(
            bundle,
            expected_session_keys={
                ("a", 20200101, "02_r1"),
                ("b", 20200102, "02_r1"),
            },
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_validate_model_bundle_rejects_nonfinite_fit_statistics(
    tmp_path: Path, bad_value: float
) -> None:
    results = _fit_rows()
    results.loc[0, "aic"] = bad_value
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, results)

    with pytest.raises(ValueError, match="non-finite aic"):
        _validate_model_bundle(
            bundle,
            expected_session_keys={
                ("a", 20200101, "02_r1"),
                ("b", 20200102, "02_r1"),
            },
        )


# ---------------------------------------------------------------------------
# Baseline parity gate (_verify_baseline)
# ---------------------------------------------------------------------------
from continuum_swr_replay.tables.io import write_table  # noqa: E402
from scripts.tables.write_regeneration_manifest import _verify_baseline  # noqa: E402


def _write_replay_bundle(
    directory: Path, *, reward_fraction: float, n_events: int = 4
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    event = pd.DataFrame(
        {
            "event_id": np.arange(n_events, dtype=np.int64),
            "peak": np.linspace(0.5, 0.9, n_events),
        }
    )
    arm = pd.DataFrame(
        {
            "event_id": np.repeat(np.arange(n_events, dtype=np.int64), 2),
            "candidate_arm": np.tile([0, 1], n_events),
            "reward_fraction_candidate_arm_last10_visits": np.full(
                n_events * 2, reward_fraction
            ),
        }
    )
    stop = pd.DataFrame(
        {
            "stop_id": np.arange(n_events, dtype=np.int64),
            "immobility_time": np.full(n_events, 1.0),
        }
    )
    write_table(event, directory / "event_table")
    write_table(arm, directory / "event_arm_table")
    write_table(stop, directory / "stop_table")


def test_verify_baseline_passes_on_identical(tmp_path: Path):
    _write_replay_bundle(tmp_path / "base", reward_fraction=0.5)
    _write_replay_bundle(tmp_path / "cand", reward_fraction=0.5)
    report = _verify_baseline(
        tmp_path / "cand", tmp_path / "base", expected_drift_columns=set()
    )
    assert all(drift == {} for drift in report.values())


def test_verify_baseline_raises_on_undeclared_drift(tmp_path: Path):
    _write_replay_bundle(tmp_path / "base", reward_fraction=0.5)
    _write_replay_bundle(tmp_path / "cand", reward_fraction=0.7)  # regressed column
    with pytest.raises(ValueError, match="undeclared"):
        _verify_baseline(
            tmp_path / "cand", tmp_path / "base", expected_drift_columns=set()
        )


def test_verify_baseline_passes_when_drift_declared(tmp_path: Path):
    _write_replay_bundle(tmp_path / "base", reward_fraction=0.5)
    _write_replay_bundle(tmp_path / "cand", reward_fraction=0.7)
    report = _verify_baseline(
        tmp_path / "cand",
        tmp_path / "base",
        expected_drift_columns={"reward_fraction_candidate_arm_last10_visits"},
    )
    assert report["event_arm_table"]["reward_fraction_candidate_arm_last10_visits"][
        "max_abs_diff"
    ] == pytest.approx(0.2)


def test_verify_baseline_raises_on_structural_drift(tmp_path: Path):
    _write_replay_bundle(tmp_path / "base", reward_fraction=0.5, n_events=4)
    _write_replay_bundle(
        tmp_path / "cand", reward_fraction=0.5, n_events=5
    )  # key set differs
    with pytest.raises(ValueError, match="row"):
        _verify_baseline(
            tmp_path / "cand", tmp_path / "base", expected_drift_columns=set()
        )


# ---------------------------------------------------------------------------
# One SWR source per release (_require_one_swr_source)
# ---------------------------------------------------------------------------
from continuum_swr_replay.analysis.events.swr_source import (  # noqa: E402
    swr_source_name,
)
from scripts.tables.write_regeneration_manifest import (  # noqa: E402
    _require_one_swr_source,
)


def _table(path: Path, table: str, swr_source: str | None) -> None:
    frame = pd.DataFrame({"x": [1]})
    frame.attrs["covariate_schema_table"] = table
    if swr_source is not None:
        frame.attrs["swr_source"] = swr_source
    write_table(frame, path)


def test_a_release_built_from_the_configured_swr_source_passes(tmp_path: Path):
    roots = {"replay": tmp_path / "replay", "contfrag": tmp_path / "contfrag"}
    _table(roots["replay"] / "event_table.parquet", "event", swr_source_name())
    _table(roots["contfrag"] / "stop_table_swr_HPC.parquet", "stop", swr_source_name())
    _table(roots["replay"] / "qvals_table.parquet", "qvals", None)

    assert _require_one_swr_source(roots) == swr_source_name()


@pytest.mark.parametrize("stamp", [None, "kay@earlier+artifacts@none"])
def test_a_release_with_a_table_from_another_swr_source_is_refused(
    tmp_path: Path, stamp
):
    roots = {"replay": tmp_path / "replay", "contfrag": tmp_path / "contfrag"}
    _table(roots["replay"] / "event_table.parquet", "event", swr_source_name())
    _table(roots["contfrag"] / "event_table_swr_HPC.parquet", "event", stamp)

    with pytest.raises(ValueError, match="event_table_swr_HPC"):
        _require_one_swr_source(roots)


def test_the_file_inventory_skips_a_symlink_at_an_old_path(tmp_path: Path):
    from scripts.tables.write_regeneration_manifest import _inventory_files

    root = tmp_path / "replay"
    (root / "models_per_bin_rate").mkdir(parents=True)
    (root / "qvals_standardization_reference.json").write_text("{}")
    (root / "models_per_bin_rate" / "qvals_standardization_reference.json").symlink_to(
        root / "qvals_standardization_reference.json"
    )

    files = _inventory_files({"replay": root}, output=tmp_path / "manifest.json")

    assert [record["path"] for record in files["replay"]] == [
        "qvals_standardization_reference.json"
    ]


def test_the_release_leaves_out_underscore_folders(tmp_path: Path):
    """Backups, golden baselines, staging runs and caches sit in ``_`` folders
    below a root; the inventory and the SWR-source check leave them out."""
    from scripts.tables.write_regeneration_manifest import _inventory_files

    root = tmp_path / "replay"
    (root / "models").mkdir(parents=True)
    (root / "models" / "fit.json").write_text("{}")
    _table(root / "event_table.parquet", "event", swr_source_name())
    for folder in ("_pre_v5_regen_backup_20260829", "_golden_baseline_v5"):
        _table(root / folder / "event_table.parquet", "event", "kay@earlier")
        (root / folder / "standardization_reference.json").write_text("{}")

    files = _inventory_files({"replay": root}, output=tmp_path / "manifest.json")

    assert [record["path"] for record in files["replay"]] == [
        "event_table.parquet",
        "models/fit.json",
    ]
    assert _require_one_swr_source({"replay": root}) == swr_source_name()
