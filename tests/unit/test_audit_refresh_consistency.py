"""The refresh's same-data audit and its tables manifest.

Nothing here reaches the Spyglass database, a GPU or a real ``Processed-Data``
tree. Every artifact is made up under the test's ``tmp_path``, for made-up
epochs (``sim20200101_*``), and so is the epoch register the audit reads
(``--register``); ``get_epoch_info`` is replaced by a frame of named epochs. Every git check runs in a throwaway repository
there, and ``check_encoding_marks`` reads its marks and encoding models through
fakes. ``check_fit_provenance`` runs the repository's real
``scripts/qa/check_fit_provenance.py`` on roots under ``tmp_path``; that script
only reads.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from non_local_detector.discrete_state_transitions import DiscreteStationaryCustom

import scripts.qa.audit_refresh_consistency as audit
from continuum_swr_replay.analysis.hippocampus.swr_source import swr_source_name
from continuum_swr_replay.data_loaders.epoch_register import read_register
from continuum_swr_replay.decoding_transitions import (
    contfrag_transition_settings,
    nonlocal_transition_settings,
)
from continuum_swr_replay.table_io import sha256_file, write_table

BASE_TIME_S = 1.58e9
"""Unix-like timestamps, as the decodes carry them, so nothing can assume time 0."""

FS = 500.0
EPOCHS = ("sim20200101_02_r1_HPC", "sim20200101_04_r2_HPC")
NO_INPUT_EPOCH = "sim20200101_06_r3_HPC"
NO_RIPPLE_BAND = "no LFPBandV1 ripple band"
SPIKE_LOADER = "src/continuum_swr_replay/data_loaders/spikes.py"


REGISTER_COLUMNS = ["nwb_file_name", "epoch", "status", "reason", "evidence"]
DEPLETION = "a depletion session"


def _key(epoch_id: str) -> tuple[str, str]:
    """``sim20200101_02_r1_HPC`` -> ``("sim20200101_.nwb", "02_r1")``."""
    stem, number, replicate, _ = epoch_id.split("_")
    return f"{stem}_.nwb", f"{number}_{replicate}"


def _write_register(
    path: Path,
    *,
    analysed: tuple[str, ...] = EPOCHS,
    pending: tuple[str, ...] = (NO_INPUT_EPOCH,),
    excluded: tuple[str, ...] = (),
) -> Path:
    """An epoch register: `pending` waits on a populate, `excluded` is depletion."""
    rows = [[*_key(epoch_id), "analysed", "", ""] for epoch_id in analysed]
    rows += [
        [*_key(epoch_id), "pending", "PENDING_POPULATE", NO_RIPPLE_BAND]
        for epoch_id in pending
    ]
    rows += [
        [*_key(epoch_id), "excluded", "DEPLETION_TASK", DEPLETION]
        for epoch_id in excluded
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=REGISTER_COLUMNS).to_csv(path, index=False)
    return path


def _register(tmp_path: Path, **statuses: tuple[str, ...]) -> pd.DataFrame:
    return read_register(_write_register(tmp_path / "epoch_register.csv", **statuses))


def _epoch_info(*epoch_ids: str) -> pd.DataFrame:
    """``get_epoch_info``'s index for these epochs."""
    return pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [_key(epoch_id) for epoch_id in epoch_ids],
            names=["nwb_file_name", "interval_list_name"],
        )
    )


# --------------------------------------------------------------------------
# Made-up artifacts
# --------------------------------------------------------------------------


def _provenance(commit: str = "a" * 40, **overrides: object) -> dict:
    """A decode's provenance record, as ``provenance_record`` writes it."""
    record = {
        "commit": commit,
        "dirty": False,
        "uv_lock_sha256": "1" * 64,
        "ripple_detection_commit": "68e1784b183f161a568d3154325cf770730bfb83",
        "non_local_detector_version": "0.6.10.dev330+g2d730c3a8",
        "cuda_plugin_version": "0.8.0",
    }
    record.update(overrides)
    return record


def _write_results(path: Path, provenance: dict | None) -> None:
    """A small ``results.nc``, stamped as a fresh save stamps it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        {"acausal_posterior": ("time", np.zeros(3))},
        coords={"time": BASE_TIME_S + np.arange(3) / FS},
    )
    if provenance is not None:
        dataset.attrs["provenance"] = json.dumps(provenance, sort_keys=True)
    dataset.to_netcdf(path, engine="h5netcdf")


def _write_log(path: Path, provenance: dict, *, resumed: bool = False) -> None:
    """A decode log in ``setup_logger``'s format, with its PROVENANCE line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "03-Oct-26 15:46:42 - sim - INFO     - "
    lines = [
        prefix + " START ".center(50, "#"),
        prefix + "PROVENANCE " + json.dumps(provenance, sort_keys=True),
        prefix + "Loading data...",
    ]
    if resumed:
        lines.append(prefix + "RESUMED from existing results in /staging")
    path.write_text("\n".join(lines) + "\n")


def _stamp(
    root: Path,
    log_dir: Path,
    decode_id: str,
    results: dict | None,
    log: dict | None,
    *,
    resumed: bool = False,
) -> None:
    """Write one decode's ``results.nc`` and log."""
    _write_results(root / f"{decode_id}_results.nc", results)
    if log is not None:
        _write_log(log_dir / f"{decode_id}.log", log, resumed=resumed)


def _provenance_frame(
    tmp_path: Path, overrides: dict[str, dict] | None = None
) -> pd.DataFrame:
    """``decode_provenance`` over both decoders of both epochs, all fresh saves.

    Each decode is stamped with ``_provenance()``, updated by
    ``overrides[decode_id]``, in its results and its log alike.
    """
    root, log_dir = tmp_path / "decodes", tmp_path / "logs"
    for epoch_id in EPOCHS:
        for decode_id in (epoch_id, f"{epoch_id}_contfrag"):
            record = _provenance() | (overrides or {}).get(decode_id, {})
            _stamp(root, log_dir, decode_id, record, record)
    return audit.decode_provenance(log_dir, root, list(EPOCHS))


def _dot(
    n: int = 600, step: float = 1 / FS, start: float = BASE_TIME_S
) -> pd.DataFrame:
    """A ``data_over_time`` frame: a time index, position and speed."""
    time = start + np.arange(n) * step
    return pd.DataFrame(
        {
            "linear_position": np.linspace(0.0, 120.0, n),
            "head_speed": 10.0 * np.abs(np.sin(np.arange(n) / 50.0)),
        },
        index=pd.Index(time, name="time"),
    )


@dataclass
class FakeTransition:
    """Stands in for a ``non_local_detector`` discrete transition type."""

    values: np.ndarray


def _detector(**overrides: object) -> SimpleNamespace:
    """A picklable stand-in for a fitted detector's discrete-transition settings."""
    settings: dict[str, object] = {
        "discrete_transition_type": FakeTransition(
            values=np.array([[0.99, 0.01], [0.02, 0.98]])
        ),
        "discrete_transition_stickiness": np.array([1e5, 300.0]),
        "discrete_transition_concentration": 1.0,
        "discrete_transition_prior_weight": 0.0,
        "discrete_transition_regularization": 1e-10,
        "frozen_discrete_transition_rows": (1,),
        "local_position_std": 2.5,
    }
    settings.update(overrides)
    return SimpleNamespace(**settings)


def _pinned_detector(decoder: str, **overrides: object) -> SimpleNamespace:
    """A stand-in for a fitted `decoder` detector, built as the decoders build it.

    It carries the pinned transition settings, the NonLocal Local state's
    2.5 cm (None for ContFrag, which has no Local state) and the library's
    ``discrete_transition_regularization`` of 1e-10, which no decoder sets.
    """
    if decoder == "nonlocal":
        settings = {**nonlocal_transition_settings(), "local_position_std": 2.5}
    else:
        settings = {**contfrag_transition_settings(), "local_position_std": None}
    settings["discrete_transition_regularization"] = 1e-10
    settings.update(overrides)
    return SimpleNamespace(**settings)


def _git_environment() -> dict[str, str]:
    """The environment without GIT_* variables, so git acts only on its cwd."""
    return {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }


def _repo(path: Path) -> Callable[..., str]:
    """A new, empty git repository at `path`; return a git runner for it."""
    path.mkdir(parents=True)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            env=_git_environment(),
        ).stdout.strip()

    git("init", "-q")
    return git


def _commit(git: Callable[..., str], repo: Path, files: dict[str, str]) -> str:
    """Write `files` (path relative to `repo` to text), commit them, return the sha."""
    for relative, text in files.items():
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text(text)
    git("add", "-A")
    git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "change",
    )
    return git("rev-parse", "HEAD")


STOP_IDS = np.array([8393993873049741010, 4684575748763187657], dtype=np.uint64)
"""One stop per made-up epoch, as uint64 hashes like the real ``stop_id``."""


def _sessions() -> pd.DataFrame:
    """A table holding both made-up epochs, one stop each, as the stop table does."""
    return pd.DataFrame(
        {
            "animal": ["sim", "sim"],
            "session_date": [20200101, 20200101],
            "epoch": ["02_r1", "04_r2"],
            "stop_id": STOP_IDS,
            "value": [1.0, 2.0],
        }
    )


def _qvals(stop_ids: np.ndarray = STOP_IDS) -> pd.DataFrame:
    """A qvals table with the real one's keys: ``stop_id`` and ``epoch`` only."""
    epochs = dict(zip(STOP_IDS, ["02_r1", "04_r2"], strict=True))
    return pd.DataFrame(
        {
            "stop_id": np.asarray(stop_ids, dtype=np.uint64),
            "epoch": [epochs.get(stop_id, "02_r1") for stop_id in stop_ids],
            "q_arm": np.linspace(0.0, 1.0, len(stop_ids)),
        }
    )


def _fit(directory: Path, name: str, inputs: dict[str, Path]) -> None:
    """A fitted artifact and its provenance sidecar, current for `inputs`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.pkl").write_bytes(b"fit")
    record = {
        "artifact": f"{name}.pkl",
        "inputs": {
            input_name: {"sha256": sha256_file(path)}
            for input_name, path in inputs.items()
        },
    }
    (directory / f"{name}.provenance.json").write_text(json.dumps(record))


def _snapshot(tree: Path, leave_out: Path) -> dict[str, tuple[int, int]]:
    """Every file under `tree` but `leave_out`, with its size and mtime."""
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(tree.rglob("*"))
        if path.is_file() and leave_out not in path.parents and path != leave_out
    }


# --------------------------------------------------------------------------
# decode_provenance
# --------------------------------------------------------------------------


def test_decode_provenance_prefers_results_nc_then_the_log(tmp_path):
    root, log_dir = tmp_path / "decodes", tmp_path / "logs"
    epoch_id = EPOCHS[0]
    _stamp(root, log_dir, epoch_id, _provenance("a" * 40), _provenance("a" * 40))
    _stamp(root, log_dir, f"{epoch_id}_contfrag", None, _provenance("c" * 40))

    frame = audit.decode_provenance(log_dir, root, [epoch_id]).set_index("decoder")

    assert frame.loc["nonlocal", "source"] == "results.nc"
    assert frame.loc["nonlocal", "commit"] == "a" * 40
    assert frame.loc["nonlocal", "decode_id"] == epoch_id
    assert frame.loc["contfrag", "source"] == "log"
    assert frame.loc["contfrag", "commit"] == "c" * 40
    assert frame.loc["contfrag", "decode_id"] == f"{epoch_id}_contfrag"
    assert not frame["resumed"].any()


def test_a_resumed_log_is_not_read_as_the_decode_provenance(tmp_path):
    root, log_dir = tmp_path / "decodes", tmp_path / "logs"
    epoch_id = EPOCHS[0]
    _stamp(root, log_dir, epoch_id, None, _provenance("b" * 40), resumed=True)
    _stamp(root, log_dir, f"{epoch_id}_contfrag", _provenance(), _provenance())

    frame = audit.decode_provenance(log_dir, root, [epoch_id])
    row = frame.set_index("decoder").loc["nonlocal"]

    # the resume rewrote the log, so its PROVENANCE line describes the resume
    assert row["resumed"]
    assert row["source"] is None
    assert row["commit"] is None
    result = audit.check_one_commit(frame, {})
    assert not result["ok"]
    assert any(
        "RESUMED" in failure and epoch_id in failure for failure in result["failures"]
    )


# --------------------------------------------------------------------------
# check_one_commit
# --------------------------------------------------------------------------


def test_one_commit_passes_and_names_the_commit(tmp_path):
    result = audit.check_one_commit(_provenance_frame(tmp_path), {})

    assert result["ok"], result["failures"]
    assert result["details"]["commit"] == "a" * 40


def test_two_commits_fail_without_an_exception(tmp_path):
    late = f"{EPOCHS[1]}_contfrag"
    frame = _provenance_frame(tmp_path, {late: {"commit": "f" * 40}})

    result = audit.check_one_commit(frame, {})

    assert not result["ok"]
    assert result["details"]["commit"] is None
    assert any(
        "commit" in failure and late in failure for failure in result["failures"]
    )


def test_a_dirty_row_fails_without_an_exception(tmp_path):
    frame = _provenance_frame(tmp_path, {EPOCHS[0]: {"dirty": True}})

    result = audit.check_one_commit(frame, {})

    assert not result["ok"]
    assert any(
        "dirty" in failure and EPOCHS[0] in failure for failure in result["failures"]
    )


def test_a_listed_exception_with_its_reason_passes(tmp_path):
    late = f"{EPOCHS[1]}_contfrag"
    frame = _provenance_frame(tmp_path, {late: {"commit": "f" * 40, "dirty": True}})
    reason = "redecoded after the batch with the loader fix; reviewed"

    result = audit.check_one_commit(frame, {late: reason})

    assert result["ok"], result["failures"]
    assert result["details"]["commit"] == "a" * 40
    assert result["details"]["exceptions"][late]["reason"] == reason


def test_another_uv_lock_or_a_missing_cuda_plugin_fails(tmp_path):
    frame = _provenance_frame(
        tmp_path,
        {
            EPOCHS[0]: {"uv_lock_sha256": "2" * 64},
            EPOCHS[1]: {"cuda_plugin_version": None},
        },
    )

    failures = audit.check_one_commit(frame, {})["failures"]

    assert any("uv_lock_sha256" in failure for failure in failures)
    assert any(
        "cuda_plugin_version" in failure and EPOCHS[1] in failure
        for failure in failures
    )


def test_a_resume_at_another_commit_fails(tmp_path):
    root, log_dir = tmp_path / "decodes", tmp_path / "logs"
    for decode_id in (EPOCHS[0], f"{EPOCHS[0]}_contfrag"):
        _stamp(root, log_dir, decode_id, _provenance(), _provenance())
    # the decode was saved at "a"*40, then resumed at another commit, which
    # rewrote data_over_time
    _stamp(root, log_dir, EPOCHS[0], _provenance(), _provenance("e" * 40), resumed=True)

    result = audit.check_one_commit(
        audit.decode_provenance(log_dir, root, [EPOCHS[0]]), {}
    )

    assert not result["ok"]
    assert any(
        "e" * 12 in failure and EPOCHS[0] in failure for failure in result["failures"]
    )


@pytest.mark.parametrize("reason", ["", None])
def test_an_exception_without_a_reason_is_refused(tmp_path, reason):
    late = f"{EPOCHS[1]}_contfrag"
    frame = _provenance_frame(tmp_path, {late: {"commit": "f" * 40}})

    result = audit.check_one_commit(frame, {late: reason})

    assert not result["ok"]
    assert (
        f"the exception for {late} has no reason, so it is not applied"
        in (result["failures"])
    )
    # and the row is checked as if it were not listed
    assert any(
        failure.startswith("2 values of commit") for failure in result["failures"]
    )
    assert result["details"]["exceptions"] == {}


def test_an_absent_decode_is_reported_with_its_epoch_sets_reason(tmp_path):
    root, log_dir = tmp_path / "decodes", tmp_path / "logs"
    _stamp(root, log_dir, EPOCHS[0], _provenance(), _provenance())
    # the ContFrag decode was never saved; its log is from a run that failed
    _write_log(log_dir / f"{EPOCHS[0]}_contfrag.log", _provenance("f" * 40))
    frame = audit.decode_provenance(log_dir, root, [EPOCHS[0]])
    row = frame.set_index("decoder").loc["contfrag"]
    assert not row["saved"]
    assert row["source"] is None

    unexplained = audit.check_one_commit(frame, {})
    reason = "ContFrag decode failed: no HPC spikes in the position span"
    explained = audit.check_one_commit(
        frame, {}, explained={f"{EPOCHS[0]}_contfrag": reason}
    )

    assert unexplained["failures"] == [
        f"{EPOCHS[0]}_contfrag: no results.nc, so the decode was never saved and has "
        "no provenance; epoch_sets records no reason for its absence"
    ]
    assert explained["ok"], explained["failures"]
    assert explained["details"]["absent"] == {
        f"{EPOCHS[0]}_contfrag": f"explained in epoch_sets: {reason}"
    }
    assert explained["details"]["commit"] == "a" * 40


def test_a_decode_without_provenance_fails(tmp_path):
    root, log_dir = tmp_path / "decodes", tmp_path / "logs"
    _stamp(root, log_dir, EPOCHS[0], _provenance(), _provenance())
    _write_results(root / f"{EPOCHS[0]}_contfrag_results.nc", None)  # no log either

    result = audit.check_one_commit(
        audit.decode_provenance(log_dir, root, [EPOCHS[0]]), {}
    )

    assert not result["ok"]
    assert any(f"{EPOCHS[0]}_contfrag" in failure for failure in result["failures"])


# --------------------------------------------------------------------------
# check_grids_match
# --------------------------------------------------------------------------


def test_identical_grids_pass():
    result = audit.check_grids_match(_dot(), _dot())

    assert result["ok"], result["failures"]


def test_a_one_sample_shifted_contfrag_grid_fails():
    shifted = _dot(start=BASE_TIME_S + 1 / FS)

    result = audit.check_grids_match(_dot(), shifted)

    assert not result["ok"]
    assert any("time" in failure for failure in result["failures"])


def test_a_differing_speed_or_position_fails():
    contfrag = _dot()
    contfrag.iloc[10, contfrag.columns.get_loc("head_speed")] += 0.5
    contfrag.iloc[20, contfrag.columns.get_loc("linear_position")] = np.nan

    failures = audit.check_grids_match(_dot(), contfrag)["failures"]

    assert any("head_speed" in failure for failure in failures)
    assert any("linear_position" in failure for failure in failures)


def test_a_grid_with_a_gap_fails_validate_time_grid():
    gapped = _dot().drop(index=_dot().index[100:110])

    result = audit.check_grids_match(gapped, gapped.copy())

    assert not result["ok"]
    assert any("validate_time_grid" in failure for failure in result["failures"])


# --------------------------------------------------------------------------
# check_minimum_sample_counts
# --------------------------------------------------------------------------


def test_one_minimum_sample_count_passes():
    times = {epoch_id: _dot().index.to_numpy() for epoch_id in EPOCHS}

    result = audit.check_minimum_sample_counts(times)

    assert result["ok"], result["failures"]
    # 15 ms is 7.5 samples at 500 Hz; on absolute timestamps the median float64
    # step is 2.0000935 ms, so the count rounds to 7 (decision R9)
    assert result["details"]["per_epoch"] == dict.fromkeys(EPOCHS, 7)


def test_minimum_sample_counts_of_8_and_7_fail():
    times = {
        EPOCHS[0]: _dot().index.to_numpy(),
        EPOCHS[1]: _dot(step=1 / 533.0).index.to_numpy(),
    }

    result = audit.check_minimum_sample_counts(times)

    assert result["details"]["per_epoch"] == {EPOCHS[0]: 7, EPOCHS[1]: 8}
    assert not result["ok"]


# --------------------------------------------------------------------------
# check_widloski_hashes
# --------------------------------------------------------------------------


def _widloski_manifest(root: Path, epochs: tuple[str, ...], run_commit: str) -> dict:
    """A Widloski run manifest whose input hashes are the files' as written now."""
    records = {}
    for epoch_id in epochs:
        inputs = {
            "detector": root / f"{epoch_id}_detector.pkl",
            "data_over_time": root / f"{epoch_id}_80_data_over_time.pkl",
        }
        records[epoch_id] = {
            "status": "ok",
            "git_commit": run_commit,
            "inputs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in inputs.items()
            },
        }
    return {
        "run_id": "sim-run",
        "settings": {"git_commit": run_commit},
        "epochs": records,
    }


def _write_detector_and_dot(root: Path, epoch_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(_detector(), root / f"{epoch_id}_detector.pkl")
    _dot().to_pickle(root / f"{epoch_id}_80_data_over_time.pkl")


def test_unchanged_decode_files_pass_the_widloski_hash_check(tmp_path):
    for epoch_id in EPOCHS:
        _write_detector_and_dot(tmp_path, epoch_id)
    manifest = _widloski_manifest(tmp_path, EPOCHS, "b" * 40)

    result = audit.check_widloski_hashes(manifest, tmp_path)

    assert result["ok"], result["failures"]
    assert result["details"]["n_files"] == 4


def test_an_edited_decode_file_fails_the_widloski_hash_check(tmp_path):
    for epoch_id in EPOCHS:
        _write_detector_and_dot(tmp_path, epoch_id)
    manifest = _widloski_manifest(tmp_path, EPOCHS, "b" * 40)
    edited = _dot()
    edited.iloc[0, 0] += 1.0
    edited.to_pickle(tmp_path / f"{EPOCHS[1]}_80_data_over_time.pkl")

    result = audit.check_widloski_hashes(manifest, tmp_path)

    assert not result["ok"]
    assert result["failures"] == [
        f"{EPOCHS[1]}: data_over_time "
        f"{tmp_path / f'{EPOCHS[1]}_80_data_over_time.pkl'} is not the file the "
        "run read: its sha256 differs from the recorded one"
    ]


# --------------------------------------------------------------------------
# check_swr_spans
# --------------------------------------------------------------------------


def _swr_manifest(
    spans: dict[str, tuple[float, float]],
    run_commit: str = "b" * 40,
    skipped: dict[str, str] | None = None,
) -> dict:
    """An SWR run manifest: ``ok`` records with spans, and skipped ones."""
    records: dict[str, dict] = {
        epoch_id: {
            "status": "ok",
            "position_span_s": list(span),
            "provenance": _provenance(run_commit),
        }
        for epoch_id, span in spans.items()
    }
    records.update(
        {
            epoch_id: {"status": "skipped", "reason": reason}
            for epoch_id, reason in (skipped or {}).items()
        }
    )
    return {
        "run_id": "sim-run",
        "settings": {"speed": "head_speed of get_epoch_position_info"},
        "epochs": records,
    }


def test_matching_swr_spans_pass():
    spans = dict.fromkeys(EPOCHS, (BASE_TIME_S, BASE_TIME_S + 600.0))

    result = audit.check_swr_spans(_swr_manifest(spans), spans)

    assert result["ok"], result["failures"]
    assert result["details"]["n_compared"] == 2


def test_a_span_off_by_10_ms_fails():
    spans = dict.fromkeys(EPOCHS, (BASE_TIME_S, BASE_TIME_S + 600.0))
    swr_spans = dict(spans)
    swr_spans[EPOCHS[0]] = (BASE_TIME_S, BASE_TIME_S + 600.01)

    result = audit.check_swr_spans(_swr_manifest(swr_spans), spans)

    assert not result["ok"]
    assert len(result["failures"]) == 1
    assert EPOCHS[0] in result["failures"][0]


# --------------------------------------------------------------------------
# record_tables and check_tables_recorded
# --------------------------------------------------------------------------


def _stamped(
    frame: pd.DataFrame, table: str, source: str | None = None
) -> pd.DataFrame:
    """`frame` as a canonical `table` built from an SWR source (the configured one
    by default), as ``publish_canonical`` writes it."""
    frame.attrs["covariate_schema_table"] = table
    frame.attrs["swr_source"] = swr_source_name() if source is None else source
    return frame


def _table_dir(path: Path, swr_source: str | None = None) -> Path:
    """A promoted table folder, with the files a manifest must and must not hold."""
    write_table(
        _stamped(_sessions(), "event", swr_source), path / "event_table.parquet"
    )
    write_table(_stamped(_sessions(), "stop", swr_source), path / "stop_table.parquet")
    (path / "standardization_reference.json").write_text("{}")
    write_table(_qvals(), path / "models_per_bin_rate" / "qvals_table.parquet")
    (path / "models_per_bin_rate" / "qvals_standardization_reference.json").write_text(
        "{}"
    )
    # fits and their sidecars change when models are refit, so they are not tables
    _fit(path, "stage6_is_stay_logistic", {"stop_table": path / "stop_table.parquet"})
    _fit(
        path / "models_per_bin_rate",
        "all_models",
        {"stop_table": path / "stop_table.parquet"},
    )
    (path / "models_per_bin_rate" / "swr_contrast_rate_meta.json").write_text("{}")
    (path / "stop_table.pkl.bak").write_bytes(b"old")
    return path


@dataclass
class History:
    """A throwaway repository's commits, in order."""

    repo: Path
    git: Callable[..., str]
    base: str
    """Before the decode commit."""
    decode: str
    """The decode commit, which adds the spike loader."""
    later: str
    """Descends from the decode commit; the spike loader is unchanged."""


def _history(tmp_path: Path) -> History:
    repo = tmp_path / "repo"
    git = _repo(repo)
    base = _commit(git, repo, {"README": "base"})
    decode = _commit(git, repo, {SPIKE_LOADER: "loader", "README": "decode"})
    later = _commit(git, repo, {"README": "later"})
    return History(repo=repo, git=git, base=base, decode=decode, later=later)


def test_record_tables_records_tables_but_not_fits(tmp_path):
    table_dir = _table_dir(tmp_path / "replay_content")

    path = audit.record_tables(table_dir, "c" * 40, {"decode_commit": "a" * 40})

    manifest = json.loads(path.read_text())
    assert path == table_dir / "tables_manifest.json"
    assert sorted(manifest["files"]) == [
        "event_table.parquet",
        "models_per_bin_rate/qvals_standardization_reference.json",
        "models_per_bin_rate/qvals_table.parquet",
        "standardization_reference.json",
        "stop_table.parquet",
    ]
    assert manifest["files"]["event_table.parquet"] == sha256_file(
        table_dir / "event_table.parquet"
    )
    assert manifest["commit"] == "c" * 40
    assert manifest["inputs"] == {
        "decode_commit": "a" * 40,
        "swr_source": swr_source_name(),
    }


def test_record_tables_records_the_swr_source_the_tables_carry(tmp_path):
    table_dir = _table_dir(tmp_path / "replay_content", "kay@earlier+artifacts@none")

    path = audit.record_tables(table_dir, "c" * 40, {"decode_commit": "a" * 40})

    assert json.loads(path.read_text())["inputs"]["swr_source"] == (
        "kay@earlier+artifacts@none"
    )
    with pytest.raises(ValueError, match="stamped"):
        audit.record_tables(
            table_dir,
            "c" * 40,
            {"decode_commit": "a" * 40, "swr_source": swr_source_name()},
        )


def test_record_tables_refuses_tables_from_two_swr_sources(tmp_path):
    table_dir = _table_dir(tmp_path / "replay_content")
    write_table(
        _stamped(_sessions(), "stop", "kay@earlier+artifacts@none"),
        table_dir / "stop_table.parquet",
    )

    with pytest.raises(ValueError, match="one SWR source"):
        audit.record_tables(table_dir, "c" * 40, {"decode_commit": "a" * 40})


def test_tables_built_from_another_swr_source_fail(tmp_path):
    history = _history(tmp_path)
    table_dir = _table_dir(tmp_path / "replay_content", "kay@earlier+artifacts@none")
    audit.record_tables(table_dir, history.later, {"decode_commit": history.decode})

    failures = audit.check_tables_recorded(
        [table_dir], history.decode, repo=history.repo
    )["failures"]

    assert failures == [
        f"{table_dir}: its tables were built from SWR source "
        f"'kay@earlier+artifacts@none', not the configured {swr_source_name()!r}"
    ]


def test_recorded_tables_built_after_the_decode_commit_pass(tmp_path):
    history = _history(tmp_path)
    table_dir = _table_dir(tmp_path / "replay_content")
    audit.record_tables(table_dir, history.later, {"decode_commit": history.decode})

    result = audit.check_tables_recorded([table_dir], history.decode, repo=history.repo)

    assert result["ok"], result["failures"]


def test_a_table_changed_after_record_tables_fails(tmp_path):
    history = _history(tmp_path)
    table_dir = _table_dir(tmp_path / "replay_content")
    audit.record_tables(table_dir, history.later, {"decode_commit": history.decode})
    changed = _sessions()
    changed.loc[0, "value"] = 5.0
    write_table(changed, table_dir / "stop_table.parquet")

    result = audit.check_tables_recorded([table_dir], history.decode, repo=history.repo)

    assert not result["ok"]
    assert result["failures"] == [
        f"{table_dir}: stop_table.parquet changed after it was recorded"
    ]


def test_a_table_added_or_removed_after_record_tables_fails(tmp_path):
    history = _history(tmp_path)
    table_dir = _table_dir(tmp_path / "replay_content")
    audit.record_tables(table_dir, history.later, {"decode_commit": history.decode})
    write_table(_sessions(), table_dir / "event_patch_table.parquet")
    (table_dir / "standardization_reference.json").unlink()

    failures = audit.check_tables_recorded(
        [table_dir], history.decode, repo=history.repo
    )["failures"]

    assert any("event_patch_table.parquet" in failure for failure in failures)
    assert any("standardization_reference.json" in failure for failure in failures)


def test_tables_not_built_from_the_decode_commit_fail(tmp_path):
    history = _history(tmp_path)
    replay = _table_dir(tmp_path / "replay_content")
    contfrag = _table_dir(tmp_path / "contfrag_replay_content")
    audit.record_tables(replay, history.later, {"decode_commit": history.decode})
    # built by code older than the decode commit
    audit.record_tables(contfrag, history.base, {"decode_commit": history.decode})

    failures = audit.check_tables_recorded(
        [replay, contfrag], history.decode, repo=history.repo
    )["failures"]

    assert len(failures) == 1
    assert failures[0].startswith(f"{contfrag}: built at {history.base}")
    assert "does not descend" in failures[0]

    # recorded as built from another decode
    audit.record_tables(replay, history.later, {"decode_commit": history.base})
    failures = audit.check_tables_recorded([replay], history.decode, repo=history.repo)[
        "failures"
    ]
    assert len(failures) == 1
    assert "decode_commit" in failures[0]


def test_record_tables_command_writes_only_its_manifest(tmp_path):
    table_dir = _table_dir(tmp_path / "replay_content")
    before = _snapshot(tmp_path, table_dir / "tables_manifest.json")

    code = audit.main(
        [
            "record-tables",
            str(table_dir),
            "--commit",
            "c" * 40,
            "--inputs",
            json.dumps({"decode_commit": "a" * 40, "behavior_cache": "cache"}),
        ]
    )

    assert code == 0
    assert _snapshot(tmp_path, table_dir / "tables_manifest.json") == before
    manifest = json.loads((table_dir / "tables_manifest.json").read_text())
    assert manifest["inputs"] == {
        "decode_commit": "a" * 40,
        "behavior_cache": "cache",
        "swr_source": swr_source_name(),
    }


def test_tables_with_an_unknown_or_missing_commit_fail(tmp_path):
    history = _history(tmp_path)
    table_dir = _table_dir(tmp_path / "replay_content")

    audit.record_tables(table_dir, "f" * 40, {"decode_commit": history.decode})
    unknown = audit.check_tables_recorded(
        [table_dir], history.decode, repo=history.repo
    )["failures"]
    audit.record_tables(table_dir, "", {"decode_commit": history.decode})
    missing = audit.check_tables_recorded(
        [table_dir], history.decode, repo=history.repo
    )["failures"]

    # git merge-base exits 128 on a commit it does not know: neither 0 nor 1
    assert len(unknown) == 1
    assert unknown[0].startswith(
        f"{table_dir}: git could not tell whether {'f' * 40} descends from "
        f"{history.decode}: fatal"
    )
    assert missing == [f"{table_dir}: the manifest records no commit"]


def test_tables_checked_outside_a_git_repository_fail(tmp_path):
    history = _history(tmp_path)
    table_dir = _table_dir(tmp_path / "replay_content")
    audit.record_tables(table_dir, history.later, {"decode_commit": history.decode})
    not_a_repository = tmp_path / "not_a_repository"
    not_a_repository.mkdir()

    failures = audit.check_tables_recorded(
        [table_dir], history.decode, repo=not_a_repository
    )["failures"]

    assert len(failures) == 1
    assert "git could not tell" in failures[0]


# --------------------------------------------------------------------------
# check_pfc_sidecars
# --------------------------------------------------------------------------


def _pfc_sidecar(epoch_name: str, event_table_sha256: str) -> dict:
    return {
        "artifact": f"sim20200101_{epoch_name}_mPFC_modulation.pkl",
        "nwb_file_name": "sim20200101_.nwb",
        "epoch_name": epoch_name,
        "canonical_event_table_sha256": event_table_sha256,
    }


def test_pfc_sidecars_naming_the_event_table_pass():
    sidecars = [_pfc_sidecar("02_r1", "e" * 64), _pfc_sidecar("04_r2", "e" * 64)]

    result = audit.check_pfc_sidecars(sidecars, "e" * 64)

    assert result["ok"], result["failures"]


def test_an_mpfc_sidecar_naming_an_older_event_table_fails():
    sidecars = [_pfc_sidecar("02_r1", "e" * 64), _pfc_sidecar("04_r2", "d" * 64)]

    result = audit.check_pfc_sidecars(sidecars, "e" * 64)

    assert not result["ok"]
    assert "sim20200101_04_r2_mPFC_modulation.pkl" in result["failures"][0]
    assert "02_r1" not in result["failures"][0]


# --------------------------------------------------------------------------
# check_epoch_sets
# --------------------------------------------------------------------------


def test_an_analysed_epoch_an_artifact_lacks_fails_unless_recorded(tmp_path):
    register = _register(tmp_path)
    span = (BASE_TIME_S, BASE_TIME_S + 600.0)
    swr_manifest = _swr_manifest({EPOCHS[0]: span})
    decodes = set(EPOCHS)

    present, named = audit.run_manifest_epochs(swr_manifest)
    result = audit.check_epoch_sets(
        {"nonlocal_decode": decodes, "swr": present}, {}, register=register, named=named
    )

    assert result["failures"] == [
        f"swr lacks 1 analysed epoch(s) with no recorded reason: {EPOCHS[1]}"
    ]

    # neither a failed nor a skipped record explains it
    for record in (
        {"status": "failed", "error": "ValueError()"},
        {"status": "skipped", "reason": NO_RIPPLE_BAND},
    ):
        swr_manifest["epochs"][EPOCHS[1]] = record
        present, named = audit.run_manifest_epochs(swr_manifest)
        result = audit.check_epoch_sets(
            {"nonlocal_decode": decodes, "swr": present},
            {},
            register=register,
            named=named,
        )
        assert not result["ok"]

    # a reason recorded for that artifact (--exceptions) does
    result = audit.check_epoch_sets(
        {"nonlocal_decode": decodes, "swr": present},
        {"swr": {EPOCHS[1]: "recorded"}},
        register=register,
        named=named,
    )
    assert result["ok"], result["failures"]
    assert result["details"]["swr"]["explained"] == {EPOCHS[1]: "recorded"}


def test_an_analysed_epoch_no_artifact_holds_fails_every_artifact(tmp_path):
    unheld = "sim20200101_08_r4_HPC"
    register = _register(tmp_path, analysed=(*EPOCHS, unheld))
    sets = {"nonlocal_decode": set(EPOCHS), "swr": set(EPOCHS)}

    result = audit.check_epoch_sets(sets, {}, register=register)

    assert result["failures"] == [
        f"{name} lacks 1 analysed epoch(s) with no recorded reason: {unheld}"
        for name in sets
    ]


def test_an_epoch_the_register_does_not_analyse_must_not_be_held(tmp_path):
    depleted = "sim20200101_10_r5_HPC"
    register = _register(tmp_path, excluded=(depleted,))
    sets = {
        "nonlocal_decode": {*EPOCHS, NO_INPUT_EPOCH},
        "replay_content/event_table.parquet": {*EPOCHS, depleted},
    }

    result = audit.check_epoch_sets(sets, {}, register=register)

    assert result["failures"] == [
        "nonlocal_decode holds 1 epoch(s) the register does not analyse: "
        f"{NO_INPUT_EPOCH} (pending, PENDING_POPULATE)",
        "replay_content/event_table.parquet holds 1 epoch(s) the register does not "
        f"analyse: {depleted} (excluded, DEPLETION_TASK)",
    ]
    assert result["details"]["nonlocal_decode"]["not_analysed"] == {
        NO_INPUT_EPOCH: f"pending, PENDING_POPULATE: {NO_RIPPLE_BAND}"
    }


def test_an_unregistered_epoch_fails_wherever_it_is_named(tmp_path):
    register = _register(tmp_path)
    stray = "sim20200101_12_r6_HPC"

    held = audit.check_epoch_sets(
        {"nonlocal_decode": {*EPOCHS, stray}}, {}, register=register
    )
    named = audit.check_epoch_sets(
        {"nonlocal_decode": set(EPOCHS)}, {}, register=register, named={stray}
    )

    for result in (held, named):
        assert (
            f"1 epoch(s) are not in the epoch register: {stray}" in (result["failures"])
        )


def test_a_skipped_record_of_an_unanalysed_epoch_passes(tmp_path):
    span = (BASE_TIME_S, BASE_TIME_S + 600.0)
    swr_manifest = _swr_manifest(
        dict.fromkeys(EPOCHS, span), skipped={NO_INPUT_EPOCH: NO_RIPPLE_BAND}
    )
    present, named = audit.run_manifest_epochs(swr_manifest)

    result = audit.check_epoch_sets(
        {"nonlocal_decode": set(EPOCHS), "swr": present},
        {},
        register=_register(tmp_path),
        named=named,
    )

    assert named == {*EPOCHS, NO_INPUT_EPOCH}
    assert result["ok"], result["failures"]


def _touch_decodes(root: Path, epoch_id: str, decoders=("nonlocal", "contfrag")):
    """Empty stand-ins for every file of `epoch_id`'s decodes under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    for decoder in decoders:
        for path in audit.decode_files(epoch_id, root, decoder).values():
            path.touch()


def test_a_decode_folder_of_analysed_epochs_alone_passes(tmp_path):
    root = tmp_path / "decodes"
    for epoch_id in EPOCHS:
        _touch_decodes(root, epoch_id)
    # a variant decode is not a canonical one, so it is not the folder's concern
    (root / f"{NO_INPUT_EPOCH}_all_detector.pkl").touch()

    result = audit.check_decode_folder(root, _register(tmp_path))

    assert result["ok"], result["failures"]
    assert result["details"]["n_epochs"] == 2


def test_any_decode_file_of_an_unanalysed_epoch_fails_the_decode_folder(tmp_path):
    """Even one file of a partly saved decode: a reader that globs the folder
    would read it."""
    root = tmp_path / "decodes"
    for epoch_id in EPOCHS:
        _touch_decodes(root, epoch_id)
    stray = "sim20200101_12_r6_HPC"
    (root / f"{NO_INPUT_EPOCH}_results.nc").touch()  # a pending epoch's
    (root / f"{stray}_contfrag_detector.pkl").touch()  # an unregistered one's

    result = audit.check_decode_folder(root, _register(tmp_path))

    assert result["failures"] == [
        "the decode folder holds files of 1 epoch(s) the register does not "
        f"analyse: {NO_INPUT_EPOCH} (pending, PENDING_POPULATE)",
        f"the decode folder holds files of 1 epoch(s) not in the epoch register: "
        f"{stray}",
    ]


# --------------------------------------------------------------------------
# check_epochs_registered and check_pending_epochs
# --------------------------------------------------------------------------


def test_every_epoch_spyglass_lists_is_registered(tmp_path):
    register = _register(tmp_path)

    ok = audit.check_epochs_registered(lambda: _epoch_info(*EPOCHS), register)
    # a pending epoch Spyglass already lists is registered too
    listed = audit.check_epochs_registered(
        lambda: _epoch_info(*EPOCHS, NO_INPUT_EPOCH), register
    )

    assert ok["ok"] and listed["ok"], (ok["failures"], listed["failures"])
    assert listed["details"]["by_status"] == {"analysed": 2, "pending": 1}


def test_an_epoch_spyglass_lists_but_the_register_does_not_fails(tmp_path):
    stray = "sim20200101_12_r6_HPC"

    result = audit.check_epochs_registered(
        lambda: _epoch_info(*EPOCHS, stray), _register(tmp_path)
    )

    assert result["failures"] == [
        "get_epoch_info lists 1 epoch(s) the register does not: sim20200101_.nwb 12_r6"
    ]


def test_an_analysed_epoch_spyglass_no_longer_lists_fails(tmp_path):
    result = audit.check_epochs_registered(
        lambda: _epoch_info(EPOCHS[0]), _register(tmp_path)
    )

    assert result["failures"] == [
        "1 analysed epoch(s) are not in get_epoch_info: sim20200101_.nwb 04_r2"
    ]


def test_an_unreadable_get_epoch_info_fails(tmp_path):
    def unreachable():
        raise ConnectionError("no database")

    result = audit.check_epochs_registered(unreachable, _register(tmp_path))

    assert result["failures"] == [
        "get_epoch_info could not be read, so no epoch it lists was checked: "
        "ConnectionError('no database')"
    ]


def test_the_pending_epochs_are_reported(tmp_path):
    result = audit.check_pending_epochs(_register(tmp_path))

    assert result["ok"]
    assert result["details"] == {
        "n_pending": 1,
        "epochs": {NO_INPUT_EPOCH: f"PENDING_POPULATE: {NO_RIPPLE_BAND}"},
    }


QVALS = "replay_content/models_per_bin_rate/qvals_table.parquet"


def test_the_qvals_table_is_read_through_the_stop_table(tmp_path):
    table_dir = _table_dir(tmp_path / "replay_content")

    sets, failures, not_checked = audit.table_epoch_sets([table_dir])

    # the real qvals table carries stop_id and epoch, not animal or session_date
    assert sets[QVALS] == set(EPOCHS)
    assert sets["replay_content/stop_table.parquet"] == set(EPOCHS)
    assert failures == []
    assert not_checked == []


def test_a_session_missing_from_qvals_fails_the_epoch_sets(tmp_path):
    table_dir = _table_dir(tmp_path / "replay_content")
    qvals_path = table_dir / "models_per_bin_rate" / "qvals_table.parquet"
    write_table(_qvals(STOP_IDS[:1]), qvals_path)

    sets, failures, _ = audit.table_epoch_sets([table_dir])
    result = audit.check_epoch_sets(sets, {}, register=_register(tmp_path))

    assert failures == []
    assert result["failures"] == [
        f"{QVALS} lacks 1 analysed epoch(s) with no recorded reason: {EPOCHS[1]}"
    ]


def test_a_qvals_stop_the_stop_table_lacks_fails(tmp_path):
    table_dir = _table_dir(tmp_path / "replay_content")
    stray = np.append(STOP_IDS, np.uint64(7))
    write_table(
        _qvals(stray), table_dir / "models_per_bin_rate" / "qvals_table.parquet"
    )

    sets, failures, not_checked = audit.table_epoch_sets([table_dir])

    assert QVALS not in sets
    assert failures == [
        f"{QVALS}: 1 of its 3 stop_ids are not in the folder's stop table, so their "
        "epochs cannot be told"
    ]
    assert not_checked == []


def test_a_table_without_epoch_keys_is_listed_as_not_checked(tmp_path):
    table_dir = _table_dir(tmp_path / "replay_content")
    weights = pd.DataFrame({"arm": [0, 1], "weight": [0.5, 0.5]})
    write_table(weights, table_dir / "arm_weights.parquet")

    sets, failures, not_checked = audit.table_epoch_sets([table_dir])

    assert "replay_content/arm_weights.parquet" not in sets
    assert failures == []
    assert len(not_checked) == 1
    assert not_checked[0].startswith("epoch_sets: replay_content/arm_weights.parquet")


# --------------------------------------------------------------------------
# check_fit_provenance
# --------------------------------------------------------------------------


def test_fresh_fits_pass_and_a_stale_one_fails(tmp_path):
    root = tmp_path / "replay_content"
    write_table(_sessions(), root / "event_table.parquet")
    _fit(root / "models", "content", {"event_table": root / "event_table.parquet"})

    assert audit.check_fit_provenance(audit.REPO, [root])["ok"]

    changed = _sessions()
    changed.loc[1, "value"] = 9.0
    write_table(changed, root / "event_table.parquet")
    result = audit.check_fit_provenance(audit.REPO, [root])

    assert not result["ok"]
    assert any("content.pkl" in failure for failure in result["failures"])


def test_a_root_level_artifact_is_checked_against_its_sidecar(tmp_path):
    root = tmp_path / "replay_content"
    write_table(_sessions(), root / "stop_table.parquet")
    _fit(root / "models", "content", {"stop_table": root / "stop_table.parquet"})
    _fit(root, "stage5b_retrospective", {"stop_table": root / "stop_table.parquet"})
    (root / "stage5b_retrospective.provenance.json").write_text(
        json.dumps({"inputs": {"stop_table": {"sha256": "0" * 64}}})
    )

    result = audit.check_fit_provenance(
        audit.REPO, [root], artifacts=[root / "stage5b_retrospective.pkl"]
    )

    assert not result["ok"]
    assert any("stage5b_retrospective.pkl" in failure for failure in result["failures"])


def test_root_level_fits_without_a_sidecar_fail_and_absent_families_are_listed(
    tmp_path,
):
    root = tmp_path / "replay_content"
    write_table(_sessions(), root / "stop_table.parquet")
    _fit(root, "stage6_is_stay_logistic", {"stop_table": root / "stop_table.parquet"})
    (root / "stage5b_retrospective.pkl").write_bytes(b"fit")  # its sidecar is gone

    artifacts, not_checked = audit.root_fits(root)
    result = audit.check_fit_provenance(audit.REPO, [], artifacts=artifacts)

    assert artifacts == [
        root / "stage5b_retrospective.pkl",
        root / "stage6_is_stay_logistic.pkl",
    ]
    assert not_checked == [
        f"fit_provenance: no stop_replay_summary fit (stop_replay_summary*.pkl) at {root}"
    ]
    assert not result["ok"]
    assert any(
        "stage5b_retrospective.pkl (missing_sidecar)" in failure
        for failure in result["failures"]
    )


# --------------------------------------------------------------------------
# check_encoding_marks
# --------------------------------------------------------------------------


def _marks_fixture(tmp_path: Path) -> tuple[Callable, dict[Path, dict]]:
    """A marks loader and both decodes' encoding models, all made up."""
    epoch_id = EPOCHS[0]
    _dot().to_pickle(tmp_path / f"{epoch_id}_80_data_over_time.pkl")
    times = [
        BASE_TIME_S + np.array([0.1, 0.2, 0.3, 50.0]),  # the last is past the span
        BASE_TIME_S + np.array([0.15, 0.25]),
    ]
    features = [
        np.arange(16, dtype=np.float64).reshape(4, 4),
        100.0 + np.arange(8, dtype=np.float64).reshape(2, 4),
    ]

    def load_marks(nwb_file_name: str) -> tuple[list, list]:
        assert nwb_file_name == "sim20200101_.nwb"
        return times, features

    encoding = {
        "encoding_spike_waveform_features": [
            features[0][:2].astype(np.float32),
            features[1][1:].astype(np.float32),
        ]
    }
    models = {
        tmp_path / f"{epoch_id}_detector.pkl": encoding,
        tmp_path / f"{epoch_id}_contfrag_detector.pkl": {
            "encoding_spike_waveform_features": [
                array.copy() for array in encoding["encoding_spike_waveform_features"]
            ]
        },
    }
    return load_marks, models


def test_encoding_marks_among_the_reloaded_ones_pass(tmp_path):
    load_marks, models = _marks_fixture(tmp_path)

    result = audit.check_encoding_marks(
        EPOCHS[0],
        tmp_path,
        load_marks=load_marks,
        load_encoding_model=models.__getitem__,
    )

    assert result["ok"], result["failures"]
    assert result["details"]["n_marks"] == 5  # the spike past the span is dropped


def test_an_encoding_mark_the_reload_lacks_fails(tmp_path):
    load_marks, models = _marks_fixture(tmp_path)
    contfrag = models[tmp_path / f"{EPOCHS[0]}_contfrag_detector.pkl"]
    # the spike past the span: the reload over the decode's span cannot hold it
    contfrag["encoding_spike_waveform_features"][0] = np.arange(
        16, dtype=np.float32
    ).reshape(4, 4)[3:]

    result = audit.check_encoding_marks(
        EPOCHS[0],
        tmp_path,
        load_marks=load_marks,
        load_encoding_model=models.__getitem__,
    )

    assert not result["ok"]
    assert len(result["failures"]) == 1
    assert result["failures"][0].startswith("contfrag:")


# --------------------------------------------------------------------------
# check_spike_loader_unchanged
# --------------------------------------------------------------------------


def test_a_spike_loader_unchanged_since_the_decode_passes(tmp_path):
    history = _history(tmp_path)

    result = audit.check_spike_loader_unchanged(
        history.repo, history.decode, history.later
    )

    assert result["ok"], result["failures"]


def test_a_changed_spike_loader_fails(tmp_path):
    history = _history(tmp_path)
    run_commit = _commit(history.git, history.repo, {SPIKE_LOADER: "another loader"})

    result = audit.check_spike_loader_unchanged(
        history.repo, history.decode, run_commit
    )

    assert not result["ok"]
    assert SPIKE_LOADER in result["failures"][0]


def test_a_dirty_or_unknown_run_commit_fails(tmp_path):
    history = _history(tmp_path)

    dirty = audit.check_spike_loader_unchanged(
        history.repo, history.decode, f"{history.later}-dirty"
    )
    unknown = audit.check_spike_loader_unchanged(history.repo, history.decode, "f" * 40)

    assert not dirty["ok"]
    assert "dirty" in dirty["failures"][0]
    assert not unknown["ok"]


# --------------------------------------------------------------------------
# check_decoder_settings
# --------------------------------------------------------------------------


def test_identical_detectors_pass(tmp_path):
    paths = {}
    for epoch_id in EPOCHS:
        paths[epoch_id] = tmp_path / f"{epoch_id}_detector.pkl"
        joblib.dump(_detector(), paths[epoch_id])

    result = audit.check_decoder_settings(paths)

    assert result["ok"], result["failures"]
    assert result["details"]["settings"]["discrete_transition_stickiness"] == [
        1e5,
        300.0,
    ]
    assert result["details"]["settings"]["discrete_transition_type"] == {
        "class": "FakeTransition",
        "values": [[0.99, 0.01], [0.02, 0.98]],
    }


def test_a_detector_with_another_stickiness_fails(tmp_path):
    third = "sim20200102_02_r1_HPC"
    paths = {
        epoch_id: tmp_path / f"{epoch_id}_detector.pkl" for epoch_id in (*EPOCHS, third)
    }
    joblib.dump(_detector(), paths[EPOCHS[0]])
    joblib.dump(_detector(), paths[EPOCHS[1]])
    joblib.dump(
        _detector(discrete_transition_stickiness=np.array([1e6, 300.0])), paths[third]
    )

    result = audit.check_decoder_settings(paths)

    assert not result["ok"]
    assert len(result["failures"]) == 1
    assert third in result["failures"][0]
    assert "discrete_transition_stickiness" in result["failures"][0]
    assert EPOCHS[0] not in result["failures"][0]


def test_settings_other_than_expected_fail(tmp_path):
    paths = {EPOCHS[0]: tmp_path / "detector.pkl"}
    joblib.dump(_detector(), paths[EPOCHS[0]])

    assert audit.check_decoder_settings(
        paths,
        {"discrete_transition_stickiness": [100000, 300], "local_position_std": 2.5},
    )["ok"]
    result = audit.check_decoder_settings(
        paths,
        {
            "frozen_discrete_transition_rows": None,
            # a self-transition of 0.96 against 0.98, the size of the drift
            # between April's library defaults and today's
            "discrete_transition_type": {
                "class": "FakeTransition",
                "values": [[0.99, 0.01], [0.02, 0.96]],
            },
        },
    )
    assert not result["ok"]
    assert "frozen_discrete_transition_rows" in result["failures"][0]
    assert "discrete_transition_type" in result["failures"][0]
    # a misspelled setting is not silently left unchecked
    assert not audit.check_decoder_settings(paths, {"stickiness": [1e5, 300.0]})["ok"]


def test_the_default_expectation_is_the_pinned_settings():
    """``expected_decoder_settings`` is the code's pin, in the audit's JSON form."""
    expected = audit.expected_decoder_settings()

    pinned = {
        "nonlocal": nonlocal_transition_settings(),
        "contfrag": contfrag_transition_settings(),
    }
    assert set(expected) == set(audit.DECODERS)
    for decoder, settings in pinned.items():
        assert set(expected[decoder]) == set(audit.DISCRETE_SETTINGS)
        # the five pinned transition values, each exactly as _plain writes it
        assert {name: expected[decoder][name] for name in settings} == {
            name: audit._plain(value) for name, value in settings.items()
        }
    # JSON types throughout: a tuple or an array would not survive the round trip
    assert json.loads(json.dumps(expected)) == expected
    assert expected["nonlocal"]["discrete_transition_type"]["class"] == (
        "DiscreteStationaryCustom"
    )
    assert expected["nonlocal"]["frozen_discrete_transition_rows"] == [1]
    # the two settings the pin does not hold: the NonLocal decode's Local-state
    # width, and the library's regularization, which no decoder sets
    assert expected["nonlocal"]["local_position_std"] == 2.5
    assert expected["contfrag"]["local_position_std"] is None
    assert expected["nonlocal"]["discrete_transition_regularization"] == 1e-10
    assert expected["contfrag"]["discrete_transition_regularization"] == 1e-10


@pytest.mark.parametrize("decoder", audit.DECODERS)
def test_a_detector_with_the_pinned_settings_passes(tmp_path, decoder):
    path = tmp_path / f"{decoder}_detector.pkl"
    joblib.dump(_pinned_detector(decoder), path)

    result = audit.check_decoder_settings(
        {EPOCHS[0]: path}, audit.expected_decoder_settings()[decoder]
    )

    assert result["ok"], result["failures"]


def test_a_detector_with_the_library_nonlocal_defaults_fails(tmp_path):
    """The library's own NonLocal defaults, which the decoders took before the pin."""
    no_spike_exit = (1 - 0.98) / 3
    non_local_exit = (1 - 0.96 - 0.005) / 2
    library_defaults = DiscreteStationaryCustom(
        values=np.array(
            [
                [0.99, 0.005, 0.0025, 0.0025],
                [no_spike_exit, 0.98, no_spike_exit, no_spike_exit],
                [non_local_exit, 0.005, 0.96, non_local_exit],
                [non_local_exit, 0.005, non_local_exit, 0.96],
            ]
        )
    )
    path = tmp_path / "detector.pkl"
    joblib.dump(
        _pinned_detector(
            "nonlocal",
            discrete_transition_type=library_defaults,
            discrete_transition_concentration=1.0,
            discrete_transition_stickiness=np.array([1e5, 1e5, 300.0, 300.0]),
        ),
        path,
    )

    result = audit.check_decoder_settings(
        {EPOCHS[0]: path}, audit.expected_decoder_settings()["nonlocal"]
    )

    assert not result["ok"]
    (failure,) = result["failures"]
    assert {name for name in audit.DISCRETE_SETTINGS if f"{name} is " in failure} == {
        "discrete_transition_type",
        "discrete_transition_concentration",
        "discrete_transition_stickiness",
    }


# --------------------------------------------------------------------------
# Nothing to check
# --------------------------------------------------------------------------


EMPTY_INPUTS: dict[str, Callable[[Path], dict]] = {
    "one_commit": lambda tmp: audit.check_one_commit(
        audit.decode_provenance(tmp, tmp, []), {}
    ),
    "decoder_settings": lambda tmp: audit.check_decoder_settings({}),
    "grids_match": lambda tmp: audit.check_grids_match(_dot(n=0), _dot(n=0)),
    "minimum_sample_counts": lambda tmp: audit.check_minimum_sample_counts({}),
    "widloski_hashes": lambda tmp: audit.check_widloski_hashes({"epochs": {}}, tmp),
    "swr_spans": lambda tmp: audit.check_swr_spans({"epochs": {}}, {}),
    "tables_recorded": lambda tmp: audit.check_tables_recorded([], "a" * 40, repo=tmp),
    "pfc_sidecars": lambda tmp: audit.check_pfc_sidecars([], "e" * 64),
    "fit_provenance, no roots": lambda tmp: audit.check_fit_provenance(audit.REPO, []),
    "fit_provenance, a root with no fit": lambda tmp: audit.check_fit_provenance(
        audit.REPO, [tmp]
    ),
    "epoch_sets": lambda tmp: audit.check_epoch_sets(
        {}, {}, register=_register(tmp, analysed=(), pending=())
    ),
    "epochs_registered": lambda tmp: audit.check_epochs_registered(
        _epoch_info, _register(tmp, analysed=(), pending=())
    ),
    "decode_folder": lambda tmp: audit.check_decode_folder(tmp, _register(tmp)),
}
"""Each check on empty input: an audit that passes on nothing has checked nothing."""


@pytest.mark.parametrize("check", list(EMPTY_INPUTS))
def test_every_check_fails_on_empty_input(tmp_path, check):
    result = EMPTY_INPUTS[check](tmp_path)

    assert result["ok"] is False
    assert result["failures"]


# --------------------------------------------------------------------------
# The audit command, end to end
# --------------------------------------------------------------------------


@dataclass
class Tree:
    """A made-up refreshed tree, consistent everywhere."""

    argv: list[str]
    output: Path
    swr_manifest: Path
    root: Path
    log_dir: Path
    replay: Path


def _consistent_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tree:
    """Decodes, logs, both run manifests, tables, fits and PFC sidecars that agree,
    with an epoch register analysing their epochs and ``get_epoch_info`` listing
    them."""
    history = _history(tmp_path)
    repo, decode_commit, run_commit = history.repo, history.decode, history.later
    root, log_dir = tmp_path / "Processed-Data", tmp_path / "logs"
    for epoch_id in EPOCHS:
        for decode_id, dot_name, decoder in (
            (epoch_id, f"{epoch_id}_80_data_over_time.pkl", "nonlocal"),
            (
                f"{epoch_id}_contfrag",
                f"{epoch_id}_contfrag_data_over_time.pkl",
                "contfrag",
            ),
        ):
            record = _provenance(decode_commit)
            _stamp(root, log_dir, decode_id, record, record)
            joblib.dump(_pinned_detector(decoder), root / f"{decode_id}_detector.pkl")
            (root / f"{decode_id}_environment.pkl").write_bytes(b"environment")
            _dot().to_pickle(root / dot_name)

    widloski_manifest = tmp_path / "memoryless" / "manifest.json"
    widloski_manifest.parent.mkdir()
    widloski_manifest.write_text(
        json.dumps(_widloski_manifest(root, EPOCHS, run_commit))
    )
    time = _dot().index.to_numpy()
    spans = dict.fromkeys(EPOCHS, (float(time[0]), float(time[-1])))
    swr_manifest = tmp_path / "swr" / "manifest.json"
    swr_manifest.parent.mkdir()
    swr_manifest.write_text(
        json.dumps(_swr_manifest(spans, run_commit, {NO_INPUT_EPOCH: NO_RIPPLE_BAND}))
    )

    replay = _table_dir(root / "replay_content")
    _fit(replay / "models", "content", {"event_table": replay / "event_table.parquet"})
    contfrag = root / "contfrag_replay_content"
    write_table(
        _stamped(_sessions(), "event"), contfrag / "event_table_swr_HPC.parquet"
    )
    _fit(
        contfrag / "models_swr_HPC",
        "content",
        {"event_table_swr_HPC": contfrag / "event_table_swr_HPC.parquet"},
    )
    for table_dir in (replay, contfrag):
        audit.record_tables(table_dir, run_commit, {"decode_commit": decode_commit})
    event_table_sha256 = sha256_file(replay / "event_table.parquet")
    for epoch_name in ("02_r1", "04_r2"):
        (root / f"sim20200101_{epoch_name}_mPFC_modulation.provenance.json").write_text(
            json.dumps(_pfc_sidecar(epoch_name, event_table_sha256))
        )

    register = _write_register(tmp_path / "register" / "epoch_register.csv")
    monkeypatch.setattr(audit, "get_epoch_info", lambda: _epoch_info(*EPOCHS))

    output = tmp_path / "refresh" / "audit.json"
    argv = [
        "audit",
        "--output",
        str(output),
        "--register",
        str(register),
        "--root",
        str(root),
        "--log-dir",
        str(log_dir),
        "--widloski-manifest",
        str(widloski_manifest),
        "--swr-manifest",
        str(swr_manifest),
        "--replay-content-dir",
        str(replay),
        "--contfrag-dir",
        str(contfrag),
        "--repo",
        str(repo),
    ]
    return Tree(
        argv=argv,
        output=output,
        swr_manifest=swr_manifest,
        root=root,
        log_dir=log_dir,
        replay=replay,
    )


def test_the_audit_passes_a_consistent_tree_and_writes_only_its_report(
    tmp_path, monkeypatch
):
    tree = _consistent_tree(tmp_path, monkeypatch)
    before = _snapshot(tmp_path, tree.output)

    code = audit.main(tree.argv)

    report = json.loads(tree.output.read_text())
    failed = {
        name: check["failures"]
        for name, check in report["checks"].items()
        if not check["ok"]
    }
    assert failed == {}
    assert code == 0
    assert report["ok"]
    assert _snapshot(tmp_path, tree.output) == before
    # every artifact's epochs were checked, the tables' too, qvals through
    # the stop table
    assert set(report["checks"]["epoch_sets"]["details"]) >= {
        "nonlocal_decode",
        "contfrag_decode",
        "widloski",
        "swr",
        "replay_content/event_table.parquet",
        "replay_content/models_per_bin_rate/qvals_table.parquet",
        "contfrag_replay_content/event_table_swr_HPC.parquet",
    }
    # without --expected-settings, every detector is held to the code's pin
    assert report["expected_settings"] == {
        "source": "expected_decoder_settings()",
        "values": audit.expected_decoder_settings(),
    }
    # every epoch Spyglass lists is registered, and the pending ones are named
    assert report["checks"]["epochs_registered"]["details"]["by_status"] == {
        "analysed": 2
    }
    assert report["checks"]["pending_epochs"]["details"]["epochs"] == {
        NO_INPUT_EPOCH: f"PENDING_POPULATE: {NO_RIPPLE_BAND}"
    }
    # and a pass says what it did not cover
    assert report["not_checked"] == [
        f"fit_provenance: no Stage 5B fit (stage5b_*.pkl) at {tree.replay}",
        "fit_provenance: no stop_replay_summary fit (stop_replay_summary*.pkl) at "
        f"{tree.replay}",
        "pending_epochs: 1 epoch(s) pending a Spyglass populate are outside the "
        "analysed cohort, so no artifact is checked for them (checks.pending_epochs "
        "lists them)",
        "encoding_marks: not run; --check-marks reloads every epoch's marks from "
        "the database",
    ]


def test_expected_settings_from_a_file_override_the_code(tmp_path, monkeypatch):
    """``--expected-settings`` replaces the code's expectation, and the report says so."""
    tree = _consistent_tree(tmp_path, monkeypatch)
    # the library's NonLocal stickiness, against the pinned detectors' P30-B values
    override = {
        "nonlocal": {"discrete_transition_stickiness": [1e5, 1e5, 300.0, 300.0]}
    }
    settings_file = tmp_path / "expected_settings.json"
    settings_file.write_text(json.dumps(override))

    code = audit.main([*tree.argv, "--expected-settings", str(settings_file)])

    report = json.loads(tree.output.read_text())
    assert code == 1
    assert report["failed_checks"] == ["decoder_settings_nonlocal"]
    (failure,) = report["checks"]["decoder_settings_nonlocal"]["failures"]
    assert "discrete_transition_stickiness is" in failure
    assert report["expected_settings"] == {
        "source": str(settings_file),
        "values": override,
    }
    # a decoder the file does not name is compared across epochs only
    assert [
        gap for gap in report["not_checked"] if gap.startswith("decoder_settings")
    ] == [
        "decoder_settings_contfrag: compared across epochs only, since "
        "--expected-settings names no contfrag settings"
    ]


def test_the_audit_fails_when_the_swr_manifest_silently_lacks_an_epoch(
    tmp_path, monkeypatch
):
    tree = _consistent_tree(tmp_path, monkeypatch)
    manifest = json.loads(tree.swr_manifest.read_text())
    del manifest["epochs"][EPOCHS[1]]
    tree.swr_manifest.write_text(json.dumps(manifest))

    code = audit.main(tree.argv)

    report = json.loads(tree.output.read_text())
    assert code == 1
    assert not report["ok"]
    assert report["checks"]["epoch_sets"]["failures"] == [
        f"swr lacks 1 analysed epoch(s) with no recorded reason: {EPOCHS[1]}"
    ]


def test_the_audit_without_the_run_manifests_fails(tmp_path, monkeypatch):
    tree = _consistent_tree(tmp_path, monkeypatch)
    argv = list(tree.argv)
    for option in ("--widloski-manifest", "--swr-manifest"):
        del argv[argv.index(option) : argv.index(option) + 2]

    code = audit.main(argv)

    report = json.loads(tree.output.read_text())
    checks = report["checks"]
    assert code == 1
    assert checks["widloski_hashes"]["failures"] == [
        "no Widloski manifest given (--widloski-manifest)"
    ]
    assert checks["swr_spans"]["failures"] == ["no SWR manifest given (--swr-manifest)"]
    assert checks["spike_loader_unchanged"]["failures"] == [
        "widloski: no run commit to compare (no manifest, or no epoch run)",
        "swr: no run commit to compare (no manifest, or no epoch run)",
    ]
    assert checks["epoch_sets"]["failures"] == [
        f"widloski lacks 2 analysed epoch(s) with no recorded reason: "
        f"{', '.join(EPOCHS)}",
        f"swr lacks 2 analysed epoch(s) with no recorded reason: {', '.join(EPOCHS)}",
    ]
    assert report["failed_checks"] == [
        "widloski_hashes",
        "swr_spans",
        "spike_loader_unchanged",
        "epoch_sets",
    ]


def test_an_absent_decode_explained_through_exceptions_passes(tmp_path, monkeypatch):
    tree = _consistent_tree(tmp_path, monkeypatch)
    absent = f"{EPOCHS[1]}_contfrag"
    for path in audit.decode_files(EPOCHS[1], tree.root, "contfrag").values():
        path.unlink()
    reason = "ContFrag decode failed: no HPC spikes in the position span"
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text(
        json.dumps({"epochs": {"contfrag_decode": {EPOCHS[1]: reason}}})
    )

    code = audit.main([*tree.argv, "--exceptions", str(exceptions)])

    report = json.loads(tree.output.read_text())
    checks = report["checks"]
    failed = {
        name: check["failures"] for name, check in checks.items() if not check["ok"]
    }
    assert failed == {}
    assert code == 0
    assert checks["one_commit"]["details"]["absent"] == {
        absent: f"explained in epoch_sets: {reason}"
    }
    explained = checks["epoch_sets"]["details"]["contfrag_decode"]["explained"]
    assert explained == {EPOCHS[1]: reason}
    assert checks["grids_match"]["details"]["n_epochs"] == 1


def test_the_audit_fails_on_an_epoch_spyglass_lists_but_the_register_does_not(
    tmp_path, monkeypatch
):
    tree = _consistent_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(
        audit, "get_epoch_info", lambda: _epoch_info(*EPOCHS, "sim20200101_12_r6_HPC")
    )

    code = audit.main(tree.argv)

    report = json.loads(tree.output.read_text())
    assert code == 1
    assert report["failed_checks"] == ["epochs_registered"]
    assert report["checks"]["epochs_registered"]["failures"] == [
        "get_epoch_info lists 1 epoch(s) the register does not: sim20200101_.nwb 12_r6"
    ]


def test_the_audit_fails_on_a_decode_of_an_epoch_it_does_not_analyse(
    tmp_path, monkeypatch
):
    """A decode of a pending epoch, saved before the register says it is
    analysed, is outside the cohort every other artifact covers."""
    tree = _consistent_tree(tmp_path, monkeypatch)
    for path in audit.decode_files(EPOCHS[1], tree.root, "nonlocal").values():
        path.rename(path.with_name(path.name.replace("04_r2", "06_r3")))

    code = audit.main(tree.argv)

    report = json.loads(tree.output.read_text())
    assert code == 1
    assert (
        "nonlocal_decode holds 1 epoch(s) the register does not analyse: "
        f"{NO_INPUT_EPOCH} (pending, PENDING_POPULATE)"
    ) in report["checks"]["epoch_sets"]["failures"]
    assert report["checks"]["decode_folder"]["failures"] == [
        "the decode folder holds files of 1 epoch(s) the register does not "
        f"analyse: {NO_INPUT_EPOCH} (pending, PENDING_POPULATE)"
    ]


def test_record_tables_refuses_a_folder_with_no_table(tmp_path):
    with pytest.raises(ValueError, match="no table"):
        audit.record_tables(tmp_path, "c" * 40, {})
