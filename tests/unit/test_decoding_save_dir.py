"""Both HPC decoders stage, resume safely, refuse a CPU fallback and log provenance.

``run_decoding_bandit.py`` (NonLocal) and ``run_decoding_contfrag.py``
(ContFrag) write into ``--save_dir``, which defaults to ``Processed-Data``. A
rerun over the same folder resumes from it: the results are read back from that
folder, never the canonical one, and only ``data_over_time`` is rewritten,
because the reopened ``results.nc`` is still open lazily (Review Focus 1 of the
full-refresh plan). A fresh save stamps the epoch's provenance into
``results.nc``, and every file is written atomically.

Offline: both scripts import ``bandit_task``, which imports Spyglass at import
time, so they are imported with ``spyglass`` stubbed in ``sys.modules``, as
``test_events.py`` does. No test starts a CUDA cluster, and each test that
depends on ``HAS_CUDA`` sets it, since whether ``dask_cuda`` imports depends on
the environment.

Nothing here may write outside the test's own ``tmp_path``. Three things keep
it there:

- Every call passes ``save_dir`` and ``log_directory`` under ``tmp_path``. The
  defaults bound at definition time (``run_decode``'s ``save_dir`` and the
  loaders' ``root``) still name the real ``Processed-Data``, so no test leaves a
  write to them.
- Every stand-in that writes refuses any path outside ``tmp_path`` before it
  writes.
- The epoch is synthetic (``testrat20990101``), so even an unguarded write
  could not land on a real decode's file.

Each script's ``PROCESSED_DATA_DIR`` global also points at an empty folder under
``tmp_path``. That catches only code that writes through the global, not
through a default. An unguarded version of these tests once overwrote real
decodes.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import types
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay import decoding_common
from continuum_swr_replay.analysis.contfrag import result_loading as contfrag_loading
from continuum_swr_replay.analysis.hippocampus import result_loading as hpc_loading
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
NWB_FILE_NAME = "testrat20990101_.nwb"
EPOCH_NAME = "02_r1"
"""A synthetic epoch: no real decode has its file names."""
PROVENANCE_KEYS = {
    "commit",
    "dirty",
    "uv_lock_sha256",
    "ripple_detection_commit",
    "non_local_detector_version",
    "cuda_plugin_version",
}
SAVED_FILES = ("results.nc", "detector.pkl", "environment.pkl")
"""The files a decode writes once, and a resume must leave untouched."""


@dataclass(frozen=True)
class Decoder:
    """What the tests need to know about one decoder script."""

    module_name: str
    loader_name: str
    epoch_identifier: str
    data_over_time_name: str
    save_kwargs: dict[str, Any] = field(default_factory=dict)
    """Arguments ``save_results_and_model`` takes beyond the shared ones."""
    run_kwargs: dict[str, Any] = field(default_factory=dict)
    """Arguments that keep ``run_decode`` off the network (no figurl)."""


DECODERS = {
    "nonlocal": Decoder(
        module_name="scripts.decoding.run_decoding_bandit",
        loader_name="load_results",
        epoch_identifier="testrat20990101_02_r1_HPC",
        data_over_time_name="testrat20990101_02_r1_HPC_80_data_over_time.pkl",
        save_kwargs={"non_local_threshold": 0.8},
        run_kwargs={"create_figurl": False},
    ),
    "contfrag": Decoder(
        module_name="scripts.decoding.run_decoding_contfrag",
        loader_name="load_contfrag_results",
        epoch_identifier="testrat20990101_02_r1_HPC_contfrag",
        data_over_time_name="testrat20990101_02_r1_HPC_contfrag_data_over_time.pkl",
    ),
}

#: Modules that bind the Spyglass stub when imported under it.
_STUB_BOUND_MODULES = (
    "continuum_swr_replay.data_loaders.bandit_task",
    *(decoder.module_name for decoder in DECODERS.values()),
)


@dataclass(frozen=True)
class Decoders:
    modules: dict[str, types.ModuleType]
    canonical: Path
    """Where each script's ``PROCESSED_DATA_DIR`` global points during the test."""


@pytest.fixture
def decoders(monkeypatch, tmp_path):
    """Both decoder scripts, imported offline with ``spyglass`` stubbed.

    ``bandit_task`` imports ``spyglass.common.PositionIntervalMap`` at import
    time, which opens a DataJoint connection. The modules imported here that
    bind the stub are forgotten afterwards, so a later import gets the real ones.

    Each script's ``PROCESSED_DATA_DIR`` global is pointed at an empty folder under
    ``tmp_path``. That catches code that writes through the global. It does not
    change the defaults bound at definition time, so every test passes
    ``save_dir`` itself.
    """
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = type("PositionIntervalMap", (), {})
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)

    data_loaders = importlib.import_module("continuum_swr_replay.data_loaders")
    fresh = [name for name in _STUB_BOUND_MODULES if name not in sys.modules]
    load_data_was_cached = "load_data" in vars(data_loaders)

    modules = {
        key: importlib.import_module(decoder.module_name)
        for key, decoder in DECODERS.items()
    }
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    for module in modules.values():
        monkeypatch.setattr(module, "PROCESSED_DATA_DIR", canonical)

    yield Decoders(modules=modules, canonical=canonical)

    for name in fresh:
        sys.modules.pop(name, None)
        parent, _, child = name.rpartition(".")
        vars(sys.modules[parent]).pop(child, None)
    if not load_data_was_cached:
        vars(data_loaders).pop("load_data", None)


def _parse(argv: list[str]) -> Namespace:
    return decoding_common.add_common_args(ArgumentParser()).parse_args(argv)


def _one_epoch() -> pd.DataFrame:
    """``get_epoch_info``'s shape, for the synthetic epoch."""
    return pd.DataFrame(
        {"animal": ["testrat"], "date": [20990101], "epoch": [2], "epoch_tag": ["r1"]},
        index=pd.MultiIndex.from_tuples(
            [(NWB_FILE_NAME, EPOCH_NAME)],
            names=["nwb_file_name", "interval_list_name"],
        ),
    )


def _no_cluster(**kwargs: Any) -> None:
    raise AssertionError("run_batch started a cluster it should not have")


def _record_kwargs(path: Path, **kwargs: Any) -> None:
    """Stand in for ``run_decode``: write the keyword arguments it got to `path`.

    Dask pickles the task graph even for an in-process cluster, so the task runs
    on a copy of the function. A list in a closure would be appended to on that
    copy; a file is seen by the test.
    """
    path.write_bytes(pickle.dumps(kwargs))


def _session_data() -> dict[str, Any]:
    """The keys ``run_decode`` reads before it hands the session to the decoder."""
    return {
        "spike_times": {"HPC": [np.array([0.1, 0.2])]},
        "spike_waveform_features": {"HPC": [np.zeros((2, 4))]},
        "position_info": None,
        "linear_edge_order": None,
        "linear_edge_spacing": None,
        "track_graph": None,
    }


def _data_over_time() -> pd.DataFrame:
    return pd.DataFrame(
        {"is_non_local": [False, True]},
        index=pd.Index([0.0, 0.002], name="time"),
    )


def _inside(path: Path, root: Path) -> Path:
    """Return `path`, refusing any path outside `root` before anything is written."""
    if not Path(path).resolve().is_relative_to(root.resolve()):
        raise AssertionError(f"refusing to write outside the test folder: {path}")
    return Path(path)


class _GuardedFrame:
    """Stands in for ``data_over_time``: it pickles only under `root`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.frame = _data_over_time()

    def to_pickle(self, path: Path) -> None:
        self.frame.to_pickle(_inside(path, self.root))


class _WritesPlaceholders:
    """Stands in for a fresh decode's results, detector and environment.

    Each save writes a few bytes to the path it is given, so saving into a folder
    that does not exist fails, and it refuses any path outside `root`. The attrs
    a ``to_netcdf`` call sees are kept in ``written_attrs``.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.attrs: dict[str, str] = {}
        self.written_attrs: dict[str, str] | None = None

    def reset_index(self, dim: str) -> _WritesPlaceholders:
        return self

    def to_netcdf(self, path: Path, engine: str) -> None:
        self.written_attrs = dict(self.attrs)
        _inside(path, self.root).write_bytes(b"results")

    def save_model(self, path: Path) -> None:
        _inside(path, self.root).write_bytes(b"detector")

    def save_environment(self, path: Path) -> None:
        _inside(path, self.root).write_bytes(b"environment")


class _FailsMidWrite(_WritesPlaceholders):
    """Writes part of the environment, then fails, as a crash or a full disk would."""

    def save_environment(self, path: Path) -> None:
        _inside(path, self.root).write_bytes(b"envir")
        raise OSError("disk full")


class _RefusesToSave:
    """Stands in for what a resume reloads: writing any of it back is the bug."""

    @property
    def attrs(self) -> dict[str, str]:
        raise AssertionError("a resume stamped results.attrs")

    def reset_index(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a resume rewrote results.nc")

    def to_netcdf(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a resume rewrote results.nc")

    def save_model(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a resume rewrote the detector")

    def save_environment(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a resume rewrote the environment")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved_file_hashes(folder: Path, epoch_identifier: str) -> dict[str, str]:
    return {
        name: _sha256(folder / f"{epoch_identifier}_{name}") for name in SAVED_FILES
    }


def _write_saved_files(folder: Path, epoch_identifier: str) -> dict[str, str]:
    """Write placeholder decode files into `folder`; return their sha256s."""
    for name in SAVED_FILES:
        (folder / f"{epoch_identifier}_{name}").write_bytes(f"saved {name}".encode())
    return _saved_file_hashes(folder, epoch_identifier)


def _log(tmp_path: Path, decoder: Decoder) -> str:
    return (tmp_path / "logs" / f"{decoder.epoch_identifier}.log").read_text()


def _logged_provenance(tmp_path: Path, decoder: Decoder) -> str:
    """The JSON of the epoch log's one ``PROVENANCE`` line."""
    (line,) = [
        line for line in _log(tmp_path, decoder).splitlines() if "PROVENANCE " in line
    ]
    return line.split("PROVENANCE ", 1)[1]


def _run_decode(module: types.ModuleType, decoder: Decoder, tmp_path: Path, **kwargs):
    return module.run_decode(
        NWB_FILE_NAME,
        EPOCH_NAME,
        log_directory=tmp_path / "logs",
        **decoder.run_kwargs,
        **kwargs,
    )


def _patch_fresh_decode(
    monkeypatch, module: types.ModuleType, tmp_path: Path, fresh: _WritesPlaceholders
) -> None:
    """Make ``run_decode`` decode afresh into `fresh`, with nothing real loaded."""
    monkeypatch.setattr(module, "load_data", lambda *args, **kwargs: _session_data())
    monkeypatch.setattr(module, "get_training_timepoints", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "perform_decoding", lambda *args, **kwargs: (fresh, fresh, fresh)
    )
    monkeypatch.setattr(
        module,
        "organize_results",
        lambda *args, **kwargs: (_GuardedFrame(tmp_path), None),
    )


def _git_repo(path: Path) -> Callable[..., str]:
    """Commit a ``uv.lock`` in a new git repo at `path`; return a git runner for it."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=path, check=True, capture_output=True, text=True
        ).stdout.strip()

    config = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    config += ["-c", "commit.gpgsign=false"]
    git("init", "-q")
    (path / "uv.lock").write_text("locked")
    git("add", "uv.lock")
    git(*config, "commit", "-q", "-m", "lock")
    return git


# --------------------------------------------------------------------------
# The command line and the batch driver
# --------------------------------------------------------------------------


def test_save_dir_and_allow_cpu_defaults():
    args = _parse([])

    assert args.save_dir == PROCESSED_DATA_DIR
    assert args.allow_cpu is False

    staged = _parse(
        ["--save_dir", "Processed-Data/_staging_redecode_20261003", "--allow_cpu"]
    )
    assert staged.save_dir == Path("Processed-Data/_staging_redecode_20261003")
    assert staged.allow_cpu is True


def test_run_batch_refuses_cpu_fallback(monkeypatch, tmp_path):
    def no_decode(**kwargs: Any) -> None:
        raise AssertionError("run_batch decoded")

    monkeypatch.setattr(decoding_common, "HAS_CUDA", False)
    monkeypatch.setattr(decoding_common, "LocalCluster", _no_cluster)
    monkeypatch.setattr(decoding_common, "LocalCUDACluster", _no_cluster, raising=False)
    scratch = tmp_path / "scratch"

    with pytest.raises(RuntimeError, match="dask_cuda"):
        decoding_common.run_batch(
            no_decode, _parse(["--local_scratch_dir", str(scratch)])
        )
    assert not scratch.exists()


@pytest.mark.parametrize(
    ("has_cuda", "flags", "cluster_name", "unused_cluster_name"),
    [
        (False, ["--allow_cpu"], "LocalCluster", "LocalCUDACluster"),
        # The GPU path, with LocalCUDACluster replaced: the refusal must not fire.
        (True, [], "LocalCUDACluster", "LocalCluster"),
    ],
    ids=["cpu_allowed", "cuda"],
)
def test_run_batch_passes_save_dir(
    monkeypatch, tmp_path, has_cuda, flags, cluster_name, unused_cluster_name
):
    local_cluster = decoding_common.LocalCluster
    clusters = []

    def in_process_cluster(**kwargs: Any):
        # Forward the scratch folder run_batch chose, so dask's scratch space
        # lands under tmp_path rather than the system temp folder.
        clusters.append(
            local_cluster(
                n_workers=1,
                processes=False,
                dashboard_address=":0",
                local_directory=kwargs["local_directory"],
            )
        )
        return clusters[-1]

    monkeypatch.setattr(decoding_common, "HAS_CUDA", has_cuda)
    monkeypatch.setattr(
        decoding_common, cluster_name, in_process_cluster, raising=False
    )
    monkeypatch.setattr(
        decoding_common, unused_cluster_name, _no_cluster, raising=False
    )
    monkeypatch.setattr(decoding_common, "get_epoch_info", _one_epoch)
    # run_batch sets this variable and makes logs/ in the working directory;
    # monkeypatch restores both after the test.
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.99")
    monkeypatch.chdir(tmp_path)
    record = tmp_path / "run_decode_kwargs.pkl"
    scratch = tmp_path / "scratch"
    args = _parse(
        [
            *flags,
            "--save_dir",
            str(tmp_path / "staging"),
            "--local_scratch_dir",
            str(scratch),
        ]
    )

    # The scheduler keeps its own scratch space in dask's temporary directory
    # (the system temp folder by default), whatever the workers are given.
    with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
        try:
            decoding_common.run_batch(functools.partial(_record_kwargs, record), args)
        finally:
            for cluster in clusters:
                cluster.close()

    kwargs = pickle.loads(record.read_bytes())
    assert kwargs["save_dir"] == args.save_dir == tmp_path / "staging"
    assert kwargs["nwb_file_name"] == NWB_FILE_NAME
    assert kwargs["epoch_name"] == EPOCH_NAME
    assert (scratch / "dask-scratch-space").is_dir()


# --------------------------------------------------------------------------
# The loaders
# --------------------------------------------------------------------------


def _saved_results() -> xr.Dataset:
    """A results set as the decoders save it, with ``state_bins`` reset."""
    return xr.Dataset(
        {"acausal_posterior": (("time", "state_bins"), np.full((2, 2), 0.5))},
        coords={
            "time": [0.0, 0.002],
            "state": ("state_bins", ["Continuous", "Fragmented"]),
            "position": ("state_bins", [0.0, 0.0]),
        },
    )


def _record_reads(monkeypatch, detector_class: type) -> list[Path]:
    """Make ``xr.open_dataset`` and ``detector_class.load_model`` record their paths."""
    reads = []

    def open_dataset(path: Path, engine: str) -> xr.Dataset:
        reads.append(Path(path))
        return _saved_results()

    def load_model(filename: Path) -> types.SimpleNamespace:
        reads.append(Path(filename))
        return types.SimpleNamespace(environments=["environment"])

    monkeypatch.setattr(xr, "open_dataset", open_dataset)
    monkeypatch.setattr(detector_class, "load_model", staticmethod(load_model))
    return reads


def test_load_results_reads_from_root(monkeypatch, tmp_path):
    reads = _record_reads(monkeypatch, hpc_loading.NonLocalClusterlessDetector)
    stem = DECODERS["nonlocal"].epoch_identifier

    *_, env = hpc_loading.load_results(NWB_FILE_NAME, EPOCH_NAME, root=tmp_path)

    assert reads == [tmp_path / f"{stem}_results.nc", tmp_path / f"{stem}_detector.pkl"]
    assert env == "environment"

    reads.clear()
    hpc_loading.load_results(NWB_FILE_NAME, EPOCH_NAME)
    assert [path.parent for path in reads] == [PROCESSED_DATA_DIR] * 2


def test_load_contfrag_results_reads_from_root(monkeypatch, tmp_path):
    reads = _record_reads(monkeypatch, contfrag_loading.ContFragClusterlessClassifier)
    stem = DECODERS["contfrag"].epoch_identifier

    *_, env = contfrag_loading.load_contfrag_results(
        NWB_FILE_NAME, EPOCH_NAME, root=tmp_path
    )

    assert reads == [tmp_path / f"{stem}_results.nc", tmp_path / f"{stem}_detector.pkl"]
    assert env == "environment"

    reads.clear()
    contfrag_loading.load_contfrag_results(NWB_FILE_NAME, EPOCH_NAME)
    assert [path.parent for path in reads] == [PROCESSED_DATA_DIR] * 2


# --------------------------------------------------------------------------
# Each decoder's run_decode and save_results_and_model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(DECODERS))
def test_run_decode_reloads_from_save_dir(decoders, monkeypatch, tmp_path, key):
    decoder, module = DECODERS[key], decoders.modules[key]
    save_dir = tmp_path / "staging"
    checked, loaded = [], []

    class Reloaded(Exception):
        """Raised by the loader once it has recorded its call."""

    def results_exist(folder: Path, epoch_identifier: str, overwrite: bool) -> bool:
        checked.append(folder)
        return True

    def loader(*args: Any, **kwargs: Any) -> None:
        loaded.append(kwargs)
        raise Reloaded

    monkeypatch.setattr(module, "load_data", lambda *args, **kwargs: _session_data())
    monkeypatch.setattr(module, "results_exist", results_exist)
    monkeypatch.setattr(module, decoder.loader_name, loader)

    failed = _run_decode(module, decoder, tmp_path, save_dir=save_dir)

    assert failed == decoder.epoch_identifier  # the loader's sentinel ended the run
    assert checked == [save_dir]
    assert [call["root"] for call in loaded] == [save_dir]


@pytest.mark.parametrize("key", list(DECODERS))
def test_resume_rewrites_only_data_over_time(decoders, tmp_path, key):
    decoder, module = DECODERS[key], decoders.modules[key]
    before = _write_saved_files(tmp_path, decoder.epoch_identifier)
    data_over_time = _GuardedFrame(tmp_path)

    module.save_results_and_model(
        logging.getLogger(__name__),
        _RefusesToSave(),
        _RefusesToSave(),
        _RefusesToSave(),
        data_over_time,
        tmp_path,
        decoder.epoch_identifier,
        **decoder.save_kwargs,
        resumed=True,
        provenance='{"commit": "unused on a resume"}',
    )

    assert _saved_file_hashes(tmp_path, decoder.epoch_identifier) == before
    pd.testing.assert_frame_equal(
        pd.read_pickle(tmp_path / decoder.data_over_time_name), data_over_time.frame
    )


@pytest.mark.parametrize("key", list(DECODERS))
def test_run_decode_resume_keeps_saved_files(decoders, monkeypatch, tmp_path, key):
    """A rerun over a staged folder resumes, says so, and leaves the results alone."""
    decoder, module = DECODERS[key], decoders.modules[key]
    save_dir = tmp_path / "staging"
    save_dir.mkdir()
    before = _write_saved_files(save_dir, decoder.epoch_identifier)
    roots = []

    def loader(*args: Any, **kwargs: Any) -> tuple[_RefusesToSave, ...]:
        roots.append(kwargs["root"])
        return _RefusesToSave(), _RefusesToSave(), _RefusesToSave()

    monkeypatch.setattr(module, "load_data", lambda *args, **kwargs: _session_data())
    monkeypatch.setattr(module, decoder.loader_name, loader)
    monkeypatch.setattr(
        module,
        "organize_results",
        lambda *args, **kwargs: (_GuardedFrame(tmp_path), None),
    )

    failed = _run_decode(module, decoder, tmp_path, save_dir=save_dir)

    assert failed is None, _log(tmp_path, decoder)
    assert roots == [save_dir]
    assert _saved_file_hashes(save_dir, decoder.epoch_identifier) == before
    assert (save_dir / decoder.data_over_time_name).exists()
    assert not any(decoders.canonical.iterdir())
    log = _log(tmp_path, decoder).splitlines()
    resumed = [line for line in log if "RESUMED" in line]
    assert len(resumed) == 1
    assert resumed[0].endswith(f"RESUMED from existing results in {save_dir.resolve()}")
    assert any(line.endswith(f"save_dir: {save_dir.resolve()}") for line in log)


@pytest.mark.parametrize("key", list(DECODERS))
def test_save_dir_is_created(decoders, monkeypatch, tmp_path, key):
    decoder, module = DECODERS[key], decoders.modules[key]
    save_dir = tmp_path / "staging" / "redecode"
    _patch_fresh_decode(monkeypatch, module, tmp_path, _WritesPlaceholders(tmp_path))

    failed = _run_decode(module, decoder, tmp_path, save_dir=save_dir)

    assert failed is None, _log(tmp_path, decoder)
    assert sorted(path.name for path in save_dir.iterdir()) == sorted(
        [f"{decoder.epoch_identifier}_{name}" for name in SAVED_FILES]
        + [decoder.data_over_time_name]
    )
    assert not any(decoders.canonical.iterdir())


@pytest.mark.parametrize("key", list(DECODERS))
def test_fresh_save_stamps_provenance_into_results(
    decoders, monkeypatch, tmp_path, key
):
    """The record outlives the epoch log, which a later resume rewrites."""
    decoder, module = DECODERS[key], decoders.modules[key]
    fresh = _WritesPlaceholders(tmp_path)
    _patch_fresh_decode(monkeypatch, module, tmp_path, fresh)

    failed = _run_decode(module, decoder, tmp_path, save_dir=tmp_path / "staging")

    assert failed is None, _log(tmp_path, decoder)
    assert fresh.written_attrs is not None  # to_netcdf was called
    assert fresh.written_attrs["provenance"] == _logged_provenance(tmp_path, decoder)


@pytest.mark.parametrize("key", list(DECODERS))
def test_a_failed_write_leaves_no_final_file(decoders, monkeypatch, tmp_path, key):
    """A write that dies part way leaves nothing ``results_exist`` would trust."""
    decoder, module = DECODERS[key], decoders.modules[key]
    save_dir = tmp_path / "staging"
    _patch_fresh_decode(monkeypatch, module, tmp_path, _FailsMidWrite(tmp_path))

    failed = _run_decode(module, decoder, tmp_path, save_dir=save_dir)

    assert failed == decoder.epoch_identifier
    stem = decoder.epoch_identifier
    assert sorted(path.name for path in save_dir.iterdir()) == [
        f"{stem}_detector.pkl",
        f"{stem}_results.nc",
    ]
    assert not decoding_common.results_exist(save_dir, stem, overwrite=False)


def test_write_atomically_keeps_the_old_file_when_a_write_fails(tmp_path):
    path = tmp_path / "testrat20990101_02_r1_HPC_detector.pkl"
    path.write_bytes(b"the previous run")

    def half_write(temporary: Path) -> None:
        _inside(temporary, tmp_path).write_bytes(b"trunc")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        decoding_common.write_atomically(path, half_write)

    assert path.read_bytes() == b"the previous run"
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_provenance_record_fields():
    record = decoding_common.provenance_record(REPO_ROOT)

    assert set(record) == PROVENANCE_KEYS
    assert re.fullmatch(r"[0-9a-f]{40}", record["commit"])
    assert isinstance(record["dirty"], bool)
    assert record["uv_lock_sha256"] == _sha256(REPO_ROOT / "uv.lock")
    commit = record["ripple_detection_commit"]
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_provenance_dirty_ignores_untracked_files(tmp_path):
    """The main tree holds untracked files from other work; they are not dirt."""
    git = _git_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("untracked")

    clean = decoding_common.provenance_record(tmp_path)
    assert clean["commit"] == git("rev-parse", "HEAD")
    assert clean["dirty"] is False

    (tmp_path / "uv.lock").write_text("edited")
    assert decoding_common.provenance_record(tmp_path)["dirty"] is True


def test_provenance_never_writes_the_git_index(tmp_path):
    """By default ``git status`` refreshes the index and writes it back under a
    lock. Every epoch of a multi-day batch computes the record, in a checkout
    that other work commits in, so it must not write.
    """
    git = _git_repo(tmp_path)
    lock = tmp_path / "uv.lock"
    stat = lock.stat()
    # A new mtime with the same content: the index's cached stat is now stale.
    os.utime(lock, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    index = tmp_path / ".git" / "index"
    before = _sha256(index)

    decoding_common.provenance_record(tmp_path)
    assert _sha256(index) == before

    # The control: a plain status does refresh and rewrite this index.
    git("status", "--porcelain")
    assert _sha256(index) != before


@pytest.mark.parametrize("key", list(DECODERS))
def test_epoch_log_has_one_provenance_line(decoders, monkeypatch, tmp_path, key):
    decoder, module = DECODERS[key], decoders.modules[key]

    def load_data(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the header is all this test reads")

    monkeypatch.setattr(module, "load_data", load_data)

    _run_decode(module, decoder, tmp_path, save_dir=tmp_path / "staging")

    lines = _log(tmp_path, decoder).splitlines()
    at = [i for i, line in enumerate(lines) if "PROVENANCE " in line]
    assert len(at) == 1
    assert "START" in lines[at[0] - 1]
    payload = lines[at[0]].split("PROVENANCE ", 1)[1]
    logged = json.loads(payload)
    assert payload == json.dumps(logged, sort_keys=True)
    # Other work shares this tree, so dirty can change between the two calls.
    now = decoding_common.provenance_record(REPO_ROOT)
    assert set(logged) == set(now) == PROVENANCE_KEYS
    assert isinstance(logged["dirty"], bool)
    assert {k: v for k, v in logged.items() if k != "dirty"} == {
        k: v for k, v in now.items() if k != "dirty"
    }
