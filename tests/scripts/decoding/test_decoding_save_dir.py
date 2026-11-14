"""Both HPC decoders stage, resume safely, refuse a CPU fallback, log provenance
and take their masks and SWR labels from the SWR source.

``run_decoding_bandit.py`` (NonLocal) and ``run_decoding_contfrag.py``
(ContFrag) share their flow (``scripts/decoding/_decode_runner.py``), each with
its own ``RUNNER`` record, and write into ``--save_dir``, which defaults to ``Processed-Data``. A
rerun over the same folder resumes from it: the results are read back from that
folder, never the canonical one, and only ``data_over_time`` is rewritten,
because the reopened ``results.nc`` is still open lazily (Review Focus 1 of the
full-refresh plan). A fresh save stamps the epoch's provenance into
``results.nc``, and every file is written atomically.

Offline: both scripts are imported with ``spyglass`` stubbed in ``sys.modules``,
as ``test_events.py`` does, so nothing they import reaches the database. No test starts a CUDA cluster, and each test that
depends on ``HAS_CUDA`` sets it, since whether ``dask_cuda`` imports depends on
the environment.

Nothing here may write outside the test's own ``tmp_path``. Three things keep
it there:

- Every call passes ``save_dir`` and ``log_directory`` under ``tmp_path``.
- Every stand-in that writes refuses any path outside ``tmp_path`` before it
  writes.
- The epoch is synthetic (``testrat20990101``), so even an unguarded write
  could not land on a real decode's file.

The data root (``paths.processed_data_dir()``, which ``run_decode``'s
``save_dir`` and the loaders' ``root`` default to) is also a folder under
``tmp_path``, ``canonical``, and that folder holds nothing but the synthetic SWR
source, so a write to the default lands there and is caught. An unguarded
version of these tests once overwrote real decodes.

An HPC decode reads the SWR source's Kay inventory and artifact run, so the
fixture points the source at synthetic run folders under ``tmp_path`` for the
synthetic epoch.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import inspect
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
from types import MappingProxyType
from typing import Any

import dask
import dask.distributed
import non_local_detector.environment
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.contfrag import result_loading as contfrag_loading
from continuum_swr_replay.analysis.event_definition import (
    analysed_time,
    decode_masks,
    swr_source,
)
from continuum_swr_replay.analysis.event_definition.immobility import (
    EventDefinitionError,
)
from continuum_swr_replay.decoding import files as decoding_files
from continuum_swr_replay.decoding import fit
from continuum_swr_replay.decoding import results as hpc_loading
from continuum_swr_replay.paths import PROCESSED_DATA_DIR_ENV, processed_data_dir
from continuum_swr_replay.runs import batch, cli, provenance
from continuum_swr_replay.tables.io import write_atomically
from tests.support._swr_inventory_fixtures import (
    ARTIFACT_RUN_ID,
    RUN_ID,
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
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

SWR_EPOCH = "testrat20990101_02_r1_HPC"
"""The synthetic epoch's id in the SWR source."""
DECODE_TIME = np.round(100.0 + np.arange(5000) / 500.0, 6)
"""The synthetic epoch's position (decode) grid: 500 Hz, 100.000-109.998 s."""
SWR_EVENTS = [(100.5, 100.56), (102.4, 102.6)]
"""Its Kay inventory."""
ARTIFACT_CORES = [("ca1", 104.0, 104.0)]
"""Its artifact run's cores: one CA1 step, [103.7, 104.3] in the one mask."""
MASK_INTERVAL = (103.7, 104.3)
SOURCE_FOLDERS = {"replay_detection_comparison", "artifact_intervals"}
"""The data root's folders the SWR source's two run folders are written under."""


@dataclass(frozen=True)
class Decoder:
    """What the tests need to know about one decoder script."""

    module_name: str
    epoch_identifier: str
    data_over_time_name: str
    save_kwargs: dict[str, Any] = field(default_factory=dict)
    """Arguments ``save_results_and_model`` takes beyond the shared ones."""
    run_kwargs: dict[str, Any] = field(default_factory=dict)
    """Arguments that keep ``run_decode`` off the network (no figurl)."""


DECODERS = {
    "nonlocal": Decoder(
        module_name="scripts.decoding.run_decoding_bandit",
        epoch_identifier="testrat20990101_02_r1_HPC",
        data_over_time_name="testrat20990101_02_r1_HPC_80_data_over_time.pkl",
        save_kwargs={"non_local_threshold": 0.8},
        run_kwargs={"create_figurl": False},
    ),
    "contfrag": Decoder(
        module_name="scripts.decoding.run_decoding_contfrag",
        epoch_identifier="testrat20990101_02_r1_HPC_contfrag",
        data_over_time_name="testrat20990101_02_r1_HPC_contfrag_data_over_time.pkl",
    ),
}

SHARED_MODULE = "scripts.decoding._decode_runner"
"""The flow both scripts run (``run_decode``), with its own module globals."""

#: Modules that bind the Spyglass stub when imported under it.
_STUB_BOUND_MODULES = (
    "continuum_swr_replay.data_loaders.bandit_task",
    SHARED_MODULE,
    *(decoder.module_name for decoder in DECODERS.values()),
)


@dataclass(frozen=True)
class Decoders:
    modules: dict[str, types.ModuleType]
    flow: types.ModuleType
    """``_decode_runner``: the names ``run_decode`` reads are patched here."""
    canonical: Path
    """The data root during the test: it holds the synthetic SWR source alone."""

    def holds_no_decode(self) -> bool:
        """Nothing but the SWR source's two run folders is in the data root."""
        return {path.name for path in self.canonical.iterdir()} == SOURCE_FOLDERS


@pytest.fixture
def decoders(monkeypatch, tmp_path):
    """Both decoder scripts, imported offline with ``spyglass`` stubbed.

    The stub keeps anything they import from reaching the database. The
    modules imported here that bind the stub are forgotten afterwards, so a
    later import gets the real ones.

    The data root is a folder under ``tmp_path`` that holds only the
    synthetic SWR source (:attr:`Decoders.canonical`); every test still passes
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
    flow = importlib.import_module(SHARED_MODULE)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    use_swr_source(
        monkeypatch,
        write_swr_inventory(
            canonical,
            {SWR_EPOCH: SWR_EVENTS},
            artifact_run_id=ARTIFACT_RUN_ID,
        ),
        write_artifact_run(canonical, {SWR_EPOCH: ARTIFACT_CORES}),
    )
    assert processed_data_dir() == canonical

    yield Decoders(modules=modules, flow=flow, canonical=canonical)

    for name in fresh:
        sys.modules.pop(name, None)
        parent, _, child = name.rpartition(".")
        vars(sys.modules[parent]).pop(child, None)
    if not load_data_was_cached:
        vars(data_loaders).pop("load_data", None)


def _parse(argv: list[str]) -> Namespace:
    return cli.add_common_args(ArgumentParser()).parse_args(argv)


def _one_epoch(pending_only: bool = False) -> pd.DataFrame:
    """``epochs_to_run``'s frame (``get_epoch_info``'s shape), for the synthetic
    epoch."""
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
    """Stands in for what a resume reloads: writing any of it back is the bug.

    Its ``attrs`` can be read, as a resume checks the saved masks, but not
    written.
    """

    def __init__(self, attrs: dict[str, str] | None = None) -> None:
        self._attrs = MappingProxyType(dict(attrs or {}))

    @property
    def attrs(self) -> MappingProxyType:
        return self._attrs

    def reset_index(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a resume rewrote results.nc")

    def to_netcdf(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a resume rewrote results.nc")

    def save_model(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a resume rewrote the detector")

    def save_environment(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a resume rewrote the environment")


def _saved_attrs(
    swr_inventory_run_id: str = RUN_ID, artifact_run_id: str = ARTIFACT_RUN_ID
) -> dict[str, str]:
    """A saved HPC decode's ``results.nc`` attrs: its masks' record, as built
    under the given run ids (by default the configured ones)."""
    record = {
        "swr_inventory_run_id": swr_inventory_run_id,
        "artifact_run_id": artifact_run_id,
    }
    return {"decode_masks": json.dumps(record, sort_keys=True)}


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


def _replace_runner(monkeypatch, module: types.ModuleType, **changes: Any) -> None:
    """Give the script's ``RUNNER`` record `changes` (its loader, its aggregation)."""
    monkeypatch.setattr(module, "RUNNER", dataclasses.replace(module.RUNNER, **changes))


def _patch_fresh_decode(
    monkeypatch, flow: types.ModuleType, tmp_path: Path, fresh: _WritesPlaceholders
) -> None:
    """Make ``run_decode`` decode afresh into `fresh`, with nothing real loaded.

    `flow` is ``_decode_runner``, whose globals ``run_decode`` reads.
    """
    monkeypatch.setattr(flow, "load_data", lambda *args, **kwargs: _session_data())
    monkeypatch.setattr(flow, "get_training_timepoints", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        flow,
        "hpc_decode_masks",
        lambda *args, **kwargs: decode_masks.HpcDecodeMasks(
            np.zeros(2, dtype=bool), np.ones(2, dtype=bool), {"stand_in": True}
        ),
    )
    monkeypatch.setattr(
        flow, "perform_decoding", lambda *args, **kwargs: (fresh, fresh, fresh)
    )
    monkeypatch.setattr(
        flow,
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


def test_save_dir_and_allow_cpu_defaults(processed_data_root):
    args = _parse([])

    assert args.save_dir == processed_data_root
    assert args.allow_cpu is False

    staged = _parse(
        ["--save_dir", "Processed-Data/_staging_redecode_20261003", "--allow_cpu"]
    )
    assert staged.save_dir == Path("Processed-Data/_staging_redecode_20261003")
    assert staged.allow_cpu is True


def test_run_batch_refuses_cpu_fallback(monkeypatch, tmp_path):
    def no_decode(**kwargs: Any) -> None:
        raise AssertionError("run_batch decoded")

    monkeypatch.setattr(batch, "cuda_cluster_class", lambda: None)
    monkeypatch.setattr(dask.distributed, "LocalCluster", _no_cluster)
    scratch = tmp_path / "scratch"

    with pytest.raises(RuntimeError, match="dask_cuda"):
        batch.run_batch(no_decode, _parse(["--local_scratch_dir", str(scratch)]))
    assert not scratch.exists()


@pytest.mark.parametrize(
    ("has_cuda", "flags"),
    [
        (False, ["--allow_cpu"]),
        # The GPU path, with LocalCUDACluster replaced: the refusal must not fire.
        (True, []),
    ],
    ids=["cpu_allowed", "cuda"],
)
def test_run_batch_passes_save_dir(monkeypatch, tmp_path, has_cuda, flags):
    local_cluster = dask.distributed.LocalCluster
    clusters = []

    def in_process_cluster(**kwargs: Any):
        # Forward the scratch folder run_batch chose, so dask's scratch space
        # lands under tmp_path rather than the system temp folder.
        clusters.append(
            local_cluster(
                n_workers=1,
                processes=False,
                # A thread worker measures the whole test process, which can pass
                # dask's memory limit late in the suite and pause the worker for good.
                memory_limit=0,
                dashboard_address=":0",
                local_directory=kwargs["local_directory"],
            )
        )
        return clusters[-1]

    _stand_in_clusters(monkeypatch, has_cuda=has_cuda, cluster=in_process_cluster)
    monkeypatch.setattr(batch, "epochs_to_run", _one_epoch)
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
            batch.run_batch(functools.partial(_record_kwargs, record), args)
        finally:
            for cluster in clusters:
                cluster.close()

    kwargs = pickle.loads(record.read_bytes())
    assert kwargs["save_dir"] == args.save_dir == tmp_path / "staging"
    assert kwargs["nwb_file_name"] == NWB_FILE_NAME
    assert kwargs["epoch_name"] == EPOCH_NAME
    assert (scratch / "dask-scratch-space").is_dir()


class _Client:
    """Stands in for ``dask.distributed.Client``: it connects to nothing."""

    def __init__(self, cluster: object) -> None:
        pass

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _stand_in_clusters(monkeypatch, *, has_cuda: bool, cluster: Callable) -> None:
    """``run_batch`` gets `cluster` for the path `has_cuda` picks, and fails the
    test if it starts the other one.

    ``run_batch`` imports dask when it starts a cluster, so the stand-ins go
    where it reads them: ``cuda_cluster_class`` (None when dask_cuda does not
    import) and ``dask.distributed.LocalCluster``.
    """
    monkeypatch.setattr(
        batch, "cuda_cluster_class", lambda: cluster if has_cuda else None
    )
    monkeypatch.setattr(
        dask.distributed, "LocalCluster", _no_cluster if has_cuda else cluster
    )


def _no_dask_scheduler(monkeypatch) -> None:
    """No client and no scheduler: each epoch's call runs as it is built."""
    monkeypatch.setattr(dask.distributed, "Client", _Client)
    monkeypatch.setattr(dask, "delayed", lambda function: function)
    monkeypatch.setattr(dask, "compute", lambda *outcomes: outcomes)


def test_run_batch_resolves_a_relative_save_dir(monkeypatch, tmp_path):
    """A relative ``--save_dir`` reaches each epoch's decode as an absolute path.

    A dask worker need not share the driver's working directory, so the
    folder is resolved before any task is built. The cluster, the client and
    dask itself are stand-ins: each epoch's call runs here, as it is built.
    """
    calls: list[dict[str, Any]] = []
    _stand_in_clusters(monkeypatch, has_cuda=False, cluster=lambda **kwargs: object())
    _no_dask_scheduler(monkeypatch)
    monkeypatch.setattr(batch, "epochs_to_run", _one_epoch)
    # run_batch sets this variable and makes logs/ in the working directory;
    # monkeypatch restores both after the test.
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.99")
    monkeypatch.chdir(tmp_path)
    args = _parse(
        ["--allow_cpu", "--save_dir", "staging", "--local_scratch_dir", "scratch"]
    )

    batch.run_batch(lambda **kwargs: calls.append(kwargs), args)

    (kwargs,) = calls
    assert kwargs["save_dir"].is_absolute()
    assert kwargs["save_dir"] == (tmp_path / "staging").resolve()
    assert kwargs["epoch_name"] == EPOCH_NAME


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


def _record_reads(monkeypatch) -> list[Path]:
    """Make ``xr.open_dataset`` and the saved-detector loader record their paths."""
    reads = []

    def open_dataset(path: Path, engine: str, **kwargs) -> xr.Dataset:
        reads.append(Path(path))
        return _saved_results()

    def load_saved_detector(path: Path) -> types.SimpleNamespace:
        reads.append(Path(path))
        return types.SimpleNamespace(environments=["environment"])

    monkeypatch.setattr(xr, "open_dataset", open_dataset)
    monkeypatch.setattr(hpc_loading, "load_saved_detector", load_saved_detector)
    return reads


def test_load_results_reads_from_root(monkeypatch, tmp_path, processed_data_root):
    reads = _record_reads(monkeypatch)
    stem = DECODERS["nonlocal"].epoch_identifier

    *_, env = hpc_loading.load_results(NWB_FILE_NAME, EPOCH_NAME, root=tmp_path)

    assert reads == [tmp_path / f"{stem}_results.nc", tmp_path / f"{stem}_detector.pkl"]
    assert env == "environment"

    reads.clear()
    hpc_loading.load_results(NWB_FILE_NAME, EPOCH_NAME)
    assert [path.parent for path in reads] == [processed_data_root] * 2


def test_load_contfrag_results_reads_from_root(
    monkeypatch, tmp_path, processed_data_root
):
    reads = _record_reads(monkeypatch)
    stem = DECODERS["contfrag"].epoch_identifier

    *_, env = contfrag_loading.load_contfrag_results(
        NWB_FILE_NAME, EPOCH_NAME, root=tmp_path
    )

    assert reads == [tmp_path / f"{stem}_results.nc", tmp_path / f"{stem}_detector.pkl"]
    assert env == "environment"

    reads.clear()
    contfrag_loading.load_contfrag_results(NWB_FILE_NAME, EPOCH_NAME)
    assert [path.parent for path in reads] == [processed_data_root] * 2


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

    monkeypatch.setattr(
        decoders.flow, "load_data", lambda *args, **kwargs: _session_data()
    )
    monkeypatch.setattr(decoders.flow, "results_exist", results_exist)
    _replace_runner(monkeypatch, module, load_saved=loader)

    failed = _run_decode(module, decoder, tmp_path, save_dir=save_dir)

    assert failed == decoder.epoch_identifier  # the loader's sentinel ended the run
    assert checked == [save_dir]
    assert [call["root"] for call in loaded] == [save_dir]


@pytest.mark.parametrize("key", list(DECODERS))
def test_resume_rewrites_only_data_over_time(decoders, tmp_path, key):
    decoder = DECODERS[key]
    before = _write_saved_files(tmp_path, decoder.epoch_identifier)
    data_over_time = _GuardedFrame(tmp_path)

    decoders.flow.save_results_and_model(
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
        return _RefusesToSave(_saved_attrs()), _RefusesToSave(), _RefusesToSave()

    monkeypatch.setattr(
        decoders.flow, "load_data", lambda *args, **kwargs: _session_data()
    )
    _replace_runner(monkeypatch, module, load_saved=loader)
    monkeypatch.setattr(
        decoders.flow,
        "organize_results",
        lambda *args, **kwargs: (_GuardedFrame(tmp_path), None),
    )

    failed = _run_decode(module, decoder, tmp_path, save_dir=save_dir)

    assert failed is None, _log(tmp_path, decoder)
    assert roots == [save_dir]
    assert _saved_file_hashes(save_dir, decoder.epoch_identifier) == before
    assert (save_dir / decoder.data_over_time_name).exists()
    assert decoders.holds_no_decode()
    log = _log(tmp_path, decoder).splitlines()
    resumed = [line for line in log if "RESUMED" in line]
    assert len(resumed) == 1
    assert resumed[0].endswith(f"RESUMED from existing results in {save_dir.resolve()}")
    assert any(line.endswith(f"save_dir: {save_dir.resolve()}") for line in log)


@pytest.mark.parametrize(
    "saved",
    [
        {},
        _saved_attrs(swr_inventory_run_id="50a98592acee"),
        _saved_attrs(artifact_run_id="otherartifacts"),
    ],
    ids=["no_masks", "other_inventory", "other_artifact_run"],
)
@pytest.mark.parametrize("key", list(DECODERS))
def test_a_resume_refuses_results_made_under_other_masks(
    decoders, monkeypatch, tmp_path, key, saved
):
    """An HPC resume rewrites data_over_time with the configured source's
    labels, so the results it reuses must have been decoded with the same two
    runs: one decoded before the masks, or under other run ids, is refused and
    left as it was."""
    decoder, module = DECODERS[key], decoders.modules[key]
    save_dir = tmp_path / "staging"
    save_dir.mkdir()
    before = _write_saved_files(save_dir, decoder.epoch_identifier)

    def organize_results(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a resume organized results decoded under other masks")

    monkeypatch.setattr(
        decoders.flow, "load_data", lambda *args, **kwargs: _session_data()
    )
    _replace_runner(
        monkeypatch,
        module,
        load_saved=lambda *args, **kwargs: (
            _RefusesToSave(saved),
            _RefusesToSave(),
            _RefusesToSave(),
        ),
    )
    monkeypatch.setattr(decoders.flow, "organize_results", organize_results)

    failed = _run_decode(module, decoder, tmp_path, save_dir=save_dir)

    assert failed == decoder.epoch_identifier
    assert _saved_file_hashes(save_dir, decoder.epoch_identifier) == before
    assert not (save_dir / decoder.data_over_time_name).exists()
    log = _log(tmp_path, decoder)
    assert "refusing to resume" in log
    if saved:
        assert "50a98592acee" in log or "otherartifacts" in log
    else:
        assert "no decode_masks" in log


@pytest.mark.parametrize("key", list(DECODERS))
def test_save_dir_is_created(decoders, monkeypatch, tmp_path, key):
    decoder, module = DECODERS[key], decoders.modules[key]
    save_dir = tmp_path / "staging" / "redecode"
    _patch_fresh_decode(
        monkeypatch, decoders.flow, tmp_path, _WritesPlaceholders(tmp_path)
    )

    failed = _run_decode(module, decoder, tmp_path, save_dir=save_dir)

    assert failed is None, _log(tmp_path, decoder)
    assert sorted(path.name for path in save_dir.iterdir()) == sorted(
        [f"{decoder.epoch_identifier}_{name}" for name in SAVED_FILES]
        + [decoder.data_over_time_name]
    )
    assert decoders.holds_no_decode()


@pytest.mark.parametrize("key", list(DECODERS))
def test_fresh_save_stamps_provenance_into_results(
    decoders, monkeypatch, tmp_path, key
):
    """The record outlives the epoch log, which a later resume rewrites."""
    decoder, module = DECODERS[key], decoders.modules[key]
    fresh = _WritesPlaceholders(tmp_path)
    _patch_fresh_decode(monkeypatch, decoders.flow, tmp_path, fresh)

    failed = _run_decode(module, decoder, tmp_path, save_dir=tmp_path / "staging")

    assert failed is None, _log(tmp_path, decoder)
    assert fresh.written_attrs is not None  # to_netcdf was called
    assert fresh.written_attrs["provenance"] == _logged_provenance(tmp_path, decoder)


@pytest.mark.parametrize("key", list(DECODERS))
def test_a_failed_write_leaves_no_final_file(decoders, monkeypatch, tmp_path, key):
    """A write that dies part way leaves nothing ``results_exist`` would trust."""
    decoder, module = DECODERS[key], decoders.modules[key]
    save_dir = tmp_path / "staging"
    _patch_fresh_decode(monkeypatch, decoders.flow, tmp_path, _FailsMidWrite(tmp_path))

    failed = _run_decode(module, decoder, tmp_path, save_dir=save_dir)

    assert failed == decoder.epoch_identifier
    stem = decoder.epoch_identifier
    assert sorted(path.name for path in save_dir.iterdir()) == [
        f"{stem}_detector.pkl",
        f"{stem}_results.nc",
    ]
    assert not decoding_files.results_exist(save_dir, stem, overwrite=False)


def test_write_atomically_keeps_the_old_file_when_a_write_fails(tmp_path):
    path = tmp_path / "testrat20990101_02_r1_HPC_detector.pkl"
    path.write_bytes(b"the previous run")

    def half_write(temporary: Path) -> None:
        _inside(temporary, tmp_path).write_bytes(b"trunc")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_atomically(path, half_write)

    assert path.read_bytes() == b"the previous run"
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_provenance_record_fields():
    record = provenance.provenance_record(REPO_ROOT)

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

    clean = provenance.provenance_record(tmp_path)
    assert clean["commit"] == git("rev-parse", "HEAD")
    assert clean["dirty"] is False

    (tmp_path / "uv.lock").write_text("edited")
    assert provenance.provenance_record(tmp_path)["dirty"] is True


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

    provenance.provenance_record(tmp_path)
    assert _sha256(index) == before

    # The control: a plain status does refresh and rewrite this index.
    git("status", "--porcelain")
    assert _sha256(index) != before


@pytest.mark.parametrize("key", list(DECODERS))
def test_epoch_log_has_one_provenance_line(decoders, monkeypatch, tmp_path, key):
    decoder, module = DECODERS[key], decoders.modules[key]

    def load_data(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the header is all this test reads")

    monkeypatch.setattr(decoders.flow, "load_data", load_data)

    _run_decode(module, decoder, tmp_path, save_dir=tmp_path / "staging")

    lines = _log(tmp_path, decoder).splitlines()
    at = [i for i, line in enumerate(lines) if "PROVENANCE " in line]
    assert len(at) == 1
    assert "START" in lines[at[0] - 1]
    payload = lines[at[0]].split("PROVENANCE ", 1)[1]
    logged = json.loads(payload)
    assert payload == json.dumps(logged, sort_keys=True)
    # Other work shares this tree, so dirty can change between the two calls.
    now = provenance.provenance_record(REPO_ROOT)
    assert set(logged) == set(now) == PROVENANCE_KEYS
    assert isinstance(logged["dirty"], bool)
    assert {k: v for k, v in logged.items() if k != "dirty"} == {
        k: v for k, v in now.items() if k != "dirty"
    }


# --------------------------------------------------------------------------
# Open files: Spyglass's NWB cache and the open-file limit
# --------------------------------------------------------------------------


class _ReleaseRecorder:
    """Stands in for ``release_nwb_files``: counts its calls, reports success."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return True


@pytest.mark.parametrize("outcome", ["ok", "fails"])
@pytest.mark.parametrize("key", list(DECODERS))
def test_run_decode_closes_spyglass_nwb_files_when_asked(
    decoders, monkeypatch, tmp_path, key, outcome
):
    """With ``release_nwb`` an epoch closes the NWB files Spyglass cached, pass or
    fail, and its log ends saying so."""
    decoder, module = DECODERS[key], decoders.modules[key]
    release = _ReleaseRecorder()
    monkeypatch.setattr(decoders.flow, "release_nwb_files", release)
    if outcome == "ok":
        _patch_fresh_decode(
            monkeypatch, decoders.flow, tmp_path, _WritesPlaceholders(tmp_path)
        )
    else:

        def load_data(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated: an NWB file could not be read")

        monkeypatch.setattr(decoders.flow, "load_data", load_data)

    failed = _run_decode(
        module, decoder, tmp_path, save_dir=tmp_path / "staging", release_nwb=True
    )

    assert failed == (None if outcome == "ok" else decoder.epoch_identifier)
    assert release.calls == 1
    log = _log(tmp_path, decoder).splitlines()
    assert log[-1].endswith("NWB files Spyglass held open closed: True")


@pytest.mark.parametrize("key", list(DECODERS))
def test_run_decode_leaves_spyglass_nwb_files_open_by_default(
    decoders, monkeypatch, tmp_path, key
):
    """Called on its own, an epoch may share its process with other readers."""
    decoder, module = DECODERS[key], decoders.modules[key]
    release = _ReleaseRecorder()
    monkeypatch.setattr(decoders.flow, "release_nwb_files", release)
    _patch_fresh_decode(
        monkeypatch, decoders.flow, tmp_path, _WritesPlaceholders(tmp_path)
    )

    failed = _run_decode(module, decoder, tmp_path, save_dir=tmp_path / "staging")

    assert failed is None, _log(tmp_path, decoder)
    assert release.calls == 0
    assert "NWB files" not in _log(tmp_path, decoder)


@pytest.mark.parametrize("key", list(DECODERS))
def test_a_failed_release_leaves_the_epochs_result(
    decoders, monkeypatch, tmp_path, key
):
    """The release is cleanup in a ``finally``: had it raised there, it would
    replace the epoch's return value and abort ``dask.compute``, and with it
    the batch. Spyglass's real ``release_nwb_files`` path runs here, against a
    stub whose ``close_nwb_files`` fails."""
    decoder, module = DECODERS[key], decoders.modules[key]
    helper = types.ModuleType("spyglass.utils.nwb_helper_fn")

    def close_nwb_files() -> None:
        raise OSError("simulated: close failed")

    helper.close_nwb_files = close_nwb_files
    utils = types.ModuleType("spyglass.utils")
    utils.nwb_helper_fn = helper
    monkeypatch.setitem(sys.modules, "spyglass.utils", utils)
    monkeypatch.setitem(sys.modules, "spyglass.utils.nwb_helper_fn", helper)
    save_dir = tmp_path / "staging"
    _patch_fresh_decode(
        monkeypatch, decoders.flow, tmp_path, _WritesPlaceholders(tmp_path)
    )

    failed = _run_decode(module, decoder, tmp_path, save_dir=save_dir, release_nwb=True)

    assert failed is None, _log(tmp_path, decoder)
    assert (save_dir / f"{decoder.epoch_identifier}_results.nc").exists()
    log = _log(tmp_path, decoder).splitlines()
    assert any(line.endswith("Done!") for line in log)
    assert log[-1].endswith("NWB files Spyglass held open closed: False")


# --------------------------------------------------------------------------
# HPC: the artifact mask, the encoding exclusion and the SWR source
# --------------------------------------------------------------------------


def _decode_position_info() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "head_position_x": 0.0,
            "head_position_y": 0.0,
            "head_speed": np.zeros(DECODE_TIME.size),
        },
        index=pd.Index(DECODE_TIME, name="time"),
    )


def _on_decode_grid(*intervals: tuple[float, float]) -> np.ndarray:
    """True at each decode bin inside a closed interval."""
    inside = np.zeros(DECODE_TIME.size, dtype=bool)
    for start, end in intervals:
        inside |= (DECODE_TIME >= start) & (DECODE_TIME <= end)
    return inside


def _logged_decode_masks(tmp_path: Path, decoder: Decoder) -> str:
    """The JSON of the epoch log's one ``DECODE MASKS`` line."""
    (line,) = [
        line for line in _log(tmp_path, decoder).splitlines() if "DECODE MASKS " in line
    ]
    return line.split("DECODE MASKS ", 1)[1]


@pytest.mark.parametrize("key", list(DECODERS))
def test_an_hpc_decode_masks_artifacts_and_leaves_swrs_out_of_encoding(
    decoders, monkeypatch, tmp_path, key
):
    """The artifact mask is ``is_missing``; every Kay event and the mask leave
    encoding; MUA detection gets the mask; the record names both runs."""
    decoder, module = DECODERS[key], decoders.modules[key]
    fresh = _WritesPlaceholders(tmp_path)
    loads, decodes = [], []

    def load_data(*args: Any, **kwargs: Any) -> dict[str, Any]:
        loads.append(kwargs)
        return {**_session_data(), "position_info": _decode_position_info()}

    signature = inspect.signature(decoders.flow.perform_decoding)

    def perform_decoding(*args: Any, **kwargs: Any) -> tuple:
        decodes.append(signature.bind(*args, **kwargs))
        return fresh, fresh, fresh

    monkeypatch.setattr(decoders.flow, "load_data", load_data)
    monkeypatch.setattr(decoders.flow, "perform_decoding", perform_decoding)
    monkeypatch.setattr(
        decoders.flow,
        "organize_results",
        lambda *args, **kwargs: (_GuardedFrame(tmp_path), None),
    )

    failed = _run_decode(module, decoder, tmp_path, save_dir=tmp_path / "staging")

    assert failed is None, _log(tmp_path, decoder)
    (load,) = loads
    np.testing.assert_allclose(
        load["mua_artifact_intervals"]["HPC"].to_numpy(), [MASK_INTERVAL]
    )
    # the ripple-power covariate is the source's: its kept tetrodes, the mask
    expected = swr_source.ripple_power_inputs(SWR_EPOCH)
    assert {
        k: v for k, v in load["ripple_power"].items() if k != "artifact_intervals"
    } == {k: v for k, v in expected.items() if k != "artifact_intervals"}
    np.testing.assert_allclose(
        load["ripple_power"]["artifact_intervals"].to_numpy(), [MASK_INTERVAL]
    )
    (bound,) = decodes
    assert bound.arguments["runner"] is module.RUNNER
    missing = _on_decode_grid(MASK_INTERVAL)
    np.testing.assert_array_equal(bound.arguments["is_missing"], missing)
    np.testing.assert_array_equal(
        bound.arguments["is_training"], ~_on_decode_grid(*SWR_EVENTS) & ~missing
    )
    record = json.loads(_logged_decode_masks(tmp_path, decoder))
    assert fresh.written_attrs["decode_masks"] == _logged_decode_masks(
        tmp_path, decoder
    )
    assert record["swr_inventory_run_id"] == RUN_ID
    assert record["artifact_run_id"] == ARTIFACT_RUN_ID
    assert record["n_missing_bins"] == int(missing.sum())
    assert record["missing_s"] == pytest.approx(missing.sum() * 0.002)
    removed = int((_on_decode_grid(*SWR_EVENTS) | missing).sum())
    assert record["encoding_removed_s"] == pytest.approx(removed * 0.002)


@pytest.mark.parametrize("unset", ["SWR_INVENTORY_RUN_ID", "ARTIFACT_INTERVALS_RUN_ID"])
@pytest.mark.parametrize("key", list(DECODERS))
def test_an_hpc_decode_refuses_to_run_without_both_runs(
    decoders, monkeypatch, tmp_path, key, unset
):
    """Fail closed: nothing is loaded, the epoch fails, and its log says why."""
    decoder, module = DECODERS[key], decoders.modules[key]
    monkeypatch.setattr(swr_source, unset, None)

    def load_data(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an HPC decode loaded data without both runs")

    monkeypatch.setattr(decoders.flow, "load_data", load_data)

    failed = _run_decode(module, decoder, tmp_path, save_dir=tmp_path / "staging")

    assert failed == decoder.epoch_identifier
    assert f"swr_source.{unset}" in _log(tmp_path, decoder)


@pytest.mark.parametrize("key", list(DECODERS))
def test_an_hpc_decode_refuses_an_inventory_masked_by_another_artifact_run(
    decoders, monkeypatch, tmp_path, key
):
    """Today's case: the artifact run is set, but the inventory is one made
    without the mask (R3's has no ``settings.artifacts``)."""
    decoder, module = DECODERS[key], decoders.modules[key]
    # the fixture's inventory, rewritten without the artifact run it was masked by
    write_swr_inventory(decoders.canonical, {SWR_EPOCH: SWR_EVENTS})

    def load_data(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an HPC decode loaded data on an unmasked inventory")

    monkeypatch.setattr(decoders.flow, "load_data", load_data)

    failed = _run_decode(module, decoder, tmp_path, save_dir=tmp_path / "staging")

    assert failed == decoder.epoch_identifier
    log = _log(tmp_path, decoder)
    assert f"The SWR inventory '{RUN_ID}' was masked with no artifact run" in log
    assert f"the configured artifact run is '{ARTIFACT_RUN_ID}'" in log


@pytest.mark.parametrize("key", list(DECODERS))
def test_main_refuses_an_hpc_batch_without_both_runs(
    decoders, monkeypatch, capsys, key
):
    """The batch stops before any cluster starts; a PFC batch is not refused."""
    module = decoders.modules[key]
    batches = []
    monkeypatch.setattr(module, "run_batch", lambda *args, **kw: batches.append(args))
    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", None)

    with pytest.raises(SystemExit) as stopped:
        module.main([])
    assert stopped.value.code == 2
    assert "swr_source.ARTIFACT_INTERVALS_RUN_ID" in capsys.readouterr().err
    assert batches == []

    module.main(["--brain_area", "mPFC"])
    assert len(batches) == 1


@pytest.mark.parametrize("brain_area", ["HPC", "mPFC"])
@pytest.mark.parametrize("key", list(DECODERS))
def test_an_hpc_decodes_swr_labels_are_the_sources(
    decoders, monkeypatch, key, brain_area
):
    """``organize_results`` writes the SWR source's labels and coverage into an
    HPC decode's ``data_over_time``, which no longer carries stored-ripple
    labels or counts; a PFC decode's frame is left as aggregated."""
    module = decoders.modules[key]
    aggregated = pd.DataFrame(
        {
            "SWR_event": 9,
            "n_SWR": 3,
            "animal": "testrat",
            "day": 20990101,
            "epoch": EPOCH_NAME,
            "head_speed": np.zeros(DECODE_TIME.size),
        },
        index=pd.Index(DECODE_TIME, name="time"),
    )
    _replace_runner(
        monkeypatch,
        module,
        aggregate=lambda *args, **kwargs: (aggregated.copy(), None),
    )
    results = xr.Dataset(coords={"time": DECODE_TIME})

    data_over_time, _ = decoders.flow.organize_results(
        module.RUNNER, {}, results, None, NWB_FILE_NAME, EPOCH_NAME, brain_area
    )

    if brain_area == "mPFC":
        pd.testing.assert_frame_equal(data_over_time, aggregated)
        return
    np.testing.assert_array_equal(
        data_over_time["SWR_event"],
        analysed_time.swr_event_labels(
            DECODE_TIME, SWR_EPOCH, speed=np.zeros(DECODE_TIME.size)
        ),
    )
    assert set(data_over_time["SWR_event"]) == {0, 1, 2}
    assert "n_SWR" not in data_over_time
    # excluded time, the step's interval in the one mask [103.7, 104.3]
    # widened by 0.3 s, is not observed, and nothing after it is
    np.testing.assert_array_equal(
        data_over_time["swr_observed"],
        analysed_time.swr_observed_mask(DECODE_TIME, SWR_EPOCH),
    )
    observed = data_over_time["swr_observed"].to_numpy()
    assert not observed[_on_decode_grid((103.4, 104.6))].any()
    assert observed[_on_decode_grid((104.61, 105.3))].all()
    assert data_over_time.attrs["swr_source"] == swr_source.swr_source_name()


@pytest.mark.parametrize("key", list(DECODERS))
def test_an_hpc_decode_keeps_every_event_it_decoded(decoders, monkeypatch, key):
    """The decode saves every event it decoded, the ones near the mask too; the
    analyses' read through the source drops those, each with its reason."""
    module = decoders.modules[key]
    near = _on_decode_grid((MASK_INTERVAL[1] + 0.1, MASK_INTERVAL[1] + 0.15))
    labels = np.where(near, 1, 0)
    columns = (
        {
            "non_local_event": labels,
            "Non-Local_Continuous_prob": 0.9,
            "Non-Local_Fragmented_prob": 0.0,
        }
        if key == "nonlocal"
        else {"MUA_event": labels, "Continuous_prob": 0.5, "Fragmented_prob": 0.5}
    )
    aggregated = pd.DataFrame(
        {
            "SWR_event": 0,
            "animal": "testrat",
            "day": 20990101,
            "epoch": EPOCH_NAME,
            "head_speed": np.zeros(DECODE_TIME.size),
            **columns,
        },
        index=pd.Index(DECODE_TIME, name="time"),
    )
    _replace_runner(
        monkeypatch,
        module,
        aggregate=lambda *args, **kwargs: (aggregated.copy(), None),
    )
    results = xr.Dataset(coords={"time": DECODE_TIME})

    data_over_time, _ = decoders.flow.organize_results(
        module.RUNNER, {}, results, None, NWB_FILE_NAME, EPOCH_NAME, "HPC"
    )

    column = next(iter(columns))
    np.testing.assert_array_equal(data_over_time[column], labels)
    assert not analysed_time.with_swr_source(data_over_time)[column].any()


def test_an_hpc_decode_checks_its_labels_at_its_own_threshold(decoders, monkeypatch):
    """The read on the decoder's own frame checks the non-local labels against
    the threshold they were made at, not the default."""
    module = decoders.modules["nonlocal"]
    labels = np.where(_on_decode_grid((101.0, 101.05)), 1, 0)
    aggregated = pd.DataFrame(
        {
            "SWR_event": 0,
            "animal": "testrat",
            "day": 20990101,
            "epoch": EPOCH_NAME,
            "head_speed": np.zeros(DECODE_TIME.size),
            "non_local_event": labels,
            "Non-Local_Continuous_prob": 0.6,  # above 0.5, below the 0.8 default
            "Non-Local_Fragmented_prob": 0.0,
        },
        index=pd.Index(DECODE_TIME, name="time"),
    )
    _replace_runner(
        monkeypatch,
        module,
        aggregate=lambda *args, **kwargs: (aggregated.copy(), None),
    )
    results = xr.Dataset(coords={"time": DECODE_TIME})

    data_over_time, _ = decoders.flow.organize_results(
        module.RUNNER, {}, results, None, NWB_FILE_NAME, EPOCH_NAME, "HPC", 0.5
    )

    np.testing.assert_array_equal(data_over_time["non_local_event"], labels)
    with pytest.raises(EventDefinitionError, match="probability-threshold"):
        decoders.flow.organize_results(
            module.RUNNER, {}, results, None, NWB_FILE_NAME, EPOCH_NAME, "HPC", 0.8
        )


@pytest.mark.parametrize("key", list(DECODERS))
def test_perform_decoding_passes_is_missing_to_the_decoder(decoders, monkeypatch, key):
    module = decoders.modules[key]
    calls = []

    class Detector:
        def estimate_parameters(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            return "results"

    monkeypatch.setattr(fit, "create_detector", lambda *args, **kwargs: Detector())
    monkeypatch.setattr(
        non_local_detector.environment, "Environment", lambda **kwargs: "environment"
    )
    is_training = np.array([True, False, True])
    is_missing = np.array([False, True, False])

    decoders.flow.perform_decoding(
        module.RUNNER,
        logging.getLogger(__name__),
        [np.array([0.1])],
        [np.zeros((1, 4))],
        _decode_position_info().iloc[:3],
        None,
        None,
        None,
        is_training,
        brain_area="HPC",
        is_missing=is_missing,
    )

    (kwargs,) = calls
    assert kwargs["is_missing"] is is_missing
    assert kwargs["is_training"] is is_training


@pytest.mark.parametrize("brain_area", ["HPC", "mPFC"])
@pytest.mark.parametrize("key", list(DECODERS))
def test_only_a_nonlocal_hpc_decode_keeps_its_log_likelihood(
    decoders, monkeypatch, key, brain_area
):
    """The NonLocal runner's HPC decode asks the detector for its per-bin
    log-likelihood, which the spike-evidence analysis reads from results.nc;
    the ContFrag runner's does not, nor does any PFC decode. The two runners
    differ here on purpose (each ``RUNNER.hpc_log_likelihood``)."""
    module = decoders.modules[key]
    calls = []

    class Detector:
        def estimate_parameters(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            return "results"

    monkeypatch.setattr(fit, "create_detector", lambda *args, **kwargs: Detector())
    monkeypatch.setattr(
        non_local_detector.environment, "Environment", lambda **kwargs: "environment"
    )

    decoders.flow.perform_decoding(
        module.RUNNER,
        logging.getLogger(__name__),
        [np.array([0.1])],
        [np.zeros((1, 4))],
        _decode_position_info().iloc[:3],
        None,
        None,
        None,
        np.ones(3, dtype=bool),
        brain_area=brain_area,
    )

    (kwargs,) = calls
    expected = (
        {"return_outputs": "log_likelihood"}
        if key == "nonlocal" and brain_area == "HPC"
        else {}
    )
    assert {k: v for k, v in kwargs.items() if k == "return_outputs"} == expected


@pytest.mark.parametrize("key", list(DECODERS))
def test_each_runner_aggregates_and_reloads_with_its_own_decode_functions(
    decoders, key
):
    """A runner whose ``aggregate`` or ``load_saved`` were swapped would build
    its frame from, or resume on, the other decode's results without a word.
    Each runner's record is pinned by name."""
    from continuum_swr_replay.analysis.contfrag.data_aggregation import (
        get_contfrag_data_over_time,
    )
    from continuum_swr_replay.analysis.contfrag.result_loading import (
        load_contfrag_results,
    )
    from continuum_swr_replay.analysis.detection.nonlocal_events import (
        get_data_over_time,
    )
    from continuum_swr_replay.decoding.results import (
        load_results,
    )

    expected = {
        "nonlocal": (get_data_over_time, load_results, "", True),
        "contfrag": (
            get_contfrag_data_over_time,
            load_contfrag_results,
            "_contfrag",
            False,
        ),
    }[key]
    runner = decoders.modules[key].RUNNER
    assert (
        runner.aggregate,
        runner.load_saved,
        runner.file_suffix,
        runner.hpc_log_likelihood,
    ) == expected


def _stand_in_batch(monkeypatch, tmp_path, *, has_cuda: bool, cluster: Callable):
    """``run_batch`` with `cluster` for the path `has_cuda` picks and no dask.

    Each epoch's call runs here, as it is built. The open-file limit is left
    alone unless the test replaces ``raise_open_file_limit`` itself.
    """
    _stand_in_clusters(monkeypatch, has_cuda=has_cuda, cluster=cluster)
    _no_dask_scheduler(monkeypatch)
    monkeypatch.setattr(batch, "epochs_to_run", _one_epoch)
    monkeypatch.setattr(batch, "raise_open_file_limit", lambda: (1024, 65536))
    # run_batch sets this variable and makes logs/ in the working directory;
    # monkeypatch restores both after the test.
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.99")
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize(
    ("has_cuda", "flags"),
    [(True, []), (False, ["--allow_cpu"])],
    ids=["cuda", "cpu_allowed"],
)
def test_run_batch_closes_nwb_files_only_on_one_task_workers(
    monkeypatch, tmp_path, has_cuda, flags
):
    """dask_cuda's workers run one epoch at a time, so each epoch closes the files
    it opened. LocalCluster's default workers run several epochs at once in
    threads that share Spyglass's cache, so their epochs leave it open."""
    clusters: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    def cluster(**kwargs: Any) -> object:
        clusters.append(kwargs)
        return object()

    _stand_in_batch(monkeypatch, tmp_path, has_cuda=has_cuda, cluster=cluster)
    args = _parse([*flags, "--save_dir", "staging", "--local_scratch_dir", "scratch"])

    batch.run_batch(lambda **kwargs: calls.append(kwargs), args)

    (kwargs,) = calls
    assert kwargs["release_nwb"] is has_cuda
    (cluster_kwargs,) = clusters
    if has_cuda:
        assert cluster_kwargs["threads_per_worker"] == 1
    else:
        assert "threads_per_worker" not in cluster_kwargs  # LocalCluster's default


@pytest.mark.parametrize(
    ("flags", "pending_only"),
    [([], False), (["--pending-only"], True)],
    ids=["analysed", "pending_only"],
)
def test_run_batch_selects_its_epochs_through_the_register(
    monkeypatch, tmp_path, flags, pending_only
):
    """The batch decodes the register's analysed epochs, or with
    ``--pending-only`` its pending ones alone: never ``get_epoch_info`` as it is."""
    asked: list[bool] = []
    calls: list[dict[str, Any]] = []

    def epochs_to_run(pending_only: bool = False) -> pd.DataFrame:
        asked.append(pending_only)
        return _one_epoch()

    _stand_in_batch(
        monkeypatch, tmp_path, has_cuda=True, cluster=lambda **kwargs: object()
    )
    monkeypatch.setattr(batch, "epochs_to_run", epochs_to_run)
    args = _parse([*flags, "--save_dir", "staging", "--local_scratch_dir", "scratch"])

    batch.run_batch(lambda **kwargs: calls.append(kwargs), args)

    assert asked == [pending_only]
    assert [(call["nwb_file_name"], call["epoch_name"]) for call in calls] == [
        (NWB_FILE_NAME, EPOCH_NAME)
    ]
    assert not hasattr(batch, "get_epoch_info")


def test_pending_epochs_are_never_decoded_into_the_canonical_folder(
    monkeypatch, tmp_path
):
    """Readers that glob the canonical folder would see a pending decode there,
    so ``--pending-only`` must name a staging ``--save_dir``. Refused before any
    epoch is read or any cluster starts."""

    def must_not_select(pending_only: bool = False) -> pd.DataFrame:
        raise AssertionError("epochs were selected")

    _stand_in_batch(monkeypatch, tmp_path, has_cuda=True, cluster=_no_cluster)
    monkeypatch.setattr(batch, "epochs_to_run", must_not_select)
    canonical = tmp_path / "Processed-Data"
    monkeypatch.setenv(PROCESSED_DATA_DIR_ENV, str(canonical))
    args = _parse(
        ["--pending-only", "--save_dir", str(canonical), "--local_scratch_dir", "s"]
    )

    with pytest.raises(ValueError, match="staging"):
        batch.run_batch(lambda **kwargs: None, args)


def test_run_batch_raises_the_open_file_limit_before_its_cluster(
    monkeypatch, tmp_path, capsys
):
    """Worker processes inherit the limit only if it is raised before they start."""
    order: list[str] = []

    def cluster(**kwargs: Any) -> object:
        order.append("cluster")
        return object()

    def raise_open_file_limit() -> tuple[int, int]:
        order.append("limit")
        return 1024, 65536

    _stand_in_batch(monkeypatch, tmp_path, has_cuda=True, cluster=cluster)
    monkeypatch.setattr(batch, "raise_open_file_limit", raise_open_file_limit)
    args = _parse(["--save_dir", "staging", "--local_scratch_dir", "scratch"])

    batch.run_batch(lambda **kwargs: None, args)

    assert order == ["limit", "cluster"]
    assert "open-file soft limit 1024 -> 65536" in capsys.readouterr().out
