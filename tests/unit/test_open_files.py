"""The batch runners' two guards against running out of file descriptors.

Spyglass caches every NWB file it opens in ``spyglass.utils.nwb_helper_fn``
and never closes one, so a long-lived dask
worker leaks one descriptor per distinct file it reads (Spyglass issue #1442).
``release_nwb_files`` closes them after an epoch; ``raise_open_file_limit``
lifts the soft limit a tmux session hands down (often 1024) before any worker
starts.

Offline: the first test stubs Spyglass in ``sys.modules``; the second opens a
small NWB file it writes under ``tmp_path`` through the installed Spyglass's own
cache, which needs no database. The limit is changed only in a subprocess, so
this process's own limit is untouched.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import subprocess
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

from continuum_swr_replay.runs import batch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _stub_spyglass(
    monkeypatch, open_files: dict, close_error: Exception | None = None
) -> list[str]:
    """Stub ``spyglass.utils.nwb_helper_fn``, whose ``close_nwb_files`` empties
    `open_files`, or raises `close_error` when one is given. Returns the list
    ``close_nwb_files`` appends to on each call.
    """
    calls: list[str] = []
    helper = types.ModuleType("spyglass.utils.nwb_helper_fn")

    def close_nwb_files() -> None:
        calls.append("close_nwb_files")
        if close_error is not None:
            raise close_error
        open_files.clear()

    helper.close_nwb_files = close_nwb_files
    utils = types.ModuleType("spyglass.utils")
    utils.nwb_helper_fn = helper
    spyglass = types.ModuleType("spyglass")
    spyglass.utils = utils
    monkeypatch.setitem(sys.modules, "spyglass", spyglass)
    monkeypatch.setitem(sys.modules, "spyglass.utils", utils)
    monkeypatch.setitem(sys.modules, "spyglass.utils.nwb_helper_fn", helper)
    return calls


def test_release_closes_the_spyglass_cache_and_says_so(monkeypatch):
    open_files = {f"/data/file_{i}.nwb": ("io", "nwbfile") for i in range(3)}
    calls = _stub_spyglass(monkeypatch, open_files)

    assert batch.release_nwb_files() is True
    assert calls == ["close_nwb_files"]
    assert open_files == {}

    assert batch.release_nwb_files() is True
    assert calls == ["close_nwb_files"] * 2


def test_a_release_that_fails_warns_and_returns_false(monkeypatch, caplog):
    """Cleanup: its failure is logged, never raised over the epoch's result."""
    open_files = {"/data/file_0.nwb": ("io", "nwbfile")}
    calls = _stub_spyglass(
        monkeypatch, open_files, close_error=OSError("simulated: close failed")
    )

    with caplog.at_level(logging.WARNING, logger=batch.__name__):
        assert batch.release_nwb_files() is False

    assert calls == ["close_nwb_files"]
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "simulated: close failed" in record.getMessage()


def _limits_after(monkeypatch, soft: int, hard: int, target: int) -> tuple:
    """``raise_open_file_limit(target)`` against stand-in limits; nothing is set.

    Returns what it returned and what it asked ``setrlimit`` for (None if it
    did not ask).
    """
    asked = []
    monkeypatch.setattr(batch.resource, "getrlimit", lambda resource: (soft, hard))
    monkeypatch.setattr(
        batch.resource,
        "setrlimit",
        lambda resource, limits: asked.append(limits),
    )
    returned = batch.raise_open_file_limit(target)
    return returned, (asked[0] if asked else None)


def test_an_unlimited_hard_limit_raises_the_soft_limit_to_the_target(monkeypatch):
    infinity = resource.RLIM_INFINITY

    assert _limits_after(monkeypatch, 1024, infinity, 65536) == (
        (1024, 65536),
        (65536, infinity),
    )


def test_an_unlimited_soft_limit_is_never_lowered(monkeypatch):
    infinity = resource.RLIM_INFINITY

    assert _limits_after(monkeypatch, infinity, infinity, 65536) == (
        (infinity, infinity),
        None,
    )


def _open_descriptors_to(path: Path) -> int:
    """How many of this process's file descriptors point at `path`."""
    fd_dir = Path("/proc/self/fd")
    count = 0
    for fd in fd_dir.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:  # closed between listing and reading
            continue
        count += target == str(path)
    return count


@pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(), reason="needs /proc to count descriptors"
)
def test_release_closes_a_file_the_installed_spyglass_cached(tmp_path):
    """Against the installed Spyglass: the file it cached is really closed."""
    pynwb = pytest.importorskip("pynwb")
    nwb_helper_fn = pytest.importorskip("spyglass.utils.nwb_helper_fn")

    path = (tmp_path / "tiny.nwb").resolve()
    nwbfile = pynwb.NWBFile(
        session_description="fd leak test",
        identifier="fdleak",
        session_start_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)

    batch.release_nwb_files()  # start from an empty cache
    nwb_helper_fn.get_nwb_file(str(path))
    assert _open_descriptors_to(path) == 1

    assert batch.release_nwb_files() is True
    assert _open_descriptors_to(path) == 0


_LIMITS = """
import json, resource, sys
from continuum_swr_replay.runs.batch import raise_open_file_limit

soft, hard = (int(value) for value in sys.argv[1:3])
resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
calls = []
for target in (int(value) for value in sys.argv[3:]):
    returned = raise_open_file_limit(target)
    calls.append([list(returned), resource.getrlimit(resource.RLIMIT_NOFILE)[0]])
print(json.dumps(calls))
"""


def _limits_in_a_subprocess(soft: int, hard: int, *targets: int) -> list:
    """Set the limits in a fresh interpreter, call the function once per target.

    Returns ``[[old_soft, new_soft], soft_after]`` per call.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _LIMITS, str(soft), str(hard), *map(str, targets)],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
    )
    calls: list = json.loads(completed.stdout.strip().splitlines()[-1])
    return calls


def test_raise_open_file_limit_raises_to_the_target_and_never_lowers():
    calls = _limits_in_a_subprocess(256, 4096, 1024, 512, 1024)

    assert calls == [
        [[256, 1024], 1024],  # raised to the target
        [[1024, 1024], 1024],  # a lower target does not lower it
        [[1024, 1024], 1024],  # already there: unchanged
    ]


def test_raise_open_file_limit_stops_at_the_hard_limit():
    calls = _limits_in_a_subprocess(256, 2048, 65536)

    assert calls == [[[256, 2048], 2048]]
