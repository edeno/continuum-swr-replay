"""The runners' per-epoch loop and task logging (``runs.batch``).

``run_batch``, ``release_nwb_files`` and ``raise_open_file_limit`` are tested in
``test_decoding_save_dir`` and ``test_open_files``; ``run_lock`` through the
SWR runner's tests.
"""

from __future__ import annotations

import functools

import dask
import pytest
from dask.distributed import LocalCluster, get_worker

from continuum_swr_replay.runs import batch


def _cluster(tmp_path):
    return functools.partial(
        LocalCluster,
        n_workers=2,
        processes=False,
        # A thread worker measures the whole test process, which can pass
        # dask's memory limit late in the suite and pause the worker for good.
        memory_limit=0,
        dashboard_address=":0",
        local_directory=str(tmp_path / "dask-scratch"),
    )


def _task_key(epoch: str) -> dict:
    """In a worker: the record names the task's dask key."""
    if epoch == "dies":
        raise RuntimeError("simulated: the worker died")
    return {"status": "ok", "key": get_worker().get_current_task()}


@pytest.mark.parametrize(("key_prefix", "key"), [(None, "{}"), ("swr", "swr-{}")])
def test_each_epoch_is_one_keyed_task_and_a_lost_one_is_recorded(
    tmp_path, key_prefix, key
):
    """A task that raises instead of returning its record is recorded as
    failed, and the other epoch still finishes."""
    with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
        records = dict(
            batch.run_epochs(
                {"a_HPC": ("lives",), "b_HPC": ("dies",)},
                _task_key,
                _cluster(tmp_path),
                key_prefix=key_prefix,
            )
        )

    assert records["a_HPC"] == {"status": "ok", "key": key.format("a_HPC")}
    failed = records["b_HPC"]
    assert failed["status"] == "failed"
    assert failed["error"] == repr(RuntimeError("simulated: the worker died"))
    assert "simulated: the worker died" in failed["traceback"]


def test_a_lost_epochs_record_can_be_the_runners_own(tmp_path):
    def lost(error: Exception) -> dict:
        return {"status": "failed", "error": repr(error), "git_commit": "c0ffee"}

    with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
        records = dict(
            batch.run_epochs(
                {"b_HPC": ("dies",)}, _task_key, _cluster(tmp_path), lost=lost
            )
        )

    assert records["b_HPC"]["git_commit"] == "c0ffee"


def test_no_epochs_make_no_cluster():
    def cluster_factory():
        raise AssertionError("a cluster was made for no epochs")

    assert list(batch.run_epochs({}, _task_key, cluster_factory)) == []


@pytest.mark.parametrize("log_provenance", [True, False])
@pytest.mark.parametrize("status", ["ok", "failed"])
def test_an_epoch_task_logs_its_outcome_and_closes_the_nwb_files(
    tmp_path, monkeypatch, log_provenance, status
):
    closed = []
    monkeypatch.setattr(batch, "release_nwb_files", lambda: closed.append(1) or True)
    record = {
        "status": status,
        "provenance": {"commit": "abc", "dirty": False},
        "traceback": "Traceback: simulated",
    }
    given = []

    def run(log):
        given.append(log.name)
        return record

    returned = batch.run_epoch_logged(
        run,
        log_name="e_HPC_test",
        log_path=tmp_path / "logs" / "e_HPC_test.log",
        start="e_HPC: run {'output_dir': 'x'}",
        describe=lambda r: f"{r['status']}: described",
        log_provenance=log_provenance,
    )

    assert returned is record
    assert given == ["e_HPC_test"]
    assert closed == [1]
    lines = (tmp_path / "logs" / "e_HPC_test.log").read_text().splitlines()
    messages = [line.split(" - ", 3)[-1] for line in lines]
    expected = [
        "e_HPC: run {'output_dir': 'x'}",
        "NWB files Spyglass held open closed: True",
        *(['PROVENANCE {"commit": "abc", "dirty": false}'] if log_provenance else []),
        f"{status}: described",
    ]
    if status == "failed":
        expected.append("Traceback: simulated")
    assert messages == expected
    outcome = lines[messages.index(f"{status}: described")]
    assert (" - ERROR " in outcome) is (status == "failed")


def test_an_epoch_task_that_raises_still_closes_the_nwb_files(tmp_path, monkeypatch):
    closed = []
    monkeypatch.setattr(batch, "release_nwb_files", lambda: closed.append(1) or True)

    def run(log):
        raise RuntimeError("simulated")

    with pytest.raises(RuntimeError, match="simulated"):
        batch.run_epoch_logged(
            run,
            log_name="e_HPC_raises",
            log_path=tmp_path / "e.log",
            start="start",
            describe=str,
            log_provenance=True,
        )
    assert closed == [1]
