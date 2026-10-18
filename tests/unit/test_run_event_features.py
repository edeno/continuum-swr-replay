"""The event-features runner's row bookkeeping and its imports' side effects.

The runner end to end is in ``tests/integration/test_run_event_features.py``.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import scripts.qa.run_event_features as runner
from continuum_swr_replay.parameters import ROOT_DIR


def test_the_row_mask_holds_each_intervals_rows_and_no_others():
    mask = runner._row_mask(10, np.array([1, 3, 8]), np.array([3, 0, 2]))

    np.testing.assert_array_equal(np.flatnonzero(mask), [1, 2, 3, 8, 9])


def test_overlapping_intervals_mark_their_rows_once():
    mask = runner._row_mask(6, np.array([0, 2]), np.array([4, 3]))

    np.testing.assert_array_equal(np.flatnonzero(mask), [0, 1, 2, 3, 4])


def test_the_evidence_rows_are_the_events_rows_never_the_last_sample():
    time = 100.0 + 0.002 * np.arange(20)
    events = pd.DataFrame(
        {
            "event_start_s": [time[2], time[10], time[19]],
            "event_end_s": [time[6], time[-1], time[-1]],
        }
    )

    rows = runner.evidence_rows(time, events)

    # rows 2 to 6, and 10 to 18: the grid's last sample is no row
    np.testing.assert_array_equal(np.flatnonzero(rows), [2, 3, 4, 5, 6, *range(10, 19)])


def test_a_row_of_a_scored_interval_not_requested_is_refused():
    rows = np.zeros(10, dtype=bool)
    rows[2:6] = True
    intervals = pd.DataFrame({"first_row": [2, 4], "n_rows": [3, 3]})

    with pytest.raises(RuntimeError, match="1 rows of scored intervals"):
        runner._require_requested(rows, intervals)
    runner._require_requested(rows, intervals.iloc[:1])
    # an interval with no row asks for none
    runner._require_requested(rows, pd.DataFrame({"first_row": [9], "n_rows": [0]}))


def test_importing_the_runner_installs_no_ignore_filter_for_a_warning_class():
    """The spike-evidence script's filters ignore every User, Future and
    Deprecation warning; the runner keeps them out of the process."""
    code = (
        "import warnings\n"
        "before = list(warnings.filters)\n"
        "import scripts.qa.run_event_features\n"
        "blanket = [f for f in warnings.filters if f not in before"
        " and f[0] == 'ignore' and f[1] is None"
        " and f[2] in (UserWarning, FutureWarning, DeprecationWarning)"
        " and f[3] is None]\n"
        "assert not blanket, blanket\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT_DIR,
        check=True,
        env={"JAX_PLATFORMS": "cpu", "PATH": ""},
    )


def test_a_failure_after_the_tables_keeps_no_field_of_an_ok_record(
    tmp_path, monkeypatch
):
    """The tables are made, then writing them fails: the record is a failure
    with the decode it read, and nothing of the ok record it nearly was."""
    decode = tmp_path / "decode.pkl"
    decode.write_bytes(b"decode")
    identifier = "sim20200101_02_r1_HPC"
    monkeypatch.setattr(runner.comparison, "decode_path", lambda _: decode)
    found = {
        "memoryless_available": True,
        "marks": {"sha256": "m"},
        "n_grid_rows": 10,
        "n_evidence_rows": 5,
        "likelihood_s": 1.0,
        "counts": {"events": {"kay": 1}},
    }
    calls = []

    def failing_write(identifier, output_dir, tables, stamp):
        calls.append(tables)
        if tables:
            raise OSError("disk full")
        return {}

    monkeypatch.setattr(runner, "replace_outputs", failing_write)
    monkeypatch.setattr(runner, "epoch_tables", lambda *a, **k: (found, {"x": 1}))

    record = runner.run_epoch(
        "sim20200101_.nwb",
        "02_r1",
        output_dir=tmp_path,
        inventory_dir=tmp_path,
        memoryless_dir=tmp_path,
        decode_sha256={identifier: runner.sha256_file(decode)},
        stop_table=tmp_path / "stops.parquet",
        provenance={"commit": "c" * 40},
    )

    assert record["status"] == "failed"
    assert "disk full" in record["error"]
    assert record["decode"]["path"] == str(decode)
    assert not set(found) & set(record)
    assert "likelihood_share" not in record
    assert calls[-1] == {}
