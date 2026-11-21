"""The bin-facts runner's wiring: what ``run_epoch`` reads, checks and writes.
The decode frame's time, the spike loader and the session cache are replaced at
their boundaries; the restriction run's record is made by hand."""

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.runs.io import output_path
from scripts.event_evidence import measure_bin_facts as runner

EPOCH = "wilbur20210329_02_r1_HPC"
TIME = 50.0 + 0.002 * np.arange(6)
SPIKES = [np.array([50.0, 50.0031]), np.array([50.005, 50.0099, 60.0])]
VISITS = pd.DataFrame({"arrival_time": [50.002], "departure_time": [50.006]})


@pytest.fixture
def folders(tmp_path, monkeypatch):
    restriction = tmp_path / "restriction"
    restriction.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(runner, "decode_time", lambda epoch_id, data_dir: TIME)
    monkeypatch.setattr(
        runner, "well_visits", lambda epoch_id: (VISITS, tmp_path / "cache.pkl")
    )
    (tmp_path / "cache.pkl").write_bytes(b"x")
    terms = types.ModuleType("terms")
    terms.load_marks_and_training_mask = lambda epoch_id: {
        "time": TIME,
        "spike_times": SPIKES,
    }
    monkeypatch.setitem(
        sys.modules,
        "continuum_swr_replay.analysis.event_evidence.state_evidence.terms",
        terms,
    )
    monkeypatch.setattr(
        runner, "decode_paths", lambda epoch_id, data_dir: types.SimpleNamespace(
            data_over_time=tmp_path / "cache.pkl"
        )
    )  # fmt: skip
    return tmp_path, restriction, output


def _record(restriction, n_spikes: int) -> None:
    output_path(EPOCH, restriction, runner.RESTRICTION_NAME).with_suffix(
        ".json"
    ).write_text(json.dumps({"n_spikes": n_spikes}))


def test_run_epoch_writes_each_bins_spikes_and_visit_flag(folders):
    data_dir, restriction, output = folders
    _record(restriction, 4)
    record = runner.run_epoch(EPOCH, data_dir, restriction, output)
    table = pd.read_parquet(output_path(EPOCH, output, runner.OUTPUT_NAME))
    # spikes at 50.0 -> 0, 50.0031 -> 1, 50.005 -> 2, 50.0099 -> 4 (the last
    # bin, 5, is never reached); 60.0 is outside the grid
    np.testing.assert_array_equal(table["n_spikes"], [1, 1, 1, 0, 1, 0])
    np.testing.assert_array_equal(
        table["in_well_visit"], [False, True, True, False, False, False]
    )
    assert record["n_spikes"] == 4 and record["restriction_n_spikes"] == 4
    assert record["n_bins_in_well_visit"] == 2
    assert record["n_spikes_in_well_visit"] == 2
    json.dumps(record, allow_nan=False)


def test_a_spike_count_other_than_the_restriction_runs_raises(folders):
    data_dir, restriction, output = folders
    _record(restriction, 5)
    with pytest.raises(ValueError, match="restriction run"):
        runner.run_epoch(EPOCH, data_dir, restriction, output)


def test_a_time_grid_that_is_not_the_decodes_raises(folders, monkeypatch):
    data_dir, restriction, output = folders
    _record(restriction, 4)
    monkeypatch.setattr(runner, "decode_time", lambda epoch_id, data_dir: TIME + 1e-3)
    with pytest.raises(ValueError, match="decode frame"):
        runner.run_epoch(EPOCH, data_dir, restriction, output)
