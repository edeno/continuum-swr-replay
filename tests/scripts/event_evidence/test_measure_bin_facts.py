"""The bin-facts runner's wiring: what ``run_epoch`` reads, checks and writes.
The decode frame's time, the spike loader and the well visits are replaced at
their boundaries; the restriction run's record (a fold fit's, which counts no
spikes) is made by hand."""

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_record,
)
from continuum_swr_replay.paths import (
    BIN_FACTS_NAME,
    EVENT_NONLOCAL_RESTRICTION_DIR,
    NONLOCAL_RESTRICTION_NAME,
)
from continuum_swr_replay.runs.io import output_path
from scripts.event_evidence import measure_bin_facts as runner
from tests.support._session_cache_fixtures import write_session_cache_entries

EPOCH = "wilbur20210329_02_r1_HPC"
TIME = 50.0 + 0.002 * np.arange(6)
SPIKES = [np.array([50.0, 50.0031]), np.array([50.005, 50.0099, 60.0])]
VISITS = (np.array([50.002]), np.array([50.006]))


@pytest.fixture
def folders(tmp_path, monkeypatch):
    restriction = tmp_path / "restriction"
    restriction.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(runner, "decode_time", lambda epoch_id, data_dir: TIME)
    monkeypatch.setattr(runner, "well_visit_times", lambda epoch_id: VISITS)
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


def _record(restriction) -> None:
    """A fold-fit record: it counts no spikes."""
    output_path(EPOCH, restriction, NONLOCAL_RESTRICTION_NAME).with_suffix(
        ".json"
    ).write_text(json.dumps({"status": "ok", "fit": "folds", "n_bins": 6}))


def test_run_epoch_writes_each_bins_spikes_and_visit_flag(folders):
    data_dir, restriction, output = folders
    _record(restriction)
    record = runner.run_epoch(EPOCH, data_dir, restriction, output)
    table = pd.read_parquet(output_path(EPOCH, output, BIN_FACTS_NAME))
    # spikes at 50.0 -> 0, 50.0031 -> 1, 50.005 -> 2, 50.0099 -> 4 (the last
    # bin, 5, is never reached); 60.0 is outside the grid
    np.testing.assert_array_equal(table["n_spikes"], [1, 1, 1, 0, 1, 0])
    np.testing.assert_array_equal(
        table["in_well_visit"], [False, True, True, False, False, False]
    )
    assert record["n_spikes"] == 4
    assert record["n_bins_in_well_visit"] == 2
    assert record["n_spikes_in_well_visit"] == 2
    assert record["n_well_visits"] == 1
    # the record the earlier runs wrote, less the restriction run's spike count
    assert list(record) == [
        "status",
        "n_bins",
        "n_spikes",
        "n_well_visits",
        "n_bins_in_well_visit",
        "n_spikes_in_well_visit",
        "epoch_inputs",
        "seconds",
    ]
    assert list(record["epoch_inputs"]) == ["data_over_time", "restriction_record"]
    json.dumps(record, allow_nan=False)


def test_a_time_grid_that_is_not_the_decodes_raises(folders, monkeypatch):
    data_dir, restriction, output = folders
    _record(restriction)
    monkeypatch.setattr(runner, "decode_time", lambda epoch_id, data_dir: TIME + 1e-3)
    with pytest.raises(ValueError, match="decode frame"):
        runner.run_epoch(EPOCH, data_dir, restriction, output)


def test_the_header_records_the_session_cache_of_every_epoch_of_the_restriction_run(
    processed_data_root, monkeypatch
):
    """The well visits are read from the session cache: the run records the
    entries of every epoch it is over (the restriction run's ok epochs, not only
    those of ``--epochs``), in its inputs, where
    ``prepare_data.session_cache_staleness`` reads it from the gathered manifest."""
    epochs = [EPOCH, "wilbur20210329_04_r2_HPC"]
    write_session_cache_entries(epochs)
    restriction = processed_data_root / EVENT_NONLOCAL_RESTRICTION_DIR
    restriction.mkdir()
    (restriction / "manifest.json").write_text(
        json.dumps({"epochs": dict.fromkeys(epochs, {"status": "ok"})})
    )
    monkeypatch.setattr(runner, "provenance_record", lambda root: {})
    headers = []
    monkeypatch.setattr(
        runner, "gather_manifest", lambda output, name, header: headers.append(header)
    )
    monkeypatch.setattr(
        runner, "run", lambda **kwargs: headers.append(kwargs["header"])
    )
    runner.main(["--gather"])
    runner.main(["--epochs", EPOCH])
    assert headers[0] == headers[1]
    # the header the run finishing under the earlier code wrote, key for key
    assert list(headers[0]) == ["produced_by", "provenance", "settings", "inputs"]
    assert headers[0]["settings"] == {
        "binning": "np.digitize(t, time[1:-1]), time[0] <= t <= time[-1]"
    }
    assert headers[0]["inputs"] == {"session_cache": session_cache_record(epochs)}
    assert "subset" not in headers[0] and "subset" not in headers[0]["settings"]
