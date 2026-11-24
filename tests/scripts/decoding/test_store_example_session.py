"""The stored example session: written whole from ``load_data``'s result, read
back as it was, and refused when it is not current."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition import swr_source
from continuum_swr_replay.paths import EXAMPLE_SESSION_DIR
from continuum_swr_replay.reported import inputs as stored_inputs
from scripts.decoding import store_example_session as runner

EPOCH = "wilbur20210329_02_r1_HPC"


def _data() -> dict:
    """A planted session: a ripple trace on a 1 ms grid and three tetrodes,
    the last with no spike."""
    time = np.arange(0.0, 0.01, 0.001)
    return {
        "zscored_ripple_consensus_trace": pd.Series(np.sin(time * 300), index=time),
        "spike_times": {
            "HPC": [np.array([0.001, 0.004]), np.array([0.002]), np.array([])],
            "mPFC": [np.array([0.003])],
        },
    }


@pytest.fixture
def stored(processed_data_root, monkeypatch):
    """The planted session stored by the runner under the data root."""
    monkeypatch.setattr(runner, "load_data", lambda *args, **kwargs: _data())
    monkeypatch.setattr(runner, "ripple_power_inputs", lambda identifier: None)
    monkeypatch.setattr(runner, "sha256_file", lambda path: "d" * 64)
    runner.main([])
    return processed_data_root / EXAMPLE_SESSION_DIR / EPOCH


def test_the_stored_session_reads_back_as_load_data_gave_it(stored):
    trace, spikes = stored_inputs.read_example_session(stored, EPOCH, "d" * 64)
    planted = _data()
    pd.testing.assert_series_equal(
        trace, planted["zscored_ripple_consensus_trace"], check_names=False
    )
    assert len(spikes) == 3  # the tetrode with no spike keeps its place
    for got, expected in zip(spikes, planted["spike_times"]["HPC"], strict=True):
        np.testing.assert_array_equal(got, expected)
    manifest = json.loads((stored / "manifest.json").read_text())
    assert manifest["epoch_id"] == EPOCH
    assert manifest["swr_source"] == swr_source.swr_source_name()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"decode_sha256": "0" * 64}, "another decode frame"),
        ({"swr_source": "kay@old+artifacts@none"}, "records the SWR source"),
        ({"epoch_id": "other_02_r1_HPC"}, "records the epoch"),
    ],
    ids=["another decode", "another SWR source", "another epoch"],
)
def test_a_stored_session_that_is_not_current_is_refused(stored, change, message):
    path = stored / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(ValueError, match=message):
        stored_inputs.read_example_session(stored, EPOCH, "d" * 64)


def test_a_missing_session_is_refused(processed_data_root):
    with pytest.raises(ValueError, match="holds no stored session"):
        stored_inputs.read_example_session(
            processed_data_root / EXAMPLE_SESSION_DIR / EPOCH, EPOCH, "d" * 64
        )
