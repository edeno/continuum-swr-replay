"""The held-out tetrode runner's wiring: the folds and margin ``run_epoch``
hands the scores, what it records, and the header's settings.

``epoch_inputs``, the windows and ``holdout_scores`` are replaced at their
boundaries; the frame, the trials and the scores are made by hand.
"""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.decoding.files import DecodePaths
from continuum_swr_replay.paths import (
    EVENT_PARTICIPATION_DIR,
    PARTICIPATION_NAME,
    TETRODE_HOLDOUT_NAME,
)
from continuum_swr_replay.runs.io import output_path
from scripts.event_evidence import score_tetrode_holdout as runner
from scripts.event_evidence._per_event_run import EpochInputs, trials_input

EPOCH = "wilbur20210329_02_r1_HPC"
N = 12
TIME = 100.0 + 0.002 * np.arange(N)
# Six trials: departures at bins 0, 2, 4, 6, 8 and 10, to wells 2, 3, 2, 3, 2, 3,
# each ending at the next departure and the last at bin 11, so bin 11 is the
# time after the last trial.
TRIALS = pd.DataFrame(
    {
        "movement_start_time": TIME[[0, 2, 4, 6, 8, 10]],
        "end_time": TIME[[2, 4, 6, 8, 10, 11]],
        "to_well": [2, 3, 2, 3, 2, 3],
    }
)
WINDOWS = pd.DataFrame(
    {
        "window_id": [0, 1],
        "s": [1, 6],
        "e": [3, 8],
        "event_id": ["a", "a"],
        "kind": ["event", "control"],
        "group": ["no SWR", "no SWR"],
    }
)
PARTICIPATION = pd.DataFrame({"event_id": ["a"], "control_s": [6]})


@pytest.fixture
def patched(tmp_path, monkeypatch):
    """Patch the runner's boundaries; return a function that runs the epoch
    and returns its record and what ``holdout_scores`` was given."""
    paths = DecodePaths(
        tmp_path / "dot.pkl",
        tmp_path / "results.nc",
        tmp_path / "detector.pkl",
        tmp_path / "environment.pkl",
    )
    participation = output_path(
        EPOCH, tmp_path / EVENT_PARTICIPATION_DIR, PARTICIPATION_NAME
    )
    participation.parent.mkdir()
    PARTICIPATION.to_parquet(participation)
    frame = pd.DataFrame(
        {
            "head_position_x": np.arange(N, dtype=float),
            "head_position_y": -np.arange(N, dtype=float),
            "linear_position": 2.0 * np.arange(N, dtype=float),
        },
        index=pd.Index(TIME, name="time"),
    )
    loaded = {
        "spike_times": [np.array([100.001])],
        "marks": [np.ones((1, 4))],
        "is_training": np.ones(N, dtype=bool),
        "is_missing": np.zeros(N, dtype=bool),
    }
    detector = object()
    monkeypatch.setattr(runner, "read_swr_table", lambda path, columns: "events")

    def epoch_windows(time, events, participation):
        np.testing.assert_array_equal(time, TIME)
        assert events == "events"
        pd.testing.assert_frame_equal(participation, PARTICIPATION)
        return WINDOWS

    monkeypatch.setattr(runner, "epoch_windows", epoch_windows)

    def run():
        given = {}

        def epoch_inputs(epoch_id, data_dir, *, trials):
            given["trials"] = trials
            return EpochInputs(
                paths,
                frame,
                frame.index.to_numpy(),
                loaded | {"trials": TRIALS},
                detector,
            )

        def holdout_scores(
            d,
            time,
            position,
            linear_position,
            spikes,
            marks,
            training,
            missing,
            windows,
            *,
            folds,
            margin_bins,
            log,
        ):
            assert d is detector
            np.testing.assert_array_equal(time, TIME)
            np.testing.assert_array_equal(
                position, frame[["head_position_x", "head_position_y"]].to_numpy()
            )
            np.testing.assert_array_equal(linear_position, 2.0 * np.arange(N))
            assert spikes is loaded["spike_times"] and marks is loaded["marks"]
            assert training is loaded["is_training"]
            assert missing is loaded["is_missing"]
            assert windows is WINDOWS
            given["folds"], given["margin_bins"] = folds, margin_bins
            scores = pd.DataFrame(
                {"window_id": [0, 1, 0, 1], "tetrode": [0, 0, 1, 1], "gain": 1.0}
            )
            return scores, {"refit_s": 1.0}

        monkeypatch.setattr(runner, "epoch_inputs", epoch_inputs)
        monkeypatch.setattr(runner.th, "holdout_scores", holdout_scores)
        output = tmp_path / "out"
        record = runner.run_epoch(
            EPOCH, tmp_path / "event_table.parquet", tmp_path, output
        )
        table = pd.read_parquet(output_path(EPOCH, output, TETRODE_HOLDOUT_NAME))
        assert list(table["kind"]) == ["event", "control", "event", "control"]
        return record, given

    return run


def test_trial_folds_are_made_from_the_trials_and_recorded(patched):
    record, given = patched()
    assert given["trials"] is True
    np.testing.assert_array_equal(
        given["folds"],
        th.trial_folds(
            TIME,
            TIME[[0, 2, 4, 6, 8, 10, 11]],
            np.r_[TRIALS["to_well"], th.AFTER_LAST_TRIAL],
            th.N_FOLDS,
        ),
    )
    assert given["margin_bins"] == 500  # 1 s of 2 ms bins
    assert record["status"] == "ok"
    assert record["n_windows"] == 2 and record["n_tetrodes"] == 2
    assert record["refit_s"] == 1.0
    assert record["fold_design"] == "trials"
    assert record["margin_bins"] == 500
    assert record["n_segments"] == 7  # six trials and the time after them
    assert record["n_groups"] == 7
    assert record["trial_bin_share"] == pytest.approx(11 / 12)
    # the digest of the trials the folds were made from
    assert record["epoch_inputs"] == {"trials": trials_input(TRIALS)}


@pytest.fixture
def header(processed_data_root, monkeypatch):
    """A function that returns the header ``main`` gathers under, for the
    given arguments."""
    monkeypatch.setattr(runner, "provenance_record", lambda root: {})
    monkeypatch.setattr(runner, "file_input", lambda path: {"sha256": "e"})
    monkeypatch.setattr(runner, "participation_input", lambda data_dir: {"sha256": "p"})
    headers = []
    monkeypatch.setattr(
        runner, "gather_manifest", lambda output, name, header: headers.append(header)
    )

    def gathered(*argv):
        runner.main(["--gather", *argv])
        return headers[-1]

    return gathered


def test_the_header_names_the_fold_design_and_its_margin(header):
    assert header()["settings"] == {
        "n_folds": th.N_FOLDS,
        **th.fold_settings(),
    }
