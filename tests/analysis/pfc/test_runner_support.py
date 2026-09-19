"""Tests for the session-selection, seeding and sidecar helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.runner_support import (
    filter_epoch_info,
    is_nonempty_file,
    json_safe,
    read_session_list,
    session_seed,
)


def _epoch_info(keys: list[tuple[str, str]], **columns) -> pd.DataFrame:
    """Session table indexed the way ``get_epoch_info`` returns it."""
    index = pd.MultiIndex.from_tuples(
        keys, names=["nwb_file_name", "interval_list_name"]
    )
    frame = pd.DataFrame(
        {
            "animal": [key[0].split("2")[0] for key in keys],
            "date": list(range(len(keys))),
        },
        index=index,
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


class TestSessionSeed:
    def test_it_is_deterministic_and_stream_specific(self):
        folds = session_seed("folds", "wilbur20210329_.nwb", "02_r1")

        assert folds == session_seed("folds", "wilbur20210329_.nwb", "02_r1")
        assert folds != session_seed("shuffles", "wilbur20210329_.nwb", "02_r1")
        assert folds != session_seed("folds", "wilbur20210329_.nwb", "04_r2")
        assert folds != session_seed("folds", "wilbur20210330_.nwb", "02_r1")

    def test_it_stays_inside_the_unsigned_64_bit_range(self):
        seeds = [
            session_seed(stream, f"rat2021032{index}_.nwb", "02_r1")
            for index in range(5)
            for stream in ("folds", "shuffles")
        ]

        assert all(0 <= seed < 2**64 for seed in seeds)
        assert len(set(seeds)) == len(seeds)


class TestReadSessionList:
    def test_it_keeps_file_order_and_drops_repeats_and_comments(self, tmp_path):
        path = tmp_path / "sessions.txt"
        path.write_text(
            "# worker 0\nc_.nwb 03_r1\n\na_.nwb,01_r1  # inline comment\nc_.nwb 03_r1\n"
        )

        assert read_session_list(path) == [("c_.nwb", "03_r1"), ("a_.nwb", "01_r1")]

    def test_it_refuses_a_malformed_line(self, tmp_path):
        path = tmp_path / "sessions.txt"
        path.write_text("a_.nwb 01_r1 extra\n")

        with pytest.raises(ValueError, match="expected"):
            read_session_list(path)

    def test_it_refuses_a_file_naming_no_sessions(self, tmp_path):
        path = tmp_path / "sessions.txt"
        path.write_text("# nothing here\n\n")

        with pytest.raises(ValueError, match="names no sessions"):
            read_session_list(path)


class TestFilterEpochInfo:
    def test_a_session_file_sets_the_run_order(self, tmp_path):
        epochs = _epoch_info(
            [("a_.nwb", "01_r1"), ("b_.nwb", "02_r1"), ("c_.nwb", "03_r1")]
        )
        session_file = tmp_path / "sessions.txt"
        session_file.write_text("c_.nwb 03_r1\na_.nwb 01_r1\n")

        selected = filter_epoch_info(epochs, sessions=session_file)

        assert list(selected.index) == [("c_.nwb", "03_r1"), ("a_.nwb", "01_r1")]

    def test_without_a_session_file_the_source_order_is_kept(self):
        epochs = _epoch_info(
            [("a_.nwb", "01_r1"), ("b_.nwb", "02_r1"), ("c_.nwb", "03_r1")]
        )

        selected = filter_epoch_info(epochs)

        assert list(selected.index) == list(epochs.index)

    def test_animal_date_and_epoch_filters_compose(self):
        epochs = _epoch_info(
            [("a20200101_.nwb", "01_r1"), ("a20200101_.nwb", "02_r1")],
            animal=["a", "a"],
            date=[20200101, 20200101],
        )

        selected = filter_epoch_info(epochs, animal="a", date=20200101, epoch="02_r1")

        assert list(selected.index) == [("a20200101_.nwb", "02_r1")]
        assert filter_epoch_info(epochs, animal="b").empty

    def test_an_unavailable_requested_session_is_refused(self, tmp_path):
        epochs = _epoch_info([("a_.nwb", "01_r1")])
        session_file = tmp_path / "sessions.txt"
        session_file.write_text("z_.nwb 09_r9\n")

        with pytest.raises(ValueError, match="unavailable session"):
            filter_epoch_info(epochs, sessions=session_file)


class TestIsNonemptyFile:
    def test_only_a_written_regular_file_counts(self, tmp_path):
        empty = tmp_path / "empty.npy"
        empty.touch()
        written = tmp_path / "written.npy"
        written.write_bytes(b"x")

        assert not is_nonempty_file(tmp_path / "missing.npy")
        assert not is_nonempty_file(empty)
        assert not is_nonempty_file(tmp_path)
        assert is_nonempty_file(written)


class TestJsonSafe:
    def test_non_finite_floats_become_none_at_every_depth(self):
        sanitized = json_safe(
            {
                "delta": float("nan"),
                "nested": {"upper": float("inf"), "lower": -np.inf},
                "series": [1.5, float("nan"), (np.float32("nan"), 2)],
                "kept": {"count": 3, "name": "model", "flag": True, "none": None},
            }
        )

        assert sanitized == {
            "delta": None,
            "nested": {"upper": None, "lower": None},
            "series": [1.5, None, [None, 2]],
            "kept": {"count": 3, "name": "model", "flag": True, "none": None},
        }

    def test_the_result_is_strict_json(self):
        def refuse(token: str) -> None:
            raise AssertionError(f"non-JSON token {token!r}")

        text = json.dumps(json_safe({"delta": float("nan"), "finite": 0.25}))

        assert json.loads(text, parse_constant=refuse) == {
            "delta": None,
            "finite": 0.25,
        }
