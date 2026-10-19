"""Threshold propagation regressions for the quality-metrics CLI."""

from __future__ import annotations

import sys

import pandas as pd

import scripts.qa.run_quality_metrics as quality_script


def test_main_converts_filename_threshold_and_forwards_it(monkeypatch, tmp_path):
    """``--threshold 50`` selects `_50_` data and means probability 0.5."""
    calls: dict[str, object] = {}
    data = pd.DataFrame({"sentinel": [1]})
    summary = pd.DataFrame({"animal": ["a"]})

    def _load_all_sessions(*, animal, threshold):
        calls["load"] = (animal, threshold)
        return data

    def _compute_summary(received, **kwargs):
        calls["summary"] = (received, kwargs)
        return summary

    def _plot_diagnostics(received_data, received_summary, **kwargs):
        calls["plot"] = (received_data, received_summary, kwargs)

    monkeypatch.setattr(quality_script, "load_all_sessions", _load_all_sessions)
    monkeypatch.setattr(
        quality_script,
        "compute_session_quality_summary",
        _compute_summary,
    )
    monkeypatch.setattr(quality_script, "maybe_merge_raw_stats", lambda frame, _: frame)
    monkeypatch.setattr(quality_script, "print_summary", lambda _: None)
    monkeypatch.setattr(quality_script, "plot_diagnostics", _plot_diagnostics)
    monkeypatch.setattr(quality_script, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_quality_metrics.py", "--threshold", "50"],
    )

    quality_script.main()

    assert calls["load"] == (None, 50)
    received, summary_kwargs = calls["summary"]
    assert received is data
    assert summary_kwargs["non_local_threshold"] == 0.5
    plotted_data, plotted_summary, _ = calls["plot"]
    assert plotted_data is data
    assert plotted_summary is summary


def test_load_all_sessions_reads_at_the_filename_threshold(monkeypatch, tmp_path):
    """``_50_`` files are read, and their labels checked, at probability 0.5."""
    path = tmp_path / "a20200101_02_r1_HPC_50_data_over_time.pkl"
    seen: dict[str, object] = {}

    def _read(received, *, non_local_threshold):
        seen["read"] = (received, non_local_threshold)
        return pd.DataFrame({"animal": ["a"], "day": [1], "epoch": ["02_r1"]})

    monkeypatch.setattr(
        quality_script, "analysed_decode_files", lambda _dir, _suffix: [path]
    )
    monkeypatch.setattr(quality_script, "read_dot_with_swr_source", _read)

    quality_script.load_all_sessions(threshold=50)

    assert seen["read"] == (path, 0.5)
