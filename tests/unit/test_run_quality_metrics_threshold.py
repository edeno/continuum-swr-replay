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
    plotted_data, plotted_summary, plot_kwargs = calls["plot"]
    assert plotted_data is data
    assert plotted_summary is summary
    assert plot_kwargs["non_local_threshold"] == 0.5


def test_plot_diagnostics_forwards_threshold_to_event_context(monkeypatch, tmp_path):
    seen: dict[str, float] = {}

    for helper_name in (
        "_plot_error_distribution",
        "_plot_state_fractions",
        "_plot_nonlocal_movement",
        "_plot_calibration_curves",
        "_plot_nospike_correlation",
        "_plot_segment_error",
    ):
        monkeypatch.setattr(quality_script, helper_name, lambda *_: None)

    def _plot_event_context(_data, _output_path, *, non_local_threshold):
        seen["threshold"] = non_local_threshold

    monkeypatch.setattr(quality_script, "_plot_event_context", _plot_event_context)
    monkeypatch.setattr(
        quality_script.figure_utilities,
        "set_figure_defaults",
        lambda **_: None,
    )
    monkeypatch.setattr(quality_script, "FIGURE_DIR", tmp_path)

    quality_script.plot_diagnostics(
        pd.DataFrame(),
        pd.DataFrame(),
        output_dir="diagnostics",
        non_local_threshold=0.5,
    )

    assert seen["threshold"] == 0.5
