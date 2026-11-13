"""Fixes to the figure scripts found in the R7a figure run (tasks 26b and 26c).

- The event gallery selects the same events in every process.
- The pipeline data-flow figure reads its frame through the SWR source, has a
  default event that has every arm indicator, refuses one that does not with a
  clear error, and fits its panel E headers in their cells.
- Figure 2's render cache is keyed on a version token, so a library change can
  invalidate it.
- Figure 3 panel a's relative-time ticks sit on round steps and each label is
  its tick's exact relative time (task 26c).
- Figure 3 panels c and e draw the events summary's ripple and speed values.
- The rate summary's panel B boxes log(RR), so one extreme session does not set
  its axis; the contrast coefficient figures keep their titles and x labels
  inside the figure; the arrival retention note is computed from the fit; and
  the rescaled-ISI autocorrelation panel draws its points (task 26d).
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import pickle
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.analysis.event_definition import (  # noqa: E402
    analysed_time,
    swr_source,
)
from continuum_swr_replay.tables.io import write_table  # noqa: E402
from scripts.figures import generate_supp_pipeline_data_flow as flow  # noqa: E402
from tests.support._swr_inventory_fixtures import (  # noqa: E402
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

REPO = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# Event gallery: a stable sample
# --------------------------------------------------------------------------

_SELECT = textwrap.dedent(
    """
    import json

    import pandas as pd

    from continuum_swr_replay.analysis.event_definition import gallery_sample as gallery

    rows = [
        {
            "animal": animal,
            "category": category,
            "event_id": 1000 * a + 100 * c + k,
            "ambiguity": (k * 37) % 41 / 41,
        }
        for a, animal in enumerate(("chimi", "j16", "peanut"))
        for c, category in enumerate(gallery.CATEGORIES)
        for k in range(40)
    ]
    selected = gallery.select_events(pd.DataFrame(rows), n_per_category=5)
    print(json.dumps(sorted(int(i) for i in selected["event_id"])))
    """
)


def _selection_in_a_new_process(hash_seed: str) -> list[int]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    done = subprocess.run(
        [sys.executable, "-c", _SELECT],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_gallery_selects_the_same_events_in_two_processes():
    """``hash()`` of a string is salted per process (``PYTHONHASHSEED``); the
    selection must not depend on it."""
    first = _selection_in_a_new_process("1")
    second = _selection_in_a_new_process("2")

    assert len(first) == 3 * 4 * 5
    assert first == second


# --------------------------------------------------------------------------
# Pipeline data-flow figure
# --------------------------------------------------------------------------


def test_the_data_flow_figure_defaults_to_an_event_with_every_indicator():
    assert flow.build_parser().parse_args([]).event_id == 275


def _event_rows(**indicators) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "event_id": 5548608985892833789,
            "candidate_arm": np.arange(9),
            "arm_count_hard": [20, 21, 0, 14, 3, 0, 0, 0, 0],
            "is_current_arm": 0.0,
            "is_prev_arm": 0.0,
            "is_next_arm": 0.0,
            "is_last_rewarded_arm": 0.0,
        }
    )
    return rows.assign(**indicators)


def test_an_event_with_a_missing_indicator_is_refused_by_name():
    """The old default, event 14, is off-well: its ``is_current_arm`` is NaN
    under the covariate contract, and the cast to int raised
    ``IntCastingNaNError``."""
    rows = _event_rows(is_current_arm=np.nan, is_prev_arm=np.nan)

    with pytest.raises(ValueError, match=r"event 14.*is_current_arm.*is_prev_arm"):
        flow.indicator_table(rows, event_id=14)


def test_an_event_with_every_indicator_gives_its_integer_table():
    rows = _event_rows(is_current_arm=(np.arange(9) == 5).astype(float))

    table = flow.indicator_table(rows, event_id=275)

    assert list(table.columns) == flow.INDICATOR_COLUMNS
    assert (table.dtypes == np.int64).all()
    assert table["is_current_arm"].tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0]


def test_the_panel_e_headers_fit_their_cells():
    import matplotlib.pyplot as plt

    fig, axes = flow.figure_axes()
    table = flow.draw_indicator_table(
        axes["E"], flow.indicator_table(_event_rows(), event_id=275)
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    headers = [cell for (row, _), cell in table.get_celld().items() if row == 0]
    assert len(headers) == len(flow.INDICATOR_COLUMNS)
    for cell in headers:
        text = cell.get_text().get_window_extent(renderer)
        box = cell.get_window_extent(renderer)
        assert box.x0 <= text.x0 and text.x1 <= box.x1, cell.get_text().get_text()
        assert box.y0 <= text.y0 and text.y1 <= box.y1, cell.get_text().get_text()
    # every column is named, in full, across the wrapped lines
    names = [cell.get_text().get_text().replace("\n", "") for cell in headers]
    assert sorted(names) == sorted(flow.INDICATOR_COLUMNS)
    plt.close(fig)


def test_panel_f_prints_each_term_s_median_over_the_sessions_its_gate_keeps():
    """Panel F printed the median over every fitted session. Session 4's
    ``is_current_arm`` is separated and session 3's ``is_prev_arm`` ran away
    (|coef| >= 10): the gate leaves each out of its term."""
    import matplotlib.pyplot as plt

    terms = flow.COEFFICIENT_TERMS
    fits = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b", "c"],
            "session_date": [1, 2, 1, 2, 1],
            "epoch": "02_r1",
            f"coef_{terms[0]}": [0.1, 0.2, 0.4, 0.8, 5.0],
            f"coef_{terms[1]}": [-0.3, -0.1, 0.2, 12.0, 0.0],
            f"coef_{terms[2]}": [0.5, 0.25, 0.75, 1.0, -1.0],
            "separated_terms": ["", "", "", "", terms[0]],
        }
    )
    fig, ax = plt.subplots()

    flow.draw_coefficients(ax, fits)

    # medians of the kept sessions: (0.2 + 0.4) / 2, (-0.1 + 0.0) / 2 and the
    # median of all five; over every session they were 0.4, 0.0 and 0.5
    assert [text.get_text() for text in ax.texts] == ["+0.30", "−0.050", "+0.50"]
    labels = [label.get_text() for label in ax.get_xticklabels()]
    assert [label.split("\n")[1] for label in labels] == [
        "(4 sessions)",
        "(4 sessions)",
        "(5 sessions)",
    ]
    plt.close(fig)


EPOCH_ID = "chimi20200213_02_r1_HPC"
TIME = np.round(100.0 + np.arange(2000) / 500.0, 6)


def test_the_data_flow_figure_reads_its_frame_through_the_source(tmp_path, monkeypatch):
    """An artifact core at 102.15 s (a spike line, padded by 0.1 s and widened
    by 0.3 s) puts [101.75, 102.55] in excluded time."""
    inventory = write_swr_inventory(tmp_path, {EPOCH_ID: []})
    artifacts = write_artifact_run(tmp_path, {EPOCH_ID: [("spk", 102.15, 102.15)]})
    use_swr_source(monkeypatch, inventory, artifacts)
    pd.DataFrame(
        {
            "animal": "chimi",
            "day": 20200213,
            "epoch": "02_r1",
            "nwb_file_name": "chimi20200213_.nwb",
            "head_speed": 0.0,
            "map_estimate": 0.0,
            "non_local_event": 0,
            "SWR_event": 0,
            "MUA_event": 0,
            "Non-Local_Continuous_prob": 0.0,
            "Non-Local_Fragmented_prob": 0.0,
        },
        index=pd.Index(TIME, name="time"),
    ).to_pickle(tmp_path / f"{EPOCH_ID}_80_data_over_time.pkl")
    (tmp_path / f"{EPOCH_ID}_environment.pkl").write_bytes(
        pickle.dumps({"environment": "stand-in"})
    )
    monkeypatch.setattr(
        flow, "map_estimate_to_arm", lambda values, env: np.full(len(values), np.nan)
    )

    dot, _ = flow.load_session("chimi", 20200213, "02_r1")

    ((start, end),) = analysed_time.excluded_intervals(EPOCH_ID).to_numpy()
    np.testing.assert_array_equal(
        dot["in_excluded_time"].to_numpy(), (TIME >= start) & (TIME <= end)
    )
    assert dot["in_excluded_time"].any()


# --------------------------------------------------------------------------
# Figure 2's render cache
# --------------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    """Import a script with ``spyglass`` stubbed; forget it afterwards."""
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = type("PositionIntervalMap", (), {})
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)
    data_loaders = importlib.import_module("continuum_swr_replay.data_loaders")
    load_data_was_cached = "load_data" in vars(data_loaders)
    bound = ("continuum_swr_replay.data_loaders.bandit_task",)
    fresh_bound = [name for name in bound if name not in sys.modules]
    imported: list[str] = []

    def load(name: str):
        if name not in sys.modules:
            imported.append(name)
        return importlib.import_module(name)

    yield load

    for name in [*imported, *fresh_bound]:
        sys.modules.pop(name, None)
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            vars(sys.modules[parent]).pop(child, None)
    if not load_data_was_cached:
        vars(data_loaders).pop("load_data", None)


def test_the_figure02_cache_is_keyed_on_a_version_token(offline, tmp_path):
    figure02 = offline("scripts.figures.generate_figure02")

    path = figure02._figure02_cache_path("wilbur20210329_.nwb", "02_r1", tmp_path)

    # the name before the token: a cache written with it is not read again
    v1 = (
        "wilbur20210329_02_r1_figure02_whole_event_speed_v1_"
        f"{swr_source.swr_source_name()}_cache.joblib"
    )
    assert path.parent == tmp_path / "intermediates"
    assert path.name != v1
    assert f"_v{figure02.FIGURE02_CACHE_VERSION}_" in path.name
    # a cache of version 2 holds no class counts to check the event table against
    assert figure02.FIGURE02_CACHE_VERSION >= 3
    assert swr_source.swr_source_name() in path.name


def _register(path: Path, epochs: list[str]) -> Path:
    pd.DataFrame(
        [
            ["wilbur20210329_.nwb", epoch, "analysed", "", "", "False"]
            for epoch in epochs
        ],
        columns=[
            "nwb_file_name",
            "epoch",
            "status",
            "reason",
            "evidence",
            "first_exposure",
        ],
    ).to_csv(path, index=False)
    return path


def _figure02_events() -> pd.DataFrame:
    """Eight events whose shares and median offsets fall half-way between two
    whole numbers: 1 of 8 SWR-contained (12.5 %), 5 of 8 non-SWR (62.5 %); the
    three overlapping events' median start offset is 2.5 ms and their median
    end offset -4.5 ms."""
    nan = np.nan
    return pd.DataFrame(
        {
            "animal": "a",
            "session_date": "d1",
            "epoch": "e1",
            "overlap_class": ["swr_contained", "swr_partial", "swr_partial"]
            + ["non_swr"] * 5,
            "whole_event_speed_eligible": True,
            "event_start_s": [0.0025, -0.25, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "event_end_s": [0.25, -0.0045, 2.0, 3.1, 4.1, 5.1, 6.1, 7.1],
            "matched_swr_start": [0.0, -0.125, 0.5, nan, nan, nan, nan, nan],
            "matched_swr_end": [0.5, 0.0, 1.5, nan, nan, nan, nan, nan],
        }
    )


#: ``_figure02_events``'s overlap classes, and the decode frames' event counts
#: by SWR overlap that agree with them.
FIGURE02_CLASS_COUNTS = {"swr_contained": 1, "swr_partial": 2, "non_swr": 5, "total": 8}
FIGURE02_FRAME_COUNTS = {"swr": 3, "non_swr": 5, "total": 8}


def test_figure02_panels_b_and_d_come_from_the_canonical_event_table(offline):
    """Panel b's counts and panel d's offsets are computed from the event table
    the summary cites, and the figure is refused when that table's event counts
    by SWR overlap are not the decode frames', or the table holds an event that
    is not immobile."""
    figure02 = offline("scripts.figures.generate_figure02")
    events = _figure02_events()
    panels = figure02.event_table_panels(events, FIGURE02_FRAME_COUNTS)
    assert panels["class_counts"] == FIGURE02_CLASS_COUNTS
    offsets = panels["offsets"]
    np.testing.assert_allclose(offsets["start_offset_s"], [0.0025, -0.125, 0.5])
    np.testing.assert_allclose(offsets["end_offset_s"], [-0.25, -0.0045, 0.5])
    with pytest.raises(ValueError, match="counts by SWR overlap are not"):
        figure02.event_table_panels(
            events, {**FIGURE02_FRAME_COUNTS, "non_swr": 6, "total": 9}
        )
    moving = events.assign(whole_event_speed_eligible=[True] * 7 + [False])
    with pytest.raises(ValueError, match="not whole-event immobile"):
        # the frames' counts are of immobile events, as the table's would be
        figure02.event_table_panels(
            moving, {**FIGURE02_FRAME_COUNTS, "non_swr": 4, "total": 7}
        )


def test_figure02_panels_b_and_d_print_the_digits_of_the_events_summary(offline):
    """The percents and medians written on the panels are rounded by the
    summary's rule (a half rounds away from zero), not by the plotter's
    format (a half rounds to the even digit: 12, 62, +2 and -4 here)."""
    import matplotlib.pyplot as plt

    from continuum_swr_replay.reported.events import (
        boundary_offset_values,
        event_cohort_values,
    )
    from continuum_swr_replay.reported.formatting import formatted
    from continuum_swr_replay.visualization.figure_panels import (
        plot_event_timing_offsets,
        plot_swr_overlap_class_bar,
    )

    figure02 = offline("scripts.figures.generate_figure02")
    events = _figure02_events()
    panels = figure02.event_table_panels(events, FIGURE02_FRAME_COUNTS)

    assert panels["class_percents"] == {
        "swr_contained": "13",
        "swr_partial": "25",
        "non_swr": "63",
    }
    assert panels["offset_medians"] == {
        "start_offset_s": "+3",
        "end_offset_s": "−5",
    }
    # the same digits as the summary's values of the same table
    cohort = {v.key: v for v in event_cohort_values(events, "s")}
    bounds = {v.key: v for v in boundary_offset_values(events, "s")}
    assert formatted(cohort["non_swr_fraction"]) == "63 %"
    assert formatted(bounds["start_offset_median_ms"]) == "3"
    assert formatted(bounds["end_offset_median_ms"]) == "−5"

    # the plotters write the text they are given
    fig, (ax_b, ax_start, ax_end) = plt.subplots(1, 3)
    plot_swr_overlap_class_bar(
        panels["class_counts"], panels["class_percents"], ax=ax_b
    )
    assert [text.get_text() for text in ax_b.texts] == [
        "13%\nSWR-\ncontained",
        "25%\nSWR-\npartial",
        "63%\nnon-SWR",
    ]
    plot_event_timing_offsets(
        panels["offsets"],
        panels["offset_median_s"],
        panels["offset_medians"],
        axes=[ax_start, ax_end],
    )
    # the dashed lines are at the summary's medians, 2.5 and -4.5 ms
    np.testing.assert_allclose(ax_start.get_lines()[-1].get_xdata(), [2.5, 2.5])
    np.testing.assert_allclose(ax_end.get_lines()[-1].get_xdata(), [-4.5, -4.5])
    assert [text.get_text() for text in ax_start.texts] == ["median +3 ms"]
    assert [text.get_text() for text in ax_end.texts] == ["median −5 ms"]
    plt.close(fig)


def test_figure02_checks_the_event_table_against_the_cached_frame_counts(
    offline, tmp_path, monkeypatch
):
    """With a cached bundle, panels a, b, d and e are still read when the figure
    is drawn -- a, b and d from the event table in the data folder, checked
    against the bundle's frame counts, and e from the stored decoder quality."""
    import joblib

    from continuum_swr_replay.paths import (
        processed_data_dir,
        replay_content_dir,
        table_path,
    )

    figure02 = offline("scripts.figures.generate_figure02")
    table = table_path(
        "event_table", tmp_path / replay_content_dir().relative_to(processed_data_dir())
    )
    table.parent.mkdir(parents=True)
    write_table(analysed_time.stamp_swr_source(_figure02_events()), table)
    cache = figure02._figure02_cache_path("wilbur20210329_.nwb", "02_r1", tmp_path)
    cache.parent.mkdir(parents=True)
    drawn = {}
    monkeypatch.setattr(
        figure02, "_render_figure", lambda inputs, window_sec: drawn.update(inputs)
    )
    read = {}
    monkeypatch.setattr(
        figure02,
        "example_events",
        lambda events, ripples, nwb, epoch: read.setdefault("a", len(events)),
    )
    monkeypatch.setattr(
        figure02, "read_decoding_error", lambda data_dir: read.setdefault("e", data_dir)
    )

    data = {"ripple_times": pd.DataFrame(columns=["start_time", "end_time"])}
    joblib.dump({"frame_event_counts": FIGURE02_FRAME_COUNTS, "data": data}, cache)
    figure02.generate_figure(data_dir=tmp_path)
    assert drawn["chosen"] == 8
    assert drawn["error_df"] == tmp_path
    assert drawn["class_counts"] == FIGURE02_CLASS_COUNTS
    assert drawn["class_percents"]["non_swr"] == "63"
    assert len(drawn["offsets"]) == 3

    # the frames give one more non-SWR event than the table holds
    joblib.dump(
        {
            "frame_event_counts": {**FIGURE02_FRAME_COUNTS, "non_swr": 6, "total": 9},
            "data": data,
        },
        cache,
    )
    with pytest.raises(ValueError, match="counts by SWR overlap are not"):
        figure02.generate_figure(data_dir=tmp_path)


def _figure02_session_events() -> pd.DataFrame:
    """The example session's four events, one of another session, and one
    non-SWR event the immobility rule does not keep."""
    return pd.DataFrame(
        {
            "animal": "wilbur",
            "session_date": [20210329] * 5 + [20210330],
            "epoch": "02_r1",
            "event_id": ["c", "p", "n1", "n2", "moving", "other"],
            "overlap_class": ["swr_contained", "swr_partial", "non_swr", "non_swr"]
            + ["non_swr", "non_swr"],
            "whole_event_speed_eligible": [True] * 4 + [False, True],
            "event_start_s": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event_end_s": [1.1, 2.2, 3.05, 4.3, 5.1, 6.1],
            "event_duration_s": [0.102, 0.202, 0.052, 0.302, 0.102, 0.102],
            "peak_nonlocal_prob": [0.95, 0.95, 0.95, 0.95, 0.99, 0.99],
        }
    )


def test_figure02_examples_are_the_sessions_rows_of_the_event_table(offline):
    figure02 = offline("scripts.figures.generate_figure02")
    ripples = pd.DataFrame({"start_time": [0.9, 2.1], "end_time": [1.2, 2.5]})

    chosen = figure02.example_events(
        _figure02_session_events(), ripples, "wilbur20210329_.nwb", "02_r1"
    )

    assert {cls: ev["event_id"] for cls, ev in chosen.items()} == {
        "swr_contained": "c",
        "swr_partial": "p",
        # the session's two kept non-SWR events are equally near their median
        # duration; the earlier is taken, and never the moving or other one
        "non_swr": "n1",
    }
    assert chosen["swr_partial"]["start_time"] == 2.0
    assert chosen["swr_partial"]["duration"] == 0.202

    # an example drawn under an SWR that contradicts its class is refused
    with pytest.raises(AssertionError):
        figure02.example_events(
            _figure02_session_events(),
            pd.DataFrame({"start_time": [0.9, 2.1, 3.0], "end_time": [1.2, 2.5, 3.1]}),
            "wilbur20210329_.nwb",
            "02_r1",
        )


def test_figure02_counts_the_frames_events_by_swr_overlap(offline):
    figure02 = offline("scripts.figures.generate_figure02")
    rows = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "e",
            # bins of three events; event 2 has two bins
            "non_local_event": [1, 2, 2, 3],
        }
    )

    counts = figure02.frame_event_counts(
        {
            "non_local_event_immobile": rows,
            "non_local_event_immobile_overlap_SWR_event": rows.iloc[1:3],
            "non_local_event_immobile_no_overlap_SWR_event": rows.iloc[[0, 3]],
        }
    )

    assert counts == {"swr": 1, "non_swr": 2, "total": 3}


def test_figure02_panel_e_reads_the_stored_decoder_quality(
    offline, tmp_path, monkeypatch
):
    from continuum_swr_replay.data_loaders import epoch_register

    figure02 = offline("scripts.figures.generate_figure02")
    monkeypatch.setattr(
        epoch_register,
        "REGISTER_PATH",
        _register(tmp_path / "register.csv", ["02_r1", "04_r2"]),
    )
    folder = tmp_path / "decoder_quality"
    folder.mkdir()
    sessions = pd.DataFrame(
        {
            "epoch_id": ["wilbur20210329_02_r1_HPC", "wilbur20210329_04_r2_HPC"],
            "median_error_cm": [3.0, 4.0],
        }
    )
    sessions.to_parquet(folder / "sessions.parquet")

    pd.testing.assert_frame_equal(figure02.read_decoding_error(tmp_path), sessions)

    sessions.iloc[:1].to_parquet(folder / "sessions.parquet")
    with pytest.raises(ValueError, match="analysed"):
        figure02.read_decoding_error(tmp_path)


def test_the_figure02_cache_is_keyed_on_the_analysed_epochs(
    offline, tmp_path, monkeypatch
):
    """Panel e summarises every analysed epoch, so a register change must make
    a new cache without a code change."""
    from continuum_swr_replay.data_loaders import epoch_register

    figure02 = offline("scripts.figures.generate_figure02")

    def cache_name(epochs: list[str]) -> str:
        register = _register(tmp_path / f"register_{len(epochs)}.csv", epochs)
        monkeypatch.setattr(epoch_register, "REGISTER_PATH", register)
        return figure02._figure02_cache_path(
            "wilbur20210329_.nwb", "02_r1", tmp_path
        ).name

    one = cache_name(["02_r1"])
    two = cache_name(["02_r1", "04_r2"])

    assert one != two
    assert cache_name(["02_r1"]) == one


def test_the_figure02_cache_is_keyed_on_the_decode_files(
    offline, tmp_path, monkeypatch
):
    """A rewritten decode (data_over_time or the example's results.nc) makes a
    new cache without a code change; untouched files keep the old one."""
    import os

    from continuum_swr_replay.data_loaders import epoch_register

    figure02 = offline("scripts.figures.generate_figure02")
    monkeypatch.setattr(
        epoch_register,
        "REGISTER_PATH",
        _register(tmp_path / "register.csv", ["02_r1", "04_r2"]),
    )
    dot = tmp_path / "wilbur20210329_04_r2_HPC_80_data_over_time.pkl"
    results = tmp_path / "wilbur20210329_02_r1_HPC_results.nc"
    for path in (dot, results):
        path.write_bytes(b"a")
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))

    def cache_name() -> str:
        return figure02._figure02_cache_path(
            "wilbur20210329_.nwb", "02_r1", tmp_path
        ).name

    before = cache_name()
    assert cache_name() == before

    dot.write_bytes(b"ab")
    after_dot = cache_name()
    assert after_dot != before

    os.utime(results, ns=(2_000_000_000, 2_000_000_000))
    assert cache_name() not in (before, after_dot)


# --------------------------------------------------------------------------
# Figure 3 panel a: the example is a canonical event
# --------------------------------------------------------------------------


def _figure03_example_events() -> pd.DataFrame:
    """The example session's three events and a longer one of another epoch."""
    return pd.DataFrame(
        {
            "animal": "wilbur",
            "session_date": 20210329,
            "epoch": ["02_r1", "02_r1", "02_r1", "04_r2"],
            "is_swr": [False, True, False, False],
            "event_start_s": [10.0, 20.0, 30.0, 40.0],
            "event_end_s": [10.2, 20.9, 30.3, 41.0],
            "event_duration_s": [0.202, 0.902, 0.302, 1.002],
        }
    )


def test_the_figure03_example_is_the_sessions_longest_non_swr_event(offline):
    figure03 = offline("scripts.figures.generate_figure03")
    ripples = pd.DataFrame({"start_time": [20.0], "end_time": [20.9]})

    window = figure03.find_non_swr_example_event(
        _figure03_example_events(), ripples, "wilbur20210329_.nwb", "02_r1"
    )

    assert window == slice(30.15 - 1.0, 30.15 + 1.0)


def test_the_figure03_example_is_refused_when_an_swr_is_drawn_over_it(offline):
    figure03 = offline("scripts.figures.generate_figure03")
    ripples = pd.DataFrame({"start_time": [30.1], "end_time": [30.2]})

    with pytest.raises(AssertionError):
        figure03.find_non_swr_example_event(
            _figure03_example_events(), ripples, "wilbur20210329_.nwb", "02_r1"
        )


# --------------------------------------------------------------------------
# Figure 3 panel a: relative-time ticks (task 26c)
# --------------------------------------------------------------------------

# A real session time is a Unix time near 1.6e9 s, so a float64 step is
# 2.4e-7 s; a label that matches to 1e-6 s is exact at the precision shown.
T0 = 1_615_000_000.123
LABEL_TOLERANCE_S = 1e-6


@pytest.mark.parametrize(
    ("before_t0", "after_t0"),
    [(0.0, 2.0), (0.05, 2.05), (0.0, 0.3), (0.0, 8.0), (0.17, 0.41)],
)
def test_the_figure03_ticks_sit_on_a_round_step_with_exact_labels(
    offline, before_t0, after_t0
):
    figure03 = offline("scripts.figures.generate_figure03")

    ticks, labels = figure03._relative_time_ticks(T0 - before_t0, T0 + after_t0, T0)

    relative = np.array([float(label) for label in labels])
    # each label is its tick's relative time
    np.testing.assert_allclose(
        relative, np.asarray(ticks) - T0, rtol=0, atol=LABEL_TOLERANCE_S
    )
    # evenly spaced, with a step of at most two decimals (0.05, 0.25, 0.5, 1 ...)
    steps = np.diff(relative)
    assert len(ticks) >= 2
    np.testing.assert_allclose(steps, steps[0], rtol=0, atol=1e-9)
    assert round(steps[0], 2) == pytest.approx(steps[0], abs=1e-9)
    assert all(len(label.partition(".")[2]) <= 2 for label in labels)
    # inside the window, and at most a handful of them
    assert min(ticks) >= T0 - before_t0 - LABEL_TOLERANCE_S
    assert max(ticks) <= T0 + after_t0 + LABEL_TOLERANCE_S
    assert len(ticks) <= 9


def test_the_figure03_first_tick_at_or_after_t0_is_zero(offline):
    figure03 = offline("scripts.figures.generate_figure03")

    ticks, labels = figure03._relative_time_ticks(T0 - 0.05, T0 + 2.05, T0)

    at_or_after = [
        (tick, label)
        for tick, label in zip(ticks, labels, strict=True)
        if tick >= T0 - LABEL_TOLERANCE_S
    ]
    tick, label = min(at_or_after)
    assert label == "0"
    assert tick == pytest.approx(T0, abs=LABEL_TOLERANCE_S)


def test_the_figure03_panel_a_axes_get_the_ticks_and_keep_their_limits(offline):
    import matplotlib.pyplot as plt

    figure03 = offline("scripts.figures.generate_figure03")
    fig = plt.figure()
    first = fig.add_subplot(3, 1, 1)
    axes = [first] + [fig.add_subplot(3, 1, i, sharex=first) for i in (2, 3)]
    for ax in axes:
        ax.plot([T0, T0 + 2.0], [0, 1])
        ax.set_xlim(T0, T0 + 2.0)

    figure03._set_relative_time_ticks(axes, T0)
    fig.canvas.draw()

    for ax in axes:
        labels = [label.get_text() for label in ax.get_xticklabels()]
        np.testing.assert_allclose(
            [float(label) for label in labels],
            ax.get_xticks() - T0,
            rtol=0,
            atol=LABEL_TOLERANCE_S,
        )
        assert ax.get_xlim() == (T0, T0 + 2.0)
    assert [t.get_text() for t in axes[-1].get_xticklabels()][0] == "0"
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3 panels c and e: the events summary's ripple and speed values
# --------------------------------------------------------------------------

#: Event-table columns the events summary's other property values read; the
#: Figure 3 test gives them a constant.
_OTHER_PROPERTY_COLUMNS = (
    "event_duration_s",
    "spatial_extent_cm",
    "posterior_peak_mean",
    "hpd_coverage_cm",
    "pause_fraction",
    "step_distance_max_cm",
    "n_reversals",
    "multimodal_fraction",
    "n_unique_arms",
)


def _figure03_events() -> pd.DataFrame:
    """Eight events in two animals, four of each class. Each class's mean
    differs from its median, and its median is the mean of the middle two:
    non-SWR ripple 0.2, 0.4, 3.0 (a) and 1.0 (b), median 0.7; SWR 2.0, 4.0 (a)
    and 5.0, 1.0 (b), median 3.0; non-SWR speed 100, 200, 900 and 300, median
    250; SWR 400, 600 and 500, 1300, median 550."""
    events = pd.DataFrame(
        {
            "animal": ["a"] * 5 + ["b"] * 3,
            "session_date": ["d1"] * 5 + ["d2"] * 3,
            "epoch": "e1",
            "is_swr": [False, False, False, True, True, False, True, True],
            "ripple_power_z": [0.2, 0.4, 3.0, 2.0, 4.0, 1.0, 5.0, 1.0],
            "replay_speed_median_cm_s": [100, 200, 900, 400, 600, 300, 500, 1300.0],
        }
    )
    events["theta_z"] = events["ripple_power_z"]
    events["mua_rate_z"] = events["ripple_power_z"]
    for column in _OTHER_PROPERTY_COLUMNS:
        events[column] = 1.0
    return events


def _figure03_reference() -> pd.DataFrame:
    """Three epochs, one with no event: the running medians 0.5, 0.1, 1.5
    (median 0.5, mean 0.7) and the rest medians -0.5, -0.9, -0.2 (median -0.5)."""
    reference = pd.DataFrame(
        {
            "animal": ["a", "b", "a"],
            "session_date": ["d1", "d2", "d1"],
            "epoch": ["e1", "e1", "e2"],
            "ripple_running_median": [0.5, 0.1, 1.5],
            "ripple_rest_median": [-0.5, -0.9, -0.2],
        }
    )
    for signal in ("theta", "mua"):
        reference[f"{signal}_running_median"] = reference["ripple_running_median"]
        reference[f"{signal}_rest_median"] = reference["ripple_rest_median"]
    return reference


def _figure03_intervals() -> pd.DataFrame:
    """The bootstrap's ``intervals.csv``: every property of the events summary,
    the speed row with its own numbers."""
    from continuum_swr_replay.reported.events import _PROPERTIES

    columns = [
        f"{word}{suffix}"
        for word in ("median_non_swr", "median_swr", "difference", "matched_difference")
        for suffix in ("", "_low", "_high", "_nan_replicates")
    ]
    intervals = pd.DataFrame(
        0.0, index=[p[0] for p in _PROPERTIES], columns=columns
    ).assign(n_replicates=1000.0, matched_replicates_skipping_strata=0.0, n_animals=5.0)
    intervals.loc["speed", ["difference", "difference_low", "difference_high"]] = [
        300.0,
        120.0,
        480.0,
    ]
    intervals.loc[
        "speed",
        ["matched_difference", "matched_difference_low", "matched_difference_high"],
    ] = [200.0, 50.0, 350.0]
    return intervals


def test_figure03_panels_c_and_e_draw_the_values_of_the_events_summary(offline):
    """The medians, references and intervals the two per-event panels draw are
    the events summary's values of the same inputs, and the values are tagged
    to those panels."""
    import matplotlib.pyplot as plt

    from continuum_swr_replay.reported.events import (
        brain_state_values,
        property_interval_values,
        property_values,
    )
    from continuum_swr_replay.visualization.figure_panels import (
        plot_event_ripple_by_swr,
        plot_event_speed_by_swr,
    )

    figure03 = offline("scripts.figures.generate_figure03")
    events, reference, intervals = (
        _figure03_events(),
        _figure03_reference(),
        _figure03_intervals(),
    )
    panels = figure03.event_panels(events, reference, intervals)

    fig, (ax_c, ax_e, ax_e_difference) = plt.subplots(1, 3)
    plot_event_ripple_by_swr(
        panels["events"], panels["ripple_medians"], panels["ripple_references"], ax=ax_c
    )
    plot_event_speed_by_swr(
        panels["events"],
        panels["speed_medians"],
        panels["speed_contrasts"],
        axes=[ax_e, ax_e_difference],
    )
    drawn_c = {line.get_gid(): line.get_xdata()[0] for line in ax_c.lines}
    drawn_e = {line.get_gid(): line.get_xdata()[0] for line in ax_e.lines}
    drawn_intervals = {}
    for container in ax_e_difference.containers:
        data_line, _, (bars,) = container.lines
        ((_, low), (_, high)) = bars.get_segments()[0]
        drawn_intervals[data_line.get_gid()] = (data_line.get_ydata()[0], low, high)
    plt.close(fig)

    ripple = {v.key: v for v in brain_state_values(events, reference, "s")}
    speed = {v.key: v for v in property_values(events, "s")}
    bounds = {v.key: v for v in property_interval_values(intervals, "s")}

    assert drawn_c == pytest.approx(
        {
            "median_non_swr": ripple["ripple_median_non_swr"].value,
            "median_swr": ripple["ripple_median_swr"].value,
            "reference_running": ripple["ripple_running_reference"].value,
            "reference_rest": ripple["ripple_rest_reference"].value,
        }
    )
    assert drawn_c == pytest.approx(
        {
            "median_non_swr": 0.7,
            "median_swr": 3.0,
            "reference_running": 0.5,
            "reference_rest": -0.5,
        }
    )
    assert drawn_e == pytest.approx(
        {
            "median_non_swr": speed["speed_median_non_swr"].value,
            "median_swr": speed["speed_median_swr"].value,
        }
    )
    assert drawn_e == pytest.approx({"median_non_swr": 250.0, "median_swr": 550.0})
    assert drawn_intervals["difference"][1:] == pytest.approx(
        (bounds["speed_difference_low"].value, bounds["speed_difference_high"].value)
    )
    assert drawn_intervals["matched_difference"] == pytest.approx(
        (
            bounds["speed_matched_difference"].value,
            bounds["speed_matched_difference_low"].value,
            bounds["speed_matched_difference_high"].value,
        )
    )
    assert drawn_intervals["difference"] == pytest.approx((300.0, 120.0, 480.0))

    tagged = {
        key: value.figures
        for values in (ripple, speed, bounds)
        for key, value in values.items()
        if key.startswith(("ripple_", "speed_"))
    }
    assert tagged and all(
        figures == (("3c",) if key.startswith("ripple_") else ("3e",))
        for key, figures in tagged.items()
    )


# --------------------------------------------------------------------------
# Figure 2 panel e: the x-axis label (task 26c)
# --------------------------------------------------------------------------


def test_the_figure02_panel_e_label_lies_inside_the_figure():
    """The label was wider than panel e's cell and ran off the right edge of the
    saved figure. Panel e sits at the right edge, so a label wider than the
    room left there is cut.

    A proxy for ``generate_figure02._render_figure``: the same figure size, the
    same row-3 grid (the script's ``gs_cde``) and the paper typography, with
    panel e drawn by the same function. The real figure needs the cached
    session, so the render check stays a manual one.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from continuum_swr_replay.visualization import figure_utilities
    from continuum_swr_replay.visualization.figure_panels import (
        plot_decoding_error_distribution,
    )

    with matplotlib.rc_context():
        figure_utilities.set_figure_defaults(context="paper")
        fig = plt.figure(
            figsize=(
                figure_utilities.TWO_COLUMN,
                min(9.0, figure_utilities.PAGE_HEIGHT),
            ),
            dpi=100,
            constrained_layout=True,
        )
        outer = GridSpec(3, 1, figure=fig, height_ratios=[2.2, 0.5, 1.1], hspace=0.3)
        gs_cde = outer[2].subgridspec(
            2, 4, width_ratios=[1.35, 0.8, 0.8, 1.05], hspace=0.05, wspace=0.38
        )
        # the neighbours that set the room panel e gets
        ax_c_top = fig.add_subplot(gs_cde[0, 0])
        ax_c_bot = fig.add_subplot(gs_cde[1, 0], sharex=ax_c_top)
        ax_c_top.set_ylabel("Non-Local\nProb.")
        ax_c_bot.set_ylabel("SWR\nZ-score")
        ax_c_bot.set_xlabel("Time from Event (s)")
        ax_d_start = fig.add_subplot(gs_cde[:, 1])
        ax_d_end = fig.add_subplot(gs_cde[:, 2], sharey=ax_d_start)
        ax_d_start.set_ylabel("Density")
        ax_d_start.set_xlabel("NL start − SWR start (ms)")
        ax_d_end.set_xlabel("NL end − SWR end (ms)")
        ax_e = fig.add_subplot(gs_cde[:, 3])
        rng = np.random.default_rng(0)
        plot_decoding_error_distribution(
            ax=ax_e,
            error_df=pd.DataFrame({"median_error_cm": rng.gamma(9, 0.4, 200)}),
        )

        fig.canvas.draw()
        label = ax_e.xaxis.label.get_window_extent(fig.canvas.get_renderer())
        figure = fig.bbox
        plt.close(fig)

    assert label.x0 >= figure.x0
    assert label.x1 <= figure.x1
    assert label.y0 >= figure.y0
    assert label.y1 <= figure.y1


# --------------------------------------------------------------------------
# Rate summary panel B: box statistics in log space (task 26d)
# --------------------------------------------------------------------------


@pytest.fixture
def default_style():
    """Draw with matplotlib's defaults, whatever an earlier test leaked.

    ``generate_figure02`` and ``generate_figure03`` call
    ``set_figure_defaults("paper")`` when they are imported, which sets font size
    7, dpi 450 and constrained layout for the rest of the process. The figure
    scripts these tests draw run on the defaults, and a test of text that must
    fit inside a figure passes for the wrong reason on the paper style.
    """
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        yield


def _log_box_statistics(values: np.ndarray) -> dict[str, float]:
    """Quartiles and 1.5 IQR whiskers of ``log(values)``, clipped to the data.

    The whisker is the most extreme data point within 1.5 IQR of the quartile,
    which is how matplotlib clips it.
    """
    beta = np.log(values)
    q1, q3 = np.percentile(beta, [25, 75])
    iqr = q3 - q1
    return {
        "q1": q1,
        "q3": q3,
        "bottom": float(beta[beta >= q1 - 1.5 * iqr].min()),
        "top": float(beta[beta <= q3 + 1.5 * iqr].max()),
    }


@pytest.mark.usefixtures("default_style")
def test_the_rate_summary_panel_b_takes_its_box_statistics_in_log_space():
    """A rate ratio is skewed, so on RR the 1.5 IQR whisker runs to the minimum:
    a near-separated session (RR 0.0006 in the R6b fit) stretched the axis over
    five decades. The panel boxes the coefficient beta = log(RR) instead.

    The axis is therefore linear in beta by design, not a log axis; its ticks
    sit at beta = log(RR) for plain RR values and are labelled with the RR."""
    import matplotlib.pyplot as plt

    summary = importlib.import_module(
        "scripts.replay_models.rate.plot_per_bin_rate_summary"
    )
    rng = np.random.default_rng(0)
    no_swr = np.exp(rng.uniform(np.log(0.5), np.log(20.0), 60))
    no_swr[0] = 1e-4  # one extreme low value among values between 0.5 and 20
    rate_ratios = {
        "swr": 6.4 * np.exp(rng.normal(0.0, 0.6, 60)),
        "non_local_swr": np.geomspace(2.0, 100.0, 61),  # median 14, top whisker 82
        "non_local_no_swr": no_swr,
    }
    frame = pd.concat(
        pd.DataFrame(
            {
                "model": summary.FULL,
                "outcome": outcome,
                f"coef_{summary.IMMEDIATE_TERM}": np.log(values),
            }
        )
        for outcome, values in rate_ratios.items()
    )

    fig, ax = plt.subplots()
    summary._panel_reward1(ax, frame)
    fig.canvas.draw()
    low, high = ax.get_ylim()
    scale = ax.get_yscale()
    # every line the box plot drew (whiskers, caps, medians) and every box
    drawn = np.concatenate(
        [
            line.get_ydata()
            for line in ax.lines
            if line.get_linestyle() in ("-", "--") and len(line.get_ydata())
        ]
    )
    box_tops = [patch.get_path().vertices[:, 1].max() for patch in ax.patches]
    # a whisker is the vertical line from a quartile to its end
    whisker_y = np.concatenate(
        [
            line.get_ydata()
            for line in ax.lines
            if len(line.get_ydata()) == 2 and line.get_ydata()[0] != line.get_ydata()[1]
        ]
    )
    one_lines = [
        line.get_ydata()
        for line in ax.lines
        if line.get_linestyle() == ":" and len(line.get_ydata())
    ]
    ticks = [tick for tick in ax.get_yticks() if low <= tick <= high]
    tick_labels = [
        label.get_text()
        for tick, label in zip(ax.get_yticks(), ax.get_yticklabels(), strict=True)
        if low <= tick <= high
    ]
    plt.close(fig)

    stats = [_log_box_statistics(values) for values in rate_ratios.values()]
    top = max(s["top"] for s in stats)
    bottom = min(s["bottom"] for s in stats)
    # the spread to cover: a top whisker near 80, and the extreme value far
    # below the log-space bottom whisker
    assert top > np.log(60.0)
    assert np.log(1e-4) < bottom - 3.0
    assert scale == "linear"  # linear in beta = log(RR), by design
    # the whiskers are 1.5 IQR in log space, clipped to the data
    expected = [s[key] for s in stats for key in ("q1", "q3", "bottom", "top")]
    np.testing.assert_allclose(np.sort(whisker_y), np.sort(expected))
    # the extreme low value is a flier, so it does not set the limit
    assert low > np.log(1e-4)
    # the limits hold every whisker, cap, median and box, with a small margin
    assert low <= bottom and top <= high
    assert drawn.min() >= low and drawn.max() <= high
    assert max(box_tops) <= high
    assert high - top <= np.log(1.3)
    assert bottom - low <= np.log(1.3)
    # RR = 1 is beta = 0: inside the limits, and drawn there
    assert low < 0.0 < high
    assert any(np.allclose(y, 0.0) for y in one_lines)
    # ticks at log of plain 1-2-5 rate ratios, labelled with the RR
    assert len(ticks) >= 3
    assert all("^" not in text and "$" not in text for text in tick_labels)
    rate_ratio_labels = [float(text) for text in tick_labels]
    np.testing.assert_allclose(ticks, np.log(rate_ratio_labels))
    for rate_ratio in rate_ratio_labels:
        mantissa = rate_ratio / 10.0 ** np.floor(np.log10(rate_ratio))
        assert round(mantissa, 6) in (1.0, 2.0, 5.0)
    assert "1" in tick_labels


# --------------------------------------------------------------------------
# Contrast coefficient figures: titles and x labels inside the figure (task 26d)
# --------------------------------------------------------------------------

CONTRAST_SCRIPT = REPO / "scripts" / "figures" / "plot_swr_contrast_coefs.py"
# The statements of the script's top level the test keeps: the plotting
# functions and the row lists they draw. Its other top-level statements (and
# ``main()``, which gates on ``Processed-Data`` and reads the fits) are left out.
_CONTRAST_FUNCTIONS = {
    "forest",
    "rate_figure",
    "content_arm_figure",
    "content_patch_figure",
}
_CONTRAST_NAMES = {
    "ANIM",
    "CMAP",
    "JITTER_RNG",
    "BOOTSTRAP_SEED",
    "N_BOOTSTRAP",
    "SWR",
    "NON",
    "handles",
    "rate_rows",
    "mod_rows",
    "con_rows",
    "patch_mod_rows",
    "patch_con_rows",
}


def _contrast_plotting_namespace() -> dict:
    """The script's own plotting code, run on synthetic coefficients.

    ``sessions`` and ``std_factor`` are the two functions that need the fits'
    source tables; they are replaced by stand-ins that give each term random
    per-animal coefficients and the dagger of its real kind.
    """
    import matplotlib.pyplot as plt

    from continuum_swr_replay.analysis.replay_models.coefficient_statistics import (
        animal_median_bootstrap,
    )
    from continuum_swr_replay.data_loaders.epoch_register import analysed_animals

    tree = ast.parse(CONTRAST_SCRIPT.read_text())
    kept = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _CONTRAST_FUNCTIONS:
            kept.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                n.id
                for target in node.targets
                for n in ast.walk(target)
                if isinstance(n, ast.Name)
            }
            if names & _CONTRAST_NAMES:
                kept.append(node)
    module = ast.Module(body=kept, type_ignores=[])

    def std_factor(term, sd):
        if term.startswith("swr_x_is_") or term.startswith("is_"):
            return 1.0, "per_category"
        if "reward_history_rcos" in term:
            return 1.0, "per_unit"
        return 1.0, "per_SD"

    rng = np.random.default_rng(1)

    def sessions(df, mod, pop, term, sd):
        return {a: rng.normal(0.0, 1.0, 30) for a in namespace["ANIM"]}

    namespace = {
        "np": np,
        "pd": pd,
        "plt": plt,
        "analysed_animals": analysed_animals,
        "animal_median_bootstrap": animal_median_bootstrap,
        "std_factor": std_factor,
        "sessions": sessions,
    }
    exec(compile(module, str(CONTRAST_SCRIPT), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.usefixtures("default_style")
@pytest.mark.parametrize(
    "figure_function", ["rate_figure", "content_arm_figure", "content_patch_figure"]
)
def test_the_contrast_figures_keep_titles_and_x_labels_inside_the_figure(
    figure_function,
):
    """The rate figure's title ran off the right edge and the content figures'
    right-hand x label was cut off, as saved (no ``bbox_inches="tight"``)."""
    import matplotlib.pyplot as plt

    fig = _contrast_plotting_namespace()[figure_function](None, {})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure = fig.bbox
    texts = {}
    for i, ax in enumerate(fig.axes):
        texts[f"axes {i} title"] = ax.title
        texts[f"axes {i} x label"] = ax.xaxis.label
    extents = {name: text.get_window_extent(renderer) for name, text in texts.items()}
    plt.close(fig)

    assert extents
    for name, box in extents.items():
        assert box.x0 >= figure.x0 - 0.5, f"{name} runs off the left edge"
        assert box.x1 <= figure.x1 + 0.5, f"{name} runs off the right edge"
        assert box.y0 >= figure.y0 - 0.5, f"{name} runs off the bottom"
        assert box.y1 <= figure.y1 + 0.5, f"{name} runs off the top"


@pytest.mark.parametrize(("n_separated", "drawn"), [(2, True), (3, False)])
def test_the_contrast_figure_draws_no_term_the_audited_table_refuses(
    n_separated, drawn
):
    """The figure used to apply only the separation filter, so it drew terms the
    audited aggregation refuses for too large an unidentified share."""
    from scripts.figures import plot_swr_contrast_coefs as figure

    n = 10
    term = "swr_x_is_current_arm"
    fits = pd.DataFrame(
        {
            "model": "CS1_current",
            "population": "all",
            "animal": [figure.ANIM[i % len(figure.ANIM)] for i in range(n)],
            f"coef_{term}": np.linspace(0.1, 1.0, n),
            "separated_terms": [""] * (n - n_separated) + [term] * n_separated,
        }
    )

    per_animal = figure.sessions(fits, "CS1_current", "all", term, {})

    assert bool(per_animal) == drawn
    if drawn:
        assert sum(len(v) for v in per_animal.values()) == n - n_separated


# --------------------------------------------------------------------------
# Model-checking figure: the retention note and the ISI autocorrelation panel
# --------------------------------------------------------------------------


def _fits_with_arrival_cohort(swr: tuple[int, int], no_swr: tuple[int, int]):
    """A fit frame whose ``attrs`` hold the arrival cohort tallies of the runner.

    Each tuple is (event bins kept by the arrival window, event bins in the
    frozen cohort) for one population; an event bin is a bin with ``count > 0``.
    """
    fits = pd.DataFrame(
        {
            "outcome": ["swr", "non_local_no_swr"],
            "model": "M0_const",
            "coef_Intercept": 0.0,
            "aic": 10.0,
            "separated": False,
            "ks_within_band": 0.5,
            "ks_statistic": 0.1,
            "dispersion": 1.0,
            "cv_bits_per_event": 0.0,
        }
    )
    fits.attrs["arrival_clock_cohort"] = {
        population: {
            "n_event_bins_arrival_window": kept,
            "n_event_bins_frozen": frozen,
            "n_bins_frozen": 1000,
            "n_bins_dropped_arrival_clock": 0,
            "n_bins_arrival_window": 900,
        }
        for population, (kept, frozen) in (
            ("swr", swr),
            ("non_local", (5, 10)),
            ("non_local_swr", (6, 10)),
            ("non_local_no_swr", no_swr),
        )
    }
    return fits


@pytest.mark.parametrize(
    ("swr", "no_swr", "swr_pct", "no_swr_pct"),
    [((190, 200), (7, 10), 95, 70), ((1, 3), (2, 3), 33, 67)],
)
def test_the_arrival_retention_note_gives_the_shares_the_fit_stored(
    swr, no_swr, swr_pct, no_swr_pct
):
    """The shares are the SWR and the non-local no-SWR populations' event bins
    kept by the arrival window, over their frozen cohort, in whole percent."""
    checking = importlib.import_module("scripts.replay_models.rate.plot_model_checking")

    note = checking.arrival_retention_note(_fits_with_arrival_cohort(swr, no_swr))

    assert (
        f"keeps {swr_pct} % of SWR event bins vs {no_swr_pct} % of non-SWR non-local"
        " event bins" in note
    )
    assert "retention" in note


def test_the_arrival_retention_note_is_words_only_without_the_cohort_tallies():
    checking = importlib.import_module("scripts.replay_models.rate.plot_model_checking")
    fits = _fits_with_arrival_cohort((190, 200), (7, 10))
    del fits.attrs["arrival_clock_cohort"]

    note = checking.arrival_retention_note(fits)

    assert note == checking.ARRIVAL_RETENTION_NOTE
    assert not re.search(r"\d|%", note)
    # No direction: the order differs between fits.
    assert "different shares of SWR and of non-SWR" in note
    assert "retention" in note


@pytest.mark.usefixtures("default_style")
def test_the_gof_ladder_title_carries_the_computed_retention_note(
    processed_data_root, monkeypatch
):
    import matplotlib.pyplot as plt

    # the figure goes beside the data root, under processed_data_root's tmp_path
    checking = importlib.import_module("scripts.replay_models.rate.plot_model_checking")
    figures = []
    # ``checking.plt`` is pyplot itself, so keep the real close for the cleanup.
    real_close = plt.close
    monkeypatch.setattr(checking.plt, "close", figures.append)

    checking.plot_gof_ladder(_fits_with_arrival_cohort((190, 200), (7, 10)))

    (fig,) = figures
    title = fig.get_suptitle()
    real_close("all")
    assert "keeps 95 % of SWR event bins vs 70 % of non-SWR non-local" in title


def _symmetric_autocorrelation(n_spikes: int) -> np.ndarray:
    """A normalised autocorrelation of length ``2 * n_spikes - 1``."""
    from scipy.signal import correlate

    values = np.random.default_rng(2).normal(size=n_spikes)
    c = correlate(values, values)
    return c / c.max()


@pytest.mark.parametrize(
    ("n_spikes", "lag_max"), [(100, 50), (100, 3), (60, 60), (30, 50)]
)
def test_the_isi_autocorrelation_window_is_centred_on_the_zero_lag(n_spikes, lag_max):
    """non_local_detector#64: ``slice(-lag_max + 1, lag_max)`` is empty for every
    ``lag_max <= n_spikes`` because its start counts from the end of the array and
    its stop from the start."""
    checking = importlib.import_module("scripts.replay_models.rate.plot_model_checking")
    autocorrelation = _symmetric_autocorrelation(n_spikes)
    assert autocorrelation.size == 2 * n_spikes - 1

    lags, values = checking.isi_autocorrelation_window(autocorrelation, lag_max)

    half = min(lag_max, n_spikes) - 1
    assert lags.size == values.size == 2 * half + 1
    np.testing.assert_array_equal(lags, np.arange(-half, half + 1))
    # lag 0 is the autocorrelation's maximum, 1.0, at index n_spikes - 1
    assert values[lags == 0] == pytest.approx(1.0)
    np.testing.assert_array_equal(values, autocorrelation[n_spikes - 1 + lags])
    np.testing.assert_allclose(values, values[::-1])


@pytest.mark.usefixtures("default_style")
def test_the_isi_autocorrelation_panel_draws_its_points_and_the_bounds():
    import matplotlib.pyplot as plt
    from non_local_detector.model_checking.sorted_spikes import TimeRescaling

    checking = importlib.import_module("scripts.replay_models.rate.plot_model_checking")
    rng = np.random.default_rng(3)
    rate = np.full(4000, 0.05)
    is_spike = rng.random(rate.size) < rate
    trial_id = np.repeat(np.arange(40), 100)
    tr = TimeRescaling(rate, is_spike, trial_id, adjust_for_short_trials=True)
    autocorrelation = tr.rescaled_ISI_autocorrelation()
    n_spikes = autocorrelation.size // 2 + 1
    assert n_spikes > 50

    fig, ax = plt.subplots()
    checking.plot_rescaled_isi_autocorrelation(ax, tr, lag_max=50)
    offsets = ax.collections[0].get_offsets()
    bounds = sorted(line.get_ydata()[0] for line in ax.lines)
    plt.close(fig)

    assert offsets.shape == (2 * 50 - 1, 2)
    np.testing.assert_array_equal(offsets[:, 0], np.arange(-49, 50))
    np.testing.assert_allclose(
        offsets[:, 1], autocorrelation[n_spikes - 1 - 49 : n_spikes + 49]
    )
    ci = 1.96 / np.sqrt(n_spikes)
    np.testing.assert_allclose(bounds, [-ci, ci])


# --------------------------------------------------------------------------
# The uncertainty-content figure reads the analysis's animal counts
# --------------------------------------------------------------------------


def _uncertainty_contrasts() -> pd.DataFrame:
    """Q4 - Q1 contrast rows of ``analyze_uncertainty_content``: the overall
    posterior unchosen-well row of each metric among rows of other responses,
    scopes and levels, each with its own counts. The other rows come first, so
    a reader that drops a filter reads one of them."""
    rows = [
        # (metric, response, scope, level, n_animals_positive, n_animals)
        ("conditional_event_share", "hard", "unchosen", "overall", 5, 5),
        ("conditional_event_share", "posterior", "all", "overall", 1, 5),
        ("conditional_event_share", "posterior", "unchosen", "animal", 1, 1),
        ("absolute_rate", "posterior", "unchosen", "overall", 4, 4),
        ("conditional_event_share", "posterior", "unchosen", "overall", 3, 5),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "response",
            "scope",
            "level",
            "n_animals_positive",
            "n_animals",
        ],
    )


def test_the_uncertainty_annotation_counts_the_overall_posterior_row():
    figure = importlib.import_module(
        "scripts.figures.generate_uncertainty_content_figure"
    )
    contrasts = _uncertainty_contrasts()
    assert (
        figure._positive_animals_note(contrasts, "conditional_event_share")
        == "Q4 − Q1 > 0 in 3/5 animals"
    )
    assert (
        figure._positive_animals_note(contrasts, "absolute_rate")
        == "Q4 − Q1 > 0 in 4/4 animals"
    )


def test_the_uncertainty_figure_refuses_contrasts_without_the_animal_counts(tmp_path):
    figure = importlib.import_module(
        "scripts.figures.generate_uncertainty_content_figure"
    )
    for name in (
        "trial_leaf_table.pkl",
        "quartile_profiles.csv",
        "mu_session_coefficients.csv",
        "mu_animal_coefficients.csv",
        "mu_overall_coefficients.csv",
        "horserace_summary.csv",
    ):
        (tmp_path / name).touch()
    (tmp_path / "cohort_audit.json").write_text(
        json.dumps({"arm_standardization_support": "leaf_arms"})
    )
    _uncertainty_contrasts().drop(columns="n_animals_positive").to_csv(
        tmp_path / "quartile_contrasts.csv", index=False
    )
    with pytest.raises(ValueError, match="animals-positive count"):
        figure._load_results(tmp_path)


def test_the_uncertainty_annotation_refuses_a_metric_without_an_overall_row():
    figure = importlib.import_module(
        "scripts.figures.generate_uncertainty_content_figure"
    )
    contrasts = _uncertainty_contrasts()
    contrasts = contrasts[~contrasts["metric"].eq("absolute_rate")]
    with pytest.raises(ValueError, match="no overall posterior row for absolute_rate"):
        figure._positive_animals_note(contrasts, "absolute_rate")
