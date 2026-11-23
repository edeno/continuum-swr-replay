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
    analyzed_time,
    swr_source,
)
from continuum_swr_replay.tables.io import write_table  # noqa: E402
from continuum_swr_replay.visualization import heldout_gain_panels  # noqa: E402
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

    ((start, end),) = analyzed_time.excluded_intervals(EPOCH_ID).to_numpy()
    np.testing.assert_array_equal(
        dot["in_excluded_time"].to_numpy(), (TIME >= start) & (TIME <= end)
    )
    assert dot["in_excluded_time"].any()


# --------------------------------------------------------------------------
# Figure 2's render cache
# --------------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    """Import a script with ``spyglass`` stubbed; forget it afterward."""
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
            ["wilbur20210329_.nwb", epoch, "analyzed", "", "", "False"]
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
            "visit_context": "in_visit",
            "whole_event_speed_eligible": True,
            "event_start_s": [0.0025, -0.25, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "event_end_s": [0.25, -0.0045, 2.0, 3.1, 4.1, 5.1, 6.1, 7.1],
            "matched_swr_start": [0.0, -0.125, 0.5, nan, nan, nan, nan, nan],
            "matched_swr_end": [0.5, 0.0, 1.5, nan, nan, nan, nan, nan],
        }
    )


#: The decode frames' event counts by SWR overlap that agree with
#: ``_figure02_events``'s overlap classes.
FIGURE02_FRAME_COUNTS = {"swr": 3, "non_swr": 5, "total": 8}


def test_figure02_refuses_an_event_table_that_is_not_the_frames(offline):
    """Panel d's counts come from the event table the summary cites, and the
    figure is refused when that table's event counts by SWR overlap are not the
    decode frames', or the table holds an event that is not immobile."""
    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        overlap_class_statistics,
    )

    figure02 = offline("scripts.figures.generate_figure02")
    events = _figure02_events()
    classes = overlap_class_statistics(events)
    figure02.check_table_against_frames(classes, FIGURE02_FRAME_COUNTS)
    with pytest.raises(ValueError, match="counts by SWR overlap are not"):
        figure02.check_table_against_frames(
            classes, {**FIGURE02_FRAME_COUNTS, "non_swr": 6, "total": 9}
        )
    moving = events.assign(whole_event_speed_eligible=[True] * 7 + [False])
    with pytest.raises(ValueError, match="not whole-event immobile"):
        overlap_class_statistics(moving)


def test_figure02_panel_d_prints_the_digits_of_the_events_summary(offline):
    """The bars are the mean over animals of the animals' dots, their percents
    rounded by the summary's rule (a half rounds away from zero: 12.5 % is
    13), the title gives the count of every event, and each animal is one dot
    per bar."""
    import matplotlib.pyplot as plt

    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        overlap_class_animal_shares,
        overlap_class_statistics,
    )
    from continuum_swr_replay.reported.events import event_cohort_values
    from continuum_swr_replay.reported.formatting import formatted

    figure02 = offline("scripts.figures.generate_figure02")
    events = _figure02_events()
    animals = overlap_class_animal_shares(events)
    fig, ax = plt.subplots()
    figure02.draw_events_bars(ax, overlap_class_statistics(events), animals)
    texts = [t.get_text() for t in ax.texts]
    assert texts == ["13%", "25%", "63%"]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["No SWR", "Partly in SWR", "Inside SWR"]  # bottom to top
    assert ax.get_title(loc="right") == "Non-local events (n = 8)"
    assert all(len(dots.get_offsets()) == len(animals) for dots in ax.collections)
    widths = [bar.get_width() for bar in ax.patches]  # top to bottom
    np.testing.assert_allclose(
        widths, 100 * animals[list(figure02.CLASS_ORDER)].mean().to_numpy()
    )
    cohort = {v.key: v for v in event_cohort_values(events, "s")}
    assert formatted(cohort["non_swr_fraction_animal_mean"]) == "63 %"
    plt.close(fig)


def test_figure02_checks_the_event_table_against_the_cached_frame_counts(
    offline, tmp_path
):
    """With a cached bundle, the event table in the data folder is still read
    when the figure is drawn and checked against the bundle's frame counts."""
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
    write_table(analyzed_time.stamp_swr_source(_figure02_events()), table)
    cache = figure02._figure02_cache_path("wilbur20210329_.nwb", "02_r1", tmp_path)
    cache.parent.mkdir(parents=True)
    data = {"ripple_times": pd.DataFrame(columns=["start_time", "end_time"])}
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


def test_figure02_checks_the_whole_table_and_draws_the_selected_events(
    offline, tmp_path, monkeypatch
):
    """Two non-SWR events on the track, between two visits: the frames hold
    them, so the check against the frames counts them; every panel is drawn
    from the other eight, the events that start inside a well visit."""
    import joblib

    from continuum_swr_replay.paths import (
        processed_data_dir,
        replay_content_dir,
        table_path,
    )

    figure02 = offline("scripts.figures.generate_figure02")
    outside = _figure02_events().iloc[[3, 4]].assign(visit_context="track_leaf_arm")
    events = pd.concat([_figure02_events(), outside], ignore_index=True)
    table = table_path(
        "event_table", tmp_path / replay_content_dir().relative_to(processed_data_dir())
    )
    table.parent.mkdir(parents=True)
    write_table(analyzed_time.stamp_swr_source(events), table)
    cache = figure02._figure02_cache_path("wilbur20210329_.nwb", "e1", tmp_path)
    cache.parent.mkdir(parents=True)
    data = {"ripple_times": pd.DataFrame(columns=["start_time", "end_time"])}
    frames = {**FIGURE02_FRAME_COUNTS, "non_swr": 7, "total": 10}
    joblib.dump({"frame_event_counts": frames, "data": data, "results": None}, cache)
    seen = []
    counted = figure02.overlap_class_statistics

    def count(table):
        seen.append(len(table))
        return counted(table)

    class Drawn(Exception):
        pass

    def stop(session_events, *args):
        seen.append(session_events)
        raise Drawn

    monkeypatch.setattr(figure02, "overlap_class_statistics", count)
    monkeypatch.setattr(figure02, "example_windows", stop)
    monkeypatch.setattr(figure02, "parse_nwb_filename", lambda name: ("a", "d1"))
    with pytest.raises(Drawn):
        figure02.generate_figure(epoch_name="e1", data_dir=tmp_path)
    # the check saw all ten; panel d's classes and the session's events are eight
    assert seen[:2] == [10, 8]
    assert len(seen[2]) == 8
    assert set(seen[2]["visit_context"]) == {"in_visit"}


def _figure02_example_table(figure02) -> pd.DataFrame:
    """One row per fixed example, each of its drawn subclass, and one other.
    The partial examples run past their SWR at the start (20 s) and at the end
    (30 s)."""
    ids = [event_id for event_id, _ in figure02.EXAMPLES.values()]
    subclasses = [subclass for _, subclass in figure02.EXAMPLES.values()]
    nan = np.nan
    return pd.DataFrame(
        {
            "event_id": [*ids, 1],
            "overlap_subclass": [*subclasses, "no_overlap"],
            "event_start_s": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "event_end_s": [10.1, 20.1, 30.1, 40.1, 50.1, 60.1],
            "matched_swr_start": [9.9, 20.05, 29.9, nan, nan, nan],
            "matched_swr_end": [10.2, 20.2, 30.05, nan, nan, nan],
        }
    )


def _figure02_local_results(figure02, local: float):
    import xarray as xr

    start = figure02.LOCAL_SWR_START_S
    time = start + np.arange(-0.05, 0.15, 0.002)
    states = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]
    probabilities = np.tile([local, 1 - local, 0.0, 0.0], (time.size, 1))
    return xr.Dataset(
        {"acausal_state_probabilities": (("time", "states"), probabilities)},
        coords={"time": time, "states": states},
    )


def test_figure02_examples_are_checked_rows_of_the_event_table(offline):
    figure02 = offline("scripts.figures.generate_figure02")
    events = _figure02_example_table(figure02)
    start = figure02.LOCAL_SWR_START_S
    ripples = pd.DataFrame({"start_time": [start], "end_time": [start + 0.1]})

    windows = figure02.example_windows(
        events, ripples, _figure02_local_results(figure02, 0.95)
    )
    assert list(windows) == [*figure02.EXAMPLES, "SWR, mostly Local"]
    assert windows["Inside SWR"] == (10.0, 10.1)
    assert windows["SWR, mostly Local"] == (start, start + 0.1)

    # an example of another subclass than it is drawn as is refused
    wrong = events.assign(
        overlap_subclass=["no_overlap", *events["overlap_subclass"][1:]]
    )
    with pytest.raises(AssertionError, match="Inside SWR"):
        figure02.example_windows(
            wrong, ripples, _figure02_local_results(figure02, 0.95)
        )
    # a "starts before" example that also ends after its SWR is refused
    both = events.assign(matched_swr_end=[10.2, 20.08, 30.05, *[np.nan] * 3])
    with pytest.raises(AssertionError, match="Starts before SWR.*start alone"):
        figure02.example_windows(both, ripples, _figure02_local_results(figure02, 0.95))
    # the Local SWR must be the source's, empty of events, and mostly Local
    with pytest.raises(AssertionError, match="not one of the source's"):
        figure02.example_windows(
            events,
            ripples.assign(start_time=start + 1.0),
            _figure02_local_results(figure02, 0.95),
        )
    overlapping = pd.concat(
        [
            events,
            pd.DataFrame(
                {
                    "event_id": [2],
                    "overlap_subclass": ["swr_contains_event"],
                    "event_start_s": [start + 0.01],
                    "event_end_s": [start + 0.05],
                }
            ),
        ]
    )
    with pytest.raises(AssertionError, match="overlaps the Local SWR"):
        figure02.example_windows(
            overlapping, ripples, _figure02_local_results(figure02, 0.95)
        )
    with pytest.raises(AssertionError, match="not mostly Local"):
        figure02.example_windows(
            events, ripples, _figure02_local_results(figure02, 0.5)
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


def test_the_figure02_cache_is_keyed_on_the_analyzed_epochs(
    offline, tmp_path, monkeypatch
):
    """Panel e summarizes every analyzed epoch, so a register change must make
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
        # the neighbors that set the room panel e gets
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
    sit at beta = log(RR) for plain RR values and are labeled with the RR."""
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
    # ticks at log of plain 1-2-5 rate ratios, labeled with the RR
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
    from continuum_swr_replay.data_loaders.epoch_register import analyzed_animals

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
        "analyzed_animals": analyzed_animals,
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
    """A normalized autocorrelation of length ``2 * n_spikes - 1``."""
    from scipy.signal import correlate

    values = np.random.default_rng(2).normal(size=n_spikes)
    c = correlate(values, values)
    return c / c.max()


@pytest.mark.parametrize(
    ("n_spikes", "lag_max"), [(100, 50), (100, 3), (60, 60), (30, 50)]
)
def test_the_isi_autocorrelation_window_is_centered_on_the_zero_lag(n_spikes, lag_max):
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


def test_figure02_panel_j_writes_the_share_the_events_summary_gives(offline):
    """The share of non-local time below the SWR threshold on the panel is the
    events summary's value (the animals' mean, the bins beyond the axis
    counted), rounded by the summary's rule; each animal is one line on each
    axis, with the animals' mean."""
    import matplotlib.pyplot as plt

    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        animal_signal_counts,
    )
    from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
        RIPPLE_POWER_EDGES,
        nonlocal_counts,
    )
    from continuum_swr_replay.data_loaders.epoch_register import analyzed_animals
    from continuum_swr_replay.reported.events import nonlocal_power_values
    from continuum_swr_replay.reported.formatting import formatted

    figure02 = offline("scripts.figures.generate_figure02")
    low, high = RIPPLE_POWER_EDGES[:-1], RIPPLE_POWER_EDGES[1:]
    animals = list(analyzed_animals())
    epochs = []
    for i, animal in enumerate(animals):
        # 8 non-local bins below the lowest finite edge, 4 above the highest,
        # and 4 just above the threshold for every animal after the first
        n_nonlocal = np.zeros(len(low), dtype=int)
        n_nonlocal[0], n_nonlocal[-1] = 8, 4
        n_nonlocal[np.flatnonzero(low == 2.0)[0]] = 4 * i
        # the stored table: each signal bin's non-local bins in the last
        # probability bin, the rest in the first
        for probability_low, n in ((0.0, 20 - n_nonlocal), (0.8, n_nonlocal)):
            epochs.append(
                pd.DataFrame(
                    {
                        "epoch_id": f"{animal}_e",
                        "animal": animal,
                        "signal": "ripple_power",
                        "signal_low": low,
                        "signal_high": high,
                        "probability_low": probability_low,
                        "probability_high": probability_low + 0.2,
                        "n_bins": n,
                    }
                )
            )
    counts = pd.concat(epochs, ignore_index=True)
    fig, (ax_spread, ax_share) = plt.subplots(2, 1, sharex=True)
    figure02.draw_share_by_power(
        ax_spread,
        ax_share,
        animal_signal_counts(nonlocal_counts(counts, "ripple_power")),
    )
    value = {v.key: v for v in nonlocal_power_values(counts, "s")}[
        "nonlocal_time_below_swr_threshold_fraction"
    ]
    # 8 / (12 + 4 i) over i = 0..4: mean 0.4371, although no bin on the axis
    # below the threshold holds a non-local bin
    assert formatted(value) == "44 %"
    assert ax_spread.texts[0].get_text() == "44% below\nthreshold"
    for ax in (ax_spread, ax_share):
        assert (
            len(ax.lines) == len(animals) + 2
        )  # the animals, their mean, the threshold
        assert ax.get_xlim() == (-2.0, 10.0)
    plt.close(fig)


def test_no_swr_events_have_one_color_in_every_figure(offline):
    """A non-local event that overlaps no SWR is drawn in ``COLORS["No SWR"]`` in
    Figures 2, 3 and 5 and in the shared per-event panels: not in the Non-Local
    state's green, which every non-local event is drawn in."""
    from continuum_swr_replay.visualization import figure_panels
    from continuum_swr_replay.visualization.palettes import COLORS

    figure02 = offline("scripts.figures.generate_figure02")
    figure03 = offline("scripts.figures.generate_figure03")
    figure05 = offline("scripts.figures.generate_figure05")
    no_swr = COLORS["No SWR"]
    assert no_swr not in (COLORS["Non-Local"], COLORS["SWR"])
    assert figure02.CLASS_COLORS["non_swr"] == no_swr
    assert {name: color for _, name, color in figure03.KINDS}["No SWR"] == no_swr
    bars = {label: COLORS[color] for label, _, color in figure05.SIGNIFICANCE_BARS}
    assert bars["No SWR"] == no_swr
    assert bars["Non-local"] == COLORS["Non-Local"]  # every non-local event
    classes = {word: color for _, word, color, _ in figure_panels._EVENT_CLASSES}
    assert classes["non_swr"] == no_swr


# ---------------------------------------------------------------------------
# Figure 3: what a reader can read off the axes
# ---------------------------------------------------------------------------


def _figure03_events(n: int = 400) -> pd.DataFrame:
    """Events of both kinds, of two animals with two epochs each, whose
    durations pass the duration axis and whose speeds pass the speed axis, as
    the real ones do."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "animal": np.where(np.arange(n) < n // 2, "a", "b"),
            "session_date": "d1",
            "epoch": np.where(np.arange(n) % 4 < 2, "e1", "e2"),
            "is_swr": np.arange(n) % 2 == 0,
            "event_duration_s": np.r_[rng.uniform(0.02, 0.3, n - 2), 0.45, 0.6],
            "spatial_extent_cm": rng.uniform(0, 220, n),
            "replay_speed_median_cm_s": np.r_[
                np.zeros(4), rng.lognormal(6, 2, n - 4)
            ].clip(0, 9e4),
        }
    )


def _tick_labels(fig, ax):
    """The drawn x tick labels of an axis, left to right, with their extents."""
    fig.draw_without_rendering()
    labels = [
        label
        for label in ax.get_xticklabels()
        if label.get_visible() and label.get_text()
    ]
    labels.sort(key=lambda label: label.get_window_extent().x0)
    return labels


@pytest.mark.parametrize("key", ["duration", "extent", "speed"])
def test_figure03_histogram_tick_labels_do_not_collide(offline, key):
    """At the panel's printed width, no x tick label runs into the next one."""
    import matplotlib.pyplot as plt

    figure03 = offline("scripts.figures.generate_figure03")
    fig = plt.figure(figsize=(figure03.WIDTH, figure03.HEIGHT))
    ax = fig.add_axes([0.1, 0.6, 0.12, 0.2])  # a row-1 panel's size
    figure03.draw_histogram(ax, _figure03_events(), key)
    extents = [label.get_window_extent() for label in _tick_labels(fig, ax)]
    assert len(extents) >= 2
    for left, right in zip(extents[:-1], extents[1:], strict=True):
        assert left.x1 < right.x0
    plt.close(fig)


def test_figure03_histogram_labels_the_ends_that_hold_clipped_events(offline):
    """An end bin that holds the events beyond the axis says so on its tick; an
    end with no such label holds none, and the panel is not drawn if it would."""
    import matplotlib.pyplot as plt

    figure03 = offline("scripts.figures.generate_figure03")
    events = _figure03_events()
    fig, (ax_duration, ax_speed, ax_extent) = plt.subplots(1, 3)
    figure03.draw_histogram(ax_duration, events, "duration")
    assert _tick_labels(fig, ax_duration)[-1].get_text() == "≥400"
    figure03.draw_histogram(ax_speed, events, "speed")
    assert _tick_labels(fig, ax_speed)[-1].get_text() == "≥1500"

    too_wide = events.assign(spatial_extent_cm=300.0)
    with pytest.raises(ValueError, match="beyond the axis, unmarked"):
        figure03.draw_histogram(ax_extent, too_wide, "extent")
    plt.close(fig)


def test_figure03_histogram_marks_each_animal_s_median(offline):
    """Under each kind's median, one tick per animal at the animal's own median
    (``event_statistics.animal_class_medians``), in the kind's color."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        animal_class_medians,
    )

    figure03 = offline("scripts.figures.generate_figure03")
    events = _figure03_events()
    fig, ax = plt.subplots()
    figure03.draw_histogram(ax, events, "extent")
    expected = animal_class_medians(events, "spatial_extent_cm")
    ticks = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(ticks) == 2
    for (_, _, color), word, drawn in zip(
        figure03.KINDS, ("non_swr", "swr"), ticks, strict=True
    ):
        x = [segment[0, 0] for segment in drawn.get_segments()]
        np.testing.assert_allclose(x, expected[word])
        np.testing.assert_allclose(drawn.get_colors()[0], to_rgba(color))
    plt.close(fig)


def test_figure03_cumulative_is_the_share_of_events_at_or_below(offline):
    """Panel d's lower axis: each kind's curve is the share of its events at or
    below each speed, so it crosses 50 % at the kind's median and ends short of
    100 % by the share of events beyond the axis; an event without a value
    raises, as in the histogram above it."""
    import matplotlib.pyplot as plt

    figure03 = offline("scripts.figures.generate_figure03")
    events = _figure03_events(n=2400)
    column, _, _, edges, _ = figure03.HISTOGRAMS["speed"]
    fig, ax = plt.subplots()
    figure03.draw_cumulative(ax, events, "speed")
    curves = [line for line in ax.lines if len(line.get_xdata()) > 2]
    assert len(curves) == 2
    for (is_swr, _, _), curve in zip(figure03.KINDS, curves, strict=True):
        speed, share = curve.get_xdata(), curve.get_ydata()
        values = events.loc[events["is_swr"] == is_swr, column].to_numpy()
        assert (np.diff(share) >= 0).all()
        assert share[-1] == pytest.approx(100 * (values <= edges[-1]).mean())
        assert share[-1] < 100  # some events lie beyond the axis
        median = np.median(values)
        assert share[speed <= median][-1] <= 50 <= share[speed >= median][0]
    assert ax.get_xlim() == (edges[0], edges[-1])

    missing = events.assign(
        replay_speed_median_cm_s=events[column].where(events.index > 0)
    )
    with pytest.raises(ValueError, match="an event has no value"):
        figure03.draw_cumulative(ax, missing, "speed")
    plt.close(fig)


def test_figure03_legend_counts_every_event_of_each_kind(offline):
    """The legend's n is each kind's number of events, and a histogram is not
    drawn from fewer: an event without a value raises."""
    import matplotlib.pyplot as plt

    figure03 = offline("scripts.figures.generate_figure03")
    events = _figure03_events(n=2400)
    assert figure03.kind_labels(events) == [
        "No SWR (n = 1,200 events)",
        "SWR (n = 1,200 events)",
    ]
    fig, ax = plt.subplots()
    missing = events.assign(
        spatial_extent_cm=events["spatial_extent_cm"].where(events.index > 0)
    )
    with pytest.raises(ValueError, match="an event has no value"):
        figure03.draw_histogram(ax, missing, "extent")
    plt.close(fig)


def _spider_track():
    """The task's track: a center (9), three patch junctions (6, 7, 8) and six
    wells (0-5), at one epoch's positions (cm); arms 0-5 are the leaf arms, 6-8
    the trunk arms. Returns the graph and its arms in drawing order."""
    import networkx as nx

    positions = {
        0: (177.0, 45.1), 1: (223.7, 122.3), 2: (181.7, 202.5), 3: (87.9, 204.2),
        4: (40.9, 126.5), 5: (86.2, 45.1), 6: (179.6, 98.7), 7: (135.4, 178.0),
        8: (86.5, 99.6), 9: (134.6, 124.8),
    }  # fmt: skip
    arms = {0: (6, 0), 1: (6, 1), 2: (7, 2), 3: (7, 3), 4: (8, 4), 5: (8, 5),
            6: (9, 6), 7: (9, 7), 8: (9, 8)}  # fmt: skip
    graph = nx.Graph()
    for node, position in positions.items():
        graph.add_node(node, pos=position)
    for arm, (first, second) in arms.items():
        graph.add_edge(first, second, edge_id=arm)
    return graph, [arms[arm] for arm in (6, 0, 1, 7, 2, 3, 8, 4, 5)]


def test_figure03_track_bands_meet_at_a_junction_without_overlapping(offline):
    """Where three arms meet, each arm's band ends on the lines halfway to its
    neighbors: every point of the track lies in exactly one segment's band, at
    the junctions as along the arms, and no segment is left without a band."""
    from matplotlib.path import Path

    figure03 = offline("scripts.figures.generate_figure03")
    graph, edge_order = _spider_track()
    values = pd.Series(
        0.0,
        index=pd.MultiIndex.from_product([range(9), range(figure03.N_SEGMENTS)]),
    )
    polygons, shown = figure03.track_bands(graph, edge_order, values)
    assert len(polygons) == len(shown) == 9 * figure03.N_SEGMENTS
    assert all(len(polygon) >= 3 for polygon in polygons)

    half = figure03.ARM_HALF_WIDTH_CM
    rng = np.random.default_rng(0)
    # points around every junction and the center, and over the whole track
    near = np.concatenate(
        [
            np.asarray(graph.nodes[node]["pos"]) + rng.uniform(-25, 25, (2000, 2))
            for node in (6, 7, 8, 9)
        ]
    )
    points = np.concatenate([near, rng.uniform((20, 30), (245, 220), (6000, 2))])
    in_band = np.zeros(len(points), dtype=bool)
    for first, second in edge_order:
        start = np.asarray(graph.nodes[first]["pos"])
        along = np.asarray(graph.nodes[second]["pos"]) - start
        length = np.hypot(*along)
        offset = points - start
        forward = offset @ along / length
        sideways = np.abs(offset @ np.array([-along[1], along[0]]) / length)
        in_band |= (forward >= 0) & (forward <= length) & (sideways <= half)
    n_holding = sum(Path(polygon).contains_points(points) for polygon in polygons)
    assert in_band.sum() > 3000
    assert (n_holding[in_band] == 1).all()
    assert (n_holding[~in_band] == 0).all()


def test_figure03_density_tracks_draw_each_kind_against_even(offline):
    """Panel e: one track per kind, each segment colored by the kind's density
    against even content, on a scale that saturates and says so at its ends."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    figure03 = offline("scripts.figures.generate_figure03")
    graph, edge_order = _spider_track()
    index = pd.MultiIndex.from_product(
        [[False, True], range(9), range(figure03.N_SEGMENTS)],
        names=["is_swr", "arm", "segment"],
    )
    against_even = pd.Series(np.linspace(-3, 3, len(index)), index=index)
    against_even[(False, 6, 0)] = -np.inf  # a segment without content
    fig, axes = plt.subplots(1, 2)
    figure03.draw_density_tracks(axes, graph, edge_order, against_even)
    limit = figure03.DENSITY_LIMIT_LOG2
    for ax, (is_swr, name, _) in zip(axes, figure03.KINDS, strict=True):
        assert ax.get_title() == name
        (bands,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
        _, values = figure03.track_bands(graph, edge_order, against_even.loc[is_swr])
        np.testing.assert_allclose(bands.get_array(), np.clip(values, -limit, limit))
        assert np.isfinite(bands.get_array()).all()
    bars = [ax for ax in fig.axes if ax not in list(axes)]
    assert len(bars) == 2
    for bar in bars:
        assert [t.get_text() for t in bar.get_xticklabels()] == ["≤¼×", "even", "≥4×"]
    plt.close(fig)


def test_figure03_track_map_names_the_center_a_junction_and_a_well(offline):
    """The profile's axis runs from the center to a junction to a well; the
    track map says where those are. Each name belongs to a node of its sort and
    sits beside it, with no line to it; at the map's printed size no name lies
    over a band or over another name."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.path import Path

    from continuum_swr_replay.visualization.simulated_run_panels import (
        overlapping_texts,
    )

    figure03 = offline("scripts.figures.generate_figure03")
    graph, edge_order = _spider_track()
    values = pd.Series(
        0.0,
        index=pd.MultiIndex.from_product([range(9), range(figure03.N_SEGMENTS)]),
    )
    polygons, _ = figure03.track_bands(graph, edge_order, values)
    fig = plt.figure(figsize=(figure03.WIDTH, figure03.HEIGHT), layout="none")
    ax = fig.add_axes([0.08, 0.15, 0.17, 0.33])  # one map, at its size in the figure
    ax.add_collection(PolyCollection(polygons))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    figure03.label_track(ax, graph)

    named = {text.get_text(): text for text in ax.texts}
    assert sorted(named) == ["center", "junction", "well"]
    position = {node: np.asarray(graph.nodes[node]["pos"]) for node in graph}
    np.testing.assert_allclose(named["center"].xy, position[9])
    assert any(np.allclose(named["junction"].xy, position[n]) for n in (6, 7, 8))
    assert any(np.allclose(named["well"].xy, position[n]) for n in range(6))
    assert all(text.arrow_patch is None for text in named.values())

    fig.draw_without_rendering()
    bands = [Path(ax.transData.transform(polygon)) for polygon in polygons]
    for text in named.values():
        box = text.get_window_extent()
        outline = Path(
            [(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)]
        )
        assert not any(outline.intersects_path(band, filled=True) for band in bands)
        # and beside its node: within two band widths of it
        node = ax.transData.transform(text.xy)
        nearest = np.hypot(
            np.clip(node[0], box.x0, box.x1) - node[0],
            np.clip(node[1], box.y0, box.y1) - node[1],
        )
        two_widths = 4 * figure03.ARM_HALF_WIDTH_CM
        assert nearest <= np.hypot(
            *(ax.transData.transform((two_widths, 0)) - ax.transData.transform((0, 0)))
        )
    assert overlapping_texts(fig) == []
    plt.close(fig)


def test_figure03_profiles_carry_the_even_content_line(offline):
    """Panel g draws what content spread evenly over the track would give: one
    density across both halves, labeled, and its share at a well beside the two
    kinds' shares."""
    import matplotlib.pyplot as plt

    figure03 = offline("scripts.figures.generate_figure03")
    index = pd.MultiIndex.from_product(
        [[False, True], ["trunk", "leaf"], range(figure03.N_SEGMENTS)],
        names=["is_swr", "arm_type", "segment"],
    )
    profiles = pd.DataFrame({"density_per_cm": 0.003}, index=index)
    shares = pd.DataFrame({"at_well": [0.5, 0.2]}, index=[False, True])
    even = {"density_per_cm": 0.002, "at_well_fraction": 0.25}
    fig, ax = plt.subplots()
    figure03.draw_profiles(ax, profiles, shares, even)
    (line,) = [line for line in ax.lines if line.get_gid() == "even"]
    np.testing.assert_allclose(line.get_ydata(), 0.2)  # % per cm
    texts = [text.get_text() for text in ax.texts]
    assert "even" in texts
    assert "even: 25% at a well" in texts
    plt.close(fig)


def test_figure03_draws_six_panels_from_the_decode_and_the_track(offline):
    """Figure 3 is a to f, and the script's letters are the question list's.
    Gamma during the events is the supplementary figure's, and content against
    the animal is the behavior figure's: neither is drawn here."""
    from continuum_swr_replay.reported.questions import FIGURES

    figure03 = offline("scripts.figures.generate_figure03")
    assert "".join(figure03.LETTER_ROWS) == FIGURES["3"].panels == "abcdef"
    assert sorted("".join(figure03.LETTER_COLUMNS)) == list("abcdef")
    assert not hasattr(figure03, "draw_brain_state")
    assert not hasattr(figure03, "draw_distance_from_animal")


def test_the_gamma_figure_draws_each_animal_between_its_rest_and_running_levels(
    offline,
):
    """The supplementary gamma figure: per band and kind, each animal's event
    level (``event_statistics.animal_signal_levels`` with ``matched``) between
    its quiet-rest and running levels, in the kind's color."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        animal_signal_levels,
    )

    figure = offline("scripts.figures.generate_supp_figure03")
    rng = np.random.default_rng(0)
    epochs = [("a", "e1"), ("a", "e2"), ("b", "e1")]
    events = pd.DataFrame(
        [
            {"animal": animal, "session_date": "d1", "epoch": epoch, "is_swr": is_swr}
            for animal, epoch in epochs
            for is_swr in (False, True)
            for _ in range(4)
        ]
    )
    reference = pd.DataFrame(epochs, columns=["animal", "epoch"]).assign(
        session_date="d1"
    )
    for band, column, _ in figure.SIGNALS:
        events[column] = rng.normal(size=len(events))
        for regime in ("rest", "running"):
            for group in ("non_swr", "swr"):
                reference[f"{band}_{regime}_matched_{group}"] = rng.normal(size=3)

    fig, axes = plt.subplots(1, 2)
    figure.draw_gamma(axes, events, reference)
    assert [ax.get_title() for ax in axes] == [
        "Slow gamma (25–55 Hz)",
        "Fast gamma (65–100 Hz)",
    ]
    for ax, (band, column, _) in zip(axes, figure.SIGNALS, strict=True):
        levels = animal_signal_levels(events, reference, band, column, matched=True)
        # per kind, three sets of dots: rest, events, running
        dots = ax.collections
        assert len(dots) == 6
        for k, (is_swr, _, color) in enumerate(figure.KINDS):
            rest, event, running = dots[3 * k : 3 * k + 3]
            for drawn, word in ((rest, "rest"), (event, "event"), (running, "running")):
                np.testing.assert_allclose(
                    drawn.get_offsets()[:, 1], levels.loc[is_swr, word]
                )
            np.testing.assert_allclose(event.get_facecolors()[0], to_rgba(color))
            np.testing.assert_allclose(rest.get_facecolors()[0], to_rgba("white"))
            np.testing.assert_allclose(running.get_facecolors()[0], to_rgba("black"))
        assert [t.get_text() for t in ax.get_xticklabels()] == ["No SWR", "SWR"]
    plt.close(fig)


def test_figure03_letters_share_a_height_by_row_and_a_left_edge_by_column(offline):
    """The letters of a row sit at one height, above the row's highest drawn
    top; the letters of a column sit at one left edge, the leftmost drawn extent
    of the column's panels, whatever each panel's own labels."""
    import matplotlib.pyplot as plt

    from continuum_swr_replay.visualization.simulated_run_panels import drawn_extent

    figure03 = offline("scripts.figures.generate_figure03")
    rows, columns = figure03.LETTER_ROWS, figure03.LETTER_COLUMNS
    fig, grid = plt.subplots(
        len(rows), len(columns), figsize=(figure03.WIDTH, figure03.HEIGHT)
    )
    panels = {
        letter: [grid[r, c]]
        for r, row in enumerate(rows)
        for c, column in enumerate(columns)
        for letter in row
        if letter in column
    }
    assert sorted(panels) == sorted("".join(rows))
    shared = next(column for column in columns if len(column) > 1 and "a" not in column)
    upper, lower = shared
    panels[upper][0].set_ylabel("a label\nof two lines")  # wider than the lower's
    panels[rows[1][0]][0].set_title("a title")  # higher than its row's other panels
    fig.draw_without_rendering()
    positions = figure03.letter_positions(fig, panels)

    for row in rows:
        tops = {positions[letter][1] for letter in row}
        assert len(tops) == 1
        assert tops.pop() > max(drawn_extent(fig, panels[letter]).y1 for letter in row)
    for column in columns:
        lefts = {positions[letter][0] for letter in column}
        assert len(lefts) == 1
        assert lefts.pop() < min(
            drawn_extent(fig, panels[letter]).x0 for letter in column
        )
    assert positions[lower][0] < drawn_extent(fig, panels[lower]).x0 - 0.01
    plt.close(fig)


def test_figure02_letters_share_a_row_height_from_the_named_parts(offline):
    """A row's letters sit just above its panels' highest drawn top; a panel
    named with no axes does not count (its title would lift the row), and a
    letter given a height sits there; columns share the leftmost edge."""
    import matplotlib.pyplot as plt

    figure02 = offline("scripts.figures.generate_figure02")
    fig, grid = plt.subplots(2, 3)
    panels = {
        "a": [grid[0, 0]],
        "b": [grid[0, 1]],
        "c": [grid[0, 2]],
        "d": [grid[1, 0]],
        "e": [grid[1, 1]],
        "f": [grid[1, 2]],
    }
    grid[0, 2].set_title("a title\nof two lines")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extent = {
        name: figure02.panel_extent(fig, renderer, axes)
        for name, axes in panels.items()
    }
    rows, columns = ["abc", "def"], ["ad", "be", "c", "f"]
    plain = figure02.place_letters(fig, renderer, panels, rows, columns)
    named = figure02.place_letters(
        fig, renderer, panels, rows, columns, heights={"c": [], "d": 0.3}
    )
    # by default the title of c sets its row's height
    assert {plain[n][1] for n in "abc"} == {extent["c"].y1 + 0.002}
    # named: c does not count, so a and b set it; d sits where it is given
    top = max(extent[n].y1 for n in "ab") + 0.002
    assert top < extent["c"].y1
    assert {named[n][1] for n in "abc"} == {top}
    assert named["d"][1] == 0.3
    assert {named[n][1] for n in "ef"} == {max(extent[n].y1 for n in "ef") + 0.002}
    assert named["a"][0] == named["d"][0] == min(extent[n].x0 for n in "ad") - 0.004
    plt.close(fig)


def _gain_statistics(no_swr: list[float], swr: list[float]) -> pd.DataFrame:
    """Statistics indexed by group, as ``likelihood_advantage_statistics``
    gives them, with each group's mean as its estimate and an interval of
    +/- 0.1."""
    rows = {}
    for group, animals in (("no SWR", no_swr), ("SWR", swr)):
        estimate = float(np.mean(animals))
        rows[group] = {
            "animals": np.asarray(animals),
            "animal_names": np.array(list("abcde"[: len(animals)])),
            "estimate": estimate,
            "ci_low": estimate - 0.1,
            "ci_high": estimate + 0.1,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")


def _measure_rows(measures: dict[str, list[float]]) -> pd.DataFrame:
    """Statistics rows indexed by family and measure: the filter family's
    per-animal values of each measure, their mean as the estimate and an
    interval of +/- a tenth of the largest value (the two-sided family's
    twice them)."""
    rows = {}
    for family, scale in (("filter", 1.0), ("two_sided", 2.0)):
        for measure, animals in measures.items():
            values = scale * np.asarray(animals)
            half = 0.1 * float(np.max(np.abs(values)))
            rows[(family, measure)] = {
                "animals": values,
                "animal_names": np.array(list("abcde"[: len(values)])),
                "estimate": float(values.mean()),
                "ci_low": float(values.mean()) - half,
                "ci_high": float(values.mean()) + half,
            }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["family", "measure"]
    return table


SWR_ONLY = [0.099, 0.103, 0.081, 0.138, 0.093]
FULL = [0.119, 0.117, 0.107, 0.172, 0.115]
PAIRED = [0.022, 0.023, 0.03, 0.042, 0.025]
OUTSIDE = [0.029, 0.02, 0.025, 0.038, 0.022]
INSIDE = [-0.02, -0.015, -0.025, -0.03, -0.018]
IN_EVENTS = [0.38, 0.39, 0.23, 0.42, 0.40]
OUTSIDE_EVENTS = [-0.003, -0.004, -0.005, -0.002, -0.004]


def _decoder_rows() -> pd.DataFrame:
    """``immobile_gain_statistics``' rows the panel reads."""
    return _measure_rows(
        {
            "swr_minus_none_per_spike": SWR_ONLY,
            "full_minus_none_per_spike": FULL,
            "full_minus_swr_per_spike": PAIRED,
        }
    )


def _outside_rows() -> pd.DataFrame:
    """``outside_swr_gain_statistics``' rows the panels read."""
    return _measure_rows(
        {"gain_per_spike": OUTSIDE, "outside_events_gain_per_spike": OUTSIDE_EVENTS}
    )


def _restriction_statistics():
    """``restriction_totals.restriction_figure_statistics``' tables panel j
    reads: the events' row indexed by group, as the window statistics are."""
    from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
        RestrictionFigureStatistics,
    )

    events = _measure_rows({"no SWR": IN_EVENTS}).loc["filter"]
    events.index.name = "group"
    return RestrictionFigureStatistics(
        decoders=_decoder_rows(),
        inside=_measure_rows({"gain_per_spike": INSIDE}),
        outside=_outside_rows(),
        events=events,
    )


def _dots_at(ax, x: float):
    """The one scatter of ``ax`` whose dots are spread about ``x``."""
    (dots,) = [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.PathCollection)
        and np.allclose(c.get_offsets()[:, 0], x, atol=0.2)
    ]
    return dots


def assert_spread_about(dots, x: float) -> None:
    """The dots sit symmetrically about ``x``, alternating sides by value."""
    xs, ys = dots.get_offsets()[:, 0], dots.get_offsets()[:, 1]
    assert np.mean(xs) == pytest.approx(x)
    by_value = xs[np.argsort(ys)] - x
    # neighbors in value never on the same side, the middle one on x
    assert all(a * b <= 0 for a, b in zip(by_value, by_value[1:], strict=False))
    assert by_value[len(by_value) // 2] == pytest.approx(0.0)


def _means_at(ax, x: float) -> list:
    """The y of the lines (the means' markers) beside the dots at ``x``."""
    return [
        line.get_ydata() for line in ax.lines if np.allclose(line.get_xdata(), x + 0.25)
    ]


def _has_zero_line(ax) -> bool:
    return any(
        line.get_linestyle() == "--" and np.allclose(line.get_ydata(), 0)
        for line in ax.lines
    )


def test_figure02_decoder_comparison_first_axis_draws_each_decoders_gain(offline):
    """Panel j's first axis: each decoder's gain over the local-only decoder,
    the SWR-only decoder's dots dark gray and the full decoder's green, spread
    about the column, the mean with its interval beside them, and the dashed
    zero line labeled "local-only"."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    figure02 = offline("scripts.figures.generate_figure02")
    fig, ax = plt.subplots()
    figure02.draw_decoder_gains(ax, _decoder_rows())
    fig.canvas.draw()
    for x, values, color in (
        (0.0, SWR_ONLY, heldout_gain_panels.NEUTRAL_DOTS),
        (1.0, FULL, figure02.COLORS["Non-Local"]),
    ):
        dots = _dots_at(ax, x)
        np.testing.assert_allclose(sorted(dots.get_offsets()[:, 1]), sorted(values))
        assert_spread_about(dots, x)
        np.testing.assert_allclose(dots.get_facecolors()[0], to_rgba(color))
        assert np.allclose(_means_at(ax, x), np.mean(values))
    assert _has_zero_line(ax)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["SWR-\nonly", "Full"]
    assert [t.get_text() for t in ax.texts] == ["local-only"]
    assert ax.get_ylabel() == "Held-out log score\nover local-only\n(nats/spike)"
    plt.close(fig)


def test_figure02_decoder_comparison_second_axis_draws_full_minus_swr_only(offline):
    """Panel j's second axis: the full minus the SWR-only decoder over all
    immobility (the paired difference), inside SWRs and outside SWRs (the main
    result, its label in bold), dark gray dots spread about each column, the
    mean with its interval beside them, and the dashed zero line with "full
    better"; the axis reaches below the values under zero."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    figure02 = offline("scripts.figures.generate_figure02")
    fig, ax = plt.subplots()
    figure02.draw_full_minus_swr(ax, _restriction_statistics())
    fig.canvas.draw()
    for x, values in ((0.0, PAIRED), (1.0, INSIDE), (2.0, OUTSIDE)):
        dots = _dots_at(ax, x)
        np.testing.assert_allclose(sorted(dots.get_offsets()[:, 1]), sorted(values))
        assert_spread_about(dots, x)
        np.testing.assert_allclose(
            dots.get_facecolors()[0], to_rgba(heldout_gain_panels.NEUTRAL_DOTS)
        )
        assert np.allclose(_means_at(ax, x), np.mean(values))
    assert _has_zero_line(ax)
    assert ax.get_ylim()[0] < min(INSIDE)
    labels = ax.get_xticklabels()
    assert [t.get_text() for t in labels] == [
        "All\nimmobility",
        "Inside\nSWRs",
        "Outside\nSWRs",
    ]
    assert [t.get_fontweight() for t in labels] == ["normal", "normal", "bold"]
    # horizontal, so that they do not hang below panel h's labels
    assert {t.get_rotation() for t in labels} == {0.0}
    assert ax.get_ylabel() == "Held-out log score,\nfull − SWR-only\n(nats/spike)"
    assert [t.get_text() for t in ax.texts] == ["↑ full better"]
    plt.close(fig)


def test_figure02_decoder_comparison_third_axis_draws_the_heldout_events(offline):
    """Panel j's third axis: the full minus the SWR-only decoder outside SWRs
    in the no-SWR held-out events and outside every held-out event, on its own
    scale, with the zero line."""
    import matplotlib.pyplot as plt

    figure02 = offline("scripts.figures.generate_figure02")
    fig, ax = plt.subplots()
    figure02.draw_heldout_event_gain(ax, _restriction_statistics())
    fig.canvas.draw()
    for x, values in ((0.0, IN_EVENTS), (1.0, OUTSIDE_EVENTS)):
        dots = _dots_at(ax, x)
        np.testing.assert_allclose(sorted(dots.get_offsets()[:, 1]), sorted(values))
        assert np.allclose(_means_at(ax, x), np.mean(values))
    assert _has_zero_line(ax)
    assert [t.get_text() for t in ax.get_xticklabels()] == list(
        figure02.EVENTS_LABELS.values()
    )
    assert ax.get_title() == figure02.EVENTS_TITLE
    plt.close(fig)


def test_figure02_decoder_comparison_draws_the_values_the_events_summary_tags(offline):
    """Panel j draws the restriction rows the events summary tags with it, by
    the summary's own definitions; shared content is panel h; the held-out
    tetrode values are the supplementary figure's."""
    from continuum_swr_replay.analysis.event_evidence import restriction_totals
    from continuum_swr_replay.reported import events

    figure02 = offline("scripts.figures.generate_figure02")
    # one definition of the drawn statistic and its choices, in the analysis
    # layer, used by both
    assert (
        figure02.restriction_figure_statistics
        is restriction_totals.restriction_figure_statistics
    )
    for name in (
        "RESTRICTION_FIGURE_DECODERS",
        "RESTRICTION_FIGURE_DIFFERENCE",
        "RESTRICTION_FIGURE_EVENT_GROUP",
        "RESTRICTION_FIGURE_FAMILY",
        "RESTRICTION_FIGURE_INSIDE_SWR",
        "RESTRICTION_FIGURE_OUTSIDE_EVENTS",
        "RESTRICTION_FIGURE_OUTSIDE_SWR",
    ):
        assert getattr(figure02, name) is getattr(restriction_totals, name), name
    for name in (
        "OUTSIDE_SWR_EVENT_WINDOWS",
        "RESTRICTION_FIGURE_DECODERS",
        "RESTRICTION_FIGURE_DIFFERENCE",
        "RESTRICTION_FIGURE_FAMILY",
        "OUTSIDE_SWR_EVENT_WINDOW_GROUPS",
        "RESTRICTION_FIGURE_OUTSIDE_SWR",
    ):
        assert getattr(events, name) is getattr(restriction_totals, name), name
    assert events.TAGS["nonlocal_restriction_figure"].figures == ("2j",)
    # Figure 2's supplement, panel by panel
    # (each panel's tag: test_generate_supp_figure02)
    assert events.TAGS["shared_content"].figures == ("2h",)
    assert events.TAGS["participation"].figures == ("S2h",)
    # panel e's detectors are the analysis layer's, whose found shares the
    # summary tags with the panel
    from continuum_swr_replay.analysis.detector_comparison import nonlocal_comparison

    assert list(figure02.DETECTOR_LABELS) == list(nonlocal_comparison.FIGURE_DETECTORS)
    assert events.TAGS["nonlocal_found"].figures == ("2e",)


@pytest.mark.parametrize(
    "module",
    ["scripts.figures.generate_figure03", "scripts.figures.generate_supp_figure03"],
)
def test_figure03_and_the_gamma_figure_read_the_events_that_start_in_a_visit(
    offline, monkeypatch, module
):
    """An event between two visits is in the table and in no panel: the first
    function given the events sees only those that start inside a visit."""
    figure = offline(module)
    outside = _figure02_events().iloc[[3]].assign(visit_context="after_visit_at_well")
    table = pd.concat([_figure02_events(), outside], ignore_index=True)
    seen = []

    class Read(Exception):
        pass

    def stop(events, *args):
        seen.append(events)
        raise Read

    monkeypatch.setattr(figure, "read_swr_table", lambda path: table)
    reader = (
        "content_along_arms_bins"
        if module.endswith("generate_figure03")
        else "with_brain_state"
    )
    monkeypatch.setattr(figure, reader, stop)
    with pytest.raises(Read):
        figure.read_inputs()
    assert len(seen[0]) == len(table) - 1
    assert set(seen[0]["visit_context"]) == {"in_visit"}


@pytest.mark.parametrize("problem", ["subset", "session_cache"])
@pytest.mark.parametrize("folder", ["brain_state_reference", "event_gamma"])
def test_the_gamma_figure_refuses_a_reference_not_made_over_the_subset(
    offline, monkeypatch, tmp_path, folder, problem
):
    """Quiet rest is the rest inside a well visit: a brain-state or gamma
    reference whose manifest records another subset, or none, or another
    session cache's visits than the current one, is refused."""
    import dataclasses

    from continuum_swr_replay.analysis.event_definition.cohort_selection import (
        ANALYZED_SUBSET,
    )
    from continuum_swr_replay.analysis.event_tables import prepare_data

    figure = offline("scripts.figures.generate_supp_figure03")
    monkeypatch.setattr(prepare_data, "SESSION_CACHE_DIR", tmp_path / "cache")
    entry = prepare_data.behavioral_context_cache_path({"session_id": "e"})
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"the visits")
    current = prepare_data.session_cache_record(["e"])
    inputs = dataclasses.replace(
        figure.default_inputs(),
        brain_state_reference=tmp_path / "brain_state_reference" / "reference.parquet",
        event_gamma=tmp_path / "event_gamma",
    )
    for name in ("brain_state_reference", "event_gamma"):
        (tmp_path / name).mkdir()
        manifest = {"subset": ANALYZED_SUBSET, "session_cache": current}
        if name == folder:
            manifest.pop(problem)
        (tmp_path / name / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(figure, "default_inputs", lambda: inputs)
    monkeypatch.setattr(figure, "read_swr_table", lambda path: _figure02_events())
    levels = pd.DataFrame({"epoch_id": ["e"]})
    monkeypatch.setattr(
        figure, "with_brain_state", lambda events, inputs, n: (events, levels, levels)
    )
    note = "was not made over" if problem == "subset" else "records no session-cache"
    with pytest.raises(ValueError, match=f"{folder}.* {note}"):
        figure.read_inputs()
    # a cache refreshed since the reference was made is refused too
    if problem == "session_cache":
        (tmp_path / folder / "manifest.json").write_text(
            json.dumps({"subset": ANALYZED_SUBSET, "session_cache": current})
        )
        assert figure.read_inputs()["reference"] is levels
        entry.write_bytes(b"other visits")
        with pytest.raises(ValueError, match="other well visits than the current"):
            figure.read_inputs()


# --------------------------------------------------------------------------
# Figure 2: the stored per-event tables are read over the selected events
# --------------------------------------------------------------------------


class _Figure02Read(Exception):
    """Stops the figure at the read under test."""


def _figure_with_one_event_outside(offline, tmp_path, monkeypatch):
    """The figure module with its eight events in a visit and a ninth on the
    track, every reader before the held-out tetrode run's stubbed, a
    held-out tetrode run held out in folds of whole trials (panel h), and a
    per-bin facts run made from the current session cache."""
    import joblib

    from continuum_swr_replay.analysis.event_tables import prepare_data
    from continuum_swr_replay.paths import (
        BRAIN_STATE_REFERENCE_DIR,
        EVENT_BIN_FACTS_DIR,
        EVENT_TETRODE_HOLDOUT_DIR,
        NONLOCAL_BY_SIGNAL_DIR,
        processed_data_dir,
        replay_content_dir,
        table_path,
    )
    from continuum_swr_replay.reported import events

    figure02 = offline("scripts.figures.generate_figure02")
    inside = _figure02_events().assign(event_id=range(8))
    outside = inside.iloc[[3]].assign(event_id=99, visit_context="track_leaf_arm")
    table = table_path(
        "event_table", tmp_path / replay_content_dir().relative_to(processed_data_dir())
    )
    table.parent.mkdir(parents=True)
    write_table(analyzed_time.stamp_swr_source(pd.concat([inside, outside])), table)
    cache = figure02._figure02_cache_path("wilbur20210329_.nwb", "e1", tmp_path)
    cache.parent.mkdir(parents=True)
    data = {"ripple_times": pd.DataFrame(columns=["start_time", "end_time"])}
    frames = {**FIGURE02_FRAME_COUNTS, "non_swr": 6, "total": 9}
    joblib.dump({"frame_event_counts": frames, "data": data, "results": None}, cache)
    for folder, name in (
        (BRAIN_STATE_REFERENCE_DIR, "reference.parquet"),
        (NONLOCAL_BY_SIGNAL_DIR, "counts.parquet"),
    ):
        (tmp_path / folder).mkdir()
        pd.DataFrame({"epoch_id": ["e"]}).to_parquet(tmp_path / folder / name)
    monkeypatch.setattr(prepare_data, "SESSION_CACHE_DIR", tmp_path / "cache")
    entry = prepare_data.behavioral_context_cache_path({"session_id": "e"})
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"the visits")
    (tmp_path / EVENT_TETRODE_HOLDOUT_DIR).mkdir(parents=True)
    (tmp_path / EVENT_TETRODE_HOLDOUT_DIR / "manifest.json").write_text(
        json.dumps({"settings": {"n_folds": 5, "fold_design": "trials"}})
    )
    (tmp_path / EVENT_BIN_FACTS_DIR).mkdir()
    (tmp_path / EVENT_BIN_FACTS_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": {"session_cache": prepare_data.session_cache_record(["e"])},
                "epochs": {"e": {"status": "ok"}},
            }
        )
    )
    # each stored per-event table holds a row of every event, the ninth too;
    # the held-out event run is read by the events summary's reader
    rows = pd.DataFrame({"event_id": [*range(8), 99]})
    monkeypatch.setattr(
        events, "read_complete_run", lambda *args: rows.assign(starts_in_well_visit=1)
    )
    for name, value in {
        "parse_nwb_filename": lambda name: ("a", "d1"),
        "example_windows": lambda *args: {},
        "checked_reference": lambda reference, folder, name: reference,
        "reference_medians": lambda *args: None,
        "animal_nonlocal_counts": lambda *args: None,
        "event_table_epochs": lambda *args: ["e"],
        "read_complete_run": lambda *args: rows,
        "window_scores": lambda table: table.assign(kind="event"),
        "agreement_by_spread_statistics": lambda windows: None,
        "well_visit_times": lambda identifier: None,
        "check_visit_context_starts": lambda table, visits: None,
        "restriction_inputs_note": lambda folder, epochs: None,
    }.items():
        monkeypatch.setattr(figure02, name, value)
    return figure02


def test_figure02_reads_the_held_out_tetrode_run_over_the_selected_events(
    offline, tmp_path, monkeypatch
):
    """Panel h's shared content is over the events that start inside a visit:
    the held-out tetrode run's rows of the ninth event (on the track) are not
    read."""
    figure02 = _figure_with_one_event_outside(offline, tmp_path, monkeypatch)
    seen = []

    def stop(windows):
        seen.append(windows)
        raise _Figure02Read

    monkeypatch.setattr(figure02, "agreement_by_spread_statistics", stop)
    with pytest.raises(_Figure02Read):
        figure02.generate_figure(epoch_name="e1", data_dir=tmp_path)
    assert sorted(seen[0]["event_id"]) == list(range(8))


def test_figure02_refuses_a_held_out_tetrode_run_not_on_trial_folds(
    offline, tmp_path, monkeypatch
):
    """Panel h, as the events summary: a held-out tetrode run whose folds are
    not whole trials is refused before it is read."""
    from continuum_swr_replay.paths import EVENT_TETRODE_HOLDOUT_DIR

    figure02 = _figure_with_one_event_outside(offline, tmp_path, monkeypatch)
    (tmp_path / EVENT_TETRODE_HOLDOUT_DIR / "manifest.json").write_text(
        json.dumps({"settings": {"n_folds": 5, "fold_design": "blocks"}})
    )
    monkeypatch.setattr(figure02, "agreement_by_spread_statistics", _refuse_drawing)
    with pytest.raises(ValueError, match="whole trials"):
        figure02.generate_figure(epoch_name="e1", data_dir=tmp_path)


def _refuse_drawing(windows):
    raise AssertionError("the held-out tetrode run was read without its check")


def _heldout_events_run(
    tmp_path, facts_sha256=None, restriction_sha256=None, restriction_settings=None
) -> None:
    """A restriction run's manifest (the fold fit on whole trials, or
    ``restriction_settings``) and a held-out event run's manifest that records
    the current per-bin facts and restriction runs' digests (or
    ``facts_sha256``, ``restriction_sha256``)."""
    from continuum_swr_replay.paths import (
        EVENT_BIN_FACTS_DIR,
        EVENT_NONLOCAL_RESTRICTION_DIR,
        EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
    )
    from continuum_swr_replay.runs.io import sha256_file

    restriction = tmp_path / EVENT_NONLOCAL_RESTRICTION_DIR
    restriction.mkdir()
    settings = restriction_settings or {"fit": "folds", "fold_design": "trials"}
    (restriction / "manifest.json").write_text(json.dumps({"settings": settings}))
    folder = tmp_path / EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR
    folder.mkdir()
    facts = facts_sha256 or sha256_file(
        tmp_path / EVENT_BIN_FACTS_DIR / "manifest.json"
    )
    restriction_digest = restriction_sha256 or sha256_file(
        restriction / "manifest.json"
    )
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "bin_facts": {"sha256": facts},
                    "nonlocal_restriction": {"sha256": restriction_digest},
                },
                "epochs": {"e": {"status": "ok", "bin_s": 0.002}},
            }
        )
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"restriction_sha256": "0" * 64},
            "made from another nonlocal_restriction run than the current one",
        ),
        ({"restriction_settings": {"n_folds": 5}}, "is not the fold fit"),
        (
            {"restriction_settings": {"fit": "folds", "fold_design": "contiguous"}},
            "not 'trials'",
        ),
    ],
    ids=["another restriction run", "not the fold fit", "contiguous blocks"],
)
def test_figure02_refuses_a_heldout_event_run_not_from_the_current_fold_fit(
    offline, tmp_path, monkeypatch, change, message
):
    """Panel j draws only the fold fit on whole trials, from a held-out event
    run made from the restriction run as it is now."""
    figure02 = _figure_with_one_event_outside(offline, tmp_path, monkeypatch)
    _heldout_events_run(tmp_path, **change)

    def refuse(*args):
        raise AssertionError("panel j was drawn from another restriction run")

    monkeypatch.setattr(figure02, "restriction_figure_statistics", refuse)
    with pytest.raises(ValueError, match=message) as refused:
        figure02.generate_figure(epoch_name="e1", data_dir=tmp_path)
    assert "detect_heldout_events.py" in str(refused.value)


def test_figure02_refuses_a_restriction_run_off_the_current_decodes_or_swr_source(
    offline, tmp_path, monkeypatch
):
    """Panel j is not drawn from a restriction run made from other decode
    frames or another SWR source than the current ones: the summary's reason,
    and the runner to make again."""
    figure02 = _figure_with_one_event_outside(offline, tmp_path, monkeypatch)
    _heldout_events_run(tmp_path)
    seen = []

    def note(folder, epochs):
        seen.append((folder.name, list(epochs)))
        return "the restriction run records another SWR source; rerun x before citing"

    def refuse(*args):
        raise AssertionError("panel j was drawn from other decodes")

    monkeypatch.setattr(figure02, "restriction_inputs_note", note)
    monkeypatch.setattr(figure02, "restriction_figure_statistics", refuse)
    with pytest.raises(ValueError) as refused:
        figure02.generate_figure(epoch_name="e1", data_dir=tmp_path)
    assert str(refused.value) == (
        "the decoder comparison (panel j) is not drawn: the restriction run "
        "records another SWR source. "
        "Rerun scripts/event_evidence/score_nonlocal_restriction.py and the runs "
        "made from it."
    )
    assert seen == [("event_nonlocal_restriction", ["e"])]


def test_figure02_reads_the_heldout_events_that_start_inside_a_visit(
    offline, tmp_path, monkeypatch
):
    """Panel j reads the held-out event run: its events that start inside a
    well visit (not the two planted outside one) and its time sums whole."""
    from continuum_swr_replay.reported import events

    figure02 = _figure_with_one_event_outside(offline, tmp_path, monkeypatch)
    _heldout_events_run(tmp_path)
    heldout = pd.DataFrame(
        {"heldout_event_id": range(5), "starts_in_well_visit": [1, 0, 1, 1, 0]}
    ).astype({"starts_in_well_visit": bool})
    read = {"heldout_events": heldout, "heldout_time_sums": pd.DataFrame(range(9))}
    monkeypatch.setattr(
        events, "read_complete_run", lambda folder, name, epochs: read[name]
    )
    seen = []

    def stop(windows, time_sums, bin_s):
        seen.append((windows, time_sums, bin_s))
        raise _Figure02Read

    monkeypatch.setattr(figure02, "restriction_figure_statistics", stop)
    with pytest.raises(_Figure02Read):
        figure02.generate_figure(epoch_name="e1", data_dir=tmp_path)
    windows, time_sums, bin_s = seen[0]
    assert windows["heldout_event_id"].tolist() == [0, 2, 3]
    assert len(time_sums) == 9  # the time sums are read whole
    assert bin_s.to_dict() == {"e": 0.002}


@pytest.mark.parametrize("problem", ["no session cache", "another session cache"])
def test_figure02_refuses_bin_facts_made_from_other_visits(
    offline, tmp_path, monkeypatch, problem
):
    """Panel j's time inside a well visit is the per-bin facts' flag: a facts
    run that records no session cache (made before it did) or another than the
    current one is refused, naming the runs to make again."""
    from continuum_swr_replay.paths import EVENT_BIN_FACTS_DIR

    figure02 = _figure_with_one_event_outside(offline, tmp_path, monkeypatch)
    # the facts' own record changes; the cache, which participation read too,
    # does not; the held-out event run read these facts
    path = tmp_path / EVENT_BIN_FACTS_DIR / "manifest.json"
    manifest = json.loads(path.read_text())
    if problem == "no session cache":
        manifest["inputs"].pop("session_cache")
    else:
        manifest["inputs"]["session_cache"] = {"sha256": "0" * 64}
    path.write_text(json.dumps(manifest))
    _heldout_events_run(tmp_path)
    note = {
        "no session cache": "records no session-cache SHA-256",
        "another session cache": "other well visits than the current",
    }[problem]

    def refuse(*args):
        raise AssertionError("panel j was drawn from the facts without their check")

    monkeypatch.setattr(figure02, "restriction_figure_statistics", refuse)
    with pytest.raises(ValueError, match=f"{EVENT_BIN_FACTS_DIR}.*{note}") as refused:
        figure02.generate_figure(epoch_name="e1", data_dir=tmp_path)
    message = str(refused.value)
    # what is not current, then the runners to make again, in that order
    assert "before citing" not in message
    assert message.endswith(
        ". Rerun scripts/event_evidence/measure_bin_facts.py, then "
        "scripts/event_evidence/detect_heldout_events.py."
    )


def test_figure02_refuses_a_heldout_event_run_not_made_from_the_current_bin_facts(
    offline, tmp_path, monkeypatch
):
    """Panel j draws the held-out event run, whose events' start and time sums
    take their well visits from the per-bin facts: a run that recorded another
    facts run than the current one is refused, naming the run to make again."""
    figure02 = _figure_with_one_event_outside(offline, tmp_path, monkeypatch)
    _heldout_events_run(tmp_path, facts_sha256="0" * 64)

    def refuse(*args):
        raise AssertionError("panel j was drawn from another facts run's events")

    monkeypatch.setattr(figure02, "restriction_figure_statistics", refuse)
    with pytest.raises(
        ValueError, match="made from another bin_facts run than the current one"
    ) as refused:
        figure02.generate_figure(epoch_name="e1", data_dir=tmp_path)
    assert "detect_heldout_events.py" in str(refused.value)


def test_figure02_checks_every_event_s_label_against_the_visits(
    offline, tmp_path, monkeypatch
):
    """The ninth event (on the track, starting at 3 s) is labeled outside every
    visit; visits that hold it make the figure refuse the table."""
    import numpy as np

    from continuum_swr_replay.analysis.event_definition import cohort_selection

    figure02 = _figure_with_one_event_outside(offline, tmp_path, monkeypatch)
    monkeypatch.setattr(
        figure02,
        "check_visit_context_starts",
        cohort_selection.check_visit_context_starts,
    )
    monkeypatch.setattr(
        figure02, "event_table_epochs", lambda table, analyzed: ["ad1_e1_HPC"]
    )
    visits = (np.array([-1.0]), np.array([10.0]))
    monkeypatch.setattr(figure02, "well_visit_times", lambda identifier: visits)
    monkeypatch.setattr(
        cohort_selection, "event_epoch_ids", lambda t: pd.Series("ad1_e1_HPC", t.index)
    )
    with pytest.raises(ValueError, match="1 of 9 events' visit_context disagrees"):
        figure02.generate_figure(epoch_name="e1", data_dir=tmp_path)


def test_figure02_cache_is_keyed_on_the_event_table_it_reads(offline, tmp_path):
    """The figure reads the event table of its data folder; a change to that
    table, not to the default root's, gives another cache path."""
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
    write_table(_figure02_events(), table)
    before = figure02._figure02_cache_path("wilbur20210329_.nwb", "e1", tmp_path)
    write_table(_figure02_events().iloc[:3], table)
    after = figure02._figure02_cache_path("wilbur20210329_.nwb", "e1", tmp_path)
    assert before != after


# --------------------------------------------------------------------------
# Figure 2h: shared content (was the supplement's panel h)
# --------------------------------------------------------------------------

SHARED = {
    ("no SWR", 0): [0.3, 0.25, 0.35, 0.2, 0.4],
    ("SWR", 0): [0.2, 0.25, 0.3, 0.15, 0.22],
}
"""Panel h's animals in the narrowest spread bin; each wider bin is 0.1 lower."""


SHARE_OF_EVENTS = {"no SWR": [0.12, 0.2, 0.43, 0.25], "SWR": [0.1, 0.3, 0.4, 0.2]}
"""Each group's share of its events in each spread bin."""


def _shared_statistics(figure) -> pd.DataFrame:
    """``agreement_by_spread_statistics``' rows: each bin and group's animals,
    their mean and an interval of 0.05 either side."""
    rows = []
    for k, label in enumerate(figure.SPREAD_BIN_LABELS):
        for group in ("no SWR", "SWR"):
            animals = np.asarray(SHARED[(group, 0)]) - 0.1 * k
            mean = float(animals.mean())
            rows.append(
                {
                    "spread_bin": label, "group": group, "animals": animals,
                    "animal_names": np.array(list("abcde")), "estimate": mean,
                    "ci_low": mean - 0.05, "ci_high": mean + 0.05,
                    "share_of_events": SHARE_OF_EVENTS[group][k],
                }
            )  # fmt: skip
    return pd.DataFrame(rows).set_index(["spread_bin", "group"])


def test_figure02_shared_content_draws_the_agreement_by_spread(offline):
    """Panel h: each group's dots per spread bin on its side of the bin, its means with
    their intervals farther out joined by a thin line in the group's color, the
    bins' labels, a dashed zero line and which way means shared content."""
    import matplotlib.pyplot as plt

    figure = offline("scripts.figures.generate_figure02")
    fig, (ax_share, ax) = plt.subplots(2, 1)
    figure.draw_shared_content(ax_share, ax, _shared_statistics(figure))
    assert [t.get_text() for t in ax.get_xticklabels()] == list(
        figure.SPREAD_BIN_LABELS
    )
    bins = np.arange(len(figure.SPREAD_BIN_LABELS))
    for side, group in ((-1, "no SWR"), (1, "SWR")):
        expected = [np.mean(SHARED[(group, 0)]) - 0.1 * k for k in bins]
        means = bins + side * (figure.SHARED_OFFSET + figure.SHARED_MEAN_OFFSET)
        (series,) = [
            line for line in ax.lines
            if len(line.get_xdata()) == bins.size
            and np.allclose(line.get_xdata(), means)
            and line.get_linestyle() != "None"  # not the means' markers
        ]  # fmt: skip
        np.testing.assert_allclose(series.get_ydata(), expected, atol=1e-12)
        for k in bins:
            (dots,) = [
                c for c in ax.collections
                if isinstance(c, matplotlib.collections.PathCollection)
                and np.allclose(
                    c.get_offsets()[:, 0], k + side * figure.SHARED_OFFSET, atol=0.1
                )
            ]  # fmt: skip
            np.testing.assert_allclose(
                sorted(dots.get_offsets()[:, 1]),
                sorted(np.asarray(SHARED[(group, 0)]) - 0.1 * k),
            )
    assert any(
        line.get_linestyle() == "--" and np.allclose(line.get_ydata(), 0)
        for line in ax.lines
    )
    # a strip above, sharing the x axis: each group's percent of its events per
    # bin as two narrow bars side by side, in the groups' colors
    from matplotlib.colors import to_rgba

    from continuum_swr_replay.visualization import heldout_gain_panels as panels

    bars = list(ax_share.patches)
    assert len(bars) == 2 * len(bins)
    for side, (group, color) in zip((-1, 1), panels.GROUP_COLORS.items(), strict=True):
        mine = [b for b in bars if to_rgba(b.get_facecolor()) == to_rgba(color)]
        np.testing.assert_allclose(
            [b.get_x() + b.get_width() / 2 for b in mine],
            bins + side * figure.SHARE_BAR_WIDTH / 2,
        )
        np.testing.assert_allclose(
            [b.get_height() for b in mine], 100 * np.asarray(SHARE_OF_EVENTS[group])
        )
    assert ax_share.get_shared_x_axes().joined(ax_share, ax)
    assert ax_share.get_ylabel() == "% of\nevents"
    assert ax_share.get_title() == "Shared content"
    assert not any(t.get_visible() and t.get_text() for t in ax_share.get_xticklabels())
    texts = [t.get_text() for t in ax.texts]
    assert "\u2191 shared\ncontent" in texts
    assert {"No SWR", "SWR"} <= set(texts)
    assert ax.get_ylabel() == "Agreement\n(nats/spike)"
    assert ax.get_xlabel() == "Width of the other tetrodes'\nposition estimate (cm)"
    plt.close(fig)
