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
    """The percents beside the bars are rounded by the summary's rule (a half
    rounds away from zero: 12.5 % is 13), the title gives the count, and each
    animal is one dot per bar."""
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
    cohort = {v.key: v for v in event_cohort_values(events, "s")}
    assert formatted(cohort["non_swr_fraction"]) == "63 %"
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

    from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
        RIPPLE_POWER_EDGES,
    )
    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        animal_power_counts,
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
        epochs.append(
            pd.DataFrame(
                {
                    "animal": animal,
                    "epoch_id": f"{animal}_e",
                    "power_low": low,
                    "power_high": high,
                    "n_bins": 20,
                    "n_nonlocal": n_nonlocal,
                }
            )
        )
    counts = pd.concat(epochs, ignore_index=True)
    fig, (ax_spread, ax_share) = plt.subplots(2, 1, sharex=True)
    figure02.draw_share_by_power(ax_spread, ax_share, animal_power_counts(counts))
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
    Figures 2, 3 and 4 and in the shared per-event panels: not in the Non-Local
    state's green, which every non-local event is drawn in."""
    from continuum_swr_replay.visualization import figure_panels
    from continuum_swr_replay.visualization.palettes import COLORS

    figure02 = offline("scripts.figures.generate_figure02")
    figure03 = offline("scripts.figures.generate_figure03")
    figure04 = offline("scripts.figures.generate_figure04")
    no_swr = COLORS["No SWR"]
    assert no_swr not in (COLORS["Non-Local"], COLORS["SWR"])
    assert figure02.CLASS_COLORS["non_swr"] == no_swr
    assert {name: color for _, name, color in figure03.KINDS}["No SWR"] == no_swr
    bars = {label: COLORS[color] for label, _, color in figure04.SIGNIFICANCE_BARS}
    assert bars["no-SWR"] == no_swr
    assert bars["Replay"] == COLORS["Non-Local"]  # every non-local event
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

    figure = offline("scripts.figures.generate_supp_event_gamma")
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
