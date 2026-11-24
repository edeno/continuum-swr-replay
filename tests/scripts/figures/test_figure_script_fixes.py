"""Fixes to the figure scripts found in the R7a figure run (tasks 26b and 26c).

- The event gallery selects the same events in every process.
- The pipeline data-flow figure reads its frame through the SWR source, has a
  default event that has every arm indicator, refuses one that does not with a
  clear error, and fits its panel E headers in their cells.
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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.analysis.event_definition import (  # noqa: E402
    analyzed_time,
)
from continuum_swr_replay.visualization.palettes import EVENT_KINDS  # noqa: E402
from scripts.figures import generate_supp_pipeline_data_flow as flow  # noqa: E402
from tests.support._figure02_tables import figure02_events  # noqa: E402
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


def test_figure05_draws_the_no_swr_events_in_the_palettes_color(offline):
    """Figure 5 colors its bars by name, not from ``palettes.EVENT_KINDS``: its
    no-SWR events are in the table's no-SWR color, not in the Non-Local
    state's green, which every non-local event is drawn in."""
    from continuum_swr_replay.visualization.palettes import COLORS

    figure05 = offline("scripts.figures.generate_figure05")
    bars = {label: COLORS[color] for label, _, color in figure05.SIGNIFICANCE_BARS}
    assert bars["No SWR"] == EVENT_KINDS[False][1]
    assert bars["Non-local"] == COLORS["Non-Local"]  # every non-local event


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
            "map_line_fit_speed_cm_s": np.r_[
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
    for (_, color), word, drawn in zip(
        EVENT_KINDS.values(), ("non_swr", "swr"), ticks, strict=True
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
    for is_swr, curve in zip(EVENT_KINDS, curves, strict=True):
        speed, share = curve.get_xdata(), curve.get_ydata()
        values = events.loc[events["is_swr"] == is_swr, column].to_numpy()
        assert (np.diff(share) >= 0).all()
        assert share[-1] == pytest.approx(100 * (values <= edges[-1]).mean())
        assert share[-1] < 100  # some events lie beyond the axis
        median = np.median(values)
        assert share[speed <= median][-1] <= 50 <= share[speed >= median][0]
    assert ax.get_xlim() == (edges[0], edges[-1])

    missing = events.assign(
        map_line_fit_speed_cm_s=events[column].where(events.index > 0)
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
    for ax, (is_swr, (name, _)) in zip(axes, EVENT_KINDS.items(), strict=True):
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

    from continuum_swr_replay.visualization.publication_style import overlapping_texts

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
    assert "even: 25%" in texts
    assert not [line for line in ax.lines if line.get_gid() == "animal"]
    # with each animal's profiles: a thin line per animal, kind and arm type
    animals = pd.concat(
        dict.fromkeys(("a", "b", "c"), profiles), names=["animal"]
    ).reorder_levels(["is_swr", "animal", "arm_type", "segment"])
    figure03.draw_profiles(ax, profiles, shares, even, animals)
    thin = [line for line in ax.lines if line.get_gid() == "animal"]
    assert len(thin) == 3 * 2 * 2
    np.testing.assert_allclose(thin[0].get_ydata(), 0.3)
    plt.close(fig)


def test_figure03_draws_seven_panels(offline):
    """Figure 3 is a to g, and the script's letters are the question list's.
    Gamma during the events is the supplementary figure's; how far the content
    lies from the animal is panel g (the lead, 2026-11-24)."""
    from continuum_swr_replay.reported.questions import FIGURES

    figure03 = offline("scripts.figures.generate_figure03")
    assert "".join(figure03.LETTER_ROWS) == FIGURES["3"].panels == "abcdefg"
    # g is alone in its column: every other letter shares a column's left edge
    assert sorted("".join(figure03.LETTER_COLUMNS)) == list("abcdef")
    assert not hasattr(figure03, "draw_brain_state")
    assert hasattr(figure03, "draw_content_distance")


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
        for k, (is_swr, (_, color)) in enumerate(EVENT_KINDS.items()):
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
    outside = figure02_events().iloc[[3]].assign(visit_context="after_visit_at_well")
    table = pd.concat([figure02_events(), outside], ignore_index=True)
    seen = []

    class Read(Exception):
        pass

    def stop(events, *args):
        seen.append(events)
        raise Read

    monkeypatch.setattr(figure, "read_swr_table", lambda path: table)
    reader = (
        "content_mass_along_arms"
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
    monkeypatch.setattr(figure, "read_swr_table", lambda path: figure02_events())
    levels = pd.DataFrame({"epoch_id": ["e"]})
    monkeypatch.setattr(
        figure, "with_brain_state", lambda events, inputs, n: (events, levels, levels)
    )
    # panel c's distribution is the analysis layer's, tested there
    monkeypatch.setattr(figure, "with_event_speed", lambda events, *_: (events, []))
    monkeypatch.setattr(figure, "transport_speed_distribution", lambda events: None)
    # panels d and e read the content-mass run, tested with its reader
    monkeypatch.setattr(
        figure, "content_mass_along_arms", lambda events, *_: (None, [])
    )
    monkeypatch.setattr(
        figure, "with_event_place_shares", lambda events, *_: (events, [])
    )
    monkeypatch.setattr(figure, "content_place_shares", lambda along: None)
    monkeypatch.setattr(figure, "place_share_intervals", lambda along: None)
    monkeypatch.setattr(figure, "remote_well_share_by_extent", lambda e, **_: None)
    monkeypatch.setattr(figure, "well_count_shares", lambda e, **_: None)
    # panel a's session and examples are read after the references are checked
    monkeypatch.setattr(figure, "read_example_decode", lambda *_: None)
    monkeypatch.setattr(figure, "example_windows", lambda events: None)
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


def _distance() -> pd.DataFrame:
    """``content_distance_distribution`` of the distance test's events."""
    from continuum_swr_replay.analysis.event_properties.event_statistics import (
        content_distance_distribution,
    )
    from tests.analysis.event_properties.test_event_statistics import (
        _distance_events,
    )

    return content_distance_distribution(_distance_events())


def test_the_figure03_distance_panel_draws_the_content_distance_of_each_class(
    offline,
):
    """Figure 3, panel g: for each class, the percent of events within each distance (the bins
    summed), a thin line per animal and their mean in bold, in the class's
    color, on a log x axis; dotted lines at 20 cm and 100 cm; the classes
    named in their colors; a y axis from 0 to 100."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from continuum_swr_replay.visualization import heldout_gain_panels as panels

    figure = offline("scripts.figures.generate_figure03")
    distribution = _distance()
    fig, ax = plt.subplots()
    figure.draw_content_distance(ax, distribution)
    n_bins = len(figure.DISTANCE_BIN_EDGES_CM) - 1
    curves = [line for line in ax.lines if len(line.get_xdata()) == n_bins]
    assert len(curves) == 2 * (2 + 1)  # two animals and their mean, per class
    for group, color in panels.GROUP_COLORS.items():
        mine = [c for c in curves if to_rgba(c.get_color()) == to_rgba(color)]
        mean = max(mine, key=lambda c: c.get_linewidth())
        within = (
            distribution[distribution["group"] == group]
            .pivot(index="bin_high", columns="animal", values="percent")
            .cumsum()
        )
        np.testing.assert_allclose(mean.get_xdata(), within.index.to_numpy())
        np.testing.assert_allclose(mean.get_ydata(), within.mean(axis=1).to_numpy())
        assert (np.diff(mean.get_ydata()) >= 0).all()
    assert ax.get_xscale() == "log"
    assert ax.get_ylim() == (0, 100)
    cuts = [line.get_xdata()[0] for line in ax.lines if line.get_linestyle() == ":"]
    assert cuts == [20.0, 100.0]
    assert {"No SWR", "SWR"} <= {t.get_text() for t in ax.texts}
    plt.close(fig)


def test_supp_figure03_draws_the_shares_at_a_remote_and_at_the_own_well(offline):
    """Panel d: per place, each animal's no-SWR and SWR share joined by a line,
    dots in the kinds' colors, and the animals' mean as a bar."""
    import matplotlib.pyplot as plt

    figure = offline("scripts.figures.generate_supp_figure03")
    index = pd.MultiIndex.from_product(
        [[False, True], ["a", "b", "c"]], names=["is_swr", "animal"]
    )
    shares = pd.DataFrame(
        {
            "remote_well": [0.4, 0.3, 0.5, 0.2, 0.1, 0.3],
            "own_well": [0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
        },
        index=index,
    )
    fig, ax = plt.subplots()
    figure.draw_well_shares(ax, shares)
    joined = [line for line in ax.lines if len(line.get_xdata()) == 2]
    # three animals and two mean bars, per place
    assert len(joined) == 2 * (3 + 2)
    means = sorted(
        round(float(line.get_ydata()[0]), 6)
        for line in joined
        if line.get_linewidth() == 1.5
    )
    assert means == [5.0, 10.0, 20.0, 40.0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["remote well", "own well"]
    # with the intervals: one line per place and kind, from low to high
    intervals = pd.DataFrame(
        {"ci_low": 0.1, "ci_high": 0.6},
        index=pd.MultiIndex.from_product(
            [["non_swr", "swr"], ["remote_well", "own_well"]], names=["group", "place"]
        ),
    )
    figure.draw_well_shares(ax, shares, intervals)
    drawn = [line for line in ax.lines if line.get_gid() == "interval"]
    assert len(drawn) == 4
    np.testing.assert_allclose(drawn[0].get_ydata(), [10.0, 60.0])
    plt.close(fig)


def test_supp_figure03_draws_the_remote_well_share_by_extent(offline):
    """Panel e: per kind a thin line per animal and their mean in bold, over
    the extent classes; a class an animal has no event in leaves a gap."""
    import matplotlib.pyplot as plt

    figure = offline("scripts.figures.generate_supp_figure03")
    rows = [
        (is_swr, animal, k, 0.1 * (k + 1) + 0.1 * (animal == "b"))
        for is_swr in (False, True)
        for animal in ("a", "b")
        for k in range(5)
        if not (animal == "b" and k == 4)
    ]
    by_extent = pd.DataFrame(
        rows, columns=["is_swr", "animal", "extent_class", "remote_well"]
    ).set_index(["is_swr", "animal", "extent_class"])
    fig, ax = plt.subplots()
    figure.draw_remote_well_by_extent(ax, by_extent)
    bold = [line for line in ax.lines if line.get_linewidth() == 1.2]
    assert len(bold) == 2 and len(ax.lines) == 2 * (2 + 1)
    np.testing.assert_allclose(bold[0].get_ydata(), [15, 25, 35, 45, 50])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["<10", "10–20", "20–50", "50–100", "≥100"]
    plt.close(fig)


def test_supp_figure03_draws_how_many_wells_an_event_represents(offline):
    """Panel f: per class (none, one, two or more) a bar per kind at the
    animals' mean, in the kind's color, with a dot per animal."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from continuum_swr_replay.visualization.palettes import EVENT_KINDS

    figure = offline("scripts.figures.generate_supp_figure03")
    index = pd.MultiIndex.from_product(
        [[False, True], ["a", "b"]], names=["is_swr", "animal"]
    )
    shares = pd.DataFrame(
        {
            "none": [0.4, 0.5, 0.6, 0.7],
            "one": [0.5, 0.45, 0.35, 0.28],
            "two or more": [0.1, 0.05, 0.05, 0.02],
        },
        index=index,
    )
    fig, ax = plt.subplots()
    figure.draw_well_counts(ax, shares)
    heights = [round(bar.get_height(), 6) for bar in ax.patches]
    assert heights == [45.0, 47.5, 7.5, 65.0, 31.5, 3.5]
    colors = {to_rgba(bar.get_facecolor()) for bar in ax.patches[:3]}
    assert colors == {to_rgba(EVENT_KINDS[False][1])}
    assert sum(len(c.get_offsets()) for c in ax.collections) == 2 * 3 * 2
    assert ax.get_ylim() == (0, 100)
    plt.close(fig)


def test_supp_figure03_examples_are_checked_against_the_event_table(offline):
    """Panel a's examples: three events with no SWR, then three with one, each
    titled with its kind and duration and drawn over its own start and end; an
    example missing from the table, or of the other kind, is refused."""
    figure = offline("scripts.figures.generate_supp_figure03")
    ids = [*figure.EXAMPLES[False], *figure.EXAMPLES[True]]
    events = pd.DataFrame(
        {
            "event_id": ids,
            "is_swr": [False] * 3 + [True] * 3,
            "event_start_s": np.arange(6, dtype=float),
            "event_duration_s": 0.05 + 0.01 * np.arange(6),
        }
    )
    events["event_end_s"] = events["event_start_s"] + events["event_duration_s"]
    windows = figure.example_windows(events)
    assert list(windows) == [
        "No SWR, 50 ms", "No SWR, 60 ms", "No SWR, 70 ms",
        "SWR, 80 ms", "SWR, 90 ms", "SWR, 100 ms",
    ]  # fmt: skip
    assert windows["SWR, 80 ms"] == (3.0, 3.08)
    with pytest.raises(AssertionError, match="matches 0 rows"):
        figure.example_windows(events.iloc[1:])
    with pytest.raises(AssertionError, match="not of its kind"):
        figure.example_windows(events.assign(is_swr=True))
