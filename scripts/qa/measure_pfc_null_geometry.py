"""What the mPFC held-out null's windows look like, beside the real events.

The empirical p-value each unit carries compares the held-out improvement a
real event class buys with the improvement a duration-matched non-event control
buys, redrawn ``R`` times. That comparison is only as good as the resemblance
between a drawn window set and a real one. Four properties are measured here
and reported for the reader of the p-values, not acted on:

1. **Onset spacing.** Real non-local onsets are separated by at least their own
   durations -- one event ends before the next begins. Control onsets are drawn
   distinct but may fall a single bin apart, because duration constrains only
   where an onset may be drawn and never enters the onset-only design. A
   regressor whose onsets bunch is not the same shape as one whose onsets are
   spread, so the spacing distributions are put beside each other.
2. **Engagement-null overlap.** The engagement statistic's null draws are
   independent of the shipped matched control they are entered beside, so a
   draw may land where that control already sits -- which two real classes
   never do. A large overlap would make that null's added regressor partly a
   copy of the baseline, and the engagement p-value anti-conservative.
3. **Units called with no out-of-fold gain.** A one-sided empirical p-value can
   be small while the observed improvement itself is zero or negative, when the
   null's draws are worse still. Those units are "significant" against the
   control without predicting held-out spikes better than a constant rate.
4. **Events the matched test could not use.** An event with no legal non-event
   window inside its own fitted bout cannot enter a matched randomisation test.
   Those events are excluded from both sides, so the p-value describes the
   tested subset, not the whole class.

Sections 2-4 read the shipped bundles and their sidecars, so they cover the
whole batch. Section 1 has to redraw the control from its recorded seed, which
needs the session's own fit mask and events: it is reported for the sessions
named on the command line.

Usage
-----
``uv run python scripts/qa/measure_pfc_null_geometry.py \
    --data-dir Processed-Data --session j1620210710_02_r1 \
    --session chimi20200223_06_r3 --out docs/audits/<date>-null-geometry.md``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc.glm import (
    EVENT_CLASSES,
    MATCHED_CONTROL_SOURCE_CLASS,
    make_matched_control_times,
    matched_control_seed,
)
from continuum_swr_replay.analysis.pfc.heldout import (
    _matchable_subset_seed,
    null_draw_seeds,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

#: Column carrying the observed statistic each empirical p-value was read
#: against, per class. The engagement test is keyed by its pair name.
OBSERVED_COLUMN = {
    "SWR": "SWR vs. constant: matched_null_observed_delta_dev",
    "non_local": "non_local vs. constant: matched_null_observed_delta_dev",
    "non_local_no_SWR": (
        "non_local_no_SWR vs. constant: matched_null_observed_delta_dev"
    ),
    "non_local_SWR": "non_local_SWR vs. constant: matched_null_observed_delta_dev",
    "engagement": (
        "matched_control + non_local_no_SWR vs. matched_control: "
        "matched_null_observed_delta_dev"
    ),
}

#: Column carrying the empirical p-value, per class.
PVALUE_COLUMN = {
    "SWR": "SWR: p_heldout",
    "non_local": "non_local: p_heldout",
    "non_local_no_SWR": "non_local_no_SWR: p_heldout",
    "non_local_SWR": "non_local_SWR: p_heldout",
    "engagement": ("matched_control + non_local_no_SWR vs. matched_control: p_heldout"),
}


def _cell(value: object, float_format: str) -> str:
    """One markdown cell: missing renders blank, a count renders as a count."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return float_format.format(value)
    return str(value)


def _md(frame: pd.DataFrame, float_format: str = "{:.4g}") -> str:
    """A markdown table, with missing values rendered as an empty cell."""
    if frame.empty:
        return "_(no rows)_\n"
    columns = {
        name: [_cell(value, float_format) for value in frame[name]]
        for name in frame.columns
    }
    header = "| " + " | ".join(str(name) for name in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(cells[i] for cells in columns.values()) + " |"
        for i in range(len(frame))
    ]
    return "\n".join([header, rule, *body]) + "\n"


def bundle_paths(data_dir: Path) -> list[Path]:
    """Canonical mPFC bundles in a tree, in session order."""
    return sorted(data_dir.glob("*_mPFC_modulation.pkl"))


def session_of(path: Path) -> str:
    """Session key from a bundle path (``chimi20200223_02_r1``)."""
    return path.name.replace("_mPFC_modulation.pkl", "")


def read_sidecar(path: Path) -> dict[str, Any]:
    """The bundle's provenance sidecar as a dict."""
    return json.loads(path.with_suffix(".provenance.json").read_text())


# ---------------------------------------------------------------------------
# 1. Onset spacing of the control, the null's draws and the real events
# ---------------------------------------------------------------------------


def spacing_summary(
    label: str, times: pd.DataFrame, n_draws: int | None = None
) -> dict[str, Any]:
    """Inter-onset spacing of one window set, against its own durations.

    Parameters
    ----------
    label : str
        What the row describes.
    times : pandas.DataFrame
        ``start_time`` / ``end_time`` of the windows, in seconds.
    n_draws : int, optional
        How many draws the rows were pooled over, for the report.

    Returns
    -------
    dict
        Median, 5th percentile and minimum inter-onset gap in ms, and the
        fraction of consecutive onsets closer together than the earlier
        window's own duration.
    """
    frame = times.sort_values("start_time")
    starts = frame["start_time"].to_numpy(dtype=float)
    durations = (frame["end_time"] - frame["start_time"]).to_numpy(dtype=float)
    gaps = np.diff(starts)
    if gaps.size == 0:
        return {
            "windows": label,
            "n_windows": int(len(frame)),
            "n_draws": n_draws,
            "median_gap_ms": np.nan,
            "p05_gap_ms": np.nan,
            "min_gap_ms": np.nan,
            "frac_closer_than_source_duration": np.nan,
        }
    closer = gaps < durations[:-1]
    return {
        "windows": label,
        "n_windows": int(len(frame)),
        "n_draws": n_draws,
        "median_gap_ms": float(np.median(gaps) * 1e3),
        "p05_gap_ms": float(np.percentile(gaps, 5) * 1e3),
        "min_gap_ms": float(gaps.min() * 1e3),
        "frac_closer_than_source_duration": float(closer.mean()),
    }


def draw_spacing_summary(label: str, draws: list[pd.DataFrame]) -> dict[str, Any]:
    """Spacing of a set of null draws, summarised across draws.

    Each draw is measured on its own time axis and the per-draw statistics are
    then combined: pooling the draws' onsets first would make every gap look
    ``len(draws)`` times smaller than any single regressor's really is.
    """
    per_draw = pd.DataFrame([spacing_summary(label, draw) for draw in draws])
    return {
        "windows": label,
        "n_windows": int(per_draw["n_windows"].median()),
        "n_draws": len(draws),
        "median_gap_ms": float(per_draw["median_gap_ms"].median()),
        "p05_gap_ms": float(per_draw["p05_gap_ms"].median()),
        "min_gap_ms": float(per_draw["min_gap_ms"].median()),
        "frac_closer_than_source_duration": float(
            per_draw["frac_closer_than_source_duration"].mean()
        ),
    }


def session_window_sets(
    nwb_file_name: str,
    epoch_name: str,
    sidecar: Mapping[str, Any],
    n_draws: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[pd.DataFrame]]]:
    """Rebuild a session's real events, shipped control and null draws.

    The draws are regenerated from the seeds the sidecar records, by the same
    calls the runner made, so the windows measured here are the windows the
    p-values were read from.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
        The session.
    sidecar : Mapping
        Its provenance record, for the two seeds and the class subsets.
    n_draws : int
        How many of the recorded null draws to rebuild per class. The draws are
        seeded independently, so a prefix of them is a fair sample of all
        ``R``.

    Returns
    -------
    real : dict of str to pandas.DataFrame
        The event classes and the shipped ``matched_control``.
    drawn : dict of str to list of pandas.DataFrame
        The rebuilt null draws per class, one frame per draw.
    """
    from continuum_swr_replay.analysis.hippocampus import load_results
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        align_results_to_position,
    )
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        load_session_canonical_events,
        load_session_swr_intervals,
    )
    from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
    from continuum_swr_replay.analysis.pfc.glm import (
        build_session_event_sets,
        build_session_fit_mask,
    )
    from continuum_swr_replay.data_loaders import load_data

    stop_cohort = str(sidecar["stop_cohort"])
    cohort_stops = (
        None
        if stop_cohort == "all_stops"
        else load_session_stop_cohort(nwb_file_name, epoch_name, cohort=stop_cohort)[0]
    )
    results, _, _ = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC", load_detector=False
    )
    data = load_data(nwb_file_name, epoch_name)
    results, position_info, _ = align_results_to_position(
        results, data["position_info"]
    )
    data = {**data, "position_info": position_info}

    _, fit_mask, _ = build_session_fit_mask(
        position_info, cohort_stops=cohort_stops, stop_cohort=stop_cohort
    )
    canonical_events, _ = load_session_canonical_events(nwb_file_name, epoch_name)
    swr_intervals, _ = load_session_swr_intervals(nwb_file_name, epoch_name)
    event_times, _ = build_session_event_sets(
        data,
        results,
        canonical_events=canonical_events,
        swr_intervals=swr_intervals,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        fit_mask=fit_mask,
        basis_half_window_s=0.25,
    )

    control = make_matched_control_times(
        event_times,
        fit_mask,
        np.random.default_rng(matched_control_seed(nwb_file_name, epoch_name)),
    )
    real = {**event_times, "matched_control": control.times}

    subset_seed = int(sidecar["control_draw_seed"])
    seeds = null_draw_seeds(subset_seed, int(sidecar["n_control_draws"]))[:n_draws]
    drawn: dict[str, list[pd.DataFrame]] = {}
    for class_name in EVENT_CLASSES:
        selection = make_matched_control_times(
            {**event_times, MATCHED_CONTROL_SOURCE_CLASS: event_times[class_name]},
            fit_mask,
            np.random.default_rng(_matchable_subset_seed(subset_seed, class_name)),
        )
        tested = event_times[class_name].iloc[selection.source_indices]
        # Kept draw by draw: pooling them first would put 20 draws' onsets on
        # one time axis and divide every gap by 20.
        drawn[class_name] = [
            make_matched_control_times(
                {**event_times, MATCHED_CONTROL_SOURCE_CLASS: tested},
                fit_mask,
                np.random.default_rng(int(seed)),
            ).times
            for seed in seeds
        ]
    return real, drawn


def spacing_section(
    sessions: Iterable[str], data_dir: Path, n_draws: int
) -> tuple[str, pd.DataFrame]:
    """Onset spacing of real events, shipped control and null draws."""
    rows: list[dict[str, Any]] = []
    for session in sessions:
        bundle = data_dir / f"{session}_mPFC_modulation.pkl"
        sidecar = read_sidecar(bundle)
        real, drawn = session_window_sets(
            str(sidecar["nwb_file_name"]),
            str(sidecar["epoch_name"]),
            sidecar,
            n_draws,
        )
        for name in ("non_local", "SWR"):
            rows.append(
                {"session": session, **spacing_summary(f"real {name}", real[name])}
            )
        rows.append(
            {
                "session": session,
                **spacing_summary("shipped matched_control", real["matched_control"]),
            }
        )
        for name, draws in drawn.items():
            rows.append(
                {
                    "session": session,
                    **draw_spacing_summary(f"null draws ({name})", draws),
                }
            )
    table = pd.DataFrame(rows)
    text = (
        "## How the drawn windows are spaced, beside the real ones\n\n"
        "Gaps are between consecutive onsets of one window set, in "
        "milliseconds, measured inside one window set; "
        "`frac_closer_than_source_duration` is the share of "
        "consecutive onsets whose gap is under the earlier window's own "
        "duration. Real events cannot overlap, so their fraction is zero by "
        "construction. Control onsets are drawn distinct but their "
        "duration-matching windows may overlap, which is what makes an exact "
        "count-matched draw possible in a dense bout; the price is a drawn "
        "regressor whose onsets can bunch where a real one's never do. A null "
        "row is the median over the first "
        f"{n_draws} recorded draws of that draw's own statistic (the last "
        "column is the mean over draws), each draw rebuilt from its own "
        "recorded seed; the draws are never pooled onto one time axis, which "
        "would divide every gap by the number of draws.\n\n"
    )
    return text + _md(table), table


# ---------------------------------------------------------------------------
# 2. Engagement-null overlap with the shipped control
# ---------------------------------------------------------------------------


def overlap_section(data_dir: Path) -> tuple[str, pd.DataFrame]:
    """Per-session overlap of the engagement null's draws with the control."""
    rows = []
    for bundle in bundle_paths(data_dir):
        record = read_sidecar(bundle)
        overlap = record.get("engagement_null_overlap") or {}
        rows.append(
            {
                "session": session_of(bundle),
                "median_overlap_fraction": overlap.get("median", np.nan),
                "max_overlap_fraction": overlap.get("max", np.nan),
            }
        )
    table = pd.DataFrame(rows)
    median = table["median_overlap_fraction"]
    summary = pd.DataFrame(
        [
            {
                "n_sessions": int(median.notna().sum()),
                "median_over_sessions": float(median.median()),
                "min_over_sessions": float(median.min()),
                "max_over_sessions": float(median.max()),
                "worst_session": str(table.loc[median.idxmax(), "session"]),
                "max_single_draw_overlap": float(table["max_overlap_fraction"].max()),
            }
        ]
    )
    text = (
        "## How much of the engagement null is a copy of its baseline\n\n"
        "The engagement statistic's null draws are independent of the shipped "
        "matched control they are entered beside, so a draw may place a window "
        "where that control already sits -- something two real event classes "
        "never do, because each class's windows avoid the others'. The cost is "
        "measured rather than removed: blocking it would draw the null by "
        "different rules from the control it stands in for. The larger this "
        "fraction, the more of the null's added regressor is a copy of the "
        "baseline it is entered beside, and the more anti-conservative the "
        "engagement p-value.\n\n"
    )
    return text + _md(summary), table


# ---------------------------------------------------------------------------
# 3. Units called without an out-of-fold gain
# ---------------------------------------------------------------------------


def non_positive_section(
    data_dir: Path, alpha: float, min_firing_rate: float | None
) -> tuple[str, pd.DataFrame]:
    """Units at ``p < alpha`` whose observed improvement is not positive.

    The population is the one the audit and the figure read -- four finite
    primary modulation curves and the same held-out provenance contract. A
    rate screen is used only when explicitly supplied for a sensitivity audit.

    Raises
    ------
    KeyError
        If the population table is missing a class's p-value or observed
        statistic. Every class in :data:`PVALUE_COLUMN` gets a row or the
        section is refused.
    """
    from continuum_swr_replay.analysis.pfc.load_population import load_population_data

    table, _, _ = load_population_data(data_dir, min_firing_rate=min_firing_rate)
    rows = []
    for name, pvalue_column in PVALUE_COLUMN.items():
        observed_column = OBSERVED_COLUMN[name]
        for column in (pvalue_column, observed_column):
            if column not in table:
                # Skipping the class would drop it from the report without
                # saying so, and read as "nothing to report" for that class.
                raise KeyError(
                    f"population table has no column {column!r}, needed to "
                    f"count units called for {name!r}"
                )
        pvalues = pd.to_numeric(table[pvalue_column], errors="coerce").to_numpy(
            dtype=float
        )
        observed = pd.to_numeric(table[observed_column], errors="coerce").to_numpy(
            dtype=float
        )
        finite_pvalue = np.isfinite(pvalues)
        called = finite_pvalue & (pvalues < alpha)
        non_positive = called & (observed <= 0)
        rows.append(
            {
                "class": name,
                "n_units": int(len(table)),
                "n_tested": int(finite_pvalue.sum()),
                "n_called": int(called.sum()),
                "n_called_with_non_positive_gain": int(non_positive.sum()),
                "frac_of_called": float(
                    non_positive.sum() / called.sum() if called.sum() else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)
    text = (
        "## Units called with no out-of-fold gain\n\n"
        "The empirical p-value is one-sided against the control-draw null, so "
        "a unit can sit below "
        f"{alpha} while its own observed held-out improvement over the "
        "constant model is zero or negative: the draws simply did worse. Those "
        "units are evidence that the class beats a content-free regressor of "
        "the same size, not that the class predicts held-out spikes better "
        "than a constant rate. The statistic counted here is the one the "
        "p-value was read against -- the improvement on the fixed matchable "
        "event subset.\n\n"
    )
    return text + _md(summary), summary


# ---------------------------------------------------------------------------
# 4. Events the matched test could not use
# ---------------------------------------------------------------------------


def excluded_section(data_dir: Path) -> tuple[str, pd.DataFrame]:
    """Per-session, per-class counts of events no legal control could match."""
    rows = []
    for bundle in bundle_paths(data_dir):
        record = read_sidecar(bundle)
        for class_name, window in (record.get("null_windows") or {}).items():
            total = int(window["n_events_total"])
            rows.append(
                {
                    "session": session_of(bundle),
                    "event_class": class_name,
                    "n_events_total": total,
                    "n_events_tested": int(window["n_events_tested"]),
                    "n_events_excluded_unmatchable": int(
                        window["n_events_excluded_unmatchable"]
                    ),
                    "frac_excluded": (
                        float(window["n_events_excluded_unmatchable"] / total)
                        if total
                        else np.nan
                    ),
                }
            )
    table = pd.DataFrame(rows)
    corpus = (
        table.groupby("event_class", as_index=False)
        .agg(
            n_sessions=("session", "size"),
            n_events_total=("n_events_total", "sum"),
            n_events_excluded=("n_events_excluded_unmatchable", "sum"),
            max_session_frac_excluded=("frac_excluded", "max"),
        )
        .assign(
            frac_excluded=lambda frame: frame["n_events_excluded"]
            / frame["n_events_total"]
        )
    )
    worst = table.loc[table["frac_excluded"].idxmax()]
    text = (
        "## Events the matched test could not use\n\n"
        "An event with no legal non-event window inside its own fitted bout "
        "cannot enter a matched randomisation test, so one fixed subset -- "
        "selected from timing and duration alone, never from spikes -- is used "
        "on both the observed and the null side. The empirical p-value is "
        "therefore a statement about that subset, not about every event of the "
        "class; the all-event held-out difference remains separately in the "
        "bundle. The worst single session/class here is "
        f"{worst['session']} / {worst['event_class']}, which excludes "
        f"{int(worst['n_events_excluded_unmatchable'])} of "
        f"{int(worst['n_events_total'])} events "
        f"({100 * worst['frac_excluded']:.1f} %).\n\n"
    )
    return text + _md(corpus), table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Tree holding the canonical mPFC bundles and their sidecars.",
    )
    parser.add_argument(
        "--session",
        dest="sessions",
        action="append",
        default=[],
        help=(
            "Session to rebuild the control and null draws for, e.g. "
            "j1620210710_02_r1. Repeatable; omit to skip the spacing section, "
            "which is the only one that reloads session data."
        ),
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=20,
        help="Null draws rebuilt per class per named session.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--min-firing-rate",
        type=float,
        default=None,
        help=(
            "Optional sensitivity screen on immobility firing rate. Omitted "
            "by default, matching the descriptive Figure 4 population."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the report here.")
    args = parser.parse_args()

    sections = ["# Geometry of the mPFC held-out null\n"]
    sections.append(
        "Generated by `scripts/qa/measure_pfc_null_geometry.py`. Every number "
        "here describes the null the shipped empirical p-values were read "
        "against. Nothing is changed by it: the draw is what it is, and these "
        "are the terms on which its p-values should be read.\n"
    )
    if args.sessions:
        text, _ = spacing_section(args.sessions, args.data_dir, args.n_draws)
        sections.append(text)
    text, _ = overlap_section(args.data_dir)
    sections.append(text)
    text, _ = non_positive_section(args.data_dir, args.alpha, args.min_firing_rate)
    sections.append(text)
    text, _ = excluded_section(args.data_dir)
    sections.append(text)

    report = "\n".join(sections)
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
