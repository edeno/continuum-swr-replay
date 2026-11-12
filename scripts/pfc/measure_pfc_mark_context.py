"""What the marked-kernel content mark is entangled with, session by session.

The marked mPFC kernels (``docs/design/pfc_marked_event_kernels.md``) score a
content mark -- ``arm_fraction`` on the animal's currently occupied arm -- by
permuting that mark freely among the events of one session and replay class.
Free permutation is exchangeable only if the mark carries no information that
the nuisance and contextual variables riding along with each event also carry.
It does carry some. Three of those associations are measured here and reported
for the reader of the label-shuffle reference, not acted on:

1. **Event duration.** The design already holds a log-duration marked kernel,
   so a content mark correlated with duration is partly redistributing a term
   that stays put under the shuffle.
2. **Session time.** Content drifts over a session; a shuffle that ignores when
   an event happened reassigns marks across that drift.
3. **Occupied arm.** ``arm_fraction`` on the current arm is defined against the
   arm the animal is sitting on, so arm identity alone explains some of its
   variance.

Every number is computed over the canonical ``event_arm_table``, which is every
canonical session -- not only the sessions with mPFC units -- restricted
to the well-anchored events the marked kernels fit (event ``occupancy_state`` of
``at_well`` or ``mixed``, the event-grain form of the PFC ``at_well`` stop
cohort; see ``docs/design/cohort_and_timing_rules.md``). The groups are
(animal, session date, epoch, ``is_swr``), the same partition the shuffle
permutes within. "Session time" is the event midpoint.

Usage
-----
``uv run python scripts/pfc/measure_pfc_mark_context.py \
    --out .claude/docs/reviews/<date>-mark-context.md``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.paths import table_path
from continuum_swr_replay.tables.io import read_table

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.qa._report import _md  # noqa: E402

#: Columns that identify one shuffle group: the shuffle permutes the mark
#: among the events of one session and one replay class.
GROUP_COLUMNS = ("animal", "session_date", "epoch", "is_swr")

#: ``|r|`` above which an association is counted as more than incidental.
CORRELATION_THRESHOLD = 0.2

#: Event-grain occupancy states of the well-anchored cohort the marked kernels
#: fit; events elsewhere carry a current-arm row but are never fit.
WELL_ANCHORED_STATES = ("at_well", "mixed")


def load_current_arm_marks(path: Path) -> pd.DataFrame:
    """Read the one row per well-anchored event that carries the content mark.

    Parameters
    ----------
    path : pathlib.Path
        The canonical ``event_arm_table`` parquet. It holds one row per event
        x candidate arm; exactly the ``is_current_arm`` rows carry the mark the
        marked kernels use. Only events whose ``occupancy_state`` is one of
        :data:`WELL_ANCHORED_STATES` are kept, matching the fitted population.

    Returns
    -------
    pandas.DataFrame, shape (n_well_anchored_events_with_a_current_arm, 8)
        The group keys, ``arm_fraction``, ``occupancy_well_arm``, and the
        event's start and end times in seconds.
    """
    table = read_table(
        path,
        columns=[
            *GROUP_COLUMNS,
            "is_current_arm",
            "occupancy_state",
            "arm_fraction",
            "occupancy_well_arm",
            "event_start_s",
            "event_end_s",
        ],
    )
    keep = table["is_current_arm"].to_numpy(dtype=bool) & table["occupancy_state"].isin(
        WELL_ANCHORED_STATES
    ).to_numpy(dtype=bool)
    return table.loc[keep].drop(columns=["is_current_arm", "occupancy_state"])


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson ``r`` of two 1-D arrays, NaN when either is constant.

    Parameters
    ----------
    left, right : numpy.ndarray, shape (n_events,)

    Returns
    -------
    float
    """
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2:
        return float("nan")
    left, right = left[finite], right[finite]
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _one_way_r2(values: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of ``values`` variance explained by the ``labels`` grouping.

    This is the one-way ANOVA ``R^2`` -- between-group sum of squares over
    total sum of squares -- for a categorical predictor with no ordering.

    Parameters
    ----------
    values : numpy.ndarray, shape (n_events,)
    labels : numpy.ndarray, shape (n_events,)
        Categorical group label per event; missing labels are dropped.

    Returns
    -------
    float
        NaN when fewer than two events remain or ``values`` is constant.
    """
    usable = pd.notna(labels) & np.isfinite(values)
    if usable.sum() < 2:
        return float("nan")
    values, labels = values[usable], labels[usable]
    total = float(((values - values.mean()) ** 2).sum())
    if total == 0:
        return float("nan")
    within = float(
        sum(
            ((values[labels == level] - values[labels == level].mean()) ** 2).sum()
            for level in pd.unique(labels)
        )
    )
    return 1.0 - within / total


def group_context_table(marks: pd.DataFrame) -> pd.DataFrame:
    """Per session x replay-class associations of the content mark.

    Parameters
    ----------
    marks : pandas.DataFrame
        Output of :func:`load_current_arm_marks`.

    Returns
    -------
    pandas.DataFrame, shape (n_groups, 8)
        One row per (animal, session date, epoch, ``is_swr``) group with the
        group size, the mark's correlation with log event duration and with
        the event midpoint time, and the one-way ``R^2`` on the occupied arm.
    """
    rows = []
    for keys, group in marks.groupby(list(GROUP_COLUMNS), sort=True):
        fraction = group["arm_fraction"].to_numpy(dtype=float)
        start = group["event_start_s"].to_numpy(dtype=float)
        end = group["event_end_s"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, keys, strict=True)),
                "n_events": int(len(group)),
                "r_log_duration": _pearson(fraction, np.log(end - start)),
                "r_midpoint_time": _pearson(fraction, 0.5 * (start + end)),
                "r2_occupancy_well_arm": _one_way_r2(
                    fraction, group["occupancy_well_arm"].to_numpy()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-group table to the figures the design doc quotes.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`group_context_table`.

    Returns
    -------
    pandas.DataFrame, shape (1, 6)
    """
    duration = table["r_log_duration"]
    midpoint = table["r_midpoint_time"]
    return pd.DataFrame(
        [
            {
                "n_groups": int(len(table)),
                "min_group_size": int(table["n_events"].min()),
                "median_r_log_duration": float(duration.median()),
                "n_groups_duration_r_above_threshold": int(
                    (duration.abs() > CORRELATION_THRESHOLD).sum()
                ),
                "n_groups_midpoint_r_above_threshold": int(
                    (midpoint.abs() > CORRELATION_THRESHOLD).sum()
                ),
                "median_r2_occupancy_well_arm": float(
                    table["r2_occupancy_well_arm"].median()
                ),
            }
        ]
    )


def build_report(summary: pd.DataFrame, table_path: Path) -> str:
    """The markdown report for one run."""
    row = summary.iloc[0]
    return (
        "# What the marked-kernel content mark is entangled with\n\n"
        "Generated by `scripts/pfc/measure_pfc_mark_context.py` over "
        f"`{table_path}`, which holds every canonical session -- not only the "
        "sessions with mPFC units -- restricted to well-anchored events "
        "(`occupancy_state` of `at_well` or `mixed`, the population the marked "
        "kernels fit). Groups are (animal, session date, epoch, "
        "`is_swr`), the partition the label shuffle permutes within; 'session "
        "time' is the event midpoint.\n\n"
        f"Across {int(row['n_groups'])} groups (smallest "
        f"{int(row['min_group_size'])} events), the content/log-duration "
        f"correlation has median {row['median_r_log_duration']:.3f} and "
        f"exceeds |r| = {CORRELATION_THRESHOLD} in "
        f"{int(row['n_groups_duration_r_above_threshold'])} groups; the "
        f"content/session-time correlation exceeds it in "
        f"{int(row['n_groups_midpoint_r_above_threshold'])} groups. Occupied-"
        "arm identity explains a median "
        f"{100 * row['median_r2_occupancy_well_arm']:.1f} % of content-mark "
        "variance.\n\n" + _md(summary)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-arm-table",
        type=Path,
        default=table_path("event_arm_table"),
        help="Canonical per-event-arm table the content mark is read from.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the report here.")
    args = parser.parse_args()

    marks = load_current_arm_marks(args.event_arm_table)
    table = group_context_table(marks)
    report = build_report(summarize(table), args.event_arm_table)
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
