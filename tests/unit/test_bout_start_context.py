"""``bouts.bout_start_context`` equals the per-bout loop it replaced.

Both stop builders read the trial context and the previous, next,
last-rewarded and last-visited wells at each bout's first row. They used to
iterate the bout groups and call ``strict_first`` (the first row's value as a
float, NaN kept) per well; ``bout_start_context`` reads the same first row with
``groupby.first(skipna=False)``. ``strict_first`` checks no agreement within a
bout: it is the first row's value whatever the later rows hold, so neither does
the helper. The reference below is the pre-helper builder code, verbatim but
for its names. The frame covers single-row bouts, all-NaN bouts, bouts whose
rows disagree (a NaN first row before a finite one, and the reverse), and bool,
object and categorical columns.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay._helpers import (
    arm_to_patch,
    single_arm_from_well,
    stay_indicator_from_patches,
    strict_first,
)
from continuum_swr_replay.analysis.replay.bouts import (
    bout_start_context,
    summarize_bout_location,
)
from continuum_swr_replay.analysis.replay.simulation import make_bandit_track_graph

WELL_COLUMNS = ("prev_well", "next_well", "last_rewarded_well", "last_visited_well")


def _reference(
    bout_rows: pd.DataFrame,
    n_bouts: int,
    track_graph: nx.Graph,
    excluded: pd.DataFrame,
) -> pd.DataFrame:
    """The builders' per-bout code before ``bout_start_context`` (fda010e8)."""
    grouped = bout_rows.groupby("_bout_idx", sort=True)
    per_bout = pd.DataFrame(index=pd.RangeIndex(n_bouts))
    bout_first_row = grouped.first(skipna=False)
    for col in (
        "did_last_completed_trip_switch_patch",
        "patch_switch_fraction_last5_completed_trips",
        "leaky_q_value_diff",
    ):
        per_bout[col] = bout_first_row[col].reindex(np.arange(n_bouts)).to_numpy()
    per_bout["is_rewarded_outcome"] = (
        bout_first_row["is_rewarded_outcome"].reindex(np.arange(n_bouts)).to_numpy()
    )
    per_bout["completed_trial_id"] = pd.array(
        bout_first_row["completed_trial_id"].reindex(np.arange(n_bouts)),
        dtype="Int64",
    )
    location = summarize_bout_location(bout_rows, n_bouts, track_graph, excluded)
    for column in location.columns:
        per_bout[column] = location[column]
    current_arm_vals = location["occupancy_well_arm"].to_numpy()

    prev_arm_vals = np.full(n_bouts, np.nan)
    next_arm_vals = np.full(n_bouts, np.nan)
    last_rew_vals = np.full(n_bouts, np.nan)
    completed_trial_arm_vals = np.full(n_bouts, np.nan)
    for bi, (_, g) in enumerate(grouped):
        prev_arm_vals[bi] = single_arm_from_well(
            strict_first(g["prev_well"]), track_graph
        )
        next_arm_vals[bi] = single_arm_from_well(
            strict_first(g["next_well"]), track_graph
        )
        last_rew_vals[bi] = single_arm_from_well(
            strict_first(g["last_rewarded_well"]), track_graph
        )
        completed_trial_arm_vals[bi] = single_arm_from_well(
            strict_first(g["last_visited_well"]), track_graph
        )

    per_bout["prev_arm"] = prev_arm_vals
    per_bout["next_arm"] = next_arm_vals
    per_bout["last_rewarded_arm"] = last_rew_vals
    per_bout["completed_trial_arm"] = completed_trial_arm_vals
    cur_patch = np.array([arm_to_patch(a) for a in current_arm_vals])
    nxt_patch = np.array([arm_to_patch(a) for a in next_arm_vals])
    per_bout["current_patch"] = cur_patch
    per_bout["prev_patch"] = [arm_to_patch(a) for a in prev_arm_vals]
    per_bout["next_patch"] = nxt_patch
    per_bout["is_stay"] = stay_indicator_from_patches(nxt_patch, cur_patch)
    return per_bout


# Rows per bout, in time order. Each tuple is one row:
# (prev, next, last_rewarded, last_visited, switch, reward, at-well well).
NAN = np.nan
BOUTS: list[list[tuple]] = [
    # 0: one row, every well known.
    [(0.0, 1.0, 0.0, 1.0, True, 1.0, 1.0)],
    # 1: one row, every well NaN (before the first trial).
    [(NAN, NAN, NAN, NAN, None, NAN, NAN)],
    # 2: all rows NaN.
    [(NAN, NAN, NAN, NAN, None, NAN, NAN)] * 3,
    # 3: NaN first row, finite later rows: the NaN is the bout's value.
    [(NAN, NAN, NAN, NAN, None, NAN, 2.0)]
    + [(2.0, 3.0, 2.0, 3.0, False, 0.0, 2.0)] * 2,
    # 4: finite first row, then disagreeing and NaN rows: the first row wins.
    [
        (4.0, 5.0, 4.0, 5.0, True, 1.0, 5.0),
        (1.0, 2.0, NAN, 0.0, False, 0.0, 5.0),
        (NAN, NAN, NAN, NAN, None, NAN, 5.0),
    ],
    # 5: next well in another patch (a switch) over two rows.
    [(3.0, 0.0, 3.0, 2.0, False, 0.0, 2.0)] * 2,
    # 6: next well in the same patch (a stay).
    [(5.0, 3.0, NAN, 2.0, True, 1.0, 2.0)],
]


def _bout_rows(categorical_wells: bool) -> pd.DataFrame:
    rows = [(b, *row) for b, bout in enumerate(BOUTS) for row in bout]
    frame = pd.DataFrame(
        rows,
        columns=[
            "_bout_idx",
            *WELL_COLUMNS,
            "did_last_completed_trip_switch_patch",
            "is_rewarded_outcome",
            "occupancy_well_id",
        ],
    )
    frame.index = pd.Index(0.1 * np.arange(len(frame)), name="time")
    # ``did_last_completed_trip_switch_patch`` stays object (bools and None).
    frame["patch_switch_fraction_last5_completed_trips"] = np.linspace(0, 1, len(frame))
    frame.loc[frame.index[::3], "patch_switch_fraction_last5_completed_trips"] = NAN
    frame["leaky_q_value_diff"] = np.where(
        frame["prev_well"].isna(), NAN, frame["prev_well"] / 10
    )
    completed = pd.array(np.arange(len(frame)) // 2, dtype="Int64")
    completed[frame["prev_well"].isna().to_numpy()] = pd.NA
    frame["completed_trial_id"] = completed
    frame["occupancy_state"] = np.where(
        frame["occupancy_well_id"].isna(), "in_transit", "at_well"
    )
    frame["nearest_well_distance_cm"] = np.where(
        frame["occupancy_well_id"].isna(), 40.0, 1.0
    )
    frame["track_segment_id"] = frame["occupancy_well_id"].fillna(6.0)
    if categorical_wells:
        for column in WELL_COLUMNS:
            frame[column] = frame[column].astype("category")
    return frame


EXCLUDED = {
    "none": pd.DataFrame({"start_time": [], "end_time": []}, dtype=float),
    # Covers bout 0's row and bout 3's first row (rows 0 and 4 at 0.0 and 0.4 s).
    "some": pd.DataFrame({"start_time": [-0.05, 0.35], "end_time": [0.05, 0.45]}),
}


@pytest.mark.parametrize("categorical_wells", [False, True])
@pytest.mark.parametrize("excluded", sorted(EXCLUDED))
def test_matches_per_bout_loop(categorical_wells: bool, excluded: str) -> None:
    bout_rows = _bout_rows(categorical_wells)
    graph = make_bandit_track_graph()
    n_bouts = len(BOUTS)

    expected = _reference(bout_rows, n_bouts, graph, EXCLUDED[excluded])
    got = bout_start_context(bout_rows, n_bouts, graph, EXCLUDED[excluded])

    pd.testing.assert_frame_equal(
        got[expected.columns], expected, check_exact=True, check_dtype=True
    )
    assert set(got.columns) == set(expected.columns)


def test_first_row_is_read_as_is() -> None:
    """The values the reference must hold, so the comparison above is not empty."""
    context = bout_start_context(
        _bout_rows(False), len(BOUTS), make_bandit_track_graph(), EXCLUDED["none"]
    )
    # Bout 3's first row is NaN; its later finite rows do not leak in.
    assert np.isnan(context.loc[3, ["prev_arm", "next_arm"]].astype(float)).all()
    # Bout 4's first row wins over its disagreeing later rows.
    assert context.loc[4, "prev_arm"] == 4.0
    assert context.loc[4, "completed_trial_arm"] == 5.0
    # Bout 5 switches patch, bout 6 stays.
    assert context.loc[5, "is_stay"] == 0.0
    assert context.loc[6, "is_stay"] == 1.0
