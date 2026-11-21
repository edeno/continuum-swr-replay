"""The Non-Local SWR split of the held-out state-evidence windows, and the bouts
that start inside a well visit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence.state_evidence import (
    bout_statistics,
    bout_visit_counts,
    mark_bouts_in_well_visits,
    split_non_local_by_swr,
)


def _windows() -> pd.DataFrame:
    """Two immobile Non-Local Continuous bouts (bins 2-4 and 10-12), one Local
    bout, and a matched baseline window for each Non-Local bout."""
    rows = [
        ("bout", "Non-Local Continuous", 2, 4, np.nan),
        ("bout", "Non-Local Continuous", 10, 12, np.nan),
        ("baseline", "Non-Local Continuous", np.nan, np.nan, 2),
        ("baseline", "Non-Local Continuous", np.nan, np.nan, 10),
        ("bout", "Local", 20, 25, np.nan),
        ("calibration", "Non-Local Continuous", np.nan, np.nan, np.nan),
    ]
    return pd.DataFrame(
        rows, columns=["kind", "state", "bout_s", "bout_e", "matched_bout_s"]
    ).assign(regime="immobile")


def test_a_bout_touching_an_swr_bin_and_its_matched_window_are_labeled_swr():
    swr = np.zeros(30, dtype=bool)
    swr[4] = True  # the last bin of the first bout
    out = split_non_local_by_swr(_windows(), swr)
    assert len(out) == 4  # Local and calibration rows are left out
    by_start = out.assign(
        key=np.where(out["kind"] == "bout", out["bout_s"], out["matched_bout_s"])
    ).set_index(["kind", "key"])["state"]
    assert by_start[("bout", 2)] == "Non-Local Continuous | SWR"
    assert by_start[("baseline", 2)] == "Non-Local Continuous | SWR"
    assert by_start[("bout", 10)] == "Non-Local Continuous | no SWR"
    assert by_start[("baseline", 10)] == "Non-Local Continuous | no SWR"


def test_an_swr_bin_just_outside_the_bout_does_not_count():
    swr = np.zeros(30, dtype=bool)
    swr[[1, 5]] = True
    out = split_non_local_by_swr(_windows(), swr)
    assert set(out["swr"]) == {"no SWR"}


def test_a_matched_window_without_its_bout_raises():
    windows = _windows()
    windows.loc[3, "matched_bout_s"] = 7
    with pytest.raises(ValueError, match="matched window has no bout"):
        split_non_local_by_swr(windows, np.zeros(30, dtype=bool))


def test_duplicate_bout_keys_raise():
    windows = _windows()
    windows.loc[1, ["bout_s", "bout_e"]] = [2, 4]
    with pytest.raises(ValueError, match="duplicate bout keys"):
        split_non_local_by_swr(windows, np.zeros(30, dtype=bool))


# --- the bouts that start inside a well visit -------------------------------------

TIME = 100.0 + 0.002 * np.arange(30)


def _labeled() -> pd.DataFrame:
    """The two Non-Local bouts and their matched windows (at bins 20 and 25),
    labeled by SWR (the first bout touches one), with their scored starts."""
    windows = _windows()
    windows["s"] = [2, 10, 20, 25, 20, 0]
    windows["start_time"] = TIME[windows["s"]]
    swr = np.zeros(30, dtype=bool)
    swr[4] = True
    return split_non_local_by_swr(windows, swr)


def test_a_bout_counts_when_its_first_sample_is_inside_a_visit_and_its_window_goes_with_it():
    # one visit from bin 8 to bin 22 (its final poke-out sample is outside)
    out = mark_bouts_in_well_visits(_labeled(), TIME, [TIME[8]], [TIME[22]])
    flags = out.set_index(
        ["kind", np.where(out["kind"] == "bout", out["bout_s"], out["matched_bout_s"])]
    )
    assert not flags.loc[("bout", 2), "starts_in_well_visit"]
    assert not flags.loc[("baseline", 2), "starts_in_well_visit"]
    assert flags.loc[("bout", 10), "starts_in_well_visit"]
    assert flags.loc[("baseline", 10), "starts_in_well_visit"]
    # the bout at 10 is matched to the window at bin 25, after the visit
    assert not flags.loc[("bout", 10), "matched_starts_in_well_visit"]
    assert flags.loc[("bout", 2), "matched_starts_in_well_visit"]


def test_a_bout_whose_first_sample_is_the_visit_s_final_poke_out_is_left_out():
    out = mark_bouts_in_well_visits(_labeled(), TIME, [TIME[0]], [TIME[10]])
    bouts = out[out["kind"] == "bout"].set_index("bout_s")
    assert bouts["starts_in_well_visit"].to_dict() == {2: True, 10: False}


def test_the_counts_per_state_and_swr_label():
    out = mark_bouts_in_well_visits(_labeled(), TIME, [TIME[8]], [TIME[22]])
    counts = bout_visit_counts(out).set_index("state")
    assert counts.loc[
        "Non-Local Continuous | SWR", ["n_bouts", "n_kept", "n_left_out"]
    ].tolist() == [1, 0, 1]
    assert counts.loc["Non-Local Continuous | no SWR"].to_dict() == {
        "regime": "immobile",
        "n_bouts": 1,
        "n_kept": 1,
        "n_left_out": 0,
        "n_kept_matched_outside_visit": 1,
    }


def test_a_planted_bout_outside_every_visit_does_not_move_the_statistics():
    """A third, immobile no-SWR bout with a huge score, before the visit: the
    statistics over the kept bouts are those without it."""
    windows = _windows().iloc[:4]
    windows["s"] = [2, 10, 20, 25]
    planted = pd.DataFrame(
        {
            "kind": ["bout", "baseline"],
            "state": "Non-Local Continuous",
            "bout_s": [0, np.nan],
            "bout_e": [1, np.nan],
            "matched_bout_s": [np.nan, 0],
            "regime": "immobile",
            "s": [0, 15],
        }
    )
    windows = pd.concat([windows, planted], ignore_index=True)
    windows["start_time"] = TIME[windows["s"]]
    windows["epoch_id"] = "e1"
    windows["animal"] = "a"
    windows["n_found"] = 1
    windows["runcv_nl_anim_mark_ps"] = [1.0, 3.0, 0.0, 0.0, 1000.0, 0.0]
    marked = mark_bouts_in_well_visits(
        split_non_local_by_swr(windows, np.zeros(30, dtype=bool)),
        TIME,
        [TIME[2]],
        [TIME[29]],
    )
    kept = marked[marked["starts_in_well_visit"]]
    stats = bout_statistics(
        kept,
        metrics={"Non-Local Continuous | no SWR": ["runcv_nl_anim_mark_ps"]},
        n_boot=10,
        n_flips=10,
    )
    row = stats[stats["regime"] == "immobile"].iloc[0]
    assert row["n"] == 2
    assert row["mean"] == 2.0


def test_windows_on_another_time_grid_refuse():
    labeled = _labeled()
    labeled["start_time"] += 0.001
    with pytest.raises(ValueError, match="not the decode time at s"):
        mark_bouts_in_well_visits(labeled, TIME, [TIME[8]], [TIME[22]])


def test_a_running_bout_is_read_wherever_it_happens():
    """Running is counted wherever it happens (the animal does not run inside a
    visit): only the immobile bouts are restricted to the visits."""
    labeled = _labeled()
    running = labeled.assign(regime="running")
    out = mark_bouts_in_well_visits(
        pd.concat([labeled, running], ignore_index=True), TIME, [TIME[8]], [TIME[22]]
    )
    bouts = out[out["kind"] == "bout"]
    immobile = bouts[bouts["regime"] == "immobile"].set_index("bout_s")
    moving = bouts[bouts["regime"] == "running"].set_index("bout_s")
    assert immobile["in_analyzed_subset"].to_dict() == {2: False, 10: True}
    assert moving["in_analyzed_subset"].to_dict() == {2: True, 10: True}
    # the matched window of a read bout is read with it
    assert out.groupby(["regime", "in_analyzed_subset"]).size().to_dict() == {
        ("immobile", False): 2,
        ("immobile", True): 2,
        ("running", True): 4,
    }
    counts = bout_visit_counts(out).set_index(["state", "regime"])
    assert counts.loc[("Non-Local Continuous | SWR", "running"), "n_kept"] == 1
    assert counts.loc[("Non-Local Continuous | SWR", "running"), "n_left_out"] == 0
    assert counts.loc[("Non-Local Continuous | SWR", "immobile"), "n_kept"] == 0
