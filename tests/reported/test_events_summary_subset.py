"""The events summary reads the events that start inside a well visit, and counts
every well-visit label, kept and left out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.constants import VISIT_CONTEXTS
from continuum_swr_replay.analysis.event_definition import swr_source
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
    count_visit_context,
    select_visit_context,
)
from continuum_swr_replay.reported import events as builder
from tests.reported.test_reported_events_values import _by_key, _events


def _labeled() -> pd.DataFrame:
    """The thirteen events of the builders' tests, all starting inside a visit,
    and one more event of every label, each an SWR-partial event of animal c,
    which would change every class count and share if it were read."""
    inside = _events().assign(
        event_id=np.arange(13, dtype="uint64"),
        visit_context=["in_visit"] * 12 + ["spans_visit_end"],
    )
    extra = inside.iloc[[12] * len(VISIT_CONTEXTS)].assign(
        event_id=np.arange(100, 100 + len(VISIT_CONTEXTS), dtype="uint64"),
        visit_context=list(VISIT_CONTEXTS),
    )
    return pd.concat([inside, extra], ignore_index=True)


def test_the_label_counts_are_count_visit_context():
    table = _labeled()
    values = _by_key(builder.visit_context_values(table, "s"))
    counts = count_visit_context(table)
    for label in VISIT_CONTEXTS:
        assert values[f"events_label_{label}"].value == counts.loc[label, "n"]
    assert values["events_label_in_visit"].value == 13
    assert values["events_label_spans_visit_end"].value == 2
    assert values["events_label_track_center"].value == 1
    assert values["events_in_subset"].value == 15
    assert values["events_left_out"].value == len(VISIT_CONTEXTS) - 2
    # the extra events are SWR-partial: every left-out one is an SWR event
    assert values["swr_events_in_subset"].value == 6 + 2
    assert values["swr_events_left_out"].value == len(VISIT_CONTEXTS) - 2
    assert values["non_swr_events_in_subset"].value == 7
    assert values["non_swr_events_left_out"].value == 0
    assert "kept" in values["events_label_in_visit"].description
    assert "left out" in values["events_label_track_leaf_arm"].description


class _Read(Exception):
    """Stops the summary after the event table's values."""


def test_summary_reads_the_selected_events(monkeypatch):
    table = _labeled()
    seen: dict[str, pd.DataFrame] = {}

    def stop(events, folder, n_epochs):
        seen["extras"] = events
        raise _Read

    monkeypatch.setattr(builder, "read_swr_table", lambda path: table)
    monkeypatch.setattr(
        builder, "file_record", lambda path, root: {"path": "p", "sha256": "0"}
    )
    monkeypatch.setattr(builder, "with_start_and_rest_theta", stop)
    # (the label check against the visits is tested in cohort_selection)
    monkeypatch.setattr(builder, "read_manifest", lambda run: {"epochs": {}})
    monkeypatch.setattr(builder, "check_visit_context_starts", lambda t, v: None)
    # (the state shares, the HPD coverage and the content distance are not
    # columns of these events)
    monkeypatch.setattr(builder, "state_composition_values", lambda events, s: [])
    monkeypatch.setattr(builder, "hpd_tight_values", lambda events, s: [])
    monkeypatch.setattr(builder, "content_distance_values", lambda events, s: [])
    added = []
    monkeypatch.setattr(
        builder.Summary, "add", lambda self, *values: added.extend(values)
    )
    with pytest.raises(_Read):
        builder.summarize(builder.default_inputs(), "by")

    selected = select_visit_context(table)
    assert seen["extras"]["event_id"].tolist() == selected["event_id"].tolist()
    values = _by_key(added)
    # the class counts of the selected rows: the thirteen and the two extra
    # events labeled in_visit and spans_visit_end (both SWR-partial)
    assert values["nonlocal_events"].value == 15
    assert values["nonlocal_events_non_swr"].value == 7
    assert values["nonlocal_events_swr_partial"].value == 6
    assert values["events_label_before_first_visit"].value == 1
    for key in ("nonlocal_events", "swr_overlapping_events"):
        assert values[key].description.endswith(f"; over {builder.EVENTS_SUBSET}")
    assert "; over" not in values["events_label_in_visit"].description


def test_content_bins_are_read_for_the_selected_events_only(tmp_path: Path):
    events = select_visit_context(_labeled())
    rows = []
    for event_id in [*events["event_id"], 100 + 5]:  # 105 is not selected
        rows += [{"event_id": event_id, "epoch_id": "e", "arm": 1}] * 2
    pd.DataFrame(rows).to_parquet(tmp_path / "e.parquet")

    bins, _ = builder.content_along_arms_bins(events, tmp_path, 1, ["arm"])

    assert set(bins["event_id"]) == set(events["event_id"])
    assert len(bins) == 2 * len(events)
    with pytest.raises(ValueError, match="do not cover"):
        builder.content_along_arms_bins(
            events.iloc[:-1].assign(
                event_id=lambda t: t["event_id"].where(t.index > 0, 999)
            ),
            tmp_path,
            1,
            ["arm"],
        )


def test_the_content_mass_is_summed_over_the_selected_events(tmp_path):
    """Each event of the run has two rows on one place (weights 1 and 3, the
    animal 10 and 30 cm away); the unselected event's rows are not read, and
    the rows of an epoch, animal and kind are one row per place and side."""
    events = select_visit_context(_labeled())
    rows = [
        {
            "event_id": event_id,
            "epoch_id": "e",
            "arm": 1,
            "arm_length_cm": 50.0,
            "place_bin": 3,
            "n_place_bins": 26,
            "distance_to_well_cm": 40.0,
            "far": True,
            "weight": weight,
            "distance_to_animal_cm": distance,
        }
        for event_id in [*events["event_id"], 100 + 5]  # 105 is not selected
        for weight, distance in ((1.0, 10.0), (3.0, 30.0))
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "e.parquet")

    bins, paths = builder.content_mass_along_arms(events, tmp_path, 1)

    assert [path.name for path in paths] == ["e.parquet"]
    n_events = events.groupby(["animal", "is_swr"]).size()
    by_kind = bins.set_index(["animal", "is_swr"])
    # one place and one side, so one row per animal and kind
    assert len(bins) == len(n_events)
    for key, n in n_events.items():
        assert by_kind.loc[key, "weight"] == 4.0 * n
        assert by_kind.loc[key, "distance_to_animal_cm"] == pytest.approx(25.0)
    with pytest.raises(ValueError, match="do not cover"):
        builder.content_mass_along_arms(
            events.iloc[:-1].assign(
                event_id=lambda t: t["event_id"].where(t.index > 0, 999)
            ),
            tmp_path,
            1,
        )


def test_the_content_mass_run_is_read_once_for_the_selected_events(tmp_path):
    """Every event of the run has, on arm 2 (patch 2), 1 time bin at a remote
    well, 1 at its own well and 2 far off the wells. The selected events get
    shares 0.5, 0.25 and 0.25 (named apart from the table's own at_well), one
    well and one patch; the content by patch is one row per kind, animal and
    distance to the well; the unselected event is not read."""
    events = select_visit_context(_labeled()).assign(current_patch=1)
    rows = [
        {
            "event_id": event_id,
            "epoch_id": "e",
            "arm": 2,
            "arm_length_cm": 100.0,
            "place_bin": place_bin,
            "n_place_bins": 50,
            "distance_to_well_cm": d_well,
            "far": far,
            "weight": weight,
            "distance_to_animal_cm": 150.0 if far else 5.0,
        }
        for event_id in [*events["event_id"], 100 + 5]  # 105 is not selected
        for far, weight, d_well, place_bin in (
            (True, 1.0, 5.0, 0),
            (False, 1.0, 5.0, 0),
            (True, 2.0, 60.0, 30),
        )
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "e.parquet")

    tables, paths = builder.content_mass_tables(events, tmp_path, 1)

    assert [path.name for path in paths] == ["e.parquet"]
    joined = tables["events"]
    assert joined["event_id"].tolist() == events["event_id"].tolist()
    assert (joined["content_at_well"] == 0.5).all()
    assert (joined["content_remote_well"] == 0.25).all()
    assert (joined["content_own_well"] == 0.25).all()
    # half of each event's content time is at arm 2's well: one well, one patch
    assert (joined["content_n_wells"] == 1).all()
    assert (joined["content_n_patches"] == 1).all()
    assert (joined["epoch_id"] == "e").all()
    along, _ = builder.content_mass_along_arms(events, tmp_path, 1)
    pd.testing.assert_frame_equal(tables["along"], along)
    by_patch = tables["by_patch"]
    assert set(by_patch["current_patch"]) == {1}
    n = events.groupby(["animal", "is_swr"]).size()
    at_well = by_patch[by_patch["distance_to_well_cm"] == 5.0].set_index(
        ["animal", "is_swr"]
    )["weight"]
    assert (at_well == 2.0 * n).all()
    assert by_patch["weight"].sum() == 4.0 * len(events)
    with pytest.raises(ValueError, match="no current patch"):
        builder.content_mass_tables(events.assign(current_patch=np.nan), tmp_path, 1)
    with pytest.raises(ValueError, match="no row"):
        builder.content_mass_tables(
            pd.concat([events, events.iloc[[0]].assign(event_id=999)]), tmp_path, 1
        )


def test_each_selected_event_gets_where_its_content_starts_and_ends(tmp_path):
    """Every event's content is 5 cm from the animal at its first time bin and
    at a well 150 cm away at its last."""
    events = select_visit_context(_labeled())
    rows = [
        {
            "event_id": event_id,
            "epoch_id": "e",
            "time": time,
            "distance_to_animal_cm": d_animal,
            "distance_to_well_cm": d_well,
        }
        for event_id in [*events["event_id"], 100 + 5]  # 105 is not selected
        for time, d_animal, d_well in ((0.0, 5.0, 5.0), (0.002, 150.0, 4.0))
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "e.parquet")

    joined, _ = builder.with_event_start_end(events, tmp_path, 1)

    assert len(joined) == len(events)
    assert joined["starts_near_animal"].all() and joined["ends_at_remote_well"].all()
    assert not joined["ends_near_animal"].any()


def test_gamma_around_event_starts_is_read_for_the_selected_events(tmp_path):
    """One epoch whose slow gamma is the sample number and fast gamma its
    negative; every selected event starts at sample 200, so each kind's median
    trace is the samples 125 to 275 (0.3 s at 500 Hz)."""
    events = select_visit_context(_labeled())
    events = events.assign(
        animal="a", session_date=20200101, epoch="02_r1", event_start_s=200 * 0.002
    )
    (epoch_id,) = set(swr_source.event_epoch_ids(events))
    sample = np.arange(400.0)
    pd.DataFrame(
        {"time": sample * 0.002, "slow_gamma_z": sample, "fast_gamma_z": -sample}
    ).to_parquet(tmp_path / f"{epoch_id}_gamma_z.parquet")

    curves, paths = builder.gamma_triggered_curves(events, tmp_path, 1)

    assert [path.name for path in paths] == [f"{epoch_id}_gamma_z.parquet"]
    np.testing.assert_allclose(
        curves["slow_gamma"]["time"][[0, 75, -1]], [-0.15, 0, 0.15]
    )
    for kind in ("overlap", "no_overlap"):
        np.testing.assert_allclose(
            curves["slow_gamma"][f"{kind}_median"], np.arange(125.0, 276.0)
        )
        np.testing.assert_allclose(
            curves["fast_gamma"][f"{kind}_q75"], -np.arange(125.0, 276.0)
        )
    with pytest.raises(ValueError, match="not 2"):
        builder.gamma_triggered_curves(events, tmp_path, 2)
    with pytest.raises(ValueError, match="no gamma for an event's epoch"):
        builder.gamma_triggered_curves(events.assign(epoch="04_r2"), tmp_path, 1)


def test_an_output_not_made_over_the_subset_is_pending():
    assert builder.subset_staleness({"subset": ANALYZED_SUBSET}, "out") is None
    assert "rerun it" in builder.subset_staleness({}, "out")
    assert "rerun it" in builder.subset_staleness({"subset": "every event"}, "out")


def test_the_spike_evidence_left_out_are_counted_per_group():
    metrics = pd.DataFrame(
        {"group": ["baseline", "baseline", "SWR-remote", "nonSWR-close", "SWR-remote"]}
    )
    kept = np.array([True, False, False, True, True])
    values = _by_key(builder.spike_evidence_visit_values(metrics, kept, "s"))
    assert values["baseline_left_out"].value == 1
    assert values["swr_remote_left_out"].value == 1
    assert values["non_swr_close_left_out"].value == 0
    assert values["non_swr_remote_left_out"].value == 0
    assert "quiet windows" in values["baseline_left_out"].description


def test_the_held_out_bouts_read_and_left_out_are_the_split_s_counts():
    bouts = pd.DataFrame(
        {
            "state": [
                "Non-Local Continuous | SWR",
                "Non-Local Continuous | no SWR",
                "Non-Local Continuous | no SWR",
            ],
            "regime": ["immobile", "immobile", "running"],
            "n_bouts": [10, 20, 99],
            "n_kept": [9, 15, 1],
            "n_left_out": [1, 5, 98],
            "n_kept_matched_outside_visit": [2, 3, 0],
        }
    )
    values = _by_key(builder.heldout_bout_values(bouts, "s"))
    assert values["heldout_swr_bouts_in_subset"].value == 9
    assert values["heldout_swr_bouts_left_out"].value == 1
    assert values["heldout_non_swr_bouts_in_subset"].value == 15
    assert values["heldout_non_swr_bouts_left_out"].value == 5
    assert values["heldout_non_swr_bouts_matched_outside_visit"].value == 3
