"""The other detectors' events against the non-local events, on a run small enough
to check by hand."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detector_comparison import (
    nonlocal_comparison as comparison,
)
from continuum_swr_replay.analysis.detector_comparison.features import (
    STATE_SHARE_COLUMNS,
)

LOCAL, NO_SPIKE, NON_LOCAL, UNCERTAIN = STATE_SHARE_COLUMNS
NAN = np.nan


def _events() -> pd.DataFrame:
    """Kay events as (animal, epoch, matched, local, no-spike, non-local,
    uncertain).

    Animal a has 3 of 5 events overlapping no non-local event: all 3 of epoch
    e1's and none of e2's 2. Animal b has 4 of 8: 4 of e1's 5 and none of e2's
    3.
    """
    rows = [
        # overlapping no non-local event: three mostly Local, one mostly
        # No-Spike, two neither (one of them exactly half Local), one unknown
        ("a", "e1", False, 0.8, 0.1, 0.0, 0.1),
        ("a", "e1", False, 0.9, 0.0, 0.0, 0.1),
        ("a", "e1", False, 0.2, 0.6, 0.0, 0.2),
        ("b", "e1", False, 0.6, 0.2, 0.0, 0.2),
        ("b", "e1", False, 0.4, 0.4, 0.0, 0.2),
        ("b", "e1", False, 0.5, 0.5, 0.0, 0.0),
        ("b", "e1", False, NAN, NAN, NAN, NAN),
        # overlapping a non-local event: five with state shares, one without
        ("a", "e2", True, 0.0, 0.1, 0.9, 0.0),
        ("a", "e2", True, 0.2, 0.1, 0.5, 0.2),
        ("b", "e1", True, 0.05, 0.4, 0.35, 0.2),
        ("b", "e2", True, 0.0, 0.0, 1.0, 0.0),
        ("b", "e2", True, 0.1, 0.3, 0.3, 0.3),
        ("b", "e2", True, NAN, NAN, NAN, NAN),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["animal", "epoch", "matched", LOCAL, NO_SPIKE, NON_LOCAL, UNCERTAIN],
    )
    return frame.assign(detector="kay", day="d1")


def test_other_detector_statistics_split_events_by_overlap_and_state():
    result = comparison.other_detector_statistics(_events()).loc["kay"]
    assert result["n_events"] == 13
    assert result["no_nonlocal_fraction"] == pytest.approx(7 / 13)
    # each animal's share is the mean of its epochs': a, 1 and 0; b, 0.8 and 0
    # (pooled, a's 3 of 5 and b's 4 of 8 would give 0.6 and 0.5)
    assert result["no_nonlocal_fraction_animal_min"] == pytest.approx(0.4)
    assert result["no_nonlocal_fraction_animal_max"] == pytest.approx(0.5)
    # Figure 2c's bars, the mean over animals: no non-local (0.5 + 0.4) / 2;
    # mostly Local, a's (2/3 + 0) / 2 and b's (1/5 + 0) / 2
    assert result["no_nonlocal_fraction_animal_mean"] == pytest.approx(0.45)
    assert result["mostly_local_fraction_animal_mean"] == pytest.approx(
        (1 / 3 + 0.1) / 2
    )
    # of the six events with state shares that overlap no non-local event (the
    # seventh has none and is left out)
    assert result["no_nonlocal_mostly_local_fraction"] == pytest.approx(3 / 6)
    assert result["no_nonlocal_mostly_no_spike_fraction"] == pytest.approx(1 / 6)
    assert result["no_nonlocal_mixed_fraction"] == pytest.approx(2 / 6)
    # the same three mostly Local events as a share of all 13 compared events
    assert result["mostly_local_fraction"] == pytest.approx(3 / 13)
    # of the five events with state shares that overlap a non-local event:
    # Local 0, 0, 0.05, 0.1, 0.2; No-Spike 0, 0.1, 0.1, 0.3, 0.4;
    # non-local 0.3, 0.35, 0.5, 0.9, 1; uncertain 0, 0, 0.2, 0.2, 0.3
    assert result["overlap_local_share_median"] == pytest.approx(0.05)
    assert result["overlap_no_spike_share_median"] == pytest.approx(0.1)
    assert result["overlap_nonlocal_share_median"] == pytest.approx(0.5)
    assert result["overlap_uncertain_share_median"] == pytest.approx(0.2)
    assert result["overlap_any_local_fraction"] == pytest.approx(3 / 5)
    assert result["overlap_any_no_spike_fraction"] == pytest.approx(4 / 5)
    assert result["overlap_all_nonlocal_fraction"] == pytest.approx(1 / 5)


def _write_manifest(run_dir) -> None:
    """A manifest marking epoch ``a`` ok and ``b`` failed; it does not know ``c``."""
    (run_dir / "manifest.json").write_text(
        json.dumps({"epochs": {"a": {"status": "ok"}, "b": {"status": "failed"}}})
    )


def _write_unread_epochs(run_dir, tables: dict[str, pd.DataFrame]) -> None:
    """The tables, as epochs ``b`` (failed) and ``c`` (not in the manifest)."""
    for identifier in ("b", "c"):
        for name, table in tables.items():
            table.to_parquet(run_dir / f"{identifier}_{name}.parquet")


def _write_memberships(run_dir) -> None:
    """One ok epoch's state shares and memberships: Kay events k1 (overlaps a
    non-local event), k2 (overlaps none) and k3 (unknown), and MUA event m1
    (overlaps one). Two epochs that are not ok hold Kay event k9."""
    run_dir.mkdir()
    _write_manifest(run_dir)
    unread = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "04_r2",
            "detector": ["kay"],
            "event_key": ["k9"],
        }
    )
    _write_unread_epochs(
        run_dir,
        {
            "state_shares": unread.assign(
                **{LOCAL: 1.0, NO_SPIKE: 0.0, NON_LOCAL: 0.0, UNCERTAIN: 0.0}
            ),
            "pair_membership": unread.assign(
                other_detector="state_space", membership="unmatched"
            ),
        },
    )
    keys = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "02_r1",
            "detector": ["kay", "kay", "kay", "mua"],
            "event_key": ["k1", "k2", "k3", "m1"],
        }
    )
    shares = keys.assign(
        **{LOCAL: [0.0, 0.7, 0.2, 0.1], NO_SPIKE: 0.1, NON_LOCAL: 0.1, UNCERTAIN: 0.1}
    )
    shares.to_parquet(run_dir / "a_state_shares.parquet")
    against_nonlocal = keys.assign(
        other_detector="state_space",
        membership=["matched", "unmatched", "unknown", "matched"],
    )
    # the same events against another inventory, which is not what is asked
    against_mua = keys.iloc[:3].assign(other_detector="mua", membership="unmatched")
    pd.concat([against_nonlocal, against_mua]).to_parquet(
        run_dir / "a_pair_membership.parquet"
    )


def test_other_detector_events_keep_the_known_memberships_against_the_nonlocal_events(
    tmp_path,
):
    _write_memberships(tmp_path / "run")
    events = comparison.other_detector_events(tmp_path / "run")
    assert list(events["event_key"]) == ["k1", "k2", "m1"]
    assert list(events["matched"]) == [True, False, True]
    assert list(events[LOCAL]) == [0.0, 0.7, 0.1]


def test_a_run_without_a_readable_manifest_is_refused(tmp_path):
    _write_memberships(tmp_path / "run")
    manifest = tmp_path / "run" / "manifest.json"
    for text in ("{not json", '{"settings": {}}'):
        manifest.write_text(text)
        with pytest.raises(ValueError, match="no readable run manifest"):
            comparison.other_detector_events(tmp_path / "run")
    manifest.unlink()
    with pytest.raises(ValueError, match="no readable run manifest"):
        comparison.other_detector_events(tmp_path / "run")
    with pytest.raises(ValueError, match="no readable run manifest"):
        comparison.left_out_seconds(tmp_path / "run", [], {})


def test_a_run_with_no_ok_epoch_is_refused(tmp_path):
    _write_memberships(tmp_path / "run")
    (tmp_path / "run" / "manifest.json").write_text(
        json.dumps({"epochs": {"a": {"status": "failed"}}})
    )
    with pytest.raises(ValueError, match="marks no epoch ok"):
        comparison.other_detector_events(tmp_path / "run")


def test_check_against_directional_refuses_other_counts(tmp_path):
    _write_memberships(tmp_path / "run")
    events = comparison.other_detector_events(tmp_path / "run")
    directional = pd.DataFrame(
        {
            "source_detector": ["kay", "kay", "mua", "kay"],
            "target_detector": ["state_space", "state_space", "state_space", "mua"],
            "n_eligible": [1, 1, 1, 50],
            "n_matched": [1, 0, 1, 50],
        }
    )
    comparison.check_against_directional(events, directional)
    for column in ("n_eligible", "n_matched"):
        other = directional.copy()
        other.loc[1, column] += 1
        with pytest.raises(ValueError, match="do not give the directional counts"):
            comparison.check_against_directional(events, other)
    # a detector the comparison counts and the memberships do not hold
    with pytest.raises(ValueError, match="do not give the directional counts"):
        comparison.check_against_directional(
            events[events["detector"] == "kay"], directional
        )


def _key(detector: str, epoch: str, native: object) -> str:
    return f"{detector}:a1_{epoch}:{native}"


#: The selected non-local events of epoch a: 1 and 2 (and 0, 3 to 8, which only
#: the inventory run's cells hold). Event 99 starts outside every visit.
SELECTED = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#: Epoch a's one well visit; the run's other epochs have none.
VISITS = {"a": (np.array([0.0]), np.array([5.0]))}


def _write_inventory_run(inventory_dir) -> None:
    """Two ok epochs. In ``a`` the non-local detector observed the epoch and its
    no-trigger cell holds the selected events 0 to 8 (0 to 3 unmatched, 4 and 5
    unknown, 6 to 8 matched) and the unselected 99 (unmatched); another cell
    holds 0 to 2 (unmatched). In ``d`` it observed nothing, so its unmatched
    event 0 is not counted. A failed epoch and one the manifest does not know
    hold an unmatched event 0 each."""
    inventory_dir.mkdir()
    (inventory_dir / "manifest.json").write_text(
        json.dumps(
            {
                "epochs": {
                    "a": {"status": "ok"},
                    "d": {"status": "ok"},
                    "b": {"status": "failed"},
                }
            }
        )
    )
    no_trigger = "sequence_matched_by_no_trigger"
    for epoch, available, memberships in (
        (
            "a",
            True,
            {
                no_trigger: ["unmatched"] * 4
                + ["unknown"] * 2
                + ["matched"] * 3
                + ["unmatched"],
                "another_cell": ["unmatched"] * 3,
            },
        ),
        ("d", False, {no_trigger: ["unmatched"]}),
        ("b", True, {no_trigger: ["unmatched"]}),
        ("c", True, {no_trigger: ["unmatched"]}),
    ):
        session = {"animal": "a", "day": 1, "epoch": epoch}
        pd.DataFrame(
            {
                **session,
                "brain_area": "HPC",
                "detector": ["state_space", "kay"],
                "run_id": "r",
                "start_s": 0.0,
                "end_s": 100.0,
                "available": [available, True],
                "reason": "",
            }
        ).to_parquet(inventory_dir / f"{epoch}_observations.parquet")
        rows = [
            {"cell": cell, "membership": m, "native": [*range(9), 99][i]}
            for cell, listed in memberships.items()
            for i, m in enumerate(listed)
        ]
        pd.DataFrame(rows).assign(
            **session,
            detector="state_space",
            event_key=[_key("state_space", epoch, row["native"]) for row in rows],
            n_against=3,
        ).drop(columns="native").to_parquet(inventory_dir / f"{epoch}_cells.parquet")


def _write_run(run_dir, inventory_dir) -> None:
    """One ok epoch, a, with one visit from 0 to 5 s: the selected non-local
    events 1 [0, 1] and 2 [2, 3], and 99 [20, 25], which starts after the
    visit; Kay covers [0.5, 1] and [10, 11] (which starts after the visit); Yu
    covers [0, 0.25]; MUA covers [2, 2.25]; the Widloski detector covers
    [2.75, 3]. Two epochs that are not ok each hold a non-local event of 5 s
    that no Kay SWR overlaps. The inventory comparison run it read is
    :func:`_write_inventory_run`'s."""
    run_dir.mkdir()
    _write_inventory_run(inventory_dir)
    _write_manifest(run_dir)
    _write_unread_epochs(
        run_dir,
        {
            "features": pd.DataFrame(
                {
                    "detector": ["state_space"],
                    "event_key": _key("state_space", "b", 1),
                    "event_start_s": 0.0,
                    "event_end_s": 5.0,
                }
            ),
            "pair_membership": pd.DataFrame(
                {
                    "detector": ["state_space"],
                    "event_key": _key("state_space", "b", 1),
                    "other_detector": "kay",
                    "membership": "unmatched",
                }
            ),
        },
    )
    detectors = ["state_space"] * 3 + ["kay", "kay", "yu", "mua", "memoryless"]
    natives = [1, 2, 99, 1, 2, 1, 1, 1]
    features = pd.DataFrame(
        {
            "detector": detectors,
            "event_key": [
                _key(d, "a", n) for d, n in zip(detectors, natives, strict=True)
            ],
            "event_start_s": [0.0, 2.0, 20.0, 0.5, 10.0, 0.0, 2.0, 2.75],
            "event_end_s": [1.0, 3.0, 25.0, 1.0, 11.0, 0.25, 2.25, 3.0],
        }
    )
    features.to_parquet(run_dir / "a_features.parquet")
    # against Kay: 1 matched, 2 unmatched, 99 unmatched; against the Widloski
    # detector: 1 and 2 unmatched, 99 matched; 1 is unknown against Kay too
    membership = pd.DataFrame(
        {
            "detector": "state_space",
            "event_key": [_key("state_space", "a", n) for n in (1, 2, 99, 1, 1, 2, 99)],
            "other_detector": ["kay", "kay", "kay", "kay", *["memoryless"] * 3],
            "membership": [
                "matched",
                "unmatched",
                "unmatched",
                "unknown",
                "unmatched",
                "unmatched",
                "matched",
            ],
        }
    )
    membership.to_parquet(run_dir / "a_pair_membership.parquet")


def _exposure() -> pd.DataFrame:
    """The non-local detector's exposure inside the visits: 3600 s in a; d's
    999 s are not summed (it observed nothing)."""
    return pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": ["a", "d"],
            "exposure_in_visit_s": [3600.0, 999.0],
        }
    )


def test_left_out_seconds_subtract_each_covering_from_the_selected_nonlocal_time(
    tmp_path,
):
    _write_run(tmp_path / "run", tmp_path / "inventory")
    seconds = comparison.left_out_seconds(tmp_path / "run", SELECTED, VISITS)
    # event 99's 5 s are not selected; the Kay SWR at 10 s starts after the
    # visit, so Kay's time is the first SWR's 0.5 s
    assert seconds == {
        "nonlocal": 2.0,
        "kay": 0.5,
        "outside_kay": 1.5,
        "outside_swr": 1.25,
        "outside_trigger": 1.0,
        "outside_widloski": 1.75,
    }


def test_left_out_seconds_refuse_an_ok_epoch_without_its_features_table_or_visits(
    tmp_path,
):
    _write_run(tmp_path / "run", tmp_path / "inventory")
    with pytest.raises(ValueError, match="no well visits for 1 epochs"):
        comparison.left_out_seconds(tmp_path / "run", SELECTED, {})
    (tmp_path / "run" / "a_features.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="a_features.parquet"):
        comparison.left_out_seconds(tmp_path / "run", SELECTED, VISITS)


def test_left_out_statistics_count_the_selected_events_time_and_rates(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "inventory")
    result = comparison.left_out_statistics(
        tmp_path / "run", tmp_path / "inventory", SELECTED, VISITS, _exposure()
    )
    # the non-local detector's exposure inside the visits, where it observed
    assert result["rest_hours"] == 1.0
    assert result["nonlocal_time_s"] == 2.0
    assert result["kay_time_s"] == 0.5
    assert result["nonlocal_time_outside_kay_s"] == 1.5
    assert result["nonlocal_time_outside_kay_fraction"] == 0.75
    assert result["nonlocal_time_outside_swr_s"] == 1.25
    assert result["nonlocal_time_outside_swr_fraction"] == 0.625
    assert result["nonlocal_time_outside_trigger_s"] == 1.0
    assert result["nonlocal_time_outside_trigger_fraction"] == 0.5
    assert result["nonlocal_time_outside_widloski_s"] == 1.75
    assert result["nonlocal_time_outside_widloski_fraction"] == 0.875
    # one of the two selected events with a known membership against Kay (99,
    # unmatched, is not selected)
    assert result["nonlocal_events_outside_kay"] == 1
    assert result["nonlocal_events_outside_kay_fraction"] == 0.5
    assert result["nonlocal_events_outside_kay_per_hour"] == 1.0
    # both against the Widloski detector (99, matched, is not selected)
    assert result["nonlocal_events_outside_widloski"] == 2
    assert result["nonlocal_events_outside_widloski_fraction"] == 1.0
    assert result["nonlocal_events_outside_widloski_per_hour"] == 2.0
    # the cell in a: 4 unmatched of the 9 - 2 selected with a known membership;
    # 99, d's, b's and c's events are not counted
    assert result["nonlocal_events_outside_trigger"] == 4
    assert result["nonlocal_events_outside_trigger_fraction"] == 4 / 7
    assert result["nonlocal_events_outside_trigger_per_hour"] == 4.0


def test_left_out_statistics_refuse_an_exposure_without_an_observed_session(
    tmp_path,
):
    _write_run(tmp_path / "run", tmp_path / "inventory")
    with pytest.raises(ValueError, match="lacks a session"):
        comparison.left_out_statistics(
            tmp_path / "run",
            tmp_path / "inventory",
            SELECTED,
            VISITS,
            _exposure().iloc[[1]],
        )


def test_an_inventory_run_with_no_ok_epoch_is_refused(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "inventory")
    manifest = tmp_path / "inventory" / "manifest.json"
    manifest.write_text(json.dumps({"epochs": {"a": {"status": "failed"}}}))
    with pytest.raises(ValueError, match="marks no epoch ok"):
        comparison.left_out_statistics(
            tmp_path / "run", tmp_path / "inventory", SELECTED, VISITS, _exposure()
        )


def test_the_exposure_inside_the_visits_is_the_stored_exposure_cut_to_the_visits():
    """Observed 0 to 100 s less an excluded event's span (10 to 12 s); still
    from 0 to 30 s and from 60 to 100 s; one visit from 20 to 70 s."""
    observations = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "02_r1",
            "brain_area": "HPC",
            "detector": ["state_space"],
            "run_id": "r",
            "start_s": 0.0,
            "end_s": 100.0,
            "available": True,
            "reason": "",
        }
    )
    set_aside = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "02_r1",
            "detector": ["state_space"],
            "interval_start_s": 10.0,
            "interval_end_s": 12.0,
        }
    )
    time = np.arange(0.0, 100.0 + 1e-9, 0.002)
    speed = np.where((time > 30.0) & (time < 60.0), 20.0, 1.0)
    exposure_s, in_visit_s = comparison.exposure_in_well_visits(
        observations, set_aside, time, speed, np.array([20.0]), np.array([70.0])
    )
    # still: [0, 30] and [60, 100] less [10, 12]: 68 s; in the visit: 10 + 10
    assert exposure_s == pytest.approx(68.0, abs=0.01)
    assert in_visit_s == pytest.approx(20.0, abs=0.01)


def test_nonlocal_events_overlapped_pools_events_and_averages_each_animals_epochs():
    # animal a: epoch 1 has 1 of 2 non-local events overlapped by Kay, epoch 2
    # has 1 of 1, so its share is the epochs' mean, 0.75, not the pooled 2/3
    memberships = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "b", "b", "a"],
            "day": [1, 1, 2, 1, 1, 1],
            "epoch": ["02_r1"] * 6,
            "other_detector": ["kay"] * 5 + ["mua"],
            "matched": [True, False, True, False, False, True],
        }
    )
    overlap = comparison.nonlocal_events_overlapped(memberships, ("kay", "mua"))
    assert list(overlap.index) == ["kay", "mua"]
    assert overlap.loc["kay", "pooled"] == pytest.approx(2 / 5)
    np.testing.assert_allclose(overlap.loc["kay", "animals"], [0.75, 0.0])
    # Figure 2e's bar: the mean of the animals' shares
    assert overlap.loc["kay", "animal_mean"] == pytest.approx(0.375)
    # an animal with no event against a detector has no dot
    np.testing.assert_allclose(overlap.loc["mua", "animals"], [1.0])


def test_events_overlapping_nonlocal_gives_the_content_classes_animal_shares():
    from continuum_swr_replay.analysis.detector_comparison.nonlocal_comparison import (
        content_class_animal_shares,
    )

    events = pd.DataFrame(
        {
            "detector": ["kay"] * 4,
            "animal": ["a", "a", "a", "b"],
            "day": ["d1", "d1", "d2", "d1"],
            "epoch": ["e1"] * 4,
            "matched": [True, False, True, False],
            LOCAL: [0.0, 0.9, 0.0, 0.2],
            NO_SPIKE: 0.0,
            NON_LOCAL: [1.0, 0.0, 1.0, 0.0],
            UNCERTAIN: 0.0,
        }
    )
    shares = comparison.events_overlapping_nonlocal(events, ("kay",))
    assert shares.loc["kay", "pooled"] == pytest.approx(2 / 4)
    np.testing.assert_allclose(
        shares.loc["kay", "animals"],
        content_class_animal_shares(events, "kay")["with_nonlocal"].to_numpy(),
    )


def test_nonlocal_event_memberships_keep_what_the_directional_table_counts(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)
    pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "02_r1",
            "detector": "state_space",
            "event_key": [f"state_space:a1_02_r1:{k}" for k in (11, 12, 13, 14)],
            "other_detector": "kay",
            "membership": ["matched", "unmatched", "unknown", "matched"],
            # n4 lies outside the two detectors' shared exposure
            "within_shared_exposure": [True, True, True, False],
        }
    ).to_parquet(run_dir / "a_pair_membership.parquet")
    directional = pd.DataFrame(
        {
            "source_detector": ["state_space"],
            "target_detector": ["kay"],
            "n_eligible": [2],
            "n_matched": [1],
        }
    )
    kept = comparison.nonlocal_event_memberships(run_dir, directional, [11, 12, 14])
    assert list(kept["event_key"]) == [
        "state_space:a1_02_r1:11",
        "state_space:a1_02_r1:12",
    ]
    assert list(kept["matched"]) == [True, False]
    # the counts are checked over every event; then the selected ones are kept
    # (event 11 starts outside a well visit here)
    kept = comparison.nonlocal_event_memberships(run_dir, directional, [12, 14])
    assert list(kept["matched"]) == [False]
    with pytest.raises(ValueError, match="directional counts"):
        comparison.nonlocal_event_memberships(
            run_dir, directional.assign(n_matched=2), [11, 12]
        )


def test_content_classes_and_their_animal_shares():
    from continuum_swr_replay.analysis.detector_comparison.nonlocal_comparison import (
        content_class_animal_shares,
        content_classes,
    )

    events = pd.DataFrame(
        {
            "detector": ["kay"] * 5 + ["yu"],
            "animal": ["a", "a", "a", "b", "b", "a"],
            "day": ["d1", "d1", "d2", "d1", "d1", "d1"],
            "epoch": ["e1"] * 6,
            "matched": [True, False, False, True, False, True],
            LOCAL: [0.0, 0.9, 0.2, 0.0, np.nan, 0.0],
            NO_SPIKE: [0.0, 0.1, 0.8, 0.0, np.nan, 0.0],
            NON_LOCAL: [1.0, 0.0, 0.0, 1.0, np.nan, 1.0],
            UNCERTAIN: [0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
        }
    )
    # an unmatched event without state shares is "other"
    assert content_classes(events).tolist() == [
        "with_nonlocal",
        "mostly_local",
        "other",
        "with_nonlocal",
        "other",
        "with_nonlocal",
    ]
    shares = content_class_animal_shares(events, "kay")
    # a: epoch (d1) 1/2 with, 1/2 mostly Local; epoch (d2) all other -> means
    assert shares.loc["a"].tolist() == pytest.approx([0.25, 0.25, 0.5])
    assert shares.loc["b"].tolist() == pytest.approx([0.5, 0.0, 0.5])
    # Figure 2c's bars: the means over animals, from the statistics the
    # events summary's kay_* values read (0.375, 0.125 and 0.5)
    means = comparison.content_class_means(
        comparison.other_detector_statistics(events).loc["kay"]
    )
    assert list(means.index) == list(comparison.CONTENT_CLASSES)
    assert means.tolist() == pytest.approx(shares.mean().tolist())
    assert means.tolist() == pytest.approx([0.375, 0.125, 0.5])


def _write_features(run_dir, starts: dict[str, float]) -> None:
    """The ok epoch ``a``'s features table: each event's start (s)."""
    pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "02_r1",
            "detector": ["kay" if key.startswith("k") else "mua" for key in starts],
            "event_key": list(starts),
            "event_start_s": list(starts.values()),
        }
    ).to_parquet(run_dir / "a_features.parquet")


def test_visit_starts_flag_the_events_whose_first_sample_is_inside_a_visit(tmp_path):
    _write_memberships(tmp_path / "run")
    # one visit, 10 s to 20 s: k1 starts at its first poke-in (inside), k2 at
    # its final poke-out (outside), m1 between two visits
    _write_features(tmp_path / "run", {"k1": 10.0, "k2": 20.0, "k3": 15.0, "m1": 25.0})
    visits = {"a": (np.array([10.0, 30.0]), np.array([20.0, 40.0]))}
    flags = comparison.visit_starts(tmp_path / "run", visits).set_index("event_key")
    assert flags["starts_in_well_visit"].to_dict() == {
        "k1": True,
        "k2": False,
        "k3": True,
        "m1": False,
    }
    with pytest.raises(ValueError, match="no well visits"):
        comparison.visit_starts(tmp_path / "run", {})


def test_read_comparison_checks_every_event_then_keeps_those_starting_in_a_visit(
    processed_data_root,
):
    run_dir = processed_data_root / "run"
    _write_memberships(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["settings"] = {"input_runs": {"inventory": "inv"}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    _write_features(run_dir, {"k1": 10.0, "k2": 25.0, "k3": 15.0, "m1": 12.0})
    report = comparison.inventory_comparison_root() / "inv" / "report"
    report.mkdir(parents=True)
    # the directional counts of every known event: k1 and k2 (one matched), m1
    pd.DataFrame(
        {
            "source_detector": ["kay", "mua"],
            "target_detector": ["state_space", "state_space"],
            "n_eligible": [2, 1],
            "n_matched": [1, 1],
        }
    ).to_csv(report / "directional_by_animal.csv", index=False)
    visits = {"a": (np.array([10.0]), np.array([20.0]))}

    read = comparison.read_comparison(run_dir, visits)

    # k2 starts after the visit's final poke-out: counted in the check, not kept
    assert list(read.others["event_key"]) == ["k1", "m1"]
    assert read.visit_counts.to_dict("index") == {
        "kay": {"n_in_visit": 1, "n_outside_visit": 1},
        "mua": {"n_in_visit": 1, "n_outside_visit": 0},
    }
    statistics = comparison.other_detector_statistics(read.others)
    assert statistics.loc["kay", "n_events"] == 1
    assert statistics.loc["kay", "no_nonlocal_fraction"] == 0.0
