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
    # of the six events with state shares that overlap no non-local event (the
    # seventh has none and is left out)
    assert result["no_nonlocal_mostly_local_fraction"] == pytest.approx(3 / 6)
    assert result["no_nonlocal_mostly_no_spike_fraction"] == pytest.approx(1 / 6)
    assert result["no_nonlocal_mixed_fraction"] == pytest.approx(2 / 6)
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
        comparison.left_out_seconds(tmp_path / "run")


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


def _inventory_rows(
    epoch: str, state_space_available: bool, exposure_s: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One epoch's observations and counts in the inventory run: the non-local
    detector's and Kay's (100 s of exposure)."""
    session = {"animal": "a", "day": 1, "epoch": epoch}
    observations = pd.DataFrame(
        {
            **session,
            "brain_area": "HPC",
            "detector": ["state_space", "kay"],
            "run_id": "r",
            "start_s": 0.0,
            "end_s": 100.0,
            "available": [state_space_available, True],
            "reason": "",
        }
    )
    counts = pd.DataFrame(
        {
            **session,
            "detector": ["state_space", "kay"],
            "run_id": "r",
            "n_native": 1,
            "n_set_aside": 0,
            "n_moving": 0,
            "n_eligible": 1,
            "exposure_s": [exposure_s, 100.0],
            "n_in_exposure": 1,
            "rate_per_s": 0.0,
            "native_note": "",
        }
    )
    return observations, counts


def _write_inventory_run(inventory_dir) -> None:
    """Two ok epochs. In ``a`` the non-local detector has 7200 s of exposure and
    its no-trigger cell 10 events: 4 unmatched, 2 unknown, 4 matched; another
    cell has 3 unmatched events. In ``d`` it was unavailable, so its 3600 s and
    its unmatched event are not summed. A failed epoch and one the manifest does
    not know hold 1800 s and an unmatched event each."""
    inventory_dir.mkdir()
    (inventory_dir / "manifest.json").write_text(
        json.dumps(
            {
                "epochs": {
                    "a": {"status": "ok"},
                    "d": {"status": "ok"},
                    "b": {"status": "failed"},
                },
                "settings": {
                    "cells": {
                        "sequence_matched_by_no_trigger": {"sources": ["state_space"]},
                        "another_cell": {"sources": ["state_space"]},
                    }
                },
            }
        )
    )

    def cells(epoch: str, memberships: dict[str, list[str]]) -> pd.DataFrame:
        rows = [
            {"cell": cell, "membership": m}
            for cell, listed in memberships.items()
            for m in listed
        ]
        return pd.DataFrame(rows).assign(
            animal="a",
            day=1,
            epoch=epoch,
            detector="state_space",
            event_key=[f"{epoch}{i}" for i in range(len(rows))],
            n_against=3,
        )

    no_trigger = "sequence_matched_by_no_trigger"
    for epoch, available, exposure, memberships in (
        (
            "a",
            True,
            7200.0,
            {
                no_trigger: ["unmatched"] * 4 + ["unknown"] * 2 + ["matched"] * 4,
                "another_cell": ["unmatched"] * 3,
            },
        ),
        ("d", False, 3600.0, {no_trigger: ["unmatched"]}),
        ("b", True, 1800.0, {no_trigger: ["unmatched"]}),
        ("c", True, 1800.0, {no_trigger: ["unmatched"]}),
    ):
        observations, counts = _inventory_rows(epoch, available, exposure)
        observations.to_parquet(inventory_dir / f"{epoch}_observations.parquet")
        counts.to_parquet(inventory_dir / f"{epoch}_counts.parquet")
        cells(epoch, memberships).to_parquet(inventory_dir / f"{epoch}_cells.parquet")


def _write_run(run_dir, inventory_dir) -> None:
    """One ok epoch: non-local events [0, 1] and [2, 3]; Kay covers [0.5, 1] and
    [10, 11]; Yu covers [0, 0.25]; MUA covers [2, 2.25]; the Widloski detector
    covers [2.75, 3]. Two epochs that are not ok each hold a non-local event of
    5 s that no Kay SWR overlaps. The inventory comparison run it read is
    :func:`_write_inventory_run`'s."""
    run_dir.mkdir()
    _write_inventory_run(inventory_dir)
    _write_manifest(run_dir)
    _write_unread_epochs(
        run_dir,
        {
            "features": pd.DataFrame(
                {"detector": ["state_space"], "event_start_s": 0.0, "event_end_s": 5.0}
            ),
            "pair_membership": pd.DataFrame(
                {
                    "detector": ["state_space"],
                    "other_detector": "kay",
                    "membership": "unmatched",
                }
            ),
        },
    )
    features = pd.DataFrame(
        {
            "detector": [
                "state_space",
                "state_space",
                "kay",
                "kay",
                "yu",
                "mua",
                "memoryless",
            ],
            "event_start_s": [0.0, 2.0, 0.5, 10.0, 0.0, 2.0, 2.75],
            "event_end_s": [1.0, 3.0, 1.0, 11.0, 0.25, 2.25, 3.0],
        }
    )
    features.to_parquet(run_dir / "a_features.parquet")
    membership = pd.DataFrame(
        {
            "detector": "state_space",
            "other_detector": ["kay", "kay", "kay", *["memoryless"] * 3],
            "membership": [
                "matched",
                "unmatched",
                "unknown",
                "unmatched",
                "unmatched",
                "matched",
            ],
        }
    )
    membership.to_parquet(run_dir / "a_pair_membership.parquet")


def test_left_out_seconds_subtract_each_covering_from_the_nonlocal_time(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "inventory")
    seconds = comparison.left_out_seconds(tmp_path / "run")
    assert seconds == {
        "nonlocal": 2.0,
        "kay": 1.5,
        "outside_kay": 1.5,
        "outside_swr": 1.25,
        "outside_trigger": 1.0,
        "outside_widloski": 1.75,
    }


def test_left_out_seconds_refuse_an_ok_epoch_without_its_features_table(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "inventory")
    (tmp_path / "run" / "a_features.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="a_features.parquet"):
        comparison.left_out_seconds(tmp_path / "run")


def test_left_out_statistics_count_events_time_and_rates(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "inventory")
    result = comparison.left_out_statistics(tmp_path / "run", tmp_path / "inventory")
    # the non-local detector's exposure, not Kay's
    assert result["rest_hours"] == 2.0
    assert result["nonlocal_time_s"] == 2.0
    assert result["kay_time_s"] == 1.5
    assert result["nonlocal_time_outside_kay_s"] == 1.5
    assert result["nonlocal_time_outside_kay_fraction"] == 0.75
    assert result["nonlocal_time_outside_swr_s"] == 1.25
    assert result["nonlocal_time_outside_swr_fraction"] == 0.625
    assert result["nonlocal_time_outside_trigger_s"] == 1.0
    assert result["nonlocal_time_outside_trigger_fraction"] == 0.5
    assert result["nonlocal_time_outside_widloski_s"] == 1.75
    assert result["nonlocal_time_outside_widloski_fraction"] == 0.875
    # one of the two events with a known membership against Kay is unmatched
    assert result["nonlocal_events_outside_kay"] == 1
    assert result["nonlocal_events_outside_kay_fraction"] == 0.5
    assert result["nonlocal_events_outside_kay_per_hour"] == 0.5
    # two of the three against the Widloski detector
    assert result["nonlocal_events_outside_widloski"] == 2
    assert result["nonlocal_events_outside_widloski_fraction"] == pytest.approx(2 / 3)
    assert result["nonlocal_events_outside_widloski_per_hour"] == 1.0
    # the comparison's cell: 4 unmatched of the 10 - 2 with a known membership
    assert result["nonlocal_events_outside_trigger"] == 4
    assert result["nonlocal_events_outside_trigger_fraction"] == 0.5
    assert result["nonlocal_events_outside_trigger_per_hour"] == 2.0


def test_the_inventory_tables_pool_the_ok_epochs_where_a_detector_was_available(
    tmp_path,
):
    _write_inventory_run(tmp_path / "inventory")
    pooled = comparison.inventory_animal_tables(tmp_path / "inventory")
    counts = pooled["counts_by_animal"].set_index("detector")
    # epoch d's 3600 s are left out (the detector was unavailable); b and c are
    # not ok epochs. Kay was available in a and d.
    assert counts.loc["state_space", "exposure_s"] == 7200.0
    assert counts.loc["state_space", "n_sessions"] == 1
    assert counts.loc["state_space", "unavailable_sessions"] == "a1_d"
    assert counts.loc["kay", "exposure_s"] == 200.0
    cells = pooled["cells_by_animal"].set_index("cell")
    no_trigger = cells.loc["sequence_matched_by_no_trigger"]
    assert no_trigger[["n_events", "n_unmatched", "n_unknown"]].tolist() == [10, 4, 2]
    assert cells.loc["another_cell", "n_unmatched"] == 3


def test_an_inventory_run_with_no_ok_epoch_is_refused(tmp_path):
    _write_inventory_run(tmp_path / "inventory")
    manifest = tmp_path / "inventory" / "manifest.json"
    record = json.loads(manifest.read_text())
    record["epochs"] = {"a": {"status": "failed"}}
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="marks no epoch ok"):
        comparison.inventory_animal_tables(tmp_path / "inventory")


def test_nonlocal_events_overlapped_pools_animals_and_keeps_each_animals_share():
    directional = pd.DataFrame(
        {
            "animal": ["a", "b", "a", "b", "a"],
            "source_detector": ["state_space"] * 4 + ["kay"],
            "target_detector": ["kay", "kay", "mua", "mua", "state_space"],
            "n_eligible": [10, 30, 10, 0, 5],
            "n_matched": [5, 15, 2, 0, 5],
        }
    )
    overlap = comparison.nonlocal_events_overlapped(directional, ("kay", "mua"))
    assert list(overlap.index) == ["kay", "mua"]
    assert overlap.loc["kay", "pooled"] == pytest.approx(20 / 40)
    np.testing.assert_allclose(overlap.loc["kay", "animals"], [0.5, 0.5])
    assert overlap.loc["mua", "pooled"] == pytest.approx(0.2)
    # an animal with no eligible event has no share, not a division by zero
    assert np.isnan(overlap.loc["mua", "animals"][1])
