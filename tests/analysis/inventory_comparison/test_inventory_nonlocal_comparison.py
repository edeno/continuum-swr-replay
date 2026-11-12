"""The other detectors' events against the non-local events, on a run small enough
to check by hand."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison import (
    nonlocal_comparison as comparison,
)
from continuum_swr_replay.analysis.inventory_comparison.features import (
    STATE_SHARE_COLUMNS,
)

LOCAL, NO_SPIKE, NON_LOCAL, UNCERTAIN = STATE_SHARE_COLUMNS
NAN = np.nan


def _events() -> pd.DataFrame:
    """Kay events as (animal, matched, local, no-spike, non-local, uncertain).

    Animal a has 3 of 5 events overlapping no non-local event, animal b 4 of 8.
    """
    rows = [
        # overlapping no non-local event: three mostly Local, one mostly
        # No-Spike, two neither (one of them exactly half Local), one unknown
        ("a", False, 0.8, 0.1, 0.0, 0.1),
        ("a", False, 0.9, 0.0, 0.0, 0.1),
        ("a", False, 0.2, 0.6, 0.0, 0.2),
        ("b", False, 0.6, 0.2, 0.0, 0.2),
        ("b", False, 0.4, 0.4, 0.0, 0.2),
        ("b", False, 0.5, 0.5, 0.0, 0.0),
        ("b", False, NAN, NAN, NAN, NAN),
        # overlapping a non-local event: five with state shares, one without
        ("a", True, 0.0, 0.1, 0.9, 0.0),
        ("a", True, 0.2, 0.1, 0.5, 0.2),
        ("b", True, 0.05, 0.4, 0.35, 0.2),
        ("b", True, 0.0, 0.0, 1.0, 0.0),
        ("b", True, 0.1, 0.3, 0.3, 0.3),
        ("b", True, NAN, NAN, NAN, NAN),
    ]
    frame = pd.DataFrame(
        rows, columns=["animal", "matched", LOCAL, NO_SPIKE, NON_LOCAL, UNCERTAIN]
    )
    return frame.assign(detector="kay")


def test_other_detector_statistics_split_events_by_overlap_and_state():
    result = comparison.other_detector_statistics(_events()).loc["kay"]
    assert result["n_events"] == 13
    assert result["no_nonlocal_fraction"] == pytest.approx(7 / 13)
    # a: 3 of 5; b: 4 of 8
    assert result["no_nonlocal_fraction_animal_min"] == pytest.approx(0.5)
    assert result["no_nonlocal_fraction_animal_max"] == pytest.approx(0.6)
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


def _write_run(run_dir, report_dir) -> None:
    """One ok epoch: non-local events [0, 1] and [2, 3]; Kay covers [0.5, 1] and
    [10, 11]; Yu covers [0, 0.25]; MUA covers [2, 2.25]; the Widloski detector
    covers [2.75, 3]. Two epochs that are not ok each hold a non-local event of
    5 s that no Kay SWR overlaps."""
    run_dir.mkdir()
    report_dir.mkdir()
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
    pd.DataFrame(
        {"detector": ["state_space", "kay"], "exposure_s": [7200.0, 100.0]}
    ).to_csv(report_dir / "counts_by_animal.csv", index=False)
    pd.DataFrame(
        {
            "cell": ["sequence_matched_by_no_trigger", "another_cell"],
            "detector": "state_space",
            "n_events": [10, 99],
            "n_unmatched": [4, 99],
            "n_unknown": [2, 99],
        }
    ).to_csv(report_dir / "cells_by_animal.csv", index=False)


def test_left_out_seconds_subtract_each_covering_from_the_nonlocal_time(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "report")
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
    _write_run(tmp_path / "run", tmp_path / "report")
    (tmp_path / "run" / "a_features.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="a_features.parquet"):
        comparison.left_out_seconds(tmp_path / "run")


def test_left_out_statistics_count_events_time_and_rates(tmp_path):
    _write_run(tmp_path / "run", tmp_path / "report")
    result = comparison.left_out_statistics(tmp_path / "run", tmp_path / "report")
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
