"""The event features report: per-session values and animal pools.

The rules are checked on per-epoch tables built by hand with the runner's own
columns and dtypes: an animal pools its sessions by summing counts, never by
averaging their proportions; an ``unknown`` membership never enters an
``unmatched`` population; a missing feature is counted and left out of its
quantiles; every ``ms_*`` and memoryless row is marked pending; a run's old
baseline rows are dropped. The last tests write the report
of a small run on disk, into temporary folders.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.detector_comparison.report_event_features as report
from continuum_swr_replay.analysis.detection.detection_inputs import (
    memoryless_output_paths as memoryless_outputs,
)
from continuum_swr_replay.analysis.detector_comparison import (
    feature_run,
    inventory_run,
)
from continuum_swr_replay.analysis.detector_comparison.features import (
    DURATION_STRATUM_LABELS,
)
from continuum_swr_replay.analysis.detector_comparison.overlap import MATCH_DTYPES
from continuum_swr_replay.analysis.detector_comparison.selection import (
    REJECTION_DTYPES,
)
from continuum_swr_replay.paths import (
    event_features_root,
    inventory_comparison_root,
)
from continuum_swr_replay.paths import memoryless_root as memoryless_runs_root
from continuum_swr_replay.runs.io import output_path
from continuum_swr_replay.tables.io import write_table
from tests.support._swr_inventory_fixtures import under_root

DAY = 20200101
DETECTORS = ("kay", "state_space", "memoryless", "memoryless_criteria")
"""The detectors of the simulated sessions."""

SETTINGS = {
    "input_runs": {"inventory": "inventory01", "memoryless": "memoryless01"},
    "swr_source": "kay@run01+artifacts@run02",
    "duration_strata": {"labels": list(DURATION_STRATUM_LABELS)},
    "tracking_confidence": "unavailable: the decode frame carries none",
}

_FEATURE_DEFAULTS = {
    object: "",
    np.int64: 0,
    np.float64: 1.0,
    "Int64": 1,
    "UInt64": 1,
    "boolean": True,
    np.bool_: True,
}
_OBJECT_DEFAULTS = {
    "brain_area": "HPC",
    "run_id": "run01",
    "duration_stratum": "50-100 ms",
    "ss_disagreement": None,
    "matched_event_key": None,
    "occupancy_state": "at_well",
}
MEMORYLESS_COLUMNS = {
    "animal": object,
    "day": np.int64,
    "epoch": object,
    "ms_event_id": np.int64,
    "is_replay": np.bool_,
    "shuffle_pass": np.bool_,
    "shuffle_testable": np.bool_,
    "n_shuffles": np.int64,
    "artifact_drop_reason": object,
    "whole_event_speed_eligible": np.bool_,
}


def key(detector: str, epoch: str, i: int, animal: str = "sim") -> str:
    return f"{detector}:{animal}{DAY}_{epoch}:{i}"


def event(
    epoch: str, detector: str, i: int, animal: str = "sim", **values: object
) -> dict:
    """One features row's identity and the values it sets."""
    row: dict = {
        "animal": animal,
        "day": DAY,
        "epoch": epoch,
        "detector": detector,
        "event_index": i,
        "event_key": key(detector, epoch, i, animal),
    }
    return row | values


def typed(dtypes: dict, records: list[dict], defaults: dict) -> pd.DataFrame:
    """``records`` with ``dtypes``; an unnamed column takes its default."""
    return pd.DataFrame(
        {
            column: pd.Series(
                [
                    record.get(
                        column,
                        _OBJECT_DEFAULTS.get(column, defaults[dtype])
                        if dtype is object
                        else defaults[dtype if dtype in defaults else str(dtype)],
                    )
                    for record in records
                ],
                dtype=dtype,
            )
            for column, dtype in dtypes.items()
        }
    )


def sessions_of(events: list[dict]) -> list[tuple[str, str]]:
    return sorted({(e["animal"], e["epoch"]) for e in events})


def build(
    events: list[dict],
    *,
    memberships: dict | None = None,
    unavailable: set | frozenset = frozenset(),
    cells: list[dict] = (),  # type: ignore[assignment]
    rejection: list[dict] = (),  # type: ignore[assignment]
    candidates: list[dict] | None = None,
    no_memoryless: set | frozenset = frozenset(),
    sessions: list[tuple[str, str]] | None = None,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """A run's manifest and the six tables the report reads.

    Parameters
    ----------
    events : list of dict
        Features rows (``event``); a ``baseline`` row stands for a run written
        before the baselines were dropped.
    memberships : dict
        ``(event_key, other_detector) -> membership``; ``unmatched`` otherwise.
    unavailable : set
        ``(animal, epoch, detector)`` whose pairs share no exposure.
    candidates : list of dict, optional
        The memoryless run's candidates; by default one per
        memoryless_criteria event, published when it has a memoryless twin.
        Every memoryless event is given a memoryless_criteria twin.
    no_memoryless : set
        ``(animal, epoch)`` with no memoryless run.
    """
    memberships = memberships or {}
    sessions = sessions or sessions_of(events)
    keys = {e["event_key"] for e in events}
    events = events + [
        event(e["epoch"], "memoryless_criteria", e["event_index"], e["animal"])
        for e in events
        if e["detector"] == "memoryless"
        and key("memoryless_criteria", e["epoch"], e["event_index"], e["animal"])
        not in keys
    ]
    if candidates is None:
        published = {e["event_key"] for e in events if e["detector"] == "memoryless"}
        candidates = [
            {
                "animal": e["animal"],
                "epoch": e["epoch"],
                "ms_event_id": e["event_index"],
                "is_replay": True,
                "n_shuffles": 100,
                "shuffle_pass": key(
                    "memoryless", e["epoch"], e["event_index"], e["animal"]
                )
                in published,
            }
            for e in events
            if e["detector"] == "memoryless_criteria"
        ]
    features = typed(dict(feature_run.FEATURE_DTYPES), events, _FEATURE_DEFAULTS)
    compared = [e for e in events if e["detector"] != "baseline"]
    pairs = []
    for e in compared:
        for other in DETECTORS:
            if other == e["detector"]:
                continue
            membership = memberships.get((e["event_key"], other), "unmatched")
            pairs.append(
                {
                    **{c: e[c] for c in ("animal", "day", "epoch", "detector")},
                    "event_key": e["event_key"],
                    "other_detector": other,
                    "within_shared_exposure": membership != "unknown",
                    "matched": membership == "matched",
                    "n_counterparts": int(membership == "matched"),
                    "membership": membership,
                }
            )
    directional = [
        {
            "animal": animal,
            "day": DAY,
            "epoch": epoch,
            "source_detector": source,
            "target_detector": target,
            "available": (animal, epoch, source) not in unavailable
            and (animal, epoch, target) not in unavailable,
        }
        for animal, epoch in sessions
        for source in DETECTORS
        for target in DETECTORS
        if source != target
    ]
    cell_rows = [{"day": DAY, "n_against": 2, **row} for row in cells]
    tables = {
        "features": features,
        "pair_membership": typed(dict(MATCH_DTYPES), pairs, _FEATURE_DEFAULTS),
        "rejection": typed(
            dict(REJECTION_DTYPES),
            [{"day": DAY, **r} for r in rejection],
            {**_FEATURE_DEFAULTS, "Int64": pd.NA, "boolean": pd.NA},
        ),
        "cells": typed(
            dict(inventory_run.TABLE_DTYPES["cells"]), cell_rows, _FEATURE_DEFAULTS
        ),
        "directional": typed(
            dict(inventory_run.TABLE_DTYPES["directional"]),
            directional,
            {**_FEATURE_DEFAULTS, np.float64: 0.0},
        ),
        "memoryless_events": typed(
            MEMORYLESS_COLUMNS,
            [{"day": DAY, "artifact_drop_reason": None, **c} for c in candidates],
            _FEATURE_DEFAULTS,
        ),
    }
    manifest = {
        "run_id": "run01",
        "settings": SETTINGS,
        "epochs": {
            f"{animal}{DAY}_{epoch}_HPC": {
                "status": "ok",
                "memoryless_available": (animal, epoch) not in no_memoryless,
            }
            for animal, epoch in sessions
        },
    }
    return manifest, tables


def row_of(table: pd.DataFrame, **values: object) -> pd.Series:
    """The one row of ``table`` with ``values``."""
    mask = np.ones(len(table), dtype=bool)
    for column, value in values.items():
        mask &= (table[column] == value).to_numpy()
    assert mask.sum() == 1, (values, int(mask.sum()))
    return table[mask].iloc[0]


class TestAnAnimalPoolsItsSessions:
    def test_by_summing_counts_not_averaging_proportions(self):
        # 90 spikes over 10 rows and 9 over 90: the sums give 99 over 100, the
        # per-session ratios average to 4.55
        manifest, tables = build(
            [
                event("02_r1", "kay", 0, n_rows=10, n_spikes=90),
                event("04_r2", "kay", 0, n_rows=90, n_spikes=9),
            ]
        )

        out = report.report_tables(manifest, tables)

        pooled = row_of(out["evidence_coverage_by_animal"], detector="kay")
        assert (pooled.n_sessions, pooled.n_spikes, pooled.n_rows) == (2, 99, 100)
        assert pooled.spikes_per_row == pytest.approx(0.99)
        assert np.mean([9.0, 0.1]) == pytest.approx(4.55)
        sessions = out["evidence_coverage_by_session"]
        assert sessions[sessions.detector == "kay"].spikes_per_row.tolist() == (
            pytest.approx([9.0, 0.1])
        )

    def test_the_rejection_counts_are_summed(self):
        rejection = [
            {
                "animal": "sim",
                "epoch": epoch,
                "detector": "state_space",
                "event_key": key("state_space", epoch, i),
                "cohort": "memoryless",
                "membership": "unmatched",
                "stage": "dispersion" if i < n_dispersion else "no_candidate",
                "unknown_reason": "",
                "n_windows": 3,
                "n_candidates": 1,
                "any_dispersion_failed": i < n_dispersion,
            }
            for epoch, n, n_dispersion in (("02_r1", 10, 9), ("04_r2", 90, 9))
            for i in range(n)
        ]
        manifest, tables = build(
            [
                event(e["epoch"], "state_space", int(e["event_key"][-1]))
                for e in rejection[:1]
            ]
            + [event("04_r2", "state_space", 0)],
            rejection=rejection,
        )

        out = report.report_tables(manifest, tables)

        pooled = row_of(
            out["rejection_by_animal"], cohort="memoryless", membership="unmatched"
        )
        assert (pooled.n_events, pooled.n_dispersion, pooled.n_no_candidate) == (
            100,
            18,
            82,
        )
        assert (pooled.n_windows, pooled.n_any_dispersion_failed) == (300, 18)


class TestUnknownMembership:
    def test_never_enters_an_unmatched_population(self):
        unknown = key("state_space", "02_r1", 0)
        manifest, tables = build(
            [
                event("02_r1", "state_space", 0, evidence_dll=100.0),
                event("02_r1", "state_space", 1, evidence_dll=1.0),
                event("02_r1", "state_space", 2, evidence_dll=3.0),
                event("02_r1", "memoryless", 0),
                event("02_r1", "kay", 0),
            ],
            memberships={(unknown, "memoryless"): "unknown"},
        )

        out = report.report_tables(manifest, tables)

        for level in ("by_session", "by_animal"):
            evidence = out[f"features_evidence_{level}"]
            unmatched = row_of(
                evidence, detector="state_space", population="unmatched_by:memoryless"
            )
            assert unmatched.n_events == 2
            assert unmatched.evidence_dll_max == 3.0
            apart = row_of(
                evidence, detector="state_space", population="unknown_by:memoryless"
            )
            assert apart.n_events == 1
            assert apart.unknown_reason == "outside_shared_exposure"

    def test_a_pair_sharing_no_exposure_gives_its_reason(self):
        manifest, tables = build(
            [event("02_r1", "kay", 0), event("02_r1", "state_space", 0)],
            memberships={(key("kay", "02_r1", 0), "memoryless"): "unknown"},
            unavailable={("sim", "02_r1", "memoryless")},
        )

        out = report.report_tables(manifest, tables)

        apart = row_of(
            out["features_timing_by_animal"],
            detector="kay",
            population="unknown_by:memoryless",
        )
        assert apart.unknown_reason == "no_shared_exposure"


class TestAMissingFeature:
    def test_is_counted_and_absent_from_its_quantiles(self):
        manifest, tables = build(
            [
                event("02_r1", "kay", 0, evidence_dll=1.0),
                event("02_r1", "kay", 1, evidence_dll=np.nan),
                event("02_r1", "kay", 2, evidence_dll=3.0),
            ]
        )

        out = report.report_tables(manifest, tables)

        for level in ("by_session", "by_animal"):
            missing = row_of(
                out[f"missing_{level}"],
                detector="kay",
                population="all",
                feature="evidence_dll",
            )
            assert (missing.n, missing.n_missing) == (3, 1)
            spread = row_of(
                out[f"features_evidence_{level}"], detector="kay", population="all"
            )
            assert spread.n_events == 3
            assert spread.evidence_dll_n == 2
            assert spread.evidence_dll_median == 2.0
            assert spread.evidence_dll_max == 3.0

    def test_a_feature_present_is_counted_with_none_missing(self):
        manifest, tables = build([event("02_r1", "kay", 0)])

        out = report.report_tables(manifest, tables)

        missing = row_of(
            out["missing_by_animal"],
            detector="kay",
            population="all",
            feature="n_spikes",
        )
        assert (missing.n, missing.n_missing) == (1, 0)
        # missing in no session, so left out of the session table
        assert "n_spikes" not in set(out["missing_by_session"].feature)


def _pending_world() -> tuple[dict, dict[str, pd.DataFrame]]:
    events = [
        event("02_r1", "kay", 0),
        event("02_r1", "state_space", 0),
        event("02_r1", "memoryless", 0),
    ]
    cells = [
        {
            "cell": "exactly_one_detector",
            "animal": "sim",
            "epoch": "02_r1",
            "detector": "kay",
            "event_key": key("kay", "02_r1", 0),
            "membership": "unmatched",
        },
        {
            "cell": "sequence_matched_by_no_trigger",
            "animal": "sim",
            "epoch": "02_r1",
            "detector": "state_space",
            "event_key": key("state_space", "02_r1", 0),
            "membership": "unmatched",
        },
    ]
    rejection = [
        {
            "animal": "sim",
            "epoch": "02_r1",
            "detector": "state_space",
            "event_key": key("state_space", "02_r1", 0),
            "cohort": "memoryless",
            "membership": "unmatched",
            "stage": "no_candidate",
            "unknown_reason": "",
        }
    ]
    return build(events, cells=cells, rejection=rejection)


class TestPending:
    def test_every_ms_and_memoryless_row_is_marked(self):
        manifest, tables = _pending_world()

        out = report.report_tables(manifest, tables)

        for level in ("by_session", "by_animal", "by_stratum_by_session"):
            ms = out[f"features_ms_{level}"]
            assert len(ms) and (ms.pending == report.PENDING).all()
        for name, table in out.items():
            if "pending" not in table:
                continue
            marked = table.pending == report.PENDING
            if "detector" in table:
                memoryless = table.detector.isin({"memoryless", "memoryless_criteria"})
                assert marked[memoryless].all(), name
            if "population" in table:
                against = table.population.str.endswith(":memoryless")
                assert marked[against].all(), name
            if "feature" in table:
                assert marked[table.feature.str.startswith("ms_")].all(), name
        for name in ("rejection_by_session", "memoryless_cohort_by_animal"):
            assert (out[name].pending == report.PENDING).all()

    def test_rows_without_a_memoryless_inventory_are_not_marked(self):
        manifest, tables = _pending_world()

        out = report.report_tables(manifest, tables)

        timing = out["features_timing_by_animal"]
        for population in ("all", "unmatched_by:state_space"):
            row = row_of(timing, detector="kay", population=population)
            assert row.pending == "", population
        # tested against memoryless, so pending
        cell = row_of(timing, detector="kay", population="exactly_one_detector")
        assert cell.pending == report.PENDING
        # tested against the triggered inventories only
        cell = row_of(
            timing, detector="state_space", population="sequence_matched_by_no_trigger"
        )
        assert cell.pending == ""

    def test_the_memoryless_context_rows_are_marked(self):
        manifest, tables = _pending_world()

        out = report.report_tables(manifest, tables)

        for level in ("by_session", "by_animal"):
            context = out[f"context_{level}"]
            for detector in ("memoryless", "memoryless_criteria"):
                row = row_of(context, detector=detector, population="all")
                assert row.pending == report.PENDING, (level, detector)
            assert row_of(context, detector="kay", population="all").pending == ""

    def test_every_caption_carries_the_pending_note(self):
        for caption in [
            *report.TABLE_CAPTIONS.values(),
            *report.FIGURE_CAPTIONS.values(),
        ]:
            assert report.PENDING in caption
        assert report.PENDING == "pending the Phase 2 pilot checks"


class TestOldBaselineRows:
    def test_are_dropped_from_every_table(self):
        manifest, tables = build(
            [
                event("02_r1", "kay", 0, evidence_dll=1.0),
                event("02_r1", "state_space", 0, evidence_dll=2.0),
                # a run written before the baselines were dropped
                event("02_r1", "baseline", 0, evidence_dll=1000.0),
            ]
        )

        out = report.report_tables(manifest, tables)

        for name, table in out.items():
            if "detector" in table:
                assert "baseline" not in set(table.detector), name
            if "population" in table:
                assert "baseline" not in set(table.population), name
        assert (out["features_evidence_by_animal"].evidence_dll_max < 1000).all()
        assert not [name for name in report.TABLE_CAPTIONS if "baseline" in name]
        assert not [name for name in report.FIGURE_CAPTIONS if "baseline" in name]


def test_session_tables_hold_the_decoders_pairs_and_animal_tables_every_pair():
    manifest, tables = build(
        [
            event("02_r1", "kay", 0),
            event("02_r1", "state_space", 0),
            event("02_r1", "memoryless", 0),
        ]
    )

    out = report.report_tables(manifest, tables)

    pairwise = {
        "unmatched_by:kay",
        "unmatched_by:state_space",
        "unmatched_by:memoryless",
    }
    session = set(out["features_timing_by_session"].population)
    animal = set(out["features_timing_by_animal"].population)
    assert session & pairwise == {"unmatched_by:state_space", "unmatched_by:memoryless"}
    assert animal & pairwise == pairwise
    assert "all" in session


def test_the_strata_take_the_runs_order():
    manifest, tables = build(
        [
            event("02_r1", "kay", 0, duration_stratum=">500 ms"),
            event("02_r1", "kay", 1, duration_stratum="0-50 ms"),
            event("02_r1", "kay", 2, duration_stratum="100-200 ms"),
            event("02_r1", "kay", 3, duration_stratum="0-50 ms"),
        ]
    )

    out = report.report_tables(manifest, tables)

    strata = out["features_timing_by_stratum_by_animal"]
    rows = strata[(strata.detector == "kay") & (strata.population == "all")]
    assert rows.duration_stratum.tolist() == ["0-50 ms", "100-200 ms", ">500 ms"]
    assert rows.n_events.tolist() == [2, 1, 1]
    assert "duration_stratum" not in out["features_timing_by_animal"]


def test_a_stratum_outside_the_runs_labels_is_refused():
    manifest, tables = build([event("02_r1", "kay", 0, duration_stratum="1-2 s")])

    with pytest.raises(ValueError, match="1-2 s"):
        report.report_tables(manifest, tables)


def test_rejection_keeps_unknown_apart_with_its_reasons():
    def stage(i: int, membership: str, stage: str, reason: str = "") -> dict:
        counts = (
            {
                "n_windows": 4,
                "n_kept": 2,
                "n_candidates": 1,
                "any_duration_failed": True,
            }
            if membership != "unknown"
            else {}
        )
        return {
            "animal": "sim",
            "epoch": "02_r1",
            "detector": "state_space",
            "event_key": key("state_space", "02_r1", i),
            "cohort": "memoryless_criteria",
            "membership": membership,
            "stage": stage,
            "unknown_reason": reason,
            **counts,
        }

    rejection = [
        stage(0, "matched", "in_cohort"),
        stage(1, "unmatched", "duration"),
        stage(2, "unmatched", "no_window"),
        stage(3, "unknown", "unknown", "no_shared_exposure"),
        stage(4, "unknown", "unknown", "outside_shared_exposure"),
        stage(5, "unknown", "unknown", "outside_shared_exposure"),
    ]
    manifest, tables = build(
        [event("02_r1", "state_space", i) for i in range(6)], rejection=rejection
    )

    out = report.report_tables(manifest, tables)

    for level in ("by_session", "by_animal"):
        table = out[f"rejection_{level}"]
        unmatched = row_of(table, cohort="memoryless_criteria", membership="unmatched")
        assert (unmatched.n_events, unmatched.n_duration, unmatched.n_no_window) == (
            2,
            1,
            1,
        )
        assert (unmatched.n_windows, unmatched.n_any_duration_failed) == (8, 2)
        assert unmatched.n_unknown == 0
        unknown = row_of(table, cohort="memoryless_criteria", membership="unknown")
        assert (unknown.n_events, unknown.n_unknown) == (3, 3)
        assert (unknown.n_no_shared_exposure, unknown.n_outside_shared_exposure) == (
            1,
            2,
        )
        assert pd.isna(unknown.n_windows)
        assert pd.isna(unknown.n_any_duration_failed)
        matched = row_of(table, cohort="memoryless_criteria", membership="matched")
        assert matched.n_in_cohort == 1


def test_the_memoryless_cohort_before_and_after_the_shuffle():
    def candidate(i: int, **values: object) -> dict:
        return {"animal": "sim", "epoch": "02_r1", "ms_event_id": i, **values}

    candidates = [
        candidate(0, is_replay=True, shuffle_pass=True, n_shuffles=100),
        candidate(1, is_replay=True, shuffle_pass=False, n_shuffles=100),
        candidate(
            2,
            is_replay=True,
            shuffle_pass=False,
            shuffle_testable=False,
            n_shuffles=100,
        ),
        candidate(
            3,
            is_replay=True,
            shuffle_pass=True,
            n_shuffles=100,
            artifact_drop_reason="in excluded time",
        ),
        candidate(4, is_replay=False, shuffle_pass=False, n_shuffles=0),
    ]
    events = [event("02_r1", "memoryless_criteria", i) for i in (0, 1, 2)] + [
        event("02_r1", "memoryless", 0),
        event("04_r2", "kay", 0),
    ]
    manifest, tables = build(
        events, candidates=candidates, no_memoryless={("sim", "04_r2")}
    )

    out = report.report_tables(manifest, tables)

    by_session = out["memoryless_cohort_by_session"]
    native = row_of(by_session, epoch="02_r1", scope="candidates")
    assert (
        native.n_candidates,
        native.n_criteria,
        native.n_published,
        native.n_shuffle_failed,
        native.n_shuffle_untestable,
    ) == (5, 4, 2, 1, 1)
    compared = row_of(by_session, epoch="02_r1", scope="compared")
    assert (
        compared.n_candidates,
        compared.n_criteria,
        compared.n_published,
        compared.n_shuffle_failed,
        compared.n_shuffle_untestable,
    ) == (3, 3, 1, 1, 1)
    assert not row_of(by_session, epoch="04_r2", scope="compared").available
    pooled = row_of(out["memoryless_cohort_by_animal"], scope="compared")
    assert (pooled.n_sessions, pooled.n_criteria) == (1, 3)
    assert pooled.unavailable_sessions == f"sim{DAY}_04_r2"


def test_the_memoryless_cohort_refuses_a_published_count_it_cannot_reproduce():
    candidates = [
        {
            "animal": "sim",
            "epoch": "02_r1",
            "ms_event_id": 0,
            "is_replay": True,
            "shuffle_pass": False,
            "n_shuffles": 100,
        }
    ]
    manifest, tables = build(
        [event("02_r1", "memoryless_criteria", 0), event("02_r1", "memoryless", 0)],
        candidates=candidates,
    )

    with pytest.raises(ValueError, match="published"):
        report.report_tables(manifest, tables)


def test_comparator_only_events_by_disagreement():
    def kay(i: int, disagreement: object, dll: float) -> dict:
        return event(
            "02_r1",
            "kay",
            i,
            ss_disagreement=disagreement,
            evidence_dll=dll,
            ss_continuous_prob_mean=0.1 * (i + 1),
        )

    events = [
        kay(0, "near", 2.0),
        kay(1, "near", -1.0),
        kay(2, "sustained", 2.0),
        kay(3, None, 0.5),
        kay(4, "subthreshold", 2.0),
        event("02_r1", "state_space", 0),
    ]
    matched = key("kay", "02_r1", 4)
    manifest, tables = build(events, memberships={(matched, "state_space"): "matched"})

    out = report.report_tables(manifest, tables)

    for level in ("by_session", "by_animal"):
        table = out[f"comparator_only_{level}"]
        near = row_of(table, detector="kay", ss_disagreement="near")
        assert (near.n_events, near.n_scorable) == (2, 2)
        assert near.evidence_dll_median == pytest.approx(0.5)
        assert near.ss_continuous_prob_mean_median == pytest.approx(0.15)
        assert row_of(table, detector="kay", ss_disagreement="sustained").n_events == 1
        assert row_of(table, detector="kay", ss_disagreement="missing").n_events == 1
        # matched by state_space, so not comparator-only
        assert "subthreshold" not in set(table.ss_disagreement)
        assert "state_space" not in set(table.detector)


def test_evidence_coverage_counts_by_reason_with_the_spikes_per_row():
    manifest, tables = build(
        [
            event("02_r1", "kay", 0, n_rows=10, evidence_n_valid_rows=8, n_spikes=5),
            event(
                "02_r1",
                "kay",
                1,
                n_rows=0,
                evidence_unscorable="no_row",
                evidence_dll=np.nan,
                evidence_n_valid_rows=pd.NA,
                n_spikes=pd.NA,
            ),
            event(
                "02_r1",
                "kay",
                2,
                n_rows=4,
                evidence_unscorable="no_valid_row",
                evidence_dll=np.nan,
                evidence_n_valid_rows=0,
                n_spikes=0,
            ),
        ]
    )

    out = report.report_tables(manifest, tables)

    for level in ("by_session", "by_animal"):
        coverage = row_of(out[f"evidence_coverage_{level}"], detector="kay")
        assert (
            coverage.n_events,
            coverage.n_scorable,
            coverage.n_unscorable_no_row,
            coverage.n_unscorable_no_valid_row,
        ) == (3, 1, 1, 1)
        assert (coverage.n_rows, coverage.n_valid_rows, coverage.n_spikes) == (14, 8, 5)
        assert coverage.spikes_per_row == pytest.approx(5 / 14)


def test_an_unscorable_reason_the_report_does_not_know_is_refused():
    manifest, tables = build(
        [event("02_r1", "kay", 0, evidence_unscorable="odd", evidence_dll=np.nan)]
    )

    with pytest.raises(ValueError, match="odd"):
        report.report_tables(manifest, tables)


# The report written from a small run on disk.


def _write_run(root) -> tuple:
    """A run, its comparison run and its memoryless run, where ``paths`` puts
    them under data root `root`; the three run folders' parents."""
    events = [
        event(
            epoch,
            detector,
            i,
            animal=animal,
            evidence_dll=float(i),
            n_rows=10,
            n_spikes=i,
            duration_stratum=DURATION_STRATUM_LABELS[i % 5],
            ss_disagreement=(None if detector == "state_space" else "near"),
        )
        for animal, epoch in (("sim", "02_r1"), ("sim", "04_r2"), ("toy", "02_r1"))
        for detector in DETECTORS
        for i in range(4)
    ] + [
        event("02_r1", "baseline", 0, matched_event_key=key("kay", "02_r1", 0)),
    ]
    rejection = [
        {
            "animal": e["animal"],
            "epoch": e["epoch"],
            "detector": "state_space",
            "event_key": e["event_key"],
            "cohort": cohort,
            "membership": "unmatched",
            "stage": "no_candidate",
            "unknown_reason": "",
            "n_windows": 2,
        }
        for e in events
        if e["detector"] == "state_space"
        for cohort in ("memoryless", "memoryless_criteria")
    ]
    manifest, tables = build(events, rejection=rejection)
    manifest["epochs"]["sim20200101_06_r3_HPC"] = {"status": "failed", "error": "e"}
    run_dir = under_root(root, event_features_root) / "run01"
    inventory_dir = under_root(root, inventory_comparison_root) / "inventory01"
    memoryless_dir = under_root(root, memoryless_runs_root) / "memoryless01"
    for folder in (run_dir, inventory_dir, memoryless_dir):
        folder.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    session = ["animal", "day", "epoch"]
    for (animal, day, epoch), _ in tables["features"].groupby(session):
        identifier = f"{animal}{day}_{epoch}_HPC"

        def of(frame: pd.DataFrame, session=(animal, day, epoch)) -> pd.DataFrame:
            mask = (frame[["animal", "day", "epoch"]] == session).all(axis=1)
            return frame[mask.to_numpy()].reset_index(drop=True)

        for name in feature_run.TABLE_DTYPES:
            write_table(of(tables[name]), output_path(identifier, run_dir, name))
        for name in ("cells", "directional"):
            write_table(
                of(tables[name]),
                output_path(identifier, inventory_dir, name),
            )
        write_table(
            of(tables["memoryless_events"]),
            memoryless_outputs(identifier, memoryless_dir)["events"],
        )
    return run_dir, inventory_dir.parent, memoryless_dir.parent


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("report")
    run_dir, inventory_root, memoryless_root = _write_run(tmp_path)
    # an earlier version of the report's outputs
    (run_dir / "report").mkdir()
    (run_dir / "report" / "evidence_vs_baseline_by_animal.csv").write_text("old")
    (tmp_path / "figures" / "run01" / "examples").mkdir(parents=True)
    (tmp_path / "figures" / "run01" / "evidence_vs_baseline.png").write_bytes(b"old")
    (tmp_path / "figures" / "run01" / "examples" / "kept.png").write_bytes(b"x")
    paths = report.write_report(
        run_dir,
        tmp_path / "figures",
        inventory_root=inventory_root,
        memoryless_root=memoryless_root,
        analysed=[
            "sim20200101_02_r1_HPC",
            "sim20200101_04_r2_HPC",
            "toy20200101_02_r1_HPC",
            "sim20200101_06_r3_HPC",
            "sim20200101_08_r4_HPC",
        ],
    )
    return run_dir / "report", tmp_path / "figures" / "run01", paths


class TestTheReportOnARunOnDisk:
    def test_every_table_figure_and_caption_is_written(self, written):
        report_dir, figure_dir, paths = written

        for name in report.TABLE_CAPTIONS:
            assert (report_dir / f"{name}.csv").exists(), name
        for name in report.FIGURE_CAPTIONS:
            assert (figure_dir / name).exists(), name
        assert (figure_dir / "captions.md").exists()
        assert len(paths["tables"]) == len(report.TABLE_CAPTIONS)
        assert len(paths["figures"]) == len(report.FIGURE_CAPTIONS)
        # nothing of an earlier version is left beside them; examples are kept
        assert sorted(p.stem for p in report_dir.glob("*.csv")) == sorted(
            report.TABLE_CAPTIONS
        )
        assert sorted(p.name for p in figure_dir.glob("*.png")) == sorted(
            report.FIGURE_CAPTIONS
        )
        assert (figure_dir / "examples" / "kept.png").exists()

    def test_report_md_lists_every_table_and_states_the_unknowns(self, written):
        report_dir, _, _ = written

        text = (report_dir / "report.md").read_text()

        for name, caption in report.TABLE_CAPTIONS.items():
            assert f"`{name}.csv`" in text
            assert caption in text
        for name in ("boundary_offsets", "match_strength", "recount"):
            for level in ("by_session", "by_animal"):
                assert f"inventory01/report/{name}_{level}.csv" in text, name
        assert "tracking confidence" in text.lower()
        assert SETTINGS["tracking_confidence"] in text
        assert "Widloski" in text
        assert f"SWR source {SETTINGS['swr_source']}" in text.split("## ")[0]
        # 3 ok, 1 failed, and 1 analysed epoch outside the run
        assert "3 ok epochs, 1 failed, and 1 analysed epochs" in text

    def test_the_tables_read_back_from_disk(self, written):
        report_dir, _, _ = written

        epochs = pd.read_csv(report_dir / "epochs.csv", keep_default_na=False)
        assert sorted(epochs.status) == ["failed", "ok", "ok", "ok"]
        timing = pd.read_csv(report_dir / "features_timing_by_animal.csv")
        assert set(timing.animal) == {"sim", "toy"}
        assert (
            row_of(timing, animal="sim", detector="kay", population="all").n_events == 8
        )


def test_main_reads_the_run_named(tmp_path, processed_data_root, monkeypatch):
    run_dir, _, _ = _write_run(processed_data_root)
    monkeypatch.setattr(report, "FIGURE_ROOT", tmp_path / "figures")
    monkeypatch.setattr(report, "analysed_epoch_ids", lambda: [])

    report.main(["--run-id", "run01"])

    assert (run_dir / "report" / "report.md").exists()
    assert (tmp_path / "figures" / "run01" / "evidence_by_spikes.png").exists()


def test_every_feature_group_names_columns_of_the_features_table():
    known = {*feature_run.FEATURE_DTYPES, *report.DERIVED_FEATURES}
    for group, columns in report.FEATURE_GROUPS.items():
        assert set(columns) <= known, group
    assert all(c.startswith("ss_") for c in report.FEATURE_GROUPS["ss"])
    assert all(c.startswith("ms_") for c in report.FEATURE_GROUPS["ms"])


def test_the_distribution_reads_the_inventory_reports_quantiles():
    from scripts.detector_comparison import (
        report_inventory_comparison as inventory_report,
    )

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "g": rng.integers(0, 7, 500).astype(str),
            "x": rng.normal(size=500),
            "y": pd.array(
                np.where(rng.random(500) < 0.2, None, rng.integers(0, 9, 500)),
                dtype="Int64",
            ),
        }
    )
    frame.loc[frame.index[::5], "x"] = np.nan

    fast = report.distribution(frame, ["g"], ["x", "y"], "n_events")
    slow = inventory_report._distribution(frame, ["g"], ["x", "y"], "n_events")

    assert list(fast.columns) == list(slow.columns)
    pd.testing.assert_frame_equal(
        fast, slow.astype(fast.dtypes.to_dict()), check_exact=False, rtol=1e-12
    )
    assert list(report.distribution(frame.iloc[:0], ["g"], ["x"], "n").columns) == list(
        inventory_report._distribution(frame.iloc[:0], ["g"], ["x"], "n").columns
    )

    # one-element and all-NaN groups, two keys, one of them categorical
    edge = pd.DataFrame(
        {
            "g": ["a", "a", "b", "c", "c"],
            "s": pd.Categorical(
                ["late", "early", "late", "early", "early"],
                categories=["late", "early"],
                ordered=True,
            ),
            "x": [1.0, 2.0, 3.0, np.nan, np.nan],
        }
    )
    fast = report.distribution(edge, ["g", "s"], ["x"], "n")
    slow = inventory_report._distribution(
        edge.assign(s=edge.s.astype(str)), ["g", "s"], ["x"], "n"
    )
    assert fast.s.tolist() == ["late", "early", "late", "early"]
    order = ["g", "s"]
    pd.testing.assert_frame_equal(
        fast.assign(s=fast.s.astype(str)).sort_values(order, ignore_index=True),
        slow.sort_values(order, ignore_index=True).astype(
            fast.dtypes.to_dict() | {"s": object}
        ),
        check_exact=False,
        rtol=1e-12,
    )
    lone = row_of(fast, g="b")
    assert (lone.n, lone.x_n, lone.x_median, lone.x_max) == (1, 1, 3.0, 3.0)
    empty = row_of(fast, g="c")
    assert (empty.n, empty.x_n) == (2, 0)
    assert np.isnan(empty.x_median) and np.isnan(empty.x_max)


def test_the_context_pools_counts_per_animal():
    events = [
        event(
            "02_r1",
            "kay",
            0,
            at_well=True,
            occupancy_state="at_well",
            current_patch=1.0,
        ),
        event(
            "02_r1",
            "kay",
            1,
            at_well=True,
            occupancy_state="at_well",
            current_patch=2.0,
        ),
        event(
            "02_r1",
            "kay",
            2,
            at_well=False,
            occupancy_state="leaving",
            is_rewarded_outcome=0.0,
        ),
        event(
            "04_r2",
            "kay",
            0,
            at_well=pd.NA,
            occupancy_state=None,
            current_patch=np.nan,
            is_rewarded_outcome=np.nan,
        ),
    ]
    manifest, tables = build(events)

    out = report.report_tables(manifest, tables)

    sessions = out["context_by_session"]
    first = row_of(sessions, epoch="02_r1", detector="kay", population="all")
    assert (first.n_events, first.n_at_well_true, first.p_at_well_true) == (
        3,
        2,
        pytest.approx(2 / 3),
    )
    pooled = row_of(out["context_by_animal"], detector="kay", population="all")
    assert (pooled.n_sessions, pooled.n_events) == (2, 4)
    assert (
        pooled.n_at_well_true,
        pooled.n_at_well_false,
        pooled.n_at_well_missing,
    ) == (2, 1, 1)
    assert pooled.p_at_well_true == pytest.approx(0.5)
    assert "p_sessions" not in out["context_by_animal"]
    assert (pooled.n_occupancy_state_at_well, pooled.n_occupancy_state_leaving) == (
        2,
        1,
    )
    assert pooled.n_occupancy_state_missing == 1
    assert (
        pooled.n_current_patch_1,
        pooled.n_current_patch_2,
        pooled.n_current_patch_3,
    ) == (2, 1, 0)
    assert pooled.n_current_patch_missing == 1
    assert (pooled.n_is_rewarded_outcome_1, pooled.n_is_rewarded_outcome_0) == (2, 1)
    assert pooled.pending == ""


def test_a_context_value_outside_its_categories_is_refused():
    manifest, tables = build([event("02_r1", "kay", 0, current_patch=4.0)])

    with pytest.raises(ValueError, match="current_patch"):
        report.report_tables(manifest, tables)


def test_an_event_without_a_stratum_is_refused():
    manifest, tables = build([event("02_r1", "kay", 0, duration_stratum=None)])

    with pytest.raises(ValueError, match="no duration_stratum"):
        report.report_tables(manifest, tables)


def test_the_session_missing_table_leaves_out_features_never_missing():
    manifest, tables = build(
        [
            event("02_r1", "kay", 0, evidence_dll=np.nan),
            event("04_r2", "kay", 0),
        ]
    )

    out = report.report_tables(manifest, tables)

    sessions = out["missing_by_session"]
    # spikes_per_row too: an event of no row has none
    assert set(sessions.feature) == {
        "evidence_dll",
        "ss_disagreement",
        "spikes_per_row",
    }
    assert set(out["missing_by_animal"].feature) == set(report.MISSING_FEATURES)
    assert "evidence_dll" not in report.never_missing(out["missing_by_animal"])
    assert "n_spikes" in report.never_missing(out["missing_by_animal"])


def test_an_empty_occupancy_state_counts_as_missing():
    manifest, tables = build(
        [
            event("02_r1", "kay", 0, occupancy_state="at_well"),
            event("02_r1", "kay", 1, occupancy_state=""),
        ]
    )

    out = report.report_tables(manifest, tables)

    context = row_of(out["context_by_animal"], detector="kay", population="all")
    assert (context.n_occupancy_state_at_well, context.n_occupancy_state_missing) == (
        1,
        1,
    )
    assert "n_occupancy_state_" not in out["context_by_animal"]
    missing = row_of(
        out["missing_by_animal"],
        detector="kay",
        population="all",
        feature="occupancy_state",
    )
    assert (missing.n, missing.n_missing) == (2, 1)


def test_the_evidence_is_read_beside_the_spikes_per_row():
    events = [
        event("02_r1", "kay", 0, n_rows=10, n_spikes=1, evidence_dll=0.1),  # 0.1
        event("02_r1", "kay", 1, n_rows=10, n_spikes=3, evidence_dll=0.3),  # 0.3
        event("02_r1", "kay", 2, n_rows=10, n_spikes=15, evidence_dll=1.0),  # 1.5
        event("04_r2", "kay", 0, n_rows=10, n_spikes=15, evidence_dll=3.0),  # 1.5
        # no row, so in no bin
        event("04_r2", "kay", 1, n_rows=0, n_spikes=pd.NA, evidence_dll=np.nan),
    ]
    manifest, tables = build(events)

    out = report.report_tables(manifest, tables)

    pooled = out["evidence_by_spikes_by_animal"]
    pooled = pooled[(pooled.detector == "kay") & (pooled.population == "all")]
    assert pooled.spikes_per_row_bin.astype(str).tolist() == [
        "0-0.25",
        "0.25-0.5",
        "1-2",
    ]
    assert pooled.n_events.tolist() == [1, 1, 2]
    assert pooled.evidence_dll_median.tolist() == pytest.approx([0.1, 0.3, 2.0])
    sessions = out["evidence_by_spikes_by_session"]
    second = row_of(
        sessions,
        epoch="04_r2",
        detector="kay",
        population="all",
        spikes_per_row_bin="1-2",
    )
    assert second.n_events == 1
    evidence = row_of(
        out["features_evidence_by_animal"], detector="kay", population="all"
    )
    assert (evidence.spikes_per_row_n, evidence.spikes_per_row_median) == (
        4,
        pytest.approx(0.9),
    )


def test_every_evidence_caption_says_what_the_score_is_and_is_not():
    evidence = [
        caption
        for name, caption in report.TABLE_CAPTIONS.items()
        if name.startswith(
            ("features_evidence_", "evidence_coverage_", "evidence_by_spikes_")
        )
        or name.startswith("comparator_only_")
    ]
    evidence += [
        report.FIGURE_CAPTIONS["evidence_by_stratum.png"],
        report.FIGURE_CAPTIONS["evidence_by_spikes.png"],
    ]
    assert len(evidence) == 12
    for caption in evidence:
        assert (
            "log-likelihood difference between the best remote real place bin "
            "and the animal's bin" in caption
        )
        assert "grows with the number of spikes" in caption
        assert "It is a diagnostic: it is not a calibrated declaration of replay" in (
            caption
        )
    assert "evidence_vs_baseline.png" not in report.FIGURE_CAPTIONS


def test_the_ripple_and_mua_traces_are_named():
    for level in ("by_session", "by_animal"):
        for name in (
            f"features_lfp_mua_{level}",
            f"features_lfp_mua_by_stratum_{level}",
        ):
            caption = report.TABLE_CAPTIONS[name]
            assert "SWR_zscore" in caption and "MUA_zscore" in caption
            assert "Kay consensus" in caption
            assert "one sample off" in caption
