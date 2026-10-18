"""The event features report: per-session values and animal pools.

The rules are checked on per-epoch tables built by hand with the runner's own
columns and dtypes: an animal pools its sessions by summing counts, never by
averaging their proportions; an ``unknown`` membership never enters an
``unmatched`` population; a missing feature is counted and left out of its
quantiles; every ``ms_*`` and memoryless row is marked pending; the baseline
windows never enter a detector's populations. The last tests write the report
of a small run on disk, into temporary folders.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.qa.report_event_features as report
from continuum_swr_replay.analysis.inventory_comparison.features import (
    DURATION_STRATUM_LABELS,
)
from continuum_swr_replay.analysis.inventory_comparison.selection import (
    REJECTION_DTYPES,
)
from continuum_swr_replay.table_io import write_table
from scripts.decoding.run_memoryless_detection import (
    output_paths as memoryless_outputs,
)
from scripts.qa import run_event_features as features_runner
from scripts.qa import run_inventory_comparison as comparison

DAY = 20200101
DETECTORS = ("kay", "state_space", "memoryless", "memoryless_criteria")
"""The detectors of the simulated sessions."""

SETTINGS = {
    "input_runs": {"inventory": "inventory01", "memoryless": "memoryless01"},
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
    "baseline_level": "location",
    "ss_disagreement": None,
    "matched_event_key": None,
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
    if detector == "baseline":
        row |= {
            "baseline_level": None,
            "baseline_n": pd.NA,
            "baseline_n_scored": pd.NA,
            "baseline_median_dll": np.nan,
            "baseline_q95_dll": np.nan,
            "evidence_rank": np.nan,
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
        Features rows (``event``), baseline windows included.
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
    features = typed(dict(features_runner.FEATURE_DTYPES), events, _FEATURE_DEFAULTS)
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
        "pair_membership": typed(
            dict(features_runner.MATCH_DTYPES), pairs, _FEATURE_DEFAULTS
        ),
        "rejection": typed(
            dict(REJECTION_DTYPES),
            [{"day": DAY, **r} for r in rejection],
            {**_FEATURE_DEFAULTS, "Int64": pd.NA, "boolean": pd.NA},
        ),
        "cells": typed(
            dict(comparison.TABLE_DTYPES["cells"]), cell_rows, _FEATURE_DEFAULTS
        ),
        "directional": typed(
            dict(comparison.TABLE_DTYPES["directional"]),
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


def scored_against_baseline(epoch: str, n: int, n_above: int) -> list[dict]:
    """``n`` state-space events, ``n_above`` of them above their q95."""
    return [
        event(
            epoch,
            "state_space",
            i,
            evidence_dll=2.0 if i < n_above else 0.5,
            baseline_median_dll=0.0,
            baseline_q95_dll=1.0,
        )
        for i in range(n)
    ]


class TestAnAnimalPoolsItsSessions:
    def test_by_summing_counts_not_averaging_proportions(self):
        # 9 of 10 and 9 of 90: the counts pool to 18 of 100, the proportions
        # average to 0.5
        manifest, tables = build(
            scored_against_baseline("02_r1", 10, 9)
            + scored_against_baseline("04_r2", 90, 9)
        )

        out = report.report_tables(manifest, tables)

        pooled = row_of(
            out["evidence_vs_baseline_by_animal"],
            detector="state_space",
            population="all",
        )
        assert (pooled.n_sessions, pooled.n_with_baseline, pooled.n_above_q95) == (
            2,
            100,
            18,
        )
        assert pooled.p_above_q95 == pytest.approx(0.18)
        assert np.mean([9 / 10, 9 / 90]) == pytest.approx(0.5)
        sessions = out["evidence_vs_baseline_by_session"]
        assert sessions[
            (sessions.detector == "state_space") & (sessions.population == "all")
        ].n_above_q95.tolist() == [9, 9]

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
            counts = out[f"evidence_vs_baseline_{level}"]
            assert (
                row_of(
                    counts,
                    detector="state_space",
                    population="unmatched_by:memoryless",
                ).n_events
                == 2
            )

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
        event("02_r1", "baseline", 0, matched_event_key=key("kay", "02_r1", 0)),
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
        window = row_of(timing, detector="baseline")
        assert window.pending == ""
        # tested against memoryless, so pending
        cell = row_of(timing, detector="kay", population="exactly_one_detector")
        assert cell.pending == report.PENDING
        # tested against the triggered inventories only
        cell = row_of(
            timing, detector="state_space", population="sequence_matched_by_no_trigger"
        )
        assert cell.pending == ""

    def test_every_caption_carries_the_pending_note(self):
        for caption in [
            *report.TABLE_CAPTIONS.values(),
            *report.FIGURE_CAPTIONS.values(),
        ]:
            assert report.PENDING in caption
        assert report.PENDING == "pending the Phase 2 pilot checks"


class TestBaselineWindows:
    def test_never_enter_a_detectors_populations(self):
        manifest, tables = build(
            [
                event("02_r1", "kay", 0, evidence_dll=1.0),
                event("02_r1", "state_space", 0, evidence_dll=2.0),
                event(
                    "02_r1",
                    "baseline",
                    0,
                    evidence_dll=1000.0,
                    matched_event_key=key("kay", "02_r1", 0),
                ),
            ]
        )

        out = report.report_tables(manifest, tables)

        for level in ("by_session", "by_animal", "by_stratum_by_animal"):
            evidence = out[f"features_evidence_{level}"]
            ours = evidence[evidence.detector != "baseline"]
            assert (ours.evidence_dll_max < 1000).all()
            assert set(ours.population) >= {"all"}
            assert "baseline" not in set(ours.population)
            window = row_of(evidence, detector="baseline")
            assert window.population == "baseline"
            assert (window.n_events, window.evidence_dll_max) == (1, 1000.0)
        assert "baseline" not in set(out["evidence_vs_baseline_by_animal"].detector)
        assert "baseline" not in set(out["baseline_matching_by_animal"].detector)
        coverage = out["evidence_coverage_by_animal"]
        assert row_of(coverage, detector="baseline").n_events == 1


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
            baseline_median_dll=0.0,
            baseline_q95_dll=1.0,
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
        assert (near.n_events, near.n_above_q95, near.n_at_or_below_median) == (2, 1, 1)
        assert near.ss_continuous_prob_mean_median == pytest.approx(0.15)
        assert row_of(table, detector="kay", ss_disagreement="sustained").n_events == 1
        assert row_of(table, detector="kay", ss_disagreement="missing").n_events == 1
        # matched by state_space, so not comparator-only
        assert "subthreshold" not in set(table.ss_disagreement)
        assert "state_space" not in set(table.detector)


def test_evidence_coverage_and_baseline_matching_count_by_reason_and_level():
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
                baseline_level="failed",
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
                baseline_level="duration_only",
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
        matching = row_of(out[f"baseline_matching_{level}"], detector="kay")
        assert (
            matching.n_events,
            matching.n_location,
            matching.n_duration_only,
            matching.n_failed,
        ) == (3, 1, 1, 1)


def test_an_unscorable_reason_the_report_does_not_know_is_refused():
    manifest, tables = build(
        [event("02_r1", "kay", 0, evidence_unscorable="odd", evidence_dll=np.nan)]
    )

    with pytest.raises(ValueError, match="odd"):
        report.report_tables(manifest, tables)


# The report written from a small run on disk.


def _write_run(tmp_path) -> tuple:
    events = [
        event(
            epoch,
            detector,
            i,
            animal=animal,
            evidence_dll=float(i),
            baseline_median_dll=0.5,
            baseline_q95_dll=1.5,
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
    run_dir = tmp_path / "event_features" / "run01"
    inventory_dir = tmp_path / "inventory" / "inventory01"
    memoryless_dir = tmp_path / "memoryless" / "memoryless01"
    for folder in (run_dir, inventory_dir, memoryless_dir):
        folder.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    session = ["animal", "day", "epoch"]
    for (animal, day, epoch), _ in tables["features"].groupby(session):
        identifier = f"{animal}{day}_{epoch}_HPC"

        def of(frame: pd.DataFrame, session=(animal, day, epoch)) -> pd.DataFrame:
            mask = (frame[["animal", "day", "epoch"]] == session).all(axis=1)
            return frame[mask.to_numpy()].reset_index(drop=True)

        for name in features_runner.TABLE_DTYPES:
            write_table(
                of(tables[name]), features_runner.output_path(identifier, run_dir, name)
            )
        for name in ("cells", "directional"):
            write_table(
                of(tables[name]),
                comparison.output_path(identifier, inventory_dir, name),
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


def test_main_reads_the_run_named(tmp_path, monkeypatch):
    run_dir, inventory_root, memoryless_root = _write_run(tmp_path)
    monkeypatch.setattr(report, "OUTPUT_ROOT", run_dir.parent)
    monkeypatch.setattr(report, "INVENTORY_ROOT", inventory_root)
    monkeypatch.setattr(report, "MEMORYLESS_ROOT", memoryless_root)
    monkeypatch.setattr(report, "FIGURE_ROOT", tmp_path / "figures")
    monkeypatch.setattr(report, "analysed_epoch_ids", lambda: [])

    report.main(["--run-id", "run01"])

    assert (run_dir / "report" / "report.md").exists()
    assert (tmp_path / "figures" / "run01" / "evidence_vs_baseline.png").exists()


def test_every_feature_group_names_columns_of_the_features_table():
    for group, columns in report.FEATURE_GROUPS.items():
        assert set(columns) <= set(features_runner.FEATURE_DTYPES), group
    assert all(c.startswith("ss_") for c in report.FEATURE_GROUPS["ss"])
    assert all(c.startswith("ms_") for c in report.FEATURE_GROUPS["ms"])


def test_the_distribution_reads_the_inventory_reports_quantiles():
    from scripts.qa import report_inventory_comparison as inventory_report

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


def test_twin_baseline_windows_count_once():
    def window(i: int, matched: str, start: float, dll: float) -> dict:
        return event(
            "02_r1",
            "baseline",
            i,
            matched_event_key=matched,
            event_start_s=start,
            event_end_s=start + 0.1,
            evidence_dll=dll,
        )

    events = [
        event("02_r1", "kay", 0),
        event("02_r1", "memoryless", 0),
        # the twins' windows are identical; the kay window is not
        window(0, key("memoryless", "02_r1", 0), 5.0, 2.0),
        window(1, key("memoryless_criteria", "02_r1", 0), 5.0, 2.0),
        window(2, key("kay", "02_r1", 0), 7.0, 4.0),
    ]
    manifest, tables = build(events)

    out = report.report_tables(manifest, tables)

    for level in ("by_session", "by_animal"):
        spread = row_of(out[f"features_evidence_{level}"], detector="baseline")
        assert (spread.n_events, spread.evidence_dll_median) == (2, 3.0)
        assert (
            row_of(out[f"evidence_coverage_{level}"], detector="baseline").n_events == 2
        )
        sources = out[f"baseline_sources_{level}"]
        assert sources.set_index("detector").n_windows.to_dict() == {
            "kay": 1,
            "memoryless": 1,
            "memoryless_criteria": 1,
        }
        assert sources.p_windows.tolist() == pytest.approx([1 / 3] * 3)
    missing = row_of(
        out["missing_by_animal"], detector="baseline", feature="evidence_dll"
    )
    assert missing.n == 2


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
    assert set(sessions.feature) == {"evidence_dll", "ss_disagreement"}
    assert set(out["missing_by_animal"].feature) == set(report.MISSING_FEATURES)
    assert "evidence_dll" not in report.never_missing(out["missing_by_animal"])
    assert "n_spikes" in report.never_missing(out["missing_by_animal"])
