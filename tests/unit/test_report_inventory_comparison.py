"""The inventory comparison's report: per-session values and animal pools.

The rules are checked on per-epoch tables built by hand with the runner's own
columns and dtypes: an animal pools its sessions by summing counts, never by
averaging their proportions; a session where the comparison was unavailable is
left out of the pool and listed; an ``unknown`` membership is never counted as
``unmatched``. The last tests run the report on the tables the runner itself
writes for its simulated epochs, into temporary folders.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.detection.report_inventory_comparison as report
import scripts.detection.run_inventory_comparison as runner

# the runner's simulated run, built once for this module
from tests.integration.test_run_inventory_comparison import (  # noqa: F401
    ANIMAL,
    DAY,
    EPOCHS,
    NO_DECODE,
    SHIFTED,
    inputs,
    module_monkeypatch,
    run,
)

MEMORYLESS_INVENTORIES = {"memoryless", "memoryless_criteria"}
DEFAULTS = {object: "", np.int64: 0, np.float64: 0.0, np.bool_: True, "Int64": pd.NA}


def rows(name: str, records: list[dict]) -> pd.DataFrame:
    """Table ``name`` as the runner types it; unnamed columns take a default."""
    dtypes = runner.TABLE_DTYPES[name]
    return pd.DataFrame(
        {
            column: pd.Series(
                [record.get(column, DEFAULTS[dtype]) for record in records],
                dtype=dtype,
            )
            for column, dtype in dtypes.items()
        }
    )


def session(epoch: str, **values) -> dict:
    return {"animal": "sim", "day": 20200101, "epoch": epoch, **values}


def observed(epochs_detectors: dict[tuple[str, str], bool]) -> pd.DataFrame:
    return rows(
        "observations",
        [
            session(epoch, detector=detector, available=available)
            for (epoch, detector), available in epochs_detectors.items()
        ],
    )


TWO_SESSIONS = [
    # 9 of 10 and 9 of 90: the counts pool to 18 of 100, the proportions
    # average to 0.5
    session("02_r1", n_source_events=10, n_eligible=10, n_matched=9),
    session("04_r2", n_source_events=95, n_eligible=90, n_excluded=5, n_matched=9),
]
UNAVAILABLE = session(
    # counts a flagged session should never carry, so leaving it out shows
    "06_r3",
    available=False,
    n_source_events=50,
    n_eligible=50,
    n_matched=50,
)


def directional(records: list[dict]) -> pd.DataFrame:
    return rows(
        "directional",
        [
            {
                "source_detector": "state_space",
                "target_detector": "kay",
                "proportion_matched": record["n_matched"] / record["n_eligible"],
                **record,
            }
            for record in records
        ],
    )


class TestAnAnimalPoolsItsSessions:
    def test_by_summing_counts_not_averaging_proportions(self):
        pooled = report.directional_by_animal(directional(TWO_SESSIONS))

        row = pooled.iloc[0]
        assert (row.n_sessions, row.n_eligible, row.n_matched) == (2, 100, 18)
        assert row.n_excluded == 5
        assert row.proportion_matched == pytest.approx(0.18)
        assert np.mean([9 / 10, 9 / 90]) == pytest.approx(0.5)

    def test_an_unavailable_session_is_left_out_and_listed(self):
        pooled = report.directional_by_animal(directional([*TWO_SESSIONS, UNAVAILABLE]))

        row = pooled.iloc[0]
        assert (row.n_sessions, row.n_eligible, row.n_matched) == (2, 100, 18)
        assert row.proportion_matched == pytest.approx(0.18)
        assert row.unavailable_sessions == "sim20200101_06_r3"

    def test_a_detector_unavailable_in_a_session_leaves_its_counts(self):
        counts = rows(
            "counts",
            [
                session("02_r1", detector="kay", n_native=4, n_eligible=4),
                session("04_r2", detector="kay", n_native=6, n_eligible=6),
                session("06_r3", detector="kay", n_native=7, n_eligible=7),
            ],
        )
        availability = report.detector_availability(
            observed(
                {
                    ("02_r1", "kay"): True,
                    ("04_r2", "kay"): True,
                    ("06_r3", "kay"): False,
                }
            )
        )

        row = report.counts_by_animal(
            report.counts_by_session(counts, availability)
        ).iloc[0]
        assert (row.n_sessions, row.n_native, row.n_eligible) == (2, 10, 10)
        assert row.unavailable_sessions == "sim20200101_06_r3"

    def test_the_swr_matrix_fraction_is_over_the_summed_eligible(self):
        matrix = rows(
            "swr_matrix",
            [
                session(
                    epoch,
                    cohort="state_space",
                    swr_definition="kay",
                    n_eligible=eligible,
                    n_non_swr=non_swr,
                    n_swr=eligible - non_swr,
                    available=available,
                )
                for epoch, eligible, non_swr, available in (
                    ("02_r1", 4, 1, True),
                    ("04_r2", 16, 12, True),
                    ("06_r3", 0, 0, False),
                )
            ],
        )

        row = report.swr_matrix_by_animal(matrix).iloc[0]
        assert (row.n_eligible, row.n_non_swr) == (20, 13)
        assert row.non_swr_fraction == pytest.approx(13 / 20)
        assert row.unavailable_sessions == "sim20200101_06_r3"


class TestUnknownMembership:
    CELLS = rows(
        "cells",
        [
            session(
                "02_r1",
                cell="swr_matched_by_no_decoder",
                detector="kay",
                event_key=f"kay:{number}",
                membership=membership,
            )
            for number, membership in enumerate(
                ["matched", "unmatched", "unknown", "unknown"]
            )
        ],
    )
    SOURCES = {"swr_matched_by_no_decoder": ["kay"]}

    def test_never_enters_an_unmatched_count(self):
        availability = report.detector_availability(observed({("02_r1", "kay"): True}))

        by_session = report.cells_by_session(self.CELLS, availability, self.SOURCES)
        by_animal = report.cells_by_animal(by_session)

        for table in (by_session, by_animal):
            row = table.iloc[0]
            assert (row.n_matched, row.n_unmatched, row.n_unknown) == (1, 1, 2)
            assert row.n_events == 4

    def test_a_session_without_its_source_detector_is_listed(self):
        availability = report.detector_availability(
            observed({("02_r1", "kay"): True, ("04_r2", "kay"): False})
        )

        by_session = report.cells_by_session(self.CELLS, availability, self.SOURCES)
        row = report.cells_by_animal(by_session).iloc[0]

        assert len(by_session) == 2
        assert (row.n_sessions, row.n_unmatched, row.n_unknown) == (1, 1, 2)
        assert row.unavailable_sessions == "sim20200101_04_r2"


class TestTheLabelTablesPoolOnTheMatrixPopulation:
    """Paired labels and definition disagreement pool the sessions the SWR
    matrix pools, cohort by cohort, and list the others."""

    MATRIX = rows(
        "swr_matrix",
        [
            session(
                epoch,
                cohort=cohort,
                swr_definition="kay",
                available=not (epoch == "04_r2" and cohort == "memoryless"),
            )
            for epoch in ("02_r1", "04_r2")
            for cohort in ("memoryless", "state_space")
        ],
    )

    def test_paired_labels_leave_a_session_without_both_populations(self):
        paired = rows(
            "paired_labels",
            [
                session(
                    epoch,
                    swr_definition="kay",
                    detector_a="memoryless",
                    detector_b="state_space",
                    agreement=agreement,
                )
                for epoch, agreement in (
                    ("02_r1", "agree"),
                    ("02_r1", "unknown"),
                    ("04_r2", "disagree"),
                )
            ],
        )

        by_session = report.paired_labels_by_session(paired, self.MATRIX)
        row = report.labels_by_animal(by_session, "paired_labels").iloc[0]

        assert by_session.available.tolist() == [True, False]
        assert (row.n_sessions, row.n_pairs, row.n_agree, row.n_disagree) == (
            1,
            2,
            1,
            0,
        )
        assert row.n_unknown == 1
        assert row.unavailable_sessions == "sim20200101_04_r2"

    def test_disagreement_leaves_a_session_without_the_population(self):
        labels = rows(
            "definition_disagreement",
            [
                session(epoch, cohort="memoryless", event_key=key, n_unknown=unknown)
                for epoch, key, unknown in (
                    ("02_r1", "a", 0),
                    ("04_r2", "b", 1),
                    ("04_r2", "c", 0),
                )
            ],
        )

        by_session = report.definition_disagreement_by_session(labels, self.MATRIX)
        pooled = report.labels_by_animal(
            by_session, "definition_disagreement"
        ).set_index("cohort")

        memoryless = pooled.loc["memoryless"]
        assert (memoryless.n_sessions, memoryless.n_events) == (1, 1)
        assert memoryless.n_with_unknown == 0
        assert memoryless.unavailable_sessions == "sim20200101_04_r2"
        # a population with no events is pooled with zeros
        assert (pooled.n_sessions["state_space"], pooled.n_events["state_space"]) == (
            2,
            0,
        )


def test_a_detector_with_no_observation_row_is_refused():
    counts = rows("counts", [session("02_r1", detector="kay")])
    availability = report.detector_availability(observed({("02_r1", "yu"): True}))

    with pytest.raises(
        ValueError, match="No observation row for kay in sim20200101_02_r1"
    ):
        report.counts_by_session(counts, availability)


def test_a_run_with_no_ok_epoch_gives_empty_tables(tmp_path, monkeypatch):
    settings = json.loads(
        json.dumps(
            {
                "roster": dict(runner.ROSTER),
                "cells": {
                    name: {"sources": sources, "against": against}
                    for name, (sources, against) in runner.CELLS.items()
                },
                "comparisons": dict(runner.COMPARISONS),
                "no_detector_time": runner.CELL_DETECTORS,
                "recount": {"detector": "state_space"},
            }
        )
    )
    run_dir = tmp_path / "run01"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "settings": settings,
                "epochs": {"sim20200101_02_r1_HPC": {"status": "failed", "error": "e"}},
            }
        )
    )
    monkeypatch.setattr(report, "analysed_epoch_ids", lambda: [])

    written = report.write_report(run_dir, tmp_path / "figures")

    assert len(written["figures"]) == len(report.FIGURE_CAPTIONS)
    for name in report.TABLE_CAPTIONS:
        table = pd.read_csv(run_dir / "report" / f"{name}.csv")
        assert len(table) == (1 if name == "epochs" else 0), name


def test_a_session_without_a_cohorts_matrix_row_is_refused():
    matrix = rows(
        "swr_matrix",
        [
            session("02_r1", cohort="memoryless", swr_definition="kay"),
            session("02_r1", cohort="state_space", swr_definition="kay"),
            session("04_r2", cohort="state_space", swr_definition="kay"),
        ],
    )

    with pytest.raises(ValueError, match="memoryless in sim20200101_04_r2"):
        report.paired_labels_by_session(rows("paired_labels", []), matrix)


def test_rows_with_a_memoryless_inventory_are_marked_pending():
    table = directional(
        [
            {**TWO_SESSIONS[0], "target_detector": target}
            for target in ("kay", "memoryless", "memoryless_criteria")
        ]
    )

    marked = report.with_pending(table, ["source_detector", "target_detector"])

    assert marked.pending.tolist() == ["", report.PENDING, report.PENDING]
    assert report.PENDING == "pending the Phase 2 pilot checks"


# The report on the runner's own tables: two ok epochs, two failed.


@pytest.fixture(scope="module")
def written(run, tmp_path_factory, module_monkeypatch):  # noqa: F811
    module_monkeypatch.setattr(
        report,
        "analysed_epoch_ids",
        lambda: [*EPOCHS.values(), SHIFTED, NO_DECODE, "sim20200101_10_r5_HPC"],
    )
    figure_root = tmp_path_factory.mktemp("figures")
    report.write_report(run, figure_root)
    return run / "report", figure_root / run.name


@pytest.mark.slow
class TestTheReportOnTheRunnersTables:
    def test_every_table_figure_and_caption_is_written(self, written):
        report_dir, figure_dir = written

        for name in report.TABLE_CAPTIONS:
            assert (report_dir / f"{name}.csv").exists(), name
        for name in report.FIGURE_CAPTIONS:
            assert (figure_dir / name).exists(), name
        assert (report_dir / "report.md").exists()
        assert (figure_dir / "captions.md").exists()

    def test_every_caption_carries_the_pending_note(self, written):
        report_dir, figure_dir = written

        for caption in [
            *report.TABLE_CAPTIONS.values(),
            *report.FIGURE_CAPTIONS.values(),
        ]:
            assert report.PENDING in caption
        assert report.PENDING in (report_dir / "report.md").read_text()
        assert report.PENDING in (figure_dir / "captions.md").read_text()

    def test_failed_epochs_are_listed_and_unrun_ones_counted(self, written):
        report_dir, _ = written

        epochs = pd.read_csv(report_dir / "epochs.csv").set_index("epoch_id")
        assert epochs.status.to_dict() == {
            EPOCHS["02_r1"]: "ok",
            EPOCHS["04_r2"]: "ok",
            SHIFTED: "failed",
            NO_DECODE: "failed",
        }
        assert "1 analysed epochs of the epoch register outside the run" in (
            (report_dir / "report.md").read_text()
        )
        assert "position span" in epochs.reason[SHIFTED]
        assert "FileNotFoundError" in epochs.reason[NO_DECODE]
        assert epochs.unavailable_detectors[EPOCHS["04_r2"]] == (
            "memoryless; memoryless_criteria"
        )
        assert epochs.pending[EPOCHS["04_r2"]] == report.PENDING
        assert pd.isna(epochs.pending[EPOCHS["02_r1"]])

    def test_the_pending_marks_of_the_cells_and_exclusion_tables(
        self,
        run,  # noqa: F811
        written,
    ):
        """Every row's mark follows the rule: a cell row is marked when its own
        detector or the cell's against set has a memoryless inventory, and a
        comparison when it holds one, the same on all its rows."""
        report_dir, _ = written
        settings = json.loads((run / "manifest.json").read_text())["settings"]

        def expected(detectors) -> str:
            return report.PENDING if MEMORYLESS_INVENTORIES & set(detectors) else ""

        for level in ("by_session", "by_animal"):
            cells = pd.read_csv(report_dir / f"cells_{level}.csv")
            assert len(cells) > 0
            want = [
                expected([detector, *settings["cells"][cell]["against"]])
                for cell, detector in zip(cells.cell, cells.detector, strict=True)
            ]
            assert cells.pending.fillna("").tolist() == want, level
            # both kinds occur, so the comparison is not vacuous
            assert set(want) == {"", report.PENDING}

            exclusion = pd.read_csv(report_dir / f"exclusion_{level}.csv")
            assert set(exclusion.comparison) == set(settings["comparisons"])
            want = [
                expected(settings["comparisons"][comparison])
                for comparison in exclusion.comparison
            ]
            assert exclusion.pending.fillna("").tolist() == want, level
            assert set(want) == {"", report.PENDING}
            assert (
                exclusion.assign(pending=exclusion.pending.fillna(""))
                .groupby("comparison")
                .pending.nunique()
                .eq(1)
                .all()
            )

    def test_the_criteria_twin_keeps_its_durations_and_its_sequence_cell(
        self,
        run,  # noqa: F811
        written,
    ):
        """The components leave memoryless_criteria out, but its compared
        events keep their duration rows, and it is a source of the sequences
        cell, pooled apart from memoryless."""
        report_dir, _ = written
        inventory = pd.read_parquet(run / f"{EPOCHS['02_r1']}_inventory.parquet")

        durations = pd.read_csv(
            report_dir / "native_durations_by_session.csv"
        ).set_index(["epoch", "detector"])
        compared = inventory[inventory.whole_event_speed_eligible]
        for detector in ("memoryless", "memoryless_criteria", "state_space"):
            assert (
                durations.n_events[("02_r1", detector)]
                == (compared.detector == detector).sum()
            ), detector
        assert durations.n_events[("02_r1", "memoryless_criteria")] == 2
        cells = pd.read_csv(report_dir / "cells_by_animal.csv").set_index(
            ["cell", "detector"]
        )
        sources = cells.loc["sequence_matched_by_no_trigger"]
        assert set(sources.index) == {
            "state_space",
            "memoryless",
            "memoryless_criteria",
        }
        assert sources.n_unmatched["memoryless_criteria"] == 2
        assert sources.pending["memoryless_criteria"] == report.PENDING

    def test_the_label_tables_list_the_session_without_memoryless(self, written):
        report_dir, _ = written

        paired = pd.read_csv(report_dir / "paired_labels_by_animal.csv")
        disagreement = pd.read_csv(
            report_dir / "definition_disagreement_by_animal.csv"
        ).set_index("cohort")

        assert set(paired.unavailable_sessions) == {f"{ANIMAL}{DAY}_04_r2"}
        assert disagreement.unavailable_sessions["memoryless"] == f"{ANIMAL}{DAY}_04_r2"
        assert pd.isna(disagreement.unavailable_sessions["state_space"])

    def test_the_animal_counts_are_the_session_counts_summed(self, run, written):  # noqa: F811
        report_dir, _ = written

        sessions = pd.concat(
            pd.read_parquet(run / f"{identifier}_counts.parquet")
            for identifier in EPOCHS.values()
        )
        animal = pd.read_csv(report_dir / "counts_by_animal.csv").set_index("detector")
        # kay looked in both epochs; memoryless only in the first
        assert (
            animal.n_native["kay"]
            == sessions[sessions.detector == "kay"].n_native.sum()
        )
        memoryless = animal.loc["memoryless"]
        assert memoryless.n_sessions == 1
        assert memoryless.unavailable_sessions == f"{ANIMAL}{DAY}_04_r2"
        assert memoryless.pending == report.PENDING
        assert pd.isna(animal.pending["kay"])
        assert animal.native_note["mua"] and animal.native_note["state_space"]

    def test_an_epoch_without_memoryless_reads_unknown_not_unmatched(
        self,
        run,  # noqa: F811
        written,
    ):
        report_dir, _ = written

        cells = pd.concat(
            pd.read_parquet(run / f"{identifier}_cells.parquet")
            for identifier in EPOCHS.values()
        )
        by_animal = pd.read_csv(report_dir / "cells_by_animal.csv").set_index(
            ["cell", "detector"]
        )
        for (cell, detector), row in by_animal.iterrows():
            source = cells[(cells.cell == cell) & (cells.detector == detector)]
            assert row.n_unmatched == (source.membership == "unmatched").sum()
            assert row.n_unknown == (source.membership == "unknown").sum()
        assert by_animal.n_unknown.sum() > 0

    def test_main_reads_the_run_named(self, run, tmp_path, monkeypatch):  # noqa: F811
        monkeypatch.setattr(report, "FIGURE_ROOT", tmp_path / "figures")

        report.main(["--run-id", run.name])

        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["run_id"] == run.name
        assert (tmp_path / "figures" / run.name / "native_durations.png").exists()


def _pair(key_a, key_b, cover_a, cover_b, iou, onset, offset, *, within_a=True):
    return session(
        "02_r1",
        **{
            "detector_a": "kay",
            "event_key_a": key_a,
            "detector_b": "zugaro",
            "event_key_b": key_b,
            "iou": iou,
            "fraction_a_covered": cover_a,
            "fraction_b_covered": cover_b,
            "onset_difference_s": onset,
            "offset_difference_s": offset,
            "a_within_shared_exposure": within_a,
            "b_within_shared_exposure": True,
        },
    )


def _directional(source, target, n_eligible, n_matched):
    return session(
        "02_r1",
        source_detector=source,
        target_detector=target,
        available=True,
        n_eligible=n_eligible,
        n_matched=n_matched,
    )


def test_match_strength_counts_each_source_event_by_its_best_match():
    """A short zugaro event nested in a kay event is fully covered (cover 1.0)
    while the kay event is only partly covered; a kay event touched by two
    zugaro events counts once, by its best match; a kay event outside the
    shared exposure does not count. Offsets read target less source."""
    pairs = rows(
        "pairs",
        [
            # k1 wraps z1: z1 starts 30 ms later and ends 20 ms earlier
            _pair("k1", "z1", 0.4, 1.0, 0.4, 0.03, -0.02),
            # k2 touched by z2 (sliver) and z3 (most of it)
            _pair("k2", "z2", 0.1, 0.2, 0.08, 0.05, 0.1),
            _pair("k2", "z3", 0.95, 0.9, 0.86, -0.004, 0.002),
            # k3 lies outside the shared exposure
            _pair("k3", "z4", 1.0, 1.0, 1.0, 0.0, 0.0, within_a=False),
        ],
    )
    directional = pd.DataFrame(
        [_directional("kay", "zugaro", 4, 2), _directional("zugaro", "kay", 4, 4)]
    )
    out = report.match_strength_by_session(pairs, directional).set_index(
        ["source_detector", "target_detector"]
    )
    kay = out.loc[("kay", "zugaro")]
    assert (kay.n_matched, kay.n_covered_50pct, kay.n_covered_90pct) == (2, 1, 1)
    assert kay.n_iou_50pct == 1
    assert kay.p_covered_50pct == pytest.approx(0.25)
    zugaro = out.loc[("zugaro", "kay")]
    # z1, z3 and z4 are covered (z4 lies within the shared exposure); z3 and
    # z4 reach the IoU
    assert (zugaro.n_covered_50pct, zugaro.n_covered_90pct, zugaro.n_iou_50pct) == (
        3,
        3,
        2,
    )

    oriented = report.oriented_pairs(pairs)
    z1 = oriented[oriented.source_event_key == "z1"].iloc[0]
    # from z1's side, kay starts 30 ms earlier and ends 20 ms later
    assert (z1.onset_difference_s, z1.offset_difference_s) == pytest.approx(
        (-0.03, 0.02)
    )


def test_match_strength_refuses_a_directional_count_it_cannot_reproduce():
    pairs = rows("pairs", [_pair("k1", "z1", 0.4, 1.0, 0.4, 0.03, -0.02)])
    directional = pd.DataFrame(
        [_directional("kay", "zugaro", 3, 2), _directional("zugaro", "kay", 3, 1)]
    )
    with pytest.raises(ValueError, match="n_matched 2 in the directional table"):
        report.match_strength_by_session(pairs, directional)
