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

import scripts.qa.report_inventory_comparison as report
import scripts.qa.run_inventory_comparison as runner

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
    figure_dir = tmp_path_factory.mktemp("figures")
    report.write_report(run, figure_dir)
    return run / "report", figure_dir


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

    def test_failed_and_unrun_epochs_are_listed(self, written):
        report_dir, _ = written

        epochs = pd.read_csv(report_dir / "epochs.csv").set_index("epoch_id")
        assert epochs.status.to_dict() == {
            EPOCHS["02_r1"]: "ok",
            EPOCHS["04_r2"]: "ok",
            SHIFTED: "failed",
            NO_DECODE: "failed",
            "sim20200101_10_r5_HPC": "not_run",
        }
        assert "position span" in epochs.reason[SHIFTED]
        assert "FileNotFoundError" in epochs.reason[NO_DECODE]
        assert epochs.unavailable_detectors[EPOCHS["04_r2"]] == (
            "memoryless; memoryless_criteria"
        )

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
        assert (tmp_path / "figures" / "native_durations.png").exists()
