"""Sessions pooled within an animal, on tables small enough to check by hand.

An animal sums its available sessions' counts, never averages their
proportions; an unavailable session is left out of the sums and listed; an
``unknown`` membership is never counted as ``unmatched``. The per-epoch tables
are built with the inventory runner's own columns and dtypes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detector_comparison import inventory_run, pooling

DEFAULTS = {object: "", np.int64: 0, np.float64: 0.0, np.bool_: True, "Int64": pd.NA}


def rows(name: str, records: list[dict]) -> pd.DataFrame:
    """Table ``name`` as the runner types it; unnamed columns take a default."""
    dtypes = inventory_run.TABLE_DTYPES[name]
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


class TestPoolSessions:
    # two animals with unequal sessions; group "b" of animal "x" was never
    # available; animal "y" has one session holding two rows of group "a"
    FRAME = pd.DataFrame(
        {
            "animal": ["x", "x", "x", "x", "y", "y"],
            "day": [1, 1, 2, 2, 5, 5],
            "epoch": ["02_r1", "04_r2", "02_r1", "02_r1", "02_r1", "02_r1"],
            "group": ["a", "a", "a", "b", "a", "a"],
            "n": np.array([1, 9, 50, 4, 2, 0], dtype=np.int64),
            "seconds": [10.0, 90.0, 5.0, 1.0, 3.5, 0.5],
        }
    )
    AVAILABLE = np.array([True, True, False, False, True, True])

    def pooled(self) -> pd.DataFrame:
        return pooling.pool_sessions(
            self.FRAME, by=["group"], sums=["n", "seconds"], available=self.AVAILABLE
        ).set_index(["animal", "group"])

    def test_sums_the_available_sessions_and_lists_the_others(self):
        pooled = self.pooled()

        assert pooled.loc[("x", "a"), "n_sessions"] == 2
        assert pooled.loc[("x", "a"), "n"] == 10
        assert pooled.loc[("x", "a"), "seconds"] == pytest.approx(100.0)
        assert pooled.loc[("x", "a"), "unavailable_sessions"] == "x2_02_r1"

    def test_a_group_with_no_available_session_is_zero_and_listed(self):
        row = self.pooled().loc[("x", "b")]

        assert (row.n_sessions, row.n, row.seconds) == (0, 0, 0.0)
        assert row.unavailable_sessions == "x2_02_r1"

    def test_a_session_counts_once_however_many_rows_it_has(self):
        row = self.pooled().loc[("y", "a")]

        assert (row.n_sessions, row.n) == (1, 2)
        assert row.unavailable_sessions == ""

    def test_keeps_the_sums_dtypes(self):
        pooled = self.pooled()

        assert pooled.n.dtype == np.int64
        assert pooled.n_sessions.dtype == np.int64
        assert pooled.seconds.dtype == np.float64

    def test_summed_leaves_no_session_out(self):
        pooled = pooling.summed(self.FRAME, ["group"], ["n"]).set_index(
            ["animal", "group"]
        )

        assert pooled.loc[("x", "a"), "n"] == 60
        assert pooled.loc[("x", "a"), "n_sessions"] == 3
        assert "unavailable_sessions" not in pooled.columns


def test_a_ratio_with_no_denominator_is_nan_not_zero():
    out = pooling.ratio(pd.Series([1, 0, 3]), pd.Series([4, 0, -1]))

    assert out[0] == pytest.approx(0.25)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_a_detector_unavailable_in_a_session_leaves_its_counts():
    counts = rows(
        "counts",
        [
            session("02_r1", detector="kay", n_native=4, n_eligible=4),
            session("04_r2", detector="kay", n_native=6, n_eligible=6),
            session("06_r3", detector="kay", n_native=7, n_eligible=7),
        ],
    )
    availability = pooling.detector_availability(
        observed(
            {
                ("02_r1", "kay"): True,
                ("04_r2", "kay"): True,
                ("06_r3", "kay"): False,
            }
        )
    )

    row = pooling.counts_by_animal(
        pooling.counts_by_session(counts, availability)
    ).iloc[0]
    assert (row.n_sessions, row.n_native, row.n_eligible) == (2, 10, 10)
    assert row.unavailable_sessions == "sim20200101_06_r3"


def test_the_rate_is_over_the_summed_exposure():
    counts = rows(
        "counts",
        [
            # 1 in 10 s and 9 in 30 s: 10 in 40 s, where the sessions' rates
            # average to 0.2
            session("02_r1", detector="kay", n_in_exposure=1, exposure_s=10.0),
            session("04_r2", detector="kay", n_in_exposure=9, exposure_s=30.0),
        ],
    )
    availability = pooling.detector_availability(
        observed({("02_r1", "kay"): True, ("04_r2", "kay"): True})
    )

    row = pooling.counts_by_animal(
        pooling.counts_by_session(counts, availability)
    ).iloc[0]
    assert row.exposure_s == pytest.approx(40.0)
    assert row.rate_per_s == pytest.approx(0.25)


def test_a_detector_with_no_observation_row_is_refused():
    counts = rows("counts", [session("02_r1", detector="kay")])
    availability = pooling.detector_availability(observed({("02_r1", "yu"): True}))

    with pytest.raises(
        ValueError, match="No observation row for kay in sim20200101_02_r1"
    ):
        pooling.counts_by_session(counts, availability)


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
        availability = pooling.detector_availability(observed({("02_r1", "kay"): True}))

        by_session = pooling.cells_by_session(self.CELLS, availability, self.SOURCES)
        by_animal = pooling.cells_by_animal(by_session)

        for table in (by_session, by_animal):
            row = table.iloc[0]
            assert (row.n_matched, row.n_unmatched, row.n_unknown) == (1, 1, 2)
            assert row.n_events == 4

    def test_a_session_without_its_source_detector_is_listed(self):
        availability = pooling.detector_availability(
            observed({("02_r1", "kay"): True, ("04_r2", "kay"): False})
        )

        by_session = pooling.cells_by_session(self.CELLS, availability, self.SOURCES)
        row = pooling.cells_by_animal(by_session).iloc[0]

        assert len(by_session) == 2
        assert (row.n_sessions, row.n_unmatched, row.n_unknown) == (1, 1, 2)
        assert row.unavailable_sessions == "sim20200101_04_r2"

    def test_a_membership_outside_the_three_is_refused(self):
        cells = self.CELLS.assign(membership="maybe")
        availability = pooling.detector_availability(observed({("02_r1", "kay"): True}))

        with pytest.raises(ValueError, match="are not"):
            pooling.cells_by_session(cells, availability, self.SOURCES)


class TestDirectionalByAnimal:
    """An animal's directional agreement pools its sessions' counts."""

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

    @staticmethod
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

    def test_by_summing_counts_not_averaging_proportions(self):
        pooled = pooling.directional_by_animal(self.directional(self.TWO_SESSIONS))

        row = pooled.iloc[0]
        assert (row.n_sessions, row.n_eligible, row.n_matched) == (2, 100, 18)
        assert row.n_excluded == 5
        assert row.proportion_matched == pytest.approx(0.18)
        assert np.mean([9 / 10, 9 / 90]) == pytest.approx(0.5)

    def test_an_unavailable_session_is_left_out_and_listed(self):
        pooled = pooling.directional_by_animal(
            self.directional([*self.TWO_SESSIONS, self.UNAVAILABLE])
        )

        row = pooled.iloc[0]
        assert (row.n_sessions, row.n_eligible, row.n_matched) == (2, 100, 18)
        assert row.proportion_matched == pytest.approx(0.18)
        assert row.unavailable_sessions == "sim20200101_06_r3"
