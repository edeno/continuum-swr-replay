"""The populations of the compared events, and the memberships they read.

One simulated epoch (0-100 s on a 500 Hz grid) is compared as the comparison
runner compares one: detector ``a`` observes all of it, ``b`` all but
40-41 s, and ``c`` did not run. The animal moves over 60-65 s. ``a`` finds
events at 10.0, 20.0, 40.5 and 62.0 s (the last one moving); ``b`` at 10.1
and 30.0 s, and at 39.95-41.05 s, which bridges its own gap and is set aside;
``a``'s event at 70.0 s was dropped by the analyzed-time rule.
"""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detector_comparison.availability import (
    coverage_from_observations,
    missing_run_table,
    observation_table,
)
from continuum_swr_replay.analysis.detector_comparison.cells import (
    membership_against,
)
from continuum_swr_replay.analysis.detector_comparison.overlap import (
    SET_ASIDE_DTYPES,
    directional_agreement,
    event_matches,
    flag_pairs,
    overlap_pairs,
    remove_bridging_events,
    set_aside_events,
)
from continuum_swr_replay.analysis.detector_comparison.populations import (
    POPULATION_DTYPES,
    _require_same,
    compared_events,
    population_rows,
    rebuild_matches,
)
from continuum_swr_replay.analysis.detector_comparison.schema import (
    SessionIdentity,
    combine_inventories,
    inventory_from_intervals,
)

SESSION = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
KEY = ("sim", 20200101, "02_r1")
KEY_COLUMNS = ("animal", "day", "epoch")
TIME = np.arange(50_001) * 0.002
SPEED = np.where((TIME >= 60.0) & (TIME < 65.0), 10.0, 0.0)
DETECTORS = ("a", "b", "c")
CELLS = {
    "exactly_one_detector": (DETECTORS, DETECTORS),
    "a_matched_by_no_b": (("a",), ("b",)),
}


def events(detector, intervals):
    starts, ends = np.array(intervals).T
    return inventory_from_intervals(
        starts,
        ends,
        native_keys=np.arange(starts.size),
        detector=detector,
        session=SESSION,
        run_id="r",
        speed_time_s=TIME,
        speed_cm_per_s=SPEED,
    )


def key_of(detector, number):
    return f"{detector}:sim20200101_02_r1:{number}"


@pytest.fixture(scope="module")
def epoch():
    """The epoch's tables as the runner stores them, its still intervals, and
    the coverage and memberships the runner compared them with."""
    observations = pd.concat(
        [
            observation_table(
                SESSION,
                detector=detector,
                run_id="r",
                epoch_bounds=(0.0, 100.0),
                observed=observed,
                unavailable_reason="missing",
            )
            for detector, observed in (
                ("a", np.array([[0.0, 100.0]])),
                ("b", np.array([[0.0, 40.0], [41.0, 100.0]])),
            )
        ]
        + [
            missing_run_table(
                SESSION, detector="c", run_id="r", epoch_bounds=(0.0, 100.0), reason="x"
            )
        ],
        ignore_index=True,
    )
    dropped = pd.DataFrame(
        {
            "animal": ["sim"],
            "day": [20200101],
            "epoch": ["02_r1"],
            "detector": ["a"],
            "event_key": ["a:sim20200101_02_r1:dropped"],
            "interval_start_s": [70.0],
            "interval_end_s": [70.2],
            "reason": ["artifact"],
        }
    ).astype(dict(SET_ASIDE_DTYPES))
    still = {KEY: np.array([[0.0, 60.0], [64.998, 100.0]])}
    eligible, moving = combine_inventories(
        [
            events("a", [(10.0, 10.2), (20.0, 20.2), (40.5, 40.6), (62.0, 62.1)]),
            events("b", [(10.1, 10.3), (30.0, 30.1), (39.95, 41.05)]),
        ]
    )
    coverage = set_aside_events(
        coverage_from_observations(observations, still), dropped
    )
    kept, coverage, bridging = remove_bridging_events(eligible, coverage)
    set_aside = pd.concat(
        [
            dropped,
            pd.DataFrame(
                {
                    **{c: bridging[c] for c in (*KEY_COLUMNS, "detector", "event_key")},
                    "interval_start_s": bridging.event_start_s,
                    "interval_end_s": bridging.event_end_s,
                    "reason": "bridging",
                }
            ).astype(dict(SET_ASIDE_DTYPES)),
        ],
        ignore_index=True,
    )
    pairs = overlap_pairs(kept)
    matches = event_matches(kept, pairs, coverage, detectors=DETECTORS)
    cells = pd.concat(
        [
            membership_against(kept, matches, sources=sources, against=against).assign(
                cell=name
            )
            for name, (sources, against) in CELLS.items()
        ],
        ignore_index=True,
    )
    tables = {
        "observations": observations,
        "set_aside": set_aside,
        "inventory": pd.concat([kept, moving], ignore_index=True).sort_values(
            ["detector", "event_start_s"], ignore_index=True
        ),
        "pairs": flag_pairs(pairs, matches),
        "directional": directional_agreement(
            kept, pairs, coverage, detectors=DETECTORS
        ),
        "cells": cells[["cell", *cells.columns.drop("cell")]],
    }
    return tables, still, coverage, matches


def _sorted(frame, columns):
    return frame.sort_values(list(columns), ignore_index=True)


class TestRebuildMatches:
    def test_it_reproduces_the_coverage_and_the_memberships(self, epoch):
        tables, still, coverage, matches = epoch

        rebuilt, recomputed = rebuild_matches(
            tables, still, detectors=DETECTORS, cells=CELLS
        )

        assert set(rebuilt) == set(coverage)
        for key, entry in coverage.items():
            np.testing.assert_array_equal(rebuilt[key].observed, entry.observed)
            np.testing.assert_array_equal(rebuilt[key].exposure, entry.exposure)
        pd.testing.assert_frame_equal(
            _sorted(recomputed, ["event_key", "other_detector"]),
            _sorted(matches, ["event_key", "other_detector"]),
        )
        # the moving event is not compared
        assert key_of("a", 3) not in set(recomputed.event_key)

    @pytest.mark.parametrize("name", ["directional", "cells"])
    def test_a_stored_table_it_does_not_reproduce_is_refused(self, epoch, name):
        tables, still, _, _ = epoch
        changed = tables[name].copy()
        if name == "cells":
            changed.loc[changed.membership == "unmatched", "membership"] = "matched"
        else:
            changed.loc[0, "n_matched"] += 1

        with pytest.raises(ValueError, match=name):
            rebuild_matches(
                {**tables, name: changed}, still, detectors=DETECTORS, cells=CELLS
            )

    def test_a_dtype_clash_is_refused_by_name_not_a_type_error(self):
        # categoricals of other categories cannot be compared: != raises a
        # TypeError, and the refusal must still be the named one
        rebuilt = pd.DataFrame({"k": [0, 1], "m": pd.Categorical(["a", "b"])})
        stored = pd.DataFrame({"k": [0, 1], "m": pd.Categorical(["a", "c"])})

        with pytest.raises(ValueError, match="stored cells table.*in dtype"):
            _require_same("cells", rebuilt, stored, ["k"])

    def test_stored_pair_flags_it_does_not_reproduce_are_refused(self, epoch):
        tables, still, _, _ = epoch
        pairs = tables["pairs"].copy()
        pairs.loc[0, "a_within_shared_exposure"] = ~pairs.loc[
            0, "a_within_shared_exposure"
        ]

        with pytest.raises(ValueError, match="stored pairs table"):
            rebuild_matches(
                {**tables, "pairs": pairs}, still, detectors=DETECTORS, cells=CELLS
            )

    def test_coverage_without_the_set_aside_spans_is_refused(self, epoch):
        """Without the bridging row, b's bridging span beside its gap reads as
        exposure it shares with a, so the pair's shared exposure differs."""
        tables, still, _, _ = epoch
        set_aside = tables["set_aside"]

        with pytest.raises(ValueError, match="directional|cells"):
            rebuild_matches(
                {**tables, "set_aside": set_aside[set_aside.reason != "bridging"]},
                still,
                detectors=DETECTORS,
                cells=CELLS,
            )


class TestPopulationRows:
    @pytest.fixture(scope="class")
    def rows(self, epoch):
        tables, _, _, matches = epoch
        inventory = tables["inventory"]
        compared = inventory[inventory.whole_event_speed_eligible]
        return population_rows(
            compared, matches, tables["cells"], tables["directional"]
        )

    def populations(self, rows, key):
        these = rows[rows.event_key == key]
        assert not these.population.duplicated().any(), key
        return dict(zip(these.population, these.unknown_reason, strict=True))

    def test_columns_and_dtypes(self, rows):
        assert list(rows.columns) == list(POPULATION_DTYPES)
        assert rows.dtypes.to_dict() == {
            name: pd.api.types.pandas_dtype(dtype)
            for name, dtype in POPULATION_DTYPES.items()
        }

    def test_all_holds_every_compared_event_once(self, rows, epoch):
        everyone = rows[rows.population == "all"]

        assert sorted(everyone.event_key) == sorted(
            [key_of("a", 0), key_of("a", 1), key_of("a", 2)]
            + [key_of("b", 0), key_of("b", 1)]
        )

    def test_each_events_populations_with_the_reasons_it_is_unknown(self, rows):
        assert self.populations(rows, key_of("a", 0)) == {
            "all": "",
            "matched_by:b": "",
            "unknown_by:c": "no_shared_exposure",
            "exactly_one_detector:matched": "",
            "a_matched_by_no_b:matched": "",
        }
        assert self.populations(rows, key_of("a", 1)) == {
            "all": "",
            "unmatched_by:b": "",
            "unknown_by:c": "no_shared_exposure",
            "exactly_one_detector:unknown": "",
            "a_matched_by_no_b": "",
        }
        # in b's gap and its bridging event's span
        assert self.populations(rows, key_of("a", 2)) == {
            "all": "",
            "unknown_by:b": "outside_shared_exposure",
            "unknown_by:c": "no_shared_exposure",
            "exactly_one_detector:unknown": "",
            "a_matched_by_no_b:unknown": "",
        }

    def test_an_unknown_membership_never_enters_an_unmatched_population(
        self, rows, epoch
    ):
        _, _, _, matches = epoch
        unknown = matches[matches.membership == "unknown"]
        unmatched = rows[rows.population.str.startswith("unmatched_by:")]

        pairs = set(zip(unmatched.event_key, unmatched.population, strict=True))
        assert not any(
            (key, f"unmatched_by:{other}") in pairs
            for key, other in zip(
                unknown.event_key, unknown.other_detector, strict=True
            )
        )
        assert (
            (rows.unknown_reason != "")
            .eq(rows.population.str.startswith("unknown_by:"))
            .all()
        )

    def test_each_event_and_population_is_one_row(self, rows):
        assert not rows.duplicated(["event_key", "population"]).any()

    def test_cells_of_other_events_are_refused(self, epoch):
        tables, _, _, matches = epoch
        compared = tables["inventory"][tables["inventory"].whole_event_speed_eligible]
        cells = tables["cells"].copy()
        cells.loc[0, "event_key"] = "a:sim20200101_02_r1:elsewhere"

        with pytest.raises(ValueError, match="cells"):
            population_rows(compared, matches, cells, tables["directional"])

    def test_a_membership_with_no_directional_row_is_refused(self, epoch):
        tables, _, _, matches = epoch
        directional = tables["directional"]

        with pytest.raises(ValueError, match="directional"):
            population_rows(
                tables["inventory"][tables["inventory"].whole_event_speed_eligible],
                matches,
                tables["cells"],
                directional[directional.target_detector != "c"],
            )


def test_the_compared_events_are_the_whole_event_immobile_ones_indexed_from_zero():
    inventory = pd.DataFrame(
        {
            "event_key": ["a", "b", "c", "d"],
            "whole_event_speed_eligible": [False, True, False, True],
        },
        index=[10, 11, 12, 13],
    )
    compared = compared_events(inventory)
    assert compared.event_key.tolist() == ["b", "d"]
    assert compared.index.tolist() == [0, 1]
