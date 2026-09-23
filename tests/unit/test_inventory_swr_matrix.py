"""The cohort x SWR-definition label matrix."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    Coverage,
    event_matches,
    overlap_pairs,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    SessionIdentity,
    inventory_from_intervals,
)
from continuum_swr_replay.analysis.inventory_comparison.swr_matrix import (
    MATRIX_DTYPES,
    definition_disagreement,
    paired_labels,
    swr_label_matrix,
    swr_labels,
)

GRID = np.arange(0, 100.001, 0.002)
EPOCH = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
KEY = ("sim", 20200101, "02_r1")
COHORTS = ["S1", "S2"]
DEFINITIONS = ["K", "Y"]


def key(detector, native):
    """The event key of an event in the test epoch."""
    return f"{detector}:sim20200101_02_r1:{native}"


def events(detector, spans):
    return inventory_from_intervals(
        np.array([a for a, _ in spans], dtype=float),
        np.array([b for _, b in spans], dtype=float),
        native_keys=np.arange(len(spans)),
        detector=detector,
        session=EPOCH,
        run_id="run",
        speed_time_s=GRID,
        speed_cm_per_s=np.zeros(GRID.size),
    )


def cover(exposure=((0.0, 100.0),)):
    return Coverage(observed=np.array([[0.0, 100.0]]), exposure=np.array(exposure))


NOWHERE = Coverage(np.empty((0, 2)), np.empty((0, 2)))
"""A detector on the roster whose run produced nothing in the epoch."""


def matrix_of(labels, coverage, cohorts=COHORTS, definitions=DEFINITIONS):
    return swr_label_matrix(
        labels, coverage, cohorts=cohorts, swr_definitions=definitions
    )


@pytest.fixture
def labelled():
    """S1: one SWR event under both definitions, one under K only, and one
    that K calls SWR during a gap in Y's LFP. S2's first event overlaps S1's
    first but not its SWR; its second overlaps S1's third, inside Y's gap."""
    inventory = pd.concat(
        [
            events("S1", [(10, 20), (40, 50), (70, 80)]),
            events("S2", [(18, 30), (70.5, 85)]),
            events("K", [(11, 12), (41, 42), (71, 72)]),
            events("Y", [(11, 12)]),
        ],
        ignore_index=True,
    )
    coverage = {(*KEY, d): cover() for d in ["S1", "S2", "K"]} | {
        (*KEY, "Y"): cover(((0.0, 60.0), (90.0, 100.0)))
    }
    pairs = overlap_pairs(inventory)
    matches = event_matches(inventory, pairs, coverage)
    labels = swr_labels(
        inventory, matches, coverage, cohorts=COHORTS, swr_definitions=DEFINITIONS
    )
    return pairs, labels, coverage


def label_of(labels, key, definition):
    row = labels[(labels.event_key == key) & (labels.swr_definition == definition)]
    return row.iloc[0]


def test_labels_follow_strict_overlap_and_coverage(labelled):
    _, labels, _ = labelled

    assert label_of(labels, key("S1", 0), "K").label == "SWR"
    assert label_of(labels, key("S1", 1), "K").label == "SWR"
    assert label_of(labels, key("S1", 1), "Y").label == "non-SWR"
    assert label_of(labels, key("S1", 2), "K").label == "SWR"
    # during missing LFP the status is unknown, never non-SWR
    assert label_of(labels, key("S1", 2), "Y").label == "unknown"


def test_the_matrix_uses_the_same_events_in_every_column(labelled):
    """S1:2 is observed by K but not by Y, so it leaves both columns."""
    _, labels, coverage = labelled

    matrix = matrix_of(labels, coverage).set_index(["cohort", "swr_definition"])

    assert matrix.available.all()
    s1_k, s1_y = matrix.loc[("S1", "K")], matrix.loc[("S1", "Y")]
    assert (s1_k.n_events, s1_k.n_eligible, s1_k.n_excluded) == (3, 2, 1)
    assert s1_y.n_eligible == s1_k.n_eligible
    assert (s1_k.n_swr, s1_k.n_non_swr) == (2, 0)
    assert (s1_y.n_swr, s1_y.n_non_swr) == (1, 1)
    assert s1_y.non_swr_fraction == pytest.approx(0.5)
    assert matrix.loc[("S2", "K")].non_swr_fraction == pytest.approx(1.0)


def test_the_matrix_carries_its_declared_columns(labelled):
    _, labels, coverage = labelled

    matrix = matrix_of(labels, coverage)

    assert list(matrix.columns) == list(MATRIX_DTYPES)
    for column, dtype in MATRIX_DTYPES.items():
        assert matrix[column].dtype == dtype, column


def test_a_cohort_with_no_eligible_event_has_no_fraction():
    inventory = pd.concat(
        [events("S1", [(10, 20)]), events("K", [])], ignore_index=True
    )
    coverage = {(*KEY, "S1"): cover(), (*KEY, "K"): cover(((50.0, 100.0),))}
    matches = event_matches(inventory, overlap_pairs(inventory), coverage)

    matrix = matrix_of(
        swr_labels(inventory, matches, coverage, cohorts=["S1"], swr_definitions=["K"]),
        coverage,
        ["S1"],
        ["K"],
    ).iloc[0]

    assert (matrix.n_eligible, matrix.n_excluded) == (0, 1)
    assert np.isnan(matrix.non_swr_fraction)


def test_a_cohort_that_looked_and_found_nothing_has_a_row_of_zeros():
    """Zero found is not the same as not run: the row is there and available."""
    inventory = pd.concat([events("S1", []), events("K", [(1, 2)])], ignore_index=True)
    coverage = {(*KEY, "S1"): cover(), (*KEY, "K"): cover()}
    matches = event_matches(inventory, overlap_pairs(inventory), coverage)

    matrix = matrix_of(
        swr_labels(inventory, matches, coverage, cohorts=["S1"], swr_definitions=["K"]),
        coverage,
        ["S1"],
        ["K"],
    ).iloc[0]

    assert matrix.available
    assert (matrix.n_events, matrix.n_eligible, matrix.n_swr, matrix.n_non_swr) == (
        0,
        0,
        0,
        0,
    )


def test_a_cohort_and_a_definition_must_be_different_detectors():
    inventory = events("S1", [(10, 20)])
    coverage = {(*KEY, "S1"): cover(), (*KEY, "K"): cover()}
    matches = event_matches(inventory, overlap_pairs(inventory), coverage)

    with pytest.raises(ValueError, match="both a cohort"):
        swr_labels(
            inventory, matches, coverage, cohorts=["S1"], swr_definitions=["S1", "K"]
        )
    with pytest.raises(ValueError, match="roster"):
        swr_labels(inventory, matches, coverage, cohorts=["S1"], swr_definitions=["k"])


def test_a_matched_pair_may_carry_different_labels_and_the_table_keeps_it(labelled):
    pairs, labels, _ = labelled

    paired = paired_labels(pairs, labels, cohorts=COHORTS).set_index(
        ["event_key_a", "event_key_b", "swr_definition"]
    )

    first = paired.loc[(key("S1", 0), key("S2", 0), "K")]
    assert (first.label_a, first.label_b, first.agreement) == (
        "SWR",
        "non-SWR",
        "disagree",
    )
    assert paired.loc[(key("S1", 0), key("S2", 0), "Y")].agreement == "disagree"
    assert paired.loc[(key("S1", 2), key("S2", 1), "K")].agreement == "agree"
    # inside Y's gap neither label is known, so neither is their agreement
    assert paired.loc[(key("S1", 2), key("S2", 1), "Y")].agreement == "unknown"
    assert len(paired) == 4


def test_where_definitions_disagree_about_an_event_it_is_reported(labelled):
    _, labels, _ = labelled

    disagreement = definition_disagreement(labels).set_index("event_key")

    assert disagreement.loc[key("S1", 0), "definitions_disagree"] == False  # noqa: E712
    assert disagreement.loc[key("S1", 1), "definitions_disagree"] == True  # noqa: E712
    # an unknown label is not a disagreement
    assert disagreement.loc[key("S1", 2), "definitions_disagree"] == False  # noqa: E712
    assert disagreement.loc[key("S1", 2), ["n_swr", "n_unknown"]].tolist() == [1, 1]


def test_a_definition_that_did_not_run_labels_nothing_non_swr():
    """On the roster with empty coverage, as a missing run is registered."""
    inventory = events("S1", [(10, 20)])
    coverage = {(*KEY, "S1"): cover(), (*KEY, "K"): NOWHERE}
    matches = event_matches(inventory, overlap_pairs(inventory), coverage)

    labels = swr_labels(
        inventory, matches, coverage, cohorts=["S1"], swr_definitions=["K"]
    )
    matrix = matrix_of(labels, coverage, ["S1"], ["K"]).iloc[0]

    assert labels.label.tolist() == ["unknown"]
    assert not labels.in_matrix.any()
    assert not matrix.available


def test_a_sequence_only_partly_observed_by_the_swr_detector_is_unknown():
    """An overlap in the observed part cannot tell what the rest held; the
    pair table keeps the overlap."""
    inventory = pd.concat(
        [events("S1", [(49, 51)]), events("K", [(49.5, 49.8)])], ignore_index=True
    )
    coverage = {
        (*KEY, "S1"): cover(),
        (*KEY, "K"): Coverage(np.array([[0.0, 50.0]]), np.array([[0.0, 50.0]])),
    }
    pairs = overlap_pairs(inventory)
    matches = event_matches(inventory, pairs, coverage)

    labels = swr_labels(
        inventory, matches, coverage, cohorts=["S1"], swr_definitions=["K"]
    )

    assert len(pairs) == 1
    assert labels.label.tolist() == ["unknown"]


def test_a_definition_missing_in_an_epoch_leaves_every_column_unavailable():
    """Every column counts the same events, those inside the cohort's
    exposure and every definition's; one definition not running empties that
    population for all of them."""
    inventory = pd.concat(
        [events("S1", [(10, 20), (30, 40)]), events("K", [(12, 13)])],
        ignore_index=True,
    )
    coverage = {(*KEY, "S1"): cover(), (*KEY, "K"): cover()} | {
        (*KEY, "Y"): Coverage(np.empty((0, 2)), np.empty((0, 2)))
    }
    matches = event_matches(inventory, overlap_pairs(inventory), coverage)
    labels = swr_labels(
        inventory, matches, coverage, cohorts=["S1"], swr_definitions=["K", "Y"]
    )

    matrix = swr_label_matrix(
        labels, coverage, cohorts=["S1"], swr_definitions=["K", "Y"]
    ).set_index("swr_definition")

    assert matrix.available.tolist() == [False, False]
    assert matrix.n_excluded.tolist() == [2, 2]
