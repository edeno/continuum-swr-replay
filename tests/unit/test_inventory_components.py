"""Connected components of the overlap graph (spec 03, "Overlap is not
transitive")."""

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.inventory_comparison.components import (
    COMPONENT_DTYPES,
    overlap_components,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import overlap_pairs
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    SessionIdentity,
    inventory_from_intervals,
)

GRID = np.arange(0, 100.001, 0.002)
EPOCH = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
OTHER = SessionIdentity(animal="sim", day=20200101, epoch="04_r2", brain_area="HPC")


def events(detector, spans, session=EPOCH):
    return inventory_from_intervals(
        np.array([a for a, _ in spans], dtype=float),
        np.array([b for _, b in spans], dtype=float),
        native_keys=np.arange(len(spans)),
        detector=detector,
        session=session,
        run_id="run",
        speed_time_s=GRID,
        speed_cm_per_s=np.zeros(GRID.size),
    )


def components_of(*frames):
    inventory = pd.concat(frames, ignore_index=True)
    return overlap_components(inventory, overlap_pairs(inventory))


def test_a_chain_is_one_component_with_its_span_not_an_interval():
    """A overlaps B and B overlaps C, while A and C do not overlap."""
    events_table, components = components_of(
        events("A", [(0, 10)]), events("B", [(8, 20)]), events("C", [(18, 30)])
    )

    assert len(components) == 1
    chain = components.iloc[0]
    assert (chain.n_events, chain.n_pairs, chain.detectors) == (3, 2, "A+B+C")
    assert (chain.span_start_s, chain.span_end_s, chain.span_s) == (0, 30, 30)
    # each event keeps its own bounds; the component only labels them
    assert events_table.set_index("event_key").loc["A:0", "event_end_s"] == 10
    assert events_table.component_id.nunique() == 1
    assert "event_start_s" not in components.columns


def test_an_unmatched_event_is_its_own_component():
    events_table, components = components_of(
        events("A", [(0, 10), (50, 60)]), events("B", [(5, 12)])
    )

    by_detectors = components.set_index("detectors")
    assert by_detectors.loc["A", "n_events"] == 1
    assert by_detectors.loc["A+B", "n_events"] == 2
    assert events_table.component_id.nunique() == 2


def test_two_events_of_one_detector_joined_through_another():
    _, components = components_of(
        events("A", [(0, 10), (12, 20)]), events("B", [(5, 15)])
    )

    only = components.iloc[0]
    assert (only.n_events, only.n_detectors, only.detectors) == (3, 2, "A+B")


def test_components_never_cross_epochs():
    _, components = components_of(
        events("A", [(0, 10)]), events("B", [(0, 10)], session=OTHER)
    )

    assert len(components) == 2
    assert sorted(components.epoch) == ["02_r1", "04_r2"]
    # numbering restarts in each epoch
    assert components.component_id.tolist() == [0, 0]


def test_components_are_numbered_by_onset_within_each_epoch():
    events_table, _ = components_of(events("A", [(50, 60), (0, 10)]))

    assert events_table.set_index("event_key").component_id.to_dict() == {
        "A:1": 0,
        "A:0": 1,
    }


def test_no_events_gives_the_declared_dtypes():
    events_table, components = components_of(events("A", []))

    assert events_table.empty and components.empty
    for column, dtype in COMPONENT_DTYPES.items():
        assert components[column].dtype == dtype, column
