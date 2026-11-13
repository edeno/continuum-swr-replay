"""The examples' sampler: which events of each cell are drawn for a figure, and
the event-feature examples' evidence subsets.

For each cell, source detector and animal the sampler draws up to two
``unmatched`` events, and as counterexamples up to two ``matched`` events of
the same cell, detector and animal, by a generator seeded from the run seed
and that triple. ``unknown`` events are never drawn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detector_comparison import selection
from tests.support._inventory_cells import cells

SEED = 7


LARGE = cells(
    [
        ("exactly_one_detector", "kay", "a", "unmatched", 30),
        ("exactly_one_detector", "kay", "a", "matched", 30),
        ("exactly_one_detector", "kay", "a", "unknown", 30),
        ("exactly_one_detector", "kay", "b", "unmatched", 30),
        ("exactly_one_detector", "kay", "b", "matched", 30),
        ("swr_matched_by_no_decoder", "kay", "a", "unmatched", 30),
        ("swr_matched_by_no_decoder", "kay", "a", "matched", 30),
        ("swr_matched_by_no_decoder", "yu", "a", "unmatched", 30),
        ("swr_matched_by_no_decoder", "yu", "a", "matched", 30),
    ]
)
UNIT = ["cell", "detector", "animal"]


def test_the_draw_is_deterministic_for_a_seed() -> None:
    first = selection.draw_examples(LARGE, seed=SEED)
    again = selection.draw_examples(LARGE, seed=SEED)
    pd.testing.assert_frame_equal(first, again)
    # the input's row order does not change what is drawn
    shuffled = LARGE.sample(frac=1.0, random_state=3).reset_index(drop=True)
    pd.testing.assert_frame_equal(selection.draw_examples(shuffled, seed=SEED), first)
    # another seed draws other events from these 30-event pools
    other = selection.draw_examples(LARGE, seed=SEED + 1)
    assert set(other.event_key) != set(first.event_key)


def test_each_unit_draws_two_unmatched_and_two_matched() -> None:
    drawn = selection.draw_examples(LARGE, seed=SEED)
    sizes = drawn.groupby([*UNIT, "membership"]).size()
    assert (sizes == 2).all()
    assert len(sizes) == 2 * len(LARGE[UNIT].drop_duplicates())
    roles = dict(zip(drawn.membership, drawn.role, strict=True))
    assert roles == {"unmatched": "example", "matched": "counterexample"}
    assert not drawn.duplicated("event_key").any()


def test_unknown_events_are_never_drawn() -> None:
    only_unknown = cells(
        [
            ("exactly_one_detector", "mua", "a", "unknown", 50),
            ("exactly_one_detector", "mua", "a", "unmatched", 1),
        ]
    )
    for seed in range(20):
        drawn = selection.draw_examples(only_unknown, seed=seed)
        assert set(drawn.membership) == {"unmatched"}
        assert len(drawn) == 1
    assert (
        selection.draw_examples(LARGE, seed=SEED)
        .membership.isin(["unmatched", "matched"])
        .all()
    )


def test_a_small_cell_gives_fewer() -> None:
    small = cells(
        [
            ("exactly_one_detector", "carey", "a", "matched", 1),
            ("exactly_one_detector", "carey", "a", "unknown", 5),
            ("sequence_matched_by_no_trigger", "memoryless", "a", "unmatched", 1),
            ("sequence_matched_by_no_trigger", "memoryless", "a", "matched", 3),
        ]
    )
    drawn = selection.draw_examples(small, seed=SEED)
    counts = drawn.groupby(["cell", "membership"]).size().to_dict()
    assert counts == {
        ("exactly_one_detector", "matched"): 1,
        ("sequence_matched_by_no_trigger", "matched"): 2,
        ("sequence_matched_by_no_trigger", "unmatched"): 1,
    }
    assert selection.draw_examples(small.iloc[:0], seed=SEED).empty


def test_one_units_draw_does_not_depend_on_the_others() -> None:
    """Each (cell, detector, animal) has its own generator."""
    alone = LARGE[(LARGE.animal == "a") & (LARGE.detector == "kay")]
    unit = (
        (LARGE.animal == "a")
        & (LARGE.detector == "kay")
        & (LARGE.cell == "exactly_one_detector")
    )
    full = selection.draw_examples(LARGE, seed=SEED)
    part = selection.draw_examples(alone, seed=SEED)
    keys = set(LARGE[unit].event_key)
    assert set(full.event_key) & keys == set(part.event_key) & keys
    assert np.isin(part.event_key, LARGE.event_key).all()


def test_the_evidence_subsets_are_quartiles_of_each_group() -> None:
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan]
    events = pd.DataFrame(
        {
            "cell": "comparator_only",
            "detector": ["kay"] * 9 + ["kay"] * 9 + ["mua"],
            "animal": ["a"] * 9 + ["b"] * 9 + ["a"],
            # animal b's values are shifted: its own quartiles apply
            "evidence_dll": values + [v + 100.0 for v in values] + [5.0],
        }
    )
    subsets = selection.evidence_subsets(events).tolist()
    # quartiles 1.75 and 5.25 of the eight finite values; a missing value is in
    # neither, as is a group of one event, whose quartiles coincide
    expected = [
        selection.LOW,
        selection.LOW,
        None,
        None,
        None,
        None,
        selection.HIGH,
        selection.HIGH,
        None,
    ]
    assert subsets == expected + expected + [None]
