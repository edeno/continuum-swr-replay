"""The event features examples' sampler: which events are drawn for a figure.

Two populations are drawn from. Comparator-only events are each detector's
events (``state_space`` and ``memoryless_criteria`` aside) unmatched by
``state_space``; ours-only events are ``state_space``'s events of the
exactly-one cell. Within each population, source detector and animal the
sampler draws up to two events of the target evidence subset and up to two
counterexamples of the other: comparator-only targets high evidence (at or
above the group's 75th percentile), ours-only low evidence (at or below its
25th). An event in neither subset, or with an ``unknown`` membership, is never
drawn. An epoch
whose inputs cannot be read, a missing memoryless run's file included, is
recorded as not written and the run goes on.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.figures.generate_event_feature_examples as examples
from continuum_swr_replay.analysis.inventory_comparison.populations import (
    POPULATION_DTYPES,
)

SEED = 7
DAY = 20200101
HIGH, LOW = examples.HIGH, examples.LOW

# evidence_dll by subset: in a group holding as many of each of the first
# three, the 75th percentile is 2.0 and the 25th -1.0
EVIDENCE = {HIGH: 2.0, LOW: -1.0, "between": 0.5, "missing": np.nan}


def tables(
    records: list[tuple[str, str, str, str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and population rows from (detector, animal, population, subset,
    count): each event in ``all`` and in the named population. A subset may
    be a list of evidence values, one event each."""
    features, populations = [], []
    for detector, animal, population, subset, count in records:
        values = (
            list(subset) if isinstance(subset, list) else [EVIDENCE[subset]] * count
        )
        for number, value in enumerate(values):
            label = subset if isinstance(subset, str) else "value"
            key = f"{detector}:{animal}{DAY}_02_r1:{population}:{label}{number}"
            ids = {
                "animal": animal,
                "day": DAY,
                "epoch": "02_r1",
                "detector": detector,
                "event_key": key,
            }
            features.append(
                {
                    **ids,
                    "brain_area": "HPC",
                    "evidence_dll": value,
                    "n_spikes": 3,
                    "n_rows": 10,
                    "ss_disagreement": "near",
                }
            )
            for label in ("all", population):
                populations.append({**ids, "population": label, "unknown_reason": ""})
    frame = pd.DataFrame(populations, columns=list(POPULATION_DTYPES))
    return pd.DataFrame(features), frame.astype(dict(POPULATION_DTYPES))


COMPARATOR = "unmatched_by:state_space"
OURS = "exactly_one_detector"
LARGE = tables(
    [
        (detector, animal, population, subset, 30)
        for detector, population in (
            ("kay", COMPARATOR),
            ("mua", COMPARATOR),
            ("state_space", OURS),
        )
        for animal in ("a", "b")
        for subset in (HIGH, LOW, "between", "missing")
    ]
)
UNIT = ["cell", "detector", "animal"]


def drawn(
    features: pd.DataFrame, populations: pd.DataFrame, seed: int = SEED
) -> pd.DataFrame:
    events = examples.sampler_events(features, populations)
    return examples.draw_population_examples(events, seed=seed)


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
    subsets = examples.evidence_subsets(events).tolist()
    # quartiles 1.75 and 5.25 of the eight finite values; a missing value is in
    # neither, as is a group of one event, whose quartiles coincide
    expected = [LOW, LOW, None, None, None, None, HIGH, HIGH, None]
    assert subsets == expected + expected + [None]


def test_the_subsets_are_descriptive_and_named_so() -> None:
    assert "descriptive" in examples.NOT_CALIBRATED
    assert "not a calibrated declaration of replay" in examples.NOT_CALIBRATED
    assert "baseline_" not in examples.SAMPLING_RULE
    assert "75th percentile" in examples.SAMPLING_RULE


def test_only_the_two_populations_enter_the_sampler() -> None:
    """Comparator-only: unmatched by state_space, memoryless_criteria aside.
    Ours-only: state_space's exactly-one events. Matched, unknown, other cells
    and other detectors' exactly-one events never enter."""
    features, populations = tables(
        [
            ("kay", "a", COMPARATOR, [2.0, -1.0], 0),
            ("memoryless", "a", COMPARATOR, [2.0, -1.0], 0),
            ("state_space", "a", OURS, [2.0, -1.0], 0),
            # none of these
            ("memoryless_criteria", "a", COMPARATOR, HIGH, 3),
            ("kay", "a", "unknown_by:state_space", HIGH, 3),
            ("kay", "a", "matched_by:state_space", HIGH, 3),
            ("kay", "a", "unmatched_by:memoryless", HIGH, 3),
            ("kay", "a", OURS, HIGH, 3),
            ("state_space", "a", f"{OURS}:unknown", LOW, 3),
            ("state_space", "a", f"{OURS}:matched", LOW, 3),
            ("state_space", "a", "swr_matched_by_no_decoder", LOW, 3),
            ("state_space", "a", "unmatched_by:kay", LOW, 3),
        ]
    )
    events = examples.sampler_events(features, populations)
    assert sorted(set(zip(events.cell, events.detector, strict=True))) == [
        ("comparator_only", "kay"),
        ("comparator_only", "memoryless"),
        ("ours_only", "state_space"),
    ]
    assert len(events) == 6
    assert (events.membership == "unmatched").all()
    assert set(events.population) == {COMPARATOR, OURS}
    assert events.epoch_id.unique().tolist() == [f"a{DAY}_02_r1_HPC"]
    assert (events.spikes_per_row == 0.3).all()


def test_each_unit_draws_two_targets_and_two_counterexamples() -> None:
    picks = drawn(*LARGE)
    sizes = picks.groupby([*UNIT, "evidence_subset"]).size()
    assert (sizes == 2).all()
    assert len(sizes) == 2 * len(picks[UNIT].drop_duplicates()) == 2 * 6
    roles = picks.groupby(["cell", "evidence_subset"]).role.unique().to_dict()
    assert {key: list(value) for key, value in roles.items()} == {
        ("comparator_only", HIGH): ["example"],
        ("comparator_only", LOW): ["counterexample"],
        ("ours_only", LOW): ["example"],
        ("ours_only", HIGH): ["counterexample"],
    }
    assert not picks.duplicated("event_key").any()
    # every draw is in its subset by its own evidence and its group's
    assert (picks.evidence_dll == picks.evidence_subset.map(EVIDENCE)).all()


def test_the_draw_is_deterministic_for_a_seed() -> None:
    first = drawn(*LARGE)
    pd.testing.assert_frame_equal(drawn(*LARGE), first)
    features, populations = LARGE
    shuffled = (
        features.sample(frac=1.0, random_state=3).reset_index(drop=True),
        populations.sample(frac=1.0, random_state=4).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(drawn(*shuffled), first)
    assert set(drawn(*LARGE, seed=SEED + 1).event_key) != set(first.event_key)
    # the run seed is this run's, not the inventory comparison's
    assert examples.EXAMPLE_SEED == 20261018
    default = examples.draw_population_examples(
        examples.sampler_events(*LARGE)
    ).event_key
    assert set(default) != set(first.event_key)


def test_events_in_neither_subset_or_unknown_are_never_drawn() -> None:
    features, populations = tables(
        [
            ("kay", "a", COMPARATOR, "between", 20),
            ("kay", "a", COMPARATOR, "missing", 20),
            ("kay", "a", "unknown_by:state_space", HIGH, 20),
            ("kay", "a", COMPARATOR, HIGH, 1),
            ("state_space", "a", f"{OURS}:unknown", LOW, 20),
            ("state_space", "a", OURS, "between", 20),
        ]
    )
    for seed in range(20):
        picks = drawn(features, populations, seed=seed)
        assert len(picks) == 1
        assert picks.event_key.iloc[0].endswith(f"{COMPARATOR}:{HIGH}0")


def test_a_small_subset_gives_fewer() -> None:
    features, populations = tables(
        [
            # quartiles 0.5 and 1.5: one high, one low
            ("kay", "a", COMPARATOR, [0.0, 1.0, 2.0], 0),
            # quartiles 1.75 and 5.25: two of each
            ("state_space", "a", OURS, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 0),
        ]
    )
    counts = drawn(features, populations).groupby(["cell", "role"]).size().to_dict()
    assert counts == {
        ("comparator_only", "counterexample"): 1,
        ("comparator_only", "example"): 1,
        ("ours_only", "counterexample"): 2,
        ("ours_only", "example"): 2,
    }
    assert drawn(features.iloc[:0], populations.iloc[:0]).empty


def test_an_epoch_without_its_inputs_is_recorded_and_the_run_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Animal b's epoch has no memoryless run: loading it raises
    FileNotFoundError, its draw is recorded as not written with the reason,
    and animal a's draws are written. A subset with no event in the run is
    named in the manifest and the captions."""
    features, populations = tables(
        [
            ("kay", "a", COMPARATOR, [2.0, -1.0], 0),
            ("kay", "b", COMPARATOR, [2.0, -1.0], 0),
            ("state_space", "a", OURS, [-1.0, 2.0], 0),
        ]
    )
    ours_key = features.event_key.iloc[4]
    rejection = pd.DataFrame(
        {
            "event_key": [ours_key, ours_key],
            "cohort": ["memoryless", "memoryless_criteria"],
            "stage": ["no_run", "duration"],
        }
    )
    manifest = {
        "settings": {"input_runs": {"inventory": "inv01", "memoryless": "ml01"}}
    }
    inventory_root = tmp_path / "inventory"
    (inventory_root / "inv01").mkdir(parents=True)
    (inventory_root / "inv01" / "manifest.json").write_text(
        json.dumps(
            {
                "epochs": {},
                "settings": {
                    "cells": {
                        "exactly_one_detector": {
                            "against": ["kay", "state_space", "mua"]
                        }
                    }
                },
            }
        )
    )
    run_dir, figure_dir = tmp_path / "run01", tmp_path / "figures"
    tables_read = {
        "features": features,
        "pair_membership": None,
        "rejection": rejection,
    }
    monkeypatch.setattr(
        examples, "read_run", lambda *args: (manifest, dict(tables_read))
    )
    monkeypatch.setattr(
        examples,
        "read_inputs",
        lambda *args: {"cells": None, "directional": None},
    )
    monkeypatch.setattr(examples, "event_populations", lambda *a: populations)
    closed = []

    def load_epoch(
        identifier: str, *, run_dir: Path, memoryless_dir: Path
    ) -> SimpleNamespace:
        if identifier.startswith("b"):
            raise FileNotFoundError(
                f"No such file: {memoryless_dir / identifier}_memoryless_candidates.pkl"
            )
        return SimpleNamespace(results=SimpleNamespace(close=lambda: closed.append(1)))

    notes: dict[str, str] = {}

    def draw_example(
        inputs: object,
        drawn: pd.Series,
        *,
        against: list[str],
        path: Path,
        run_id: str,
        seed: int,
        note: str,
    ) -> dict:
        notes[drawn.event_key] = note
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return {**drawn.to_dict(), "against": against, "path": str(path)}

    monkeypatch.setattr(examples.inventory_examples, "load_epoch", load_epoch)
    monkeypatch.setattr(examples.inventory_examples, "draw_example", draw_example)

    records = examples.write_examples(
        run_dir,
        figure_dir,
        inventory_root=inventory_root,
        memoryless_root=tmp_path / "memoryless",
    )

    status = {r["animal"] + r["cell"] + r["role"]: r for r in records}
    assert len(records) == 6 and closed == [1]
    for role in ("example", "counterexample"):
        missing = status[f"bcomparator_only{role}"]
        assert missing["status"] == "not_written"
        assert missing["reason"].startswith("No such file:")
        assert not Path(missing["path"]).exists()
    for name in ("acomparator_onlyexample", "aours_onlyexample"):
        assert status[name]["status"] == "written"
        assert Path(status[name]["path"]).exists()
    assert status["acomparator_onlyexample"]["against"] == ["state_space"]
    assert status["aours_onlyexample"]["against"] == ["kay", "mua"]
    assert status["aours_onlyexample"]["rejection_stage"] == {
        "memoryless": "no_run",
        "memoryless_criteria": "duration",
    }
    assert "memoryless† no_run, memoryless_criteria† duration" in notes[ours_key]
    # the title gives the evidence and the spikes per row, no baseline
    assert notes[ours_key].startswith(
        "low evidence: evidence_dll -1, 0.3 spikes per row"
    )
    assert "baseline" not in notes[ours_key]

    written = json.loads((figure_dir / "manifest.json").read_text())
    assert written["empty_subsets"] == []
    assert written["seed"] == examples.EXAMPLE_SEED
    assert [r["status"] for r in written["figures"]] == [r["status"] for r in records]
    captions = (figure_dir / "captions.md").read_text()
    assert examples.CELL_NOTE in captions
    assert captions.count("(not written: No such file:") == 2


def test_a_population_whose_groups_are_constant_has_empty_subsets() -> None:
    features, populations = tables(
        [
            ("kay", "a", COMPARATOR, [2.0, -1.0], 0),
            # one value throughout: its quartiles coincide, so neither subset
            ("state_space", "a", OURS, [1.0, 1.0, 1.0], 0),
        ]
    )
    events = examples.sampler_events(features, populations)

    assert examples.empty_subsets(events) == [
        "No ours-only event is low evidence in this run.",
        "No ours-only event is high evidence in this run.",
    ]
