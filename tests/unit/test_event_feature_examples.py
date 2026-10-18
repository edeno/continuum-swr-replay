"""The event features examples' sampler: which events are drawn for a figure.

Two populations are drawn from. Comparator-only events are each detector's
events (``state_space`` and ``memoryless_criteria`` aside) unmatched by
``state_space``; ours-only events are ``state_space``'s events of the
exactly-one cell. Within each population, source detector and animal the
sampler draws up to two events of the target evidence subset and up to two
counterexamples of the other: comparator-only targets evidence above the
baseline's q95, ours-only evidence at or below the baseline's median. An event
in neither subset, or with an ``unknown`` membership, is never drawn. An epoch
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
ABOVE, AT_OR_BELOW = examples.ABOVE, examples.AT_OR_BELOW

# evidence_dll for each subset against a baseline median 0 and q95 1
EVIDENCE = {ABOVE: 2.0, AT_OR_BELOW: -1.0, "between": 0.5, "missing": np.nan}


def tables(
    records: list[tuple[str, str, str, str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and population rows from (detector, animal, population, subset,
    count): each event in ``all`` and in the named population."""
    features, populations = [], []
    for detector, animal, population, subset, count in records:
        for number in range(count):
            key = f"{detector}:{animal}{DAY}_02_r1:{population}:{subset}{number}"
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
                    "evidence_dll": EVIDENCE[subset],
                    "baseline_median_dll": 0.0,
                    "baseline_q95_dll": 1.0,
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
        for subset in (ABOVE, AT_OR_BELOW, "between", "missing")
    ]
)
UNIT = ["cell", "detector", "animal"]


def drawn(
    features: pd.DataFrame, populations: pd.DataFrame, seed: int = SEED
) -> pd.DataFrame:
    events = examples.sampler_events(features, populations)
    return examples.draw_population_examples(events, seed=seed)


def test_the_evidence_subsets() -> None:
    features = pd.DataFrame(
        {
            "evidence_dll": [2.0, 1.0, 0.5, 0.0, -1.0, np.nan, 2.0, -1.0],
            "baseline_median_dll": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
            "baseline_q95_dll": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
        }
    )
    subsets = examples.evidence_subsets(features)
    # above q95 is strict; at the median is at or below; a missing value is in
    # neither, as is everything between the median and q95
    assert subsets.tolist() == [
        ABOVE,
        None,
        None,
        AT_OR_BELOW,
        AT_OR_BELOW,
        None,
        None,
        AT_OR_BELOW,
    ]


def test_only_the_two_populations_enter_the_sampler() -> None:
    """Comparator-only: unmatched by state_space, memoryless_criteria aside.
    Ours-only: state_space's exactly-one events. Matched, unknown, other cells
    and other detectors' exactly-one events never enter."""
    features, populations = tables(
        [
            ("kay", "a", COMPARATOR, ABOVE, 1),
            ("memoryless", "a", COMPARATOR, ABOVE, 1),
            ("state_space", "a", OURS, AT_OR_BELOW, 1),
            # none of these
            ("memoryless_criteria", "a", COMPARATOR, ABOVE, 3),
            ("kay", "a", "unknown_by:state_space", ABOVE, 3),
            ("kay", "a", "matched_by:state_space", ABOVE, 3),
            ("kay", "a", "unmatched_by:memoryless", ABOVE, 3),
            ("kay", "a", OURS, ABOVE, 3),
            ("state_space", "a", f"{OURS}:unknown", AT_OR_BELOW, 3),
            ("state_space", "a", f"{OURS}:matched", AT_OR_BELOW, 3),
            ("state_space", "a", "swr_matched_by_no_decoder", AT_OR_BELOW, 3),
            ("state_space", "a", "unmatched_by:kay", AT_OR_BELOW, 3),
        ]
    )
    events = examples.sampler_events(features, populations)
    assert sorted(zip(events.cell, events.detector, strict=True)) == [
        ("comparator_only", "kay"),
        ("comparator_only", "memoryless"),
        ("ours_only", "state_space"),
    ]
    assert (events.membership == "unmatched").all()
    assert set(events.population) == {COMPARATOR, OURS}
    assert events.epoch_id.unique().tolist() == [f"a{DAY}_02_r1_HPC"]
    # an epoch's baseline windows are rows of detector baseline: never drawn
    baseline = features.iloc[:1].assign(detector="baseline", event_key="baseline:0")
    with_baseline = pd.concat([features, baseline], ignore_index=True)
    rows = populations.iloc[:1].assign(
        detector="baseline", event_key="baseline:0", population="baseline"
    )
    again = examples.sampler_events(
        with_baseline, pd.concat([populations, rows], ignore_index=True)
    )
    pd.testing.assert_frame_equal(again, events)


def test_each_unit_draws_two_targets_and_two_counterexamples() -> None:
    picks = drawn(*LARGE)
    sizes = picks.groupby([*UNIT, "evidence_subset"]).size()
    assert (sizes == 2).all()
    assert len(sizes) == 2 * len(picks[UNIT].drop_duplicates()) == 2 * 6
    roles = picks.groupby(["cell", "evidence_subset"]).role.unique().to_dict()
    assert {key: list(value) for key, value in roles.items()} == {
        ("comparator_only", ABOVE): ["example"],
        ("comparator_only", AT_OR_BELOW): ["counterexample"],
        ("ours_only", AT_OR_BELOW): ["example"],
        ("ours_only", ABOVE): ["counterexample"],
    }
    assert not picks.duplicated("event_key").any()
    # every draw is in its subset by its own evidence
    assert (examples.evidence_subsets(picks) == picks.evidence_subset).all()


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
            ("kay", "a", "unknown_by:state_space", ABOVE, 20),
            ("kay", "a", COMPARATOR, ABOVE, 1),
            ("state_space", "a", f"{OURS}:unknown", AT_OR_BELOW, 20),
            ("state_space", "a", OURS, "between", 20),
        ]
    )
    for seed in range(20):
        picks = drawn(features, populations, seed=seed)
        assert len(picks) == 1
        assert picks.event_key.iloc[0].endswith(f"{COMPARATOR}:{ABOVE}0")


def test_a_small_subset_gives_fewer() -> None:
    features, populations = tables(
        [
            ("kay", "a", COMPARATOR, ABOVE, 1),
            ("kay", "a", COMPARATOR, AT_OR_BELOW, 5),
            ("state_space", "a", OURS, ABOVE, 3),
        ]
    )
    counts = drawn(features, populations).groupby(["cell", "role"]).size().to_dict()
    assert counts == {
        ("comparator_only", "counterexample"): 2,
        ("comparator_only", "example"): 1,
        ("ours_only", "counterexample"): 2,
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
            ("kay", "a", COMPARATOR, ABOVE, 1),
            ("kay", "b", COMPARATOR, ABOVE, 1),
            ("state_space", "a", OURS, ABOVE, 1),
        ]
    )
    ours_key = features.event_key.iloc[2]
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
                "settings": {
                    "cells": {
                        "exactly_one_detector": {
                            "against": ["kay", "state_space", "mua"]
                        }
                    }
                }
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
        examples.report, "read_run", lambda *args: (manifest, dict(tables_read))
    )
    monkeypatch.setattr(
        examples.report,
        "read_inputs",
        lambda *args: {"cells": None, "directional": None},
    )
    monkeypatch.setattr(examples.report, "event_populations", lambda *a: populations)
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

    status = {r["animal"] + r["cell"]: r for r in records}
    assert len(records) == 3 and closed == [1]
    missing = status["bcomparator_only"]
    assert missing["status"] == "not_written"
    assert missing["reason"].startswith("No such file:")
    assert not Path(missing["path"]).exists()
    for name in ("acomparator_only", "aours_only"):
        assert status[name]["status"] == "written"
        assert Path(status[name]["path"]).exists()
    assert status["acomparator_only"]["against"] == ["state_space"]
    assert status["aours_only"]["against"] == ["kay", "mua"]
    assert status["aours_only"]["rejection_stage"] == {
        "memoryless": "no_run",
        "memoryless_criteria": "duration",
    }
    assert "memoryless† no_run, memoryless_criteria† duration" in notes[ours_key]

    empty = [
        "No comparator-only event is evidence at or below baseline in this run.",
        "No ours-only event is evidence at or below baseline in this run.",
    ]
    written = json.loads((figure_dir / "manifest.json").read_text())
    assert written["empty_subsets"] == empty
    assert written["seed"] == examples.EXAMPLE_SEED
    assert [r["status"] for r in written["figures"]] == [r["status"] for r in records]
    captions = (figure_dir / "captions.md").read_text()
    assert all(sentence in captions for sentence in empty)
    assert examples.CELL_NOTE in captions
    assert captions.count("(not written: No such file:") == 1
