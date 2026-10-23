#!/usr/bin/env python
"""Sampled comparator-only and ours-only events of one event features run.

Reads the per-epoch ``features``, ``pair_membership`` and ``rejection`` tables
that ``scripts/detection/run_event_features.py`` wrote for the run's ``ok`` epochs,
and the ``cells`` and ``directional`` tables of the inventory comparison run
it read, and draws events from two populations by :data:`SAMPLING_RULE`.
Design: ``docs/design/replay_detection_comparison/04-event-features.md``
(examples follow 03's sampling rule).

- **Comparator-only**: each detector's events unmatched by ``state_space``
  (``unmatched_by:state_space``), for every detector of the comparison's
  exactly-one cell but ``state_space``.
- **Ours-only**: ``state_space``'s events of the exactly-one cell.

Each event is in one of two descriptive subsets by its spike evidence within
its population, source detector and animal (high: at or above the group's 75th
percentile; low: at or below its 25th), or in neither. No baseline is compared:
the lead dropped the baseline windows (2026-10-18). Each figure is
``generate_inventory_examples.draw_example``'s, with the event's evidence, its
spikes per row, and its ``ss_disagreement`` (comparator-only) or
its rejection stage in each memoryless cohort (ours-only) in the title. A
figure whose recomputed memoryless windows are not the stored ones is not
written, and its record in the manifest says why.

Figures, ``captions.md`` and ``manifest.json`` (the rule, the seed and every
draw) go to ``paper/figures/event_features/<run id>/examples/``.

Usage:
    python scripts/figures/generate_event_feature_examples.py \
        --run-id <event features run id>
"""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continuum_swr_replay.analysis.inventory_comparison.overlap import (  # noqa: E402
    UNMATCHED,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (  # noqa: E402
    STATE_SPACE,
)
from continuum_swr_replay.analysis.inventory_comparison.selection import (  # noqa: E402
    REJECTION_COHORTS,
)
from continuum_swr_replay.parameters import FIGURE_DIR  # noqa: E402
from continuum_swr_replay.paths import (  # noqa: E402
    EVENT_FEATURES_ROOT,
    INVENTORY_COMPARISON_ROOT,
    MEMORYLESS_ROOT,
)
from continuum_swr_replay.runs.cli import parse_run_id  # noqa: E402
from continuum_swr_replay.runs.io import json_ready  # noqa: E402
from continuum_swr_replay.runs.provenance import produced_by  # noqa: E402
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY  # noqa: E402
from scripts.detection import report_event_features as report  # noqa: E402
from scripts.detection import run_event_features as features_runner  # noqa: E402
from scripts.detection.run_inventory_comparison import CELL_DETECTORS  # noqa: E402
from scripts.figures import (  # noqa: E402
    generate_inventory_examples as inventory_examples,
)

PRODUCED_BY = produced_by(__file__)
FIGURE_ROOT = FIGURE_DIR / "event_features"
EXAMPLES_DIR = "examples"
"""The examples' folder inside the run's figure folder."""

EXAMPLE_SEED = 20261018
"""The run seed every draw is seeded from, with its population, detector and
animal."""
N_PER_SIDE = inventory_examples.N_PER_SIDE

HIGH = "high evidence"
LOW = "low evidence"
SUBSET_COLUMN = "evidence_subset"
SUBSET_QUANTILES: Mapping[str, float] = MappingProxyType({HIGH: 0.75, LOW: 0.25})
"""The quantile of each subset's bound, over the scorable events of the event's
group (:data:`GROUP`)."""
SUBSETS: Mapping[str, str] = MappingProxyType(
    {
        HIGH: "evidence_dll at or above the 75th percentile of the group's "
        "scorable events",
        LOW: "evidence_dll at or below the 25th percentile of the group's "
        "scorable events",
    }
)
"""Each descriptive evidence subset and its rule. An event with no evidence is
in neither; so is one meeting both, as every event of a group whose 25th and
75th percentiles coincide does."""
GROUP = ("cell", "detector", "animal")
"""An event's group: its population, source detector and animal, the unit the
examples are drawn by."""

COMPARATOR_ONLY = "comparator_only"
OURS_ONLY = "ours_only"
EXACTLY_ONE = "exactly_one_detector"
POPULATIONS: Mapping[str, str] = MappingProxyType(
    {COMPARATOR_ONLY: f"unmatched_by:{STATE_SPACE}", OURS_ONLY: EXACTLY_ONE}
)
"""Each population drawn from and its ``populations.population_rows`` label."""
COMPARATOR_DETECTORS = tuple(d for d in CELL_DETECTORS if d != STATE_SPACE)
"""The comparator-only population's source detectors."""
ROLES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        COMPARATOR_ONLY: MappingProxyType({HIGH: "example", LOW: "counterexample"}),
        OURS_ONLY: MappingProxyType({LOW: "example", HIGH: "counterexample"}),
    }
)
"""Each population's target subset (``example``) and the other subset
(``counterexample``), in the order drawn."""

CELL_NOTE = (
    "In manifest.json, a figure's cell is the name of its population "
    f"({COMPARATOR_ONLY} or {OURS_ONLY}) and its population is that "
    f"population's label ({POPULATIONS[COMPARATOR_ONLY]} or "
    f"{POPULATIONS[OURS_ONLY]})."
)

SESSION = list(SESSION_COLUMNS_DAY)
EVIDENCE = ("evidence_dll", "n_spikes", "n_rows")

NOT_CALIBRATED = (
    "The two subsets are descriptive. evidence_dll, the mean per-row "
    "log-likelihood difference between the best remote real place bin and the "
    "animal's bin, grows with the number of spikes, so each title gives the "
    "spikes per row beside it; it is a diagnostic, not a calibrated declaration "
    "of replay."
)
SAMPLING_RULE = (
    "Two populations are drawn from: comparator-only events, each detector's "
    "events unmatched by state_space (unmatched_by:state_space; detectors "
    f"{', '.join(map(inventory_examples._label, COMPARATOR_DETECTORS))}), and "
    "ours-only events, state_space's "
    f"events of the {EXACTLY_ONE} cell. Within each population, source detector "
    f"and animal, an event's evidence is {HIGH} when its evidence_dll is at or "
    "above the 75th percentile of that group's scorable events, and "
    f"{LOW} when it is at or below their 25th percentile; an event with no "
    "evidence, or meeting both (a group whose two percentiles coincide), is in "
    "neither subset and is not drawn. For each population, source detector and animal, "
    f"{N_PER_SIDE} events of the target subset are drawn (comparator-only: "
    f"{HIGH}; ours-only: {LOW}) and, as counterexamples, "
    f"{N_PER_SIDE} events of the other subset; fewer when a subset has fewer. "
    "Events with membership unknown are in neither population and are never "
    "drawn. The draws are made without replacement from the events sorted by "
    "session and event_key, by numpy.random.default_rng seeded from the run "
    f"seed {EXAMPLE_SEED} and the (population, detector, animal) triple."
)


def evidence_subsets(events: pd.DataFrame) -> pd.Series:
    """Each event's evidence subset by :data:`SUBSETS`, or None in neither.

    Parameters
    ----------
    events : DataFrame
        With ``evidence_dll`` and the columns of :data:`GROUP`.

    Returns
    -------
    Series of object
        :data:`HIGH`, :data:`LOW` or None, on ``events``' index.
    """
    dll = events.evidence_dll.astype(np.float64)
    groups = dll.groupby([events[column] for column in GROUP], sort=False)
    # quantiles of each group's finite values; NaN is skipped
    high = (dll >= groups.transform("quantile", SUBSET_QUANTILES[HIGH])).to_numpy()
    low = (dll <= groups.transform("quantile", SUBSET_QUANTILES[LOW])).to_numpy()
    labels = np.full(len(events), None, dtype=object)
    labels[high & ~low] = HIGH
    labels[low & ~high] = LOW
    return pd.Series(labels, index=events.index, dtype=object)


def sampler_events(features: pd.DataFrame, populations: pd.DataFrame) -> pd.DataFrame:
    """The comparator-only and ours-only events, with their evidence subsets.

    Parameters
    ----------
    features : DataFrame
        The run's ``features`` rows: ``animal``, ``day``, ``epoch``,
        ``brain_area``, ``detector``, ``event_key``, ``evidence_dll``,
        ``n_spikes``, ``n_rows`` and ``ss_disagreement``.
    populations : DataFrame
        ``report_event_features.event_populations`` of the run.

    Returns
    -------
    DataFrame
        One row per event of the two populations: ``cell`` (the population's
        name, :data:`COMPARATOR_ONLY` or :data:`OURS_ONLY`), the session,
        ``detector``, ``event_key``, ``population`` (its label),
        ``membership`` (``unmatched``), ``brain_area``, the evidence columns,
        ``ss_disagreement``, ``spikes_per_row`` (``n_spikes / n_rows``, NaN
        with no row), :data:`SUBSET_COLUMN` and ``epoch_id``, sorted by
        population, detector, session and event key.
    """
    ids = [*SESSION, "detector", "event_key"]
    population, detector = populations.population, populations.detector
    comparator = (population == POPULATIONS[COMPARATOR_ONLY]) & detector.isin(
        COMPARATOR_DETECTORS
    )
    ours = (population == POPULATIONS[OURS_ONLY]) & (detector == STATE_SPACE)
    rows = populations.loc[(comparator | ours).to_numpy(), [*ids, "population"]]
    rows.insert(
        0,
        "cell",
        np.where(
            rows.population == POPULATIONS[COMPARATOR_ONLY], COMPARATOR_ONLY, OURS_ONLY
        ),
    )
    events = rows.merge(
        features[[*ids, "brain_area", *EVIDENCE, "ss_disagreement"]],
        on=ids,
        how="left",
        validate="one_to_one",
    ).assign(membership=UNMATCHED)
    n_rows = events.n_rows.to_numpy(np.float64)
    n_spikes = events.n_spikes.astype("Float64").to_numpy(np.float64, na_value=np.nan)
    events["spikes_per_row"] = np.where(
        n_rows > 0, n_spikes / np.maximum(n_rows, 1), np.nan
    )
    events[SUBSET_COLUMN] = evidence_subsets(events)
    events["epoch_id"] = (
        events.animal
        + events.day.astype(str)
        + "_"
        + events.epoch
        + "_"
        + events.brain_area
    )
    return events.sort_values(["cell", "detector", *ids[:3], "event_key"]).reset_index(
        drop=True
    )


def draw_population_examples(
    events: pd.DataFrame, *, seed: int = EXAMPLE_SEED, n: int = N_PER_SIDE
) -> pd.DataFrame:
    """The events :data:`SAMPLING_RULE` draws from :func:`sampler_events`.

    Each population is drawn by ``generate_inventory_examples.draw_examples``
    by :data:`SUBSET_COLUMN`, with that population's :data:`ROLES`.

    Parameters
    ----------
    events : DataFrame
        :func:`sampler_events` of the run.
    seed : int, optional
        The run seed; each draw's stream is seeded from it and its
        (population, detector, animal).
    n : int, optional
        Events drawn per subset, population, detector and animal, at most.

    Returns
    -------
    DataFrame
        The drawn rows with ``role`` and ``draw``, comparator-only first.
    """
    drawn = [
        inventory_examples.draw_examples(
            events[(events.cell == name).to_numpy()],
            seed=seed,
            n=n,
            column=SUBSET_COLUMN,
            roles=ROLES[name],
        )
        for name in POPULATIONS
    ]
    return pd.concat([d for d in drawn if len(d)] or drawn[:1], ignore_index=True)


def empty_subsets(events: pd.DataFrame) -> list[str]:
    """A sentence for each population's subset with no event in ``events``.

    Parameters
    ----------
    events : DataFrame
        :func:`sampler_events` of the run.
    """
    return [
        f"No {name.replace('_', '-')} event is {subset} in this run."
        for name, roles in ROLES.items()
        for subset in roles
        if not ((events.cell == name) & (events[SUBSET_COLUMN] == subset)).any()
    ]


def rejection_stages(rejection: pd.DataFrame) -> dict[str, dict[str, str]]:
    """``event_key -> {cohort: stage}`` of the run's ``rejection`` table."""
    stages: dict[str, dict[str, str]] = {}
    for key, cohort, stage in zip(
        rejection.event_key, rejection.cohort, rejection.stage, strict=True
    ):
        stages.setdefault(key, {})[cohort] = stage
    return stages


def figure_note(drawn: pd.Series, stages: Mapping[str, str], run_id: str) -> str:
    """The title's last lines: the event's evidence and what it disagrees by.

    Parameters
    ----------
    drawn : Series
        One row of :func:`draw_population_examples`.
    stages : mapping
        The event's rejection stage by memoryless cohort
        (:func:`rejection_stages`); a cohort it lacks reads ``missing``.
    run_id : str
        The event features run.

    Returns
    -------
    str
        Two lines: the evidence subset and values with the ``ss_disagreement``
        (comparator-only) or the rejection stages (ours-only); then the run and
        the reminder that the evidence is a diagnostic, not a calibrated
        declaration of replay.
    """
    values = (
        f"{drawn.evidence_subset}: evidence_dll {drawn.evidence_dll:.3g}, "
        f"{drawn.spikes_per_row:.3g} spikes per row"
    )
    if drawn.cell == COMPARATOR_ONLY:
        disagreement = drawn.ss_disagreement
        detail = (
            f"ss_disagreement {disagreement if pd.notna(disagreement) else 'missing'}"
        )
    else:
        detail = "rejection stage " + ", ".join(
            f"{inventory_examples._label(cohort)} {stages.get(cohort, 'missing')}"
            for cohort in REJECTION_COHORTS
        )
    return (
        f"{values}; {detail}\nevent features run {run_id}; the evidence is a "
        "diagnostic, not a calibrated declaration of replay"
    )


CAPTION = (
    "Sampled comparator-only and ours-only events of the event features run. "
    + SAMPLING_RULE
    + " "
    + NOT_CALIBRATED
    + " Each title gives the event's evidence_dll, its spikes per row, and its "
    "ss_disagreement (comparator-only) or its rejection "
    "stage in each memoryless† cohort (ours-only)." + inventory_examples.PANELS_CAPTION
)


def write_examples(
    run_dir: Path,
    figure_dir: Path,
    *,
    inventory_root: Path | None = None,
    memoryless_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Draw every example of the run and write them, captions and manifest.

    A draw whose figure cannot be made (its recomputed windows are not the
    stored ones, or its epoch's inputs cannot be read, a missing memoryless
    run's file included) is recorded with
    ``status`` ``not_written`` and its ``reason``, and the run goes on.

    Parameters
    ----------
    run_dir : Path
        The event features run's folder.
    figure_dir : Path
    inventory_root, memoryless_root : Path, optional
        Where the comparison and memoryless runs are; the event features
        runner's by default.

    Returns
    -------
    list of dict
        Each draw's record, as the manifest holds it: ``status`` ``written``
        or ``not_written``.
    """
    inventory_root = (
        INVENTORY_COMPARISON_ROOT if inventory_root is None else inventory_root
    )
    memoryless_root = MEMORYLESS_ROOT if memoryless_root is None else memoryless_root
    manifest, tables = report.read_run(
        run_dir, features_runner.TABLE_DTYPES, features_runner.output_path
    )
    tables |= report.read_inputs(manifest, inventory_root, memoryless_root)
    tables["features"] = report.without_baseline_rows(tables["features"])
    populations = report.event_populations(
        tables["features"],
        tables["pair_membership"],
        tables["cells"],
        tables["directional"],
    )
    events = sampler_events(tables["features"], populations)
    drawn = draw_population_examples(events)
    empty = empty_subsets(events)
    stages = rejection_stages(tables["rejection"])
    runs = manifest["settings"]["input_runs"]
    inventory_dir = inventory_root / runs["inventory"]
    memoryless_dir = memoryless_root / runs["memoryless"]
    cells = json.loads((inventory_dir / "manifest.json").read_text())["settings"][
        "cells"
    ]
    against = {
        COMPARATOR_ONLY: [STATE_SPACE],
        OURS_ONLY: sorted(set(cells[EXACTLY_ONE]["against"]) - {STATE_SPACE}),
    }
    run_id = run_dir.name
    records = []
    for identifier, group in drawn.groupby("epoch_id", sort=True):
        try:
            inputs: inventory_examples.EpochInputs | None = (
                inventory_examples.load_epoch(
                    identifier, run_dir=inventory_dir, memoryless_dir=memoryless_dir
                )
            )
            epoch_error: str | None = None
        except (ValueError, FileNotFoundError) as error:
            # the epoch's draws fail, not the run; a missing file is, for
            # example, an epoch the memoryless run did not observe
            inputs, epoch_error = None, str(error)
        for _, row in group.iterrows():
            path = figure_dir / inventory_examples.figure_name(row)
            event_stages = (
                stages.get(row.event_key, {}) if row.cell == OURS_ONLY else {}
            )
            note = figure_note(row, event_stages, run_id)
            reason = epoch_error
            if inputs is not None:
                try:
                    record = inventory_examples.draw_example(
                        inputs,
                        row,
                        against=against[row.cell],
                        path=path,
                        run_id=runs["inventory"],
                        seed=EXAMPLE_SEED,
                        note=note,
                    )
                except ValueError as error:
                    reason = str(error)
            if reason is not None:
                # no figure of an earlier run stands beside this record
                path.unlink(missing_ok=True)
                record = {
                    **row.to_dict(),
                    "against": against[row.cell],
                    "path": str(path),
                    "reason": reason,
                }
            status = "written" if reason is None else "not_written"
            records.append(
                {
                    "status": status,
                    **record,
                    "rejection_stage": event_stages,
                    "caption": note.split("\n")[0],
                }
            )
            print(f"{identifier}: {path.name} {status}", flush=True)
        if inputs is not None:
            inputs.results.close()
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "manifest.json").write_text(
        json.dumps(
            json_ready(
                {
                    "produced_by": PRODUCED_BY,
                    "event_features_run_id": run_id,
                    "input_runs": runs,
                    "seed": EXAMPLE_SEED,
                    "n_per_side": N_PER_SIDE,
                    "sampling_rule": SAMPLING_RULE,
                    "populations": {
                        name: {"label": label, "roles": ROLES[name]}
                        for name, label in POPULATIONS.items()
                    },
                    "comparator_detectors": COMPARATOR_DETECTORS,
                    "cell_column": CELL_NOTE,
                    "empty_subsets": empty,
                    "subsets": SUBSETS,
                    "not_calibrated": NOT_CALIBRATED,
                    "pad_s": inventory_examples.PAD_S,
                    "memoryless_check": {
                        "spread_rtol": inventory_examples.SPREAD_RTOL,
                        "center_tie_rtol": inventory_examples.CENTER_TIE_RTOL,
                        **inventory_examples.recompute_device(),
                    },
                    "pending": inventory_examples.PENDING,
                    "figures": records,
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )
    lines = [f"# Event features examples: run {run_id}", "", CAPTION, ""]
    if empty:
        lines += [" ".join(empty), ""]
    lines += [CELL_NOTE, ""]
    lines += [
        f"- `{Path(r['path']).name}`: {r['event_key']} ({r['role']}); {r['caption']}"
        + ("" if r["status"] == "written" else f" (not written: {r['reason']})")
        for r in records
    ]
    (figure_dir / "captions.md").write_text("\n".join(lines) + "\n")
    return records


def main(argv: list[str] | None = None) -> None:
    """Draw and write the examples of the run ``--run-id`` names.

    Parameters
    ----------
    argv : list of str, optional
        The command line; ``sys.argv[1:]`` by default.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-id",
        type=parse_run_id,
        required=True,
        help="The event features run's folder name, under "
        "replay_detection_comparison/event_features/.",
    )
    args = parser.parse_args(argv)
    inventory_examples.use_memoryless_device()
    run_dir = EVENT_FEATURES_ROOT / args.run_id
    if not (run_dir / "manifest.json").exists():
        sys.exit(f"No event features run manifest at {run_dir}.")
    figure_dir = FIGURE_ROOT / args.run_id / EXAMPLES_DIR
    records = write_examples(run_dir, figure_dir)
    written = sum(r["status"] == "written" for r in records)
    print(f"{written} of {len(records)} examples written -> {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
