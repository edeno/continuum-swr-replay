#!/usr/bin/env python
"""Report one inventory comparison run: its counts per session and per animal.

Reads the per-epoch tables ``scripts/detection/run_inventory_comparison.py`` wrote
for the run's ``ok`` epochs, concatenated, and its ``manifest.json``. Writes
plain tables, one CSV each, to ``<inventory run>/report/`` with a
``report.md`` that captions them, and a few figures to
``paper/figures/inventory_comparison/<run id>/`` with a ``captions.md``.
Design:
``docs/design/replay_detection_comparison/03-inventory-comparison.md``
(Reporting).

Every table comes per session (one analysed epoch, nested in its animal) and
per animal. An animal pools its sessions by summing counts and seconds and
takes proportions from those sums (``n_matched`` over ``n_eligible``), never
by averaging the sessions' proportions. A session where the comparison was
unavailable is not summed; it is listed in ``unavailable_sessions``. The
pooling, and the counts and cells tables the events summary reads, are
``analysis.detector_comparison.pooling``'s. An
``unknown`` membership or label is counted apart, never as ``unmatched`` or
``non-SWR``. Durations, overlap lengths, IoU and component sizes are given as
distributions of the events or pairs pooled within the animal.

Every row with a ``memoryless`` or ``memoryless_criteria`` inventory carries
``pending`` = "pending the Phase 2 pilot checks", and every caption says so.
The report states counts only.

Usage:
    python scripts/detection/report_inventory_comparison.py --run-id <inventory run id>
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from continuum_swr_replay.analysis.detector_comparison import (  # noqa: E402
    inventory_run,
)
from continuum_swr_replay.analysis.detector_comparison.pooling import (  # noqa: E402
    cells_by_animal,
    counts_by_animal,
    directional_by_animal,
    pool_sessions,
    ratio,
    session_names,
    session_tables,
    summed,
)
from continuum_swr_replay.analysis.detector_comparison.populations import (  # noqa: E402
    compared_events,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analysed_epoch_ids,
)
from continuum_swr_replay.parameters import FIGURE_DIR  # noqa: E402
from continuum_swr_replay.paths import inventory_comparison_root  # noqa: E402
from continuum_swr_replay.runs.cli import parse_run_id  # noqa: E402
from continuum_swr_replay.runs.device import default_jax_platforms  # noqa: E402
from continuum_swr_replay.runs.provenance import produced_by  # noqa: E402
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY  # noqa: E402
from scripts.detection._report import (  # noqa: E402
    PENDING,
    PENDING_DETECTORS,
    PENDING_SENTENCE,
    QUANTILE_COLUMNS,
    QUANTILES,
    caption_with_pending,
    per_session,
    read_run,
    with_pending,
)

PRODUCED_BY = produced_by(__file__)
FIGURE_ROOT = FIGURE_DIR / "inventory_comparison"
REPORT_DIR = "report"
"""The report's folder inside the inventory run's."""
SESSION = list(SESSION_COLUMNS_DAY)

AGREEMENTS = ("agree", "disagree", "unknown")

_ANIMAL_SUMS = (
    "Per animal: counts and seconds summed over its sessions, proportions taken "
    "from those sums, never averaged over sessions. A session where the "
    "comparison was unavailable is not summed; it is listed in "
    "unavailable_sessions, and n_sessions counts the sessions summed."
)
_SESSION_ROWS = "Per session (one analysed epoch, nested in its animal)."


_TOPICS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        "native_durations": (
            "Native duration distributions of each detector's compared events "
            "(speed-eligible and not set aside): interval_span_s (event_end_s - "
            "event_start_s, defined the same for every detector) and "
            "native_duration_s (the detector's own convention), both shown. They "
            "come before any view split by duration.",
            _SESSION_ROWS,
            "Per animal, the compared events of all its sessions pooled.",
        ),
        "counts": (
            "Native counts beside eligible counts: n_native = n_eligible + "
            "n_moving + n_set_aside; n_in_exposure counts the eligible events "
            "wholly inside the detector's own exposure (exposure_s), and "
            "rate_per_s is n_in_exposure over exposure_s. native_note says why "
            "the mua and state_space native counts equal their eligible counts.",
            _SESSION_ROWS + " available says whether the detector observed any "
            "of the epoch.",
            _ANIMAL_SUMS + " A detector with no observation of a session is "
            "unavailable there.",
        ),
        "set_aside": (
            "Events set aside, by detector and reason: the analysed-time rule's "
            "drop reasons, and bridging (an event spanning its own detector's "
            "unobserved time).",
            _SESSION_ROWS + " Only reasons with events have rows.",
            "Per animal, summed over its sessions. A detector unavailable in a "
            "session sets nothing aside there; those sessions are listed in "
            "counts_by_animal.",
        ),
        "exclusion": (
            "Exposure and excluded time by reason for each comparison (every "
            "pair, all nine, and each SWR-matrix population): the "
            "shared_exposure row is the comparison's shared exposure, and every "
            "other second of the epoch is under the first reason that explains "
            "it, so a comparison's rows sum to the epoch span. A detector's "
            "unavailability appears as its own reason, '<detector>: <reason>'.",
            _SESSION_ROWS,
            "Per animal, seconds summed over all its sessions; n_sessions counts them.",
        ),
        "directional": (
            "Directional agreement for each ordered pair: of the source "
            "detector's events wholly inside the pair's shared exposure "
            "(n_eligible; n_excluded are outside it), n_matched overlap any "
            "event of the target detector for a positive duration. "
            "proportion_matched = n_matched / n_eligible.",
            _SESSION_ROWS + " available is False where the shared exposure is empty.",
            _ANIMAL_SUMS,
        ),
        "overlap": (
            "Overlap length (intersection_s) and temporal IoU of every "
            "overlapping pair of events, by detector pair. Pair counts describe "
            "the matching relation; they are not event counts. "
            "n_a_within_shared_exposure and n_b_within_shared_exposure count "
            "the pairs whose first or second event lies wholly inside the "
            "pair's shared exposure. " + QUANTILE_COLUMNS,
            _SESSION_ROWS,
            "Per animal, the pairs of all its sessions pooled.",
        ),
        "match_strength": (
            "Directional agreement under stricter rules than any overlap: of the "
            "source detector's events wholly inside the pair's shared exposure "
            "(n_eligible, as in directional), n_matched overlap a target event "
            "for a positive duration, n_covered_50pct and n_covered_90pct have "
            "one target event covering at least half or 90 % of their span, and "
            "n_iou_50pct have one target event with a temporal IoU of at least "
            "0.5. Covering tells a source event nested inside a longer target "
            "event (high cover, low IoU) apart from two events that only touch. "
            "Each p_ column is its count over n_eligible.",
            _SESSION_ROWS + " available as in directional.",
            _ANIMAL_SUMS,
        ),
        "boundary_offsets": (
            "Boundary offsets of every overlapping pair, by ordered detector "
            "pair, with the source event wholly inside the pair's shared "
            "exposure: onset_difference_s is the target event's start less the "
            "source event's start, offset_difference_s the target's end less "
            "the source's end, so a target that wraps its source has a negative "
            "onset and a positive offset difference. Pair counts describe the "
            "matching relation; they are not event counts. " + QUANTILE_COLUMNS,
            _SESSION_ROWS,
            "Per animal, the pairs of all its sessions pooled.",
        ),
        "time_coverage": (
            "Time coverage of each pair of inventories: the union and "
            "intersection of their events, clipped to the pair's shared "
            "exposure, in seconds and as fractions of it.",
            _SESSION_ROWS + " available is False where the shared exposure is empty.",
            _ANIMAL_SUMS,
        ),
        "recount": (
            "The state-space fragment recount against the native count: "
            "directional agreement of every ordered pair with state_space as "
            "source or target, with the native state-space events "
            "(segmentation native) and with state-space events merged across "
            "gaps shorter than each stated gap (recount, gap <g> ms). For "
            "state_space as source, n_source_events is the native or recounted "
            "event count. The native count remains primary.",
            _SESSION_ROWS,
            _ANIMAL_SUMS,
        ),
        "cells": (
            "The cells pairwise counts miss: each source detector's events found "
            "by exactly one detector (exactly_one_detector, over the eight "
            "without memoryless_criteria), SWRs matched by no decoder "
            "(swr_matched_by_no_decoder), and sequences (state_space, "
            "memoryless and memoryless_criteria, each its own source) matched "
            "by no LFP- or rate-triggered inventory "
            "(sequence_matched_by_no_trigger). "
            "n_unmatched counts unmatched memberships only; n_unknown, the "
            "events not observed by every detector tested against, is reported "
            "apart and never enters n_unmatched.",
            _SESSION_ROWS + " available says whether the source detector "
            "observed any of the epoch.",
            _ANIMAL_SUMS + " A session where the source detector is unavailable "
            "is listed.",
        ),
        "no_detector_time": (
            "Time accounted for by no detector: the shared exposure of the "
            "eight without memoryless_criteria, less the union of their events "
            "(undetected_s); undetected_fraction is undetected_s over "
            "shared_exposure_s.",
            _SESSION_ROWS + " available is False where the shared exposure is empty.",
            _ANIMAL_SUMS,
        ),
        "swr_matrix": (
            "The SWR label matrix: each sequence cohort labelled under each of "
            "the five SWR definitions. n_eligible counts the cohort's events "
            "wholly inside its exposure and every definition's (the same events "
            "in every column; n_excluded are the rest); non_swr_fraction = "
            "n_non_swr / n_eligible, with the definition named in its row. "
            "Unknown labels never count as non-SWR.",
            _SESSION_ROWS + " available is False where the matrix population "
            "does not exist.",
            _ANIMAL_SUMS,
        ),
        "paired_labels": (
            "Paired label agreement: for each overlapping pair of sequence "
            "events and each SWR definition, whether their two labels agree, "
            "disagree, or either is unknown (n_unknown, counted apart).",
            _SESSION_ROWS + " available says whether both cohorts' SWR-matrix "
            "populations exist in the session.",
            _ANIMAL_SUMS,
        ),
        "definition_disagreement": (
            "Definition disagreement: per cohort, the sequence events labelled "
            "SWR by at least one definition and non-SWR by another "
            "(n_disagree); n_with_unknown counts events with at least one "
            "unknown label, and n_agree_all_known the events every definition "
            "labelled the same.",
            _SESSION_ROWS + " available says whether the cohort's SWR-matrix "
            "population exists in the session, as in swr_matrix.",
            _ANIMAL_SUMS,
        ),
        "components": (
            "Connected components of the overlap graph over the eight "
            "inventories without memoryless_criteria (every published "
            "memoryless event is also a criteria event, so with both a "
            "component would count one detector twice): their sizes (n_events, n_detectors) and spans "
            "(span_s, first onset to last offset). Components are diagnostics, "
            "not event intervals. " + QUANTILE_COLUMNS,
            _SESSION_ROWS,
            "Per animal, the components of all its sessions pooled.",
        ),
    }
)

TABLE_CAPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "epochs": caption_with_pending(
            "Every epoch of the run: ok or failed (with the manifest's reason "
            "or error), and the detectors unavailable in each ok epoch. The "
            "analysed epochs the run does not hold are counted in report.md."
        ),
        "unavailable_detectors": caption_with_pending(
            "Each session and detector with no observed time, and the reason "
            "its observation table gives."
        ),
        **{
            f"{topic}_{level}": caption_with_pending(what, detail)
            for topic, (what, by_session, by_animal) in _TOPICS.items()
            for level, detail in (("by_session", by_session), ("by_animal", by_animal))
        },
    }
)
"""Every report table, in the order ``report.md`` lists it, with its caption."""

FIGURE_CAPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "native_durations.png": caption_with_pending(
            "Native duration distributions per detector, before any view split "
            "by duration: interval_span_s and native_duration_s of each "
            "detector's compared events, pooled over every session, on "
            "logarithmic bins from 1 ms to 10 s; values outside that range are "
            "drawn in the edge bins. n is the number of compared events."
        ),
        "directional_agreement.png": caption_with_pending(
            "Directional agreement per animal: the proportion of the source "
            "detector's eligible events (rows) that overlap any event of the "
            "target detector (columns), n_matched over n_eligible summed over "
            "the animal's available sessions. Grey: no eligible events. "
            "Counts in directional_by_animal.csv."
        ),
        "overlap_distributions.png": caption_with_pending(
            "Overlap length (intersection_s) and IoU of every overlapping pair "
            "of events, by detector pair, pooled over every session: the median "
            "(dot) and the 10th to 90th percentiles (bar); n is the number of "
            "pairs, which describe the matching relation, not events."
        ),
        "swr_matrix.png": caption_with_pending(
            "The SWR label matrix per animal: non_swr_fraction of each sequence "
            "cohort (rows) under each SWR definition (columns), n_non_swr over "
            "n_eligible summed over the animal's available sessions, printed in "
            "each cell to two places; the counts are in "
            "swr_matrix_by_animal.csv. Grey: no eligible events."
        ),
        "components.png": caption_with_pending(
            "Overlap-graph components over the eight inventories without "
            "memoryless_criteria, pooled over every session: the number of "
            "events in each (left) and its span, first onset to last offset "
            "(right, logarithmic bins from 1 ms to 10 s, values outside drawn in "
            "the edge bins)."
        ),
    }
)
"""Every figure, by file name, with its caption."""


def epoch_status(manifest: Mapping) -> pd.DataFrame:
    """Every epoch of the run: its status, reason and unavailable detectors.

    A failed epoch's reason is its record's ``reason``, or else its
    ``error``. ``pending`` is :data:`PENDING` where a memoryless inventory is
    among the unavailable detectors.
    """
    rows = []
    for identifier, record in sorted(manifest["epochs"].items()):
        unavailable = sorted(record.get("unavailable", {}))
        rows.append(
            {
                "epoch_id": identifier,
                "status": record["status"],
                "reason": record.get("reason") or record.get("error", ""),
                "unavailable_detectors": "; ".join(unavailable),
                "pending": PENDING if PENDING_DETECTORS & set(unavailable) else "",
            }
        )
    return pd.DataFrame(
        rows,
        columns=["epoch_id", "status", "reason", "unavailable_detectors", "pending"],
    )


def _distribution(
    frame: pd.DataFrame, keys: Sequence[str], columns: Sequence[str], count: str
) -> pd.DataFrame:
    """Per ``keys`` group, the row count and each column's percentiles."""
    named: dict = {count: (columns[0], "size")}
    for column in columns:
        named[f"{column}_n"] = (column, "count")
        for label, q in QUANTILES.items():
            named[f"{column}_{label}"] = (column, lambda x, q=q: x.quantile(q))
        named[f"{column}_max"] = (column, "max")
    if frame.empty:
        return pd.DataFrame(columns=[*keys, *named])
    return frame.groupby(list(keys)).agg(**named).reset_index()


COVERED_FRACTIONS = (0.5, 0.9)
"""The fractions of a source event one target event must cover (match_strength)."""
MATCH_IOU = 0.5
"""The temporal IoU one target event must reach (match_strength)."""
MATCH_STRENGTH_COUNTS = (
    "n_matched",
    *(f"n_covered_{round(f * 100)}pct" for f in COVERED_FRACTIONS),
    f"n_iou_{round(MATCH_IOU * 100)}pct",
)


def oriented_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Every overlapping pair from each event's side, as source and target.

    Returns
    -------
    DataFrame
        Two rows per pair: the session columns, ``source_detector``,
        ``target_detector``, ``source_event_key``, ``fraction_covered`` (of the
        source), ``iou``, ``onset_difference_s`` and ``offset_difference_s``
        (target less source) and ``source_within_shared_exposure``.
    """

    def side(source: str, target: str, sign: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                **{column: pairs[column].to_numpy() for column in SESSION},
                "source_detector": pairs[f"detector_{source}"].to_numpy(),
                "target_detector": pairs[f"detector_{target}"].to_numpy(),
                "source_event_key": pairs[f"event_key_{source}"].to_numpy(),
                "fraction_covered": pairs[f"fraction_{source}_covered"].to_numpy(),
                "iou": pairs["iou"].to_numpy(),
                # the pair table's differences are b less a
                "onset_difference_s": sign * pairs["onset_difference_s"].to_numpy(),
                "offset_difference_s": sign * pairs["offset_difference_s"].to_numpy(),
                "source_within_shared_exposure": pairs[
                    f"{source}_within_shared_exposure"
                ].to_numpy(dtype=bool),
            }
        )

    return pd.concat([side("a", "b", 1.0), side("b", "a", -1.0)], ignore_index=True)


def match_strength_by_session(
    pairs: pd.DataFrame, directional: pd.DataFrame
) -> pd.DataFrame:
    """Each ordered pair's directional agreement under the stricter rules.

    A source event's best match is the largest fraction of it one target
    event covers and the largest IoU one target event reaches.

    Raises
    ------
    ValueError
        The source events the pair table matches are not the directional
        table's ``n_matched``: the two tables disagree on the eligible events.
    """
    ordered = ["source_detector", "target_detector"]
    oriented = oriented_pairs(pairs)
    oriented = oriented[oriented.source_within_shared_exposure]
    best = oriented.groupby([*SESSION, *ordered, "source_event_key"]).agg(
        fraction_covered=("fraction_covered", "max"), iou=("iou", "max")
    )
    flags = {"n_paired": np.ones(len(best), dtype=bool)} | {
        f"n_covered_{round(f * 100)}pct": best.fraction_covered.to_numpy() >= f
        for f in COVERED_FRACTIONS
    }
    flags[f"n_iou_{round(MATCH_IOU * 100)}pct"] = best.iou.to_numpy() >= MATCH_IOU
    counts = (
        pd.DataFrame(flags, index=best.index)
        .groupby(level=[*SESSION, *ordered])
        .sum()
        .reset_index()
    )
    merged = directional[[*SESSION, *ordered, "available", "n_eligible", "n_matched"]]
    merged = merged.merge(counts, on=[*SESSION, *ordered], how="left", validate="1:1")
    stricter = [c for c in MATCH_STRENGTH_COUNTS if c != "n_matched"]
    merged[["n_paired", *stricter]] = (
        merged[["n_paired", *stricter]].fillna(0).astype(np.int64)
    )
    disagree = merged.n_paired != merged.n_matched
    if disagree.any():
        bad = merged[disagree].iloc[0]
        raise ValueError(
            f"{session_names(merged[disagree]).iloc[0]}: {bad.source_detector} -> "
            f"{bad.target_detector} has {bad.n_paired} source events in the pair "
            f"table but n_matched {bad.n_matched} in the directional table."
        )
    return _with_proportions(merged.drop(columns="n_paired"))


def _with_proportions(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with ``p_<x>`` = ``n_<x>`` / ``n_eligible`` for each count."""
    return frame.assign(
        **{
            "p_" + c.removeprefix("n_"): ratio(frame[c], frame.n_eligible)
            for c in MATCH_STRENGTH_COUNTS
        }
    )


def match_strength_by_animal(by_session: pd.DataFrame) -> pd.DataFrame:
    """The stricter-rule counts pooled per animal and ordered pair."""
    pooled = pool_sessions(
        by_session,
        by=["source_detector", "target_detector"],
        sums=["n_eligible", *MATCH_STRENGTH_COUNTS],
        available=by_session.available,
    )
    return _with_proportions(pooled)


def swr_matrix_by_animal(matrix: pd.DataFrame) -> pd.DataFrame:
    """The SWR matrix pooled per animal; the fraction is over summed eligible."""
    pooled = pool_sessions(
        matrix,
        by=["cohort", "swr_definition"],
        sums=["n_events", "n_eligible", "n_excluded", "n_swr", "n_non_swr"],
        available=matrix.available,
    )
    return pooled.assign(non_swr_fraction=ratio(pooled.n_non_swr, pooled.n_eligible))


def matrix_population(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per session and cohort, whether the SWR-matrix population exists.

    The runner gives every definition of a cohort the same ``available``;
    the population is taken as available only where all of them are.
    """
    return matrix.groupby([*SESSION, "cohort"]).available.all().reset_index()


def _with_counts(
    skeleton: pd.DataFrame, counted: pd.DataFrame, keys: Sequence[str]
) -> pd.DataFrame:
    """``counted`` on ``skeleton``'s rows, zeros where it has none.

    Raises
    ------
    ValueError
        ``counted`` has a row ``skeleton`` does not.
    """
    out = skeleton.merge(counted, on=list(keys), how="outer", indicator=True)
    if (out._merge == "right_only").any():
        raise ValueError(
            "Label counts of a session or cohort the matrix does not name."
        )
    counts = [c for c in counted.columns if c not in keys]
    out[counts] = out[counts].fillna(0).astype(np.int64)
    return out.drop(columns="_merge").sort_values(list(keys), ignore_index=True)


def paired_labels_by_session(
    paired_labels: pd.DataFrame, matrix: pd.DataFrame
) -> pd.DataFrame:
    """Paired label agreement counted per session, definition and cohort pair.

    One row per session, SWR definition and pair of matrix cohorts (in name
    order), zeros where no pair overlapped. ``available`` is whether both
    cohorts' matrix populations exist in the session.

    Raises
    ------
    ValueError
        A session has no matrix row for one of the cohorts.
    """
    population = matrix_population(matrix)
    cohorts = sorted(population.cohort.unique())
    skeleton = (
        population[SESSION]
        .drop_duplicates()
        .merge(
            pd.DataFrame(
                {"swr_definition": sorted(matrix.swr_definition.unique())},
                dtype=object,
            ),
            how="cross",
        )
        .merge(
            pd.DataFrame(
                list(combinations(cohorts, 2)),
                columns=["detector_a", "detector_b"],
                dtype=object,
            ),
            how="cross",
        )
    )
    for side in ("a", "b"):
        skeleton = skeleton.merge(
            population.rename(
                columns={"cohort": f"detector_{side}", "available": f"_{side}"}
            ),
            on=[*SESSION, f"detector_{side}"],
            how="left",
        )
        missing = skeleton[f"_{side}"].isna()
        if missing.any():
            rows = skeleton[missing].drop_duplicates([*SESSION, f"detector_{side}"])
            raise ValueError(
                "No SWR-matrix row for "
                + ", ".join(
                    f"{cohort} in {session}"
                    for cohort, session in zip(
                        rows[f"detector_{side}"], session_names(rows), strict=True
                    )
                )
                + "; its population is unknown."
            )
    skeleton["available"] = skeleton._a.astype(bool) & skeleton._b.astype(bool)
    keys = [*SESSION, "swr_definition", "detector_a", "detector_b"]
    counted = _count_by(paired_labels, keys, "agreement", AGREEMENTS).rename(
        columns={"n": "n_pairs"}
    )
    return _with_counts(skeleton.drop(columns=["_a", "_b"]), counted, keys)


def definition_disagreement_by_session(
    labels: pd.DataFrame, matrix: pd.DataFrame
) -> pd.DataFrame:
    """Definition disagreement counted per session and cohort.

    One row per session and matrix cohort, with the matrix population's
    ``available``.
    """
    keys = [*SESSION, "cohort"]
    counted = (
        labels.assign(
            n_disagree=labels.definitions_disagree,
            n_with_unknown=labels.n_unknown > 0,
            n_agree_all_known=(labels.n_unknown == 0) & ~labels.definitions_disagree,
        )
        .groupby(keys)
        .agg(
            n_events=("event_key", "size"),
            n_disagree=("n_disagree", "sum"),
            n_with_unknown=("n_with_unknown", "sum"),
            n_agree_all_known=("n_agree_all_known", "sum"),
        )
        .astype(np.int64)
        .reset_index()
    )
    return _with_counts(matrix_population(matrix), counted, keys)


LABEL_SUMS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "paired_labels": ("n_pairs", *(f"n_{a}" for a in AGREEMENTS)),
        "definition_disagreement": (
            "n_events",
            "n_disagree",
            "n_with_unknown",
            "n_agree_all_known",
        ),
    }
)


def labels_by_animal(by_session: pd.DataFrame, table: str) -> pd.DataFrame:
    """Paired-label or disagreement counts summed over each animal's sessions
    with the matrix population; the others are listed."""
    by = (
        ["swr_definition", "detector_a", "detector_b"]
        if table == "paired_labels"
        else ["cohort"]
    )
    return pool_sessions(
        by_session, by=by, sums=LABEL_SUMS[table], available=by_session.available
    )


def _count_by(
    frame: pd.DataFrame, keys: Sequence[str], column: str, values: Sequence[str]
) -> pd.DataFrame:
    """Per ``keys`` group, the rows and the rows of each ``column`` value."""
    return (
        frame.assign(**{f"n_{v}": frame[column] == v for v in values})
        .groupby(list(keys))
        .agg(n=(column, "size"), **{f"n_{v}": (f"n_{v}", "sum") for v in values})
        .reset_index()
    )


def report_tables(
    manifest: Mapping, tables: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Every report table, by name, in the order of :data:`TABLE_CAPTIONS`."""
    settings = manifest["settings"]
    cell_against = {name: cell["against"] for name, cell in settings["cells"].items()}
    pairs_of = ["detector_a", "detector_b"]
    ordered = ["source_detector", "target_detector"]
    by_session = session_tables(manifest, tables)
    availability = by_session["availability"]
    out: dict[str, pd.DataFrame] = {
        "epochs": epoch_status(manifest),
        "unavailable_detectors": with_pending(
            per_session(availability[~availability.available]), ["detector"]
        ),
    }

    compared = compared_events(tables["inventory"])
    durations = ["interval_span_s", "native_duration_s"]
    for level, keys in (("by_session", SESSION), ("by_animal", ["animal"])):
        out[f"native_durations_{level}"] = _distribution(
            compared, [*keys, "detector"], durations, "n_events"
        )
    counts = by_session["counts"]
    out["counts_by_session"] = per_session(counts)
    out["counts_by_animal"] = counts_by_animal(counts)

    set_aside = (
        tables["set_aside"]
        .groupby([*SESSION, "detector", "reason"])
        .size()
        .rename("n_events")
        .reset_index()
    )
    out["set_aside_by_session"] = per_session(set_aside)
    out["set_aside_by_animal"] = summed(set_aside, ["detector", "reason"], ["n_events"])

    exclusion = tables["exclusion_by_reason"]
    out["exclusion_by_session"] = per_session(exclusion)
    out["exclusion_by_animal"] = summed(
        exclusion, ["comparison", "reason"], ["seconds"]
    )

    directional = tables["directional"]
    out["directional_by_session"] = per_session(directional)
    out["directional_by_animal"] = directional_by_animal(directional)

    pairs = tables["pairs"]
    for level, keys in (("by_session", SESSION), ("by_animal", ["animal"])):
        group = [*keys, *pairs_of]
        within = (
            pairs.groupby(group)[
                ["a_within_shared_exposure", "b_within_shared_exposure"]
            ]
            .sum()
            .add_prefix("n_")
            .reset_index()
        )
        spread = _distribution(pairs, group, ["intersection_s", "iou"], "n_pairs")
        out[f"overlap_{level}"] = spread.merge(within, on=group, how="left")

    strength = match_strength_by_session(pairs, directional)
    out["match_strength_by_session"] = per_session(strength)
    out["match_strength_by_animal"] = match_strength_by_animal(strength)
    oriented = oriented_pairs(pairs)
    oriented = oriented[oriented.source_within_shared_exposure]
    for level, keys in (("by_session", SESSION), ("by_animal", ["animal"])):
        out[f"boundary_offsets_{level}"] = _distribution(
            oriented,
            [*keys, *ordered],
            ["onset_difference_s", "offset_difference_s"],
            "n_pairs",
        )

    coverage = tables["time_coverage"]
    out["time_coverage_by_session"] = per_session(coverage)
    pooled = pool_sessions(
        coverage,
        by=pairs_of,
        sums=["shared_exposure_s", "union_s", "intersection_s"],
        available=coverage.available,
    )
    out["time_coverage_by_animal"] = pooled.assign(
        union_fraction=ratio(pooled.union_s, pooled.shared_exposure_s),
        intersection_fraction=ratio(pooled.intersection_s, pooled.shared_exposure_s),
    )

    recounted = settings["recount"]["detector"]
    recount = tables["recount_directional"]
    both = pd.concat(
        [
            directional.assign(segmentation="native", gap_s=np.nan),
            recount.assign(
                segmentation="recount, gap "
                + (recount.gap_s * 1000).round().astype(np.int64).astype(str)
                + " ms"
            ),
        ],
        ignore_index=True,
    )
    both = both[
        (both.source_detector == recounted) | (both.target_detector == recounted)
    ]
    out["recount_by_session"] = per_session(both)
    out["recount_by_animal"] = directional_by_animal(both, by=["segmentation"])

    cells = by_session["cells"]
    out["cells_by_session"] = per_session(cells)
    out["cells_by_animal"] = cells_by_animal(cells)

    undetected = tables["no_detector_time"]
    out["no_detector_time_by_session"] = per_session(undetected)
    pooled = pool_sessions(
        undetected,
        by=[],
        sums=["shared_exposure_s", "detected_s", "undetected_s"],
        available=undetected.available,
    )
    out["no_detector_time_by_animal"] = pooled.assign(
        undetected_fraction=ratio(pooled.undetected_s, pooled.shared_exposure_s)
    )

    matrix = tables["swr_matrix"]
    out["swr_matrix_by_session"] = per_session(matrix)
    out["swr_matrix_by_animal"] = swr_matrix_by_animal(matrix)

    paired = paired_labels_by_session(tables["paired_labels"], matrix)
    out["paired_labels_by_session"] = per_session(paired)
    out["paired_labels_by_animal"] = labels_by_animal(paired, "paired_labels")
    disagreement = definition_disagreement_by_session(
        tables["definition_disagreement"], matrix
    )
    out["definition_disagreement_by_session"] = per_session(disagreement)
    out["definition_disagreement_by_animal"] = labels_by_animal(
        disagreement, "definition_disagreement"
    )

    components = tables["components"]
    for level, keys in (("by_session", SESSION), ("by_animal", ["animal"])):
        out[f"components_{level}"] = _distribution(
            components, keys, ["n_events", "n_detectors", "span_s"], "n_components"
        )

    marks = {
        "native_durations": {"columns": ["detector"]},
        "counts": {"columns": ["detector"]},
        "set_aside": {"columns": ["detector"]},
        "exclusion": {"groups": {"comparison": settings["comparisons"]}},
        "directional": {"columns": ordered},
        "overlap": {"columns": pairs_of},
        "match_strength": {"columns": ordered},
        "boundary_offsets": {"columns": ordered},
        "time_coverage": {"columns": pairs_of},
        "recount": {"columns": ordered},
        "cells": {"columns": ["detector"], "groups": {"cell": cell_against}},
        "swr_matrix": {"columns": ["cohort"]},
        "paired_labels": {"columns": pairs_of},
        "definition_disagreement": {"columns": ["cohort"]},
    }
    every_detector = PENDING_DETECTORS & set(settings["roster"])
    for name in list(out):
        topic = name.rsplit("_by_", 1)[0]
        if topic in marks:
            out[name] = with_pending(out[name], **marks[topic])
        elif topic in {"no_detector_time", "components"}:
            # no-detector time and the components are over the eight with
            # memoryless; a component can chain through a memoryless event
            # whatever its label
            involved = (
                settings["no_detector_time"]
                if topic == "no_detector_time"
                else every_detector
            )
            out[name] = out[name].assign(
                pending=PENDING if PENDING_DETECTORS & set(involved) else ""
            )
    # the listed sessions last, before the pending note
    return {
        name: out[name][
            [
                c
                for c in out[name].columns
                if c not in ("unavailable_sessions", "pending")
            ]
            + [c for c in ("unavailable_sessions", "pending") if c in out[name].columns]
        ]
        for name in TABLE_CAPTIONS
    }


def _label(detector: str) -> str:
    return f"{detector}†" if detector in PENDING_DETECTORS else detector


def _footnote(fig: plt.Figure, run_id: str) -> None:
    """The pending note and the run, below the panels."""
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    fig.text(
        0.01, 0.01, f"† {PENDING}\ninventory run {run_id}", fontsize=6, va="bottom"
    )


def _log_bins() -> np.ndarray:
    return np.logspace(-3, 1, 41)


def _clipped(values: pd.Series, bins: np.ndarray) -> np.ndarray:
    finite = values.to_numpy(dtype=np.float64)
    return np.clip(finite[np.isfinite(finite)], bins[0], bins[-1])


def _heatmap_panels(n_animals: int, width: float) -> tuple[plt.Figure, np.ndarray]:
    return plt.subplots(
        1,
        max(n_animals, 1),
        figsize=(max(width * n_animals, 3.6), width + 0.9),
        squeeze=False,
        constrained_layout=True,
    )


def draw_figures(
    out: Mapping[str, pd.DataFrame],
    tables: Mapping[str, pd.DataFrame],
    roster: Sequence[str],
    figure_dir: Path,
    run_id: str,
) -> list[Path]:
    """Every figure of :data:`FIGURE_CAPTIONS`, written to ``figure_dir``."""
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    paths = []

    def save(fig: plt.Figure, name: str) -> None:
        _footnote(fig, run_id)
        path = figure_dir / name
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    # native durations
    compared = compared_events(tables["inventory"])
    bins = _log_bins()
    n_rows = -(-len(roster) // 3)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(7.2, 1.7 * n_rows + 0.4),
        sharex=True,
        constrained_layout=True,
    )
    for ax, detector in zip(axes.flat, roster, strict=False):
        events = compared[compared.detector == detector]
        if events.empty:
            ax.text(
                0.5, 0.5, "no events", ha="center", transform=ax.transAxes, fontsize=6
            )
        for column, color, style in (
            ("interval_span_s", "C0", "-"),
            ("native_duration_s", "C1", "--"),
        ):
            if events.empty:
                continue
            ax.hist(
                _clipped(events[column], bins),
                bins=bins,
                histtype="step",
                color=color,
                linestyle=style,
                label=column,
            )
        ax.set_xscale("log")
        ax.set_title(f"{_label(detector)} (n = {len(events)})", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(roster) :]:
        ax.axis("off")
    axes.flat[0].legend(
        handles=[
            plt.Line2D([], [], color="C0", label="interval_span_s"),
            plt.Line2D([], [], color="C1", linestyle="--", label="native_duration_s"),
        ],
        fontsize=6,
        frameon=False,
    )
    fig.supxlabel("duration (s)", fontsize=7)
    fig.supylabel("events", fontsize=7)
    save(fig, "native_durations.png")

    # directional agreement
    directional = out["directional_by_animal"]
    animals = sorted(directional.animal.unique())
    fig, axes = _heatmap_panels(len(animals), 3.0)
    image = None
    for ax, animal in zip(axes[0], animals, strict=False):
        grid = (
            directional[directional.animal == animal]
            .pivot(
                index="source_detector",
                columns="target_detector",
                values="proportion_matched",
            )
            .reindex(index=list(roster), columns=list(roster))
        )
        image = ax.imshow(grid.to_numpy(np.float64), vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(
            range(len(roster)), [_label(d) for d in roster], rotation=90, fontsize=6
        )
        ax.set_yticks(range(len(roster)), [_label(d) for d in roster], fontsize=6)
        ax.set_xlabel("target", fontsize=7)
        ax.set_ylabel("source", fontsize=7)
        ax.set_title(animal, fontsize=8)
    if image is None:
        axes[0, 0].text(0.5, 0.5, "no directional rows", ha="center", fontsize=7)
    else:
        bar = fig.colorbar(image, ax=axes[0], shrink=0.8)
        bar.set_label("proportion of eligible source events matched", fontsize=7)
        bar.ax.tick_params(labelsize=6)
    save(fig, "directional_agreement.png")

    # overlap distributions
    pairs = tables["pairs"]
    groups = list(pairs.groupby(["detector_a", "detector_b"]))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 0.16 * max(len(groups), 4) + 0.9),
        sharey=True,
        constrained_layout=True,
    )
    for ax, column, scale in (
        (axes[0], "intersection_s", "log"),
        (axes[1], "iou", "linear"),
    ):
        for y, (_pair, rows) in enumerate(groups):
            low, mid, high = rows[column].quantile([0.1, 0.5, 0.9])
            ax.hlines(y, low, high, color="0.4", lw=1)
            ax.plot(mid, y, "o", color="k", ms=2.5)
        ax.set_xscale(scale)
        if scale == "log":
            ax.xaxis.set_major_locator(
                matplotlib.ticker.FixedLocator(
                    [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
                )
            )
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(column, fontsize=7)
        ax.tick_params(which="both", labelsize=6)
    axes[0].set_yticks(
        range(len(groups)),
        [f"{_label(a)} + {_label(b)} (n = {len(rows)})" for (a, b), rows in groups],
    )
    axes[0].invert_yaxis()
    if not groups:
        axes[0].text(
            0.5, 0.5, "no overlapping pairs", transform=axes[0].transAxes, fontsize=7
        )
    save(fig, "overlap_distributions.png")

    # SWR matrix
    matrix = out["swr_matrix_by_animal"]
    cohorts = list(dict.fromkeys(matrix.cohort))
    definitions = list(dict.fromkeys(tables["swr_matrix"].swr_definition))
    animals = sorted(matrix.animal.unique())
    fig, axes = _heatmap_panels(len(animals), 2.4)
    image = None
    for ax, animal in zip(axes[0], animals, strict=False):
        rows = matrix[matrix.animal == animal].set_index(["cohort", "swr_definition"])
        grid = np.full((len(cohorts), len(definitions)), np.nan)
        for i, cohort in enumerate(cohorts):
            for j, definition in enumerate(definitions):
                if (cohort, definition) not in rows.index:
                    continue
                row = rows.loc[(cohort, definition)]
                grid[i, j] = row.non_swr_fraction
                ax.text(
                    j,
                    i,
                    f"{row.non_swr_fraction:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="k" if row.non_swr_fraction > 0.6 else "w",
                )
        image = ax.imshow(grid, vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(range(len(definitions)), definitions, rotation=90, fontsize=6)
        ax.set_yticks(range(len(cohorts)), [_label(c) for c in cohorts], fontsize=6)
        ax.set_title(animal, fontsize=8)
    if image is None:
        axes[0, 0].text(0.5, 0.5, "no matrix rows", ha="center", fontsize=7)
    else:
        bar = fig.colorbar(image, ax=axes[0], shrink=0.8)
        bar.set_label("non_swr_fraction", fontsize=7)
        bar.ax.tick_params(labelsize=6)
    save(fig, "swr_matrix.png")

    # components
    components = tables["components"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), constrained_layout=True)
    largest = int(components.n_events.max()) if len(components) else 1
    axes[0].hist(components.n_events, bins=np.arange(0.5, largest + 1.5), color="0.4")
    if len(components):
        axes[0].set_yscale("log")
    axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_xlabel("events per component", fontsize=7)
    axes[1].hist(_clipped(components.span_s, bins), bins=bins, color="0.4")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("component span (s)", fontsize=7)
    for ax in axes:
        ax.set_ylabel("components", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(
        f"{len(components)} components over the eight without memoryless_criteria†",
        fontsize=8,
    )
    save(fig, "components.png")
    return paths


def write_report(run_dir: Path, figure_root: Path) -> dict[str, list[Path]]:
    """Write the run's report tables, figures and captions.

    Tables go to ``run_dir/report/``; figures and ``captions.md`` to
    ``figure_root/<run id>/``, so runs do not overwrite each other.

    Returns
    -------
    dict
        ``tables`` and ``figures``: the paths written.
    """
    manifest, tables = read_run(run_dir, inventory_run.TABLE_DTYPES)
    out = report_tables(manifest, tables)
    run_id = run_dir.name
    figure_dir = figure_root / run_id
    outside = len(set(analysed_epoch_ids()) - set(manifest["epochs"]))
    report_dir = run_dir / REPORT_DIR
    report_dir.mkdir(exist_ok=True)
    table_paths = []
    for name, frame in out.items():
        path = report_dir / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.10g")
        table_paths.append(path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = draw_figures(
        out, tables, list(manifest["settings"]["roster"]), figure_dir, run_id
    )

    status = out["epochs"].status.value_counts()
    lines = [
        f"# Inventory comparison report: run {run_id}",
        "",
        f"Written by `{PRODUCED_BY}` from the per-epoch tables of `{run_dir}` "
        f"({status.get('ok', 0)} ok epochs, {status.get('failed', 0)} failed, "
        f"and {outside} analysed epochs of the epoch register outside the run).",
        "",
        PENDING_SENTENCE,
        "",
        "## Tables",
        "",
    ]
    for name, caption in TABLE_CAPTIONS.items():
        lines += [f"### `{name}.csv` ({len(out[name])} rows)", "", caption, ""]
    lines += ["## Figures", ""]
    for name, caption in FIGURE_CAPTIONS.items():
        lines += [f"### `{figure_dir / name}`", "", caption, ""]
    (report_dir / "report.md").write_text("\n".join(lines))
    captions = [f"# Inventory comparison figures: run {run_id}", ""]
    for name, caption in FIGURE_CAPTIONS.items():
        captions += [f"## `{name}`", "", caption, ""]
    (figure_dir / "captions.md").write_text("\n".join(captions))
    return {"tables": table_paths, "figures": figure_paths}


def main(argv: list[str] | None = None) -> None:
    # CPU only: the package imports JAX, which the report never needs
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-id",
        type=parse_run_id,
        required=True,
        help="The inventory run's folder name, under "
        "replay_detection_comparison/inventory/.",
    )
    args = parser.parse_args(argv)
    run_dir = inventory_comparison_root() / args.run_id
    if not (run_dir / "manifest.json").exists():
        sys.exit(f"No inventory run manifest at {run_dir}.")
    written = write_report(run_dir, FIGURE_ROOT)
    print(
        f"{len(written['tables'])} tables -> {run_dir / REPORT_DIR}; "
        f"{len(written['figures'])} figures -> {FIGURE_ROOT / args.run_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
