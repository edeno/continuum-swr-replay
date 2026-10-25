#!/usr/bin/env python
"""Report one event features run: its features per session and per animal.

Reads the per-epoch tables ``scripts/detection/run_event_features.py`` wrote for the
run's ``ok`` epochs (``features``, ``pair_membership``, ``rejection``) and its
``manifest.json``, the stored ``cells`` and ``directional`` tables of the
inventory comparison run it read, and the memoryless run's candidate tables.
Writes plain tables, one CSV each, to ``<features run>/report/`` with a
``report.md`` that captions them, and a few figures to
``paper/figures/event_features/<run id>/`` with a ``captions.md``. Design:
``docs/design/replay_detection_comparison/04-event-features.md`` (Initial
report).

The populations are ``populations.population_rows`` of the compared events,
rebuilt from the run's pair memberships and the comparison's cells and
directional tables: ``all``, ``matched_by:<B>``, ``unmatched_by:<B>``,
``unknown_by:<B>`` (with its ``unknown_reason``), each cell's unmatched
events (``<cell>``) with ``<cell>:matched`` and ``<cell>:unknown``. An
``unknown`` membership is its own population, never pooled as unmatched.
There are no baseline windows: the lead dropped the baseline comparison from
this stage (2026-10-18), since windows matched on duration and location but
not on spike count measured engagement more than evidence. A run written
before that decision holds baseline rows (detector ``baseline``); they are
dropped on reading. The evidence is reported beside the spikes per row.

Every table comes per session (one analysed epoch, nested in its animal) and
per animal. Counts are summed over each animal's sessions and proportions are
taken from those sums, never averaged over sessions. Feature distributions are
quantiles of the events, per session and pooled per animal. The session tables
hold the pairwise populations against ``state_space`` and ``memoryless`` only;
the animal tables hold them against every detector.

Every row with a memoryless or memoryless_criteria inventory (as the detector,
or as the B or a cell's test of a population) and every ``ms_*`` feature carry
``pending`` = "pending the Phase 2 pilot checks", and every caption says so.
The report states counts only.

Usage:
    python scripts/detection/report_event_features.py --run-id <event features run id>
"""

import argparse
import os
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

# CPU only: the package imports JAX, which the report never needs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from continuum_swr_replay.analysis.hippocampus.detection_inputs import (  # noqa: E402
    memoryless_output_paths as memoryless_outputs,
)
from continuum_swr_replay.analysis.inventory_comparison.populations import (  # noqa: E402
    ALL,
    BASELINE,
    NO_SHARED_EXPOSURE,
    OUTSIDE_SHARED_EXPOSURE,
    POPULATION_DTYPES,
    population_rows,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (  # noqa: E402
    MEMORYLESS,
    STATE_SPACE,
    _event_keys,
    typed_table,
)
from continuum_swr_replay.analysis.inventory_comparison.selection import (  # noqa: E402
    REJECTION_DTYPES,
    REJECTION_STAGES,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analysed_epoch_ids,
)
from continuum_swr_replay.decoding.identifiers import (  # noqa: E402
    parse_epoch_identifier,
)
from continuum_swr_replay.parameters import FIGURE_DIR  # noqa: E402
from continuum_swr_replay.paths import (  # noqa: E402
    EVENT_FEATURES_ROOT,
    INVENTORY_COMPARISON_ROOT,
    MEMORYLESS_ROOT,
)
from continuum_swr_replay.runs.cli import parse_run_id  # noqa: E402
from continuum_swr_replay.runs.provenance import produced_by  # noqa: E402
from continuum_swr_replay.tables.io import read_table  # noqa: E402
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY  # noqa: E402
from scripts.detection import run_event_features as features_runner  # noqa: E402
from scripts.detection import run_inventory_comparison as comparison  # noqa: E402
from scripts.detection._report import (  # noqa: E402
    MEMBERSHIPS,
    PENDING,
    PENDING_DETECTORS,
    PENDING_SENTENCE,
    QUANTILE_COLUMNS,
    QUANTILES,
    caption_with_pending,
    per_session,
    pool_sessions,
    ratio,
    read_run,
    summed,
    with_pending,
)
from scripts.figures.generate_inventory_examples import (  # noqa: E402
    detector_color,
)

PRODUCED_BY = produced_by(__file__)
OUTPUT_ROOT = EVENT_FEATURES_ROOT
"""Where the event features runs are."""
INVENTORY_ROOT = INVENTORY_COMPARISON_ROOT
FIGURE_ROOT = FIGURE_DIR / "event_features"
REPORT_DIR = "report"
"""The report's folder inside the run's."""
SESSION = list(SESSION_COLUMNS_DAY)
POPULATION = ["detector", "population", "unknown_reason"]

SESSION_PAIRS = (STATE_SPACE, MEMORYLESS)
"""The detectors B whose pairwise populations the session tables hold; the
animal tables hold every B."""
BOUNDARY_TABLES = ("boundary_offsets", "match_strength", "recount")
"""The inventory comparison report's tables for the boundaries, linked from
``report.md``."""

FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "timing": (
            "interval_span_s",
            "native_duration_s",
            "previous_gap_s",
            "next_gap_s",
        ),
        "engagement": ("n_spikes", "n_active_tetrodes", "spike_rate_hz"),
        "lfp_mua": ("ripple_z_max", "ripple_z_mean", "mua_z_max", "mua_z_mean"),
        "evidence": (
            "evidence_dll",
            "evidence_n_valid_rows",
            "evidence_valid_fraction",
            "spikes_per_row",
        ),
        "ss": (
            "ss_local_prob_mean",
            "ss_no_spike_prob_mean",
            "ss_continuous_prob_mean",
            "ss_fragmented_prob_mean",
            "ss_nonlocal_prob_max",
            "ss_frac_rows_nonlocal",
            "ss_extent_cm",
            "ss_distance_cm_mean",
            "ss_hpd_cm_mean",
            "ss_n_runs_nonlocal",
            "ss_longest_run_nonlocal_s",
            "ss_nearest_event_gap_s",
        ),
        "ms": (
            "ms_n_windows",
            "ms_n_windows_decoded",
            "ms_n_windows_with_center",
            "ms_spread_cm_mean",
            "ms_extent_cm",
            "ms_distance_cm_mean",
            "ms_frac_windows_kept",
        ),
        "context": ("linear_position_cm", "time_in_stop_s", "time_since_arrival_s"),
    }
)
"""The numeric features of each group, whose quantiles the features tables
give."""
DERIVED_FEATURES = ("spikes_per_row",)
"""Features the report derives from the features table: ``spikes_per_row`` is
``n_spikes / n_rows``, NaN for an interval with no row."""
SPIKES_PER_ROW_BINS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf)
"""Bounds of the spikes-per-row bins, each closed on the left."""
SPIKES_PER_ROW_LABELS = ("0-0.25", "0.25-0.5", "0.5-1", "1-2", "2-4", ">=4")
LABELLED_FEATURES = (
    "track_segment_id",
    "patch_id",
    "stop_id",
    "at_well",
    "occupancy_state",
    "occupancy_well_id",
    "current_patch",
    "is_rewarded_outcome",
    "ss_disagreement",
)
"""Features that are labels, not quantities: counted in ``missing_by_*`` only
(``ss_disagreement`` also in ``comparator_only_by_*``)."""
MISSING_FEATURES = (
    *(column for columns in FEATURE_GROUPS.values() for column in columns),
    *LABELLED_FEATURES,
)

UNSCORABLE_REASONS = ("no_row", "no_valid_row")
"""``evidence_unscorable``'s reasons; ``""`` is a scored interval."""
UNKNOWN_REASONS = (NO_SHARED_EXPOSURE, OUTSIDE_SHARED_EXPOSURE)
REJECTION_COUNTS = tuple(
    name for name, dtype in REJECTION_DTYPES.items() if dtype == "Int64"
)
"""The rejection table's window and candidate counts, summed."""
REJECTION_FLAGS = tuple(
    name for name, dtype in REJECTION_DTYPES.items() if dtype == "boolean"
)
"""The rejection table's candidate flags, counted as events."""
DISAGREEMENTS = ("near", "subthreshold", "sustained", "missing")
"""``ss_disagreement``'s classes; ``missing`` where the runs are unknown."""
COMPOSITION = (
    "ss_local_prob_mean",
    "ss_no_spike_prob_mean",
    "ss_continuous_prob_mean",
    "ss_fragmented_prob_mean",
    "ss_nonlocal_prob_max",
    "ss_frac_rows_nonlocal",
    "ss_nearest_event_gap_s",
)
"""The state-space composition of the comparator-only events."""
CANDIDATE_COLUMNS: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "ms_event_id": np.int64,
        "is_replay": np.bool_,
        "shuffle_pass": np.bool_,
        "shuffle_testable": np.bool_,
        "n_shuffles": np.int64,
        "artifact_drop_reason": object,
        "whole_event_speed_eligible": np.bool_,
    }
)
"""The columns of the memoryless run's candidate tables the report reads."""
COHORT_COUNTS = (
    "n_candidates",
    "n_criteria",
    "n_published",
    "n_shuffle_failed",
    "n_shuffle_untestable",
)
CONTEXT_CATEGORIES: Mapping[str, tuple[object, ...] | None] = MappingProxyType(
    {
        "at_well": (True, False),
        "occupancy_state": None,
        "current_patch": (1, 2, 3),
        "is_rewarded_outcome": (1, 0),
    }
)
"""The categorical context counted per category, each with its missing
events; ``None`` takes every value the run holds."""
COMPARATOR_COUNTS = ("n_events", "n_scorable")

_POPULATIONS = (
    "Populations: all (every compared event of the detector); matched_by:<B> "
    "and unmatched_by:<B> (its membership against detector B); unknown_by:<B>, "
    "counted apart and never as unmatched, with unknown_reason "
    "(no_shared_exposure: the pair shares no exposure in the epoch; "
    "outside_shared_exposure: the event is not wholly inside it); each cell's "
    "unmatched events (exactly_one_detector, swr_matched_by_no_decoder, "
    "sequence_matched_by_no_trigger) with <cell>:matched and <cell>:unknown "
    "beside it."
)
_SESSION_ROWS = (
    "Per session (one analysed epoch, nested in its animal), with the pairwise "
    "populations against state_space and memoryless only."
)
_ANIMAL_EVENTS = (
    "Per animal, the events of all its sessions pooled, with the pairwise "
    "populations against every detector."
)
_ANIMAL_SUMS = (
    "Per animal: counts summed over its sessions (n_sessions), proportions "
    "taken from those sums, never averaged over sessions."
)
_EVIDENCE_NOTE = (
    "evidence_dll is the mean, over an interval's rows with evidence, of each "
    "row's log-likelihood difference between the best remote real place bin "
    "and the animal's bin. It grows with the number of spikes in the row, so "
    "it is read beside spikes_per_row (n_spikes / n_rows; evidence_coverage_by_* "
    "and evidence_by_spikes_*). It is a diagnostic: it is not a calibrated "
    "declaration of replay. No baseline is compared: the lead dropped the "
    "baseline windows, pending a design matched on spike count."
)
_LFP_MUA = (
    "ripple_z_* reads the decode frame's SWR_zscore: the Kay consensus over "
    "the tetrodes rule R-T kept, normalized outside the artifact mask and "
    "nearest-sampled onto the 500 Hz decode grid, so a peak can be one sample "
    "off. mua_z_* reads the frame's MUA_zscore."
)


def _feature_topic(group: str, columns: Sequence[str], stratified: bool) -> str:
    what = (
        f"Quantiles of the {group} features ({', '.join(columns)}) of each "
        "detector's events, by population."
    )
    if stratified:
        what += (
            " Split by duration_stratum (the interval span's stratum, in the "
            "run's order), beside the native tables."
        )
    if group == "ms":
        what += " Every ms_* feature reads the memoryless run, so every row is pending."
    if group == "lfp_mua":
        what += " " + _LFP_MUA
    if group == "evidence":
        what += " " + _EVIDENCE_NOTE
    return " ".join((what, _POPULATIONS, QUANTILE_COLUMNS))


_TOPICS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        **{
            f"features_{group}{'_by_stratum' if stratified else ''}": (
                _feature_topic(group, columns, stratified),
                _SESSION_ROWS,
                _ANIMAL_EVENTS,
            )
            for group, columns in FEATURE_GROUPS.items()
            for stratified in (False, True)
        },
        "missing": (
            "Per detector, population and feature: n events, and n_missing of "
            "them with the feature missing (NaN or NA). A value missing by its "
            "definition is counted the same: the gaps at an epoch's first and "
            "last event, ss_longest_run_nonlocal_s with no run, "
            "ss_nearest_event_gap_s and ss_disagreement of state_space's own "
            "events, the stop context of an event in no stop, and "
            "spikes_per_row of an interval with no row. Labels "
            f"({', '.join(LABELLED_FEATURES)}) are counted here and in "
            "context_by_*. " + _POPULATIONS,
            _SESSION_ROWS + " Features with no missing value in any session of "
            "the run are left out; report.md names them.",
            _ANIMAL_SUMS + " Every feature.",
        ),
        "evidence_coverage": (
            "The spike evidence's coverage per detector: n_events; n_scorable "
            "(an evidence score); n_unscorable_no_row (no grid row) and "
            "n_unscorable_no_valid_row (rows, none valid); the rows (n_rows), "
            "valid rows (n_valid_rows) and spikes (n_spikes) summed, a missing "
            "count (an interval with no row) summed as 0; and spikes_per_row, "
            "n_spikes over n_rows. " + _EVIDENCE_NOTE,
            "Per session (one analysed epoch, nested in its animal).",
            _ANIMAL_SUMS + " spikes_per_row is taken from the sums.",
        ),
        "evidence_by_spikes": (
            "Quantiles of evidence_dll of each detector's events, by population "
            "and by spikes_per_row bin ("
            + ", ".join(SPIKES_PER_ROW_LABELS)
            + "; each bin closed on the left; an interval with no row is in "
            "none). " + _EVIDENCE_NOTE + " " + _POPULATIONS + " " + QUANTILE_COLUMNS,
            _SESSION_ROWS,
            _ANIMAL_EVENTS,
        ),
        "context": (
            "The categorical context of each detector's events, per population: "
            "n_events and, per category, n_<feature>_<value> and n_<feature>_"
            "missing (the event's onset in no stop, or the stop's value "
            "missing) for at_well (true, false), occupancy_state (each value), "
            "current_patch (1, 2, 3) and is_rewarded_outcome (1, 0); each p_ "
            "is its count over n_events. " + _POPULATIONS,
            _SESSION_ROWS,
            _ANIMAL_SUMS,
        ),
        "rejection": (
            "How far the memoryless detector got on each compared state_space "
            "event, per cohort (memoryless: the published cohort; "
            "memoryless_criteria: the replay criteria without the shuffle) and "
            "membership against it. n_<stage> counts each event once, at the "
            "furthest stage any candidate overlapping it reached (in_cohort "
            "exactly when matched). The reason columns count every reason: the "
            "window counts and overlapping candidates summed "
            f"({', '.join(REJECTION_COUNTS)}), and the events with any "
            "overlapping candidate of each kind (n_any_*). unknown is its own "
            "row, with its reasons (n_no_shared_exposure, "
            "n_outside_shared_exposure) and no rejection reason (left "
            "missing).",
            "Per session (one analysed epoch, nested in its animal).",
            _ANIMAL_SUMS + " The unknown rows' reason columns stay missing.",
        ),
        "memoryless_cohort": (
            "The memoryless cohort before and after the shuffle filter. Scope "
            "candidates: every candidate of the memoryless run; scope "
            "compared: the compared memoryless_criteria events (no "
            "artifact drop, whole-event speed eligible, not set aside). "
            "n_criteria meet the replay criteria, n_published pass the shuffle "
            "as well, n_shuffle_failed meet the criteria and did not pass "
            "(unshuffled ones included), n_shuffle_untestable were shuffled "
            "but could not be tested. The adapted shuffle permutes encoding "
            "pairs; it does not reproduce the published sorted-cell null.",
            "Per session; available is False for a session with no memoryless "
            "run, whose counts are missing.",
            _ANIMAL_SUMS + " A session with no memoryless run is not summed; "
            "it is listed in unavailable_sessions.",
        ),
        "comparator_only": (
            "Each detector's events unmatched by state_space, by "
            "ss_disagreement: near (a compared state_space event within near_s, "
            "50 ms), subthreshold (a run of non-local probability above the "
            "threshold inside it), sustained (neither), missing (the runs "
            "unknown): n_events, and n_scorable of them with an evidence score. "
            "Then quantiles of their state-space composition ("
            + ", ".join(COMPOSITION)
            + "), their evidence_dll and their spikes_per_row. "
            + _EVIDENCE_NOTE
            + " "
            + QUANTILE_COLUMNS,
            "Per session (one analysed epoch, nested in its animal).",
            _ANIMAL_SUMS + " The quantiles pool the animal's events.",
        ),
    }
)

TABLE_CAPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "epochs": caption_with_pending(
            "Every epoch of the run: ok or failed (with the manifest's reason "
            "or error), and whether the memoryless run observed it. The "
            "analysed epochs the run does not hold are counted in report.md."
        ),
        **{
            f"{topic}_{level}": caption_with_pending(what, detail)
            for topic, (what, by_session, by_animal) in _TOPICS.items()
            for level, detail in (("by_session", by_session), ("by_animal", by_animal))
        },
    }
)
"""Every report table, in the order ``report.md`` lists it, with its caption."""

DISTRIBUTION_FEATURES = (
    ("interval_span_s", "log"),
    ("spike_rate_hz", "linear"),
    ("ripple_z_max", "linear"),
    ("mua_z_max", "linear"),
    ("evidence_dll", "linear"),
    ("ss_hpd_cm_mean", "linear"),
)
"""The features of ``feature_distributions.png``, with their axis scales."""
DECODER_PAIRS = (
    (STATE_SPACE, MEMORYLESS),
    (MEMORYLESS, STATE_SPACE),
)
"""The two decoders' pairs: (detector, B) of the populations drawn."""

FIGURE_CAPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "feature_distributions.png": caption_with_pending(
            "Feature distributions of the two decoders' pairwise populations, "
            "per animal (rows): state_space events matched and unmatched by "
            "memoryless†, and memoryless† events matched and unmatched by "
            "state_space, for each feature (columns): the median (dot; filled "
            "matched, open unmatched) and the 10th to 90th percentiles (bar) "
            "of the animal's events pooled. Colors by detector family. Counts "
            "in features_*_by_animal.csv."
        ),
        "rejection_stages.png": caption_with_pending(
            "Rejection stages of the state_space events unmatched by each "
            "memoryless cohort (panels), per animal (rows) and exclusive stage "
            "(columns, in the detector's order): the fraction of those events "
            "at the stage, printed in every cell (0 for none, <0.01 for a "
            "nonzero share below 0.005), each row summing to 1. n counts the "
            "unmatched events; unknown memberships are counted apart in the "
            "label and never drawn as a stage. Counts in "
            "rejection_by_animal.csv."
        ),
        "comparator_only.png": caption_with_pending(
            "Comparator-only events per animal (panels): each detector's "
            "events unmatched by state_space, by ss_disagreement class, as "
            "fractions of them, printed where the bar has room; n counts the "
            "events. Counts in comparator_only_by_animal.csv."
        ),
        "evidence_by_stratum.png": caption_with_pending(
            "Spike evidence (evidence_dll) by duration stratum, per detector "
            "(panels), every compared event: the median (marker, one per "
            "animal) and the 25th to 75th percentiles (bar) of the animal's "
            "events pooled. Colors by detector family. "
            + _EVIDENCE_NOTE
            + " Counts in features_evidence_by_stratum_by_animal.csv."
        ),
        "evidence_by_spikes.png": caption_with_pending(
            "Spike evidence (evidence_dll) against spikes per row, per detector "
            "(panels), every compared event: in each spikes_per_row bin, the "
            "median (marker, one per animal) and the 25th to 75th percentiles "
            "(bar) of the animal's events pooled. Colors by detector family. "
            + _EVIDENCE_NOTE
            + " Counts in evidence_by_spikes_by_animal.csv."
        ),
    }
)
"""Every figure, by file name, with its caption."""


def distribution(
    frame: pd.DataFrame, keys: Sequence[str], columns: Sequence[str], count: str
) -> pd.DataFrame:
    """Per ``keys`` group, the row count and each column's percentiles.

    The columns and values of the inventory report's ``_distribution`` (the
    count of non-missing values ``_n``, :data:`QUANTILES` by pandas' linear
    rule, and the maximum), read with grouped quantiles: it calls no Python
    function per group, so it is several hundred times faster on a run's
    many groups. A categorical key keeps its order and its observed values.
    """
    named = [count, *(f"{c}_{s}" for c in columns for s in ("n", *QUANTILES, "max"))]
    if frame.empty:
        return pd.DataFrame(columns=[*keys, *named])
    values = frame[list(columns)].astype(np.float64)
    grouped = values.groupby([frame[key] for key in keys], observed=True, sort=True)
    quantiles = grouped.quantile(list(QUANTILES.values()))
    present, top = grouped.count(), grouped.max()
    parts: dict[str, pd.Series] = {count: grouped.size()}
    for column in columns:
        parts[f"{column}_n"] = present[column]
        for label, q in QUANTILES.items():
            parts[f"{column}_{label}"] = quantiles[column].xs(q, level=-1)
        parts[f"{column}_max"] = top[column]
    return pd.DataFrame(parts).reset_index()


def epoch_status(manifest: Mapping) -> pd.DataFrame:
    """Every epoch of the run: its status, reason and memoryless run.

    ``pending`` is :data:`PENDING` where the memoryless run did not observe
    an ``ok`` epoch.
    """
    rows = []
    for identifier, record in sorted(manifest["epochs"].items()):
        observed = record.get("memoryless_available")
        rows.append(
            {
                "epoch_id": identifier,
                "status": record["status"],
                "reason": record.get("reason") or record.get("error", ""),
                "memoryless_available": "" if observed is None else bool(observed),
                "pending": PENDING if observed is False else "",
            }
        )
    return pd.DataFrame(
        rows,
        columns=["epoch_id", "status", "reason", "memoryless_available", "pending"],
    )


def _ok(manifest: Mapping) -> list[str]:
    return sorted(i for i, r in manifest["epochs"].items() if r["status"] == "ok")


def _session_of(identifier: str) -> tuple[str, int, str]:
    """``(animal, day, epoch)`` of ``{animal}{day}_{epoch}_HPC``."""
    key = parse_epoch_identifier(identifier)
    return key.animal, key.date, key.epoch_name


def read_inputs(
    manifest: Mapping, inventory_root: Path, memoryless_root: Path
) -> dict[str, pd.DataFrame]:
    """The comparison's ``cells`` and ``directional`` tables and the memoryless
    run's candidates, for the run's ``ok`` epochs, concatenated.

    The runs are the ones the manifest's ``input_runs`` name. Candidates are
    read for the epochs the memoryless run observed.
    """
    runs = manifest["settings"]["input_runs"]
    inventory_dir = inventory_root / runs["inventory"]
    memoryless_dir = memoryless_root / runs["memoryless"]
    ok = _ok(manifest)
    tables = {
        name: typed_table(
            (
                read_table(comparison.output_path(identifier, inventory_dir, name))
                for identifier in ok
            ),
            comparison.TABLE_DTYPES[name],
        )
        for name in ("cells", "directional")
    }
    tables["memoryless_events"] = typed_table(
        (
            read_table(
                memoryless_outputs(identifier, memoryless_dir)["events"],
                columns=list(CANDIDATE_COLUMNS),
            )
            for identifier in ok
            if manifest["epochs"][identifier].get("memoryless_available")
        ),
        CANDIDATE_COLUMNS,
    )
    return tables


def without_baseline_rows(features: pd.DataFrame) -> pd.DataFrame:
    """``features`` less the baseline windows (detector ``baseline``) that a
    run written before the baselines were dropped holds."""
    kept: pd.DataFrame = features[(features.detector != BASELINE).to_numpy()]
    return kept.reset_index(drop=True)


def event_populations(
    features: pd.DataFrame,
    pair_membership: pd.DataFrame,
    cells: pd.DataFrame,
    directional: pd.DataFrame,
) -> pd.DataFrame:
    """Each compared event once per population it belongs to.

    Returns
    -------
    DataFrame
        Columns and dtypes as ``populations.POPULATION_DTYPES``.
    """
    rows = population_rows(features, pair_membership, cells, directional)
    return rows[list(POPULATION_DTYPES)].astype(dict(POPULATION_DTYPES))


def context_by_session(
    populations: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Per session, detector and population, the events by each category of
    :data:`CONTEXT_CATEGORIES`, and the missing ones.

    Raises
    ------
    ValueError
        A value is not one of its feature's categories.
    """
    frame = _with(populations, features, list(CONTEXT_CATEGORIES))
    counts: dict[str, object] = {"n_events": 1}
    for feature, categories in CONTEXT_CATEGORIES.items():
        values = frame[feature]
        present = values.notna().to_numpy()
        if categories is None:
            categories = tuple(sorted(set(values[present])))
        _refuse_values(values[present], categories, feature)
        for category in categories:
            label = str(category).lower()
            counts[f"n_{feature}_{label}"] = present & (values == category).to_numpy(
                dtype=bool, na_value=False
            )
        counts[f"n_{feature}_missing"] = ~present
    counted = pd.DataFrame(counts, index=frame.index).astype(np.int64)
    return _grouped_sum(counted, frame[[*SESSION, *POPULATION]])


def _with_context_shares(frame: pd.DataFrame) -> pd.DataFrame:
    counts = [
        c
        for c in frame.columns
        if c.startswith("n_") and c not in ("n_events", "n_sessions")
    ]
    return frame.assign(
        **{"p_" + c.removeprefix("n_"): ratio(frame[c], frame.n_events) for c in counts}
    )


def _at_session(populations: pd.DataFrame) -> pd.DataFrame:
    """The populations a session table holds: pairwise only against
    :data:`SESSION_PAIRS`."""
    against = populations.population.str.split("_by:", n=1).str[1].fillna("")
    return populations[((against == "") | against.isin(SESSION_PAIRS)).to_numpy()]


def _with(
    populations: pd.DataFrame, features: pd.DataFrame, columns: Collection[str]
) -> pd.DataFrame:
    """``populations`` with each event's ``columns``."""
    return populations.join(
        features.set_index("event_key")[list(columns)], on="event_key"
    )


def _population_detectors(labels: Collection[str]) -> dict[str, tuple[str, ...]]:
    """Each population label -> the detectors its membership involves."""
    out: dict[str, tuple[str, ...]] = {}
    for label in labels:
        if "_by:" in label:
            out[label] = (label.partition("_by:")[2],)
        else:
            cell = label.partition(":")[0]
            out[label] = (
                tuple(comparison.CELLS[cell][1]) if cell in comparison.CELLS else ()
            )
    return out


def _mark(frame: pd.DataFrame, *, every_row: bool = False) -> pd.DataFrame:
    """``frame`` with ``pending``: on every row, or on the rows of a
    memoryless detector, a population tested against a memoryless detector,
    or an ``ms_*`` feature."""
    if every_row:
        return frame.assign(pending=PENDING)
    groups: dict[str, Mapping[str, Collection[str]]] = {}
    if "population" in frame:
        groups["population"] = _population_detectors(set(frame.population))
    if "feature" in frame:
        groups["feature"] = {
            f: PENDING_DETECTORS for f in set(frame.feature) if f.startswith("ms_")
        }
    columns = ["detector"] if "detector" in frame else []
    return with_pending(frame, columns, groups)


def feature_tables(
    populations: pd.DataFrame, features: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Every ``features_<group>[_by_stratum]_by_<level>`` table."""
    out = {}
    levels = (
        ("by_session", SESSION, _at_session(populations)),
        ("by_animal", ["animal"], populations),
    )
    for group, columns in FEATURE_GROUPS.items():
        for level, keys, rows in levels:
            frame = _with(rows, features, [*columns, "duration_stratum"])
            for by in ((), ("duration_stratum",)):
                table = distribution(
                    frame, [*keys, *POPULATION, *by], columns, "n_events"
                )
                if level == "by_session":
                    table = per_session(table)
                name = f"features_{group}{'_by_stratum' if by else ''}_{level}"
                out[name] = _mark(table, every_row=group == "ms")
    return out


def _grouped_sum(
    values: pd.DataFrame, by: pd.DataFrame, *, min_count: int = 0
) -> pd.DataFrame:
    """``values`` summed per distinct row of ``by``, ``by``'s columns first."""
    if by.empty:
        return pd.DataFrame(columns=[*by.columns, *values.columns])
    summed = values.groupby(
        [by[column].to_numpy() for column in by.columns], sort=True
    ).sum(min_count=min_count)
    summed.index = summed.index.set_names(list(by.columns))
    return summed.reset_index()


def _refuse_values(values: pd.Series, known: Collection[object], name: str) -> None:
    """Raise unless every value is one of ``known``."""
    strange = set(values) - set(known)
    if strange:
        raise ValueError(
            f"{name} values {sorted(map(str, strange))} are not {tuple(known)}."
        )


def missing_by_session(
    populations: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Per session, detector, population and feature: ``n`` events and
    ``n_missing`` of them with the feature missing; every population."""
    keys = [*SESSION, *POPULATION]
    frame = _with(populations, features, MISSING_FEATURES)
    missing = _grouped_sum(
        frame[list(MISSING_FEATURES)].isna().assign(n=1).astype(np.int64),
        frame[keys],
    )
    long = missing.melt(
        id_vars=[*keys, "n"],
        value_vars=list(MISSING_FEATURES),
        var_name="feature",
        value_name="n_missing",
    )
    return long[[*keys, "feature", "n", "n_missing"]].sort_values(
        keys, kind="stable", ignore_index=True
    )


def never_missing(missing: pd.DataFrame) -> list[str]:
    """The features no event of the run misses, in table order."""
    total = missing.groupby("feature", sort=False).n_missing.sum()
    return [f for f in MISSING_FEATURES if f in total.index and total[f] == 0]


EVIDENCE_COLUMNS = ("evidence_dll", "evidence_unscorable", "spikes_per_row")


def with_spikes_per_row(features: pd.DataFrame) -> pd.DataFrame:
    """``features`` with ``spikes_per_row`` (``n_spikes / n_rows``, NaN with
    no row) and its bin of :data:`SPIKES_PER_ROW_BINS`."""
    n_rows = features.n_rows.to_numpy(np.float64)
    n_spikes = features.n_spikes.astype("Float64").to_numpy(np.float64, na_value=np.nan)
    per_row = np.where(n_rows > 0, n_spikes / np.maximum(n_rows, 1), np.nan)
    return features.assign(
        spikes_per_row=per_row,
        spikes_per_row_bin=pd.cut(
            per_row,
            bins=list(SPIKES_PER_ROW_BINS),
            labels=list(SPIKES_PER_ROW_LABELS),
            right=False,
        ),
    )


def _comparator_counts(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Per ``keys`` group, the events and the scorable ones."""
    flags = pd.DataFrame(
        {
            "n_events": np.ones(len(frame), dtype=np.int64),
            "n_scorable": (frame.evidence_unscorable == "").to_numpy(),
        }
    ).astype(np.int64)
    return _grouped_sum(flags, frame[list(keys)])


def evidence_coverage_by_session(features: pd.DataFrame) -> pd.DataFrame:
    """Per session and detector: the events, the scorable ones, the unscorable ones by reason, and the summed rows, valid
    rows and spikes.

    Raises
    ------
    ValueError
        An unscorable reason is not one of :data:`UNSCORABLE_REASONS`.
    """
    reason = features.evidence_unscorable
    _refuse_values(reason, ("", *UNSCORABLE_REASONS), "evidence_unscorable")
    counted = pd.DataFrame(
        {
            "n_events": 1,
            "n_scorable": (reason == "").to_numpy(),
            **{
                f"n_unscorable_{r}": (reason == r).to_numpy()
                for r in UNSCORABLE_REASONS
            },
            "n_rows": features.n_rows.to_numpy(np.int64),
            "n_valid_rows": features.evidence_n_valid_rows.fillna(0).to_numpy(np.int64),
            "n_spikes": features.n_spikes.fillna(0).to_numpy(np.int64),
        },
        index=features.index,
    ).astype(np.int64)
    return _grouped_sum(counted, features[[*SESSION, "detector"]])


REJECTION_STAGE_COUNTS = tuple(f"n_{s}" for s in (*REJECTION_STAGES, "unknown"))
REJECTION_REASON_COUNTS = (*REJECTION_COUNTS, *(f"n_{f}" for f in REJECTION_FLAGS))
"""Every reason's count: the summed window and candidate counts, and the
events with each candidate flag."""
REJECTION_SUMS = (
    "n_events",
    *REJECTION_STAGE_COUNTS,
    *(f"n_{r}" for r in UNKNOWN_REASONS),
    *REJECTION_REASON_COUNTS,
)


def rejection_by_session(rejection: pd.DataFrame) -> pd.DataFrame:
    """Per session, cohort and membership: the events, each exclusive stage,
    each unknown reason and every rejection reason.

    The reason counts are missing on the ``unknown`` rows, which have none.

    Raises
    ------
    ValueError
        A stage, membership or unknown reason is not one the report knows.
    """
    stages = (*REJECTION_STAGES, "unknown")
    _refuse_values(rejection.stage, stages, "stage")
    _refuse_values(rejection.membership, MEMBERSHIPS, "membership")
    _refuse_values(rejection.unknown_reason, ("", *UNKNOWN_REASONS), "unknown_reason")
    keys = rejection[[*SESSION, "cohort", "membership"]]
    counts = pd.DataFrame(
        {
            "n_events": 1,
            **{f"n_{s}": (rejection.stage == s).to_numpy() for s in stages},
            **{
                f"n_{r}": (rejection.unknown_reason == r).to_numpy()
                for r in UNKNOWN_REASONS
            },
        },
        index=rejection.index,
    ).astype(np.int64)
    reasons = pd.concat(
        [
            rejection[list(REJECTION_COUNTS)],
            rejection[list(REJECTION_FLAGS)].add_prefix("n_"),
        ],
        axis=1,
    ).astype("Int64")
    counted = _grouped_sum(counts, keys)
    summed = _grouped_sum(reasons, keys, min_count=1)
    return counted.merge(summed, on=list(keys.columns), validate="one_to_one")


def rejection_by_animal(by_session: pd.DataFrame) -> pd.DataFrame:
    """The rejection counts summed per animal, cohort and membership; the
    reasons stay missing on the ``unknown`` rows."""
    pooled = summed(
        by_session.astype(dict.fromkeys(REJECTION_REASON_COUNTS, "Int64")),
        ["cohort", "membership"],
        REJECTION_SUMS,
    )
    unknown = (pooled.membership == "unknown").to_numpy()
    for column in REJECTION_REASON_COUNTS:
        pooled[column] = pooled[column].mask(unknown)
    return pooled


def memoryless_cohort_by_session(
    candidates: pd.DataFrame, features: pd.DataFrame, manifest: Mapping
) -> pd.DataFrame:
    """Per ok session and scope, the memoryless candidates by the cohort's
    criteria and the shuffle; missing where the memoryless run did not
    observe the session.

    Raises
    ------
    ValueError
        A compared memoryless_criteria event is not among the run's
        candidates, or the compared published events are not the
        memoryless events of the features table.
    """
    keyed = candidates.assign(
        event_key=_event_keys(
            comparison.MEMORYLESS_CRITERIA,
            candidates.animal,
            candidates.day,
            candidates.epoch,
            candidates.ms_event_id,
        )
    )
    criteria = set(
        features.event_key[(features.detector == comparison.MEMORYLESS_CRITERIA)]
    )
    absent = criteria - set(keyed.event_key)
    if absent:
        raise ValueError(
            f"{len(absent)} compared memoryless_criteria events, such as "
            f"{sorted(absent)[0]}, are not among the memoryless run's candidates."
        )
    untestable = keyed.is_replay & (keyed.n_shuffles > 0) & ~keyed.shuffle_testable
    flags = pd.DataFrame(
        {
            "n_candidates": 1,
            "n_criteria": keyed.is_replay,
            "n_published": keyed.is_replay & keyed.shuffle_pass,
            "n_shuffle_failed": keyed.is_replay & ~keyed.shuffle_pass & ~untestable,
            "n_shuffle_untestable": untestable,
        },
        index=keyed.index,
    ).astype(np.int64)
    compared = keyed.event_key.isin(criteria).to_numpy()
    keys = [*SESSION, "scope"]
    counts = pd.concat(
        [
            _grouped_sum(flags, keyed[SESSION].assign(scope="candidates")),
            _grouped_sum(
                flags[compared], keyed.loc[compared, SESSION].assign(scope="compared")
            ),
        ],
        ignore_index=True,
    ).astype(dict.fromkeys(COHORT_COUNTS, np.int64))
    skeleton = pd.DataFrame(
        [
            (
                *_session_of(identifier),
                scope,
                bool(manifest["epochs"][identifier].get("memoryless_available")),
            )
            for identifier in _ok(manifest)
            for scope in ("candidates", "compared")
        ],
        columns=[*keys, "available"],
    ).astype({"day": np.int64, "available": np.bool_})
    out = skeleton.merge(
        counts.astype({"day": np.int64}), on=keys, how="left", validate="one_to_one"
    )
    unavailable = ~out.available.to_numpy()
    for column in COHORT_COUNTS:
        out[column] = out[column].fillna(0).astype("Int64").mask(unavailable)

    rows = out[(out.scope == "compared").to_numpy() & ~unavailable].set_index(SESSION)
    published = (
        features[(features.detector == MEMORYLESS).to_numpy()]
        .groupby(SESSION)
        .size()
        .reindex(rows.index, fill_value=0)
    )
    differ = rows.n_published.to_numpy(np.int64) != published.to_numpy(np.int64)
    if differ.any():
        first = np.flatnonzero(differ)[0]
        raise ValueError(
            f"{rows.index[first]}: the compared memoryless_criteria events hold "
            f"{rows.n_published.iloc[first]} published events; the features "
            f"table holds {published.iloc[first]}."
        )
    return out


def comparator_only_events(
    populations: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """The events each detector has unmatched by state_space, with their
    ``ss_disagreement`` (``missing`` where unknown), evidence and
    composition.

    Raises
    ------
    ValueError
        A class is not one of :data:`DISAGREEMENTS`.
    """
    rows = populations[
        (populations.population == f"unmatched_by:{STATE_SPACE}").to_numpy()
    ]
    events = _with(rows, features, ["ss_disagreement", *EVIDENCE_COLUMNS, *COMPOSITION])
    events["ss_disagreement"] = events.ss_disagreement.fillna("missing")
    _refuse_values(events.ss_disagreement, DISAGREEMENTS, "ss_disagreement")
    return events


def comparator_only_table(
    counts: pd.DataFrame, events: pd.DataFrame, keys: Sequence[str]
) -> pd.DataFrame:
    """``counts`` with the quantiles of the composition, the evidence and the
    spikes per row of ``events`` pooled by ``keys``, detector and class."""
    by = [*keys, "detector", "ss_disagreement"]
    spread = distribution(
        events, by, [*COMPOSITION, "evidence_dll", "spikes_per_row"], "n_events"
    ).drop(columns="n_events")
    return counts.merge(spread, on=by, how="left")


def stratum_labels(features: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """``features`` with ``duration_stratum`` an ordered categorical of the
    run's ``labels``.

    Raises
    ------
    ValueError
        A stratum is not one of ``labels``.
    """
    unstratified = features.duration_stratum.isna()
    if unstratified.any():
        raise ValueError(
            f"{int(unstratified.sum())} rows, such as "
            f"{features.event_key[unstratified].iloc[0]}, have no duration_stratum; "
            "a compared event always has a positive span."
        )
    present = set(features.duration_stratum)
    if present - set(labels):
        raise ValueError(
            f"duration_stratum {sorted(present - set(labels))} is not among the "
            f"run's strata {list(labels)}."
        )
    return features.assign(
        duration_stratum=pd.Categorical(
            features.duration_stratum, categories=list(labels), ordered=True
        )
    )


def report_tables(
    manifest: Mapping, tables: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Every report table, by name, in the order of :data:`TABLE_CAPTIONS`.

    Parameters
    ----------
    manifest : mapping
        The run's manifest.
    tables : mapping
        ``features``, ``pair_membership`` and ``rejection`` of the run's ok
        epochs (``read_run``), and ``cells``, ``directional`` and
        ``memoryless_events`` (:func:`read_inputs`).

    Raises
    ------
    ValueError
        Two features rows share an event key, a value is not one the report
        knows, or the tables disagree (see the functions called).
    """
    features = without_baseline_rows(tables["features"])
    if not features.event_key.is_unique:
        raise ValueError("Two features rows share an event key.")
    features = stratum_labels(
        features, manifest["settings"]["duration_strata"]["labels"]
    )
    # an empty label is no state
    features["occupancy_state"] = features.occupancy_state.replace("", None)
    features = with_spikes_per_row(features)
    populations = event_populations(
        features, tables["pair_membership"], tables["cells"], tables["directional"]
    )
    out: dict[str, pd.DataFrame] = {"epochs": epoch_status(manifest)}
    out |= feature_tables(populations, features)

    missing = missing_by_session(populations, features)
    kept = ~missing.feature.isin(never_missing(missing)).to_numpy()
    out["missing_by_session"] = per_session(_at_session(missing[kept]))
    out["missing_by_animal"] = summed(
        missing, [*POPULATION, "feature"], ["n", "n_missing"]
    )
    context = context_by_session(populations, features)
    out["context_by_session"] = _with_context_shares(per_session(_at_session(context)))
    out["context_by_animal"] = _with_context_shares(
        summed(context, POPULATION, [c for c in context.columns if c.startswith("n_")])
    )
    coverage = evidence_coverage_by_session(features)
    out["evidence_coverage_by_session"] = per_session(coverage).assign(
        spikes_per_row=lambda t: ratio(t.n_spikes, t.n_rows)
    )
    out["evidence_coverage_by_animal"] = summed(
        coverage, ["detector"], [c for c in coverage.columns if c.startswith("n_")]
    ).assign(spikes_per_row=lambda t: ratio(t.n_spikes, t.n_rows))
    for level, keys, rows in (
        ("by_session", SESSION, _at_session(populations)),
        ("by_animal", ["animal"], populations),
    ):
        binned = _with(rows, features, ["evidence_dll", "spikes_per_row_bin"])
        table = distribution(
            # an interval with no row is in no bin
            binned[binned.spikes_per_row_bin.notna().to_numpy()],
            [*keys, *POPULATION, "spikes_per_row_bin"],
            ["evidence_dll"],
            "n_events",
        )
        out[f"evidence_by_spikes_{level}"] = (
            per_session(table) if level == "by_session" else table
        )

    rejection = rejection_by_session(tables["rejection"])
    out["rejection_by_session"] = per_session(rejection)
    out["rejection_by_animal"] = rejection_by_animal(rejection)
    cohort = memoryless_cohort_by_session(
        tables["memoryless_events"], features, manifest
    )
    out["memoryless_cohort_by_session"] = per_session(cohort)
    out["memoryless_cohort_by_animal"] = pool_sessions(
        cohort, by=["scope"], sums=COHORT_COUNTS, available=cohort.available
    )

    events = comparator_only_events(populations, features)
    keys = [*SESSION, "detector", "ss_disagreement"]
    counts = _comparator_counts(events, keys)
    out["comparator_only_by_session"] = per_session(
        comparator_only_table(counts, events, SESSION)
    )
    out["comparator_only_by_animal"] = comparator_only_table(
        summed(counts, ["detector", "ss_disagreement"], COMPARATOR_COUNTS),
        events,
        ["animal"],
    )

    every_row = {"rejection", "memoryless_cohort"}
    for name in list(out):
        if name == "epochs" or name.startswith("features_"):
            continue
        out[name] = _mark(out[name], every_row=name.rsplit("_by_", 1)[0] in every_row)
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


def _subplots(*args: Any, **kwargs: Any) -> tuple[plt.Figure, Any]:
    """``plt.subplots`` under constrained layout, its panels and legends kept
    above the footnote's 0.32 inch strip."""
    fig, axes = plt.subplots(*args, **kwargs)
    bottom = 0.32 / fig.get_figheight()
    fig.set_layout_engine("constrained", rect=(0, bottom, 1, 1 - bottom))
    return fig, axes


def _footnote(fig: plt.Figure, run_id: str) -> None:
    """The pending note and the run, in the strip below the panels."""
    fig.text(
        0.01,
        0.005,
        f"† {PENDING}\nevent features run {run_id}",
        fontsize=6,
        va="bottom",
    )


def _no_data(ax: plt.Axes, text: str = "no events") -> None:
    ax.text(
        0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=6
    )


def _detectors_of(frame: pd.DataFrame) -> list[str]:
    """The detectors of ``frame``, in the roster's order."""
    present = set(frame.detector)
    return [d for d in comparison.DETECTORS if d in present]


def _stacked(
    ax: plt.Axes,
    shares: pd.DataFrame,
    labels: Sequence[str],
    colors: Mapping[str, Any],
    hatches: Mapping[str, str] = MappingProxyType({}),
) -> None:
    """Horizontal stacked bars of ``shares`` (rows by columns, each row
    summing to 1), the share printed where its segment has room."""
    left = np.zeros(len(shares))
    for column in shares.columns:
        width = shares[column].to_numpy(np.float64)
        ax.barh(
            range(len(shares)),
            width,
            left=left,
            color=colors[column],
            edgecolor="0.2",
            linewidth=0.3,
            hatch=hatches.get(column),
            label=column,
        )
        for y, (x, w) in enumerate(zip(left, width, strict=True)):
            if w >= 0.09:
                ax.text(
                    x + w / 2,
                    y,
                    f"{w:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="w"
                    if np.mean(matplotlib.colors.to_rgb(colors[column])) < 0.6
                    else "k",
                )
        left = left + width
    ax.set_yticks(range(len(shares)), labels, fontsize=6)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.tick_params(labelsize=6)


def draw_figures(
    out: Mapping[str, pd.DataFrame], figure_dir: Path, run_id: str
) -> list[Path]:
    """Every figure of :data:`FIGURE_CAPTIONS`, written to ``figure_dir``."""
    paths = []

    def save(fig: plt.Figure, name: str) -> None:
        _footnote(fig, run_id)
        path = figure_dir / name
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    animals = sorted(out["features_timing_by_animal"].animal.unique())

    # feature distributions of the two decoders' pairs
    groups = {
        column: group for group, columns in FEATURE_GROUPS.items() for column in columns
    }
    populations = [
        (detector, f"{membership}_by:{other}")
        for detector, other in DECODER_PAIRS
        for membership in ("matched", "unmatched")
    ]
    n_rows = max(len(animals), 1)
    fig, axes = _subplots(
        n_rows,
        len(DISTRIBUTION_FEATURES),
        figsize=(7.2, 0.95 * n_rows + 0.9),
        sharex="col",
        squeeze=False,
    )
    for col, (feature, scale) in enumerate(DISTRIBUTION_FEATURES):
        table = out[f"features_{groups[feature]}_by_animal"]
        for row in range(n_rows):
            ax = axes[row, col]
            animal = animals[row] if animals else None
            counts = []
            for y, (detector, population) in enumerate(populations):
                match = table[
                    (table.animal == animal)
                    & (table.detector == detector)
                    & (table.population == population)
                ]
                counts.append(0 if match.empty else int(match.n_events.iloc[0]))
                if match.empty or not match[f"{feature}_n"].iloc[0]:
                    ax.text(
                        0.5,
                        y,
                        "no events" if match.empty else "no values",
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="0.4",
                        transform=ax.get_yaxis_transform(),
                    )
                    continue
                values = match.iloc[0]
                color = detector_color(detector)
                ax.hlines(
                    y,
                    values[f"{feature}_q10"],
                    values[f"{feature}_q90"],
                    color=color,
                    lw=1,
                )
                ax.plot(
                    values[f"{feature}_median"],
                    y,
                    "o",
                    ms=3,
                    color=color,
                    mfc=color if population.startswith("matched") else "w",
                )
            ax.set_xscale(scale)
            if scale == "log":
                ax.xaxis.set_major_locator(
                    matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0))
                )
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            else:
                # few ticks, so four-digit values (spike_rate_hz) do not run together
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.set_ylim(len(populations) - 0.5, -0.5)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_yticks(
                    range(len(populations)),
                    [
                        f"{_label(d)}\n{p.partition('_by:')[0]} by "
                        f"{_label(p.partition('_by:')[2])} (n = {n})"
                        for (d, p), n in zip(populations, counts, strict=True)
                    ],
                    fontsize=5,
                )
            else:
                ax.set_yticks(range(len(populations)), [""] * len(populations))
            if col == len(DISTRIBUTION_FEATURES) - 1 and animal is not None:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(animal, fontsize=7)
            if row == 0:
                ax.set_title(feature, fontsize=6)
    if not animals:
        _no_data(axes[0, 0])
    save(fig, "feature_distributions.png")

    # rejection stages of the ours-only events: one cell per animal and
    # stage, each labelled by its column and printed, so a thin stage shows
    rejection = out["rejection_by_animal"]
    stages = list(REJECTION_STAGES[:-1])
    cohorts = list(features_runner.REJECTION_COHORTS)
    greys = plt.get_cmap("Greys")
    fig, axes = _subplots(
        1,
        len(cohorts),
        figsize=(7.2, 0.3 * max(len(animals), 2) + 1.9),
        squeeze=False,
    )
    image = None
    for ax, cohort in zip(axes[0], cohorts, strict=True):
        rows = rejection[(rejection.cohort == cohort)].set_index(
            ["animal", "membership"]
        )
        counts = pd.DataFrame(
            [
                [
                    int(rows.loc[(a, "unmatched"), f"n_{s}"])
                    if (a, "unmatched") in rows.index
                    else 0
                    for s in stages
                ]
                for a in animals
            ],
            index=animals,
            columns=stages,
            dtype=np.float64,
        )
        totals = counts.sum(axis=1)
        unknown = [
            int(rows.loc[(a, "unknown"), "n_events"])
            if (a, "unknown") in rows.index
            else 0
            for a in animals
        ]
        shares = counts.div(totals.where(totals > 0), axis=0).to_numpy(np.float64)
        image = ax.imshow(
            shares.reshape(len(animals), len(stages)),
            vmin=0,
            vmax=1,
            cmap=greys,
            aspect="auto",
        )
        for (i, j), value in np.ndenumerate(shares):
            if not np.isfinite(value):
                continue
            text = "0" if value == 0 else "<0.01" if value < 0.005 else f"{value:.2f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=5,
                color="w" if value > 0.5 else "k",
            )
        ax.set_xticks(range(len(stages)), stages, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(
            range(len(animals)),
            [
                f"{a} (n = {int(t)}; unknown {u})"
                for a, t, u in zip(animals, totals, unknown, strict=True)
            ],
            fontsize=6,
        )
        ax.set_title(f"state_space unmatched by {_label(cohort)}", fontsize=7)
        if not animals:
            _no_data(ax)
    if image is not None:
        bar = fig.colorbar(image, ax=axes[0], shrink=0.8)
        bar.set_label("fraction (of the animal's unmatched events)", fontsize=6)
        bar.ax.tick_params(labelsize=6)
    save(fig, "rejection_stages.png")

    # comparator-only classes
    comparator = out["comparator_only_by_animal"]
    # neutral greys, no family hue; missing white and hatched
    class_colors = dict(zip(DISAGREEMENTS, ["0.3", "0.6", "0.85", "w"], strict=True))
    class_hatches = {"missing": "////"}
    detectors = [d for d in _detectors_of(comparator) if d != STATE_SPACE]
    fig, axes = _subplots(
        n_rows,
        1,
        figsize=(7.2, (0.22 * max(len(detectors), 2) + 0.45) * n_rows + 0.8),
        squeeze=False,
    )
    for ax, animal in zip(axes[:, 0], animals or [None], strict=True):
        rows = comparator[comparator.animal == animal]
        counts = (
            rows.pivot(index="detector", columns="ss_disagreement", values="n_events")
            .reindex(index=detectors, columns=list(DISAGREEMENTS))
            .fillna(0)
        )
        totals = counts.sum(axis=1)
        _stacked(
            ax,
            counts.div(totals.where(totals > 0), axis=0).fillna(0),
            [
                f"{_label(d)} (n = {int(t)})"
                for d, t in zip(detectors, totals, strict=True)
            ],
            class_colors,
            class_hatches,
        )
        ax.set_title(animal or "", fontsize=7)
        if not detectors:
            _no_data(ax)
    axes[-1, 0].set_xlabel(
        "fraction of the detector's events unmatched by state_space", fontsize=6
    )
    fig.legend(
        handles=[
            matplotlib.patches.Patch(
                facecolor=class_colors[c],
                edgecolor="0.2",
                linewidth=0.3,
                hatch=class_hatches.get(c),
                label=c,
            )
            for c in DISAGREEMENTS
        ],
        loc="outside right center",
        ncol=1,
        fontsize=6,
        frameon=False,
    )
    save(fig, "comparator_only.png")

    # evidence by duration stratum, and against spikes per row
    for table, category, xlabel, name in (
        (
            out["features_evidence_by_stratum_by_animal"],
            "duration_stratum",
            "duration stratum",
            "evidence_by_stratum.png",
        ),
        (
            out["evidence_by_spikes_by_animal"],
            "spikes_per_row_bin",
            "spikes per row",
            "evidence_by_spikes.png",
        ),
    ):
        save(_evidence_by(table, category, xlabel, animals), name)
    return paths


def _evidence_by(
    table: pd.DataFrame, category: str, xlabel: str, animals: Sequence[str]
) -> plt.Figure:
    """One panel per detector of ``evidence_dll`` by the bins of ``category``:
    each animal's median (marker) and 25th to 75th percentiles (bar), from
    ``table``'s ``all`` rows."""
    rows_of_all = table[(table.population == ALL).to_numpy()]
    detectors = _detectors_of(rows_of_all)
    labels = list(rows_of_all[category].cat.categories) if len(rows_of_all) else []
    n_cols = 4
    n_panel_rows = max(-(-len(detectors) // n_cols), 1)
    fig, axes = _subplots(
        n_panel_rows,
        n_cols,
        figsize=(7.2, 1.55 * n_panel_rows + 0.9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    markers = dict(zip(animals, "osD^vP", strict=False))
    offsets = dict(
        zip(animals, np.linspace(-0.25, 0.25, max(len(animals), 1)), strict=False)
    )
    for ax, detector in zip(axes.flat, detectors, strict=False):
        rows = rows_of_all[rows_of_all.detector == detector]
        color = detector_color(detector)
        for animal, values in rows.groupby("animal"):
            x = values[category].cat.codes.to_numpy() + offsets[animal]
            ax.vlines(
                x, values.evidence_dll_q25, values.evidence_dll_q75, color=color, lw=0.8
            )
            ax.plot(
                x,
                values.evidence_dll_median,
                markers[animal],
                ms=2.5,
                color=color,
                linestyle="none",
                label=animal,
            )
        ax.set_title(_label(detector), fontsize=7)
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(detectors) :]:
        ax.axis("off")
    for index, ax in enumerate(axes.flat[: len(detectors)]):
        # the lowest panel of each column carries the bins
        if index + n_cols >= len(detectors):
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xticks(
                range(len(labels)), labels, rotation=45, ha="right", fontsize=6
            )
    if not detectors:
        _no_data(axes[0, 0])
    fig.supylabel("evidence_dll", fontsize=7)
    fig.supxlabel(xlabel, fontsize=7)
    fig.legend(
        handles=[
            plt.Line2D([], [], marker=markers[a], color="0.3", ls="none", ms=3, label=a)
            for a in animals
        ],
        loc="outside right center",
        ncol=1,
        fontsize=6,
        frameon=False,
    )
    return fig


def write_report(
    run_dir: Path,
    figure_root: Path,
    *,
    inventory_root: Path | None = None,
    memoryless_root: Path | None = None,
    analysed: Collection[str] | None = None,
) -> dict[str, list[Path]]:
    """Write the run's report tables, figures and captions.

    Tables go to ``run_dir/report/``; figures and ``captions.md`` to
    ``figure_root/<run id>/``, so runs do not overwrite each other.

    Parameters
    ----------
    run_dir : Path
        The event features run's folder.
    figure_root : Path
    inventory_root, memoryless_root : Path, optional
        Where the comparison and memoryless runs are;
        :data:`INVENTORY_ROOT` and :data:`MEMORYLESS_ROOT` by default.
    analysed : collection of str, optional
        The analysed epochs; the epoch register's by default.

    Returns
    -------
    dict
        ``tables`` and ``figures``: the paths written.
    """
    inventory_root = INVENTORY_ROOT if inventory_root is None else inventory_root
    memoryless_root = MEMORYLESS_ROOT if memoryless_root is None else memoryless_root
    analysed = analysed_epoch_ids() if analysed is None else analysed
    manifest, tables = read_run(
        run_dir, features_runner.TABLE_DTYPES, features_runner.output_path
    )
    tables |= read_inputs(manifest, inventory_root, memoryless_root)
    out = report_tables(manifest, tables)
    run_id = run_dir.name
    figure_dir = figure_root / run_id
    report_dir = run_dir / REPORT_DIR
    report_dir.mkdir(exist_ok=True)
    # a table or figure an earlier version of the report wrote, and this one
    # does not, must not stand beside it
    for stale in report_dir.glob("*.csv"):
        if stale.stem not in out:
            stale.unlink()
    table_paths = []
    for name, frame in out.items():
        path = report_dir / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.10g")
        table_paths.append(path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for stale in figure_dir.glob("*.png"):
        if stale.name not in FIGURE_CAPTIONS:
            stale.unlink()
    figure_paths = draw_figures(out, figure_dir, run_id)

    settings = manifest["settings"]
    inventory_run = settings["input_runs"]["inventory"]
    boundary_dir = inventory_root / inventory_run / "report"
    status = out["epochs"].status.value_counts()
    outside = len(set(analysed) - set(manifest["epochs"]))
    lines = [
        f"# Event features report: run {run_id}",
        "",
        f"Written by `{PRODUCED_BY}` from the per-epoch tables of `{run_dir}` "
        f"({status.get('ok', 0)} ok epochs, {status.get('failed', 0)} failed, "
        f"and {outside} analysed epochs of the epoch register outside the run), "
        f"the cells and directional tables of inventory comparison run "
        f"{inventory_run} and the candidates of memoryless run "
        f"{settings['input_runs']['memoryless']}, read through SWR source "
        f"{settings['swr_source']}. It states counts and makes no claim.",
        "",
        PENDING_SENTENCE,
        "",
        "## What is unknown",
        "",
        f"- **Tracking confidence**: {settings['tracking_confidence']}.",
        "- **The Widloski columns** are pending the Phase 2 pilot checks: the "
        "memoryless and memoryless_criteria inventories (the Widloski-style "
        "detector), the populations tested against them, and every ms_* "
        "feature, which reads that detector's window posteriors. No claim is "
        "made from them.",
        "- **No baseline**: the lead dropped the baseline windows (2026-10-18), "
        "pending a design matched on spike count, so the evidence score is read "
        "beside the spikes per row and against nothing.",
        "- **Unknown memberships** are their own populations (unknown_by:<B>, "
        "<cell>:unknown) with their reasons, never counted as unmatched.",
        "",
        "## Boundaries",
        "",
        "Signed onset and offset differences, agreement under stricter match "
        "rules, one-to-many matches and the fragment-merge recount are not "
        f"recomputed here; inventory comparison run {inventory_run} reports them:",
        "",
        *(
            f"- `{boundary_dir / f'{name}_{level}.csv'}`"
            + (
                ""
                if (boundary_dir / f"{name}_{level}.csv").exists()
                else " (not found)"
            )
            for name in BOUNDARY_TABLES
            for level in ("by_session", "by_animal")
        ),
        "",
        "## Tables",
        "",
    ]
    dropped = never_missing(out["missing_by_animal"])
    for name, caption in TABLE_CAPTIONS.items():
        if name == "missing_by_session":
            caption += (
                " Left out here, missing in no session: "
                f"{', '.join(dropped) or 'none'}."
            )
        lines += [f"### `{name}.csv` ({len(out[name])} rows)", "", caption, ""]
    lines += ["## Figures", ""]
    for name, caption in FIGURE_CAPTIONS.items():
        lines += [f"### `{figure_dir / name}`", "", caption, ""]
    (report_dir / "report.md").write_text("\n".join(lines))
    captions = [f"# Event features figures: run {run_id}", ""]
    for name, caption in FIGURE_CAPTIONS.items():
        captions += [f"## `{name}`", "", caption, ""]
    (figure_dir / "captions.md").write_text("\n".join(captions))
    return {"tables": table_paths, "figures": figure_paths}


def main(argv: list[str] | None = None) -> None:
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
    run_dir = OUTPUT_ROOT / args.run_id
    if not (run_dir / "manifest.json").exists():
        sys.exit(f"No event features run manifest at {run_dir}.")
    written = write_report(
        run_dir,
        FIGURE_ROOT,
        inventory_root=INVENTORY_ROOT,
        memoryless_root=MEMORYLESS_ROOT,
        analysed=analysed_epoch_ids(),
    )
    print(
        f"{len(written['tables'])} tables -> {run_dir / REPORT_DIR}; "
        f"{len(written['figures'])} figures -> {FIGURE_ROOT / args.run_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
