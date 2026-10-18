#!/usr/bin/env python
"""Report one event features run: its features per session and per animal.

Reads the per-epoch tables ``scripts/qa/run_event_features.py`` wrote for the
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
``unknown`` membership is its own population, never pooled as unmatched. The
matched baseline windows are the population ``baseline`` of detector
``baseline`` and enter no detector's population.

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
    python scripts/qa/report_event_features.py --run-id <event features run id>
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
)
from continuum_swr_replay.analysis.inventory_comparison.selection import (  # noqa: E402
    DURATION_ONLY,
    FAILED,
    LOCATION,
    REJECTION_DTYPES,
    REJECTION_STAGES,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analysed_epoch_ids,
)
from continuum_swr_replay.parameters import FIGURE_DIR  # noqa: E402
from continuum_swr_replay.table_io import read_table  # noqa: E402
from scripts.decoding.run_memoryless_detection import (  # noqa: E402
    output_paths as memoryless_outputs,
)
from scripts.figures.generate_inventory_examples import (  # noqa: E402
    detector_color,
)
from scripts.qa import run_event_features as features_runner  # noqa: E402
from scripts.qa import run_inventory_comparison as comparison  # noqa: E402
from scripts.qa.report_inventory_comparison import (  # noqa: E402
    _QUANTILE_COLUMNS,
    MEMBERSHIPS,
    PENDING,
    PENDING_DETECTORS,
    PENDING_SENTENCE,
    QUANTILES,
    _caption,
    _per_session,
    _ratio,
    _summed,
    _typed,
    pool_sessions,
    read_run,
    with_pending,
)

PRODUCED_BY = "scripts/qa/report_event_features.py"
OUTPUT_ROOT = features_runner.OUTPUT_ROOT
"""Where the event features runs are."""
INVENTORY_ROOT = features_runner.INVENTORY_ROOT
MEMORYLESS_ROOT = features_runner.MEMORYLESS_ROOT
FIGURE_ROOT = FIGURE_DIR / "event_features"
REPORT_DIR = "report"
"""The report's folder inside the run's."""
SESSION = ["animal", "day", "epoch"]
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
            "evidence_rank",
            "baseline_median_dll",
            "baseline_q95_dll",
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
BASELINE_LEVELS = (LOCATION, DURATION_ONLY, FAILED)
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
EVIDENCE_COUNTS = (
    "n_events",
    "n_scorable",
    "n_with_baseline",
    "n_above_q95",
    "n_at_or_below_median",
)

_POPULATIONS = (
    "Populations: all (every compared event of the detector); matched_by:<B> "
    "and unmatched_by:<B> (its membership against detector B); unknown_by:<B>, "
    "counted apart and never as unmatched, with unknown_reason "
    "(no_shared_exposure: the pair shares no exposure in the epoch; "
    "outside_shared_exposure: the event is not wholly inside it); each cell's "
    "unmatched events (exactly_one_detector, swr_matched_by_no_decoder, "
    "sequence_matched_by_no_trigger) with <cell>:matched and <cell>:unknown "
    "beside it; and baseline, the matched baseline windows (detector "
    "baseline), which enter no detector's population. The baseline pool is the "
    "shared exposure of all nine inventories less every event of each."
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
_AGAINST_BASELINE = (
    "n_scorable have an evidence score; n_with_baseline are scorable and have "
    "a baseline median and 95th percentile; n_above_q95 score above their own "
    "baseline_q95_dll and n_at_or_below_median at or below their own "
    "baseline_median_dll. These are descriptive labels: an excess over "
    "baseline is not a calibrated declaration of replay."
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
    return " ".join((what, _POPULATIONS, _QUANTILE_COLUMNS))


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
            "events, the stop context of an event in no stop, and a baseline "
            "window's native duration, gaps and baseline columns. Labels "
            f"({', '.join(LABELLED_FEATURES)}) are counted here only. " + _POPULATIONS,
            _SESSION_ROWS,
            _ANIMAL_SUMS,
        ),
        "evidence_coverage": (
            "The spike evidence's coverage per detector, baseline windows "
            "included: n_events; n_scorable (an evidence score); "
            "n_unscorable_no_row (no grid row) and n_unscorable_no_valid_row "
            "(rows, none valid); and the rows (n_rows), valid rows "
            "(n_valid_rows) and spikes (n_spikes) summed.",
            "Per session (one analysed epoch, nested in its animal).",
            _ANIMAL_SUMS,
        ),
        "baseline_matching": (
            "Each detector's compared events by the level of their matched "
            "baselines: n_location (matched in duration and in the animal's "
            "location), n_duration_only (in duration only) and n_failed (no "
            "baseline). The baseline pool involves every inventory.",
            "Per session (one analysed epoch, nested in its animal).",
            _ANIMAL_SUMS,
        ),
        "evidence_vs_baseline": (
            "Each event's spike evidence against its own matched baselines, "
            "counted per detector and population: n_events; "
            + _AGAINST_BASELINE
            + " p_above_q95 and p_at_or_below_median are those counts over "
            "n_with_baseline. " + _POPULATIONS,
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
            "unknown). Counts against their baselines: "
            + _AGAINST_BASELINE
            + " Then quantiles of their state-space composition ("
            + ", ".join(COMPOSITION)
            + "). "
            + _QUANTILE_COLUMNS,
            "Per session (one analysed epoch, nested in its animal).",
            _ANIMAL_SUMS + " The quantiles pool the animal's events.",
        ),
    }
)

TABLE_CAPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "epochs": _caption(
            "Every epoch of the run: ok or failed (with the manifest's reason "
            "or error), and whether the memoryless run observed it. The "
            "analysed epochs the run does not hold are counted in report.md."
        ),
        **{
            f"{topic}_{level}": _caption(what, detail)
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
        "feature_distributions.png": _caption(
            "Feature distributions of the two decoders' pairwise populations, "
            "per animal (rows): state_space events matched and unmatched by "
            "memoryless†, and memoryless† events matched and unmatched by "
            "state_space, for each feature (columns): the median (dot; filled "
            "matched, open unmatched) and the 10th to 90th percentiles (bar) "
            "of the animal's events pooled. Colors by detector family. Counts "
            "in features_*_by_animal.csv."
        ),
        "evidence_vs_baseline.png": _caption(
            "Spike evidence against each event's own matched baselines, per "
            "animal (columns) and detector (rows), every compared event: the "
            "fraction above its baseline 95th percentile (left) and at or "
            "below its baseline median (right), of the events scorable with a "
            "baseline, from counts summed over the animal's sessions; printed "
            "in each cell to two places. Grey: no such events. Descriptive "
            "labels, not a calibrated declaration of replay. Counts in "
            "evidence_vs_baseline_by_animal.csv."
        ),
        "rejection_stages.png": _caption(
            "Rejection stages of the state_space events unmatched by each "
            "memoryless cohort, per animal: the fraction of those events at "
            "each exclusive stage, printed where the bar has room. n counts "
            "the unmatched events; unknown memberships are counted apart in "
            "the label and never drawn as a stage. Counts in "
            "rejection_by_animal.csv."
        ),
        "comparator_only.png": _caption(
            "Comparator-only events per animal (panels): each detector's "
            "events unmatched by state_space, by ss_disagreement class, as "
            "fractions of them, printed where the bar has room; n counts the "
            "events. Counts in comparator_only_by_animal.csv."
        ),
        "evidence_by_stratum.png": _caption(
            "Spike evidence (evidence_dll) by duration stratum, per detector "
            "(panels), every compared event, and the baseline windows: the "
            "median (marker, one per animal) and the 25th to 75th percentiles "
            "(bar) of the animal's events pooled. Colors by detector family. "
            "Counts in features_evidence_by_stratum_by_animal.csv."
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
    stem, number, replicate = identifier.split("_")[:3]
    return stem[:-8], int(stem[-8:]), f"{number}_{replicate}"


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
        name: _typed(
            (
                read_table(comparison.output_path(identifier, inventory_dir, name))
                for identifier in ok
            ),
            comparison.TABLE_DTYPES[name],
        )
        for name in ("cells", "directional")
    }
    tables["memoryless_events"] = _typed(
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


def event_populations(
    features: pd.DataFrame,
    pair_membership: pd.DataFrame,
    cells: pd.DataFrame,
    directional: pd.DataFrame,
) -> pd.DataFrame:
    """Each compared event once per population it belongs to, and each
    baseline window once, as population ``baseline``.

    Returns
    -------
    DataFrame
        Columns and dtypes as ``populations.POPULATION_DTYPES``.
    """
    windows = (features.detector == BASELINE).to_numpy()
    rows = population_rows(features[~windows], pair_membership, cells, directional)
    baselines = features.loc[windows, [*SESSION, "detector", "event_key"]].assign(
        population=BASELINE, unknown_reason=""
    )
    together = pd.concat([rows, baselines], ignore_index=True)
    return together[list(POPULATION_DTYPES)].astype(dict(POPULATION_DTYPES))


def _at_session(populations: pd.DataFrame) -> pd.DataFrame:
    """The populations a session table holds: pairwise only against
    :data:`SESSION_PAIRS`."""
    against = populations.population.str.partition("_by:")[2]
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
                    table = _per_session(table)
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


def _refuse_values(values: pd.Series, known: Collection[str], name: str) -> None:
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


EVIDENCE_COLUMNS = (
    "evidence_dll",
    "evidence_unscorable",
    "baseline_median_dll",
    "baseline_q95_dll",
)


def _against_baseline(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Per ``keys`` group, :data:`EVIDENCE_COUNTS` of the events against their
    own baselines."""
    dll = frame.evidence_dll.to_numpy(np.float64)
    scorable = (frame.evidence_unscorable == "").to_numpy()
    median = frame.baseline_median_dll.to_numpy(np.float64)
    q95 = frame.baseline_q95_dll.to_numpy(np.float64)
    with_baseline = scorable & np.isfinite(median) & np.isfinite(q95)
    flags = pd.DataFrame(
        {
            "n_events": np.ones(len(frame), dtype=np.int64),
            "n_scorable": scorable,
            "n_with_baseline": with_baseline,
            "n_above_q95": with_baseline & (dll > q95),
            "n_at_or_below_median": with_baseline & (dll <= median),
        }
    ).astype(np.int64)
    return _grouped_sum(flags, frame[list(keys)])


def _with_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with the two shares over ``n_with_baseline``."""
    return frame.assign(
        p_above_q95=_ratio(frame.n_above_q95, frame.n_with_baseline),
        p_at_or_below_median=_ratio(frame.n_at_or_below_median, frame.n_with_baseline),
    )


def evidence_vs_baseline_by_session(
    populations: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Per session, detector and population, the events against their
    baselines; every population but ``baseline``."""
    rows = populations[(populations.population != BASELINE).to_numpy()]
    return _against_baseline(
        _with(rows, features, EVIDENCE_COLUMNS), [*SESSION, *POPULATION]
    )


def evidence_coverage_by_session(features: pd.DataFrame) -> pd.DataFrame:
    """Per session and detector, baseline windows included: the events, the
    scorable ones, the unscorable ones by reason, and the summed rows, valid
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


def baseline_matching_by_session(features: pd.DataFrame) -> pd.DataFrame:
    """Per session and detector, its compared events by baseline level.

    Raises
    ------
    ValueError
        A compared event's level is not one of :data:`BASELINE_LEVELS`.
    """
    events = features[(features.detector != BASELINE).to_numpy()]
    _refuse_values(events.baseline_level, BASELINE_LEVELS, "baseline_level")
    counted = pd.DataFrame(
        {
            "n_events": 1,
            **{
                f"n_{level}": (events.baseline_level == level).to_numpy()
                for level in BASELINE_LEVELS
            },
        },
        index=events.index,
    ).astype(np.int64)
    return _grouped_sum(counted, events[[*SESSION, "detector"]])


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
    pooled = _summed(
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
    """``counts`` with their shares and the composition's quantiles of
    ``events`` pooled by ``keys``, detector and class."""
    by = [*keys, "detector", "ss_disagreement"]
    spread = distribution(events, by, COMPOSITION, "n_events").drop(columns="n_events")
    return _with_shares(counts).merge(spread, on=by, how="left")


def stratum_labels(features: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """``features`` with ``duration_stratum`` an ordered categorical of the
    run's ``labels``.

    Raises
    ------
    ValueError
        A stratum is not one of ``labels``.
    """
    present = set(features.duration_stratum.dropna())
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
    features = tables["features"]
    if not features.event_key.is_unique:
        raise ValueError("Two features rows share an event key.")
    features = stratum_labels(
        features, manifest["settings"]["duration_strata"]["labels"]
    )
    populations = event_populations(
        features, tables["pair_membership"], tables["cells"], tables["directional"]
    )
    out: dict[str, pd.DataFrame] = {"epochs": epoch_status(manifest)}
    out |= feature_tables(populations, features)

    missing = missing_by_session(populations, features)
    out["missing_by_session"] = _per_session(_at_session(missing))
    out["missing_by_animal"] = _summed(
        missing, [*POPULATION, "feature"], ["n", "n_missing"]
    )
    for topic, counted in (
        ("evidence_coverage", evidence_coverage_by_session(features)),
        ("baseline_matching", baseline_matching_by_session(features)),
    ):
        out[f"{topic}_by_session"] = _per_session(counted)
        out[f"{topic}_by_animal"] = _summed(
            counted, ["detector"], [c for c in counted.columns if c.startswith("n_")]
        )
    against = evidence_vs_baseline_by_session(populations, features)
    out["evidence_vs_baseline_by_session"] = _with_shares(
        _per_session(_at_session(against))
    )
    out["evidence_vs_baseline_by_animal"] = _with_shares(
        _summed(against, POPULATION, EVIDENCE_COUNTS)
    )

    rejection = rejection_by_session(tables["rejection"])
    out["rejection_by_session"] = _per_session(rejection)
    out["rejection_by_animal"] = rejection_by_animal(rejection)
    cohort = memoryless_cohort_by_session(
        tables["memoryless_events"], features, manifest
    )
    out["memoryless_cohort_by_session"] = _per_session(cohort)
    out["memoryless_cohort_by_animal"] = pool_sessions(
        cohort, by=["scope"], sums=COHORT_COUNTS, available=cohort.available
    )

    events = comparator_only_events(populations, features)
    keys = [*SESSION, "detector", "ss_disagreement"]
    counts = _against_baseline(events, keys)
    out["comparator_only_by_session"] = _per_session(
        comparator_only_table(counts, events, SESSION)
    )
    out["comparator_only_by_animal"] = comparator_only_table(
        _summed(counts, ["detector", "ss_disagreement"], EVIDENCE_COUNTS),
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
            edgecolor="w",
            linewidth=0.5,
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
            for y, (detector, population) in enumerate(populations):
                match = table[
                    (table.animal == animal)
                    & (table.detector == detector)
                    & (table.population == population)
                ]
                if match.empty or not match[f"{feature}_n"].iloc[0]:
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
            ax.set_ylim(len(populations) - 0.5, -0.5)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_yticks(
                    range(len(populations)),
                    [
                        f"{_label(d)}\n{p.partition('_by:')[0]} by "
                        f"{_label(p.partition('_by:')[2])}"
                        for d, p in populations
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

    # evidence against baseline
    against = out["evidence_vs_baseline_by_animal"]
    against = against[(against.population == ALL).to_numpy()]
    detectors = _detectors_of(against)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    fig, axes = _subplots(
        1,
        2,
        figsize=(7.2, 0.28 * max(len(detectors), 3) + 1.4),
        squeeze=False,
    )
    image = None
    for ax, (column, title) in zip(
        axes[0],
        (
            ("p_above_q95", "above its baseline 95th percentile"),
            ("p_at_or_below_median", "at or below its baseline median"),
        ),
        strict=True,
    ):
        grid = (
            against.pivot(index="detector", columns="animal", values=column)
            .reindex(index=detectors, columns=animals)
            .to_numpy(np.float64)
        )
        image = ax.imshow(grid, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for (i, j), value in np.ndenumerate(grid):
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="k" if value > 0.6 else "w",
                )
        ax.set_xticks(range(len(animals)), animals, fontsize=6)
        ax.set_yticks(range(len(detectors)), [_label(d) for d in detectors], fontsize=6)
        ax.set_title(f"fraction {title}", fontsize=7)
        if not detectors:
            _no_data(ax)
    if image is not None:
        bar = fig.colorbar(image, ax=axes[0], shrink=0.8)
        bar.set_label("fraction of events scorable with a baseline", fontsize=6)
        bar.ax.tick_params(labelsize=6)
    save(fig, "evidence_vs_baseline.png")

    # rejection stages of the ours-only events
    rejection = out["rejection_by_animal"]
    stages = list(REJECTION_STAGES[:-1])
    stage_colors = {s: plt.get_cmap("tab10")(i) for i, s in enumerate(stages)}
    cohorts = list(features_runner.REJECTION_COHORTS)
    fig, axes = _subplots(
        1,
        len(cohorts),
        figsize=(7.2, 0.32 * max(len(animals), 2) + 1.6),
        squeeze=False,
    )
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
        _stacked(
            ax,
            counts.div(totals.where(totals > 0), axis=0).fillna(0),
            [
                f"{a} (n = {int(t)}; unknown {u})"
                for a, t, u in zip(animals, totals, unknown, strict=True)
            ],
            stage_colors,
        )
        ax.set_title(f"state_space unmatched by {_label(cohort)}", fontsize=7)
        ax.set_xlabel("fraction of unmatched events", fontsize=6)
        if not animals:
            _no_data(ax)
    fig.legend(
        handles=[
            matplotlib.patches.Patch(color=stage_colors[s], label=s) for s in stages
        ],
        loc="outside right center",
        ncol=1,
        fontsize=6,
        frameon=False,
    )
    save(fig, "rejection_stages.png")

    # comparator-only classes
    comparator = out["comparator_only_by_animal"]
    class_colors = dict(
        zip(DISAGREEMENTS, ["#0072B2", "#E69F00", "#CC79A7", "0.75"], strict=True)
    )
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
        )
        ax.set_title(animal or "", fontsize=7)
        if not detectors:
            _no_data(ax)
    axes[-1, 0].set_xlabel(
        "fraction of the detector's events unmatched by state_space", fontsize=6
    )
    fig.legend(
        handles=[
            matplotlib.patches.Patch(color=class_colors[c], label=c)
            for c in DISAGREEMENTS
        ],
        loc="outside right center",
        ncol=1,
        fontsize=6,
        frameon=False,
    )
    save(fig, "comparator_only.png")

    # evidence by duration stratum
    strata = out["features_evidence_by_stratum_by_animal"]
    strata = strata[strata.population.isin([ALL, BASELINE]).to_numpy()]
    detectors = [
        *_detectors_of(strata),
        *([BASELINE] if (strata.detector == BASELINE).any() else []),
    ]
    labels = list(strata.duration_stratum.cat.categories) if len(strata) else []
    n_cols = 4
    n_panel_rows = max(-(-len(detectors) // n_cols), 1)
    fig, axes = _subplots(
        n_panel_rows,
        n_cols,
        figsize=(7.2, 1.55 * n_panel_rows + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    markers = dict(zip(animals, "osD^vP", strict=False))
    offsets = dict(
        zip(animals, np.linspace(-0.25, 0.25, max(len(animals), 1)), strict=False)
    )
    for ax, detector in zip(axes.flat, detectors, strict=False):
        rows = strata[strata.detector == detector]
        color = "0.4" if detector == BASELINE else detector_color(detector)
        for animal, values in rows.groupby("animal"):
            x = values.duration_stratum.cat.codes.to_numpy() + offsets[animal]
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
        # the lowest panel of each column carries the strata
        if index + n_cols >= len(detectors):
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xticks(
                range(len(labels)), labels, rotation=45, ha="right", fontsize=6
            )
    if not detectors:
        _no_data(axes[0, 0])
    fig.supylabel("evidence_dll", fontsize=7)
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
    save(fig, "evidence_by_stratum.png")
    return paths


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
    table_paths = []
    for name, frame in out.items():
        path = report_dir / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.10g")
        table_paths.append(path)
    figure_dir.mkdir(parents=True, exist_ok=True)
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
        f"{settings['input_runs']['memoryless']}. It states counts and makes no "
        "claim.",
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
    for name, caption in TABLE_CAPTIONS.items():
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
