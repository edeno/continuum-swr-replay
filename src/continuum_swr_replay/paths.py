"""Output locations and the canonical table-name registry.

One place spells where the pipeline's outputs live, so moving an output
(staging, a backup, a renamed directory) is a change here rather than a hunt
through every reader and writer. Every location is under the processed-data
root, :func:`processed_data_dir`, and is resolved when called: the root is
``$CSR_PROCESSED_DATA_DIR`` when set, else the checkout's ``Processed-Data/``.
That environment variable is the one place a run (or a test) moves every data
root, its child processes included. No library module, and no supported
script's function or default, binds a root at import.

The canonical tables are named once, by :data:`CANONICAL_TABLES`; their
basenames (:data:`CANONICAL_TABLE_BASENAMES`), ``tables.io.SWR_SOURCE_TABLES``
and the covariate registry's row keys follow from it. :func:`table_path` turns
a name and a root into the file, and refuses a name outside the registry.

The upper-case names this module once bound at import (``REPLAY_CONTENT_DIR``
and the rest of :data:`_LEGACY_ROOTS`) still resolve, through the module's
``__getattr__``, for notebooks and exploratory scripts; library code and
supported scripts call the functions.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

__all__ = [
    "ROOT_DIR",
    "BASENAME_TO_TABLE",
    "CANONICAL_TABLES",
    "CANONICAL_TABLE_BASENAMES",
    "MANUSCRIPT",
    "MODEL_SUBDIRS",
    "PROCESSED_DATA_DIR_ENV",
    "QVALS_STANDARDIZATION_REFERENCE",
    "STANDARDIZATION_REFERENCE",
    "SUMMARY_DIR",
    "TEX_PATH",
    "artifact_intervals_root",
    "carey_root",
    "content_occupancy_dir",
    "contfrag_replay_content_dir",
    "detection_comparison_dir",
    "event_features_root",
    "event_spike_evidence_dir",
    "figure_dir",
    "inventory_comparison_root",
    "memoryless_root",
    "model_dir",
    "processed_data_dir",
    "replay_content_dir",
    "swr_inventory_root",
    "table_path",
]

ROOT_DIR = Path(__file__).parent.parent.parent
"""The project root (continuum-swr-replay/, not src/); also
``parameters.ROOT_DIR``."""

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"
"""Where the summaries of the reported values live (``<name>.json``)."""

TEX_PATH = ROOT_DIR / "paper" / "reported_values.tex"
"""The LaTeX macros the summaries' values are emitted to."""

MANUSCRIPT = ROOT_DIR / "paper" / "main.tex"
"""The manuscript's LaTeX source."""

PROCESSED_DATA_DIR_ENV = "CSR_PROCESSED_DATA_DIR"
"""The environment variable that moves the processed-data root."""


def processed_data_dir() -> Path:
    """The processed-data root, read when called.

    ``$CSR_PROCESSED_DATA_DIR`` when set, so a run from any checkout can read
    and write one root; otherwise the checkout's ``Processed-Data/``.
    """
    return Path(os.environ.get(PROCESSED_DATA_DIR_ENV, ROOT_DIR / "Processed-Data"))


def replay_content_dir() -> Path:
    """The non-local replay-content tables and the fits made from them."""
    return processed_data_dir() / "replay_content"


def contfrag_replay_content_dir() -> Path:
    """The ContFrag (SWR- and MUA-labelled) replay-content tables and their fits."""
    return processed_data_dir() / "contfrag_replay_content"


def detection_comparison_dir() -> Path:
    """The event-detector runs (SWR, memoryless, Carey) and their comparison."""
    return processed_data_dir() / "replay_detection_comparison"


def swr_inventory_root() -> Path:
    """Where the SWR runner writes its run folders, one per run id."""
    return detection_comparison_dir() / "swr"


def memoryless_root() -> Path:
    """Where the memoryless runner writes its run folders (and its marks cache)."""
    return detection_comparison_dir() / "memoryless"


def carey_root() -> Path:
    """Where the Carey runner writes its run folders."""
    return detection_comparison_dir() / "carey"


def inventory_comparison_root() -> Path:
    """Where the inventory-comparison runner writes its run folders."""
    return detection_comparison_dir() / "inventory"


def event_features_root() -> Path:
    """Where the event-feature runner writes its run folders."""
    return detection_comparison_dir() / "event_features"


def artifact_intervals_root() -> Path:
    """Where artifact-interval run folders go, one per run id."""
    return processed_data_dir() / "artifact_intervals"


def event_spike_evidence_dir() -> Path:
    """The spike-evidence run that scores every event: one folder per epoch."""
    return processed_data_dir() / "event_spike_evidence"


def content_occupancy_dir() -> Path:
    """Where the content-occupancy analysis writes its per-session table."""
    return processed_data_dir() / "content_occupancy"


# The events measurements' folders, under the processed-data root: each is
# written by its script and read by the events summary.
BRAIN_STATE_REFERENCE_DIR = "brain_state_reference"
"""Each epoch's running and quiet-rest levels
(``scripts/event_properties/measure_brain_state_reference.py``)."""
EVENT_GAMMA_DIR = "event_gamma"
"""The gamma pass over the events (``scripts/event_properties/measure_event_gamma.py``)."""
RIPPLE_CLASS_STRUCTURE_DIR = "ripple_class_structure"
"""The one-class-or-two tests
(``scripts/event_properties/measure_ripple_class_structure.py``)."""
EVENT_START_REST_THETA_DIR = "event_start_rest_theta"
"""Each event's start distance and rest-theta share
(``scripts/event_properties/measure_event_start_and_rest_theta.py``)."""
EVENT_PROPERTY_BOOTSTRAP_DIR = "event_property_bootstrap"
"""The event properties' bootstrap intervals
(``scripts/event_properties/bootstrap_event_properties.py``)."""
DECODER_QUALITY_DIR = "decoder_quality"
"""Each analysed epoch's decoder quality
(``scripts/decoding/measure_decoder_quality.py``)."""
STATE_EVIDENCE_DIR = "state_evidence"
"""A held-out tetrode run's folder inside its run directory
(``scripts/event_evidence/run_state_evidence.py``); the detection summary reads it."""
STATE_EVIDENCE_RUN = "_refresh_20261016"
"""The held-out tetrode run, under the processed-data root, that the detection
and events summaries read (242 epochs, all ``ok``)."""
NL_BOUTS_BY_SWR_STEM = "nl_bouts_by_swr"
"""The stem of the files the SWR split of a held-out tetrode run writes in the
run's folder (``scripts/event_evidence/split_state_evidence_by_swr.py``); the events
summary reads ``<stem>_stats.csv`` and ``<stem>.provenance.json``."""
EVENT_LIKELIHOOD_ADVANTAGE_DIR = "event_likelihood_advantage"
"""The spike likelihood at decoded and actual positions of the canonical
non-local events, under the processed-data root
(``scripts/event_evidence/score_event_likelihood_advantage.py``)."""
EVENT_TETRODE_HOLDOUT_DIR = "event_tetrode_holdout"
"""The held-out tetrode score of the canonical non-local events and their
Local-rest controls, under the processed-data root
(``scripts/event_evidence/score_tetrode_holdout.py``)."""
EVENT_PARTICIPATION_DIR = "event_participation"
"""The tetrode participation and spike rate of the canonical non-local events
and of Local-rest control windows, under the processed-data root
(``scripts/event_evidence/measure_event_participation.py``)."""

# The behavior summary's inputs that are not one of MODEL_SUBDIRS: each folder is
# written by its script and read by the behavior summary.
CONTENT_ESTIMATOR_COMPARISON_DIR = "content_estimator_comparison"
"""The three content estimators compared, under the replay-content root
(``scripts/replay_models/content/compare_content_estimators.py``)."""
EVENT_SOURCE_COMPARISON_DIR = "event_source_comparison"
"""The choice fits by event source, under the replay-content root
(``scripts/replay_models/content/compare_event_source_models.py``)."""


def event_source_comparison_fits_name(event_source: str, brain_area: str) -> str:
    """One event source's comparison fits, in ``EVENT_SOURCE_COMPARISON_DIR``.

    ``brain_area`` is the area whose data the source's events come from (HPC
    for ``non_local``). The behavior summary reads ``non_local`` on HPC.
    """
    return f"{event_source}_{brain_area}_comparison_fits.pkl"


UNCERTAINTY_CONTENT_DIR = "uncertainty_content"
"""The within-arm horse race and the uncertainty tables, under the
replay-content root (``scripts/replay_models/content/analyze_uncertainty_content.py``)."""
CROSS_HISTORY_RIDGE_CV_DIR = "cross_history_ridge_cv"
"""The cross-history ridge cross-validation, under ``model_dir("per_bin_rate")``
(``scripts/replay_models/rate/fit_cross_history_ridge_cv.py``)."""

# The mpfc summary's inputs beside the unit bundles, under the processed-data
# root: each is written by its script and read by the mpfc summary.
PFC_TIMING_CALIBRATION = "pfc_response_followups/timing_calibration.json"
"""The timing calibration (``scripts/pfc/calibrate_pfc_kernel_timing.py``)."""
PFC_SWR_WITHOUT_REPLAY_DIR = "pfc_response_followups/swr_without_replay"
"""The extended-event-set bundles (``scripts/pfc/run_pfc_analysis_bandit.py
--event-set with_swr_without_replay --output-dir`` this folder)."""
PFC_STOP_CONTEXT_DIR = "pfc_response_followups/stop_context"
"""The stop-level context fits (``scripts/pfc/run_pfc_stop_context.py fit``)."""
PFC_STOP_CONTEXT_RESULTS = "results.csv"
"""The stop-context fits' table, in their folder."""
PFC_MARKED_FAMILY_DIR = "pfc_response_followups/marked_family_well_unknown"
"""The follow-up marked models fit with the per-stop baseline prior and the
``well_unknown`` baseline column (bundle contract 3;
``scripts/pfc/run_pfc_marked_model_family.py --output-dir`` this folder)."""
PFC_MARKED_FAMILY_SUMMARY = "summary.parquet"
"""The marked-family summary (``scripts/pfc/summarize_pfc_marked_family.py``),
in its bundle folder."""


def pfc_cohort_name(brain_area: str) -> str:
    """The PFC cohort record's file name, beside the bundles of ``brain_area``.

    ``scripts/pfc/run_pfc_analysis_bandit.py`` writes it; the mpfc summary and
    ``scripts/pfc/run_pfc_stop_context.py`` read it.
    """
    return f"pfc_cohort_{brain_area}.json"


_LEGACY_ROOTS: dict[str, Callable[[], Path]] = {
    "REPLAY_CONTENT_DIR": replay_content_dir,
    "CONTFRAG_REPLAY_CONTENT_DIR": contfrag_replay_content_dir,
    "DETECTION_COMPARISON_DIR": detection_comparison_dir,
    "SWR_INVENTORY_ROOT": swr_inventory_root,
    "MEMORYLESS_ROOT": memoryless_root,
    "CAREY_ROOT": carey_root,
    "INVENTORY_COMPARISON_ROOT": inventory_comparison_root,
    "EVENT_FEATURES_ROOT": event_features_root,
    "ARTIFACT_INTERVALS_DIR": artifact_intervals_root,
}
"""The constants this module bound at import, by the function now behind each.

Kept for notebooks and exploratory scripts. ``paths.REPLAY_CONTENT_DIR`` is
resolved at each access; ``from ... import REPLAY_CONTENT_DIR`` binds the value
at that import, as the constant did. ``tests/test_paths.py`` keeps them out
of ``src/`` and the supported scripts."""


def __getattr__(name: str) -> Path:
    if name in _LEGACY_ROOTS:
        return _LEGACY_ROOTS[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


MODEL_SUBDIRS: dict[str, str] = {
    "content_arm": "models",
    "content_patch": "models_patch",
    "per_bin_rate": "models_per_bin_rate",
    "per_event_swr_power": "models_per_event_swr_power",
    "per_stop_rate": "models_per_stop_rate",
    "swr_contrast": "models_swr_contrast",
}
"""The fitted-model directories, by contents, relative to a replay-content root.

``swr_contrast`` holds the content is_swr-contrast fits (arm and patch) and
their audit tables. They have their own directory because the arm content
ladder publishes ``content_arm`` by replacing the whole directory, which would
delete anything else written there.

:func:`model_dir` joins one to a root; use it rather than joining by hand.
"""


def model_dir(key: str, root: Path | None = None) -> Path:
    """Return fitted-model directory `key` under `root`.

    Parameters
    ----------
    key : str
        One of :data:`MODEL_SUBDIRS`.
    root : Path, optional
        The replay-content root (the canonical one, a staging root, a backup);
        :func:`replay_content_dir` when None, resolved at the call.

    Returns
    -------
    Path

    Raises
    ------
    KeyError
        If `key` is not a fitted-model directory.
    """
    if key not in MODEL_SUBDIRS:
        raise KeyError(
            f"{key!r} is not a fitted-model directory; expected one of "
            f"{tuple(MODEL_SUBDIRS)}"
        )
    return (replay_content_dir() if root is None else root) / MODEL_SUBDIRS[key]


def figure_dir(topic: str, root: Path | None = None) -> Path:
    """Return the figure directory for `topic` beside data root `root`.

    Figures land at ``<root>.parent / "paper" / "figures" / <topic>``, beside
    the data they were drawn from, not at ``parameters.FIGURE_DIR``. On the
    default root the two are the same folder; a run on another root
    (``CSR_PROCESSED_DATA_DIR``, e.g. a ``run_refits`` staging run) writes its
    figures beside that root instead. That keeps the paper's figures only when
    the staging root's parent is not the repository, and two roots that share
    a parent share a figure folder.

    Parameters
    ----------
    topic : str
        The figure topic subdirectory, e.g. ``"rate"``.
    root : Path, optional
        The processed-data root; :func:`processed_data_dir` when None,
        resolved at the call.

    Returns
    -------
    Path
    """
    return (root or processed_data_dir()).parent / "paper" / "figures" / topic


CANONICAL_TABLES: tuple[str, ...] = (
    "event",
    "event_arm",
    "event_patch",
    "event_sequence",
    "stop",
    "qvals",
    "event_onset",
    "stop_timebin",
)
"""The canonical tables, by covariate-registry key, in the order every
freeze, manifest and parity sweep iterates them. The one list: the registry's
row keys (``tables.covariate_registry``) must name the same tables."""

BASENAME_TO_TABLE: dict[str, str] = {
    f"{table}_table": table for table in CANONICAL_TABLES
}
"""On-disk table basename -> the registry table key that owns its row key."""

CANONICAL_TABLE_BASENAMES: tuple[str, ...] = tuple(BASENAME_TO_TABLE)

STANDARDIZATION_REFERENCE = "standardization_reference.json"
"""The replay-content tables' ``_z`` standardization reference, beside them at
a replay-content root."""

QVALS_STANDARDIZATION_REFERENCE = "qvals_standardization_reference.json"
"""``qvals_table``'s ``_z`` standardization reference, beside it at the root."""


def table_path(name: str, root: Path | None = None, *, fmt: str = "parquet") -> Path:
    """Return the file of canonical table `name` under `root`.

    Every canonical table sits at `root`, so a pipeline root (the canonical
    one, a staging one) and a frozen golden baseline share one flat layout.

    Parameters
    ----------
    name : str
        One of :data:`CANONICAL_TABLE_BASENAMES`.
    root : Path, optional
        The replay-content root; :func:`replay_content_dir` when None, resolved
        at the call.
    fmt : str, optional
        The file suffix without its dot: ``"parquet"``, or ``"pkl"`` for a
        retired pickle (only ``scripts/tables/convert_tables_to_parquet.py``
        reads those, with ``pandas.read_pickle``).

    Returns
    -------
    Path

    Raises
    ------
    KeyError
        If `name` is not a canonical table.
    """
    if name not in BASENAME_TO_TABLE:
        raise KeyError(
            f"{name!r} is not a canonical table; expected one of "
            f"{CANONICAL_TABLE_BASENAMES}"
        )
    return (replay_content_dir() if root is None else root) / f"{name}.{fmt}"
