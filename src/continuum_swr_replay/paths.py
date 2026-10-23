"""Output locations and the canonical table-name registry.

One place spells where the pipeline's outputs live, so moving an output
(staging, a backup, a renamed directory) is a change here rather than a hunt
through every reader and writer. Every constant is a path under
:data:`~continuum_swr_replay.parameters.PROCESSED_DATA_DIR`.

The canonical tables are named once, by :data:`CANONICAL_TABLES`; their
basenames (:data:`CANONICAL_TABLE_BASENAMES`), ``tables.io.SWR_SOURCE_TABLES``
and the covariate registry's row keys follow from it. :func:`table_path` turns
a name and a root into the file, and refuses a name outside the registry.
"""

from __future__ import annotations

from pathlib import Path

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

__all__ = [
    "ARTIFACT_INTERVALS_DIR",
    "BASENAME_TO_TABLE",
    "CAREY_ROOT",
    "CANONICAL_TABLES",
    "CANONICAL_TABLE_BASENAMES",
    "CONTFRAG_REPLAY_CONTENT_DIR",
    "DETECTION_COMPARISON_DIR",
    "EVENT_FEATURES_ROOT",
    "INVENTORY_COMPARISON_ROOT",
    "MEMORYLESS_ROOT",
    "MODEL_SUBDIRS",
    "QVALS_STANDARDIZATION_REFERENCE",
    "REPLAY_CONTENT_DIR",
    "STANDARDIZATION_REFERENCE",
    "SWR_INVENTORY_ROOT",
    "model_dir",
    "table_path",
]

REPLAY_CONTENT_DIR = PROCESSED_DATA_DIR / "replay_content"
"""The non-local replay-content tables and the fits made from them."""

CONTFRAG_REPLAY_CONTENT_DIR = PROCESSED_DATA_DIR / "contfrag_replay_content"
"""The ContFrag (SWR- and MUA-labelled) replay-content tables and their fits."""

DETECTION_COMPARISON_DIR = PROCESSED_DATA_DIR / "replay_detection_comparison"
"""The event-detector runs (SWR, memoryless, Carey) and their comparison."""

SWR_INVENTORY_ROOT = DETECTION_COMPARISON_DIR / "swr"
"""Where the SWR runner writes its run folders, one per run id."""

MEMORYLESS_ROOT = DETECTION_COMPARISON_DIR / "memoryless"
"""Where the memoryless runner writes its run folders (and its marks cache)."""

CAREY_ROOT = DETECTION_COMPARISON_DIR / "carey"
"""Where the Carey runner writes its run folders."""

INVENTORY_COMPARISON_ROOT = DETECTION_COMPARISON_DIR / "inventory"
"""Where the inventory-comparison runner writes its run folders."""

EVENT_FEATURES_ROOT = DETECTION_COMPARISON_DIR / "event_features"
"""Where the event-feature runner writes its run folders."""

ARTIFACT_INTERVALS_DIR = PROCESSED_DATA_DIR / "artifact_intervals"
"""Where artifact-interval run folders go, one per run id."""

MODEL_SUBDIRS: dict[str, str] = {
    "content_arm": "models",
    "content_patch": "models_patch",
    "per_bin_rate": "models_per_bin_rate",
    "per_event_swr_power": "models_per_event_swr_power",
    "per_stop_rate": "models_per_stop_rate",
}
"""The fitted-model directories, by contents, relative to a replay-content root.

:func:`model_dir` joins one to a root; use it rather than joining by hand.
"""


def model_dir(key: str, root: Path = REPLAY_CONTENT_DIR) -> Path:
    """Return fitted-model directory `key` under `root`.

    Parameters
    ----------
    key : str
        One of :data:`MODEL_SUBDIRS`.
    root : Path, optional
        The replay-content root (the canonical one, a staging root, a backup);
        :data:`REPLAY_CONTENT_DIR` by default. The default is bound at import,
        so a test redirects by passing its own root, not by patching this
        module.

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
    return root / MODEL_SUBDIRS[key]


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


def table_path(
    name: str, root: Path = REPLAY_CONTENT_DIR, *, fmt: str = "parquet"
) -> Path:
    """Return the file of canonical table `name` under `root`.

    Every canonical table sits at `root`, so a pipeline root (the canonical
    one, a staging one) and a frozen golden baseline share one flat layout.

    Parameters
    ----------
    name : str
        One of :data:`CANONICAL_TABLE_BASENAMES`.
    root : Path, optional
        The replay-content root; :data:`REPLAY_CONTENT_DIR` by default.
    fmt : str, optional
        The file suffix without its dot (``"parquet"`` or ``"pkl"``).

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
    return root / f"{name}.{fmt}"
