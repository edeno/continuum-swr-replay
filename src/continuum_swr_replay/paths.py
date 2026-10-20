"""Output locations and the canonical table-name registry.

One place spells where the pipeline's outputs live, so moving an output
(staging, a backup, the deferred ``models_per_bin_rate/`` rename) is a change
here rather than a hunt through every reader and writer. Every constant is a
path under :data:`~continuum_swr_replay.parameters.PROCESSED_DATA_DIR`.

The canonical tables are named by :data:`CANONICAL_TABLE_BASENAMES`;
:func:`table_path` turns a name and a root into the file, and refuses a name
outside the registry.
"""

from __future__ import annotations

from pathlib import Path

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

__all__ = [
    "ARTIFACT_INTERVALS_DIR",
    "BASENAME_TO_TABLE",
    "CANONICAL_TABLE_BASENAMES",
    "CONTFRAG_REPLAY_CONTENT_DIR",
    "DETECTION_COMPARISON_DIR",
    "MODEL_SUBDIRS",
    "REPLAY_CONTENT_DIR",
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

ARTIFACT_INTERVALS_DIR = PROCESSED_DATA_DIR / "artifact_intervals"
"""Where artifact-interval run folders go, one per run id."""

MODEL_SUBDIRS: dict[str, str] = {
    "content_arm": "models",
    "content_patch": "models_patch",
    # Besides the per-bin rate ladder this directory holds ``qvals_table``,
    # ``swr_contrast_rate``, ``uncertainty_rate`` and ``swr_power_model``; its
    # rename is deferred (lead decision Q7).
    "per_bin_rate": "models_per_bin_rate",
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


# On-disk table basename -> the registry table key that owns its row key.
BASENAME_TO_TABLE: dict[str, str] = {
    "event_table": "event",
    "event_arm_table": "event_arm",
    "event_patch_table": "event_patch",
    "event_sequence_table": "event_sequence",
    "stop_table": "stop",
    "qvals_table": "qvals",
    "event_onset_table": "event_onset",
    "stop_timebin_table": "stop_timebin",
}
CANONICAL_TABLE_BASENAMES: tuple[str, ...] = tuple(BASENAME_TO_TABLE)


def table_path(
    name: str, root: Path = REPLAY_CONTENT_DIR, *, fmt: str = "parquet"
) -> Path:
    """Return the file of canonical table `name` under `root`.

    ``qvals_table`` sits one level down, in the per-bin rate model directory;
    every other table sits at `root`. That is the layout of every pipeline
    root (the canonical one and a staging one alike); a frozen golden baseline
    is flat and is read by ``root / name`` instead.

    Parameters
    ----------
    name : str
        One of :data:`CANONICAL_TABLE_BASENAMES`.
    root : Path, optional
        The replay-content root; :data:`REPLAY_CONTENT_DIR` by default.
    fmt : str, optional
        The file suffix without its dot (``"parquet"`` or ``"pkl"``). An empty
        string returns the bare basename, which ``tables.io.read_table`` and
        ``write_table`` resolve to the parquet file.

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
    directory = model_dir("per_bin_rate", root) if name == "qvals_table" else root
    return directory / (f"{name}.{fmt}" if fmt else name)
