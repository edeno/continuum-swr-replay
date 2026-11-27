"""The checks of the stored held-out runs, and the held-out tetrode run's reader.

A held-out run (the held-out tetrode run, the restriction runs) records its
fold design in its settings; the values and figures describe folds of whole
trials, so a run of another design is refused (:func:`require_trial_folds`).
The held-out tetrode run is read only through :func:`read_tetrode_holdout`,
which refuses a run off trial folds and a table without every column the
runner writes (``tetrode_holdout.RUN_COLUMNS``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_events_in,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    TRIAL_DESIGN,
    require_run_columns,
)
from continuum_swr_replay.paths import TETRODE_HOLDOUT_NAME
from continuum_swr_replay.runs.io import read_complete_run
from continuum_swr_replay.runs.staleness import read_output_manifest


def require_trial_folds(manifest: dict, what: str, rerun: str) -> None:
    """Refuse a held-out run whose folds are not whole trials.

    A held-out run records its fold design in its settings (``fold_design``).
    A run of an older design (contiguous blocks of time, or no design recorded)
    cannot give values that describe folds of whole trials.

    Raises
    ------
    ValueError
        If ``manifest``'s settings do not record ``"fold_design": "trials"``;
        the message ends with ``rerun``, what to run instead.
    """
    design = manifest.get("settings", {}).get("fold_design")
    if design != TRIAL_DESIGN:
        raise ValueError(
            f"{what} is not held out in folds of whole trials (its settings "
            f"record fold_design {design!r}, not {TRIAL_DESIGN!r}). {rerun}"
        )


def require_tetrode_holdout_folds(manifest: dict, what: str) -> None:
    """Refuse a held-out tetrode run whose folds are not whole trials
    (:func:`require_trial_folds`).

    Raises
    ------
    ValueError
        If ``manifest``'s settings do not record ``"fold_design": "trials"``.
    """
    require_trial_folds(
        manifest,
        what,
        "Put the held-out tetrode run on folds of whole trials "
        "(scripts/event_evidence/score_tetrode_holdout.py) in this folder, then "
        "rerun scripts/event_evidence/sum_nonlocal_restriction_windows.py on it.",
    )


def read_tetrode_holdout(
    folder: Path, epoch_ids: Iterable[str], selected: object, what: str
) -> pd.DataFrame:
    """The held-out tetrode run's per-event table over the selected events: the
    run refused when its folds are not whole trials
    (:func:`require_tetrode_holdout_folds`), read over ``epoch_ids``
    (``read_complete_run``) and selected by ``cohort_selection.select_events_in``
    with ``selected``; refused without every column of the run
    (``tetrode_holdout.require_run_columns``). ``what`` names the folder in the
    messages."""
    require_tetrode_holdout_folds(read_output_manifest(folder), what)
    table = select_events_in(
        read_complete_run(folder, TETRODE_HOLDOUT_NAME, epoch_ids), selected
    )
    require_run_columns(table, what)
    return table
