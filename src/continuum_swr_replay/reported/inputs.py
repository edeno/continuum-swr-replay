"""The stored outputs the events summary and the figures read, and why each
is stale.

Where each output is (:func:`default_inputs`), why an output is not made from
the current inputs (each ``*_staleness`` and ``*_note`` gives a reason or None:
the summary marks a value pending on it, a figure refuses to draw), and the
readers that both use: the held-out event run (:func:`read_heldout_restriction`)
and the per-epoch measurements checked against the decodes
(:func:`read_checked`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detector_comparison.inventory_run import (
    decode_path,
)
from continuum_swr_replay.analysis.event_definition import swr_source
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_dot_with_swr_source,
    swr_intervals,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    subset_staleness,
)
from continuum_swr_replay.analysis.event_evidence.heldout_runs import (
    require_trial_folds,
)
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import FIT
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_staleness,
    time_mask_staleness,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.decoding.identifiers import format_epoch_identifier
from continuum_swr_replay.decoding.results import load_results
from continuum_swr_replay.paths import (
    BRAIN_STATE_REFERENCE_DIR,
    CONTENT_ALONG_ARMS_DIR,
    CONTENT_MASS_ALONG_ARMS_DIR,
    DECODER_QUALITY_DIR,
    EVENT_BIN_FACTS_DIR,
    EVENT_CONTENT_PLACE_RATE_DIR,
    EVENT_FEATURES_RUN,
    EVENT_GAMMA_DIR,
    EVENT_GAMMA_TRIGGER_WINDOWS_DIR,
    EVENT_LIKELIHOOD_ADVANTAGE_DIR,
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
    EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
    EVENT_PARTICIPATION_DIR,
    EVENT_PROPERTY_BOOTSTRAP_DIR,
    EVENT_RULE_SWEEP_DIR,
    EVENT_SPEED_DIR,
    EVENT_START_REST_THETA_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    EVENT_TRIGGER_WINDOWS_DIR,
    EXAMPLE_SESSION_DIR,
    EXPOSURE_IN_VISITS_DIR,
    HELDOUT_EVENTS_NAME,
    HELDOUT_TIME_SUMS_NAME,
    NL_BOUTS_BY_SWR_STEM,
    NONLOCAL_BY_SIGNAL_DIR,
    NONLOCAL_PROBABILITY_DIR,
    RIPPLE_CLASS_STRUCTURE_DIR,
    ROOT_DIR,
    STATE_BY_SIGNAL_DIR,
    STATE_CALIBRATION_IN_VISITS_DIR,
    STATE_EVIDENCE_DIR,
    STATE_EVIDENCE_RUN,
    event_features_root,
    event_spike_evidence_dir,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.values import (
    file_record,
    rel,
)
from continuum_swr_replay.runs.epoch_records import MANIFEST
from continuum_swr_replay.runs.io import (
    ok_epochs,
    read_complete_run,
    read_manifest,
    sha256_file,
)
from continuum_swr_replay.runs.staleness import read_output_manifest, recorded_sha256


@dataclass(frozen=True)
class EventsInputs:
    """The files the events summary reads."""

    event_table: Path
    event_features_run: Path
    spike_evidence_dir: Path
    heldout_by_swr: Path
    content_along_arms: Path
    content_mass_along_arms: Path
    brain_state_reference: Path
    event_gamma: Path
    class_structure: Path
    nonlocal_by_signal: Path
    nonlocal_probability: Path
    state_by_signal: Path
    event_rule_sweep: Path
    state_calibration_in_visits: Path
    event_start_rest_theta: Path
    event_speed: Path
    property_bootstrap: Path
    decoder_quality: Path
    event_participation: Path
    likelihood_advantage: Path
    tetrode_holdout: Path
    content_place_rate: Path
    nonlocal_restriction: Path
    nonlocal_restriction_windows: Path
    nonlocal_restriction_heldout_events: Path
    bin_facts: Path
    exposure_in_visits: Path
    trigger_windows: Path
    gamma_trigger_windows: Path


def default_inputs() -> EventsInputs:
    return EventsInputs(
        event_table=table_path("event_table", replay_content_dir()),
        event_features_run=event_features_root() / EVENT_FEATURES_RUN,
        spike_evidence_dir=event_spike_evidence_dir(),
        heldout_by_swr=processed_data_dir()
        / STATE_EVIDENCE_RUN
        / STATE_EVIDENCE_DIR
        / f"{NL_BOUTS_BY_SWR_STEM}_stats.csv",
        content_along_arms=processed_data_dir() / CONTENT_ALONG_ARMS_DIR,
        content_mass_along_arms=processed_data_dir() / CONTENT_MASS_ALONG_ARMS_DIR,
        brain_state_reference=processed_data_dir()
        / BRAIN_STATE_REFERENCE_DIR
        / "reference.parquet",
        event_gamma=processed_data_dir() / EVENT_GAMMA_DIR,
        class_structure=processed_data_dir() / RIPPLE_CLASS_STRUCTURE_DIR,
        nonlocal_by_signal=processed_data_dir() / NONLOCAL_BY_SIGNAL_DIR,
        nonlocal_probability=processed_data_dir() / NONLOCAL_PROBABILITY_DIR,
        state_by_signal=processed_data_dir() / STATE_BY_SIGNAL_DIR,
        event_rule_sweep=processed_data_dir() / EVENT_RULE_SWEEP_DIR,
        state_calibration_in_visits=processed_data_dir()
        / STATE_CALIBRATION_IN_VISITS_DIR,
        event_start_rest_theta=processed_data_dir() / EVENT_START_REST_THETA_DIR,
        event_speed=processed_data_dir() / EVENT_SPEED_DIR,
        property_bootstrap=processed_data_dir() / EVENT_PROPERTY_BOOTSTRAP_DIR,
        decoder_quality=processed_data_dir() / DECODER_QUALITY_DIR,
        event_participation=processed_data_dir() / EVENT_PARTICIPATION_DIR,
        likelihood_advantage=processed_data_dir() / EVENT_LIKELIHOOD_ADVANTAGE_DIR,
        tetrode_holdout=processed_data_dir() / EVENT_TETRODE_HOLDOUT_DIR,
        content_place_rate=processed_data_dir() / EVENT_CONTENT_PLACE_RATE_DIR,
        nonlocal_restriction=processed_data_dir() / EVENT_NONLOCAL_RESTRICTION_DIR,
        nonlocal_restriction_windows=processed_data_dir()
        / EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
        nonlocal_restriction_heldout_events=processed_data_dir()
        / EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
        bin_facts=processed_data_dir() / EVENT_BIN_FACTS_DIR,
        exposure_in_visits=processed_data_dir() / EXPOSURE_IN_VISITS_DIR,
        trigger_windows=processed_data_dir() / EVENT_TRIGGER_WINDOWS_DIR,
        gamma_trigger_windows=processed_data_dir() / EVENT_GAMMA_TRIGGER_WINDOWS_DIR,
    )


def input_staleness(
    manifest: dict, name: str, current_sha256: str, what: str, adjective: str, noun: str
) -> str | None:
    """Why an output is stale for one of its inputs, or None.

    ``manifest`` records the input's SHA-256 under ``name`` or ``inputs.<name>``
    (``runs.staleness.recorded_sha256``); an output recording none cannot be
    tied to the current input and is treated as stale.
    """
    recorded = recorded_sha256(manifest, name)
    if recorded is None:
        return f"{what} records no {adjective} SHA-256; rerun it before citing"
    if recorded != current_sha256:
        return (
            f"{what} was made from another {noun} than the current one "
            f"(sha256 {recorded[:12]}, current {current_sha256[:12]}); rerun it "
            "before citing"
        )
    return None


def event_table_staleness(manifest: dict, current_sha256: str, what: str) -> str | None:
    """Why an output made from the canonical event table is stale, or None."""
    return input_staleness(
        manifest, "event_table", current_sha256, what, "event-table", "event table"
    )


def require_fold_fit(manifest: dict, what: str) -> None:
    """Refuse a restriction run that is not the fold fit on folds of whole trials.

    The values describe scores under a decoder fit on the other folds, five
    folds of whole trials balanced by destination well; the fold-fit run's
    settings record ``"fit": "folds"`` and ``"fold_design": "trials"``. A run of
    the earlier design, which held out only the encodings and ran the models
    with the transitions fit on the whole epoch, records no ``fit``; a run of
    another fold design is refused by :func:`require_trial_folds`. Neither can
    give these values.

    Raises
    ------
    ValueError
        If ``manifest``'s settings do not record ``"fit": "folds"`` and
        ``"fold_design": "trials"``.
    """
    fit = manifest.get("settings", {}).get("fit")
    rerun = (
        "Put the fold-fit run (scripts/event_evidence/score_nonlocal_restriction.py) "
        "in this folder, then rerun scripts/event_evidence/measure_bin_facts.py, "
        "scripts/event_evidence/detect_heldout_events.py and "
        "scripts/event_evidence/sum_nonlocal_restriction_windows.py on it."
    )
    if fit != FIT:
        raise ValueError(
            f"{what} is not the fold fit (its settings record fit {fit!r}, not "
            f"{FIT!r}): it is the earlier design, which held out only the "
            f"encodings. {rerun}"
        )
    require_trial_folds(manifest, what, rerun)


def run_staleness(manifest: dict, folder: Path, name: str, run_dir: Path) -> str | None:
    """Why the output in ``folder`` (its ``manifest``) is stale for the run
    ``name`` in ``run_dir`` that it read, or None: it records no digest of that
    run's manifest, or another than the current one."""
    return input_staleness(
        manifest,
        name,
        file_record(run_dir / MANIFEST, ROOT_DIR)["sha256"],
        rel(folder),
        f"{name} run",
        f"{name} run",
    )


@dataclass(frozen=True)
class HeldoutRestriction:
    """The held-out event run's inputs to the restriction values and Figure 2k
    (:func:`read_heldout_restriction`), and why a value read from them is
    pending (each None when it is not)."""

    events: pd.DataFrame
    """The events detected from the held-out decode that start inside a well
    visit (``starts_in_well_visit``)."""
    time_sums: pd.DataFrame
    """The time sums of every epoch (``restriction_totals.restriction_time_sums``)."""
    bin_s: pd.Series
    """Each epoch's decode time step (s), indexed by ``epoch_id``, as the run read
    it from the decode frame's time (checked there against the per-bin table's)."""
    restriction_run: str | None
    """The run was not made from the current restriction run."""
    bin_facts_run: str | None
    """The run was not made from the current per-bin facts run."""
    session_cache: str | None
    """The per-bin facts, which give the time sums, the events' start and the
    windows' bins their well-visit flag, were not made from the current
    session cache."""
    event_table: str | None
    """The run was not made from the current event table."""


def restriction_inputs_note(
    restriction_dir: Path,
    epochs: Iterable[str],
    label: str | None = None,
    features_manifest: dict | None = None,
) -> str | None:
    """Why the restriction run in ``restriction_dir`` was not made from the
    current decode frames and SWR source, or None: :func:`restriction_input_staleness`
    with the decode frames the event features run records for each epoch
    (``features_manifest``, read from the canonical run when None). ``label``
    names the run in the note (by default ``the restriction run (<folder>)``).
    The figures that draw the restriction run refuse it then
    (:func:`heldout_restriction_refusal`)."""
    if features_manifest is None:
        features_manifest = read_manifest(event_features_root() / EVENT_FEATURES_RUN)
    return restriction_input_staleness(
        read_output_manifest(restriction_dir),
        {
            epoch_id: record.get("decode", {}).get("sha256")
            for epoch_id, record in features_manifest["epochs"].items()
        },
        epochs,
        f"the restriction run ({restriction_dir})" if label is None else label,
    )


def checked_reference(reference: pd.DataFrame, folder: Path, name: str) -> pd.DataFrame:
    """A per-epoch measurement (the brain-state reference, the counts by signal
    of Figure 2i and its supplement), if it was made over the time inside the well visits, from the
    current session cache's visits and from the decodes the event features run
    records for every analyzed epoch.

    Parameters
    ----------
    reference : pd.DataFrame
        The measurement, with each row's ``epoch_id`` and ``decode_sha256``.
    folder : Path
        Its folder, whose ``manifest.json`` records the subset it is over and
        the session-cache entries it read the visits from.
    name : str
        What it is, for the error.

    Raises
    ------
    ValueError
        If it was not made over the analyses' subset or from the current
        session cache (``prepare_data.time_mask_staleness``), an analyzed epoch
        is missing or extra, or its decode hash differs.
    """
    note = time_mask_staleness(
        read_output_manifest(folder), reference["epoch_id"], f"the {name} ({folder})"
    )
    if note is not None:
        raise ValueError(note)
    manifest = read_manifest(event_features_root() / EVENT_FEATURES_RUN)
    expected = {
        epoch_id: manifest["epochs"][epoch_id]["decode"]["sha256"]
        for epoch_id in analyzed_epoch_ids()
    }
    recorded = dict(zip(reference["epoch_id"], reference["decode_sha256"], strict=True))
    if recorded != expected:
        differ = sorted(
            epoch_id
            for epoch_id in expected.keys() | recorded.keys()
            if expected.get(epoch_id) != recorded.get(epoch_id)
        )
        raise ValueError(
            f"the {name} was not made from the decodes of event "
            f"features run {EVENT_FEATURES_RUN}: {len(differ)} epochs differ, "
            f"e.g. {differ[:3]}"
        )
    return reference


def read_checked(folder: Path, name: str, file: str = "counts.parquet") -> pd.DataFrame:
    """A per-epoch measurement's table (``file`` in ``folder``), refused as
    :func:`checked_reference` refuses it (``name`` says what it is)."""
    return checked_reference(pd.read_parquet(folder / file), folder, name)


def restriction_input_staleness(
    manifest: dict,
    decode_sha256: Mapping[str, str],
    epoch_ids: Iterable[str],
    what: str,
) -> str | None:
    """Why the restriction run was not made from the current decode frames and
    SWR source, or None.

    The run's settings record the SWR source by name (``swr_source``, which
    names the SWR inventory run and the artifact run) and each epoch's record
    the SHA-256 of the decode frame it read (``epoch_inputs.data_over_time``).
    The current source is ``swr_source.swr_source_name()``. The current decode
    frames are the ones the event features run records (``decode_sha256``),
    which the figures also take as the decodes (:func:`checked_reference`); no decode frame is hashed here, so a decode remade
    without a new event features run is not seen.

    Parameters
    ----------
    manifest : dict
        The restriction run's manifest.
    decode_sha256 : Mapping of str to str
        Each epoch's decode frame's SHA-256, as the event features run records it.
    epoch_ids : iterable of str
        The epochs read; one the run has no record of, or the features run no
        digest of, counts as another decode.
    what : str
        The restriction run, for the note.
    """
    notes = []
    recorded = manifest.get("settings", {}).get("swr_source")
    current = swr_source.swr_source_name()
    if recorded != current:
        notes.append(
            f"its settings record the SWR source {recorded!r}, not the current "
            f"{current!r}"
        )
    records = manifest.get("epochs", {})

    def recorded_decode(epoch_id: str) -> str | None:
        inputs = records.get(epoch_id, {}).get("epoch_inputs", {})
        return inputs.get("data_over_time", {}).get("sha256")

    differ = sorted(
        epoch_id
        for epoch_id in epoch_ids
        if recorded_decode(epoch_id) is None
        or recorded_decode(epoch_id) != decode_sha256.get(epoch_id)
    )
    if differ:
        notes.append(
            f"{len(differ)} of its epochs record another decode frame than the "
            f"one event features run {EVENT_FEATURES_RUN} records (e.g. "
            f"{differ[:3]})"
        )
    if not notes:
        return None
    return (
        f"{what} was not made from the current decodes and SWR source "
        f"({'; '.join(notes)}); rerun scripts/event_evidence/"
        "score_nonlocal_restriction.py and the runs made from it before citing"
    )


HELDOUT_EVENTS_RERUN = "scripts/event_evidence/detect_heldout_events.py"
HELDOUT_RESTRICTION_RERUN = {
    "restriction_inputs": "scripts/event_evidence/score_nonlocal_restriction.py "
    "and the runs made from it",
    "restriction_run": HELDOUT_EVENTS_RERUN,
    "bin_facts_run": HELDOUT_EVENTS_RERUN,
    "session_cache": "scripts/event_evidence/measure_bin_facts.py, then "
    + HELDOUT_EVENTS_RERUN,
}
"""The runners to make again when a figure refuses the held-out event run, by
what is not current (:func:`heldout_restriction_refusal`)."""


def heldout_restriction_refusal(
    heldout: HeldoutRestriction, restriction_inputs: str | None
) -> tuple[str, str] | None:
    """Why a figure drawing the held-out event run refuses it, or None.

    Parameters
    ----------
    heldout : HeldoutRestriction
        The run's inputs and reasons (:func:`read_heldout_restriction`); its
        event-table reason is the summary's alone.
    restriction_inputs : str or None
        Why the restriction run was not made from the current decodes and SWR
        source (:func:`restriction_inputs_note`).

    Returns
    -------
    tuple of (str, str) or None
        The first reason that is not None, without its closing advice to rerun
        before citing, and the runners to make again
        (:data:`HELDOUT_RESTRICTION_RERUN`).
    """
    for name, reason in (
        ("restriction_inputs", restriction_inputs),
        ("restriction_run", heldout.restriction_run),
        ("bin_facts_run", heldout.bin_facts_run),
        ("session_cache", heldout.session_cache),
    ):
        if reason is not None:
            return reason.split("; rerun ")[0], HELDOUT_RESTRICTION_RERUN[name]
    return None


def read_heldout_restriction(
    heldout_dir: Path,
    restriction_dir: Path,
    bin_facts_dir: Path,
    epoch_ids: Iterable[str],
    event_table_sha256: str,
) -> HeldoutRestriction:
    """The held-out event run's events, time sums and time steps, with the
    reasons a value read from them is pending.

    Parameters
    ----------
    heldout_dir, restriction_dir, bin_facts_dir : Path
        The held-out event run, the restriction run and the per-bin facts run.
    epoch_ids : iterable of str
        The epochs read (every analyzed epoch with an event).
    event_table_sha256 : str
        The current event table's SHA-256.

    Raises
    ------
    ValueError
        If the restriction run is not the fold fit on folds of whole trials
        (:func:`require_fold_fit`).
    """
    require_fold_fit(read_output_manifest(restriction_dir), rel(restriction_dir))
    epoch_ids = list(epoch_ids)
    manifest = read_output_manifest(heldout_dir)
    events = read_complete_run(heldout_dir, HELDOUT_EVENTS_NAME, epoch_ids)
    facts_manifest = read_output_manifest(bin_facts_dir)
    return HeldoutRestriction(
        events=events[events["starts_in_well_visit"].astype(bool)],
        time_sums=read_complete_run(heldout_dir, HELDOUT_TIME_SUMS_NAME, epoch_ids),
        bin_s=pd.Series(
            {
                epoch_id: record["bin_s"]
                for epoch_id, record in manifest["epochs"].items()
            }
        ),
        restriction_run=run_staleness(
            manifest, heldout_dir, "nonlocal_restriction", restriction_dir
        ),
        bin_facts_run=run_staleness(manifest, heldout_dir, "bin_facts", bin_facts_dir),
        session_cache=session_cache_staleness(
            facts_manifest, ok_epochs(facts_manifest), rel(bin_facts_dir)
        ),
        event_table=event_table_staleness(
            manifest, event_table_sha256, rel(heldout_dir)
        ),
    )


def output_staleness(
    folder: Path,
    *,
    event_table: Path | None = None,
    epoch_ids: Iterable[str] | None = None,
) -> str | None:
    """Why a stored output is stale, or None: the first of its reasons.

    It was not made from the current event table (checked when ``event_table``
    is given), or not over the analyses' subset
    (``cohort_selection.subset_staleness``), or, for a time-masked output
    (``epoch_ids`` given, the epochs it holds), not over the subset from the
    current well visits (``prepare_data.time_mask_staleness``).

    Parameters
    ----------
    folder : Path
        The output's folder, with its ``manifest.json``.
    event_table : Path, optional
        The current canonical event table.
    epoch_ids : iterable of str, optional
        The epochs a time-masked output holds.
    """
    manifest = read_output_manifest(folder)
    what = rel(folder)
    reasons = (
        None
        if event_table is None
        else event_table_staleness(manifest, sha256_file(event_table), what),
        subset_staleness(manifest, what)
        if epoch_ids is None
        else time_mask_staleness(manifest, epoch_ids, what),
    )
    return next((reason for reason in reasons if reason is not None), None)


def participation_staleness(
    manifest: dict, epoch_ids: Iterable[str], what: str
) -> str | None:
    """Why the participation run's controls are not Local rest inside the
    current well visits, or None: its settings record no subset or another
    (``cohort_selection.subset_staleness``; the run records the subset in its
    settings), or it read other visits than the current session cache's
    (``prepare_data.session_cache_staleness``)."""
    notes = (
        subset_staleness(manifest.get("settings", {}), what),
        session_cache_staleness(manifest, epoch_ids, what),
    )
    return "; ".join(note for note in notes if note) or None


def read_figure_restriction(
    inputs: EventsInputs, epoch_ids: Iterable[str]
) -> tuple[HeldoutRestriction, tuple[str, str] | None]:
    """The held-out event run as a figure reads it, with why the figure
    refuses it (:func:`heldout_restriction_refusal`), or None.

    Parameters
    ----------
    inputs : EventsInputs
    epoch_ids : iterable of str
        The epochs read (every analyzed epoch with an event).
    """
    epoch_ids = list(epoch_ids)
    heldout = read_heldout_restriction(
        inputs.nonlocal_restriction_heldout_events,
        inputs.nonlocal_restriction,
        inputs.bin_facts,
        epoch_ids,
        sha256_file(inputs.event_table),
    )
    return heldout, heldout_restriction_refusal(
        heldout, restriction_inputs_note(inputs.nonlocal_restriction, epoch_ids)
    )


def panel_refusal(figure: str, letter: str, cause: str, rerun: str) -> str:
    """Why a figure's panel is not drawn: ``"<figure>: panel <letter> is not
    drawn: <cause>. Rerun <rerun>."``"""
    return f"{figure}: panel {letter} is not drawn: {cause}. Rerun {rerun}."


def read_trigger_windows(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stored windows around event starts and No-Spike starts
    (``scripts/event_properties/measure_trigger_windows.py``), and one row per
    epoch of its record.

    Parameters
    ----------
    folder : Path
        The run's folder, with ``windows.parquet`` and ``manifest.json``.

    Returns
    -------
    windows : pd.DataFrame
        One row per window and signal
        (``triggered_analysis.epoch_trigger_windows``), with ``epoch_id``,
        ``animal``, ``session_date`` and ``epoch``.
    epochs : pd.DataFrame
        One row per epoch: ``epoch_id``, ``animal``, ``decode_sha256``,
        ``no_spike_level``, and ``n_swr``, ``n_non_swr`` and ``n_events``, the
        frame's events by SWR overlap.

    Raises
    ------
    ValueError
        As :func:`checked_reference` refuses a per-epoch measurement: not made
        over the analyses' subset from the current well visits, an analyzed
        epoch missing or extra, or another decode than the event features run's.
    """
    windows = pd.read_parquet(folder / "windows.parquet")
    records = read_output_manifest(folder)["epochs"]
    epochs = (
        windows.groupby("epoch_id", sort=False)[["animal", "decode_sha256"]]
        .first()
        .reset_index()
    )
    epochs = epochs.assign(
        no_spike_level=[records[e]["no_spike_level"] for e in epochs["epoch_id"]],
        n_swr=[records[e]["n_events"]["swr"] for e in epochs["epoch_id"]],
        n_non_swr=[records[e]["n_events"]["non_swr"] for e in epochs["epoch_id"]],
        n_events=[records[e]["n_events"]["total"] for e in epochs["epoch_id"]],
    )
    return windows, checked_reference(epochs, folder, "trigger windows")


def read_gamma_trigger_windows(folder: Path) -> pd.DataFrame:
    """The stored gamma windows around event starts
    (``scripts/event_properties/measure_gamma_trigger_windows.py``): one row
    per window and band (``triggered_analysis.event_start_windows``), with
    ``epoch_id``, ``animal``, ``session_date`` and ``epoch``.

    Raises
    ------
    ValueError
        As :func:`checked_reference` refuses a per-epoch measurement: not made
        over the analyses' subset from the current well visits, an analyzed
        epoch missing or extra, or another decode than the event features run's.
    """
    windows = pd.read_parquet(folder / "windows.parquet")
    epochs = windows.groupby("epoch_id", sort=False)["decode_sha256"].first()
    checked_reference(epochs.reset_index(), folder, "gamma trigger windows")
    return windows


EXAMPLE_SESSION_RERUN = "scripts/decoding/store_example_session.py"
"""The runner that stores an example session's recording."""


def read_example_session(
    folder: Path, epoch_id: str, decode_sha256: str
) -> tuple[pd.Series, list[np.ndarray]]:
    """An example epoch's stored ripple trace and HPC spikes
    (``scripts/decoding/store_example_session.py``).

    Parameters
    ----------
    folder : Path
        The epoch's folder, with ``ripple.parquet``, ``spikes.parquet`` and
        ``manifest.json``.
    epoch_id : str
        The epoch.
    decode_sha256 : str
        The epoch's decode frame's sha256, as the event features run records it.

    Returns
    -------
    trace : pd.Series
        The z-scored ripple consensus trace, indexed by time (s).
    spikes : list of np.ndarray
        Each HPC tetrode's spike times (s), in the stored order.

    Raises
    ------
    ValueError
        If the folder holds no stored session, or it records another epoch,
        another decode frame or another SWR source than the current one.
    """
    if not (folder / "manifest.json").exists():
        raise ValueError(f"{rel(folder)} holds no stored session")
    manifest = read_output_manifest(folder)
    what = rel(folder)
    notes = [
        f"it records the epoch {manifest.get('epoch_id')!r}, not {epoch_id!r}"
        if manifest.get("epoch_id") != epoch_id
        else None,
        f"it was made from another decode frame (sha256 "
        f"{str(manifest.get('decode_sha256'))[:12]}, current {decode_sha256[:12]})"
        if manifest.get("decode_sha256") != decode_sha256
        else None,
        f"it records the SWR source {manifest.get('swr_source')!r}, not the "
        f"current {swr_source.swr_source_name()!r}"
        if manifest.get("swr_source") != swr_source.swr_source_name()
        else None,
    ]
    notes = [note for note in notes if note is not None]
    if notes:
        raise ValueError(f"{what} is not current: {'; '.join(notes)}")
    ripple = pd.read_parquet(folder / "ripple.parquet")
    spikes = pd.read_parquet(folder / "spikes.parquet")
    trace = pd.Series(ripple["zscore"].to_numpy(), index=ripple["time"].to_numpy())
    by_tetrode = [
        spikes.loc[spikes["tetrode"] == k, "time"].to_numpy()
        for k in range(manifest["n_tetrodes"])
    ]
    return trace, by_tetrode


EXAMPLE_POSITION_COLUMNS = (
    "head_position_x",
    "head_position_y",
    "head_speed",
    "linear_position",
)
"""The position an example session's panels draw, from its decode frame."""


def read_example_decode(nwb_file_name: str, epoch_name: str) -> dict:
    """An example session for the example-event columns
    (``visualization.example_event_panels``), from stored outputs alone.

    ``data``: the position (:data:`EXAMPLE_POSITION_COLUMNS`) and the SWR
    source's SWRs (``ripple_times``) from the epoch's decode frame; the track
    graph, its edge order and spacing from the decode's environment; the ripple
    trace of the SWR source and the HPC tetrodes' spikes stored by
    ``scripts/decoding/store_example_session.py``
    (:func:`read_example_session`). ``results``, ``detector`` and ``env``: the
    decode's (``load_results``). Figure 2's ``read_session`` reads the same
    things, with its own refusal.

    Raises
    ------
    ValueError
        If the stored session is not current (:func:`read_example_session`).
    """
    identifier = format_epoch_identifier(nwb_file_name, epoch_name, "HPC")
    features = read_manifest(event_features_root() / EVENT_FEATURES_RUN)
    trace, spikes = read_example_session(
        processed_data_dir() / EXAMPLE_SESSION_DIR / identifier,
        identifier,
        features["epochs"][identifier]["decode"]["sha256"],
    )
    frame = read_dot_with_swr_source(decode_path(identifier))
    results, detector, env = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC", root=processed_data_dir()
    )
    data = {
        "position_info": frame[list(EXAMPLE_POSITION_COLUMNS)],
        "track_graph": env.track_graph,
        "linear_edge_order": env.edge_order,
        "linear_edge_spacing": env.edge_spacing,
        "zscored_ripple_consensus_trace": trace,
        "ripple_times": swr_intervals(frame),
        "spike_times": {"HPC": spikes},
    }
    return {"data": data, "results": results, "detector": detector, "env": env}
