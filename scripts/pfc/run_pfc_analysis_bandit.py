"""Batch PFC GLM analysis of neuron modulation during non-local hippocampal events.

This script runs GLM-based analysis of prefrontal cortex (mPFC/OFC) neuron firing
rate modulation around sharp-wave ripples (SWR) and non-local hippocampal events.
For each session, it fits Poisson GLM models with convolutional spline basis functions
to quantify how each neuron's firing rate changes relative to event onsets.

The analysis pipeline per session:
1. Load the session's non-local events from the canonical event table
2. Split them on that table's ``is_swr`` column (strict positive-duration
   overlap with the speed-gated ``SWR_event`` labels), so Figure 4 describes
   the same events, with the same classification, as the content and rate
   families
3. Fit GLMs for each neuron comparing models: constant, SWR, non-local, combined
4. Compute theta modulation during running
5. Save per-session neuron statistics and modulation time courses

Requires HPC decoding results to already exist (run run_decoding_bandit.py first).

Usage
-----
# Analyze all mPFC sessions
python run_pfc_analysis_bandit.py --brain_area mPFC

# Analyze specific animal
python run_pfc_analysis_bandit.py --animal wilbur --brain_area mPFC

# Analyze specific session
python run_pfc_analysis_bandit.py --animal wilbur --date 20210329 --epoch 02_r1

# Overwrite existing results
python run_pfc_analysis_bandit.py --brain_area mPFC --overwrite

# Use more Dask workers
python run_pfc_analysis_bandit.py --brain_area mPFC --n_workers 16

# Sensitivity bundle: no stop restriction, written as *_modulation_allstops.*
python run_pfc_analysis_bandit.py --brain_area mPFC --stop-cohort all_stops

# Smoke one session into a scratch directory
python run_pfc_analysis_bandit.py --max-sessions 1 --output-dir /tmp/pfc_smoke

# Sensitivity bundle at another prior, written as *_modulation_sigma1.*
python run_pfc_analysis_bandit.py --brain_area mPFC --penalty-sigma 1

# One worker of a partitioned batch: its own sessions, its own card
python run_pfc_analysis_bandit.py --brain_area mPFC --n_workers 1 \
    --gpu 2 --sessions worker0.sessions
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import warnings
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

# Pin BLAS/OpenMP threads to 1 per process to prevent thread oversubscription
# when running multiple Dask workers. Each worker runs one session serially,
# and statsmodels GLM (IRLS) is single-threaded anyway.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from pathlib import Path

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc import (
    build_session_event_sets,
    build_session_fit_mask,
    make_design_matrices,
    make_matched_control_times,
    matched_control_seed,
    run_session,
)
from continuum_swr_replay.analysis.pfc.batched_glm import (
    DEFAULT_PENALTY_SIGMA,
    fit_poisson_glms,
    implied_prior_sd,
)
from continuum_swr_replay.analysis.pfc.canonical_events import (
    IS_SWR_DEFINITION,
    canonical_event_input_hashes,
)
from continuum_swr_replay.analysis.pfc.cohort import (
    DEFAULT_STOP_TABLE_PATH,
    load_session_stop_cohort,
)
from continuum_swr_replay.analysis.pfc.glm import (
    EVENT_CLASSES,
    get_spikecount_per_time_bin,
    is_separated,
    model_penalties,
    peak_log_modulation,
    peak_modulation_se,
    prior_binds,
)
from continuum_swr_replay.analysis.pfc.gpu import configure_jax_gpu
from continuum_swr_replay.analysis.pfc.heldout import (
    assign_blocked_folds,
    control_draw_null,
    empirical_pvalue,
    heldout_deviance,
    null_draw_seeds,
)
from continuum_swr_replay.analysis.replay.cohort_selection import stop_cohort_suffix
from continuum_swr_replay.analysis.replay.fit_provenance import (
    fit_provenance_path,
    write_fit_provenance,
)
from continuum_swr_replay.data_loaders import get_epoch_info
from continuum_swr_replay.decoding_common import setup_logger
from continuum_swr_replay.parameters import (
    MAX_SPEED_THRESHOLD_CM_PER_SEC,
    PROCESSED_DATA_DIR,
    SAMPLING_FREQUENCY,
)

#: Populations the PFC GLMs may be fit on. ``at_well`` is the canonical shared
#: cohort and publishes the unsuffixed artifact names; ``all_stops`` applies no
#: stop restriction and publishes ``_allstops`` names. ``in_transit`` is not
#: offered here -- there is no power for it at the single-unit level.
STOP_COHORTS = ("at_well", "all_stops")

warnings.filterwarnings("ignore", message=".*Skipped checksum.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="datajoint")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="hdmf")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pynwb")

#: Prior standard deviation of an event-basis coefficient, on the log-rate
#: scale, that the canonical bundle is fit at. What it says about a *curve* is
#: not this number -- the coefficients are B-spline weights -- so the sidecar
#: records the implied per-lag and sustained standard deviations beside it,
#: from ``implied_prior_sd``.
CANONICAL_PENALTY_SIGMA = DEFAULT_PENALTY_SIGMA

#: The estimator and the inference each bundle is produced by, recorded in the
#: provenance sidecar and compared by the skip check: a bundle fit by maximum
#: likelihood and tested by a likelihood ratio is not this runner's bundle.
ESTIMATOR = "ridge_poisson_glm"
INFERENCE = "blocked_heldout_deviance"

#: Nested ``(full, reduced)`` pairs the held-out comparison is reported for.
#: Every pair is nested -- the reduced design's columns lie in the full
#: design's span -- so the difference of held-out deviances is a statement
#: about the added regressors alone. ``("SWR + non_local_no_SWR",
#: "non_local")`` is absent for the same reason it always was: it is only
#: nested when the session has no SWR-only events. The last two are the
#: engagement decomposition, whose baseline is the matched non-event control.
NESTED_MODEL_PAIRS = (
    ("SWR", "constant"),
    ("non_local", "constant"),
    ("non_local_no_SWR", "constant"),
    ("non_local_SWR", "constant"),
    ("SWR + non_local_no_SWR", "constant"),
    ("SWR + non_local_no_SWR", "SWR"),
    ("SWR + non_local_no_SWR", "non_local_no_SWR"),
    ("non_local_SWR + non_local_no_SWR", "constant"),
    ("non_local_SWR + non_local_no_SWR", "non_local_SWR"),
    ("non_local_SWR + non_local_no_SWR", "non_local_no_SWR"),
    ("non_local_SWR + non_local_no_SWR", "non_local"),
    ("matched_control + non_local_no_SWR", "matched_control"),
    ("matched_control", "constant"),
)

#: The engagement comparison: does non-SWR replay predict held-out spikes
#: beyond a duration-matched non-event baseline drawn from the same bouts?
ENGAGEMENT_PAIR = ("matched_control + non_local_no_SWR", "matched_control")

#: Best model per unit, by held-out deviance, within each decomposition. Only
#: comparable models compete, as they did under AIC.
BEST_MODEL_CANDIDATES = {
    "min_heldout_model": (
        "constant",
        "SWR",
        "non_local",
        "SWR + non_local_no_SWR",
    ),
    "min_heldout_model_replay_split": (
        "constant",
        "non_local",
        "non_local_SWR + non_local_no_SWR",
    ),
}

#: The five peri-event curves a bundle ships, and where each comes from: the
#: model whose fit it is read off, and which basis block of that model's
#: parameters. The block order is the *design's* own, which is what the shipped
#: curves are built from, so the maximum-likelihood peak reported beside a
#: curve is the peak of the same block.
#:
#: For the replay-split model that order is ``[constant, non_local_no_SWR,
#: non_local_SWR]`` -- the reverse of the order the model's name lists the
#: classes in. Reading it by the name is the swap that mislabelled both of that
#: model's curves in every bundle written before 2026-09-15.
CURVE_SOURCES = {
    "SWR": ("SWR + non_local_no_SWR", 0),
    "no_SWR": ("SWR + non_local_no_SWR", 1),
    "non_local_SWR": ("non_local_SWR + non_local_no_SWR", 1),
    "non_local_no_SWR": ("non_local_SWR + non_local_no_SWR", 0),
    "matched_control": ("matched_control + non_local_no_SWR", 0),
}

#: Columns the maximum-likelihood ladder wrote that this estimator cannot
#: produce. A penalised fit has no likelihood-ratio reference distribution and
#: no AIC at a stated number of parameters, so the statistics table must not
#: carry either: a reader who found a ``: pval`` column beside a ridge fit
#: would read an asymptotic test of a model that was never fit unpenalised.
RETIRED_STATISTIC_SUFFIXES = (": lr", ": pval", ": aic")
RETIRED_STATISTIC_COLUMNS = ("min_aic_model", "min_aic_model_replay_split")


def pair_name(full: str, reduced: str) -> str:
    """Column stem for one nested comparison, as the statistics table spells it."""
    return f"{full} vs. {reduced}"


def penalty_sigma_suffix(penalty_sigma: float) -> str:
    """Artifact-name suffix naming the prior a sensitivity bundle was fit at.

    The canonical prior publishes the unsuffixed name every reader hard-codes,
    so a bundle fit at another sigma can never be mistaken for it.

    Parameters
    ----------
    penalty_sigma : float
        Prior standard deviation of an event-basis coefficient, on the log-rate
        scale.

    Returns
    -------
    str
        ``""`` at the canonical prior, else ``"_sigma<value>"`` with any
        decimal point written ``p`` -- the bundle's companion arrays are named
        by suffixing the stem, so a dot in the stem would truncate them.
    """
    if float(penalty_sigma) == CANONICAL_PENALTY_SIGMA:
        return ""
    return "_sigma" + f"{float(penalty_sigma):g}".replace(".", "p")


def session_seed(stream: str, nwb_file_name: str, epoch_name: str) -> int:
    """Per-session, per-stream seed derived from the session key.

    Derived rather than counted, exactly as the matched control's own seed is:
    a session refit on its own -- to repair one bundle, say -- draws the same
    folds and the same null as the batch drew, and two sessions never share a
    draw. The stream name keeps the fold split and the null draws independent
    of each other and of the shipped control.

    Parameters
    ----------
    stream : str
        Which draw the seed is for, e.g. ``"folds"`` or ``"control_draws"``.
    nwb_file_name : str
        NWB file name, e.g. ``"wilbur20210329_.nwb"``.
    epoch_name : str
        Epoch name, e.g. ``"02_r1"``.

    Returns
    -------
    int
        The low 64 bits of ``sha256(f"{stream}|{nwb_file_name}|{epoch_name}")``.
    """
    key = f"{stream}|{nwb_file_name}|{epoch_name}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def modulation_path(
    save_folder: Path,
    epoch_identifier: str,
    stop_cohort: str,
    penalty_sigma: float = CANONICAL_PENALTY_SIGMA,
) -> Path:
    """Artifact path for one session, named for how it was cut and fit.

    The canonical at-well bundle at the canonical prior keeps the unsuffixed
    name every reader already hard-codes, so a sensitivity bundle -- another
    population, or another prior -- can never be mistaken for it.
    """
    suffix = stop_cohort_suffix(stop_cohort) + penalty_sigma_suffix(penalty_sigma)
    return save_folder / f"{epoch_identifier}_modulation{suffix}.pkl"


def results_exist(
    save_folder: Path,
    epoch_identifier: str,
    overwrite: bool,
    stop_cohort: str = "at_well",
    stop_table_sha256: str | None = None,
    event_table_sha256: str | None = None,
    data_over_time_sha256: str | None = None,
    fitter: str | None = None,
    *,
    penalty_sigma: float = CANONICAL_PENALTY_SIGMA,
    n_folds: int | None = None,
    fold_seed: int | None = None,
    n_control_draws: int | None = None,
    control_draw_seed: int | None = None,
) -> bool:
    """Check if this session's bundle already exists *for this population*.

    Distinct from ``_decoding_common.results_exist`` because the PFC
    pipeline writes a different set of files (modulation .npy arrays
    plus a single .pkl) than the decoder pipeline (results .nc, detector
    .pkl, environment .pkl). The file names alone never settle it. A bundle
    written before the matched non-event control existed is missing that
    array, so it is refit rather than kept under a name a current run wants. An at-well
    bundle never satisfies an ``all_stops`` request; a bundle that does not
    *state* its population in a provenance sidecar is not accepted as one (the
    canonical name is unsuffixed, so a bundle fit before the cohort existed
    would otherwise be kept silently by a skip); the stop table the bundle was
    cut from must be the one on disk now, because regenerating the stop table
    changes which bins and events the cohort contains; and the bundle must
    state the canonical event definition and have been cut from the event
    table and cached decode frame now on disk, because a bundle fit on a
    run-time re-detection of events, or on an earlier generation of those two
    inputs, describes different events.

    Parameters
    ----------
    stop_table_sha256 : str or None
        Hash of the stop table this run would use, as
        ``load_session_stop_cohort`` reports it. ``None`` for ``all_stops``,
        which reads no stop table -- and which therefore only matches a sidecar
        that also recorded no stop table.
    event_table_sha256, data_over_time_sha256 : str or None
        Hashes of the canonical event table and the session's cached decode
        frame this run would read. ``None`` matches no existing bundle, so a
        session whose inputs could not be hashed is refit rather than skipped.
    fitter : str or None
        The backend this run would fit with. A bundle fit with the other one is
        not this run's bundle: the two agree to the equivalence audit's
        tolerances, not bit-for-bit, so a batch that means to be one estimator
        throughout must not silently keep sessions from the other. ``None``
        (the default) matches a sidecar that names no fitter, which is how a
        bundle written before the flag existed is read.
    penalty_sigma : float
        Prior this run would fit at. It is already in the artifact name for a
        sensitivity bundle, but the canonical name is unsuffixed, so the
        sidecar is what tells a ridge bundle from the maximum-likelihood one
        that occupied the same name before.
    n_folds, fold_seed, n_control_draws, control_draw_seed : int or None
        The inference this run would produce. A bundle scored on a different
        number of folds, a different split, or a null of a different size is
        not this run's bundle: its held-out differences and its empirical
        p-values are not the ones a current batch means to ship. ``None``
        matches a sidecar that records no value, which is how a bundle written
        before the held-out inference existed is read -- and such a bundle also
        fails the estimator comparison above, so it is refit.
    """
    if overwrite:
        return False
    artifact = modulation_path(
        save_folder, epoch_identifier, stop_cohort, penalty_sigma
    )
    save_base = artifact.with_suffix("")
    required_files = [
        save_base.with_suffix(".pkl"),
        save_base.with_suffix(".SWR_modulation.npy"),
        save_base.with_suffix(".no_SWR_modulation.npy"),
        save_base.with_suffix(".non_local_SWR_modulation.npy"),
        save_base.with_suffix(".non_local_no_SWR_modulation.npy"),
        save_base.with_suffix(".matched_control_modulation.npy"),
    ]
    if not all(f.exists() for f in required_files):
        return False
    sidecar = fit_provenance_path(artifact)
    if not sidecar.exists():
        return False
    record = json.loads(sidecar.read_text())
    return (
        record.get("stop_cohort") == stop_cohort
        and record.get("stop_table_sha256") == stop_table_sha256
        and record.get("is_swr_definition") == IS_SWR_DEFINITION
        and record.get("canonical_event_table_sha256") == event_table_sha256
        and record.get("data_over_time_sha256") == data_over_time_sha256
        and record.get("fitter") == fitter
        and record.get("estimator") == ESTIMATOR
        and record.get("inference") == INFERENCE
        and record.get("penalty_sigma_log_rate") == penalty_sigma
        and record.get("n_folds") == n_folds
        and record.get("fold_seed") == fold_seed
        and record.get("n_control_draws") == n_control_draws
        and record.get("control_draw_seed") == control_draw_seed
    )


#: Half-width of the peri-event spline basis, in samples at 500 Hz. The models
#: are fit over +/-250 ms so the spline's boundary behaviour falls outside the
#: +/-200 ms the modulation curves are reported on. The scoring designs must be
#: the ones the bundle was fit on, so they are built on the same lags.
FIT_LAGS = np.arange(-125, 126, 1)

#: Half-width of the window the curves are reported on, in seconds. The models
#: are fit over a wider one so the spline's boundary behaviour falls outside it.
REPORT_HALF_WINDOW_S = 0.2


def build_scoring_context(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str,
    *,
    stop_cohort: str,
    cohort_stops: pd.DataFrame | None,
    sampling_frequency: float,
    immobility_threshold: float,
    logger: logging.Logger,
) -> dict[str, object]:
    """Rebuild the population, the designs and the counts a session was fit on.

    The held-out score and its null are fit on the same bins, the same events
    and the same designs the shipped fit used, so they are assembled here from
    the same functions and the same session-keyed control seed. Nothing is
    re-detected and nothing is re-cut: the events are the canonical table's
    rows for this session and the cohort is the stop cohort the caller
    resolved.

    Parameters
    ----------
    nwb_file_name, epoch_name, brain_area : str
        The session key and the population of units.
    stop_cohort : {"at_well", "all_stops"}
        Population the models were fit on.
    cohort_stops : pandas.DataFrame or None
        The session's kept stops; ``None`` for ``all_stops``.
    sampling_frequency : float
        Sampling frequency in Hz.
    immobility_threshold : float
        Speed threshold the fit mask and the SWR gate are cut at (cm/s).
    logger : logging.Logger

    Returns
    -------
    dict
        ``fit_mask`` (bool Series on the session's time axis), ``counts``
        (``(n_time, n_units)`` spike counts), ``designs`` (full-length design
        matrices by model name), ``spline_basis``, ``lags``, ``lag_time``,
        ``event_times``,
        ``n_basis_per_class``, and the ``bin_counts`` / ``event_counts`` /
        ``n_matched_control`` / ``matched_control_seed`` the shipped fit is
        checked against.
    """
    # Deferred, as they are in ``run_session``: these open Spyglass at import
    # time, and this module is imported by tests that never touch it.
    from continuum_swr_replay.analysis.constants import DEFAULT_NON_LOCAL_THRESHOLD
    from continuum_swr_replay.analysis.hippocampus import load_results
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        align_results_to_position,
    )
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        load_session_canonical_events,
        load_session_swr_intervals,
    )
    from continuum_swr_replay.data_loaders import load_data

    logger.info("Rebuilding the session's designs for the held-out score...")
    results, _, _ = load_results(
        nwb_file_name, epoch_name, decoding_brain_area="HPC", load_detector=False
    )
    data = load_data(nwb_file_name, epoch_name)
    results, position_info, _ = align_results_to_position(
        results, data["position_info"]
    )
    data = {**data, "position_info": position_info}

    is_immobile, fit_mask, bin_counts = build_session_fit_mask(
        position_info,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        immobility_threshold=immobility_threshold,
    )
    del is_immobile

    canonical_events, _ = load_session_canonical_events(nwb_file_name, epoch_name)
    swr_intervals, _ = load_session_swr_intervals(
        nwb_file_name,
        epoch_name,
        non_local_threshold=DEFAULT_NON_LOCAL_THRESHOLD,
        speed_threshold=immobility_threshold,
    )
    event_times, event_counts = build_session_event_sets(
        data,
        results,
        canonical_events=canonical_events,
        swr_intervals=swr_intervals,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        fit_mask=fit_mask,
        basis_half_window_s=float(np.abs(FIT_LAGS).max()) / sampling_frequency,
    )
    control_seed = matched_control_seed(nwb_file_name, epoch_name)
    event_times["matched_control"] = make_matched_control_times(
        event_times, fit_mask, np.random.default_rng(control_seed)
    ).times

    designs, spline_basis, lag_time = make_design_matrices(
        event_times,
        position_info,
        sampling_frequency=sampling_frequency,
        lags=FIT_LAGS,
    )

    time = position_info.index.to_numpy(dtype=float)
    neurons = data["spike_times"].get(brain_area) or []
    counts = (
        np.column_stack(
            [get_spikecount_per_time_bin(neuron, time) for neuron in neurons]
        ).astype(float)
        if neurons
        else np.zeros((len(time), 0))
    )

    return {
        "fit_mask": fit_mask,
        "counts": counts,
        "designs": designs,
        "spline_basis": spline_basis,
        "lags": FIT_LAGS,
        "lag_time": lag_time,
        "event_times": event_times,
        "n_basis_per_class": int(spline_basis.shape[1]),
        "bin_counts": bin_counts,
        "event_counts": event_counts,
        "n_matched_control": int(len(event_times["matched_control"])),
        "matched_control_seed": control_seed,
    }


#: Counts the rebuilt scoring context and the shipped fit must agree on. They
#: come from two independent passes over the same inputs, so a disagreement is
#: a drift between the two -- not something to average over.
CONTEXT_AGREEMENT_KEYS = (
    "n_bins_fit",
    "n_nonlocal_after",
    "n_nonlocal_swr",
    "n_nonlocal_no_swr",
    "n_ripple_after",
)


def verify_scoring_context(
    context: dict[str, object], cohort_info: dict[str, object]
) -> None:
    """Refuse a scoring context that is not the population the fit was on.

    The context is rebuilt from the same functions and the same seeds the fit
    used, so every count must match exactly. If one does not, the held-out
    score would describe a different set of bins or events than the modulation
    curves beside it, and the bundle would be internally inconsistent in a way
    no reader could see.

    Raises
    ------
    ValueError
        Naming every count that disagrees.
    """
    rebuilt = {
        **context["bin_counts"],
        **context["event_counts"],
        "n_matched_control": context["n_matched_control"],
        "matched_control_seed": context["matched_control_seed"],
    }
    keys = (*CONTEXT_AGREEMENT_KEYS, "n_matched_control", "matched_control_seed")
    disagreements = [
        f"{key}: fit {cohort_info.get(key)!r}, rebuilt {rebuilt.get(key)!r}"
        for key in keys
        if cohort_info.get(key) != rebuilt.get(key)
    ]
    if disagreements:
        raise ValueError(
            "the rebuilt scoring designs do not describe the population the "
            "models were fit on: " + "; ".join(disagreements)
        )


def separated_column_counts(design: NDArray, counts: NDArray) -> NDArray:
    """Basis columns with no spikes anywhere under their support, per unit.

    A column whose support holds no spikes is the configuration that drives an
    unpenalised coefficient to minus infinity; under the prior it is finite but
    uninformed, so the count is reported beside the fit rather than acted on.
    The intercept is not counted -- its support is every bin.

    Parameters
    ----------
    design : NDArray, shape (n_fit_bins, n_features)
        Design on the bins the models were fit on; column 0 is the intercept.
    counts : NDArray, shape (n_fit_bins, n_units)
        Spike counts on those bins.

    Returns
    -------
    NDArray of int, shape (n_units,)
    """
    support = np.asarray(design[:, 1:], dtype=float) > 0
    spikes_under_support = support.T.astype(float) @ np.asarray(counts, dtype=float)
    return (spikes_under_support == 0).sum(axis=0).astype(int)


def score_session(
    context: dict[str, object],
    *,
    modulation_curves: dict[str, NDArray],
    penalty_sigma: float,
    n_folds: int,
    fold_seed: int,
    n_control_draws: int,
    control_draw_seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score every model out of fold, and read the improvements against a null.

    Each model is refit on ``K - 1`` folds of contiguous immobility runs and
    scored by the Poisson deviance of the fold left out, under the same prior
    the shipped fit used. A nested pair's ``heldout_delta_dev`` is the reduced
    model's held-out deviance minus the full model's: positive means the added
    regressors predict spikes the fit never saw. The improvement a real event
    class buys is read against ``R`` draws of the matched non-event control --
    the same rules, no event content -- which is the only null available for a
    statistic that is neither a likelihood ratio nor independent across bins.

    Parameters
    ----------
    context : dict
        From :func:`build_scoring_context`.
    modulation_curves : dict[str, NDArray]
        The shipped penalised curves, keyed as :data:`CURVE_SOURCES` is, so the
        prior's effect on each can be read against the same curve fit without
        it.
    penalty_sigma : float
        The prior the shipped fit used, applied to every training fit.
    n_folds : int
        Folds ``K``.
    fold_seed : int
        Seed of the run-to-fold assignment, recorded in the sidecar.
    n_control_draws : int
        Draws ``R`` of the null. ``0`` skips the null entirely, and the
        empirical p-value columns are then not written at all rather than
        written empty.
    control_draw_seed : int
        Seed the ``R`` draw seeds are derived from, recorded in the sidecar.
    logger : logging.Logger

    Returns
    -------
    statistics : pandas.DataFrame
        One row per unit, in unit order, carrying the held-out deviances, the
        nested differences (total and per fold), the empirical p-values and the
        best model per decomposition.
    diagnostics : dict
        ``fold_sizes``, ``events_per_fold`` for each event class, and the
        models found non-estimable -- what the choice of ``K`` is read from.
    """
    fit_mask = context["fit_mask"]
    designs: dict[str, NDArray] = context["designs"]
    fit_rows = np.asarray(fit_mask, dtype=bool)
    counts = np.asarray(context["counts"], dtype=float)
    fit_counts = counts[fit_rows]
    n_units = counts.shape[1]

    folds = assign_blocked_folds(fit_mask, n_folds, fold_seed)
    penalties = model_penalties(designs, penalty_sigma)

    # A design whose event class has no events leaves an all-zero basis block,
    # which is rank-deficient against the intercept. Such a model is not an
    # estimated zero effect, it is no estimate at all, so every statistic that
    # reads it is NaN -- the same rule the modulation curves are held to.
    non_estimable = {
        name
        for name, design in designs.items()
        if np.linalg.matrix_rank(design[fit_rows]) < design.shape[1]
    }
    if non_estimable:
        logger.warning(
            "Non-estimable designs (rank-deficient on the fit bins): %s",
            sorted(non_estimable),
        )

    deviance: dict[str, NDArray] = {}
    per_fold: dict[str, NDArray] = {}
    for name, design in designs.items():
        if name in non_estimable:
            deviance[name] = np.full(n_units, np.nan)
            per_fold[name] = np.full((n_folds, n_units), np.nan)
            continue
        scored = heldout_deviance(
            design[fit_rows], fit_counts, penalties.get(name), folds
        )
        deviance[name] = scored.total
        per_fold[name] = scored.per_fold
    logger.info("Scored %d models on %d folds", len(designs), n_folds)

    statistics: dict[str, object] = {}
    for name in designs:
        statistics[f"{name}: heldout_deviance"] = deviance[name]
        statistics[f"{name}: separated_columns"] = separated_column_counts(
            designs[name][fit_rows], fit_counts
        )

    for full, reduced in NESTED_MODEL_PAIRS:
        stem = pair_name(full, reduced)
        statistics[f"{stem}: heldout_delta_dev"] = deviance[reduced] - deviance[full]
        difference = per_fold[reduced] - per_fold[full]
        statistics[f"{stem}: heldout_delta_dev_folds"] = list(difference.T)

    for column, candidates in BEST_MODEL_CANDIDATES.items():
        usable = [name for name in candidates if name not in non_estimable]
        if usable:
            stacked = np.column_stack([deviance[name] for name in usable])
            best = np.argmin(np.where(np.isnan(stacked), np.inf, stacked), axis=1)
            statistics[column] = [
                None if not np.isfinite(row).any() else usable[index]
                for row, index in zip(stacked, best, strict=True)
            ]
        else:
            statistics[column] = [None] * n_units

    diagnostics: dict[str, object] = {
        "fold_sizes": np.bincount(folds, minlength=n_folds).tolist(),
        "events_per_fold": events_per_fold(context, folds),
        "non_estimable_models": sorted(non_estimable),
    }

    statistics.update(
        maximum_likelihood_comparison(
            designs, fit_rows, fit_counts, context, modulation_curves, logger
        )
    )

    if n_control_draws > 0:
        seeds = null_draw_seeds(control_draw_seed, n_control_draws)
        logger.info(
            "Drawing the null: %d matched-control draws per class on %d folds...",
            n_control_draws,
            n_folds,
        )
        null = control_draw_null(
            context["event_times"],
            fit_mask,
            counts,
            folds,
            rng_seeds=seeds,
            spline_basis=context["spline_basis"],
            lags=context["lags"],
            penalty_sigma=penalty_sigma,
            baseline_control=context["event_times"]["matched_control"],
        )
        # Each class is read against a null drawn to its own onset count and
        # its own durations: a regressor built from 30 windows buys less than
        # one built from 300 whatever either contains, so one shared null would
        # hold classes of different size to the same reference.
        for event_class in EVENT_CLASSES:
            stem = pair_name(event_class, "constant")
            statistics[f"{event_class}: p_heldout"] = empirical_pvalue(
                statistics[f"{stem}: heldout_delta_dev"],
                null.improvement[event_class],
            )
        engagement_stem = pair_name(*ENGAGEMENT_PAIR)
        statistics[f"{engagement_stem}: p_heldout"] = empirical_pvalue(
            statistics[f"{engagement_stem}: heldout_delta_dev"], null.engagement
        )
        diagnostics["null_windows"] = {
            name: {
                "n_events": int(null.n_events[name]),
                "n_placed_median": float(np.median(null.n_placed[name])),
                "n_placed_min": int(np.min(null.n_placed[name])),
            }
            for name in null.n_placed
        }

    return pd.DataFrame(statistics), diagnostics


def maximum_likelihood_comparison(
    designs: dict[str, NDArray],
    fit_rows: NDArray,
    fit_counts: NDArray,
    context: dict[str, object],
    modulation_curves: dict[str, NDArray],
    logger: logging.Logger,
) -> dict[str, NDArray]:
    """What the same curve looks like without the prior, and whether it binds.

    The shipped curve is the penalised one; the question a prior has to answer
    is where it did the work. So each curve-bearing model is fit unpenalised as
    well, and the two peaks are differenced against the estimation error the
    *unpenalised* fit itself reports -- the standard error of the kernel at its
    own peak lag, off-diagonal terms included, which a diagonal ``bse`` cannot
    give. A unit whose unpenalised fit ran away has no peak to difference
    against: it is flagged instead, and left out of the binding count.

    Parameters
    ----------
    designs : dict[str, NDArray]
        Full-length designs by model name.
    fit_rows : NDArray of bool, shape (n_time,)
        The bins the models were fit on.
    fit_counts : NDArray, shape (n_fit_bins, n_units)
        Spike counts on those bins.
    context : dict
        From :func:`build_scoring_context`; its ``spline_basis`` and
        ``lag_time`` say what a coefficient block means.
    modulation_curves : dict[str, NDArray]
        The shipped penalised curves, ``(n_report_lags, n_units)`` each, keyed
        as :data:`CURVE_SOURCES` is.
    logger : logging.Logger

    Returns
    -------
    dict[str, NDArray]
        Per curve: the penalised peak, the unpenalised peak and its standard
        error, whether the unpenalised fit is separated, and whether the prior
        binds.
    """
    spline_basis = context["spline_basis"]
    lag_time = context["lag_time"]
    n_basis = spline_basis.shape[1]
    wanted = sorted({model for model, _ in CURVE_SOURCES.values()})
    logger.info(
        "Fitting %d models unpenalised, to read the prior against", len(wanted)
    )
    unpenalised = {
        model: fit_poisson_glms(designs[model][fit_rows], fit_counts, penalty=None)
        for model in wanted
    }

    columns: dict[str, NDArray] = {}
    for curve, (model, block) in CURVE_SOURCES.items():
        start = 1 + block * n_basis
        params = np.asarray(unpenalised[model].params)[:, start : start + n_basis]
        covariance = np.asarray(unpenalised[model].covariance)[
            :, start : start + n_basis, start : start + n_basis
        ]
        mle = peak_log_modulation(params, spline_basis, lag_time)
        mle_se = peak_modulation_se(covariance, spline_basis, mle.index)
        penalised_peak = curve_peaks(modulation_curves[curve])
        columns[f"{curve}: peak_log_modulation"] = penalised_peak
        columns[f"{curve}: peak_log_modulation_mle"] = mle.value
        columns[f"{curve}: peak_log_modulation_mle_se"] = mle_se
        columns[f"{curve}: mle_separated"] = is_separated(params)
        columns[f"{curve}: prior_binds"] = prior_binds(
            penalised_peak, mle.value, mle_se
        )
    return columns


def curve_peaks(curves: NDArray) -> NDArray:
    """Peak log-modulation per unit: the curve's value where ``|curve|`` is largest.

    The shipped curve is ``spline_basis @ params`` over the reported window, so
    this is what a coefficient-level peak helper returns for the coefficients
    that produced it -- read off the curve the bundle actually ships.

    Parameters
    ----------
    curves : NDArray, shape (n_report_lags, n_units)

    Returns
    -------
    NDArray, shape (n_units,)
        ``nan`` for a unit whose curve is not finite.
    """
    curves = np.asarray(curves, dtype=float)
    peaks = np.full(curves.shape[1], np.nan)
    usable = np.isfinite(curves).all(axis=0)
    if usable.any():
        block = curves[:, usable]
        peaks[usable] = block[np.abs(block).argmax(axis=0), np.arange(block.shape[1])]
    return peaks


def events_per_fold(
    context: dict[str, object], folds: NDArray
) -> dict[str, list[int]]:
    """Event onsets falling in each fold, per event class.

    The rarest class's count is what a choice of ``K`` has to live with: a fold
    holding a handful of events of the class under test cannot say much about
    a model built on it.
    """
    fit_mask = context["fit_mask"]
    time = np.asarray(fit_mask.index, dtype=float)
    fit_rows = np.asarray(fit_mask, dtype=bool)
    fold_of_bin = np.full(len(time), -1, dtype=np.int64)
    fold_of_bin[fit_rows] = folds
    n_folds = int(folds.max()) + 1
    per_class: dict[str, list[int]] = {}
    for name, times in context["event_times"].items():
        if not len(times):
            per_class[name] = [0] * n_folds
            continue
        index = np.searchsorted(time, times["start_time"].to_numpy(dtype=float))
        index = np.clip(index, 0, len(time) - 1)
        assigned = fold_of_bin[index]
        per_class[name] = [
            int((assigned == fold).sum()) for fold in range(n_folds)
        ]
    return per_class


def drop_retired_statistics(neuron_stats: pd.DataFrame) -> pd.DataFrame:
    """Remove the maximum-likelihood ladder's columns from a statistics table.

    The per-unit fits are penalised and the inference is out of fold, so the
    likelihood-ratio statistics, their asymptotic p-values and the AICs are not
    quantities this estimator produces. They are dropped before the bundle is
    written rather than shipped alongside the held-out columns, where a reader
    could pick either.
    """
    retired = [
        column
        for column in neuron_stats.columns
        if column.endswith(RETIRED_STATISTIC_SUFFIXES)
        or column in RETIRED_STATISTIC_COLUMNS
    ]
    return neuron_stats.drop(columns=retired)


def run_analysis(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "mPFC",
    sampling_frequency: int = SAMPLING_FREQUENCY,
    immobility_threshold: float = MAX_SPEED_THRESHOLD_CM_PER_SEC,
    log_directory: str | Path = "logs",
    overwrite: bool = False,
    git_hash: str = "",
    stop_cohort: str = "at_well",
    output_dir: Path = PROCESSED_DATA_DIR,
    fitter: str = "jax",
    fit_diagnostics: bool = False,
    gpu: int | None = None,
    allow_cpu: bool = False,
    penalty_sigma: float = CANONICAL_PENALTY_SIGMA,
    n_folds: int = 5,
    n_control_draws: int = 40,
    fold_seed: int | None = None,
    control_draw_seed: int | None = None,
) -> None:
    """Run PFC GLM analysis for a single session.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name (e.g., "wilbur20210329_.nwb")
    epoch_name : str
        Epoch name (e.g., "02_r1")
    brain_area : str
        Brain area to analyze ("mPFC" or "OFC")
    sampling_frequency : int
        Sampling frequency in Hz
    immobility_threshold : float
        Speed threshold for immobility classification (cm/s)
    log_directory : str | Path
        Directory for log files
    overwrite : bool
        Whether to overwrite existing results
    git_hash : str
        Pre-computed git hash for logging
    stop_cohort : {"at_well", "all_stops"}
        Population to fit: the well-anchored stops the hippocampal ladders are
        read on, or no stop restriction at all.
    output_dir : Path
        Directory the artifacts and their provenance sidecar are written to.
    fitter : {"jax", "statsmodels"}
        How the GLMs are fit. ``"jax"`` fits every neuron of the session
        against each design at once; ``"statsmodels"`` keeps the original
        per-neuron IRLS. The sidecar records which was used.
    fit_diagnostics : bool
        Also write ``<artifact>.fit_diagnostics.npz`` -- per-neuron, per-model
        parameters and fit criteria, for the fitter equivalence audit. Forces
        the serial neuron path.
    gpu : int, optional
        Fit on this GPU instead of the emptiest free one. The card is still
        claimed against sibling workers, but its occupancy is not checked.
    allow_cpu : bool
        Permit the batched fitter to run on CPU when no GPU is free. Off by
        default: the CPU path is many times slower, and a batch that quietly
        fell back to it would not be noticed until it overran.
    penalty_sigma : float
        Prior standard deviation of an event-basis coefficient, on the log-rate
        scale. A value other than the canonical one writes a suffixed
        sensitivity bundle and skips the null, whose draws would then describe
        a prior the canonical bundle is not fit under.
    n_folds : int
        Folds the held-out deviance is summed over. Whole contiguous
        immobility runs are assigned to folds, so no fold is predicted from
        bins 2 ms either side of it.
    n_control_draws : int
        Draws of the matched non-event control the observed held-out
        improvements are read against. ``0`` skips the null.
    fold_seed, control_draw_seed : int, optional
        Seeds of the fold split and of the null draws. Derived from the session
        key when omitted, so a session refit on its own reproduces the batch's
        split and null exactly.

    Notes
    -----
    With ``fitter="jax"`` the device is chosen before anything is loaded, and a
    device that cannot be chosen ends the session rather than degrading it: no
    free card (and no ``allow_cpu``) raises ``NoFreeGPUError``, and a process
    that had already initialised a JAX backend on every card raises
    ``WrongJAXDeviceError``. Both propagate out of this function -- they are
    raised before the ``try`` that turns a per-session failure into a log line,
    because they are a fact about the host rather than about the session, and a
    batch that hits them should stop rather than work through 70 more sessions
    failing the same way.
    """
    epoch_identifier = (
        f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{brain_area}"
    )
    save_file = modulation_path(
        output_dir, epoch_identifier, stop_cohort, penalty_sigma
    )
    if fold_seed is None:
        fold_seed = session_seed("folds", nwb_file_name, epoch_name)
    if control_draw_seed is None:
        control_draw_seed = session_seed("control_draws", nwb_file_name, epoch_name)
    # A bundle fit under another prior is a sensitivity bundle: it is written
    # for its modulation curves, and a null drawn under a prior the canonical
    # bundle is not fit under would not be a null for anything shipped.
    if float(penalty_sigma) != CANONICAL_PENALTY_SIGMA:
        n_control_draws = 0

    logger = setup_logger(
        name_logfile=epoch_identifier,
        path_logfile=Path(log_directory) / f"{epoch_identifier}.log",
    )

    # The cohort is resolved before the skip check, not after: the check needs
    # the stop table's hash to tell a bundle cut from the current table from one
    # cut from a previous generation of it. Reading and hashing the table is
    # seconds against a multi-hour fit.
    if stop_cohort == "all_stops":
        cohort_stops, cohort_report = None, {"cohort": stop_cohort}
    else:
        try:
            cohort_stops, cohort_report = load_session_stop_cohort(
                nwb_file_name, epoch_name, cohort=stop_cohort
            )
        except Exception as e:
            # One session missing from the stop table must not abort the batch;
            # the log is the per-session failure record, as it is for a failed
            # fit below.
            logger.exception(f"No stop cohort for {epoch_identifier}: {e}")
            return

    # The event definition is resolved before the skip check for the same
    # reason the cohort is: a bundle cut from a previous generation of the
    # canonical event table or of the cached decode frame describes different
    # events, so it is not this run's bundle.
    event_inputs = canonical_event_input_hashes(nwb_file_name, epoch_name)

    if results_exist(
        output_dir,
        epoch_identifier,
        overwrite,
        stop_cohort,
        cohort_report.get("stop_table_sha256"),
        event_inputs["canonical_event_table_sha256"],
        event_inputs["data_over_time_sha256"],
        fitter,
        penalty_sigma=penalty_sigma,
        n_folds=n_folds,
        fold_seed=fold_seed,
        n_control_draws=n_control_draws,
        control_draw_seed=control_draw_seed,
    ):
        logger.info(f"Results already exist for {epoch_identifier}. Skipping...")
        return

    logger.info(" START ".center(50, "#"))
    logger.info(f"Git Hash: {git_hash}")
    logger.info(f"brain_area: {brain_area}")
    logger.info(f"sampling_frequency: {sampling_frequency}")
    logger.info(f"immobility_threshold: {immobility_threshold}")
    logger.info(f"stop_cohort: {stop_cohort}")
    logger.info(f"fitter: {fitter}")
    logger.info(f"penalty_sigma_log_rate: {penalty_sigma}")
    logger.info(f"n_folds: {n_folds} (fold_seed {fold_seed})")
    logger.info(
        f"n_control_draws: {n_control_draws} "
        f"(control_draw_seed {control_draw_seed})"
    )

    diagnostics: dict | None = {} if fit_diagnostics else None

    # Before JAX initialises a backend -- later than the import, and checked
    # afterwards -- pick the card this process will fit on and bound what it
    # takes of it. Each Dask worker runs this for itself; the claim keeps two
    # of them off the same card, and a worker handed a second session gets the
    # card it is already on rather than a second one.
    gpu_record: dict[str, object] = {}
    if fitter == "jax":
        gpu_record = configure_jax_gpu(gpu=gpu, allow_cpu=allow_cpu)
        logger.info(
            "JAX device: %s%s",
            gpu_record["jax_backend"],
            (
                f" (GPU {gpu_record['gpu_index']}, {gpu_record['gpu_name']}, "
                f"{gpu_record['gpu_uuid']})"
                if gpu_record["gpu_index"] is not None
                else ""
            ),
        )

    try:
        logger.info(f"Processing {epoch_identifier}...")
        if cohort_stops is not None:
            logger.info(
                "Stop cohort %s: %d/%d stops (%s)",
                stop_cohort,
                cohort_report["n_stops_after"],
                cohort_report["n_stops_before"],
                cohort_report["rule"],
            )
        (
            neuron_stats,
            SWR_modulation,
            no_SWR_modulation,
            non_local_SWR_modulation,
            non_local_no_SWR_modulation,
            matched_control_modulation,
            cohort_info,
        ) = run_session(
            nwb_file_name,
            epoch_name,
            brain_area=brain_area,
            sampling_frequency=sampling_frequency,
            immobility_threshold=immobility_threshold,
            run_parallel=False,
            stop_cohort=stop_cohort,
            cohort_stops=cohort_stops,
            fitter=fitter,
            fit_diagnostics_out=diagnostics,
            penalty_sigma=penalty_sigma,
        )

        logger.info(f"Analyzed {len(neuron_stats)} neurons")

        # The statistics table's inference is the held-out score, so the
        # likelihood-ratio columns the maximum-likelihood ladder wrote are
        # dropped and the held-out ones take their place.
        neuron_stats = drop_retired_statistics(neuron_stats)
        n_basis_per_class: int | None = None
        implied_prior: dict[str, float] = {}
        scoring_diagnostics: dict[str, object] = {}
        if len(neuron_stats):
            context = build_scoring_context(
                nwb_file_name,
                epoch_name,
                brain_area,
                stop_cohort=stop_cohort,
                cohort_stops=cohort_stops,
                sampling_frequency=sampling_frequency,
                immobility_threshold=immobility_threshold,
                logger=logger,
            )
            verify_scoring_context(context, cohort_info)
            n_basis_per_class = context["n_basis_per_class"]
            # What the prior says about a curve, not about a coefficient: the
            # basis is a partition of unity, so the prior on the kernel at one
            # lag is wider than sigma and the prior on a modulation held across
            # the window is narrower. Both are recorded, over the reported
            # window rather than the wider one the models are fit on.
            reported = (
                np.abs(context["lag_time"]) <= REPORT_HALF_WINDOW_S + 1e-12
            )
            implied_prior = implied_prior_sd(
                penalty_sigma, context["spline_basis"], reported
            )
            heldout_stats, scoring_diagnostics = score_session(
                context,
                modulation_curves={
                    "SWR": SWR_modulation,
                    "no_SWR": no_SWR_modulation,
                    "non_local_SWR": non_local_SWR_modulation,
                    "non_local_no_SWR": non_local_no_SWR_modulation,
                    "matched_control": matched_control_modulation,
                },
                penalty_sigma=penalty_sigma,
                n_folds=n_folds,
                fold_seed=fold_seed,
                n_control_draws=n_control_draws,
                control_draw_seed=control_draw_seed,
                logger=logger,
            )
            if len(heldout_stats) != len(neuron_stats):
                raise ValueError(
                    f"the held-out score describes {len(heldout_stats)} units "
                    f"and the fit {len(neuron_stats)}; they are the same "
                    "neurons in the same order or they are not comparable"
                )
            neuron_stats = pd.concat(
                [neuron_stats, heldout_stats.set_index(neuron_stats.index)], axis=1
            )
            logger.info(
                "Fold sizes: %s; events per fold: %s",
                scoring_diagnostics["fold_sizes"],
                scoring_diagnostics["events_per_fold"],
            )

        logger.info("Saving results...")

        save_file.parent.mkdir(parents=True, exist_ok=True)
        neuron_stats.to_pickle(save_file)
        np.save(
            save_file.with_suffix(".SWR_modulation.npy"),
            SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".no_SWR_modulation.npy"),
            no_SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".non_local_SWR_modulation.npy"),
            non_local_SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".non_local_no_SWR_modulation.npy"),
            non_local_no_SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".matched_control_modulation.npy"),
            matched_control_modulation,
        )

        if diagnostics:
            # Per-neuron, per-model parameters and fit criteria. Not part of
            # the bundle any reader loads; written only when a run asks for it,
            # so the fitter audit can compare the two backends at the fit level
            # rather than only through the statistics they feed.
            diagnostics_file = save_file.with_suffix(".fit_diagnostics.npz")
            np.savez_compressed(
                diagnostics_file,
                **{
                    f"{model_name}|{quantity}": values
                    for model_name, record in diagnostics.items()
                    for quantity, values in record.items()
                },
            )
            logger.info(f"Wrote fit diagnostics to {diagnostics_file}")

        # Every artifact states the population it was fit on: which cohort,
        # which stop table, and how many bins and events survived it.
        sidecar = write_fit_provenance(
            save_file,
            {"stop_table": DEFAULT_STOP_TABLE_PATH} if cohort_stops is not None else {},
            estimator=ESTIMATOR,
            cli=sys.argv,
            extra={
                "stop_cohort": stop_cohort,
                # The estimator and the inference, in full: the prior the
                # coefficients were fit under, the blocked split the models
                # were scored out of, and the null the improvements are read
                # against. A bundle that does not state these cannot be told
                # from the maximum-likelihood one that held the same name.
                "inference": INFERENCE,
                "penalty_sigma_log_rate": float(penalty_sigma),
                "n_folds": int(n_folds),
                "fold_seed": int(fold_seed),
                "n_control_draws": int(n_control_draws),
                "control_draw_seed": int(control_draw_seed),
                "n_basis_per_class": n_basis_per_class,
                "implied_prior_sd": implied_prior,
                "fold_sizes": scoring_diagnostics.get("fold_sizes"),
                "events_per_fold": scoring_diagnostics.get("events_per_fold"),
                "non_estimable_models": scoring_diagnostics.get(
                    "non_estimable_models"
                ),
                # How many windows each class's null draws actually placed,
                # against the real onset count they were matched to: a draw
                # that placed far fewer is a thinner regressor, and its
                # improvement is not the reference the observed one needs.
                "null_windows": scoring_diagnostics.get("null_windows"),
                "stop_table_sha256": cohort_report.get("stop_table_sha256"),
                "stop_cohort_report": cohort_report,
                "cohort_counts": cohort_info,
                "brain_area": brain_area,
                "nwb_file_name": nwb_file_name,
                "epoch_name": epoch_name,
                # How the models were fit: which backend, the convergence
                # tolerance both backends stop on, and -- for the batched
                # fitter -- the device and JAX version, since a GPU and a CPU
                # run are not bit-identical.
                "fitter": cohort_info.get("fitter"),
                "fitter_tol": cohort_info.get("fitter_tol"),
                "jax_backend": cohort_info.get("jax_backend"),
                "jax_version": cohort_info.get("jax_version"),
                # Which card the batched fits ran on, so a bundle names the
                # device as well as the estimator.
                "gpu_index": gpu_record.get("gpu_index"),
                "gpu_uuid": gpu_record.get("gpu_uuid"),
                "gpu_name": gpu_record.get("gpu_name"),
                # How many of the shipped fits never settled. A unit x model
                # that never settled is non-estimable for that neuron alone --
                # every statistic that reads it is NaN -- so these counts are
                # the only record of how much was dropped and from where.
                "n_model_fits": cohort_info.get("n_model_fits"),
                "n_nonconverged_model_fits": cohort_info.get(
                    "n_nonconverged_model_fits"
                ),
                "n_neurons_with_nonconverged_model": cohort_info.get(
                    "n_neurons_with_nonconverged_model"
                ),
                # Which events the fit describes, from the same block the skip
                # check above compares against.
                **event_inputs,
                # The cohort-restricted event counts, so the sidecar states the
                # split without a reader recomputing it.
                "n_events_total": cohort_info.get("n_nonlocal_after"),
                "n_events_swr": cohort_info.get("n_nonlocal_swr"),
                "n_events_non_swr": cohort_info.get("n_nonlocal_no_swr"),
                # The matched non-event control the engagement test is read
                # against: how it was drawn, and how many of the session's
                # events got one.
                "matched_control_seed": cohort_info.get("matched_control_seed"),
                "n_matched_control": cohort_info.get("n_matched_control"),
                "n_matched_control_unplaced": cohort_info.get(
                    "n_matched_control_unplaced"
                ),
                # ...and split by why: an onset outside every fitted bout is a
                # cohort fact, a saturated bout is an event-density one.
                "n_matched_control_onset_outside_fitted_bout": cohort_info.get(
                    "n_matched_control_onset_outside_fitted_bout"
                ),
                "n_matched_control_bout_saturated": cohort_info.get(
                    "n_matched_control_bout_saturated"
                ),
            },
        )

        logger.info(f"Saved results to {save_file}")
        logger.info(f"Wrote provenance to {sidecar}")
        logger.info("Done!")
    except Exception as e:
        logger.exception(f"Error processing {epoch_identifier}: {e}")


def get_command_line_arguments() -> Namespace:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--local_scratch_dir", type=str, default="/tmp/dask_scratch",
        help="Local scratch directory for Dask workers (default: /tmp/dask_scratch)",
    )
    parser.add_argument(
        "--n_workers", type=int, default=4,
        help="Number of Dask workers (default: 4)",
    )
    parser.add_argument(
        "--animal", type=str, default=None,
        help="Filter to specific animal (e.g., wilbur)",
    )
    parser.add_argument(
        "--date", type=int, default=None,
        help="Filter to specific date (e.g., 20210329)",
    )
    parser.add_argument(
        "--epoch", type=str, default=None,
        help="Filter to specific epoch (e.g., 02_r1)",
    )
    parser.add_argument(
        "--brain_area", type=str, default="mPFC",
        choices=["mPFC", "OFC"],
        help="Brain area to analyze: mPFC or OFC (default: mPFC)",
    )
    parser.add_argument(
        "--log_directory", type=str, default="logs",
        help="Directory for log files (default: logs)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--stop-cohort",
        type=str,
        default="at_well",
        choices=list(STOP_COHORTS),
        help=(
            "Population to fit (default: at_well). 'at_well' keeps only the "
            "immobility bins and events inside the well-anchored stops the "
            "hippocampal rate and content ladders are read on, and publishes "
            "the canonical artifact names; 'all_stops' applies no stop "
            "restriction and publishes '_allstops' names."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=(
            "Directory for the modulation artifacts and their provenance "
            f"sidecars (default: {PROCESSED_DATA_DIR})"
        ),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Process at most this many sessions (smoke runs)",
    )
    parser.add_argument(
        "--fitter",
        type=str,
        default="jax",
        choices=["jax"],
        help=(
            "How the GLMs are fit (default and only choice: jax, the batched "
            "IRLS that fits every neuron of a session against each design at "
            "once). The per-neuron statsmodels IRLS has no penalised path, so "
            "a batch cannot be launched on it; it survives in the library as "
            "the control the fitter equivalence tests are read against."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=(
            "Fit on this GPU instead of the emptiest free one. The card is "
            "still claimed against sibling workers, but its occupancy is not "
            "checked."
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Permit the batched fitter to run on CPU when no GPU is free. Off "
            "by default: the CPU path is many times slower, and a batch that "
            "quietly fell back to it would not be noticed until it overran."
        ),
    )
    parser.add_argument(
        "--fit-diagnostics",
        action="store_true",
        help=(
            "Also write <artifact>.fit_diagnostics.npz: per-neuron, per-model "
            "parameters and fit criteria, for the fitter equivalence audit. "
            "Forces the serial neuron path."
        ),
    )
    parser.add_argument(
        "--penalty-sigma",
        type=float,
        default=CANONICAL_PENALTY_SIGMA,
        help=(
            "Prior standard deviation of an event-basis coefficient, on the "
            f"log-rate scale (default: {CANONICAL_PENALTY_SIGMA}). A "
            "coefficient of this basis is a log-rate modulation near its knot, "
            "so the value reads as a rate multiplier. Another value writes a "
            "'_sigma<value>' sensitivity bundle and skips the null, which "
            "would otherwise describe a prior nothing is shipped under."
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help=(
            "Folds the held-out deviance is summed over (default: 5). Whole "
            "contiguous immobility runs are assigned to folds, so no fold is "
            "predicted from the bins 2 ms either side of it."
        ),
    )
    parser.add_argument(
        "--n-control-draws",
        type=int,
        default=40,
        help=(
            "Draws of the matched non-event control that make the null each "
            "held-out improvement is read against (default: 40). 0 skips the "
            "null, and no empirical p-value is written."
        ),
    )
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=None,
        help=(
            "Seed of the run-to-fold assignment. Derived from the session key "
            "when omitted, so a session refit on its own reproduces the "
            "batch's split."
        ),
    )
    parser.add_argument(
        "--control-draw-seed",
        type=int,
        default=None,
        help=(
            "Seed the null's draw seeds are derived from. Derived from the "
            "session key when omitted."
        ),
    )
    parser.add_argument(
        "--sessions",
        type=Path,
        default=None,
        help=(
            "Fit only the sessions named in this file: one "
            "'<nwb_file_name> <epoch_name>' per line (comma or whitespace "
            "separated), '#' comments and blank lines ignored. For splitting "
            "a corpus across workers that cannot be named by animal or date. "
            "A named session that does not exist is an error, so a mistyped "
            "partition cannot quietly fit nothing."
        ),
    )

    args = parser.parse_args()
    if args.penalty_sigma <= 0:
        parser.error(
            f"--penalty-sigma {args.penalty_sigma} is not positive; it is a "
            "prior standard deviation on the log-rate scale."
        )
    if args.n_folds < 2:
        parser.error(
            f"--n-folds {args.n_folds} leaves nothing held out; the held-out "
            "deviance needs at least two folds."
        )
    if args.n_control_draws < 0:
        parser.error(
            f"--n-control-draws {args.n_control_draws} is negative; 0 skips "
            "the null."
        )
    if args.gpu is not None and args.n_workers > 1:
        # --gpu names one card and every worker would be handed the same one:
        # they would all claim it, all preallocate a share of it, and every
        # sidecar would name a card most of them are not really alone on. One
        # worker per card is the layout that works -- separate processes, each
        # with its own --gpu and its own session list.
        parser.error(
            f"--gpu {args.gpu} names one card but --n_workers is "
            f"{args.n_workers}: every worker would be sent to that card. Run "
            "one process per card (--n_workers 1 --gpu N --sessions <list>), "
            "or drop --gpu and let each worker claim a free card of its own."
        )
    return args


def read_session_list(path: Path) -> list[tuple[str, str]]:
    """Session keys from a partition file, in file order.

    Parameters
    ----------
    path : Path
        One ``<nwb_file_name> <epoch_name>`` per line, comma or whitespace
        separated. Blank lines and ``#`` comments are ignored.

    Returns
    -------
    list of (str, str)
        ``(nwb_file_name, epoch_name)`` pairs, duplicates removed, first
        occurrence kept.

    Raises
    ------
    ValueError
        On a line that is not two fields, naming the line so the file can be
        fixed rather than guessed at.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [field for field in line.replace(",", " ").split() if field]
        if len(fields) != 2:
            raise ValueError(
                f"{path}:{number}: expected '<nwb_file_name> <epoch_name>', "
                f"got {raw.strip()!r}"
            )
        key = (fields[0], fields[1])
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    if not pairs:
        raise ValueError(f"{path} names no sessions")
    return pairs


def _session_keys(epoch_info: pd.DataFrame) -> list[tuple[str, str]]:
    """``(nwb_file_name, epoch_name)`` for every row, in row order."""
    return list(
        zip(
            epoch_info.index.get_level_values("nwb_file_name"),
            epoch_info.index.get_level_values("interval_list_name"),
            strict=True,
        )
    )


def _filter_epoch_info(epoch_info: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    """Filter epoch_info based on command line arguments.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        All available epochs
    args : Namespace
        Parsed command line arguments

    Returns
    -------
    pd.DataFrame
        Filtered epochs
    """
    is_epoch = np.ones(len(epoch_info), dtype=bool)

    if getattr(args, "sessions", None) is not None:
        wanted = read_session_list(args.sessions)
        keys = _session_keys(epoch_info)
        missing = [key for key in wanted if key not in set(keys)]
        if missing:
            raise ValueError(
                f"{args.sessions} names {len(missing)} session(s) that do not "
                "exist: "
                + ", ".join(f"{name} {epoch}" for name, epoch in missing[:10])
                + (", ..." if len(missing) > 10 else "")
            )
        requested = set(wanted)
        is_epoch = np.logical_and(
            np.array([key in requested for key in keys], dtype=bool), is_epoch
        )

    if args.animal is not None:
        is_epoch = np.logical_and(epoch_info.animal == args.animal, is_epoch)

    if args.date is not None:
        is_epoch = np.logical_and(epoch_info.date == args.date, is_epoch)

    if args.epoch is not None:
        is_epoch = np.logical_and(
            epoch_info.index.get_level_values("interval_list_name") == args.epoch,
            is_epoch,
        )

    epoch_info = epoch_info.loc[is_epoch, :]

    if isinstance(epoch_info, pd.Series):
        epoch_info = epoch_info.to_frame().T

    return epoch_info


def _filter_to_sorted_sessions(epoch_info: pd.DataFrame) -> pd.DataFrame:
    """Filter to sessions that have curated spike sorting results.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        Available epochs

    Returns
    -------
    pd.DataFrame
        Epochs with curated spike sorting
    """
    try:
        from spyglass.spikesorting.v0 import CuratedSpikeSorting

        restriction = {
            "preproc_params_name": "default",
            "sort_interval_name": (
                "sleeps_runs_noPrePostTrialTimes raw data valid times"
            ),
            "sorter_params_name": (
                "franklab_probe_ctx_30KHz_115rad_new_mountainsort2"
            ),
            "team_name": "ac_em_xs",
            "sorter": "mountainsort4",
        }

        sorted_nwb_file_names = np.unique(
            (CuratedSpikeSorting & restriction).fetch("nwb_file_name")
        )
        epoch_info = epoch_info[
            epoch_info.index.get_level_values("nwb_file_name").isin(
                sorted_nwb_file_names
            )
        ]
    except ImportError:
        logging.getLogger(__name__).warning(
            "Spyglass not available. Running on all sessions without filtering "
            "by curated spike sorting."
        )

    return epoch_info


def main() -> None:
    """Main entry point for batch PFC GLM analysis."""
    args = get_command_line_arguments()

    Path(args.local_scratch_dir).mkdir(exist_ok=True, parents=True)
    Path(args.log_directory).mkdir(exist_ok=True, parents=True)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.rstrip()

    epoch_info = get_epoch_info()
    epoch_info = _filter_epoch_info(epoch_info, args)
    epoch_info = _filter_to_sorted_sessions(epoch_info)
    if args.max_sessions is not None:
        epoch_info = epoch_info.iloc[: args.max_sessions]

    print(
        f"Processing {len(epoch_info)} sessions for {args.brain_area} "
        f"(stop cohort: {args.stop_cohort}, fitter: {args.fitter}, "
        f"penalty sigma: {args.penalty_sigma}, folds: {args.n_folds}, "
        f"control draws: {args.n_control_draws})..."
    )

    if args.n_workers <= 1 or len(epoch_info) <= 1:
        for epoch in epoch_info.itertuples():
            run_analysis(
                nwb_file_name=epoch.Index[0],
                epoch_name=epoch.Index[1],
                brain_area=args.brain_area,
                sampling_frequency=SAMPLING_FREQUENCY,
                immobility_threshold=MAX_SPEED_THRESHOLD_CM_PER_SEC,
                log_directory=args.log_directory,
                overwrite=args.overwrite,
                git_hash=git_hash,
                stop_cohort=args.stop_cohort,
                output_dir=args.output_dir,
                fitter=args.fitter,
                fit_diagnostics=args.fit_diagnostics,
                gpu=args.gpu,
                allow_cpu=args.allow_cpu,
                penalty_sigma=args.penalty_sigma,
                n_folds=args.n_folds,
                n_control_draws=args.n_control_draws,
                fold_seed=args.fold_seed,
                control_draw_seed=args.control_draw_seed,
            )
    else:
        cluster = LocalCluster(
            local_directory=args.local_scratch_dir,
            n_workers=args.n_workers,
            threads_per_worker=1,
            processes=True,
            memory_limit="400GB",
        )

        with Client(cluster):
            results = [
                dask.delayed(run_analysis)(
                    nwb_file_name=epoch.Index[0],
                    epoch_name=epoch.Index[1],
                    brain_area=args.brain_area,
                    sampling_frequency=SAMPLING_FREQUENCY,
                    immobility_threshold=MAX_SPEED_THRESHOLD_CM_PER_SEC,
                    log_directory=args.log_directory,
                    overwrite=args.overwrite,
                    git_hash=git_hash,
                    stop_cohort=args.stop_cohort,
                    output_dir=args.output_dir,
                    fitter=args.fitter,
                    fit_diagnostics=args.fit_diagnostics,
                    gpu=args.gpu,
                    allow_cpu=args.allow_cpu,
                    penalty_sigma=args.penalty_sigma,
                    n_folds=args.n_folds,
                    n_control_draws=args.n_control_draws,
                    fold_seed=args.fold_seed,
                    control_draw_seed=args.control_draw_seed,
                )
                for epoch in epoch_info.itertuples()
            ]

            dask.compute(*results)


if __name__ == "__main__":
    sys.exit(main())
