"""Batch PFC GLM analysis of neuron modulation during non-local hippocampal events.

This script runs GLM-based analysis of prefrontal cortex (mPFC/OFC) neuron firing
rate modulation around sharp-wave ripples (SWR) and non-local hippocampal events.
For each session, it fits Poisson GLM models with convolutional spline basis functions
to quantify how each neuron's firing rate changes relative to event onsets.

The analysis pipeline per session:
1. Load the session's non-local events from the canonical event table
2. Split them on that table's ``is_swr`` column (strict positive-duration
   overlap with the SWR source's speed-gated ``SWR_event`` labels, the direct
   Kay inventory of ``analysis/events/swr_source.py``), so Figure 4
   describes the same events, with the same classification, as the content
   and rate families
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
import sys
import warnings
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
from typing import Any

from continuum_swr_replay.runs.threads import pin_blas_threads

# Pin BLAS/OpenMP threads to 1 per process to prevent thread oversubscription
# when running multiple Dask workers. Each worker runs one session serially,
# and statsmodels GLM (IRLS) is single-threaded anyway.
pin_blas_threads("NUMEXPR_NUM_THREADS")

import dask  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dask.distributed import Client, LocalCluster  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from continuum_swr_replay.analysis.events.cohort_selection import (  # noqa: E402
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.pfc import run_session  # noqa: E402
from continuum_swr_replay.analysis.pfc.batched_glm import (  # noqa: E402
    DEFAULT_PENALTY_SIGMA,
    fit_poisson_glms,
    implied_prior_sd,
    prepare_heldout_response,
    score_poisson_designs_heldout,
)
from continuum_swr_replay.analysis.pfc.canonical_events import (  # noqa: E402
    IS_SWR_DEFINITION,
    canonical_event_input_hashes,
)
from continuum_swr_replay.analysis.pfc.cohort import (  # noqa: E402
    DEFAULT_STOP_TABLE_PATH,
    load_session_stop_cohort,
)
from continuum_swr_replay.analysis.pfc.event_sets import EVENT_CLASSES  # noqa: E402
from continuum_swr_replay.analysis.pfc.fit import model_penalties  # noqa: E402
from continuum_swr_replay.analysis.pfc.gpu import configure_jax_gpu  # noqa: E402
from continuum_swr_replay.analysis.pfc.heldout import (  # noqa: E402
    BEST_MODEL_CANDIDATES,
    CONTROL_NULL_EVENT_POLICY,
    DEFAULT_CONTROL_DRAWS,
    DEFAULT_NULL_DESIGN_BATCH_SIZE,
    EMPIRICAL_PVALUE_METHOD,
    ENGAGEMENT_PAIR,
    MIN_USABLE_NULL_DRAWS,
    assign_blocked_folds,
    control_draw_null,
    empirical_pvalue,
    null_draw_seeds,
    required_usable_null_draws,
)
from continuum_swr_replay.analysis.pfc.load_population import (  # noqa: E402
    penalty_sigma_suffix,
)
from continuum_swr_replay.analysis.pfc.matched_control import (  # noqa: E402
    MATCHED_CONTROL_PLACEMENT_METHOD,
)
from continuum_swr_replay.analysis.pfc.modulation import (  # noqa: E402
    CURVE_SOURCES,
    NESTED_MODEL_PAIRS,
    REPORT_HALF_WINDOW_S,
    curve_coefficients,
    is_separated,
    peak_log_modulation,
    peak_modulation_se,
    prior_binds,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (  # noqa: E402
    fit_provenance_path,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (  # noqa: E402
    git_hash as loaded_commit,
)
from continuum_swr_replay.data_loaders.epoch_register import (  # noqa: E402
    analysed_epochs,
)
from continuum_swr_replay.decoding.identifiers import (  # noqa: E402
    format_epoch_identifier,
)
from continuum_swr_replay.parameters import (  # noqa: E402
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    PROCESSED_DATA_DIR,
    SAMPLING_FREQUENCY,
)
from continuum_swr_replay.runs.batch import (  # noqa: E402
    raise_open_file_limit,
    release_nwb_files,
    setup_logger,
)
from continuum_swr_replay.runs.provenance import produced_by  # noqa: E402
from continuum_swr_replay.tables.io import write_atomically  # noqa: E402

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
    swr_interval_source: str | None = None,
    swr_events_sha256: str | None = None,
) -> bool:
    """Check if this session's bundle already exists *for this population*.

    Distinct from ``decoding.files.results_exist`` because the PFC
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
    swr_interval_source, swr_events_sha256 : str or None
        The SWR source (``swr_source_name()``) and the hash of its events file this
        run would read the SWR intervals from: a bundle fit on the stored
        ripples, or on another inventory run, describes other SWRs.
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

    The sidecar must also name the current matched-control placement method.
    The former disjoint-duration-window policy can leave every null draw
    thinner than the real class on dense bouts, so those bundles do not carry
    the same empirical test even when their p-value column names match.
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
    expected_usable_draws = (
        None if n_control_draws is None else required_usable_null_draws(n_control_draws)
    )
    return (
        record.get("stop_cohort") == stop_cohort
        and record.get("stop_table_sha256") == stop_table_sha256
        and record.get("is_swr_definition") == IS_SWR_DEFINITION
        and record.get("canonical_event_table_sha256") == event_table_sha256
        and record.get("data_over_time_sha256") == data_over_time_sha256
        and record.get("swr_interval_source") == swr_interval_source
        and record.get("swr_events_sha256") == swr_events_sha256
        and record.get("fitter") == fitter
        and record.get("estimator") == ESTIMATOR
        and record.get("inference") == INFERENCE
        and record.get("penalty_sigma_log_rate") == penalty_sigma
        and record.get("n_folds") == n_folds
        and record.get("fold_seed") == fold_seed
        and record.get("n_control_draws") == n_control_draws
        and record.get("control_draw_seed") == control_draw_seed
        and record.get("empirical_pvalue_method") == EMPIRICAL_PVALUE_METHOD
        and record.get("control_null_event_policy") == CONTROL_NULL_EVENT_POLICY
        and record.get("min_usable_null_draws") == expected_usable_draws
        and (
            n_control_draws in (None, 0)
            or isinstance(record.get("null_usable_draws"), dict)
        )
        and record.get("matched_control_placement_method")
        == MATCHED_CONTROL_PLACEMENT_METHOD
    )


def separated_column_counts(design: NDArray, counts: NDArray) -> NDArray:
    """Basis columns with no spikes anywhere under their support, per unit.

    A column whose support holds no spikes is the configuration that drives an
    unpenalised coefficient to minus infinity; under the prior it is finite but
    uninformed, so the count is reported beside the fit rather than acted on.
    The intercept is not counted -- its support is every bin.

    The bin set is every bin the models were fit on -- the whole fit mask, not
    one fold of it -- so the count describes the shipped full-data fit.

    Parameters
    ----------
    design : NDArray, shape (n_fit_bins, n_features)
        Design restricted to the fit mask; column 0 is the intercept.
    counts : NDArray, shape (n_fit_bins, n_units)
        Spike counts on those same bins.

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

    Each model is refit on ``K - 1`` folds of support-separated immobility
    blocks and
    scored by the Poisson deviance of the fold left out, under the same prior
    the shipped fit used. A nested pair's ``heldout_delta_dev`` is the reduced
    model's held-out deviance minus the full model's: positive means the added
    regressors predict spikes the fit never saw. The improvement a real event
    class buys is read against ``R`` draws of the matched non-event control --
    the same rules, no event content -- which is the only null available for a
    statistic that is neither a likelihood ratio nor independent across bins.
    If an event has no legal within-bout match, the empirical test uses one
    fixed matchable subset on both its observed and null sides; the all-event
    held-out difference remains separately reported.

    The model sets it reads -- :data:`ENGAGEMENT_PAIR`,
    :data:`BEST_MODEL_CANDIDATES` and :data:`CURVE_SOURCES` -- are defined in
    ``continuum_swr_replay.analysis.pfc.heldout`` and ``.modulation``, beside
    the folds, the null and the p-value it is built from.

    Parameters
    ----------
    context : dict
        The session's fit material, as ``run_session`` hands it back through
        ``scoring_out``.
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

    support_radius_bins = int(np.max(np.abs(np.asarray(context["lags"]))))
    folds = assign_blocked_folds(
        fit_mask,
        n_folds,
        fold_seed,
        support_radius_bins=support_radius_bins,
    )
    penalties = model_penalties(designs, penalty_sigma)
    scoring_response = prepare_heldout_response(fit_counts, folds)

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
    for name in non_estimable:
        deviance[name] = np.full(n_units, np.nan)
        per_fold[name] = np.full((n_folds, n_units), np.nan)
    widths = sorted(
        {
            design.shape[1]
            for name, design in designs.items()
            if name not in non_estimable
        }
    )
    for width in widths:
        names = [
            name
            for name, design in designs.items()
            if name not in non_estimable and design.shape[1] == width
        ]
        design_batch = np.stack([designs[name][fit_rows] for name in names])
        penalty_batch = np.stack(
            [penalties.get(name, np.zeros(width)) for name in names]
        )
        scored = score_poisson_designs_heldout(
            design_batch,
            scoring_response,
            penalty_batch,
            design_batch_size=DEFAULT_NULL_DESIGN_BATCH_SIZE,
        )
        for index, name in enumerate(names):
            deviance[name] = scored.total[index]
            per_fold[name] = scored.per_fold[index]
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
        "fold_support_radius_bins": support_radius_bins,
    }

    statistics.update(
        maximum_likelihood_comparison(
            designs, fit_rows, fit_counts, context, modulation_curves, logger
        )
    )

    if n_control_draws > 0:
        required_draws = required_usable_null_draws(n_control_draws)
        seeds = null_draw_seeds(control_draw_seed, n_control_draws)
        logger.info(
            "Drawing the null: %d matched-control draws per class on %d folds...",
            n_control_draws,
            n_folds,
        )
        null = control_draw_null(
            context["event_times"],
            fit_mask,
            None,
            None,
            rng_seeds=seeds,
            subset_seed=control_draw_seed,
            spline_basis=context["spline_basis"],
            lags=context["lags"],
            penalty_sigma=penalty_sigma,
            baseline_control=context["event_times"]["matched_control"],
            model_deviance=deviance,
            response=scoring_response,
            avoid=context["artifact_dropped"],
        )
        # Each class is read against a null drawn to its own fixed matchable
        # onset subset and durations: a regressor built from 30 windows buys
        # less than one built from 300 whatever either contains, so one shared
        # null would hold classes of different size to the same reference.
        usable_draws: dict[str, NDArray] = {}
        for event_class in EVENT_CLASSES:
            stem = pair_name(event_class, "constant")
            statistics[f"{stem}: matched_null_observed_delta_dev"] = (
                null.observed_improvement[event_class]
            )
            usable = np.isfinite(null.improvement[event_class]).sum(axis=0)
            usable_draws[event_class] = usable
            statistics[f"{event_class}: n_usable_null_draws"] = usable
            statistics[f"{event_class}: p_heldout"] = empirical_pvalue(
                null.observed_improvement[event_class],
                null.improvement[event_class],
                min_usable_draws=required_draws,
            )
        engagement_stem = pair_name(*ENGAGEMENT_PAIR)
        statistics[f"{engagement_stem}: matched_null_observed_delta_dev"] = (
            null.observed_engagement
        )
        usable_draws["engagement"] = np.isfinite(null.engagement).sum(axis=0)
        statistics[f"{engagement_stem}: n_usable_null_draws"] = usable_draws[
            "engagement"
        ]
        statistics[f"{engagement_stem}: p_heldout"] = empirical_pvalue(
            null.observed_engagement,
            null.engagement,
            min_usable_draws=required_draws,
        )
        diagnostics["null_usable_draws"] = {
            name: values.astype(int).tolist() for name, values in usable_draws.items()
        }
        diagnostics["null_windows"] = {
            name: {
                "n_events_total": int(null.n_events[name]),
                "n_events_tested": int(null.n_tested[name]),
                "n_events_excluded_unmatchable": int(
                    null.n_events[name] - null.n_tested[name]
                ),
                "tested_source_indices": (
                    null.tested_source_indices[name].astype(int).tolist()
                ),
                "n_placed_median": float(np.median(null.n_placed[name])),
                "n_placed_min": int(np.min(null.n_placed[name])),
                "n_exact_draws": int(
                    np.sum(null.n_placed[name] == null.n_tested[name])
                ),
                "n_required_draws": required_draws,
            }
            for name in null.n_placed
        }
        # The engagement null's draws are independent of the shipped control
        # they are entered beside, so they may land on it -- which two real
        # event classes never do. The cost of drawing the null by the control's
        # own rules is measured rather than assumed away.
        overlap = np.asarray(null.engagement_overlap, dtype=float)
        diagnostics["engagement_null_overlap"] = (
            {
                "median": float(np.median(overlap)),
                "max": float(np.max(overlap)),
            }
            if overlap.size
            else None
        )

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
        The session's fit material from ``run_session``; its ``spline_basis``
        and ``lag_time`` say what a coefficient block means.
    modulation_curves : dict[str, NDArray]
        The shipped penalised curves, ``(n_report_lags, n_units)`` each, keyed
        as :data:`CURVE_SOURCES` is.
    logger : logging.Logger

    Returns
    -------
    dict[str, NDArray]
        Per curve: the penalised peak, the unpenalised peak and its standard
        error, whether the unpenalised fit converged or separated, and whether
        the prior binds.
    """
    spline_basis = context["spline_basis"]
    lag_time = context["lag_time"]
    n_basis = spline_basis.shape[1]
    wanted = sorted({model for model, _ in CURVE_SOURCES.values()})
    logger.info("Fitting %d models unpenalised, to read the prior against", len(wanted))
    unpenalised = {
        model: fit_poisson_glms(designs[model][fit_rows], fit_counts, penalty=None)
        for model in wanted
    }

    columns: dict[str, NDArray] = {}
    for curve in CURVE_SOURCES:
        model, block = curve_coefficients(curve, n_basis)
        result = unpenalised[model]
        raw_params = np.asarray(result.params)
        model_separated = is_separated(raw_params[:, 1:])
        converged = np.asarray(result.converged, dtype=bool)
        usable = converged & ~model_separated
        params = raw_params[:, block].copy()
        covariance = np.asarray(result.covariance)[:, block, block].copy()
        params[~usable] = np.nan
        covariance[~usable] = np.nan
        mle = peak_log_modulation(params, spline_basis, lag_time)
        mle_se = peak_modulation_se(covariance, spline_basis, mle.index)
        penalised_peak = curve_peaks(modulation_curves[curve])
        columns[f"{curve}: peak_log_modulation"] = penalised_peak
        columns[f"{curve}: peak_log_modulation_mle"] = mle.value
        columns[f"{curve}: peak_log_modulation_mle_se"] = mle_se
        columns[f"{curve}: mle_converged"] = converged
        columns[f"{curve}: mle_separated"] = model_separated
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


def events_per_fold(context: dict[str, object], folds: NDArray) -> dict[str, list[int]]:
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
        per_class[name] = [int((assigned == fold).sum()) for fold in range(n_folds)]
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


def session_identifier(nwb_file_name: str, epoch_name: str, brain_area: str) -> str:
    """The session's name in its artifacts and its log: ``wilbur20210329_02_r1_mPFC``."""
    return format_epoch_identifier(nwb_file_name, epoch_name, brain_area)


def run_analysis(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "mPFC",
    sampling_frequency: int = SAMPLING_FREQUENCY,
    immobility_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    log_directory: str | Path = "logs",
    overwrite: bool = False,
    git_hash: str | None = None,
    stop_cohort: str = "at_well",
    output_dir: Path = PROCESSED_DATA_DIR,
    fitter: str = "jax",
    fit_diagnostics: bool = False,
    gpu: int | None = None,
    allow_cpu: bool = False,
    penalty_sigma: float = CANONICAL_PENALTY_SIGMA,
    n_folds: int = 5,
    n_control_draws: int = DEFAULT_CONTROL_DRAWS,
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
    git_hash : str, optional
        The commit ``main`` loaded, logged and recorded in the sidecar. A dask
        worker imports the code later than ``main`` does, after the checkout
        may have moved on, so ``main`` passes its own. Omitted, it is the
        commit this process loaded.
    stop_cohort : {"at_well", "all_stops"}
        Population to fit: the well-anchored stops the hippocampal ladders are
        read on, or no stop restriction at all.
    output_dir : Path
        Directory the artifacts and their provenance sidecar are written to.
    fitter : {"jax"}
        How the GLMs are fit: the batched backend, which fits every neuron of
        the session against each design at once and is the only one with a
        penalised path. The sidecar records it.
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
        Folds the held-out deviance is summed over. Whole immobility runs whose
        event-kernel supports overlap are assigned together, so one event
        cannot enter both train and test.
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
    epoch_identifier = session_identifier(nwb_file_name, epoch_name, brain_area)
    save_file = modulation_path(
        output_dir, epoch_identifier, stop_cohort, penalty_sigma
    )
    # A dask worker imports the library when it starts, which can be hours
    # after main did; if the checkout moved on in between, it runs newer code
    # than the commit main passed in.
    try:
        worker_commit = loaded_commit()
    except RuntimeError:
        # A worker that could not read git must not fail a session main has a
        # commit for; run on its own, the session has no commit to record.
        if git_hash is None:
            raise
        worker_commit = "unknown"
    if git_hash is None:
        git_hash = worker_commit
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
        swr_interval_source=event_inputs["swr_interval_source"],
        swr_events_sha256=event_inputs["swr_events_sha256"],
    ):
        logger.info(f"Results already exist for {epoch_identifier}. Skipping...")
        return

    logger.info(" START ".center(50, "#"))
    logger.info(f"Git Hash: {git_hash}")
    if worker_commit != git_hash:
        logger.warning(
            f"this process loaded {worker_commit}, not {git_hash}: the "
            "checkout changed after main started; the sidecar records both"
        )
    logger.info(f"brain_area: {brain_area}")
    logger.info(f"sampling_frequency: {sampling_frequency}")
    logger.info(f"immobility_threshold: {immobility_threshold}")
    logger.info(f"stop_cohort: {stop_cohort}")
    logger.info(f"fitter: {fitter}")
    logger.info(f"penalty_sigma_log_rate: {penalty_sigma}")
    logger.info(f"n_folds: {n_folds} (fold_seed {fold_seed})")
    logger.info(
        f"n_control_draws: {n_control_draws} (control_draw_seed {control_draw_seed})"
    )

    diagnostics: dict | None = {} if fit_diagnostics else None
    # Filled by ``run_session`` with the bins, counts, designs and events it
    # fit on, so the out-of-fold score is taken on exactly that material
    # rather than on a second assembly of it.
    scoring_context: dict[str, object] = {}

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
            scoring_out=scoring_context,
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
            context = scoring_context
            n_basis_per_class = int(context["spline_basis"].shape[1])
            # What the prior says about a curve, not about a coefficient: the
            # basis is a partition of unity, so the prior on the kernel at one
            # lag is wider than sigma and the prior on a modulation held across
            # the window is narrower. Both are recorded, over the reported
            # window rather than the wider one the models are fit on.
            reported = np.abs(context["lag_time"]) <= REPORT_HALF_WINDOW_S + 1e-12
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
            commit=git_hash,
            extra={
                **(
                    {"worker_git_hash": worker_commit}
                    if worker_commit != git_hash
                    else {}
                ),
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
                "empirical_pvalue_method": EMPIRICAL_PVALUE_METHOD,
                "control_null_event_policy": CONTROL_NULL_EVENT_POLICY,
                "min_usable_null_draws": required_usable_null_draws(n_control_draws),
                "matched_control_placement_method": (MATCHED_CONTROL_PLACEMENT_METHOD),
                "n_basis_per_class": n_basis_per_class,
                "implied_prior_sd": implied_prior,
                "fold_sizes": scoring_diagnostics.get("fold_sizes"),
                "fold_support_radius_bins": scoring_diagnostics.get(
                    "fold_support_radius_bins"
                ),
                "events_per_fold": scoring_diagnostics.get("events_per_fold"),
                "non_estimable_models": scoring_diagnostics.get("non_estimable_models"),
                # Which real onsets enter the fixed matched test, and how many
                # windows each null draw actually placed against that tested
                # count. A thinner draw is not the reference the subset-
                # matched observed statistic needs.
                "null_windows": scoring_diagnostics.get("null_windows"),
                "null_usable_draws": scoring_diagnostics.get("null_usable_draws"),
                "engagement_null_overlap": scoring_diagnostics.get(
                    "engagement_null_overlap"
                ),
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


def run_analysis_then_release(
    nwb_file_name: str, epoch_name: str, brain_area: str = "mPFC", **kwargs: Any
) -> None:
    """``run_analysis``, then close the NWB files Spyglass opened for it.

    ``main``'s workers run one session at a time (``threads_per_worker=1``), and
    its serial path runs them one after another, so no other session in the
    process can still be reading those files (``release_nwb_files``). The release
    wraps ``run_analysis`` rather than living in it because that returns early
    and turns most failures into a log line: here it runs whether the session
    is fit, skipped, failed or raised. How many files were closed goes to the
    session's log.
    """
    try:
        run_analysis(nwb_file_name, epoch_name, brain_area=brain_area, **kwargs)
    finally:
        logging.getLogger(
            session_identifier(nwb_file_name, epoch_name, brain_area)
        ).info("closed %d NWB files Spyglass held open", release_nwb_files())


def get_command_line_arguments() -> Namespace:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--local_scratch_dir",
        type=str,
        default="/tmp/dask_scratch",
        help="Local scratch directory for Dask workers (default: /tmp/dask_scratch)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of Dask workers (default: 4)",
    )
    parser.add_argument(
        "--animal",
        type=str,
        default=None,
        help="Filter to specific animal (e.g., wilbur)",
    )
    parser.add_argument(
        "--date",
        type=int,
        default=None,
        help="Filter to specific date (e.g., 20210329)",
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default=None,
        help="Filter to specific epoch (e.g., 02_r1)",
    )
    parser.add_argument(
        "--brain_area",
        type=str,
        default="mPFC",
        choices=["mPFC", "OFC"],
        help="Brain area to analyze: mPFC or OFC (default: mPFC)",
    )
    parser.add_argument(
        "--log_directory",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
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
            "immobility runs whose event-kernel supports overlap are assigned "
            "together, so one event cannot enter both train and test."
        ),
    )
    parser.add_argument(
        "--n-control-draws",
        type=int,
        default=DEFAULT_CONTROL_DRAWS,
        help=(
            "Draws of the matched non-event control that make the null each "
            f"held-out improvement is read against (default: "
            f"{DEFAULT_CONTROL_DRAWS}). 0 skips the null, and no empirical "
            "p-value is written."
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
            f"--n-control-draws {args.n_control_draws} is negative; 0 skips the null."
        )
    if 0 < args.n_control_draws < MIN_USABLE_NULL_DRAWS:
        parser.error(
            f"--n-control-draws {args.n_control_draws} is below the "
            f"{MIN_USABLE_NULL_DRAWS}-draw reporting floor; use 0 to skip "
            "the null or request at least that many draws."
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
                f"{args.sessions} names {len(missing)} session(s) that are not "
                "analysed epochs of the epoch register: "
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


NO_SORTED_UNITS_REASON = (
    "the session has no curated cortical spike sorting (CuratedSpikeSorting, "
    "mountainsort4), a PFC-level condition, not an epoch exclusion"
)
"""Why an analysed epoch is outside the PFC cohort."""


def write_cohort_record(
    path: Path,
    *,
    n_analysed: int,
    pfc_cohort: pd.DataFrame,
    ran: pd.DataFrame,
    left_out: pd.Index,
    git_hash: str,
) -> None:
    """Record the PFC cohort beside the bundles: what ran, and why the rest of
    the register's analysed epochs did not.

    Every partition of a run computes the same cohort, so every field but
    ``ran`` is the same whichever partition writes it last. ``ran`` is per
    writer: each invocation overwrites the file when it starts, so a batch
    split across parallel partitions (``--sessions``) leaves the last
    partition's share only. The union is the bundles on disk.
    """
    record = {
        "produced_by": produced_by(__file__),
        "git_hash": git_hash,
        "n_analysed": n_analysed,
        "n_pfc_cohort": len(pfc_cohort),
        "ran": [f"{name} {epoch}" for name, epoch in ran.index],
        "left_out": {
            f"{name} {epoch}": NO_SORTED_UNITS_REASON for name, epoch in left_out
        },
    }
    text = json.dumps(record, indent=2) + "\n"
    write_atomically(path, lambda temporary: temporary.write_text(text))


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
            "sorter_params_name": ("franklab_probe_ctx_30KHz_115rad_new_mountainsort2"),
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
    # before any cluster starts: its worker processes inherit this limit
    old_soft, new_soft = raise_open_file_limit()
    print(f"open-file soft limit {old_soft} -> {new_soft}", flush=True)

    Path(args.local_scratch_dir).mkdir(exist_ok=True, parents=True)
    Path(args.log_directory).mkdir(exist_ok=True, parents=True)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Read when this script imported its code, before any worker starts.
    git_hash = loaded_commit()

    # The PFC cohort: the register's analysed epochs, whose session has sorted
    # units. An analysed epoch without them is left out here, with that reason,
    # which pfc_cohort_<area>.json records beside the bundles.
    analysed = analysed_epochs()
    pfc_cohort = _filter_to_sorted_sessions(analysed)
    unsorted = analysed.index.difference(pfc_cohort.index, sort=False)
    if len(unsorted):
        print(
            f"{len(unsorted)} analysed epoch(s) left out at the PFC level, "
            f"{NO_SORTED_UNITS_REASON}: "
            + ", ".join(f"{name} {epoch}" for name, epoch in unsorted)
        )
    epoch_info = _filter_epoch_info(analysed, args)
    epoch_info = epoch_info[epoch_info.index.isin(pfc_cohort.index)]
    if args.max_sessions is not None:
        epoch_info = epoch_info.iloc[: args.max_sessions]
    write_cohort_record(
        args.output_dir / f"pfc_cohort_{args.brain_area}.json",
        n_analysed=len(analysed),
        pfc_cohort=pfc_cohort,
        ran=epoch_info,
        left_out=unsorted,
        git_hash=git_hash,
    )

    print(
        f"Processing {len(epoch_info)} sessions for {args.brain_area} "
        f"(stop cohort: {args.stop_cohort}, fitter: {args.fitter}, "
        f"penalty sigma: {args.penalty_sigma}, folds: {args.n_folds}, "
        f"control draws: {args.n_control_draws})..."
    )

    if args.n_workers <= 1 or len(epoch_info) <= 1:
        for epoch in epoch_info.itertuples():
            run_analysis_then_release(
                nwb_file_name=epoch.Index[0],
                epoch_name=epoch.Index[1],
                brain_area=args.brain_area,
                sampling_frequency=SAMPLING_FREQUENCY,
                immobility_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
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
                dask.delayed(run_analysis_then_release)(
                    nwb_file_name=epoch.Index[0],
                    epoch_name=epoch.Index[1],
                    brain_area=args.brain_area,
                    sampling_frequency=SAMPLING_FREQUENCY,
                    immobility_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
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
