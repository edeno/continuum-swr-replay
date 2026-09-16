"""Generalized Linear Model analysis for PFC neural activity.

Fits Poisson GLMs relating single-unit spiking activity to hippocampal
replay events (SWR-coincident and non-SWR) during immobility, and tests
theta phase modulation during running.
"""

import hashlib
import logging
import warnings
from collections.abc import Mapping
from typing import Any, NamedTuple

import dask
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from numpy.typing import NDArray
from patsy import dmatrix

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.replay.cohort_selection import (
    events_in_stops,
    stop_interval_mask,
)

# ``load_data`` and the hippocampus helpers below open a Spyglass /
# DataJoint connection at import time. We defer those imports into
# ``run_session`` (the only function that uses them) so that the pure
# helpers in this module — ``get_spikecount_per_time_bin``,
# ``process_neuron``, the design-matrix builders — stay offline-
# importable for unit tests on machines without VPN.

logger = logging.getLogger(__name__)


def get_spikecount_per_time_bin(spike_times: NDArray, time: NDArray) -> NDArray:
    """Bin spike times into time bins.

    Parameters
    ----------
    spike_times : NDArray, shape (n_spikes,)
        Spike times in seconds.
    time : NDArray, shape (n_time_bins,)
        Time bin starts (left edges) in seconds — i.e. each ``time[i]``
        is the start of bin ``i``, not its center. Matches the
        sample-bin-start convention used by the data loader's
        multiunit pipeline. Spike ``s`` is assigned to bin ``i`` where
        ``time[i] <= s < time[i+1]``; spikes at exactly ``time[-1]``
        fall in the last bin.

    Returns
    -------
    NDArray, shape (n_time_bins,)
        Spike counts per time bin.

    """
    spike_times = spike_times[
        np.logical_and(spike_times >= time[0], spike_times <= time[-1])
    ]
    # ``np.digitize(s, time[1:])`` produces indices ``0..N-1`` against
    # a ``minlength=N`` bincount, so all output bins are reachable.
    # The previous ``time[1:-1]`` form maxed out at index ``N-2``,
    # leaving the final bin unreachable: e.g. with
    # ``time=[0,1,2,3]``, spikes at 2.5 and 3.0 both landed in bin 2
    # and bin 3 stayed at zero.
    return np.bincount(
        np.digitize(spike_times, time[1:]),
        minlength=time.shape[0],
    )


def reindex_by(
    df: pd.Series | pd.DataFrame,
    index: pd.Index,
    tolerance: float | None = None,
) -> pd.Series | pd.DataFrame:
    """Reindex pandas object to new index using nearest neighbor.

    Uses pd.merge_asof for efficient O(n log n) nearest-neighbor matching.
    The original pd.reindex(method='nearest') can use O(n*m) memory when
    source and target indices have different sampling rates.

    Parameters
    ----------
    df : pd.Series | pd.DataFrame
        Data to reindex. Index must be sorted.
    index : pd.Index
        Target index to align to. Must be sorted.
    tolerance : float | None, optional
        Maximum distance for nearest neighbor match (in index units, typically
        seconds for time indices). If None, uses unlimited matching which may
        match to distant points. Recommended to set this to the sampling period
        (e.g., 0.002 seconds for 500 Hz data) to prevent spurious matches.

    Returns
    -------
    pd.Series | pd.DataFrame
        Reindexed data with values from nearest time point within tolerance.
        Values outside tolerance will be NaN.

    """
    if tolerance is None:
        warnings.warn(
            "reindex_by called with tolerance=None; nearest neighbor matching will "
            "match to arbitrarily distant points. Consider setting tolerance to the "
            "sampling period (e.g., 0.002 for 500 Hz data) to prevent spurious matches.",
            UserWarning,
            stacklevel=2,
        )

    target_df = pd.DataFrame(index=index)
    source_df = df.copy() if isinstance(df, pd.DataFrame) else df.to_frame()

    # Ensure index dtypes match for merge_asof (empty Series may have int index)
    if len(source_df) == 0:
        source_df.index = source_df.index.astype(target_df.index.dtype)

    merged = pd.merge_asof(
        target_df,
        source_df,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=tolerance,
    )

    if isinstance(df, pd.Series):
        return merged.iloc[:, 0].rename(df.name)
    return merged


def likelihood_ratio_test(
    deviance_full: float,
    deviance_restricted: float,
    degrees_of_freedom_full: float,
    degrees_of_freedom_restricted: float,
) -> tuple[float, float]:
    """Compare goodness of fit of nested models.

    Parameters
    ----------
    deviance_full : float
        Deviance of the more complicated model
    deviance_restricted : float
        Deviance of the simpler model
    degrees_of_freedom_full : float
        Degrees of freedom of the more complicated model
    degrees_of_freedom_restricted : float
        Degrees of freedom of the simpler model

    Returns
    -------
    likelihood_ratio : float
    p_value : float

    """
    degrees_of_freedom = degrees_of_freedom_restricted - degrees_of_freedom_full
    likelihood_ratio = deviance_restricted - deviance_full
    p_value = scipy.stats.chi2.sf(likelihood_ratio, df=degrees_of_freedom)

    return likelihood_ratio, p_value


def _lr_test(full_model: Any, restricted_model: Any) -> tuple[float, float]:
    return likelihood_ratio_test(
        full_model.deviance,
        restricted_model.deviance,
        full_model.df_resid,
        restricted_model.df_resid,
    )


#: The three-column theta design, as a patsy formula.
THETA_FORMULA = "1 + np.cos(theta_phase) + np.sin(theta_phase)"


def build_theta_design(
    theta_phase: pd.Series,
    is_immobile: pd.Series,
    sampling_frequency: float = 500.0,
) -> tuple[NDArray, NDArray]:
    """The running-bin mask and theta design the theta test is fit on.

    Split out of :func:`test_theta_modulation` so a session can build the one
    design every unit shares once, and fit every unit against it in a single
    batched call.

    Parameters
    ----------
    theta_phase : pd.Series, shape (n_lfp_samples,)
        Theta phase timeseries (radians), on its own clock.
    is_immobile : pd.Series, shape (n_time,)
        Boolean series indicating immobility at each time bin.
    sampling_frequency : float
        Sampling frequency in Hz, used to set the reindex tolerance.

    Returns
    -------
    joint_mask : NDArray of bool, shape (n_time,)
        Running AND theta phase available.
    design_matrix : NDArray, shape (n_joint, 3)
        Intercept, cosine and sine of theta phase on those bins.
    """
    reindexed_theta = reindex_by(
        theta_phase,
        is_immobile.index,
        tolerance=1.0 / sampling_frequency,
    )
    # Joint mask: running AND theta phase available
    valid = reindexed_theta.notna().to_numpy().ravel()
    joint_mask = (~is_immobile).to_numpy() & valid

    design_matrix = dmatrix(
        THETA_FORMULA,
        {"theta_phase": reindexed_theta[joint_mask]},
        return_type="dataframe",
    )
    return joint_mask, design_matrix.values


#: Fitter backends. ``"statsmodels"`` is the per-unit IRLS the analysis was
#: built on; ``"jax"`` is the batched IRLS in
#: :mod:`continuum_swr_replay.analysis.pfc.batched_glm`, which mirrors its
#: starting values, step and convergence rule.
FITTERS = ("jax", "statsmodels")


def _check_fitter(fitter: str) -> None:
    if fitter not in FITTERS:
        raise ValueError(f"fitter must be one of {FITTERS}, got {fitter!r}")


def fitter_provenance(fitter: str) -> dict[str, Any]:
    """How the models were fit, for the artifact's provenance sidecar.

    Both backends stop on the same absolute deviance change, so the tolerance
    is recorded once for either; the JAX backend additionally records which
    device it ran on, because a GPU and a CPU run are not bit-identical.

    Parameters
    ----------
    fitter : {"jax", "statsmodels"}

    Returns
    -------
    dict
        ``fitter``, ``fitter_tol``, ``jax_backend``, ``jax_version``. The last
        two are ``None`` for the statsmodels backend.
    """
    _check_fitter(fitter)
    record: dict[str, Any] = {
        "fitter": fitter,
        "jax_backend": None,
        "jax_version": None,
    }
    if fitter == "jax":
        import jax

        from continuum_swr_replay.analysis.pfc.batched_glm import (
            DEFAULT_TOL,
            jax_backend,
        )

        record["fitter_tol"] = DEFAULT_TOL
        record["jax_backend"] = jax_backend()
        record["jax_version"] = jax.__version__
    else:
        # statsmodels' ``GLM.fit`` default, the tolerance the batched fitter
        # mirrors.
        record["fitter_tol"] = 1e-8
    return record


def fit_models_batched(
    design_matrices: Mapping[str, NDArray],
    counts: NDArray,
    penalties: Mapping[str, NDArray] | None = None,
) -> dict[str, Any]:
    """Fit every unit against every design in one batched call per design.

    Parameters
    ----------
    design_matrices : Mapping[str, NDArray]
        Design matrices keyed by model name, each already restricted to the
        bins being fit, shape (n_fit_bins, n_features).
    counts : NDArray, shape (n_fit_bins, n_units)
        Spike counts on those bins, one column per unit.
    penalties : Mapping[str, NDArray], optional
        Per-model ridge penalty vectors, one value per column of that model's
        design. Omitted, or missing a model, the fit is unpenalised. See
        :func:`~continuum_swr_replay.analysis.pfc.batched_glm.make_ridge_penalty`.

    Returns
    -------
    dict[str, BatchedGLMResults]
        Keyed by model name.

    Raises
    ------
    UnconfiguredDeviceError
        From the batched solver, when no JAX device has been chosen and CUDA
        is still reachable -- it would otherwise take every card on a shared
        host.
    """
    from continuum_swr_replay.analysis.pfc.batched_glm import fit_poisson_glms

    penalties = penalties or {}
    return {
        model_name: fit_poisson_glms(design, counts, penalty=penalties.get(model_name))
        for model_name, design in design_matrices.items()
    }


def model_penalties(
    design_matrices: Mapping[str, NDArray], sigma: float | None
) -> dict[str, NDArray]:
    """The ridge penalty for each design: free intercept, ``1/sigma**2`` elsewhere.

    Parameters
    ----------
    design_matrices : Mapping[str, NDArray]
        Design matrices keyed by model name. Every design's first column is the
        intercept.
    sigma : float or None
        Prior standard deviation of an event-basis coefficient on the log-rate
        scale. ``None`` leaves every model unpenalised.

    Returns
    -------
    dict[str, NDArray]
        Empty when ``sigma`` is ``None``.
    """
    if sigma is None:
        return {}
    from continuum_swr_replay.analysis.pfc.batched_glm import make_ridge_penalty

    return {
        model_name: make_ridge_penalty(design.shape[1], sigma)
        for model_name, design in design_matrices.items()
    }


def _fit_theta_models(
    design_matrix: NDArray,
    counts: NDArray,
    fitter: str = "statsmodels",
) -> tuple[Any, Any]:
    """The constant and theta fits the theta LR test compares.

    Parameters
    ----------
    design_matrix : NDArray, shape (n_joint, 3)
    counts : NDArray, shape (n_joint,) or (n_joint, n_units)
        Spike counts on the running bins. A 2-D array fits every unit at once
        and returns batched results.
    fitter : {"statsmodels", "jax"}

    Returns
    -------
    constant_result, theta_result
        Single-unit results for 1-D ``counts``, batched results for 2-D.

    Raises
    ------
    UnconfiguredDeviceError
        From the batched path, when no JAX device has been chosen.
    """
    _check_fitter(fitter)
    constant_design = np.ones((design_matrix.shape[0], 1))
    if fitter == "jax":
        from continuum_swr_replay.analysis.pfc.batched_glm import fit_poisson_glms

        constant_result = fit_poisson_glms(constant_design, counts)
        theta_result = fit_poisson_glms(design_matrix, counts)
        if counts.ndim != 1:
            return constant_result, theta_result
        return constant_result.for_unit(0), theta_result.for_unit(0)
    if counts.ndim != 1:
        raise ValueError(
            "the statsmodels theta fitter takes one unit's counts at a time; "
            "use fitter='jax' to fit a whole session at once"
        )
    constant_result = sm.GLM(
        counts,
        np.ones_like(counts),
        family=sm.families.Poisson(),
    ).fit()
    theta_result = sm.GLM(
        counts,
        design_matrix,
        family=sm.families.Poisson(),
    ).fit()
    return constant_result, theta_result


def test_theta_modulation(
    theta_phase: pd.Series,
    is_immobile: pd.Series,
    neuron_spike_count: NDArray,
    sampling_frequency: float = 500.0,
    *,
    fitter: str = "statsmodels",
    prefitted: tuple[Any, Any] | None = None,
    design: tuple[NDArray, NDArray] | None = None,
) -> tuple[float, float, float, float]:
    """Test theta phase modulation of neuron firing during running.

    Uses an unregularized GLM for the likelihood ratio test (correct for
    hypothesis testing), then a ridge-regularized fit (alpha=1e-20, L1_wt=0)
    for extracting modulation amplitude and preferred phase to avoid
    overfitting.

    The regularized fit is statsmodels' coordinate-descent elastic net under
    both fitters: its answer differs from the unregularized MLE by ~1e-4, so
    ``theta_mod`` and ``theta_preferred_phase`` are that estimator's, not
    IRLS's, and swapping the IRLS backend must not quietly change them.

    Parameters
    ----------
    theta_phase : pd.Series, shape (n_lfp_samples,)
        Theta phase timeseries (radians). May have different sampling
        rate from ``is_immobile``; reindexed internally.
    is_immobile : pd.Series, shape (n_time,)
        Boolean series indicating immobility at each time bin.
    neuron_spike_count : NDArray, shape (n_time,)
        Binned spike counts aligned to ``is_immobile.index``.
    sampling_frequency : float
        Sampling frequency in Hz, used to set reindex tolerance.
    fitter : {"statsmodels", "jax"}
        Backend for the two unregularized fits the LR test is read from.
        Ignored when ``prefitted`` is given.
    prefitted : tuple, optional
        ``(constant_result, theta_result)`` for this unit, already fit on the
        design :func:`build_theta_design` returns for this session. Passed by
        :func:`run_session`, which fits every unit at once.
    design : tuple, optional
        ``(joint_mask, design_matrix)`` from :func:`build_theta_design`. The
        design is the same for every unit of a session and reindexing the theta
        phase onto the time axis is not free, so a session builds it once and
        passes it here; omitted, it is built from ``theta_phase``.

    Returns
    -------
    likelihood_ratio : float
        Likelihood ratio statistic (theta vs constant), or ``nan`` if either
        unregularized fit ran to ``maxiter`` without converging -- a pair of
        fits that never settled has no likelihood ratio to report. The
        regularized fit ``theta_mod`` and ``theta_preferred_phase`` come from
        is a different estimator (statsmodels' coordinate descent, which
        reports no convergence flag) and is not gated.
    p_value : float
        P-value from chi-square test.
    modulation : float
        Theta modulation strength (sqrt of sum of squared cos/sin coefficients).
    preferred_phase : float
        Preferred theta phase in radians (-pi to pi).

    """
    if design is None:
        design = build_theta_design(
            theta_phase, is_immobile, sampling_frequency=sampling_frequency
        )
    joint_mask, design_matrix = design

    if prefitted is not None:
        constant_glm_results, theta_glm_results = prefitted
    else:
        constant_glm_results, theta_glm_results = _fit_theta_models(
            design_matrix, neuron_spike_count[joint_mask], fitter=fitter
        )

    # Gated on convergence for the same reason the immobility models are: a
    # pair of fits that ran to ``maxiter`` has no likelihood ratio to report,
    # and a finite number here would be read as one.
    if all(
        bool(getattr(result, "converged", True))
        for result in (constant_glm_results, theta_glm_results)
    ):
        likelihood_ratio, p_value = _lr_test(theta_glm_results, constant_glm_results)
    else:
        likelihood_ratio, p_value = np.nan, np.nan

    theta_glm_model = sm.GLM(
        neuron_spike_count[joint_mask],
        design_matrix,
        family=sm.families.Poisson(),
    )

    # Fit regularized model to avoid overfitting for modulation and preferred phase
    theta_glm_results = theta_glm_model.fit_regularized(alpha=1e-20, L1_wt=0.0)

    modulation = np.sqrt(
        theta_glm_results.params[1] ** 2 + theta_glm_results.params[2] ** 2
    )
    preferred_phase = np.arctan2(
        theta_glm_results.params[2], theta_glm_results.params[1]
    )

    return likelihood_ratio, p_value, modulation, preferred_phase


#: Spacing of the B-spline's internal knots, in seconds. 50 ms resolves the
#: sharp peri-SWR responses (~50 ms peaks) without giving the kernel more
#: freedom than the events can support.
KNOT_SPACING_S = 0.050


def event_basis_functions(
    lags: NDArray, sampling_frequency: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Cubic B-spline basis over the lag window, zero at both ends.

    The kernel a design column carries is ``sum_b beta_b B_b(lag)``. A clamped
    cubic B-spline has exactly two functions that are non-zero at the window's
    edges -- the first, which is 1 at the left edge, and the last, which is 1
    at the right -- and dropping *both* is what makes the kernel return to
    baseline at both ends of the fit window rather than being pinned at one end
    and free at the other. What is left sums to one between the first and last
    internal knot, so a flat modulation over that span is representable by
    equal coefficients, and each coefficient reads as the log-rate modulation
    near its knot.

    Parameters
    ----------
    lags : NDArray, shape (n_lags,)
        Lag indices in samples, positive meaning after onset.
    sampling_frequency : float
        Sampling frequency in Hz.

    Returns
    -------
    spline_basis : NDArray, shape (n_lags, n_basis)
        ``n_basis = len(knots) + 2`` for a cubic spline.
    lag_time : NDArray, shape (n_lags,)
        Lag times in seconds.
    knots : NDArray, shape (n_knots,)
        Internal knot positions in seconds.
    """
    lag_time = np.asarray(lags, dtype=float) / sampling_frequency
    knots = np.arange(lag_time[0], lag_time[-1], KNOT_SPACING_S)[1:]
    # ``include_intercept=True`` asks patsy for the whole basis (patsy drops
    # the first function otherwise); both edge functions are dropped here.
    spline_basis = np.asarray(
        dmatrix(
            "bs(lag_time, knots=knots, include_intercept=True) - 1",
            {"lag_time": lag_time, "knots": knots},
        )
    )[:, 1:-1]
    return spline_basis, lag_time, knots


def convolve_event_basis(
    onset_counts: NDArray, spline_basis: NDArray, lags: NDArray
) -> NDArray:
    """Convolve an onset indicator with each basis function.

    Column ``b`` is ``sum_L indicator[t - L] B_b(L)`` over the lags ``L``, so a
    positive lag is time *after* an onset. Event onsets are sparse, so each
    non-zero bin adds the small basis block at its affected rows. This avoids
    both the old ``(n_time, n_lags)`` matrix of shifted copies (about 1.4 GB
    per event class on the longest sessions) and a dense full-session
    convolution for every basis function. It is exact: all three forms sum the
    same terms.

    Parameters
    ----------
    onset_counts : NDArray, shape (n_time,)
        Event onsets per time bin.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Basis functions evaluated at the lags, from
        :func:`event_basis_functions`.
    lags : NDArray, shape (n_lags,)
        Contiguous lag indices in samples, ascending by one sample.

    Returns
    -------
    NDArray, shape (n_time, n_basis)
    """
    signal = np.asarray(onset_counts, dtype=float)
    basis = np.asarray(spline_basis, dtype=float)
    lag_indices = np.asarray(lags)
    if signal.ndim != 1:
        raise ValueError(f"onset_counts must be 1-D, got shape {signal.shape}")
    if basis.ndim != 2:
        raise ValueError(f"spline_basis must be 2-D, got shape {basis.shape}")
    if lag_indices.ndim != 1 or len(lag_indices) != basis.shape[0]:
        raise ValueError(
            "lags must be 1-D with one value per spline_basis row; got "
            f"{lag_indices.shape} and {basis.shape}"
        )
    if not np.issubdtype(lag_indices.dtype, np.integer) or (
        len(lag_indices) > 1 and not np.all(np.diff(lag_indices) == 1)
    ):
        raise ValueError("lags must be contiguous ascending integer sample offsets")

    n_time = signal.shape[0]
    convolved = np.zeros((n_time, basis.shape[1]), dtype=float)
    for onset in np.flatnonzero(signal):
        positions = onset + lag_indices
        valid = (positions >= 0) & (positions < n_time)
        convolved[positions[valid]] += signal[onset] * basis[valid]
    return convolved


def event_onset_counts(times: pd.DataFrame, time: NDArray) -> NDArray:
    """Event onsets per time bin, from a table of event windows.

    Parameters
    ----------
    times : pandas.DataFrame
        Event windows with a ``start_time`` column, in seconds.
    time : NDArray, shape (n_time,)
        Time bin starts in seconds.

    Returns
    -------
    NDArray, shape (n_time,)
    """
    return get_spikecount_per_time_bin(times.start_time, time)


def _make_conv_spline_basis(
    times: dict[str, pd.DataFrame],
    position_info: pd.DataFrame,
    sampling_frequency: float = 500,
    lags: NDArray | None = None,
) -> tuple[dict[str, NDArray], NDArray, NDArray]:
    if lags is None:
        lags = np.arange(-100, 101, 1)
    if not times:
        raise ValueError("times dict must be non-empty")

    # The basis depends only on the lags, not on the per-class onsets.
    spline_basis, lag_time, _ = event_basis_functions(lags, sampling_frequency)

    time = position_info.index.to_numpy()
    conv_spline_basis: dict[str, NDArray] = {
        times_name: convolve_event_basis(
            event_onset_counts(times_df, time), spline_basis, lags
        )
        for times_name, times_df in times.items()
    }

    return conv_spline_basis, spline_basis, lag_time


def _make_design_matrices(
    times: dict[str, pd.DataFrame],
    position_info: pd.DataFrame,
    sampling_frequency: float = 500,
    lags: NDArray | None = None,
) -> tuple[dict[str, NDArray], NDArray, NDArray]:
    """Build GLM design matrices for all model variants.

    Parameters
    ----------
    times : dict[str, pd.DataFrame]
        Event times keyed by name. Each DataFrame must have ``start_time``.
        The five keys the models are built from are ``non_local_no_SWR``,
        ``non_local_SWR``, ``non_local``, ``SWR`` and ``matched_control`` --
        the last from :func:`make_matched_control_times`, so a caller that has
        not drawn the control fails here rather than silently fitting a ladder
        without its non-event baseline.
    position_info : pd.DataFrame
        Position data (index defines time bins).
    sampling_frequency : float
        Sampling frequency in Hz.
    lags : NDArray, shape (n_lags,), optional
        Lag indices in samples.

    Returns
    -------
    design_matrices : dict[str, NDArray]
        Design matrices keyed by model name. Each shape
        (n_time, 1 + n_basis * n_event_types_in_model).
    spline_basis : NDArray, shape (n_lags, n_basis)
        B-spline basis functions.
    lag_time : NDArray, shape (n_lags,)
        Lag times in seconds.

    """
    constant_design_matrix = np.ones((len(position_info), 1))

    conv_spline_basis, spline_basis, lag_time = _make_conv_spline_basis(
        times, position_info, sampling_frequency=sampling_frequency, lags=lags
    )

    non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_no_SWR"],
        ],
        axis=1,
    )

    non_local_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_SWR"],
        ],
        axis=1,
    )

    swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["SWR"],
        ],
        axis=1,
    )

    non_local_swr_and_non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local_no_SWR"],
            conv_spline_basis["non_local_SWR"],
        ],
        axis=1,
    )

    swr_and_non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["SWR"],
            conv_spline_basis["non_local_no_SWR"],
        ],
        axis=1,
    )

    non_local_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["non_local"],
        ],
        axis=1,
    )

    # The matched non-event control, built exactly as ``non_local_no_SWR`` is,
    # and jointly with it: the pair is what makes the replay question nested
    # ("does replay add beyond a duration-matched non-event baseline?") rather
    # than a comparison of two single-regressor models the LR test cannot
    # license.
    matched_control_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["matched_control"],
        ],
        axis=1,
    )

    matched_control_and_non_local_no_swr_design_matrix = np.concatenate(
        [
            constant_design_matrix,
            conv_spline_basis["matched_control"],
            conv_spline_basis["non_local_no_SWR"],
        ],
        axis=1,
    )

    design_matrices = {
        "constant": constant_design_matrix,
        "non_local_no_SWR": non_local_no_swr_design_matrix,
        "non_local_SWR": non_local_swr_design_matrix,
        "SWR": swr_design_matrix,
        "non_local_SWR + non_local_no_SWR": non_local_swr_and_non_local_no_swr_design_matrix,
        "SWR + non_local_no_SWR": swr_and_non_local_no_swr_design_matrix,
        "non_local": non_local_design_matrix,
        "matched_control": matched_control_design_matrix,
        "matched_control + non_local_no_SWR": (
            matched_control_and_non_local_no_swr_design_matrix
        ),
    }

    return design_matrices, spline_basis, lag_time


def _fit_immobility_models(
    fit_mask: pd.Series,
    design_matrices: dict[str, NDArray],
    neuron_spike_count: NDArray,
    *,
    fitter: str = "statsmodels",
    prefitted: Mapping[str, Any] | None = None,
    penalties: Mapping[str, NDArray] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit Poisson GLMs for each design matrix on the fitted time bins.

    Parameters
    ----------
    fit_mask : pd.Series, shape (n_time,)
        Boolean mask for the time bins the models are fit on: immobility, and
        (when a stop cohort is applied) immobility inside a cohort stop.
    design_matrices : dict[str, NDArray]
        Design matrices keyed by model name, each shape (n_time, n_features).
    neuron_spike_count : NDArray, shape (n_time,)
        Spike counts per time bin.
    fitter : {"statsmodels", "jax"}
        Backend. Ignored when ``prefitted`` is given.
    prefitted : Mapping[str, Any], optional
        This unit's already-fitted results keyed by model name, as
        :func:`run_session` produces by fitting the whole session at once.
        Nothing is fit when they are supplied.
    penalties : Mapping[str, NDArray], optional
        Per-model ridge penalty vectors for the batched fitter, as
        :func:`model_penalties` builds them. The statsmodels backend has no
        penalised path and refuses them.

    Returns
    -------
    glm_results : dict[str, Any]
        Fitted model results keyed by model name. Whatever the backend, each
        exposes ``params``, ``deviance``, ``llf``, ``aic`` and ``df_resid``.
    glm_models : dict[str, statsmodels GLM]
        Unfitted model objects keyed by model name. Empty unless the
        statsmodels backend did the fitting -- the batched backend has no
        per-unit model object, and nothing downstream reads one.

    Raises
    ------
    ValueError
        If ``prefitted`` is missing a design's results, or if a penalty is
        asked of the statsmodels backend.
    """
    if prefitted is not None:
        missing = sorted(set(design_matrices) - set(prefitted))
        if missing:
            raise ValueError(f"prefitted results are missing models: {missing}")
        return {name: prefitted[name] for name in design_matrices}, {}

    _check_fitter(fitter)
    if fitter == "jax":
        batched = fit_models_batched(
            {
                model_name: design_matrix[fit_mask]
                for model_name, design_matrix in design_matrices.items()
            },
            np.asarray(neuron_spike_count)[fit_mask][:, None],
            penalties,
        )
        return {name: result.for_unit(0) for name, result in batched.items()}, {}

    if penalties:
        raise ValueError(
            "the statsmodels backend fits unpenalised models only; use "
            "fitter='jax' for a penalised fit"
        )
    glm_models = {
        model_name: sm.GLM(
            neuron_spike_count[fit_mask],
            design_matrix[fit_mask],
            family=sm.families.Poisson(),
        )
        for model_name, design_matrix in design_matrices.items()
    }
    glm_results = {model_name: model.fit() for model_name, model in glm_models.items()}

    return glm_results, glm_models


def get_frac_null_deviance(
    full_model: Any, reduced_model: Any, constant_model: Any
) -> float:
    """Compute fraction of total (null) deviance explained by additional terms.

    Parameters
    ----------
    full_model : statsmodels GLMResults
        More complex model (lower deviance).
    reduced_model : statsmodels GLMResults
        Simpler nested model (higher deviance).
    constant_model : statsmodels GLMResults
        Intercept-only model (highest deviance = total deviance).

    Returns
    -------
    float
        (reduced.deviance - full.deviance) / constant.deviance.
        Non-negative for properly nested models.

    """
    return float(
        (reduced_model.deviance - full_model.deviance) / constant_model.deviance
    )


def get_frac_explained_deviance(
    full_model: Any, reduced_model: Any, constant_model: Any
) -> float:
    """Compute fraction of model-explained deviance attributable to reduced terms.

    Parameters
    ----------
    full_model : statsmodels GLMResults
        Most complex model.
    reduced_model : statsmodels GLMResults
        Simpler nested model.
    constant_model : statsmodels GLMResults
        Intercept-only model.

    Returns
    -------
    float
        (reduced.deviance - full.deviance) / (constant.deviance - full.deviance).
        Returns 0.0 if the full model explains no deviance.

    """
    denominator = constant_model.deviance - full_model.deviance
    if abs(denominator) < 1e-10:
        return 0.0
    return float((reduced_model.deviance - full_model.deviance) / denominator)


#: Half-width of the window peak modulations are read over, in seconds. The
#: models are fit on a wider window (+/-250 ms) so the kernel's ends are outside
#: what is reported; a peak taken over the fit window would be reporting the
#: part that was widened to be thrown away.
REPORT_HALF_WINDOW_S = 0.200

#: A coefficient past this on the log-rate scale is a rate multiplier of 5e8:
#: a basis column whose support holds no spikes, whose maximum-likelihood
#: estimate is at -infinity, and whose fitted value says only where the solver
#: stopped. Used to separate units that have a maximum-likelihood fit to
#: compare against from units that do not.
SEPARATED_LOG_RATE = 20.0


class PeakModulation(NamedTuple):
    """The largest excursion of a peri-event kernel, and where it is.

    Attributes
    ----------
    value : NDArray, shape (n_units,)
        Signed log-rate modulation at the peak -- the value at the largest
        absolute excursion, not the largest value.
    lag : NDArray, shape (n_units,)
        Lag of that peak, in seconds.
    index : NDArray of int, shape (n_units,)
        Row of ``spline_basis`` the peak sits at, for evaluating the basis
        there.
    """

    value: NDArray
    lag: NDArray
    index: NDArray


def peak_log_modulation(
    params: NDArray,
    spline_basis: NDArray,
    lag_time: NDArray,
    half_window_s: float = REPORT_HALF_WINDOW_S,
) -> PeakModulation:
    """Peak of ``spline_basis @ params`` over the reported window, per unit.

    Parameters
    ----------
    params : NDArray, shape (n_units, n_basis) or (n_basis,)
        One event class's basis coefficients -- the block of a design's
        parameters that multiplies ``spline_basis``, not the whole parameter
        vector.
    spline_basis : NDArray, shape (n_lags, n_basis)
    lag_time : NDArray, shape (n_lags,)
        Lag of each basis row, in seconds.
    half_window_s : float
        Peaks are taken over ``|lag| <= half_window_s``.

    Returns
    -------
    PeakModulation
        With ``nan`` value and lag, and index ``-1``, for a unit whose
        coefficients are not finite.

    Raises
    ------
    ValueError
        If no lag falls inside the window.
    """
    coefficients = np.atleast_2d(np.asarray(params, dtype=float))
    lag_time = np.asarray(lag_time, dtype=float)
    inside = np.flatnonzero(np.abs(lag_time) <= half_window_s + 1e-12)
    if inside.size == 0:
        raise ValueError(
            f"no lag inside +/-{half_window_s} s; the basis covers "
            f"{lag_time.min()} to {lag_time.max()}"
        )
    curves = coefficients @ np.asarray(spline_basis, dtype=float)[inside].T
    usable = np.isfinite(curves).all(axis=1)
    at = np.argmax(np.abs(np.nan_to_num(curves)), axis=1)
    index = np.where(usable, inside[at], -1)
    value = np.where(usable, curves[np.arange(len(curves)), at], np.nan)
    lag = np.where(usable, lag_time[index], np.nan)
    return PeakModulation(value=value, lag=lag, index=index)


def peak_modulation_se(
    covariance: NDArray, spline_basis: NDArray, peak_index: NDArray
) -> NDArray:
    """Standard error of the kernel at each unit's peak lag.

    The kernel at one lag is ``b(lag)' beta``, a linear function of the
    coefficients, so its variance is ``b(lag)' cov(beta) b(lag)`` -- the
    off-diagonal terms included, which is why the diagonal ``bse`` cannot
    answer this. The peak lag is treated as fixed, as a delta-method standard
    error does.

    Parameters
    ----------
    covariance : NDArray, shape (n_units, n_basis, n_basis)
        Covariance of the same coefficient block ``params`` came from.
    spline_basis : NDArray, shape (n_lags, n_basis)
    peak_index : NDArray of int, shape (n_units,)
        Basis row of each unit's peak, from :func:`peak_log_modulation`.
        ``-1`` gives ``nan``.

    Returns
    -------
    NDArray, shape (n_units,)
    """
    covariance = np.asarray(covariance, dtype=float)
    basis = np.asarray(spline_basis, dtype=float)
    peak_index = np.asarray(peak_index, dtype=int)
    at_peak = basis[np.clip(peak_index, 0, len(basis) - 1)]
    variance = np.einsum("np,npq,nq->n", at_peak, covariance, at_peak)
    return np.where(peak_index >= 0, np.sqrt(np.clip(variance, 0.0, None)), np.nan)


def is_separated(params: NDArray, threshold: float = SEPARATED_LOG_RATE) -> NDArray:
    """Units whose fit has no maximum-likelihood estimate to compare against.

    Parameters
    ----------
    params : NDArray, shape (n_units, n_basis) or (n_basis,)
        Basis coefficients of an **unpenalised** fit.
    threshold : float
        Absolute log-rate a coefficient has to pass to count as separated.

    Returns
    -------
    NDArray of bool, shape (n_units,)
        True where the fit ran away (or is not finite).
    """
    coefficients = np.atleast_2d(np.asarray(params, dtype=float))
    return ~np.all(np.abs(coefficients) <= threshold, axis=1)


def prior_binds(
    penalised_peak: NDArray, mle_peak: NDArray, mle_peak_se: NDArray
) -> NDArray:
    """Where the prior moved the curve by more than the data could resolve.

    "The prior binds" is a statement about the modulation, not about a
    coefficient: the peak of the penalised kernel differs from the peak of the
    maximum-likelihood one by more than the latter's own estimation error. A
    unit with no maximum-likelihood fit (see :func:`is_separated`) has nothing
    to difference and is not counted here -- report it separately.

    Parameters
    ----------
    penalised_peak, mle_peak : NDArray, shape (n_units,)
        Peak log-rate modulations, from :func:`peak_log_modulation`.
    mle_peak_se : NDArray, shape (n_units,)
        Standard error of ``mle_peak``, from :func:`peak_modulation_se`.

    Returns
    -------
    NDArray of bool, shape (n_units,)
        False wherever any of the three is not finite.
    """
    difference = np.abs(np.asarray(penalised_peak) - np.asarray(mle_peak))
    finite = np.isfinite(difference) & np.isfinite(mle_peak_se)
    return finite & (difference > np.asarray(mle_peak_se))


def _fit_diagnostics(results: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-model fit summary, read the same way off either backend's results.

    The quantities the fitter equivalence audit is run on: the parameters and
    the three fit criteria, plus the rank and the convergence record that say
    whether the numbers mean anything.

    Parameters
    ----------
    results : Mapping[str, Any]
        Fitted results keyed by model name (statsmodels ``GLMResults`` or
        :class:`~continuum_swr_replay.analysis.pfc.batched_glm.UnitGLMResult`).

    Returns
    -------
    dict[str, dict[str, Any]]
    """
    diagnostics: dict[str, dict[str, Any]] = {}
    for name, result in results.items():
        history = getattr(result, "fit_history", None)
        diagnostics[name] = {
            "params": np.asarray(result.params, dtype=float),
            "deviance": float(result.deviance),
            "llf": float(result.llf),
            "aic": float(result.aic),
            "df_resid": float(result.df_resid),
            "df_model": float(result.df_model),
            "converged": bool(getattr(result, "converged", True)),
            "n_iter": int(
                getattr(result, "n_iter", None)
                if getattr(result, "n_iter", None) is not None
                else (history or {}).get("iteration", -1)
            ),
        }
    return diagnostics


def process_neuron(
    neuron_spike_times: NDArray,
    is_immobile: pd.Series,
    theta_phase: pd.Series,
    time: NDArray,
    immobility_design_matrices: dict[str, NDArray],
    spline_basis: NDArray,
    lag_time: NDArray,
    logger_name: str | None = None,
    fit_mask: pd.Series | None = None,
    fitter: str = "statsmodels",
    prefitted_immobility: Mapping[str, Any] | None = None,
    prefitted_theta: tuple[Any, Any] | None = None,
    theta_design: tuple[NDArray, NDArray] | None = None,
    design_ranks: Mapping[str, int] | None = None,
    fit_diagnostics_out: dict[str, Any] | None = None,
    penalty_sigma: float | None = None,
    prebinned_spike_count: NDArray | None = None,
) -> tuple[dict[str, Any], NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Fit GLM models for a single neuron and compute modulation statistics.

    Fits 9 Poisson GLM models during immobility to test how the neuron's
    firing rate relates to SWR and non-local (replay) events. Models are
    organized into two decomposition strategies:

    **SWR-controlled decomposition** — separates the generic ripple response
    from replay-specific modulation:

    - ``constant``: intercept only (baseline)
    - ``SWR``: all whole-event-speed-eligible detected ripples
    - ``non_local``: all replay events regardless of SWR overlap
    - ``SWR + non_local_no_SWR``: ripples + non-SWR replay (key model;
      isolates replay-specific effects after controlling for SWR)

    **Replay-split decomposition** — tests whether SWR-coincident and
    non-SWR replay drive different responses:

    - ``non_local_SWR``: replay events that co-occur with a ripple
    - ``non_local_no_SWR``: replay events without a ripple
    - ``non_local_SWR + non_local_no_SWR``: both replay types with
      separate regressors

    **Engagement decomposition** -- separates replay from the immobility
    it happens in:

    - ``matched_control``: duration-matched non-event windows drawn from
      the same fitted immobility bouts (:func:`make_matched_control_times`)
    - ``matched_control + non_local_no_SWR``: the baseline plus non-SWR
      replay, the model the engagement test is run on

    The analysis addresses a sequence of scientific questions:

    **Q1. Do PFC neurons respond to hippocampal events during immobility?**
    ``SWR vs constant``, ``non_local vs constant``

    **Q2. Are non-local events and SWRs redundant or independent?**
    ``SWR + non_local_no_SWR vs SWR`` (replay adds beyond ripples),
    ``SWR + non_local_no_SWR vs non_local_no_SWR`` (ripples add beyond
    replay).

    **Q3. Do PFC neurons respond to non-local events lacking a detected SWR?**
    ``non_local_no_SWR vs constant``. Modulation: ``no_swr_modulation``.

    **Q4. Is the non-SWR replay response robust to sub-threshold ripple
    confounds?**
    Compare ``no_swr_modulation`` (controlling for all SWRs) vs
    ``non_local_no_swr_modulation`` (controlling for replay-coincident
    SWRs only). ``non_local_SWR + non_local_no_SWR vs non_local_SWR``.

    **Q5. Do SWR-coincident and non-SWR replay drive different responses?**
    ``non_local_SWR + non_local_no_SWR vs non_local`` (does splitting
    improve fit?). Compare
    ``non_local_swr_modulation`` vs ``non_local_no_swr_modulation`` shapes.

    **Q6. Does replay engage the neuron beyond the immobility it occurs
    in?**
    ``matched_control + non_local_no_SWR vs. matched_control`` (the
    engagement test: non-SWR replay adds beyond a duration-matched
    non-event baseline drawn from the same bouts), and
    ``matched_control vs. constant`` (does that baseline itself modulate
    firing -- a positive answer means the peri-event window is not
    exchangeable with the rest of the bout, and the engagement test is the
    comparison to read). The control is matched one per *non-local* event,
    SWR-coincident ones included, so it is more numerous than the
    ``non_local_no_SWR`` class it is compared against; a source event's
    duration fixes its control's duration and constrains where the window
    fits, and nothing else about the event is matched.

    **Q7. What is the relationship to theta modulation during running?**
    ``theta_mod``, ``theta_preferred_phase``, firing rates.

    Each nested pair carries an effect size -- ``frac_null_dev`` and
    ``frac_explained_dev``, ratios of the deviances the added regressors buy.
    No likelihood-ratio statistic, p-value or AIC is produced here: the fits
    are penalised, so a likelihood ratio has no chi-square reference
    distribution at any stated degrees of freedom and an AIC has no stated
    number of parameters. What the models are compared and selected by is the
    held-out deviance the runner scores them on, read against a null drawn by
    redrawing the matched non-event control. Multiple-comparison correction
    across neurons and the population layer are downstream.

    Parameters
    ----------
    neuron_spike_times : NDArray, shape (n_spikes,)
        Spike times in seconds.
    is_immobile : pd.Series, shape (n_time,)
        Boolean series indicating immobility at each time bin.
    theta_phase : pd.Series, shape (n_lfp_samples,)
        Theta phase timeseries (radians).
    time : NDArray, shape (n_time,)
        Time bin centers in seconds (from ``position_info.index``).
    immobility_design_matrices : dict[str, NDArray]
        Pre-computed design matrices keyed by model name.
        Each value has shape ``(n_time, n_features)``.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Spline basis functions for reconstructing peri-event modulation.
    lag_time : NDArray, shape (n_lags,)
        Lag times in seconds corresponding to spline_basis rows.
    logger_name : str | None, optional
        Logger name for per-neuron messages. If None, uses module logger.
    fit_mask : pd.Series | None, shape (n_time,), optional
        Bins the immobility GLMs are fit on. Defaults to ``is_immobile``, i.e.
        every immobile bin in the epoch; a stop-cohort run passes
        ``is_immobile & in_cohort_stop`` so the models describe the same
        population the hippocampal ladders do. ``is_immobile`` itself still
        defines the *running* comparison (theta modulation and
        ``run_avg_firing_rate``): a bin outside the cohort is not running, so it
        must not be counted as such.
    fitter : {"statsmodels", "jax"}, optional
        Backend for the IRLS fits. Ignored for any model whose results are
        supplied prefitted.
    prefitted_immobility : Mapping[str, Any] | None, optional
        This neuron's immobility-model results, keyed by model name, already
        fit on ``fit_mask``. :func:`run_session` fits every neuron of a session
        at once and passes each neuron's slice in here.
    prefitted_theta : tuple | None, optional
        ``(constant_result, theta_result)`` for the theta LR test, likewise
        already fit. The regularized fit ``theta_mod`` and
        ``theta_preferred_phase`` come from is always run here, in statsmodels.
    theta_design : tuple | None, optional
        ``(joint_mask, design_matrix)`` from :func:`build_theta_design`, built
        once for the session rather than once per neuron.
    design_ranks : Mapping[str, int] | None, optional
        ``numpy.linalg.matrix_rank`` of each design on ``fit_mask``. The rank
        is a property of the design, not of the neuron, so a session computes
        it once; when omitted it is computed here, as it always was.
    fit_diagnostics_out : dict | None, optional
        When given, filled with this neuron's per-model fit summary (params,
        deviance, llf, aic, rank, convergence). For the fitter equivalence
        audit; never part of the shipped bundle.
    penalty_sigma : float | None, optional
        Prior standard deviation of an event-basis coefficient, on the log-rate
        scale, when this neuron's immobility models are fit here. ``None`` fits
        them unpenalised. Ignored when ``prefitted_immobility`` is given -- the
        prior is then whatever the session was prefit with.
    prebinned_spike_count : NDArray, shape (n_time,), optional
        Counts already binned on ``time``. The session runner bins every unit
        once for the batched and held-out fits and passes the corresponding
        column here; direct callers may continue to pass spike times only.

    Returns
    -------
    neuron_stats : dict[str, Any]
        Per-neuron statistics including:

        - ``run_avg_firing_rate``: Hz over every moving bin.
        - ``immobility_avg_firing_rate``: Hz over the bins the models were fit
          on -- with a stop cohort, the cohort's immobile bins, not every
          immobile bin. This is the rate the downstream ``min_firing_rate``
          inclusion gate screens on, so it describes the sample the estimates
          come from; its value therefore depends on the cohort the bundle was
          fit on (named in the bundle's provenance sidecar).
        - ``immobility_avg_firing_rate_all_immobile``: Hz over every immobile
          bin, cohort or not. Equal to ``immobility_avg_firing_rate`` exactly
          when no stop restriction was applied.
        - ``theta_mod``, ``theta_preferred_phase``: theta modulation during running
        - ``{model} vs. {baseline}: frac_null_dev, frac_explained_dev``:
          fractional deviance measures
        - ``SWR_max_mod``, ``non_local_no_SWR_max_mod``: peak modulation
          from SWR-controlled model
        - ``non_local_SWR_max_mod``, ``non_local_no_SWR_split_max_mod``:
          peak modulation from replay-split model
        - ``matched_control_max_mod``: peak modulation of the matched
          non-event baseline from the engagement model
        - ``*_max_mod_time``: corresponding peak lag times (s)

        A model with no usable fit -- NaN parameters, or an iteration that
        never settled -- is non-estimable for this neuron: every statistic
        that reads it is NaN, and the neuron's other models are reported as
        usual. Empty dict with only firing rates if the fitting call itself
        raised.

    swr_modulation : NDArray
        SWR peri-event modulation from SWR-controlled model
        (``SWR + non_local_no_SWR``). Shape (n_lag_times,).
    no_swr_modulation : NDArray
        Non-SWR replay modulation from SWR-controlled model, controlling
        for all whole-event-speed-eligible ripples. Shape (n_lag_times,).
    non_local_swr_modulation : NDArray
        SWR-coincident replay modulation from replay-split model
        (``non_local_SWR + non_local_no_SWR``). Shape (n_lag_times,).
    non_local_no_swr_modulation : NDArray
        Non-SWR replay modulation from replay-split model, controlling
        for replay-coincident ripples only. Shape (n_lag_times,).
        Comparing this with ``no_swr_modulation`` tests whether the
        non-SWR replay response is robust to the choice of SWR control.
    matched_control_modulation : NDArray
        Matched non-event baseline modulation from the engagement model
        (``matched_control + non_local_no_SWR``), i.e. the response around
        duration-matched control windows after accounting for real non-SWR
        replay. Shape (n_lag_times,). A curve that is flat while
        ``no_swr_modulation`` is not is the evidence that the replay
        response is to replay rather than to the immobility around it.

    """
    _logger = logging.getLogger(logger_name) if logger_name else logger

    neuron_spike_count = (
        get_spikecount_per_time_bin(neuron_spike_times, time)
        if prebinned_spike_count is None
        else np.asarray(prebinned_spike_count, dtype=float)
    )
    if neuron_spike_count.shape != np.asarray(time).shape:
        raise ValueError(
            "prebinned_spike_count must have one value per time bin; got "
            f"{neuron_spike_count.shape} for time {np.asarray(time).shape}"
        )
    sampling_frequency = 1 / np.median(np.diff(time))
    if fit_mask is None:
        fit_mask = is_immobile

    try:
        neuron_stats: dict[str, Any] = {}

        # Get run theta statistics
        (
            neuron_stats["theta vs. constant: lr"],
            neuron_stats["theta vs. constant: pval"],
            neuron_stats["theta_mod"],
            neuron_stats["theta_preferred_phase"],
        ) = test_theta_modulation(
            theta_phase,
            is_immobile,
            neuron_spike_count,
            sampling_frequency=sampling_frequency,
            fitter=fitter,
            prefitted=prefitted_theta,
            design=theta_design,
        )
        neuron_stats["run_avg_firing_rate"] = (
            neuron_spike_count[~is_immobile].mean() * sampling_frequency
        )

        # Get immobility SWR/non-local statistics. This is the mean rate over
        # the bins the models are actually fit on, so the downstream
        # minimum-firing-rate inclusion gate screens the sample the estimates
        # come from rather than a wider one. The column name is the same across
        # cohorts, so the unrestricted rate is carried alongside it: the pair
        # makes a bundle's cohort readable from the table itself (the two are
        # equal exactly when no stop restriction was applied).
        neuron_stats["immobility_avg_firing_rate"] = (
            neuron_spike_count[fit_mask].mean() * sampling_frequency
        )
        neuron_stats["immobility_avg_firing_rate_all_immobile"] = (
            neuron_spike_count[is_immobile].mean() * sampling_frequency
        )

        results, _ = _fit_immobility_models(
            fit_mask,
            immobility_design_matrices,
            neuron_spike_count,
            fitter=fitter,
            prefitted=prefitted_immobility,
            penalties=model_penalties(immobility_design_matrices, penalty_sigma),
        )

        # A model whose fit produced nothing usable -- NaN parameters, or an
        # iteration that never settled -- is marked non-estimable for THIS
        # neuron and routed to NaN below, exactly as an absent event class is.
        # The neuron's other models are unaffected: they are separate fits of
        # separate designs, and a neuron whose ``matched_control`` fit failed
        # still has a ``SWR`` fit that says something. Failing the whole neuron
        # would discard that, and would do it silently -- the neuron would come
        # back with firing rates and no statistics at all.
        failed_models = {
            name
            for name, res in results.items()
            if np.any(np.isnan(res.params)) or not bool(getattr(res, "converged", True))
        }
        if failed_models:
            _logger.warning(
                "No usable fit for models %s (NaN parameters, or no "
                "convergence); their statistics are NaN for this neuron. "
                "Immobility firing rate: %.3f Hz.",
                sorted(failed_models),
                neuron_stats["immobility_avg_firing_rate"],
            )

        if fit_diagnostics_out is not None:
            fit_diagnostics_out.update(_fit_diagnostics(results))

        # Detect non-estimable models. ``_make_conv_spline_basis``
        # builds spline regressors for every event class regardless
        # of whether that class has any events; an empty event class
        # leaves its spline columns all-zero, so the design matrix is
        # rank-deficient against the intercept. The fitter still
        # returns finite (zero) parameters for those columns and
        # finite degenerate LR / fractional-deviance values, so the
        # failed-fit check above doesn't catch them. Mark every
        # rank-deficient design as non-estimable up front and route
        # the affected deviance fractions and modulation
        # extractions to NaN below — so downstream consumers can
        # tell "absent event class" from "estimated zero effect".
        non_estimable_models = set(failed_models)
        for name, design in immobility_design_matrices.items():
            X_im = design[fit_mask]
            if X_im.shape[0] == 0:
                non_estimable_models.add(name)
                continue
            # The rank is a property of the design, which every neuron of a
            # session shares, so a session-level caller supplies it rather than
            # paying for the same decomposition once per neuron.
            rank = (
                np.linalg.matrix_rank(X_im)
                if design_ranks is None
                else design_ranks[name]
            )
            if rank < X_im.shape[1]:
                non_estimable_models.add(name)

        ## Effect sizes: the deviance each nested pair's added regressors buy
        # frac_null_dev: (reduced - full) / constant deviance
        #   = fraction of total deviance uniquely explained by added regressors
        # frac_explained_dev: (reduced - full) / (constant - full)
        #   = fraction of full model's explained deviance due to added
        #     regressors
        #
        # These are ratios of deviances, not tests. No likelihood ratio and no
        # AIC is computed here: the fits carry a prior on the event-basis
        # coefficients, so a likelihood ratio has no chi-square reference
        # distribution at any stated degrees of freedom and an AIC has no
        # stated number of parameters. What the models are compared by is the
        # held-out deviance the runner scores them on.
        for model1, model2 in NESTED_MODEL_PAIRS:
            test_name = f"{model1} vs. {model2}"
            if (
                model1 in non_estimable_models
                or model2 in non_estimable_models
                or "constant" in non_estimable_models
            ):
                neuron_stats[f"{test_name}: frac_null_dev"] = np.nan
                neuron_stats[f"{test_name}: frac_explained_dev"] = np.nan
            else:
                neuron_stats[f"{test_name}: frac_null_dev"] = get_frac_null_deviance(
                    results[model1], results[model2], results["constant"]
                )
                neuron_stats[f"{test_name}: frac_explained_dev"] = (
                    get_frac_explained_deviance(
                        results[model1], results[model2], results["constant"]
                    )
                )

        ## Modulation time courses
        # Share fits with the deviance-fraction path so neuron_stats and
        # the modulation arrays can never disagree due to divergent
        # IRLS convergence. When the source model is non-estimable
        # (rank-deficient design — typically an absent event class),
        # the spline-by-coefficient product would silently emit a
        # zero curve from statsmodels' degenerate fit; route those
        # to NaN so consumers can tell "absent event class" from
        # "estimated zero modulation".
        n_basis_functions = spline_basis.shape[1]
        nan_modulation = np.full(spline_basis.shape[0], np.nan)

        # SWR-controlled model: SWR and non-SWR replay modulation
        # controlling for all whole-event-speed-eligible ripples
        if "SWR + non_local_no_SWR" in non_estimable_models:
            swr_modulation = nan_modulation
            no_swr_modulation = nan_modulation
            neuron_stats["SWR_max_mod"] = np.nan
            neuron_stats["SWR_max_mod_time"] = np.nan
            neuron_stats["non_local_no_SWR_max_mod"] = np.nan
            neuron_stats["non_local_no_SWR_max_mod_time"] = np.nan
        else:
            swr_ctrl_params = results["SWR + non_local_no_SWR"].params
            swr_modulation = spline_basis @ swr_ctrl_params[1 : n_basis_functions + 1]
            no_swr_modulation = spline_basis @ swr_ctrl_params[n_basis_functions + 1 :]

            swr_max_mod_ind = np.argmax(np.abs(swr_modulation))
            neuron_stats["SWR_max_mod"] = swr_modulation[swr_max_mod_ind]
            neuron_stats["SWR_max_mod_time"] = lag_time[swr_max_mod_ind]

            no_swr_max_mod_ind = np.argmax(np.abs(no_swr_modulation))
            neuron_stats["non_local_no_SWR_max_mod"] = no_swr_modulation[
                no_swr_max_mod_ind
            ]
            neuron_stats["non_local_no_SWR_max_mod_time"] = lag_time[no_swr_max_mod_ind]

        # Replay-split model: SWR-replay and non-SWR replay modulation
        # controlling for replay-coincident ripples only. The blocks are read
        # in the order ``_make_design_matrices`` concatenates them --
        # ``[constant, non_local_no_SWR, non_local_SWR]`` -- which is NOT the
        # order the model's name lists them in. Reading them by the name's
        # order is exactly the bug this comment exists to prevent: between
        # 2026-04-24 and 2026-09-15 each curve carried the other class's
        # kernel, and so did ``non_local_SWR_max_mod`` and
        # ``non_local_no_SWR_split_max_mod``.
        if "non_local_SWR + non_local_no_SWR" in non_estimable_models:
            non_local_swr_modulation = nan_modulation
            non_local_no_swr_modulation = nan_modulation
            neuron_stats["non_local_SWR_max_mod"] = np.nan
            neuron_stats["non_local_SWR_max_mod_time"] = np.nan
            neuron_stats["non_local_no_SWR_split_max_mod"] = np.nan
            neuron_stats["non_local_no_SWR_split_max_mod_time"] = np.nan
        else:
            replay_split_params = results["non_local_SWR + non_local_no_SWR"].params
            non_local_no_swr_modulation = (
                spline_basis @ replay_split_params[1 : n_basis_functions + 1]
            )
            non_local_swr_modulation = (
                spline_basis @ replay_split_params[n_basis_functions + 1 :]
            )

            non_local_swr_max_ind = np.argmax(np.abs(non_local_swr_modulation))
            neuron_stats["non_local_SWR_max_mod"] = non_local_swr_modulation[
                non_local_swr_max_ind
            ]
            neuron_stats["non_local_SWR_max_mod_time"] = lag_time[non_local_swr_max_ind]

            non_local_no_swr_max_ind = np.argmax(np.abs(non_local_no_swr_modulation))
            neuron_stats["non_local_no_SWR_split_max_mod"] = (
                non_local_no_swr_modulation[non_local_no_swr_max_ind]
            )
            neuron_stats["non_local_no_SWR_split_max_mod_time"] = lag_time[
                non_local_no_swr_max_ind
            ]

        # Matched-control model: the non-event baseline's own peri-window
        # modulation, taken from the joint engagement model so it is the
        # baseline response after accounting for real non-SWR replay --
        # the same model the engagement comparison is run on.
        if "matched_control + non_local_no_SWR" in non_estimable_models:
            matched_control_modulation = nan_modulation
            neuron_stats["matched_control_max_mod"] = np.nan
            neuron_stats["matched_control_max_mod_time"] = np.nan
        else:
            engagement_params = results["matched_control + non_local_no_SWR"].params
            matched_control_modulation = (
                spline_basis @ engagement_params[1 : n_basis_functions + 1]
            )
            matched_control_max_ind = np.argmax(np.abs(matched_control_modulation))
            neuron_stats["matched_control_max_mod"] = matched_control_modulation[
                matched_control_max_ind
            ]
            neuron_stats["matched_control_max_mod_time"] = lag_time[
                matched_control_max_ind
            ]

    except (ValueError, np.linalg.LinAlgError):
        _logger.warning(
            "GLM fitting failed for neuron; falling back to firing rates only",
            exc_info=True,
        )
        neuron_stats = {}
        neuron_stats["run_avg_firing_rate"] = (
            neuron_spike_count[~is_immobile].mean() * sampling_frequency
        )
        neuron_stats["immobility_avg_firing_rate"] = (
            neuron_spike_count[fit_mask].mean() * sampling_frequency
        )
        neuron_stats["immobility_avg_firing_rate_all_immobile"] = (
            neuron_spike_count[is_immobile].mean() * sampling_frequency
        )
        nan_modulation = np.full(spline_basis.shape[0], np.nan)
        swr_modulation = nan_modulation
        no_swr_modulation = nan_modulation
        non_local_swr_modulation = nan_modulation
        non_local_no_swr_modulation = nan_modulation
        matched_control_modulation = nan_modulation

    return (
        neuron_stats,
        swr_modulation,
        no_swr_modulation,
        non_local_swr_modulation,
        non_local_no_swr_modulation,
        matched_control_modulation,
    )


def build_session_fit_mask(
    position_info: pd.DataFrame,
    *,
    cohort_stops: pd.DataFrame | None = None,
    stop_cohort: str = "at_well",
    immobility_threshold: float = 4.0,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Bins the immobility GLMs are fit on, for one session.

    Two restrictions compose. Immobility is the long-standing one. The stop
    cohort is the new one: it keeps only the time the animal spends stopped at a
    well, which is the population the hippocampal rate and content ladders are
    read on, so a PFC claim and a replay claim describe the same stops.

    Parameters
    ----------
    position_info : pandas.DataFrame
        Time-indexed position table carrying ``head_speed``. The index is
        absolute seconds, the same clock the stop table's bounds are on.
    cohort_stops : pandas.DataFrame, optional
        Stops kept by
        :func:`~continuum_swr_replay.analysis.pfc.cohort.load_session_stop_cohort`.
        Required unless ``stop_cohort`` is ``"all_stops"``.
    stop_cohort : str
        Cohort name, recorded in the returned counts. ``"all_stops"`` means *no
        stop restriction at all* -- every immobile bin in the epoch is fit,
        reproducing the pre-cohort mask exactly.
    immobility_threshold : float
        Speed threshold for immobility (cm/s).

    Returns
    -------
    is_immobile : pandas.Series of bool, shape (n_time,)
        Immobility alone; still the definition of the *running* comparison.
    fit_mask : pandas.Series of bool, shape (n_time,)
        ``is_immobile & in_cohort_stop``, or ``is_immobile`` for ``all_stops``.
    counts : dict
        ``stop_cohort``, ``n_bins_total``, ``n_bins_immobile``, ``n_bins_fit``.

    Raises
    ------
    ValueError
        If a stop cohort is requested without ``cohort_stops``.
    """
    is_immobile = position_info.head_speed < immobility_threshold
    if stop_cohort == "all_stops":
        fit_mask = is_immobile
    else:
        if cohort_stops is None:
            raise ValueError(
                f"stop_cohort={stop_cohort!r} needs cohort_stops; pass the stops "
                "load_session_stop_cohort returned, or use 'all_stops' for no "
                "stop restriction"
            )
        in_cohort = stop_interval_mask(
            position_info.index.to_numpy(dtype=float), cohort_stops
        )
        fit_mask = is_immobile & pd.Series(in_cohort, index=position_info.index)
    counts = {
        "stop_cohort": stop_cohort,
        "n_bins_total": int(len(is_immobile)),
        "n_bins_immobile": int(is_immobile.sum()),
        "n_bins_fit": int(fit_mask.sum()),
    }
    return is_immobile, fit_mask, counts


#: Half-width of the peri-event spline basis, in seconds. The fit window is
#: +/-125 samples at 500 Hz; a dropped event within this of a fitted bin is an
#: event whose basis would have covered bins the model does fit.
DEFAULT_BASIS_HALF_WINDOW_S = 0.25


def count_events_covering_fit_bins(
    onsets: NDArray,
    times: NDArray,
    fit_mask: NDArray,
    *,
    basis_half_window_s: float = DEFAULT_BASIS_HALF_WINDOW_S,
) -> int:
    """How many of ``onsets`` have a fitted bin within the basis half-window.

    The cohort restriction drops events by onset, but the regressor basis spans
    +/-250 ms around each event, so an event dropped just before a stop's
    leading edge would have covered bins that *are* fit. Those bins are then fit
    with that event's contribution set to zero. This counts the events for which
    that is true, so a run can state the size of the effect rather than assume
    it away.

    Parameters
    ----------
    onsets : numpy.ndarray, shape (n_events,)
        Onset times of the dropped events, in seconds.
    times : numpy.ndarray, shape (n_time,)
        Sample times, ascending, on the same clock.
    fit_mask : numpy.ndarray of bool, shape (n_time,)
        The bins the models are fit on.
    basis_half_window_s : float
        Half-width of the peri-event basis, in seconds.

    Returns
    -------
    int
        Number of onsets with at least one fitted bin in
        ``[onset - basis_half_window_s, onset + basis_half_window_s]``.
    """
    onsets = np.asarray(onsets, dtype=float)
    if onsets.size == 0:
        return 0
    # Prefix sums make the per-event window test a pair of searchsorted lookups
    # rather than a scan over the time axis.
    fitted_before = np.concatenate(
        [[0], np.cumsum(np.asarray(fit_mask, dtype=bool), dtype=np.int64)]
    )
    first = np.searchsorted(times, onsets - basis_half_window_s, side="left")
    last = np.searchsorted(times, onsets + basis_half_window_s, side="right")
    return int((fitted_before[last] - fitted_before[first] > 0).sum())


def _basis_coverage_counts(
    dropped_non_local_onsets: NDArray,
    dropped_ripple_onsets: NDArray,
    *,
    kept_non_local: int,
    kept_ripple: int,
    time: NDArray,
    fit_mask: pd.Series | None,
    basis_half_window_s: float,
) -> dict[str, Any]:
    """The leading-edge coverage diagnostic, as four report fields.

    Named per event class and expressed both as a count and as a fraction of the
    events that were kept, because the count alone is not interpretable without
    the cohort's size.
    """
    if fit_mask is None:
        return {
            "n_nonlocal_dropped_overlapping_fit": None,
            "n_ripple_dropped_overlapping_fit": None,
            "n_nonlocal_dropped_overlapping_fit_frac_of_kept": None,
            "n_ripple_dropped_overlapping_fit_frac_of_kept": None,
        }
    mask = np.asarray(fit_mask, dtype=bool)
    n_non_local = count_events_covering_fit_bins(
        dropped_non_local_onsets,
        time,
        mask,
        basis_half_window_s=basis_half_window_s,
    )
    n_ripple = count_events_covering_fit_bins(
        dropped_ripple_onsets, time, mask, basis_half_window_s=basis_half_window_s
    )
    return {
        "n_nonlocal_dropped_overlapping_fit": n_non_local,
        "n_ripple_dropped_overlapping_fit": n_ripple,
        "n_nonlocal_dropped_overlapping_fit_frac_of_kept": (
            n_non_local / kept_non_local if kept_non_local else None
        ),
        "n_ripple_dropped_overlapping_fit_frac_of_kept": (
            n_ripple / kept_ripple if kept_ripple else None
        ),
    }


def _restrict_to_frame(
    events: pd.DataFrame, position_time: pd.Index, *, what: str
) -> tuple[pd.DataFrame, int]:
    """Drop events not wholly inside the aligned position frame, and count them.

    The canonical events were built on the decode's own time axis. A session
    whose decode and position axes were trimmed to their shared run could in
    principle carry an event in a dropped tail; such an event has no bins to be
    modelled on, so keeping it would put an onset outside the design matrix.
    The 27 trimmed sessions were checked and none has one, so a non-zero count
    here is a data change worth seeing rather than routine attrition.
    """
    if events.empty:
        return events, 0
    first, last = float(position_time[0]), float(position_time[-1])
    inside = (events.start_time.to_numpy(dtype=float) >= first) & (
        events.end_time.to_numpy(dtype=float) <= last
    )
    n_outside = int((~inside).sum())
    if n_outside:
        logger.warning(
            "%d of %d %s lie outside the aligned position frame [%f, %f] and "
            "were dropped; the decode and position axes disagree over more "
            "than an empty tail",
            n_outside,
            len(events),
            what,
            first,
            last,
        )
    return events.loc[inside].reset_index(drop=True), n_outside


def build_session_event_sets(
    data: Mapping[str, Any],
    results: Any,
    *,
    canonical_events: pd.DataFrame,
    swr_intervals: pd.DataFrame,
    cohort_stops: pd.DataFrame | None = None,
    stop_cohort: str = "at_well",
    fit_mask: pd.Series | None = None,
    basis_half_window_s: float = DEFAULT_BASIS_HALF_WINDOW_S,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build the four event sets the PFC design matrices are keyed on.

    Both populations are canonical and are read, not re-detected here: the
    non-local events are rows of the shipped event table and the SWR intervals
    are its speed-gated ``SWR_event`` labels, both loaded by
    :mod:`continuum_swr_replay.analysis.pfc.canonical_events`. The SWR split is
    the event table's own ``is_swr`` column -- strict positive-duration overlap
    with those same labels -- so a Figure 4 event and a content- or rate-family
    event with the same ``event_id`` are the same event, classified the same
    way.

    Events are restricted to the stop cohort by onset, BEFORE the split is
    read, so ``non_local_SWR`` and ``non_local_no_SWR`` partition exactly the
    events that are kept.

    Parameters
    ----------
    data : Mapping
        Session data from ``data_loaders.load_data``; only ``position_info`` is
        read.
    results : xarray.Dataset
        HPC decoding results, used only to align the position frame.
    canonical_events : pandas.DataFrame
        The session's rows of the canonical event table, from
        :func:`~continuum_swr_replay.analysis.pfc.canonical_events.load_session_canonical_events`:
        ``event_id``, ``start_time``, ``end_time``, ``is_swr``,
        ``occupancy_state``. Passed in rather than loaded here so this function
        stays testable without the shipped tables.
    swr_intervals : pandas.DataFrame
        The session's speed-gated SWR intervals, from
        :func:`~continuum_swr_replay.analysis.pfc.canonical_events.load_session_swr_intervals`:
        ``start_time``, ``end_time``.
    cohort_stops : pandas.DataFrame, optional
        Stops kept for this session; required unless ``stop_cohort`` is
        ``"all_stops"``.
    stop_cohort : str
        Cohort name; ``"all_stops"`` applies no restriction.
    fit_mask : pandas.Series of bool, optional
        The bins the models will be fit on. Used only to measure how many
        dropped events would have had basis coverage over fitted bins; when
        omitted those counts are ``None``.
    basis_half_window_s : float
        Half-width of the peri-event basis used for that measurement.

    Returns
    -------
    event_times : dict of str to pandas.DataFrame
        ``non_local_no_SWR``, ``non_local_SWR``, ``non_local``, ``SWR``.
    counts : dict
        ``stop_cohort`` and the event counts before and after the restriction
        (``n_nonlocal_before``/``n_nonlocal_after``,
        ``n_ripple_before``/``n_ripple_after``), the split
        (``n_nonlocal_swr``, ``n_nonlocal_no_swr``), the leading-edge
        coverage diagnostic ``n_nonlocal_dropped_overlapping_fit`` /
        ``n_ripple_dropped_overlapping_fit`` with their fractions of the kept
        counts (``*_frac_of_kept``), and how many canonical events and SWR
        intervals fell outside the aligned position frame
        (``n_events_outside_frame``, ``n_swr_intervals_outside_frame``).

    Raises
    ------
    ValueError
        If a stop cohort is requested without ``cohort_stops``.
    """
    # Deferred import: this helper opens Spyglass / DataJoint at import time.
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        align_results_to_position,
    )

    # Idempotent when the caller already aligned the pair (run_session does):
    # the shared run is then the whole of both axes and this is the identity.
    _, position_info, _ = align_results_to_position(results, data["position_info"])
    position_time = position_info.index

    non_local_times, n_events_outside_frame = _restrict_to_frame(
        canonical_events, position_time, what="canonical events"
    )
    swr_intervals, n_swr_outside_frame = _restrict_to_frame(
        swr_intervals, position_time, what="SWR intervals"
    )

    counts: dict[str, Any] = {
        "stop_cohort": stop_cohort,
        "n_events_outside_frame": n_events_outside_frame,
        "n_swr_intervals_outside_frame": n_swr_outside_frame,
        "n_nonlocal_before": int(len(non_local_times)),
        "n_ripple_before": int(len(swr_intervals)),
    }
    dropped_non_local_onsets = np.empty(0, dtype=float)
    dropped_ripple_onsets = np.empty(0, dtype=float)
    if stop_cohort != "all_stops":
        if cohort_stops is None:
            raise ValueError(
                f"stop_cohort={stop_cohort!r} needs cohort_stops; pass the stops "
                "load_session_stop_cohort returned, or use 'all_stops' for no "
                "stop restriction"
            )
        keep_non_local = events_in_stops(
            non_local_times.start_time.to_numpy(dtype=float), cohort_stops
        )
        keep_ripple = events_in_stops(
            swr_intervals.start_time.to_numpy(dtype=float), cohort_stops
        )
        dropped_non_local_onsets = non_local_times.start_time.to_numpy(dtype=float)[
            ~keep_non_local
        ]
        dropped_ripple_onsets = swr_intervals.start_time.to_numpy(dtype=float)[
            ~keep_ripple
        ]
        non_local_times = non_local_times.loc[keep_non_local]
        swr_intervals = swr_intervals.loc[keep_ripple]
    counts["n_nonlocal_after"] = int(len(non_local_times))
    counts["n_ripple_after"] = int(len(swr_intervals))
    counts.update(
        _basis_coverage_counts(
            dropped_non_local_onsets,
            dropped_ripple_onsets,
            kept_non_local=counts["n_nonlocal_after"],
            kept_ripple=counts["n_ripple_after"],
            time=position_time.to_numpy(dtype=float),
            fit_mask=fit_mask,
            basis_half_window_s=basis_half_window_s,
        )
    )

    # The split is read from the canonical table, not recomputed against the
    # interval list: the column already encodes strict positive-duration
    # overlap with exactly these speed-gated labels, and recomputing it here
    # would let the PFC path drift from the content and rate families again.
    is_swr = non_local_times.is_swr.to_numpy(dtype=bool)
    non_local_swr_times = non_local_times.loc[is_swr].reset_index(drop=True)
    non_local_no_swr_times = non_local_times.loc[~is_swr].reset_index(drop=True)
    counts["n_nonlocal_swr"] = int(len(non_local_swr_times))
    counts["n_nonlocal_no_swr"] = int(len(non_local_no_swr_times))

    # Both populations use the same whole-event immobility definition, so the
    # SWR-controlled decomposition cannot reintroduce a movement-spanning SWR
    # as a secondary event source.
    event_times = {
        "non_local_no_SWR": non_local_no_swr_times,
        "non_local_SWR": non_local_swr_times,
        "non_local": non_local_times,
        "SWR": swr_intervals,
    }
    return event_times, counts


#: Nested ``(full, reduced)`` model pairs the comparisons are reported for.
#: Every pair is nested -- the reduced design's columns lie in the full
#: design's span -- so a difference between them is a statement about the added
#: regressors alone. ``("SWR + non_local_no_SWR", "non_local")`` is
#: intentionally absent: it is only nested when the session has no SWR-only
#: events. The last two are the engagement decomposition, whose baseline is the
#: matched non-event control.
#:
#: One list, read by :func:`process_neuron` for its deviance fractions and by
#: the runner for the held-out differences, so the two can never describe
#: different sets of comparisons.
NESTED_MODEL_PAIRS: tuple[tuple[str, str], ...] = (
    # Individual event types vs baseline
    ("SWR", "constant"),
    ("non_local", "constant"),
    ("non_local_no_SWR", "constant"),
    ("non_local_SWR", "constant"),
    # SWR-controlled: does separating SWR and replay help?
    ("SWR + non_local_no_SWR", "constant"),
    ("SWR + non_local_no_SWR", "SWR"),
    ("SWR + non_local_no_SWR", "non_local_no_SWR"),
    # Replay-split: does splitting replay by SWR status help?
    ("non_local_SWR + non_local_no_SWR", "constant"),
    ("non_local_SWR + non_local_no_SWR", "non_local_SWR"),
    ("non_local_SWR + non_local_no_SWR", "non_local_no_SWR"),
    ("non_local_SWR + non_local_no_SWR", "non_local"),
    # Engagement: does non-SWR replay add beyond a duration-matched non-event
    # baseline drawn from the same fitted immobility bouts, and does that
    # baseline itself modulate firing?
    ("matched_control + non_local_no_SWR", "matched_control"),
    ("matched_control", "constant"),
)

#: The event classes the design matrices are keyed on. A matched control may
#: not overlap a window of any of them -- including ``SWR``, whose windows are
#: not all covered by the non-local classes.
EVENT_CLASSES = ("non_local_no_SWR", "non_local_SWR", "non_local", "SWR")

#: The class matched controls are drawn one-for-one against.
MATCHED_CONTROL_SOURCE_CLASS = "non_local"

#: Placement semantics written to every bundle.  The duration-matching window
#: for each control must avoid real events, but two controls may have
#: overlapping windows because the fitted regressor contains only their
#: distinct onsets.  Treating those auxiliary windows as occupied time makes
#: an exact count-matched null impossible in dense bouts even when hundreds of
#: distinct event-free onset bins exist.
MATCHED_CONTROL_PLACEMENT_METHOD = (
    "same_bout_real_event_free_distinct_onsets_windows_may_overlap"
)


#: Whole-draw placement attempts before the best incomplete matching is kept.
#: Each attempt visits source events in a new random order and samples directly
#: from their legal onset bins; it is not rejection sampling over illegal bins.
DEFAULT_MATCHED_CONTROL_ATTEMPTS = 50


def matched_control_seed(nwb_file_name: str, epoch_name: str) -> int:
    """Per-session seed for the matched-control draw.

    Derived from the session key rather than taken from a counter so a session
    refit on its own -- to repair one bundle, say -- redraws exactly the control
    population the batch drew, and two sessions never share a draw.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name, e.g. ``"wilbur20210329_.nwb"``.
    epoch_name : str
        Epoch name, e.g. ``"02_r1"``.

    Returns
    -------
    int
        The low 64 bits of ``sha256(f"{nwb_file_name}|{epoch_name}")``.
    """
    digest = hashlib.sha256(f"{nwb_file_name}|{epoch_name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _contiguous_true_runs(mask: NDArray) -> NDArray:
    """Half-open ``[start, stop)`` sample bounds of each contiguous True run.

    Parameters
    ----------
    mask : NDArray of bool, shape (n_time,)

    Returns
    -------
    NDArray of int, shape (n_runs, 2)
        One row per run; ``mask[start:stop]`` is all True and is bounded by
        False (or by the array's ends) on both sides.
    """
    flat = np.asarray(mask, dtype=bool)
    if flat.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    padded = np.concatenate([[False], flat, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges.reshape(-1, 2).astype(np.int64)


def _bout_containing(runs: NDArray, index: int) -> int:
    """Row of ``runs`` whose half-open bounds contain ``index``, or ``-1``.

    Parameters
    ----------
    runs : NDArray of int, shape (n_runs, 2)
        Ascending, non-overlapping half-open bounds, as
        :func:`_contiguous_true_runs` returns them.
    index : int
        Sample index to locate.

    Returns
    -------
    int
        Row index into ``runs``, or ``-1`` when ``index`` is in no run.
    """
    if len(runs) == 0 or index < 0:
        return -1
    row = int(np.searchsorted(runs[:, 0], index, side="right")) - 1
    if row < 0 or index >= int(runs[row, 1]):
        return -1
    return row


def _within_immobile(
    start_time: float, end_time: float, times: NDArray, run: NDArray
) -> bool:
    """Whether ``[start_time, end_time]`` lies inside one run's sampled span.

    Every sample between a run's first and last is True by construction, so a
    window bounded by those two times covers True bins only.

    Parameters
    ----------
    start_time, end_time : float
        Window bounds in seconds.
    times : NDArray, shape (n_time,)
        Ascending sample times in seconds.
    run : NDArray of int, shape (2,)
        One row of :func:`_contiguous_true_runs`.

    Returns
    -------
    bool
    """
    run_start, run_stop = int(run[0]), int(run[1])
    if run_stop <= run_start:
        return False
    return bool(start_time >= times[run_start] and end_time <= times[run_stop - 1])


def _overlaps_any(
    start_time: float, end_time: float, starts: NDArray, ends: NDArray
) -> bool:
    """Whether ``[start_time, end_time)`` overlaps any of the given windows.

    Windows are half-open, so abutting one exactly is not an overlap.

    Parameters
    ----------
    start_time, end_time : float
        Window bounds in seconds.
    starts, ends : NDArray, shape (n_windows,)
        Ascending, non-overlapping window bounds (as :func:`_merge_windows`
        returns them). ``ends`` is then ascending too, so the only window that
        can reach past ``start_time`` among those starting before ``end_time``
        is the last of them.

    Returns
    -------
    bool
    """
    if len(starts) == 0:
        return False
    n_starting_before = int(np.searchsorted(starts, end_time, side="left"))
    if n_starting_before == 0:
        return False
    return bool(ends[n_starting_before - 1] > start_time)


def _merge_windows(starts: NDArray, ends: NDArray) -> tuple[NDArray, NDArray]:
    """Sort and union windows into ascending, non-overlapping bounds.

    Parameters
    ----------
    starts, ends : NDArray, shape (n_windows,)
        Window bounds in seconds, in any order, possibly overlapping.

    Returns
    -------
    merged_starts, merged_ends : NDArray, shape (n_merged,)
        Ascending and disjoint; touching windows are merged.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if starts.size == 0:
        return starts, ends
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    # A window that starts after every end seen so far opens a new merged group.
    reach = np.maximum.accumulate(ends)
    opens_group = np.concatenate([[True], starts[1:] > reach[:-1]])
    group_start = np.flatnonzero(opens_group)
    return starts[opens_group], np.maximum.reduceat(ends, group_start)


class MatchedControlDraw(NamedTuple):
    """One draw of the matched non-event control, and why events went unplaced.

    The two shortfall counts are kept apart because they mean different things
    about a session. An onset outside every fitted bout is a *cohort* fact --
    the event is kept by its onset bin, which can sit outside the fit mask --
    and says nothing about how crowded the session is. A saturated bout is a
    *density* fact: that bout is too short or too full of real events to offer
    a legal onset for this event's duration, or the finite whole-draw matching
    could not give every source a distinct onset. Control duration windows may
    overlap one another; only their onsets enter the fitted design, and those
    onsets are distinct.

    Attributes
    ----------
    times : pandas.DataFrame
        Columns ``start_time``, ``end_time`` in seconds, one row per placed
        control.
    source_indices : NDArray of int, shape (n_placed,)
        Positional indices of the source events represented by ``times``, in
        the same order. This makes a placement shortfall auditable and lets a
        caller restrict both the observed statistic and subsequent null draws
        to one fixed, jointly matchable event subset.
    n_onset_outside_fitted_bout : int
        Source events whose onset bin belongs to no fitted immobility bout.
    n_bout_saturated : int
        Source events whose bout could not supply a distinct event-free onset
        for a window of their duration after ``max_attempts`` whole-draw
        matchings.
    """

    times: pd.DataFrame
    source_indices: NDArray
    n_onset_outside_fitted_bout: int
    n_bout_saturated: int

    @property
    def n_unplaced(self) -> int:
        """Source events with no control window, for either reason."""
        return self.n_onset_outside_fitted_bout + self.n_bout_saturated


def make_matched_control_times(
    event_times: Mapping[str, pd.DataFrame],
    is_immobile: pd.Series,
    rng: np.random.Generator,
    max_attempts: int = DEFAULT_MATCHED_CONTROL_ATTEMPTS,
) -> MatchedControlDraw:
    """Time-shifted-onset matched non-event control.

    The non-SWR replay regressor asks whether a neuron responds to replay. On
    its own it cannot say whether the response is to replay or to *being in the
    stretch of immobility where replay happens* -- events are not uniform over
    a stop, and neither is firing. This draws, for every real non-local event, a
    window of the same duration inside the same fitted immobility bout that
    holds no real event at all. Entered as its own event class, it turns the
    replay question into a nested one: does replay add beyond a duration-matched
    non-event baseline drawn from the same bins.

    For each source event the control onset is drawn uniformly from the legal
    sample times of its own bout, where a legal onset has these properties:

    - the control window lies entirely inside that bout (so every bin it covers
      is a bin the models are fit on),
    - it overlaps no window of any real event class, ``SWR`` included, and
    - no other control in the draw uses that exact onset bin.

    The duration-matching windows may overlap one another. Duration constrains
    where an onset may be drawn, but the GLM design is onset-only: it does not
    contain an event-in-progress box. Requiring the auxiliary windows to be
    mutually disjoint can therefore make a count-matched null impossible in a
    dense bout even when many distinct, individually event-free onsets exist.
    Directly enumerating legal onsets also avoids stochastic shortfalls from
    missing a narrow valid interval during rejection sampling.

    An event is dropped -- never placed somewhere else -- when its onset falls
    in no fitted bout at all (the cohort keeps an event by onset, and an onset
    bin can still be outside the fit mask) or when no distinct legal onset can
    be assigned in ``max_attempts`` whole-draw matchings. The returned count is
    therefore ``<=`` the number of source events, and the caller must record
    the shortfall rather than assume a one-to-one match.

    Parameters
    ----------
    event_times : Mapping[str, pandas.DataFrame]
        The event sets :func:`build_session_event_sets` returns, keyed
        ``non_local_no_SWR``, ``non_local_SWR``, ``non_local``, ``SWR``, each
        with ``start_time`` / ``end_time`` in seconds. ``non_local`` supplies
        the source events; every class supplies windows a control may not
        overlap.
    is_immobile : pandas.Series of bool, shape (n_time,)
        The bins a control may be drawn from, indexed by the session's sample
        times in seconds. Callers pass the *fit mask*
        (:func:`build_session_fit_mask`'s second return), not raw immobility:
        the control has to sample the same bins the models are fit on. A "bout"
        is a contiguous True run of this mask.
    rng : numpy.random.Generator
        Seeded generator; see :func:`matched_control_seed`.
    max_attempts : int
        Random source-order matchings attempted before the best incomplete
        matching is kept. Zero deliberately places nothing (used by the
        incomplete-null regression test).

    Returns
    -------
    MatchedControlDraw
        ``times`` holds columns ``start_time``, ``end_time`` in seconds, one
        row per *placed* control, in source-event order. Rows are deliberately
        not re-sorted by time; ``source_indices[k]`` identifies the source
        event matched by row ``k``. ``n_onset_outside_fitted_bout`` and
        ``n_bout_saturated`` split the shortfall by its two causes.

    Raises
    ------
    KeyError
        If ``event_times`` lacks one of the four event classes.
    """
    times = np.asarray(is_immobile.index, dtype=float)
    runs = _contiguous_true_runs(np.asarray(is_immobile, dtype=bool))

    source = event_times[MATCHED_CONTROL_SOURCE_CLASS]
    real_starts, real_ends = _merge_windows(
        np.concatenate(
            [
                event_times[name].start_time.to_numpy(dtype=float)
                for name in EVENT_CLASSES
            ]
        ),
        np.concatenate(
            [event_times[name].end_time.to_numpy(dtype=float) for name in EVENT_CLASSES]
        ),
    )

    if max_attempts < 0:
        raise ValueError(f"max_attempts must be non-negative, got {max_attempts}")

    source_starts = source.start_time.to_numpy(dtype=float)
    source_ends = source.end_time.to_numpy(dtype=float)
    candidate_indices: list[NDArray] = []
    onset_outside = np.zeros(len(source), dtype=bool)
    for source_index, (start, end) in enumerate(
        zip(source_starts, source_ends, strict=True)
    ):
        duration = end - start
        # The bin holding the onset, and the bout that bin belongs to.
        onset_index = int(np.searchsorted(times, start, side="right")) - 1
        run_row = _bout_containing(runs, onset_index)
        if run_row < 0:
            onset_outside[source_index] = True
            candidate_indices.append(np.empty(0, dtype=np.int64))
            continue
        run_start, run_stop = int(runs[run_row, 0]), int(runs[run_row, 1])
        # Onsets late enough in the bout that the window would run past its
        # last sample are not candidates.
        last_offset = (
            int(
                np.searchsorted(
                    times[run_start:run_stop],
                    times[run_stop - 1] - duration,
                    side="right",
                )
            )
            - 1
        )
        if last_offset < 0:
            candidate_indices.append(np.empty(0, dtype=np.int64))
            continue

        indices = np.arange(run_start, run_start + last_offset + 1, dtype=np.int64)
        starts = times[indices]
        ends = starts + duration
        # For each candidate, the only merged real window that can overlap is
        # the last one starting before candidate_end.  This vectorized search
        # enumerates all legal onset bins without a candidate x event matrix.
        n_starting_before = np.searchsorted(real_starts, ends, side="left")
        overlaps = np.zeros(len(indices), dtype=bool)
        has_predecessor = n_starting_before > 0
        overlaps[has_predecessor] = (
            real_ends[n_starting_before[has_predecessor] - 1] > starts[has_predecessor]
        )
        candidate_indices.append(indices[~overlaps])

    # This is a randomized matching between source events and legal onset
    # bins.  Windows are allowed to overlap, but onset bins are not: duplicate
    # onsets would be summed by ``event_onset_counts`` and would not represent
    # the requested number of events.  Retrying source orders avoids a greedy
    # early choice starving a later event whose legal set is narrower.
    target = int(sum(len(candidates) > 0 for candidates in candidate_indices))
    best: dict[int, int] = {}
    for _ in range(max_attempts):
        used = np.zeros(len(times), dtype=bool)
        placed: dict[int, int] = {}
        for source_index in rng.permutation(len(source)):
            candidates = candidate_indices[int(source_index)]
            available = candidates[~used[candidates]]
            if not len(available):
                continue
            chosen = int(available[int(rng.integers(len(available)))])
            placed[int(source_index)] = chosen
            used[chosen] = True
        if len(placed) > len(best):
            best = placed
        if len(best) == target:
            break

    placed_source_indices = sorted(best)
    control_starts = [float(times[best[index]]) for index in placed_source_indices]
    control_ends = [
        control_start + (source_ends[index] - source_starts[index])
        for index, control_start in zip(
            placed_source_indices, control_starts, strict=True
        )
    ]
    n_onset_outside_fitted_bout = int(onset_outside.sum())
    n_bout_saturated = (
        len(source) - n_onset_outside_fitted_bout - len(placed_source_indices)
    )

    if n_onset_outside_fitted_bout or n_bout_saturated:
        logger.warning(
            "%d of %d matched controls could not be placed in %d attempts: "
            "%d events have an onset in no fitted bout, %d sit in a bout with "
            "no free window of the event's duration",
            n_onset_outside_fitted_bout + n_bout_saturated,
            len(source),
            max_attempts,
            n_onset_outside_fitted_bout,
            n_bout_saturated,
        )
    return MatchedControlDraw(
        times=pd.DataFrame(
            {"start_time": control_starts, "end_time": control_ends}, dtype=float
        ),
        source_indices=np.asarray(placed_source_indices, dtype=np.int64),
        n_onset_outside_fitted_bout=n_onset_outside_fitted_bout,
        n_bout_saturated=n_bout_saturated,
    )


def shipped_fit_counts(
    prefitted_immobility: list[dict[str, Any] | None],
) -> dict[str, int]:
    """How many of a session's shipped immobility fits never settled.

    A fit that runs to ``maxiter`` is not an estimate: ``process_neuron`` marks
    it non-estimable for that neuron and every statistic that reads it is NaN
    (the neuron's other models are reported as usual). The NaNs say a number is
    missing but not how many, or over how many neurons, so unless the count is
    taken here -- once, over the results that will actually be read -- there is
    no record of what a session dropped.

    A neuron with no fit at all (no spikes on the fitted bins) enters neither
    total, which is why ``n_model_fits`` is reported beside the failures rather
    than assumed to be neurons x models.

    Parameters
    ----------
    prefitted_immobility : list of (dict or None)
        Per neuron, the results keyed by model name, as
        :func:`_prefit_session` assembles them.

    Returns
    -------
    dict[str, int]
        ``n_model_fits``, ``n_nonconverged_model_fits``,
        ``n_neurons_with_nonconverged_model``.
    """
    n_model_fits = 0
    n_nonconverged = 0
    n_neurons_nonconverged = 0
    for fits in prefitted_immobility:
        if fits is None:
            continue
        failures = sum(
            not bool(getattr(fit, "converged", True)) for fit in fits.values()
        )
        n_model_fits += len(fits)
        n_nonconverged += failures
        n_neurons_nonconverged += failures > 0
    return {
        "n_model_fits": int(n_model_fits),
        "n_nonconverged_model_fits": int(n_nonconverged),
        "n_neurons_with_nonconverged_model": int(n_neurons_nonconverged),
    }


def _prefit_session(
    spike_times: list[NDArray],
    time: NDArray,
    is_immobile: pd.Series,
    fit_mask: pd.Series,
    immobility_design_matrices: Mapping[str, NDArray],
    theta_phase: pd.Series,
    logger: logging.Logger,
    penalty_sigma: float | None = None,
    counts: NDArray | None = None,
) -> tuple[
    list[dict[str, Any] | None],
    list[tuple[Any, Any] | None],
    dict[str, int],
    tuple[NDArray, NDArray],
    dict[str, int],
]:
    """Fit every neuron of a session against every design, in batches.

    Parameters
    ----------
    spike_times : list of NDArray
        Spike times per neuron, in seconds.
    time : NDArray, shape (n_time,)
        Time bin starts, the axis the designs and masks are on.
    is_immobile, fit_mask : pd.Series of bool, shape (n_time,)
        The running comparison's mask and the immobility models' mask.
    immobility_design_matrices : Mapping[str, NDArray]
        Full-length designs keyed by model name.
    theta_phase : pd.Series
        Theta phase timeseries, on its own clock.
    logger : logging.Logger
    penalty_sigma : float or None
        Prior standard deviation of an event-basis coefficient, on the log-rate
        scale, for the immobility models. ``None`` fits them unpenalised. The
        two theta models carry no event basis and are never penalised.
    counts : NDArray, shape (n_time, n_units), optional
        Spike counts already binned on ``time``. Supplying the session-wide
        matrix avoids binning the same spikes again for scoring and for each
        per-neuron summary.

    Returns
    -------
    prefitted_immobility : list of dict
        Per neuron, the immobility results keyed by model name.
    prefitted_theta : list of tuple
        Per neuron, ``(constant_result, theta_result)`` for the theta LR test.
    design_ranks : dict[str, int]
        Rank of each design on the fit mask.
    theta_design : tuple
        ``(joint_mask, design_matrix)``, built once for the session.
    fit_counts : dict[str, int]
        What the shipped fits say about themselves: ``n_model_fits`` (neuron x
        model cells that produced a fit at all), ``n_nonconverged_model_fits``
        and ``n_neurons_with_nonconverged_model``. A cell that never settled is
        reported per model rather than discarding the neuron, so the only place
        that fact survives is here.
    """
    if counts is None:
        counts = np.column_stack(
            [get_spikecount_per_time_bin(neuron, time) for neuron in spike_times]
        ).astype(float)
    else:
        counts = np.asarray(counts, dtype=float)
        if counts.shape != (len(time), len(spike_times)):
            raise ValueError(
                "counts must have shape (n_time, n_units); got "
                f"{counts.shape}, expected {(len(time), len(spike_times))}"
            )

    fit_rows = np.asarray(fit_mask, dtype=bool)
    logger.info(
        "Batch-fitting %d neurons x %d models on %d bins...",
        counts.shape[1],
        len(immobility_design_matrices),
        int(fit_rows.sum()),
    )
    batched = fit_models_batched(
        {
            model_name: design[fit_rows]
            for model_name, design in immobility_design_matrices.items()
        },
        counts[fit_rows],
        model_penalties(immobility_design_matrices, penalty_sigma),
    )
    design_ranks = {name: result.rank for name, result in batched.items()}

    # ``process_neuron`` derives the sampling frequency from the time axis and
    # passes it to the theta design's reindex tolerance. Derive it the same way
    # here, so the bins the batch is fit on are exactly the bins each neuron's
    # own call will read the regularized fit off.
    sampling_frequency = 1 / np.median(np.diff(time))
    joint_mask, theta_design = build_theta_design(
        theta_phase, is_immobile, sampling_frequency=sampling_frequency
    )
    logger.info(
        "Batch-fitting %d neurons x 2 theta models on %d running bins...",
        counts.shape[1],
        int(joint_mask.sum()),
    )
    theta_constant, theta_full = _fit_theta_models(
        theta_design, counts[joint_mask], fitter="jax"
    )

    prefitted_immobility: list[dict[str, Any]] = [
        {name: result.for_unit(i) for name, result in batched.items()}
        for i in range(counts.shape[1])
    ]
    prefitted_theta: list[tuple[Any, Any]] = [
        (theta_constant.for_unit(i), theta_full.for_unit(i))
        for i in range(counts.shape[1])
    ]

    return (
        prefitted_immobility,
        prefitted_theta,
        design_ranks,
        (joint_mask, theta_design),
        shipped_fit_counts(prefitted_immobility),
    )


def _stack_fit_diagnostics(
    per_neuron: list[dict[str, Any]],
    design_matrices: Mapping[str, NDArray],
) -> dict[str, dict[str, NDArray]]:
    """Stack per-neuron fit summaries into per-model arrays over neurons.

    A neuron whose fit failed contributes NaN/absent entries rather than
    shortening the arrays, so every model's arrays are indexed by neuron.

    Parameters
    ----------
    per_neuron : list of dict
        One :func:`_fit_diagnostics` mapping per neuron; ``{}`` where the fit
        raised.
    design_matrices : Mapping[str, NDArray]
        The designs, for the model names and their widths.

    Returns
    -------
    dict[str, dict[str, NDArray]]
    """
    n_neurons = len(per_neuron)
    stacked: dict[str, dict[str, NDArray]] = {}
    for name, design in design_matrices.items():
        n_features = design.shape[1]
        params = np.full((n_neurons, n_features), np.nan)
        scalars = {
            key: np.full(n_neurons, np.nan)
            for key in ("deviance", "llf", "aic", "df_resid", "df_model")
        }
        converged = np.zeros(n_neurons, dtype=bool)
        n_iter = np.full(n_neurons, -1, dtype=int)
        for index, diagnostics in enumerate(per_neuron):
            record = (diagnostics or {}).get(name)
            if record is None:
                continue
            params[index] = record["params"]
            for key, values in scalars.items():
                values[index] = record[key]
            converged[index] = record["converged"]
            n_iter[index] = record["n_iter"]
        stacked[name] = {
            "params": params,
            "converged": converged,
            "n_iter": n_iter,
            **scalars,
        }
    return stacked


def run_session(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "mPFC",
    sampling_frequency: float = 500,
    immobility_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    run_parallel: bool = True,
    stop_cohort: str = "at_well",
    cohort_stops: pd.DataFrame | None = None,
    fitter: str = "jax",
    fit_diagnostics_out: dict[str, Any] | None = None,
    penalty_sigma: float | None = None,
    scoring_out: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, NDArray, NDArray, NDArray, NDArray, NDArray, dict[str, Any]]:
    """Run PFC GLM analysis for all neurons in a single session.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name (e.g., "wilbur20210329_.nwb").
    epoch_name : str
        Epoch name (e.g., "02_r1").
    brain_area : str
        Brain area to analyze ("mPFC" or "OFC").
    sampling_frequency : float
        Sampling frequency in Hz.
    immobility_threshold : float
        Speed threshold for immobility classification (cm/s). The rule is
        strictly ``<``, on the fit mask and on the SWR speed gate alike. It
        gates the SWR intervals the canonical ``is_swr`` column was computed
        against, so a value other than
        :data:`~continuum_swr_replay.analysis.constants.DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC`
        is refused rather than silently applied to one side of the split only.
    non_local_threshold : float
        Probability threshold the events were cut at. It is not applied here:
        non-local events are read from the canonical event table, which is cut
        at :data:`~continuum_swr_replay.analysis.constants.DEFAULT_NON_LOCAL_THRESHOLD`,
        and a different value is refused rather than silently ignored. It does
        select the cached decode frame the SWR labels are read from.
    run_parallel : bool
        Whether to use Dask for parallel neuron processing.
    stop_cohort : str
        Population to fit. ``"at_well"`` (canonical) keeps only immobility bins
        and events inside the well-anchored stops the hippocampal rate and
        content ladders are read on; ``"all_stops"`` applies no stop restriction
        at all, fitting every immobile bin and every speed-eligible event in the
        epoch. Under ``"all_stops"`` the fit mask and the event sets reproduce
        the pre-cohort pipeline exactly, and so does every fitted number when
        ``fitter="statsmodels"`` -- the batched fitter reaches the same point
        to the equivalence audit's tolerances rather than bit-for-bit, because
        it is a different arithmetic order on a different device. What differs
        in either case is the shape of the output -- the stats table gains the
        provenance columns ``stop_cohort`` and
        ``immobility_avg_firing_rate_all_immobile``, and this function returns a
        7-tuple rather than a 5-tuple.
    cohort_stops : pd.DataFrame, optional
        The session's kept stops from
        :func:`~continuum_swr_replay.analysis.pfc.cohort.load_session_stop_cohort`.
        Required unless ``stop_cohort`` is ``"all_stops"``.
    fitter : {"jax", "statsmodels"}
        How the GLMs are fit. ``"jax"`` (default) fits every neuron of the
        session against each design in one batched IRLS -- every neuron shares
        the design, so the per-neuron loop was refitting the same weighted
        cross-products once per neuron. ``"statsmodels"`` keeps the original
        per-neuron path, unchanged, for the equivalence audit. The two
        converge to the same point by construction (same starting values, same
        step, same convergence rule); the audit measures how close on real
        sessions.
    fit_diagnostics_out : dict, optional
        When given, filled with ``{model_name: {quantity: array over neurons}}``
        for the fitter equivalence audit. Never part of the shipped bundle.
    penalty_sigma : float or None
        Prior standard deviation of an event-basis coefficient, on the log-rate
        scale, for the batched fitter. ``None`` fits the models unpenalised.
        The statsmodels backend has no penalised path and refuses a value.
    scoring_out : dict, optional
        When given, filled with everything an out-of-fold score and its null
        need from this session: ``fit_mask`` (bool Series on the session's time
        axis), ``counts`` (``(n_time, n_units)`` spike counts in unit order),
        ``designs`` (the full-length design matrices by model name),
        ``spline_basis``, ``lag_time``, ``lags`` and ``event_times`` (the
        matched control included, as drawn here). It is the same material the
        fits were built from, handed over rather than rebuilt: a caller that
        reassembled it from the same functions would pay a second load of the
        session and could drift from what was actually fit. Left empty for a
        session with no units in ``brain_area``.

    Returns
    -------
    neuron_stats : pd.DataFrame
        Per-neuron statistics. Shape (n_neurons, n_stats). Includes GLM
        comparison results, theta modulation and firing rates.
    swr_modulation : NDArray
        SWR peri-event modulation from SWR-controlled model.
        Shape (n_lag_times, n_neurons).
    no_swr_modulation : NDArray
        Non-SWR replay modulation from SWR-controlled model, controlling
        for all whole-event-speed-eligible ripples. Shape (n_lag_times, n_neurons).
    non_local_swr_modulation : NDArray
        SWR-coincident replay modulation from replay-split model.
        Shape (n_lag_times, n_neurons).
    non_local_no_swr_modulation : NDArray
        Non-SWR replay modulation from replay-split model, controlling
        for replay-coincident ripples only.
        Shape (n_lag_times, n_neurons).
    matched_control_modulation : NDArray
        Matched non-event baseline modulation from the engagement model.
        Shape (n_lag_times, n_neurons).
    cohort_info : dict
        How the population was cut and which event definition it was cut on:
        ``stop_cohort``, the bin counts (``n_bins_total``, ``n_bins_immobile``,
        ``n_bins_fit``), the event counts before and after the restriction, and
        the canonical-event provenance (``is_swr_definition``,
        ``immobility_inequality``, ``canonical_event_table_path`` /
        ``_sha256``, ``swr_interval_source``, ``data_over_time_path`` /
        ``_sha256``), plus the matched non-event control's draw
        (``matched_control_seed``, ``n_matched_control``,
        ``n_matched_control_unplaced`` and the two causes that total,
        ``n_matched_control_onset_outside_fitted_bout`` and
        ``n_matched_control_bout_saturated``). Written to the run's provenance
        sidecar so an artifact states both the population and the events it
        was fit on. Also records how the models were fit: ``fitter``,
        ``fitter_tol`` and, for the batched fitter, ``jax_backend`` and
        ``jax_version``.

    Raises
    ------
    ValueError
        If ``non_local_threshold`` or ``immobility_threshold`` differs from the
        threshold the canonical events were cut and speed-gated at; neither can
        be re-cut at run time.
    UnconfiguredDeviceError
        With ``fitter="jax"``, if no JAX device was chosen first. Call
        :func:`~continuum_swr_replay.analysis.pfc.gpu.configure_jax_gpu` (the
        runner does) or pin ``JAX_PLATFORMS=cpu``; otherwise JAX would take
        every card on a shared host.

    Notes
    -----
    Non-local events and their SWR classification are the canonical ones: the
    rows of ``Processed-Data/replay_content/event_table.parquet`` for this
    session, and that table's ``is_swr`` column (strict positive-duration
    overlap with the speed-gated ``SWR_event`` labels). Figure 4 therefore
    describes the same ``event_id``s as the content and rate families.

    The matched non-event control is drawn here rather than by the caller so
    every fit on this path carries one, with the seed
    :func:`matched_control_seed` derives from the session key -- a refit of one
    session redraws the batch's control population exactly.
    """
    # Deferred imports: these helpers open Spyglass / DataJoint at
    # import time. Keeping them function-local lets the rest of the
    # module (pure GLM helpers + design-matrix builders) stay
    # offline-importable for unit tests.
    from continuum_swr_replay.analysis.hippocampus import load_results
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        align_results_to_position,
    )
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        DEFAULT_EVENT_TABLE_PATH,
        canonical_event_provenance,
        load_session_canonical_events,
        load_session_swr_intervals,
        session_data_over_time_path,
    )
    from continuum_swr_replay.data_loaders import load_data

    _check_fitter(fitter)
    if penalty_sigma is not None and fitter != "jax":
        raise ValueError(
            f"penalty_sigma={penalty_sigma!r} needs fitter='jax'; the "
            f"{fitter!r} backend fits unpenalised models only"
        )
    if immobility_threshold != DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC:
        raise ValueError(
            f"immobility_threshold={immobility_threshold!r} is not the "
            "threshold the canonical events were speed-gated at "
            f"({DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC}). The event table's "
            "``is_swr`` column and the SWR intervals it was computed against "
            "are cut at that one threshold, so re-gating the SWR population "
            "here would split the events on a population the canonical column "
            "never saw."
        )
    if non_local_threshold != DEFAULT_NON_LOCAL_THRESHOLD:
        raise ValueError(
            f"non_local_threshold={non_local_threshold!r} is not the threshold "
            f"the canonical event table is cut at "
            f"({DEFAULT_NON_LOCAL_THRESHOLD}). Non-local events are read from "
            "that table, not re-detected, so a different threshold cannot be "
            "honoured here."
        )

    logger = logging.getLogger(
        f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{brain_area}"
    )

    logger.info("Loading HPC decoding results...")
    results, _, _ = load_results(
        nwb_file_name,
        epoch_name,
        decoding_brain_area="HPC",
        load_detector=False,
    )

    logger.info("Loading session data...")
    data = load_data(nwb_file_name, epoch_name)

    # Decodes saved before get_position_info was trimmed carry a few more time
    # bins than load_data now returns. Intersect the two axes once, here, so
    # every downstream consumer (fit mask, event sets, design matrices, spike
    # binning) sees the same shared run.
    results, position_info, alignment_counts = align_results_to_position(
        results, data["position_info"]
    )
    data = {**data, "position_info": position_info}
    if alignment_counts["dropped_at"] != "none":
        logger.warning(
            "Trimmed decode/position axes to their shared timestamps: dropped "
            "%d results bins (%.3fs) and %d position rows (%.3fs) at the %s",
            alignment_counts["n_results_bins_dropped"],
            alignment_counts["dropped_seconds_results"],
            alignment_counts["n_position_rows_dropped"],
            alignment_counts["dropped_seconds_position"],
            alignment_counts["dropped_at"],
        )

    position_time = position_info.index
    time = position_time.to_numpy(dtype=float)

    # Fit with wider lag window (±250ms) so spline boundary artifacts
    # fall outside the reported window (±200ms). The returned modulation
    # arrays are trimmed to the reporting window after fitting.
    fit_lags = np.arange(-125, 126, 1)
    report_lags = np.arange(-100, 101, 1)

    is_immobile, fit_mask, bin_counts = build_session_fit_mask(
        data["position_info"],
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        immobility_threshold=immobility_threshold,
    )
    logger.info(
        "Fit bins: %d immobile, %d in cohort (%.3f of %d time bins)",
        bin_counts["n_bins_immobile"],
        bin_counts["n_bins_fit"],
        bin_counts["n_bins_fit"] / bin_counts["n_bins_total"],
        bin_counts["n_bins_total"],
    )

    logger.info("Loading canonical non-local events (stop cohort: %s)...", stop_cohort)
    canonical_events, event_table_sha256 = load_session_canonical_events(
        nwb_file_name, epoch_name
    )
    swr_intervals, dot_sha256 = load_session_swr_intervals(
        nwb_file_name,
        epoch_name,
        non_local_threshold=non_local_threshold,
        speed_threshold=immobility_threshold,
    )
    event_provenance = canonical_event_provenance(
        event_table_path=DEFAULT_EVENT_TABLE_PATH,
        event_table_sha256=event_table_sha256,
        data_over_time_path=session_data_over_time_path(
            nwb_file_name, epoch_name, non_local_threshold=non_local_threshold
        ),
        data_over_time_sha256=dot_sha256,
    )

    event_times, event_counts = build_session_event_sets(
        data,
        results,
        canonical_events=canonical_events,
        swr_intervals=swr_intervals,
        cohort_stops=cohort_stops,
        stop_cohort=stop_cohort,
        fit_mask=fit_mask,
        basis_half_window_s=float(np.abs(fit_lags).max()) / sampling_frequency,
    )
    # The matched non-event control is drawn after the cohort restriction and
    # from the fit mask, so its windows sample exactly the bins the models are
    # fit on and avoid exactly the events that survived the restriction.
    control_seed = matched_control_seed(nwb_file_name, epoch_name)
    control_draw = make_matched_control_times(
        event_times, fit_mask, np.random.default_rng(control_seed)
    )
    event_times["matched_control"] = control_draw.times
    n_matched_control = int(len(control_draw.times))
    cohort_info = {
        **bin_counts,
        **event_counts,
        **event_provenance,
        **fitter_provenance(fitter),
        # Filled in after the fits: how many of the shipped fits ran to
        # ``maxiter`` without settling. ``None`` when no neuron was fit at all,
        # or on the statsmodels path, which never reaches the batched fitter.
        "n_model_fits": None,
        "n_nonconverged_model_fits": None,
        "n_neurons_with_nonconverged_model": None,
        "matched_control_seed": control_seed,
        "n_matched_control": n_matched_control,
        "n_matched_control_unplaced": control_draw.n_unplaced,
        "n_matched_control_onset_outside_fitted_bout": (
            control_draw.n_onset_outside_fitted_bout
        ),
        "n_matched_control_bout_saturated": control_draw.n_bout_saturated,
        "alignment": alignment_counts,
    }
    logger.info(
        "Events in cohort: %d/%d non-local (%d with SWR, %d without SWR), %d/%d SWRs",
        event_counts["n_nonlocal_after"],
        event_counts["n_nonlocal_before"],
        event_counts["n_nonlocal_swr"],
        event_counts["n_nonlocal_no_swr"],
        event_counts["n_ripple_after"],
        event_counts["n_ripple_before"],
    )
    logger.info(
        "Dropped events whose basis would have covered fitted bins: "
        "%s non-local, %s SWR",
        event_counts["n_nonlocal_dropped_overlapping_fit"],
        event_counts["n_ripple_dropped_overlapping_fit"],
    )
    logger.info(
        "Matched non-event controls: %d placed, %d unplaced "
        "(%d onset outside a fitted bout, %d bout saturated; seed %d)",
        cohort_info["n_matched_control"],
        cohort_info["n_matched_control_unplaced"],
        cohort_info["n_matched_control_onset_outside_fitted_bout"],
        cohort_info["n_matched_control_bout_saturated"],
        control_seed,
    )
    logger.info(
        "Canonical events outside the aligned position frame: %d (%d SWR intervals)",
        event_counts["n_events_outside_frame"],
        event_counts["n_swr_intervals_outside_frame"],
    )

    logger.info("Building design matrices...")
    immobility_design_matrices, spline_basis, lag_time = _make_design_matrices(
        event_times,
        data["position_info"],
        sampling_frequency=sampling_frequency,
        lags=fit_lags,
    )

    if brain_area not in data["spike_times"] or not data["spike_times"][brain_area]:
        logger.warning("No %s neurons found for this session", brain_area)
        empty = np.empty((0, 0))
        return pd.DataFrame(), empty, empty, empty, empty, empty, cohort_info

    spike_times = data["spike_times"][brain_area]
    # The JAX fit, held-out scorer and serial per-neuron summaries all consume
    # the same binned response. Bin it once instead of repeating the histogram
    # pass up to three times over every unit.
    session_counts = None
    if fitter == "jax" or scoring_out is not None:
        session_counts = np.column_stack(
            [get_spikecount_per_time_bin(neuron, time) for neuron in spike_times]
        ).astype(float)

    if scoring_out is not None:
        # Handed over, not rebuilt: an out-of-fold score has to be taken on the
        # bins, events and designs the shipped fit used, and reassembling them
        # from the same functions would cost a second load of the session and
        # could drift from what was fit.
        scoring_out.update(
            {
                "fit_mask": fit_mask,
                "counts": session_counts,
                "designs": immobility_design_matrices,
                "spline_basis": spline_basis,
                "lag_time": lag_time,
                "lags": fit_lags,
                "event_times": event_times,
            }
        )

    n_neurons = len(spike_times)
    logger.info(
        "Fitting GLMs for %d %s neurons (fitter=%s, parallel=%s)...",
        n_neurons,
        brain_area,
        fitter,
        run_parallel,
    )

    prefitted_immobility: list[dict[str, Any]] | None = None
    prefitted_theta: list[tuple[Any, Any]] | None = None
    design_ranks: dict[str, int] | None = None
    theta_design: tuple[NDArray, NDArray] | None = None
    # The statsmodels path fits inside ``process_neuron`` and never reaches the
    # batched fitter, so it has nothing to say here; the sidecar carries
    # ``None`` rather than a zero that would read as "nothing failed".
    fit_counts: dict[str, int | None] = {
        "n_model_fits": None,
        "n_nonconverged_model_fits": None,
        "n_neurons_with_nonconverged_model": None,
    }
    if fitter == "jax":
        # Every neuron shares the design; only the response column differs. Fit
        # the whole session against each design once, then hand each neuron its
        # own slice. The rank of each design is a property of the design, so it
        # is taken from the same batched fit rather than recomputed per neuron.
        (
            prefitted_immobility,
            prefitted_theta,
            design_ranks,
            theta_design,
            fit_counts,
        ) = _prefit_session(
            spike_times,
            time,
            is_immobile,
            fit_mask,
            immobility_design_matrices,
            data["theta_phase"],
            logger=logger,
            penalty_sigma=penalty_sigma,
            counts=session_counts,
        )

    def _neuron_args(index: int) -> dict[str, Any]:
        prefit = None if prefitted_immobility is None else prefitted_immobility[index]
        return {
            "logger_name": logger.name,
            "fit_mask": fit_mask,
            "fitter": fitter,
            "design_ranks": design_ranks,
            "theta_design": theta_design,
            "prefitted_immobility": prefit,
            "prefitted_theta": (
                None if prefitted_theta is None else prefitted_theta[index]
            ),
            "fit_diagnostics_out": (None if fit_diagnostics_out is None else {}),
            # Passing a full-length count vector to each Dask task is more
            # expensive than re-binning its sparse spikes. The production
            # runner is serial here; that path reuses the session matrix.
            "prebinned_spike_count": (
                None
                if run_parallel or session_counts is None
                else session_counts[:, index]
            ),
        }

    neuron_diagnostics: list[dict[str, Any]] = []
    if run_parallel:
        if fit_diagnostics_out is not None:
            # Dask returns copies, so an out-parameter cannot be filled through
            # a delayed call; the diagnostics are a serial-path option.
            raise ValueError("fit_diagnostics_out requires run_parallel=False")
        tasks = [
            dask.delayed(process_neuron)(
                neuron_spike_times,
                is_immobile,
                data["theta_phase"],
                time,
                immobility_design_matrices,
                spline_basis,
                lag_time,
                **_neuron_args(i),
            )
            for i, neuron_spike_times in enumerate(spike_times)
        ]

        result = dask.compute(*tasks)
    else:
        result = []
        for i, neuron_spike_times in enumerate(spike_times):
            logger.info("  Neuron %d/%d...", i + 1, n_neurons)
            kwargs = _neuron_args(i)
            neuron_diagnostics.append(kwargs["fit_diagnostics_out"])
            result.append(
                process_neuron(
                    neuron_spike_times,
                    is_immobile,
                    data["theta_phase"],
                    time,
                    immobility_design_matrices,
                    spline_basis,
                    lag_time,
                    **kwargs,
                )
            )
    if fit_diagnostics_out is not None:
        fit_diagnostics_out.update(
            _stack_fit_diagnostics(neuron_diagnostics, immobility_design_matrices)
        )
    cohort_info.update(fit_counts)
    if fit_counts["n_nonconverged_model_fits"]:
        logger.info(
            "%d of %d shipped model fits ran to maxiter without converging, "
            "over %d of %d neurons. Each is non-estimable for its neuron: "
            "every statistic that reads it is NaN, and this count is the only "
            "record of what was dropped.",
            fit_counts["n_nonconverged_model_fits"],
            fit_counts["n_model_fits"],
            fit_counts["n_neurons_with_nonconverged_model"],
            n_neurons,
        )

    (
        neuron_stats_list,
        swr_modulation_list,
        no_swr_modulation_list,
        non_local_swr_modulation_list,
        non_local_no_swr_modulation_list,
        matched_control_modulation_list,
    ) = zip(*result, strict=True)

    neuron_stats = pd.DataFrame(list(neuron_stats_list))
    swr_modulation = np.stack(swr_modulation_list, axis=1)
    no_swr_modulation = np.stack(no_swr_modulation_list, axis=1)
    non_local_swr_modulation = np.stack(non_local_swr_modulation_list, axis=1)
    non_local_no_swr_modulation = np.stack(non_local_no_swr_modulation_list, axis=1)
    matched_control_modulation = np.stack(matched_control_modulation_list, axis=1)

    # Trim modulation arrays from fit window (±250ms) to report window (±200ms)
    # to remove spline boundary artifacts at the edges
    report_mask = np.isin(
        np.round(lag_time, 6), np.round(report_lags / sampling_frequency, 6)
    )
    swr_modulation = swr_modulation[report_mask]
    no_swr_modulation = no_swr_modulation[report_mask]
    non_local_swr_modulation = non_local_swr_modulation[report_mask]
    non_local_no_swr_modulation = non_local_no_swr_modulation[report_mask]
    matched_control_modulation = matched_control_modulation[report_mask]

    neuron_stats["nwb_file_name"] = nwb_file_name
    neuron_stats["epoch_name"] = epoch_name
    neuron_stats["brain_area"] = brain_area
    neuron_stats["stop_cohort"] = stop_cohort

    logger.info("Session complete: %d neurons analyzed", len(neuron_stats))

    return (
        neuron_stats,
        swr_modulation,
        no_swr_modulation,
        non_local_swr_modulation,
        non_local_no_swr_modulation,
        matched_control_modulation,
        cohort_info,
    )
