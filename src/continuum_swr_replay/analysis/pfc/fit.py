"""Fitters, penalties and fit diagnostics for the PFC GLMs.

The per-unit statsmodels IRLS and the batched JAX backend, the ridge
penalties on the event-basis coefficients, the theta-modulation fit, and
the per-model fit summaries the fitter equivalence audit reads.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.design import build_theta_design
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

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


def test_theta_modulation(
    theta_phase: pd.Series,
    is_immobile: pd.Series,
    neuron_spike_count: NDArray,
    sampling_frequency: float = SAMPLING_FREQUENCY,
    *,
    design: tuple[NDArray, NDArray] | None = None,
) -> tuple[float, float]:
    """Theta modulation depth and preferred phase of a neuron's running bins.

    A ridge-regularized fit (``alpha=1e-20``, ``L1_wt=0``) supplies both, as it
    always has: statsmodels' coordinate descent lands ~1e-4 from the
    unregularized MLE, so ``theta_mod`` and ``theta_preferred_phase`` are that
    estimator's and swapping an IRLS backend must not quietly change them.

    No hypothesis test is read off these fits. The bundles' inference is the
    blocked held-out deviance of the immobility models, and an asymptotic
    likelihood ratio over autocorrelated 2 ms bins is not a quantity this
    estimator reports, so the two unregularized fits it would need are not run.

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
    design : tuple, optional
        ``(joint_mask, design_matrix)`` from :func:`build_theta_design`. The
        design is the same for every unit of a session and reindexing the theta
        phase onto the time axis is not free, so a session builds it once and
        passes it here; omitted, it is built from ``theta_phase``.

    Returns
    -------
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

    theta_glm_results = sm.GLM(
        neuron_spike_count[joint_mask],
        design_matrix,
        family=sm.families.Poisson(),
    ).fit_regularized(alpha=1e-20, L1_wt=0.0)

    modulation = np.sqrt(
        theta_glm_results.params[1] ** 2 + theta_glm_results.params[2] ** 2
    )
    preferred_phase = np.arctan2(
        theta_glm_results.params[2], theta_glm_results.params[1]
    )

    return modulation, preferred_phase


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
        Backend. Ignored when ``prefitted`` is given. The default is the
        unpenalised control; the canonical path fits with ``fitter="jax"`` and
        ``penalties=model_penalties(design_matrices, sigma)``.
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


def _fit_diagnostics(results: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-model fit summary, read the same way off either backend's results.

    The quantities the fitter equivalence audit is run on: the parameters and
    the three fit criteria, plus the rank and the convergence record that say
    whether the numbers mean anything. A result with no ``converged`` flag is
    recorded as not converged.

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
            "converged": bool(getattr(result, "converged", False)),
            "n_iter": int(
                getattr(result, "n_iter", None)
                if getattr(result, "n_iter", None) is not None
                else (history or {}).get("iteration", -1)
            ),
        }
    return diagnostics


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
