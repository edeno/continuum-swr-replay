"""Deviance and modulation statistics for the PFC GLMs.

Deviance fractions for the nested model pairs, and the peak peri-event
log-rate modulation, its standard error and the separation checks.
"""

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray


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
#:
#: Read through :func:`curve_coefficients`; ``session._modulation_curves``
#: slices the same blocks.
CURVE_SOURCES = {
    "SWR": ("SWR + non_local_no_SWR", 0),
    "no_SWR": ("SWR + non_local_no_SWR", 1),
    "non_local_SWR": ("non_local_SWR + non_local_no_SWR", 1),
    "non_local_no_SWR": ("non_local_SWR + non_local_no_SWR", 0),
    "matched_control": ("matched_control + non_local_no_SWR", 0),
}


def curve_coefficients(curve: str, n_basis: int) -> tuple[str, slice]:
    """The model a :data:`CURVE_SOURCES` curve is read from, and its parameter slice.

    Parameter 0 is the intercept; basis block ``b`` is the ``n_basis``
    parameters after it, starting at ``1 + b * n_basis``.
    """
    model, block = CURVE_SOURCES[curve]
    start = 1 + block * n_basis
    return model, slice(start, start + n_basis)


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
