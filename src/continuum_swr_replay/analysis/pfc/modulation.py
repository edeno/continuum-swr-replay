"""Deviance and modulation statistics for the PFC GLMs.

Deviance fractions for the nested model pairs, the peak peri-event log-rate
modulation, its standard error and the separation checks, one unit's kernel with
its pointwise standard error, and a unit's firing rate around events.
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


def in_report_window(
    lag_time: NDArray, half_window_s: float = REPORT_HALF_WINDOW_S
) -> NDArray:
    """Which lags of a fit window's lag axis are reported.

    The one rule for the reported lag set: ``|lag| <= half_window_s``, with a
    1e-12 s tolerance so a lag computed as ``k / sampling_frequency`` that
    lands on the edge is kept.

    Parameters
    ----------
    lag_time : NDArray, shape (n_lags,)
        Lag of each row of the fit window's basis, in seconds.
    half_window_s : float
        Half-width of the reported window, in seconds.

    Returns
    -------
    NDArray of bool, shape (n_lags,)
    """
    return np.abs(np.asarray(lag_time, dtype=float)) <= half_window_s + 1e-12


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
    inside = np.flatnonzero(in_report_window(lag_time, half_window_s))
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


def curve_peaks(curves: NDArray) -> NDArray:
    """Peak log-modulation per unit: the curve's value where ``|curve|`` is largest.

    The shipped curve is ``spline_basis @ params`` over the reported window, so
    this is what :func:`peak_log_modulation` returns for the coefficients that
    produced it -- read off the curve the bundle actually ships.

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


def curve_standard_error(
    spline_basis: NDArray, params: NDArray, covariance: NDArray
) -> tuple[NDArray, NDArray]:
    """One unit's kernel at every lag, and its pointwise standard error.

    The kernel at a lag is ``b(lag)' beta`` and its variance
    ``b(lag)' cov(beta) b(lag)``, the off-diagonal terms included (as
    :func:`peak_modulation_se` at the peak alone). A variance rounded below
    zero is taken as zero.

    Parameters
    ----------
    spline_basis : NDArray, shape (n_lags, n_basis)
    params : NDArray, shape (n_basis,)
        One basis block of the fit's coefficients (:func:`curve_coefficients`).
    covariance : NDArray, shape (n_basis, n_basis)
        Covariance of that block.

    Returns
    -------
    curve, standard_error : NDArray, shape (n_lags,)
    """
    curve = spline_basis @ params
    variance = np.einsum("lp,pq,lq->l", spline_basis, covariance, spline_basis)
    return curve, np.sqrt(np.clip(variance, 0.0, None))


def peri_event_rate(
    spike_times: NDArray,
    event_times: NDArray,
    lag_bounds: tuple[float, float],
    bin_s: float,
) -> tuple[NDArray, NDArray]:
    """A unit's firing rate around events, averaged over events (a PSTH).

    Each spike within ``lag_bounds`` of an event (both bounds included) counts
    at its lag from that event, once per event it falls near; the rate is the
    count per bin divided by the number of events and the bin width.

    Parameters
    ----------
    spike_times : NDArray, shape (n_spikes,)
        The unit's spike times, s.
    event_times : NDArray, shape (n_events,)
        The times the spikes are aligned to, s; at least one.
    lag_bounds : tuple of float
        First and last lag, s.
    bin_s : float
        Bin width, s. The bins run from the first lag in steps of ``bin_s``
        (``np.arange``) up to the last lag.

    Returns
    -------
    centers, rate : NDArray, shape (n_bins,)
        Each bin's middle lag, s, and the rate in it, Hz.
    """
    edges = np.arange(lag_bounds[0], lag_bounds[1] + bin_s, bin_s)
    lags = [
        spike_times[
            (spike_times >= start + lag_bounds[0])
            & (spike_times <= start + lag_bounds[1])
        ]
        - start
        for start in event_times
    ]
    counts, _ = np.histogram(np.concatenate(lags), bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), counts / (len(event_times) * bin_s)


def is_separated(params: NDArray, threshold: float = SEPARATED_LOG_RATE) -> NDArray:
    """Units whose fit has no maximum-likelihood estimate to compare against.

    Parameters
    ----------
    params : NDArray, shape (n_units, n_basis) or (n_basis,)
        Basis coefficients of an **unpenalized** fit.
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
    penalized_peak: NDArray, mle_peak: NDArray, mle_peak_se: NDArray
) -> NDArray:
    """Where the prior moved the curve by more than the data could resolve.

    "The prior binds" is a statement about the modulation, not about a
    coefficient: the peak of the penalized kernel differs from the peak of the
    maximum-likelihood one by more than the latter's own estimation error. A
    unit with no maximum-likelihood fit (see :func:`is_separated`) has nothing
    to difference and is not counted here -- report it separately.

    Parameters
    ----------
    penalized_peak, mle_peak : NDArray, shape (n_units,)
        Peak log-rate modulations, from :func:`peak_log_modulation`.
    mle_peak_se : NDArray, shape (n_units,)
        Standard error of ``mle_peak``, from :func:`peak_modulation_se`.

    Returns
    -------
    NDArray of bool, shape (n_units,)
        False wherever any of the three is not finite.
    """
    difference = np.abs(np.asarray(penalized_peak) - np.asarray(mle_peak))
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
#: classes in. Reading it by the name is the swap that mislabeled both of that
#: model's curves in every bundle written before 2026-09-15.
#:
#: Read through :func:`curve_coefficients`.
CURVE_SOURCES = {
    "SWR": ("SWR + non_local_no_SWR", 0),
    "no_SWR": ("SWR + non_local_no_SWR", 1),
    "non_local_SWR": ("non_local_SWR + non_local_no_SWR", 1),
    "non_local_no_SWR": ("non_local_SWR + non_local_no_SWR", 0),
    "matched_control": ("matched_control + non_local_no_SWR", 0),
}


#: The extended event set's joint model: the two replay classes and the SWRs
#: that overlap no non-local event, each with its own kernel, in this block
#: order.
THREE_CLASS_MODEL = "non_local_no_SWR + non_local_SWR + SWR_no_non_local"

#: The extended event set's extra curves, both read off :data:`THREE_CLASS_MODEL`
#: so the two can be compared within one fit: SWRs without replay (block 2) and
#: SWR-coupled replay in the same model (block 1).
EXTENDED_CURVE_SOURCES = {
    "SWR_no_non_local": (THREE_CLASS_MODEL, 2),
    "non_local_SWR_3class": (THREE_CLASS_MODEL, 1),
}


def curve_coefficients(
    curve: str, n_basis: int, sources: dict[str, tuple[str, int]] | None = None
) -> tuple[str, slice]:
    """The model a curve is read from, and its parameter slice.

    ``sources`` defaults to :data:`CURVE_SOURCES`. Parameter 0 is the
    intercept; basis block ``b`` is the ``n_basis`` parameters after it,
    starting at ``1 + b * n_basis``.
    """
    model, block = (CURVE_SOURCES if sources is None else sources)[curve]
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

#: The extended event set's nested pairs, beside :data:`NESTED_MODEL_PAIRS`:
#: does mPFC respond to SWRs without replay at all, do they add to the replay
#: split, and does the replay split add to them.
EXTENDED_NESTED_MODEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("SWR_no_non_local", "constant"),
    (THREE_CLASS_MODEL, "non_local_SWR + non_local_no_SWR"),
    (THREE_CLASS_MODEL, "SWR_no_non_local"),
)
