"""Planted scenarios for the simulation validation of the mPFC unit models.

Each scenario takes a session's real design (:class:`~.unit_design.UnitDesign`,
built on the session's real fit bins and events) and returns, for each
simulated unit, its expected spike count in every fit bin, with the planted
value of each quantity the models report. A simulation replaces only the
spike counts (a Poisson draw of the expected counts); the real fit, the real
tests and the real reduction then run on them. These functions are not yet
wired to the simulation engine (``analysis/regression/simulation.py``).

The expected count of unit ``u`` in fit bin ``t`` is

.. code-block:: text

    mu_ut = (rate_u / fs) * exp( sum_k sum_e w_ke c_ku(t - onset_e) )

with ``rate_u`` the unit's baseline rate (Hz), ``fs`` the sampling rate,
``c_ku`` the planted log-rate curve of kernel ``k`` over the lags and
``w_ke`` the event's weight in that kernel (1, or its mark for a marked
kernel): the same convolution the design's columns are made of
(``design.convolve_event_basis``), so a curve need not lie in the basis.

:data:`SCENARIOS` lists the scenarios, what each validates and what result
counts as a failure. Each runs under both settings of the open kernel-penalty
question (:data:`~.unit_design.KERNEL_PENALTY_RULES`) as two arms, so the
simulation shows the lead what each setting does on the real designs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from continuum_swr_replay.analysis.pfc.design import convolve_event_basis
from continuum_swr_replay.analysis.pfc.modulation import curve_peaks, in_report_window
from continuum_swr_replay.analysis.pfc.unit_design import UnitDesign

__all__ = [
    "SCENARIOS",
    "PlantedUnits",
    "ScenarioSpec",
    "animal_session_gains",
    "baseline_rates",
    "expected_counts",
    "kernel_by_kind_and_mark",
    "low_rate_units",
    "no_modulation",
    "outside_basis_kernel",
    "planted_share",
    "response_curve",
]

#: A realistic peri-event response: a Gaussian bump of this log-rate gain,
#: peak lag and SD (s). A gain of 0.7 is a rate multiplier of 2.0, between the
#: median absolute peak of the matched-control curves and that of the SWR
#: curves in the shipped bundles (both pending diagnostics, not results).
REALISTIC_GAIN = 0.7
REALISTIC_PEAK_S = 0.05
REALISTIC_WIDTH_S = 0.04

#: The median baseline rate of a simulated unit (Hz) and the log-normal SD of
#: the rates across units.
MEDIAN_RATE_HZ = 5.0
RATE_LOG_SD = 0.8

#: The rates of the low-rate units (Hz): few spikes, to test the penalties.
LOW_RATES_HZ = (0.2, 0.5, 1.0)


@dataclass(frozen=True)
class ScenarioSpec:
    """One scenario: its name, what it validates, what counts as a failure."""

    name: str
    validates: str
    failure: str


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        "no_modulation",
        "the false-positive rate of every reported quantity: each unit's "
        "empirical p against the matched controls, each called fraction (per "
        "session and reduced to the population), the best-model fractions, "
        "the peak sizes (the noise floor)",
        "a called fraction whose binomial 95 % interval over the simulated "
        "units lies above alpha, or a reduced fraction whose interval excludes "
        "alpha; a best model other than the constant above alpha",
    ),
    ScenarioSpec(
        "planted_share",
        "power per unit, the called fraction against the planted share, the "
        "recovered curve and peak (bias, coverage of the pointwise band), and "
        "that the unmodulated units stay at the false-positive rate",
        "the reduced called fraction's interval excludes share * power + "
        "(1 - share) * alpha; the recovered peak's median bias beyond 20 % of "
        "the planted gain; unmodulated units called above alpha",
    ),
    ScenarioSpec(
        "kernel_by_kind_and_mark",
        "that the SWR and No SWR kernels and a marked kernel are estimated "
        "apart (no leakage between kinds or into the mark), and the kind "
        "difference of the peak",
        "a kernel planted at zero called above alpha, or recovered with a "
        "mean curve beyond its pointwise band, when the other kind or the "
        "mark carries the modulation",
    ),
    ScenarioSpec(
        "low_rate_units",
        "the penalties with few spikes: each fit has a finite maximum, the "
        "curve is shrunk toward zero rather than given the wrong sign, and the "
        "false-positive rate holds",
        "a fit that ends without a finite maximum or a converged status; a "
        "recovered peak of the wrong sign in more than alpha of the "
        "modulated units; the false-positive rate above alpha",
    ),
    ScenarioSpec(
        "outside_basis_kernel",
        "the estimator on a response the basis cannot represent (a unit's "
        "peri-event histogram, lightly smoothed): the bias of the peak, its "
        "lag and the onset the curve gives",
        "a peak bias beyond 30 % of the planted peak, or a peak lag off by "
        "more than one knot spacing (50 ms)",
    ),
    ScenarioSpec(
        "animal_session_gains",
        "the reduction session -> day -> animal: the coverage of the "
        "population interval when the gain differs between animals and "
        "sessions, and the count of animals with the estimate's sign",
        "coverage of the nominal 95 % interval below 90 % over the simulated "
        "replicates",
    ),
)


@dataclass(frozen=True)
class PlantedUnits:
    """What a scenario plants in one session.

    Attributes
    ----------
    mu : NDArray, shape (n_fit_bins, n_units)
        Each unit's expected spike count per fit bin.
    rate_hz : NDArray, shape (n_units,)
        Each unit's baseline rate.
    curves : dict of str to NDArray, shape (n_lags, n_units)
        Each kernel's planted log-rate curve over the design's lags, by kernel
        name; zero for a unit without modulation.
    modulated : dict of str to NDArray of bool, shape (n_units,)
        Which units carry each kernel (a non-zero curve).
    peaks : dict of str to NDArray, shape (n_units,)
        Each kernel's planted peak over the reported window (signed, the
        curve's value where its absolute value is largest; ``curve_peaks``).
    modulated_fraction : dict of str to float
        The session's share of units with each kernel: the planted value of
        the called fraction (the called fraction should be near
        ``fraction * power + (1 - fraction) * alpha``).
    """

    mu: NDArray
    rate_hz: NDArray
    curves: dict[str, NDArray]
    modulated: dict[str, NDArray]
    peaks: dict[str, NDArray]
    modulated_fraction: dict[str, float]


def response_curve(
    lag_time: NDArray,
    gain: float = REALISTIC_GAIN,
    peak_s: float = REALISTIC_PEAK_S,
    width_s: float = REALISTIC_WIDTH_S,
) -> NDArray:
    """A Gaussian bump of log-rate gain ``gain`` at ``peak_s``, SD ``width_s``.

    Returns
    -------
    NDArray, shape (n_lags,)
    """
    lag_time = np.asarray(lag_time, dtype=float)
    return gain * np.exp(-0.5 * ((lag_time - peak_s) / width_s) ** 2)


def baseline_rates(
    rng: np.random.Generator,
    n_units: int,
    median_hz: float = MEDIAN_RATE_HZ,
    log_sd: float = RATE_LOG_SD,
) -> NDArray:
    """Log-normal baseline rates (Hz), shape ``(n_units,)``."""
    return median_hz * np.exp(log_sd * rng.standard_normal(n_units))


def expected_counts(
    design: UnitDesign,
    rate_hz: NDArray,
    curves: Mapping[str, NDArray],
    sampling_frequency: float,
) -> PlantedUnits:
    """Each unit's expected count per fit bin, from planted curves.

    Parameters
    ----------
    design : UnitDesign
        The session's design: its fit bins, lags and kernels' onsets.
    rate_hz : NDArray, shape (n_units,)
    curves : mapping of str to NDArray, shape (n_lags, n_units)
        A planted curve per kernel name (a kernel not named is planted at
        zero).
    sampling_frequency : float
        Hz.

    Returns
    -------
    PlantedUnits
    """
    rate_hz = np.asarray(rate_hz, dtype=float)
    n_units = rate_hz.size
    n_lags = design.lags.size
    unknown = sorted(set(curves) - set(design.onsets))
    if unknown:
        raise ValueError(f"curves for kernels the design lacks: {unknown}")
    full = {
        kernel.name: np.asarray(
            curves.get(kernel.name, np.zeros((n_lags, n_units))), dtype=float
        )
        for kernel in design.kernels
    }
    eta = np.tile(np.log(rate_hz / sampling_frequency), (int(design.rows.sum()), 1))
    for name, curve in full.items():
        if curve.shape != (n_lags, n_units):
            raise ValueError(
                f"{name}: curve of shape {curve.shape}, not {(n_lags, n_units)}"
            )
        if curve.any():
            eta += convolve_event_basis(design.onsets[name], curve, design.lags)[
                design.rows
            ]
    report = in_report_window(design.lag_time)
    modulated = {name: np.abs(c).max(axis=0) > 0 for name, c in full.items()}
    return PlantedUnits(
        mu=np.exp(eta),
        rate_hz=rate_hz,
        curves=full,
        modulated=modulated,
        peaks={name: curve_peaks(c[report]) for name, c in full.items()},
        modulated_fraction={name: float(m.mean()) for name, m in modulated.items()},
    )


def no_modulation(
    design: UnitDesign,
    rng: np.random.Generator,
    n_units: int,
    sampling_frequency: float,
) -> PlantedUnits:
    """No unit is modulated by any kernel; rates log-normal."""
    return expected_counts(design, baseline_rates(rng, n_units), {}, sampling_frequency)


def planted_share(
    design: UnitDesign,
    rng: np.random.Generator,
    n_units: int,
    sampling_frequency: float,
    *,
    kernel: str,
    share: float,
    curve: NDArray | None = None,
) -> PlantedUnits:
    """A share of the units carries one kernel; the rest none.

    The modulated units are ``round(share * n_units)`` drawn at random; half
    of them (rounded down) carry the curve with its sign flipped (a
    suppression). ``curve`` defaults to :func:`response_curve`.
    """
    curve = response_curve(design.lag_time) if curve is None else np.asarray(curve)
    n_modulated = int(round(share * n_units))
    units = rng.permutation(n_units)[:n_modulated]
    signs = np.ones(n_modulated)
    signs[: n_modulated // 2] = -1.0
    planted = np.zeros((design.lags.size, n_units))
    planted[:, units] = curve[:, None] * signs
    return expected_counts(
        design, baseline_rates(rng, n_units), {kernel: planted}, sampling_frequency
    )


def kernel_by_kind_and_mark(
    design: UnitDesign,
    rng: np.random.Generator,
    n_units: int,
    sampling_frequency: float,
    *,
    curves: Mapping[str, NDArray],
) -> PlantedUnits:
    """Every unit carries the given curve of each named kernel (one per kind,
    one per mark), the same for every unit; the kernels not named are zero.

    For example the SWR kernel as an excitation at +50 ms, the No SWR kernel
    as a smaller suppression at +100 ms, and the No SWR marked kernel at zero
    (or the reverse), so a leak between them shows.
    """
    n_lags = design.lags.size
    planted = {
        name: np.broadcast_to(np.asarray(c, dtype=float)[:, None], (n_lags, n_units))
        for name, c in curves.items()
    }
    return expected_counts(
        design, baseline_rates(rng, n_units), planted, sampling_frequency
    )


def low_rate_units(
    design: UnitDesign,
    rng: np.random.Generator,
    n_units: int,
    sampling_frequency: float,
    *,
    kernel: str,
    rates_hz: Sequence[float] = LOW_RATES_HZ,
    share: float = 0.5,
) -> PlantedUnits:
    """Units at low rates (cycling through ``rates_hz``), a share of them with
    the realistic kernel, the rest none."""
    rate = np.resize(np.asarray(rates_hz, dtype=float), n_units)
    n_modulated = int(round(share * n_units))
    planted = np.zeros((design.lags.size, n_units))
    planted[:, rng.permutation(n_units)[:n_modulated]] = response_curve(
        design.lag_time
    )[:, None]
    return expected_counts(design, rate, {kernel: planted}, sampling_frequency)


def outside_basis_kernel(
    design: UnitDesign,
    sampling_frequency: float,
    *,
    kernel: str,
    peri_event_rate: NDArray,
    smoothing_s: float = 0.004,
) -> PlantedUnits:
    """Each unit's kernel taken from a peri-event rate, lightly smoothed.

    Parameters
    ----------
    peri_event_rate : NDArray, shape (n_lags, n_units)
        Each unit's rate around the events over the design's lags, in Hz (for
        example its peri-event histogram), positive. The planted curve is the
        log of it over its mean at the window's edges (the outer 50 ms each
        side), smoothed by a Gaussian of SD ``smoothing_s``: a shape at the
        sampling resolution, outside the 50 ms basis.
    smoothing_s : float
        s.

    Returns
    -------
    PlantedUnits
        With each unit's baseline rate the edge mean of its peri-event rate.
    """
    rate = np.asarray(peri_event_rate, dtype=float)
    if rate.shape[0] != design.lags.size or (rate <= 0).any():
        raise ValueError("one positive rate per lag and unit")
    edges = np.abs(design.lag_time) >= np.abs(design.lag_time).max() - 0.05
    baseline = rate[edges].mean(axis=0)
    curve = gaussian_filter1d(
        np.log(rate / baseline), smoothing_s * sampling_frequency, axis=0
    )
    return expected_counts(design, baseline, {kernel: curve}, sampling_frequency)


def animal_session_gains(
    rng: np.random.Generator,
    animals: NDArray,
    *,
    mean_gain: float = REALISTIC_GAIN,
    animal_sd: float = 0.2,
    session_sd: float = 0.1,
) -> tuple[NDArray, dict[str, float]]:
    """A kernel gain for each session that differs between animals and sessions.

    Parameters
    ----------
    animals : NDArray, shape (n_sessions,)
        Each session's animal.
    mean_gain, animal_sd, session_sd : float
        The population's mean gain, and the SDs of the animals' gains about it
        and of the sessions' about their animal's.

    Returns
    -------
    gains : NDArray, shape (n_sessions,)
        Pass each to :func:`planted_share` (``curve=response_curve(lag_time,
        gain)``) for its session.
    planted : dict
        ``animal -> gain`` and ``"population"``: the mean of the animals'
        gains, the animal-level value the reduced interval should cover.
    """
    animals = np.asarray(animals)
    names = sorted(set(animals.tolist()))
    animal_gain = {a: mean_gain + animal_sd * rng.standard_normal() for a in names}
    gains = np.array([animal_gain[a] for a in animals]) + session_sd * (
        rng.standard_normal(animals.size)
    )
    return gains, {
        **animal_gain,
        "population": float(np.mean(list(animal_gain.values()))),
    }
