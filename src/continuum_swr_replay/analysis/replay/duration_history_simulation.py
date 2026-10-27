"""Simulated non-local events with duration, for the per-bin history design.

Simulation only: nothing here reads data. Non-local events are intervals, and
one decoder's events never overlap, while the per-bin rate design
(:func:`per_bin_design.aggregate_to_timebins`) treats each event as an
instantaneous onset, counts every bin of a stop as at risk (including the bins
inside an ongoing event) and builds the self- and cross-history from past
ONSETS. This module simulates two streams of non-overlapping events (an
SWR-coupled one and a non-SWR one) under known history dependence, so the
current design and its duration-aware twin (at-risk time plus offset history,
``MODEL_CROSS_HISTORY_DURATION_RATE``) can be fit to the same data with the
real fitter and compared with the truth.

The generative model, per stop (seconds since stop onset): while no event is
ongoing, stream ``s`` starts an event at the hazard

``lambda_s(t) = base_s * exp(sum_i k(t - anchor_i))``,

summed over past events ``i`` and the scenario's kernels ``k`` whose target is
``s`` and whose source is event ``i``'s stream; ``anchor_i`` is the event's
onset or offset, as the kernel says. While an event is ongoing no stream can
start one. An event's duration is lognormal per stream (clipped below at
:data:`MIN_DURATION_S`); an event that runs past the stop's end keeps its end.
The process is simulated exactly for this piecewise-constant (``dt``) hazard
by inverting the cumulative hazard after each event ends.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .design_constants import (
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
    TMAX,
)
from .model_specs import (
    MODEL_CROSS_HISTORY_DURATION_RATE,
    MODEL_CROSS_HISTORY_RATE,
    PerBinRateModelSpec,
    longest_history_window_s,
)
from .per_bin_design import (
    NO_EXCLUDED_TIME,
    aggregate_to_timebins,
    raised_cosine_history_basis,
)
from .per_bin_rate_model import fit_per_bin_rate_model, freeze_per_bin_cohort

__all__ = [
    "ANCHOR_SETS",
    "HISTORY_BLOCKS",
    "MIN_DURATION_S",
    "SCENARIOS",
    "STREAMS",
    "HistoryKernel",
    "Scenario",
    "fit_history_direction",
    "history_kernels",
    "lag_support",
    "onset_offset_collinearity",
    "simulate_interval_streams",
    "summarize_direction",
    "true_lag_kernel",
]

#: The two streams of one decoder: SWR-coupled and non-SWR non-local events.
STREAMS: tuple[str, str] = ("swr", "no_swr")
#: Shortest event (the decoder's 7-sample minimum at 2 ms is a 12 ms span).
MIN_DURATION_S = 0.012
#: Every kernel is zero beyond this many seconds after its anchor.
KERNEL_SUPPORT_S = 0.25
# Window over which the hazard is evaluated after an event ends. Longer than
# the kernel support, so past it the hazard is the base rate.
_WINDOW_S = 0.3

#: Fitted history blocks: name -> (basis column prefix, anchor, self/cross).
HISTORY_BLOCKS: dict[str, tuple[str, str, str]] = {
    "self_onset": ("hist_rcos_", "onset", "self"),
    "self_offset": ("offhist_rcos_", "offset", "self"),
    "cross_onset": ("cross_hist_rcos_", "onset", "cross"),
    "cross_offset": ("cross_offhist_rcos_", "offset", "cross"),
}


@dataclass(frozen=True)
class HistoryKernel:
    """A Gaussian bump in one stream's log onset hazard after a past event.

    Attributes
    ----------
    target : str
        Stream whose onset hazard the kernel modulates.
    source : str
        Stream of the past event that drives it.
    anchor : str
        ``"onset"`` or ``"offset"``: the past event's time the lag is measured
        from.
    log_gain : float
        Peak log hazard ratio.
    center_s, width_s : float
        Peak lag and Gaussian width (seconds); zero at negative lags.
    """

    target: str
    source: str
    anchor: str
    log_gain: float
    center_s: float
    width_s: float

    def __post_init__(self) -> None:
        if self.anchor not in ("onset", "offset"):
            raise ValueError(f"anchor must be 'onset' or 'offset': {self.anchor!r}")
        if {self.target, self.source} - set(STREAMS):
            raise ValueError(f"streams must be in {STREAMS}")
        if self.center_s + 4 * self.width_s > KERNEL_SUPPORT_S:
            raise ValueError("kernel reaches past KERNEL_SUPPORT_S")

    def __call__(self, lag_s: np.ndarray) -> np.ndarray:
        """Log hazard ratio at ``lag_s`` seconds after the anchor."""
        lag_s = np.asarray(lag_s, dtype=float)
        bump = self.log_gain * np.exp(
            -0.5 * ((lag_s - self.center_s) / self.width_s) ** 2
        )
        return np.where(
            (lag_s > 0) & (lag_s <= self.center_s + 4 * self.width_s), bump, 0.0
        )


@dataclass(frozen=True)
class Scenario:
    """Ground truth for one simulation: base rates, durations and kernels.

    Attributes
    ----------
    name, description : str
    base_rate_hz : dict[str, float]
        Per stream, the onset hazard while no event is ongoing and no kernel
        is active.
    duration_median_s, duration_sigma : dict[str, float]
        Per stream, the lognormal duration's median (s) and log-SD (0 gives a
        constant duration).
    kernels : tuple of HistoryKernel
    """

    name: str
    description: str
    base_rate_hz: dict[str, float] = field(
        default_factory=lambda: {"swr": 0.4, "no_swr": 0.6}
    )
    duration_median_s: dict[str, float] = field(
        default_factory=lambda: {"swr": 0.09, "no_swr": 0.05}
    )
    duration_sigma: dict[str, float] = field(
        default_factory=lambda: {"swr": 0.45, "no_swr": 0.45}
    )
    kernels: tuple[HistoryKernel, ...] = ()

    def with_durations(self, sigma: float, median_s: float | None = None) -> Scenario:
        """The same scenario with every stream's duration log-SD (and median)."""
        medians = (
            dict(self.duration_median_s)
            if median_s is None
            else dict.fromkeys(STREAMS, median_s)
        )
        return dataclasses.replace(
            self,
            name=f"{self.name}_sigma{sigma:g}",
            duration_median_s=medians,
            duration_sigma=dict.fromkeys(STREAMS, sigma),
        )


def _kernels_all_pairs(anchor: str, **bump: float) -> tuple[HistoryKernel, ...]:
    return tuple(
        HistoryKernel(target=t, source=s, anchor=anchor, **bump)
        for t in STREAMS
        for s in STREAMS
    )


#: The ground-truth scenarios of the duration-aware history simulation.
SCENARIOS: dict[str, Scenario] = {
    "a_independent": Scenario(
        name="a_independent",
        description=(
            "No dependence: onsets homogeneous in the time no event is ongoing; "
            "the only history effect is non-overlap."
        ),
    ),
    "b_fragmentation": Scenario(
        name="b_fragmentation",
        description=(
            "Fragmentation: after ANY event ends, both streams' onset hazard is "
            "raised 6x, peaking 40 ms after the END (about 20-60 ms)."
        ),
        kernels=_kernels_all_pairs(
            "offset", log_gain=float(np.log(6.0)), center_s=0.04, width_s=0.01
        ),
    ),
    "c_onset_rhythm": Scenario(
        name="c_onset_rhythm",
        description=(
            "Onset-locked rhythm: each stream's onset hazard is raised 4x, "
            "peaking 110 ms after its own previous ONSET, whatever that "
            "event's duration."
        ),
        kernels=tuple(
            HistoryKernel(
                target=s,
                source=s,
                anchor="onset",
                log_gain=float(np.log(4.0)),
                center_s=0.11,
                width_s=0.015,
            )
            for s in STREAMS
        ),
    ),
    "d_cross_swr_to_nonswr": Scenario(
        name="d_cross_swr_to_nonswr",
        description=(
            "Cross-stream coupling: non-SWR onsets raised 5x, peaking 50 ms "
            "after an SWR-coupled event ENDS; nothing the other way."
        ),
        kernels=(
            HistoryKernel(
                target="no_swr",
                source="swr",
                anchor="offset",
                log_gain=float(np.log(5.0)),
                center_s=0.05,
                width_s=0.012,
            ),
        ),
    ),
}


def simulate_interval_streams(
    scenario: Scenario,
    *,
    n_stops: int,
    seed: int,
    stop_median_s: float = 3.0,
    stop_sigma: float = 0.4,
    dt: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate stops and the two streams' non-overlapping events in them.

    Parameters
    ----------
    scenario : Scenario
    n_stops : int
        Number of stops, one synthetic session.
    seed : int
        Seed of the ``numpy`` generator; the output is a function of it.
    stop_median_s, stop_sigma : float
        Lognormal stop duration (median seconds, log-SD), clipped to
        ``[0.5, 10]`` s.
    dt : float
        Resolution of the piecewise-constant hazard (seconds).

    Returns
    -------
    stop_table : pd.DataFrame, shape (n_stops, ...)
        ``animal``, ``session_date``, ``epoch``, ``stop_id``, ``stop_start``,
        ``stop_end``, ``immobility_duration_s`` and the reward-history basis
        columns (uniform noise with no effect on the rate, so the shipped
        rungs' reward block has something to fit).
    events : pd.DataFrame, shape (n_events, ...)
        The session keys, ``stop_id``, ``stream``, ``t_since_stop_onset``,
        ``t_end_since_stop_onset`` (seconds since stop onset) and
        ``duration_s``, in time order within each stop.
    """
    rng = np.random.default_rng(seed)
    streams = np.asarray(STREAMS)
    base = np.array([scenario.base_rate_hz[s] for s in STREAMS])
    medians = np.array([scenario.duration_median_s[s] for s in STREAMS])
    sigmas = np.array([scenario.duration_sigma[s] for s in STREAMS])
    stream_index = {s: i for i, s in enumerate(STREAMS)}
    kernels = [
        (stream_index[k.target], stream_index[k.source], k) for k in scenario.kernels
    ]
    stop_dur = np.clip(
        stop_median_s * np.exp(stop_sigma * rng.standard_normal(n_stops)), 0.5, 10.0
    )

    rows: list[tuple[int, int, float, float]] = []
    for stop_id, length in enumerate(stop_dur):
        t = 0.0
        past: list[tuple[int, float, float]] = []  # (stream, onset, offset)
        while t < length:
            n = int(np.ceil(min(_WINDOW_S, length - t) / dt))
            grid = t + dt * (np.arange(n) + 0.5)
            log_h = np.repeat(np.log(base)[:, None], n, axis=1)  # (2, n)
            for src, onset, offset in past:
                if offset < t - KERNEL_SUPPORT_S:
                    continue
                for target, source, kernel in kernels:
                    if source == src:
                        anchor = onset if kernel.anchor == "onset" else offset
                        log_h[target] += kernel(grid - anchor)
            hazard = np.exp(log_h)
            total = hazard.sum(axis=0)
            cum = np.cumsum(total) * dt
            draw = rng.exponential()
            if draw <= cum[-1]:
                i = int(np.searchsorted(cum, draw))
                before = cum[i - 1] if i else 0.0
                onset = t + i * dt + (draw - before) / total[i]
                p = hazard[:, i] / total[i]
            else:
                if t + n * dt >= length:
                    break
                # Past the window every kernel is zero: the base rate.
                onset = t + n * dt + (draw - cum[-1]) / base.sum()
                p = base / base.sum()
            if onset >= length:
                break
            s = int(rng.choice(2, p=p))
            duration = max(
                MIN_DURATION_S, medians[s] * np.exp(sigmas[s] * rng.standard_normal())
            )
            past.append((s, onset, onset + duration))
            rows.append((stop_id, s, onset, onset + duration))
            t = onset + duration

    gap_s = 5.0
    stop_start = np.concatenate([[0.0], np.cumsum(stop_dur + gap_s)[:-1]])
    session = {"animal": "sim", "session_date": 20261026, "epoch": "02_r1"}
    stop_table = pd.DataFrame(
        {
            **session,
            "stop_id": np.arange(n_stops),
            "stop_start": stop_start,
            "stop_end": stop_start + stop_dur,
            "immobility_duration_s": stop_dur,
        }
    )
    for col in REWARD_HISTORY_COLUMNS:
        stop_table[col] = rng.uniform(size=n_stops)
    stop_table[REWARD_HISTORY_INCOMPLETE] = (rng.uniform(size=n_stops) < 0.1).astype(
        float
    )
    arr = np.array(rows, dtype=float).reshape(-1, 4)
    events = pd.DataFrame(
        {
            **session,
            "stop_id": arr[:, 0].astype(int),
            "stream": streams[arr[:, 1].astype(int)],
            "t_since_stop_onset": arr[:, 2],
            "t_end_since_stop_onset": arr[:, 3],
        }
    )
    events["duration_s"] = (
        events["t_end_since_stop_onset"] - events["t_since_stop_onset"]
    )
    return stop_table, events


def true_lag_kernel(
    scenario: Scenario,
    *,
    target: str,
    source: str,
    anchor: str,
    n_lags: int = N_HISTORY_LAGS,
    bin_s: float = BIN_S,
) -> np.ndarray:
    """The scenario's log hazard ratio at each design lag ``1..n_lags``.

    An anchor ``j`` bins back lies ``(j + U1 - U2) * bin_s`` before a time in
    the current bin (``U1``, ``U2`` uniform within the bins), a triangular lag
    on ``[(j - 1), (j + 1)] * bin_s``; the truth at lag ``j`` is the kernels'
    mean over it.

    Returns
    -------
    np.ndarray, shape (n_lags,)
    """
    u = (np.arange(200) + 0.5) / 200
    offsets = (u[:, None] - u[None, :]).ravel()  # triangular on (-1, 1)
    lags = (np.arange(1, n_lags + 1)[:, None] + offsets[None, :]) * bin_s
    out = np.zeros(n_lags)
    for kernel in scenario.kernels:
        if (kernel.target, kernel.source, kernel.anchor) == (target, source, anchor):
            out += kernel(lags).mean(axis=1)
    return out


def history_kernels(fit: dict) -> dict[str, np.ndarray]:
    """Per-lag log rate ratio of each fitted history block.

    Parameters
    ----------
    fit : dict
        A :func:`per_bin_rate_model.fit_per_bin_rate_model` result.

    Returns
    -------
    dict
        Block name (:data:`HISTORY_BLOCKS`) -> shape ``(N_HISTORY_LAGS,)``
        log rate ratio of one past event at each lag (the raw raised-cosine
        bumps weighted by the block's coefficients), for each block the fit
        carries.
    """
    basis = raised_cosine_history_basis(N_HISTORY_LAGS, N_HISTORY_BASIS)
    out = {}
    for name, (prefix, _, _) in HISTORY_BLOCKS.items():
        cols = [f"coef_{prefix}{k}" for k in range(N_HISTORY_BASIS)]
        if all(c in fit for c in cols):
            out[name] = np.array([fit[c] for c in cols]) @ basis
    return out


#: Per-lag count columns of each history block in the design.
_LAG_COLUMNS = {
    "self_onset": "onset_lag",
    "self_offset": "offset_lag",
    "cross_onset": "cross_lag",
    "cross_offset": "cross_offset_lag",
}
#: Anchor sets a history block is fit with, by label.
ANCHOR_SETS: dict[str, tuple[str, ...]] = {
    "onset": ("onset",),
    "offset": ("offset",),
    "both": ("onset", "offset"),
}


def lag_support(design: pd.DataFrame) -> dict[str, np.ndarray]:
    """Expected response onsets at each lag of each history block.

    The design's mean onset rate times the exposure of the bins that have a
    past event at that lag: how many onsets a lag's coefficient is estimated
    from if history had no effect. In the at-risk design a short onset lag has
    almost no support (the event that began then is usually still ongoing, so
    the bin is not at risk), and its coefficient is not identified.

    Returns
    -------
    dict
        Block name -> shape ``(N_HISTORY_LAGS,)``, for each block whose lag
        columns ``design`` carries.
    """
    rate = design["count"].sum() / design["exposure"].sum()
    exposure = design["exposure"].to_numpy(dtype=float)
    out = {}
    for name, prefix in _LAG_COLUMNS.items():
        cols = [f"{prefix}{j}" for j in range(1, N_HISTORY_LAGS + 1)]
        if all(c in design.columns for c in cols):
            present = design[cols].to_numpy() > 0  # (n_bins, n_lags)
            out[name] = rate * (exposure @ present)
    return out


def onset_offset_collinearity(design: pd.DataFrame) -> dict[str, float]:
    """Largest variance inflation of the onset and offset blocks of one stream.

    For the self and the cross stream, the VIFs of the eight columns of the
    onset (``*hist_rcos_k``) and offset (``*offhist_rcos_k``) blocks within
    the design's (at-risk) bins, ``diag(inv(corr))``. With constant durations
    an offset is an onset shifted by a fixed lag, so the two blocks span nearly
    the same columns and the VIFs grow; varying durations break the tie.

    Returns
    -------
    dict
        ``max_vif_self`` / ``max_vif_cross`` for each stream whose two blocks
        ``design`` carries.
    """
    out = {}
    for rel, prefix in (("self", ""), ("cross", "cross_")):
        cols = [
            f"{prefix}{anchor}hist_rcos_{k}"
            for anchor in ("", "off")
            for k in range(N_HISTORY_BASIS)
        ]
        if all(c in design.columns for c in cols):
            corr = np.corrcoef(design[cols].to_numpy(dtype=float), rowvar=False)
            out[f"max_vif_{rel}"] = float(np.max(np.diag(np.linalg.inv(corr))))
    return out


def _rung(
    base: PerBinRateModelSpec,
    name: str,
    *,
    self_anchors: str | None,
    cross_anchors: str | None,
) -> PerBinRateModelSpec:
    """``base`` with the given self / cross anchor sets (None drops the block)."""
    return dataclasses.replace(
        base,
        name=name,
        self_history_lags=N_HISTORY_LAGS if self_anchors else 0,
        self_history_basis=N_HISTORY_BASIS if self_anchors else 0,
        self_history_anchors=ANCHOR_SETS[self_anchors or "onset"],
        cross_history_lags=N_HISTORY_LAGS if cross_anchors else 0,
        cross_history_basis=N_HISTORY_BASIS if cross_anchors else 0,
        cross_history_anchors=ANCHOR_SETS[cross_anchors or "onset"],
    )


def _fit_or_raise(design: pd.DataFrame, spec: PerBinRateModelSpec) -> dict:
    fit = fit_per_bin_rate_model(design, spec)
    if fit is None:
        raise RuntimeError(f"{spec.name} did not fit")
    fit.pop("predicted")
    return fit


def fit_history_direction(
    stop_table: pd.DataFrame,
    events: pd.DataFrame,
    *,
    response: str,
    duration_aware: bool,
) -> dict:
    """Fit one response stream's history rungs, as the cross runner does.

    Mirrors ``scripts/rate/fit_cross_history_rate.py:_fit_direction`` on a
    simulated session: :func:`aggregate_to_timebins` at ``BIN_S`` / ``TMAX``
    with the reward basis and the raised-cosine self- and cross-history, a
    frozen cohort, and :func:`fit_per_bin_rate_model`.

    The current design fits the shipped rungs ``Mself`` and ``Mselfcross``
    (``MODEL_CROSS_HISTORY_RATE``) and the same rung without history
    (``none``). The duration-aware design adds the at-risk time over both
    streams' events and the offset history, and chooses each block's anchors
    by AIC, built from ``MODEL_CROSS_HISTORY_DURATION_RATE``'s rungs: the self
    block as ``none`` / ``self_onset`` / ``self_offset`` / ``self_both`` (no
    cross), then, on the best of those, the cross block as onset / offset /
    both (``{self}+cross_{anchors}``). Every rung of a direction is fit to the
    same frozen bins, so their AICs compare.

    Returns
    -------
    dict
        ``response``, ``cross``, ``duration_aware``, ``aic`` (rung -> AIC),
        ``fits`` (rung -> fit dict without ``predicted``), ``kernels`` (rung
        -> :func:`history_kernels`), ``support`` (:func:`lag_support` of the
        design), ``collinearity`` (:func:`onset_offset_collinearity`; empty
        for the current design), ``best_self`` (the duration-aware self rung chosen; ``self``
        for the current design), ``best`` (the lowest-AIC rung), ``n_bins``,
        ``n_events`` and ``design_attrs``.
    """
    cross = next(s for s in STREAMS if s != response)
    resp = events[events["stream"] == response]
    xs = events[events["stream"] == cross]
    if duration_aware:
        full = MODEL_CROSS_HISTORY_DURATION_RATE["Mselfcross_dur"]
        self_rungs = {
            "none": _rung(full, "none", self_anchors=None, cross_anchors=None)
        } | {
            f"self_{a}": _rung(full, f"self_{a}", self_anchors=a, cross_anchors=None)
            for a in ANCHOR_SETS
        }
    else:
        m_self, full = MODEL_CROSS_HISTORY_RATE.values()
        self_rungs = {
            "none": _rung(m_self, "none", self_anchors=None, cross_anchors=None),
            "self": m_self,
        }
    design = aggregate_to_timebins(
        stop_table,
        resp,
        bin_s=BIN_S,
        tmax=TMAX,
        reward_basis=True,
        self_history_lags=N_HISTORY_LAGS,
        self_history_basis=N_HISTORY_BASIS,
        cross_onset_times=xs,
        cross_history_lags=N_HISTORY_LAGS,
        cross_history_basis=N_HISTORY_BASIS,
        history_window_s=longest_history_window_s([full]),
        excluded_intervals=NO_EXCLUDED_TIME,
        arrival_clocks=False,
        outcome_shift=False,
        at_risk_events=events if duration_aware else None,
        offset_history=duration_aware,
    )
    design, _ = freeze_per_bin_cohort(design, [full])
    fits = {name: _fit_or_raise(design, spec) for name, spec in self_rungs.items()}
    if duration_aware:
        best_self = min(fits, key=lambda name: fits[name]["aic"])
        self_anchors = None if best_self == "none" else best_self.removeprefix("self_")
        for a in ANCHOR_SETS:
            name = f"{best_self}+cross_{a}"
            spec = _rung(full, name, self_anchors=self_anchors, cross_anchors=a)
            fits[name] = _fit_or_raise(design, spec)
    else:
        best_self = "self"
        fits["self+cross"] = _fit_or_raise(design, full)
    aic = {name: fit["aic"] for name, fit in fits.items()}
    return {
        "response": response,
        "cross": cross,
        "duration_aware": duration_aware,
        "aic": aic,
        "fits": fits,
        "kernels": {name: history_kernels(fit) for name, fit in fits.items()},
        "support": lag_support(design),
        "collinearity": onset_offset_collinearity(design),
        "best_self": best_self,
        "best": min(aic, key=aic.get),
        "n_bins": int(len(design)),
        "n_events": int(design["count"].sum()),
        "design_attrs": dict(design.attrs),
    }


def summarize_direction(result: dict) -> dict:
    """The AIC contrasts of one :func:`fit_history_direction` result.

    Positive values favour the more complex rung, or (``*_offset_vs_onset``)
    the offset anchor.

    Returns
    -------
    dict
        ``delta_aic_self`` (no history -> the self block: ``Mself`` for the
        current design, both anchors for the duration-aware one),
        ``delta_aic_cross`` (current: ``Mself`` -> ``Mselfcross``;
        duration-aware: the chosen self rung -> it plus the cross block with
        both anchors, a fixed block so the test is not chosen by the data),
        and for the duration-aware design ``self_offset_vs_onset`` (AIC of
        onset-only minus offset-only self history), ``cross_offset_vs_onset``
        (the same for the cross block, on the chosen self rung),
        ``best_self`` and ``best``.
    """
    aic = result["aic"]
    if not result["duration_aware"]:
        return {
            "delta_aic_self": aic["none"] - aic["self"],
            "delta_aic_cross": aic["self"] - aic["self+cross"],
            "best_self": "self",
            "best": result["best"],
        }
    best_self = result["best_self"]
    cross = {a: aic[f"{best_self}+cross_{a}"] for a in ANCHOR_SETS}
    return {
        "delta_aic_self": aic["none"] - aic["self_both"],
        "delta_aic_cross": aic[best_self] - cross["both"],
        "self_offset_vs_onset": aic["self_onset"] - aic["self_offset"],
        "cross_offset_vs_onset": cross["onset"] - cross["offset"],
        "best_self": best_self,
        "best": result["best"],
    }
