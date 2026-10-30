"""The duration-aware per-bin history design on real events, and its readouts.

Events have duration and one detector's events never overlap. The shipped
per-bin design counts every bin as at risk and reads history from onsets
only; the duration-aware design (``aggregate_to_timebins(at_risk_events=...,
offset_history=True)``, the ``MODEL_CROSS_HISTORY_DURATION_RATE`` rungs) drops
the bins inside an ongoing event and reads history from onsets, ends or both.
Simulation behind it: :mod:`.duration_history_simulation`; review
``.claude/docs/reviews/2026-10-26-duration-aware-history-simulation.md``.

This module holds what the runners and the Part 3 summary share:

- :func:`attach_event_ends` gives each non-local and SWR onset its event's end;
- :func:`build_design` is one session's frozen design for a population;
- :func:`lag_support`, :func:`onset_offset_collinearity` and
  :func:`history_kernels` read a design or a fit;
- :func:`fit_pooled_history` fits a rung to several sessions' designs at once
  with a session fixed effect, for kernels a single session is too sparse to
  measure (about one expected onset per lag).
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from continuum_swr_replay.analysis.events.swr_source import (
    read_swr_inventory,
    swr_epoch_id_of_session,
)
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ._fit_diagnostics import (
    EXPECTED_FIT_ERRORS,
    capture_fit_warnings,
    count_fit_quality_warnings,
)
from .design_constants import BIN_S, N_HISTORY_BASIS, N_HISTORY_LAGS, TMAX
from .model_specs import (
    MODEL_CROSS_HISTORY_DURATION_RATE,
    MODEL_CROSS_HISTORY_RATE,
    STOP_COVARIATES,
    PerBinRateModelSpec,
    longest_history_window_s,
)
from .per_bin_design import (
    POPULATIONS,
    aggregate_to_timebins,
    raised_cosine_history_basis,
)
from .per_bin_rate_model import freeze_per_bin_cohort

__all__ = [
    "CROSS_RUNGS",
    "DECODE_SAMPLE_S",
    "END_COLUMN",
    "HISTORY_BLOCKS",
    "POPULATION_PLAN",
    "SELF_RUNGS",
    "attach_event_ends",
    "build_design",
    "fit_pooled_history",
    "history_kernels",
    "lag_support",
    "onset_offset_collinearity",
]

LADDER = MODEL_CROSS_HISTORY_DURATION_RATE
SELF_RUNGS = ("Mnohist_dur", "Mself_onset_dur", "Mself_offset_dur", "Mself_dur")
CROSS_RUNGS = ("Mselfcross_onset_dur", "Mselfcross_offset_dur", "Mselfcross_dur")
HISTORY_WINDOW_S = longest_history_window_s(LADDER.values())
#: An event's end, seconds since its stop's onset.
END_COLUMN = "t_end_since_stop_onset"
#: The decode's sample (2 ms at 500 Hz).
DECODE_SAMPLE_S = 1 / SAMPLING_FREQUENCY
#: An SWR onset matches its Kay event when their starts differ by at most this
#: (one 2 ms sample, with rounding).
SWR_START_TOLERANCE_S = 0.0025

#: Population -> (event type whose events set the at-risk time, cross population).
POPULATION_PLAN: dict[str, tuple[str, str | None]] = {
    "swr": ("SWR", None),
    "non_local": ("non_local", None),
    "non_local_swr": ("non_local", "non_local_no_swr"),
    "non_local_no_swr": ("non_local", "non_local_swr"),
}

#: Fitted history blocks: name -> (basis column prefix, anchor, self/cross).
HISTORY_BLOCKS: dict[str, tuple[str, str, str]] = {
    "self_onset": ("hist_rcos_", "onset", "self"),
    "self_offset": ("offhist_rcos_", "offset", "self"),
    "cross_onset": ("cross_hist_rcos_", "onset", "cross"),
    "cross_offset": ("cross_offhist_rcos_", "offset", "cross"),
}
#: Per-lag count columns of each history block in the design.
_LAG_COLUMNS = {
    "self_onset": "onset_lag",
    "self_offset": "offset_lag",
    "cross_onset": "cross_lag",
    "cross_offset": "cross_offset_lag",
}


def attach_event_ends(onsets: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    """Add ``t_end_since_stop_onset`` to the non-local and SWR onset rows.

    Non-local: the matched ``event_table`` event's duration (``event_end_s -
    event_start_s``) added to its onset, matched on (session, stop, onset time
    == ``event_start_s``) exactly. SWR: the Kay inventory event whose start is
    nearest the onset (within :data:`SWR_START_TOLERANCE_S`). MUA rows keep NaN.

    Raises
    ------
    ValueError
        If a non-local onset has no event, or an SWR onset no Kay event within
        tolerance.
    """
    out = onsets.copy()
    out[END_COLUMN] = np.nan
    keys = [*SESSION_COLUMNS, "stop_id"]
    nl = out["event_type"] == "non_local"
    ev = event_table[[*keys, "event_start_s", "event_end_s"]].rename(
        columns={"event_start_s": "onset_time"}
    )
    merged = (
        out.loc[nl, [*keys, "onset_time"]]
        .reset_index()
        .merge(ev, on=[*keys, "onset_time"], how="left", validate="one_to_one")
        .set_index("index")
    )
    if merged["event_end_s"].isna().any():
        raise ValueError(
            f"{int(merged['event_end_s'].isna().sum())} non-local onsets have no "
            "event_table event"
        )
    out.loc[merged.index, END_COLUMN] = (
        out.loc[merged.index, "t_since_stop_onset"].to_numpy()
        + (merged["event_end_s"] - merged["onset_time"]).to_numpy()
    )

    swr = out["event_type"] == "SWR"
    for session, g in out.loc[swr].groupby(list(SESSION_COLUMNS), sort=False):
        inv = read_swr_inventory(swr_epoch_id_of_session(*session))
        starts = inv["start_time"].to_numpy(dtype=float)
        ends = inv["end_time"].to_numpy(dtype=float)
        t = g["onset_time"].to_numpy(dtype=float)
        right = np.clip(np.searchsorted(starts, t), 0, len(starts) - 1)
        left = np.clip(right - 1, 0, len(starts) - 1)
        pick = np.where(
            np.abs(starts[left] - t) < np.abs(starts[right] - t), left, right
        )
        err = np.abs(starts[pick] - t)
        if (err > SWR_START_TOLERANCE_S).any():
            raise ValueError(
                f"{session}: {int((err > SWR_START_TOLERANCE_S).sum())} SWR onsets "
                f"have no Kay event within {SWR_START_TOLERANCE_S} s"
            )
        out.loc[g.index, END_COLUMN] = (
            g["t_since_stop_onset"].to_numpy() + ends[pick] - starts[pick]
        )
    return out


def build_design(
    keys: tuple,
    stops_g: pd.DataFrame,
    onsets_g: pd.DataFrame,
    outcome: str,
    *,
    duration_aware: bool = True,
    ladder: Mapping[str, PerBinRateModelSpec] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """One session's frozen design for ``outcome`` and the rungs it carries.

    The design matches ``fit_per_bin_rate_models.py``'s (bins, TMAX, reward
    basis with the outcome shift, excluded time, history window). With
    ``duration_aware`` the bins inside an event of the response's detector are
    dropped and offset history is added: each event covers ``[start + one
    sample, end + one sample)`` (``end`` is its last sample), so the sample in
    which it starts is the sample it was at risk in. ``duration_aware=False``
    builds the shipped design on the same arguments; its frame still carries
    the cross block for the non-local streams.

    Parameters
    ----------
    keys : tuple
        The session's (animal, session_date, epoch).
    stops_g, onsets_g : pd.DataFrame
        The session's stops and onsets; the onsets carry ``is_swr`` and
        :data:`END_COLUMN` (:func:`attach_event_ends`).
    outcome : str
        A key of :data:`POPULATION_PLAN`.
    ladder : mapping of str to PerBinRateModelSpec, optional
        Duration-aware rungs to build and freeze the design for instead of
        ``MODEL_CROSS_HISTORY_DURATION_RATE`` (e.g.
        ``MODEL_CROSS_HISTORY_LONG_RATE``): the cross block's lags and basis
        and the history window are the ladder's longest. Requires a population
        with a cross stream.

    Returns
    -------
    design : pd.DataFrame
        The frozen (stop x bin) design.
    rungs : tuple of str
        The ``MODEL_CROSS_HISTORY_DURATION_RATE`` (or ``ladder``) rungs the
        population fits.

    Raises
    ------
    ValueError
        If the freeze drops a stop (every required column is complete by
        construction).
    """
    event_type, cross_pop = POPULATION_PLAN[outcome]
    response = POPULATIONS[outcome](onsets_g)
    at_risk = onsets_g.loc[
        onsets_g["event_type"] == event_type,
        ["stop_id", "t_since_stop_onset", END_COLUMN],
    ].copy()
    at_risk[["t_since_stop_onset", END_COLUMN]] += DECODE_SAMPLE_S
    cross_kwargs: dict = {}
    rungs = SELF_RUNGS
    cross_lags, cross_basis, window_s = (
        N_HISTORY_LAGS,
        N_HISTORY_BASIS,
        HISTORY_WINDOW_S,
    )
    if ladder is not None:
        if cross_pop is None or not duration_aware:
            raise ValueError(f"{outcome}: a ladder needs a duration-aware cross design")
        cross_lags = max(spec.cross_history_lags for spec in ladder.values())
        cross_basis = max(spec.cross_history_basis for spec in ladder.values())
        window_s = longest_history_window_s(ladder.values())
    if cross_pop is not None:
        cross_kwargs = {
            "cross_onset_times": POPULATIONS[cross_pop](onsets_g),
            "cross_history_lags": cross_lags,
            "cross_history_basis": cross_basis,
        }
        rungs = SELF_RUNGS + CROSS_RUNGS if ladder is None else tuple(ladder)
    design = aggregate_to_timebins(
        stops_g,
        response,
        bin_s=BIN_S,
        tmax=TMAX,
        reward_basis=True,
        self_history_lags=N_HISTORY_LAGS,
        self_history_basis=N_HISTORY_BASIS,
        stop_covariates=STOP_COVARIATES,
        history_window_s=window_s,
        arrival_clocks=False,
        at_risk_events=at_risk if duration_aware else None,
        offset_history=duration_aware,
        **cross_kwargs,
    )
    specs = (
        list(ladder.values())
        if ladder is not None
        else [LADDER[name] for name in rungs]
        if duration_aware
        else [
            MODEL_CROSS_HISTORY_RATE["Mselfcross" if cross_pop is not None else "Mself"]
        ]
    )
    design, freeze = freeze_per_bin_cohort(design, specs)
    if freeze["n_stops_before"] != freeze["n_stops_after"]:
        raise ValueError(
            f"{keys}/{outcome}: freezing dropped "
            f"{freeze['n_stops_before'] - freeze['n_stops_after']} stops"
        )
    return design, rungs


def history_kernels(fit: dict) -> dict[str, np.ndarray]:
    """Per-lag log rate ratio of each fitted history block.

    Parameters
    ----------
    fit : dict
        A :func:`per_bin_rate_model.fit_per_bin_rate_model` result (or any
        mapping with ``coef_{prefix}{k}`` entries).

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
        Block name -> shape ``(n_lags,)``, for each block whose lag columns
        ``design`` carries (``n_lags`` is the block's own window: the long
        cross block has more lags than the self block).
    """
    rate = design["count"].sum() / design["exposure"].sum()
    exposure = design["exposure"].to_numpy(dtype=float)
    out = {}
    for name, prefix in _LAG_COLUMNS.items():
        n_lags = 0
        while f"{prefix}{n_lags + 1}" in design.columns:
            n_lags += 1
        cols = [f"{prefix}{j}" for j in range(1, n_lags + 1)]
        if n_lags:
            present = design[cols].to_numpy() > 0  # (n_bins, n_lags)
            out[name] = rate * (exposure @ present)
    return out


def onset_offset_collinearity(design: pd.DataFrame) -> dict[str, float]:
    """How nearly the onset block's span contains the offset block's.

    For the self and the cross stream: the first canonical correlation between
    the onset (``*hist_rcos_k``) and offset (``*offhist_rcos_k``) blocks, over
    the design's bins with any history in either block. With constant durations
    an offset is an onset shifted by a fixed lag, so some combination of one
    block nearly reproduces one of the other and the correlation nears 1 (the
    two anchors cannot be told apart); varying durations lower it. The
    raised-cosine bumps within one block are correlated by construction, which
    this measure, unlike a VIF over all eight columns, does not count.

    Returns
    -------
    dict
        ``canonical_corr_self`` / ``canonical_corr_cross`` for each stream whose
        two blocks ``design`` carries.
    """
    out = {}
    for rel, prefix in (("self", ""), ("cross", "cross_")):
        blocks = [
            [f"{prefix}{anchor}hist_rcos_{k}" for k in range(N_HISTORY_BASIS)]
            for anchor in ("", "off")
        ]
        if not all(c in design.columns for cols in blocks for c in cols):
            continue
        onset, offset = (design[cols].to_numpy(dtype=float) for cols in blocks)
        rows = (onset != 0).any(axis=1) | (offset != 0).any(axis=1)
        bases = []
        for x in (onset[rows], offset[rows]):
            u, s, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
            bases.append(u[:, s > s[0] * 1e-8])
        out[f"canonical_corr_{rel}"] = float(
            np.linalg.svd(bases[0].T @ bases[1], compute_uv=False)[0]
        )
    return out


def fit_pooled_history(
    design: pd.DataFrame, spec: PerBinRateModelSpec, *, group: str = "session"
) -> dict:
    """Fit ``spec`` to many sessions' stacked designs with a session effect.

    The rung's formula plus ``C(group)``, Poisson with a ``log(exposure)``
    offset: the time spline, reward basis and history kernels are shared over
    the stacked sessions, each session keeps its own baseline rate. A single
    session carries about one expected onset per history lag, too few to read a
    kernel; stacking an animal's sessions makes the kernel measurable.

    Parameters
    ----------
    design : pd.DataFrame
        Stacked :func:`build_design` frames with a ``group`` column.
    spec : PerBinRateModelSpec
    group : str
        The column naming each row's session.

    Returns
    -------
    dict
        ``ok``; when the fit ran: ``aic``, ``llf``, ``n_params``,
        ``converged``, ``n_convergence_warnings`` (convergence and separation
        warnings statsmodels raised), ``warnings`` (their distinct messages),
        ``seconds`` and ``kernels`` (block -> ``log_rr`` and its Wald ``se``
        per lag, and the basis ``coef``); otherwise ``error``.
    """
    cols = [c for c in spec.design_columns() if c != "count"]
    df = design[["count", "exposure", group, *cols]]
    t0 = time.perf_counter()
    try:
        with capture_fit_warnings() as caught:
            res = smf.glm(
                spec.formula() + f" + C({group})",
                data=df,
                family=sm.families.Poisson(),
                offset=np.log(df["exposure"].to_numpy(dtype=float)),
            ).fit()
    except EXPECTED_FIT_ERRORS as exc:
        return {"ok": False, "error": repr(exc)}
    basis = raised_cosine_history_basis(N_HISTORY_LAGS, N_HISTORY_BASIS)
    cov = res.cov_params()
    kernels = {}
    for block, (prefix, _, _) in HISTORY_BLOCKS.items():
        names = [f"{prefix}{k}" for k in range(N_HISTORY_BASIS)]
        if all(n in res.params.index for n in names):
            beta = res.params[names].to_numpy()
            sigma = cov.loc[names, names].to_numpy()
            kernels[block] = {
                "log_rr": (beta @ basis).tolist(),
                "se": np.sqrt(np.einsum("kj,kl,lj->j", basis, sigma, basis)).tolist(),
                "coef": beta.tolist(),
            }
    return {
        "ok": True,
        "aic": float(res.aic),
        "llf": float(res.llf),
        "n_params": int(len(res.params)),
        "converged": bool(res.converged),
        "n_convergence_warnings": count_fit_quality_warnings(caught),
        "warnings": sorted({str(w.message) for w in caught}),
        "seconds": round(time.perf_counter() - t0, 1),
        "kernels": kernels,
    }
