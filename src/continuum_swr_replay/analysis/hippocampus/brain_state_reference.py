"""Theta, ripple and MUA levels of running and of quiet rest, the references for an event's.

Each event's ``theta_z``, ``ripple_power_z`` and ``mua_rate_z`` in the event
table are means, over its bins, of the decode frame's per-epoch z-scores
(``theta_power_zscore``, ``SWR_zscore``, ``MUA_zscore``). On that same scale,
this gives one epoch's levels during running and during quiet rest, so an event
can be read against them: an event whose theta reaches the running level is
theta-like.

- **Running**: bins above 10 cm/s (``state_evidence.windows.regimes``).
- **Quiet rest**: bins below 4 cm/s that are in no non-local, SWR or MUA event.

Bins in excluded time (``in_excluded_time``) are left out of both.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.state_evidence.windows import regimes

SIGNALS = {
    "theta": "theta_power_zscore",
    "ripple": "SWR_zscore",
    "mua": "MUA_zscore",
}
"""Each signal's word and its decode-frame column."""

EVENT_COLUMNS = ("non_local_event", "SWR_event", "MUA_event")
"""The frame's event labels; 0 is no event."""


def regime_masks(dot: pd.DataFrame) -> dict[str, NDArray[np.bool_]]:
    """The running and quiet-rest bins of a decode frame.

    Running: above 10 cm/s; quiet rest: below 4 cm/s and in no non-local, SWR
    or MUA event (:data:`EVENT_COLUMNS`); neither in excluded time.

    Parameters
    ----------
    dot : pd.DataFrame
        A decode frame read through the SWR source, with ``head_speed``,
        ``in_excluded_time`` and :data:`EVENT_COLUMNS`.

    Returns
    -------
    masks : dict
        ``"running"`` and ``"rest"``, each a bool array of shape (n_time,).
    """
    usable = ~dot["in_excluded_time"].to_numpy(dtype=bool)
    regime = regimes(dot["head_speed"].to_numpy(dtype=np.float64))
    in_event = np.zeros(len(dot), dtype=bool)
    for column in EVENT_COLUMNS:
        in_event |= dot[column].to_numpy() > 0
    return {
        "running": usable & (regime == "running"),
        "rest": usable & (regime == "immobile") & ~in_event,
    }


def matched_window_means(
    values: NDArray,
    eligible: NDArray[np.bool_],
    lengths: NDArray[np.int64],
    n_windows: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Means of ``values`` over random windows of eligible bins, one length per event.

    For each length ``L``, ``n_windows`` windows of ``L`` consecutive bins
    that are all ``eligible`` (for example quiet rest), drawn uniformly over
    every such window. A window's value is the mean of its finite ``values``,
    as an event's ``theta_z`` is the mean over its bins.

    Parameters
    ----------
    values : NDArray, shape (n_time,)
    eligible : NDArray of bool, shape (n_time,)
    lengths : NDArray of int, shape (n_events,)
        Each event's number of bins.
    n_windows : int
    rng : np.random.Generator

    Returns
    -------
    means : NDArray, shape (n_events, n_windows)
        NaN for an event whose length no eligible run holds, or a window with
        no finite value.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    total = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0))])
    count = np.concatenate([[0], np.cumsum(finite)])
    edges = np.diff(np.concatenate([[0], np.asarray(eligible, dtype=np.int8), [0]]))
    run_start = np.flatnonzero(edges == 1)
    run_length = np.flatnonzero(edges == -1) - run_start
    means = np.full((len(lengths), n_windows), np.nan)
    for k, length in enumerate(np.asarray(lengths, dtype=np.int64)):
        n_starts = np.clip(run_length - length + 1, 0, None)
        available = n_starts.sum()
        if available == 0:
            continue
        offsets = rng.integers(0, available, n_windows)
        cumulative = np.cumsum(n_starts)
        run = np.searchsorted(cumulative, offsets, side="right")
        starts = run_start[run] + offsets - (cumulative[run] - n_starts[run])
        n_finite = count[starts + length] - count[starts]
        with np.errstate(invalid="ignore", divide="ignore"):
            means[k] = np.where(
                n_finite > 0,
                (total[starts + length] - total[starts]) / n_finite,
                np.nan,
            )
    return means


def brain_state_reference(
    dot: pd.DataFrame, signals: Mapping[str, str] = SIGNALS
) -> dict[str, float]:
    """One epoch's signal levels during running and during quiet rest.

    Parameters
    ----------
    dot : pd.DataFrame
        A decode frame read through the SWR source
        (``analysed_time.with_swr_source``), with ``head_speed``,
        ``in_excluded_time``, :data:`EVENT_COLUMNS` and the columns of
        ``signals``.
    signals : Mapping[str, str]
        Each signal's word and its column in ``dot``; :data:`SIGNALS` by
        default.

    Returns
    -------
    reference : dict
        ``n_running_bins``, ``n_rest_bins``, and for each signal and regime
        ``{signal}_{regime}_median`` and ``{signal}_{regime}_mean`` (NaN when
        the regime has no bin).
    """
    masks = regime_masks(dot)
    reference: dict[str, float] = {
        f"n_{name}_bins": int(mask.sum()) for name, mask in masks.items()
    }
    for word, column in signals.items():
        values = dot[column].to_numpy(dtype=np.float64)
        for name, mask in masks.items():
            chosen = values[mask]
            chosen = chosen[np.isfinite(chosen)]
            reference[f"{word}_{name}_median"] = (
                float(np.median(chosen)) if chosen.size else np.nan
            )
            reference[f"{word}_{name}_mean"] = (
                float(chosen.mean()) if chosen.size else np.nan
            )
    return reference
