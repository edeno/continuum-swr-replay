"""Spike likelihood at decoded and actual positions, over each whole non-local event.

During a detected event, how much more strongly does the population activity
favor the decoded positions than the animal's measured positions? For each time
bin t of the event, with the decoder's stored clusterless log-likelihood
``l(t, x)`` (every tetrode's spike waveforms and expected firing, the
Non-Local Continuous state's columns; ``spike_evidence.stored_likelihood_columns``):

    delta(t) = l(t, decoded(t)) - l(t, animal(t))

where ``decoded(t)`` is the frame's ``map_estimate`` (the mode of the
conditional non-local posterior) and ``animal(t)`` the animal's linear position,
each at its nearest interior place bin. The event's

- ``D`` is the sum of ``delta`` over its bins (nats),
- ``D_bin`` their mean (nats per bin),
- ``D_spike`` = ``D`` / the event's spike count (nats per spike), undefined for
  an event without spikes.

A bin whose decoded or animal position is not finite is left out; the event
records how many of its bins were scored. The same spikes helped detect the
event and estimate its positions, which favors positive scores: the score
measures the size of the likelihood advantage within detected events.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    nearest_interior_bins,
)
from continuum_swr_replay.analysis.shared_statistics import (
    animal_session_values,
    bootstrap_interval,
    nested_bootstrap_draws,
    nested_codes,
    weighted_median,
)

GROUPS = ("no SWR", "SWR")
"""The event groups, by the event table's ``is_swr``."""


def advantage_from_bins(
    log_likelihood: NDArray, decoded_bin: NDArray, animal_bin: NDArray
) -> NDArray:
    """``delta(t)`` from each row's decoded and animal place-bin columns.

    Parameters
    ----------
    log_likelihood : ndarray, shape (n_rows, n_bins)
        The stored log-likelihood of each scored time bin over interior place
        bins.
    decoded_bin, animal_bin : ndarray of int, shape (n_rows,)

    Returns
    -------
    ndarray, shape (n_rows,)
    """
    rows = np.arange(len(decoded_bin))
    return log_likelihood[rows, decoded_bin] - log_likelihood[rows, animal_bin]


def per_bin_advantage(
    log_likelihood: NDArray, decoded_cm: NDArray, animal_cm: NDArray, env: Any
) -> NDArray:
    """``delta(t)`` of each time bin, NaN where either position is not finite.

    Parameters
    ----------
    log_likelihood : ndarray, shape (n_rows, n_interior_bins)
    decoded_cm, animal_cm : ndarray, shape (n_rows,)
        Linear positions, cm.
    env : Environment
        Fitted; its interior bins are the columns of ``log_likelihood``.
    """
    decoded_cm, animal_cm = np.asarray(decoded_cm), np.asarray(animal_cm)
    valid = np.isfinite(decoded_cm) & np.isfinite(animal_cm)
    delta = np.full(len(decoded_cm), np.nan)
    if valid.any():
        delta[valid] = advantage_from_bins(
            np.asarray(log_likelihood)[valid],
            nearest_interior_bins(env, decoded_cm[valid]),
            nearest_interior_bins(env, animal_cm[valid]),
        )
    return delta


def event_scores(delta: NDArray, lengths: NDArray, n_spikes: NDArray) -> pd.DataFrame:
    """Each event's ``D``, ``D_bin`` and ``D_spike`` from its bins' ``delta``.

    Parameters
    ----------
    delta : ndarray, shape (sum(lengths),)
        The events' bins, event after event (NaN for a bin not scored).
    lengths : ndarray of int, shape (n_events,)
        Each event's bins.
    n_spikes : ndarray, shape (n_events,)
        Each event's spike count over every tetrode.

    Returns
    -------
    pd.DataFrame, shape (n_events, 6)
        ``n_bins``, ``n_scored_bins``, ``D`` (NaN when no bin is scored),
        ``D_bin``, ``n_spikes`` and ``D_spike`` (NaN without spikes or a ``D``).
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    scored = np.isfinite(delta)
    n_scored = np.add.reduceat(scored.astype(np.int64), starts) if len(lengths) else []
    total = (
        np.add.reduceat(np.where(scored, delta, 0.0), starts) if len(lengths) else []
    )
    n_scored = np.asarray(n_scored)
    D = np.where(n_scored > 0, total, np.nan)
    n_spikes = np.asarray(n_spikes, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        D_bin = np.where(n_scored > 0, D / n_scored, np.nan)
        D_spike = np.where(n_spikes > 0, D / n_spikes, np.nan)
    return pd.DataFrame(
        {
            "n_bins": lengths,
            "n_scored_bins": n_scored,
            "D": D,
            "D_bin": D_bin,
            "n_spikes": n_spikes.astype(np.int64),
            "D_spike": D_spike,
        }
    )


def _animal_median_mean(
    epoch_values: NDArray, animal_codes: NDArray, weights: NDArray, animal_weights
) -> float:
    """The mean over animals of each animal's weighted median of its epochs."""
    total, mass = 0.0, 0.0
    for a in np.flatnonzero(animal_weights > 0):
        mine = animal_codes == a
        order = np.argsort(epoch_values[mine])
        w = weights[mine][order]
        if w.sum() == 0:
            continue
        total += animal_weights[a] * weighted_median(epoch_values[mine][order], w)
        mass += animal_weights[a]
    return total / mass if mass else float("nan")


def likelihood_advantage_statistics(
    table: pd.DataFrame,
    *,
    column: str = "D_spike",
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Each group's ``column``: one value per animal, their mean and its interval.

    An animal's value is the median over its epochs of each epoch's median
    ``column`` (``animal_session_values(..., "median")``). The estimate is the
    mean over animals; its 95 % interval is ``bootstrap_interval`` of the same
    statistic on nested bootstrap replicates of the epochs' medians (animals,
    then their epochs).

    Parameters
    ----------
    table : pd.DataFrame
        One row per event: ``animal``, ``epoch_id``, ``group`` and ``column``
        (``D_spike`` by default; the held-out score's ``gain_per_spike`` too).
    n_boot : int, optional
    seed : int, optional

    Returns
    -------
    pd.DataFrame
        Indexed by group: ``animals`` (the animals' values, sorted animal
        order), ``animal_names``, ``estimate``, ``ci_low``, ``ci_high``,
        ``n_events``, ``n_scored`` (events with a value) and ``n_animals``.
    """
    rows = {}
    for group, mine in table.groupby("group"):
        scored = mine.dropna(subset=[column])
        per_animal = animal_session_values(
            scored[column], scored["animal"], scored["epoch_id"], "median"
        )
        epochs = scored.groupby(["animal", "epoch_id"])[column].median().reset_index()
        animal_codes, epoch_codes = nested_codes(epochs["animal"], epochs["epoch_id"])
        draws = nested_bootstrap_draws(
            animal_codes, epoch_codes, np.random.default_rng(seed), n_boot
        )
        values = epochs[column].to_numpy()
        replicate = np.array(
            [
                _animal_median_mean(
                    values, animal_codes, draws.rows[i], draws.animals[i]
                )
                for i in range(n_boot)
            ]
        )
        estimate = float(per_animal.mean())
        interval = bootstrap_interval(estimate, replicate, len(per_animal))
        rows[group] = {
            "animals": per_animal.to_numpy(),
            "animal_names": per_animal.index.to_numpy(),
            "estimate": estimate,
            "ci_low": interval.low,
            "ci_high": interval.high,
            "n_events": len(mine),
            "n_scored": len(scored),
            "n_animals": len(per_animal),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")
