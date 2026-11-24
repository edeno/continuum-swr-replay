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
    nested_bootstrap_copies,
    nested_codes,
    session_copy_values,
    weighted_medians,
)


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


def _replicates(
    values: NDArray, animal: NDArray, epoch: NDArray, rng: np.random.Generator, n: int
) -> NDArray:
    """The statistic on ``n`` nested bootstrap replicates of the events.

    Each replicate draws the animals with replacement, then each drawn animal's
    epochs, then each drawn epoch's events, every drawn copy on its own
    (``nested_bootstrap_copies``, ``session_copy_values``): an animal drawn
    twice is two animals, each with its own epochs, and an epoch drawn twice
    is two epochs, each with its own events. Each drawn epoch takes the median
    of its events, each drawn animal the median of its epochs' medians, and
    the replicate the mean over its animals, as the statistic does on the data.

    Parameters
    ----------
    values : ndarray, shape (n_events,)
        Finite.
    animal, epoch : ndarray of int, shape (n_events,)
        Codes (``nested_codes``).

    Returns
    -------
    ndarray, shape (n,)
    """
    n_epochs = int(epoch.max()) + 1
    epoch_animal = np.zeros(n_epochs, dtype=np.int64)
    epoch_animal[epoch] = animal
    copies = nested_bootstrap_copies(epoch_animal, rng, n)
    sorted_events = [np.sort(values[epoch == e]) for e in range(n_epochs)]

    def epoch_medians(e: int, n_copies: int) -> NDArray:
        events = sorted_events[e]
        drawn = rng.multinomial(
            events.size, np.full(events.size, 1.0 / events.size), size=n_copies
        )
        return weighted_medians(np.broadcast_to(events, drawn.shape), drawn)

    animal_medians = np.nanmedian(session_copy_values(copies, epoch_medians), axis=2)
    return animal_medians.mean(axis=1)


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
    statistic on nested bootstrap replicates of the events (:func:`_replicates`:
    the animals, then their epochs, then each epoch's events).

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
        ``n_events``, ``n_scored`` (events with a value), ``n_animals`` and
        ``n_animals_positive`` (animals whose value is above zero).
    """
    rows = {}
    for group, mine in table.groupby("group"):
        scored = mine.dropna(subset=[column])
        per_animal = animal_session_values(
            scored[column], scored["animal"], scored["epoch_id"], "median"
        )
        animal_codes, epoch_codes = nested_codes(scored["animal"], scored["epoch_id"])
        replicate = _replicates(
            scored[column].to_numpy(dtype=float),
            animal_codes,
            epoch_codes,
            np.random.default_rng(seed),
            n_boot,
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
            "n_animals_positive": int((per_animal > 0).sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")
