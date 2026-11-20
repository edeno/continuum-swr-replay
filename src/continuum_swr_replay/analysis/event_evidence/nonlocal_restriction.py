"""Is the non-local activity of rest the activity of sharp-wave ripples? Three
decoders that differ only in when the non-local states may be occupied, scored
by how well each predicts each bin's spikes.

The models share the epoch's decoder: its encoding, its continuous transitions
``C`` and its fitted discrete transitions ``D`` (no per-model refit). They differ
only in the discrete states allowed at each time bin, ``a_t`` in {0, 1}^S:

- ``none``: the local states (Local and No-Spike) only, at every bin;
- ``swr``: an SWR is non-local activity. Inside an SWR bin (not padded) only
  the non-local states, outside SWRs only the local states: the SWR times set
  the state;
- ``full``: every state at every bin (the paper's decoder).

**The restricted model.** State bins s = 1..N (the track-interior bins of every
state), with discrete state ``d(s)``. The library's joint transition is
``T[s', s] = C[s', s] D[d(s'), d(s)]`` (row: the bin at t - 1). Where a state is
forbidden at t, the transitions into it are removed and the rows of the
*discrete* matrix renormalized over the allowed destinations:

- ``D~_t[i, j] = D[i, j] a_t[j] / sum_k D[i, k] a_t[k]``;
- ``T~_t[s', s] = C[s', s] D~_t[d(s'), d(s)]``, ``a_t`` the allowed set at the
  destination time t.

So at an SWR's first bin ``swr`` enters a non-local state with probability 1,
split between Continuous and Fragmented as the row's renormalized entries say,
and at the first bin after it returns to Local or No-Spike the same way. A row
with no allowed destination is an error. With every state allowed, ``D~ = D``
and the model is the library's decoder. Each restricted model is a hidden
Markov model whose transitions depend on a known input (the SWR indicator). It
is a proper one (rows of ``T~_t`` sum to 1) because each block of ``C``, the
columns of one destination state, has rows that sum to 1
(:func:`continuous_block_deviation` reports how far they are from it).

**Reading the contrasts.** ``swr - none`` is what calling every SWR non-local
buys. ``full`` and ``swr`` differ in two places: inside SWRs ``swr`` must be
non-local where ``full`` may stay local, and outside SWRs ``swr`` cannot be
non-local where ``full`` may be. ``full - swr`` over a set of bins mixes the
two, so it is read per category of bins (:func:`restriction_sums`): over bins
inside SWRs it is the cost of forcing the state, over bins outside them what
the non-local states buy there. The share of the non-local gain that lies
outside SWRs (:func:`share_outside_swr`) is taken from ``full - none`` alone.
Allowing a state is not free: a model that may leave Local puts predictive
mass on the non-local states, which costs it wherever activity stays local.

**Two scores**, both read from the library's filter and smoother
(``chunked_filter_smoother_covariate_dependent``, started from
``pi a_0[d(.)]`` renormalized; this module has no recursion of its own). With
``pred_t`` the predictive posterior, ``gamma_t`` the smoothed one and ``L_t =
exp(ll[t])``:

- the filter score, ``log p(y_t | y_1..y_{t-1}) = log sum_s pred_t[s] L_t[s]``.
  The per-bin scores add up to the model's log marginal likelihood of the
  epoch, so a sum over any set of bins is that set's share of it;
- the two-sided score, ``log p(y_t | y_{-t})``: the bin's spikes predicted from
  every other bin, earlier and later. With ``r_t[s] = sum_{s'} T~_{t+1}[s, s']
  gamma_{t+1}[s'] / pred_{t+1}[s']`` (the factor of the smoother's step),
  ``gamma_t = pred_t L_t r_t / c_t`` with ``c_t`` the filter score's
  probability, and ``p(x_t | y_{-t})`` is proportional to ``pred_t r_t``, so
  ``p(y_t | y_{-t}) = c_t / sum_s pred_t[s] r_t[s]`` and ``sum_s pred_t[s]
  r_t[s] = sum_s q_{t+1}[s] gamma_{t+1}[s] / pred_{t+1}[s]``, ``q_{t+1} = pred_t
  T~_{t+1}``. The term of a state bin with ``pred_{t+1}[s] = 0`` is dropped.
  That is exact: ``gamma_{t+1}`` is proportional to ``pred_{t+1}``, so it is 0
  there too, and ``q_{t+1}[s]`` is 0 wherever the bin is forbidden at t + 1;
  only a predicted mass that underflows to 0 in float32 where the exact one is
  positive (and its smoothed mass with it) is lost. The last bin's two-sided
  score is its filter score. The two-sided scores do not add up to a
  likelihood.

The log-likelihood ``ll`` is held out in time, as in :mod:`.tetrode_holdout`:
the epoch is cut into :data:`~.tetrode_holdout.N_FOLDS` blocks and each block's
bins are scored with encodings fit on the other blocks, with every tetrode in
the model. A missing bin's likelihood is 0, so it scores 0 on both, to float32
rounding of the posterior's mass.

What the score does not show: that the activity it credits is coherent spatial
content. Any activity the Local state predicts poorly and a non-local state
predicts less poorly is credited: in simulation
(``tests/integration/test_nonlocal_restriction_simulated.py``) a burst whose
tetrodes each follow their own random place gains over ``none`` about as much
as one whose tetrodes share a random place, because at 2 ms a bin holds about
one spike and the Non-Local Fragmented state can follow each spike's place.
SWRs are not padded, so non-local activity that begins before its SWR or ends
after it is counted outside SWRs. Nor does it show anything about a transition
matrix other than the fitted one, which was fit on the same epoch with every
state allowed.
"""

from __future__ import annotations

import time as clock
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    analyzed_bins,
    immobile_analyzed_bins,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    refit_encoding,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    LOCAL,
    N_FOLDS,
    NO_SPIKE,
    heldout_terms,
    state_log_likelihood,
)
from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

MODELS = ("none", "swr", "full")
"""The three models: never non-local, non-local exactly in SWRs, unrestricted."""

LOCAL_STATES = (LOCAL, NO_SPIKE)
"""The local states; every other state is a non-local one."""

FAMILY_SUFFIX = {"filter": "", "two_sided": "_two_sided"}
"""The two scores, ``log p(y_t | y_1..y_{t-1})`` and ``log p(y_t | y_{-t})``,
and the suffix of their columns: a model's filter score is column ``<model>``,
its two-sided score ``<model>_two_sided``."""

SCORE_COLUMNS = tuple(
    f"{model}{suffix}" for suffix in FAMILY_SUFFIX.values() for model in MODELS
)
"""The six score columns, the filter scores first."""

IMMOBILE_IN_SWR = "immobile_in_swr"
IMMOBILE_OUTSIDE_SWR = "immobile_outside_swr"
CATEGORIES = ("excluded", IMMOBILE_IN_SWR, IMMOBILE_OUTSIDE_SWR, "moving")
"""Each bin's category (:func:`bin_categories`)."""

CAUSAL_MASS_TOLERANCE = 1e-3
"""Largest allowed distance of a model's filtered posterior mass from 1."""


def bin_categories(
    frame: pd.DataFrame,
    in_swr: NDArray[np.bool_],
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.Categorical:
    """Each bin's category: excluded, immobile in or outside an SWR, or moving.

    Parameters
    ----------
    frame : pd.DataFrame
        A decode frame read through ``analyzed_time.read_dot_with_swr_source``
        (``in_excluded_time`` and ``head_speed``), one row per bin.
    in_swr : ndarray of bool, shape (n_time,)
        Whether each bin lies inside an SWR.
    speed_threshold : float
        cm/s; a bin is immobile when its speed is strictly below it
        (``analyzed_time.immobile_analyzed_bins``).

    Returns
    -------
    pd.Categorical, length n_time
        Categories :data:`CATEGORIES`: ``excluded`` where the bin is not
        analyzed; else :data:`IMMOBILE_IN_SWR` / :data:`IMMOBILE_OUTSIDE_SWR`
        where it is immobile; else ``moving`` (a bin without a speed included).
    """
    in_swr = np.asarray(in_swr, dtype=bool)
    if in_swr.shape != (len(frame),):
        raise ValueError(f"in_swr has shape {in_swr.shape}, not ({len(frame)},)")
    analyzed = analyzed_bins(frame)
    immobile = immobile_analyzed_bins(frame, speed_threshold)
    labels = np.select(
        [~analyzed, immobile & in_swr, immobile],
        ["excluded", IMMOBILE_IN_SWR, IMMOBILE_OUTSIDE_SWR],
        default="moving",
    )
    return pd.Categorical(labels, categories=CATEGORIES)


def allowed_states(
    state_names: Sequence[str], in_swr: NDArray
) -> dict[str, NDArray[np.bool_]]:
    """Each model's allowed discrete states at each bin.

    Parameters
    ----------
    state_names : sequence of str, length n_states
    in_swr : ndarray of bool, shape (n_time,)

    Returns
    -------
    dict of str to ndarray of bool, shape (n_time, n_states)
        ``none`` (the local states at every bin), ``swr`` (the non-local
        states inside SWRs, the local states outside them) and ``full`` (every
        state at every bin) (:data:`MODELS`).
    """
    in_swr = np.asarray(in_swr, dtype=bool)
    local = np.isin(np.asarray(state_names), LOCAL_STATES)
    every = np.ones(local.size, dtype=bool)
    return {
        "none": np.broadcast_to(local, (in_swr.size, local.size)),
        "swr": np.where(in_swr[:, None], ~local, local),
        "full": np.broadcast_to(every, (in_swr.size, local.size)),
    }


def _row_normalizers(discrete: NDArray, patterns: NDArray) -> NDArray:
    """``sum_k D[i, k] a[k]`` for every pattern and row, shape (n_patterns,
    n_states); raises where a row has no allowed destination."""
    normalizers = np.asarray(patterns, dtype=float) @ np.asarray(discrete).T
    empty = np.argwhere(normalizers <= 0)
    if empty.size:
        p, i = empty[0]
        raise ValueError(
            f"allowed pattern {np.asarray(patterns, dtype=int)[p].tolist()} leaves "
            f"discrete state {int(i)} with no allowed destination"
        )
    return normalizers


def restricted_discrete_transitions(discrete: NDArray, patterns: NDArray) -> NDArray:
    """The renormalized discrete transitions of each allowed pattern,
    ``D~[p, i, j] = D[i, j] a_p[j] / sum_k D[i, k] a_p[k]``.

    Parameters
    ----------
    discrete : ndarray, shape (n_states, n_states)
        ``D[i, j]``, from state i to state j.
    patterns : ndarray of bool, shape (n_patterns, n_states)
        Each pattern's allowed destination states.

    Returns
    -------
    ndarray, shape (n_patterns, n_states, n_states)

    Raises
    ------
    ValueError
        When a pattern leaves a row of ``D`` with no allowed destination.
    """
    discrete = np.asarray(discrete, dtype=float)
    patterns = np.asarray(patterns, dtype=bool)
    normalizers = _row_normalizers(discrete, patterns)
    return discrete[None] * patterns[:, None, :] / normalizers[:, :, None]


def continuous_block_deviation(continuous: NDArray, state_ind: NDArray) -> float:
    """How far the rows of each destination state's block of ``C`` are from
    summing to 1: ``max |sum_{s: d(s) = j} C[s', s] - 1|`` over rows s' and
    states j.

    Parameters
    ----------
    continuous : ndarray, shape (n_state_bins, n_state_bins)
    state_ind : ndarray of int, shape (n_state_bins,)
    """
    continuous = np.asarray(continuous, dtype=float)
    state_ind = np.asarray(state_ind)
    sums = continuous @ (state_ind[:, None] == np.unique(state_ind)[None, :])
    return float(np.abs(sums - 1.0).max())


#: Bins per batch of the two-sided step.
CHUNK_BINS = 20_000

MARGINAL_TOLERANCE = 5e-3
"""Allowed relative difference between the library's marginal log-likelihood
and the sum of the per-bin filter scores.

The check guards the wiring (the order of the arrays the library returns, the
alignment of the likelihood with the transitions), not precision: it could not
catch an error the size of a model contrast, and no contrast uses the library's
total. The tolerance is loose because the pinned library (non_local_detector
2d730c3a in ``uv.lock``) adds its per-bin terms into a float32 total whose
rounding grows with the number of bins: 1.8e-4 of the total over 93,000 bins of
the simulated recording, and about 8e-4 over the 654,000 bins of a real epoch
(``wilbur20210329_02_r1_HPC``), where the total drifted by 5,100 to 6,600 nats,
by a different amount for each model. The per-bin terms themselves agree to
about 1e-5 nats. Upstream main fixed the total (23369aa3, 2026-11-11); the
tolerance stays harmless after an upgrade."""


@jax.jit
def _bin_scores(
    predicted: jnp.ndarray,
    log_likelihood: jnp.ndarray,
    predicted_next: jnp.ndarray,
    smoothed_next: jnp.ndarray,
    next_pattern: jnp.ndarray,
    joint: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A batch of bins' filter and two-sided scores from the library's
    posteriors.

    Parameters
    ----------
    predicted, log_likelihood : jnp.ndarray, shape (n_batch, n_state_bins)
        ``pred[t]`` and ``ll[t]``.
    predicted_next, smoothed_next : jnp.ndarray, shape (n_batch, n_state_bins)
        ``pred[t + 1]`` and ``gamma[t + 1]``.
    next_pattern : jnp.ndarray of int, shape (n_batch,)
        The allowed pattern of bin t + 1.
    joint : jnp.ndarray, shape (n_patterns, n_state_bins, n_state_bins)
        ``T~`` of each pattern.

    Returns
    -------
    filter_score, two_sided : jnp.ndarray, shape (n_batch,)
    """
    has_mass = predicted > 0
    terms = jnp.where(
        has_mass,
        jnp.log(jnp.where(has_mass, predicted, 1.0)) + log_likelihood,
        -jnp.inf,
    )
    top = terms.max(axis=1, keepdims=True)
    filter_score = jnp.log(jnp.exp(terms - top).sum(axis=1)) + top[:, 0]
    # q[t + 1] = pred[t] @ T~_{t+1}: the prediction of bin t + 1 that skips
    # bin t's spikes
    # at full float32 precision: at the GPU's default (reduced) precision this
    # product shifted the two-sided scores by about 5e-4 nats a bin, by a
    # different amount for each model, which moved a two-sided model contrast
    # of a real epoch by about 1 % (the filter score was unaffected)
    skipped = jnp.zeros_like(predicted)
    for p in range(joint.shape[0]):
        skipped = skipped + (next_pattern == p)[:, None] * jnp.matmul(
            predicted, joint[p], precision=jax.lax.Precision.HIGHEST
        )
    next_mass = predicted_next > 0
    ratio = jnp.where(
        next_mass, smoothed_next / jnp.where(next_mass, predicted_next, 1.0), 0.0
    )
    return filter_score, filter_score - jnp.log((skipped * ratio).sum(axis=1))


def predictive_log_scores(
    log_likelihood: NDArray,
    initial: NDArray,
    continuous: NDArray,
    discrete: NDArray,
    state_ind: NDArray,
    allowed: NDArray,
    *,
    chunk_bins: int = CHUNK_BINS,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict]:
    """Each bin's filter and two-sided scores under the model that allows the
    states ``allowed`` at each bin, from the library's filter and smoother.

    Parameters
    ----------
    log_likelihood : ndarray, shape (n_time, n_state_bins)
        ``ll[t, s]``; a row of zeros (a missing bin) scores 0 on both, to
        float32 rounding.
    initial : ndarray, shape (n_state_bins,)
        ``pi``, restricted to the states allowed at the first bin and
        renormalized.
    continuous : ndarray, shape (n_state_bins, n_state_bins)
        ``C[s', s]``, from bin s' to bin s.
    discrete : ndarray, shape (n_states, n_states)
        ``D[i, j]``.
    state_ind : ndarray of int, shape (n_state_bins,)
        Each state bin's discrete state.
    allowed : ndarray of bool, shape (n_time, n_states)
        ``a_t``: the states that may be occupied at bin t.
    chunk_bins : int
        Bins per batch of the two-sided step.

    Returns
    -------
    filter_score : ndarray of float32, shape (n_time,)
        ``log p(y_t | y_1..y_{t-1})``.
    two_sided : ndarray of float32, shape (n_time,)
        ``log p(y_t | y_{-t})``.
    record : dict
        ``library_s``, ``two_sided_s``, ``marginal`` (the library's),
        ``filter_score_sum`` (the float64 sum of the filter scores),
        ``marginal_difference`` (``marginal`` minus ``filter_score_sum``),
        ``marginal_agrees`` (within :data:`MARGINAL_TOLERANCE` of the total) and
        ``causal_mass_deviation`` (the largest distance of a filtered
        posterior's sum from 1).

    Raises
    ------
    ValueError
        When ``allowed`` and ``log_likelihood`` differ in their number of bins,
        ``discrete`` is not 2-D, a bin's allowed states leave a row of ``D``
        with no allowed destination, or none of the first bin's allowed states
        has initial mass.
    """
    from non_local_detector.core import chunked_filter_smoother_covariate_dependent

    if len(allowed) != len(log_likelihood):
        raise ValueError(
            f"allowed has {len(allowed)} bins, log_likelihood {len(log_likelihood)}"
        )
    if np.ndim(discrete) != 2:
        raise ValueError(f"discrete must be 2-D, not {np.ndim(discrete)}-D")
    record: dict[str, Any] = {}
    state_ind = np.asarray(state_ind)
    patterns, pattern = np.unique(
        np.asarray(allowed, dtype=bool), axis=0, return_inverse=True
    )
    pattern = pattern.ravel()
    n_time = pattern.size
    restricted = restricted_discrete_transitions(discrete, patterns)
    destination = patterns[:, state_ind]
    first = np.asarray(initial, dtype=float) * destination[pattern[0]]
    if first.sum() <= 0:
        raise ValueError("no allowed state of the first bin has initial mass")
    # the library pairs transition t with the move from bin t to bin t + 1, so
    # index t holds D~ of the allowed set at t + 1; the last is unused
    next_pattern = np.r_[pattern[1:], pattern[-1]]
    log_likelihood = jnp.asarray(log_likelihood, dtype=jnp.float32)
    # A forbidden bin holds no predicted mass, so its likelihood is not part of
    # the model; it is set to -inf so that the library subtracts the largest
    # likelihood of a bin that can hold mass before exponentiating. The pinned
    # library (non_local_detector 2d730c3a) adds 1e-15 in its normalization,
    # which loses mass when every bin with mass is far below the subtracted
    # maximum; upstream main fixed that (23369aa3, 2026-11-11), and the mask is
    # harmless after an upgrade.
    if not patterns.all():
        masked = jnp.where(
            jnp.asarray(destination)[jnp.asarray(pattern)], log_likelihood, -jnp.inf
        )
    else:
        masked = log_likelihood

    started = clock.time()
    (smoothed, _, marginal, _, _, _, causal, predicted) = (
        chunked_filter_smoother_covariate_dependent(
            time=np.arange(n_time),
            state_ind=state_ind,
            initial_distribution=first / first.sum(),
            discrete_transition_matrix=restricted[next_pattern],
            continuous_transition_matrix=np.asarray(continuous),
            log_likelihood_func=None,
            log_likelihood_args=(),
            log_likelihoods=masked,
        )
    )
    del masked
    record["library_s"] = round(clock.time() - started, 2)
    record["causal_mass_deviation"] = float(np.abs(causal.sum(axis=1) - 1.0).max())
    del causal

    started = clock.time()
    joint = jnp.asarray(
        np.asarray(continuous)[None] * restricted[:, state_ind][:, :, state_ind],
        dtype=jnp.float32,
    )
    filter_score = np.empty(n_time, dtype=np.float32)
    two_sided = np.empty(n_time, dtype=np.float32)
    for start in range(0, n_time, chunk_bins):
        stop = min(start + chunk_bins, n_time)
        # the next bin of the last bin is a stand-in; its two-sided score is set
        # below
        following = np.minimum(np.arange(start, stop) + 1, n_time - 1)
        scores = _bin_scores(
            jnp.asarray(predicted[start:stop]),
            log_likelihood[start:stop],
            jnp.asarray(predicted[following]),
            jnp.asarray(smoothed[following]),
            jnp.asarray(pattern[following]),
            joint,
        )
        filter_score[start:stop], two_sided[start:stop] = (
            np.asarray(s) for s in scores
        )
    two_sided[-1] = filter_score[-1]
    record["two_sided_s"] = round(clock.time() - started, 2)
    del smoothed, predicted

    total = float(filter_score.astype(np.float64).sum())
    record["marginal"] = float(marginal)
    record["filter_score_sum"] = total
    record["marginal_difference"] = float(marginal) - total
    record["marginal_agrees"] = bool(
        abs(record["marginal_difference"]) <= MARGINAL_TOLERANCE * abs(total)
    )
    return filter_score, two_sided, record


def heldout_log_likelihood(
    detector: Any,
    time: NDArray,
    head_position: NDArray,
    spike_times: Sequence[NDArray],
    marks: Sequence[NDArray],
    is_training: NDArray,
    is_missing: NDArray,
    *,
    n_folds: int = N_FOLDS,
    refit: Callable[..., dict] = refit_encoding,
    log: Callable[[str], None] = print,
) -> tuple[NDArray[np.float32], NDArray[np.int64], dict]:
    """The decoder's four-state log-likelihood, every tetrode in the model,
    each bin under the encodings of its time block (fit on the other blocks).

    The held-out stage of :func:`.tetrode_holdout.holdout_scores`
    (:func:`.tetrode_holdout.heldout_terms`), with no tetrode left out.

    Parameters
    ----------
    detector : NonLocalClusterlessDetector
        The epoch's fitted decoder (its states and environment).
    time : ndarray, shape (n_time,)
        Its time axis, s.
    head_position : ndarray, shape (n_time, 2)
        The position it was decoded with.
    spike_times, marks : sequence of ndarray, one per tetrode
    is_training, is_missing : ndarray of bool, shape (n_time,)
        The decode's masks; a missing bin's likelihood is 0.
    n_folds : int
    refit : callable
        ``refit(detector, time, position, spike_times, marks, is_training)``,
        an encoding model.
    log : callable

    Returns
    -------
    log_likelihood : ndarray of float32, shape (n_time, n_state_bins)
    blocks : ndarray of int, shape (n_time,)
        Each bin's time block.
    record : dict
        ``refit_s``, ``terms_s``, ``n_spikes``
        (:func:`.tetrode_holdout.heldout_terms`).
    """
    terms, no_spike, kernel, blocks, record = heldout_terms(
        detector,
        time,
        head_position,
        spike_times,
        marks,
        is_training,
        n_folds=n_folds,
        refit=refit,
        log=log,
    )
    state_ind = np.asarray(detector.state_ind_)[detector.is_track_interior_state_bins_]
    likelihood = state_log_likelihood(
        terms.total(),
        no_spike.sum(axis=1),
        kernel,
        state_ind,
        list(detector.state_names),
        np.asarray(is_missing, dtype=bool),
    )
    return likelihood, blocks, record


#: Each per-model key of :func:`predictive_log_scores`' record and its name in
#: :func:`model_scores`' record.
_MODEL_RECORD_KEYS = {
    "library_s": "library_{}_s",
    "two_sided_s": "two_sided_{}_s",
    "marginal": "marginal_log_likelihood_{}",
    "filter_score_sum": "filter_score_sum_{}",
    "marginal_difference": "marginal_difference_{}",
    "marginal_agrees": "marginal_agrees_{}",
    "causal_mass_deviation": "causal_mass_deviation_{}",
}


def model_scores(
    detector: Any,
    log_likelihood: NDArray,
    in_swr: NDArray,
    *,
    models: Sequence[str] = MODELS,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, NDArray[np.float32]], dict]:
    """Every bin's scores under the three models (or ``models``), on a given
    likelihood.

    The decoder's ``C``, ``D``, ``pi`` and state bins are read from
    ``detector``, restricted to its track-interior state bins.

    Parameters
    ----------
    detector : NonLocalClusterlessDetector
        The epoch's fitted decoder (its states and transitions).
    log_likelihood : ndarray, shape (n_time, n_state_bins)
        ``ll[t, s]`` (:func:`heldout_log_likelihood`).
    in_swr : ndarray of bool, shape (n_time,)
        The bins inside an SWR, where ``swr`` allows only the non-local states.
    models : sequence of str
        The models to score, of :data:`MODELS`.
    log : callable

    Returns
    -------
    scores : dict of str to ndarray of float32, shape (n_time,)
        Each model's columns of :data:`SCORE_COLUMNS`.
    record : dict
        ``continuous_block_deviation``; per model the library's
        ``marginal_log_likelihood_<model>`` as it returned it, the
        ``filter_score_sum_<model>``, their ``marginal_difference_<model>``,
        ``marginal_agrees_<model>`` (:data:`MARGINAL_TOLERANCE`),
        ``causal_mass_deviation_<model>``, ``library_<model>_s`` and
        ``two_sided_<model>_s``.
    """
    likelihood = jnp.asarray(log_likelihood)  # on the device once for the models
    interior = np.asarray(detector.is_track_interior_state_bins_)
    state_ind = np.asarray(detector.state_ind_)[interior]
    continuous = np.asarray(detector.continuous_state_transitions_)[
        np.ix_(interior, interior)
    ]
    discrete = np.asarray(detector.discrete_state_transitions_)
    initial = np.asarray(detector.initial_conditions_)[interior]
    record: dict[str, Any] = {
        "continuous_block_deviation": continuous_block_deviation(continuous, state_ind)
    }
    scores: dict[str, NDArray[np.float32]] = {}
    allowed = allowed_states(detector.state_names, in_swr)
    for model in models:
        filter_score, two_sided, part = predictive_log_scores(
            likelihood, initial, continuous, discrete, state_ind, allowed[model]
        )
        scores[f"{model}{FAMILY_SUFFIX['filter']}"] = filter_score
        scores[f"{model}{FAMILY_SUFFIX['two_sided']}"] = two_sided
        for key, name in _MODEL_RECORD_KEYS.items():
            record[name.format(model)] = part[key]
        log(
            f"{model}: library {part['library_s']} s, two-sided {part['two_sided_s']} s"
        )
    return scores, record


def restriction_scores(
    detector: Any,
    time: NDArray,
    head_position: NDArray,
    spike_times: Sequence[NDArray],
    marks: Sequence[NDArray],
    is_training: NDArray,
    is_missing: NDArray,
    in_swr: NDArray,
    *,
    n_folds: int = N_FOLDS,
    refit: Callable[..., dict] = refit_encoding,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, dict]:
    """Every bin's score under the three models, with a likelihood held out
    in time (:func:`heldout_log_likelihood`, then :func:`model_scores`).

    Parameters
    ----------
    detector : NonLocalClusterlessDetector
        The epoch's fitted decoder (its states, transitions and environment).
    time : ndarray, shape (n_time,)
        Its time axis, s.
    head_position : ndarray, shape (n_time, 2)
        The position it was decoded with.
    spike_times, marks : sequence of ndarray, one per tetrode
    is_training, is_missing : ndarray of bool, shape (n_time,)
        The decode's masks.
    in_swr : ndarray of bool, shape (n_time,)
        The bins inside an SWR, where ``swr`` allows only the non-local states.
    n_folds : int
    refit : callable
        ``refit(detector, time, position, spike_times, marks, is_training)``,
        an encoding model.
    log : callable

    Returns
    -------
    scores : pd.DataFrame
        One row per bin: ``time``; ``none``, ``swr``, ``full`` (float32
        filter scores, ``log p(y_t | y_1..y_{t-1})``) and ``none_two_sided``,
        ``swr_two_sided``, ``full_two_sided`` (``log p(y_t | y_{-t})``);
        ``in_swr`` and ``block`` (the bin's time block).
    record : dict
        :func:`heldout_log_likelihood`'s (``refit_s``, ``terms_s``,
        ``n_spikes``) and :func:`model_scores`' records, and ``n_in_swr``,
        ``n_missing``, ``total_s``.

    Raises
    ------
    ValueError
        When ``in_swr`` and ``time`` differ in length, or as
        :func:`predictive_log_scores`.
    """
    started = clock.time()
    in_swr = np.asarray(in_swr, dtype=bool)
    if in_swr.shape != np.shape(time):
        raise ValueError(f"in_swr has shape {in_swr.shape}, time {np.shape(time)}")
    likelihood, blocks, record = heldout_log_likelihood(
        detector,
        time,
        head_position,
        spike_times,
        marks,
        is_training,
        is_missing,
        n_folds=n_folds,
        refit=refit,
        log=log,
    )
    scores, part = model_scores(detector, likelihood, in_swr, log=log)
    record.update(part)
    record["n_in_swr"] = int(in_swr.sum())
    record["n_missing"] = int(np.asarray(is_missing, dtype=bool).sum())
    record["total_s"] = round(clock.time() - started, 1)
    table = pd.DataFrame(
        {
            "time": np.asarray(time),
            **{column: scores[column] for column in SCORE_COLUMNS},
            "in_swr": in_swr,
            "block": blocks,
        }
    )
    return table, record


def check_failures(record: dict) -> list[str]:
    """The model checks a :func:`restriction_scores` record fails, as messages.

    A model's library marginal must agree with its summed filter scores
    (``marginal_agrees_<model>``), and its filtered posterior mass must be
    within :data:`CAUSAL_MASS_TOLERANCE` of 1.
    """
    failures = []
    for model in MODELS:
        if not record[f"marginal_agrees_{model}"]:
            failures.append(
                f"{model}: library marginal {record[f'marginal_log_likelihood_{model}']}"
                f" differs from the summed filter scores by "
                f"{record[f'marginal_difference_{model}']}"
            )
        deviation = record[f"causal_mass_deviation_{model}"]
        if not deviation <= CAUSAL_MASS_TOLERANCE:
            failures.append(
                f"{model}: filtered posterior mass is {deviation} from 1 "
                f"(tolerance {CAUSAL_MASS_TOLERANCE})"
            )
    return failures


def restriction_sums(scores: pd.DataFrame, labels: Sequence) -> pd.DataFrame:
    """The epoch's bins and summed scores in each category of bins.

    Parameters
    ----------
    scores : pd.DataFrame
        :func:`restriction_scores`' table, one row per bin.
    labels : sequence, length n_time
        Each bin's category (for example ``immobile_in_swr``,
        ``immobile_outside_swr``, ``moving``, ``excluded``); a missing label
        (None or NaN) leaves the bin out of every category.

    Returns
    -------
    pd.DataFrame
        One row per category (index ``category``): ``n_bins`` and every model's
        summed score (nats) of both families, ``<model>`` (filter) and
        ``<model>_two_sided``.

    Raises
    ------
    ValueError
        When a labeled bin has a score that is not finite.
    """
    labels = pd.Series(np.asarray(labels, dtype=object), index=scores.index)
    columns = list(SCORE_COLUMNS)
    kept = labels.notna()
    not_finite = ~np.isfinite(scores.loc[kept, columns].to_numpy(np.float64))
    if not_finite.any():
        counts = {
            column: n
            for column, n in zip(columns, not_finite.sum(axis=0).tolist(), strict=True)
            if n
        }
        raise ValueError(f"labeled bins with a score that is not finite: {counts}")
    grouped = (
        scores.loc[kept, columns]
        .astype(np.float64)
        .groupby(labels[kept].rename("category"))
    )
    sums = grouped.sum()
    sums.insert(0, "n_bins", grouped.size())
    return sums


def restriction_contrasts(
    sums: pd.DataFrame, categories: Sequence[str], bin_s: float
) -> dict[str, float]:
    """The model contrasts over the bins of ``categories``, both score families.

    Parameters
    ----------
    sums : pd.DataFrame
        :func:`restriction_sums`' output.
    categories : sequence of str
        The categories to combine (a category with no bins adds nothing).
    bin_s : float
        The bin width, s.

    Returns
    -------
    dict
        ``n_bins``, ``seconds``; for each family (no suffix for the filter score,
        ``_two_sided`` for the two-sided one): ``swr_minus_none``,
        ``full_minus_swr`` and ``full_minus_none`` (nats), each per second
        (``..._per_s``, NaN without bins).
    """
    combined = sums.reindex(list(categories)).fillna(0.0).sum()
    seconds = float(combined["n_bins"]) * bin_s
    out: dict[str, float] = {"n_bins": int(combined["n_bins"]), "seconds": seconds}
    for suffix in FAMILY_SUFFIX.values():
        for name, (a, b) in {
            "swr_minus_none": ("swr", "none"),
            "full_minus_swr": ("full", "swr"),
            "full_minus_none": ("full", "none"),
        }.items():
            difference = float(combined[f"{a}{suffix}"] - combined[f"{b}{suffix}"])
            out[f"{name}{suffix}"] = difference
            out[f"{name}{suffix}_per_s"] = (
                difference / seconds if seconds > 0 else float("nan")
            )
    return out


def share_outside_swr(
    sums: pd.DataFrame,
    inside: str = IMMOBILE_IN_SWR,
    outside: str = IMMOBILE_OUTSIDE_SWR,
) -> dict[str, float]:
    """Of the nats the non-local states buy over the bins of two categories,
    the part in the bins outside SWRs, both score families.

    ``(full - none)`` summed over ``outside`` divided by ``(full - none)``
    summed over ``inside`` and ``outside`` together. Only ``full`` and ``none``
    enter, so the share is a split of the non-local gain by where it lies, not
    a model contrast.

    Parameters
    ----------
    sums : pd.DataFrame
        :func:`restriction_sums`' output.
    inside, outside : str
        The categories of the bins inside and outside SWRs (by default
        :data:`IMMOBILE_IN_SWR` and :data:`IMMOBILE_OUTSIDE_SWR`); a category
        with no bins adds nothing.

    Returns
    -------
    dict
        ``share_outside_swr`` (filter) and ``share_outside_swr_two_sided``,
        NaN when the gain over both categories is not positive.
    """
    rows = sums.reindex([inside, outside]).fillna(0.0)
    out: dict[str, float] = {}
    for suffix in FAMILY_SUFFIX.values():
        gain = rows[f"full{suffix}"] - rows[f"none{suffix}"]
        total = float(gain.sum())
        out[f"share_outside_swr{suffix}"] = (
            float(gain[outside]) / total if total > 0 else float("nan")
        )
    return out
