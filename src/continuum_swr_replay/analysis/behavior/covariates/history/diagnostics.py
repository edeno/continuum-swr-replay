"""Diagnostic per-animal fitted-alpha leaky-Q learner. Produces no cataloged covariate; used only for offline model checking."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _q_chosen(
    arm: np.ndarray,
    reward: np.ndarray,
    grp: np.ndarray,
    alpha: float,
    init: float = 0.5,
) -> np.ndarray:
    """Per-leaf delta-rule Q value of the chosen leaf, pre-update.

    Q[a] += alpha*(reward - Q[a]); Q resets to ``init`` at each new group (session),
    so within a session it updates continuously across contingency blocks -- the
    within-block-updating subjective value the block-constant nominal value lacks.

    This is the chosen-arm column of :func:`q_all_arm` by construction, so it
    delegates to that rollout rather than re-implementing the delta-rule.
    """
    return q_all_arm(arm, reward, grp, alpha, init)[np.arange(len(arm)), arm]


def q_all_arm(  # DIAGNOSTIC per-animal fitted-alpha learner; not a fitted covariate
    arm: np.ndarray,
    reward: np.ndarray,
    grp: np.ndarray,
    alpha: float,
    init: float = 0.5,
) -> np.ndarray:
    """Per-leaf delta-rule Q **level of every arm**, pre-update, for all trials.

    The canonical leaky delta-rule rollout (``Q[a] += alpha*(reward - Q[a])``,
    ``Q`` reset to ``init`` at each new group), snapshotting the full 6-arm value
    vector *before* each trial's update. The chosen-arm value is
    ``[i, arm[i]]`` (which :func:`_q_chosen` returns); the full output is the
    local comparison learner's per-arm value and is not the fitted behavioral
    model's canonical reward-belief column.

    Parameters
    ----------
    arm : np.ndarray, shape (n_trials,)
        Chosen leaf arm 0..5 at each trial, in trial order within group.
    reward : np.ndarray, shape (n_trials,)
        Binary reward (0/1) at each trial.
    grp : np.ndarray, shape (n_trials,)
        Group label (session); ``Q`` resets to ``init`` at each new group.
    alpha : float
        Leaky delta-rule learning rate.
    init : float
        Initial Q for every arm at the start of each group.

    Returns
    -------
    np.ndarray, shape (n_trials, 6)
        Row ``i`` is the pre-update Q value of each of the 6 leaf arms at trial ``i``.
    """
    out = np.empty((len(arm), 6))
    Q, prev = None, None
    for i in range(len(arm)):
        if grp[i] != prev:
            Q, prev = np.full(6, init), grp[i]
        out[i] = Q  # numpy row-assignment copies values, so the later in-place
        a = arm[i]  # update of Q does not alias the stored snapshot
        Q[a] += alpha * (reward[i] - Q[a])
    return out


def _fit_alpha(arm: np.ndarray, reward: np.ndarray, grp: np.ndarray) -> float:
    """alpha (per animal) that makes q_chosen best predict reward (Bernoulli NLL)."""

    def nll(a: float) -> float:
        p = np.clip(_q_chosen(arm, reward, grp, a), 1e-4, 1 - 1e-4)
        return -float(np.sum(reward * np.log(p) + (1 - reward) * np.log(1 - p)))

    return float(minimize_scalar(nll, bounds=(0.01, 0.99), method="bounded").x)
