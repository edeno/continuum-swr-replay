"""Synthetic-data generators for validating the conditional Poisson model.

Tier A — ``simulate_counts``: sufficient-statistic simulation. Draws
``arm_count ~ Multinomial(L_e, softmax(X β_true))`` directly, bypassing the
posterior/MAP pipeline. Self-consistency check on the estimator.

Tier C — ``simulate_posterior_bumps``: posterior/MAP-space simulation. For
each timebin the chosen arm is drawn from softmax(X β_true), then a MAP
linear position is sampled inside that arm with Gaussian spread. The MAPs
are mapped to arms by ``map_estimate_to_arm`` against a ``SyntheticEnv``
with junction bins (edge_id=-1), exposing pipeline artifacts that Tier A
hides: junction handling, MAP-to-bin discretization, and bumps that
straddle arm boundaries.

Tier B — ``simulate_autocorrelated_counts``: trajectory-level simulation.
For each event, an arm sequence is drawn from a 9-state Markov chain with
self-transition probability ``rho`` and stationary distribution
softmax(X β_true). Marginal arm counts are unbiased (same expectation as
Tier A), but consecutive timebins are correlated, so per-event count
variance is *smaller* than Multinomial. This breaks the
within-event-independence assumption of the conditional Poisson and can
make ``bse`` from the fit too narrow — exactly the calibration story we
want to quantify.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

from .spatial_mapping import map_estimate_to_arm
from .summaries import arm_count_hard as _arm_count_hard_fn

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())

DEFAULT_PREDICTORS: tuple[str, ...] = (
    "is_current",
    "is_prev",
    "is_next",
    "is_last_rewarded",
)


# Predictor name → ("base" arm-varying, "event" event-level) decomposition.
# Used by ``simulate_with_event_covariates`` to construct interaction columns
# the same way ``add_interaction_columns`` and ``add_temporal_interactions``
# do in the production pipeline.
_INTERACTION_FACTORS: dict[str, tuple[str, str]] = {
    "swr_x_current": ("is_current", "is_swr"),
    "swr_x_prev": ("is_prev", "is_swr"),
    "swr_x_next": ("is_next", "is_swr"),
    "swr_x_last_rewarded": ("is_last_rewarded", "is_swr"),
    "mua_x_last_rewarded": ("is_last_rewarded", "is_mua"),
    "switch_x_current": ("is_current", "is_after_switch"),
    "switch_x_prev": ("is_prev", "is_after_switch"),
    "omission_x_last_rewarded": ("is_last_rewarded", "is_after_omission"),
    "time_x_current": ("is_current", "time_since_stop"),
    "time_x_prev": ("is_prev", "time_since_stop"),
    "time_x_next": ("is_next", "time_since_stop"),
}

# Arm-level × arm-level interactions (both factors vary across arms).
_ARM_ARM_INTERACTIONS: dict[str, tuple[str, str]] = {
    "current_x_last_rewarded": ("is_current", "is_last_rewarded"),
}

_BASE_BINARY = ("is_current", "is_prev", "is_next", "is_last_rewarded")
_CONTINUOUS_ARM = ("visit_recency", "cumulative_reward", "dist_to_arm")
_EVENT_LEVEL = ("is_swr", "is_after_switch", "is_after_omission", "time_since_stop")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis with the standard max-shift for stability."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def sample_arm_design(
    n_events: int,
    predictors: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an arm-feature design tensor of shape (n_events, n_arms, n_predictors).

    Each event independently picks a "current" arm uniformly at random over
    the 6 leaf arms (the animal's physical location is always at a well).
    ``is_prev`` and ``is_next`` pick distinct random leaf arms (the trial
    departure/destination wells); ``is_last_rewarded`` picks another random
    leaf arm. Trunk arms (6, 7, 8) are always 0 for these binary features
    — matching the real-data convention that ``well_to_arm_set`` only
    returns leaf arms.

    Continuous predictors (``visit_recency``, ``cumulative_reward``,
    ``dist_to_arm``) are sampled from plausible ranges if requested.
    """
    n_arms = len(ALL_ARMS)
    leaf_arms = np.array([0, 1, 2, 3, 4, 5])

    feature_cols: dict[str, np.ndarray] = {}

    # Independent random leaf-arm assignments per event for each binary slot.
    current = rng.choice(leaf_arms, size=n_events)
    prev = rng.choice(leaf_arms, size=n_events)
    nxt = rng.choice(leaf_arms, size=n_events)
    rewarded = rng.choice(leaf_arms, size=n_events)

    arm_grid = np.arange(n_arms)[None, :]  # (1, n_arms)
    feature_cols["is_current"] = (arm_grid == current[:, None]).astype(float)
    feature_cols["is_prev"] = (arm_grid == prev[:, None]).astype(float)
    feature_cols["is_next"] = (arm_grid == nxt[:, None]).astype(float)
    feature_cols["is_last_rewarded"] = (arm_grid == rewarded[:, None]).astype(float)

    # Continuous: independent across events and arms with arbitrary scale.
    if "visit_recency" in predictors:
        feature_cols["visit_recency"] = rng.uniform(0, 10, size=(n_events, n_arms))
    if "cumulative_reward" in predictors:
        feature_cols["cumulative_reward"] = rng.uniform(0, 5, size=(n_events, n_arms))
    if "dist_to_arm" in predictors:
        feature_cols["dist_to_arm"] = rng.uniform(0, 100, size=(n_events, n_arms))

    missing = [p for p in predictors if p not in feature_cols]
    if missing:
        raise ValueError(
            f"sample_arm_design does not know how to sample: {missing}. "
            "Pass a custom design tensor instead."
        )

    return np.stack([feature_cols[p] for p in predictors], axis=-1)


def simulate_counts(
    beta_true: dict[str, float],
    n_events: int = 200,
    L_e: int | Sequence[int] = 20,
    *,
    predictors: Sequence[str] | None = None,
    design: np.ndarray | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate Tier-A synthetic long-format data with known coefficients.

    Parameters
    ----------
    beta_true : dict[str, float]
        Map from predictor name to true coefficient. Predictor order is
        taken from ``predictors`` (or the dict's insertion order if
        ``predictors`` is None).
    n_events : int
        Number of synthetic events to generate.
    L_e : int or sequence of int
        Total timebin count per event. Scalar broadcasts to all events;
        sequence must have length ``n_events``.
    predictors : sequence of str, optional
        Predictor names and order. Defaults to ``beta_true.keys()``.
    design : ndarray, optional
        Pre-built design tensor of shape (n_events, n_arms, n_predictors)
        if you want to control the feature distribution exactly. If None,
        ``sample_arm_design`` is used.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format frame with one row per (event, arm). Columns:
        ``event_uid``, ``arm_id``, ``arm_count``, plus one column per
        predictor. Compatible with ``fit_conditional_poisson``.
    """
    rng = np.random.default_rng(seed)
    predictor_list = list(predictors) if predictors is not None else list(beta_true)
    beta_vec = np.array([beta_true[p] for p in predictor_list], dtype=float)

    if design is None:
        design = sample_arm_design(n_events, predictor_list, rng)
    else:
        design = np.asarray(design, dtype=float)
        expected_shape = (n_events, len(ALL_ARMS), len(predictor_list))
        if design.shape != expected_shape:
            raise ValueError(
                f"design shape {design.shape} != expected {expected_shape}"
            )

    if np.isscalar(L_e):
        L_e_arr = np.full(n_events, int(L_e))
    else:
        L_e_arr = np.asarray(L_e, dtype=int)
        if L_e_arr.shape != (n_events,):
            raise ValueError(f"L_e length {L_e_arr.shape} != n_events {n_events}")

    logits = design @ beta_vec  # (n_events, n_arms)
    probs = _softmax(logits)

    n_arms = len(ALL_ARMS)
    counts = np.empty((n_events, n_arms), dtype=int)
    for e in range(n_events):
        counts[e] = rng.multinomial(L_e_arr[e], probs[e])

    event_ids = np.repeat(np.arange(n_events), n_arms)
    arm_ids = np.tile(ALL_ARMS, n_events)

    data: dict[str, np.ndarray] = {
        "event_uid": event_ids,
        "arm_id": arm_ids,
        "arm_count": counts.reshape(-1),
    }
    for k, name in enumerate(predictor_list):
        data[name] = design[:, :, k].reshape(-1)

    return pd.DataFrame(data)


def simulate_with_event_covariates(
    beta_true: dict[str, float],
    *,
    n_events: int = 200,
    L_e: int | Sequence[int] = 20,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate long-format data with event-level covariates and interactions.

    Extends ``simulate_counts`` to support the full predictor vocabulary
    used by ``model_specs.MODEL_HIERARCHY`` — including the continuous
    arm-varying predictors (``visit_recency``, ``cumulative_reward``,
    ``dist_to_arm``) and the event×arm interaction columns
    (``swr_x_*``, ``switch_x_*``, ``omission_x_last_rewarded``, ``time_x_*``).

    Per event we sample event-level covariates (``is_swr``,
    ``is_after_switch``, ``is_after_omission``, ``time_since_stop``) and
    arm-varying base predictors as in ``sample_arm_design``. Interaction
    columns are then constructed by element-wise product of their base
    arm-varying predictor and their event-level scalar — exactly the
    construction in ``event_collapsing.add_interaction_columns`` and
    ``predictors.add_temporal_interactions``. Counts are drawn from
    ``Multinomial(L_e, softmax(X β_true))`` over the requested predictor
    set.

    The returned long-format frame contains *every* model_specs predictor
    (not just the ones with non-zero β), so a single simulator output can
    be used to compare nested models in the M0–M7 hierarchy.
    """
    rng = np.random.default_rng(seed)
    predictor_list = list(beta_true)
    n_arms = len(ALL_ARMS)
    leaf = np.array([0, 1, 2, 3, 4, 5])

    # Arm-varying base binary predictors, sampled per event.
    current = rng.choice(leaf, size=n_events)
    prev = rng.choice(leaf, size=n_events)
    nxt = rng.choice(leaf, size=n_events)
    rewarded = rng.choice(leaf, size=n_events)
    arm_grid = np.arange(n_arms)[None, :]
    arm_features: dict[str, np.ndarray] = {
        "is_current": (arm_grid == current[:, None]).astype(float),
        "is_prev": (arm_grid == prev[:, None]).astype(float),
        "is_next": (arm_grid == nxt[:, None]).astype(float),
        "is_last_rewarded": (arm_grid == rewarded[:, None]).astype(float),
        "visit_recency": rng.uniform(0, 10, size=(n_events, n_arms)),
        "cumulative_reward": rng.uniform(0, 5, size=(n_events, n_arms)),
        "dist_to_arm": rng.uniform(0, 100, size=(n_events, n_arms)),
    }

    # Event-level scalars, broadcast to all arms.
    event_scalars: dict[str, np.ndarray] = {
        "is_swr": (rng.random(n_events) < 0.5).astype(float),
        "is_mua": (rng.random(n_events) < 0.5).astype(float),
        "is_after_switch": (rng.random(n_events) < 0.3).astype(float),
        "is_after_omission": (rng.random(n_events) < 0.3).astype(float),
        "time_since_stop": rng.uniform(0, 5, size=n_events),
    }
    event_broadcast: dict[str, np.ndarray] = {
        k: np.tile(v[:, None], (1, n_arms)) for k, v in event_scalars.items()
    }

    # Interaction columns from the canonical decomposition.
    interaction_cols: dict[str, np.ndarray] = {}
    for name, (base, ev) in _INTERACTION_FACTORS.items():
        interaction_cols[name] = arm_features[base] * event_broadcast[ev]

    # Arm × arm interactions (both factors vary across arms).
    for name, (left, right) in _ARM_ARM_INTERACTIONS.items():
        interaction_cols[name] = arm_features[left] * arm_features[right]

    all_columns: dict[str, np.ndarray] = {
        **arm_features,
        **event_broadcast,
        **interaction_cols,
    }

    missing = [p for p in predictor_list if p not in all_columns]
    if missing:
        raise ValueError(
            f"simulate_with_event_covariates does not know how to build: {missing}"
        )

    beta_vec = np.array([beta_true[p] for p in predictor_list], dtype=float)
    design = np.stack([all_columns[p] for p in predictor_list], axis=-1)
    logits = design @ beta_vec
    probs = _softmax(logits)

    if np.isscalar(L_e):
        L_e_arr = np.full(n_events, int(L_e))
    else:
        L_e_arr = np.asarray(L_e, dtype=int)
        if L_e_arr.shape != (n_events,):
            raise ValueError(f"L_e length {L_e_arr.shape} != n_events {n_events}")

    counts = np.empty((n_events, n_arms), dtype=int)
    for e in range(n_events):
        counts[e] = rng.multinomial(L_e_arr[e], probs[e])

    # Long-format with EVERY canonical column populated, so callers can
    # fit any nested submodel of the truth.
    event_ids = np.repeat(np.arange(n_events), n_arms)
    arm_ids = np.tile(ALL_ARMS, n_events)
    data: dict[str, np.ndarray] = {
        "event_uid": event_ids,
        "arm_id": arm_ids,
        "arm_count": counts.reshape(-1),
    }
    for name, arr in all_columns.items():
        data[name] = arr.reshape(-1)
    return pd.DataFrame(data)


# ── Tier B: trajectory-level simulation with within-event autocorrelation ──


def _markov_arm_sequence(
    pi: np.ndarray,
    rho: float,
    n_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an arm sequence whose stationary distribution is ``pi``.

    Uses a "lazy" Markov chain: at each step, with probability ``rho``
    repeat the current arm, otherwise draw a fresh sample from ``pi``.
    The resulting chain has stationary distribution exactly ``pi`` (since
    every fresh draw comes from pi) but introduces controllable within-
    chain autocorrelation. ``rho=0`` recovers iid sampling; ``rho→1``
    produces nearly constant trajectories.

    The autocorrelation at lag k is ``rho**k``, so the effective sample
    size per chain of length L is approximately L * (1-rho) / (1+rho).
    """
    out = np.empty(n_bins, dtype=int)
    out[0] = rng.choice(len(pi), p=pi)
    if n_bins == 1:
        return out
    repeat = rng.random(n_bins - 1) < rho
    fresh_draws = rng.choice(len(pi), size=n_bins - 1, p=pi)
    for t in range(1, n_bins):
        out[t] = out[t - 1] if repeat[t - 1] else fresh_draws[t - 1]
    return out


def simulate_autocorrelated_counts(
    beta_true: dict[str, float],
    *,
    n_events: int = 200,
    L_e: int | Sequence[int] = 20,
    rho: float = 0.0,
    predictors: Sequence[str] | None = None,
    design: np.ndarray | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Tier-B generator: per-event arm sequences with controllable
    within-event temporal autocorrelation, returned in long format.

    The marginal per-event arm distribution still equals softmax(X β_true)
    so β̂ from ``fit_conditional_poisson`` should remain unbiased. What
    changes is the per-event count variance: with rho > 0, the counts are
    *less* dispersed than Multinomial(L_e, π_e), so the conditional
    Poisson's information matrix overstates how much information each
    event carries, and ``bse`` becomes too narrow.

    Use ``rho=0`` to reproduce Tier A behavior. Use ``rho=0.7`` (effective
    sample size ≈ 0.18 × L_e) to simulate the kind of strong temporal
    smoothness expected for a coherent trajectory replay.
    """
    rng = np.random.default_rng(seed)
    predictor_list = list(predictors) if predictors is not None else list(beta_true)
    beta_vec = np.array([beta_true[p] for p in predictor_list], dtype=float)

    if design is None:
        design = sample_arm_design(n_events, predictor_list, rng)
    else:
        design = np.asarray(design, dtype=float)
        expected_shape = (n_events, len(ALL_ARMS), len(predictor_list))
        if design.shape != expected_shape:
            raise ValueError(
                f"design shape {design.shape} != expected {expected_shape}"
            )

    if np.isscalar(L_e):
        L_e_arr = np.full(n_events, int(L_e))
    else:
        L_e_arr = np.asarray(L_e, dtype=int)
        if L_e_arr.shape != (n_events,):
            raise ValueError(f"L_e length {L_e_arr.shape} != n_events {n_events}")

    logits = design @ beta_vec
    probs = _softmax(logits)
    n_arms = len(ALL_ARMS)

    counts = np.zeros((n_events, n_arms), dtype=int)
    for e in range(n_events):
        seq = _markov_arm_sequence(probs[e], rho, int(L_e_arr[e]), rng)
        for a in range(n_arms):
            counts[e, a] = int((seq == a).sum())

    event_ids = np.repeat(np.arange(n_events), n_arms)
    arm_ids = np.tile(ALL_ARMS, n_events)
    data: dict[str, np.ndarray] = {
        "event_uid": event_ids,
        "arm_id": arm_ids,
        "arm_count": counts.reshape(-1),
    }
    for k, name in enumerate(predictor_list):
        data[name] = design[:, :, k].reshape(-1)
    return pd.DataFrame(data)


# ── Tier C: posterior/MAP-space simulation ──


@dataclass
class SyntheticEnv:
    """Minimal stub of a non_local_detector ``Environment`` for simulation.

    Exposes ``place_bin_centers_nodes_df_`` with the two columns that
    ``map_estimate_to_arm`` actually reads: ``linear_position`` and
    ``edge_id``. Lays the 9 arms out sequentially in linear position with
    optional junction bins (edge_id=-1) interleaved between adjacent arms,
    mirroring the linearization artifacts in real ``Environment`` objects.

    Attributes
    ----------
    place_bin_centers_nodes_df_ : pd.DataFrame
        Bin table with linear_position (cm) and edge_id (0-8 for real bins,
        -1 for junction-interpolation bins).
    arm_extents : dict[int, tuple[float, float]]
        Per-arm (start, end) in linear cm. Useful for sampling MAPs inside
        a chosen arm.
    arm_length : float
        Common arm length in cm.
    """

    place_bin_centers_nodes_df_: pd.DataFrame
    arm_extents: dict[int, tuple[float, float]]
    arm_length: float


def make_synthetic_env(
    n_arms: int = 9,
    arm_length: float = 50.0,
    bins_per_arm: int = 25,
    junction_bins: int = 4,
    junction_gap: float = 4.0,
) -> SyntheticEnv:
    """Build a synthetic env with sequential arms and inter-arm junction bins.

    The layout concatenates arms 0..n_arms-1 along a single linear axis,
    each ``arm_length`` cm long with ``bins_per_arm`` evenly-spaced real
    bins. Between consecutive arms a gap of ``junction_gap`` cm holds
    ``junction_bins`` placeholder bins with edge_id=-1, matching how
    ``non_local_detector`` interpolates across edge boundaries.

    Parameters
    ----------
    n_arms : int
        Number of arms (default 9, matching the bandit task).
    arm_length : float
        Length of each arm in cm.
    bins_per_arm : int
        Number of real place bins per arm.
    junction_bins : int
        Number of junction (edge_id=-1) bins between consecutive arms.
    junction_gap : float
        Linear-cm extent of each inter-arm junction region.
    """
    rows = []
    arm_extents: dict[int, tuple[float, float]] = {}
    cursor = 0.0
    bin_step = arm_length / bins_per_arm

    for arm in range(n_arms):
        start = cursor
        for j in range(bins_per_arm):
            rows.append(
                {
                    "linear_position": start + (j + 0.5) * bin_step,
                    "edge_id": arm,
                }
            )
        end = cursor + arm_length
        arm_extents[arm] = (start, end)
        cursor = end

        if arm < n_arms - 1 and junction_bins > 0:
            jstep = junction_gap / junction_bins
            for k in range(junction_bins):
                rows.append(
                    {
                        "linear_position": cursor + (k + 0.5) * jstep,
                        "edge_id": -1,
                    }
                )
            cursor += junction_gap

    return SyntheticEnv(
        place_bin_centers_nodes_df_=pd.DataFrame(rows),
        arm_extents=arm_extents,
        arm_length=arm_length,
    )


def simulate_posterior_bumps(
    beta_true: dict[str, float],
    *,
    n_events: int = 200,
    L_e: int | Sequence[int] = 20,
    sigma: float = 5.0,
    env: SyntheticEnv | None = None,
    predictors: Sequence[str] | None = None,
    design: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate Tier-C synthetic data via per-timebin MAP estimates.

    For each timebin we (1) pick a "chosen" arm from the per-event softmax,
    (2) draw a MAP linear position from a Gaussian centered at the arm's
    midpoint with std ``sigma`` (no clipping, so large sigma values *do*
    leak across arm boundaries — this is the artifact the simulator is
    designed to expose), and (3) snap the MAP to the nearest real bin via
    ``map_estimate_to_arm``. At sigma=0 every MAP lands cleanly inside its
    chosen arm. As sigma grows, samples can land in junction-bin regions
    (where they snap to whichever neighbouring arm has the closer real
    bin) or directly inside a neighbouring arm.

    Parameters
    ----------
    beta_true : dict[str, float]
        True coefficients keyed by predictor name.
    n_events : int
        Number of synthetic events.
    L_e : int or sequence
        Per-event timebin counts.
    sigma : float
        Std (cm) of the Gaussian bump within the chosen arm. sigma=0 puts
        every MAP at the arm midpoint; large sigma bleeds into neighbors.
    env : SyntheticEnv, optional
        Synthetic environment. Defaults to ``make_synthetic_env()``.
    predictors, design, seed : as in ``simulate_counts``.

    Returns
    -------
    long_df : pd.DataFrame
        Long-format frame matching the schema of ``simulate_counts`` —
        one row per (event, arm) with ``event_uid``, ``arm_id``,
        ``arm_count``, and one column per predictor. The arm counts come
        from running every per-timebin MAP through ``map_estimate_to_arm``.
    diagnostics : dict
        ``fraction_on_chosen``: fraction of timebins whose decoded arm
        equals the generative chosen arm (1.0 at sigma=0, decreases as
        sigma grows). Useful to confirm the simulator is exercising the
        spatial-mapping artifact.
    """
    rng = np.random.default_rng(seed)
    predictor_list = list(predictors) if predictors is not None else list(beta_true)
    beta_vec = np.array([beta_true[p] for p in predictor_list], dtype=float)
    if env is None:
        env = make_synthetic_env()

    if design is None:
        design = sample_arm_design(n_events, predictor_list, rng)
    else:
        design = np.asarray(design, dtype=float)
        expected_shape = (n_events, len(ALL_ARMS), len(predictor_list))
        if design.shape != expected_shape:
            raise ValueError(
                f"design shape {design.shape} != expected {expected_shape}"
            )

    if np.isscalar(L_e):
        L_e_arr = np.full(n_events, int(L_e))
    else:
        L_e_arr = np.asarray(L_e, dtype=int)
        if L_e_arr.shape != (n_events,):
            raise ValueError(f"L_e length {L_e_arr.shape} != n_events {n_events}")

    logits = design @ beta_vec
    probs = _softmax(logits)
    n_arms = len(ALL_ARMS)

    # Sample chosen arms and MAP estimates per timebin per event.
    all_chosen: list[np.ndarray] = []
    all_maps: list[np.ndarray] = []
    for e in range(n_events):
        chosen = rng.choice(n_arms, size=int(L_e_arr[e]), p=probs[e])
        midpoints = np.array(
            [(env.arm_extents[a][0] + env.arm_extents[a][1]) / 2 for a in chosen]
        )
        maps = midpoints + rng.normal(0.0, sigma, size=int(L_e_arr[e]))
        all_chosen.append(chosen)
        all_maps.append(maps)

    chosen_arr = np.concatenate(all_chosen)
    map_arr = np.concatenate(all_maps)
    decoded_arm = map_estimate_to_arm(map_arr, env)

    fraction_on_chosen = float(np.mean(decoded_arm == chosen_arr))

    # Aggregate decoded arms into per-event counts, then build long format.
    event_repeats = np.concatenate(
        [np.full(int(L_e_arr[e]), e) for e in range(n_events)]
    )
    bin_df = pd.DataFrame(
        {"event_uid": event_repeats, "decoded_arm": decoded_arm.astype(int)}
    )
    counts_wide = (
        bin_df.groupby(["event_uid", "decoded_arm"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ALL_ARMS, fill_value=0)
        .reindex(index=np.arange(n_events), fill_value=0)
    )
    counts = counts_wide.values  # (n_events, n_arms)

    event_ids = np.repeat(np.arange(n_events), n_arms)
    arm_ids = np.tile(ALL_ARMS, n_events)
    data: dict[str, np.ndarray] = {
        "event_uid": event_ids,
        "arm_id": arm_ids,
        "arm_count": counts.reshape(-1),
    }
    for k, name in enumerate(predictor_list):
        data[name] = design[:, :, k].reshape(-1)

    return pd.DataFrame(data), {"fraction_on_chosen": fraction_on_chosen}


def simulate_hard_vs_soft_counts(
    beta_true: dict[str, float],
    *,
    n_events: int = 200,
    L_e: int | Sequence[int] = 20,
    sigma: float = 5.0,
    env: SyntheticEnv | None = None,
    predictors: Sequence[str] | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate paired hard/soft long-format frames from the same posteriors.

    For each timebin we draw a chosen arm from softmax(X β_true) and a
    Gaussian bump centered at that arm's midpoint with std ``sigma``. We
    then report the per-arm posterior in two ways:

    - **Hard** (``hard_long_df``): take the MAP via ``map_estimate_to_arm``
      against the SyntheticEnv (the production pipeline). One arm receives
      a count of 1 per timebin; junction-bin snapping decides ties.
    - **Soft** (``soft_long_df``): integrate the Gaussian bump analytically
      over each arm's linear extent (using the cumulative normal across
      ``arm_extents``). Per-timebin posterior probabilities sum to ≤ 1
      (the residual mass falls in inter-arm junction gaps and is dropped
      to mirror the production pipeline's junction handling). Fractional
      counts are scaled up by 1000 and rounded to integers so the result
      is admissible to ``ConditionalPoisson`` (which requires integer y).

    Both frames share schema and predictors with ``simulate_counts``, so
    they're drop-in inputs for ``fit_conditional_poisson``. Comparing the
    two β̂'s isolates how much of the σ-attenuation in the hard path is
    intrinsic to MAP-collapse (and would be relieved by using the full
    posterior) versus what survives even with the full posterior.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    predictor_list = list(predictors) if predictors is not None else list(beta_true)
    beta_vec = np.array([beta_true[p] for p in predictor_list], dtype=float)
    if env is None:
        env = make_synthetic_env()

    design = sample_arm_design(n_events, predictor_list, rng)

    if np.isscalar(L_e):
        L_e_arr = np.full(n_events, int(L_e))
    else:
        L_e_arr = np.asarray(L_e, dtype=int)
        if L_e_arr.shape != (n_events,):
            raise ValueError(f"L_e length {L_e_arr.shape} != n_events {n_events}")

    logits = design @ beta_vec
    probs = _softmax(logits)
    n_arms = len(ALL_ARMS)
    arm_starts = np.array([env.arm_extents[a][0] for a in ALL_ARMS])
    arm_ends = np.array([env.arm_extents[a][1] for a in ALL_ARMS])

    all_chosen, all_maps, all_event_ids = [], [], []
    for e in range(n_events):
        n_bins = int(L_e_arr[e])
        chosen = rng.choice(n_arms, size=n_bins, p=probs[e])
        midpoints = np.array(
            [(env.arm_extents[a][0] + env.arm_extents[a][1]) / 2 for a in chosen]
        )
        maps = midpoints + rng.normal(0.0, sigma, size=n_bins)
        all_chosen.append(chosen)
        all_maps.append(maps)
        all_event_ids.append(np.full(n_bins, e))

    map_arr = np.concatenate(all_maps)
    event_per_bin = np.concatenate(all_event_ids)

    # --- Hard: MAP through the production spatial mapping ---
    decoded_arm = map_estimate_to_arm(map_arr, env)
    hard_counts = np.zeros((n_events, n_arms), dtype=int)
    for t, e in enumerate(event_per_bin):
        a = int(decoded_arm[t])
        hard_counts[e, a] += 1

    # --- Soft: per-arm posterior probability per timebin, summed per event ---
    # P(MAP in arm a | bump centered at MAP_t, std sigma) =
    #     Φ((end_a - MAP_t)/sigma) - Φ((start_a - MAP_t)/sigma)
    # Junction-region mass is excluded (matches the hard path).
    if sigma > 0:
        diffs_end = (arm_ends[None, :] - map_arr[:, None]) / sigma
        diffs_start = (arm_starts[None, :] - map_arr[:, None]) / sigma
        per_bin_arm_prob = norm.cdf(diffs_end) - norm.cdf(diffs_start)
    else:
        # Degenerate bump: posterior mass is entirely on the one arm whose
        # extent contains the MAP. Fall back to indicator counts.
        per_bin_arm_prob = (
            (map_arr[:, None] >= arm_starts[None, :])
            & (map_arr[:, None] < arm_ends[None, :])
        ).astype(float)
    soft_counts_float = np.zeros((n_events, n_arms), dtype=float)
    for e in range(n_events):
        mask = event_per_bin == e
        soft_counts_float[e] = per_bin_arm_prob[mask].sum(axis=0)
    # Rescale to admissible integer counts (ConditionalPoisson requires ints).
    # Scale chosen so the smallest typical event still has > a few counts on
    # the dominant arm; the conditional likelihood is invariant to a uniform
    # scaling of all counts within an event.
    scale = 1000.0
    soft_counts = np.round(soft_counts_float * scale).astype(int)

    def _to_long(counts: np.ndarray) -> pd.DataFrame:
        event_ids = np.repeat(np.arange(n_events), n_arms)
        arm_ids = np.tile(ALL_ARMS, n_events)
        d: dict[str, np.ndarray] = {
            "event_uid": event_ids,
            "arm_id": arm_ids,
            "arm_count": counts.reshape(-1),
        }
        for k, name in enumerate(predictor_list):
            d[name] = design[:, :, k].reshape(-1)
        return pd.DataFrame(d)

    chosen_arr = np.concatenate(all_chosen)
    fraction_on_chosen = float(np.mean(decoded_arm == chosen_arr))

    return (
        _to_long(hard_counts),
        _to_long(soft_counts),
        {
            "fraction_on_chosen": fraction_on_chosen,
            "mean_total_soft_per_event": float(soft_counts.sum(axis=1).mean()),
        },
    )


# ── Full-pipeline simulation: exercise the real transformation chain ──


def make_bandit_track_graph():
    """Return a NetworkX graph matching the real bandit task topology.

    6 wells (nodes 0-5, all degree-1), 3 patch junctions (6-8), 1 center
    junction (9). Each well connects to its patch junction via a leaf arm
    (edge_id = well_id), and each patch junction connects to the center
    via a trunk arm (edge_id 6, 7, 8). Mirrors the fixture in
    ``tests/unit/test_replay_content.py``.
    """
    import networkx as nx

    g = nx.Graph()
    well_positions = {
        0: (0, 0), 1: (10, 0),
        2: (0, 10), 3: (10, 10),
        4: (0, 20), 5: (10, 20),
    }
    for node, pos in well_positions.items():
        g.add_node(node, pos=pos)
    for node, pos in [(6, (5, 2)), (7, (5, 12)), (8, (5, 22)), (9, (5, 12))]:
        g.add_node(node, pos=pos)
    leaf_edges = [(0, 6, 0), (1, 6, 1), (2, 7, 2), (3, 7, 3), (4, 8, 4), (5, 8, 5)]
    trunk_edges = [(6, 9, 6), (7, 9, 7), (8, 9, 8)]
    for u, v, eid in leaf_edges + trunk_edges:
        g.add_edge(u, v, edge_id=eid, distance=10.0)
    return g


def simulate_full_pipeline(
    beta_true: dict[str, float],
    *,
    n_events: int = 200,
    L_e: int | Sequence[int] = 20,
    sigma: float = 0.0,
    env: SyntheticEnv | None = None,
    track_graph=None,
    predictors: Sequence[str] | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run synthetic data through the *real* transformation chain.

    Generates a per-timebin synthetic ``data_over_time``-shaped frame with
    all columns the production pipeline reads, then routes it through the
    actual ``map_estimate_to_arm`` → ``compute_arm_counts`` →
    ``collapse_to_events`` → ``make_long_format_arm`` → ``add_event_uid``
    sequence used in ``data_preparation.py``. This is the only simulator
    that exercises every transformation step, so any bug in column joins,
    event-onset semantics, junction-bin handling, or wide↔long melts will
    surface here as biased recovery.

    Generative model: per event, an animal is "stopped" at a leaf well
    chosen uniformly at random (this becomes ``track_segment_id``,
    matching real-data semantics where the modal arm equals the well).
    The previous trial's from_well, the next trial's to_well, and the
    last_rewarded_well are independently drawn from leaf wells (allowing
    coincidence). For each timebin within the event, a chosen arm is
    sampled from softmax(X β_true), and a MAP linear position is drawn
    from N(arm midpoint, σ) before being mapped back to an arm by the
    production ``map_estimate_to_arm``.

    .. note::
       Trial-context columns (``prev_trial_from_well``, ``next_trial_to_well``,
       ``last_rewarded_well``) are injected directly per timebin rather than
       derived from a synthetic ``trials`` frame via ``add_trial_context``.
       That means this simulator validates the count-aggregation chain
       (``compute_arm_counts`` → ``collapse_to_events`` → ``make_long_format_arm``
       → ``add_event_uid``) but does **not** exercise the trial-context
       computation in ``add_trial_context`` (the ``searchsorted`` "in-progress
       trip" semantics). A dedicated trial-context simulator would be a
       useful complement.

    Parameters mirror ``simulate_posterior_bumps``; ``track_graph`` defaults
    to the real bandit topology via ``make_bandit_track_graph()``.

    Returns
    -------
    long_count : pd.DataFrame
        Output of ``make_long_format_arm`` with ``event_uid``,
        ``arm_count``, and the binary indicator predictors. Ready to pass
        to ``fit_conditional_poisson``.
    events : pd.DataFrame
        Output of ``collapse_to_events`` (one row per event, modal arm).
    diagnostics : dict
        ``fraction_on_chosen``, ``mean_arms_per_event`` (mean number of
        arms with arm_count > 0 — compare to real data to gauge the
        salt-and-pepper-vs-trajectory shape), and ``n_events_used``.
    """
    rng = np.random.default_rng(seed)
    predictor_list = list(predictors) if predictors is not None else list(beta_true)
    beta_vec = np.array([beta_true[p] for p in predictor_list], dtype=float)
    if env is None:
        env = make_synthetic_env()
    if track_graph is None:
        track_graph = make_bandit_track_graph()

    leaf_wells = np.array([0, 1, 2, 3, 4, 5])
    current_well = rng.choice(leaf_wells, size=n_events)
    prev_well = rng.choice(leaf_wells, size=n_events)
    next_well = rng.choice(leaf_wells, size=n_events)
    rewarded_well = rng.choice(leaf_wells, size=n_events)

    # Build the design tensor consistent with the well draws so that the
    # binary indicators *during the fit* match the trial-context columns
    # *during collapse_to_events*. This is the contract the real pipeline
    # relies on.
    n_arms = len(ALL_ARMS)
    arm_grid = np.arange(n_arms)[None, :]
    feature_cols: dict[str, np.ndarray] = {
        "is_current": (arm_grid == current_well[:, None]).astype(float),
        "is_prev": (arm_grid == prev_well[:, None]).astype(float),
        "is_next": (arm_grid == next_well[:, None]).astype(float),
        "is_last_rewarded": (arm_grid == rewarded_well[:, None]).astype(float),
    }
    missing = [p for p in predictor_list if p not in feature_cols]
    if missing:
        raise ValueError(
            f"simulate_full_pipeline only supports binary indicator "
            f"predictors right now; got: {missing}"
        )
    design = np.stack([feature_cols[p] for p in predictor_list], axis=-1)

    if np.isscalar(L_e):
        L_e_arr = np.full(n_events, int(L_e))
    else:
        L_e_arr = np.asarray(L_e, dtype=int)
        if L_e_arr.shape != (n_events,):
            raise ValueError(f"L_e length {L_e_arr.shape} != n_events {n_events}")

    logits = design @ beta_vec
    probs = _softmax(logits)

    # Sample per-timebin chosen arms and MAP estimates.
    rows = []
    chosen_all = []
    for e in range(n_events):
        n_bins = int(L_e_arr[e])
        chosen = rng.choice(n_arms, size=n_bins, p=probs[e])
        midpoints = np.array(
            [(env.arm_extents[a][0] + env.arm_extents[a][1]) / 2 for a in chosen]
        )
        maps = midpoints + rng.normal(0.0, sigma, size=n_bins)
        chosen_all.append(chosen)
        for b in range(n_bins):
            rows.append(
                {
                    "animal": "sim",
                    "day": 20200101,
                    "epoch": "01_r1",
                    "non_local_event": e + 1,  # 1-indexed; collapse drops 0
                    "SWR_event": 0,
                    "map_estimate": float(maps[b]),
                    "track_segment_id": int(current_well[e]),
                    "prev_trial_from_well": float(prev_well[e]),
                    "next_trial_to_well": float(next_well[e]),
                    "last_rewarded_well": float(rewarded_well[e]),
                    "is_after_switch": 0.0,
                    "is_after_omission": 0.0,
                }
            )
    nl = pd.DataFrame(rows)
    # Canonical pipeline: map MAP estimate → arm via the same
    # ``map_estimate_to_arm`` the production code uses, then aggregate
    # per-event counts via ``summaries.arm_count_hard``.
    decoded_arm = map_estimate_to_arm(nl["map_estimate"].values, env)
    nl["decoded_arm"] = decoded_arm
    chosen_arr = np.concatenate(chosen_all)
    fraction_on_chosen = float(np.mean(decoded_arm == chosen_arr))

    # Filter out junction bins (decoded_arm not in ALL_ARMS → NaN from
    # map_estimate_to_arm when sigma pushes MAP onto a junction).
    real_mask = np.isin(decoded_arm, ALL_ARMS) & ~np.isnan(decoded_arm)
    event_ids = nl["non_local_event"].values[real_mask].astype(np.int64)
    arm_map = decoded_arm[real_mask].astype(np.int64)
    # Use a dummy nonlocal_prob of 1.0 (all timebins are "in event")
    # and threshold=0.0 (count every row).
    ids_sorted, counts = _arm_count_hard_fn(
        np.ones(real_mask.sum()),
        arm_map,
        event_ids,
        n_arms=len(ALL_ARMS),
        threshold=0.0,
    )

    # Build the long-format DataFrame with the KNOWN design-matrix
    # indicators (not re-derived from trials, because the test
    # controls the design). This mirrors the canonical
    # ``event_arm_table`` layout.
    n_arms = len(ALL_ARMS)
    long_rows = []
    for idx, ev_id in enumerate(ids_sorted):
        e = int(ev_id) - 1  # 0-indexed into the synthetic event arrays
        for j, arm in enumerate(ALL_ARMS):
            row = {
                "event_uid": int(ev_id),
                "arm_id": arm,
                "arm_count": int(counts[idx, j]),
            }
            for p_idx, p_name in enumerate(predictor_list):
                row[p_name] = float(design[e, j, p_idx])
            long_rows.append(row)

    long_count = pd.DataFrame(long_rows)

    # Filter out zero-total events (all timebins on junction bins).
    event_totals = long_count.groupby("event_uid")["arm_count"].transform("sum")
    long_count = long_count[event_totals > 0].reset_index(drop=True)

    mean_arms_per_event = float(
        (counts > 0).sum(axis=1).mean()
    )

    # Minimal events frame for backward compat with callers that
    # inspect the second return value.
    events = pd.DataFrame({
        "non_local_event": ids_sorted,
        "total_count": counts.sum(axis=1),
    })
    events = events[events["total_count"] > 0].copy()

    diagnostics = {
        "fraction_on_chosen": fraction_on_chosen,
        "mean_arms_per_event": mean_arms_per_event,
        "n_events_used": int(len(events)),
    }
    return long_count, events, diagnostics
