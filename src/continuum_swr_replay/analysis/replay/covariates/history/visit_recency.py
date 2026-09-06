"""Visit-recency covariates: how recently (and whether ever) each arm was visited, at event time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.model_specs import (
    RECENCY_SPLINE_BOUNDS,
    RECENCY_SPLINE_KNOTS,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    WELL_ARMS,
    WELL_PATCH_MAPPING,
)

from ._common import _prepare_port_experiences

# ── Candidate-arm histories evaluated at event time ──


def center_visit_recency_within_event(
    recency: np.ndarray,
    never_visited: np.ndarray,
    *,
    log_scale: bool = False,
) -> np.ndarray:
    """Center leaf-arm recency within each event without using future events.

    Only *visited* reward-site arms participate in the mean. A never-visited
    arm carries the right-censoring fill (``n_completed + 1``), which is a
    property of the censoring rule rather than a measured recency; letting it
    into the mean would displace every other arm in the event by an arbitrary
    amount. In a session with one durably avoided well the fill can exceed the
    visited arms by two orders of magnitude, so the resulting coefficient would
    be estimated off the censoring rule instead of off recency.
    ``arm_never_visited`` carries the censored state on its own.

    Structural trunk values remain zero once history exists, and an event with
    no visited leaf arm remains all-NaN. Guarantees that appending later
    events cannot change an earlier row.

    Parameters
    ----------
    recency : np.ndarray, shape (n_events, len(ALL_ARMS))
        Per-event per-arm visit recency.
    never_visited : np.ndarray, shape (n_events, len(ALL_ARMS))
        1.0 where the arm has never been visited by this event, else 0.0.
        NaN is treated as not-censored, matching ``recency``'s own NaN.
    log_scale : bool
        Apply ``log1p`` to the raw recency *before* centering. Recency is a
        trial count spanning 1 to ~200, and it is deterministic on the two arms
        that matter -- exactly 1 on the current arm and exactly 2 on the
        previous arm -- so a linear term forces the current-versus-previous
        contrast to one unit while the data show far more. Within event, among
        the arms that are neither current nor previous, the relationship with
        replay share is monotone (median within-event r = +0.24, positive in
        70% of events), so a monotone transform is the right shape; a spline
        is not needed. Transforming before centering keeps the censoring rule
        below unchanged -- the mean is still taken over visited leaves only.

    Returns
    -------
    np.ndarray, shape (n_events, len(ALL_ARMS))
        Recency with visited leaf arms centered within each event.
    """
    values = np.asarray(recency, dtype=float)
    censored = np.asarray(never_visited, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(ALL_ARMS):
        raise ValueError(
            f"recency must have shape (n_events, len(ALL_ARMS)); got {values.shape}"
        )
    if censored.shape != values.shape:
        raise ValueError(
            "never_visited must match recency shape "
            f"{values.shape}; got {censored.shape}"
        )
    if log_scale:
        values = np.log1p(values)
    output = values.copy()
    arm_positions = {arm: index for index, arm in enumerate(ALL_ARMS)}
    leaf_positions = np.asarray([arm_positions[arm] for arm in WELL_ARMS])
    leaf_values = values[:, leaf_positions]
    visited = np.isfinite(leaf_values) & (censored[:, leaf_positions] != 1.0)
    counts = visited.sum(axis=1)
    totals = np.where(visited, leaf_values, 0.0).sum(axis=1)
    means = np.divide(
        totals,
        counts,
        out=np.full(len(values), np.nan),
        where=counts > 0,
    )
    output[:, leaf_positions] = leaf_values - means[:, None]
    return output


def spline_visit_recency_within_event(
    recency: np.ndarray,
    never_visited: np.ndarray,
    is_current: np.ndarray,
    is_prev: np.ndarray,
    arm: np.ndarray,
    event_id: np.ndarray,
) -> np.ndarray:
    """Event-centered natural-spline basis of recency, on its identified support.

    The basis is built on the raw trial-count scale (see the knot constants in
    ``model_specs``), so a coefficient reads per unit of the basis at knots that
    are themselves trial counts.

    All inputs are flat per-(event, arm) row arrays of equal length. Returns
    ``(n_rows, len(RECENCY_SPLINE_COLUMNS))``, zero outside the masked support so
    the columns carry no weight on arms whose recency is already determined.

    The within-event mean is taken over the masked rows only, mirroring
    :func:`center_visit_recency_within_event`: an arm outside the support must
    not displace the arms inside it.
    """
    from patsy import dmatrix

    values = np.asarray(recency, dtype=float)
    mask = (
        (np.asarray(arm) < len(WELL_ARMS))
        & (np.asarray(never_visited, dtype=float) == 0.0)
        & (np.asarray(is_current, dtype=float) == 0.0)
        & (np.asarray(is_prev, dtype=float) == 0.0)
        & np.isfinite(values)
    )
    basis = np.asarray(
        dmatrix(
            "cr(x, knots=k, lower_bound=lo, upper_bound=hi) - 1",
            {
                "x": np.where(np.isfinite(values), values, RECENCY_SPLINE_BOUNDS[0]),
                "k": list(RECENCY_SPLINE_KNOTS),
                "lo": RECENCY_SPLINE_BOUNDS[0],
                "hi": RECENCY_SPLINE_BOUNDS[1],
            },
            return_type="dataframe",
        )
    )[:, 1:]
    out = basis * mask[:, None]
    events = pd.Series(np.asarray(event_id))
    for column in range(out.shape[1]):
        masked = pd.Series(np.where(mask, out[:, column], np.nan))
        means = masked.groupby(events).transform("mean").fillna(0.0).to_numpy()
        out[:, column] = np.where(mask, out[:, column] - means, 0.0)
    return out


def visit_recency_all_events(
    event_times: np.ndarray,
    port_experiences: pd.DataFrame,
    well_to_arm: dict[int, int],
) -> np.ndarray:
    """Completed port experiences since each arm was last visited.

    A reachable arm not yet visited at an event receives
    ``n_arrived_experiences + 1``. This right-censored value depends only on
    history available at the event; appending later experiences cannot change it.
    Trunk/non-reachable arms are structural ``0`` once history exists, and all
    arms are ``NaN`` before the first recorded port occupancy.

    """
    recency, _ = visit_recency_and_never_visited_all_events(
        event_times, port_experiences, well_to_arm
    )
    return recency


def arm_never_visited_all_events(
    event_times: np.ndarray,
    port_experiences: pd.DataFrame,
    well_to_arm: dict[int, int],
) -> np.ndarray:
    """Indicate arms not visited before each event.

    Returns a float ``(n_events, len(ALL_ARMS))`` array so unknown history can
    remain ``NaN``. With completed history, reachable never-visited arms are
    ``1``, visited arms are ``0``, and trunk/non-reachable arms are structural
    ``0``.
    """
    _, never_visited = visit_recency_and_never_visited_all_events(
        event_times, port_experiences, well_to_arm
    )
    return never_visited


def visit_recency_and_never_visited_all_events(
    event_times: np.ndarray,
    port_experiences: pd.DataFrame,
    well_to_arm: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute time-local recency and novelty from one shared history pass.

    This is the batched source of truth for the two coupled visit-history
    covariates.  Both outputs use the same arrival-time cutoffs, arm mapping,
    and right-censoring rule, so callers that need both should use this helper
    rather than scanning the experience stream once per output.

    Returns
    -------
    recency, never_visited : tuple[np.ndarray, np.ndarray]
        Two float arrays with shape ``(n_events, len(ALL_ARMS))``.  Rows before
        any completed arrival are all-NaN; after history begins, trunk arms are
        structural zero and unvisited leaf arms carry the time-local censored
        recency plus a novelty indicator of one.
    """
    n_events = len(event_times)
    n_arms = len(ALL_ARMS)
    experiences = _prepare_port_experiences(port_experiences)
    n_experiences = len(experiences)
    recency = np.full((n_events, n_arms), np.nan)
    never_visited = np.full((n_events, n_arms), np.nan)

    if n_experiences == 0:
        return recency, never_visited

    arm_idx = {arm: j for j, arm in enumerate(ALL_ARMS)}
    mapped_wells = set(well_to_arm)
    expected_wells = set(WELL_PATCH_MAPPING)
    if mapped_wells != expected_wells:
        missing = sorted(expected_wells - mapped_wells)
        unexpected = sorted(mapped_wells - expected_wells)
        raise ValueError(
            "well_to_arm must cover every canonical reward-site well exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )
    mapped_leaf_arms = set(well_to_arm.values())
    expected_leaf_arms = set(WELL_ARMS)
    if mapped_leaf_arms != expected_leaf_arms:
        missing = sorted(expected_leaf_arms - mapped_leaf_arms)
        unexpected = sorted(mapped_leaf_arms - expected_leaf_arms)
        raise ValueError(
            "well_to_arm must cover every canonical reward-site leaf arm "
            f"exactly; missing={missing}, unexpected={unexpected}"
        )
    if len(well_to_arm) != len(expected_leaf_arms):
        raise ValueError(
            "well_to_arm must map exactly one well to each canonical leaf arm"
        )
    reachable_arms = expected_leaf_arms

    cutoff_times = experiences["arrival_time"].to_numpy(dtype=float)
    ports = experiences["port"].to_numpy(dtype=np.int64)
    # ``cutoffs[e]`` = how many port experiences have already ARRIVED by event e.
    # Example: arrivals at t=[1, 3, 5], event at t=4 -> searchsorted(side="right")
    # = 2 (the t=1 and t=3 experiences are history; t=5 is not yet). has_history
    # is False for events before the very first arrival (cutoff 0).
    cutoffs = np.searchsorted(cutoff_times, event_times, side="right")
    has_history = cutoffs > 0

    visit_lists: list[list[int]] = [[] for _ in ALL_ARMS]
    for experience_idx, port in enumerate(ports):
        arm = well_to_arm.get(int(port))
        if arm in arm_idx:
            visit_lists[arm_idx[arm]].append(experience_idx)

    for j, arm in enumerate(ALL_ARMS):
        if arm not in reachable_arms:
            recency[has_history, j] = 0.0
            never_visited[has_history, j] = 0.0
            continue

        # Right-censored fill for a reachable arm with no visit yet. Unlike a
        # session-wide default, this can be known at the event time.
        recency[has_history, j] = cutoffs[has_history] + 1.0
        never_visited[has_history, j] = 1.0

        visits = np.asarray(visit_lists[j], dtype=np.intp)
        if visits.size == 0:
            continue
        n_visits = np.searchsorted(visits, cutoffs, side="left")
        was_visited = has_history & (n_visits > 0)
        last_visit = visits[np.clip(n_visits[was_visited] - 1, 0, None)]
        recency[was_visited, j] = cutoffs[was_visited] - last_visit
        never_visited[was_visited, j] = 0.0

    return recency, never_visited
