"""Canonical qvals value-block and arm-uncertainty joins.

Reads the precomputed per-stop value artifact (``qvals_table``) and joins the
candidate-arm scheduled reward probability, reward-belief mean, and pre-outcome
belief variance onto the event/stop tables. This is a join of a precomputed
artifact, distinct from the compute-from-behavior covariates in
:mod:`value_uncertainty`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    WELL_ARMS,
    port_columns,
)
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table, table_exists

_P_REWARD_PORT_COLS = port_columns("task_reward_probability")
_Q_PORT_COLS = port_columns("reward_belief_mean")
_BETADIST_VAR_COLS = port_columns("reward_belief_variance")


def load_contingency(path: Path) -> pd.DataFrame | None:
    """Load the per-stop value artifact (``qvals_table``) for the value block.

    Shared by the NonLocal (``prepare_data``) and ContFrag prep entry points.
    ``qvals_table`` is the sole source for the candidate-arm scheduled reward
    probability and reward-belief mean, so its
    absence must degrade visibly, not silently: present -> returns the frame and
    logs the row count; absent -> returns ``None`` and logs a warning (the
    builders then emit those columns as NaN, dropping uniformly at fit time).
    """
    path = Path(path)
    if table_exists(path):
        cont = read_table(path)
        require_covariate_schema(cont, table="qvals")
        logging.info(
            "loaded qvals_table (%d stop rows) for task probability / belief mean",
            len(cont),
        )
        return cont
    logging.warning(
        "no qvals_table at %s — candidate-arm task probability / belief mean will be NaN",
        path,
    )
    return None


def _validate_candidate_arms(arm: np.ndarray) -> np.ndarray:
    """Coerce arm ids to a validated int64 array of canonical arms (0..8)."""
    arm = np.asarray(arm, dtype=float)
    if arm.ndim != 1 or not np.isfinite(arm).all():
        raise ValueError("arm IDs must be a finite one-dimensional array")
    if not np.equal(arm, np.floor(arm)).all() or not np.isin(arm, ALL_ARMS).all():
        raise ValueError(f"arm IDs must be canonical members of {tuple(ALL_ARMS)}")
    return arm.astype(np.int64)


def _merge_contingency_on_stop(
    stop_id: np.ndarray,
    arm: np.ndarray,
    contingency: pd.DataFrame,
    port_cols: list[str],
    *,
    alignment: str,
) -> pd.DataFrame:
    """Left-join the per-port value columns onto (event, arm) rows on ``stop_id``.

    Requires one contingency row per stop (``alignment`` names the value block in
    the duplicate-stop error). The returned frame carries the ``port_cols`` plus a
    ``_qvals_merge`` indicator marking which query rows matched a contingency row.
    """
    if contingency["stop_id"].isna().any():
        raise ValueError("contingency stop_id must not be missing")
    n_unique = contingency["stop_id"].nunique(dropna=False)
    if n_unique != len(contingency):
        raise ValueError(
            f"contingency has {len(contingency) - n_unique} duplicate stop_id "
            f"row(s); {alignment} alignment requires one row per stop"
        )
    cont = contingency[["stop_id", *port_cols]]
    return pd.DataFrame({"stop_id": np.asarray(stop_id), "candidate_arm": arm}).merge(
        cont, on="stop_id", how="left", validate="many_to_one", indicator="_qvals_merge"
    )


def _pick_leaf_port_value(
    arm: np.ndarray,
    present: np.ndarray,
    value_matrix: np.ndarray,
) -> np.ndarray:
    """Pick each leaf arm's own port value from a matched per-port matrix.

    Leaf arms 0..5 take their own port column; trunk arms 6..8 are not reward
    sites, so they take ``0.0`` where the stop matched the contingency and
    ``NaN`` where it did not -- a stop with no contingency row drops uniformly
    across all nine arms rather than injecting a spurious 0.0. The caller is
    responsible for the value-specific finite/range guards on the matched rows.
    """
    leaf = arm < len(WELL_ARMS)
    leaf_idx = np.clip(arm, 0, len(WELL_ARMS) - 1).astype(int)
    row = np.arange(len(arm))
    trunk_fill = np.where(present, 0.0, np.nan)
    return np.where(leaf, value_matrix[row, leaf_idx], trunk_fill)


def attach_contingency_value_block(
    arm: np.ndarray,
    stop_id: np.ndarray,
    contingency: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(event, arm) nominal p(R) and model value (Q) from the contingency.

    Single source for the per-arm value block shared by the NonLocal
    (``tables.py``) and ContFrag (``contfrag/tables.py``) event-arm builders
    (``docs/covariate_dictionary.md`` D2/D3). Both builders previously inlined
    this join, which let the join key silently diverge.

    Parameters
    ----------
    arm : np.ndarray, shape (n_rows,)
        Arm id (0-8) for each (event, arm) row.
    stop_id : np.ndarray, shape (n_rows,)
        Join key matching ``contingency["stop_id"]``. The caller supplies the
        key that matches the artifact: the NonLocal pipeline passes the event-arm
        table's own ``stop_id``; ContFrag passes the *plain*
        ``hash_stop_id(animal, day, epoch, stop_idx)`` because its own
        ``stop_id`` mixes in ``brain_area`` / ``event_source`` and so matches
        nothing in the (NonLocal-keyed) contingency.
    contingency : pd.DataFrame | None
        Per-stop artifact with ``stop_id`` plus the canonical nominal reward
        probabilities and reward-belief means for ports 0..5. ``None`` (no
        artifact supplied) yields all-NaN. A
        supplied frame that is *missing* those columns raises ``ValueError``:
        ``qvals_table`` is the sole source, so a stale/partial artifact must
        fail loudly rather than silently blanking the whole value block.

    Returns
    -------
    task_probability, belief_mean : np.ndarray, each shape (n_rows,)
        Leaf arm ``j`` takes ``task_reward_probability_port_{j}`` /
        ``reward_belief_mean_port_{j}``; trunk arms
        (6-8) are not reward sites and take ``0.0`` where the stop is present in
        the contingency, ``NaN`` where absent — so a stop with no contingency row
        drops uniformly (all nine arms NaN) at fit time rather than contributing
        a spurious 0.0.

    Raises
    ------
    ValueError
        If ``contingency`` is supplied but lacks the canonical per-port
        probability or reward-belief columns.
    """
    arm = _validate_candidate_arms(arm)
    n_rows = len(arm)
    if contingency is None:
        return np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    missing = {"stop_id", *_P_REWARD_PORT_COLS, *_Q_PORT_COLS} - set(
        contingency.columns
    )
    if missing:
        raise ValueError(
            "contingency is missing per-arm value columns "
            f"{sorted(missing)}; qvals_table is the sole source and must carry "
            "task_reward_probability_port_{0..5} / "
            "reward_belief_mean_port_{0..5} (regenerate with "
            "`uv run python scripts/behavior/build_qvals_table.py --write`)"
        )
    merged = _merge_contingency_on_stop(
        stop_id,
        arm,
        contingency,
        [*_P_REWARD_PORT_COLS, *_Q_PORT_COLS],
        alignment="q/value",
    )
    port_mat = merged[_P_REWARD_PORT_COLS].to_numpy(dtype=float)
    q_mat = merged[_Q_PORT_COLS].to_numpy(dtype=float)
    present = merged["_qvals_merge"].eq("both").to_numpy()
    if present.any() and not np.isfinite(port_mat[present]).all():
        raise ValueError(
            "matched qvals rows must carry finite task reward probabilities"
        )
    if present.any() and not np.isfinite(q_mat[present]).all():
        raise ValueError(
            "matched contingency rows must carry finite reward-belief means"
        )
    if present.any() and (
        ((port_mat[present] < 0.0) | (port_mat[present] > 1.0)).any()
        or ((q_mat[present] < 0.0) | (q_mat[present] > 1.0)).any()
    ):
        raise ValueError(
            "matched contingency probabilities and Q values must lie in [0, 1]"
        )
    task_probability = _pick_leaf_port_value(arm, present, port_mat)
    belief_mean = _pick_leaf_port_value(arm, present, q_mat)
    return task_probability, belief_mean


def attach_arm_uncertainty(
    arm: np.ndarray,
    stop_id: np.ndarray,
    contingency: pd.DataFrame | None,
) -> np.ndarray:
    """Candidate-arm Beta reward-belief variance.

    It is the variance counterpart of
    ``reward_belief_mean_candidate_arm_preoutcome``: both come from the same
    Beta(alpha, beta) posterior,
    so a value+uncertainty model stays construct-coherent. Built
    with the SAME leaf->arm mapping as :func:`attach_contingency_value_block`,
    sourced from ``reward_belief_variance_port_{0..5}`` (the per-leaf Beta posterior variance
    ``alpha*beta / ((alpha+beta)^2 (alpha+beta+1))``). Leaf arm ``j`` takes
    port ``j``; trunk arms (6-8) are not reward sites and take ``0.0``
    where the stop is present in the contingency and ``NaN`` where absent, so a
    stop missing the value block drops uniformly across all nine arms at fit time
    (matching the candidate-arm belief mean).

    Parameters
    ----------
    arm : np.ndarray, shape (n_rows,)
        Arm id per row (0-5 leaf, 6-8 trunk).
    stop_id : np.ndarray, shape (n_rows,)
        Stop id per row, keyed into ``contingency``.
    contingency : pd.DataFrame or None
        ``qvals_table`` rows carrying the canonical per-port reward-belief
        variances; ``None`` -> all NaN.

    Returns
    -------
    reward_belief_variance : np.ndarray, shape (n_rows,)
    """
    arm = _validate_candidate_arms(arm)
    n_rows = len(arm)
    if contingency is None:
        return np.full(n_rows, np.nan)
    missing = {"stop_id", *_BETADIST_VAR_COLS} - set(contingency.columns)
    if missing:
        raise ValueError(
            "contingency is missing per-leaf beta-variance columns "
            f"{sorted(missing)}; regenerate qvals_table with "
            "`uv run python scripts/behavior/build_qvals_table.py --write` "
            "(carries reward_belief_variance_port_{0..5})."
        )
    merged = _merge_contingency_on_stop(
        stop_id, arm, contingency, list(_BETADIST_VAR_COLS), alignment="uncertainty"
    )
    var_mat = merged[_BETADIST_VAR_COLS].to_numpy(dtype=float)
    present = merged["_qvals_merge"].eq("both").to_numpy()
    if present.any() and not np.isfinite(var_mat[present]).all():
        raise ValueError(
            "matched contingency rows must carry finite betadist variance values"
        )
    if present.any() and ((var_mat[present] < 0.0) | (var_mat[present] > 0.25)).any():
        raise ValueError("matched Beta posterior variances must lie in [0, 0.25]")
    return _pick_leaf_port_value(arm, present, var_mat)


QVALS_TIME_TOLERANCE_SEC: float = 1e-6
"""Maximum ``|qvals_context_time - stop_start|`` accepted as the same stop."""

SESSION_SPAN_PAD_SEC: float = 600.0
"""Slack on each end of a stop table's own span when claiming qvals rows."""


def _qvals_rows_expected_in(
    stop_table: pd.DataFrame, contingency: pd.DataFrame
) -> np.ndarray:
    """Mask the qvals rows the given stop table is expected to account for.

    ``qvals_table`` has no animal/date columns, so a single-session stop table
    cannot be matched against the corpus-wide artifact by key alone. A qvals
    row is claimed by a stop table when either its ``stop_id`` names one of the
    table's stops -- those are joined, so their clock is always checked -- or
    its epoch appears in the table and its ``qvals_context_time`` falls inside
    that epoch's span of ``stop_start``, which is how a row whose ``stop_id``
    went stale is still recognised as belonging here. A corpus-wide stop table
    therefore claims every qvals row of every epoch it covers, and a
    one-session stop table claims only that session's rows.

    The span is padded by ``SESSION_SPAN_PAD_SEC`` on each end because the
    bouts that went missing can be the session's own first and last, which
    would otherwise shrink the span past them and let them escape unnoticed;
    the smallest gap between sessions in this corpus is 61,589 s, so ten
    minutes of slack cannot reach a neighbouring session.
    """
    spans = stop_table.groupby("epoch", sort=False)["stop_start"].agg(["min", "max"])
    epoch = contingency["epoch"].to_numpy()
    time = contingency["qvals_context_time"].to_numpy(dtype=float)
    low = spans["min"].reindex(epoch).to_numpy(dtype=float) - SESSION_SPAN_PAD_SEC
    high = spans["max"].reindex(epoch).to_numpy(dtype=float) + SESSION_SPAN_PAD_SEC
    with np.errstate(invalid="ignore"):
        in_span = (time >= low) & (time <= high)
    joined = contingency["stop_id"].isin(set(stop_table["stop_id"])).to_numpy()
    return in_span | joined


def assert_qvals_aligned(
    stop_table: pd.DataFrame,
    contingency: pd.DataFrame | None,
    *,
    tolerance: float = QVALS_TIME_TOLERANCE_SEC,
) -> None:
    """Fail if the qvals value block is not on the live stop table's clock.

    ``qvals_table`` is keyed by ``stop_id``, a hash of
    ``(animal, day, epoch, stop_idx)``. Renumbering the immobility bouts --
    a re-decode, a changed speed threshold, a changed cohort -- reassigns
    those hashes, and the join then silently attaches one stop's value block
    to a different stop, or drops it. The stop id carries no timestamp, so the
    only way to notice is to compare the two clocks: ``qvals_context_time`` is
    the bout start the value artifact was built against and must equal the
    live ``stop_start`` for the same ``stop_id``.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Live stop table (one session or the whole corpus) with ``stop_id``,
        ``stop_start``, and ``epoch``.
    contingency : pd.DataFrame or None
        ``qvals_table`` rows, or ``None`` when the artifact is absent (nothing
        is joined, so there is nothing to check).
    tolerance : float, optional
        Maximum accepted absolute difference between the two clocks, seconds.

    Raises
    ------
    ValueError
        If a qvals row the stop table is expected to account for has no
        matching ``stop_id``, or if a matched row's ``qvals_context_time``
        differs from ``stop_start`` by more than ``tolerance``.
    """
    if contingency is None or contingency.empty or stop_table.empty:
        return
    missing_cols = {"stop_id", "qvals_context_time", "epoch"} - set(contingency.columns)
    if missing_cols:
        raise ValueError(f"qvals table is missing columns {sorted(missing_cols)}")
    missing_cols = {"stop_id", "stop_start", "epoch"} - set(stop_table.columns)
    if missing_cols:
        raise ValueError(f"stop table is missing columns {sorted(missing_cols)}")

    expected = contingency.loc[
        _qvals_rows_expected_in(stop_table, contingency),
        ["stop_id", "qvals_context_time"],
    ]
    if expected.empty:
        return
    merged = expected.merge(
        stop_table[["stop_id", "stop_start"]],
        on="stop_id",
        how="left",
        validate="one_to_one",
    )
    absent = merged["stop_start"].isna()
    if absent.any():
        first = merged.loc[absent].iloc[0]
        raise ValueError(
            f"{int(absent.sum())} qvals row(s) name a stop that is not in the "
            f"stop table; first stop_id={first['stop_id']} at "
            f"qvals_context_time={first['qvals_context_time']:.6f}. Regenerate "
            "qvals_table against the live stop table."
        )
    drift = (merged["qvals_context_time"] - merged["stop_start"]).abs()
    offending = drift > tolerance
    if offending.any():
        first = merged.loc[offending].iloc[0]
        raise ValueError(
            f"{int(offending.sum())} qvals row(s) are off the stop table's "
            f"clock by more than {tolerance:g} s; first stop_id="
            f"{first['stop_id']} has qvals_context_time="
            f"{first['qvals_context_time']:.6f} but stop_start="
            f"{first['stop_start']:.6f}. Regenerate qvals_table against the "
            "live stop table."
        )
