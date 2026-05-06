"""Switch-aligned timing analysis for patch-switch prediction.

Identifies unique switching episodes and collects replay content in
the stops surrounding each switch. Provides matched-control
comparisons for assessing whether replay content shifts before or
after patch transitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

from .choice_prediction import predict_arm_fractions


def build_switch_aligned_table(
    stop_table: pd.DataFrame,
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    *,
    content_fit: dict | None = None,
    content_predictors: list[str] | None = None,
    fraction_col: str = "arm_fraction",
    max_lag: int = 20,
) -> pd.DataFrame:
    """Align replay content to stops preceding patch switches.

    Identifies unique switching episodes (first switch in a run of
    consecutive switches), then collects the preceding ``max_lag``
    stops. For each stop with replay, computes patch-level replay
    content: fraction on the current patch vs non-current patches.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Canonical stop table (includes zero-event stops).
    event_table : pd.DataFrame
        Canonical event table.
    event_arm_table : pd.DataFrame
        Canonical event-arm table (9 rows per event).
    content_fit : dict, optional
        Fitted content model for residualization. If provided along
        with *content_predictors*, adds ``replay_frac_residual_noncurrent``.
    content_predictors : list[str], optional
        Predictors used in *content_fit*.
    max_lag : int
        Maximum number of stops before the switch to include.

    Returns
    -------
    pd.DataFrame
        One row per (episode, lag) for stops with replay. Columns:

        - ``episode_id`` — unique switching episode identifier
        - ``animal``, ``day``, ``epoch`` — session identifiers
        - ``switch_stop_idx`` — stop_idx of the switch
        - ``stop_idx``, ``stop_id`` — the replay stop
        - ``lag`` — stops before switch (negative; -1 = immediately before)
        - ``current_patch`` — patch the animal was in during the run
        - ``replay_frac_current_patch`` — fraction of replay on current patch
        - ``replay_frac_noncurrent`` — fraction on non-current patches
        - ``n_events`` — events in this stop
        - ``replay_frac_residual_noncurrent`` — (if content model provided)
    """
    st = stop_table.copy()

    valid = st[st["is_stay"].notna()].copy()
    if len(valid) == 0:
        return pd.DataFrame()

    valid["is_switch"] = (valid["is_stay"] == 0).astype(int)
    valid["prev_is_switch"] = (
        valid.groupby(["animal", "day", "epoch"])["is_switch"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )
    first_switches = valid[(valid["is_switch"] == 1) & (valid["prev_is_switch"] == 0)]

    if len(first_switches) == 0:
        return pd.DataFrame()

    eat = event_arm_table
    if content_fit is not None and content_predictors is not None:
        eat = predict_arm_fractions(eat, content_fit, content_predictors)

    stop_replay = stop_patch_replay(eat, event_table, fraction_col=fraction_col)

    rows = []
    for ep_i, (_idx, sw_row) in enumerate(first_switches.iterrows()):
        sw_stop_idx = sw_row["stop_idx"]
        current_patch = sw_row["current_patch"]
        animal = sw_row["animal"]
        day = sw_row["day"]
        epoch = sw_row["epoch"]

        # Restrict to the switch's own session — stop_idx is only
        # session-local, so without this filter a switch in (animal A,
        # day D, epoch E) could match preceding stops from any other
        # session that happens to have a stop_idx within the lag
        # window. Lines 116-118 then write the switch's session keys
        # onto every row, masking the cross-session contamination.
        preceding = st[
            (st["animal"] == animal)
            & (st["day"] == day)
            & (st["epoch"] == epoch)
            & (st["stop_idx"] < sw_stop_idx)
            & (st["stop_idx"] >= sw_stop_idx - max_lag)
            & st["has_replay"]
        ]

        for _, ps_row in preceding.iterrows():
            lag = int(ps_row["stop_idx"] - sw_stop_idx)
            sid = ps_row["stop_id"]

            if sid not in stop_replay.index:
                continue

            sr = stop_replay.loc[sid]
            row = {
                "episode_id": ep_i,
                "animal": animal,
                "day": day,
                "epoch": epoch,
                "switch_stop_idx": int(sw_stop_idx),
                "stop_idx": int(ps_row["stop_idx"]),
                "stop_id": sid,
                "lag": lag,
                "current_patch": current_patch,
                "replay_frac_current_patch": sr.get(
                    "replay_frac_current_patch", np.nan
                ),
                "replay_frac_noncurrent": sr.get("replay_frac_noncurrent", np.nan),
                "n_events": int(ps_row["n_events"]),
            }
            if "replay_frac_residual_noncurrent" in sr.index:
                row["replay_frac_residual_noncurrent"] = sr[
                    "replay_frac_residual_noncurrent"
                ]
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_control_aligned_table(
    stop_table: pd.DataFrame,
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    switch_episodes: pd.DataFrame,
    *,
    content_fit: dict | None = None,
    content_predictors: list[str] | None = None,
    fraction_col: str = "arm_fraction",
    max_lag: int = 20,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build matched control table for switch-aligned comparison.

    For each switching episode, finds a random stretch of ``max_lag``
    consecutive stay stops in the same session. Returns the same
    columns as :func:`build_switch_aligned_table`.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Canonical stop table.
    event_table, event_arm_table
        Canonical tables.
    switch_episodes : pd.DataFrame
        Output of :func:`build_switch_aligned_table`.
    content_fit, content_predictors
        For residualization.
    max_lag : int
        Window size for control stretch.
    rng : numpy Generator, optional
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        Same schema as ``build_switch_aligned_table``.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    st = stop_table.copy()

    eat = event_arm_table
    if content_fit is not None and content_predictors is not None:
        eat = predict_arm_fractions(eat, content_fit, content_predictors)

    stop_replay = stop_patch_replay(eat, event_table, fraction_col=fraction_col)

    if len(switch_episodes) == 0:
        return pd.DataFrame()
    ep_meta = switch_episodes.groupby("episode_id").first()[
        ["animal", "day", "epoch", "current_patch", "switch_stop_idx"]
    ]

    rows = []
    for ep_id, meta in ep_meta.iterrows():
        animal = meta["animal"]
        day = meta["day"]
        epoch = meta["epoch"]
        patch = meta["current_patch"]
        sw_idx = meta["switch_stop_idx"]

        session_stays = st[
            (st["animal"] == animal)
            & (st["day"] == day)
            & (st["epoch"] == epoch)
            & (st["is_stay"] == 1)
            & (
                (st["stop_idx"] < sw_idx - max_lag)
                | (st["stop_idx"] > sw_idx + max_lag)
            )
        ].sort_values("stop_idx")

        if len(session_stays) < 5:
            continue

        possible_anchors = session_stays.iloc[5:]
        if len(possible_anchors) == 0:
            continue
        anchor_idx = rng.integers(0, len(possible_anchors))
        anchor_stop_idx = possible_anchors.iloc[anchor_idx]["stop_idx"]

        # Match the switch-aligned builder: stop_idx is session-local,
        # so the preceding-stop filter must restrict to the anchor's
        # own session. Otherwise the writer at lines ~237-241 stamps
        # this episode's session keys onto stops drawn from other
        # sessions that happened to share the stop_idx range.
        preceding = st[
            (st["animal"] == animal)
            & (st["day"] == day)
            & (st["epoch"] == epoch)
            & (st["stop_idx"] < anchor_stop_idx)
            & (st["stop_idx"] >= anchor_stop_idx - max_lag)
            & st["has_replay"]
        ]

        for _, ps_row in preceding.iterrows():
            lag = int(ps_row["stop_idx"] - anchor_stop_idx)
            sid = ps_row["stop_id"]
            if sid not in stop_replay.index:
                continue
            sr = stop_replay.loc[sid]
            row = {
                "episode_id": ep_id,
                "animal": animal,
                "day": day,
                "epoch": epoch,
                "switch_stop_idx": int(anchor_stop_idx),
                "stop_idx": int(ps_row["stop_idx"]),
                "stop_id": sid,
                "lag": lag,
                "current_patch": patch,
                "replay_frac_current_patch": sr.get(
                    "replay_frac_current_patch", np.nan
                ),
                "replay_frac_noncurrent": sr.get("replay_frac_noncurrent", np.nan),
                "n_events": int(ps_row["n_events"]),
            }
            if "replay_frac_residual_noncurrent" in sr.index:
                row["replay_frac_residual_noncurrent"] = sr[
                    "replay_frac_residual_noncurrent"
                ]
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def stop_patch_replay(
    event_arm_table: pd.DataFrame,
    event_table: pd.DataFrame,
    *,
    fraction_col: str = "arm_fraction",
) -> pd.DataFrame:
    """Compute per-stop replay fraction on current vs non-current patches.

    Vectorized implementation — avoids per-group ``.apply()`` lambdas.

    Parameters
    ----------
    event_arm_table : pd.DataFrame
        Canonical event-arm table.
    event_table : pd.DataFrame
        Canonical event table.
    fraction_col : str
        Column with observed arm fractions. ``"arm_fraction"``
        (default, MAP-based) or ``"arm_fraction_posterior"``.

    Returns a DataFrame indexed by ``stop_id`` with columns:
    ``replay_frac_current_patch``, ``replay_frac_noncurrent``, and
    optionally ``replay_frac_residual_noncurrent`` if
    ``residual_frac`` is present in *event_arm_table*.
    """
    # Both ``stop_id`` and ``event_id`` must be globally unique across
    # sessions. The per-stop average over events conflates sessions
    # otherwise, and per-event sums of arm fractions can climb above
    # 1 as multiple sessions' events with the same id are added
    # together. Reuse the shared precondition helper.
    from .choice_prediction import _require_session_unique_ids

    _require_session_unique_ids(
        event_arm_table, "event_id", helper_name="stop_patch_replay"
    )
    _require_session_unique_ids(
        event_arm_table, "stop_id", helper_name="stop_patch_replay"
    )

    eat = event_arm_table.copy()

    # Map arms to patches
    eat["_patch"] = eat["arm"].map(TRACK_SEGMENT_TO_PATCH)

    # current_patch is already in event_arm_table; fall back to merge.
    if "current_patch" not in eat.columns:
        evt_patch = event_table[["event_id", "current_patch"]].drop_duplicates()
        eat = eat.merge(evt_patch, on="event_id", how="left")

    # Flag arms in the current patch. ``NaN == anything`` is False
    # in pandas/numpy, so a missing ``current_patch`` would silently
    # mark every arm as "non-current" and shovel the whole event's
    # replay fraction into ``replay_frac_noncurrent`` while reporting
    # ``replay_frac_current_patch == 0.0``. The current-vs-non-current
    # split is undefined when the current patch is unknown, so route
    # those rows to NaN explicitly: NaN × (anything) propagates
    # through the ``skipna=False`` per-event sum below.
    cp = eat["current_patch"]
    p = eat["_patch"]
    is_cp_float = np.where(
        cp.isna() | p.isna(),
        np.nan,
        (p.to_numpy() == cp.to_numpy()).astype(float),
    )

    # Per-event: sum fraction for current-patch and noncurrent arms
    eat["_frac_cp"] = eat[fraction_col].to_numpy() * is_cp_float
    eat["_frac_nc"] = eat[fraction_col].to_numpy() * (1.0 - is_cp_float)

    # Per-event sums use ``skipna=False`` so an event with any NaN
    # arm fraction (or NaN residual) propagates NaN to the per-event
    # value instead of pandas' default skip-NaN behavior, which
    # silently fabricates a real zero from missing data. The
    # per-stop mean below uses the same strict semantics so a stop
    # containing any contaminated event surfaces NaN to downstream
    # consumers (matching the contract enforced in
    # ``build_residualized_choice_dataset``).
    event_agg = pd.DataFrame(
        {
            "replay_frac_current_patch": (
                eat.groupby(["stop_id", "event_id"])["_frac_cp"]
                .apply(lambda s: s.sum(skipna=False))
            ),
            "replay_frac_noncurrent": (
                eat.groupby(["stop_id", "event_id"])["_frac_nc"]
                .apply(lambda s: s.sum(skipna=False))
            ),
        }
    )

    # Average across events within each stop, NaN-aware.
    result = (
        event_agg.groupby("stop_id")[
            ["replay_frac_current_patch", "replay_frac_noncurrent"]
        ]
        .agg(lambda s: s.mean(skipna=False))
    )

    # Residualized version if available
    if "residual_frac" in eat.columns:
        # Same NaN-aware multiplication: missing ``current_patch``
        # routes the residual mass to NaN, not to "non-current".
        eat["_resid_nc"] = eat["residual_frac"].to_numpy() * (1.0 - is_cp_float)
        event_resid = (
            eat.groupby(["stop_id", "event_id"])["_resid_nc"]
            .apply(lambda s: s.sum(skipna=False))
            .rename("replay_frac_residual_noncurrent")
            .to_frame()
        )
        resid_agg = (
            event_resid.groupby("stop_id")["replay_frac_residual_noncurrent"]
            .apply(lambda s: s.mean(skipna=False))
            .rename("replay_frac_residual_noncurrent")
            .to_frame()
        )
        result = result.join(resid_agg)

    return result
