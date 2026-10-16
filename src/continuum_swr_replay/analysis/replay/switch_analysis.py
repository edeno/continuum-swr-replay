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

from ._helpers import strict_mean, strict_sum
from .choice_prediction import predict_arm_fractions


def _stop_replay_per_patch(
    event_arm_table: pd.DataFrame,
    event_table: pd.DataFrame,
    *,
    fraction_col: str,
) -> pd.DataFrame:
    """Per-stop fraction of replay mass on each patch label.

    Returns DataFrame indexed by ``stop_id`` with columns ``1``, ``2``,
    ``3`` (patch labels) holding the strict-NaN-aware mean across
    events of the per-event sum of ``fraction_col`` over arms in that
    patch.

    Used by :func:`build_switch_aligned_table` and
    :func:`build_control_aligned_table` to look up replay relative to
    a TARGET patch (the switch's patch) rather than each individual
    replay stop's own current_patch — otherwise replay mass that
    landed on the previous patch in a multi-patch run would be
    silently re-labeled as "replay on the switch patch".

    Requires ``stop_id`` and ``event_id`` to be globally unique across
    sessions, like the rest of the per-stop / per-event helpers in
    this module — without that, the ``groupby(['stop_id',
    'event_id'])`` and ``groupby('stop_id')`` reductions silently
    average across sessions whenever a value is reused.
    """
    from .choice_prediction import _require_session_unique_ids

    eat = event_arm_table.copy()
    eat["_patch"] = eat["candidate_arm"].map(TRACK_SEGMENT_TO_PATCH)

    # Bring ``stop_id`` in BEFORE the uniqueness guards. The guards
    # are no-ops when ``id_col`` is absent from the frame, so checking
    # ``event_arm_table`` directly would silently skip the
    # ``stop_id`` check on the fallback-merge path — and a session-
    # local ``stop_id`` from ``event_table`` would then mix across
    # sessions in the per-stop reduction below.
    if "stop_id" not in eat.columns:
        eat = eat.merge(
            event_table[["event_id", "stop_id"]].drop_duplicates(),
            on="event_id",
            how="left",
        )

    _require_session_unique_ids(eat, "event_id", helper_name="_stop_replay_per_patch")
    _require_session_unique_ids(eat, "stop_id", helper_name="_stop_replay_per_patch")

    p_arr = eat["_patch"].to_numpy(dtype=float)
    f_arr = eat[fraction_col].to_numpy(dtype=float)

    out: dict[int, pd.Series] = {}
    for patch in (1, 2, 3):
        is_target = np.where(
            np.isnan(p_arr),
            np.nan,
            (p_arr == patch).astype(float),
        )
        contrib = f_arr * is_target
        per_event = (
            pd.DataFrame(
                {
                    "stop_id": eat["stop_id"].to_numpy(),
                    "event_id": eat["event_id"].to_numpy(),
                    "_contrib": contrib,
                }
            )
            .groupby(["stop_id", "event_id"])["_contrib"]
            .apply(strict_sum)
        )
        per_stop = per_event.groupby("stop_id").apply(strict_mean)
        out[patch] = per_stop

    return pd.DataFrame(out)


def _patch_relative_fracs(
    sid: int,
    target_patch: float,
    per_stop_per_patch: pd.DataFrame,
    per_stop_per_patch_residual: pd.DataFrame | None,
) -> tuple[float, float, float | None]:
    """Look up patch-relative replay fractions for a given stop.

    Returns ``(frac_target, frac_nontarget, residual_nontarget)``
    where the third element is ``None`` if no residual table was
    provided. Strict NaN propagation: a stop missing from the
    per-stop tables, or a target_patch outside ``{1, 2, 3}``,
    yields all NaN.
    """
    if (
        sid not in per_stop_per_patch.index
        or pd.isna(target_patch)
        or int(target_patch) not in (1, 2, 3)
    ):
        return (
            float("nan"),
            float("nan"),
            (None if per_stop_per_patch_residual is None else float("nan")),
        )
    tp = int(target_patch)
    sp = per_stop_per_patch.loc[sid]
    frac_target = float(sp[tp])
    non_targets = [p for p in (1, 2, 3) if p != tp]
    frac_nontarget = float(sp[non_targets].sum(skipna=False))
    residual_nt: float | None = None
    if per_stop_per_patch_residual is not None:
        if sid in per_stop_per_patch_residual.index:
            spr = per_stop_per_patch_residual.loc[sid]
            residual_nt = float(spr[non_targets].sum(skipna=False))
        else:
            residual_nt = float("nan")
    return frac_target, frac_nontarget, residual_nt


def _detect_first_switches(stop_table: pd.DataFrame) -> pd.DataFrame | None:
    """Per-session first-switch rows: the stops that start a run of switches.

    A stop starts a run when it is a switch (``is_stay == 0``), the trip
    that brought the animal to it stayed in its patch
    (``did_last_completed_trip_switch_patch == 0``), and no earlier stop
    after that same trip (same session and ``completed_trial_id``) was
    already a switch. Returns ``None`` when no rows have ``is_stay``
    defined.

    Whether the run started here is read from the stop's own arriving
    trip, not from the ``is_stay`` of the stop before it. That neighbour's
    ``is_stay`` is NaN for reasons that say nothing about switching: the
    stop was in transit or mixed (no current patch), before the first trial
    (no next well yet), or wholly in excluded time. When the stop before
    is a known at-well stop one trip earlier, the two readings agree
    exactly. A stop whose own ``is_stay`` or arriving-trip record is NaN
    starts no episode; that includes a session's first stops before any
    trip has completed.

    Stops after the same trip share its departure and so its ``is_stay``;
    only the first switch among them starts the episode. When an earlier
    stop of that visit is NaN (in excluded time, say), the episode starts
    at the first switch stop that is known.

    Sorting by ``[animal, session_date, epoch, stop_idx]`` is load-bearing:
    "earlier stop after the same trip" is read in chronological stop order,
    not caller row order. ``completed_trial_id`` restarts in every session,
    so the visit is keyed by session as well.
    """
    if not stop_table["is_stay"].notna().any():
        return None
    session = ["animal", "session_date", "epoch"]
    ordered = stop_table.sort_values([*session, "stop_idx"]).reset_index(drop=True)
    is_switch = ordered["is_stay"].eq(0)
    arrived_by_stay = ordered["did_last_completed_trip_switch_patch"].eq(0)
    visit = [ordered[c] for c in [*session, "completed_trial_id"]]
    switches_before = (
        is_switch.astype(int).groupby(visit, dropna=False).cumsum() - is_switch
    )
    return ordered[is_switch & arrived_by_stay & switches_before.eq(0)]


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
    stops. A run starts at a switch stop reached by a trip that stayed in
    its patch (see ``_detect_first_switches``). For each stop with replay,
    computes patch-level replay content: fraction on the current patch vs
    non-current patches.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Canonical stop table (includes zero-event stops), with
        ``is_stay``, ``did_last_completed_trip_switch_patch`` and
        ``completed_trial_id``.
    event_table : pd.DataFrame
        Canonical event table.
    event_arm_table : pd.DataFrame
        Canonical event-arm table (9 rows per event).
    content_fit : dict, optional
        Fitted content model for residualization. If provided along
        with *content_predictors*, adds ``replay_fraction_residual_noncurrent``.
    content_predictors : list[str], optional
        Predictors used in *content_fit*.
    max_lag : int
        Maximum number of stops before the switch to include.

    Returns
    -------
    pd.DataFrame
        One row per (episode, lag) for stops with replay. Columns:

        - ``episode_id`` — unique switching episode identifier
        - ``animal``, ``session_date``, ``epoch`` — session identifiers
        - ``switch_stop_idx`` — stop_idx of the switch
        - ``stop_idx``, ``stop_id`` — the replay stop
        - ``lag`` — stops before switch (negative; -1 = immediately before)
        - ``current_patch`` — patch the animal was in during the run
        - ``replay_fraction_current_patch`` — fraction of replay on current patch
        - ``replay_fraction_noncurrent`` — fraction on non-current patches
        - ``n_selected_events`` — events in this stop
        - ``replay_fraction_residual_noncurrent`` — (if content model provided)
    """
    st = stop_table.copy()

    first_switches = _detect_first_switches(st)
    if first_switches is None:
        return pd.DataFrame()

    if len(first_switches) == 0:
        return pd.DataFrame()

    eat = event_arm_table
    if content_fit is not None and content_predictors is not None:
        eat = predict_arm_fractions(eat, content_fit, content_predictors)

    # Per-stop fraction-on-each-patch table. Looking up by the
    # SWITCH's current_patch is what makes this switch-relative —
    # the previous form pre-aggregated using each replay stop's
    # OWN current_patch via ``stop_patch_replay``, then stamped the
    # switch's ``current_patch`` onto the lag rows, silently
    # re-labeling replay that landed on the previous patch as
    # "replay on the switch patch".
    per_stop_per_patch = _stop_replay_per_patch(
        eat, event_table, fraction_col=fraction_col
    )
    per_stop_per_patch_residual = (
        _stop_replay_per_patch(eat, event_table, fraction_col="residual_frac")
        if "residual_frac" in eat.columns
        else None
    )

    rows = []
    for ep_i, (_idx, sw_row) in enumerate(first_switches.iterrows()):
        sw_stop_idx = sw_row["stop_idx"]
        current_patch = sw_row["current_patch"]
        animal = sw_row["animal"]
        session_date = sw_row["session_date"]
        epoch = sw_row["epoch"]

        # Restrict to the switch's own session — stop_idx is only
        # session-local, so without this filter a switch in (animal A,
        # session_date D, epoch E) could match preceding stops from any other
        # session that happens to have a stop_idx within the lag
        # window. Lines 116-118 then write the switch's session keys
        # onto every row, masking the cross-session contamination.
        preceding = st[
            (st["animal"] == animal)
            & (st["session_date"] == session_date)
            & (st["epoch"] == epoch)
            & (st["stop_idx"] < sw_stop_idx)
            & (st["stop_idx"] >= sw_stop_idx - max_lag)
            & st["has_replay"]
        ]

        for _, ps_row in preceding.iterrows():
            lag = int(ps_row["stop_idx"] - sw_stop_idx)
            sid = ps_row["stop_id"]

            if sid not in per_stop_per_patch.index:
                continue

            frac_cp, frac_nc, residual_nc = _patch_relative_fracs(
                sid, current_patch, per_stop_per_patch, per_stop_per_patch_residual
            )
            row = {
                "episode_id": ep_i,
                "animal": animal,
                "session_date": session_date,
                "epoch": epoch,
                "switch_stop_idx": int(sw_stop_idx),
                "stop_idx": int(ps_row["stop_idx"]),
                "stop_id": sid,
                "lag": lag,
                "current_patch": current_patch,
                "replay_fraction_current_patch": frac_cp,
                "replay_fraction_noncurrent": frac_nc,
                "n_selected_events": int(ps_row["n_selected_events"]),
            }
            if residual_nc is not None:
                row["replay_fraction_residual_noncurrent"] = residual_nc
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

    For each switching episode, picks a random stay anchor in the
    same session and emits one row per stay stop with replay in the
    ``[anchor - max_lag, anchor)`` window. The anchor and every
    contributing preceding stop must satisfy ``is_stay == 1``, so the
    control window contains no switch dynamics. Returns the same
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

    # Per-stop fraction-on-each-patch table, looked up by the SWITCH's
    # current_patch so the control fractions are measured against the
    # same reference as the switch-aligned table (matched-control
    # contract). Pre-fix used ``stop_patch_replay``, which aggregates
    # against each individual control stop's OWN current_patch, so
    # the matched-control comparison was actually against a moving
    # reference frame.
    per_stop_per_patch = _stop_replay_per_patch(
        eat, event_table, fraction_col=fraction_col
    )
    per_stop_per_patch_residual = (
        _stop_replay_per_patch(eat, event_table, fraction_col="residual_frac")
        if "residual_frac" in eat.columns
        else None
    )

    if len(switch_episodes) == 0:
        return pd.DataFrame()
    ep_meta = switch_episodes.groupby("episode_id").first()[
        ["animal", "session_date", "epoch", "current_patch", "switch_stop_idx"]
    ]

    rows = []
    for ep_id, meta in ep_meta.iterrows():
        animal = meta["animal"]
        session_date = meta["session_date"]
        epoch = meta["epoch"]
        patch = meta["current_patch"]
        sw_idx = meta["switch_stop_idx"]

        session_stays = st[
            (st["animal"] == animal)
            & (st["session_date"] == session_date)
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
        # Also restrict to ``is_stay == 1``: the matched-control
        # contract is "stay stops aligned to a stay anchor", and the
        # anchor was already drawn from ``session_stays``. Without
        # this filter, a switch stop sitting in [anchor - max_lag,
        # anchor) would be re-labeled as a control row at its lag,
        # contaminating the switch-vs-control comparison with the
        # very switch dynamics it's meant to baseline against.
        preceding = st[
            (st["animal"] == animal)
            & (st["session_date"] == session_date)
            & (st["epoch"] == epoch)
            & (st["is_stay"] == 1)
            & (st["stop_idx"] < anchor_stop_idx)
            & (st["stop_idx"] >= anchor_stop_idx - max_lag)
            & st["has_replay"]
        ]

        for _, ps_row in preceding.iterrows():
            lag = int(ps_row["stop_idx"] - anchor_stop_idx)
            sid = ps_row["stop_id"]
            if sid not in per_stop_per_patch.index:
                continue
            frac_cp, frac_nc, residual_nc = _patch_relative_fracs(
                sid, patch, per_stop_per_patch, per_stop_per_patch_residual
            )
            row = {
                "episode_id": ep_id,
                "animal": animal,
                "session_date": session_date,
                "epoch": epoch,
                "switch_stop_idx": int(anchor_stop_idx),
                "stop_idx": int(ps_row["stop_idx"]),
                "stop_id": sid,
                "lag": lag,
                "current_patch": patch,
                "replay_fraction_current_patch": frac_cp,
                "replay_fraction_noncurrent": frac_nc,
                "n_selected_events": int(ps_row["n_selected_events"]),
            }
            if residual_nc is not None:
                row["replay_fraction_residual_noncurrent"] = residual_nc
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
    ``replay_fraction_current_patch``, ``replay_fraction_noncurrent``, and
    optionally ``replay_fraction_residual_noncurrent`` if
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
    eat["_patch"] = eat["candidate_arm"].map(TRACK_SEGMENT_TO_PATCH)

    # current_patch is already in event_arm_table; fall back to merge.
    if "current_patch" not in eat.columns:
        evt_patch = event_table[["event_id", "current_patch"]].drop_duplicates()
        eat = eat.merge(evt_patch, on="event_id", how="left")

    # Flag arms in the current patch. ``NaN == anything`` is False
    # in pandas/numpy, so a missing ``current_patch`` would silently
    # mark every arm as "non-current" and shovel the whole event's
    # replay fraction into ``replay_fraction_noncurrent`` while reporting
    # ``replay_fraction_current_patch == 0.0``. The current-vs-non-current
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
            "replay_fraction_current_patch": (
                eat.groupby(["stop_id", "event_id"])["_frac_cp"].apply(strict_sum)
            ),
            "replay_fraction_noncurrent": (
                eat.groupby(["stop_id", "event_id"])["_frac_nc"].apply(strict_sum)
            ),
        }
    )

    # Average across events within each stop, NaN-aware.
    result = event_agg.groupby("stop_id")[
        ["replay_fraction_current_patch", "replay_fraction_noncurrent"]
    ].agg(strict_mean)

    # Residualized version if available
    if "residual_frac" in eat.columns:
        # Same NaN-aware multiplication: missing ``current_patch``
        # routes the residual mass to NaN, not to "non-current".
        eat["_resid_nc"] = eat["residual_frac"].to_numpy() * (1.0 - is_cp_float)
        event_resid = (
            eat.groupby(["stop_id", "event_id"])["_resid_nc"]
            .apply(strict_sum)
            .rename("replay_fraction_residual_noncurrent")
            .to_frame()
        )
        resid_agg = (
            event_resid.groupby("stop_id")["replay_fraction_residual_noncurrent"]
            .apply(strict_mean)
            .rename("replay_fraction_residual_noncurrent")
            .to_frame()
        )
        result = result.join(resid_agg)

    return result
