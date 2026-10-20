"""Canonical arm- and patch-grain builders: one row per (event, arm) / (event, patch)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.events.immobility import (
    require_canonical_speed_threshold,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    TRACK_SEGMENT_TO_PATCH,
    WELL_ARMS,
)
from continuum_swr_replay.tables.covariate_registry import stamp_covariate_schema

from ._helpers import (
    align_count_matrix as _align_count_matrix,
)
from ._helpers import (
    align_posterior_to_dot as _align_posterior_to_dot,
)
from ._helpers import (
    assert_event_label_alignment as _assert_event_label_alignment,
)
from ._helpers import (
    assert_single_session_alignment as _assert_single_session_alignment,
)
from ._tables_common import (
    _EVENT_ARM_TABLE_BUILD_COLUMNS,
    EVENT_ARM_STANDARDIZED_COLS,
    _in_event_mask,
    _prepare_dot,
)
from .covariates.candidate_arm import (
    attach_candidate_arm_covariates,
    broadcast_candidate_arms,
)
from .summaries import (
    arm_count_hard,
    arm_count_soft,
    arm_fraction_from_event_total,
    arm_posterior_mass,
    posterior_to_arm_marginal,
)
from .table_contracts import (
    EVENT_BIN_TOTAL_COLUMNS,
    event_bin_total_column,
    validate_event_arm_table,
    validate_event_patch_table,
)


def build_event_arm_table(
    event_table: pd.DataFrame,
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    *,
    conditional_posterior=None,
    contingency: pd.DataFrame | None = None,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> pd.DataFrame:
    """One row per (event, arm) — nine rows per event.

    Assumes ``event_table`` was produced by :func:`build_event_table`
    on the same ``dot`` / ``trials`` / ``track_graph``. Broadcasts
    event-level fields (including ``stop_id``, context flags, and
    event-level posterior columns) across arms and augments each
    (event, arm) row with the three arm-count responses, the binary
    arm indicators, spatial and reward predictors, and the interaction
    terms.

    ``task_reward_probability_candidate_arm`` and
    ``reward_belief_mean_candidate_arm_preoutcome`` are populated from the
    qvals contingency block when one is supplied, and emitted as NaN otherwise.

    Parameters
    ----------
    event_table : pd.DataFrame
        Output of :func:`build_event_table`.
    dot, env, trials, port_experiences, track_graph
        Same as :func:`build_event_table`. ``env`` is needed for
        ``map_estimate_to_arm`` and, when ``conditional_posterior`` is
        passed, for the ``place_bin_centers_nodes_df_`` edge_id vector.
    conditional_posterior : xr.DataArray, optional
        When supplied, the ``arm_posterior_mass`` column is populated
        via :func:`summaries.arm_posterior_mass`; otherwise it is NaN.
    speed_threshold, nonlocal_threshold : float
        Kept in lockstep with :func:`build_event_table`. Any drift
        would invalidate the ``event_id`` join.

    Returns
    -------
    pd.DataFrame
        Columns: see :data:`_EVENT_ARM_TABLE_COLUMNS`. Empty (correct
        schema) if ``event_table`` is empty.

    Raises
    ------
    ValueError
        If `speed_threshold` is not the canonical
        ``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC``.
    """
    require_canonical_speed_threshold(speed_threshold, "build_event_arm_table")
    canonical_column_order = list(_EVENT_ARM_TABLE_BUILD_COLUMNS)
    if len(event_table) == 0:
        return stamp_covariate_schema(
            pd.DataFrame(columns=canonical_column_order), table="event_arm"
        )

    n_arms = len(ALL_ARMS)
    n_events = len(event_table)

    dot = _prepare_dot(dot, env)
    in_event = _in_event_mask(dot, speed_threshold, nonlocal_threshold)
    nl = dot.loc[in_event]

    # Arm-count responses via the shared helpers. Junction bins (NaN
    # ``decoded_arm``) are excluded from the MAP-arm counts — they can
    # still contribute to the posterior-mass response via
    # :func:`posterior_to_arm_marginal`, which zeroes junction bins
    # and renormalizes per timebin.
    arm_series = nl["decoded_arm"]
    real_mask = arm_series.isin(ALL_ARMS).to_numpy()
    arm_map_real = arm_series.to_numpy()[real_mask].astype(np.int64)
    event_id_real = nl["non_local_event"].to_numpy()[real_mask]
    nonlocal_prob_real = nl["nonlocal_prob"].to_numpy()[real_mask]
    nonlocal_prob_all = nl["nonlocal_prob"].to_numpy()
    event_id_all = nl["non_local_event"].to_numpy()

    _assert_single_session_alignment(
        dot,
        event_table,
        source_name="NonLocal dot",
        target_session_cols=("animal", "session_date", "epoch"),
    )
    _assert_event_label_alignment(
        event_id_all,
        event_table["non_local_event"].to_numpy(),
        source_name="NonLocal qualified bins",
    )

    ids_real, counts_hard = arm_count_hard(
        nonlocal_prob_real, arm_map_real, event_id_real, n_arms=n_arms, threshold=0.0
    )
    _, counts_soft = arm_count_soft(
        nonlocal_prob_real, arm_map_real, event_id_real, n_arms=n_arms
    )

    # Align count matrices with event_table's event order. ``ids_real``
    # comes from np.unique (sorted), so missing events (those whose
    # every in-event timebin was on a junction bin — extremely rare)
    # need zero rows filled in.
    ev_ids = event_table["non_local_event"].to_numpy().astype(np.int64)
    hard_aligned = _align_count_matrix(counts_hard, ids_real, ev_ids, n_arms)
    soft_aligned = _align_count_matrix(counts_soft, ids_real, ev_ids, n_arms)

    if conditional_posterior is not None:
        edge_id = np.asarray(env.place_bin_centers_nodes_df_["edge_id"])
        post_vals = _align_posterior_to_dot(conditional_posterior, dot)
        post_in_event = post_vals[in_event.to_numpy()]
        arm_posterior = posterior_to_arm_marginal(post_in_event, edge_id, n_arms=n_arms)
        ids_all, mass = arm_posterior_mass(
            nonlocal_prob_all, arm_posterior, event_id_all
        )
        mass_aligned = _align_count_matrix(mass, ids_all, ev_ids, n_arms)
    else:
        mass_aligned = np.full((n_events, n_arms), np.nan)

    # Broadcast event-level fields to 9-per-event, then attach per-arm
    # columns. ``np.repeat`` on the numpy representation avoids copying
    # the whole frame's index repeatedly.
    long = broadcast_candidate_arms(event_table)
    long["arm_count_hard"] = hard_aligned.flatten()
    long["arm_count_soft"] = soft_aligned.flatten()
    long["arm_posterior_mass"] = mass_aligned.flatten()

    # arm_fraction: numerator and denominator must cover the same set
    # of timebins. ``arm_count_hard`` excludes NaN-``decoded_arm``
    # timebins (counted as ``n_junction_bins``); the denominator must
    # exclude them too, or the fraction deflates below 1 whenever any
    # MAP failed to attribute to a real arm. NaN when no real-arm
    # bins exist (denominator <= 0).
    n_real_bins_long = (
        long["n_nonlocal_bins"].to_numpy() - long["n_undecoded_bins"].to_numpy()
    )
    long["arm_fraction"] = arm_fraction_from_event_total(
        long["arm_count_hard"].to_numpy(),
        n_real_bins_long,
    )

    # arm_fraction_posterior: posterior-mass-based fraction. Uses
    # ``nonlocal_probability_sum`` is the exact event-weight denominator.
    nws_long = long["nonlocal_probability_sum"].to_numpy()
    apm = long["arm_posterior_mass"].to_numpy()
    long["arm_fraction_posterior"] = arm_fraction_from_event_total(
        apm, nws_long, denominator_floor=1e-10
    )

    long = attach_candidate_arm_covariates(
        long,
        event_table=event_table,
        port_experiences=port_experiences,
        track_graph=track_graph,
        contingency_stop_id=long["stop_id"].to_numpy(),
        contingency=contingency,
        include_belief_variance=True,
    )

    # Standardized (<col>_z) placeholders: the real pooled-GLOBAL z is computed
    # post-concat in prepare_data.run_prepare_data (a per-session build cannot know the global
    # mean/SD). Fill NaN here so the canonical column order carries the names.
    for c in EVENT_ARM_STANDARDIZED_COLS:
        long[f"{c}_z"] = np.nan

    validate_event_arm_table(long, event_table=event_table)
    canonical = stamp_covariate_schema(long, table="event_arm")
    return canonical[canonical_column_order]


def build_event_patch_table(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse event-arm table to event-patch level (3 rows per event).

    Sums ``arm_count_hard`` across arms within each patch, normalizes it
    to ``patch_fraction`` over the event's real-arm bins, and aggregates
    per-arm predictors to patch level. The result is ready for
    patch-level conditional Poisson fitting (``MODEL_PATCH``).

    Parameters
    ----------
    event_table : pd.DataFrame
        Canonical event table (for ``last_rewarded_arm`` → patch mapping and
        the ``patch_fraction`` denominator). The denominator is the event's
        bin total minus ``n_undecoded_bins``. The bin total is
        ``n_nonlocal_bins`` for non-local tables and ``n_timebins`` for
        ContFrag tables (``n_nonlocal_bins`` wins when both exist). Either way
        it is the same denominator as that table's ``arm_fraction``.
    event_arm_table : pd.DataFrame
        Canonical event-arm table (9 rows per event).

    Returns
    -------
    pd.DataFrame
        3 rows per event (one per patch), with columns:
        ``event_id``, ``patch``, ``patch_count_hard``,
        ``patch_fraction``, ``is_current_patch``, ``is_prev_patch``,
        ``is_last_rewarded_patch``, ``is_last_rewarded_noncurrent_patch``,
        canonical patch-level reward/value/recency aggregates, and canonical
        SWR-by-patch-role products.
    """
    # All groupbys / merges below key on ``event_id`` only. If a
    # caller passes session-local IDs (event_id=1 in two sessions),
    # the per-(event_id, patch) groupby silently fuses the two
    # sessions' arm counts and the merges to ``event_table`` then
    # produce a Cartesian product (a tiny two-session repro produced
    # 48 rows instead of the expected 6 session-patch rows). Mirror
    # the precondition already enforced in ``predict_arm_fractions``,
    # ``build_choice_dataset``, and ``stop_patch_replay``.
    from .choice_prediction import _require_session_unique_ids

    _require_session_unique_ids(
        event_arm_table, "event_id", helper_name="build_event_patch_table"
    )
    _require_session_unique_ids(
        event_table, "event_id", helper_name="build_event_patch_table"
    )
    validate_event_arm_table(event_arm_table, event_table=event_table)

    eat = event_arm_table.copy()
    eat["patch"] = eat["candidate_arm"].map(TRACK_SEGMENT_TO_PATCH)

    # ``cumulative_reward`` is sometimes NaN (no reward history known
    # yet for a session). The default pandas sum has ``skipna=True``,
    # which collapses an all-NaN group to ``0.0`` — turning unknown
    # reward history into a real zero. ``min_count=1`` preserves NaN
    # when no non-null observations exist in the group, which is what
    # the downstream P3 patch model needs to drop those rows from its
    # complete-case fit instead of conditioning on a fabricated zero.
    agg_spec: dict = {
        "patch_count_hard": ("arm_count_hard", "sum"),
        "is_current_patch": ("is_same_patch_as_current", "max"),
        "reward_count_candidate_arm_last10_known_trials_patch": (
            "reward_count_candidate_arm_last10_known_trials",
            lambda s: s.sum(min_count=1),
        ),
    }
    # Value aggregates: patch mean over the patch's LEAF arms only (trunk arms
    # 6,7,8 are not reward sites) per the aggregator rule. Mask trunk to NaN so
    # the skipna mean averages the leaves; a patch whose leaves are all NaN
    # (unmapped stop / never-visited) stays NaN and drops at fit time.
    leaf = eat["candidate_arm"].isin(WELL_ARMS)
    # Recency is defined for reachable leaf arms; trunks carry structural zero.
    # Mask trunks before taking a patch minimum so zero cannot make every patch
    # appear maximally recent. A patch is novel only if both of its leaf arms
    # are novel, hence the minimum over leaf novelty indicators.
    for col, out_col in (
        (
            "reward_fraction_candidate_arm_last10_visits",
            "reward_fraction_candidate_arm_last10_visits_patch",
        ),
        (
            "task_reward_probability_candidate_arm",
            "task_reward_probability_candidate_arm_patch",
        ),
        (
            "reward_belief_mean_candidate_arm_preoutcome",
            "reward_belief_mean_candidate_arm_preoutcome_patch",
        ),
    ):
        if col in eat.columns:
            eat[f"_leaf_{col}"] = eat[col].where(leaf)
            agg_spec[out_col] = (f"_leaf_{col}", "mean")
    patch_eat = eat.groupby(["event_id", "patch"]).agg(**agg_spec).reset_index()

    # patch_fraction: the event's share of decoded content allocated to this
    # patch. Same denominator as ``arm_fraction`` in build_event_arm_table --
    # the in-event timebins whose MAP landed on a real arm -- so a patch
    # fraction is exactly the sum of its three arms' fractions and an event's
    # three patch fractions sum to one. NaN when the event has no real-arm bin.
    # The bin total is ``n_nonlocal_bins`` on a non-local event table and
    # ``n_timebins`` on a ContFrag event table; either way it is the total that
    # table's own ``arm_fraction`` divides by. Both bin counts are event-grain,
    # so the denominator is carried from event_table rather than reconstructed
    # from the response's own total.
    total_column = event_bin_total_column(event_table)
    if total_column is None or "n_undecoded_bins" not in event_table.columns:
        accepted = " or ".join(f"'{name}'" for name in EVENT_BIN_TOTAL_COLUMNS)
        raise ValueError(
            "build_event_patch_table requires an event_table bin-total column "
            f"({accepted}) and 'n_undecoded_bins'; patch_fraction has no "
            "denominator without them"
        )
    real_bins_per_event = (
        pd.to_numeric(event_table[total_column], errors="coerce")
        - pd.to_numeric(event_table["n_undecoded_bins"], errors="coerce")
    ).set_axis(event_table["event_id"])
    patch_eat.insert(
        patch_eat.columns.get_loc("patch_count_hard") + 1,
        "patch_fraction",
        arm_fraction_from_event_total(
            patch_eat["patch_count_hard"].to_numpy(dtype=float),
            real_bins_per_event.reindex(patch_eat["event_id"]).to_numpy(dtype=float),
        ),
    )
    leaf_history = (
        eat.loc[leaf]
        .groupby(["event_id", "patch"])
        .agg(
            visit_recency_trials_patch=(
                "visit_recency_trials",
                lambda values: values.min(skipna=False),
            ),
            is_patch_never_visited=(
                "is_candidate_arm_never_visited",
                lambda values: values.min(skipna=False),
            ),
        )
        .reset_index()
    )
    patch_eat = patch_eat.merge(
        leaf_history,
        on=["event_id", "patch"],
        how="left",
        validate="one_to_one",
    )

    # is_last_rewarded_patch: is the last-rewarded arm in this patch?
    last_rew = event_table[["event_id", "last_rewarded_arm"]].copy()
    last_rew["last_rewarded_patch"] = last_rew["last_rewarded_arm"].map(
        lambda a: float(TRACK_SEGMENT_TO_PATCH.get(int(a), np.nan))
        if pd.notna(a)
        else np.nan
    )
    patch_eat = patch_eat.merge(
        last_rew[["event_id", "last_rewarded_patch"]],
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    patch_eat["is_last_rewarded_patch"] = (
        patch_eat["patch"] == patch_eat["last_rewarded_patch"]
    ).astype(int)
    patch_eat = patch_eat.drop(columns=["last_rewarded_patch"])

    # is_last_rewarded_noncurrent_patch: the last-rewarded patch, but
    # only where it is NOT the patch the animal currently occupies.
    # ``is_last_rewarded_patch`` overlaps ``is_current_patch`` heavily
    # (the animal is usually rewarded in its current patch); the patch
    # hierarchy uses this disjoint-support residual at P2/P3, mirroring
    # ``is_last_rewarded_noncurrent_arm``. See model_specs.py.
    # NaN where is_current_patch is unobserved -- casting to int here
    # would turn that NaN into a sentinel integer.
    patch_eat["is_last_rewarded_noncurrent_patch"] = patch_eat[
        "is_last_rewarded_patch"
    ].to_numpy() * (1 - patch_eat["is_current_patch"].to_numpy())

    # is_prev_patch: is this the patch the animal just came from?
    prev_p = event_table[["event_id", "prev_patch"]].copy()
    patch_eat = patch_eat.merge(
        prev_p,
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    patch_eat["is_prev_patch"] = (patch_eat["patch"] == patch_eat["prev_patch"]).astype(
        int
    )
    patch_eat = patch_eat.drop(columns=["prev_patch"])

    # SWR × patch interactions (P4, PS3). is_swr is event-level — bring
    # it in from event_table once and reuse for the three products. The
    # column is RETAINED in the output so the patch-level fitter's
    # ``--swr-subset`` filter (which gates on an ``is_swr`` column) works
    # for ``--hierarchy patch`` as it does for arm fits.
    if "is_swr" not in event_table.columns:
        raise ValueError(
            "build_event_patch_table requires canonical event_table column "
            "'is_swr'; refusing to fabricate non-SWR labels"
        )
    swr_col = event_table[["event_id", "is_swr"]].copy()
    patch_eat = patch_eat.merge(
        swr_col,
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    # Every patch-table event_id comes from event_table, so the merge
    # must be total. Fail loudly rather than fabricate is_swr=False for
    # an unmatched id (which would silently mislabel SWR status).
    n_unmatched = int(patch_eat["is_swr"].isna().sum())
    if n_unmatched > 0:
        raise ValueError(
            f"build_event_patch_table: {n_unmatched} patch rows have an "
            "event_id absent from event_table; is_swr cannot be assigned."
        )
    is_swr_i = patch_eat["is_swr"].astype(int).to_numpy()
    patch_eat["swr_x_is_current_patch"] = (
        is_swr_i * patch_eat["is_current_patch"].to_numpy()
    )
    patch_eat["swr_x_is_last_rewarded_noncurrent_patch"] = (
        is_swr_i * patch_eat["is_last_rewarded_noncurrent_patch"].to_numpy()
    )
    patch_eat["swr_x_is_prev_patch"] = is_swr_i * patch_eat["is_prev_patch"].to_numpy()

    # Broadcast event-level identity/location so patch analyses can apply the
    # same explicit at-well/in-transit/mixed cohort filters as arm analyses.
    # Session keys are also required by the patch fitter's per-session groups.
    event_context_cols = [
        column
        for column in (
            "animal",
            "session_date",
            "epoch",
            "occupancy_state",
            "at_well",
            "occupancy_well_id",
            "occupancy_well_arm",
            "modal_observed_track_segment_id",
            "nearest_well_distance_cm_min",
            "nearest_well_distance_cm_mean",
            "nearest_well_distance_cm_max",
        )
        if column in event_table.columns
    ]
    if event_context_cols:
        patch_eat = patch_eat.merge(
            event_table[["event_id", *event_context_cols]],
            on="event_id",
            how="left",
            validate="many_to_one",
        )

    validate_event_patch_table(patch_eat, event_table=event_table)
    return stamp_covariate_schema(patch_eat, table="event_patch")
