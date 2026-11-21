"""Small canonical content tables and simulated content sessions.

The content runner's tests and the tests of the package functions it calls
build their inputs here, so a function and its runner are tested on the same
tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    stamp_swr_source,
)
from continuum_swr_replay.analysis.replay_models.content.simulation import (
    simulate_dirichlet_multinomial_counts,
    simulate_with_event_covariates,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (
    MODEL_HIERARCHY,
    MODEL_PATCH,
    default_content_model_names,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    TRACK_SEGMENT_TO_PATCH,
    WELL_ARMS,
)
from continuum_swr_replay.tables.covariate_registry import stamp_covariate_schema


def canonical_arm_table(
    *,
    n_events: int = 4,
    fraction_scale: float = 1.0,
    undecoded_event: str | None = None,
    partial_event: str | None = None,
    crossing_event: str | None = None,
) -> pd.DataFrame:
    """A minimal table satisfying the canonical arm contract.

    ``fraction_scale`` rescales every event's fractions (1.0 is the contract).
    ``undecoded_event`` leaves one event's fraction missing on every candidate,
    as an event with no real-arm bins; ``partial_event`` leaves it missing on
    one candidate only, which the contract forbids. ``crossing_event`` starts
    inside its visit and crosses the final poke-out: ``mixed``,
    ``spans_visit_end``, no whole-window well, and the visit's well as its
    current arm.
    """
    rows = []
    for index in range(n_events):
        event = f"e{index}"
        current = index % len(WELL_ARMS)
        other = (current + 1) % len(WELL_ARMS)
        counts = np.zeros(len(ALL_ARMS))
        counts[current] = 6.0
        counts[other] = 2.0
        fractions = counts / counts.sum() * fraction_scale
        for arm in ALL_ARMS:
            fraction = fractions[arm]
            if event == undecoded_event or (
                event == partial_event and arm == len(ALL_ARMS) - 1
            ):
                fraction = np.nan
            rows.append(
                {
                    "animal": "chimi",
                    "session_date": "20200212",
                    "epoch": "02_r1",
                    "event_id": event,
                    "candidate_arm": arm,
                    "arm_count_hard": counts[arm],
                    "arm_fraction": fraction,
                    **_location(current, crossing=event == crossing_event),
                    "occupancy_well_arm": current,
                    "is_current_arm": float(arm == current),
                    "visit_recency_trials_event_leaf_centered": float(arm),
                    "is_candidate_arm_never_visited": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _location(well: int, *, crossing: bool) -> dict[str, object]:
    """The whole-window location of an event at ``well``, or crossing the
    final poke-out of a visit there."""
    if crossing:
        return {
            "occupancy_state": "mixed",
            "visit_context": "spans_visit_end",
            "at_well": False,
            "occupancy_well_id": np.nan,
        }
    return {
        "occupancy_state": "at_well",
        "visit_context": "in_visit",
        "at_well": True,
        "occupancy_well_id": well,
    }


def canonical_patch_table(
    *, n_events: int = 4, crossing_event: str | None = None
) -> pd.DataFrame:
    """A minimal table satisfying the canonical patch contract.

    ``crossing_event`` is as in :func:`canonical_arm_table`.
    """
    patches = sorted(set(TRACK_SEGMENT_TO_PATCH.values()))
    rows = []
    for index in range(n_events):
        current_arm = index % len(WELL_ARMS)
        current_patch = TRACK_SEGMENT_TO_PATCH[current_arm]
        counts = dict.fromkeys(patches, 0.0)
        counts[current_patch] = 5.0
        counts[patches[(patches.index(current_patch) + 1) % len(patches)]] = 3.0
        total = sum(counts.values())
        for patch in patches:
            rows.append(
                {
                    "animal": "chimi",
                    "session_date": "20200212",
                    "epoch": "02_r1",
                    "event_id": f"e{index}",
                    "patch": patch,
                    "patch_count_hard": counts[patch],
                    "patch_fraction": counts[patch] / total,
                    **_location(current_arm, crossing=f"e{index}" == crossing_event),
                    "occupancy_well_arm": current_arm,
                    "is_current_patch": float(patch == current_patch),
                    "visit_recency_trials_patch": float(patch),
                    "is_patch_never_visited": 0.0,
                }
            )
    return pd.DataFrame(rows)


def default_ladder_arm_table(n_events: int = 2) -> pd.DataFrame:
    """Small canonical arm table with every default-ladder predictor."""
    arms = np.tile(np.arange(9), n_events)
    event_ids = np.repeat(np.arange(n_events), 9)
    frame = pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "event_id": event_ids,
            "candidate_arm": arms,
            "arm_count_hard": (arms == (event_ids % 9)).astype(float),
            # One decoded bin per event, so the allocation fraction is the
            # count: each event puts all of its mass on one arm.
            "arm_fraction": (arms == (event_ids % 9)).astype(float),
            "occupancy_state": "at_well",
            "visit_context": "in_visit",
            "at_well": True,
            "occupancy_well_id": 0.0,
            "occupancy_well_arm": 0.0,
            "is_current_arm": (arms == 0).astype(float),
            "visit_recency_trials_event_leaf_centered": 0.0,
            "is_candidate_arm_never_visited": 0.0,
        }
    )
    requested = default_content_model_names(MODEL_HIERARCHY)
    for model_name in requested:
        for predictor in MODEL_HIERARCHY[model_name].predictors:
            if predictor.startswith("arm_fe_") or predictor in frame:
                continue
            frame[predictor] = 0.0
    return stamp_swr_source(stamp_covariate_schema(frame, table="event_arm"))


def default_ladder_patch_table(n_events: int = 2) -> pd.DataFrame:
    """Small canonical patch table with every default-ladder predictor."""
    patches = np.tile([1, 2, 3], n_events)
    event_ids = np.repeat(np.arange(n_events), 3)
    frame = pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "event_id": event_ids,
            "patch": patches,
            "patch_count_hard": (patches == ((event_ids % 3) + 1)).astype(float),
            "patch_fraction": (patches == ((event_ids % 3) + 1)).astype(float),
            "occupancy_state": "at_well",
            "visit_context": "in_visit",
            "at_well": True,
            "occupancy_well_id": 0.0,
            "occupancy_well_arm": 0.0,
            "is_current_patch": (patches == 1).astype(float),
            "visit_recency_trials_patch": 0.0,
            "is_patch_never_visited": 0.0,
        }
    )
    requested = default_content_model_names(MODEL_PATCH)
    for model_name in requested:
        for predictor in MODEL_PATCH[model_name].predictors:
            if predictor in frame:
                continue
            frame[predictor] = 0.0
    return stamp_swr_source(stamp_covariate_schema(frame, table="event_patch"))


def simulated_session(response_col: str, seed: int = 5) -> pd.DataFrame:
    """One simulated session of 60 events, with both the counts and fractions."""
    df = simulate_with_event_covariates(
        {"is_current_arm": 0.7}, n_events=60, L_e=20, seed=seed
    )
    df["arm_fraction"] = df["arm_count_hard"] / df.groupby("event_id")[
        "arm_count_hard"
    ].transform("sum")
    df["animal"] = "chimi"
    df["session_date"] = "20200212"
    df["epoch"] = "02_r1"
    assert response_col in df.columns
    return df


def dm_session(
    n_events: int, *, seed: int, animal: str = "chimi", epoch: str = "02_r1"
) -> pd.DataFrame:
    """One session of Dirichlet-multinomial counts with globally unique event ids."""
    df = simulate_dirichlet_multinomial_counts(
        {"is_current_arm": 0.8},
        log_phi_intercept=1.2,
        n_events=n_events,
        L_e=20,
        seed=seed,
    )
    df["animal"] = animal
    df["session_date"] = "20200212"
    df["epoch"] = epoch
    # Event ids are globally unique; the simulator numbers each session from 0.
    df["event_id"] = [f"{epoch}-{index}" for index in df["event_id"]]
    # The simulator leaves its drawn allocations in attrs, which pandas compares
    # element-wise (and so raises on) when concatenating two sessions.
    df.attrs.clear()
    return df


def logit_session(epoch: str, seed: int) -> pd.DataFrame:
    """One session of 60 events with a current-arm and a previous-arm effect."""
    df = simulate_with_event_covariates(
        {"is_current_arm": 0.7, "is_prev_arm": -0.3}, n_events=60, L_e=20, seed=seed
    )
    df["arm_fraction"] = df["arm_count_hard"] / df.groupby("event_id")[
        "arm_count_hard"
    ].transform("sum")
    df["event_id"] = epoch + "_" + df["event_id"].astype(str)
    df["animal"] = "chimi"
    df["session_date"] = "20200212"
    df["epoch"] = epoch
    return df


def fail_logit_for(monkeypatch, epoch: str, n_predictors: int, error: Exception):
    """Make ConditionalLogit raise ``error`` for one session at one rung."""
    from continuum_swr_replay.analysis.replay_models import fitting

    real = fitting.ConditionalLogit

    def picky(y, X, groups):
        if X.shape[1] == n_predictors and str(groups[0]).startswith(epoch):
            raise error
        return real(y, X, groups=groups)

    monkeypatch.setattr(fitting, "ConditionalLogit", picky)


def dm_contrast_session(n_events: int = 60, *, epoch: str, seed: int) -> pd.DataFrame:
    """One session of nine-arm counts drawn at a known within-event concentration.

    SWR events are the more concentrated ones (``log_phi_is_swr = -0.8``).
    """
    df = simulate_dirichlet_multinomial_counts(
        {"is_current_arm": 0.8},
        log_phi_intercept=1.2,
        log_phi_is_swr=-0.8,
        n_events=n_events,
        L_e=20,
        seed=seed,
    )
    df["animal"] = "chimi"
    df["session_date"] = "20200212"
    df["epoch"] = epoch
    df["event_id"] = [f"{epoch}-{index}" for index in df["event_id"]]
    df["is_swr"] = df["is_swr"].astype(bool)
    df.attrs.clear()
    return df


def with_event_length(cohort: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Attach a z-scored log event length that is constant within each event.

    SWR events run longer, which is the reason the length adjustment exists.
    """
    rng = np.random.default_rng(seed)
    events = sorted(cohort["event_id"].unique())
    is_swr = (
        cohort.drop_duplicates("event_id").set_index("event_id")["is_swr"].astype(bool)
    )
    values = pd.Series(
        rng.normal(size=len(events)) + 0.6 * is_swr.loc[events].to_numpy(),
        index=events,
    )
    values = (values - values.mean()) / values.std(ddof=0)
    return cohort.assign(log_n_nonlocal_bins_z=cohort["event_id"].map(values))


#: The estimator comparison's simulated cohort. The planted concentration is
#: small enough that each event's timebins are strongly correlated, so an
#: estimator that treats them as independent understates its uncertainty.
COMPARISON_PHI_TRUE = 3.0
COMPARISON_LOG_PHI_IS_SWR_TRUE = -0.5
COMPARISON_N_EVENTS_PER_SESSION = 60
COMPARISON_BINS_PER_EVENT = 32
COMPARISON_BETA_TRUE = {
    "is_current_arm": 1.0,
    "is_prev_arm": -0.5,
    "reward_fraction_candidate_arm_last10_visits": 0.4,
}


def _comparison_session(
    animal: str, session_date: str, seed: int, offset: int
) -> pd.DataFrame:
    frame = simulate_dirichlet_multinomial_counts(
        COMPARISON_BETA_TRUE,
        log_phi_intercept=float(np.log(COMPARISON_PHI_TRUE)),
        log_phi_is_swr=COMPARISON_LOG_PHI_IS_SWR_TRUE,
        n_events=COMPARISON_N_EVENTS_PER_SESSION,
        L_e=COMPARISON_BINS_PER_EVENT,
        seed=seed,
    ).copy()
    # ``attrs["p_true"]`` is an array and pandas compares attrs element-wise on
    # concat; drop it rather than work around the comparison.
    frame.attrs = {}
    frame["event_id"] = frame["event_id"] + offset
    frame["animal"] = animal
    frame["session_date"] = session_date
    frame["epoch"] = "02_r1"
    return frame


def estimator_comparison_cohort() -> pd.DataFrame:
    """Two simulated sessions from two animals, both with SWR and non-SWR events."""
    return pd.concat(
        [
            _comparison_session("alpha", "20200101", seed=11, offset=0),
            _comparison_session("bravo", "20200102", seed=22, offset=10_000),
        ],
        ignore_index=True,
    )
