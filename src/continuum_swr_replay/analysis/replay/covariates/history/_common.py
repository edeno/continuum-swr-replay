"""Shared port-experience validation for the history covariates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import (
    WELL_PATCH_MAPPING,
)

_PORT_EXPERIENCE_COLUMNS = (
    "experience_id",
    "movement_trial_number",
    "is_initial_experience",
    "is_boundary_derived",
    "outcome_observed",
    "port",
    "patch",
    "arrival_time",
    "departure_time",
    "reward_on_time",
    "outcome_known_time",
    "is_reward",
)


def _prepare_port_experiences(port_experiences: pd.DataFrame) -> pd.DataFrame:
    """Validate and chronologically order the canonical experience stream."""
    missing = sorted(set(_PORT_EXPERIENCE_COLUMNS) - set(port_experiences.columns))
    if missing:
        raise KeyError(
            "history covariates require the canonical port-experience stream; "
            f"missing {missing}. Regenerate the behavioral session cache."
        )
    out = port_experiences.copy()
    if out.empty:
        return out
    numeric = (
        "experience_id",
        "port",
        "patch",
        "arrival_time",
        "departure_time",
        "reward_on_time",
        "outcome_known_time",
    )
    for column in numeric:
        original = out[column]
        converted = pd.to_numeric(original, errors="coerce")
        if (original.notna() & converted.isna()).any():
            raise ValueError(f"port experience {column} must be numeric")
        out[column] = converted
    ids = out["experience_id"].to_numpy(dtype=float)
    if (
        not np.isfinite(ids).all()
        or not np.equal(ids, np.floor(ids)).all()
        or not np.array_equal(np.sort(ids.astype(np.int64)), np.arange(len(out)))
    ):
        raise ValueError("port experience IDs must be unique and contiguous from zero")
    out = out.sort_values("experience_id", kind="stable").reset_index(drop=True)
    ports = out["port"].to_numpy(dtype=float)
    if (
        not np.isfinite(ports).all()
        or not np.equal(ports, np.floor(ports)).all()
        or not np.isin(ports, tuple(WELL_PATCH_MAPPING)).all()
    ):
        raise ValueError("port experiences must use canonical integer reward ports")
    patches = out["patch"].to_numpy(dtype=float)
    expected_patches = out["port"].map(WELL_PATCH_MAPPING).to_numpy(dtype=float)
    if (
        not np.isfinite(patches).all()
        or not np.equal(patches, np.floor(patches)).all()
        or not np.array_equal(patches, expected_patches)
    ):
        raise ValueError("port-experience patch must exactly match its reward port")
    arrival = out["arrival_time"].to_numpy(dtype=float)
    departure = out["departure_time"].to_numpy(dtype=float)
    outcome = out["outcome_known_time"].to_numpy(dtype=float)
    if (
        not np.isfinite(arrival).all()
        or not np.isfinite(departure).all()
        or np.any(np.diff(arrival) <= 0)
        or np.any(departure < arrival)
        or (len(out) > 1 and np.any(departure[:-1] > arrival[1:]))
    ):
        raise ValueError(
            "port-experience dwell clocks must be finite, strictly ordered, "
            "and nonoverlapping"
        )
    observed = pd.array(out["outcome_observed"], dtype="boolean")
    if observed.isna().any():
        raise ValueError("port experience outcome_observed must be finite boolean")
    observed_mask = observed.to_numpy(dtype=bool)
    out["outcome_observed"] = observed
    if not observed_mask.all():
        raise ValueError(
            "canonical port experiences require a physical poke and observed "
            "outcome; epoch boundaries cannot create scientific experiences"
        )
    if (
        not np.isfinite(outcome).all()
        or np.any(np.diff(outcome) < 0)
        or np.any(outcome < arrival)
        or np.any(outcome > departure)
    ):
        raise ValueError(
            "port outcomes require ordered finite clocks within their dwell"
        )
    reward = pd.to_numeric(out["is_reward"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(reward).all() or not np.isin(reward, (0.0, 1.0)).all():
        raise ValueError("port experiences require a finite binary reward")
    out["is_reward"] = pd.array(reward, dtype="boolean")
    reward_on = out["reward_on_time"].to_numpy(dtype=float)
    rewarded = reward == 1.0
    omitted = reward == 0.0
    if (
        (~np.isfinite(reward_on[rewarded])).any()
        or np.isfinite(reward_on[omitted]).any()
        or not np.array_equal(reward_on[rewarded], outcome[rewarded])
    ):
        raise ValueError(
            "rewarded port experiences require reward_on_time equal to "
            "outcome_known_time; omissions require reward_on_time=NaN"
        )
    initial = out["is_initial_experience"].to_numpy()
    if (
        not np.isin(initial, (False, True)).all()
        or initial.sum() != 1
        or not initial[0]
    ):
        raise ValueError("exactly experience 0 must be the initial port experience")
    boundary = pd.array(out["is_boundary_derived"], dtype="boolean")
    if boundary.isna().any():
        raise ValueError("port experience is_boundary_derived must be finite boolean")
    boundary_mask = boundary.to_numpy(dtype=bool)
    if boundary_mask.any():
        raise ValueError(
            "canonical port experiences require a physical poke; "
            "is_boundary_derived must be false"
        )
    out["is_boundary_derived"] = boundary
    movement_ids = pd.to_numeric(
        out["movement_trial_number"], errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    if (
        np.isfinite(movement_ids[0])
        or not np.isfinite(movement_ids[1:]).all()
        or not np.equal(movement_ids[1:], np.floor(movement_ids[1:])).all()
        or len(np.unique(movement_ids[1:])) != len(movement_ids[1:])
    ):
        raise ValueError(
            "only the initial port experience may lack a movement trial; "
            "destination movement_trial_number values must be unique integers"
        )
    return out
