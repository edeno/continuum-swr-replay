"""Summary statistics of the controls sweep.

The controls run the two replay detectors on simulated activity that is not one
clean replay: rest with nothing added, place-less and jumping control events,
and replays with a jump or in a chain
(``scripts/detection/sweep_memoryless_controls.py``, ``sweep_controls``). The
sweep writes a long-format summary, one value per row
(``sweep_summaries.controls_summary``); each function here takes it and returns
named numbers, which the detection summary reports
(``continuum_swr_replay.reported.detection``).

A share is a fraction in [0, 1]. Durations and gaps are in seconds.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "carving_statistics",
    "control_event_share_range",
    "controls_value",
    "no_replay_statistics",
    "part_recovery_statistics",
    "parts_by_level",
]


def _only_value(rows: pd.DataFrame, what: str) -> float:
    """The value of the one row of ``rows``.

    Raises
    ------
    ValueError
        If ``rows`` is not exactly one row.
    """
    if len(rows) != 1:
        raise ValueError(f"{what}: {len(rows)} rows; expected one")
    return float(rows["value"].iloc[0])


def controls_value(
    controls: pd.DataFrame,
    block: str,
    measure: str,
    condition: str,
    level: float | None,
    detector: str,
) -> float:
    """One value of the controls summary.

    Parameters
    ----------
    controls : pd.DataFrame, shape (n_rows, n_columns)
        The controls summary, with ``summary`` (the block), ``measure``,
        ``condition``, ``level`` (a jump in cm, a gap or a control event duration in
        seconds; missing on a row pooled over levels), ``detector`` and
        ``value``.
    block, measure, condition, detector : str
        The row's block, measure, injected condition and detector output.
    level : float or None
        The row's level; None for the row pooled over levels.

    Returns
    -------
    value : float

    Raises
    ------
    ValueError
        If the summary does not hold exactly one such row.
    """
    rows = controls[
        (controls["summary"] == block)
        & (controls["measure"] == measure)
        & (controls["condition"] == condition)
        & (controls["detector"] == detector)
    ]
    rows = (
        rows[rows["level"].isna()]
        if level is None
        else rows[np.isclose(rows["level"], level)]
    )
    return _only_value(
        rows,
        f"controls summary: {block}, {measure}, {condition}, {level}, {detector}",
    )


def _levels(
    controls: pd.DataFrame,
    condition: str,
    block: str = "parts",
    measure: str | None = None,
) -> list[float]:
    """The levels a condition was run at in a block of the summary (a
    multi-part condition's, by default), lowest first; only those with a row of
    ``measure`` when it is given."""
    rows = controls[
        (controls["summary"] == block) & (controls["condition"] == condition)
    ]
    if measure is not None:
        rows = rows[rows["measure"] == measure]
    return sorted(rows["level"].dropna().unique())


def no_replay_statistics(controls: pd.DataFrame) -> dict[str, float]:
    """What the detectors' outputs reported in stretches of rest with no replay:
    nothing, or the summary is refused.

    Parameters
    ----------
    controls : pd.DataFrame, shape (n_rows, n_columns)
        The controls summary (:func:`controls_value`).

    Returns
    -------
    statistics : dict
        - ``minutes``: the simulated immobility with no replay, in minutes.
        - ``n_recordings``: the simulated recordings it comes from.
        - ``n_outputs``: the detector outputs run on it (the rows of its event
          counts, one per output), none of which reported an event.
        - ``rate_bound_per_minute``: the 95 % upper bound on an output's false
          event rate, by the rule of three (3 over the minutes), in events per
          minute.

    Raises
    ------
    ValueError
        If no output was run there or an output reported an event there (the
        rule of three bounds a rate only when no event was seen), the summary
        holds two event counts for one output, or not exactly one row of the
        minutes or of the recordings.
    """
    no_replay = controls[controls["summary"] == "no_replay"]
    n_events = no_replay[no_replay["measure"] == "n_events"]
    if n_events["detector"].duplicated().any():
        raise ValueError(
            "controls summary: no_replay, n_events has several rows for one "
            "detector output"
        )
    n_with_events = int((n_events["value"] != 0).sum())
    if n_events.empty or n_with_events:
        raise ValueError(
            "controls summary: no_replay, n_events: expected outputs run there and "
            f"no output reported an event; {len(n_events)} outputs were run, "
            f"{n_with_events} reported events"
        )
    minutes = _only_value(
        no_replay[no_replay["measure"] == "immobile_minutes"],
        "controls summary: no_replay, immobile_minutes",
    )
    n_recordings = _only_value(
        controls[
            (controls["summary"] == "provenance")
            & (controls["measure"] == "n_recordings")
        ],
        "controls summary: provenance, n_recordings",
    )
    return {
        "minutes": minutes,
        "n_recordings": n_recordings,
        "n_outputs": len(n_events),
        "rate_bound_per_minute": 3 / minutes,
    }


def control_event_share_range(
    controls: pd.DataFrame, condition: str, share: str, detector: str
) -> dict[str, float]:
    """The range, over control event durations, of the share of an event's time in a state.

    Parameters
    ----------
    controls : pd.DataFrame, shape (n_rows, n_columns)
        The controls summary (:func:`controls_value`).
    condition : str
        A control event condition (``"placeless_event"`` or ``"jumping_event"``).
    share : str
        ``"fragmented_share"`` or ``"continuous_share"``: the mean share of a
        event's time the detector output gives that state.
    detector : str
        The detector output (a state-space one; the sliding window has no
        states).

    Returns
    -------
    statistics : dict
        ``min`` and ``max``, the lowest and highest share over the control event
        durations the summary has a row of the share for.

    Raises
    ------
    ValueError
        If the summary has no such share at any control event duration, or not exactly
        one of the detector output's at a duration.
    """
    durations = _levels(controls, condition, "control_events", share)
    if not durations:
        raise ValueError(
            f"controls summary: no {share} of {condition} at any control event duration"
        )
    shares = np.array(
        [
            controls_value(
                controls, "control_events", share, condition, duration, detector
            )
            for duration in durations
        ]
    )
    return {"min": float(shares.min()), "max": float(shares.max())}


def parts_by_level(
    controls: pd.DataFrame, conditions: Sequence[str], measure: str, detector: str
) -> pd.Series:
    """A measure of two-part trials at each level, pooled over conditions.

    At each level (a jump size in cm or a gap in seconds) the value is the
    mean, over ``conditions``, of each condition's value there.

    Parameters
    ----------
    controls : pd.DataFrame, shape (n_rows, n_columns)
        The controls summary (:func:`controls_value`).
    conditions : sequence of str
        The two-part conditions pooled; all run at the first one's levels.
    measure : str
        A measure of the ``parts`` block given at each level, such as
        ``"segmentation_spanning"`` (the share of trials the detector's events
        carve as one event over both parts) or ``"part_recovery"``.
    detector : str
        The detector output.

    Returns
    -------
    values : pd.Series, shape (n_levels,)
        Indexed by level, lowest first.

    Raises
    ------
    ValueError
        If a condition does not hold exactly one row of the measure at one of
        the first condition's levels.
    """
    levels = _levels(controls, conditions[0])
    return pd.Series(
        [
            np.mean(
                [
                    controls_value(
                        controls, "parts", measure, condition, level, detector
                    )
                    for condition in conditions
                ]
            )
            for level in levels
        ],
        index=pd.Index(levels, name="level"),
        name=measure,
        dtype=float,
    )


def carving_statistics(
    controls: pd.DataFrame, conditions: Sequence[str], label: str, detector: str
) -> dict[str, float]:
    """How often a detector carves two-part trials one way, over levels.

    At each level (a jump size or a gap) the share is the mean, over
    ``conditions``, of the share of trials the detector's events carve as
    ``label`` (:func:`parts_by_level`).

    Parameters
    ----------
    controls : pd.DataFrame, shape (n_rows, n_columns)
        The controls summary (:func:`controls_value`).
    conditions : sequence of str
        The two-part conditions pooled; all run at the first one's levels.
    label : str
        The carving: ``"spanning"`` (one event over both parts),
        ``"one_per_part"`` or ``"parts_missing"``.
    detector : str
        The detector output.

    Returns
    -------
    statistics : dict
        - ``min``, ``max``: the lowest and highest share over levels.
        - ``at_lowest_level``, ``at_highest_level``: the share at the lowest
          and at the highest level.
    """
    shares = parts_by_level(
        controls, conditions, f"segmentation_{label}", detector
    ).to_numpy()
    return {
        "min": float(shares.min()),
        "max": float(shares.max()),
        "at_lowest_level": float(shares[0]),
        "at_highest_level": float(shares[-1]),
    }


def part_recovery_statistics(
    controls: pd.DataFrame, conditions: Sequence[str], detector: str
) -> dict[str, float]:
    """How many parts of two-part trials a detector recovers, inside the event
    and with each part decoded alone.

    Parameters
    ----------
    controls : pd.DataFrame, shape (n_rows, n_columns)
        The controls summary (:func:`controls_value`).
    conditions : sequence of str
        The two-part conditions.
    detector : str
        The detector output.

    Returns
    -------
    statistics : dict
        - ``recovery_in_event``: the share of parts recovered inside the event,
          the mean over every condition and level of that cell's share.
        - ``recovery_alone``: the same with each part decoded alone.
        - ``n_trials``: the trials, over the conditions.
        - ``n_trials_fewer_in_event``, ``n_trials_more_in_event``: the trials
          in which fewer, or more, parts are recovered inside the event than
          alone.
    """

    def mean_over_cells(measure: str) -> float:
        return float(
            np.mean(
                [
                    controls_value(
                        controls, "parts", measure, condition, level, detector
                    )
                    for condition in conditions
                    for level in _levels(controls, condition)
                ]
            )
        )

    def total(measure: str) -> float:
        return sum(
            controls_value(controls, "parts", measure, condition, None, detector)
            for condition in conditions
        )

    return {
        "recovery_in_event": mean_over_cells("part_recovery"),
        "recovery_alone": mean_over_cells("part_recovery_alone"),
        "n_trials": total("n_trials"),
        "n_trials_fewer_in_event": total("n_trials_fewer_in_context"),
        "n_trials_more_in_event": total("n_trials_more_in_context"),
    }
