"""Summary statistics of the memoryless sweeps.

The three sweeps of the two replay detectors on simulated data each write a
long-format summary, one value per row: the injected-replay sweep
(``scripts/detection/sweep_memoryless_detection.py``), its controls
(``sweep_memoryless_controls.py``) and the held-out check
(``sweep_memoryless_heldout.py``). Each function here takes one of those
summaries and returns named numbers: one value, a dict, or a series with one
value per measure (:func:`heldout_event_classes` returns the names of the
classes of event instead). They are the statistics the detection summary
reports (``continuum_swr_replay.reported.detection`` names and describes them).

A share is a fraction in [0, 1]. Durations are in seconds; the replay sweep
stores its timing errors in milliseconds, and the statistics of them stay in
milliseconds.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "burst_share_range",
    "carving_statistics",
    "controls_value",
    "heldout_event_classes",
    "heldout_fold_statistics",
    "lowest_estimator_agreement",
    "no_replay_statistics",
    "part_recovery_statistics",
    "pooled_heldout_rate",
    "sweep_grid_statistics",
    "sweep_jump_pass_statistics",
    "sweep_measures",
    "sweep_timing_statistics",
]

# --- the injected-replay sweep --------------------------------------------------


def sweep_measures(summary: pd.DataFrame, block: str) -> pd.Series:
    """The values of one block of the replay sweep's summary, by measure.

    Parameters
    ----------
    summary : pd.DataFrame, shape (n_rows, n_columns)
        The replay sweep's summary, with ``summary`` (the block a row belongs
        to), ``measure`` and ``value``.
    block : str
        A block with one row per measure: ``"overall"`` (every swept replay) or
        ``"overall_in_domain"`` (the replays the published definition can
        accept).

    Returns
    -------
    values : pd.Series, shape (n_measures,)
        ``value``, indexed by ``measure``.
    """
    return summary[summary["summary"] == block].set_index("measure")["value"]


def _by_duration(summary: pd.DataFrame, block: str, measure: str) -> pd.Series:
    """One measure of a by-duration block: its value at each true duration (s),
    shortest first."""
    rows = summary[
        (summary["summary"] == block) & (summary["measure"] == measure)
    ].sort_values("duration_s")
    return rows.set_index("duration_s")["value"]


def _whole_count(share: float, n: float) -> int:
    """The count a share of ``n`` stands for.

    Raises
    ------
    ValueError
        If ``share * n`` is not a whole number to within 1e-6, the rounding error
        of a share stored as a float.
    """
    count = share * n
    if not np.isclose(count, round(count), rtol=0, atol=1e-6):
        raise ValueError(f"{share} of {n} is not a whole count")
    return int(round(count))


def sweep_grid_statistics(summary: pd.DataFrame) -> dict[str, float]:
    """How the swept replays divide between the two detectors' reports.

    Parameters
    ----------
    summary : pd.DataFrame, shape (n_rows, n_columns)
        The replay sweep's summary (:func:`sweep_measures`).

    Returns
    -------
    statistics : dict
        - ``n_replays``: the swept replays (grid cells).
        - ``domain_fraction``: the share of them the published definition can
          accept.
        - ``n_sliding_only``: replays the sliding-window detector reported and
          the state-space detector did not.
        - ``n_sliding_only_before_test``: the same, counting the sliding
          window's reports before its significance test.

    Raises
    ------
    ValueError
        If a stored share of the swept replays is not a whole number of them.
    """
    overall = sweep_measures(summary, "overall")
    domain = sweep_measures(summary, "overall_in_domain")
    n = overall["n_cells"]
    return {
        "n_replays": n,
        "domain_fraction": domain["n_cells"] / n,
        "n_sliding_only": _whole_count(overall["found_by_sliding_window_only"], n),
        "n_sliding_only_before_test": _whole_count(
            overall["found_by_sliding_window_only_before_shuffle"], n
        ),
    }


def sweep_jump_pass_statistics(summary: pd.DataFrame) -> dict[str, float]:
    """The jump half of the significance test at the shortest and longest replays.

    Parameters
    ----------
    summary : pd.DataFrame, shape (n_rows, n_columns)
        The replay sweep's summary, with ``duration_s`` on the rows of its
        ``shuffle_pass_rate_by_duration`` block.

    Returns
    -------
    statistics : dict
        - ``shortest_duration_s``, ``longest_duration_s``: the shortest and
          longest true durations swept, in seconds.
        - ``shortest_fraction``, ``longest_fraction``: at each, the share
          passing the jump half among the replays the sliding window reported
          before its test.
    """
    jump_pass = _by_duration(
        summary, "shuffle_pass_rate_by_duration", "sliding_jump_pass"
    )
    return {
        "shortest_duration_s": jump_pass.index[0],
        "shortest_fraction": jump_pass.iloc[0],
        "longest_duration_s": jump_pass.index[-1],
        "longest_fraction": jump_pass.iloc[-1],
    }


def sweep_timing_statistics(summary: pd.DataFrame) -> dict[str, float]:
    """Where each detector puts a replay's start and end, over true durations.

    A timing error is reported minus true, on the replays both detectors
    reported: the sliding window starts early when its start error is negative
    and ends late when its end error is positive. The summary holds the median
    and quartiles of each error at each true duration.

    Parameters
    ----------
    summary : pd.DataFrame, shape (n_rows, n_columns)
        The replay sweep's summary, with ``duration_s`` on the rows of its
        ``timing_by_duration`` and ``median_duration_ratio_by_duration`` blocks.

    Returns
    -------
    statistics : dict
        All in milliseconds; "over durations" is over the true durations swept.

        - ``state_space_edge_error_max_ms``: the largest absolute median start
          or end error of the state-space detector over durations.
        - ``sliding_start_early_min_ms``, ``sliding_start_early_max_ms``: the
          smallest and largest median amount the sliding window starts early,
          over durations (the start error's sign reversed).
        - ``sliding_end_late_min_ms``, ``sliding_end_late_max_ms``: the smallest
          and largest median amount it ends late, over durations.
        - ``sliding_end_late_upper_quartile_max_ms``: the largest upper quartile
          of its end error, over durations.
        - ``shortest_start_early_ms``, ``shortest_end_late_ms``: the median
          amounts at the shortest duration.
        - ``shortest_reported_duration_ms``: the median duration the sliding
          window reports for the shortest replays (the shortest duration times
          the median ratio of reported to true duration there).
    """

    def errors(detector: str, edge: str, statistic: str) -> np.ndarray:
        return _by_duration(
            summary, "timing_by_duration", f"{detector}_{edge}_error_ms_{statistic}"
        ).to_numpy()

    sliding_start = errors("sliding", "start", "median")
    sliding_end = errors("sliding", "end", "median")
    state_space_edges = np.concatenate(
        [errors("state_space", edge, "median") for edge in ("start", "end")]
    )
    ratio = _by_duration(
        summary, "median_duration_ratio_by_duration", "sliding_duration_ratio"
    )
    return {
        "state_space_edge_error_max_ms": np.abs(state_space_edges).max(),
        "sliding_start_early_min_ms": -sliding_start.max(),
        "sliding_start_early_max_ms": -sliding_start.min(),
        "sliding_end_late_min_ms": sliding_end.min(),
        "sliding_end_late_max_ms": sliding_end.max(),
        "sliding_end_late_upper_quartile_max_ms": errors("sliding", "end", "q75").max(),
        "shortest_start_early_ms": -sliding_start[0],
        "shortest_end_late_ms": sliding_end[0],
        "shortest_reported_duration_ms": 1000 * ratio.index[0] * ratio.iloc[0],
    }


# --- the controls: activity that is not one clean replay -------------------------


def controls_value(
    controls: pd.DataFrame,
    summary: str,
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
        ``condition``, ``level`` (a jump in cm, a gap or a burst duration in
        seconds; missing on a row pooled over levels), ``detector`` and
        ``value``.
    summary, measure, condition, detector : str
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
        (controls["summary"] == summary)
        & (controls["measure"] == measure)
        & (controls["condition"] == condition)
        & (controls["detector"] == detector)
    ]
    rows = (
        rows[rows["level"].isna()]
        if level is None
        else rows[np.isclose(rows["level"], level)]
    )
    if len(rows) != 1:
        raise ValueError(
            f"controls summary: {len(rows)} rows for {summary}, {measure}, "
            f"{condition}, {level}, {detector}; expected one"
        )
    return float(rows["value"].iloc[0])


def _levels(controls: pd.DataFrame, condition: str) -> list[float]:
    """The levels a multi-part condition was run at, lowest first."""
    rows = controls[
        (controls["summary"] == "parts") & (controls["condition"] == condition)
    ]
    return sorted(rows["level"].dropna().unique())


def no_replay_statistics(controls: pd.DataFrame) -> dict[str, float]:
    """What the detectors' outputs reported in stretches of rest with no replay.

    Parameters
    ----------
    controls : pd.DataFrame, shape (n_rows, n_columns)
        The controls summary (:func:`controls_value`).

    Returns
    -------
    statistics : dict
        - ``minutes``: the simulated immobility with no replay, in minutes.
        - ``n_recordings``: the simulated recordings it comes from.
        - ``n_outputs``: the detector outputs run on it.
        - ``n_outputs_with_events``: of those, the outputs that reported any
          event there.
        - ``rate_bound_per_minute``: the 95 % upper bound on an output's false
          event rate when it reported none, by the rule of three (3 over the
          minutes), in events per minute.
    """
    n_events = controls[
        (controls["summary"] == "no_replay") & (controls["measure"] == "n_events")
    ]["value"]
    minutes = float(
        controls[
            (controls["summary"] == "no_replay")
            & (controls["measure"] == "immobile_minutes")
        ]["value"].iloc[0]
    )
    n_recordings = float(
        controls[
            (controls["summary"] == "provenance")
            & (controls["measure"] == "n_recordings")
        ]["value"].iloc[0]
    )
    return {
        "minutes": minutes,
        "n_recordings": n_recordings,
        "n_outputs": len(n_events),
        "n_outputs_with_events": int((n_events != 0).sum()),
        "rate_bound_per_minute": 3 / minutes,
    }


def burst_share_range(
    controls: pd.DataFrame, condition: str, share: str
) -> dict[str, float]:
    """The range, over burst durations, of the share of a burst's time in a state.

    Parameters
    ----------
    controls : pd.DataFrame, shape (n_rows, n_columns)
        The controls summary (:func:`controls_value`).
    condition : str
        A burst condition (``"placeless_burst"`` or ``"jumping_burst"``).
    share : str
        ``"fragmented_share"`` or ``"continuous_share"``: the mean share of a
        burst's time the state-space detector gives that state.

    Returns
    -------
    statistics : dict
        ``min`` and ``max``, the lowest and highest share over burst durations.
    """
    values = controls[
        (controls["summary"] == "bursts")
        & (controls["measure"] == share)
        & (controls["condition"] == condition)
    ]["value"]
    return {"min": values.min(), "max": values.max()}


def carving_statistics(
    controls: pd.DataFrame, conditions: Sequence[str], label: str, detector: str
) -> dict[str, float]:
    """How often a detector carves two-part trials one way, over levels.

    At each level (a jump size or a gap) the share is the mean, over
    ``conditions``, of the share of trials the detector's events carve as
    ``label``.

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
    shares = np.array(
        [
            np.mean(
                [
                    controls_value(
                        controls,
                        "parts",
                        f"segmentation_{label}",
                        condition,
                        level,
                        detector,
                    )
                    for condition in conditions
                ]
            )
            for level in _levels(controls, conditions[0])
        ]
    )
    return {
        "min": shares.min(),
        "max": shares.max(),
        "at_lowest_level": shares[0],
        "at_highest_level": shares[-1],
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


# --- the held-out check ---------------------------------------------------------


def pooled_heldout_rate(
    heldout: pd.DataFrame,
    fold: str,
    classes: Sequence[str],
    truth: str,
    measure: str = "support_rate",
    suffix: str = "",
    estimator: str = "state_space",
) -> dict[str, float]:
    """A held-out rate over the state-space detector's events of one fold,
    pooled over event classes.

    Each event is counted once, as the state-space detector bounds it. A rate
    over informative events is weighted by how many are informative, a share
    of all events by how many there are.

    Parameters
    ----------
    heldout : pd.DataFrame, shape (n_rows, n_columns)
        The held-out summary, with ``summary`` (the block), ``measure``,
        ``fold``, ``estimator``, ``detector``, ``event_class``, ``truth`` and
        ``value``.
    fold : str
        The fold of discovery and evaluation cells.
    classes : sequence of str
        The event classes pooled (``"ours_only"``, ``"shared"``).
    truth : str
        What the events truly are (``"continuous_replay"``, ``"non_local"``,
        ``"no_content"``).
    measure : str, optional
        ``"support_rate"``, the share of informative events with held-out
        support, or ``"informative_share"``, the share of all events the
        evaluation cells can judge.
    suffix : str, optional
        The coverage floor of the informative events: ``""`` for the main
        floor, ``"_floor2"`` for 2 expected spikes.
    estimator : str, optional
        The content estimate scored: ``"state_space"`` or ``"sliding_window"``.

    Returns
    -------
    statistics : dict
        - ``rate``: the pooled rate; NaN when no event counts.
        - ``n_events``: the events the rate rests on.
    """
    rows = heldout[
        (heldout["summary"] == "by_class")
        & (heldout["fold"] == fold)
        & (heldout["estimator"] == estimator)
        & (heldout["detector"] == "state_space")
        & heldout["event_class"].isin(classes)
        & (heldout["truth"] == truth)
    ].pivot_table(index="event_class", columns="measure", values="value")
    weight = "n_events_all" if measure == "informative_share" else f"n_events{suffix}"
    if rows.empty or weight not in rows or rows[weight].sum() == 0:
        return {"rate": float("nan"), "n_events": 0}
    n = rows[weight].fillna(0)
    rate = float((rows[f"{measure}{suffix}"].fillna(0) * n).sum() / n.sum())
    return {"rate": rate, "n_events": int(n.sum())}


def heldout_fold_statistics(
    heldout: pd.DataFrame,
    folds: Sequence[str],
    classes: Sequence[str],
    truth: str,
    measure: str = "support_rate",
    suffix: str = "",
    estimator: str = "state_space",
) -> dict[str, float]:
    """A pooled held-out rate over several folds.

    Parameters
    ----------
    heldout : pd.DataFrame, shape (n_rows, n_columns)
        The held-out summary (:func:`pooled_heldout_rate`).
    folds : sequence of str
        The folds.
    classes, truth, measure, suffix, estimator
        As in :func:`pooled_heldout_rate`, which gives each fold's rate and the
        events it rests on.

    Returns
    -------
    statistics : dict
        - ``rate_min``, ``rate_max``: the lowest and highest of the folds'
          rates, when every fold has an event that counts.
        - ``rate_spread``: the highest minus the lowest; NaN when a fold has no
          event that counts.
        - ``n_events``: the events the rates rest on, over the folds.
        - ``n_supported``: of those, the events the rates count (each fold's
          rate times its events, rounded to a whole number).
        - ``n_folds_with_support``: the folds whose rate is above zero.
    """
    pooled = [
        pooled_heldout_rate(heldout, fold, classes, truth, measure, suffix, estimator)
        for fold in folds
    ]
    rates = [fold["rate"] for fold in pooled]
    counted = [fold for fold in pooled if fold["n_events"]]
    return {
        "rate_min": min(rates),
        "rate_max": max(rates),
        "rate_spread": float(np.ptp(rates)),
        "n_events": sum(fold["n_events"] for fold in pooled),
        "n_supported": sum(round(fold["rate"] * fold["n_events"]) for fold in counted),
        "n_folds_with_support": sum(fold["rate"] > 0 for fold in counted),
    }


def lowest_estimator_agreement(heldout: pd.DataFrame) -> float:
    """The lowest agreement of the two content estimates, over folds.

    Parameters
    ----------
    heldout : pd.DataFrame, shape (n_rows, n_columns)
        The held-out summary (:func:`pooled_heldout_rate`). Its ``agreement``
        block holds, per fold, the share of the events informative under both
        content estimates on which the two verdicts agree.

    Returns
    -------
    agreement : float
        The lowest fold's share.
    """
    agreement = heldout[
        (heldout["summary"] == "agreement")
        & (heldout["measure"] == "estimator_agreement")
    ]["value"]
    return agreement.min()


def heldout_event_classes(heldout: pd.DataFrame) -> set[str]:
    """The classes of event the held-out summary holds.

    Parameters
    ----------
    heldout : pd.DataFrame, shape (n_rows, n_columns)
        The held-out summary (:func:`pooled_heldout_rate`).

    Returns
    -------
    classes : set of str
        The ``event_class`` of its by-class rows: ``"shared"`` (both detectors
        reported the event), ``"ours_only"`` or ``"comparator_only"``.
    """
    by_class = heldout[heldout["summary"] == "by_class"]
    return set(by_class["event_class"])
