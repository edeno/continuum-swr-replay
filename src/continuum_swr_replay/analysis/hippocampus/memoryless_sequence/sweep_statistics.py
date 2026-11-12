"""Summary statistics of the memoryless sweeps.

The three sweeps of the two replay detectors on simulated data each write a
long-format summary, one value per row: the injected-replay sweep
(``scripts/detection/sweep_memoryless_detection.py``), its controls
(``sweep_memoryless_controls.py``) and the held-out check
(``sweep_memoryless_heldout.py``). Each function here takes one of those
summaries and returns named numbers: one value, a dict, a series with one
value per measure or level, or a table with one row per true duration
(:func:`heldout_event_classes` returns the names of the classes of event
instead). They are the statistics the detection summary reports
(``continuum_swr_replay.reported.detection`` names and describes them) and the
memoryless comparison figure draws
(``scripts/figures/generate_memoryless_comparison.py``).

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
    "parts_by_level",
    "pooled_heldout_rate",
    "sweep_by_duration",
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
    """One measure of a by-duration block: its value at each true duration (s)
    the block has a row for, shortest first.

    Raises
    ------
    ValueError
        If the block has no row of the measure, or one with no ``duration_s``.
    """
    rows = summary[(summary["summary"] == block) & (summary["measure"] == measure)]
    if rows.empty or rows["duration_s"].isna().any():
        raise ValueError(
            f"replay sweep summary: {block}, {measure} has no row, or a row with "
            "no duration_s"
        )
    return rows.sort_values("duration_s").set_index("duration_s")["value"]


def sweep_by_duration(
    summary: pd.DataFrame, block: str, measures: Sequence[str]
) -> pd.DataFrame:
    """Measures of one by-duration block of the replay sweep's summary, side by
    side.

    Parameters
    ----------
    summary : pd.DataFrame, shape (n_rows, n_columns)
        The replay sweep's summary (:func:`sweep_measures`), with ``duration_s``
        on the rows of ``block``.
    block : str
        A block with one row per measure and true duration, such as
        ``"timing_by_duration"``.
    measures : sequence of str
        The measures read, such as ``"sliding_start_error_ms_median"``.

    Returns
    -------
    values : pd.DataFrame, shape (n_durations, n_measures)
        One column per measure, in the order given, indexed by the true
        duration in seconds, shortest first; in the units the summary stores
        (the timing errors in milliseconds).

    Raises
    ------
    ValueError
        If a measure has no row or a row with no ``duration_s``, or the
        measures are not all at the same durations.
    """
    columns = {measure: _by_duration(summary, block, measure) for measure in measures}
    durations = columns[measures[0]].index
    if not all(column.index.equals(durations) for column in columns.values()):
        raise ValueError(
            f"replay sweep summary: the {block} measures are not all at the same "
            "durations"
        )
    return pd.DataFrame(columns)


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
    n = float(overall["n_cells"])
    return {
        "n_replays": n,
        "domain_fraction": float(domain["n_cells"]) / n,
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
          longest true durations the block has a jump-half row for, in
          seconds.
        - ``shortest_fraction``, ``longest_fraction``: at each, the share
          passing the jump half among the replays the sliding window reported
          before its test.

    Raises
    ------
    ValueError
        If the block has no jump-half row, or one with no ``duration_s``.
    """
    jump_pass = _by_duration(
        summary, "shuffle_pass_rate_by_duration", "sliding_jump_pass"
    )
    return {
        "shortest_duration_s": float(jump_pass.index[0]),
        "shortest_fraction": float(jump_pass.iloc[0]),
        "longest_duration_s": float(jump_pass.index[-1]),
        "longest_fraction": float(jump_pass.iloc[-1]),
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
        The errors and the reported duration are in milliseconds, as the
        summary stores them; "over durations" is over the true durations the
        two blocks have rows for, the same for every measure read.

        - ``shortest_duration_s``, ``longest_duration_s``: the shortest and
          longest of those durations, in seconds.
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

    Raises
    ------
    ValueError
        If a measure has no row or a row with no ``duration_s``, or the
        measures are not all at the same durations.
    """
    errors = sweep_by_duration(
        summary,
        "timing_by_duration",
        [
            "sliding_start_error_ms_median",
            "sliding_end_error_ms_median",
            "sliding_end_error_ms_q75",
            "state_space_start_error_ms_median",
            "state_space_end_error_ms_median",
        ],
    )
    ratio = _by_duration(
        summary, "median_duration_ratio_by_duration", "sliding_duration_ratio"
    )
    durations = errors.index
    if not ratio.index.equals(durations):
        raise ValueError(
            "replay sweep summary: the timing errors and the duration ratio are "
            "not all at the same durations"
        )
    # as arrays, so that a missing value is NaN in the result and not skipped
    start = errors["sliding_start_error_ms_median"].to_numpy()
    end = errors["sliding_end_error_ms_median"].to_numpy()
    state_space = errors[
        ["state_space_start_error_ms_median", "state_space_end_error_ms_median"]
    ].to_numpy()
    return {
        "shortest_duration_s": float(durations[0]),
        "longest_duration_s": float(durations[-1]),
        "state_space_edge_error_max_ms": float(np.abs(state_space).max()),
        "sliding_start_early_min_ms": float(-start.max()),
        "sliding_start_early_max_ms": float(-start.min()),
        "sliding_end_late_min_ms": float(end.min()),
        "sliding_end_late_max_ms": float(end.max()),
        "sliding_end_late_upper_quartile_max_ms": float(
            errors["sliding_end_error_ms_q75"].to_numpy().max()
        ),
        "shortest_start_early_ms": float(-start[0]),
        "shortest_end_late_ms": float(end[0]),
        "shortest_reported_duration_ms": float(1000 * durations[0] * ratio.iloc[0]),
    }


# --- the controls: activity that is not one clean replay -------------------------


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
        ``condition``, ``level`` (a jump in cm, a gap or a burst duration in
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


def burst_share_range(
    controls: pd.DataFrame, condition: str, share: str, detector: str
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
        burst's time the detector output gives that state.
    detector : str
        The detector output (a state-space one; the sliding window has no
        states).

    Returns
    -------
    statistics : dict
        ``min`` and ``max``, the lowest and highest share over the burst
        durations the summary has a row of the share for.

    Raises
    ------
    ValueError
        If the summary has no such share at any burst duration, or not exactly
        one of the detector output's at a duration.
    """
    durations = _levels(controls, condition, "bursts", share)
    if not durations:
        raise ValueError(
            f"controls summary: no {share} of {condition} at any burst duration"
        )
    shares = np.array(
        [
            controls_value(controls, "bursts", share, condition, duration, detector)
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
        - ``rate``: the pooled rate, a share of ``n_events``; NaN when no event
          counts (the fold has no event of these classes, or none informative).
        - ``n_events``: the events the rate rests on, its denominator: the
          informative events for a support rate, all events for an
          informative share.
        - ``n_counted``: of those, the events the rate counts, its numerator:
          the events with held-out support, or the informative ones.

    Raises
    ------
    ValueError
        If the by-class block has no row of the fold, estimator, truth, a
        class or a measure read; two rows for one class and measure; a class
        with no number of events, or with events and no rate; or a class whose
        rate is not a whole number of its events.
    """
    rate_name = f"{measure}{suffix}"
    weight = "n_events_all" if measure == "informative_share" else f"n_events{suffix}"
    by_class = heldout[heldout["summary"] == "by_class"]
    for column, wanted in (
        ("fold", fold),
        ("estimator", estimator),
        ("truth", truth),
        ("measure", rate_name),
        ("measure", weight),
        *(("event_class", event_class) for event_class in classes),
    ):
        if wanted not in set(by_class[column]):
            raise ValueError(
                f"held-out summary: no by-class row with {column} {wanted!r}"
            )
    rows = by_class[
        (by_class["fold"] == fold)
        & (by_class["estimator"] == estimator)
        & (by_class["detector"] == "state_space")
        & by_class["event_class"].isin(classes)
        & (by_class["truth"] == truth)
        & by_class["measure"].isin([rate_name, weight])
    ]
    where = f"held-out summary, fold {fold}, {estimator}, {truth}"
    if rows.duplicated(["event_class", "measure"]).any():
        raise ValueError(f"{where}: several rows for one class and measure")
    rows = rows.pivot(index="event_class", columns="measure", values="value").reindex(
        columns=[rate_name, weight]
    )
    n, rates = rows[weight], rows[rate_name]
    counted = n > 0
    if n.isna().any() or rates[counted].isna().any():
        raise ValueError(
            f"{where}: a class has no {weight}, or has events and no {rate_name}"
        )
    if not counted.any():
        return {"rate": float("nan"), "n_events": 0, "n_counted": 0}
    n, rates = n[counted], rates[counted]
    return {
        "rate": float((rates * n).sum() / n.sum()),
        "n_events": int(n.sum()),
        "n_counted": sum(
            _whole_count(rate, events) for rate, events in zip(rates, n, strict=True)
        ),
    }


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
          rates; NaN when a fold has no rate (no event that counts).
        - ``rate_spread``: the highest minus the lowest; NaN when a fold has no
          rate.
        - ``n_events``: the events the rates rest on, summed over the folds.
        - ``n_counted``: of those, the events the rates count (with held-out
          support, or informative), summed over the folds.
        - ``n_folds_with_support``: the folds whose rate is above zero.
    """
    pooled = [
        pooled_heldout_rate(heldout, fold, classes, truth, measure, suffix, estimator)
        for fold in folds
    ]
    rates = [fold["rate"] for fold in pooled]
    return {
        "rate_min": float(np.min(rates)),
        "rate_max": float(np.max(rates)),
        "rate_spread": float(np.ptp(rates)),
        "n_events": sum(fold["n_events"] for fold in pooled),
        "n_counted": sum(fold["n_counted"] for fold in pooled),
        "n_folds_with_support": sum(fold["rate"] > 0 for fold in pooled),
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

    Raises
    ------
    ValueError
        If the summary holds no fold's agreement, or a fold's is missing.
    """
    agreement = heldout[
        (heldout["summary"] == "agreement")
        & (heldout["measure"] == "estimator_agreement")
    ]["value"]
    if agreement.empty or agreement.isna().any():
        raise ValueError(
            "held-out summary: no estimator agreement, or a fold without one"
        )
    return float(agreement.min())


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
