"""Summary statistics of the memoryless sweeps.

The injected-replay sweep of the two replay detectors on simulated data
(``scripts/detection/sweep_memoryless_detection.py``) writes a long-format
summary, one value per row. Each function here takes that summary and returns
named numbers: a dict, or a series with one value per measure. They are the
statistics the detection summary reports
(``continuum_swr_replay.reported.detection`` names and describes them).

A share is a fraction in [0, 1]. Durations are in seconds; the replay sweep
stores its timing errors in milliseconds, and the statistics of them stay in
milliseconds.
"""

import numpy as np
import pandas as pd

__all__ = [
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
