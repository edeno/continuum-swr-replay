"""Summary statistics of the injected-replay sweep.

The sweep of the two replay detectors over injected replays
(``scripts/detection/sweep_memoryless_detection.py``) writes a long-format
summary, one value per row (``sweep_summaries.replay_sweep_summary``). Each
function here takes that summary and returns named numbers: a dict, a series
with one value per measure, or a table with one row per threshold. They are the
statistics the detection summary reports
(``continuum_swr_replay.reported.detection`` names and describes them) and the
supplementary comparison figure draws. The controls' statistics are in
``controls_statistics`` and the held-out check's in
``detector_comparison.heldout_statistics``.

A share is a fraction in [0, 1]. Durations are in seconds; the timing errors
are in milliseconds, as the summary stores them.
"""

import re

import numpy as np
import pandas as pd

__all__ = [
    "sweep_grid_statistics",
    "sweep_jump_pass_statistics",
    "sweep_measures",
    "sweep_pooled_timing_statistics",
    "sweep_position_by_speed",
    "sweep_position_statistics",
    "sweep_timing_exceedance",
    "whole_count",
]


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


def whole_count(share: float, n: float) -> int:
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
        "n_sliding_only": whole_count(overall["found_by_sliding_window_only"], n),
        "n_sliding_only_before_test": whole_count(
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


def sweep_pooled_timing_statistics(summary: pd.DataFrame) -> dict[str, float]:
    """The start and end errors of both detectors, pooled over the replays both
    report.

    Parameters
    ----------
    summary : pd.DataFrame, shape (n_rows, n_columns)
        The replay sweep's summary, with its ``timing_pooled`` block.

    Returns
    -------
    statistics : dict
        The block's values by measure, as
        ``sweep_summaries.pooled_timing_errors`` names them: errors in
        milliseconds, reported minus true, the truth being the replay's own
        first and last spike.

    Raises
    ------
    ValueError
        If the summary has no such block, or a value of it is missing.
    """
    pooled = sweep_measures(summary, "timing_pooled")
    if pooled.empty or pooled.isna().any():
        raise ValueError(
            "replay sweep summary: the timing_pooled block is absent, or a value "
            "of it is missing"
        )
    return {str(measure): float(value) for measure, value in pooled.items()}


def sweep_position_statistics(summary: pd.DataFrame) -> dict[str, float]:
    """How well each detector's decoded position follows the replay, pooled over
    the replays both report.

    Parameters
    ----------
    summary : pd.DataFrame, shape (n_rows, n_columns)
        The replay sweep's summary, with its ``position_pooled`` block.

    Returns
    -------
    statistics : dict
        The block's values by measure, as
        ``sweep_summaries.pooled_position_errors`` names them: distances in cm,
        the speed ratio a pure number.

    Raises
    ------
    ValueError
        If the summary has no such block, or a value of it is missing.
    """
    pooled = sweep_measures(summary, "position_pooled")
    if pooled.empty or pooled.isna().any():
        raise ValueError(
            "replay sweep summary: the position_pooled block is absent, or a value "
            "of it is missing"
        )
    return {str(measure): float(value) for measure, value in pooled.items()}


def sweep_position_by_speed(summary: pd.DataFrame) -> pd.DataFrame:
    """Each detector's median position error at each replay speed.

    Parameters
    ----------
    summary : pd.DataFrame, shape (n_rows, n_columns)
        The replay sweep's summary, with its ``position_by_speed`` block
        (``sweep_summaries.position_error_by_speed``).

    Returns
    -------
    by_speed : pd.DataFrame, shape (n_bins, 3)
        Indexed by ``speed_m_per_s``, the lower edge of each speed bin,
        slowest first; the columns ``sliding_position_error_cm`` and
        ``state_space_position_error_cm`` hold the medians, in cm, and
        ``n_replays`` the replays in the bin.

    Raises
    ------
    ValueError
        If the block is absent, a row has no speed, or a detector has no
        median at some speed.
    """
    block = summary[summary["summary"] == "position_by_speed"]
    if block.empty or block["speed_m_per_s"].isna().any():
        raise ValueError(
            "replay sweep summary: the position_by_speed block is absent, or a row "
            "of it has no speed"
        )
    by_speed = block.pivot(index="speed_m_per_s", columns="measure", values="value")
    by_speed.columns.name = None
    if by_speed.isna().any().any():
        raise ValueError(
            "replay sweep summary: position_by_speed lacks a value at some speed"
        )
    return by_speed.sort_index()


_EXCEEDANCE_MEASURE = re.compile(
    r"(?P<detector>sliding|state_space)_(?P<edge>start|end)_error_beyond_"
    r"(?P<threshold_ms>\d+)ms"
)


def sweep_timing_exceedance(summary: pd.DataFrame) -> pd.DataFrame:
    """The share of replays whose start or end error is beyond each size.

    Parameters
    ----------
    summary : pd.DataFrame, shape (n_rows, n_columns)
        The replay sweep's summary, with its ``timing_exceedance`` block
        (``sweep_summaries.timing_error_exceedance``).

    Returns
    -------
    shares : pd.DataFrame, shape (n_thresholds, 4)
        Indexed by ``threshold_ms``, smallest first; the columns
        ``sliding_start``, ``sliding_end``, ``state_space_start`` and
        ``state_space_end`` hold the share of the replays both detectors report
        whose absolute error is above the threshold.

    Raises
    ------
    ValueError
        If the block is absent, a measure of it is not one of these, or a share
        is missing.
    """
    block = sweep_measures(summary, "timing_exceedance")
    rows = []
    for measure, value in block.items():
        match = _EXCEEDANCE_MEASURE.fullmatch(str(measure))
        if match is None:
            raise ValueError(
                f"replay sweep summary: timing_exceedance holds {measure!r}, which "
                "is not a share beyond a threshold"
            )
        rows.append(
            {
                "threshold_ms": int(match["threshold_ms"]),
                "column": f"{match['detector']}_{match['edge']}",
                "share": float(value),
            }
        )
    if not rows:
        raise ValueError("replay sweep summary: the timing_exceedance block is absent")
    shares = pd.DataFrame(rows).pivot(
        index="threshold_ms", columns="column", values="share"
    )
    shares.columns.name = None
    if shares.isna().any().any():
        raise ValueError(
            "replay sweep summary: timing_exceedance lacks a share at some threshold"
        )
    return shares.sort_index()
