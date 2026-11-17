"""The long-format summaries of the three memoryless sweeps, from their raw tables.

Each sweep of the two replay detectors on simulated data writes its raw table
under ``Processed-Data/replay_detection_comparison/`` and a summary, one value
per row, to ``paper/reported/inputs/detection/``: the injected-replay sweep
(``scripts/detection/sweep_memoryless_detection.py``,
:func:`replay_sweep_summary`), its controls (``sweep_memoryless_controls.py``,
:func:`controls_summary`) and the held-out check (``sweep_memoryless_heldout.py``,
whose summary is ``detector_comparison.heldout.heldout_summary``, beside the
held-out rule it summarizes). The scripts add the provenance rows and write the
files; the statistics the detection summary reports are read from these
summaries by ``sweep_statistics``.

A rate or share is a mean of booleans, in [0, 1]; durations are in seconds and
the replay sweep's timing errors in milliseconds.
"""

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detection.memoryless_sequence.scoring import (
    SEGMENTATION_LABELS,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    SIGNIFICANCE_LEVEL,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep import CONFIG

IN_DOMAIN_PATH_CM = 45.0
IN_DOMAIN_DURATION_S = 0.12
"""The smallest swept path and duration the published replay definition can
accept. Dispersion must exceed 12 cm and a uniform traverse of length L has
dispersion L / sqrt(12), so a path under about 42 cm cannot qualify however
well it is decoded; duration must exceed 100 ms. Three quarters of the grid
lies outside this, by design, to show where each procedure's limit is -- but
the aggregate over the whole grid is not a rate of anything the published
method claims to detect."""


def share(numerator: float, denominator: float) -> float:
    """A rate that is undefined, not zero, when nothing was counted."""
    return float(numerator / denominator) if denominator > 0 else np.nan


def melted(frame: pd.DataFrame, id_vars: list[str], summary: str) -> pd.DataFrame:
    """One block of the long-format summary, its grouping columns kept."""
    return frame.melt(id_vars=id_vars, var_name="measure", value_name="value").assign(
        summary=summary
    )


LARGE_TIMING_ERROR_MS = 100.0
"""An edge placed further than this from the truth counts as far off
(:func:`pooled_timing_errors`)."""


EXCEEDANCE_THRESHOLDS_MS = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80,
    100, 120, 150, 200, 250, 300, 400, 500, 700, 1000,
)  # fmt: skip
"""The absolute timing errors at which :func:`timing_error_exceedance` gives the
share of replays beyond; :data:`LARGE_TIMING_ERROR_MS` is one of them."""


def timing_error_exceedance(found: pd.DataFrame) -> dict[str, float]:
    """How far off each detector's starts and ends are, as a distribution.

    The share of replays whose absolute error is above each threshold: the
    whole distribution of the error's size, tail included, which a median and
    quartiles leave out.

    Parameters
    ----------
    found : pd.DataFrame, shape (n_replays, n_columns)
        The replays both detectors report, as :func:`pooled_timing_errors`
        reads them.

    Returns
    -------
    shares : dict
        ``<detector>_<edge>_error_beyond_<threshold>ms`` for ``sliding`` and
        ``state_space``, the edges ``start`` and ``end``, and each threshold of
        :data:`EXCEEDANCE_THRESHOLDS_MS`: the share of replays whose absolute
        error is above it.
    """
    shares = {}
    for detector in ("sliding", "state_space"):
        for edge in ("start", "end"):
            absolute_ms = 1000.0 * found[f"{detector}_{edge}_error_s"].abs()
            for threshold_ms in EXCEEDANCE_THRESHOLDS_MS:
                shares[f"{detector}_{edge}_error_beyond_{threshold_ms}ms"] = float(
                    (absolute_ms > threshold_ms).mean()
                )
    return shares


def against_spike_span(table: pd.DataFrame) -> pd.DataFrame:
    """The replays' timing measured against their own first and last spike.

    A replay's nominal start and end hold silent time when it is sparse: its
    first spike can come well after the nominal start. No detector can report
    what has not fired, so the truth a start, an end or a duration is held to
    is the span of the replay's own spikes.

    Parameters
    ----------
    table : pd.DataFrame, shape (n_replays, n_columns)
        One row per injected replay: ``at_s`` and ``duration_s`` (the nominal
        start and duration), ``first_spike_s`` and ``last_spike_s``, and for
        ``sliding`` and ``state_space`` the ``<detector>_start_error_s`` and
        ``<detector>_end_error_s`` against the nominal bounds and the reported
        ``<detector>_duration_s``.

    Returns
    -------
    table : pd.DataFrame, shape (n_replays, n_columns + 3)
        A copy whose start and end errors are against the first and the last
        spike, with ``spike_duration_s`` (last minus first spike) and each
        detector's ``<detector>_duration_ratio``, its reported duration over
        that one.
    """
    table = table.copy()
    start_shift_s = table["first_spike_s"] - table["at_s"]
    end_shift_s = table["last_spike_s"] - (table["at_s"] + table["duration_s"])
    table["spike_duration_s"] = table["last_spike_s"] - table["first_spike_s"]
    for detector in ("sliding", "state_space"):
        table[f"{detector}_start_error_s"] -= start_shift_s
        table[f"{detector}_end_error_s"] -= end_shift_s
        table[f"{detector}_duration_ratio"] = (
            table[f"{detector}_duration_s"] / table["spike_duration_s"]
        )
    return table


def pooled_timing_errors(found: pd.DataFrame) -> dict[str, float]:
    """Each detector's start and end errors over the replays both report, pooled.

    An error is reported minus true, in milliseconds: negative for an edge
    placed early.

    Parameters
    ----------
    found : pd.DataFrame, shape (n_replays, n_columns)
        The replays both detectors report, one row each, with for ``sliding``
        and ``state_space`` the ``<detector>_start_error_s`` and
        ``<detector>_end_error_s`` (:func:`against_spike_span`).

    Returns
    -------
    statistics : dict
        - ``n_replays``: the replays pooled.
        - ``<detector>_<edge>_error_ms_<statistic>`` for the edges ``start``
          and ``end`` and the statistics ``min``, ``q25``, ``median``, ``q75``,
          ``max``, ``abs_median`` (the median absolute error) and ``far_share``
          (the share of replays with an absolute error above
          :data:`LARGE_TIMING_ERROR_MS`).
        - ``<detector>_duration_error_ms_<q25|median|q75>``: reported minus
          true duration (the end error minus the start error).
        - ``sliding_larger_<edge>_share``: the share of replays on which the
          sliding window's absolute error is the larger of the two.
    """
    statistics: dict[str, float] = {"n_replays": float(len(found))}
    for detector in ("sliding", "state_space"):
        for edge in ("start", "end"):
            error_ms = 1000.0 * found[f"{detector}_{edge}_error_s"]
            key = f"{detector}_{edge}_error_ms"
            statistics |= {
                f"{key}_min": float(error_ms.min()),
                f"{key}_q25": float(error_ms.quantile(0.25)),
                f"{key}_median": float(error_ms.median()),
                f"{key}_q75": float(error_ms.quantile(0.75)),
                f"{key}_max": float(error_ms.max()),
                f"{key}_abs_median": float(error_ms.abs().median()),
                f"{key}_far_share": float(
                    (error_ms.abs() > LARGE_TIMING_ERROR_MS).mean()
                ),
            }
        duration_error_ms = 1000.0 * (
            found[f"{detector}_end_error_s"] - found[f"{detector}_start_error_s"]
        )
        for name, quantile in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
            statistics[f"{detector}_duration_error_ms_{name}"] = float(
                duration_error_ms.quantile(quantile)
            )
    for edge in ("start", "end"):
        statistics[f"sliding_larger_{edge}_share"] = float(
            (
                found[f"sliding_{edge}_error_s"].abs()
                > found[f"state_space_{edge}_error_s"].abs()
            ).mean()
        )
    return statistics


def replay_sweep_summary(table: pd.DataFrame) -> pd.DataFrame:
    """The aggregates quoted from the injected-replay sweep, in one long table.

    Parameters
    ----------
    table : pd.DataFrame, shape (n_cells, n_columns)
        The sweep's raw table, one row per injected replay (cell): its
        ``path_cm``, ``duration_s`` and ``distance_from_animal_cm``, and each
        detector's outcome (the ``sliding_*`` and ``state_space_*`` columns
        ``scripts/detection/sweep_memoryless_detection.py`` writes).

    Returns
    -------
    summary : pd.DataFrame, shape (n_values, 7)
        ``summary`` (the block), ``measure``, the block's grouping columns
        ``path_cm``, ``duration_s`` and ``distance_bin_cm`` (NaN where the
        block is not grouped by it), ``value`` and an empty ``note``. The
        blocks: detection rates by path and duration
        (``detection_rate_by_path_and_duration``) and by path (``by_path``),
        the shuffle's pass rates by duration, median duration ratios by
        duration and by distance from the animal, the rates over every cell
        (``overall``) and over the cells the published definition can accept
        (``overall_in_domain``, :data:`IN_DOMAIN_PATH_CM`,
        :data:`IN_DOMAIN_DURATION_S`), and the start and end errors' quartiles
        by duration (``timing_by_duration``, ms), pooled (``timing_pooled``,
        :func:`pooled_timing_errors`) and as a distribution
        (``timing_exceedance``, :func:`timing_error_exceedance`). The duration, distance and
        timing blocks read the replays both detectors report as the injection
        (``sliding_matches`` and ``state_space_matches``), their durations and
        errors against each replay's own first and last spike
        (:func:`against_spike_span`); ``duration_s`` on a row is still the
        nominal duration the sweep's grid names.
    """
    # the replays both detectors report, the sliding window under the whole
    # published definition: the detection rates count those, and a duration or
    # timing drawn beside them must be of the same events
    found = against_spike_span(table[table.sliding_matches & table.state_space_matches])
    missed = table[table.state_space_matches & ~table.sliding_matches]

    parts = []

    by_path_duration = (
        table.groupby(["path_cm", "duration_s"])[
            [
                "sliding_found",
                "sliding_matches",
                "sliding_matches_before_shuffle",
                "sliding_matches_spread_only",
                "sliding_any_candidate",
                "state_space_found",
                "state_space_matches",
                "state_space_replay_matches",
            ]
        ]
        .mean()
        .reset_index()
    )
    parts.append(
        melted(
            by_path_duration,
            ["path_cm", "duration_s"],
            "detection_rate_by_path_and_duration",
        )
    )

    # the published replay definition applied to the state-space detector's
    # own events, by path, so the dispersion floor can be read off the file
    by_path = (
        # before the shuffle, sliding_matches_before_shuffle is the like-for-like
        # partner of state_space_replay_matches: both are the duration and
        # dispersion criteria alone
        table.groupby("path_cm")[
            [
                "sliding_matches",
                "sliding_matches_before_shuffle",
                "state_space_replay_matches",
            ]
        ]
        .mean()
        .join(
            table[table.state_space_matches]
            .groupby("path_cm")
            .state_space_dispersion_cm.median()
            .rename("state_space_median_dispersion_cm")
        )
        .reset_index()
    )
    parts.append(melted(by_path, ["path_cm"], "by_path"))

    # each half of the significance test, among the replays reported before
    # it, by duration: a short replay covers its path in few windows, so its
    # jumps are large and the jump half is where it fails
    reported = table[table.sliding_matches_before_shuffle]
    shuffle_by_duration = (
        reported.assign(
            sliding_spread_pass=reported.sliding_p_spread.le(SIGNIFICANCE_LEVEL),
            sliding_jump_pass=reported.sliding_p_jump.le(SIGNIFICANCE_LEVEL),
        )
        .groupby("duration_s")[["sliding_spread_pass", "sliding_jump_pass"]]
        .mean()
        .reset_index()
    )
    parts.append(
        melted(shuffle_by_duration, ["duration_s"], "shuffle_pass_rate_by_duration")
    )

    by_duration = (
        found.groupby("duration_s")[
            # the sliding window's reported duration itself, beside its ratio
            # to the replay's spike span
            [
                "sliding_duration_ratio",
                "state_space_duration_ratio",
                "sliding_duration_s",
            ]
        ]
        .median()
        .reset_index()
    )
    parts.append(
        melted(by_duration, ["duration_s"], "median_duration_ratio_by_duration")
    )

    found["distance_bin_cm"] = pd.cut(
        found.distance_from_animal_cm, [0, 40, 80, 120, 200]
    ).astype(str)
    by_distance = (
        found.groupby("distance_bin_cm", observed=True)[
            [
                "sliding_duration_ratio",
                "sliding_local_fraction",
                "state_space_duration_ratio",
            ]
        ]
        .median()
        .reset_index()
    )
    parts.append(
        melted(by_distance, ["distance_bin_cm"], "median_by_distance_from_animal")
    )

    overall = pd.DataFrame(
        {
            "measure": [
                "n_cells",
                "sliding_found",
                "sliding_matches",
                "sliding_matches_before_shuffle",
                "sliding_matches_spread_only",
                "sliding_shuffle_spread_pass_rate",
                "sliding_shuffle_jump_pass_rate",
                "missed_failed_shuffle",
                "sliding_any_candidate",
                "state_space_found",
                "state_space_matches",
                "state_space_replay_matches",
                "found_by_state_space_only",
                "found_by_sliding_window_only",
                "found_by_sliding_window_only_before_shuffle",
                "sliding_accepted_but_not_the_injection",
                "sliding_cells_with_several_accepted_overlaps",
                "state_space_reports_in_several_intervals",
                "missed_with_candidate_below_dispersion",
                "missed_with_no_candidate",
                "missed_median_spread_pass_rate",
            ],
            "value": [
                float(len(table)),
                float(table.sliding_found.mean()),
                float(table.sliding_matches.mean()),
                float(table.sliding_matches_before_shuffle.mean()),
                float(table.sliding_matches_spread_only.mean()),
                # each half's pass rate among the injected replays the
                # detector reported before the shuffle
                float(
                    table[table.sliding_matches_before_shuffle]
                    .sliding_p_spread.le(SIGNIFICANCE_LEVEL)
                    .mean()
                ),
                float(
                    table[table.sliding_matches_before_shuffle]
                    .sliding_p_jump.le(SIGNIFICANCE_LEVEL)
                    .mean()
                ),
                float(missed.sliding_matches_before_shuffle.mean()),
                float(table.sliding_any_candidate.mean()),
                float(table.state_space_found.mean()),
                float(table.state_space_matches.mean()),
                float(table.state_space_replay_matches.mean()),
                float((table.state_space_matches & ~table.sliding_matches).mean()),
                float((table.sliding_matches & ~table.state_space_matches).mean()),
                float(
                    (
                        table.sliding_matches_before_shuffle
                        & ~table.state_space_matches
                    ).mean()
                ),
                share(
                    (table.sliding_found & ~table.sliding_matches_before_shuffle).sum(),
                    table.sliding_found.sum(),
                ),
                float(table.sliding_n_accepted.gt(1).mean()),
                share(
                    table.state_space_n_intervals.gt(1).sum(),
                    table.state_space_found.sum(),
                ),
                float(
                    (
                        missed.sliding_n_candidates.gt(0)
                        & missed.sliding_best_dispersion_cm.lt(
                            CONFIG.dispersion_threshold_cm
                        )
                    ).mean()
                ),
                float(missed.sliding_n_candidates.eq(0).mean()),
                float(missed.sliding_spread_ok.median()),
            ],
        }
    )
    overall["summary"] = "overall"
    parts.append(overall)

    in_domain = table[
        (table.path_cm >= IN_DOMAIN_PATH_CM)
        & (table.duration_s >= IN_DOMAIN_DURATION_S)
    ]
    domain = pd.DataFrame(
        {
            "measure": [
                "n_cells",
                "sliding_matches",
                "sliding_matches_before_shuffle",
                "sliding_matches_spread_only",
                "state_space_matches",
                "state_space_replay_matches",
            ],
            "value": [
                float(len(in_domain)),
                float(in_domain.sliding_matches.mean()),
                float(in_domain.sliding_matches_before_shuffle.mean()),
                float(in_domain.sliding_matches_spread_only.mean()),
                float(in_domain.state_space_matches.mean()),
                float(in_domain.state_space_replay_matches.mean()),
            ],
        }
    )
    domain["summary"] = "overall_in_domain"
    parts.append(domain)

    # where each detector puts an event's start and end against the truth, on
    # the replays both reported: signed, so absorbed local coding shows as an
    # early start or a late end rather than only as a longer event
    timing = []
    for duration_s, group in found.groupby("duration_s"):
        row = {"duration_s": duration_s}
        for detector in ("sliding", "state_space"):
            for edge in ("start", "end"):
                error_ms = 1000.0 * group[f"{detector}_{edge}_error_s"]
                for name, quantile in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
                    row[f"{detector}_{edge}_error_ms_{name}"] = float(
                        error_ms.quantile(quantile)
                    )
        timing.append(row)
    parts.append(melted(pd.DataFrame(timing), ["duration_s"], "timing_by_duration"))

    # the same errors pooled over those replays
    pooled = (
        pd.Series(pooled_timing_errors(found), name="value")
        .rename_axis("measure")
        .reset_index()
    )
    pooled["summary"] = "timing_pooled"
    parts.append(pooled)
    exceedance = (
        pd.Series(timing_error_exceedance(found), name="value")
        .rename_axis("measure")
        .reset_index()
    )
    exceedance["summary"] = "timing_exceedance"
    parts.append(exceedance)

    # the grouping column is not called "aggregate": that name shadows
    # DataFrame.aggregate, so frame.aggregate == "x" compares a method and
    # quietly selects nothing
    table = pd.concat(parts, ignore_index=True)
    table["note"] = ""
    return table[
        [
            "summary",
            "measure",
            "path_cm",
            "duration_s",
            "distance_bin_cm",
            "value",
            "note",
        ]
    ]


def controls_summary(
    trial_table: pd.DataFrame, no_replay: pd.DataFrame
) -> pd.DataFrame:
    """The aggregates of the controls sweep, in one long table.

    Parameters
    ----------
    trial_table : pd.DataFrame, shape (n_trial_rows, n_columns)
        One row per injected trial and scored detector output: ``condition``
        (a ``*_event`` control event, a jump or a chain), ``level``,
        ``peak_rate_hz``, ``detector``, ``called``, the canonical output's
        ``continuous_share`` and ``fragmented_share``, and for multi-part
        trials ``n_parts``, ``n_recovered``, ``n_recovered_alone`` and
        ``segmentation``.
    no_replay : pd.DataFrame, shape (n_stretch_rows, n_columns)
        One row per immobile stretch and detector output: ``recording``,
        ``stretch``, ``exposure_s``, ``detector``, ``n_events`` and
        ``event_time_s``.

    Returns
    -------
    summary : pd.DataFrame, shape (n_values, 7)
        ``summary`` (``no_replay``, ``control_events`` or ``parts``), ``measure``,
        ``condition``, ``level`` (NaN for a value over every level),
        ``detector`` (empty for the immobile minutes), ``value`` and an empty
        ``note``. The no-replay exposure counts each stretch once; a part
        recovery is the mean over trials of each trial's recovered share, and
        the paired counts are the trials recovering fewer, or more, parts in
        context than alone.
    """
    rows = []

    def add(summary, measure, condition, level, detector, value):
        rows.append(
            {
                "summary": summary,
                "measure": measure,
                "condition": condition,
                "level": level,
                "detector": detector,
                "value": float(value),
            }
        )

    exposure_min = (
        no_replay.groupby(["recording", "stretch"]).exposure_s.first().sum() / 60
    )
    add("no_replay", "immobile_minutes", "no_replay", np.nan, "", exposure_min)
    for detector, group in no_replay.groupby("detector"):
        add(
            "no_replay", "n_events", "no_replay", np.nan, detector, group.n_events.sum()
        )
        add(
            "no_replay",
            "events_per_minute",
            "no_replay",
            np.nan,
            detector,
            group.n_events.sum() / exposure_min,
        )
        add(
            "no_replay",
            "time_share",
            "no_replay",
            np.nan,
            detector,
            group.event_time_s.sum() / (60 * exposure_min),
        )

    control_events = trial_table[trial_table.condition.str.endswith("_event")]
    for (condition, level, detector), group in control_events.groupby(
        ["condition", "level", "detector"]
    ):
        add(
            "control_events",
            "called_rate",
            condition,
            level,
            detector,
            group.called.mean(),
        )
        if detector == "state_space_canonical":
            for share in ("continuous_share", "fragmented_share"):
                add(
                    "control_events",
                    share,
                    condition,
                    level,
                    detector,
                    group[share].mean(),
                )
    for (condition, detector), group in control_events.groupby(
        ["condition", "detector"]
    ):
        add(
            "control_events",
            "called_rate",
            condition,
            np.nan,
            detector,
            group.called.mean(),
        )
        for rate_hz, at_rate in group.groupby("peak_rate_hz"):
            add(
                "control_events",
                f"called_rate_at_{rate_hz:g}hz",
                condition,
                np.nan,
                detector,
                at_rate.called.mean(),
            )

    multi = trial_table[~trial_table.condition.str.endswith("_event")]
    for (condition, level, detector), group in multi.groupby(
        ["condition", "level", "detector"]
    ):
        add(
            "parts",
            "part_recovery",
            condition,
            level,
            detector,
            (group.n_recovered / group.n_parts).mean(),
        )
        add(
            "parts",
            "part_recovery_alone",
            condition,
            level,
            detector,
            (group.n_recovered_alone / group.n_parts).mean(),
        )
        for label in SEGMENTATION_LABELS:
            add(
                "parts",
                f"segmentation_{label}",
                condition,
                level,
                detector,
                (group.segmentation == label).mean(),
            )
    for (condition, detector), group in multi.groupby(["condition", "detector"]):
        # paired by trial: pooled recovery can rise while some trials lose parts
        change = group.n_recovered - group.n_recovered_alone
        add("parts", "n_trials", condition, np.nan, detector, len(group))
        for measure, count in (
            ("n_trials_fewer_in_context", (change < 0).sum()),
            ("n_trials_more_in_context", (change > 0).sum()),
        ):
            add("parts", measure, condition, np.nan, detector, count)
    table = pd.DataFrame(rows)
    table["note"] = ""
    return table
