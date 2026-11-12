"""The behavior summary's stop- and onset-table values: dwell by outcome,
survival, the outcome boundary and the onset times the rate docs quote. The
statistics are computed in ``analysis.behavior.rate.stop_statistics``."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.behavior.model_specs import (
    TMAX,
)
from continuum_swr_replay.analysis.behavior.rate import (
    stop_statistics,
)
from continuum_swr_replay.analysis.events.cohort_selection import (
    select_stop_cohort,
)
from continuum_swr_replay.reported.behavior.common import TAGS
from continuum_swr_replay.reported.values import (
    Value,
    value_factory,
)


def stop_table_values(
    stops: pd.DataFrame, onsets: pd.DataFrame, source: str
) -> list[Value]:
    """Stop- and onset-table facts the rate docs quote: dwell by outcome on the
    well-anchored cohort (the analysed immobility: the stop's span less the
    excluded time inside it), survival past 2 s and 4 s (from the stop's
    start, excluded time not taken out), the outcome-boundary shift
    (:func:`~continuum_swr_replay.analysis.behavior.rate.stop_statistics.outcome_boundary_statistics`), and the share of onsets
    inside the ``TMAX`` truncation."""
    make = value_factory(stop_table_values, source, TAGS["rate"])
    anchored, _ = select_stop_cohort(stops, cohort="at_well")
    dwell = stop_statistics.dwell_statistics(anchored)
    boundary = stop_statistics.outcome_boundary_statistics(stops)
    onset_times = stop_statistics.onset_time_statistics(onsets, TMAX)
    incomplete = stop_statistics.reward_history_incomplete_statistics(
        stops, onsets, "non_local"
    )
    analysed = "analysed immobility (the stop's span less the excluded time inside it)"
    values = [
        make(
            "stop_duration_mean_rewarded",
            dwell["mean_rewarded_s"],
            "quantity",
            "sig2",
            f"well-anchored stops with a rewarded outcome: mean {analysed}",
            "s",
        ),
        make(
            "stop_duration_mean_unrewarded",
            dwell["mean_omitted_s"],
            "quantity",
            "sig2",
            f"well-anchored stops with an omitted outcome: mean {analysed}",
            "s",
        ),
        make(
            "stop_duration_mean_ratio",
            dwell["mean_ratio"],
            "quantity",
            "sig2",
            "ratio of those means",
            "ratio",
        ),
        make(
            "stop_duration_median_rewarded",
            dwell["median_rewarded_s"],
            "quantity",
            "sig2",
            f"well-anchored stops with a rewarded outcome: median {analysed}",
            "s",
        ),
        make(
            "stop_duration_median_unrewarded",
            dwell["median_omitted_s"],
            "quantity",
            "sig2",
            f"well-anchored stops with an omitted outcome: median {analysed}",
            "s",
        ),
        make(
            "stop_duration_median_ratio",
            dwell["median_ratio"],
            "quantity",
            "sig2",
            "ratio of those medians",
            "ratio",
        ),
    ]
    values += [
        make(
            f"stop_survival_{word}_s",
            stop_statistics.still_immobile_fraction(anchored, at_s),
            "fraction",
            "pct",
            f"share of well-anchored stops still immobile {at_s} s after they began "
            f"(stop_end - stop_start at least {at_s} s; excluded time inside the "
            "stop is not taken out)",
        )
        for word, at_s in (("two", 2), ("four", 4))
    ]
    values += [
        make(
            "stops_in_transit",
            stop_statistics.stop_cohort_counts(stops)["n_in_transit"],
            "count",
            "count",
            "in-transit stops in the canonical stop table",
        ),
        make(
            "stops_crossing_outcome",
            boundary["n_crossing"],
            "count",
            "count",
            "stops with stop_start < outcome_known_time_s <= stop_end (all stops)",
        ),
        make(
            "stops_crossing_outcome_fraction",
            boundary["crossing_fraction"],
            "fraction",
            "pct",
            "their share of all stops",
        ),
        make(
            "stops_outcome_shifted",
            boundary["n_shifted"],
            "count",
            "count",
            "crossing stops whose per-bin reward history can be shifted: no later "
            "arrival inside, and a bin at or past the boundary in the stop-level "
            "design (before a fitted population's bins inside its ongoing events "
            "are dropped, so an upper bound for each fitted population)",
        ),
        make(
            "stops_outcome_shifted_fraction",
            boundary["shifted_fraction"],
            "fraction",
            "pct",
            "their share of all stops",
        ),
        make(
            "stops_spanning_arrival",
            boundary["n_spanning_arrival"],
            "count",
            "count",
            "stops containing a later well arrival",
        ),
        make(
            "stops_spanning_arrival_crossing",
            boundary["n_spanning_arrival_crossing"],
            "count",
            "count",
            "of those, stops that also cross their outcome boundary",
        ),
    ]
    for word, event_type in (
        ("non_local", "non_local"),
        ("swr", "SWR"),
        ("mua", "MUA"),
    ):
        values.append(
            make(
                f"onsets_within_tmax_fraction_{word}",
                onset_times.loc[event_type, "within_window_fraction"],
                "fraction",
                "pct",
                f"share of {event_type} onsets within the first {TMAX:g} s of their stop (event_onset_table)",
            )
        )
    values += [
        make(
            "stops_reward_history_incomplete_fraction",
            incomplete["stop_fraction"],
            "fraction",
            "sig2",
            "share of stops flagged reward_history_incomplete (fewer than two known outcomes)",
        ),
        make(
            "onsets_reward_history_incomplete_fraction_non_local",
            incomplete["onset_fraction"],
            "fraction",
            "sig2",
            "share of non-local onsets in those stops",
        ),
    ]
    values.append(
        make(
            "onset_median_s_non_local",
            onset_times.loc["non_local", "median_s"],
            "quantity",
            "sig2",
            "median time of a non-local onset after its stop's speed-threshold crossing",
            "s",
        )
    )
    return values
