"""The events summary's event properties (part 2d): duration, extent, the two
speeds, certainty, stillness and location by SWR class, the state composition, the HPD
and content-distance values, the start distance and rest theta, and the nested
bootstrap intervals of the property contrasts. The statistics are computed in
``analysis.event_properties.event_statistics``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.constants import REMOTE_CM
from continuum_swr_replay.analysis.event_properties import event_statistics
from continuum_swr_replay.analysis.event_properties.property_contrast import (
    MAX_NAN_REPLICATE_SHARE,
)
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.events.common import (
    _RUN_GROUPS,
    EVENTS_SUBSET,
    TAGS,
    TIME_SUBSET,
    SummaryReads,
    _animal_range_values,
    _share_interval,
)
from continuum_swr_replay.reported.formatting import MS_PER_S, format_number
from continuum_swr_replay.reported.values import (
    Value,
    file_record,
    files_record,
    over,
    pend,
    rel,
    value_factory,
)


def state_composition_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.10: the mean share of an SWR or no-SWR event in Continuous and in
    Fragmented (Figure 3b)."""
    make = value_factory(state_composition_values, source, TAGS["state_composition"])
    statistics = event_statistics.state_composition_statistics(events)
    values = []
    for word, group in _RUN_GROUPS.items():
        for state in event_statistics.STATE_SHARE_COLUMNS:
            values.append(
                make(
                    f"{word}_{state}_share",
                    statistics.loc[group, f"{state}_animal_mean"],
                    "fraction",
                    "pct",
                    f"mean share of a {group} event in {state.capitalize()} (each "
                    "event weighted alike, whatever its length): the mean "
                    "over animals of each animal's mean over epochs of each epoch's "
                    "mean over events of the event's mean state probability",
                )
            )
    return values


#: (word, column, scale, unit, style, the expected direction for non-SWR against
#: SWR, tags)
_PROPERTIES = (
    (
        "duration",
        "event_duration_s",
        MS_PER_S,
        "ms",
        "int",
        "lower",
        TAGS["duration"],
    ),
    (
        "extent",
        "spatial_extent_cm",
        1.0,
        "cm",
        "sig2",
        "lower",
        TAGS["extent"],
    ),
    (
        "speed",
        "map_line_fit_speed_cm_s",
        1.0,
        "cm/s",
        "sig2",
        "lower",
        TAGS["speed"],
    ),
    (
        "transport_speed",
        "transport_slope_speed_cm_s",
        1.0,
        "cm/s",
        "sig2",
        "lower",
        TAGS["transport_speed"],
    ),
    (
        "peak",
        "posterior_peak_mean",
        1.0,
        "",
        "dec2",
        "higher",
        TAGS["certainty"],
    ),
    ("hpd", "hpd_coverage_cm", 1.0, "cm", "sig2", "lower", TAGS["certainty"]),
    (
        "pause",
        "pause_fraction",
        1.0,
        "",
        "dec2",
        "higher",
        TAGS["stillness"],
    ),
    (
        "max_step",
        "step_distance_max_cm",
        1.0,
        "cm",
        "sig2",
        "lower",
        TAGS["stillness"],
    ),
    (
        "reversals",
        "n_reversals",
        1.0,
        "",
        "int",
        "lower",
        TAGS["stillness"],
    ),
    (
        "multimodal",
        "multimodal_fraction",
        1.0,
        "",
        "dec2",
        "lower",
        TAGS["certainty"],
    ),
    (
        "start_distance",
        "start_distance_cm",
        1.0,
        "cm",
        "sig2",
        "lower",
        TAGS["location"],
    ),
    (
        "arms",
        "n_unique_arms",
        1.0,
        "",
        "int",
        "lower",
        TAGS["arms"],
    ),
)


_PROPERTY_NAMES = {
    "duration": "event duration",
    "extent": "spatial extent of the decoded content",
    "speed": "replay speed (the slope, at least 0, of the track distance from the "
    "event's first decoded position against time)",
    "transport_speed": "speed of the position distribution (the slope, at least 0, "
    "of the earth mover's distance on the track between the Continuous-state "
    "distributions of two time bins against their lag, 10 to 20 ms; over the "
    "events with one: an event too short or too Fragmented has none)",
    "peak": "posterior peak (mean over the event)",
    "hpd": "highest-density-region coverage",
    "pause": "pause fraction",
    "max_step": "largest step of the decoded position",
    "reversals": "number of direction reversals",
    "multimodal": "multimodal fraction of the posterior",
    "start_distance": "start distance (track-graph distance from the animal to the "
    "event's first decoded position)",
    "arms": "number of arms the decoded position visits",
}
"""How each property's values describe it."""


def property_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Questions 2d.1–2d.6: per-event properties, non-SWR against SWR.

    Plain medians of all events, and the number of animals whose own medians
    differ in the claim's direction. Not duration-controlled. The properties
    are the event table's own columns; the start distance and the two speeds,
    read from other outputs, are made by :func:`start_distance_values` and
    :func:`speed_values`.
    """
    values = []
    for prop in _PROPERTIES:
        if prop[0] not in ("start_distance", *_SPEED_PROPERTIES):
            values += _class_median_values(
                value_factory(property_values, source, prop[6]), events, prop
            )
    return values


def start_distance_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.1: the start distance of events, non-SWR against SWR, as
    :func:`property_values`; read from the start-distance output's
    ``start_distance_cm``."""
    (prop,) = (p for p in _PROPERTIES if p[0] == "start_distance")
    return _class_median_values(
        value_factory(start_distance_values, source, prop[6]), events, prop
    )


_SPEED_PROPERTIES = ("speed", "transport_speed")
"""The properties read from the event-speed output."""


def speed_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.2: how fast the events' content moves, non-SWR against SWR,
    as :func:`property_values`: the MAP line-fit speed (Figure 3d) and the
    transport-slope speed (the supplement's S3b), read from the event-speed
    output (``analysis/event_properties/event_speed.py`` has the two
    definitions)."""
    values = []
    for prop in _PROPERTIES:
        if prop[0] in _SPEED_PROPERTIES:
            values += _class_median_values(
                value_factory(speed_values, source, prop[6]), events, prop
            )
    return values


def _class_median_values(
    make: Callable[..., Value], events: pd.DataFrame, prop: tuple
) -> list[Value]:
    """One property's class medians and animals in direction, made by ``make``;
    ``prop`` is its row of :data:`_PROPERTIES`."""
    word, column, scale, unit, style, direction, _ = prop
    stats = event_statistics.class_medians(events, column)
    name = _PROPERTY_NAMES[word]
    return [
        make(
            f"{word}_median_non_swr",
            scale * stats["median_non_swr"],
            "quantity",
            style,
            f"median {name} of immobile non-SWR non-local events: the mean over "
            "animals of each animal's median (the median over its epochs of the "
            "epoch's median)",
            unit,
        ),
        make(
            f"{word}_median_swr",
            scale * stats["median_swr"],
            "quantity",
            style,
            f"median {name} of immobile SWR-overlapping non-local events: the "
            "mean over animals of each animal's median (the median over its "
            "epochs of the epoch's median)",
            unit,
        ),
        make(
            f"{word}_animals_in_direction",
            event_statistics.n_animals_in_direction(events, column, direction),
            "count",
            "count",
            f"animals whose non-SWR median {name} is {direction} than their SWR "
            "median, each the median over the animal's epochs of the epoch's median",
        ),
    ]


def hpd_tight_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.5 and the posterior-spread panel of Figure 2's supplement (S2k):
    the share of each class's events with a tight posterior, HPD coverage below
    ``event_statistics.HPD_TIGHT_CM``
    (``event_statistics.hpd_tight_share_statistics``), the animals in which the
    no-SWR share is the higher, and the cut. Each class holds a tight group of
    events and a diffuse one, so a median falls between them; the pooled class
    medians (``hpd_median_*``, :func:`property_values`) carry no figure tag."""
    make = value_factory(hpd_tight_values, source, TAGS["hpd"])
    result = event_statistics.hpd_tight_share_statistics(events)
    rows = result["rows"]
    cut = event_statistics.HPD_TIGHT_CM
    what = (
        f"share of the events whose HPD coverage, the length of track that holds "
        f"95 % of the decoded probability averaged over the event's bins, is below "
        f"{cut:g} cm"
    )
    reduction = (
        "each epoch's share, each animal's mean over its epochs, the mean over "
        "animals; 95 % interval from the nested bootstrap of the animals and "
        "their epochs (each replicate takes the same estimator), small-sample t "
        "correction"
    )
    values = [
        make(
            "hpd_tight_cut_cm",
            cut,
            "quantity",
            "int",
            "the HPD coverage below which an event's posterior counts as tight, "
            "between the tight and the diffuse group of events",
            "cm",
        )
    ]
    for group, word, name in (
        ("no SWR", "non_swr", "non-SWR"),
        ("SWR", "swr", "SWR-overlapping"),
    ):
        row = rows.loc[group]
        key = f"hpd_tight_share_{word}"
        values += _share_interval(
            make, key, row, f"{what}, of immobile {name} non-local events: {reduction}"
        )
    values.append(
        make(
            "hpd_tight_share_animals_non_swr_higher",
            result["n_animals_non_swr_higher"],
            "count",
            "count",
            "animals whose share of non-SWR events with a tight posterior (the "
            "mean over its epochs) is above their share of SWR events",
        )
    )
    return values


#: Each distance band's word in a key and what it holds
#: (``event_statistics.DISTANCE_BANDS_CM``).
_DISTANCE_BANDS = {
    "near": (
        "under",
        f"under {event_statistics.NEAR_CM:g} cm from the animal",
    ),
    "middle": (
        "between",
        f"{event_statistics.NEAR_CM:g} cm to under {REMOTE_CM:g} cm from the animal",
    ),
    "remote": ("beyond", f"{REMOTE_CM:g} cm or more from the animal"),
}


def content_distance_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.1 and the distance panel of Figure 3 (3g): for
    each class, the share of the events whose content lies under 20 cm, 20 to
    100 cm, and 100 cm or more from the animal (the mean over the event's bins
    of the absolute distance along the track of the decoded place from the
    animal's; ``event_statistics.content_distance_band_statistics``): the mean
    over animals of each animal's share with the lowest and highest animal,
    tagged to the panel; the headline share is the same number."""
    make = value_factory(content_distance_values, source, TAGS["content_distance"])
    table = event_statistics.content_distance_band_statistics(events)
    values = []
    for group, word, name in (
        ("no SWR", "non_swr", "non-SWR"),
        ("SWR", "swr", "SWR-overlapping"),
    ):
        for band, (band_word, where) in _DISTANCE_BANDS.items():
            row = table.loc[(group, band)]
            what = (
                f"share of the immobile {name} non-local events whose content "
                f"(the mean over the event's bins of the distance along the track "
                f"of the decoded place from the animal's) lies {where}"
            )
            values += _animal_range_values(
                make,
                f"content_distance_{band_word}_{word}_fraction",
                (row["share"], row["animal_min"], row["animal_max"]),
                (
                    f"{what}: the mean over animals of each animal's share (the "
                    "mean over its epochs of the epoch's share), the same number "
                    f"as content_distance_{band_word}_{word}_fraction_animal_mean",
                    f"lowest animal's {what} (an animal's share is the mean over "
                    "its epochs of the epoch's share)",
                    f"highest animal's {what} (the mean over its epochs)",
                ),
                animal_mean=(
                    row["share_animal_mean"],
                    f"{what}, the mean over animals of each animal's share (the "
                    "mean over its epochs of the epoch's share)",
                ),
                pooled_tags=TAGS["content_distance_pooled"],
            )
    values += [
        make(
            "content_distance_near_cm",
            event_statistics.NEAR_CM,
            "quantity",
            "int",
            "the distance of the content from the animal under which it counts as near",
            "cm",
        ),
        make(
            "content_distance_remote_cm",
            REMOTE_CM,
            "quantity",
            "int",
            "the distance of the content from the animal at or beyond which it "
            "counts as remote (the spike-evidence analysis's remote place)",
            "cm",
        ),
    ]
    return values


def property_interval_values(intervals: pd.DataFrame, source: str) -> list[Value]:
    """Questions 2d.1–2d.6: the uncertainty of each property contrast, and
    the contrast at matched duration.

    A nested bootstrap (animals, then epochs within animal, then events within
    epoch; 1000 replicates, fixed seed) of the SWR minus non-SWR difference of
    class medians, and of the duration-matched difference: within ten quantile
    strata of the pooled event durations, averaged by stratum size
    (``analysis.event_properties.property_contrast``). 95 % intervals: the estimate
    plus or minus a small-sample t correction over the animals times the
    replicates' standard deviation. The class medians' own intervals stay in
    the bootstrap's ``intervals.csv``.
    """
    if set(intervals.index) != {p[0] for p in _PROPERTIES}:
        raise ValueError(
            f"the bootstrap holds {sorted(intervals.index)}, not the events summary's properties"
        )
    nan_columns = [c for c in intervals.columns if c.endswith("_nan_replicates")]
    if not nan_columns:
        raise ValueError("the bootstrap does not count its replicates without estimate")
    if "n_animals" not in intervals.columns:
        raise ValueError(
            "the bootstrap does not record its animals: its intervals are percentile "
            "intervals, not the small-sample t rule"
        )
    share = intervals[nan_columns].div(intervals["n_replicates"], axis=0)
    if (share > MAX_NAN_REPLICATE_SHARE).any(axis=None):
        raise ValueError(f"too many bootstrap replicates without estimate:\n{share}")
    values: list[Value] = []
    for word, _, _, unit, _, _, tags in _PROPERTIES:
        make = value_factory(property_interval_values, source, tags)
        row = intervals.loc[word]
        name = _PROPERTY_NAMES[word]
        interval = (
            "95 % interval (nested bootstrap, small-sample t correction over the "
            f"{format_number(row.n_animals, 'int')} animals)"
        )
        skipped = (
            f"; {format_number(row.matched_replicates_skipping_strata, 'int')} of "
            f"{format_number(row.n_replicates, 'int')} replicates left out a stratum "
            "that drew no event of a class, "
            f"{format_number(row.matched_difference_nan_replicates, 'int')} had no "
            "estimate"
        )
        values += [
            make(
                f"{word}_difference_low",
                row.difference_low,
                "quantity",
                "sig2s",
                f"lower bound of the {interval} of the SWR minus non-SWR difference "
                f"of median {name} (each class's median the mean over animals of "
                "each animal's median over its epochs of the epoch's median)",
                unit,
            ),
            make(
                f"{word}_difference_high",
                row.difference_high,
                "quantity",
                "sig2s",
                "upper bound of that interval",
                unit,
            ),
        ]
        if word == "duration":
            continue
        values += [
            make(
                f"{word}_matched_difference",
                row.matched_difference,
                "quantity",
                "sig2s",
                f"SWR minus non-SWR difference of median {name} at matched duration "
                "(within ten duration-quantile strata of the pooled events, "
                "averaged by stratum size; pooled over events by design)",
                unit,
            ),
            make(
                f"{word}_matched_difference_low",
                row.matched_difference_low,
                "quantity",
                "sig2s",
                f"lower bound of its {interval}{skipped}",
                unit,
            ),
            make(
                f"{word}_matched_difference_high",
                row.matched_difference_high,
                "quantity",
                "sig2s",
                f"upper bound of its {interval}",
                unit,
            ),
        ]
    return values


def rest_theta_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.8: the theta-like share of events against duration-matched quiet rest.

    For each event, ten windows of quiet rest (below 4 cm/s, in no non-local,
    SWR or MUA event, outside excluded time) with as many bins as the event,
    drawn uniformly in its epoch; a window is theta-like when its mean theta
    reaches the epoch's median running theta, as an event is. The reference
    share of a class is the mean over its events of their windows' theta-like
    share, so it carries the class's durations. Every event must have its
    windows: an event whose length no rest stretch of its epoch holds would
    have none, and raises rather than leaving the comparison silently.
    """
    make = value_factory(rest_theta_values, source, TAGS["brain_state"])
    statistics = event_statistics.rest_matched_theta_statistics(events)
    values: list[Value] = []
    for group, is_swr in (("non_swr", False), ("swr", True)):
        stats = statistics.loc[is_swr]
        name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
        values += [
            make(
                f"theta_like_fraction_rest_matched_{group}",
                stats["rest_theta_like_fraction"],
                "fraction",
                "pct",
                f"share of quiet-rest windows, matched in duration to the {name} events, "
                "whose theta reaches the epoch's median running theta: each epoch's "
                "mean over its events, each animal's mean over its epochs, the mean "
                "over animals",
            ),
            make(
                f"theta_like_animals_above_rest_matched_{group}",
                stats["n_animals_above_rest"],
                "count",
                "count",
                f"animals whose share of theta-like {name} events is above that of "
                "their duration-matched quiet-rest windows, each the mean over the "
                "animal's epochs of the epoch's share",
            ),
        ]
    return values


def with_start_and_rest_theta(
    events: pd.DataFrame, folder: Path, n_epochs: int
) -> tuple[pd.DataFrame, list[Path]]:
    """``events`` with each one's start distance and rest-theta columns
    (``Processed-Data/event_start_rest_theta``), and the files read.

    Raises
    ------
    ValueError
        If ``folder`` does not hold ``n_epochs`` epochs, or an event has no row.
    """
    paths = sorted(folder.glob("*.parquet"))
    if len(paths) != n_epochs:
        raise ValueError(f"{folder} holds {len(paths)} epochs, not {n_epochs}")
    extras = pd.concat(map(pd.read_parquet, paths), ignore_index=True)
    joined = events.merge(
        extras.drop(columns=["epoch_id", "decode_sha256"]),
        on="event_id",
        how="left",
        validate="1:1",
    )
    if joined.n_bins.isna().any():
        raise ValueError("an event has no start-distance and rest-theta row")
    return joined, paths


EVENT_SPEED_COLUMNS = ("map_line_fit_speed_cm_s", "transport_slope_speed_cm_s")
"""The speeds read from the event-speed output
(``analysis/event_properties/event_speed.py``)."""


def with_event_speed(
    events: pd.DataFrame, folder: Path, n_epochs: int
) -> tuple[pd.DataFrame, list[Path]]:
    """``events`` with each one's :data:`EVENT_SPEED_COLUMNS`
    (``Processed-Data/event_speed``), and the files read.

    Raises
    ------
    ValueError
        If ``folder`` does not hold ``n_epochs`` epochs, or an event has no row
        or no MAP line-fit speed (the transport-slope speed may be missing: an
        event too short or too Fragmented has none).
    """
    paths = sorted(folder.glob("*.parquet"))
    if len(paths) != n_epochs:
        raise ValueError(f"{folder} holds {len(paths)} epochs, not {n_epochs}")
    speeds = pd.concat(
        (pd.read_parquet(p, columns=["event_id", *EVENT_SPEED_COLUMNS]) for p in paths),
        ignore_index=True,
    )
    joined = events.merge(speeds, on="event_id", how="left", validate="1:1")
    if joined[EVENT_SPEED_COLUMNS[0]].isna().any():
        raise ValueError("an event has no MAP line-fit speed")
    return joined, paths


def add_properties(reads: SummaryReads) -> None:
    """Add the event properties, start distance, rest theta and their intervals."""
    summary = reads.summary
    inputs = reads.inputs
    events = reads.events
    event_table_record = reads.event_table_record
    n_epochs = reads.n_epochs
    source = reads.source
    stale = reads.stale
    off_subset = reads.off_subset
    off_time_mask = reads.off_time_mask
    summary.add(*over(hpd_tight_values(events, source), EVENTS_SUBSET))
    summary.add(*over(content_distance_values(events, source), EVENTS_SUBSET))
    summary.add(*over(state_composition_values(events, source), EVENTS_SUBSET))
    summary.inputs.append(event_table_record)
    events_with_extras, extra_paths = with_start_and_rest_theta(
        events, inputs.event_start_rest_theta, n_epochs
    )
    extras_source = (
        f"{source} and {rel(inputs.event_start_rest_theta)} "
        "(scripts/event_properties/measure_event_start_and_rest_theta.py)"
    )
    extras_stale = stale(inputs.event_start_rest_theta)
    summary.add(
        *over(property_values(events_with_extras, extras_source), EVENTS_SUBSET)
    )
    summary.add(
        *over(
            pend(
                start_distance_values(events_with_extras, extras_source),
                extras_stale,
            ),
            EVENTS_SUBSET,
        )
    )
    summary.add(
        *over(
            pend(
                rest_theta_values(events_with_extras, extras_source),
                extras_stale,
                off_time_mask(
                    inputs.event_start_rest_theta, (path.stem for path in extra_paths)
                ),
            ),
            f"{EVENTS_SUBSET}, against quiet rest in {TIME_SUBSET}",
        )
    )
    summary.inputs += [
        files_record("*.parquet", extra_paths, ROOT_DIR),
        file_record(inputs.event_start_rest_theta / "manifest.json", ROOT_DIR),
    ]
    events_with_speed, speed_paths = with_event_speed(
        events, inputs.event_speed, n_epochs
    )
    summary.add(
        *over(
            pend(
                speed_values(
                    events_with_speed,
                    f"{source} and {rel(inputs.event_speed)} "
                    "(scripts/event_properties/measure_event_speed.py)",
                ),
                stale(inputs.event_speed),
                off_subset(inputs.event_speed),
            ),
            EVENTS_SUBSET,
        )
    )
    summary.inputs += [
        files_record("*.parquet", speed_paths, ROOT_DIR),
        file_record(inputs.event_speed / "manifest.json", ROOT_DIR),
    ]
    intervals = pd.read_csv(inputs.property_bootstrap / "intervals.csv", index_col=0)
    summary.add(
        *over(
            pend(
                property_interval_values(
                    intervals,
                    f"{rel(inputs.property_bootstrap)} "
                    "(scripts/event_properties/bootstrap_event_properties.py)",
                ),
                stale(inputs.property_bootstrap),
                off_subset(inputs.property_bootstrap),
            ),
            EVENTS_SUBSET,
        )
    )
    summary.inputs += [
        file_record(inputs.property_bootstrap / name, ROOT_DIR)
        for name in ("intervals.csv", "manifest.json")
    ]
