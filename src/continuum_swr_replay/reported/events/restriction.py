"""The events summary's restriction values (question 2c.3): the local-only,
SWR-only, full and SWR-allowed decoders' held-out scores over the canonical
events' windows and their controls, over the events detected from the held-out
decode, and over time. The statistics are computed in
``analysis.event_evidence.restriction_windows``,
``analysis.event_evidence.restriction_totals`` and
``analysis.event_evidence.heldout_events``."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import pandas as pd

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_events_in,
)
from continuum_swr_replay.analysis.event_evidence import (
    heldout_events,
    restriction_totals,
    restriction_windows,
)
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    ALL_CONTRASTS,
    FAMILY_SUFFIX,
)
from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
    OUTSIDE_SWR_EVENT_WINDOW_GROUPS,
    OUTSIDE_SWR_EVENT_WINDOWS,
    RESTRICTION_FIGURE_EVENT_GROUP,
    RESTRICTION_FIGURE_FAMILY,
    RESTRICTION_FIGURE_INSIDE_SWR,
    RESTRICTION_FIGURE_OUTSIDE_EVENTS,
    RESTRICTION_FIGURE_OUTSIDE_SWR,
    SWR_ALLOWED_FIGURE_DECODERS,
    SWR_ALLOWED_FIGURE_DIFFERENCES,
    SWR_ALLOWED_TIME_CONTRASTS,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.analysis.regression.folds import MAX_SEGMENT_S
from continuum_swr_replay.paths import (
    CANONICAL_EVENTS_FOUND_NAME,
    HELDOUT_EVENTS_NAME,
    HELDOUT_TIME_SUMS_NAME,
    NONLOCAL_RESTRICTION_WINDOWS_NAME,
    ROOT_DIR,
)
from continuum_swr_replay.reported.events.common import (
    _RUN_GROUPS,
    CANONICAL_WINDOWS_SUBSET,
    EVENTS_SUBSET,
    HELDOUT_EVENTS_SUBSET,
    TAGS,
    TIME_SUBSET,
    SummaryReads,
    _interval_and_animals_positive,
    _share_interval,
)
from continuum_swr_replay.reported.events.tetrode_holdout import ParticipationReads
from continuum_swr_replay.reported.inputs import (
    read_heldout_restriction,
    restriction_inputs_note,
    run_staleness,
)
from continuum_swr_replay.reported.values import (
    Value,
    file_record,
    files_record,
    over,
    pend,
    rel,
    value_factory,
)
from continuum_swr_replay.runs.epoch_records import MANIFEST
from continuum_swr_replay.runs.io import ok_table_paths, read_complete_run
from continuum_swr_replay.runs.staleness import read_output_manifest


def window_run_note_staleness(manifest: dict, what: str) -> str | None:
    """Why the restriction window run's values are pending for its note, or None.

    The window run records a note (``restriction_windows.EVENT_TABLE_NOTE``, in
    its settings) when the event table it read is not the one the held-out
    tetrode run read, whose spike counts it adds; its windows still matched that
    run's window by window, but the two runs are not on one table.
    """
    note = manifest.get("settings", {}).get(restriction_windows.EVENT_TABLE_NOTE)
    if note is None:
        return None
    theirs = note["tetrode_holdout_event_table_sha256"]
    return (
        f"{what} read another event table (sha256 "
        f"{note['event_table_sha256'][:12]}) than the held-out tetrode run whose "
        f"spike counts it adds (sha256 {'none' if theirs is None else theirs[:12]}); "
        "rerun the held-out tetrode run and then the window run before citing"
    )


#: The three decoders of the restriction run (``nonlocal_restriction.MODELS``),
#: in words.
_RESTRICTION_MODELS = {
    "none": "the local-only decoder (Local and No-Spike states at every bin)",
    "swr": "the SWR-only decoder (only the non-local states in SWR bins, only the "
    "local states elsewhere)",
    "full": "the full decoder (every state at every bin)",
    "swr_allowed": "the SWR-allowed decoder (every state in SWR bins, only the "
    "local states elsewhere)",
}


#: The two score families (``nonlocal_restriction.FAMILY_SUFFIX``), in words.
_RESTRICTION_SCORES = {
    "filter": "one-step-ahead (filter) predictive log score of each 2 ms bin's spikes",
    "two_sided": "two-sided predictive log score of each 2 ms bin's spikes (from "
    "every other bin; summed, two-sided scores are not a log likelihood)",
}


_RESTRICTION_HELD_OUT = (
    "with each of five folds of whole trials (balanced by destination well; a "
    "trial's run to a well with its whole stay there, when longer than "
    f"{MAX_SEGMENT_S:g} s, is cut into pieces of at most "
    f"{MAX_SEGMENT_S:g} s that can fall in different folds) "
    "scored under a decoder whose place-field encodings were fit on the decode's "
    "training bins (outside every SWR and artifact) of the other folds, more than "
    "1 s from the fold, and whose state transitions and initial conditions were "
    "fit by EM with the fold's bins missing"
)


#: The window comparisons reported: kind of window, group, contrast.
_RESTRICTION_WINDOWS = (
    ("event", SWR_LABELS[False], "full_minus_swr"),
    ("event", SWR_LABELS[True], "swr_minus_none"),
    ("event", SWR_LABELS[True], "full_minus_none"),
    ("event", SWR_LABELS[True], "full_minus_swr"),
    ("control", SWR_LABELS[False], "full_minus_swr"),
    ("control", SWR_LABELS[True], "full_minus_swr"),
)


_RESTRICTION_KINDS = {
    "event": ("events", "events"),
    "control": ("controls", "matched Local-rest controls"),
}


#: Each unit of a restriction value: its word in a key, what the sum is divided
#: by, and the value's unit.
_RESTRICTION_UNITS = (
    ("per_spike", "their spikes", "nats per spike"),
    ("per_s", "their duration", "nats per s"),
)


#: Each group's bins outside SWRs, in words.
_OUTSIDE_SWR_WHERE = {
    SWR_LABELS[False]: "over each of the events with no SWR (its bins outside "
    "SWRs: all but a few inside an SWR the SWR-only decoder used)",
    SWR_LABELS[True]: "over the bins outside the SWR of each of the events that "
    "overlap an SWR",
}


#: The bins every value over the immobile time outside SWRs is summed over, in
#: words (``restriction_windows.outside_swr_bins``).
_OUTSIDE_SWR_BINS = (
    "immobile bins outside SWRs (outside every SWR of the source, where SWR "
    "detection was observed)"
)


#: How a value over time is reduced over epochs and animals, in words.
_EPOCH_REDUCTION = (
    "mean over animals of each animal's median over its epochs; 95 % interval "
    "from the nested bootstrap of the animals and their epochs, small-sample t "
    "correction over the animals"
)


#: Each SWR group's word in a key (the inverse of ``_RUN_GROUPS``).
_GROUP_WORDS = {group: word for word, group in _RUN_GROUPS.items()}


#: The full decoder's score minus the SWR-only decoder's, in words.
_FULL_MINUS_SWR = (
    f"under {_RESTRICTION_MODELS['full']} minus under {_RESTRICTION_MODELS['swr']}"
)


def _full_minus_swr_values(
    make: Callable[..., Value],
    key: str,
    row: pd.Series,
    unit: tuple[str, str, str],
    scored: str,
    time: str,
    epochs: str,
    positive: str = "epochs whose gain per second is above zero",
    difference: str = _FULL_MINUS_SWR,
) -> list[Value]:
    """The full minus the SWR-only decoder (or another ``difference``, in
    words) over ``time``, per ``unit`` (an entry of ``_RESTRICTION_UNITS``):
    :func:`_interval_and_animals_positive`, and per second also the epochs
    whose value is above zero (``positive``) and the epochs with a value
    (``epochs``, what they have)."""
    name, divided, words = unit
    values = _interval_and_animals_positive(
        make,
        f"{key}_{name}",
        row,
        f"{scored}, {difference}, summed over {time} and divided by "
        f"{divided}: {_EPOCH_REDUCTION}",
        unit=words,
    )
    if name == "per_s":
        values += [
            make(
                f"{key}_epochs_positive",
                row["n_epochs_positive"],
                "count",
                "count",
                positive,
            ),
            make(
                f"{key}_epochs",
                row["n_epochs"],
                "count",
                "count",
                f"epochs with {epochs}",
            ),
        ]
    return values


class WindowValues(NamedTuple):
    """The restriction window values, by what they read."""

    windows: list[Value]
    """The values that read the window run only."""
    after_visit: list[Value]
    """The counts of events by their bins after the final poke-out of their
    visit: of the window values, only these read the well visits (through the
    per-bin facts)."""


def restriction_window_values(windows: pd.DataFrame, source: str) -> WindowValues:
    """Question 2c.3: the three restriction decoders compared per spike over the
    canonical events of ``windows`` (those that start inside a well visit,
    ``cohort_selection.select_events_in``) and their Local-rest controls, in
    both score families, and the full decoder against the SWR-only decoder per
    spike and per second over the events' bins outside SWRs. No figure draws
    these, so none carries a figure's tag. Every key starts ``restriction_canonical_``, and the summary adds
    :data:`CANONICAL_WINDOWS_SUBSET` to every description."""
    make = value_factory(
        restriction_window_values, source, TAGS["nonlocal_restriction"]
    )
    # only the per-spike rows are read; each row's bootstrap is seeded on its
    # own, so the rows do not depend on which others are computed
    statistics = restriction_windows.restriction_window_statistics(
        windows, units=["per_spike"]
    ).set_index(["family", "contrast", "unit", "kind", "group"])
    values, after_visit = [], []
    for family, suffix in FAMILY_SUFFIX.items():
        for kind, group, contrast in _RESTRICTION_WINDOWS:
            row = statistics.loc[(family, contrast, "per_spike", kind, group)]
            kind_word, label = _RESTRICTION_KINDS[kind]
            a, b = restriction_windows.CONTRASTS[contrast]
            values += _interval_and_animals_positive(
                make,
                f"restriction_canonical_{_GROUP_WORDS[group]}_{kind_word}_{contrast}"
                f"{suffix}_per_spike",
                row,
                f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}, under "
                f"{_RESTRICTION_MODELS[a]} minus under {_RESTRICTION_MODELS[b]}, "
                f"summed over each of the {group} {label} and divided by its "
                "spikes: mean over animals of each animal's "
                "median over epochs of each epoch's median",
            )
    # full - swr per spike and per second over the events' bins outside SWRs
    outside = restriction_windows.restriction_window_statistics(
        windows,
        **{
            **OUTSIDE_SWR_EVENT_WINDOWS,
            "families": list(FAMILY_SUFFIX),
            "units": ["per_spike", "per_s"],
        },
    ).set_index(["family", "unit", "group"])
    counts = restriction_windows.outside_swr_window_counts(windows)
    a, b = restriction_windows.CONTRASTS["full_minus_swr"]
    for family, suffix in FAMILY_SUFFIX.items():
        for unit, divided, words in _RESTRICTION_UNITS:
            for group in OUTSIDE_SWR_EVENT_WINDOW_GROUPS:
                row = outside.loc[(family, unit, group)]
                word = _GROUP_WORDS[group]
                values += _interval_and_animals_positive(
                    make,
                    f"restriction_canonical_{word}_events_outside_swr_full_minus_swr"
                    f"{suffix}_{unit}",
                    row,
                    f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}, under "
                    f"{_RESTRICTION_MODELS[a]} minus under {_RESTRICTION_MODELS[b]}, "
                    f"summed {_OUTSIDE_SWR_WHERE[group]} and divided by {divided}: "
                    "mean over animals of each animal's median over epochs of each "
                    "epoch's median",
                    unit=words,
                )
                if family == RESTRICTION_FIGURE_FAMILY:
                    values.append(
                        make(
                            f"restriction_canonical_{word}_events_outside_swr_scored"
                            f"_{unit}",
                            row["n_scored"],
                            "count",
                            "count",
                            f"{group} events with a value "
                            + (
                                "per spike outside SWRs (immobile bins there and a "
                                "spike in them)"
                                if unit == "per_spike"
                                else "per second outside SWRs (immobile bins there)"
                            ),
                        )
                    )
    for group in OUTSIDE_SWR_EVENT_WINDOW_GROUPS:
        row = counts.loc[("event", group)]
        word = _GROUP_WORDS[group]
        values.append(
            make(
                f"restriction_canonical_{word}_events_outside_swr_without_spikes",
                row["n_without_spikes_outside_swr"],
                "count",
                "count",
                f"{group} events with immobile bins outside SWRs but no spike in "
                "them: no value per spike there",
            )
        )
        after_visit.append(
            make(
                f"restriction_canonical_{word}_events_outside_swr_after_visit",
                row["n_with_bins_outside_swr_outside_visit"],
                "count",
                "count",
                f"{group} events with immobile bins outside SWRs after the final "
                "poke-out of their visit (left out of "
                "the shares of immobile time inside a well visit)",
            )
        )
    # the windows and those with spikes are the same for every contrast and family
    for kind, (kind_word, label) in _RESTRICTION_KINDS.items():
        for word, group in _RUN_GROUPS.items():
            row = statistics.loc[("filter", "full_minus_swr", "per_spike", kind, group)]
            values += [
                make(
                    f"restriction_canonical_{word}_{kind_word}",
                    row["n_events"],
                    "count",
                    "count",
                    f"{group} {label} summed",
                ),
                make(
                    f"restriction_canonical_{word}_{kind_word}_scored",
                    row["n_scored"],
                    "count",
                    "count",
                    f"{group} {label} with a per-spike value (at least one spike)",
                ),
            ]
    values.append(
        make(
            "restriction_canonical_animals",
            row["n_animals"],
            "count",
            "count",
            "animals in the restriction comparison",
        )
    )
    return WindowValues(values, after_visit)


def restriction_heldout_event_values(windows: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the full decoder minus the SWR-only decoder per spike
    over the bins outside SWRs of the events detected from the held-out decode
    (``windows``, those that start inside a well visit), events with no SWR and
    events that overlap one, in :data:`RESTRICTION_FIGURE_FAMILY`
    (:data:`OUTSIDE_SWR_EVENT_WINDOWS`); with the events of each group, those with a
    value and those with bins outside SWRs but no spike in them. The gain per
    spike in the events with no SWR is drawn on Figure 2k's second axis
    (``restriction_totals.RestrictionFigureStatistics.events``) and carries its
    tag, the counts with it; the rest is for the text."""
    make_text = value_factory(
        restriction_heldout_event_values, source, TAGS["nonlocal_restriction"]
    )
    make_figure = value_factory(
        restriction_heldout_event_values, source, TAGS["nonlocal_restriction_figure"]
    )
    statistics = restriction_windows.restriction_window_statistics(
        windows, **OUTSIDE_SWR_EVENT_WINDOWS
    ).set_index("group")
    counts = restriction_windows.outside_swr_window_counts(windows)
    a, b = restriction_windows.CONTRASTS["full_minus_swr"]
    values = []
    for group in OUTSIDE_SWR_EVENT_WINDOW_GROUPS:
        row = statistics.loc[group]
        word = _GROUP_WORDS[group]
        make = make_figure if group == RESTRICTION_FIGURE_EVENT_GROUP else make_text
        values += _interval_and_animals_positive(
            make,
            f"restriction_heldout_{word}_events_outside_swr_full_minus_swr_per_spike",
            row,
            f"{_RESTRICTION_SCORES[RESTRICTION_FIGURE_FAMILY]}, "
            f"{_RESTRICTION_HELD_OUT}, under {_RESTRICTION_MODELS[a]} minus under "
            f"{_RESTRICTION_MODELS[b]}, summed {_OUTSIDE_SWR_WHERE[group]} and "
            "divided by their spikes: mean over animals of each animal's median "
            "over epochs of each epoch's median",
        )
        values += [
            make(
                f"restriction_heldout_{word}_events",
                row["n_events"],
                "count",
                "count",
                f"{group} events detected from the held-out decode",
            ),
            make(
                f"restriction_heldout_{word}_events_outside_swr_scored_per_spike",
                row["n_scored"],
                "count",
                "count",
                f"{group} events with a value per spike outside SWRs (immobile "
                "bins there and a spike in them)",
            ),
            make(
                f"restriction_heldout_{word}_events_outside_swr_without_spikes",
                counts.loc[("event", group), "n_without_spikes_outside_swr"],
                "count",
                "count",
                f"{group} events with immobile bins outside SWRs but no spike in "
                "them: no value per spike there",
            ),
        ]
    return values


def restriction_agreement_values(
    heldout: pd.DataFrame, canonical: pd.DataFrame, source: str
) -> list[Value]:
    """Question 2c.3: how the events detected from the held-out decode and the
    canonical events agree. Of the canonical events of ``canonical``, the share
    a held-out event shares a bin with; of the held-out events of ``heldout``,
    the share a canonical event shares a bin with; each over every event and
    by SWR group, the mean over animals and the lowest and highest animal
    (``heldout_events.found_again_statistics``); and the held-out events."""
    make = value_factory(
        restriction_agreement_values, source, TAGS["nonlocal_restriction"]
    )
    words = {"all": "", SWR_LABELS[False]: "_non_swr", SWR_LABELS[True]: "_swr"}
    values = []
    for events, found, stem, what, partner in (
        (
            canonical,
            "shares_bin_with_heldout",
            "restriction_canonical{}_events_found_again",
            "canonical events (found from the decode fit on all the data)",
            "an event detected from the held-out decode",
        ),
        (
            heldout,
            "shares_bin_with_canonical",
            "restriction_heldout{}_events_with_canonical",
            "events detected from the held-out decode",
            "a canonical event",
        ),
    ):
        statistics = heldout_events.found_again_statistics(events, found)
        for group, word in words.items():
            row = statistics.loc[group]
            key = stem.format(word)
            which = what if group == "all" else f"{group} {what}"
            values += [
                make(
                    key,
                    row["share"],
                    "fraction",
                    "pct",
                    f"share of the {which} that share at least one bin with "
                    f"{partner} (of any well-visit start): each epoch's share, each "
                    "animal's mean over its epochs, the mean over animals",
                ),
                make(
                    f"{key}_animal_min",
                    row["share_animal_min"],
                    "fraction",
                    "pctnum",
                    "lowest of the animals' shares, each animal's share the mean "
                    "over its epochs of the epoch's share",
                ),
                make(
                    f"{key}_animal_max",
                    row["share_animal_max"],
                    "fraction",
                    "pct",
                    "highest of the animals' shares, each animal's share the mean "
                    "over its epochs of the epoch's share",
                ),
            ]
    row = heldout_events.found_again_statistics(
        heldout, "shares_bin_with_canonical"
    ).loc["all"]
    values.append(
        make(
            "restriction_heldout_events",
            row["n_events"],
            "count",
            "count",
            "events detected from the held-out decode",
        )
    )
    return values


class OutsideSwrValues(NamedTuple):
    """The outside-SWR values, by what they are over."""

    with_events: list[Value]
    """The values over the time and the selected held-out events."""
    outside_events: list[Value]
    """The values over the time no held-out event holds: the held-out event run
    marks that time from every event detected from the held-out decode,
    wherever it starts, so these are over the time inside a well visit alone."""


def restriction_outside_swr_values(
    time_sums: pd.DataFrame, bin_s: pd.Series, windows: pd.DataFrame, source: str
) -> OutsideSwrValues:
    """Question 2c.3: the full decoder against the SWR-only decoder over the
    immobile time outside SWRs inside a well visit (time chosen by speed, ripple
    detection and the well visits, not by the decoder), per spike and per
    second, the share of it in the events of ``windows`` (the events detected
    from the held-out decode that start inside a well visit), and the rest
    outside every held-out event (``time_sums``' ``in_event_window``, which
    marks every held-out event, wherever it starts). The gain per spike and
    the three shares (in the no-SWR events, in the time outside the SWR of the
    events that overlap one, and the rest outside those events) in
    :data:`RESTRICTION_FIGURE_FAMILY` is drawn on Figure 2k and carries its
    tag, as does the gain per spike outside every held-out event (its second
    axis); the shares are for the text."""
    make_text = value_factory(
        restriction_outside_swr_values, source, TAGS["nonlocal_restriction"]
    )
    make_figure = value_factory(
        restriction_outside_swr_values, source, TAGS["nonlocal_restriction_figure"]
    )
    statistics = restriction_totals.outside_swr_statistics(time_sums, bin_s, windows)
    time = f"every one of the epoch's {_OUTSIDE_SWR_BINS}"
    values, outside_events = [], []
    for family, suffix in FAMILY_SUFFIX.items():
        key = f"restriction_outside_swr_full_minus_swr{suffix}"
        scored = f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}"
        figure = family == RESTRICTION_FIGURE_FAMILY
        for unit in _RESTRICTION_UNITS:
            name, divided, words = unit
            values += _full_minus_swr_values(
                make_figure
                if figure and f"gain_{name}" == RESTRICTION_FIGURE_OUTSIDE_SWR
                else make_text,
                key,
                statistics.loc[(family, f"gain_{name}")],
                unit,
                scored,
                time,
                _OUTSIDE_SWR_BINS,
            )
            row = statistics.loc[(family, f"outside_events_gain_{name}")]
            outside_events += _interval_and_animals_positive(
                make_figure
                if figure
                and f"outside_events_gain_{name}" == RESTRICTION_FIGURE_OUTSIDE_EVENTS
                else make_text,
                f"{key}_outside_heldout_events_{name}",
                row,
                f"{scored}, {_FULL_MINUS_SWR}, summed over the epoch's immobile bins "
                "outside SWRs that no event detected from the held-out decode "
                f"holds (wherever it starts) and divided by {divided}: "
                f"{_EPOCH_REDUCTION}",
                unit=words,
            )
        for measure, name, part, what in (
            (
                "share_in_swr_events_outside_swr",
                "share_in_heldout_swr_events_outside_swr",
                "the time outside the SWR of the events detected from the held-out "
                "decode that overlap an SWR",
                "a share: in the data every epoch's lies between 0 and 1",
            ),
            (
                "share_in_no_swr_events",
                "share_in_heldout_non_swr_events",
                "the events detected from the held-out decode with no SWR",
                "a ratio, not a share: it can lie outside 0 to 1 (above 1 when "
                "the full decoder does worse than the SWR-only decoder in the "
                "rest of the time)",
            ),
        ):
            values += _share_interval(
                make_text,
                f"{key}_{name}",
                statistics.loc[(family, measure)],
                f"{scored}; the gain is the score {_FULL_MINUS_SWR}. Per epoch, "
                f"the gain summed over {part} (their {_OUTSIDE_SWR_BINS} "
                "inside a well visit; an event that runs past the final poke-out "
                "adds its bins before it only), divided by the gain summed over "
                f"every one of the epoch's {_OUTSIDE_SWR_BINS} inside a well "
                "visit (no value when that sum is not positive). Both are signed "
                f"sums, so the two need not add to 1; this one is {what}. "
                f"Over the cohort: {_EPOCH_REDUCTION}",
            )
        # a ratio that is below 0 where the full decoder does worse: a
        # unitless quantity, as a fraction must lie in [0, 1]
        row = statistics.loc[(family, "share_outside_events")]
        values += [
            make_text(
                f"{key}_share_outside_heldout_events",
                row["estimate"],
                "quantity",
                "dec2",
                f"{scored}; the gain is the score {_FULL_MINUS_SWR}. Per epoch, 1 "
                "minus the ratio of the gain in the events detected from the "
                "held-out decode with no SWR and minus its share in the time "
                "outside the SWR of those that overlap an SWR, so the three add to "
                f"1: the ratio of the gain over the epoch's {_OUTSIDE_SWR_BINS} "
                "inside a well visit that lies outside those events to the whole "
                "gain (no value when the gain summed over those bins is not "
                "positive). A ratio, not a share: it can lie outside 0 to 1, below "
                "0 when the full decoder does worse than the SWR-only decoder "
                f"there. Over the cohort: {_EPOCH_REDUCTION}",
            ),
            make_text(
                f"{key}_share_outside_heldout_events_ci_low",
                row["ci_low"],
                "quantity",
                "dec2",
                "lower bound of its 95 % interval",
            ),
            make_text(
                f"{key}_share_outside_heldout_events_ci_high",
                row["ci_high"],
                "quantity",
                "dec2",
                "upper bound of that interval",
            ),
        ]
    return OutsideSwrValues(values, outside_events)


#: Each part of the time outside SWRs: its word in a key (which holds no digit)
#: and what it keeps, in words, by kind of part and threshold (s).
_OUTSIDE_SWR_PARTS = {
    "beyond_swr_margin": {
        threshold: (
            f"beyond_{word}",
            f"farther than {threshold * 1000:g} ms (on each side) from every bin "
            "of the scoring mask (every SWR of the source)",
        )
        for threshold, word in zip(
            restriction_totals.OUTSIDE_SWR_MARGINS_S,
            ("hundred_ms", "quarter_s", "half_s"),
            strict=True,
        )
    },
    "place_sampling": {
        threshold: (
            f"sampled_{word}",
            f"whose place was sampled for at least {threshold:g} s in the "
            "training data of the bin's own fold (the seconds of the decode's "
            "training bins outside the fold and its margin at the animal's place "
            "bin)",
        )
        for threshold, word in zip(
            restriction_totals.PLACE_SAMPLING_THRESHOLDS_S,
            ("ten_s", "minute"),
            strict=True,
        )
    },
}


def restriction_outside_swr_part_values(
    time_sums: pd.DataFrame, bin_s: pd.Series, source: str
) -> list[Value]:
    """Question 2c.3: the full decoder against the SWR-only decoder over parts
    of the immobile time outside SWRs inside a well visit
    (``restriction_totals.outside_swr_subset_statistics``): away from the SWRs
    (farther than a margin from every bin of the scoring mask) and where the
    animal's place was well sampled in training, per spike and per second,
    with the epochs above zero and the share of the bins each part keeps; and
    the immobile bins outside every SWR where SWR detection was not observed,
    which every value outside SWRs leaves out
    (``restriction_totals.outside_swr_not_observed_counts``). The gains per
    spike in :data:`RESTRICTION_FIGURE_FAMILY` away from the SWRs are drawn on
    Figure 2's supplement and carry its panel's tag (S2i); the gain at
    well-sampled places is for the text."""
    make = value_factory(
        restriction_outside_swr_part_values, source, TAGS["nonlocal_restriction"]
    )
    make_distance = value_factory(
        restriction_outside_swr_part_values,
        source,
        TAGS["nonlocal_restriction_distance"],
    )
    counts = restriction_totals.outside_swr_not_observed_counts(time_sums)
    values = [
        make(
            "restriction_outside_swr_not_observed_bins",
            counts["n_bins"],
            "count",
            "count",
            "immobile bins outside every SWR of the source where SWR detection "
            "was not observed (a gap in the stored ripple band): not known to be "
            "outside an SWR, so every value over the immobile time outside SWRs "
            "leaves them out",
        ),
        make(
            "restriction_outside_swr_not_observed_epochs",
            counts["n_epochs"],
            "count",
            "count",
            "epochs with immobile bins outside every SWR where SWR detection was "
            "not observed",
        ),
    ]
    statistics = restriction_totals.outside_swr_subset_statistics(
        time_sums, bin_s
    ).set_index(["subset", "threshold_s", "family", "measure"])
    shares = restriction_totals.outside_swr_subset_shares(time_sums, bin_s)
    for subset, parts in _OUTSIDE_SWR_PARTS.items():
        for threshold, (word, kept) in parts.items():
            time = f"the epoch's {_OUTSIDE_SWR_BINS} {kept}"
            for family, suffix in FAMILY_SUFFIX.items():
                key = f"restriction_outside_swr_{word}_full_minus_swr{suffix}"
                scored = f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}"
                for unit in _RESTRICTION_UNITS:
                    drawn = (
                        subset == "beyond_swr_margin"
                        and family == RESTRICTION_FIGURE_FAMILY
                        and unit[0] == "per_spike"
                    )
                    values += _full_minus_swr_values(
                        make_distance if drawn else make,
                        key,
                        statistics.loc[(subset, threshold, family, f"gain_{unit[0]}")],
                        unit,
                        scored,
                        time,
                        f"{_OUTSIDE_SWR_BINS} {kept}",
                    )
            row = shares.loc[(subset, threshold)]
            where = f"that lie {kept}" if subset == "beyond_swr_margin" else kept
            share = f"share of the epoch's {_OUTSIDE_SWR_BINS} {where}"
            values += [
                make(
                    f"restriction_outside_swr_{word}_share_of_bins",
                    row["estimate"],
                    "fraction",
                    "pct",
                    f"{share}: mean over animals of each animal's mean over its "
                    "epochs; no interval is given (a t interval of a share near 1 "
                    "can pass 1), the lowest animal's value is given beside it",
                ),
                make(
                    f"restriction_outside_swr_{word}_share_of_bins_animal_min",
                    row["animal_min"],
                    "fraction",
                    "pct",
                    f"lowest of the animals' values of the {share}, each animal's "
                    "the mean over its epochs",
                ),
            ]
    return values


_INSIDE_SWR_BINS = (
    "immobile bins inside SWRs (inside an SWR of the source, where the SWR-only "
    "decoder must use the non-local states and the full decoder may use any state)"
)


def restriction_inside_swr_values(
    time_sums: pd.DataFrame, bin_s: pd.Series, source: str
) -> list[Value]:
    """Question 2c.3: the full decoder against the SWR-only decoder over the
    immobile time inside SWRs inside a well visit, per spike and per second,
    with the statistic, interval and epoch counts of the gain outside SWRs
    (``restriction_totals.inside_swr_gain_statistics``). The gain per spike in
    :data:`RESTRICTION_FIGURE_FAMILY` is drawn on Figure 2k's first axis and
    carries its tag; the rest is for the text."""
    make_text = value_factory(
        restriction_inside_swr_values, source, TAGS["nonlocal_restriction"]
    )
    make_figure = value_factory(
        restriction_inside_swr_values, source, TAGS["nonlocal_restriction_figure"]
    )
    statistics = restriction_totals.inside_swr_gain_statistics(
        restriction_totals.epoch_inside_swr_gains(time_sums, bin_s)
    ).set_index(["family", "measure"])
    values = []
    for family, suffix in FAMILY_SUFFIX.items():
        key = f"restriction_inside_swr_full_minus_swr{suffix}"
        scored = f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}"
        for unit in _RESTRICTION_UNITS:
            drawn = (
                family == RESTRICTION_FIGURE_FAMILY
                and f"gain_{unit[0]}" == RESTRICTION_FIGURE_INSIDE_SWR
            )
            values += _full_minus_swr_values(
                make_figure if drawn else make_text,
                key,
                statistics.loc[(family, f"gain_{unit[0]}")],
                unit,
                scored,
                f"every one of the epoch's {_INSIDE_SWR_BINS} inside a well visit",
                f"{_INSIDE_SWR_BINS} inside a well visit",
                "epochs whose gain per second inside SWRs is above zero",
            )
    return values


def restriction_immobile_values(
    time_sums: pd.DataFrame, bin_s: pd.Series, source: str
) -> list[Value]:
    """Question 2c.3: the SWR-only and full decoders against the local-only
    decoder over all immobile time inside a well visit (time chosen by speed and
    the well visits, not by the decoder or the SWRs), per spike and per second,
    their difference, and the share of the full decoder's gain the SWR-only
    decoder captures. No panel of Figure 2 draws these; Figure 2's supplement,
    panel l, draws the SWR-only and full decoders' gains over the local-only
    decoder per spike (``restriction_totals.IMMOBILE_DECODER_GAINS``, in
    :data:`RESTRICTION_FIGURE_FAMILY`), and those values carry its tag."""
    make_text = value_factory(
        restriction_immobile_values, source, TAGS["nonlocal_restriction"]
    )
    make_figure = value_factory(
        restriction_immobile_values, source, TAGS["nonlocal_restriction_swr_allowed"]
    )
    statistics = restriction_totals.immobile_gain_statistics(
        restriction_totals.epoch_immobile_gains(time_sums, bin_s)
    ).set_index(["family", "measure"])
    values = []
    for family, suffix in FAMILY_SUFFIX.items():
        scored = (
            f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}, summed over "
            "every immobile bin of the epoch (inside and outside SWRs)"
        )
        for unit, divided, words in _RESTRICTION_UNITS:
            for contrast in ("swr_minus_none", "full_minus_none", "full_minus_swr"):
                measure = f"{contrast}_{unit}"
                a, b = restriction_windows.CONTRASTS[contrast]
                drawn = (
                    family == RESTRICTION_FIGURE_FAMILY
                    and unit == "per_spike"
                    and contrast in restriction_totals.IMMOBILE_DECODER_GAINS
                )
                values += _interval_and_animals_positive(
                    make_figure if drawn else make_text,
                    f"restriction_immobile_{contrast}{suffix}_{unit}",
                    statistics.loc[(family, measure)],
                    f"{scored}, under {_RESTRICTION_MODELS[a]} minus under "
                    f"{_RESTRICTION_MODELS[b]}, divided by {divided}: "
                    f"{_EPOCH_REDUCTION}",
                    unit=words,
                )
        row = statistics.loc[(family, "full_minus_swr_per_spike")]
        values.append(
            make_text(
                f"restriction_immobile_epochs_full_above_swr_only{suffix}",
                row["n_epochs_full_above_swr_only"],
                "count",
                "count",
                f"epochs whose summed score ({family}) over their immobile bins "
                "is higher under "
                f"{_RESTRICTION_MODELS['full']} than under "
                f"{_RESTRICTION_MODELS['swr']}",
            )
        )
        values += _share_interval(
            make_text,
            f"restriction_immobile_share_captured_by_swr_only{suffix}",
            statistics.loc[(family, "share_captured_by_swr_only")],
            f"{scored}: the gain of {_RESTRICTION_MODELS['swr']} over "
            f"{_RESTRICTION_MODELS['none']} as a share of the gain of "
            f"{_RESTRICTION_MODELS['full']} over it: {_EPOCH_REDUCTION}",
        )
    row = statistics.loc[(RESTRICTION_FIGURE_FAMILY, "full_minus_swr_per_spike")]
    values.append(
        make_text(
            "restriction_immobile_epochs",
            row["n_epochs"],
            "count",
            "count",
            "epochs with immobile bins in the restriction comparison",
        )
    )
    return values


#: Each time of ``restriction_totals.CONTRAST_TIMES``, in words: the bins a
#: contrast is summed over.
_CONTRAST_TIME_BINS = {
    "inside_swr": f"{_INSIDE_SWR_BINS} inside a well visit",
    "outside_swr": f"{_OUTSIDE_SWR_BINS} inside a well visit",
    "immobile": "immobile bins inside a well visit (inside SWRs, outside SWRs, "
    "and where SWR detection was not observed, which the sum outside SWRs "
    "leaves out)",
}


_SWR_ALLOWED_RUN = (
    "every decoder scored in one pass under the stored fold fits "
    "(scripts/event_evidence/score_nonlocal_restriction.py --from-stored-fit)"
)


def restriction_swr_allowed_values(
    windows: pd.DataFrame, time_sums: pd.DataFrame, bin_s: pd.Series, source: str
) -> list[Value]:
    """Question 2c.3: the SWR-allowed decoder (every state inside SWRs, only the
    local states outside them) against the other three, from the restriction
    run, which scores all four (``restriction_totals.swr_allowed_statistics``).

    Over the immobile time inside SWRs, outside SWRs and all of it inside a well
    visit (``restriction_totals.CONTRAST_TIMES``), each contrast of
    ``nonlocal_restriction.SWR_ALLOWED_CONTRASTS`` per spike and per second, in
    both score families, with the statistic, interval and epoch counts of the
    gain inside SWRs (keys ``restriction_<time>_<contrast>...``); and the full
    minus the SWR-allowed decoder per spike over the bins outside SWRs of the
    no-SWR events detected from the run's held-out decode (``windows``, those
    that start inside a well visit). The SWR-only and full decoders' gains over
    the local-only decoder are :func:`restriction_immobile_values`'. The
    values Figure 2's supplement draws on its fourth-decoder panel (``restriction_totals.SWR_ALLOWED_FIGURE_*``, in
    :data:`RESTRICTION_FIGURE_FAMILY` per spike) carry its tag (S2l); the rest
    is for the text."""
    make_text = value_factory(
        restriction_swr_allowed_values, source, TAGS["nonlocal_restriction"]
    )
    make_figure = value_factory(
        restriction_swr_allowed_values,
        source,
        TAGS["nonlocal_restriction_swr_allowed"],
    )
    statistics = restriction_totals.swr_allowed_statistics(windows, time_sums, bin_s)
    values = []
    for time, contrasts in SWR_ALLOWED_TIME_CONTRASTS.items():
        bins = _CONTRAST_TIME_BINS[time]
        drawn = (
            *SWR_ALLOWED_FIGURE_DIFFERENCES,
            *(SWR_ALLOWED_FIGURE_DECODERS if time == "immobile" else ()),
        )
        for contrast in contrasts:
            a, b = ALL_CONTRASTS[contrast]
            for family, suffix in FAMILY_SUFFIX.items():
                for unit in _RESTRICTION_UNITS:
                    measure = f"{contrast}_{unit[0]}"
                    figure = family == RESTRICTION_FIGURE_FAMILY and measure in drawn
                    values += _full_minus_swr_values(
                        make_figure if figure else make_text,
                        f"restriction_{time}_{contrast}{suffix}",
                        statistics.times.loc[(time, family, measure)],
                        unit,
                        f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}, "
                        f"{_SWR_ALLOWED_RUN}",
                        f"every one of the epoch's {bins}",
                        bins,
                        "epochs whose value per second is above zero",
                        difference=f"under {_RESTRICTION_MODELS[a]} minus under "
                        f"{_RESTRICTION_MODELS[b]}",
                    )
    group = RESTRICTION_FIGURE_EVENT_GROUP
    row = statistics.events.loc[group]
    word = _GROUP_WORDS[group]
    a, b = ALL_CONTRASTS[restriction_totals.SWR_ALLOWED_FIGURE_EVENT_CONTRAST]
    values += _interval_and_animals_positive(
        make_figure,
        f"restriction_heldout_{word}_events_outside_swr_full_minus_swr_allowed"
        "_per_spike",
        row,
        f"{_RESTRICTION_SCORES[RESTRICTION_FIGURE_FAMILY]}, "
        f"{_RESTRICTION_HELD_OUT}, {_SWR_ALLOWED_RUN}, under "
        f"{_RESTRICTION_MODELS[a]} minus under {_RESTRICTION_MODELS[b]}, summed "
        f"{_OUTSIDE_SWR_WHERE[group]} and divided by their spikes: mean over "
        "animals of each animal's median over epochs of each epoch's median",
    )
    return values


def add_restriction(reads: SummaryReads, participation: ParticipationReads) -> None:
    """Add the restriction decoders' values, over the canonical windows and over
    the events detected from the held-out decode and the time."""
    summary = reads.summary
    inputs = reads.inputs
    features_manifest = reads.features_manifest
    current_sha256 = reads.current_sha256
    stale = reads.stale
    analyzed = participation.analyzed
    participation_off = participation.off
    made_from_participation = participation.made_from
    selected = reads.events["event_id"]
    # the restriction scores over the canonical events' windows and their
    # controls: the window run sums the restriction run's bins and takes its
    # spike counts from the held-out tetrode run, so it must be made from both
    # runs as they are now, and on the held-out tetrode run's event table (the
    # window run's note when it is not; the event-table check above covers the
    # window run's own table)
    restriction_dir = inputs.nonlocal_restriction
    restriction_windows_dir = inputs.nonlocal_restriction_windows
    heldout_dir = inputs.nonlocal_restriction_heldout_events
    # the held-out event run (Figure 2k): its events that start inside a well
    # visit, its time sums and time steps, and why a value read from them is
    # pending; a restriction run of the earlier design is refused (the window
    # and held-out event runs must have read this run, staleness below)
    heldout = read_heldout_restriction(
        heldout_dir, restriction_dir, inputs.bin_facts, analyzed, current_sha256
    )
    # every restriction value reads the restriction run: pending when it was not
    # made from the current decode frames (as the event features run records
    # them) and SWR source
    restriction_inputs = restriction_inputs_note(
        restriction_dir, analyzed, rel(restriction_dir), features_manifest
    )
    # the run holds the windows of every event: those of the selected events
    # (and their controls, under their event's id) are read
    restriction_windows_table = select_events_in(
        read_complete_run(
            restriction_windows_dir, NONLOCAL_RESTRICTION_WINDOWS_NAME, analyzed
        ),
        selected,
    )
    windows_manifest = read_output_manifest(restriction_windows_dir)
    restriction_stale = (
        stale(restriction_windows_dir),
        made_from_participation(restriction_windows_dir),
        run_staleness(
            windows_manifest,
            restriction_windows_dir,
            "tetrode_holdout",
            inputs.tetrode_holdout,
        ),
        run_staleness(
            windows_manifest,
            restriction_windows_dir,
            "nonlocal_restriction",
            restriction_dir,
        ),
        run_staleness(
            windows_manifest, restriction_windows_dir, "bin_facts", inputs.bin_facts
        ),
        window_run_note_staleness(windows_manifest, rel(restriction_windows_dir)),
        restriction_inputs,
    )
    restriction_source = (
        f"{rel(restriction_windows_dir)} "
        "(scripts/event_evidence/sum_nonlocal_restriction_windows.py, from "
        f"{rel(restriction_dir)}, scripts/event_evidence/score_nonlocal_restriction.py, "
        f"and {rel(inputs.bin_facts)}, scripts/event_evidence/measure_bin_facts.py)"
    )
    # the windows' controls come from the participation tables the held-out
    # tetrode run read (sum_nonlocal_restriction_windows.check_holdout_inputs):
    # Local rest inside one well visit only when that run records the subset
    window_values = restriction_window_values(
        restriction_windows_table, restriction_source
    )
    window_stale = (*restriction_stale, participation_off)
    summary.add(
        *over(pend(window_values.windows, *window_stale), CANONICAL_WINDOWS_SUBSET)
    )
    # of the window values, only the counts after the visit read the visits
    # (through the per-bin facts)
    summary.add(
        *over(
            pend(window_values.after_visit, *window_stale, heldout.session_cache),
            CANONICAL_WINDOWS_SUBSET,
        )
    )

    # the canonical events found again, over the selected events
    found_table = select_events_in(
        read_complete_run(heldout_dir, CANONICAL_EVENTS_FOUND_NAME, analyzed),
        selected,
    )
    heldout_selected = heldout.events
    restriction_time_sums = heldout.time_sums
    restriction_bin_s = heldout.bin_s
    heldout_runs_stale = (
        heldout.restriction_run,
        heldout.bin_facts_run,
        restriction_inputs,
    )
    heldout_stale = (heldout.event_table, *heldout_runs_stale, heldout.session_cache)
    heldout_source = (
        f"{rel(heldout_dir)} (scripts/event_evidence/detect_heldout_events.py, from "
        f"{rel(restriction_dir)}, scripts/event_evidence/score_nonlocal_restriction.py, "
        f"and {rel(inputs.bin_facts)}, scripts/event_evidence/measure_bin_facts.py)"
    )
    summary.add(
        *over(
            pend(
                restriction_heldout_event_values(heldout_selected, heldout_source),
                *heldout_stale,
            ),
            HELDOUT_EVENTS_SUBSET,
        )
    )
    summary.add(
        *over(
            pend(
                restriction_agreement_values(
                    heldout_selected,
                    found_table,
                    f"{heldout_source} (its held-out events and the canonical "
                    "events found again)",
                ),
                *heldout_stale,
            ),
            f"{HELDOUT_EVENTS_SUBSET}, and {EVENTS_SUBSET}; whether an event "
            "shares a bin with an event of the other set is read against all of "
            "that set's events, so its partner may start outside a visit",
        )
    )
    outside_swr_values = restriction_outside_swr_values(
        restriction_time_sums,
        restriction_bin_s,
        heldout_selected,
        f"{heldout_source} (its held-out events and time sums)",
    )
    summary.add(
        *over(
            pend(outside_swr_values.with_events, *heldout_stale),
            f"{TIME_SUBSET}, and {HELDOUT_EVENTS_SUBSET}",
        )
    )
    # the time no event holds is outside every held-out event, selected or not:
    # those values are over the time alone
    summary.add(
        *over(pend(outside_swr_values.outside_events, *heldout_stale), TIME_SUBSET)
    )
    # over parts of the time outside SWRs, from the time sums alone: no event
    # enters
    summary.add(
        *over(
            pend(
                restriction_outside_swr_part_values(
                    restriction_time_sums,
                    restriction_bin_s,
                    f"{heldout_source} (its time sums)",
                ),
                *heldout_runs_stale,
                heldout.session_cache,
            ),
            TIME_SUBSET,
        )
    )
    # inside SWRs, from the time sums alone: no event enters
    summary.add(
        *over(
            pend(
                restriction_inside_swr_values(
                    restriction_time_sums,
                    restriction_bin_s,
                    f"{heldout_source} (its time sums)",
                ),
                *heldout_runs_stale,
                heldout.session_cache,
            ),
            TIME_SUBSET,
        )
    )
    # over immobile time only: no event enters
    summary.add(
        *over(
            pend(
                restriction_immobile_values(
                    restriction_time_sums,
                    restriction_bin_s,
                    f"{heldout_source} (its time sums)",
                ),
                *heldout_runs_stale,
                heldout.session_cache,
            ),
            TIME_SUBSET,
        )
    )
    # the fourth decoder (SWR-allowed) against the other three, from the same
    # runs
    summary.add(
        *over(
            pend(
                restriction_swr_allowed_values(
                    heldout.events,
                    heldout.time_sums,
                    heldout.bin_s,
                    f"{heldout_source[:-1]}; its held-out events and time sums)",
                ),
                *heldout_stale,
            ),
            f"{TIME_SUBSET}, and {HELDOUT_EVENTS_SUBSET}",
        )
    )
    summary.inputs += [
        file_record(restriction_dir / MANIFEST, ROOT_DIR),
        files_record(
            f"*_{NONLOCAL_RESTRICTION_WINDOWS_NAME}.parquet",
            ok_table_paths(restriction_windows_dir, NONLOCAL_RESTRICTION_WINDOWS_NAME),
            ROOT_DIR,
        ),
        file_record(restriction_windows_dir / MANIFEST, ROOT_DIR),
        *(
            files_record(
                f"*_{name}.parquet", ok_table_paths(heldout_dir, name), ROOT_DIR
            )
            for name in (
                HELDOUT_EVENTS_NAME,
                CANONICAL_EVENTS_FOUND_NAME,
                HELDOUT_TIME_SUMS_NAME,
            )
        ),
        file_record(heldout_dir / MANIFEST, ROOT_DIR),
        file_record(inputs.bin_facts / MANIFEST, ROOT_DIR),
    ]
