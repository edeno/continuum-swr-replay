"""The events summary's held-out evidence (questions 2c.3, 2d.9, 2b.2): the
participation and likelihood-advantage runs and the held-out tetrode run (its
scores, the shared content, the events at low-rate places and those with a
resolved place). They share the participation run: the held-out tetrode values
are pending when it is. The statistics are computed in
``analysis.event_evidence`` and ``analysis.event_properties``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from continuum_swr_replay.analysis.event_definition import swr_source
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_events_in,
)
from continuum_swr_replay.analysis.event_evidence import (
    likelihood_advantage,
    participation,
    shared_content,
    tetrode_holdout,
)
from continuum_swr_replay.analysis.event_evidence.heldout_runs import (
    read_tetrode_holdout,
)
from continuum_swr_replay.analysis.event_properties import (
    event_statistics,
    low_rate_places,
    resolved_content,
)
from continuum_swr_replay.analysis.event_properties.content_place_rate import (
    LOW_RATE_RANK,
)
from continuum_swr_replay.analysis.regression.folds import MAX_SEGMENT_S, N_FOLDS
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.paths import (
    CONTENT_PLACE_RATE_NAME,
    PARTICIPATION_NAME,
    ROOT_DIR,
    TETRODE_HOLDOUT_NAME,
)
from continuum_swr_replay.reported.events.common import (
    _RUN_GROUPS,
    CONTROL_SUBSET,
    EVENTS_SUBSET,
    TAGS,
    SummaryReads,
    _interval_and_animals_positive,
    _per_spike_interval,
)
from continuum_swr_replay.reported.inputs import (
    input_staleness,
    participation_staleness,
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
from continuum_swr_replay.runs.io import files_sha256, ok_table_paths, read_complete_run
from continuum_swr_replay.runs.staleness import read_output_manifest


class ParticipationReads(NamedTuple):
    """What the restriction values read of the participation run."""

    analyzed: list[str]
    """The analyzed epochs with an event."""
    off: str | None
    """Why the run's controls are not Local rest inside the current well
    visits, or None."""
    made_from: Callable[[Path], str | None]
    """Why a run made from the participation tables is stale, or None."""


def participation_values(table: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.9: tetrode participation and spike rate of the events against
    Local-rest windows of their length (Figure 2h shows
    the participation)."""
    makers = {
        "participation": value_factory(
            participation_values, source, TAGS["participation"]
        ),
        "spike_rate_hz": value_factory(
            participation_values, source, TAGS["spike_rate"]
        ),
    }
    statistics = participation.participation_statistics(table)
    differences = participation.participation_differences(statistics)
    values = []
    for word, group in _RUN_GROUPS.items():
        for measure, style, what, unit in (
            ("participation", "pct", "share of tetrodes with a spike", ""),
            ("spike_rate_hz", "int", "spike rate summed over tetrodes", "spikes per s"),
        ):
            kind = "fraction" if measure == "participation" else "quantity"
            make = makers[measure]
            for window, label in (
                ("event", "events"),
                ("control", "Local-rest windows"),
            ):
                values.append(
                    make(
                        f"{word}_{window}_{measure}",
                        statistics.loc[(group, window), f"{measure}_animal_mean"],
                        kind,
                        style,
                        f"{what} of {group} {label}: mean over animals of each "
                        "animal's median over epochs of each epoch's median",
                        unit,
                    )
                )
            values.append(
                make(
                    f"{word}_{measure}_animals_above_rest",
                    differences.loc[group, f"{measure}_n_animals_above"],
                    "count",
                    "count",
                    f"animals whose {group} events' {what} is above their "
                    "Local-rest windows'",
                )
            )
        values.append(
            makers["participation"](
                f"{word}_participation_animals",
                differences.loc[group, "n_animals"],
                "count",
                "count",
                f"animals with {group} events and Local-rest windows",
            )
        )
    values.append(
        makers["participation"](
            "participation_events_without_control",
            int((table["kind"] == "event").sum() - (table["kind"] == "control").sum()),
            "count",
            "count",
            "events longer than any stretch of Local rest inside one well visit of "
            "their epoch (no control)",
        )
    )
    return values


def likelihood_advantage_values(table: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the spike likelihood at the decoded positions minus at the
    animal's, per spike, over each whole event (in sample; the held-out tetrode
    supplementary figure shows the held-out score)."""
    make = value_factory(
        likelihood_advantage_values, source, TAGS["likelihood_advantage"]
    )
    statistics = likelihood_advantage.likelihood_advantage_statistics(table)
    values = []
    for word, group in _RUN_GROUPS.items():
        row = statistics.loc[group]
        values += _per_spike_interval(
            make,
            f"{word}_likelihood_advantage_per_spike",
            row,
            f"log-likelihood of {group} events' spikes at the decoded positions "
            "minus at the animal's, summed over each event and divided by its "
            "spikes: mean over animals of each animal's median over epochs of "
            "each epoch's median",
        )
        values += [
            make(
                f"{word}_likelihood_advantage_events_scored",
                row["n_scored"],
                "count",
                "count",
                f"{group} events with a per-spike score (at least one spike)",
            ),
            make(
                f"{word}_likelihood_advantage_events",
                row["n_events"],
                "count",
                "count",
                f"{group} events scored",
            ),
        ]
    return values


_TETRODE_HELD_OUT = (
    "each tetrode is left out of the decoder in turn; at a scored bin, that "
    "tetrode's and the other tetrodes' place-field encodings are fit on the other "
    f"folds ({N_FOLDS} folds of whole trials balanced by "
    f"destination well, a segment longer than {MAX_SEGMENT_S:g} s "
    "cut into pieces, training bins within "
    f"{tetrode_holdout.TRIAL_MARGIN_S:g} s of the fold left out); the state "
    "transitions and initial conditions are the canonical decoder's, fit on the "
    "whole epoch with every tetrode; the posterior is smoothed over the whole "
    "epoch; the windows are the canonical events and their controls, chosen from "
    "the decode of every tetrode, so the choice of time is not held out"
)
"""How the held-out tetrode score is held out, in words
(``tetrode_holdout.holdout_scores``)."""


_CONTROL_AGREEMENT = (
    "; in the Local-rest controls the tetrodes share content, the animal's own "
    "place, so the controls are not a zero reference for the agreement (they are "
    "one for the gain, which is measured against local coding)"
)
"""What the controls' agreement values add to their description."""


def tetrode_holdout_values(windows: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the held-out tetrode gain over local coding and the
    agreement, per spike, of the events and their Local-rest controls. The
    events' gain is drawn on Figure 2j (the fit to the decoded place over the
    animal's place) and, with its parts in the bins with and without a spike,
    on Figure 2's supplement (S2h), and carries their tags; the controls' gain
    and the agreement are drawn on no figure."""
    make_figure = value_factory(tetrode_holdout_values, source, TAGS["tetrode_holdout"])
    make_split = value_factory(
        tetrode_holdout_values, source, TAGS["tetrode_holdout_split"]
    )
    make_text = value_factory(
        tetrode_holdout_values, source, TAGS["tetrode_holdout_text"]
    )
    values = []
    estimates = (
        ("gain", "gain", "event", "events"),
        ("spiking_gain", "spiking_gain", "event", "events"),
        ("silent_gain", "silent_gain", "event", "events"),
        ("control_gain", "gain", "control", "Local-rest controls"),
        ("agreement", "agreement", "event", "events"),
        ("control_agreement", "agreement", "control", "Local-rest controls"),
    )
    for name, measure, kind, label in estimates:
        column, words = _HELDOUT_MEASURES[measure]
        note = _CONTROL_AGREEMENT if (measure, kind) == ("agreement", "control") else ""
        if measure == "agreement" or kind == "control":
            make = make_text
        elif name == "gain":
            make = make_figure
        else:
            make = make_split
        statistics = tetrode_holdout.holdout_statistics(windows, kind, column)
        for word, group in _RUN_GROUPS.items():
            row = statistics.loc[group]
            values += _per_spike_interval(
                make,
                f"{word}_heldout_{name}_per_spike",
                row,
                f"held-out tetrode score of {group} {label}: {words}, summed over "
                "tetrodes and bins and divided by the window's spikes; "
                f"{_TETRODE_HELD_OUT}{note}; mean over animals of each animal's "
                "median over epochs of each epoch's median",
                # the controls' values are near zero: two significant digits
                style="sig2" if kind == "control" else "dec2",
            )
            if name == "gain":
                values.append(
                    make(
                        f"{word}_heldout_events_scored",
                        row["n_scored"],
                        "count",
                        "count",
                        f"{group} events with a held-out score per spike",
                    )
                )
    return values


#: Each width bin's word in a key (which holds no digit) and its range in words,
#: in the bins' order (``shared_content.WIDTH_BIN_LABELS``).
_WIDTH_BINS = dict(
    zip(
        shared_content.WIDTH_BIN_LABELS,
        (
            ("below_twenty_five_cm", "under 25 cm"),
            ("twenty_five_to_fifty_cm", "25 to 50 cm"),
            ("fifty_to_hundred_cm", "50 to 100 cm"),
            ("hundred_to_two_hundred_cm", "100 to 200 cm"),
            ("two_hundred_cm_or_more", "200 cm or more"),
        ),
        strict=True,
    )
)


_WIDTH_WORDS = (
    "the event's width: the mean over its left-out tetrodes of the mean over the "
    "window's scored bins of the 95 % HPD length of the other tetrodes' position "
    "distribution"
)


_REFERENCE_WORDS = (
    "the shuffled-event reference: the window's spikes scored under the other "
    "tetrodes' position distribution of another event window of the same group "
    "and fold (bins matched by offset from the window's start), minus under no "
    "place preference, summed over tetrodes and the bins scored and divided by "
    "their spikes; a window without a donor has none"
)


_UNMATCHED_DONOR = (
    "the donor is any event window of the same group and fold, not matched for "
    "the width of its estimate, so over all events the reference is a "
    "permutation reference; it is lower the wider the donor's estimate"
)


_MATCHED_DONOR = (
    "width-matched: only the windows whose donor's estimate (its 95 % HPD "
    "length, as the event's width) falls in the same width bin, a random subset "
    "of the bin's windows since the donor is drawn at random within the group "
    "and fold; the reference depends on the donor's width, not the window's"
)


#: Each held-out tetrode measure reported: its column of
#: ``tetrode_holdout.window_scores`` and the score in words.
_HELDOUT_MEASURES = {
    "gain": (
        "gain_per_spike",
        "each tetrode's log predictive score under the other tetrodes' position "
        "distribution minus under local coding around the animal",
    ),
    "spiking_gain": (
        "spiking_gain_per_spike",
        "each tetrode's log predictive score under the other tetrodes' position "
        "distribution minus under local coding around the animal in the bins "
        "where the scored tetrode spiked (its spikes' waveform terms with the "
        "expected-firing term of those bins); with the silent bins' part it adds "
        "to the gain for each event, but each of the three is a median over "
        "events, so over the cohort they need not add up",
    ),
    "silent_gain": (
        "silent_gain_per_spike",
        "each tetrode's log predictive score under the other tetrodes' position "
        "distribution minus under local coding around the animal in the bins "
        "where the scored tetrode was silent (the expected-firing term alone: "
        "the gain from the absence of spikes); with the spiking bins' part it "
        "adds to the gain for each event, but each of the three is a median over "
        "events, so over the cohort they need not add up",
    ),
    "agreement": (
        "agreement_per_spike",
        "the agreement, each tetrode's log predictive score under the other "
        "tetrodes' position distribution minus under no place preference (the "
        "mean over place bins): above zero when the other tetrodes' "
        "distribution predicts where the scored tetrode points better than no "
        "preference does, that is when the tetrodes share content, local or "
        "not (the gain over local coding is above zero for any non-local "
        "activity the distribution covers, coherent or not)",
    ),
    "gain_rate": (
        "gain_rate_per_spike",
        "the part of the gain over local coding that the firing-rate term gives "
        "(the expected spikes and the spike count, not the waveform marks)",
    ),
    "gain_mark": (
        "gain_mark_per_spike",
        "the part of the gain over local coding that only the waveform marks "
        "give (the gain minus its firing-rate part)",
    ),
    "agreement_rate": (
        "agreement_rate_per_spike",
        "the part of the agreement that the firing-rate term gives",
    ),
    "agreement_mark": (
        "agreement_mark_per_spike",
        "the part of the agreement that only the waveform marks give",
    ),
    "agreement_reference": (
        "agreement_null_per_spike",
        f"{_REFERENCE_WORDS}; {_UNMATCHED_DONOR}",
    ),
    "agreement_minus_reference": (
        "agreement_minus_null_per_spike",
        "the agreement minus the shuffled-event reference, event by event (an "
        f"event without a donor has no value); {_UNMATCHED_DONOR}",
    ),
    "mark_information": (
        "mark_information_per_spike",
        "the place information of the scored spikes' marks: for each spike, log "
        "of the places minus the entropy of the conditional density of its mark "
        "over place (the rate map divided out), under its fold's encoding, "
        "summed over the window's spikes and divided by the spikes counted",
    ),
    "spike_information": (
        "spike_information_per_spike",
        "the place information of the scored spikes: for each spike, log of the "
        "places minus the entropy of its whole per-spike term over place (rate "
        "map and mark together), under its fold's encoding, summed over the "
        "window's spikes and divided by the spikes counted",
    ),
}


def shared_content_values(windows: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: during an event, do the tetrodes point to
    the same remote place? Over the events of each SWR group by the width of
    the other tetrodes' position distribution in fixed bins
    (``shared_content.shared_content_statistics``, on the held-out tetrode
    run's ``windows``): the held-out tetrode agreement per spike, its
    width-matched shuffled-event reference and the agreement minus it event by
    event (over the windows whose donor's estimate is in the same width bin,
    ``shared_content.width_matched``), each with its interval, and the
    matched windows; the share of events whose agreement is above zero; and
    the share of the group's events in the bin, each animal's then their mean
    with the lowest and highest animal. No figure draws these by
    width (Figure 2j draws the difference over all widths, from
    :func:`tetrode_holdout_extra_values`); they are for the text."""
    make = value_factory(shared_content_values, source, TAGS["shared_content"])
    make_raw = value_factory(shared_content_values, source, TAGS["shared_content_raw"])
    make_text = value_factory(
        shared_content_values, source, TAGS["shared_content_text"]
    )
    statistics = shared_content.shared_content_statistics(
        windows[windows["kind"] == "event"]
    )
    _, words = _HELDOUT_MEASURES["agreement"]
    reduction = (
        "mean over animals of each animal's median over epochs of each epoch's median"
    )
    values = []
    for label, (bin_word, width) in _WIDTH_BINS.items():
        for word, group in _RUN_GROUPS.items():
            row = statistics.agreement.loc[(label, group)]
            reference = statistics.reference.loc[(label, group)]
            difference = statistics.difference.loc[(label, group)]
            key = f"{word}_shared_content_{bin_word}"
            where = f"{group} events whose width is {width} ({_WIDTH_WORDS})"
            values += _per_spike_interval(
                make_raw,
                f"{key}_agreement_per_spike",
                row,
                f"held-out tetrode agreement of the {where}: {words}, summed over "
                "tetrodes and bins and divided by the event's spikes; above zero "
                "only when the left-out tetrode's activity is where the other "
                f"tetrodes point; {_TETRODE_HELD_OUT}; {reduction}",
            )
            values += _per_spike_interval(
                make_raw,
                f"{key}_reference_per_spike",
                reference,
                f"{_REFERENCE_WORDS}, of the {where} whose donor's estimate is "
                f"in the same width bin ({_MATCHED_DONOR}); {_TETRODE_HELD_OUT}; "
                f"{reduction}",
            )
            values += _interval_and_animals_positive(
                make,
                f"{key}_agreement_minus_reference_per_spike",
                difference,
                "held-out tetrode agreement minus the shuffled-event reference, "
                f"event by event, of the {where} whose donor's estimate is in the "
                f"same width bin ({_MATCHED_DONOR}; {words}; {_REFERENCE_WORDS}); "
                f"{_TETRODE_HELD_OUT}; {reduction}",
            )
            values += [
                make(
                    f"{key}_share_positive",
                    row["share_positive"],
                    "fraction",
                    "pct",
                    f"share of the {where} whose agreement per spike is above zero: "
                    "per epoch, then each animal's mean over its epochs, then the "
                    "mean over animals",
                ),
                make(
                    f"{key}_share_positive_animal_min",
                    row["share_positive_animal_min"],
                    "fraction",
                    "pctnum",
                    "lowest of the animals' values of that share",
                ),
                make(
                    f"{key}_share_positive_animal_max",
                    row["share_positive_animal_max"],
                    "fraction",
                    "pct",
                    "highest of the animals' values of that share",
                ),
                make_text(
                    f"{key}_share_of_events",
                    row["share_of_events"],
                    "fraction",
                    "pct",
                    f"share of the {group} events with a held-out score per spike "
                    f"whose width is {width}: each epoch's share, each animal's mean "
                    "over its epochs, then the mean over animals (the same number as "
                    f"{key}_share_of_events_animal_mean)",
                ),
                make(
                    f"{key}_share_of_events_animal_mean",
                    row["share_of_events_animal_mean"],
                    "fraction",
                    "pct",
                    f"share of the {group} events with a held-out score per spike "
                    f"whose width is {width}: each epoch's share, each animal's mean "
                    "over its epochs, then the mean over animals",
                ),
                make(
                    f"{key}_share_of_events_animal_min",
                    row["share_of_events_animal_min"],
                    "fraction",
                    "pctnum",
                    "lowest of the animals' values of that share",
                ),
                make(
                    f"{key}_share_of_events_animal_max",
                    row["share_of_events_animal_max"],
                    "fraction",
                    "pct",
                    "highest of the animals' values of that share",
                ),
                make(
                    f"{key}_epochs",
                    row["n_epochs"],
                    "count",
                    "count",
                    f"epochs with {group} events whose width is {width}",
                ),
                make(
                    f"{key}_events",
                    row["n_scored"],
                    "count",
                    "count",
                    f"{group} events with a held-out score per spike whose width "
                    f"is {width}",
                ),
                make(
                    f"{key}_width_matched_events",
                    reference["n_scored"],
                    "count",
                    "count",
                    "of those, events with a width-matched shuffled-event "
                    "reference (a donor whose estimate is in the same width bin "
                    "and a spike in the bins it scored)",
                ),
            ]
    return values


def tetrode_holdout_extra_values(windows: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: what the held-out tetrode evidence of the events is made
    of, from the held-out tetrode run's extra scores (``windows``,
    ``tetrode_holdout.window_scores``): the gain over local coding and the
    agreement each split into their firing-rate and mark parts, the agreement's
    shuffled-event reference and the agreement minus it, and the place
    information of the scored spikes in the events, in their Local-rest
    controls and the event minus its control
    (``low_rate_places.event_minus_control``), by group, with the statistic of
    the neighboring held-out tetrode values. Figure 2j draws the agreement minus
    the reference (the fit to the decoded place over another event's place)
    and carries its tag; no figure draws the others."""
    make = value_factory(
        tetrode_holdout_extra_values, source, TAGS["tetrode_holdout_text"]
    )
    make_figure = value_factory(
        tetrode_holdout_extra_values, source, TAGS["tetrode_holdout_extra_figure"]
    )
    reduction = (
        "mean over animals of each animal's median over epochs of each epoch's median"
    )
    values = []
    for name in (
        "gain_rate",
        "gain_mark",
        "agreement_rate",
        "agreement_mark",
        "agreement_reference",
        "agreement_minus_reference",
        "mark_information",
        "spike_information",
    ):
        column, words = _HELDOUT_MEASURES[name]
        kinds = [("event", "events", "")]
        if name.endswith("information"):
            kinds.append(("control", "Local-rest controls", "control_"))
        for kind, label, prefix in kinds:
            statistics = tetrode_holdout.holdout_statistics(windows, kind, column)
            for word, group in _RUN_GROUPS.items():
                description = (
                    f"held-out tetrode score of {group} {label}: {words}; "
                    f"{_TETRODE_HELD_OUT}; {reduction}"
                )
                key = f"{word}_heldout_{prefix}{name}_per_spike"
                row = statistics.loc[group]
                if name == "agreement_minus_reference":
                    values += _interval_and_animals_positive(
                        make_figure, key, row, description
                    )
                else:
                    values += _per_spike_interval(make, key, row, description)
        if name.endswith("information"):
            paired = low_rate_places.event_minus_control(windows, column)
            statistics = tetrode_holdout.holdout_statistics(
                paired, "event", f"{column}_minus_control"
            )
            for word, group in _RUN_GROUPS.items():
                values += _interval_and_animals_positive(
                    make,
                    f"{word}_heldout_{name}_minus_control_per_spike",
                    statistics.loc[group],
                    f"{words}, in each {group} event minus in its Local-rest "
                    f"control (paired on the event); {_TETRODE_HELD_OUT}; "
                    f"{reduction}",
                )
    return values


#: The tenths of the place-rate rank, by their word in a key (which holds no
#: digit), lowest first.
_TENTHS = (
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
)  # fmt: skip


#: Each place group of ``low_rate_places.PLACES``: its word in a key and in text.
_PLACES = {
    "low_rate": ("low_rate", "decoded at a low-rate place"),
    "other": ("other_place", "decoded elsewhere"),
}


_PLACE_RATE_WORDS = (
    "an event's two places are the track's real places nearest its replay start "
    "and end positions, each ranked among the real places by the encoding "
    "model's summed ground intensity (0 the lowest rate); a low-rate event has "
    f"both places ranked below {LOW_RATE_RANK:g} (an event with one position is "
    "judged on that one)"
)


def low_rate_values(
    place_rate: pd.DataFrame,
    events: pd.DataFrame,
    windows: pd.DataFrame,
    source: str,
) -> list[Value]:
    """Question 2c.3: the events decoded at low-rate places
    (``low_rate_places.low_rate_statistics``). The share of each group's events
    that are low-rate events, each animal's then their mean, with the lowest
    and highest animal and the counts; the mean share in each tenth of the
    rank; the share of events with no SWR
    without the low-rate events (``event_statistics.overlap_class_statistics``
    of ``low_rate_places.events_outside_low_rate``); and the held-out tetrode
    scores of ``low_rate_places.LOW_RATE_MEASURES`` for low-rate and other
    events of each group (``windows``, the held-out tetrode run's). No
    figure draws these."""
    make_text = value_factory(low_rate_values, source, TAGS["tetrode_holdout_text"])
    statistics = low_rate_places.low_rate_statistics(windows, place_rate)
    values = []
    for word, group in _RUN_GROUPS.items():
        row = statistics.shares.loc[group]
        share = f"share of the {group} events that are low-rate events ({_PLACE_RATE_WORDS})"
        values += [
            make_text(
                f"{word}_low_rate_share_animal_mean",
                row["share_animal_mean"],
                "fraction",
                "pct",
                f"{share}: each epoch's share, each animal's mean over its epochs, "
                "then the mean over animals",
            ),
            make_text(
                f"{word}_low_rate_share_animal_min",
                row["share_animal_min"],
                "fraction",
                "pctnum",
                "lowest of the animals' values of that share",
            ),
            make_text(
                f"{word}_low_rate_share_animal_max",
                row["share_animal_max"],
                "fraction",
                "pct",
                "highest of the animals' values of that share",
            ),
            make_text(
                f"{word}_low_rate_events",
                row["n_low_rate_events"],
                "count",
                "count",
                f"{group} low-rate events",
            ),
            make_text(
                f"{word}_place_rate_ranked_events",
                row["n_events"],
                "count",
                "count",
                f"{group} events with a ranked place",
            ),
        ]
        for k, tenth in enumerate(_TENTHS):
            values.append(
                make_text(
                    f"{word}_place_rate_{tenth}_tenth_share",
                    statistics.tenths.loc[(group, k), "animal_mean"],
                    "fraction",
                    "pct",
                    f"share of the {group} events' places whose rank is in the "
                    f"{tenth} tenth from the lowest ({_PLACE_RATE_WORDS}; each "
                    "event counts its two places half each, its one place whole): "
                    "each epoch's share, each animal's mean over its epochs, then "
                    "the mean over animals; chance is the share of the track's "
                    "places in the tenth, about 10 %",
                )
            )
    outside = event_statistics.overlap_class_statistics(
        low_rate_places.events_outside_low_rate(events, place_rate)
    )
    values += [
        make_text(
            "non_swr_fraction_animal_mean_without_low_rate",
            outside["non_swr_fraction_animal_mean"],
            "fraction",
            "pct",
            "share of immobile non-local events overlapping no SWR, the mean over "
            "animals of each animal's share (the mean over its epochs of the "
            "epoch's share), over the events that are not low-rate events "
            f"({_PLACE_RATE_WORDS})",
        ),
        make_text(
            "nonlocal_events_without_low_rate",
            outside["n_events"],
            "count",
            "count",
            "immobile non-local events that are not low-rate events",
        ),
    ]
    reduction = (
        "mean over animals of each animal's median over epochs of each epoch's median"
    )
    names = {column: name for name, (column, _) in _HELDOUT_MEASURES.items()}
    for column in low_rate_places.LOW_RATE_MEASURES:
        name = names[column]
        words = _HELDOUT_MEASURES[name][1]
        make = make_text
        split = statistics.measures[column]
        for place, (place_word, place_words) in _PLACES.items():
            for word, group in _RUN_GROUPS.items():
                row = split.loc[(place, group)]
                key = f"{word}_{place_word}_heldout_{name}_per_spike"
                values += _interval_and_animals_positive(
                    make,
                    key,
                    row,
                    f"held-out tetrode score of the {group} events {place_words} "
                    f"({_PLACE_RATE_WORDS}): {words}; {_TETRODE_HELD_OUT}; "
                    f"{reduction}",
                )
                if name == "gain":
                    values.append(
                        make_text(
                            f"{word}_{place_word}_heldout_events_scored",
                            row["n_scored"],
                            "count",
                            "count",
                            f"{group} events {place_words} with a held-out score "
                            "per spike",
                        )
                    )
    return values


def resolved_content_values(
    events: pd.DataFrame, windows: pd.DataFrame, source: str
) -> list[Value]:
    """Question 2b.2: the share of events with no SWR among the events whose
    content is resolved (the other tetrodes' estimate under
    ``resolved_content.RESOLVED_WIDTH_CM`` wide, in the held-out tetrode run,
    ``windows``) and among the wider ones, by the rule
    of ``non_swr_fraction_animal_mean``; the share of events that are
    resolved; the events of each and those without a width
    (``resolved_content.resolved_share_statistics``). No figure draws these."""
    make = value_factory(resolved_content_values, source, TAGS["cohort_pooled"])
    statistics = resolved_content.resolved_share_statistics(events, windows)
    cut = f"{resolved_content.RESOLVED_WIDTH_CM:g} cm"
    rule = (
        "the mean over animals of each animal's share (the mean over its epochs "
        "of the epoch's share)"
    )
    width = (
        "the event's width: the mean over its left-out tetrodes and bins of the "
        "95 % HPD length of the other tetrodes' position estimate"
    )
    values = []
    for name, group, words in (
        ("resolved", "resolved", f"whose width is under {cut} ({width})"),
        ("wide", "wide", f"whose width is {cut} or more ({width})"),
    ):
        row = statistics[group]
        values += [
            make(
                f"non_swr_fraction_animal_mean_{name}",
                row["non_swr_fraction_animal_mean"],
                "fraction",
                "pct",
                f"share of the immobile non-local events {words} that overlap no "
                f"SWR: {rule}",
            ),
            make(
                f"non_swr_fraction_animal_min_{name}",
                row["non_swr_fraction_animal_min"],
                "fraction",
                "pctnum",
                "lowest of the animals' values of that share",
            ),
            make(
                f"non_swr_fraction_animal_max_{name}",
                row["non_swr_fraction_animal_max"],
                "fraction",
                "pct",
                "highest of the animals' values of that share",
            ),
            make(
                f"nonlocal_events_{name}",
                row["n_events"],
                "count",
                "count",
                f"immobile non-local events {words}",
            ),
        ]
    share = f"share of the immobile non-local events with a width whose width is under {cut} ({width})"
    values += [
        make(
            "resolved_share_animal_mean",
            statistics["resolved_share_animal_mean"],
            "fraction",
            "pct",
            f"{share}: {rule}",
        ),
        make(
            "resolved_share_animal_min",
            statistics["resolved_share_animal_min"],
            "fraction",
            "pctnum",
            "lowest of the animals' values of that share",
        ),
        make(
            "resolved_share_animal_max",
            statistics["resolved_share_animal_max"],
            "fraction",
            "pct",
            "highest of the animals' values of that share",
        ),
        make(
            "nonlocal_events_without_width",
            statistics["n_without_width"],
            "count",
            "count",
            "immobile non-local events without a width in the held-out tetrode "
            "run (left out of the shares by width)",
        ),
    ]
    return values


def add_tetrode_holdout(reads: SummaryReads) -> ParticipationReads:
    """Add the participation, likelihood-advantage and held-out tetrode values;
    return what the restriction values read of the participation run."""
    summary = reads.summary
    inputs = reads.inputs
    table = reads.table
    events = reads.events
    stale = reads.stale
    # every analyzed epoch with an event: an epoch with none has no table
    analyzed = swr_source.event_table_epochs(table, analyzed_epoch_ids())
    selected = events["event_id"]
    participation_table = select_events_in(
        read_complete_run(inputs.event_participation, PARTICIPATION_NAME, analyzed),
        selected,
    )
    advantage_table = select_events_in(
        read_complete_run(
            inputs.likelihood_advantage, "likelihood_advantage", analyzed
        ),
        selected,
    )
    # the held-out tetrode run, reduced to its windows at the read (no name
    # keeps the per-tetrode table)
    holdout_windows = tetrode_holdout.window_scores(
        read_tetrode_holdout(
            inputs.tetrode_holdout, analyzed, selected, rel(inputs.tetrode_holdout)
        )
    )
    # the place-rate run: every event of the table, read over the selected ones
    place_rate_table = select_events_in(
        read_complete_run(inputs.content_place_rate, CONTENT_PLACE_RATE_NAME, analyzed),
        selected,
    )
    participation_paths = ok_table_paths(inputs.event_participation, PARTICIPATION_NAME)
    participation_sha256 = files_sha256(participation_paths)

    # why the participation run's controls are not Local rest inside the
    # current well visits, or None (read once)
    participation_off = participation_staleness(
        read_output_manifest(inputs.event_participation),
        analyzed,
        rel(inputs.event_participation),
    )

    def made_from_participation(folder: Path) -> str | None:
        """Why a run made from the participation tables is stale, or None."""
        return input_staleness(
            read_output_manifest(folder),
            "event_participation",
            participation_sha256,
            rel(folder),
            "participation",
            "participation run",
        )

    summary.add(
        *over(
            pend(
                participation_values(
                    participation_table,
                    f"{rel(inputs.event_participation)} "
                    "(scripts/event_evidence/measure_event_participation.py)",
                ),
                stale(inputs.event_participation),
                participation_off,
            ),
            CONTROL_SUBSET,
        )
    )
    summary.add(
        *over(
            pend(
                likelihood_advantage_values(
                    advantage_table,
                    f"{rel(inputs.likelihood_advantage)} "
                    "(scripts/event_evidence/score_event_likelihood_advantage.py)",
                ),
                stale(inputs.likelihood_advantage),
                made_from_participation(inputs.likelihood_advantage),
            ),
            EVENTS_SUBSET,
        )
    )
    holdout_source = (
        f"{rel(inputs.tetrode_holdout)} "
        "(scripts/event_evidence/score_tetrode_holdout.py)"
    )
    holdout_stale = (
        stale(inputs.tetrode_holdout),
        made_from_participation(inputs.tetrode_holdout),
    )
    summary.add(
        *over(
            pend(
                tetrode_holdout_values(holdout_windows, holdout_source),
                *holdout_stale,
                participation_off,
            ),
            CONTROL_SUBSET,
        )
    )
    # the shared content: the events alone, by the width of the others' estimate
    summary.add(
        *over(
            pend(
                shared_content_values(holdout_windows, holdout_source), *holdout_stale
            ),
            EVENTS_SUBSET,
        )
    )
    # the parts of the evidence: the events and their controls
    summary.add(
        *over(
            pend(
                tetrode_holdout_extra_values(holdout_windows, holdout_source),
                *holdout_stale,
                participation_off,
            ),
            CONTROL_SUBSET,
        )
    )
    # the share with no SWR among the events with a resolved place
    summary.add(
        *over(
            pend(
                resolved_content_values(events, holdout_windows, holdout_source),
                *holdout_stale,
            ),
            EVENTS_SUBSET,
        )
    )
    # the events at low-rate places: the place-rate run and the held-out
    # tetrode run
    summary.add(
        *over(
            pend(
                low_rate_values(
                    place_rate_table,
                    events,
                    holdout_windows,
                    f"{rel(inputs.content_place_rate)} "
                    "(scripts/event_properties/measure_content_place_rate.py) and "
                    f"{holdout_source}",
                ),
                stale(inputs.content_place_rate),
                *holdout_stale,
            ),
            EVENTS_SUBSET,
        )
    )
    summary.inputs += [
        files_record(
            f"*_{TETRODE_HOLDOUT_NAME}.parquet",
            ok_table_paths(inputs.tetrode_holdout, TETRODE_HOLDOUT_NAME),
            ROOT_DIR,
        ),
        file_record(inputs.tetrode_holdout / "manifest.json", ROOT_DIR),
        files_record(
            f"*_{CONTENT_PLACE_RATE_NAME}.parquet",
            ok_table_paths(inputs.content_place_rate, CONTENT_PLACE_RATE_NAME),
            ROOT_DIR,
        ),
        file_record(inputs.content_place_rate / "manifest.json", ROOT_DIR),
        files_record(f"*_{PARTICIPATION_NAME}.parquet", participation_paths, ROOT_DIR),
        file_record(inputs.event_participation / "manifest.json", ROOT_DIR),
        files_record(
            "*_likelihood_advantage.parquet",
            ok_table_paths(inputs.likelihood_advantage, "likelihood_advantage"),
            ROOT_DIR,
        ),
        file_record(inputs.likelihood_advantage / "manifest.json", ROOT_DIR),
    ]
    return ParticipationReads(analyzed, participation_off, made_from_participation)
