"""The events summary's checks of the method over immobile time (questions
2a.5, 2a.6, 2a.7): the time by non-local probability, the event rule's
sensitivity sweep, and the held-out state calibration. The statistics are
computed in ``analysis.decode_quality`` and ``analysis.event_properties``."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.decode_quality import nonlocal_probability
from continuum_swr_replay.analysis.event_properties import rule_sweep
from continuum_swr_replay.data_loaders.epoch_register import require_analyzed_epochs
from continuum_swr_replay.paths import ROOT_DIR, STATE_EVIDENCE_DIR, STATE_EVIDENCE_RUN
from continuum_swr_replay.reported.events.common import (
    EVENTS_SUBSET,
    TAGS,
    TIME_SUBSET,
    SummaryReads,
    _animal_range_values,
)
from continuum_swr_replay.reported.values import (
    Value,
    file_record,
    over,
    pend,
    rel,
    value_factory,
)
from continuum_swr_replay.runs.staleness import read_output_manifest


def nonlocal_probability_values(counts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.6 and Figure 2's supplement (S2c): how the immobile time
    inside the well visits is spread over the decoder's non-local probability,
    as three shares of the time, at most 0.2, between 0.2 and 0.8 (the event
    rule's cut), and above 0.8, each with the smallest and largest animal
    (``nonlocal_probability.probability_share_statistics``: each epoch's share,
    each animal's mean over its epochs, then the mean over animals)."""
    make = value_factory(
        nonlocal_probability_values, source, TAGS["nonlocal_probability"]
    )
    statistics = nonlocal_probability.probability_share_statistics(counts)
    reduction = (
        "each epoch's share, each animal's mean over its epochs, the mean over animals"
    )
    values = []
    for part, words in nonlocal_probability.SHARES.items():
        row = statistics.loc[part]
        what = (
            "share of the immobile time inside the well visits whose non-local "
            f"probability (Continuous plus Fragmented) is {words}"
        )
        values += _animal_range_values(
            make,
            f"nonlocal_probability_{part}_share",
            (row["animal_mean"], row["animal_min"], row["animal_max"]),
            (
                f"{what}: {reduction}",
                f"smallest animal's {what} (the mean over its epochs)",
                f"largest animal's {what} (the mean over its epochs)",
            ),
        )
    return values


#: Each swept setting's word in a key and in a description, and each value's
#: word in a key (which holds no digit).
_SWEEP_SETTINGS = {
    "probability_threshold": (
        "probability_threshold",
        "the event rule's non-local probability an event stays strictly above",
        dict(
            zip(
                rule_sweep.PROBABILITY_THRESHOLDS,
                ("half", "point_six", "point_seven", "point_eight", "point_nine",
                 "point_nine_five"),
                strict=True,
            )
        ),
    ),
    "minimum_samples": (
        "minimum_samples",
        "the event rule's minimum run length (2 ms decode samples; 7 samples, "
        "the rule's 15 ms, is a shortest event of 14 ms, and n samples one of "
        "2n ms)",
        dict(
            zip(
                rule_sweep.MINIMUM_SAMPLES,
                ("seven", "ten", "fifteen", "twenty_five", "thirty_eight", "fifty"),
                strict=True,
            )
        ),
    ),
    "swr_threshold_z": (
        "swr_threshold",
        "the SWR detector's threshold, not a setting of the event rule (z of the "
        "Kay consensus ripple power; an event is counted as overlapping an SWR "
        "when its Kay criterion on the decode grid is at or above it; the share "
        "with no SWR changes strongly with it, as a threshold on one "
        "distribution must)",
        dict(
            zip(
                rule_sweep.SWR_THRESHOLDS_Z,
                ("zero", "half", "one", "one_and_half", "two", "two_and_half",
                 "three", "three_and_half", "four", "four_and_half", "five",
                 "five_and_half", "six"),
                strict=True,
            )
        ),
    ),
}  # fmt: skip


def rule_sweep_values(counts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.5 and the sweep panel of Figure 2's supplement: the share of
    the immobile non-local events that start inside a well visit with no SWR,
    and the events against the rule's, at each setting of the event rule
    (``rule_sweep.sweep_statistics``, which the panel draws), and how often the
    Kay criterion's label at 2 z is the canonical one
    (``rule_sweep.criterion_agreement``). A sensitivity check: the canonical
    events and every other value are the rule's. The share with no SWR changes
    little with the event rule's two settings and strongly with the SWR
    detector's threshold, as a threshold on one distribution must (the text
    cites the threshold axis with the ripple-power panels); about a third of
    the events have no SWR even at 1 z."""
    make = value_factory(rule_sweep_values, source, TAGS["rule_sweep"])
    values = []
    for row in rule_sweep.sweep_statistics(counts).itertuples():
        setting_word, words, value_words = _SWEEP_SETTINGS[row.setting]
        value = int(row.value) if row.setting == "minimum_samples" else row.value
        key = f"rule_sweep_{setting_word}_{value_words[value]}"
        at = f"with {words} at {value:g}" + (
            " (the value the paper uses)" if row.is_rule else ""
        )
        values += [
            *_animal_range_values(
                make,
                f"{key}_no_swr_fraction",
                (row.no_swr_fraction, row.animal_min, row.animal_max),
                (
                    f"share of the events detected again {at} that overlap no SWR: "
                    "the mean over animals of each animal's share (the mean over its "
                    "epochs of the epoch's share), the same number as "
                    f"{key}_no_swr_fraction_animal_mean",
                    f"lowest per-animal share with no SWR {at} (an animal's share is the "
                    "mean over its epochs of the epoch's share)",
                    "highest per-animal share",
                ),
                animal_mean=(
                    row.no_swr_fraction_animal_mean,
                    f"share of the events detected again {at} that overlap no SWR, "
                    "the mean over animals of each animal's share (the mean over its "
                    "epochs of the epoch's share; the panel's bold line)",
                ),
                pooled_tags=TAGS["rule_sweep_pooled"],
            ),
            make(
                f"{key}_events_vs_rule",
                row.events_vs_rule,
                "quantity",
                "dec2",
                f"the events detected again {at}, as a multiple of the rule's",
            ),
            make(
                f"{key}_events",
                row.n_events,
                "count",
                "count",
                f"the events detected again {at}, summed over the analyzed epochs",
            ),
        ]
    values.append(
        make(
            "rule_sweep_swr_criterion_agreement",
            rule_sweep.criterion_agreement(counts),
            "fraction",
            "pct",
            "share of the events whose Kay criterion on the decode grid gives their "
            "canonical SWR label at the detector's 2 z (the stand-in by which the "
            "SWR threshold is swept, since the SWRs are not detected again): each "
            "epoch's share, each animal's mean over its epochs, the mean over animals",
        )
    )
    return values


#: Each calibrated state's word in a key, and what its evidence compares.
_CALIBRATED_STATES = {
    "Local": (
        "local",
        "the held-out tetrodes' spikes scored at the animal's place against at "
        "mismatched places (other windows' animal positions at least 20 cm away)",
    ),
    "Non-Local Continuous": (
        "continuous",
        "the place decoded from one half of the tetrodes scored on the other half "
        "against mismatched places (other windows' decoded places)",
    ),
    "Non-Local Fragmented": (
        "fragmented",
        "the place decoded from one half of the tetrodes scored on the other half "
        "against mismatched places (other windows' decoded places)",
    ),
}


_CALIBRATION_CAVEATS = (
    "the probability bins come from the decode with every tetrode, so the "
    "scored half's spikes helped make the state probability; only the remote "
    "place is held out by the tetrode split (one half chooses it, the other "
    "scores it); the place-field encodings are fit on running time in other "
    "time blocks (alternating 30 s blocks), so a window's own spikes are not in "
    "its encoding"
)


def state_calibration_values(
    bins: pd.DataFrame, trends: pd.DataFrame, source: str
) -> list[Value]:
    """Question 2a.7 and the state calibration panel of Figure 2's supplement:
    over the immobile calibration windows inside a well visit, binned by a
    state's probability, the held-out evidence for what the state claims, in
    the top and the bottom bins, and the rise over the bins (each epoch's
    slope), as ``state_evidence.calibration_statistics`` wrote them (the
    animal-level estimate and its nested bootstrap interval)."""
    make = value_factory(state_calibration_values, source, TAGS["state_calibration"])
    reduction = (
        "per epoch the median over its windows, then the mean over animals of "
        "each animal's median epoch value; 95 % interval from the nested bootstrap "
        "of the animals and their epochs (each replicate with the same estimator), "
        "small-sample t correction"
    )
    values = []
    for state, (word, what) in _CALIBRATED_STATES.items():
        mine = bins[bins["state"] == state].set_index("pbin")
        trend = trends[trends["state"] == state].iloc[0]
        key = f"state_calibration_{word}"
        for end, label, place in (
            ("top_bin", mine.index[-1], "the top"),
            ("bottom_bin", mine.index[0], "the bottom"),
        ):
            row = mine.loc[label]
            values.append(
                make(
                    f"{key}_{end}_per_spike",
                    row["animal_estimate"],
                    "quantity",
                    "sig2",
                    f"held-out evidence for {state} over the random 10-bin "
                    f"immobile windows whose {state} probability is in {place} bin "
                    f"({label}): {what}, the spike-mark log likelihood ratio per "
                    f"spike of the scored tetrodes (each run's ratio over the scored "
                    f"half's spikes, averaged over the runs in which that half "
                    f"spiked; a window in which it never spiked has no value); "
                    f"{reduction}. Caveats: {_CALIBRATION_CAVEATS}",
                    "nats per spike",
                )
            )
            if end == "top_bin":
                values += [
                    make(f"{key}_top_bin_per_spike_ci_low", row["animal_ci_lo"], "quantity",
                         "sig2", "lower bound of its 95 % interval", "nats per spike"),
                    make(f"{key}_top_bin_per_spike_ci_high", row["animal_ci_hi"], "quantity",
                         "sig2", "upper bound of that interval", "nats per spike"),
                ]  # fmt: skip
        values += [
            make(
                f"{key}_slope_per_spike",
                trend["slope_animal_estimate"],
                "quantity",
                "sig2",
                f"rise of that held-out evidence for {state} with the state's "
                "probability: each epoch's least-squares slope over its windows "
                "(nats per spike per unit of probability), then the mean over "
                "animals of each animal's median epoch slope",
                "nats per spike per unit probability",
            ),
            make(f"{key}_slope_per_spike_ci_low", trend["slope_animal_ci_lo"], "quantity",
                 "sig2", "lower bound of its 95 % interval (nested bootstrap)",
                 "nats per spike per unit probability"),
            make(f"{key}_slope_per_spike_ci_high", trend["slope_animal_ci_hi"], "quantity",
                 "sig2", "upper bound of that interval",
                 "nats per spike per unit probability"),
        ]  # fmt: skip
    first = bins[bins["state"] == next(iter(_CALIBRATED_STATES))]
    values.append(
        make(
            "state_calibration_windows",
            int(first["n"].sum()),
            "count",
            "count",
            "random 10-bin immobile calibration windows that start inside a well "
            "visit and have a held-out score for Local (a scored spike), over the "
            "analyzed "
            "epochs",
        )
    )
    return values


def add_validation(reads: SummaryReads) -> None:
    """Add the immobile time by non-local probability, the rule sweep and the
    held-out state calibration."""
    summary = reads.summary
    inputs = reads.inputs
    stale = reads.stale
    off_time_mask = reads.off_time_mask
    add_counts = reads.add_counts
    add_counts(
        inputs.nonlocal_probability,
        "scripts/decoding/measure_nonlocal_probability.py: each analyzed epoch's "
        "decode",
        (nonlocal_probability_values,),
        f"the immobile {TIME_SUBSET}",
    )
    add_counts(
        inputs.event_rule_sweep,
        "scripts/event_properties/sweep_event_rule.py: each analyzed epoch's "
        "decode, detected again at each setting",
        (rule_sweep_values,),
        EVENTS_SUBSET,
        stale(inputs.event_rule_sweep),
    )

    folder = inputs.state_calibration_in_visits
    calibration_epochs = list(read_output_manifest(folder)["epochs"])
    require_analyzed_epochs(calibration_epochs, str(folder))
    names = ("calibration_bins.csv", "calibration_trends.csv", "manifest.json")
    summary.add(
        *over(
            pend(
                state_calibration_values(
                    pd.read_csv(folder / names[0]),
                    pd.read_csv(folder / names[1]),
                    f"{rel(folder)} (scripts/event_evidence/"
                    "summarize_state_calibration_in_visits.py, from the held-out "
                    f"tetrode run {STATE_EVIDENCE_RUN}/{STATE_EVIDENCE_DIR})",
                ),
                off_time_mask(folder, calibration_epochs),
            ),
            f"the immobile {TIME_SUBSET}",
        )
    )
    summary.inputs += [file_record(folder / name, ROOT_DIR) for name in names]
