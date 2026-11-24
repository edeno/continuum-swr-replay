"""The events summary's decoder values (question 2a.1, 2a.6): the Local-state
decoding error, the calibration of that error against Local probability, and
the state fractions moving and immobile, over each analyzed epoch's decode. The
statistics are computed in ``analysis.decode_quality.quality_metrics``."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from continuum_swr_replay.analysis.decode_quality import quality_metrics
from continuum_swr_replay.data_loaders.epoch_register import require_analyzed_epochs
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.events.common import (
    TAGS,
    TIME_SUBSET,
    SummaryReads,
    _animal_range_values,
)
from continuum_swr_replay.reported.formatting import format_number
from continuum_swr_replay.reported.values import (
    Value,
    file_record,
    over,
    pend,
    rel,
    value_factory,
)

#: The ten equal Local-probability bins of the calibration curve, in order.
CALIBRATION_BINS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)


#: Each decoder state's word in a state-fraction key.
_STATE_WORDS = {
    "Local_prob": "local",
    "No-Spike_prob": "no_spike",
    "Non-Local_Continuous_prob": "continuous",
    "Non-Local_Fragmented_prob": "fragmented",
}


def decoding_error_values(sessions: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.1: the Local-state decoding error over sessions."""
    make = value_factory(decoding_error_values, source, TAGS["decoding_error"])
    stats = quality_metrics.local_decoding_error_statistics(sessions)
    return [
        make(
            "decoding_error_median_cm",
            stats["median_error_cm"],
            "quantity",
            "sig2",
            "each session's median Local-state decoding error (the track-graph "
            "distance from the decoded to the actual position, over time bins with "
            "Local probability above 0.5 outside excluded time), each animal's "
            "median over its sessions, then the mean over animals",
            "cm",
        ),
        make(
            "decoding_error_sessions",
            stats["n_sessions"],
            "count",
            "count",
            "sessions with a Local-state decoding error",
        ),
    ]


def calibration_values(calibration: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.1 and Figure 2's supplement: how far the decoded place is
    from the animal in each of ten Local-probability bins, over all analyzed
    time (``quality_metrics.animal_distance_by_local_probability``: each
    session's median distance between the animal and the maximum of the
    position estimate conditional on the non-local states, the median over an
    animal's sessions, then the mean over animals, with the smallest and
    largest animal). The keys keep their old name; the quantity is not a calibration."""
    make = value_factory(calibration_values, source, TAGS["calibration"])
    _, curve = quality_metrics.animal_distance_by_local_probability(calibration)
    if len(curve) != len(CALIBRATION_BINS):
        raise ValueError(
            f"the distance by Local probability has {len(curve)} bins, not "
            f"{len(CALIBRATION_BINS)}"
        )
    values = []
    for index, (word, (center, row)) in enumerate(
        zip(CALIBRATION_BINS, curve.iterrows(), strict=True), start=1
    ):
        where = (
            f"time bins whose Local probability is in the 0.1-wide bin centered on "
            f"{format_number(center, 'dec2')} (bin {index} of "
            f"{len(CALIBRATION_BINS)}), outside excluded time, moving and immobile"
        )
        what = (
            "each session's median distance along the track between the animal "
            "and the maximum of the position estimate conditional on the "
            f"non-local states, at {where}"
        )
        key = f"calibration_error_bin_{word}"
        values += _animal_range_values(
            make,
            key,
            (row["animal_mean"], row["animal_min"], row["animal_max"]),
            (
                f"mean over animals of each animal's median over its sessions of {what}",
                f"smallest animal's value of that mean over sessions of {what}",
                f"largest animal's value of that mean over sessions of {what}",
            ),
            "sig2",
            style="sig2",
            kind="quantity",
            unit="cm",
        )
    return values


def _state_fraction_values(
    make: Callable[..., Value], sessions: pd.DataFrame, movement: str, where: str
) -> list[Value]:
    """One movement state's four state fractions (``movement`` of
    ``quality_metrics.MOVEMENT_STATES``), made by ``make``."""
    fractions = quality_metrics.state_fraction_statistics(sessions)
    return [
        make(
            f"state_fraction_{movement}_{_STATE_WORDS[state]}",
            fractions.loc[movement, state],
            "fraction",
            "pct1",
            f"each session's mean {state} over its {where} outside excluded time, "
            "each animal's mean over its sessions, then the mean over animals",
        )
        for state in quality_metrics.STATE_PROBABILITY_COLUMNS
    ]


def state_fraction_values(sessions: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.6: the decoder's state fractions while the animal moves (the
    decoder-validation figure's bars)."""
    make = value_factory(state_fraction_values, source, TAGS["state_fractions"])
    return _state_fraction_values(make, sessions, "moving", "bins faster than 4 cm/s")


def immobile_state_fraction_values(sessions: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.6: the decoder's state fractions while the animal is still
    inside a well visit, beside Figure 2a, where everything else is inside the
    visits (the stored immobile fractions are over the immobile bins inside a
    visit, ``measure_decoder_quality.py``)."""
    make = value_factory(
        immobile_state_fraction_values, source, TAGS["state_fractions_immobile"]
    )
    return _state_fraction_values(
        make, sessions, "immobile",
        "bins slower than 4 cm/s inside a well visit (from the first poke-in at "
        "a well to the final poke-out)",
    )  # fmt: skip


def add_decoder(reads: SummaryReads) -> None:
    """Add the decoder's error, calibration and state fractions."""
    summary = reads.summary
    inputs = reads.inputs
    off_time_mask = reads.off_time_mask
    folder = inputs.decoder_quality
    quality_sessions = pd.read_parquet(folder / "sessions.parquet")
    calibration = pd.read_parquet(folder / "calibration.parquet")
    for name, quality_table in (
        ("sessions.parquet", quality_sessions),
        ("calibration.parquet", calibration),
    ):
        require_analyzed_epochs(quality_table["epoch_id"], str(folder / name))
    quality_source = (
        f"{rel(folder)} (scripts/decoding/measure_decoder_quality.py: each analyzed "
        "epoch's decode)"
    )
    summary.add(*decoding_error_values(quality_sessions, quality_source))
    summary.add(*calibration_values(calibration, quality_source))
    summary.add(*state_fraction_values(quality_sessions, quality_source))
    # the immobile fractions are over the time inside the well visits
    summary.add(
        *over(
            pend(
                immobile_state_fraction_values(quality_sessions, quality_source),
                off_time_mask(folder, quality_sessions["epoch_id"]),
            ),
            f"the immobile {TIME_SUBSET}",
        )
    )
    summary.inputs += [
        file_record(folder / name, ROOT_DIR)
        for name in ("sessions.parquet", "calibration.parquet", "manifest.json")
    ]
