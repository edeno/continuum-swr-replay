"""The events summary's one-class-or-two values (question 2b.5): the peak
ripple power's distribution, the step tests, and the immobile time by ripple
power, by MUA and by decoder state. The statistics are computed in
``analysis.event_properties``."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.detector_comparison.feature_run import (
    STATE_SPACE_FEATURES_RUN,
)
from continuum_swr_replay.analysis.event_properties import (
    event_statistics,
    nonlocal_by_signal,
    state_by_signal,
)
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_animals
from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.events.common import (
    EVENTS_SUBSET,
    TAGS,
    TIME_SUBSET,
    SummaryReads,
    _animal_range_values,
)
from continuum_swr_replay.reported.inputs import input_staleness
from continuum_swr_replay.reported.values import (
    Value,
    file_record,
    over,
    pend,
    rel,
    value_factory,
)
from continuum_swr_replay.runs.staleness import read_output_manifest


def spike_evidence_staleness(
    manifest: dict, current_sha256: str, what: str
) -> str | None:
    """Why an output made from the spike-evidence run is stale, or None.

    ``current_sha256`` is the one digest over the run's per-epoch metrics
    (``runs.io.files_sha256``), which the output's manifest records under
    ``inputs.spike_evidence`` and this summary records for the same files.
    """
    return input_staleness(
        manifest,
        "spike_evidence",
        current_sha256,
        what,
        "spike-evidence",
        "spike-evidence run",
    )


#: The properties of the step test (``class_or_continuum.STEP_PROPERTIES``)
#: other than spike evidence, by their word in ``step_fits.csv``.
OTHER_STEP_PROPERTIES = {
    "duration": "duration",
    "extent": "spatial extent",
    "speed": "replay speed",
    "posterior": "posterior peak",
    "hpd": "HPD coverage",
    "spikes": "spike count",
    "breadth": "arms visited",
}


STEP_KNOTS = 8


STEP_WINDOW_Z = 1.5
"""The step test's main settings (``measure_ripple_class_structure.py``'s defaults)."""


def class_distribution_values(distribution: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.5, test 1: is the events' peak ripple power unimodal, and where is
    its mode against the Kay threshold?

    Per animal, Hartigan's dip test and the KDE's main mode of the event's peak
    ripple power (the z-scored Kay consensus on the decode grid). The dip
    p-values, the other power measures and the simulation checks stay in the
    analysis's CSV outputs. No figure draws these: the dip test has no power
    at the observed separation (the analysis's simulation); the step test
    (:func:`class_step_values`) is for the text too.
    """
    make = value_factory(class_distribution_values, source, TAGS["class_distribution"])
    stats = event_statistics.peak_ripple_mode_statistics(
        distribution, n_animals=len(analyzed_animals())
    )
    return [
        make(
            "ripple_peak_animals_unimodal",
            stats["n_animals_unimodal"],
            "count",
            "count",
            "animals in which Hartigan's dip test does not reject one mode of the "
            "events' peak ripple power (p at or above 0.05)",
        ),
        make(
            "ripple_peak_mode_animal_min",
            stats["mode_animal_min"],
            "quantity",
            "dec2",
            "lowest per-animal location of the main KDE mode of the events' peak "
            "ripple power",
            "z",
        ),
        make(
            "ripple_peak_mode_animal_max",
            stats["mode_animal_max"],
            "quantity",
            "dec2",
            "highest per-animal location of that mode",
            "z",
        ),
    ]


def nonlocal_power_values(counts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.5: how much of the non-local time has ripple power below the
    Kay threshold (Figure 2i)?

    Over the immobile time bins decoded non-local: each epoch's share, each
    animal's mean over its epochs, the mean over animals (``counts``: the
    stored per-epoch counts by signal and non-local probability,
    ``nonlocal_by_signal.rest_signal_counts``).
    """
    make = value_factory(nonlocal_power_values, source, TAGS["nonlocal_power"])
    stats = event_statistics.nonlocal_below_threshold_statistics(
        nonlocal_by_signal.animal_nonlocal_shares(counts, "ripple_power"),
        KAY_ZSCORE_THRESHOLD,
    )
    key = "nonlocal_time_below_swr_threshold_fraction"
    return [
        make(
            key,
            stats["fraction_animal_mean"],
            "fraction",
            "pct",
            "share of the immobile time bins decoded non-local (non-local "
            f"probability above {DEFAULT_NON_LOCAL_THRESHOLD:g}) whose ripple power "
            f"is below the Kay SWR threshold ({KAY_ZSCORE_THRESHOLD:g} z): each "
            "epoch's share of its bins, each animal's mean over its epochs, the mean "
            "over animals",
        ),
        make(
            f"{key}_animal_min",
            stats["fraction_animal_min"],
            "fraction",
            "pct",
            "lowest per-animal share of non-local time bins below the Kay threshold "
            "(the mean over its epochs)",
        ),
        make(
            f"{key}_animal_max",
            stats["fraction_animal_max"],
            "fraction",
            "pct",
            "highest per-animal share",
        ),
    ]


#: Each signal of ``state_by_signal``: its word in a key, its name in text and
#: its detection level in words (the level is
#: ``nonlocal_by_signal.DETECTION_LEVELS``').
_STATE_SIGNALS = {
    "ripple_power": ("ripple", "ripple power", "the SWR threshold"),
    "mua": ("mua", "MUA", "the MUA event threshold"),
}


#: Each measure of ``state_by_signal.detection_level_statistics``: its word in
#: a key, its style and what it is (``{signal}`` the signal, ``{level}`` the
#: detection level in words).
_STATE_MEASURES = {
    "nonlocal_share_at_level": (
        "nonlocal_share_at_detection_level", "pct",
        "share of the confident non-local time (non-local probability above 0.8) "
        "with {signal} at or above {level}",
    ),
    "local_share_at_level": (
        "local_share_at_detection_level", "pct1",
        "share of the confident Local time (Local probability above 0.8) with "
        "{signal} at or above {level}",
    ),
}  # fmt: skip


def state_by_signal_values(counts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.5: how much of the confident non-local time and of the
    confident Local time has ripple power, or MUA, at or above the detection
    level (the SWR threshold; the MUA event threshold), over the immobile time
    inside the well visits (``state_by_signal.detection_level_statistics``:
    each epoch's share, each animal's mean over its epochs, the mean over
    animals with the smallest and largest animal). No figure draws them (the
    top strips of Figure 2i and of its supplement's MUA panel draw the two
    distributions)."""
    make = value_factory(state_by_signal_values, source, TAGS["state_by_signal"])
    reduction = (
        "each epoch's share, each animal's mean over its epochs, the mean over animals"
    )
    values = []
    for signal, (word, name, level_words) in _STATE_SIGNALS.items():
        level = nonlocal_by_signal.DETECTION_LEVELS[signal]
        statistics = state_by_signal.detection_level_statistics(counts, signal, level)
        for measure, (key_word, style, what) in _STATE_MEASURES.items():
            row = statistics.loc[measure]
            text = what.format(signal=name, level=f"{level_words} ({level:g} z)")
            key = f"{word}_{key_word}"
            values += _animal_range_values(
                make,
                key,
                (row["animal_mean"], row["animal_min"], row["animal_max"]),
                (
                    f"{text}: {reduction}",
                    f"smallest animal's {text} (the mean over its epochs)",
                    f"largest animal's {text} (the mean over its epochs)",
                ),
                "pctnum" if style == "pct" else style,
                style=style,
            )
    return values


def nonlocal_mua_values(counts: pd.DataFrame, source: str) -> list[Value]:
    """Non-local time against MUA, the MUA panel of Figure 2's supplement: the
    share of the immobile time bins with MUA at or above
    ``nonlocal_by_signal.MUA_LEVEL_Z`` that are decoded non-local, and the
    share of the bins decoded non-local whose MUA is below it
    (``event_statistics.nonlocal_share_above_statistics`` and
    ``nonlocal_below_threshold_statistics`` of the stored counts, as the panel
    reads them). Most non-local time is below the MUA level, and nearly all
    time above it is decoded non-local; this does not show that non-local
    detection is independent of population bursts (the held-out tetrode values
    address whether the spikes point to the decoded place)."""
    make = value_factory(nonlocal_mua_values, source, TAGS["nonlocal_mua"])
    level = nonlocal_by_signal.MUA_LEVEL_Z
    above = event_statistics.nonlocal_share_above_statistics(
        nonlocal_by_signal.nonlocal_counts(counts, "mua"), level
    )
    below = event_statistics.nonlocal_below_threshold_statistics(
        nonlocal_by_signal.animal_nonlocal_shares(counts, "mua"), level
    )
    nonlocal_ = f"non-local probability above {DEFAULT_NON_LOCAL_THRESHOLD:g}"
    level_words = (
        f"{level:g} z of MUA (MUA_zscore, the z-scored multiunit rate of the decoded "
        "tetrodes; the value of the MUA event detector's threshold, which z-scores "
        "its own smoothed rate)"
    )
    key_above = "nonlocal_share_mua_above_level"
    key_below = "nonlocal_time_below_mua_level_fraction"
    return [
        make(
            "mua_level_z",
            level,
            "quantity",
            "int",
            f"the MUA level of the values below: {level_words}",
            "z",
        ),
        *_animal_range_values(
            make,
            key_above,
            (
                above["share_animal_mean"],
                above["share_animal_min"],
                above["share_animal_max"],
            ),
            (
                f"share of the immobile time bins with MUA at or above {level_words} "
                f"that are decoded non-local ({nonlocal_}): each epoch's share of its "
                "bins, each animal's mean over its epochs, the mean over animals. "
                "Nearly all time above the MUA level is "
                "decoded non-local: the decoder needs spikes, so this does not show "
                "that non-local detection is independent of population bursts",
                f"lowest per-animal share of the bins with MUA at or above {level:g} z "
                "that are decoded non-local (the mean over its epochs)",
                "highest per-animal share",
            ),
            lowest_style="pct",
        ),
        *_animal_range_values(
            make,
            key_below,
            (
                below["fraction_animal_mean"],
                below["fraction_animal_min"],
                below["fraction_animal_max"],
            ),
            (
                f"share of the immobile time bins decoded non-local ({nonlocal_}) whose "
                f"MUA is below {level_words}: each epoch's share of its bins, each "
                "animal's mean over its epochs, the mean over animals. Most non-local "
                "time is below the MUA level",
                f"lowest per-animal share of non-local time bins with MUA below {level:g} z "
                "(the mean over its epochs)",
                "highest per-animal share",
            ),
            lowest_style="pct",
        ),
    ]


def _step_statistics(steps: pd.DataFrame) -> pd.DataFrame:
    """The step test's statistics per property, at the main settings."""
    return event_statistics.threshold_step_statistics(
        steps,
        n_knots=STEP_KNOTS,
        window_z=STEP_WINDOW_Z,
        n_animals=len(analyzed_animals()),
    )


def class_step_values(steps: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.5 and the property questions, test 2: is there a step at the threshold?

    A sharp discontinuity design on the Kay criterion around each event (the
    level the ripple it touches sustains for 14 ms): per animal and property,
    ``y ~ s(criterion)`` against ``y ~ s(criterion) + (criterion >= 2 z)``
    (cubic B-spline, 8 knots) on the events within 1.5 z of the threshold. An
    animal has a step when the term's 95 % interval (clustered by epoch)
    excludes 0 and it lowers the held-out squared error (5 folds grouped by
    epoch). The share is the term over the animal's SWR minus non-SWR difference
    over all its events. The per-property counts, the version with log
    duration, the version on the event's own power, the sensitivity settings and
    the simulation checks stay in the CSV outputs.

    This is the value over the properties other than spike evidence, which are
    read from the event table and the event features run;
    :func:`class_step_evidence_values` holds spike evidence's. No figure draws
    them.
    """
    most, attaining = event_statistics.most_steps(
        _step_statistics(steps), list(OTHER_STEP_PROPERTIES)
    )
    # the note names the property (or properties) that attain the maximum
    attained = ", ".join(OTHER_STEP_PROPERTIES[p] for p in attaining)
    make = value_factory(class_step_values, source, TAGS["class_step"])
    return [
        make(
            "step_other_properties_criterion_animals_max",
            most,
            "count",
            "count",
            "the most animals with a step at the threshold in any property but spike "
            "evidence (duration, spatial extent, replay speed, posterior peak, HPD "
            "coverage, spike count, arms visited): the term's interval excludes 0 and "
            f"it lowers the held-out error; attained by {attained}",
        ),
    ]


def class_step_evidence_values(steps: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.5, test 2 (:func:`class_step_values`) for spike
    evidence per spike: the one property of the step test read from the
    spike-evidence run, so these values are stale when that run has changed
    since the step fits were made (:func:`spike_evidence_staleness`)."""
    evidence = _step_statistics(steps).loc["evidence"]
    make_evidence = value_factory(
        class_step_evidence_values, source, TAGS["class_step_evidence"]
    )
    return [
        make_evidence(
            "step_evidence_criterion_animals",
            evidence["n_animals_with_step"],
            "count",
            "count",
            "animals with a step at the threshold in spike evidence per spike (remote "
            "events): the term's interval excludes 0 and it lowers the held-out error",
        ),
        make_evidence(
            "step_evidence_criterion_animals_same_sign",
            evidence["n_animals_same_sign"],
            "count",
            "count",
            "animals whose jump in spike evidence per spike at the threshold points "
            "the way of their SWR minus non-SWR difference, whatever its interval",
        ),
        make_evidence(
            "step_evidence_criterion_share_median",
            evidence["step_share_median"],
            "quantity",
            "sig2s",
            "median over animals of the jump in spike evidence per spike at the "
            "threshold, as a share of the animal's SWR minus non-SWR difference",
        ),
    ]


def class_step_values_with_staleness(
    manifest: dict,
    spike_evidence_sha256: str,
    event_table_stale: str | None,
    steps: pd.DataFrame,
    source: str,
    what: str,
) -> list[Value]:
    """The step test's values (:func:`class_step_values` and
    :func:`class_step_evidence_values`), pending where the step fits are stale.

    Every value is pending with ``event_table_stale`` (the step fits were made
    from another event table); spike evidence's values are also pending when the
    step fits' ``manifest`` does not record the current spike-evidence run
    (``spike_evidence_sha256``, :func:`spike_evidence_staleness`; ``what`` names
    the step fits' folder in the note).
    """
    evidence_stale = spike_evidence_staleness(manifest, spike_evidence_sha256, what)
    return [
        *pend(class_step_values(steps, source), event_table_stale),
        *pend(
            class_step_evidence_values(steps, source), event_table_stale, evidence_stale
        ),
    ]


def add_class_structure(
    reads: SummaryReads, spike_evidence_record: dict[str, object]
) -> None:
    """Add the one-class-or-two values and the immobile time by ripple power, MUA
    and state."""
    summary = reads.summary
    inputs = reads.inputs
    stale = reads.stale
    off_subset = reads.off_subset
    add_counts = reads.add_counts
    folder = inputs.class_structure
    names = ("distribution.csv", "step_fits.csv", "manifest.json")
    source = (
        f"{rel(folder)} (scripts/event_properties/measure_ripple_class_structure.py: the "
        f"event table, event features run {STATE_SPACE_FEATURES_RUN}, the decode frames "
        "and the spike-evidence run)"
    )
    class_stale = (
        "; ".join(note for note in (stale(folder), off_subset(folder)) if note) or None
    )
    summary.add(
        *over(
            pend(
                class_distribution_values(
                    pd.read_csv(folder / "distribution.csv"), source
                ),
                class_stale,
            ),
            EVENTS_SUBSET,
        )
    )
    summary.add(
        *over(
            class_step_values_with_staleness(
                read_output_manifest(folder),
                spike_evidence_record["sha256"],
                class_stale,
                pd.read_csv(folder / "step_fits.csv"),
                source,
                rel(folder),
            ),
            EVENTS_SUBSET,
        )
    )
    summary.inputs += [file_record(folder / name, ROOT_DIR) for name in names]

    add_counts(
        inputs.nonlocal_by_signal,
        "scripts/event_properties/measure_nonlocal_by_signal.py: each analyzed "
        "epoch's decode",
        (nonlocal_power_values, nonlocal_mua_values),
        f"the immobile {TIME_SUBSET}",
    )
    add_counts(
        inputs.state_by_signal,
        "scripts/event_properties/measure_state_by_signal.py: each analyzed "
        "epoch's decode",
        (state_by_signal_values,),
        f"the immobile {TIME_SUBSET}",
    )
