"""The mpfc summary's timing values: when the responses of the called units
peak and start, and the timing calibration's checks. The statistics are
computed in ``analysis.pfc.timing``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc import population_statistics, timing
from continuum_swr_replay.analysis.pfc.load_population import (
    CALLED_ALPHA as ALPHA,
)
from continuum_swr_replay.reported.mpfc.common import (
    ANIMAL_WORDS,
    CLASS_LABELS,
    OWN_CURVE,
    TIMING_BOOTSTRAP_DRAWS,
    TIMING_BOOTSTRAP_SEED,
    TIMING_RESAMPLES,
    _finite_adder,
)
from continuum_swr_replay.reported.values import (
    Value,
)

#: The curves the timing values are read from, by the word their keys use: the
#: class (``HELDOUT_P_COLUMNS`` key) a unit must be called for, and the curve
#: ``load_population_data`` returns for it.
TIMING_CLASSES = {
    "swr": ("SWR", "swr"),
    "swr_replay": ("non_local_SWR", "non_local_swr"),
    "non_swr_replay": ("non_local_no_SWR", "non_local_no_swr"),
}


# --- timing ----------------------------------------------------------------------


def timing_values(
    population: pd.DataFrame,
    curves: dict[str, np.ndarray],
    lag_time: np.ndarray,
    calibration: dict[str, Any] | None,
    make: Callable[..., Value],
) -> list[Value]:
    """When does the response peak, and does it start before event onset?

    For SWR-coupled replay, non-SWR replay and all SWRs, over the units called
    for that class (raw empirical p below ``ALPHA`` against its own matched null),
    split by the sign of each unit's peak (enhanced or suppressed):

    - per unit, the peak lag (largest ``|curve|`` on +/-200 ms) and the
      half-maximum onset (:func:`~continuum_swr_replay.analysis.pfc.timing.peak_and_onset`),
      their medians and the shares before event onset;
    - for the population, the peak lag and onset of the sign-aligned mean
      curve, with a 95 % interval from the nested bootstrap (animals, their
      recording days, those days' units), and the animals whose own mean curve
      (the mean over its days of each day's mean curve) has its onset before
      event onset.

    Beside them: the share of the units with a matched quiet-rest control curve
    whose curve peaks before onset (the floor for a peak lag drawn by noise),
    units called
    for both replay classes compared pairwise, and the timing calibration
    (simulated responses of known onset fit by the same estimator), which says
    how the estimator moves an onset and how often noise puts a unit's onset
    before zero when the response starts at zero.
    """
    values: list[Value] = []

    add = _finite_adder(values, make)

    # Resample recording days (NWB files), not epochs: the epochs of one day
    # are recorded on the same tetrodes and need not be independent.
    sessions = population["nwb_file_name"].astype(str).to_numpy()
    animals = population["animal"].to_numpy()
    onset_rule = (
        "the half-maximum onset: the earliest lag of the run, ending at the peak, "
        "over which the curve keeps the peak's sign and at least half its size"
    )

    def add_with_interval(
        stats: pd.Series, statistic: str, stem: str, description: str
    ) -> None:
        """A population lag, in ms, and its interval over the resamples."""
        add(
            f"timing_{statistic}_{stem}",
            stats[statistic],
            "quantity",
            "int",
            description,
            "ms",
        )
        add(
            f"timing_{statistic}_ci_low_{stem}",
            stats[f"{statistic}_ci_low"],
            "quantity",
            "int",
            f"2.5th percentile of that over {TIMING_BOOTSTRAP_DRAWS:,} "
            f"{TIMING_RESAMPLES} (seed {TIMING_BOOTSTRAP_SEED})",
            "ms",
        )
        add(
            f"timing_{statistic}_ci_high_{stem}",
            stats[f"{statistic}_ci_high"],
            "quantity",
            "int",
            "97.5th percentile of that over the same resamples",
            "ms",
        )

    called_by_class = {}
    for word, (name, curve_key) in TIMING_CLASSES.items():
        label = CLASS_LABELS[name]
        called = population_statistics.called_units(population, name, ALPHA)
        called_by_class[word] = called
        statistics, by_animal = timing.called_response_timing(
            curves[curve_key],
            curves["matched_control"],
            lag_time,
            called,
            sessions,
            animals,
            n_boot=TIMING_BOOTSTRAP_DRAWS,
            seed=TIMING_BOOTSTRAP_SEED,
        )
        for direction, sign in timing.DIRECTIONS.items():
            stats = statistics.loc[direction]
            stem = f"{word}_{direction}"
            who = f"units called for {label} (raw p < {ALPHA:g}) whose peak is {'an increase' if sign > 0 else 'a decrease'}"
            add(f"timing_units_{stem}", stats["n_units"], "count", "count", who)
            if not stats["n_units"]:
                continue
            add(
                f"timing_peak_lag_median_ms_{stem}",
                stats["peak_lag_median_ms"],
                "quantity",
                "int",
                f"median peak lag (largest |modulation| within +/-200 ms of onset) of {who}",
                "ms",
            )
            add(
                f"timing_peak_lag_quartile_low_ms_{stem}",
                stats["peak_lag_quartile_low_ms"],
                "quantity",
                "int",
                f"25th percentile of the peak lag of {who}",
                "ms",
            )
            add(
                f"timing_peak_lag_quartile_high_ms_{stem}",
                stats["peak_lag_quartile_high_ms"],
                "quantity",
                "int",
                f"75th percentile of the peak lag of {who}",
                "ms",
            )
            add(
                f"timing_peak_before_onset_fraction_{stem}",
                stats["peak_before_onset_fraction"],
                "fraction",
                "pct",
                f"share of {who} whose peak lag is before event onset",
            )
            add(
                f"timing_onset_median_ms_{stem}",
                stats["onset_median_ms"],
                "quantity",
                "int",
                f"median response onset of {who}; onset is {onset_rule}",
                "ms",
            )
            add(
                f"timing_onset_before_onset_fraction_{stem}",
                stats["onset_before_onset_fraction"],
                "fraction",
                "pct",
                f"share of {who} whose response onset is before event onset",
            )
            add(
                f"timing_onset_at_edge_fraction_{stem}",
                stats["onset_at_edge_fraction"],
                "fraction",
                "pct",
                f"share of {who} whose half-maximum run reaches the window edge "
                "(-200 ms), so their onset is at or before it",
            )
            add(
                f"timing_peak_at_edge_fraction_{stem}",
                stats["peak_at_edge_fraction"],
                "fraction",
                "pct",
                f"share of {who} whose peak lies at the edge of the +/-200 ms window",
            )
            add(
                f"timing_rise_onset_median_ms_{stem}",
                stats["rise_onset_median_ms"],
                "quantity",
                "int",
                f"median rise onset of {who}: the onset rule measured from the curve's "
                "value at -200 ms instead of from zero",
                "ms",
            )
            add(
                f"timing_population_pre_event_level_fraction_{stem}",
                stats["population_pre_event_level_fraction"],
                "fraction",
                "pct",
                f"value at -200 ms of the sign-aligned mean curve of {who}, as a share "
                "of its peak, clipped to the range 0 to 1: how far the curve sits "
                "from zero before the event",
            )
            for what, text in (
                (
                    "rise_onset",
                    f"rise onset of the sign-aligned mean curve of {who}: half way "
                    "from its value at -200 ms to its peak",
                ),
                (
                    "peak",
                    f"peak lag of the mean curve of {who}, each unit's curve signed "
                    "by its peak",
                ),
                (
                    "onset",
                    f"onset ({onset_rule}) of the mean curve of {who}, each unit's "
                    "curve signed by its peak",
                ),
            ):
                add_with_interval(stats, f"population_{what}_ms", stem, text)
            add(
                f"timing_steepest_median_ms_{stem}",
                stats["steepest_median_ms"],
                "quantity",
                "int",
                f"median lag of the steepest approach to the peak of {who} (the "
                "level-free rule: where the curve moves fastest toward its peak)",
                "ms",
            )
            add_with_interval(
                stats,
                "population_steepest_ms",
                stem,
                f"lag of the steepest approach to the peak of the sign-aligned mean "
                f"curve of {who}",
            )
            for rule, text in (
                ("size", "its peak, from zero"),
                ("rise_size", "its peak less its value at -200 ms"),
            ):
                add(
                    f"timing_population_{rule}_{stem}",
                    stats[f"population_{rule}"],
                    "quantity",
                    "dec2",
                    f"size of the sign-aligned mean curve of {who}: {text} (log-rate "
                    "units; the peak is the curve's largest absolute value, and each "
                    "unit's curve is signed by its own peak, so a decrease counts as "
                    "a positive size)",
                    "log rate",
                )
                for bound in ("low", "high"):
                    add(
                        f"timing_population_{rule}_ci_{bound}_{stem}",
                        stats[f"population_{rule}_ci_{bound}"],
                        "quantity",
                        "dec2",
                        f"{'2.5th' if bound == 'low' else '97.5th'} percentile of that "
                        f"over {TIMING_BOOTSTRAP_DRAWS:,} {TIMING_RESAMPLES}",
                        "log rate",
                    )
            add(
                f"timing_population_size_vs_control_{stem}",
                stats["population_size_vs_control"],
                "quantity",
                "dec2",
                f"the sign-aligned mean curve of {who} at its peak, less the same "
                "units' matched quiet-rest control curves (signed the same way) at "
                "that lag",
                "log rate",
            )
            add(
                f"timing_population_control_pre_event_level_fraction_{stem}",
                stats["population_control_pre_event_level_fraction"],
                "fraction",
                "pct",
                "the same units' matched-control mean curve at -200 ms, signed by "
                "the event curve, as a share of the event curve's peak, clipped to "
                "the range 0 to 1: how much of the event curve's pre-event level "
                "quiet-rest windows also show",
            )
            for animal, own in by_animal.loc[direction].iterrows():
                add(
                    f"timing_population_rise_onset_ms_{stem}_{ANIMAL_WORDS[animal]}",
                    own["rise_onset_ms"],
                    "quantity",
                    "int",
                    f"rise onset of the sign-aligned mean curve of {animal}'s {who}",
                    "ms",
                )
            add(
                f"timing_population_onset_animals_{stem}",
                stats["n_animals"],
                "count",
                "count",
                f"animals with at least one of the {who}",
            )
            add(
                f"timing_population_onset_animals_negative_{stem}",
                stats["n_animals_onset_before_event"],
                "count",
                "count",
                f"of those animals, how many have the half-maximum onset of their own "
                f"mean curve of {who} ({OWN_CURVE}) before event onset",
            )
            add(
                f"timing_population_steepest_animals_negative_{stem}",
                stats["n_animals_steepest_before_event"],
                "count",
                "count",
                f"of those animals, how many have the steepest approach of their own "
                f"mean curve of {who} ({OWN_CURVE}) before event onset",
            )
            add(
                f"timing_population_rise_onset_animals_negative_{stem}",
                stats["n_animals_rise_onset_before_event"],
                "count",
                "count",
                f"of those animals, how many have the rise onset of their own mean curve "
                f"of {who} ({OWN_CURVE}) before event onset",
            )

    paired = timing.paired_peak_lag_statistics(
        curves[TIMING_CLASSES["swr_replay"][1]],
        curves[TIMING_CLASSES["non_swr_replay"][1]],
        lag_time,
        called_by_class["swr_replay"],
        called_by_class["non_swr_replay"],
    )
    add(
        "timing_paired_units",
        paired["n_paired"],
        "count",
        "count",
        "units called for both SWR-coupled and non-SWR replay whose two peaks have the "
        "same sign",
    )
    add(
        "timing_peak_lag_difference_median_ms",
        paired["difference_median_ms"],
        "quantity",
        "int",
        "over those units, the median of the non-SWR replay peak lag minus the "
        "SWR-coupled replay peak lag",
        "ms",
    )
    add(
        "timing_non_swr_later_fraction",
        paired["second_later_fraction"],
        "fraction",
        "pct",
        "share of those units whose non-SWR replay peak is later than their "
        "SWR-coupled replay peak",
    )
    add(
        "timing_matched_control_peak_before_onset_fraction",
        timing.peak_before_onset_fraction(curves["matched_control"], lag_time),
        "fraction",
        "pct",
        "share of the units with a matched quiet-rest control curve (finite, and "
        "not zero everywhere) whose control curve peaks before the control "
        "window's onset (what a peak lag drawn by noise gives)",
    )

    if calibration is not None:
        _calibration_values(calibration, add)
    return values


#: Gain words of the calibration's planted responses.
CALIBRATION_GAIN_WORDS = {
    0.7: "enhanced",
    -0.7: "suppressed",
    -0.3: "shallow_suppressed",
}

#: Calibration rules quoted, by key word and description.
CALIBRATION_RULES = {
    "onset_zero": "half-maximum onset from zero",
    "onset_rise": "rise onset (half way from the -200 ms value to the peak)",
    "steepest": "steepest approach to the peak",
    "peak": "peak lag",
}


def _calibration_values(calibration: dict[str, Any], add: Callable[..., None]) -> None:
    """The timing calibration's reference values (simulated units, known truth).

    Read from ``scripts/pfc/calibrate_pfc_kernel_timing.py``'s output, canonical
    window only: for responses that start exactly at event onset, each rule's
    population bias (sign-aligned mean curve, against the truth read by the same
    rule) and the size bias; the unit-level noise floor of "onset before zero";
    and what a stop-level rate difference with no event-locked response, alone
    and under a planted decrease, looks like to each rule.
    """
    for scenario in calibration["scenarios"]:
        if scenario["true_onset_ms"] != 0.0:
            continue
        rules = scenario["rules"]
        if scenario["window"] == "wide":
            # Only what says whether a wider window would change the reading.
            if scenario["scenario"] == "response":
                stem = f"{scenario['shape']}_{CALIBRATION_GAIN_WORDS[scenario['gain']]}"
                add(
                    f"timing_calibration_wide_bias_ms_onset_rise_{stem}",
                    rules["onset_rise"]["population_bias"],
                    "quantity",
                    "int",
                    "rise onset minus the truth's, as the canonical-window key, for a "
                    "kernel fit on +/-1.05 s and read on +/-1.0 s",
                    "ms",
                )
            elif scenario["scenario"] == "stop_offset":
                add(
                    "timing_calibration_wide_stop_offset_size",
                    rules["size_zero"]["population"],
                    "quantity",
                    "dec2",
                    "size from zero of the stop-offset scenario's sign-aligned mean "
                    "curve for a kernel fit on +/-1.05 s and read on +/-1.0 s",
                    "log rate",
                )
            continue
        if scenario["scenario"] == "response":
            stem = f"{scenario['shape']}_{CALIBRATION_GAIN_WORDS[scenario['gain']]}"
            response = (
                "a 100 ms boxcar"
                if scenario["shape"] == "boxcar"
                else "an alpha function peaking 30 ms after it starts"
            )
            what = (
                f"simulated units ({scenario['n_units']}) whose response, {response} "
                f"of log-rate gain {scenario['gain']:+g}, starts exactly at event "
                "onset, fit by the shipped basis, prior and fitter"
            )
            for rule, text in CALIBRATION_RULES.items():
                if rule == "peak" and scenario["shape"] == "boxcar":
                    continue  # a boxcar's peak lag is not defined
                add(
                    f"timing_calibration_bias_ms_{rule}_{stem}",
                    rules[rule]["population_bias"],
                    "quantity",
                    "int",
                    f"{text} of the sign-aligned mean curve of {what}, minus the "
                    "truth read by the same rule",
                    "ms",
                )
            add(
                f"timing_calibration_size_bias_{stem}",
                rules["size_rise"]["population_bias"],
                "quantity",
                "dec2",
                f"size (peak less the -200 ms value) of the sign-aligned mean curve "
                f"of {what}, minus the truth's",
                "log rate",
            )
            add(
                f"timing_calibration_onset_before_zero_fraction_{stem}",
                scenario["fraction_onset_before_zero"],
                "fraction",
                "pct",
                f"share of {what} whose half-maximum onset is before event onset: "
                "the noise floor of the per-unit share",
            )
            continue
        name = scenario["scenario"]
        what = (
            "simulated units whose rate is lower (by log 0.3) in 40 % of 20 s "
            "stretches, with 80 % of the events in those stretches"
            + (
                ", plus a -0.7 boxcar decrease from event onset"
                if name == "stop_offset_plus_response"
                else ", and no event-locked response"
            )
        )
        add(
            f"timing_calibration_{name}_size",
            rules["size_zero"]["population"],
            "quantity",
            "dec2",
            f"size from zero of the sign-aligned mean curve of {what}",
            "log rate",
        )
        add(
            f"timing_calibration_{name}_rise_size",
            rules["size_rise"]["population"],
            "quantity",
            "dec2",
            f"size (peak less the -200 ms value) of the sign-aligned mean curve of "
            f"{what}",
            "log rate",
        )
        add(
            f"timing_calibration_{name}_suppressed_fraction",
            scenario["fraction_peak_sign_negative"],
            "fraction",
            "pct",
            f"share of {what} whose peak is a decrease",
        )
        if name == "stop_offset_plus_response":
            for rule in ("onset_zero", "onset_rise", "steepest"):
                add(
                    f"timing_calibration_{name}_bias_ms_{rule}",
                    rules[rule]["population_bias"],
                    "quantity",
                    "int",
                    f"{CALIBRATION_RULES[rule]} of the sign-aligned mean curve of "
                    f"{what}, minus the truth's",
                    "ms",
                )
