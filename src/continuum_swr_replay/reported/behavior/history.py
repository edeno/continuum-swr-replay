"""The behavior summary's event-history values: the self- and cross-history
kernels of the duration-aware refit and its ridge cross-validated comparison
(question 3.3). The statistics are computed in
``analysis.replay_models.rate.history_kernel_statistics``."""

from __future__ import annotations

from continuum_swr_replay.analysis.replay_models.rate import (
    history_kernel_statistics,
)
from continuum_swr_replay.reported.behavior.common import TAGS, _require
from continuum_swr_replay.reported.behavior.rate import RATE_POPULATIONS
from continuum_swr_replay.reported.values import (
    Value,
    value_factory,
)

# --- event history on the duration-aware design -------------------------------

#: The caveat every pooled history value carries.
UPPER_BOUND = (
    "pooled within animal with a session fixed effect; an upper bound, since rate "
    "variation between stops that the model does not capture inflates history terms"
)
#: Cross directions: key word -> (response population, label).
CROSS_DIRECTIONS = {
    "non_swr_from_swr": ("non_local_no_swr", "non-SWR rate after SWR-coupled events"),
    "swr_from_non_swr": ("non_local_swr", "SWR-coupled rate after non-SWR events"),
}
CROSS_LAG_MS = 45
#: The ridge cross-validated rungs (``analysis/replay_models/rate/ridge_history_cv.py``).
RIDGE_SELF_RUNG = "Mself_long_dur"
RIDGE_CROSS_RUNG = "Mselfcross_long_dur"


def duration_aware_history_values(
    ridge: dict, pooled: dict, source: str
) -> list[Value]:
    """Question 3.3: refractoriness, burst and cross-coupling once duration is handled.

    ``pooled`` is ``duration_aware_history_pooled.pkl``: per animal and
    population, the rungs fit to the animal's stacked sessions. The
    self-history kernels are the both-anchor rung's (``Mself_dur``), read at
    measurable lags only (expected onsets >= ``HISTORY_MIN_SUPPORT``, Wald
    SE <= ``HISTORY_MAX_SE``) of converged fits; an animal whose lag is not
    measurable is left out of that lag's values, and the description says how
    many were kept. ``ridge`` is ``cross_history_ridge_cv.pkl``: per animal
    and non-local stream, the self-only and self+cross rungs with the cross
    block reaching 1 s, fit with a ridge prior on the history terms; the cross
    values are the self+cross rung's (:data:`RIDGE_CROSS_RUNG`) end-locked
    rate ratio at :data:`CROSS_LAG_MS`, read where that lag has at least
    ``HISTORY_MIN_SUPPORT`` expected onsets, whatever its posterior SD, and
    the rung's held-out gain (outer five-fold cross-validation over whole
    stops). Every ridge fit must have converged.
    """
    make = value_factory(duration_aware_history_values, source, TAGS["rate"])
    by_population = history_kernel_statistics.self_history_statistics(
        pooled["results"], "Mself_dur"
    )
    outcomes = {outcome for outcome, _ in RATE_POPULATIONS.values()}
    _require(
        set(by_population.index) == outcomes,
        f"pooled populations {sorted(by_population.index)} != {sorted(outcomes)}",
    )
    overall = history_kernel_statistics.self_history_range_statistics(by_population)
    n_fits, n_measured = overall["n_fits"], overall["n_first_lag_measurable"]
    end_peaks = (
        f"{overall['n_end_peaks']} of the {n_fits} animal x population fits have a "
        "measurable end-locked lag"
    )
    values: list[Value] = []
    first_ms = history_kernel_statistics.lag_ms(0)
    for word, (outcome, label) in RATE_POPULATIONS.items():
        with_peak = (
            f"the {by_population.loc[outcome, 'n_onset_peaks']} of "
            f"{by_population.loc[outcome, 'n_fits']} animals with a measurable "
            "onset-locked lag"
        )
        values += [
            make(
                f"history_onset_peak_{word}",
                by_population.loc[outcome, "onset_peak_rate_ratio_median"],
                "quantity",
                "dec2",
                f"{label}: onset-locked part of the self-history (both-anchor rung "
                f"Mself_dur, at-risk design): median over {with_peak} of each "
                f"animal's peak rate ratio over its measurable lags; {UPPER_BOUND}",
                "rate ratio",
            ),
            make(
                f"history_onset_peak_lag_ms_{word}",
                by_population.loc[outcome, "onset_peak_lag_ms_median"],
                "quantity",
                "int",
                f"{label}: median over {with_peak} of that onset-locked peak's lag",
                "ms",
            ),
            make(
                f"history_onset_animals_{word}",
                by_population.loc[outcome, "n_onset_peaks_above_one"],
                "count",
                "count",
                f"{label}: animals, of {with_peak}, whose onset-locked peak has its "
                "95 % CI above 1, the interval widened for the measurable lags "
                "searched (Bonferroni)",
            ),
        ]
    values += [
        make(
            "history_end_first_lag_rr_low",
            overall["first_lag_rate_ratio_low"],
            "quantity",
            "dec2",
            "lowest over the four populations of the median (over animals) rate ratio "
            f"{first_ms} ms after the previous event's end (end-locked part of "
            f"Mself_dur, at-risk design), over the {n_measured} of {n_fits} animal x "
            f"population fits whose {first_ms} ms lag is measurable; {UPPER_BOUND}",
            "rate ratio",
        ),
        make(
            "history_end_first_lag_rr_high",
            overall["first_lag_rate_ratio_high"],
            "quantity",
            "dec2",
            "highest over the four populations of that median rate ratio "
            f"{first_ms} ms after the previous event's end ({n_measured} of {n_fits} "
            "fits measurable)",
            "rate ratio",
        ),
        make(
            "history_end_first_lag_suppressed_fits",
            overall["n_first_lag_suppressed"],
            "count",
            "count",
            f"animal x population fits whose rate {first_ms} ms after the previous "
            "event's end is below 1 with its 95 % CI (suppression after the end), of "
            "the history_fits fits where that lag is measurable",
        ),
        make(
            "history_end_first_lag_raised_fits",
            overall["n_first_lag_raised"],
            "count",
            "count",
            f"animal x population fits whose rate {first_ms} ms after the previous "
            "event's end is above 1 with its 95 % CI, of the history_fits fits where "
            "that lag is measurable",
        ),
        make(
            "history_fits",
            n_measured,
            "count",
            "count",
            f"animal x population pooled fits ({n_fits} converged) whose {first_ms} ms "
            "end-locked lag is measurable: the denominator of the raised and "
            "suppressed counts",
        ),
        make(
            "history_end_peak_low",
            overall["end_peak_rate_ratio_low"],
            "quantity",
            "dec2",
            "lowest over the four populations of the median (over the animals with "
            "a measurable end-locked lag) peak rate ratio of the end-locked "
            "self-history (Mself_dur, at-risk design, measurable lags); "
            f"{end_peaks}; {UPPER_BOUND}",
            "rate ratio",
        ),
        make(
            "history_end_peak_high",
            overall["end_peak_rate_ratio_high"],
            "quantity",
            "dec2",
            f"highest over the four populations of that end-locked peak ({end_peaks})",
            "rate ratio",
        ),
        make(
            "history_end_peak_lag_ms_low",
            overall["end_peak_lag_ms_low"],
            "quantity",
            "int",
            "shortest over the populations of the median end-locked peak lag "
            f"({end_peaks})",
            "ms",
        ),
        make(
            "history_end_peak_lag_ms_high",
            overall["end_peak_lag_ms_high"],
            "quantity",
            "int",
            "longest over the populations of the median end-locked peak lag "
            f"({end_peaks})",
            "ms",
        ),
        make(
            "history_end_peak_animals_min",
            overall["n_end_peaks_above_one_min"],
            "count",
            "count",
            "fewest, over the populations, animals whose end-locked peak has its 95 % "
            "CI above 1, the interval widened for the measurable lags searched "
            f"(Bonferroni); {end_peaks}",
        ),
    ]
    for word, (outcome, label) in CROSS_DIRECTIONS.items():
        cross = history_kernel_statistics.cross_history_statistics(
            ridge["results"], outcome, RIDGE_CROSS_RUNG, CROSS_LAG_MS
        )
        n_animals = cross["n_animals"]
        kept = (
            f"over the {cross['n_measurable']} of {n_animals} animals whose "
            f"{CROSS_LAG_MS} ms lag has at least "
            f"{history_kernel_statistics.HISTORY_MIN_SUPPORT:g} expected onsets"
        )
        values += [
            make(
                f"history_cross_end_rr_{word}",
                cross["rate_ratio_median"],
                "quantity",
                "dec2",
                f"{label}: rate ratio {CROSS_LAG_MS} ms after the other stream's event "
                f"ended (end-locked cross block of {RIDGE_CROSS_RUNG}, 1 s cross "
                "window, at-risk design, ridge prior on the history terms with its "
                f"strength chosen by cross-validation), median {kept}; the prior "
                f"shrinks it toward 1; {UPPER_BOUND}",
                "rate ratio",
            ),
            make(
                f"history_cross_end_animals_{word}",
                cross["n_animals_above_one"],
                "count",
                "count",
                f"{label}: animals whose rate ratio at {CROSS_LAG_MS} ms has its 95 % "
                f"interval (posterior mode +- 1.96 posterior SD) above 1, {kept}",
            ),
            make(
                f"history_cross_heldout_gain_{word}",
                cross["heldout_gain_median"],
                "quantity",
                "sig2s",
                f"{label}: median over the {n_animals} animals of the held-out "
                f"log-likelihood gain per held-out event of {RIDGE_CROSS_RUNG} over "
                f"{RIDGE_SELF_RUNG} (five folds of whole stops, each rung's ridge "
                "strength chosen by inner cross-validation); positive = the cross "
                "block predicts new data",
                "nats per event",
            ),
            make(
                f"history_cross_heldout_animals_positive_{word}",
                cross["n_animals_gain_positive"],
                "count",
                "count",
                f"{label}: animals, of {n_animals}, whose held-out cross gain is "
                "positive",
            ),
        ]
    return values
