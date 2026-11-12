"""The mpfc summary's values for SWRs without replay: the extended event set
that adds SWRs with no non-local event as a third class. The statistics are
computed in ``analysis.pfc.population_statistics``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc import population_statistics, timing
from continuum_swr_replay.analysis.pfc.load_population import (
    CALLED_ALPHA as ALPHA,
)
from continuum_swr_replay.analysis.pfc.modulation import THREE_CLASS_MODEL
from continuum_swr_replay.reported.mpfc.common import (
    ANIMAL_WORDS,
    TIMING_BOOTSTRAP_DRAWS,
    TIMING_BOOTSTRAP_SEED,
    _finite_adder,
)
from continuum_swr_replay.reported.values import (
    Value,
)

#: The new class's extra curves, as ``load_population_data`` reads them.
SWR_WITHOUT_REPLAY_CURVES = ("SWR_no_non_local", "non_local_SWR_3class")

#: A session-level event floor for the new class's check: the main values are
#: over every session, and the same values over sessions with at least this
#: many SWRs without replay are the check.
SWR_WITHOUT_REPLAY_MIN_EVENTS = 10


# --- SWRs without replay ------------------------------------------------------------


def swr_without_replay_values(
    population: pd.DataFrame,
    curves: dict[str, np.ndarray],
    lag_time: np.ndarray,
    sidecars: list[dict[str, Any]],
    make: Callable[..., Value],
) -> list[Value]:
    """Does mPFC respond to Kay SWRs that overlap no non-local event, and as it
    does to SWR-coupled replay?

    Read from the extended-event-set bundles, in which ``SWR_no_non_local`` has
    its own count- and duration-matched null and is fit beside the two replay
    classes in one three-class model, from which both compared curves come.
    Every value is given twice: over every session (``_all``) and over the
    sessions with at least :data:`SWR_WITHOUT_REPLAY_MIN_EVENTS` such SWRs
    (``_min_events``). All sessions are the main numbers; the subset is a check. In sessions with
    very few such SWRs many low-rate units have no p-value: the canonical rule
    needs all R null draws finite, and a draw's fit can fail when a unit has no
    spikes under the few windows. A curve fit on few events also wiggles more,
    so its largest excursion over lags is larger by noise alone: compare the
    called fractions (each against its own null) before the peak sizes.
    """
    values: list[Value] = []

    add = _finite_adder(values, make)

    # A sidecar without a count gives a missing value, which the statistics refuse.
    sessions = pd.DataFrame(
        [
            {
                name: s.get(name)
                for name in (
                    "nwb_file_name",
                    "epoch_name",
                    "n_swr_no_non_local",
                    "n_swr_with_non_local",
                    "n_swr_no_non_local_unknown",
                )
            }
            for s in sidecars
        ]
    )
    events = population_statistics.swr_without_replay_event_statistics(
        sessions, SWR_WITHOUT_REPLAY_MIN_EVENTS
    )
    add(
        "swr_no_replay_sessions",
        len(sidecars),
        "count",
        "count",
        "sessions in the extended-event-set tree",
    )
    add(
        "swr_no_replay_events",
        events["n_events"],
        "count",
        "count",
        "Kay SWRs (kept, in the stop cohort) that overlap no non-local event, summed "
        "over sessions",
    )
    add(
        "swr_no_replay_events_fraction",
        events["events_fraction"],
        "fraction",
        "pct",
        "share of the kept, cohort Kay SWRs that overlap no non-local event",
    )
    add(
        "swr_no_replay_events_unknown",
        events["n_unknown"],
        "count",
        "count",
        "kept SWRs left out of the class because they overlap a window the "
        "analysed-time rule dropped",
    )
    add(
        "swr_no_replay_events_session_median",
        events["events_session_median"],
        "quantity",
        "int",
        "median per session of those SWRs",
        "events",
    )
    add(
        "swr_no_replay_sessions_min_events",
        events["n_sessions_min_events"],
        "count",
        "count",
        f"sessions with at least {SWR_WITHOUT_REPLAY_MIN_EVENTS} such SWRs",
    )

    settings = {"alpha": ALPHA, "min_events": SWR_WITHOUT_REPLAY_MIN_EVENTS}
    statistics = population_statistics.swr_without_replay_unit_statistics(
        population, sessions, three_class_model=THREE_CLASS_MODEL, **settings
    )
    by_animal = population_statistics.swr_without_replay_called_by_animal(
        population, sessions, **settings
    )
    timing_statistics = population_statistics.swr_without_replay_timing_statistics(
        population,
        curves["swr_no_non_local"],
        lag_time,
        sessions,
        n_boot=TIMING_BOOTSTRAP_DRAWS,
        seed=TIMING_BOOTSTRAP_SEED,
        **settings,
    )
    for subset, label in (
        ("all", "every session"),
        (
            "min_events",
            f"sessions with at least {SWR_WITHOUT_REPLAY_MIN_EVENTS} SWRs without replay",
        ),
    ):
        where = f"({label})"
        stats = statistics.loc[subset]
        add(
            f"swr_no_replay_units_{subset}",
            stats["n_units"],
            "count",
            "count",
            f"units in {label} (Figure 4's population)",
        )
        add(
            f"swr_no_replay_units_tested_{subset}",
            stats["n_tested"],
            "count",
            "count",
            f"of those, units with a finite p-value for SWRs without replay {where}",
        )
        add(
            f"swr_no_replay_called_fraction_{subset}",
            stats["called_fraction"],
            "fraction",
            "pct",
            f"share of tested units at raw empirical p < {ALPHA:g} for SWRs "
            f"without replay, against their own matched null {where}",
        )
        add(
            f"swr_no_replay_units_called_{subset}",
            stats["n_called"],
            "count",
            "count",
            f"units called for SWRs without replay {where}",
        )
        for (_, animal), own in by_animal[
            by_animal.index.get_level_values("subset") == subset
        ].iterrows():
            add(
                f"swr_no_replay_called_fraction_{subset}_{ANIMAL_WORDS[animal]}",
                own["called_fraction"],
                "fraction",
                "pct",
                f"the same for {animal}'s tested units {where}: the mean over "
                "its sessions of each session's share",
            )
        add(
            f"swr_no_replay_overlap_tested_{subset}",
            stats["n_overlap_tested"],
            "count",
            "count",
            f"units tested for both SWRs without replay and SWR-coupled replay {where}",
        )
        for word, text in (
            ("both", "called for both"),
            ("swr_no_replay_only", "called for SWRs without replay only"),
            ("swr_replay_only", "called for SWR-coupled replay only"),
            ("neither", "called for neither"),
        ):
            add(
                f"swr_no_replay_overlap_{word}_{subset}",
                stats[f"n_overlap_{word}"],
                "count",
                "count",
                f"of those, units {text} {where}",
            )
        add(
            f"swr_no_replay_overlap_expected_{subset}",
            stats["n_overlap_expected"],
            "quantity",
            "int",
            f"units called for both expected if the two calls were independent {where}",
            "units",
        )
        add(
            f"swr_no_replay_overlap_odds_ratio_{subset}",
            stats["overlap_odds_ratio"],
            "quantity",
            "sig2",
            "odds ratio of being called for SWR-coupled replay given called for "
            f"SWRs without replay (1 = independent) {where}",
        )
        add(
            f"swr_no_replay_rank_correlation_{subset}",
            stats["rank_correlation"],
            "quantity",
            "dec2",
            "Spearman correlation over units tested for both of the peak log "
            "modulation for SWRs without replay and for SWR-coupled replay, both "
            f"from the three-class model {where}",
        )
        add(
            f"swr_no_replay_same_sign_fraction_{subset}",
            stats["same_sign_fraction"],
            "fraction",
            "pct",
            f"of the units called for both, the share whose two peaks have the "
            f"same sign {where}",
        )
        for word, text in (
            ("swr_no_replay", "SWRs without replay"),
            ("swr_replay", "SWR-coupled replay"),
        ):
            add(
                f"swr_no_replay_effect_abs_peak_median_{word}_{subset}",
                stats[f"abs_peak_median_{word}"],
                "quantity",
                "dec2",
                f"median over units of the absolute peak log-rate modulation for {text}, "
                f"three-class model {where}",
                "log rate",
            )
            add(
                f"swr_no_replay_effect_rate_multiplier_median_{word}_{subset}",
                stats[f"rate_multiplier_median_{word}"],
                "quantity",
                "sig2",
                f"exp of that median: the median unit's largest rate change for {text} "
                f"{where}",
                "x",
            )
            add(
                f"swr_no_replay_effect_abs_peak_median_called_{word}_{subset}",
                stats[f"abs_peak_median_called_{word}"],
                "quantity",
                "dec2",
                f"the same over the units called for SWRs without replay {where}",
                "log rate",
            )
        add(
            f"swr_no_replay_adds_to_split_fraction_{subset}",
            stats["adds_to_split_fraction"],
            "fraction",
            "pct",
            "share of units whose held-out deviance improves when SWRs without "
            f"replay are added to the replay split {where}",
        )
        add(
            f"swr_no_replay_best_model_fraction_{subset}",
            stats["best_model_fraction"],
            "fraction",
            "pct",
            "share of units whose best model by held-out deviance (constant, "
            "non_local, the replay split, or the three-class model) is the three-"
            f"class model {where}",
        )
        for direction, sign in timing.DIRECTIONS.items():
            rise = timing_statistics.loc[(subset, direction)]
            stem = f"{direction}_{subset}"
            add(
                f"swr_no_replay_timing_units_{stem}",
                rise["n_units"],
                "count",
                "count",
                f"units called for SWRs without replay whose peak is "
                f"{'an increase' if sign > 0 else 'a decrease'} {where}",
            )
            for key, unit, style, text in (
                ("peak_ms", "ms", "int", "peak lag"),
                ("rise_onset_ms", "ms", "int", "rise onset"),
                (
                    "rise_onset_ms_ci_low",
                    "ms",
                    "int",
                    "2.5th percentile over recording-day resamples of the rise onset",
                ),
                (
                    "rise_onset_ms_ci_high",
                    "ms",
                    "int",
                    "97.5th percentile over recording-day resamples of the rise onset",
                ),
                (
                    "rise_size",
                    "log rate",
                    "dec2",
                    "size from the -200 ms value",
                ),
            ):
                add(
                    f"swr_no_replay_timing_{key}_{stem}",
                    rise[key],
                    "quantity",
                    style,
                    f"{text} of the sign-aligned mean curve of the units called for SWRs "
                    f"without replay whose peak is {'an increase' if sign > 0 else 'a decrease'} "
                    f"{where}",
                    unit,
                )
    return values
