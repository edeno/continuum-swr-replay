"""The behavior summary's rate values (question 3.3): how often events occur.

The per-bin conditional-intensity ladder (Family B; :func:`per_bin_values`,
:func:`per_bin_detail_values`), its ``is_swr`` contrast
(:func:`rate_contrast_separation_values`, :func:`rate_contrast_audit_values`),
the per-stop ladder (Family A; :func:`per_stop_values`) and the cohort
audits of both (:func:`cohort_audit_values`). The statistics are computed in
``analysis.behavior.rate.rate_fit_statistics`` and
``analysis.behavior.coefficient_statistics``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from continuum_swr_replay.analysis.behavior.coefficient_statistics import (
    aic_difference_statistics,
    audited_drop_statistics,
    fit_rows,
    fits_carrying,
    gated_coefficient_statistics,
    gated_exp_median_ratio,
    group_range,
    known_fraction,
    rung_separation_range,
    separated_fraction,
    term_separation_statistics,
)
from continuum_swr_replay.analysis.behavior.model_specs import (
    BIN_S,
    PER_BIN_HISTORY_COLUMNS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_LAGS,
    TMAX_ARRIVAL,
)
from continuum_swr_replay.analysis.behavior.rate import (
    rate_fit_statistics,
)
from continuum_swr_replay.reported.behavior.common import (
    TAGS,
    _gate_note,
    _require,
    _require_every_animal,
)
from continuum_swr_replay.reported.behavior.inputs import BehaviorInputs
from continuum_swr_replay.reported.formatting import format_number, milliseconds
from continuum_swr_replay.reported.values import (
    Value,
    pend,
    rel,
    value_factory,
)

#: Rate populations: the word their keys use, the fits' ``outcome``, the label.
RATE_POPULATIONS = {
    "swr": ("swr", "detected SWR"),
    "non_local": ("non_local", "non-local (all)"),
    "non_local_swr": ("non_local_swr", "non-local with SWR"),
    "non_local_no_swr": ("non_local_no_swr", "non-local without SWR"),
}
FULL_RATE_RUNG = "M7r_time_histcos_rew"
IMMEDIATE_COEF = f"coef_{REWARD_HISTORY_COLUMNS[0]}"
#: The per-bin model's bin width, in ms, as the descriptions write it.
_BIN_MS = format_number(milliseconds(BIN_S), "int")
#: What every value read off one reward basis coefficient says about itself.
BASIS_COEFFICIENT_NOTE = (
    "a basis coefficient, not the effect of a reward at any lag: the three "
    "coefficients are strongly collinear and a 0 to 1 change in one feature with "
    "the others held fixed is no reward history that occurs; the effect of a reward "
    "at lag 1 is the reconstructed kernel (rate_kernel_lag_one_*)"
)
STOP_TIME_RUNG = "M5_time"
ARRIVAL_TIME_RUNG = "M5a_time_arrival"


# --- rate, per time bin (Family B) --------------------------------------------


def per_bin_values(
    fits: pd.DataFrame, all_stops: pd.DataFrame, source: str
) -> list[Value]:
    """Question 3.3: the per-bin conditional-intensity ladder (Family B).

    Read on the well-anchored cohort; ``allstops`` keys read the all-stops
    sensitivity fit with the same rules.
    """
    make = value_factory(per_bin_values, source, TAGS["rate"])
    values: list[Value] = []
    cohort = rate_fit_statistics.frozen_cohort_statistics(
        fit_rows(fits, model="M0_const")
    )
    ladder = rate_fit_statistics.ladder_fit_statistics(fits)
    late_lags = f"{rate_fit_statistics.FIRST_LATE_REWARD_LAG}-{REWARD_HISTORY_N_LAGS}"
    # what each _RANGED value is, without its population
    ranged_what = {
        "rate_kernel_late_min": f"lowest median reconstructed kernel over lags {late_lags}",
        "rate_arrival_separated_fraction": f"share of {ARRIVAL_TIME_RUNG} fits flagged separated",
    }
    values += [
        make(
            "rate_sessions",
            cohort["n_sessions"],
            "count",
            "count",
            "sessions in the per-bin rate fit",
        ),
        make(
            "rate_stops",
            cohort["n_stops"],
            "count",
            "count",
            "stops in the frozen common cohort (M0_const, every population)",
        ),
        make(
            "rate_bins",
            cohort["n_bins_non_local"],
            "count",
            "count",
            f"{_BIN_MS} ms bins at risk in the frozen common cohort, non-local "
            "populations (the bins inside a non-local event are not at risk)",
        ),
        make(
            "rate_bins_swr",
            cohort["n_bins_swr"],
            "count",
            "count",
            f"{_BIN_MS} ms bins at risk in the frozen common cohort, detected "
            "SWR (the bins inside a Kay SWR are not at risk)",
        ),
    ]
    for word, (outcome, label) in RATE_POPULATIONS.items():
        null = fit_rows(fits, model="M0_const", outcome=outcome)
        full = fit_rows(fits, model=FULL_RATE_RUNG, outcome=outcome)
        baseline = rate_fit_statistics.baseline_rate_statistics(null)
        kernel = rate_fit_statistics.reward_kernel_statistics(full)
        immediate = gated_coefficient_statistics(full, REWARD_HISTORY_COLUMNS[0])
        immediate_note = _gate_note(immediate, REWARD_HISTORY_COLUMNS[0])
        for stats in (kernel, immediate):
            _require_every_animal(stats["n_animals"], outcome)
        marginal = aic_difference_statistics(
            fit_rows(fits, model="M1_rew", outcome=outcome), null
        )
        conditional = aic_difference_statistics(
            full, fit_rows(fits, model="M6r_time_histcos", outcome=outcome)
        )
        values += [
            make(
                f"rate_baseline_{word}",
                baseline["median"],
                "quantity",
                "dec2",
                f"{label}: median over sessions of exp(intercept) of M0_const, events per at-risk second (no event of its detector ongoing)",
                "events/s",
            ),
            make(
                f"rate_full_sessions_{word}",
                ladder.loc[(FULL_RATE_RUNG, outcome), "n_fitted"],
                "count",
                "count",
                f"{label}: sessions with a fit of {FULL_RATE_RUNG}",
            ),
            make(
                f"rate_time_sessions_{word}",
                ladder.loc[(STOP_TIME_RUNG, outcome), "n_fitted"],
                "count",
                "count",
                f"{label}: sessions with a fit of {STOP_TIME_RUNG}",
            ),
            make(
                f"rate_kernel_lag_one_{word}",
                kernel["rate_ratio_lag_1"],
                "quantity",
                "dec2",
                f"{label}: reward kernel at lag 1, reconstructed per session from the three basis "
                f"coefficients of {FULL_RATE_RUNG}, median over the sessions whose three "
                "reward terms pass the per-term separation gate (session-pooled)",
                "rate ratio",
            ),
            make(
                f"rate_kernel_lag_one_animal_{word}",
                kernel["lag_one_animal_rate_ratio"],
                "quantity",
                "dec2",
                f"{label}: the same lag-1 kernel, median over animals of each animal's median "
                "(animal-balanced)",
                "rate ratio",
            ),
            make(
                f"rate_kernel_lag_one_animals_positive_{word}",
                kernel["n_animals_lag_one_positive"],
                "count",
                "count",
                f"{label}: animals whose median lag-1 kernel is above 1",
            ),
            make(
                f"rate_kernel_lag_one_positive_fraction_{word}",
                kernel["lag_one_positive_fraction"],
                "fraction",
                "pct",
                f"{label}: share of sessions whose lag-1 kernel is above 1",
            ),
            make(
                f"rate_kernel_lag_one_p_{word}",
                kernel["lag_one_signed_rank_p"],
                "quantity",
                "pval",
                f"{label}: signed-rank p of the log lag-1 kernel against 0 across sessions, "
                "treating sessions as independent",
            ),
            make(
                f"rate_kernel_sessions_{word}",
                kernel["n_sessions"],
                "count",
                "count",
                f"{label}: sessions whose {FULL_RATE_RUNG} reward terms all pass the "
                "per-term separation gate (the kernel values' sessions)",
            ),
            make(
                f"rate_kernel_dropped_{word}",
                kernel["n_dropped"],
                "count",
                "count",
                f"{label}: fitted sessions the gate left out because a reward term was "
                "separated or ran away",
            ),
            make(
                f"rate_kernel_nonconverged_{word}",
                kernel["n_nonconverged"],
                "count",
                "count",
                f"{label}: of the kernel values' sessions, fits the optimizer flagged "
                "as not converged (kept: convergence is not part of the gate)",
            ),
            make(
                f"rate_kernel_late_min_{word}",
                kernel["late_rate_ratio_min"],
                "quantity",
                "dec2",
                f"{label}: {ranged_what['rate_kernel_late_min']}",
                "rate ratio",
            ),
            make(
                f"rate_kernel_late_max_{word}",
                kernel["late_rate_ratio_max"],
                "quantity",
                "dec2",
                f"{label}: highest median reconstructed kernel over lags {late_lags}",
                "rate ratio",
            ),
            make(
                f"rate_kernel_lag_ten_{word}",
                kernel[f"rate_ratio_lag_{REWARD_HISTORY_N_LAGS}"],
                "quantity",
                "dec2",
                f"{label}: median reconstructed kernel at lag {REWARD_HISTORY_N_LAGS}",
                "rate ratio",
            ),
            make(
                f"rate_immediate_session_{word}",
                immediate["exp_median"],
                "quantity",
                "sig2",
                f"{label}: exp of the across-session median of the first reward basis "
                f"coefficient {IMMEDIATE_COEF} in {FULL_RATE_RUNG}; {BASIS_COEFFICIENT_NOTE}",
                "rate ratio",
                pending=immediate_note,
            ),
            make(
                f"rate_immediate_session_positive_fraction_{word}",
                immediate["positive_fraction"],
                "fraction",
                "pct",
                f"{label}: share of sessions whose first reward basis coefficient is "
                "positive (a basis coefficient, not an effect size)",
                pending=immediate_note,
            ),
            make(
                f"rate_immediate_session_animals_{word}",
                immediate["n_animals_same_sign"],
                "count",
                "count",
                f"{label}: animals whose median first reward basis coefficient has the "
                "sign of the across-session median",
                pending=immediate_note,
            ),
            make(
                f"rate_immediate_session_p_{word}",
                immediate["signed_rank_p"],
                "quantity",
                "pval",
                f"{label}: signed-rank p of the first reward basis coefficient against 0 "
                "across sessions (a basis coefficient, not an effect size)",
                pending=immediate_note,
            ),
            make(
                f"rate_immediate_animal_{word}",
                immediate["exp_animal_median"],
                "quantity",
                "sig2",
                f"{label}: exp of the animal-balanced median first reward basis "
                f"coefficient (rr_immediate); {BASIS_COEFFICIENT_NOTE}",
                "rate ratio",
                pending=immediate_note,
            ),
            make(
                f"rate_immediate_animals_positive_{word}",
                immediate["n_animals_positive"],
                "count",
                "count",
                f"{label}: animals whose median first reward basis coefficient is positive",
                pending=immediate_note,
            ),
            make(
                f"rate_marginal_reward_delta_aic_{word}",
                marginal["median"],
                "quantity",
                "sig2s",
                f"{label}: median per-session AIC(M1_rew) - AIC(M0_const); negative = child better",
                "AIC",
            ),
            make(
                f"rate_conditional_reward_delta_aic_{word}",
                conditional["median"],
                "quantity",
                "sig2s",
                f"{label}: median per-session AIC({FULL_RATE_RUNG}) - AIC(M6r_time_histcos); "
                "negative = child better",
                "AIC",
            ),
        ]
        window = rate_fit_statistics.arrival_window_statistics(
            fits.attrs["arrival_clock_cohort"][outcome]
        )
        window_all = rate_fit_statistics.arrival_window_statistics(
            all_stops.attrs["arrival_clock_cohort"][outcome]
        )
        values += [
            make(
                f"rate_arrival_event_share_{word}",
                window["event_share"],
                "fraction",
                "dec2",
                f"{label}: share of its event bins the arrival window "
                f"[0, {TMAX_ARRIVAL:g}] s keeps",
            ),
            make(
                f"rate_allstops_arrival_event_share_{word}",
                window_all["event_share"],
                "fraction",
                "dec2",
                f"{label}: the same share on the all-stops fit",
            ),
            make(
                f"rate_arrival_separated_fraction_{word}",
                ladder.loc[(ARRIVAL_TIME_RUNG, outcome), "separated_fraction"],
                "fraction",
                "pct",
                f"{label}: {ranged_what['rate_arrival_separated_fraction']}",
            ),
        ]
        for prefix, frame in (("rate", fits), ("rate_allstops", all_stops)):
            shape = rate_fit_statistics.clock_shape_statistics(
                fit_rows(frame, model=STOP_TIME_RUNG, outcome=outcome),
                fit_rows(frame, model=ARRIVAL_TIME_RUNG, outcome=outcome),
            )
            cohort_label = "" if prefix == "rate" else ", all-stops fit"
            for clock, rung in (
                ("time", STOP_TIME_RUNG),
                ("arrival", ARRIVAL_TIME_RUNG),
            ):
                edge = shape[f"{clock}_peak_edge"]
                interval = (
                    f"[{shape[f'{clock}_interval_start_s']:g}, "
                    f"{shape[f'{clock}_interval_end_s']:g}] s"
                )
                ranged_what[f"{prefix}_{clock}_peak_s"] = (
                    f"argmax of the median fitted {rung} time course, read on "
                    f"{interval}{cohort_label}"
                )
                values += [
                    make(
                        f"{prefix}_{clock}_peak_s_{word}",
                        shape[f"{clock}_peak_s"],
                        "quantity",
                        "dec2",
                        f"{label}: {ranged_what[f'{prefix}_{clock}_peak_s']}"
                        + (
                            "; it sits on the interval's edge, so it bounds the peak rather "
                            "than locating it"
                            if edge
                            else ""
                        ),
                        "s",
                    ),
                    make(
                        f"{prefix}_{clock}_peak_edge_{word}",
                        edge,
                        "count",
                        "count",
                        f"{label}: 1 if that argmax sits on the edge of {interval}{cohort_label}",
                    ),
                    make(
                        f"{prefix}_{clock}_peak_sessions_{word}",
                        shape[f"{clock}_peak_sessions"],
                        "count",
                        "count",
                        f"{label}: unflagged session curves behind that median{cohort_label}",
                    ),
                ]
            ranged_what[f"{prefix}_clock_ks"] = (
                "KS distance between the two clocks' median time courses, each "
                "normalized over its own supported interval and read on relative "
                f"position{cohort_label}"
            )
            values.append(
                make(
                    f"{prefix}_clock_ks_{word}",
                    shape["clock_ks"],
                    "quantity",
                    "dec2",
                    f"{label}: {ranged_what[f'{prefix}_clock_ks']}",
                )
            )
    no_swr_all = fit_rows(all_stops, model=FULL_RATE_RUNG, outcome="non_local_no_swr")
    paired = rate_fit_statistics.paired_lag_one_kernel_statistics(
        rate_fit_statistics.paired_lag_one_reward_kernels(
            fit_rows(fits, model=FULL_RATE_RUNG, outcome="non_local_swr"),
            fit_rows(fits, model=FULL_RATE_RUNG, outcome="non_local_no_swr"),
        )
    )
    paired_all = rate_fit_statistics.paired_lag_one_kernel_statistics(
        rate_fit_statistics.paired_lag_one_reward_kernels(
            fit_rows(all_stops, model=FULL_RATE_RUNG, outcome="non_local_swr"),
            no_swr_all,
        )
    )
    for stats in (paired, paired_all):
        _require_every_animal(stats["n_animals"], "the paired test")
    immediate_all_by_outcome = {
        outcome: gated_coefficient_statistics(
            fit_rows(all_stops, model=FULL_RATE_RUNG, outcome=outcome),
            REWARD_HISTORY_COLUMNS[0],
        )
        for outcome, _ in RATE_POPULATIONS.values()
    }
    immediate_all = immediate_all_by_outcome["non_local_no_swr"]
    immediate_all_note = _gate_note(immediate_all, REWARD_HISTORY_COLUMNS[0])
    kernel_all = rate_fit_statistics.reward_kernel_statistics(no_swr_all)
    _require_every_animal(kernel_all["n_animals"], "the non-SWR reward kernel")
    paired_what = (
        "the reconstructed lag-1 reward kernel, SWR-coupled minus non-SWR non-local "
        "events, within session, over the sessions in which both streams' reward "
        "terms pass the per-term separation gate"
    )
    values += [
        make(
            "rate_paired_sessions",
            paired["n_sessions"],
            "count",
            "count",
            f"sessions in the paired comparison of {paired_what}",
        ),
        make(
            "rate_paired_positive_fraction",
            paired["positive_fraction"],
            "fraction",
            "pct",
            f"share of paired sessions in which {paired_what} is positive",
        ),
        make(
            "rate_paired_animals_positive",
            paired["n_animals_positive"],
            "count",
            "count",
            "animals whose median paired difference is positive",
        ),
        make(
            "rate_paired_p",
            paired["signed_rank_p"],
            "quantity",
            "pval",
            "signed-rank p of the paired difference, treating sessions as independent",
        ),
        make(
            "rate_allstops_paired_sessions",
            paired_all["n_sessions"],
            "count",
            "count",
            "the same, all-stops fit: paired sessions",
        ),
        make(
            "rate_allstops_paired_positive_fraction",
            paired_all["positive_fraction"],
            "fraction",
            "pct",
            "the same, all-stops fit: share positive",
        ),
        make(
            "rate_allstops_paired_p",
            paired_all["signed_rank_p"],
            "quantity",
            "pval",
            "the same, all-stops fit: signed-rank p",
        ),
        make(
            "rate_immediate_ratio_swr_to_no_swr",
            gated_exp_median_ratio(
                fit_rows(fits, model=FULL_RATE_RUNG, outcome="non_local_swr"),
                fit_rows(fits, model=FULL_RATE_RUNG, outcome="non_local_no_swr"),
                REWARD_HISTORY_COLUMNS[0],
            ),
            "quantity",
            "sig2",
            "rate_immediate_session_non_local_swr divided by "
            f"rate_immediate_session_non_local_no_swr; {BASIS_COEFFICIENT_NOTE}",
            "ratio",
        ),
        make(
            "rate_allstops_immediate_session_no_swr",
            immediate_all["exp_median"],
            "quantity",
            "sig2",
            "all-stops fit, non-local without SWR: exp of the across-session median first "
            f"reward basis coefficient; {BASIS_COEFFICIENT_NOTE}",
            "rate ratio",
            pending=immediate_all_note,
        ),
        make(
            "rate_allstops_immediate_session_p_no_swr",
            immediate_all["signed_rank_p"],
            "quantity",
            "pval",
            "its signed-rank p (a basis coefficient, not an effect size)",
            pending=immediate_all_note,
        ),
        make(
            "rate_allstops_immediate_session_animals_no_swr",
            immediate_all["n_animals_same_sign"],
            "count",
            "count",
            "its animals with the median's sign",
            pending=immediate_all_note,
        ),
        make(
            "rate_allstops_kernel_lag_one_no_swr",
            kernel_all["rate_ratio_lag_1"],
            "quantity",
            "dec2",
            "all-stops fit, non-local without SWR: median reconstructed kernel at lag 1 "
            "over the sessions passing the reward gate",
            "rate ratio",
        ),
        make(
            "rate_allstops_kernel_lag_one_p_no_swr",
            kernel_all["lag_one_signed_rank_p"],
            "quantity",
            "pval",
            "all-stops fit, non-local without SWR: signed-rank p of the log lag-1 kernel "
            "against 0 across sessions",
        ),
        make(
            "rate_allstops_kernel_lag_one_animals_positive_no_swr",
            kernel_all["n_animals_lag_one_positive"],
            "count",
            "count",
            "all-stops fit, non-local without SWR: animals whose median lag-1 kernel is "
            "above 1",
        ),
    ]
    for word, (outcome, label) in RATE_POPULATIONS.items():
        stats = immediate_all_by_outcome[outcome]
        _require_every_animal(stats["n_animals"], outcome)
        values += pend(
            [
                make(
                    f"rate_allstops_immediate_animal_{word}",
                    stats["exp_animal_median"],
                    "quantity",
                    "sig2",
                    f"all-stops fit, {label}: rr_immediate (animal-balanced); "
                    f"{BASIS_COEFFICIENT_NOTE}",
                    "rate ratio",
                ),
                make(
                    f"rate_allstops_immediate_animals_positive_{word}",
                    stats["n_animals_positive"],
                    "count",
                    "count",
                    f"all-stops fit, {label}: animals positive",
                ),
            ],
            _gate_note(stats, REWARD_HISTORY_COLUMNS[0]),
        )
    for word, model in (
        ("const", "M0_const"),
        ("time", STOP_TIME_RUNG),
        ("full", FULL_RATE_RUNG),
        ("reward_by_time", "M9r_rew_x_time"),
    ):
        values += [
            make(
                f"rate_ladder_aic_{word}",
                ladder.loc[(model, "non_local"), "aic_median"],
                "quantity",
                "int",
                f"non-local (all): median AIC of {model}",
                "AIC",
            ),
            make(
                f"rate_ladder_cv_bits_{word}",
                ladder.loc[(model, "non_local"), "cv_bits_median"],
                "quantity",
                "dec2",
                f"non-local (all): median held-out bits per event of {model}, over sessions with a finite score",
                "bits/event",
            ),
            make(
                f"rate_ladder_cv_sessions_{word}",
                ladder.loc[(model, "non_local"), "n_cv_finite"],
                "count",
                "count",
                f"non-local (all): sessions with a finite held-out score for {model}",
            ),
            make(
                f"rate_ladder_ks_band_{word}",
                ladder.loc[(model, "non_local"), "ks_band_fraction"],
                "fraction",
                "dec2",
                f"non-local (all): share of {model} fits inside the KS band",
            ),
        ]
    return values + _population_ranges(values, make, ranged_what)


_LAG_WORDS = (
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
#: The stop-clock ladder rungs, by the word their keys use.
LADDER = {
    "const": "M0_const",
    "rew": "M1_rew",
    "time": STOP_TIME_RUNG,
    "histcos": "M3r_histcos",
    "time_histcos": "M6r_time_histcos",
    "full": FULL_RATE_RUNG,
    "reward_by_time": "M9r_rew_x_time",
}


def per_bin_detail_values(
    fits: pd.DataFrame, all_stops: pd.DataFrame, in_transit: pd.DataFrame, source: str
) -> list[Value]:
    """Question 3.3: the rest of the per-bin ladder's readout.

    The quartiles of the baseline rate, the single reward-basis coefficients
    over the sessions the per-term gate keeps, the reconstructed kernel at the
    lags between the first and the last, the self-history block's separation,
    the arrival-clock rungs' fit counts, the within-session baseline rate
    ratios, each stop-clock rung's fit quality, and what the arrival clock
    keeps of the all-stops and in-transit fits.
    """
    make = value_factory(per_bin_detail_values, source, TAGS["rate"])
    values: list[Value] = []
    ladder = rate_fit_statistics.ladder_fit_statistics(fits)
    transit_shares: dict[str, float] = {}
    for word, (outcome, label) in RATE_POPULATIONS.items():
        null = fit_rows(fits, model="M0_const", outcome=outcome)
        reward_only = fit_rows(fits, model="M1_rew", outcome=outcome)
        full = fit_rows(fits, model=FULL_RATE_RUNG, outcome=outcome)
        baseline = rate_fit_statistics.baseline_rate_statistics(null)
        raw = gated_coefficient_statistics(reward_only, REWARD_HISTORY_COLUMNS[0])
        raw_note = _gate_note(raw, REWARD_HISTORY_COLUMNS[0])
        kernel = rate_fit_statistics.reward_kernel_statistics(full)
        history = term_separation_statistics(
            fits_carrying(full, PER_BIN_HISTORY_COLUMNS), PER_BIN_HISTORY_COLUMNS
        )
        marginal = aic_difference_statistics(reward_only, null)
        values += [
            make(
                f"rate_baseline_lower_quartile_{word}",
                baseline["lower_quartile"],
                "quantity",
                "dec2",
                f"{label}: 25th percentile over sessions of the M0_const rate",
                "events/s",
            ),
            make(
                f"rate_baseline_upper_quartile_{word}",
                baseline["upper_quartile"],
                "quantity",
                "dec2",
                f"{label}: 75th percentile over sessions of the M0_const rate",
                "events/s",
            ),
            make(
                f"rate_bin_count_{word}",
                baseline["events_per_bin"],
                "quantity",
                "sig2",
                f"{label}: median M0_const rate times the {_BIN_MS} ms bin, the "
                "mean per-bin count",
                "events/bin",
            ),
            make(
                f"rate_raw_immediate_session_{word}",
                raw["exp_median"],
                "quantity",
                "sig2",
                f"{label}: exp of the across-session median {IMMEDIATE_COEF} of M1_rew "
                f"(uncontrolled); {BASIS_COEFFICIENT_NOTE}",
                "rate ratio",
                pending=raw_note,
            ),
            make(
                f"rate_marginal_reward_child_better_fraction_{word}",
                marginal["negative_fraction"],
                "fraction",
                "pct",
                f"{label}: share of sessions in which M1_rew has the lower AIC than M0_const",
            ),
            make(
                f"rate_hist_sessions_{word}",
                history["n_fits"],
                "count",
                "count",
                f"{label}: sessions whose {FULL_RATE_RUNG} fit carries the self-history block",
            ),
            make(
                f"rate_hist_separated_fraction_{word}",
                history["flagged_fraction"],
                "fraction",
                "pct",
                f"{label}: share of those fits flagging a self-history term "
                "(counted from event ends) separated",
            ),
        ]
        for k, bump in ((1, "one"), (2, "two")):
            agg = gated_coefficient_statistics(full, REWARD_HISTORY_COLUMNS[k])
            values += pend(
                [
                    make(
                        f"rate_bump_{bump}_session_{word}",
                        agg["exp_median"],
                        "quantity",
                        "sig2",
                        f"{label}: exp of the across-session median coef_{REWARD_HISTORY_COLUMNS[k]} "
                        f"of {FULL_RATE_RUNG}; {BASIS_COEFFICIENT_NOTE}",
                        "rate ratio",
                    ),
                    make(
                        f"rate_bump_{bump}_positive_fraction_{word}",
                        agg["positive_fraction"],
                        "fraction",
                        "pct",
                        f"{label}: share of sessions with that coefficient positive",
                    ),
                    make(
                        f"rate_bump_{bump}_animals_{word}",
                        agg["n_animals_same_sign"],
                        "count",
                        "count",
                        f"{label}: animals whose median has the across-session median's sign",
                    ),
                ],
                _gate_note(agg, REWARD_HISTORY_COLUMNS[k]),
            )
        for i, lag in enumerate(_LAG_WORDS[1:-1], start=1):  # lag ten is above
            values.append(
                make(
                    f"rate_kernel_lag_{lag}_{word}",
                    kernel[f"rate_ratio_lag_{i + 1}"],
                    "quantity",
                    "dec2",
                    f"{label}: median reconstructed reward kernel at lag {i + 1}",
                    "rate ratio",
                )
            )
        for rung_word, rung in (
            ("const", "M0a_const_arrival"),
            ("time", ARRIVAL_TIME_RUNG),
            ("full", "M7ra_time_arrival_histcos_rew"),
        ):
            values.append(
                make(
                    f"rate_arrival_{rung_word}_sessions_{word}",
                    ladder.loc[(rung, outcome), "n_finite_aic"],
                    "count",
                    "count",
                    f"{label}: {rung} fits with a finite AIC",
                )
            )
        transit = in_transit.attrs["arrival_clock_cohort"].get(outcome)
        if transit and transit["n_event_bins_frozen"]:
            transit_shares[word] = rate_fit_statistics.arrival_window_statistics(
                transit
            )["event_share"]
            values.append(
                make(
                    f"rate_intransit_arrival_event_share_{word}",
                    transit_shares[word],
                    "fraction",
                    "dec2",
                    f"{label}: share of its event bins the arrival window keeps on the in-transit fit",
                )
            )
    pairs = (
        ("swr_coupled_to_no_swr", "non_local_swr", "non_local_no_swr"),
        ("swr_to_non_local", "swr", "non_local"),
        ("swr_coupled_to_swr", "non_local_swr", "swr"),
        ("no_swr_to_swr", "non_local_no_swr", "swr"),
        ("swr_coupled_to_non_local", "non_local_swr", "non_local"),
    )
    for word, a, b in pairs:
        ratio = rate_fit_statistics.baseline_rate_ratio_statistics(
            fit_rows(fits, model="M0_const", outcome=a),
            fit_rows(fits, model="M0_const", outcome=b),
        )
        values += [
            make(
                f"rate_baseline_ratio_{word}",
                ratio["median_ratio"],
                "quantity",
                "dec2",
                f"median within-session M0_const rate ratio, {a} over {b}",
                "ratio",
            ),
            make(
                f"rate_baseline_ratio_{word}_p",
                ratio["signed_rank_p"],
                "quantity",
                "pval",
                "its paired signed-rank p",
            ),
        ]
    for word, model in LADDER.items():
        values.append(
            make(
                f"rate_ladder_sessions_{word}",
                ladder.loc[(model, "non_local"), "n_finite_aic"],
                "count",
                "count",
                f"non-local (all): sessions with a finite-AIC fit of {model}",
            )
        )
        if word in ("rew", "histcos", "time_histcos"):
            values += [
                make(
                    f"rate_ladder_aic_{word}",
                    ladder.loc[(model, "non_local"), "aic_median"],
                    "quantity",
                    "int",
                    f"non-local (all): median AIC of {model}",
                    "AIC",
                ),
                make(
                    f"rate_ladder_cv_bits_{word}",
                    ladder.loc[(model, "non_local"), "cv_bits_median"],
                    "quantity",
                    "dec2",
                    f"non-local (all): median held-out bits per event of {model}, over sessions with a finite score",
                    "bits/event",
                ),
                make(
                    f"rate_ladder_cv_sessions_{word}",
                    ladder.loc[(model, "non_local"), "n_cv_finite"],
                    "count",
                    "count",
                    f"non-local (all): sessions with a finite held-out score for {model}",
                ),
                make(
                    f"rate_ladder_ks_band_{word}",
                    ladder.loc[(model, "non_local"), "ks_band_fraction"],
                    "fraction",
                    "dec2",
                    f"non-local (all): share of {model} fits inside the KS band",
                ),
            ]
    for word, model in (
        ("time", ARRIVAL_TIME_RUNG),
        ("full", "M7ra_time_arrival_histcos_rew"),
    ):
        for pop in ("non_local", "non_local_swr"):
            values.append(
                make(
                    f"rate_arrival_{word}_cv_nan_fraction_{pop}",
                    ladder.loc[(model, pop), "cv_not_finite_fraction"],
                    "fraction",
                    "pct",
                    f"{RATE_POPULATIONS[pop][1]}: share of {model} fits whose held-out score is not finite",
                )
            )
    marginal_all = aic_difference_statistics(
        fit_rows(all_stops, model="M1_rew", outcome="non_local_no_swr"),
        fit_rows(all_stops, model="M0_const", outcome="non_local_no_swr"),
    )
    cohort = all_stops.attrs["arrival_clock_cohort"]["non_local"]
    window = rate_fit_statistics.arrival_window_statistics(cohort)
    transit_range = group_range(transit_shares)
    values += [
        make(
            "rate_allstops_marginal_reward_delta_aic_non_local_no_swr",
            marginal_all["median"],
            "quantity",
            "sig2s",
            "all-stops fit, non-local without SWR: median per-session AIC(M1_rew) - AIC(M0_const)",
            "AIC",
        ),
        make(
            "rate_allstops_marginal_reward_child_better_fraction_non_local_no_swr",
            marginal_all["negative_fraction"],
            "fraction",
            "pct",
            "all-stops fit, non-local without SWR: share of sessions in which M1_rew has the lower AIC",
        ),
        make(
            "rate_allstops_arrival_bins_dropped",
            cohort["n_bins_dropped_arrival_clock"],
            "count",
            "count",
            "all-stops fit: bins the arrival clock removed from the common cohort",
        ),
        make(
            "rate_allstops_arrival_bins_before",
            window["n_bins_before"],
            "count",
            "count",
            "all-stops fit: bins before that removal",
        ),
        make(
            "rate_allstops_arrival_bins_dropped_fraction",
            window["dropped_bin_fraction"],
            "fraction",
            "pct",
            "all-stops fit: share of bins the arrival clock removed",
        ),
        make(
            "rate_allstops_arrival_stops_dropped",
            cohort["n_stops_dropped_arrival_clock"],
            "count",
            "count",
            "all-stops fit: stops the arrival clock removed",
        ),
        make(
            "rate_allstops_arrival_window_bins",
            cohort["n_bins_arrival_window"],
            "count",
            "count",
            "all-stops fit: frozen-cohort bins inside the arrival window",
        ),
        make(
            "rate_allstops_frozen_bins",
            cohort["n_bins_frozen"],
            "count",
            "count",
            "all-stops fit: bins in the frozen common cohort",
        ),
        make(
            "rate_allstops_arrival_window_fraction",
            window["bin_share"],
            "fraction",
            "pct",
            "all-stops fit: share of frozen-cohort bins inside the arrival window",
        ),
        make(
            "rate_intransit_arrival_event_share_low",
            transit_range["low"],
            "fraction",
            "dec2",
            "lowest over populations of the in-transit fit's arrival-window event share",
        ),
        make(
            "rate_intransit_arrival_event_share_high",
            transit_range["high"],
            "fraction",
            "dec2",
            "highest over populations of the in-transit fit's arrival-window event share",
        ),
    ]
    return values


#: Per-population rate values whose range over the four populations is quoted,
#: each with the stem of its edge flags when it is a time-course peak.
_RANGED = {
    "rate_kernel_late_min": None,
    "rate_time_peak_s": "rate_time_peak_edge",
    "rate_arrival_peak_s": "rate_arrival_peak_edge",
    "rate_clock_ks": None,
    "rate_allstops_time_peak_s": "rate_allstops_time_peak_edge",
    "rate_allstops_arrival_peak_s": "rate_allstops_arrival_peak_edge",
    "rate_allstops_clock_ks": None,
    "rate_arrival_separated_fraction": None,
}


def _population_ranges(
    values: list[Value], make: Callable[..., Value], what: Mapping[str, str]
) -> list[Value]:
    """The lowest and highest of each :data:`_RANGED` value over the populations.

    A time-course peak on the edge of its supported interval bounds the peak
    rather than locating it, so the peak ranges are over the located peaks only,
    with the number of edge peaks beside them. ``what`` holds, by stem, what
    the value is without its population.
    """
    by_key = {v.key: v for v in values}
    ranges = []
    for stem, edge_stem in _RANGED.items():
        members = {word: by_key[f"{stem}_{word}"] for word in RATE_POPULATIONS}
        on_edge = None
        if edge_stem is not None:
            on_edge = {
                word: by_key[f"{edge_stem}_{word}"].value for word in RATE_POPULATIONS
            }
        span = group_range(
            {word: member.value for word, member in members.items()}, on_edge
        )
        if on_edge is not None:
            ranges.append(
                make(
                    f"{edge_stem}_count",
                    span["n_excluded"],
                    "count",
                    "count",
                    f"populations whose {what[stem]} sits on the interval's edge",
                )
            )
        for end in ("low", "high"):
            chosen = members[span[f"{end}_group"]]
            ranges.append(
                make(
                    f"{stem}_{end}",
                    chosen.value,
                    chosen.kind,
                    chosen.style,
                    f"{'lowest' if end == 'low' else 'highest'} over the populations"
                    + (
                        " whose peak is located (not on the edge)"
                        if edge_stem is not None
                        else ""
                    )
                    + f": {what[stem]}",
                    chosen.unit,
                    pending=chosen.pending,
                )
            )
    return ranges


# --- rate, per stop (Family A) -----------------------------------------------


def rate_contrast_separation_values(fits: pd.DataFrame, source: str) -> list[Value]:
    """How heavily the per-bin ``is_swr``-contrast rate ladder is separated: the
    share of its ``contrast`` fits flagged separated, per cohort, and the share of
    fits carrying ``is_swr:reward_history_rcos_1`` that flag that term."""
    make = value_factory(rate_contrast_separation_values, source, TAGS["rate"])
    contrast = fit_rows(fits, population="contrast")
    term = "is_swr:reward_history_rcos_1"
    carrying = term_separation_statistics(fits_carrying(fits, [term]), [term])
    values = [
        make(
            f"rate_contrast_separated_fraction_{cohort}",
            separated_fraction(fit_rows(contrast, cohort=cohort)),
            "fraction",
            "pct",
            f"share of the is_swr-contrast rate ladder's {cohort} contrast fits flagged "
            "separated",
        )
        for cohort in ("primary", "post_outcome")
    ]
    expanded = rung_separation_range(contrast, term)
    plain = term_separation_statistics(
        fit_rows(contrast, model="S1_rew"), ["reward_history_rcos_1"]
    )
    for column, what in (
        ("fits", "fits"),
        ("separated", "fits flagged separated"),
        ("flagged", f"fits flagging {term}"),
    ):
        for end, pick in (("low", "min"), ("high", "max")):
            values.append(
                make(
                    f"rate_contrast_expanded_{column}_{end}",
                    expanded[f"n_{column}_{pick}"],
                    "count",
                    "count",
                    f"{'fewest' if end == 'low' else 'most'} {what} over the "
                    f"{expanded['n_rungs']} contrast rungs with a fit carrying {term}",
                )
            )
    values += [
        make(
            "rate_contrast_plain_fits",
            plain["n_fits"],
            "count",
            "count",
            "S1_rew contrast fits",
        ),
        make(
            "rate_contrast_plain_flagged",
            plain["n_flagged"],
            "count",
            "count",
            "S1_rew contrast fits flagging reward_history_rcos_1 separated",
        ),
    ]
    values.append(
        make(
            "rate_contrast_term_separated_fraction",
            carrying["flagged_fraction"],
            "fraction",
            "pct",
            f"share of the fits carrying {term} that flag it separated",
        )
    )
    return values


def rate_contrast_audit_values(audited: pd.DataFrame, source: str) -> list[Value]:
    """The is_swr-contrast rate ladder's audited aggregate: how many rows it
    publishes, and how many sessions its key contrast terms lose to the gate."""
    make = value_factory(rate_contrast_audit_values, source, TAGS["rate"])
    drops = audited_drop_statistics(fit_rows(audited, population="contrast"))
    worst = f"{drops['most_dropped_model']} {drops['most_dropped_term']}"
    return [
        make(
            "rate_contrast_audited_rows",
            len(audited),
            "count",
            "count",
            "rows the audited is_swr-contrast rate aggregate publishes",
        ),
        make(
            "rate_contrast_terms_losing_sessions",
            drops["n_terms_losing_sessions"],
            "count",
            "count",
            "key is_swr contrast terms that lose any session to the per-term gate",
        ),
        make(
            "rate_contrast_most_dropped",
            drops["most_dropped"],
            "count",
            "count",
            f"most sessions a key contrast term loses to the gate ({worst})",
        ),
        make(
            "rate_contrast_most_dropped_of",
            drops["most_dropped_of"],
            "count",
            "count",
            "sessions with a coefficient for that term, kept and dropped together "
            f"({drops['most_dropped_model']})",
        ),
    ]


def cohort_audit_values(inputs: BehaviorInputs) -> list[Value]:
    """What the rate families' common-cohort freeze cost, as their runners'
    cohort audits record it (population non-local; every population shares it)."""
    values = []
    for prefix, path, rows in (
        ("rate_audit", inputs.per_bin_audit, {"modeled": "M0_const"}),
        ("per_stop_audit", inputs.per_stop_audit, {"modeled": "R0", "full": "R1"}),
    ):
        audit = pd.read_csv(path)
        audit = audit[audit["population"] == "non_local"].set_index("model")
        make = value_factory(
            cohort_audit_values, f"{rel(path)} (cohort audit)", TAGS["rate"]
        )
        first = audit.loc[rows["modeled"]]
        values += [
            make(
                f"{prefix}_stops_in",
                first["n_in"],
                "count",
                "count",
                f"{path.name}: stops entering the cohort",
            ),
            make(
                f"{prefix}_stops_own_drop",
                first["n_drop_own_predictor_missing"],
                "count",
                "count",
                f"{path.name}: stops dropped for a missing own predictor",
            ),
            make(
                f"{prefix}_stops_common_drop",
                first["n_drop_common_cohort_other"],
                "count",
                "count",
                f"{path.name}: stops dropped by the common-cohort freeze (another rung's covariate, here the arrival clock)",
            ),
        ]
        for word, model in rows.items():
            row = audit.loc[model]
            values += [
                make(
                    f"{prefix}_stops_{word}",
                    row["n_modeled"],
                    "count",
                    "count",
                    f"{path.name}: stops modelled on {model}",
                ),
                make(
                    f"{prefix}_stops_{word}_session_excluded",
                    row["n_excluded_session"],
                    "count",
                    "count",
                    f"{path.name}: stops in sessions {model} gates out",
                ),
            ]
    return values


def per_stop_values(fits: pd.DataFrame, source: str) -> list[Value]:
    """Question 3.3: the per-stop reward step ``R1 - R0`` on the well-anchored
    cohort (:func:`~continuum_swr_replay.analysis.behavior.coefficient_statistics.aic_difference_statistics`): a session counts
    only when both rungs saw the same stops; negative = child better."""
    make = value_factory(per_stop_values, source, TAGS["rate"])
    _require(
        set(fits["stop_cohort"]) == {"at_well"}, "per-stop fit is not the at_well one"
    )
    values = [
        make(
            "per_stop_cv_finite_fraction_full",
            known_fraction(fit_rows(fits, model="R1"), "cv_ll_per_stop_mean"),
            "fraction",
            "pct",
            "share of R1 session-fits (all populations) whose cv_ll_per_stop_mean is "
            "not missing",
        ),
    ]
    for word, (outcome, label) in RATE_POPULATIONS.items():
        step = aic_difference_statistics(
            fit_rows(fits, model="R1", outcome=outcome),
            fit_rows(fits, model="R0", outcome=outcome),
        )
        _require_every_animal(step["n_animals"], outcome)
        values += [
            make(
                f"per_stop_delta_aic_animal_{word}",
                step["animal_median"],
                "quantity",
                "sig2s",
                f"{label}: median over animals of the per-animal median AIC(R1) - AIC(R0)",
                "AIC",
            ),
            make(
                f"per_stop_delta_aic_session_{word}",
                step["median"],
                "quantity",
                "sig2s",
                f"{label}: median over sessions of AIC(R1) - AIC(R0)",
                "AIC",
            ),
            make(
                f"per_stop_sessions_{word}",
                step["n_sessions"],
                "count",
                "count",
                f"{label}: sessions with both rungs fit on the same stops",
            ),
            make(
                f"per_stop_child_better_fraction_{word}",
                step["negative_fraction"],
                "fraction",
                "pct",
                f"{label}: share of sessions in which R1 has the lower AIC",
            ),
            make(
                f"per_stop_animals_child_better_{word}",
                step["n_animals_negative"],
                "count",
                "count",
                f"{label}: animals whose median step is negative",
            ),
        ]
    return values
