"""The behavior summary's choice values: the event-source suite's choice block
(question 3.4, :func:`choice_values`) and the stay/leave fits (question 3.5,
:func:`stay_leave_values`). The statistics are computed in
``analysis.behavior.choice.learning`` and
``analysis.behavior.coefficient_statistics``."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.behavior.choice.learning import (
    cross_session_role_summary,
    fit_status_counts,
    pooled_gee_role_summary,
    smallest_adjusted_p,
)
from continuum_swr_replay.analysis.behavior.coefficient_statistics import (
    gated_coefficient_statistics,
)
from continuum_swr_replay.analysis.behavior.rate import (
    stop_statistics,
)
from continuum_swr_replay.reported.behavior.common import (
    TAGS,
    _gate_note,
    _require_every_animal,
)
from continuum_swr_replay.reported.values import (
    Value,
    pend,
    value_factory,
)

#: Stay/leave content strata, by the file suffix their bundles carry.
STAY_STRATA = {"all": "", "swr": "_swr", "non_swr": "_non_swr", "mua": "_mua"}


# --- choice and stay/leave --------------------------------------------------------


def choice_values(fits: dict, source: str) -> list[Value]:
    """Question 3.4: replay fraction in the conditional-logit choice block."""
    make = value_factory(choice_values, source, TAGS["choice"])
    values = []
    for word, block, model, term in (
        (
            "choice_non_swr",
            "choice_swr_split",
            "C_full_swr_split",
            "replay_fraction_non_swr",
        ),
        ("choice_swr", "choice_swr_split", "C_full_swr_split", "replay_fraction_swr"),
        ("choice_full", "choice", "C_full", "replay_fraction"),
        ("choice_raw", "choice", "C_replay_only", "replay_fraction"),
    ):
        stats = gated_coefficient_statistics(fits[block][model], term)
        _require_every_animal(stats["n_animals"], term)
        what = f"{model} {term}"
        values += pend(
            [
                make(
                    word,
                    stats["median"],
                    "quantity",
                    "sig2s",
                    f"{what}: session median coefficient (gated)",
                    "log odds",
                ),
                make(
                    f"{word}_p",
                    stats["signed_rank_p"],
                    "quantity",
                    "pval",
                    f"{what}: signed-rank p across sessions",
                ),
                make(
                    f"{word}_sessions",
                    stats["n_sessions"],
                    "count",
                    "count",
                    f"{what}: sessions kept",
                ),
                make(
                    f"{word}_positive_sessions",
                    stats["n_positive"],
                    "count",
                    "count",
                    f"{what}: sessions with a positive coefficient",
                ),
                make(
                    f"{word}_negative_sessions",
                    stats["n_negative"],
                    "count",
                    "count",
                    f"{what}: sessions with a negative coefficient",
                ),
                make(
                    f"{word}_animals_positive",
                    stats["n_animals_positive"],
                    "count",
                    "count",
                    f"{what}: animals with a positive median",
                ),
                make(
                    f"{word}_significant_fraction",
                    stats["significant_fraction"],
                    "fraction",
                    "pct",
                    f"{what}: share of sessions individually significant",
                ),
            ],
            _gate_note(stats, term),
        )
    return values


def stay_leave_values(
    bundles: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    stops: pd.DataFrame,
    source: str,
) -> list[Value]:
    """Question 3.5: stay/leave against content, per stratum.

    The per-session Wilcoxon is ``learning.cross_session_role_summary`` (the
    runner's own); the pooled GEE's q is BH across converged roles within a
    stratum, as ``run_stay_leave.py`` logs it.
    """
    make = value_factory(stay_leave_values, source, TAGS["stay_leave"])
    values = []
    for stratum, (logistic, _) in bundles.items():
        fits = fit_status_counts(logistic)
        values += [
            make(
                f"stay_converged_{stratum}",
                fits["n_converged"],
                "count",
                "count",
                f"stratum {stratum}: converged session-by-role logistic fits",
            ),
            make(
                f"stay_fits_{stratum}",
                fits["n_fits"],
                "count",
                "count",
                f"stratum {stratum}: session-by-role rows of the logistic fits, the "
                "sessions skipped before fitting (too few stops, a constant stay "
                "or outcome) included",
            ),
        ]
    signrank = smallest_adjusted_p(
        {
            stratum: cross_session_role_summary(logistic)
            for stratum, (logistic, _) in bundles.items()
        }
    )
    gee = smallest_adjusted_p(
        {
            stratum: pooled_gee_role_summary(pooled)
            for stratum, (_, pooled) in bundles.items()
        }
    )
    cohort = stop_statistics.stop_cohort_counts(stops)
    values += [
        make(
            "stay_min_q_signrank",
            signrank["p_fdr"],
            "quantity",
            "pval",
            f"smallest FDR-adjusted per-session Wilcoxon p over roles and strata (role {signrank['role']}, stratum {signrank['stratum']})",
        ),
        make(
            "stay_min_q_gee",
            gee["p_fdr"],
            "quantity",
            "pval",
            f"smallest FDR-adjusted pooled-GEE p over roles and strata (role {gee['role']}, stratum {gee['stratum']})",
        ),
        make(
            "stops_total",
            cohort["n_stops"],
            "count",
            "count",
            "stops in the canonical stop table",
            tags=TAGS["stay_leave_rate"],
        ),
        make(
            "stops_well_anchored",
            cohort["n_well_anchored"],
            "count",
            "count",
            "stops with occupancy_state at_well or mixed (the shared cohort)",
            tags=TAGS["stay_leave_rate"],
        ),
        make(
            "stops_dropped",
            cohort["n_not_anchored"],
            "count",
            "count",
            "stops the well-anchored cohort leaves out",
        ),
        make(
            "stops_dropped_with_stay",
            cohort["n_not_anchored_with_stay"],
            "count",
            "count",
            "of those, stops carrying an is_stay",
        ),
        make(
            "stops_mixed",
            cohort["n_mixed"],
            "count",
            "count",
            f"mixed stops ({cohort['n_mixed_with_stay']} of them with an is_stay)",
        ),
        make(
            "stops_mixed_with_stay",
            cohort["n_mixed_with_stay"],
            "count",
            "count",
            "mixed stops carrying an is_stay",
        ),
        make(
            "stops_at_well_with_stay",
            cohort["n_at_well_with_stay"],
            "count",
            "count",
            "at_well stops carrying an is_stay (Stage 6's population)",
        ),
    ]
    return values
