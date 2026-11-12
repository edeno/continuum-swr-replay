"""The behavior summary's content values (questions 3.1, 3.2, 3.6 and 3.7):
which arms the events represent.

The content ``is_swr``-contrast ladders (:func:`content_contrast_values`,
:func:`breadth_derived_values`), the three-estimator comparison
(:func:`estimator_values`), the leaf-only arm ladder (:func:`leaf_values`),
the within-arm horse race (:func:`horserace_values`), the trunk shares
(:func:`trunk_values`) and the paired current-arm difference
(:func:`current_arm_paired_values`). The statistics are computed in
``analysis.behavior.content`` and ``analysis.behavior.coefficient_statistics``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.coefficient_statistics import (
    fit_rows,
    gated_coefficient_difference_statistics,
    gated_coefficient_statistics,
)
from continuum_swr_replay.analysis.behavior.content import content_statistics
from continuum_swr_replay.analysis.behavior.content.model_comparison import (
    compare_nested_models,
    summarize_comparison,
)
from continuum_swr_replay.reported.behavior.common import TAGS, _gate_note, _require
from continuum_swr_replay.reported.values import (
    Tags,
    Value,
    pend,
    value_factory,
)

#: Content contrast specs: key word -> (audited file suffix, model, population, term).
#: The suffix picks the bundle: "" fraction / all stratum, "_rewarded" and
#: "_omitted" the outcome strata, "_dm" the Dirichlet-multinomial, "_dm_len" the
#: Dirichlet-multinomial with the length-dependent concentration.
_POPS = {"all": "all", "swr": "SWR", "non_swr": "nonSWR"}
_SPLIT_RUNGS = {
    "base_current": ("C2_base", "is_current_arm"),
    "base_current_no_controls": ("C1_current", "is_current_arm"),
    "armreward": ("C3_armreward", "reward_fraction_candidate_arm_last10_visits"),
    "armvalue": ("C4_armvalue", "reward_belief_mean_candidate_arm_preoutcome"),
    "rpe_current": ("C6_rpe", "rpe_postoutcome_z_x_is_current_arm"),
    "relative_value_current": (
        "C7_value",
        "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
    ),
    "entropy_current": (
        "C8_uncert",
        "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm",
    ),
    "update_current": (
        "C9_update",
        "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
    ),
    "periswitch_current": (
        "C10_periswitch",
        "periswitch_index_trials_z_x_is_current_arm",
    ),
    "block_current": ("C11_block", "trials_since_block_change_z_x_is_current_arm"),
    "switchrate_current": (
        "C12_switchrate",
        "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm",
    ),
    "armuncert": ("C13_armuncert", "reward_belief_variance_candidate_arm_preoutcome_z"),
}
_POOLED_RUNGS = {
    "swr_current_direct": ("CS1_current", "swr_x_is_current_arm"),
    "swr_current_total": ("CS1nt_current", "swr_x_is_current_arm"),
    "swr_prev_direct": ("CS2_prev", "swr_x_is_prev_arm"),
    "swr_armreward_direct": (
        "CS4_armreward",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
    ),
    "swr_armreward_total": (
        "CS4nt_armreward",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
    ),
    "swr_armvalue_direct": (
        "CS5_armvalue",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    ),
    "swr_armvalue_total": (
        "CS5nt_armvalue",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    ),
    "swr_prevgoal": (
        "CS5b_valuevsprevgoal",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    ),
    "swr_armuncert_direct": (
        "CS6_armuncert",
        "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
    ),
    "swr_armuncert_total": (
        "CS6nt_armuncert",
        "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
    ),
    "patch_current": ("CP0_base", "is_current_patch"),
    "swr_patch_current_direct": ("CPS1_current", "swr_x_is_current_patch"),
    "swr_patch_current_total": ("CPS1nt_current", "swr_x_is_current_patch"),
}


def _content_specs() -> dict[str, tuple[str, str, str, str]]:
    specs: dict[str, tuple[str, str, str, str]] = {}
    for word, (model, term) in _SPLIT_RUNGS.items():
        for pop_word, pop in _POPS.items():
            specs[f"{word}_{pop_word}"] = ("", model, pop, term)
    for word, (model, term) in _POOLED_RUNGS.items():
        specs[word] = ("", model, "all", term)
    for word in ("swr_current_direct", "swr_armvalue_direct", "swr_armvalue_total"):
        model, term = _POOLED_RUNGS[word]
        specs[f"{word}_rewarded"] = ("_rewarded", model, "all", term)
    model, term = _POOLED_RUNGS["swr_current_direct"]
    specs["swr_current_direct_omitted"] = ("_omitted", model, "all", term)
    for word in (
        "swr_armvalue_direct",
        "swr_armvalue_total",
        "swr_prevgoal",
        "swr_armuncert_direct",
        "swr_armuncert_total",
        "swr_armreward_direct",
    ):
        model, term = _POOLED_RUNGS[word]
        specs[f"dm_{word}"] = ("_dm", model, "all", term)
    specs["dm_breadth"] = ("_dm", "C2_base", "all", "log_phi_is_swr")
    specs["dm_breadth_length_adjusted"] = (
        "_dm_len",
        "C2_base",
        "all",
        "log_phi_is_swr",
    )
    specs["dm_breadth_length_term"] = (
        "_dm_len",
        "C2_base",
        "all",
        "log_phi_log_n_nonlocal_bins_z",
    )
    return specs


CONTENT_SPECS = _content_specs()
"""Every content-contrast aggregate the behavior summary reports."""

#: Estimator-comparison terms: key word -> (rung, term).
ESTIMATOR_TERMS = {
    "current": ("C2_base", "is_current_arm"),
    "swr_current": ("CS1_current", "swr_x_is_current_arm"),
    "swr_armreward": (
        "CS4_armreward",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
    ),
    "swr_armvalue": (
        "CS5_armvalue",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    ),
    "armvalue": ("CS5_armvalue", "reward_belief_mean_candidate_arm_preoutcome"),
}
ESTIMATORS = {
    "count": "conditional_poisson_count",
    "fraction": "conditional_poisson_fraction",
    "dm": "dirichlet_multinomial",
}

#: Leaf-ladder steps: key word -> (parent, child).
LEAF_STEPS = {"uncertainty": ("M4", "Mu"), "value": ("M4", "Mv"), "both": ("Mu", "Mvu")}

#: The leaf ladder's rung files the summary reads: the steps' rungs, each once,
#: in the order the steps name them.
LEAF_RUNGS = tuple(dict.fromkeys(rung for step in LEAF_STEPS.values() for rung in step))


def trunk_values(arm: pd.DataFrame, source: str) -> list[Value]:
    """Question 3.1: SWR content shifts onto the trunk arms, and which trunk.

    Each animal's mean share over its events, SWR and non-SWR; the value is the
    median over animals, with the animals where the SWR share is higher.
    """
    make = value_factory(trunk_values, source, TAGS["anchoring"])
    events = content_statistics.trunk_shares(arm)
    shares = content_statistics.trunk_share_statistics(events)
    current_patch = content_statistics.current_patch_trunk_fraction(events)
    values: list[Value] = []
    labels = {
        "trunk": "all three trunks",
        "current_patch_trunk": "the trunk of the current arm's patch",
        "other_trunks": "the other two trunks",
    }
    for column, label in labels.items():
        word = {"trunk": "", "current_patch_trunk": "_current_patch"}.get(
            column, "_other_patches"
        )
        values += [
            make(
                f"trunk_share{word}_swr",
                shares.loc[column, "swr_median"],
                "fraction",
                "pct",
                f"SWR events: mean share of an event's content on {label}, median "
                f"over the {shares.loc[column, 'n_animals']} animals (at-well events)",
            ),
            make(
                f"trunk_share{word}_non_swr",
                shares.loc[column, "non_swr_median"],
                "fraction",
                "pct",
                f"non-SWR events: mean share of an event's content on {label}, "
                "median over animals",
            ),
            make(
                f"trunk_share{word}_animals_swr_higher",
                shares.loc[column, "n_animals_swr_higher"],
                "count",
                "count",
                f"animals whose SWR share on {label} is above their non-SWR share",
            ),
        ]
    for flag, word in ((True, "swr"), (False, "non_swr")):
        values.append(
            make(
                f"trunk_current_patch_fraction_{word}",
                current_patch[word],
                "fraction",
                "pct",
                f"{'SWR' if flag else 'non-SWR'} events: the current patch's trunk "
                "as a share of all trunk content (1/3 if the three trunks shared it "
                "evenly), median over animals",
            )
        )
    return values


def current_arm_paired_values(fits: pd.DataFrame, source: str) -> list[Value]:
    """Question 3.1: the within-session SWR - non-SWR difference of the separately
    fitted current-arm preference (``C2_base`` ``is_current_arm``).

    A session enters when both of its split fits keep the term
    (:func:`gate_term_sessions`); each animal is the median of its sessions'
    differences, and the value is the median over animals.
    """
    make = value_factory(current_arm_paired_values, source, TAGS["anchoring"])
    stats = gated_coefficient_difference_statistics(
        fit_rows(fits, model="C2_base", population="SWR"),
        fit_rows(fits, model="C2_base", population="nonSWR"),
        "is_current_arm",
    )
    _require(
        stats["n_sessions"] > 0, "no session keeps is_current_arm in both split fits"
    )
    return pend(
        [
            make(
                "current_arm_paired_difference",
                stats["animal_median"],
                "quantity",
                "sig2s",
                "SWR minus non-SWR is_current_arm (C2_base, fitted separately), the "
                "difference taken within each session, median over the session "
                f"differences per animal, then over the {stats['n_animals']} animals "
                f"({stats['n_sessions']} sessions with both fits kept)",
                "log weight",
            ),
            make(
                "current_arm_paired_difference_animals_positive",
                stats["n_animals_positive"],
                "count",
                "count",
                "animals whose median within-session difference is positive",
            ),
        ],
        _gate_note(stats, "is_current_arm"),
    )


# --- content -------------------------------------------------------------------


#: The tags of each content key word, by its leading words.
_CONTENT_TAGS = (
    ("swr_armreward", TAGS["content_reward_value"]),
    ("swr_armvalue", TAGS["content_value"]),
    ("swr_prevgoal", TAGS["content_value"]),
    ("swr_armuncert", TAGS["content_uncertainty"]),
    ("armreward", TAGS["content_reward"]),
    ("armvalue", TAGS["content_reward_uncertainty"]),
    ("armuncert", TAGS["content_uncertainty"]),
    ("switchrate", TAGS["content_value"]),
)


def _content_tags(word: str) -> Tags:
    """The tags of a content key: by its leading words, else content anchoring."""
    return next(
        (tags for lead, tags in _CONTENT_TAGS if word.startswith(lead)),
        TAGS["anchoring"],
    )


def content_contrast_values(
    audited: dict[str, pd.DataFrame], sources: dict[str, str]
) -> list[Value]:
    """Questions 3.1, 3.2, 3.6 and 3.7: the content ``is_swr``-contrast ladders.

    Read from each bundle's audited aggregate as the fitting script wrote it:
    the animal-balanced median, the sessions kept by the per-term gate, the
    animals with a positive median, and the session signed-rank BH-FDR q.
    """
    values = []
    for word, (suffix, model, pop, term) in CONTENT_SPECS.items():
        frame = audited[suffix]
        rows = frame[
            (frame["model"] == model)
            & (frame["population"] == pop)
            & (frame["term"] == term)
        ]
        _require(
            len(rows) == 1, f"{word}: {len(rows)} audited rows for {model}/{pop}/{term}"
        )
        row = rows.iloc[0]
        make = value_factory(
            content_contrast_values,
            sources[suffix],
            _content_tags(word.removeprefix("dm_")),
        )
        estimator = "Dirichlet-multinomial" if suffix.startswith("_dm") else "fraction"
        stratum = suffix.removeprefix("_dm").removeprefix("_len").strip("_") or "all"
        what = (
            f"{model} {term} ({estimator}, population {pop}, outcome stratum {stratum})"
        )
        if not suffix == "_omitted":
            _require(
                int(row["n_animals"]) == 5, f"{word}: {row['n_animals']} animals, not 5"
            )
        values += [
            make(
                word,
                row["coef_animal_median"],
                "quantity",
                "sig2s",
                f"{what}: animal-balanced median coefficient",
                "log weight",
            ),
            make(
                f"{word}_sessions",
                row["n_sessions"],
                "count",
                "count",
                f"{what}: sessions the per-term gate keeps",
            ),
            make(
                f"{word}_animals_positive",
                row["n_animals_pos"],
                "count",
                "count",
                f"{what}: animals with a positive median",
            ),
        ]
        if word.startswith("dm_breadth"):
            values.append(
                make(
                    f"{word}_positive_fraction",
                    row["pct_positive_session"],
                    "fraction",
                    "pct",
                    f"{what}: share of sessions with a positive coefficient",
                )
            )
        if word.endswith("prevgoal"):
            values.append(
                make(
                    f"{word}_separated",
                    row["n_sessions_flagged_separated"],
                    "count",
                    "count",
                    f"{what}: session fits flagged separated",
                )
            )
        if suffix == "_omitted":
            values.append(
                make(
                    f"{word}_animals",
                    row["n_animals"],
                    "count",
                    "count",
                    f"{what}: animals contributing",
                )
            )
        if np.isfinite(row["session_signrank_fdr_q"]):
            values.append(
                make(
                    f"{word}_q",
                    row["session_signrank_fdr_q"],
                    "quantity",
                    "pval",
                    f"{what}: session signed-rank BH-FDR q (across the bundle's rungs per population)",
                )
            )
    return values


def breadth_derived_values(
    values: dict[str, Value], events: pd.DataFrame, source: str
) -> list[Value]:
    """Question 3.1: what the concentration contrast means, and event length."""
    make = value_factory(breadth_derived_values, source, TAGS["anchoring"])
    contrast = content_statistics.concentration_contrast_statistics(
        values["dm_breadth"].value, values["dm_breadth_length_adjusted"].value
    )
    length = content_statistics.event_length_statistics(events)
    return [
        make(
            "dm_breadth_ratio",
            contrast["ratio"],
            "quantity",
            "sig2",
            "exp of the concentration contrast log_phi_is_swr on C2_base: an SWR event's Dirichlet concentration over a non-SWR event's",
            "ratio",
        ),
        make(
            "dm_breadth_length_adjusted_ratio",
            contrast["length_adjusted_ratio"],
            "quantity",
            "sig2",
            "the same with the length-dependent concentration",
            "ratio",
        ),
        make(
            "dm_breadth_length_share",
            contrast["length_share"],
            "fraction",
            "pct",
            "share of the concentration contrast that the length term absorbs: 1 - adjusted / total",
        ),
        make(
            "event_bins_median_swr",
            length["median_bins_swr"],
            "quantity",
            "int",
            "median non-local bins of SWR-overlapping events (event table)",
            "bins",
        ),
        make(
            "event_bins_median_non_swr",
            length["median_bins_non_swr"],
            "quantity",
            "int",
            "median non-local bins of non-SWR events (event table)",
            "bins",
        ),
        make(
            "event_bins_median",
            length["median_bins"],
            "quantity",
            "int",
            "median non-local bins per event, all events (event table)",
            "bins",
        ),
        make(
            "event_bins_ratio",
            length["swr_to_non_swr_ratio"],
            "quantity",
            "sig2",
            "ratio of those medians",
            "ratio",
        ),
    ]


def estimator_values(
    summary: pd.DataFrame, bootstrap: pd.DataFrame, source: str
) -> list[Value]:
    """Questions 3.1, 3.6 and 3.7: the three-estimator comparison on one shared event
    set. Coefficients are animal-balanced medians as the comparison writes them;
    intervals are its animal bootstrap's 2.5 and 97.5 percentiles."""
    make = value_factory(estimator_values, source, TAGS["estimators"])
    values = []
    for word, (rung, term) in ESTIMATOR_TERMS.items():
        (row,) = summary[
            (summary["model"] == rung) & (summary["term"] == term)
        ].itertuples()
        for est_word, estimator in ESTIMATORS.items():
            what = f"{rung} {term}, {estimator}"
            _require(
                getattr(row, f"n_animals_{est_word}") == 5, f"{what}: not 5 animals"
            )
            values += [
                make(
                    f"est_{word}_{est_word}",
                    getattr(row, f"median_coef_{est_word}"),
                    "quantity",
                    "sig2s",
                    f"{what}: animal-balanced median",
                    "log weight",
                ),
                make(
                    f"est_{word}_{est_word}_animals_positive",
                    getattr(row, f"n_animals_pos_{est_word}"),
                    "count",
                    "count",
                    f"{what}: animals positive",
                ),
                make(
                    f"est_{word}_{est_word}_sessions",
                    getattr(row, f"n_sessions_{est_word}"),
                    "count",
                    "count",
                    f"{what}: sessions",
                ),
            ]
            boot = bootstrap[
                (bootstrap["model"] == rung)
                & (bootstrap["term"] == term)
                & (bootstrap["estimator"] == estimator)
            ]
            _require(len(boot) == 1, f"{what}: {len(boot)} bootstrap rows")
            values += [
                make(
                    f"est_{word}_{est_word}_low",
                    boot["p2_5"].iloc[0],
                    "quantity",
                    "sig2s",
                    f"{what}: animal bootstrap 2.5th percentile",
                    "log weight",
                ),
                make(
                    f"est_{word}_{est_word}_high",
                    boot["p97_5"].iloc[0],
                    "quantity",
                    "sig2s",
                    f"{what}: animal bootstrap 97.5th percentile",
                    "log weight",
                ),
            ]
        if word == "swr_armvalue":
            for est_word in ESTIMATORS:
                values.append(
                    make(
                        f"est_armvalue_rung_delta_aic_{est_word}",
                        getattr(row, f"median_delta_aic_{est_word}"),
                        "quantity",
                        "sig2s",
                        f"CS5_armvalue against C2_base_swr_leaf (the same base and SWR arm baselines, without the two value terms), {ESTIMATORS[est_word]}: animal-balanced median per-session AIC(rung) - AIC(parent); negative = rung better",
                        "AIC",
                    )
                )
    boot = bootstrap[
        (bootstrap["model"] == "C2_base")
        & (bootstrap["term"] == "log_phi_is_swr")
        & (bootstrap["estimator"] == ESTIMATORS["dm"])
    ]
    _require(len(boot) == 1, f"C2_base log_phi_is_swr: {len(boot)} bootstrap rows")
    for end, column in (("low", "p2_5"), ("high", "p97_5")):
        values.append(
            make(
                f"est_breadth_dm_{end}",
                boot[column].iloc[0],
                "quantity",
                "sig2s",
                f"C2_base log_phi_is_swr, {ESTIMATORS['dm']}: animal bootstrap "
                f"{'2.5th' if end == 'low' else '97.5th'} percentile",
                "log concentration",
                tags=TAGS["estimators_anchoring"],
            )
        )
    return values


def leaf_values(leaf_dir: Path, source: str) -> list[Value]:
    """Question 3.7: the leaf-only arm ladder's uncertainty and value rungs.

    Each step is compared from the two rungs' per-session fits
    (:func:`~continuum_swr_replay.analysis.behavior.content.model_comparison.summarize_comparison`),
    not read from the bundle's ``model_comparison.pkl``: the AIC difference is
    the child's minus the parent's, and a stored comparison written before that
    convention holds the opposite sign."""
    make = value_factory(leaf_values, source, TAGS["leaf_ladder"])
    rungs = {name: pd.read_pickle(leaf_dir / f"{name}.pkl") for name in LEAF_RUNGS}
    values = []
    for word, model, term in (
        ("leaf_uncertainty", "Mu", "reward_belief_variance_candidate_arm_preoutcome_z"),
        ("leaf_value", "Mv", "reward_belief_mean_candidate_arm_preoutcome"),
    ):
        stats = gated_coefficient_statistics(rungs[model], term)
        _require(stats["n_animals"] == 5, f"{term}: {stats['n_animals']} animals")
        what = f"leaf ladder {model} {term}"
        values += pend(
            [
                make(
                    word,
                    stats["median"],
                    "quantity",
                    "sig2s",
                    f"{what}: session median coefficient",
                    "log weight",
                ),
                make(
                    f"{word}_sessions",
                    stats["n_sessions"],
                    "count",
                    "count",
                    f"{what}: sessions kept",
                ),
                make(
                    f"{word}_animals_positive",
                    stats["n_animals_positive"],
                    "count",
                    "count",
                    f"{what}: animals positive",
                ),
                make(
                    f"{word}_se",
                    stats["session_se_median"],
                    "quantity",
                    "sig2",
                    f"{what}: median event-clustered session SE",
                    "log weight",
                ),
            ],
            _gate_note(stats, term),
        )
    for word, (parent, child) in LEAF_STEPS.items():
        step = summarize_comparison(
            compare_nested_models(rungs[parent], rungs[child]), parent, child
        )
        what = f"leaf ladder {parent} to {child}"
        _require(step["n_animals"] == 5, f"{what}: not 5 animals")
        values += [
            make(
                f"leaf_step_{word}",
                step["delta_aic_median"],
                "quantity",
                "sig2s",
                f"{what}: median per-session AIC(child) - AIC(parent); negative = child better",
                "AIC",
            ),
            make(
                f"leaf_step_{word}_clear",
                step["child_better_aic"],
                "count",
                "count",
                f"{what}: sessions in which the child's AIC is lower by more than 2",
            ),
            make(
                f"leaf_step_{word}_sessions",
                step["n_sessions"],
                "count",
                "count",
                f"{what}: sessions compared",
            ),
            make(
                f"leaf_step_{word}_animals",
                step["n_animals_child_better"],
                "count",
                "count",
                f"{what}: animals whose median step is negative (favours the child)",
            ),
        ]
    return values


# --- the within-arm horse race ------------------------------------------------

#: Horse-race predictors: key word -> (summary column, label).
HORSERACE_PREDICTORS = {
    "uncertainty": (
        "reward_belief_variance_candidate_arm_preoutcome_z",
        "reward-belief variance (uncertainty)",
    ),
    "value": (
        "reward_belief_mean_candidate_arm_preoutcome_z",
        "reward-belief mean (value)",
    ),
    "distance": ("distance_to_candidate_arm_cm_z", "distance to the arm"),
    "recency": ("visit_recency_trials_z", "visit recency"),
}
HORSERACE_POPULATIONS = {"all": "all", "swr": "SWR", "non_swr": "non_SWR"}


def horserace_values(summary: pd.DataFrame, source: str) -> list[Value]:
    """Question 3.7: are uncertainty, value and visit history separated?

    ``summary`` is ``horserace_summary.csv`` (``uncertainty_content.
    summarize_horserace``): per population and predictor, the median over
    sessions of the within-arm OLS coefficient (all four predictors entered
    together, arm-demeaned within session; non-current leaf arms), the animals
    whose own median has that sign, and the median design condition number.
    """
    make = value_factory(horserace_values, source, TAGS["content_uncertainty"])
    values: list[Value] = []
    for pop_word, pop in HORSERACE_POPULATIONS.items():
        rows = summary[summary["population"] == pop].set_index("predictor")
        _require(len(rows) == len(HORSERACE_PREDICTORS), f"horse race: {pop} rows")
        for word, (column, label) in HORSERACE_PREDICTORS.items():
            row = rows.loc[column]
            values += [
                make(
                    f"horserace_{word}_{pop_word}",
                    row["median"],
                    "quantity",
                    "sig2s",
                    f"{pop} events: within-arm coefficient of {label}, the four "
                    "predictors entered together; median over sessions",
                    "arm fraction per SD",
                ),
                make(
                    f"horserace_{word}_{pop_word}_animals",
                    int(row["n_animals_consistent"]),
                    "count",
                    "count",
                    f"{pop} events: animals whose own median {label} coefficient has "
                    "the sign of the overall median",
                ),
            ]
        values.append(
            make(
                f"horserace_condition_number_{pop_word}",
                float(rows["median_cond_number"].iloc[0]),
                "quantity",
                "dec2",
                f"{pop} events: median per-session condition number of the "
                "four-predictor design (collinearity check; below 10 = separable)",
            )
        )
    return values
