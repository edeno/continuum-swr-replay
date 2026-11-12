"""Coefficient figures for the is_swr rate + content contrast ladders.

For each effect: the coefficient of every session the per-term gate keeps is a faint
jittered dot (colored by animal; a term the gate refuses is drawn empty and labelled
"(not reported)"); the 5 larger dots are per-animal medians; the black bar is the
across-animal median; the horizontal whisker is its 95% interval (nested bootstrap:
the animals, then each drawn animal's sessions, each draw's statistic the median over
animals of their medians -- the animal is the inferential unit, n=5; the median plus
or minus a small-sample t correction over the animals times the draws' SD;
``coefficient_statistics.animal_median_bootstrap``, 4000 draws). Continuous-predictor
coefficients are shown STANDARDIZED (per 1 SD of the predictor: raw_coef * SD, which
is exact for these Poisson GLMs) so effect sizes are comparable across predictors,
value constructs, and the two ladders. Binary-predictor terms (marked with a dagger)
are per-category and are not standardized. The reward-history basis features are
weighted reward fractions in [0, 1] (marked with a double dagger); their coefficients
are per unit of the fraction -- the effect of a fully rewarded recent history versus a
fully unrewarded one -- and are likewise not standardized.

Per-SD units are read from each fit's own record, never recomputed here: the
per-arm covariate SDs from the content fit's provenance (the per-leaf-arm unit),
and the rate covariate SDs from the rate fit's meta (the at-well stop-cohort
unit), so this figure states its effects in exactly the units the audited tables
use. Content state-modulation rungs (rpe/relval/entropy/
update/tss/block/switchrate x current) are already z-scored at fit time, so their
standardization factor is 1.0.

    uv run python scripts/figures/plot_swr_contrast_coefs.py
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.coefficient_statistics import (
    animal_median_bootstrap,
)
from continuum_swr_replay.analysis.behavior.contrast_aggregation import (
    gate_term_sessions,
    load_fit_time_sd_reference,
    load_per_sd_scale,
    std_factor,
)
from continuum_swr_replay.analysis.behavior.fit_provenance import require_fresh_inputs
from continuum_swr_replay.analysis.behavior.model_specs import (
    POST_OUTCOME_COHORT,
    POST_OUTCOME_RATE_RUNGS,
    PRIMARY_COHORT,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_animals
from continuum_swr_replay.paths import (
    figure_dir,
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.cli import fit_figure_parser


def fit_artifact_paths() -> dict[str, Path]:
    """The fitted artifacts the figures are drawn from, by name.

    ``rate`` and its ``rate_meta``: the per-SD units the audited rate table
    states its coefficients in, SDs over the at-well stop cohort the fits name,
    recorded by the rate fitter beside the audited table (the full-table SDs
    differ by up to ~7 %). ``content_provenance``: the per-SD units the audited
    contrast table states its arm coefficients in. They are frozen by the fitter
    over the six leaf arms and are NOT recomputable here: the same covariates
    measured over all nine candidates have a 2.1x larger SD (the trunks are
    structural zeros), which would shrink every per-SD arm effect in this figure
    by that factor relative to the audited table. ``content_arm`` and
    ``content_patch``: the content contrast fits. Under
    ``paths.replay_content_dir()``, read when called.
    """
    rate, contrast = model_dir("per_bin_rate"), model_dir("swr_contrast")
    return {
        "rate": rate / "swr_contrast_rate.pkl",
        "rate_meta": rate / "swr_contrast_rate_meta.json",
        "content_provenance": contrast / "swr_contrast_content_provenance.json",
        "content_arm": contrast / "swr_contrast_content_arm.pkl",
        "content_patch": contrast / "swr_contrast_content_patch.pkl",
    }


def rate_input_paths() -> dict[str, Path]:
    """The tables fit_swr_contrast_rate.py fit the rate artifact on.

    Checked against its provenance sidecar here, before any table or fit is
    read, so a figure cannot be drawn from coefficients that predate the tables
    on disk.
    """
    rc = replay_content_dir()
    return {
        "stop_table": table_path("stop_table", rc),
        "qvals_table": table_path("qvals_table", rc),
        "event_onset_table": table_path("event_onset_table", rc),
        "event_table": table_path("event_table", rc),
    }


def content_input_paths() -> dict[str, Path]:
    """The tables fit_swr_contrast_content.py fit the two content artifacts on,
    gated the same way as the rate half."""
    rc = replay_content_dir()
    return {
        "event_arm_table": table_path("event_arm_table", rc),
        "event_patch_table": table_path("event_patch_table", rc),
        "stop_table": table_path("stop_table", rc),
        "qvals_table": table_path("qvals_table", rc),
    }


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (also the provenance gate's opt-out)."""
    return fit_figure_parser(__doc__)


def _per_sd_units(artifacts: dict[str, Path]):
    """Covariate -> SD the figure states its per-SD effects in.

    Both sets come from the fits' own records (`artifacts`, as
    :func:`fit_artifact_paths` names them): the rate covariates from the rate
    meta's ``fit_time_sd_reference``, the per-arm covariates from the content
    provenance's ``per_sd_scale``. The two key sets are disjoint.
    """
    return {
        **load_fit_time_sd_reference(artifacts["rate_meta"]),
        **load_per_sd_scale(artifacts["content_provenance"], "arm_cohort"),
    }


ANIM = list(analysed_animals())
CMAP = dict(zip(ANIM, plt.cm.tab10(np.linspace(0, 0.5, 5)), strict=True))
# The dots' vertical jitter only. Each row's bootstrap has its own generator,
# seeded BOOTSTRAP_SEED, so a row's CI does not depend on the rows drawn before it.
JITTER_RNG = np.random.default_rng(0)
BOOTSTRAP_SEED = 0
N_BOOTSTRAP = 4000


def sessions(df, mod, pop, term, sd):
    """Per-animal standardized session coefficients the audited table aggregates.

    Returns an empty dict when the audited aggregation refuses the term (too few
    kept sessions, or too large an unidentified share), so the row is drawn
    empty instead of showing an effect the audited table does not report.
    """
    g = df[(df.model == mod) & (df.population == pop)]
    # The rate ladder fits its post-outcome rungs on their own cohort and refits
    # the ladder null beside them, so a rung name alone can match two rows per
    # session. Keep the cohort the rung's coefficients are reported from.
    if "cohort" in g.columns:
        home = POST_OUTCOME_COHORT if mod in POST_OUTCOME_RATE_RUNGS else PRIMARY_COHORT
        g = g[g.cohort == home]
    c = f"coef_{term}"
    if c not in g.columns:
        raise KeyError(
            f"coef column {c!r} absent from fitted pickle for {mod}/{pop} — the row "
            f"spec references a term the fit did not produce (likely a rename). "
            f"Row would render blank."
        )
    # The audited aggregation's own session filter and emission gates, so the
    # figure shows exactly the sessions -- and only the terms -- the audited
    # table reports.
    gate = gate_term_sessions(g, term)
    if gate.refusal is not None:
        print(
            f"  {mod}/{pop}/{term}: not drawn, refused by the audited aggregation "
            f"({gate.refusal}: {gate.n_kept} kept, {gate.n_dropped} separated or "
            f"runaway, {gate.n_unidentified} not informed, of {gate.n_fitted})"
        )
        return {}
    g = g[gate.kept]
    f, _ = std_factor(term, sd)
    return {
        a: g.loc[g.animal == a, c].to_numpy() * f for a in ANIM if (g.animal == a).any()
    }


def forest(ax, df, rows, title, sd):
    y = np.arange(len(rows))[::-1]
    allv = []
    labels = []
    for yi, (label, mod, pop, term) in zip(y, rows, strict=True):
        _, unit = std_factor(term, sd)
        labels.append(label + {"per_category": " †", "per_unit": " ‡"}.get(unit, ""))
        s = sessions(df, mod, pop, term, sd)
        if not s:
            labels[-1] += " (not reported)"
            continue
        drawn = [a for a in ANIM if len(s.get(a, ()))]
        for a in drawn:
            allv.append(s[a])
            ax.scatter(
                s[a],
                yi + JITTER_RNG.uniform(-0.28, 0.28, len(s[a])),
                s=6,
                color=CMAP[a],
                alpha=0.20,
                edgecolors="none",
                zorder=2,
            )
        summary = animal_median_bootstrap(
            pd.Series(np.concatenate([s[a] for a in drawn])),
            pd.Series(np.repeat(drawn, [len(s[a]) for a in drawn])),
            rng=np.random.default_rng(BOOTSTRAP_SEED),
            n_draws=N_BOOTSTRAP,
        )
        for a in drawn:
            m = summary["animal_medians"][a]
            ax.plot(m, yi, "o", ms=7, color=CMAP[a], mec="k", mew=0.6, zorder=5)
        gm = summary["animal_median"]
        lo, hi = summary["low"], summary["high"]
        ax.plot([lo, hi], [yi, yi], color="k", lw=1.4, zorder=4)
        ax.plot([gm, gm], [yi - 0.34, yi + 0.34], color="k", lw=2.6, zorder=6)
    ax.axvline(0, color="0.4", lw=1, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    if allv:  # robust x-limits so per-animal medians stay legible
        cat = np.concatenate(allv)
        lo, hi = np.percentile(cat, [1, 99])
        pad = 0.1 * (hi - lo + 1e-9)
        ax.set_xlim(lo - pad, hi + pad)
    # two lines: on one line the label ran past the right edge of the right panel
    ax.set_xlabel(
        "coefficient  (continuous: per 1 SD;  † binary: per category;\n"
        "‡ reward fraction: per unit)"
    )
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(axis="x", alpha=0.25)


handles = [plt.Line2D([], [], marker="o", ls="", color=CMAP[a], label=a) for a in ANIM]

# ---- RATE ----
SWR, NON = "non_local_swr", "non_local_no_swr"
rate_rows = [
    ("immediate reward history · SWR", "G0_base", SWR, "reward_history_rcos_0"),
    ("immediate reward history · nonSWR", "G0_base", NON, "reward_history_rcos_0"),
    ("RPE (beta) · SWR", "G5_rpe", SWR, "rpe_postoutcome"),
    ("RPE (beta) · nonSWR", "G5_rpe", NON, "rpe_postoutcome"),
    (
        "rel. value (beta) · SWR",
        "G4_value",
        SWR,
        "relative_value_switch_minus_stay_preoutcome",
    ),
    (
        "rel. value (beta) · nonSWR",
        "G4_value",
        NON,
        "relative_value_switch_minus_stay_preoutcome",
    ),
    (
        "belief-entropy (beta) · SWR",
        "G6_uncert",
        SWR,
        "reward_belief_differential_entropy_leaf_mean_preoutcome",
    ),
    (
        "belief-entropy (beta) · nonSWR",
        "G6_uncert",
        NON,
        "reward_belief_differential_entropy_leaf_mean_preoutcome",
    ),
    (
        "value-update (beta) · SWR",
        "G7_update",
        SWR,
        "reward_belief_mean_update_l1_all_ports_postoutcome",
    ),
    (
        "value-update (beta) · nonSWR",
        "G7_update",
        NON,
        "reward_belief_mean_update_l1_all_ports_postoutcome",
    ),
    (
        "switch-rate · SWR",
        "G3_switchrate",
        SWR,
        "patch_switch_fraction_last5_completed_trips",
    ),
    (
        "switch-rate · nonSWR",
        "G3_switchrate",
        NON,
        "patch_switch_fraction_last5_completed_trips",
    ),
]


def rate_figure(R, sd):
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    forest(
        ax,
        R,
        rate_rows,
        "Rate ladder — is_swr contrast\n(standardized; gated sessions + animal 95% CI)",
        sd,
    )
    ax.legend(
        handles=handles, fontsize=8, loc="lower right", title="animal", framealpha=0.95
    )
    fig.tight_layout()
    return fig


# ---- CONTENT ----
mod_rows = [
    ("current-arm (C1)", "C1_current", "all", "is_current_arm"),
    (
        "reward_rate_arm (C3)",
        "C3_armreward",
        "all",
        "reward_fraction_candidate_arm_last10_visits",
    ),
    (
        "arm belief value (C4)",
        "C4_armvalue",
        "all",
        "reward_belief_mean_candidate_arm_preoutcome",
    ),
    ("RPE×current (C6)", "C6_rpe", "all", "rpe_postoutcome_z_x_is_current_arm"),
    (
        "rel.value×current (C7)",
        "C7_value",
        "all",
        "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
    ),
    (
        "entropy×current (C8)",
        "C8_uncert",
        "all",
        "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm",
    ),
    (
        "update×current (C9)",
        "C9_update",
        "all",
        "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
    ),
    (
        "periswitch×current (C10)",
        "C10_periswitch",
        "all",
        "periswitch_index_trials_z_x_is_current_arm",
    ),
    (
        "block×current (C11)",
        "C11_block",
        "all",
        "trials_since_block_change_z_x_is_current_arm",
    ),
    (
        "switch fraction×current (C12)",
        "C12_switchrate",
        "all",
        "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm",
    ),
]
con_rows = [
    ("swr × current (CS1)", "CS1_current", "all", "swr_x_is_current_arm"),
    ("swr × prev (CS2)", "CS2_prev", "all", "swr_x_is_prev_arm"),
    (
        "swr × last-rewarded (CS3)",
        "CS3_lastrew",
        "all",
        "swr_x_is_last_rewarded_noncurrent_arm",
    ),
    (
        "swr × reward fraction (CS4)",
        "CS4_armreward",
        "all",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
    ),
    (
        "swr × belief value (CS5)",
        "CS5_armvalue",
        "all",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    ),
]


def content_arm_figure(C, sd):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    forest(
        axes[0],
        C,
        mod_rows,
        "Content (arm) — dimension effects (all events, standardized)",
        sd,
    )
    forest(
        axes[1],
        C,
        con_rows,
        "Content (arm) — SWR contrasts (swr_x_*, standardized)",
        sd,
    )
    axes[1].legend(
        handles=handles, fontsize=8, loc="lower left", title="animal", framealpha=0.95
    )
    fig.tight_layout()
    return fig


# ---- CONTENT: PATCH ladder ----
patch_mod_rows = [
    ("current-patch (CP0)", "CP0_base", "all", "is_current_patch"),
    ("last-rew-patch (CP1)", "CP1_lastrew", "all", "is_last_rewarded_noncurrent_patch"),
    (
        "current patch×peri-switch (CP3)",
        "CP3_newgrowth",
        "all",
        "periswitch_index_trials_z_x_is_current_patch",
    ),
    (
        "current patch×switch fraction (CP4)",
        "CP4_switchrate",
        "all",
        "patch_switch_fraction_last5_completed_trips_z_x_is_current_patch",
    ),
]
patch_con_rows = [
    ("swr × current-patch (CPS1)", "CPS1_current", "all", "swr_x_is_current_patch"),
    (
        "swr × lastrew-patch (CPS2)",
        "CPS2_lastrew",
        "all",
        "swr_x_is_last_rewarded_noncurrent_patch",
    ),
]


def content_patch_figure(P, sd):
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
    forest(
        axes[0],
        P,
        patch_mod_rows,
        "Content (patch) — dimension effects (all events, standardized)",
        sd,
    )
    forest(
        axes[1],
        P,
        patch_con_rows,
        "Content (patch) — SWR contrasts (standardized)",
        sd,
    )
    axes[1].legend(
        handles=handles, fontsize=8, loc="lower left", title="animal", framealpha=0.95
    )
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    allow_stale = build_parser().parse_args(argv).allow_stale
    artifacts = fit_artifact_paths()
    rate, rate_meta = artifacts["rate"], artifacts["rate_meta"]
    # The provenance gate fires ahead of the first table read.
    require_fresh_inputs(rate, rate_input_paths(), allow_stale=allow_stale)
    for content_fit in (artifacts["content_arm"], artifacts["content_patch"]):
        require_fresh_inputs(
            content_fit, content_input_paths(), allow_stale=allow_stale
        )
    # The fitter writes the meta after the per-session fits, so a meta older
    # than them states the units of an earlier fit (an aggregation that failed).
    if not allow_stale and (rate_meta.stat().st_mtime < rate.stat().st_mtime):
        raise RuntimeError(
            f"{rate_meta.name} predates {rate.name}, so "
            "its per-SD units may belong to an earlier fit; re-run "
            "scripts/behavior/rate/fit_swr_contrast_rate.py --aggregate-only"
        )
    sd = _per_sd_units(artifacts)
    out = figure_dir("swr_contrast")
    out.mkdir(parents=True, exist_ok=True)
    rate_figure(pd.read_pickle(rate), sd).savefig(
        out / "rate_swr_contrast_coefs.png", dpi=150
    )
    content_arm_figure(pd.read_pickle(artifacts["content_arm"]), sd).savefig(
        out / "content_swr_contrast_coefs.png", dpi=150
    )
    content_patch_figure(pd.read_pickle(artifacts["content_patch"]), sd).savefig(
        out / "content_patch_swr_contrast_coefs.png", dpi=150
    )
    print("wrote:")
    for p in sorted(out.glob("*.png")):
        print("  ", p)


if __name__ == "__main__":
    main()
