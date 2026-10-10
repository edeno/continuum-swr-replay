"""Coefficient figures for the is_swr rate + content contrast ladders.

For each effect: every session's coefficient is a faint jittered dot (colored by
animal); the 5 larger dots are per-animal medians; the black bar is the
across-animal median; the horizontal whisker is its 95% CI (bootstrap over the 5
per-animal medians -- the animal is the inferential unit, n=5). Continuous-predictor
coefficients are shown STANDARDIZED (per 1 SD of the predictor: raw_coef * SD, which
is exact for these Poisson GLMs) so effect sizes are comparable across predictors,
value constructs, and the two ladders. Binary-predictor terms (marked with a dagger)
are per-category and are not standardized. The reward-history basis features are
weighted reward fractions in [0, 1] (marked with a double dagger); their coefficients
are per unit of the fraction -- the effect of a fully rewarded recent history versus a
fully unrewarded one -- and are likewise not standardized.

Per-arm covariate SDs are read from the content fit's provenance record, so this
figure states its arm effects in exactly the per-leaf-arm SD unit the audited
table uses; the rate ladder's per-stop covariate SDs are computed here from the
source tables. Content state-modulation rungs (rpe/relval/entropy/
update/tss/block/switchrate x current) are already z-scored at fit time, so their
standardization factor is 1.0.

    uv run python scripts/figures/plot_swr_contrast_coefs.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.swr_source import require_swr_source
from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    PER_UNIT_TERMS,
    SEPARATION_COEF_ABS,
    load_per_sd_scale,
    separated_for_term,
)
from continuum_swr_replay.analysis.replay.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.analysis.replay.fit_provenance import require_fresh_inputs
from continuum_swr_replay.analysis.replay.model_specs import (
    POST_OUTCOME_COHORT,
    POST_OUTCOME_RATE_RUNGS,
    PRIMARY_COHORT,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import read_table

RC = PROCESSED_DATA_DIR / "replay_content"
OUT = Path("paper/figures/swr_contrast")
OUT.mkdir(parents=True, exist_ok=True)

SWR_CONTRAST_RATE = RC / "models_per_bin_rate" / "swr_contrast_rate.pkl"
# The tables fit_swr_contrast_rate.py fit that artifact on; checked against its
# provenance sidecar here, before any table or fit is read, so a figure cannot be
# drawn from coefficients that predate the tables on disk.
SWR_CONTRAST_RATE_INPUTS = {
    "stop_table": RC / "stop_table.pkl",
    "qvals_table": RC / "models_per_bin_rate" / "qvals_table.pkl",
    "event_onset_table": RC / "event_onset_table.pkl",
    "event_table": RC / "event_table.pkl",
}

# The per-SD units the audited contrast table states its arm coefficients in.
# They are frozen by the fitter over the six leaf arms and are NOT recomputable
# here: the same covariates measured over all nine candidates have a 2.1x larger
# SD (the trunks are structural zeros), which would shrink every per-SD arm
# effect in this figure by that factor relative to the audited table.
SWR_CONTRAST_CONTENT_PROVENANCE = RC / "models" / "swr_contrast_content_provenance.json"

SWR_CONTRAST_CONTENT_ARM = RC / "models" / "swr_contrast_content_arm.pkl"
SWR_CONTRAST_CONTENT_PATCH = RC / "models" / "swr_contrast_content_patch.pkl"
# The tables fit_swr_contrast_content.py fit those two artifacts on, gated the
# same way as the rate half.
SWR_CONTRAST_CONTENT_INPUTS = {
    "event_arm_table": RC / "event_arm_table.pkl",
    "event_patch_table": RC / "event_patch_table.pkl",
    "stop_table": RC / "stop_table.pkl",
    "qvals_table": RC / "models_per_bin_rate" / "qvals_table.pkl",
}


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (also the provenance gate's opt-out)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help=(
            "Plot the fit even if its input tables have changed since it was "
            "fit (or carry no provenance sidecar at all)."
        ),
    )
    return parser


# This script runs top to bottom at import; the gate has to fire here, ahead of
# the first table read, rather than inside a main() it does not have.
_ALLOW_STALE = build_parser().parse_args().allow_stale
require_fresh_inputs(
    SWR_CONTRAST_RATE, SWR_CONTRAST_RATE_INPUTS, allow_stale=_ALLOW_STALE
)
for _content_fit in (SWR_CONTRAST_CONTENT_ARM, SWR_CONTRAST_CONTENT_PATCH):
    require_fresh_inputs(
        _content_fit, SWR_CONTRAST_CONTENT_INPUTS, allow_stale=_ALLOW_STALE
    )


def _global_sd(covariates):
    """Global (pooled) SD of each continuous covariate from its source table."""
    from continuum_swr_replay.analysis.replay.covariates import (
        add_uncertainty_covariates,
    )

    arm = read_table(RC / "event_arm_table.pkl")
    qv = read_table(RC / "models_per_bin_rate" / "qvals_table.pkl")
    stop = read_table(RC / "stop_table.pkl")
    require_covariate_schema(arm, table="event_arm")
    require_covariate_schema(qv, table="qvals")
    require_covariate_schema(stop, table="stop")
    require_swr_source(arm, what=str(RC / "event_arm_table.pkl"))
    require_swr_source(stop, what=str(RC / "stop_table.pkl"))
    stop = add_uncertainty_covariates(stop)
    sd = {}
    for c in covariates:
        for src in (arm, qv, stop):
            if c in src.columns:
                sd[c] = float(np.nanstd(src[c].to_numpy(dtype=float)))
                break
    return sd


# Rate-ladder covariates: per-stop quantities with no candidate-arm support
# question, so their pooled SD is computed here from the source tables. The
# per-arm covariates come from the fit's own frozen units instead.
SD = {
    **_global_sd(
        [
            "relative_value_switch_minus_stay_preoutcome",
            "rpe_postoutcome",
            "reward_belief_differential_entropy_leaf_mean_preoutcome",
            "reward_belief_mean_update_l1_all_ports_postoutcome",
            "periswitch_index_trials",
            "trials_since_block_change",
            "patch_switch_fraction_last5_completed_trips",
        ]
    ),
    **load_per_sd_scale(SWR_CONTRAST_CONTENT_PROVENANCE, "arm_cohort"),
}
ANIM = ["chimi", "j16", "peanut", "senor", "wilbur"]
CMAP = dict(zip(ANIM, plt.cm.tab10(np.linspace(0, 0.5, 5)), strict=True))
RNG = np.random.default_rng(0)


def std_factor(term):
    """Scaling to standardized units; ``(factor, unit)``.

    Mirrors the three-way labelling of
    ``analysis.replay.contrast_aggregation.std_factor`` -- ``"per_category"`` for
    binary predictors, ``"per_unit"`` for the reward-history basis features
    (weighted reward fractions, so a per-SD rescaling would misstate their unit),
    ``"per_SD"`` otherwise -- but keeps this module's fail-loud behaviour: an
    unresolved continuous term raises rather than rendering at raw scale under a
    "per 1 SD" axis.
    """
    BIN = {
        "is_current_arm",
        "swr_x_is_current_arm",
        "swr_x_is_prev_arm",
        "swr_x_is_last_rewarded_noncurrent_arm",
        "is_current_patch",
        "is_last_rewarded_noncurrent_patch",
        "swr_x_is_current_patch",
        "swr_x_is_last_rewarded_noncurrent_patch",
    }
    if term in BIN:
        return 1.0, "per_category"
    if term in PER_UNIT_TERMS:
        return 1.0, "per_unit"
    # per-arm / rate continuous covariates scaled by their global SD
    for cov, s in SD.items():
        if term == cov or term == f"swr_x_{cov}":
            return s, "per_SD"
    # Interaction terms (content/patch `*_x_*`) are built from z-scored scalars at
    # fit time, so factor 1.0 is correct. Continuous MAIN-effect terms, by contrast,
    # must be in SD — so a renamed main effect (not an interaction) raises here rather
    # than silently plotting at raw scale under a "per 1 SD" axis.
    if "_x_" in term:
        return 1.0, "per_SD"
    raise KeyError(
        f"plot term {term!r} resolves to no SD entry and is not a binary, a "
        f"reward-history basis feature, or a z-scored-at-fit interaction (`*_x_*`) — "
        f"a main-effect rename would silently mis-scale it. Add it to the SD "
        f"(_global_sd) list."
    )


def sessions(df, mod, pop, term):
    g = df[(df.model == mod) & (df.population == pop)]
    # The rate ladder fits its post-outcome rungs on their own cohort and refits
    # the ladder null beside them, so a rung name alone can match two rows per
    # session. Keep the cohort the rung's coefficients are reported from.
    if "cohort" in g.columns:
        home = (
            POST_OUTCOME_COHORT if mod in POST_OUTCOME_RATE_RUNGS else PRIMARY_COHORT
        )
        g = g[g.cohort == home]
    c = f"coef_{term}"
    if c not in g.columns:
        raise KeyError(
            f"coef column {c!r} absent from fitted pickle for {mod}/{pop} — the row "
            f"spec references a term the fit did not produce (likely a rename). "
            f"Row would render blank."
        )
    g = g.dropna(subset=[c])
    # Identifiability filter, matching contrast_aggregation.aggregate_contrast so
    # the figure shows exactly the sessions the audited table aggregates. A term
    # the fitter named as separated is dropped; so is a coefficient whose effect
    # over one SD of its own predictor runs away. Judging the raw coefficient
    # instead dropped identified sessions whose predictor has a small SD and kept
    # unidentified ones whose predictor has a large one.
    sdcol = f"sd_{term}"
    runaway = (
        g[c].abs() * g[sdcol] >= SEPARATION_COEF_ABS
        if sdcol in g.columns
        else g[c].abs() >= SEPARATION_COEF_ABS
    )
    g = g[~runaway & ~separated_for_term(g, term)]
    f, _ = std_factor(term)
    return {a: g.loc[g.animal == a, c].to_numpy() * f for a in ANIM if (g.animal == a).any()}


def animal_ci(medians, n_boot=4000):
    v = np.asarray([m for m in medians if np.isfinite(m)])
    if len(v) < 2:
        return np.nan, np.nan
    boot = [np.median(RNG.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return np.percentile(boot, [2.5, 97.5])


def forest(ax, df, rows, title):
    y = np.arange(len(rows))[::-1]
    allv = []
    labels = []
    for yi, (label, mod, pop, term) in zip(y, rows, strict=True):
        _, unit = std_factor(term)
        labels.append(label + {"per_category": " †", "per_unit": " ‡"}.get(unit, ""))
        s = sessions(df, mod, pop, term)
        amed = []
        for a in ANIM:
            v = s.get(a, np.empty(0))
            if not len(v):
                amed.append(np.nan)
                continue
            allv.append(v)
            ax.scatter(v, yi + RNG.uniform(-0.28, 0.28, len(v)), s=6, color=CMAP[a],
                       alpha=0.20, edgecolors="none", zorder=2)
            m = np.median(v)
            amed.append(m)
            ax.plot(m, yi, "o", ms=7, color=CMAP[a], mec="k", mew=0.6, zorder=5)
        gm = np.nanmedian(amed)
        lo, hi = animal_ci(amed)
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
    ax.set_xlabel(
        "coefficient  (continuous: per 1 SD;  † binary: per category;  "
        "‡ reward fraction: per unit)"
    )
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(axis="x", alpha=0.25)


handles = [plt.Line2D([], [], marker="o", ls="", color=CMAP[a], label=a) for a in ANIM]

# ---- RATE ----
R = pd.read_pickle(SWR_CONTRAST_RATE)
SWR, NON = "non_local_swr", "non_local_no_swr"
rate_rows = [
    ("immediate reward history · SWR", "G0_base", SWR, "reward_history_rcos_0"),
    ("immediate reward history · nonSWR", "G0_base", NON, "reward_history_rcos_0"),
    ("RPE (beta) · SWR", "G5_rpe", SWR, "rpe_postoutcome"),
    ("RPE (beta) · nonSWR", "G5_rpe", NON, "rpe_postoutcome"),
    ("rel. value (beta) · SWR", "G4_value", SWR, "relative_value_switch_minus_stay_preoutcome"),
    ("rel. value (beta) · nonSWR", "G4_value", NON, "relative_value_switch_minus_stay_preoutcome"),
    ("belief-entropy (beta) · SWR", "G6_uncert", SWR, "reward_belief_differential_entropy_leaf_mean_preoutcome"),
    ("belief-entropy (beta) · nonSWR", "G6_uncert", NON, "reward_belief_differential_entropy_leaf_mean_preoutcome"),
    ("value-update (beta) · SWR", "G7_update", SWR, "reward_belief_mean_update_l1_all_ports_postoutcome"),
    ("value-update (beta) · nonSWR", "G7_update", NON, "reward_belief_mean_update_l1_all_ports_postoutcome"),
    ("switch-rate · SWR", "G3_switchrate", SWR, "patch_switch_fraction_last5_completed_trips"),
    ("switch-rate · nonSWR", "G3_switchrate", NON, "patch_switch_fraction_last5_completed_trips"),
]
fig, ax = plt.subplots(figsize=(8.5, 6.6))
forest(ax, R, rate_rows, "Rate ladder — is_swr contrast (standardized; all sessions + animal 95% CI)")
ax.legend(handles=handles, fontsize=8, loc="lower right", title="animal", framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT / "rate_swr_contrast_coefs.png", dpi=150)

# ---- CONTENT ----
C = pd.read_pickle(SWR_CONTRAST_CONTENT_ARM)
mod_rows = [
    ("current-arm (C1)", "C1_current", "all", "is_current_arm"),
    ("reward_rate_arm (C3)", "C3_armreward", "all", "reward_fraction_candidate_arm_last10_visits"),
    ("value q_arm (C4)", "C4_armvalue", "all", "reward_belief_mean_candidate_arm_preoutcome"),
    ("RPE×current (C6)", "C6_rpe", "all", "rpe_postoutcome_z_x_is_current_arm"),
    ("rel.value×current (C7)", "C7_value", "all", "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm"),
    ("entropy×current (C8)", "C8_uncert", "all", "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm"),
    ("update×current (C9)", "C9_update", "all", "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm"),
    ("periswitch×current (C10)", "C10_periswitch", "all", "periswitch_index_trials_z_x_is_current_arm"),
    ("block×current (C11)", "C11_block", "all", "trials_since_block_change_z_x_is_current_arm"),
    ("switch fraction×current (C12)", "C12_switchrate", "all", "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm"),
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
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
forest(axes[0], C, mod_rows, "Content (arm) — dimension effects (all events, standardized)")
forest(axes[1], C, con_rows, "Content (arm) — SWR contrasts (swr_x_*, standardized)")
axes[1].legend(handles=handles, fontsize=8, loc="lower left", title="animal", framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT / "content_swr_contrast_coefs.png", dpi=150)

# ---- CONTENT: PATCH ladder ----
P = pd.read_pickle(SWR_CONTRAST_CONTENT_PATCH)
patch_mod_rows = [
    ("current-patch (CP0)", "CP0_base", "all", "is_current_patch"),
    ("last-rew-patch (CP1)", "CP1_lastrew", "all", "is_last_rewarded_noncurrent_patch"),
    ("current patch×peri-switch (CP3)", "CP3_newgrowth", "all", "periswitch_index_trials_z_x_is_current_patch"),
    ("current patch×switch fraction (CP4)", "CP4_switchrate", "all", "patch_switch_fraction_last5_completed_trips_z_x_is_current_patch"),
]
patch_con_rows = [
    ("swr × current-patch (CPS1)", "CPS1_current", "all", "swr_x_is_current_patch"),
    ("swr × lastrew-patch (CPS2)", "CPS2_lastrew", "all", "swr_x_is_last_rewarded_noncurrent_patch"),
]
fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
forest(axes[0], P, patch_mod_rows, "Content (patch) — dimension effects (all events, standardized)")
forest(axes[1], P, patch_con_rows, "Content (patch) — SWR contrasts (standardized)")
axes[1].legend(handles=handles, fontsize=8, loc="lower left", title="animal", framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT / "content_patch_swr_contrast_coefs.png", dpi=150)

print("wrote:")
for p in sorted(OUT.glob("*.png")):
    print("  ", p)
