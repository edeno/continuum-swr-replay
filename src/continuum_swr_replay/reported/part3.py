"""Part 3's reported values: how the events relate to the task, SWR against non-SWR.

Every value is read from an output a fit already wrote; nothing is refitted
here. The inputs (:class:`Part3Inputs`), with their runs in
``scripts/qa/run_refits.py``:

- the per-bin rate ladder, Family B (``models_per_bin_rate/all_models.pkl``,
  ``rate.per_bin.at_well``; ``all_models_allstops.pkl``,
  ``rate.per_bin.all_stops``, the sensitivity cohort);
- the per-stop rate ladder, Family A (``models_per_stop_rate/all_models.pkl``,
  ``rate.per_stop.at_well``);
- the content ``is_swr``-contrast ladders' audited aggregates
  (``models_swr_contrast/swr_contrast_content_audited*.pkl``,
  ``content.swr_contrast.*`` and ``content.breadth_contrast.*``);
- the three-estimator comparison (``content_estimator_comparison/``,
  ``content.estimator_comparison``);
- the leaf-only arm ladder (``models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu/``,
  ``content.arm_ladder.leaf_arms``);
- the event-source suite's choice block
  (``event_source_comparison/non_local_HPC_comparison_fits.pkl``,
  ``behavior.event_source_suite.non_local``);
- the stay/leave fits (``stage6_is_stay_*.pkl``, ``behavior.stay_leave.*``);
- the SWR-power model (``models_per_event_swr_power/swr_power_model.pkl``,
  ``coupling.ripple_power``);
- the canonical event and stop tables, for event lengths and cohort sizes.

Each aggregate is computed with the rule the producing script or its audit
states, named in the value's description. The content-ladder, leaf-ladder and
choice coefficients this module aggregates itself are read through the shipped
per-term identifiability gate
(:func:`~continuum_swr_replay.analysis.replay.contrast_aggregation.gate_term_sessions`).
The per-bin rate kernels are read as ``summarize_per_bin_rate_models.py`` and
``compare_fits.py`` read them, over every fitted session with no per-term gate;
the time courses drop fits flagged separated, as ``compare_fits`` does. The
self-history kernel is many sessions' separated term, so its values carry a
pending note and a gated reading beside them. "Animals positive" counts the
animals whose own median is above zero.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    gate_term_sessions,
    pct_significant_sessions,
    session_se,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    check_fit_provenance,
    fit_provenance_path,
)
from continuum_swr_replay.analysis.replay.learning import cross_session_role_summary
from continuum_swr_replay.analysis.replay.model_specs import (
    ARRIVAL_CLOCK_COLUMN,
    ARRIVAL_SPLINE_KNOTS,
    BIN_S,
    MODEL_PER_BIN_RATE,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
    SPLINE_KNOTS,
    TMAX,
    TMAX_ARRIVAL,
    longest_history_window_s,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (
    aggregate_to_timebins,
    reconstruct_reward_kernels,
    reward_kernel_rate_ratios,
    self_history_rate_ratios,
)
from continuum_swr_replay.paths import (
    ROOT_DIR,
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.values import Summary, Value, file_record
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

SC = list(SESSION_COLUMNS)

NO_SIDECAR_PENDING = (
    "{name} has no provenance sidecar, so it cannot be checked against the "
    "current tables; give its runner one before citing"
)
CHOICE_RULE_PENDING = (
    NO_SIDECAR_PENDING.format(name="the event-source suite's choice fit")
    + "; the summary rule (session median, signed-rank test) is the project's "
    "per-session convention, which the fitting script does not state itself"
)

#: Rate populations: the word their keys use, the fits' ``outcome``, the label.
RATE_POPULATIONS = {
    "swr": ("swr", "detected SWR"),
    "non_local": ("non_local", "non-local (all)"),
    "non_local_swr": ("non_local_swr", "non-local with SWR"),
    "non_local_no_swr": ("non_local_no_swr", "non-local without SWR"),
}
FULL_RATE_RUNG = "M7r_time_histcos_rew"
HISTORY_PENDING = (
    "read over every session, as the producing summarizer "
    "(summarize_per_bin_rate_models.py) reads it, without the per-term separation "
    "gate; many sessions flag a hist_rcos term separated, and leaving them out moves "
    "the kernel (rate_hist_separated_fraction_*, rate_burst_peak_gated_*)"
)
IMMEDIATE_COEF = f"coef_{REWARD_HISTORY_COLUMNS[0]}"
STOP_TIME_RUNG = "M5_time"
ARRIVAL_TIME_RUNG = "M5a_time_arrival"

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
"""Every content-contrast aggregate Part 3 reports."""

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

#: Stay/leave content strata, by the file suffix their bundles carry.
STAY_STRATA = {"all": "", "swr": "_swr", "non_swr": "_non_swr", "mua": "_mua"}

#: The SWR-power model's covariates Part 3 reports: key word -> column.
POWER_COVARIATES = {
    "reward": "known_reward_outcome_lag_1",
    "rpe": "rpe_postoutcome",
    "exploration": "patch_switch_fraction_last5_completed_trips",
}


@dataclass(frozen=True)
class Part3Inputs:
    """The files Part 3 reads."""

    per_bin: Path
    per_bin_all_stops: Path
    per_bin_in_transit: Path
    rate_contrast: Path
    per_bin_audit: Path
    per_stop_audit: Path
    per_stop: Path
    content_contrast_dir: Path
    estimator_summary: Path
    estimator_bootstrap: Path
    leaf_dir: Path
    choice_fits: Path
    stay_leave_dir: Path
    swr_power: Path
    event_table: Path
    stop_table: Path
    onset_table: Path
    trunk_per_event: Path


def default_inputs() -> Part3Inputs:
    rc = replay_content_dir()
    estimators = rc / "content_estimator_comparison"
    return Part3Inputs(
        per_bin=model_dir("per_bin_rate") / "all_models.pkl",
        per_bin_all_stops=model_dir("per_bin_rate") / "all_models_allstops.pkl",
        per_bin_in_transit=model_dir("per_bin_rate") / "all_models_intransit.pkl",
        rate_contrast=model_dir("per_bin_rate") / "swr_contrast_rate.pkl",
        per_bin_audit=model_dir("per_bin_rate") / "per_bin_rate_cohort_audit.csv",
        per_stop_audit=model_dir("per_stop_rate") / "all_models_cohort_audit.csv",
        per_stop=model_dir("per_stop_rate") / "all_models.pkl",
        content_contrast_dir=rc / "models_swr_contrast",
        estimator_summary=estimators / "content_estimator_comparison_summary.csv",
        estimator_bootstrap=estimators / "content_estimator_bootstrap.csv",
        leaf_dir=rc / "models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu",
        choice_fits=rc
        / "event_source_comparison"
        / "non_local_HPC_comparison_fits.pkl",
        stay_leave_dir=rc,
        swr_power=model_dir("per_event_swr_power") / "swr_power_model.pkl",
        event_table=table_path("event_table", rc),
        stop_table=table_path("stop_table", rc),
        onset_table=table_path("event_onset_table", rc),
        trunk_per_event=rc / "_trunk_zscore_per_event.pkl",
    )


def _value_factory(source: str, claims: tuple[str, ...], used_in: tuple[str, ...]):
    """A :class:`Value` builder with the shared source, claims and use filled in."""

    def make(
        key: str,
        value: float | int,
        kind: str,
        style: str,
        description: str,
        unit: str = "",
        *,
        pending: str | None = None,
        claims_: tuple[str, ...] | None = None,
        used_in_: tuple[str, ...] | None = None,
    ) -> Value:
        if kind == "count" and float(value).is_integer():
            value = int(value)
        elif kind != "count":
            value = float(value)
        return Value(
            key=key,
            value=value,
            kind=kind,
            style=style,
            unit=unit,
            description=description,
            claims=claims_ or claims,
            used_in=used_in_ or used_in,
            source=source,
            pending=pending,
        )

    return make


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# --- 3a: rate, per time bin (Family B) ----------------------------------------


def _rung(fits: pd.DataFrame, model: str, outcome: str) -> pd.DataFrame:
    return fits[(fits["model"] == model) & (fits["outcome"] == outcome)]


def _lag_one_kernel(frame: pd.DataFrame) -> pd.Series:
    """Per-session reconstructed reward kernel at lag 1 (log rate), by animal."""
    columns = [f"coef_{c}" for c in REWARD_HISTORY_COLUMNS]
    fitted = frame.dropna(subset=columns)
    kernels = reconstruct_reward_kernels(
        fitted[columns].to_numpy(dtype=float),
        REWARD_HISTORY_N_LAGS,
        REWARD_HISTORY_N_BASIS,
    )
    return pd.Series(kernels[:, 0], index=fitted["animal"].to_numpy())


def _paired_immediate(fits: pd.DataFrame) -> dict[str, float]:
    """The within-session comparison of the immediate basis coefficient,
    SWR-coupled against non-SWR non-local (``compare_fits.reward_quantities``)."""
    full = fits[fits["model"] == FULL_RATE_RUNG]
    a = full.loc[full["outcome"] == "non_local_swr", SC + [IMMEDIATE_COEF]]
    b = full.loc[full["outcome"] == "non_local_no_swr", SC + [IMMEDIATE_COEF]]
    merged = a.merge(b, on=SC, suffixes=("_swr", "_no_swr")).dropna()
    difference = merged[f"{IMMEDIATE_COEF}_swr"] - merged[f"{IMMEDIATE_COEF}_no_swr"]
    per_animal = merged.assign(d=difference).groupby("animal")["d"].median()
    _require(
        len(per_animal) == 5, f"paired test spans {len(per_animal)} animals, not 5"
    )
    return {
        "n": len(merged),
        "positive": float((difference > 0).mean()),
        "animals_positive": int((per_animal > 0).sum()),
        "p": float(
            wilcoxon(
                merged[f"{IMMEDIATE_COEF}_swr"], merged[f"{IMMEDIATE_COEF}_no_swr"]
            ).pvalue
        ),
    }


def _immediate_session(frame: pd.DataFrame) -> dict[str, float]:
    """Across-session median of the immediate coefficient, the share positive,
    the animals whose median shares its sign, and the signed-rank p
    (``summarize_per_bin_rate_models._one_sample``)."""
    v = frame[IMMEDIATE_COEF].dropna()
    median = float(v.median())
    per_animal = frame.groupby("animal")[IMMEDIATE_COEF].median().dropna()
    return {
        "rr": float(np.exp(median)),
        "positive": float((v > 0).mean()),
        "animals_same_sign": int((np.sign(per_animal) == np.sign(median)).sum()),
        "p": float(wilcoxon(v).pvalue),
        "n": len(v),
    }


def _immediate_animal(frame: pd.DataFrame) -> tuple[float, int]:
    """``exp`` of the animal-balanced median immediate coefficient, and the
    animals whose own median is positive (``compare_fits._animal_balanced``)."""
    per_animal = (
        frame.dropna(subset=[IMMEDIATE_COEF]).groupby("animal")[IMMEDIATE_COEF].median()
    )
    _require(len(per_animal) == 5, f"{len(per_animal)} animals, not 5")
    return float(np.exp(per_animal.median())), int((per_animal > 0).sum())


def _delta_aic(fits: pd.DataFrame, child: str, parent: str, outcome: str) -> pd.Series:
    """Per-session ``AIC(child) - AIC(parent)`` (negative = child better)."""
    a = _rung(fits, child, outcome)[SC + ["aic"]]
    b = _rung(fits, parent, outcome)[SC + ["aic"]]
    merged = a.merge(b, on=SC, suffixes=("_child", "_parent"))
    return (merged["aic_child"] - merged["aic_parent"]).dropna()


def _spline_columns(fits: pd.DataFrame, clock: str) -> list[str]:
    prefix = f"coef_bs({clock},"
    cols = [
        c
        for c in fits.columns
        if isinstance(c, str)
        and c.startswith(prefix)
        and c.endswith("]")
        and ":" not in c
    ]
    return sorted(cols, key=lambda c: int(c.rsplit("[", 1)[1][:-1]))


def _time_course(
    fits: pd.DataFrame, model: str, outcome: str, clock: str, knots, tmax: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Across-session median fitted rate on the clock's supported interior, the
    outermost interior knots (``compare_fits._fitted_time_courses``): sessions
    flagged separated or with a non-finite curve are left out. Also returns the
    number of session curves the median is over."""
    grid = np.linspace(float(knots[0]), float(knots[-1]), 401)
    sub = _rung(fits, model, outcome)
    columns = _spline_columns(fits, clock)
    beta = sub[columns].to_numpy(dtype=float)
    intercept = sub["coef_Intercept"].to_numpy(dtype=float)
    ok = np.isfinite(beta).all(axis=1) & np.isfinite(intercept)
    ok &= ~(sub["separated"] == True).to_numpy()  # noqa: E712 (NaN must not count)
    basis = np.asarray(
        dmatrix(
            f"bs(x, knots={tuple(knots)}, lower_bound=0.0, upper_bound={tmax}) - 1",
            {"x": grid},
            return_type="dataframe",
        )
    )
    with np.errstate(over="ignore"):
        curves = np.exp(intercept[ok, None] + beta[ok] @ basis.T)
    curves = curves[np.isfinite(curves).all(axis=1)]
    return grid, np.median(curves, axis=0), len(curves)


def _clock_shape(fits: pd.DataFrame, outcome: str) -> dict[str, float]:
    """Where the stop-clock and arrival-clock time courses peak, whether each
    peak sits on the edge of its supported interval, the session curves behind
    each, and the KS distance between the two shapes
    (``compare_fits.arrival_clock_shape_table``; a peak on the edge is starred
    there by ``_flag_support_edge``: it bounds the burst rather than locating it).
    """
    grid, stop_curve, n_stop = _time_course(
        fits, STOP_TIME_RUNG, outcome, "tc", SPLINE_KNOTS, TMAX
    )
    arr_grid, arr_curve, n_arr = _time_course(
        fits,
        ARRIVAL_TIME_RUNG,
        outcome,
        ARRIVAL_CLOCK_COLUMN,
        ARRIVAL_SPLINE_KNOTS,
        TMAX_ARRIVAL,
    )
    ks = np.nanmax(
        np.abs(_shape_cdf(grid, stop_curve) - _shape_cdf(arr_grid, arr_curve))
    )
    stop_i, arr_i = int(np.argmax(stop_curve)), int(np.argmax(arr_curve))
    return {
        "time_peak_s": float(grid[stop_i]),
        "time_peak_edge": int(stop_i in (0, len(grid) - 1)),
        "time_peak_sessions": n_stop,
        "arrival_peak_s": float(arr_grid[arr_i]),
        "arrival_peak_edge": int(arr_i in (0, len(arr_grid) - 1)),
        "arrival_peak_sessions": n_arr,
        "clock_ks": float(ks),
    }


def _shape_cdf(grid: np.ndarray, curve: np.ndarray) -> np.ndarray:
    increments = np.diff(grid) * 0.5 * (curve[1:] + curve[:-1])
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    return cdf / cdf[-1]


def per_bin_values(
    fits: pd.DataFrame, all_stops: pd.DataFrame, source: str
) -> list[Value]:
    """Claim 19c: the per-bin conditional-intensity ladder (Family B).

    Read on the well-anchored cohort; ``allstops`` keys read the all-stops
    sensitivity fit with the same rules.
    """
    make = _value_factory(source, ("19c",), ("paper", "supplement"))
    values: list[Value] = []
    null = fits[fits["model"] == "M0_const"]
    stops = null.groupby("outcome")["n_stops"].sum()
    bins = null.groupby("outcome")["n_bins"].sum()
    _require(
        stops.nunique() == 1 and bins.nunique() == 1, "populations differ in cohort"
    )
    values += [
        make(
            "rate_sessions",
            null.groupby(SC).ngroups,
            "count",
            "count",
            "sessions in the per-bin rate fit",
        ),
        make(
            "rate_stops",
            stops.iloc[0],
            "count",
            "count",
            "stops in the frozen common cohort (M0_const, every population)",
        ),
        make(
            "rate_bins",
            bins.iloc[0],
            "count",
            "count",
            "15 ms bins in the frozen common cohort",
        ),
    ]
    for word, (outcome, label) in RATE_POPULATIONS.items():
        base = _rung(fits, "M0_const", outcome)
        full = _rung(fits, FULL_RATE_RUNG, outcome)
        kernel = reward_kernel_rate_ratios(full)
        lag_one = _lag_one_kernel(full)
        per_animal = lag_one.groupby(level=0).median()
        _require(len(per_animal) == 5, f"{outcome}: {len(per_animal)} animals")
        immediate = _immediate_session(full)
        rr_animal, animals_pos = _immediate_animal(full)
        history = self_history_rate_ratios(full)
        refractory = 0
        while refractory < len(history) and history[refractory] < 1.0:
            refractory += 1
        peak_lag = int(np.nanargmax(history)) + 1
        values += [
            make(
                f"rate_baseline_{word}",
                np.exp(base["coef_Intercept"].dropna()).median(),
                "quantity",
                "dec2",
                f"{label}: median over sessions of exp(intercept) of M0_const, events per second of immobility",
                "events/s",
            ),
            make(
                f"rate_full_sessions_{word}",
                full["coef_Intercept"].notna().sum(),
                "count",
                "count",
                f"{label}: sessions with a fit of {FULL_RATE_RUNG}",
            ),
            make(
                f"rate_time_sessions_{word}",
                _rung(fits, STOP_TIME_RUNG, outcome)["coef_Intercept"].notna().sum(),
                "count",
                "count",
                f"{label}: sessions with a fit of {STOP_TIME_RUNG}",
            ),
            make(
                f"rate_kernel_lag_one_{word}",
                kernel[0],
                "quantity",
                "dec2",
                f"{label}: reward kernel at lag 1, reconstructed per session from the three basis "
                f"coefficients of {FULL_RATE_RUNG}, median over sessions (session-pooled)",
                "rate ratio",
            ),
            make(
                f"rate_kernel_lag_one_animal_{word}",
                np.exp(per_animal.median()),
                "quantity",
                "dec2",
                f"{label}: the same lag-1 kernel, median over animals of each animal's median "
                "(animal-balanced)",
                "rate ratio",
            ),
            make(
                f"rate_kernel_lag_one_animals_positive_{word}",
                (per_animal > 0).sum(),
                "count",
                "count",
                f"{label}: animals whose median lag-1 kernel is above 1",
            ),
            make(
                f"rate_kernel_late_min_{word}",
                kernel[2:].min(),
                "quantity",
                "dec2",
                f"{label}: lowest median reconstructed kernel over lags 3-10",
                "rate ratio",
            ),
            make(
                f"rate_kernel_late_max_{word}",
                kernel[2:].max(),
                "quantity",
                "dec2",
                f"{label}: highest median reconstructed kernel over lags 3-10",
                "rate ratio",
            ),
            make(
                f"rate_kernel_lag_ten_{word}",
                kernel[-1],
                "quantity",
                "dec2",
                f"{label}: median reconstructed kernel at lag 10",
                "rate ratio",
            ),
            make(
                f"rate_immediate_session_{word}",
                immediate["rr"],
                "quantity",
                "sig2",
                f"{label}: exp of the across-session median of the immediate basis coefficient "
                f"{IMMEDIATE_COEF} in {FULL_RATE_RUNG} (the full 0 to 1 contrast on bump 0)",
                "rate ratio",
            ),
            make(
                f"rate_immediate_session_positive_fraction_{word}",
                immediate["positive"],
                "fraction",
                "pct",
                f"{label}: share of sessions whose immediate coefficient is positive",
            ),
            make(
                f"rate_immediate_session_animals_{word}",
                immediate["animals_same_sign"],
                "count",
                "count",
                f"{label}: animals whose median immediate coefficient has the sign of the "
                "across-session median",
            ),
            make(
                f"rate_immediate_session_p_{word}",
                immediate["p"],
                "quantity",
                "pval",
                f"{label}: signed-rank p of the immediate coefficient against 0 across sessions",
            ),
            make(
                f"rate_immediate_animal_{word}",
                rr_animal,
                "quantity",
                "sig2",
                f"{label}: exp of the animal-balanced median immediate coefficient (rr_immediate)",
                "rate ratio",
            ),
            make(
                f"rate_immediate_animals_positive_{word}",
                animals_pos,
                "count",
                "count",
                f"{label}: animals whose median immediate coefficient is positive",
            ),
            make(
                f"rate_marginal_reward_delta_aic_{word}",
                _delta_aic(fits, "M1_rew", "M0_const", outcome).median(),
                "quantity",
                "sig2s",
                f"{label}: median per-session AIC(M1_rew) - AIC(M0_const); negative = child better",
                "AIC",
            ),
            make(
                f"rate_conditional_reward_delta_aic_{word}",
                _delta_aic(fits, FULL_RATE_RUNG, "M6r_time_histcos", outcome).median(),
                "quantity",
                "sig2s",
                f"{label}: median per-session AIC({FULL_RATE_RUNG}) - AIC(M6r_time_histcos); "
                "negative = child better",
                "AIC",
            ),
            make(
                f"rate_burst_peak_{word}",
                history[peak_lag - 1],
                "quantity",
                "dec2",
                f"{label}: peak of the median reconstructed self-history kernel of {FULL_RATE_RUNG}",
                "rate ratio",
                pending=HISTORY_PENDING,
            ),
            make(
                f"rate_burst_peak_lag_ms_{word}",
                peak_lag * 15,
                "quantity",
                "int",
                f"{label}: lag of that peak (15 ms lags)",
                "ms",
                pending=HISTORY_PENDING,
            ),
            make(
                f"rate_refractory_ms_{word}",
                refractory * 15,
                "quantity",
                "int",
                f"{label}: end of the first run of lags, from lag 1, at which the median "
                "self-history kernel is below 1",
                "ms",
                pending=HISTORY_PENDING,
            ),
        ]
        cohort = fits.attrs["arrival_clock_cohort"][outcome]
        cohort_all = all_stops.attrs["arrival_clock_cohort"][outcome]
        arrival_rung = _rung(fits, ARRIVAL_TIME_RUNG, outcome)
        values += [
            make(
                f"rate_arrival_event_share_{word}",
                cohort["n_event_bins_arrival_window"] / cohort["n_event_bins_frozen"],
                "fraction",
                "dec2",
                f"{label}: share of its event bins the arrival window [0, 8] s keeps",
            ),
            make(
                f"rate_allstops_arrival_event_share_{word}",
                cohort_all["n_event_bins_arrival_window"]
                / cohort_all["n_event_bins_frozen"],
                "fraction",
                "dec2",
                f"{label}: the same share on the all-stops fit",
            ),
            make(
                f"rate_arrival_separated_fraction_{word}",
                (arrival_rung["separated"] == True).mean(),  # noqa: E712
                "fraction",
                "pct",
                f"{label}: share of {ARRIVAL_TIME_RUNG} fits flagged separated",
            ),
        ]
        for prefix, frame in (("rate", fits), ("rate_allstops", all_stops)):
            shape = _clock_shape(frame, outcome)
            cohort_label = "" if prefix == "rate" else ", all-stops fit"
            for clock, rung, window in (
                ("time", STOP_TIME_RUNG, "[1, 3] s"),
                ("arrival", ARRIVAL_TIME_RUNG, "[1, 6] s"),
            ):
                edge = shape[f"{clock}_peak_edge"]
                values += [
                    make(
                        f"{prefix}_{clock}_peak_s_{word}",
                        shape[f"{clock}_peak_s"],
                        "quantity",
                        "dec2",
                        f"{label}: argmax of the median fitted {rung} time course, read on "
                        f"{window}{cohort_label}"
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
                        f"{label}: 1 if that argmax sits on the edge of {window}{cohort_label}",
                    ),
                    make(
                        f"{prefix}_{clock}_peak_sessions_{word}",
                        shape[f"{clock}_peak_sessions"],
                        "count",
                        "count",
                        f"{label}: unflagged session curves behind that median{cohort_label}",
                    ),
                ]
            values.append(
                make(
                    f"{prefix}_clock_ks_{word}",
                    shape["clock_ks"],
                    "quantity",
                    "dec2",
                    f"{label}: KS distance between the two clocks' median time courses, each "
                    "normalized over its own supported interval and read on relative "
                    f"position{cohort_label}",
                )
            )
    paired = _paired_immediate(fits)
    paired_all = _paired_immediate(all_stops)
    no_swr_all = _rung(all_stops, FULL_RATE_RUNG, "non_local_no_swr")
    immediate_all = _immediate_session(no_swr_all)
    swr_session = _immediate_session(_rung(fits, FULL_RATE_RUNG, "non_local_swr"))
    no_swr_session = _immediate_session(_rung(fits, FULL_RATE_RUNG, "non_local_no_swr"))
    values += [
        make(
            "rate_paired_sessions",
            paired["n"],
            "count",
            "count",
            "sessions with both non-local streams fit, paired",
        ),
        make(
            "rate_paired_positive_fraction",
            paired["positive"],
            "fraction",
            "pct",
            f"share of paired sessions whose immediate coefficient {IMMEDIATE_COEF} is larger for "
            "SWR-coupled than for non-SWR non-local events",
        ),
        make(
            "rate_paired_animals_positive",
            paired["animals_positive"],
            "count",
            "count",
            "animals whose median paired difference is positive",
        ),
        make(
            "rate_paired_p",
            paired["p"],
            "quantity",
            "pval",
            "signed-rank p of the paired difference",
        ),
        make(
            "rate_allstops_paired_sessions",
            paired_all["n"],
            "count",
            "count",
            "the same, all-stops fit: paired sessions",
        ),
        make(
            "rate_allstops_paired_positive_fraction",
            paired_all["positive"],
            "fraction",
            "pct",
            "the same, all-stops fit: share positive",
        ),
        make(
            "rate_allstops_paired_p",
            paired_all["p"],
            "quantity",
            "pval",
            "the same, all-stops fit: signed-rank p",
        ),
        make(
            "rate_immediate_ratio_swr_to_no_swr",
            swr_session["rr"] / no_swr_session["rr"],
            "quantity",
            "sig2",
            "across-session immediate RR of non-local events with SWR divided by that of "
            "non-local events without SWR",
            "ratio",
        ),
        make(
            "rate_allstops_immediate_session_no_swr",
            immediate_all["rr"],
            "quantity",
            "sig2",
            "all-stops fit, non-local without SWR: exp of the across-session median immediate coefficient",
            "rate ratio",
        ),
        make(
            "rate_allstops_immediate_session_p_no_swr",
            immediate_all["p"],
            "quantity",
            "pval",
            "its signed-rank p",
        ),
        make(
            "rate_allstops_immediate_session_animals_no_swr",
            immediate_all["animals_same_sign"],
            "count",
            "count",
            "its animals with the median's sign",
        ),
        make(
            "rate_allstops_kernel_lag_one_no_swr",
            reward_kernel_rate_ratios(no_swr_all)[0],
            "quantity",
            "dec2",
            "all-stops fit, non-local without SWR: median reconstructed kernel at lag 1",
            "rate ratio",
        ),
    ]
    for word, (outcome, label) in RATE_POPULATIONS.items():
        rr, pos = _immediate_animal(_rung(all_stops, FULL_RATE_RUNG, outcome))
        values += [
            make(
                f"rate_allstops_immediate_animal_{word}",
                rr,
                "quantity",
                "sig2",
                f"all-stops fit, {label}: rr_immediate (animal-balanced)",
                "rate ratio",
            ),
            make(
                f"rate_allstops_immediate_animals_positive_{word}",
                pos,
                "count",
                "count",
                f"all-stops fit, {label}: animals positive",
            ),
        ]
    nl = fits[fits["outcome"] == "non_local"]
    for word, model in (
        ("const", "M0_const"),
        ("time", STOP_TIME_RUNG),
        ("full", FULL_RATE_RUNG),
        ("reward_by_time", "M9r_rew_x_time"),
    ):
        rung = nl[nl["model"] == model]
        cv = rung["cv_bits_per_event"][np.isfinite(rung["cv_bits_per_event"])]
        values += [
            make(
                f"rate_ladder_aic_{word}",
                rung["aic"].median(),
                "quantity",
                "int",
                f"non-local (all): median AIC of {model}",
                "AIC",
            ),
            make(
                f"rate_ladder_cv_bits_{word}",
                cv.median(),
                "quantity",
                "dec2",
                f"non-local (all): median held-out bits per event of {model}, over sessions with a finite score",
                "bits/event",
            ),
            make(
                f"rate_ladder_cv_sessions_{word}",
                len(cv),
                "count",
                "count",
                f"non-local (all): sessions with a finite held-out score for {model}",
            ),
            make(
                f"rate_ladder_ks_band_{word}",
                rung["ks_within_band"].mean(),
                "fraction",
                "dec2",
                f"non-local (all): share of {model} fits inside the KS band",
            ),
        ]
    return values + _population_ranges(values, make)


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


def _coef_session(frame: pd.DataFrame, column: str) -> dict[str, float]:
    """``_immediate_session`` for any one coefficient column."""
    if column == IMMEDIATE_COEF:
        return _immediate_session(frame)
    return _immediate_session(
        frame.drop(columns=IMMEDIATE_COEF).rename(columns={column: IMMEDIATE_COEF})
    )


def per_bin_detail_values(
    fits: pd.DataFrame, all_stops: pd.DataFrame, in_transit: pd.DataFrame, source: str
) -> list[Value]:
    """The rest of the per-bin ladder's readout (``docs/per_bin_rate_results.md``,
    ``docs/rate_models.md``), each read with ``summarize_per_bin_rate_models.py``'s
    rule unless the description says otherwise."""
    make = _value_factory(source, ("19c",), ("supplement",))
    values: list[Value] = []
    null = fits[fits["model"] == "M0_const"]
    hist_columns = [f"coef_hist_rcos_{k}" for k in range(4)]
    for word, (outcome, label) in RATE_POPULATIONS.items():
        rate = np.exp(_rung(fits, "M0_const", outcome)["coef_Intercept"].dropna())
        raw = _coef_session(_rung(fits, "M1_rew", outcome), IMMEDIATE_COEF)
        full = _rung(fits, FULL_RATE_RUNG, outcome)
        kernel = reward_kernel_rate_ratios(full)
        hist_fits = full.dropna(subset=hist_columns)
        flagged = hist_fits["separated_terms"].map(
            lambda v: isinstance(v, str)
            and any(t.startswith("hist_rcos") for t in v.split("|"))
        )
        gated = self_history_rate_ratios(hist_fits[~flagged])
        raw_history = self_history_rate_ratios(_rung(fits, "M3r_histcos", outcome))
        marginal = _delta_aic(fits, "M1_rew", "M0_const", outcome)
        values += [
            make(
                f"rate_baseline_lower_quartile_{word}",
                rate.quantile(0.25),
                "quantity",
                "dec2",
                f"{label}: 25th percentile over sessions of the M0_const rate",
                "events/s",
            ),
            make(
                f"rate_baseline_upper_quartile_{word}",
                rate.quantile(0.75),
                "quantity",
                "dec2",
                f"{label}: 75th percentile over sessions of the M0_const rate",
                "events/s",
            ),
            make(
                f"rate_bin_count_{word}",
                rate.median() * BIN_S,
                "quantity",
                "sig2",
                f"{label}: median M0_const rate times the 15 ms bin, the mean per-bin count",
                "events/bin",
            ),
            make(
                f"rate_raw_immediate_session_{word}",
                raw["rr"],
                "quantity",
                "sig2",
                f"{label}: exp of the across-session median {IMMEDIATE_COEF} of M1_rew (uncontrolled)",
                "rate ratio",
            ),
            make(
                f"rate_marginal_reward_child_better_fraction_{word}",
                (marginal < 0).mean(),
                "fraction",
                "pct",
                f"{label}: share of sessions in which M1_rew has the lower AIC than M0_const",
            ),
            make(
                f"rate_hist_sessions_{word}",
                len(hist_fits),
                "count",
                "count",
                f"{label}: sessions whose {FULL_RATE_RUNG} fit carries the self-history block",
            ),
            make(
                f"rate_hist_separated_fraction_{word}",
                flagged.mean(),
                "fraction",
                "pct",
                f"{label}: share of those fits flagging a hist_rcos term separated",
            ),
            make(
                f"rate_burst_peak_gated_{word}",
                gated.max(),
                "quantity",
                "dec2",
                f"{label}: peak of the median self-history kernel over the fits that flag no hist_rcos term",
                "rate ratio",
            ),
            make(
                f"rate_burst_peak_gated_lag_ms_{word}",
                (int(np.argmax(gated)) + 1) * 15,
                "quantity",
                "int",
                f"{label}: lag of that gated peak",
                "ms",
            ),
            make(
                f"rate_raw_burst_peak_{word}",
                raw_history.max(),
                "quantity",
                "dec2",
                f"{label}: peak of the median self-history kernel of M3r_histcos (uncontrolled)",
                "rate ratio",
                pending=HISTORY_PENDING,
            ),
        ]
        for k, bump in ((1, "one"), (2, "two")):
            agg = _coef_session(full, f"coef_{REWARD_HISTORY_COLUMNS[k]}")
            values += [
                make(
                    f"rate_bump_{bump}_session_{word}",
                    agg["rr"],
                    "quantity",
                    "sig2",
                    f"{label}: exp of the across-session median coef_{REWARD_HISTORY_COLUMNS[k]} of {FULL_RATE_RUNG}",
                    "rate ratio",
                ),
                make(
                    f"rate_bump_{bump}_positive_fraction_{word}",
                    agg["positive"],
                    "fraction",
                    "pct",
                    f"{label}: share of sessions with that coefficient positive",
                ),
                make(
                    f"rate_bump_{bump}_animals_{word}",
                    agg["animals_same_sign"],
                    "count",
                    "count",
                    f"{label}: animals whose median has the across-session median's sign",
                ),
            ]
        for i, lag in enumerate(_LAG_WORDS[1:-1], start=1):  # lag ten is above
            values.append(
                make(
                    f"rate_kernel_lag_{lag}_{word}",
                    kernel[i],
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
                    np.isfinite(_rung(fits, rung, outcome)["aic"]).sum(),
                    "count",
                    "count",
                    f"{label}: {rung} fits with a finite AIC",
                )
            )
        transit = in_transit.attrs["arrival_clock_cohort"].get(outcome)
        if transit and transit["n_event_bins_frozen"]:
            values.append(
                make(
                    f"rate_intransit_arrival_event_share_{word}",
                    transit["n_event_bins_arrival_window"]
                    / transit["n_event_bins_frozen"],
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
    )
    for word, a, b in pairs:
        A = null.loc[null["outcome"] == a, SC + ["coef_Intercept"]]
        B = null.loc[null["outcome"] == b, SC + ["coef_Intercept"]]
        merged = A.merge(B, on=SC, suffixes=("_a", "_b")).dropna()
        diff = merged["coef_Intercept_a"] - merged["coef_Intercept_b"]
        values += [
            make(
                f"rate_baseline_ratio_{word}",
                np.exp(diff.median()),
                "quantity",
                "dec2",
                f"median within-session M0_const rate ratio, {a} over {b}",
                "ratio",
            ),
            make(
                f"rate_baseline_ratio_{word}_p",
                wilcoxon(merged["coef_Intercept_a"], merged["coef_Intercept_b"]).pvalue,
                "quantity",
                "pval",
                "its paired signed-rank p",
            ),
        ]
    nl = fits[fits["outcome"] == "non_local"]
    for word, model in LADDER.items():
        rung = nl[nl["model"] == model]
        values.append(
            make(
                f"rate_ladder_sessions_{word}",
                np.isfinite(rung["aic"]).sum(),
                "count",
                "count",
                f"non-local (all): sessions with a finite-AIC fit of {model}",
            )
        )
        if word in ("rew", "histcos", "time_histcos"):
            cv = rung["cv_bits_per_event"][np.isfinite(rung["cv_bits_per_event"])]
            values += [
                make(
                    f"rate_ladder_aic_{word}",
                    rung["aic"].median(),
                    "quantity",
                    "int",
                    f"non-local (all): median AIC of {model}",
                    "AIC",
                ),
                make(
                    f"rate_ladder_cv_bits_{word}",
                    cv.median(),
                    "quantity",
                    "dec2",
                    f"non-local (all): median held-out bits per event of {model}, over sessions with a finite score",
                    "bits/event",
                ),
                make(
                    f"rate_ladder_cv_sessions_{word}",
                    len(cv),
                    "count",
                    "count",
                    f"non-local (all): sessions with a finite held-out score for {model}",
                ),
                make(
                    f"rate_ladder_ks_band_{word}",
                    rung["ks_within_band"].mean(),
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
            rung = _rung(fits, model, pop)
            values.append(
                make(
                    f"rate_arrival_{word}_cv_nan_fraction_{pop}",
                    (~np.isfinite(rung["cv_bits_per_event"])).mean(),
                    "fraction",
                    "pct",
                    f"{RATE_POPULATIONS[pop][1]}: share of {model} fits whose held-out score is not finite",
                )
            )
    marginal_all = _delta_aic(all_stops, "M1_rew", "M0_const", "non_local_no_swr")
    cohort = all_stops.attrs["arrival_clock_cohort"]["non_local"]
    before = cohort["n_bins_frozen"] + cohort["n_bins_dropped_arrival_clock"]
    transit_shares = [
        v.value
        for v in values
        if v.key.startswith("rate_intransit_arrival_event_share_")
    ]
    values += [
        make(
            "rate_allstops_marginal_reward_delta_aic_non_local_no_swr",
            marginal_all.median(),
            "quantity",
            "sig2s",
            "all-stops fit, non-local without SWR: median per-session AIC(M1_rew) - AIC(M0_const)",
            "AIC",
        ),
        make(
            "rate_allstops_marginal_reward_child_better_fraction_non_local_no_swr",
            (marginal_all < 0).mean(),
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
            before,
            "count",
            "count",
            "all-stops fit: bins before that removal",
        ),
        make(
            "rate_allstops_arrival_bins_dropped_fraction",
            cohort["n_bins_dropped_arrival_clock"] / before,
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
            cohort["n_bins_arrival_window"] / cohort["n_bins_frozen"],
            "fraction",
            "pct",
            "all-stops fit: share of frozen-cohort bins inside the arrival window",
        ),
        make(
            "rate_intransit_arrival_event_share_low",
            min(transit_shares),
            "fraction",
            "dec2",
            "lowest over populations of the in-transit fit's arrival-window event share",
        ),
        make(
            "rate_intransit_arrival_event_share_high",
            max(transit_shares),
            "fraction",
            "dec2",
            "highest over populations of the in-transit fit's arrival-window event share",
        ),
    ]
    return values


def stop_table_values(
    stops: pd.DataFrame, onsets: pd.DataFrame, source: str
) -> list[Value]:
    """Stop- and onset-table facts the rate docs quote: dwell by outcome on the
    well-anchored cohort, survival past 2 s and 4 s, the outcome-boundary shift
    (``compare_fits.outcome_boundary_shift_cohort``), and the share of onsets
    inside the ``TMAX`` truncation."""
    make = _value_factory(source, ("19c",), ("supplement",))
    anchored = stops[stops["occupancy_state"].isin(["at_well", "mixed"])]
    known = anchored.dropna(subset=["is_rewarded_outcome"])
    duration = known.groupby("is_rewarded_outcome")["immobility_duration_s"]
    means, medians = duration.mean(), duration.median()
    crosses = (
        (stops["stop_start"] < stops["outcome_known_time_s"])
        & (stops["outcome_known_time_s"] <= stops["stop_end"])
    ).fillna(False)
    ordered = stops.sort_values(SC + ["stop_start"], kind="stable")
    next_arrival = ordered.groupby(SC, sort=False, observed=True)[
        "last_well_arrival_time_s"
    ].shift(-1)
    spans = (
        ((next_arrival > ordered["stop_start"]) & (next_arrival <= ordered["stop_end"]))
        .fillna(False)
        .reindex(stops.index)
    )
    delta = (stops["outcome_known_time_s"] - stops["stop_start"]).to_numpy(float)
    bins = aggregate_to_timebins(
        stops,
        pd.DataFrame({"stop_id": [], "t_since_stop_onset": []}),
        bin_s=BIN_S,
        tmax=TMAX,
        history_window_s=longest_history_window_s(MODEL_PER_BIN_RATE.values()),
    )
    last_bin = stops["stop_id"].map(bins.groupby("stop_id")["binj"].max())
    with np.errstate(invalid="ignore"):
        has_bin = last_bin.to_numpy(float) >= np.ceil(delta / BIN_S)
    shifted = (
        crosses
        & ~spans
        & pd.Series(np.nan_to_num(has_bin, nan=False).astype(bool), stops.index)
    )
    values = [
        make(
            "stop_duration_mean_rewarded",
            means[1.0],
            "quantity",
            "sig2",
            "well-anchored stops with a rewarded outcome: mean immobility duration",
            "s",
        ),
        make(
            "stop_duration_mean_unrewarded",
            means[0.0],
            "quantity",
            "sig2",
            "well-anchored stops with an omitted outcome: mean immobility duration",
            "s",
        ),
        make(
            "stop_duration_mean_ratio",
            means[1.0] / means[0.0],
            "quantity",
            "sig2",
            "ratio of those means",
            "ratio",
        ),
        make(
            "stop_duration_median_rewarded",
            medians[1.0],
            "quantity",
            "sig2",
            "well-anchored stops with a rewarded outcome: median immobility duration",
            "s",
        ),
        make(
            "stop_duration_median_unrewarded",
            medians[0.0],
            "quantity",
            "sig2",
            "well-anchored stops with an omitted outcome: median immobility duration",
            "s",
        ),
        make(
            "stop_duration_median_ratio",
            medians[1.0] / medians[0.0],
            "quantity",
            "sig2",
            "ratio of those medians",
            "ratio",
        ),
        make(
            "stop_survival_two_s",
            (anchored["immobility_duration_s"] >= 2).mean(),
            "fraction",
            "pct",
            "share of well-anchored stops still immobile at 2 s",
        ),
        make(
            "stop_survival_four_s",
            (anchored["immobility_duration_s"] >= 4).mean(),
            "fraction",
            "pct",
            "share of well-anchored stops still immobile at 4 s",
        ),
        make(
            "stops_in_transit",
            (stops["occupancy_state"] == "in_transit").sum(),
            "count",
            "count",
            "in-transit stops in the canonical stop table",
        ),
        make(
            "stops_crossing_outcome",
            crosses.sum(),
            "count",
            "count",
            "stops with stop_start < outcome_known_time_s <= stop_end (all stops)",
        ),
        make(
            "stops_crossing_outcome_fraction",
            crosses.mean(),
            "fraction",
            "pct",
            "their share of all stops",
        ),
        make(
            "stops_outcome_shifted",
            shifted.sum(),
            "count",
            "count",
            "crossing stops whose per-bin reward history is shifted (no later arrival inside, a bin at or past the boundary)",
        ),
        make(
            "stops_outcome_shifted_fraction",
            shifted.mean(),
            "fraction",
            "pct",
            "their share of all stops",
        ),
        make(
            "stops_spanning_arrival",
            spans.sum(),
            "count",
            "count",
            "stops containing a later well arrival",
        ),
        make(
            "stops_spanning_arrival_crossing",
            (spans & crosses).sum(),
            "count",
            "count",
            "of those, stops that also cross their outcome boundary",
        ),
    ]
    for word, event_type in (
        ("non_local", "non_local"),
        ("swr", "SWR"),
        ("mua", "MUA"),
    ):
        t = onsets.loc[
            onsets["event_type"] == event_type, "t_since_stop_onset"
        ].dropna()
        values.append(
            make(
                f"onsets_within_tmax_fraction_{word}",
                (t < TMAX).mean(),
                "fraction",
                "pct",
                f"share of {event_type} onsets within the first {TMAX:g} s of their stop (event_onset_table)",
            )
        )
    t = onsets.loc[onsets["event_type"] == "non_local", "t_since_stop_onset"].dropna()
    values.append(
        make(
            "onset_median_s_non_local",
            t.median(),
            "quantity",
            "sig2",
            "median time of a non-local onset after its stop's speed-threshold crossing",
            "s",
        )
    )
    return values


TRUNK_PENDING = (
    "read from an exploratory script's output (scripts/exploratory/_trunk_zscore_extract.py, "
    "_trunk_zscore_per_event.pkl, written 2026-06-08 on the pre-refresh tables): stale, "
    "and the rule behind the earlier 3.55x figure is not recoverable from it"
)


def trunk_values(per_event: pd.DataFrame, source: str) -> list[Value]:
    """The trunk-arm share of each event's posterior mass, SWR against non-SWR."""
    make = _value_factory(source, ("14",), ("docs: replay_rate_content_swr_synthesis",))
    mean = per_event.groupby("is_swr")["trunk_share"].mean()
    return [
        make(
            "trunk_share_mean_swr",
            mean[True],
            "fraction",
            "pct",
            "mean share of an SWR event's posterior mass on trunk arms",
            pending=TRUNK_PENDING,
        ),
        make(
            "trunk_share_mean_non_swr",
            mean[False],
            "fraction",
            "pct",
            "mean share of a non-SWR event's posterior mass on trunk arms",
            pending=TRUNK_PENDING,
        ),
        make(
            "trunk_share_ratio",
            mean[True] / mean[False],
            "quantity",
            "sig2",
            "ratio of those means",
            "ratio",
            pending=TRUNK_PENDING,
        ),
    ]


#: Per-population rate values whose range over the four populations is quoted.
_RANGED = (
    "rate_kernel_late_min",
    "rate_time_peak_s",
    "rate_arrival_peak_s",
    "rate_clock_ks",
    "rate_allstops_time_peak_s",
    "rate_allstops_arrival_peak_s",
    "rate_allstops_clock_ks",
    "rate_arrival_separated_fraction",
    "rate_burst_peak",
    "rate_burst_peak_lag_ms",
)


def _population_ranges(values: list[Value], make: Callable[..., Value]) -> list[Value]:
    """The lowest and highest of each :data:`_RANGED` value over the populations.

    A time-course peak on the edge of its supported interval bounds the peak
    rather than locating it, so the peak ranges are over the located peaks only,
    with the number of edge peaks beside them.
    """
    by_key = {v.key: v for v in values}
    ranges = []
    for stem in _RANGED:
        members = [by_key[f"{stem}_{word}"] for word in RATE_POPULATIONS]
        if stem.endswith("_peak_s"):
            edge_stem = stem.replace("_peak_s", "_peak_edge")
            edges = [by_key[f"{edge_stem}_{word}"].value for word in RATE_POPULATIONS]
            ranges.append(
                make(
                    f"{edge_stem}_count",
                    sum(edges),
                    "count",
                    "count",
                    "populations whose "
                    + members[0].description.split(": ", 1)[1].split(";")[0]
                    + " sits on the interval's edge",
                )
            )
            members = [m for m, e in zip(members, edges, strict=True) if not e]
            _require(bool(members), f"{stem}: every population's peak is on the edge")
        for end, pick in (("low", min), ("high", max)):
            chosen = pick(members, key=lambda v: v.value)
            ranges.append(
                make(
                    f"{stem}_{end}",
                    chosen.value,
                    chosen.kind,
                    chosen.style,
                    f"{'lowest' if end == 'low' else 'highest'} over the populations"
                    + (
                        " whose peak is located (not on the edge)"
                        if stem.endswith("_peak_s")
                        else ""
                    )
                    + ": "
                    + chosen.description.split(": ", 1)[1].split(";")[0],
                    chosen.unit,
                    pending=chosen.pending,
                )
            )
    return ranges


# --- 3a: rate, per stop (Family A) -------------------------------------------


def rate_contrast_separation_values(fits: pd.DataFrame, source: str) -> list[Value]:
    """How heavily the per-bin ``is_swr``-contrast rate ladder is separated: the
    share of its ``contrast`` fits flagged separated, per cohort, and the share of
    fits carrying ``is_swr:reward_history_rcos_1`` that flag that term."""
    make = _value_factory(
        source, ("19c",), ("docs: replay_rate_content_swr_synthesis",)
    )
    contrast = fits[fits["population"] == "contrast"]
    term = "is_swr:reward_history_rcos_1"
    carries = fits[fits[f"coef_{term}"].notna()]
    flagged = carries["separated_terms"].map(
        lambda v: isinstance(v, str) and term in v.split("|")
    )
    values = [
        make(
            f"rate_contrast_separated_fraction_{cohort}",
            (contrast.loc[contrast["cohort"] == cohort, "separated"] == True).mean(),  # noqa: E712
            "fraction",
            "pct",
            f"share of the is_swr-contrast rate ladder's {cohort} contrast fits flagged "
            "separated",
        )
        for cohort in ("primary", "post_outcome")
    ]
    values.append(
        make(
            "rate_contrast_term_separated_fraction",
            flagged.mean(),
            "fraction",
            "pct",
            f"share of the fits carrying {term} that flag it separated",
        )
    )
    return values


def cohort_audit_values(inputs: Part3Inputs) -> list[Value]:
    """What the rate families' common-cohort freeze cost, as their runners'
    cohort audits record it (population non-local; every population shares it)."""
    values = []
    for prefix, path, rows in (
        ("rate_audit", inputs.per_bin_audit, {"modeled": "M0_const"}),
        ("per_stop_audit", inputs.per_stop_audit, {"modeled": "R0", "full": "R1"}),
    ):
        audit = pd.read_csv(path)
        audit = audit[audit["population"] == "non_local"].set_index("model")
        make = _value_factory(f"{rel(path)} (cohort audit)", ("19c",), ("supplement",))
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
    """Claim 19c: the per-stop reward step ``R1 - R0`` on the well-anchored
    cohort (``compare_fits.cohort_per_stop_delta_aic_table``): a session counts
    only when both rungs saw the same stops; negative = child better."""
    make = _value_factory(source, ("19c",), ("supplement",))
    _require(
        set(fits["stop_cohort"]) == {"at_well"}, "per-stop fit is not the at_well one"
    )
    values = []
    for word, (outcome, label) in RATE_POPULATIONS.items():
        sub = fits[fits["outcome"] == outcome]
        a = sub.loc[sub["model"] == "R1", SC + ["aic", "n_stops"]]
        b = sub.loc[sub["model"] == "R0", SC + ["aic", "n_stops"]]
        merged = a.merge(b, on=SC, suffixes=("_child", "_parent"))
        paired = merged[merged["n_stops_child"] == merged["n_stops_parent"]]
        delta = (paired["aic_child"] - paired["aic_parent"]).dropna()
        per_animal = (
            paired.assign(d=delta).dropna(subset=["d"]).groupby("animal")["d"].median()
        )
        _require(len(per_animal) == 5, f"{outcome}: {len(per_animal)} animals")
        values += [
            make(
                f"per_stop_delta_aic_animal_{word}",
                per_animal.median(),
                "quantity",
                "sig2s",
                f"{label}: median over animals of the per-animal median AIC(R1) - AIC(R0)",
                "AIC",
            ),
            make(
                f"per_stop_delta_aic_session_{word}",
                delta.median(),
                "quantity",
                "sig2s",
                f"{label}: median over sessions of AIC(R1) - AIC(R0)",
                "AIC",
            ),
            make(
                f"per_stop_sessions_{word}",
                len(delta),
                "count",
                "count",
                f"{label}: sessions with both rungs fit on the same stops",
            ),
            make(
                f"per_stop_child_better_fraction_{word}",
                (delta < 0).mean(),
                "fraction",
                "pct",
                f"{label}: share of sessions in which R1 has the lower AIC",
            ),
            make(
                f"per_stop_animals_child_better_{word}",
                (per_animal < 0).sum(),
                "count",
                "count",
                f"{label}: animals whose median step is negative",
            ),
        ]
    return values


# --- 3b: content ---------------------------------------------------------------


#: The claims-table rows each content key word supports, by its leading words.
_CONTENT_CLAIMS = (
    ("swr_armreward", ("19b", "19f")),
    ("swr_armvalue", ("19f",)),
    ("swr_prevgoal", ("19f",)),
    ("swr_armuncert", ("19g",)),
    ("armreward", ("19b",)),
    ("armvalue", ("19b", "19g")),
    ("armuncert", ("19g",)),
    ("switchrate", ("19f",)),
)


def _content_claims(word: str) -> tuple[str, ...]:
    """The claims a content key supports; 19a (content anchoring) otherwise."""
    return next(
        (claims for lead, claims in _CONTENT_CLAIMS if word.startswith(lead)), ("19a",)
    )


def content_contrast_values(
    audited: dict[str, pd.DataFrame], sources: dict[str, str]
) -> list[Value]:
    """Claims 19a, 19b, 19f, 19g: the content ``is_swr``-contrast ladders.

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
        claims = _content_claims(word.removeprefix("dm_"))
        make = _value_factory(sources[suffix], claims, ("supplement",))
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
    """Claim 19a: what the concentration contrast means, and event length."""
    make = _value_factory(source, ("19a",), ("supplement",))
    total = values["dm_breadth"].value
    adjusted = values["dm_breadth_length_adjusted"].value
    bins = events.groupby("is_swr")["n_nonlocal_bins"].median()
    return [
        make(
            "dm_breadth_ratio",
            np.exp(total),
            "quantity",
            "sig2",
            "exp of the concentration contrast log_phi_is_swr on C2_base: an SWR event's Dirichlet concentration over a non-SWR event's",
            "ratio",
        ),
        make(
            "dm_breadth_length_adjusted_ratio",
            np.exp(adjusted),
            "quantity",
            "sig2",
            "the same with the length-dependent concentration",
            "ratio",
        ),
        make(
            "dm_breadth_length_share",
            1 - adjusted / total,
            "fraction",
            "pct",
            "share of the concentration contrast that the length term absorbs: 1 - adjusted / total",
        ),
        make(
            "event_bins_median_swr",
            bins[True],
            "quantity",
            "int",
            "median non-local bins of SWR-overlapping events (event table)",
            "bins",
            used_in_=("supplement",),
        ),
        make(
            "event_bins_median_non_swr",
            bins[False],
            "quantity",
            "int",
            "median non-local bins of non-SWR events (event table)",
            "bins",
        ),
        make(
            "event_bins_median",
            events["n_nonlocal_bins"].median(),
            "quantity",
            "int",
            "median non-local bins per event, all events (event table)",
            "bins",
        ),
        make(
            "event_bins_ratio",
            bins[True] / bins[False],
            "quantity",
            "sig2",
            "ratio of those medians",
            "ratio",
        ),
    ]


def estimator_values(
    summary: pd.DataFrame, bootstrap: pd.DataFrame, source: str
) -> list[Value]:
    """Claims 19a, 19f, 19g: the three-estimator comparison on one shared event
    set. Coefficients are animal-balanced medians as the comparison writes them;
    intervals are its animal bootstrap's 2.5 and 97.5 percentiles."""
    make = _value_factory(
        source, ("19a", "19f", "19g"), ("supplement", "figure: content_estimators")
    )
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
                        f"CS5_armvalue against C2_base_leaf, {ESTIMATORS[est_word]}: animal-balanced median per-session AIC(rung) - AIC(parent); negative = rung better",
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
                claims_=("19a",),
            )
        )
    return values


def _session_aggregate(fits: pd.DataFrame, term: str) -> dict[str, float]:
    """Session median of ``coef_<term>`` over the sessions the per-term gate
    keeps, with the animals positive, the median session SE, the sessions'
    signs, the share significant and the signed-rank p."""
    gate = gate_term_sessions(fits, term)
    kept = fits[gate.kept]
    coef = kept[f"coef_{term}"]
    se, _ = session_se(kept, term)
    per_animal = kept.groupby("animal")[f"coef_{term}"].median()
    _require(len(per_animal) == 5, f"{term}: {len(per_animal)} animals")
    return {
        "median": float(coef.median()),
        "animal_median": float(per_animal.median()),
        "sessions": int(len(coef)),
        "animals_positive": int((per_animal > 0).sum()),
        "se": float(se.median()) if se is not None else float("nan"),
        "positive": int((coef > 0).sum()),
        "negative": int((coef < 0).sum()),
        "significant": float(
            pct_significant_sessions(coef, se, kept.get(f"pval_{term}"))
        ),
        "p": float(wilcoxon(coef).pvalue),
    }


def leaf_values(leaf_dir: Path, source: str) -> list[Value]:
    """Claim 19g: the leaf-only arm ladder's uncertainty and value rungs."""
    make = _value_factory(
        source, ("19g",), ("supplement", "figure: uncertainty_content")
    )
    values = []
    for word, model, term in (
        ("leaf_uncertainty", "Mu", "reward_belief_variance_candidate_arm_preoutcome_z"),
        ("leaf_value", "Mv", "reward_belief_mean_candidate_arm_preoutcome"),
    ):
        agg = _session_aggregate(pd.read_pickle(leaf_dir / f"{model}.pkl"), term)
        what = f"leaf ladder {model} {term}"
        values += [
            make(
                word,
                agg["median"],
                "quantity",
                "sig2s",
                f"{what}: session median coefficient",
                "log weight",
            ),
            make(
                f"{word}_sessions",
                agg["sessions"],
                "count",
                "count",
                f"{what}: sessions kept",
            ),
            make(
                f"{word}_animals_positive",
                agg["animals_positive"],
                "count",
                "count",
                f"{what}: animals positive",
            ),
            make(
                f"{word}_se",
                agg["se"],
                "quantity",
                "sig2",
                f"{what}: median event-clustered session SE",
                "log weight",
            ),
        ]
    comparison = pd.read_pickle(leaf_dir / "model_comparison.pkl")
    for word, (parent, child) in LEAF_STEPS.items():
        (row,) = comparison[
            (comparison["parent"] == parent) & (comparison["child"] == child)
        ].itertuples()
        what = f"leaf ladder {parent} to {child}"
        _require(row.n_animals == 5, f"{what}: not 5 animals")
        values += [
            make(
                f"leaf_step_{word}",
                row.delta_aic_median,
                "quantity",
                "sig2s",
                f"{what}: median per-session AIC(parent) - AIC(child); positive = child better",
                "AIC",
            ),
            make(
                f"leaf_step_{word}_clear",
                row.child_better_aic,
                "count",
                "count",
                f"{what}: sessions in which the child's AIC is lower by more than 2",
            ),
            make(
                f"leaf_step_{word}_sessions",
                row.n_sessions,
                "count",
                "count",
                f"{what}: sessions compared",
            ),
            make(
                f"leaf_step_{word}_animals",
                row.n_animals_child_better,
                "count",
                "count",
                f"{what}: animals whose median step favours the child",
            ),
        ]
    return values


# --- 3c: choice and stay/leave ----------------------------------------------------


def choice_values(fits: dict, source: str) -> list[Value]:
    """Claim 19d: replay fraction in the conditional-logit choice block."""
    make = _value_factory(source, ("19d",), ("supplement",))
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
        agg = _session_aggregate(fits[block][model], term)
        what = f"{model} {term}"
        values += [
            make(
                word,
                agg["median"],
                "quantity",
                "sig2s",
                f"{what}: session median coefficient (gated)",
                "log odds",
                pending=CHOICE_RULE_PENDING,
            ),
            make(
                f"{word}_p",
                agg["p"],
                "quantity",
                "pval",
                f"{what}: signed-rank p across sessions",
                pending=CHOICE_RULE_PENDING,
            ),
            make(
                f"{word}_sessions",
                agg["sessions"],
                "count",
                "count",
                f"{what}: sessions kept",
                pending=CHOICE_RULE_PENDING,
            ),
            make(
                f"{word}_positive_sessions",
                agg["positive"],
                "count",
                "count",
                f"{what}: sessions with a positive coefficient",
                pending=CHOICE_RULE_PENDING,
            ),
            make(
                f"{word}_negative_sessions",
                agg["negative"],
                "count",
                "count",
                f"{what}: sessions with a negative coefficient",
                pending=CHOICE_RULE_PENDING,
            ),
            make(
                f"{word}_animals_positive",
                agg["animals_positive"],
                "count",
                "count",
                f"{what}: animals with a positive median",
                pending=CHOICE_RULE_PENDING,
            ),
            make(
                f"{word}_significant_fraction",
                agg["significant"],
                "fraction",
                "pct",
                f"{what}: share of sessions individually significant",
                pending=CHOICE_RULE_PENDING,
            ),
        ]
    return values


def stay_leave_values(
    bundles: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    stops: pd.DataFrame,
    source: str,
) -> list[Value]:
    """Claim 19e: stay/leave against content, per stratum.

    The per-session Wilcoxon is ``learning.cross_session_role_summary`` (the
    runner's own); the pooled GEE's q is BH across converged roles within a
    stratum, as ``run_stay_leave.py`` logs it.
    """
    make = _value_factory(source, ("19e",), ("supplement",))
    signrank, gee = [], []
    values = []
    for stratum, (logistic, pooled) in bundles.items():
        summary = cross_session_role_summary(logistic)
        signrank += [
            (q, stratum, role)
            for q, role in zip(summary["p_fdr"], summary["role"], strict=True)
            if np.isfinite(q)
        ]
        converged = pooled[pooled["status"] == "ok"]
        q = multipletests(converged["p_inter"].fillna(1.0).to_numpy(), method="fdr_bh")[
            1
        ]
        gee += [
            (qi, stratum, role) for qi, role in zip(q, converged["role"], strict=True)
        ]
        values += [
            make(
                f"stay_converged_{stratum}",
                (logistic["status"] == "ok").sum(),
                "count",
                "count",
                f"stratum {stratum}: converged session-by-role logistic fits",
            ),
            make(
                f"stay_fits_{stratum}",
                len(logistic),
                "count",
                "count",
                f"stratum {stratum}: session-by-role logistic fits attempted",
            ),
        ]
    q_min, stratum_min, role_min = min(signrank)
    g_min, g_stratum, g_role = min(gee)
    occupancy = stops["occupancy_state"]
    anchored = occupancy.isin(["at_well", "mixed"])
    values += [
        make(
            "stay_min_q_signrank",
            q_min,
            "quantity",
            "pval",
            f"smallest FDR-adjusted per-session Wilcoxon p over roles and strata (role {role_min}, stratum {stratum_min})",
        ),
        make(
            "stay_min_q_gee",
            g_min,
            "quantity",
            "pval",
            f"smallest FDR-adjusted pooled-GEE p over roles and strata (role {g_role}, stratum {g_stratum})",
        ),
        make(
            "stops_total",
            len(stops),
            "count",
            "count",
            "stops in the canonical stop table",
            claims_=("19e", "19c"),
        ),
        make(
            "stops_well_anchored",
            anchored.sum(),
            "count",
            "count",
            "stops with occupancy_state at_well or mixed (the shared cohort)",
            claims_=("19e", "19c"),
        ),
        make(
            "stops_dropped",
            (~anchored).sum(),
            "count",
            "count",
            "stops the well-anchored cohort leaves out",
        ),
        make(
            "stops_dropped_with_stay",
            stops.loc[~anchored, "is_stay"].notna().sum(),
            "count",
            "count",
            "of those, stops carrying an is_stay",
        ),
        make(
            "stops_mixed",
            (occupancy == "mixed").sum(),
            "count",
            "count",
            "mixed stops (every one without an is_stay)",
        ),
        make(
            "stops_mixed_with_stay",
            stops.loc[occupancy == "mixed", "is_stay"].notna().sum(),
            "count",
            "count",
            "mixed stops carrying an is_stay",
        ),
        make(
            "stops_at_well_with_stay",
            stops.loc[occupancy == "at_well", "is_stay"].notna().sum(),
            "count",
            "count",
            "at_well stops carrying an is_stay (Stage 6's population)",
        ),
    ]
    return values


# --- the SWR-power mark -----------------------------------------------------------


def power_values(coefs: pd.DataFrame, source: str) -> list[Value]:
    """The SWR-coupling mark: per-session OLS of event ripple power on
    standardized covariates, summarized as the fitting script prints it (median,
    share positive, animals whose median is positive, signed-rank p)."""
    pending = NO_SIDECAR_PENDING.format(name="the SWR-power model")
    make = _value_factory(
        source, ("synthesis Layer 2",), ("docs: replay_rate_content_swr_synthesis",)
    )
    values = [
        make(
            "power_sessions",
            len(coefs),
            "count",
            "count",
            "sessions in the SWR-power model",
            pending=pending,
        )
    ]
    for word, column in POWER_COVARIATES.items():
        s = coefs[column].dropna()
        per_animal = coefs.groupby("animal")[column].median()
        values += [
            make(
                f"power_{word}",
                s.median(),
                "quantity",
                "sig2s",
                f"{column}: median per-session coefficient on ripple_power_z",
                "SD of power per SD",
                pending=pending,
            ),
            make(
                f"power_{word}_animals_positive",
                (per_animal > 0).sum(),
                "count",
                "count",
                f"{column}: animals with a positive median",
                pending=pending,
            ),
            make(
                f"power_{word}_p",
                wilcoxon(s).pvalue,
                "quantity",
                "pval",
                f"{column}: signed-rank p",
                pending=pending,
            ),
        ]
    return values


# --- the summary -------------------------------------------------------------------


def _check_fresh(artifact: Path) -> None:
    """Refuse a fit whose provenance sidecar says a table it read has changed."""
    recorded = json.loads(fit_provenance_path(artifact).read_text())["inputs"]
    inputs = {name: table_path(name, replay_content_dir()) for name in recorded}
    stale = check_fit_provenance(artifact, inputs)
    if stale:
        raise ValueError(f"{artifact}: stale against {stale}; refit before summarizing")


def rel(path: Path) -> str:
    """``path`` relative to the repository when it is under it."""
    return (
        str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path)
    )


def summarize(inputs: Part3Inputs, produced_by: str, commit: str) -> Summary:
    """Part 3's summary from ``inputs``."""
    summary = Summary(part=3, produced_by=produced_by, commit=commit)

    def record(*paths: Path) -> None:
        summary.inputs.extend(file_record(p, ROOT_DIR) for p in paths)

    rate_fits = (
        inputs.per_bin,
        inputs.per_bin_all_stops,
        inputs.per_bin_in_transit,
        inputs.rate_contrast,
        inputs.per_stop,
    )
    for artifact in rate_fits:
        _check_fresh(artifact)
    summary.add(
        *rate_contrast_separation_values(
            pd.read_pickle(inputs.rate_contrast),
            f"{rel(inputs.rate_contrast)} (rate.swr_contrast.at_well)",
        )
    )
    in_transit = pd.read_pickle(inputs.per_bin_in_transit)
    make_transit = _value_factory(
        f"{rel(inputs.per_bin_in_transit)} (rate.per_bin.in_transit)",
        ("19c",),
        ("supplement",),
    )
    summary.add(
        make_transit(
            "rate_intransit_sessions",
            in_transit.groupby(SC).ngroups,
            "count",
            "count",
            "sessions with any per-bin fit on the in-transit stops (the sample gates "
            "leave the rest out)",
        ),
        *(
            make_transit(
                f"rate_intransit_sessions_{word}",
                in_transit[in_transit["outcome"] == outcome].groupby(SC).ngroups,
                "count",
                "count",
                f"{label}: sessions with a per-bin fit on the in-transit stops",
            )
            for word, (outcome, label) in RATE_POPULATIONS.items()
        ),
    )
    summary.add(*cohort_audit_values(inputs))
    record(inputs.per_bin_audit, inputs.per_stop_audit)
    per_bin = pd.read_pickle(inputs.per_bin)
    per_bin_all = pd.read_pickle(inputs.per_bin_all_stops)
    summary.add(
        *per_bin_values(
            per_bin,
            per_bin_all,
            f"{rel(inputs.per_bin)} (rate.per_bin.at_well) and {rel(inputs.per_bin_all_stops)} (rate.per_bin.all_stops)",
        )
    )
    summary.add(
        *per_bin_detail_values(
            per_bin,
            per_bin_all,
            in_transit,
            f"{rel(inputs.per_bin)}, {rel(inputs.per_bin_all_stops)} and "
            f"{rel(inputs.per_bin_in_transit)} (rate.per_bin.*)",
        )
    )
    summary.add(
        *per_stop_values(
            pd.read_pickle(inputs.per_stop),
            f"{rel(inputs.per_stop)} (rate.per_stop.at_well)",
        )
    )
    record(*rate_fits)

    audited, sources = {}, {}
    for suffix in sorted({spec[0] for spec in CONTENT_SPECS.values()}):
        path = inputs.content_contrast_dir / f"swr_contrast_content_audited{suffix}.pkl"
        _check_fresh(path)
        audited[suffix] = pd.read_pickle(path)
        sources[suffix] = f"{rel(path)} (audited aggregate)"
        record(path)
    content = content_contrast_values(audited, sources)
    summary.add(*content)
    events = pd.read_parquet(inputs.event_table, columns=["is_swr", "n_nonlocal_bins"])
    summary.add(
        *breadth_derived_values(
            {v.key: v for v in content},
            events,
            f"{sources['_dm']}, {sources['_dm_len']} and {rel(inputs.event_table)}",
        )
    )
    record(inputs.event_table)

    _check_fresh(
        inputs.estimator_summary.with_name("content_estimator_comparison.parquet")
    )
    summary.add(
        *estimator_values(
            pd.read_csv(inputs.estimator_summary),
            pd.read_csv(inputs.estimator_bootstrap),
            f"{rel(inputs.estimator_summary.parent)} (content.estimator_comparison)",
        )
    )
    record(inputs.estimator_summary, inputs.estimator_bootstrap)

    for name in ("Mu", "Mv", "model_comparison"):
        _check_fresh(inputs.leaf_dir / f"{name}.pkl")
    summary.add(
        *leaf_values(
            inputs.leaf_dir, f"{rel(inputs.leaf_dir)} (content.arm_ladder.leaf_arms)"
        )
    )
    record(*(inputs.leaf_dir / f"{n}.pkl" for n in ("Mu", "Mv", "model_comparison")))

    summary.add(
        *choice_values(
            pd.read_pickle(inputs.choice_fits),
            f"{rel(inputs.choice_fits)} (behavior.event_source_suite.non_local)",
        )
    )
    record(inputs.choice_fits)

    bundles = {}
    for stratum, sfx in STAY_STRATA.items():
        logistic = inputs.stay_leave_dir / f"stage6_is_stay_logistic{sfx}.pkl"
        pooled = inputs.stay_leave_dir / f"stage6_is_stay_pooled_gee{sfx}.pkl"
        for path in (logistic, pooled):
            _check_fresh(path)
        bundles[stratum] = (pd.read_pickle(logistic), pd.read_pickle(pooled))
        record(logistic, pooled)
    stops = pd.read_parquet(inputs.stop_table)
    summary.add(
        *stay_leave_values(
            bundles,
            stops,
            f"{rel(inputs.stay_leave_dir)}/stage6_is_stay_*.pkl (behavior.stay_leave.at_well*) and {rel(inputs.stop_table)}",
        )
    )
    record(inputs.stop_table)
    onsets = pd.read_parquet(
        inputs.onset_table, columns=["event_type", "t_since_stop_onset"]
    )
    summary.add(
        *stop_table_values(
            stops, onsets, f"{rel(inputs.stop_table)} and {rel(inputs.onset_table)}"
        )
    )
    record(inputs.onset_table)
    summary.add(
        *trunk_values(
            pd.read_pickle(inputs.trunk_per_event),
            f"{rel(inputs.trunk_per_event)} (exploratory)",
        )
    )
    record(inputs.trunk_per_event)

    summary.add(
        *power_values(
            pd.read_pickle(inputs.swr_power),
            f"{rel(inputs.swr_power)} (coupling.ripple_power)",
        )
    )
    record(inputs.swr_power)
    return summary
