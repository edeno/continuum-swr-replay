"""Part 4's reported values: how mPFC units respond to SWR and non-SWR non-local events.

Every value is read from an output an analysis already wrote; nothing is
refitted here. The inputs (:class:`Part4Inputs`):

- the at-well mPFC unit bundles (``Processed-Data/{epoch}_mPFC_modulation.pkl``
  and their ``*.provenance.json`` sidecars, from
  ``scripts/pfc/run_pfc_analysis_bandit.py``), read through Figure 4's loader
  (:func:`~continuum_swr_replay.analysis.pfc.load_population.load_population_data`,
  firing-rate gate 1 Hz) for every per-unit value;
- the PFC cohort record the runner writes (``pfc_cohort_mPFC.json``);
- the canonical event and stop tables, only to check that the bundles were fit
  on the current ones;
- the superseded maximum-likelihood tree (``_pre_ridge_backup_20260915``), for
  the one comparison the prior is justified by (marked pending: it was fit on
  earlier tables).

The derived values follow the rules of the scripts that first reported them:
Figure 4 (``scripts/figures/generate_figure04.py``) for the unit fractions,
``scripts/pfc/compare_pfc_cohorts.py`` for the cohort, fit-count and
prior-binding values, and ``scripts/pfc/measure_pfc_null_geometry.py`` for the
null's geometry. Unit fractions are descriptive and pooled over sessions and
animals; they are not population inference.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control

from continuum_swr_replay.analysis.pfc.load_population import (
    HELDOUT_P_COLUMNS,
    MATCHED_NULL_OBSERVED_COLUMNS,
    load_population_data,
    significant_fraction,
)
from continuum_swr_replay.analysis.pfc.modulation import REPORT_HALF_WINDOW_S
from continuum_swr_replay.analysis.pfc.timing import (
    peak_and_onset,
    population_timing,
    sign_aligned_mean,
)
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
from continuum_swr_replay.paths import (
    ROOT_DIR,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.values import (
    Summary,
    Value,
    file_record,
    files_record,
)

STEM = "_mPFC_modulation"
"""The canonical (at-well, sigma 3) bundle name; other cohorts and priors carry
a suffix after it."""

ALPHA = 0.05
"""Figure 4's raw empirical-p threshold."""

MIN_FIRING_RATE_HZ = 1.0
"""Figure 4's firing-rate gate."""

ML_TREE = "_pre_ridge_backup_20260915"
ML_PENDING = (
    "read from the superseded maximum-likelihood tree (Processed-Data/"
    "_pre_ridge_backup_20260915), fit on the tables before the R11 re-decode; "
    "it is a comparison of estimators, not refit on the current tables"
)

#: The unit tests Figure 4 reports, by the word their keys use.
CLASSES = {
    "swr": "SWR",
    "nonlocal": "non_local",
    "non_swr_replay": "non_local_no_SWR",
    "swr_replay": "non_local_SWR",
    "engagement": "engagement",
}
CLASS_LABELS = {
    "SWR": "SWR",
    "non_local": "all non-local events",
    "non_local_no_SWR": "non-SWR replay",
    "non_local_SWR": "SWR-coupled replay",
    "engagement": (
        "the engagement test (matched_control + non_local_no_SWR vs. matched_control)"
    ),
}

#: Animals by the word their keys use (a key holds no digits).
ANIMAL_WORDS = {
    "chimi": "chimi",
    "j16": "jsixteen",
    "peanut": "peanut",
    "senor": "senor",
    "wilbur": "wilbur",
}

#: The curves whose prior binding the audit counts (compare_pfc_cohorts.CURVE_NAMES).
CURVES = ("SWR", "no_SWR", "non_local_SWR", "non_local_no_SWR", "matched_control")

#: Figure 4d's best-model bars.
BEST_MODELS = {
    "constant": "constant",
    "swr": "SWR",
    "nonlocal": "non_local",
    "joint": "SWR + non_local_no_SWR",
}


TIMING_CALIBRATION = "pfc_response_followups/timing_calibration.json"
"""The timing calibration ``scripts/pfc/calibrate_pfc_kernel_timing.py`` writes,
relative to the processed-data root."""

#: The curves the timing values are read from, by the word their keys use: the
#: class (``HELDOUT_P_COLUMNS`` key) a unit must be called for, and the curve
#: ``load_population_data`` returns for it.
TIMING_CLASSES = {
    "swr": ("SWR", "swr"),
    "swr_replay": ("non_local_SWR", "non_local_swr"),
    "non_swr_replay": ("non_local_no_SWR", "non_local_no_swr"),
}

#: Session resamples behind the population timing intervals, and their seed.
TIMING_BOOTSTRAP_DRAWS = 1_000
TIMING_BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Part4Inputs:
    """The files Part 4 reads."""

    bundle_dir: Path
    pfc_cohort: Path
    event_table: Path
    stop_table: Path
    ml_tree: Path
    timing_calibration: Path


def default_inputs() -> Part4Inputs:
    data = processed_data_dir()
    return Part4Inputs(
        bundle_dir=data,
        pfc_cohort=data / "pfc_cohort_mPFC.json",
        event_table=table_path("event_table", replay_content_dir()),
        stop_table=table_path("stop_table", replay_content_dir()),
        ml_tree=data / ML_TREE,
        timing_calibration=data / TIMING_CALIBRATION,
    )


def _value_factory(
    source: str,
    claims: tuple[str, ...],
    used_in: tuple[str, ...],
    pending: str | None = None,
):
    """A :class:`Value` builder with the shared source, claims, use and pending
    note filled in."""

    def make(
        key: str,
        value: float | int,
        kind: str,
        style: str,
        description: str,
        unit: str = "",
        *,
        used_in_: tuple[str, ...] | None = None,
    ) -> Value:
        return Value(
            key=key,
            value=(
                int(value)
                if kind == "count" and float(value).is_integer()
                else value
                if kind == "count"
                else float(value)
            ),
            kind=kind,
            style=style,
            unit=unit,
            description=description,
            claims=claims,
            used_in=used_in_ or used_in,
            source=source,
            pending=pending,
        )

    return make


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sidecars(bundle_dir: Path) -> list[dict[str, Any]]:
    """Every canonical bundle's provenance sidecar, in session order.

    Raises
    ------
    ValueError
        A bundle has no sidecar or a sidecar no bundle.
    """
    bundles = {
        p.name.removesuffix(f"{STEM}.pkl") for p in bundle_dir.glob(f"*{STEM}.pkl")
    }
    paths = sorted(bundle_dir.glob(f"*{STEM}.provenance.json"))
    described = {p.name.removesuffix(f"{STEM}.provenance.json") for p in paths}
    if not bundles or bundles != described:
        raise ValueError(
            f"bundles and sidecars in {bundle_dir} differ: "
            f"{sorted(bundles ^ described)}"
        )
    return [json.loads(p.read_text()) for p in paths]


def staleness(
    sidecars: list[dict[str, Any]], event_table: Path, stop_table: Path
) -> str | None:
    """Why the bundles are stale against the current tables, or None.

    A bundle records the SHA-256 of the canonical event table and the stop
    table it was fit on; a bundle fit on other tables describes other events.
    """
    current = {
        "canonical_event_table_sha256": _sha256(event_table),
        "stop_table_sha256": _sha256(stop_table),
    }
    stale = {
        name: sum(s.get(name) != sha for s in sidecars) for name, sha in current.items()
    }
    if not any(stale.values()):
        return None
    return (
        "the mPFC bundles were fit on other tables than the current ones "
        f"({', '.join(f'{n} differs in {k} bundles' for n, k in stale.items() if k)}); "
        "refit before citing"
    )


def _uniform(sidecars: list[dict[str, Any]], name: str) -> Any:
    """The one value every sidecar records for ``name``."""
    values = {json.dumps(s.get(name), sort_keys=True) for s in sidecars}
    if len(values) != 1:
        raise ValueError(f"the bundles disagree on {name}: {sorted(values)}")
    return sidecars[0].get(name)


# --- the cohort -----------------------------------------------------------------


def cohort_values(
    sidecars: list[dict[str, Any]],
    population: pd.DataFrame,
    cohort: dict[str, Any],
    make: Callable[..., Value],
) -> list[Value]:
    """Which sessions, units, bins and events the unit GLM was fit on.

    Bins and events are summed over sessions, before and after the at-well
    stop restriction, as ``compare_pfc_cohorts.session_cohort_table`` totals
    them.
    """
    if len(sidecars) != cohort["n_pfc_cohort"]:
        raise ValueError(
            f"{len(sidecars)} bundles, but the PFC cohort holds "
            f"{cohort['n_pfc_cohort']} epochs"
        )
    if _uniform(sidecars, "stop_cohort") != "at_well":
        raise ValueError("the canonical bundles are not the at-well cohort")
    counts = pd.DataFrame([s["cohort_counts"] for s in sidecars])
    totals = counts[
        [
            "n_bins_fit",
            "n_bins_immobile",
            "n_nonlocal_after",
            "n_nonlocal_before",
            "n_ripple_after",
            "n_ripple_before",
        ]
    ].sum()
    swr_share = counts["n_nonlocal_swr"] / counts["n_nonlocal_after"]
    return [
        make(
            "sessions",
            len(sidecars),
            "count",
            "count",
            "epochs with an mPFC unit bundle (the PFC cohort: analysed epochs with "
            "curated cortical spike sorting)",
        ),
        make(
            "analysed_epochs",
            cohort["n_analysed"],
            "count",
            "count",
            "analysed epochs the PFC cohort is drawn from",
        ),
        make(
            "animals",
            population["animal"].nunique(),
            "count",
            "count",
            "animals in the mPFC population",
        ),
        make(
            "units",
            len(population),
            "count",
            "count",
            "mPFC units in Figure 4's population: a usable fit, an immobility firing "
            f"rate of at least {MIN_FIRING_RATE_HZ:g} Hz on the fitted bins, and "
            "finite modulation curves",
            used_in_=("paper", "figure: 4c", "figure: 4d"),
        ),
        make(
            "bins_fit",
            totals["n_bins_fit"],
            "count",
            "count",
            "immobile 2 ms bins inside a kept stop, the bins the GLM is fit on, "
            "summed over sessions",
        ),
        make(
            "bins_immobile",
            totals["n_bins_immobile"],
            "count",
            "count",
            "immobile 2 ms bins before the stop restriction, summed over sessions",
        ),
        make(
            "bins_fit_fraction",
            totals["n_bins_fit"] / totals["n_bins_immobile"],
            "fraction",
            "pct",
            "share of the immobile bins the stop restriction keeps, pooled",
        ),
        make(
            "nonlocal_events_kept",
            totals["n_nonlocal_after"],
            "count",
            "count",
            "non-local events whose onset is in a kept stop, summed over sessions",
        ),
        make(
            "nonlocal_events_before",
            totals["n_nonlocal_before"],
            "count",
            "count",
            "the session's non-local events before the stop restriction, summed",
        ),
        make(
            "nonlocal_events_kept_fraction",
            totals["n_nonlocal_after"] / totals["n_nonlocal_before"],
            "fraction",
            "pct",
            "share of the non-local events the stop restriction keeps, pooled",
        ),
        make(
            "swrs_kept_fraction",
            totals["n_ripple_after"] / totals["n_ripple_before"],
            "fraction",
            "pct",
            "share of the SWRs the stop restriction keeps, pooled",
        ),
        make(
            "swr_coupled_share_session_median",
            swr_share.median(),
            "fraction",
            "dec2",
            "median over sessions of the share of kept non-local events that are "
            "SWR-coupled (the canonical event table's is_swr)",
        ),
    ]


# --- the estimator and its fits ------------------------------------------------


def estimator_values(
    sidecars: list[dict[str, Any]], make: Callable[..., Value]
) -> list[Value]:
    """The prior, folds and null the bundles state, and what the fits did.

    Raises
    ------
    ValueError
        The bundles disagree on a setting, or were not fit with the ridge
        estimator and the held-out inference.
    """
    if (_uniform(sidecars, "estimator"), _uniform(sidecars, "inference")) != (
        "ridge_poisson_glm",
        "blocked_heldout_deviance",
    ):
        raise ValueError("the bundles are not the ridge, held-out bundles")
    if _uniform(sidecars, "empirical_pvalue_method") != "plus_one":
        raise ValueError("the bundles' p-values are not plus-one corrected")
    prior = _uniform(sidecars, "implied_prior_sd")
    draws = _uniform(sidecars, "n_control_draws")
    fits = pd.DataFrame(
        [
            {
                name: s[name]
                for name in (
                    "n_model_fits",
                    "n_nonconverged_model_fits",
                    "n_neurons_with_nonconverged_model",
                )
            }
            for s in sidecars
        ]
    ).sum()
    return [
        make(
            "penalty_sigma",
            _uniform(sidecars, "penalty_sigma_log_rate"),
            "quantity",
            "int",
            "prior standard deviation of an event-basis coefficient",
            "log rate",
        ),
        make(
            "prior_sd_per_lag",
            prior["per_lag"],
            "quantity",
            "dec2",
            "the prior's implied standard deviation of a curve at one lag",
            "log rate",
        ),
        make(
            "prior_multiplier_per_lag_low",
            prior["per_lag_multiplier_low"],
            "quantity",
            "sig2",
            "lower end of the rate multiplier that per-lag sd allows at 95 %",
            "rate multiplier",
        ),
        make(
            "prior_multiplier_per_lag_high",
            prior["per_lag_multiplier_high"],
            "quantity",
            "sig2",
            "upper end of that multiplier",
            "rate multiplier",
        ),
        make(
            "prior_sd_sustained",
            prior["sustained"],
            "quantity",
            "dec2",
            "the prior's implied standard deviation of a curve sustained over the "
            "reported window",
            "log rate",
        ),
        make(
            "prior_multiplier_sustained_low",
            prior["sustained_multiplier_low"],
            "quantity",
            "sig2",
            "lower end of the rate multiplier that sustained sd allows at 95 %",
            "rate multiplier",
        ),
        make(
            "prior_multiplier_sustained_high",
            prior["sustained_multiplier_high"],
            "quantity",
            "sig2",
            "upper end of that multiplier",
            "rate multiplier",
        ),
        make(
            "folds",
            _uniform(sidecars, "n_folds"),
            "count",
            "count",
            "blocked folds of whole immobility runs the held-out deviance is scored on",
        ),
        make(
            "null_draws",
            draws,
            "count",
            "count",
            "matched-control draws in each unit's null",
        ),
        make(
            "min_pvalue_denominator",
            draws + 1,
            "count",
            "count",
            "R + 1: the smallest plus-one empirical p-value is one over it",
        ),
        make(
            "min_pvalue",
            1 / (draws + 1),
            "fraction",
            "sig2",
            "the smallest plus-one empirical p-value, 1/(R + 1)",
        ),
        make(
            "model_fits",
            fits["n_model_fits"],
            "count",
            "count",
            "neuron x model fits in the bundles",
        ),
        make(
            "nonconverged_model_fits",
            fits["n_nonconverged_model_fits"],
            "count",
            "count",
            "of those, fits that ran to maxiter without converging",
        ),
        make(
            "nonconverged_neurons",
            fits["n_neurons_with_nonconverged_model"],
            "count",
            "count",
            "neurons carrying at least one non-converged fit",
        ),
        make(
            "nonconverged_percent",
            100 * fits["n_nonconverged_model_fits"] / fits["n_model_fits"],
            "quantity",
            "sig2",
            "non-converged fits as a percent of model fits",
            "%",
        ),
    ]


def ml_comparator_values(ml_tree: Path, make: Callable[..., Value]) -> list[Value]:
    """Non-convergence in the maximum-likelihood tree the prior replaced."""
    sidecars = read_sidecars(ml_tree)
    if _uniform(sidecars, "estimator") != "poisson_glm":
        raise ValueError(f"{ml_tree} is not the maximum-likelihood tree")
    fits = sum(s["n_model_fits"] for s in sidecars)
    nonconverged = sum(s["n_nonconverged_model_fits"] for s in sidecars)
    return [
        make(
            "ml_nonconverged_percent",
            100 * nonconverged / fits,
            "quantity",
            "sig2",
            "non-converged fits as a percent of model fits in the maximum-likelihood "
            f"tree ({nonconverged} of {fits}, {len(sidecars)} sessions)",
            "%",
        ),
        make(
            "ml_sessions",
            len(sidecars),
            "count",
            "count",
            "sessions in the maximum-likelihood tree",
        ),
    ]


def prior_binding_values(
    population: pd.DataFrame, make: Callable[..., Value]
) -> list[Value]:
    """How often the prior moves a curve's peak, per curve.

    The rule of ``compare_pfc_cohorts.shrinkage_table``: among the well-posed
    units of a curve (unpenalised fit converged and no separated column), the
    share whose penalised peak differs from the unpenalised one by more than
    that peak's standard error (the bundle's ``prior_binds``). The values are
    the lowest and highest share over the five curves.
    """
    shares = {}
    for curve in CURVES:
        converged = population[f"{curve}: mle_converged"].fillna(False).astype(bool)
        separated = population[f"{curve}: mle_separated"].fillna(False).astype(bool)
        well_posed = converged & ~separated
        binds = population[f"{curve}: prior_binds"].fillna(False).astype(bool)
        shares[curve] = 100 * (binds & well_posed).sum() / well_posed.sum()
    shares = pd.Series(shares)
    return [
        make(
            "prior_binds_percent_min",
            shares.min(),
            "quantity",
            "sig2",
            f"lowest per-curve percent of well-posed units the prior binds on "
            f"({shares.idxmin()})",
            "%",
        ),
        make(
            "prior_binds_percent_max",
            shares.max(),
            "quantity",
            "sig2",
            f"highest per-curve percent of well-posed units the prior binds on "
            f"({shares.idxmax()})",
            "%",
        ),
    ]


# --- the unit fractions (Figure 4d) -----------------------------------------------


def called_fraction_values(
    population: pd.DataFrame, make: Callable[..., Value]
) -> list[Value]:
    """Figure 4d: the share of units at raw empirical p < 0.05, per test.

    The denominator is the units tested for that class
    (:func:`~continuum_swr_replay.analysis.pfc.load_population.significant_fraction`).
    Engagement is also given per animal.
    """
    fractions = significant_fraction(population, alpha=ALPHA)
    values = []
    for word, name in CLASSES.items():
        n_called, n_tested = fractions[name]
        label = CLASS_LABELS[name]
        values += [
            make(
                f"{word}_called_fraction",
                n_called / n_tested,
                "fraction",
                "pct",
                f"share of tested units at raw empirical p < {ALPHA:g} for {label}, "
                "pooled (descriptive)",
            ),
            make(
                f"{word}_units_called",
                n_called,
                "count",
                "count",
                f"units at raw empirical p < {ALPHA:g} for {label}",
            ),
            make(
                f"{word}_units_tested",
                n_tested,
                "count",
                "count",
                f"units with a finite p-value for {label}",
            ),
        ]
    for animal, frame in population.groupby("animal"):
        n_called, n_tested = significant_fraction(frame, alpha=ALPHA)["engagement"]
        values.append(
            make(
                f"engagement_called_fraction_{ANIMAL_WORDS[animal]}",
                n_called / n_tested,
                "fraction",
                "pct",
                f"share of {animal}'s tested units at raw empirical p < {ALPHA:g} for "
                "the engagement test",
            )
        )
    pvalues = population[HELDOUT_P_COLUMNS["engagement"]].to_numpy(dtype=float)
    pvalues = pvalues[np.isfinite(pvalues)]
    if len(pvalues) != fractions["engagement"][1]:
        raise ValueError("the engagement p-values are not the units it was tested on")
    values.append(
        make(
            "engagement_bh_fraction",
            (false_discovery_control(pvalues, method="bh") < ALPHA).mean(),
            "fraction",
            "pct",
            "share of the tested units with a Benjamini-Hochberg q below "
            f"{ALPHA:g} for the engagement test, BH applied once over the pooled "
            "plus-one empirical p-values of every tested unit "
            "(scipy.stats.false_discovery_control). For reference only: the units "
            "are not independent, and the p-values sit on a 1/(R + 1) grid",
            used_in_=("paper",),
        )
    )
    counts = population["min_heldout_model"].value_counts()
    for word, model in BEST_MODELS.items():
        values.append(
            make(
                f"best_model_{word}_fraction",
                counts.get(model, 0) / len(population),
                "fraction",
                "pct",
                f"share of units whose best model by held-out deviance is {model}",
                used_in_=("figure: 4d",),
            )
        )
    return values


# --- the null's geometry -----------------------------------------------------------


def null_geometry_values(
    population: pd.DataFrame,
    sidecars: list[dict[str, Any]],
    make: Callable[..., Value],
) -> list[Value]:
    """Three caveats on the empirical p-values, by the rules of
    ``scripts/pfc/measure_pfc_null_geometry.py``.

    - Units called at raw p < 0.05 whose own observed held-out improvement (on
      the matchable subset) is not positive.
    - How much of the engagement null's windows land on the shipped control:
      each session's median over draws, then the median and range over
      sessions.
    - Events with no legal within-bout control, which enter neither side:
      per class, summed over sessions.
    """
    shares = {}
    for name, column in HELDOUT_P_COLUMNS.items():
        called = population[column] < ALPHA
        observed = population[MATCHED_NULL_OBSERVED_COLUMNS[name]]
        shares[name] = ((called & (observed <= 0)).sum(), called.sum())
    n_flat, n_called = shares.pop("engagement")
    class_shares = pd.Series({name: a / b for name, (a, b) in shares.items()})

    overlap = pd.Series([s["engagement_null_overlap"]["median"] for s in sidecars])

    windows = pd.DataFrame(
        [
            {
                "session": s["artifact"].removesuffix(f"{STEM}.pkl"),
                "event_class": event_class,
                "total": w["n_events_total"],
                "excluded": w["n_events_excluded_unmatchable"],
            }
            for s in sidecars
            for event_class, w in s["null_windows"].items()
        ]
    )
    by_class = windows.groupby("event_class")[["total", "excluded"]].sum()
    excluded = 100 * by_class["excluded"] / by_class["total"]
    with_events = windows[windows["total"] > 0]
    if with_events.empty:
        raise ValueError("no session holds an event of any class")
    worst = with_events.loc[(with_events["excluded"] / with_events["total"]).idxmax()]
    worst_label = f"{worst['session']} / {worst['event_class']}"
    return [
        make(
            "engagement_called_flat_fraction",
            n_flat / n_called,
            "fraction",
            "pct",
            "of the units called by the engagement test, the share whose own "
            "observed held-out improvement is not positive",
        ),
        make(
            "engagement_called_flat",
            n_flat,
            "count",
            "count",
            "units called by the engagement test with no positive observed improvement",
        ),
        make(
            "class_called_flat_fraction_min",
            class_shares.min(),
            "fraction",
            "pctnum",
            f"lowest such share over the four event classes ({class_shares.idxmin()})",
        ),
        make(
            "class_called_flat_fraction_max",
            class_shares.max(),
            "fraction",
            "pct",
            f"highest such share over the four event classes ({class_shares.idxmax()})",
        ),
        make(
            "engagement_null_overlap_median",
            overlap.median(),
            "fraction",
            "pct",
            "median over sessions of each session's median share of an engagement "
            "null draw's windows that land on the shipped matched control",
        ),
        make(
            "engagement_null_overlap_session_min",
            overlap.min(),
            "fraction",
            "pctnum",
            "lowest session median of that share",
        ),
        make(
            "engagement_null_overlap_session_max",
            overlap.max(),
            "fraction",
            "pct",
            "highest session median of that share",
        ),
        make(
            "unmatchable_percent_class_min",
            excluded.min(),
            "quantity",
            "sig2",
            "lowest percent, over event classes, of a class's events with no legal "
            f"within-bout control, summed over sessions ({excluded.idxmin()})",
            "%",
        ),
        make(
            "unmatchable_percent_class_max",
            excluded.max(),
            "quantity",
            "sig2",
            f"highest such percent ({excluded.idxmax()})",
            "%",
        ),
        make(
            "unmatchable_worst_excluded",
            worst["excluded"],
            "count",
            "count",
            f"events with no legal control in the worst session and class ({worst_label})",
        ),
        make(
            "unmatchable_worst_total",
            worst["total"],
            "count",
            "count",
            f"events of that class in that session ({worst_label})",
        ),
    ]


# --- the summary -------------------------------------------------------------------


def overlap_and_effect_values(population: pd.DataFrame, make) -> list[Value]:
    """Do the same mPFC units respond to SWR-coupled and non-SWR replay, how much,
    and do they respond alike?

    - **Overlap:** units called (raw empirical p < 0.05) for SWR-coupled replay
      (``non_local_SWR``), for non-SWR replay (``non_local_no_SWR``), for both or
      neither, among units tested for both; the count of both expected if the two
      calls were independent; their odds ratio, pooled and per animal.
    - **Effect size:** each unit's ``peak_log_modulation`` (the ridge fit's
      largest log-rate change over the reported window) for each class, as a
      median of its absolute value; the matched quiet-rest control's is the
      noise floor. ``exp`` of a log modulation is a rate multiplier.
    - **Alike:** across units, the rank correlation of the two classes' peak
      modulations, and among units called for both the share with the same sign;
      the share of units whose held-out deviance improves when the non-local
      response is split by SWR status (``non_local_SWR + non_local_no_SWR vs.
      non_local``), and whose best model by held-out deviance is that split.
    """
    swr_p = population[HELDOUT_P_COLUMNS["non_local_SWR"]]
    non_p = population[HELDOUT_P_COLUMNS["non_local_no_SWR"]]
    tested = swr_p.notna() & non_p.notna()
    frame = population[tested]
    swr_called = (swr_p[tested] < ALPHA).to_numpy()
    non_called = (non_p[tested] < ALPHA).to_numpy()
    both, swr_only = (
        int((swr_called & non_called).sum()),
        int((swr_called & ~non_called).sum()),
    )
    non_only, neither = (
        int((~swr_called & non_called).sum()),
        int((~swr_called & ~non_called).sum()),
    )
    n = int(tested.sum())

    def odds_ratio(a: int, b: int, c: int, d: int) -> float:
        return (a * d) / (b * c)

    animals = (
        frame["nwb_file_name"].map(lambda name: parse_nwb_filename(name)[0]).to_numpy()
    )
    per_animal = []
    for animal in np.unique(animals):
        sc, nc = swr_called[animals == animal], non_called[animals == animal]
        per_animal.append(
            odds_ratio(
                int((sc & nc).sum()),
                int((sc & ~nc).sum()),
                int((~sc & nc).sum()),
                int((~sc & ~nc).sum()),
            )
        )
    swr_peak = frame["non_local_SWR: peak_log_modulation"]
    non_peak = frame["non_local_no_SWR: peak_log_modulation"]
    either = swr_called | non_called
    split = population[
        "non_local_SWR + non_local_no_SWR vs. non_local: heldout_delta_dev"
    ]
    best = population["min_heldout_model_replay_split"]
    values = [
        make(
            "overlap_units_tested",
            n,
            "count",
            "count",
            "units with a finite p-value for both SWR-coupled and non-SWR replay",
        ),
        make(
            "overlap_units_both",
            both,
            "count",
            "count",
            "of those, units called (raw p < 0.05) for both",
        ),
        make(
            "overlap_units_swr_replay_only",
            swr_only,
            "count",
            "count",
            "units called for SWR-coupled replay and not for non-SWR replay",
        ),
        make(
            "overlap_units_non_swr_replay_only",
            non_only,
            "count",
            "count",
            "units called for non-SWR replay and not for SWR-coupled replay",
        ),
        make(
            "overlap_units_neither",
            neither,
            "count",
            "count",
            "units called for neither",
        ),
        make(
            "overlap_units_both_expected",
            swr_called.mean() * non_called.mean() * n,
            "quantity",
            "int",
            "units called for both expected if the two calls were independent "
            "(product of the two called shares times the units tested)",
            "units",
        ),
        make(
            "overlap_odds_ratio",
            odds_ratio(both, swr_only, non_only, neither),
            "quantity",
            "sig2",
            "odds ratio of being called for non-SWR replay given called for SWR-coupled "
            "replay, pooled over units (1 = independent)",
        ),
        make(
            "overlap_odds_ratio_animal_min",
            min(per_animal),
            "quantity",
            "sig2",
            "lowest per-animal odds ratio",
        ),
        make(
            "overlap_odds_ratio_animal_max",
            max(per_animal),
            "quantity",
            "sig2",
            "highest per-animal odds ratio",
        ),
        make(
            "overlap_animals_odds_ratio_above_one",
            sum(r > 1 for r in per_animal),
            "count",
            "count",
            "animals whose odds ratio is above 1",
        ),
        make(
            "overlap_rank_correlation",
            stats.spearmanr(swr_peak, non_peak).statistic,
            "quantity",
            "dec2",
            "Spearman correlation over units tested for both of the peak log modulation "
            "for SWR-coupled replay and for non-SWR replay",
        ),
        make(
            "overlap_rank_correlation_called",
            stats.spearmanr(swr_peak[either], non_peak[either]).statistic,
            "quantity",
            "dec2",
            "the same over units called for either",
        ),
        make(
            "overlap_same_sign_fraction_both",
            float(
                (
                    np.sign(swr_peak[swr_called & non_called])
                    == np.sign(non_peak[swr_called & non_called])
                ).mean()
            ),
            "fraction",
            "pct",
            "of the units called for both, the share whose two peak modulations have the "
            "same sign (both up or both down)",
        ),
        make(
            "split_improves_fraction",
            float((split.dropna() > 0).mean()),
            "fraction",
            "pct",
            "share of units whose held-out deviance improves when the non-local response "
            "is split by SWR status (non_local_SWR + non_local_no_SWR vs. non_local)",
        ),
        make(
            "split_best_model_fraction",
            float(
                (best == "non_local_SWR + non_local_no_SWR").sum() / best.notna().sum()
            ),
            "fraction",
            "pct",
            "share of units whose best model by held-out deviance (among constant, "
            "non_local, and the split) is the split by SWR status",
        ),
        make(
            "split_swr_larger_fraction_called",
            float((swr_peak[either].abs() > non_peak[either].abs()).mean()),
            "fraction",
            "pct",
            "of units called for either, the share whose absolute peak modulation is "
            "larger for SWR-coupled replay than for non-SWR replay",
        ),
    ]
    for word, column in (
        ("swr_replay", "non_local_SWR"),
        ("non_swr_replay", "non_local_no_SWR"),
        ("swr", "SWR"),
        ("matched_control", "matched_control"),
    ):
        peak = population[f"{column}: peak_log_modulation"].dropna()
        label = {
            "swr_replay": "SWR-coupled replay",
            "non_swr_replay": "non-SWR replay",
            "swr": "SWRs",
            "matched_control": "matched quiet-rest control windows (the noise floor)",
        }[word]
        values += [
            make(
                f"effect_abs_peak_median_{word}",
                peak.abs().median(),
                "quantity",
                "dec2",
                f"median over units of the absolute peak log-rate modulation for {label}",
                "log rate",
            ),
            make(
                f"effect_rate_multiplier_median_{word}",
                float(np.exp(peak.abs().median())),
                "quantity",
                "sig2",
                f"exp of that median: the median unit's largest rate change for {label}, as "
                "a multiplier (up or down)",
                "x",
            ),
            make(
                f"effect_suppressed_fraction_{word}",
                float((peak < 0).mean()),
                "fraction",
                "pct",
                f"share of units whose peak modulation for {label} is a decrease",
            ),
        ]
    for word, column, called in (
        ("swr_replay", "non_local_SWR", swr_called),
        ("non_swr_replay", "non_local_no_SWR", non_called),
    ):
        peak = frame[f"{column}: peak_log_modulation"][called]
        values.append(
            make(
                f"effect_abs_peak_median_called_{word}",
                peak.abs().median(),
                "quantity",
                "dec2",
                f"median absolute peak log-rate modulation of the units called for "
                f"{'SWR-coupled' if word == 'swr_replay' else 'non-SWR'} replay",
                "log rate",
            )
        )
    return values


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
    for that class (raw empirical p < 0.05 against its own matched null),
    split by the sign of each unit's peak (enhanced or suppressed):

    - per unit, the peak lag (largest ``|curve|`` on +/-200 ms) and the
      half-maximum onset (:func:`~continuum_swr_replay.analysis.pfc.timing.peak_and_onset`),
      their medians and the shares before event onset;
    - for the population, the peak lag and onset of the sign-aligned mean
      curve, with a 95 % interval from resampling sessions, and the animals
      whose own mean curve has its onset before event onset.

    Beside them: the share of all units whose matched quiet-rest control curve
    peaks before onset (the floor for a peak lag drawn by noise), units called
    for both replay classes compared pairwise, and the timing calibration
    (simulated responses of known onset fit by the same estimator), which says
    how the estimator moves an onset and how often noise puts a unit's onset
    before zero when the response starts at zero.
    """
    values: list[Value] = []

    def add(key: str, value: float, *args: Any, **kwargs: Any) -> None:
        # A statistic over an empty set is absent, never invented.
        if np.isfinite(float(value)):
            values.append(make(key, value, *args, **kwargs))

    # Resample recording days (NWB files), not epochs: the epochs of one day
    # are recorded on the same tetrodes and need not be independent.
    sessions = population["nwb_file_name"].astype(str).to_numpy()
    animals = population["animal"].to_numpy()
    ms = 1e3
    rule = (
        "the half-maximum onset: the earliest lag of the run, ending at the peak, "
        "over which the curve keeps the peak's sign and at least half its size"
    )
    timing_by_class = {}
    for word, (name, curve_key) in TIMING_CLASSES.items():
        label = CLASS_LABELS[name]
        pvalue = population[HELDOUT_P_COLUMNS[name]].to_numpy(dtype=float)
        called = pvalue < ALPHA
        timing = peak_and_onset(curves[curve_key], lag_time)
        timing_by_class[word] = (called, timing)
        for direction, sign in (("enhanced", 1.0), ("suppressed", -1.0)):
            chosen = called & (np.sign(timing.peak) == sign)
            stem = f"{word}_{direction}"
            who = f"units called for {label} (raw p < {ALPHA:g}) whose peak is {'an increase' if sign > 0 else 'a decrease'}"
            peak_ms = timing.peak_lag[chosen] * ms
            onset_ms = timing.onset_lag[chosen] * ms
            add(f"timing_units_{stem}", int(chosen.sum()), "count", "count", who)
            if not chosen.any():
                continue
            add(
                f"timing_peak_lag_median_ms_{stem}",
                np.median(peak_ms),
                "quantity",
                "int",
                f"median peak lag (largest |modulation| within +/-200 ms of onset) of {who}",
                "ms",
            )
            low, high = np.percentile(peak_ms, [25, 75])
            add(
                f"timing_peak_lag_quartile_low_ms_{stem}",
                low,
                "quantity",
                "int",
                f"25th percentile of the peak lag of {who}",
                "ms",
            )
            add(
                f"timing_peak_lag_quartile_high_ms_{stem}",
                high,
                "quantity",
                "int",
                f"75th percentile of the peak lag of {who}",
                "ms",
            )
            add(
                f"timing_peak_before_onset_fraction_{stem}",
                float((peak_ms < 0).mean()),
                "fraction",
                "pct",
                f"share of {who} whose peak lag is before event onset",
            )
            add(
                f"timing_onset_median_ms_{stem}",
                np.median(onset_ms),
                "quantity",
                "int",
                f"median response onset of {who}; onset is {rule}",
                "ms",
            )
            add(
                f"timing_onset_before_onset_fraction_{stem}",
                float((onset_ms < 0).mean()),
                "fraction",
                "pct",
                f"share of {who} whose response onset is before event onset",
            )
            add(
                f"timing_onset_at_edge_fraction_{stem}",
                float(timing.onset_at_edge[chosen].mean()),
                "fraction",
                "pct",
                f"share of {who} whose half-maximum run reaches the window edge "
                "(-200 ms), so their onset is at or before it",
            )
            add(
                f"timing_peak_at_edge_fraction_{stem}",
                float((np.abs(peak_ms) >= REPORT_HALF_WINDOW_S * ms - 1e-6).mean()),
                "fraction",
                "pct",
                f"share of {who} whose peak lies at the edge of the +/-200 ms window",
            )
            rise_ms = (
                peak_and_onset(
                    curves[curve_key][:, chosen], lag_time, baseline="window_start"
                ).onset_lag
                * ms
            )
            add(
                f"timing_rise_onset_median_ms_{stem}",
                np.nanmedian(rise_ms),
                "quantity",
                "int",
                f"median rise onset of {who}: the onset rule measured from the curve's "
                "value at -200 ms instead of from zero",
                "ms",
            )
            population_curve = population_timing(
                curves[curve_key][:, chosen],
                timing.peak[chosen],
                sessions[chosen],
                lag_time,
                n_boot=TIMING_BOOTSTRAP_DRAWS,
                seed=TIMING_BOOTSTRAP_SEED,
            )
            population_rise = population_timing(
                curves[curve_key][:, chosen],
                timing.peak[chosen],
                sessions[chosen],
                lag_time,
                n_boot=TIMING_BOOTSTRAP_DRAWS,
                seed=TIMING_BOOTSTRAP_SEED,
                baseline="window_start",
            )
            add(
                f"timing_population_pre_event_level_fraction_{stem}",
                float(
                    np.clip(
                        population_curve.curve[0] / np.max(population_curve.curve),
                        0.0,
                        1.0,
                    )
                ),
                "fraction",
                "pct",
                f"value at -200 ms of the sign-aligned mean curve of {who}, as a share "
                "of its peak: how far the curve sits from zero before the event",
            )
            add(
                f"timing_population_rise_onset_ms_{stem}",
                population_rise.onset_lag * ms,
                "quantity",
                "int",
                f"rise onset of the sign-aligned mean curve of {who}: half way from its "
                "value at -200 ms to its peak",
                "ms",
            )
            add(
                f"timing_population_rise_onset_ms_ci_low_{stem}",
                population_rise.onset_lag_interval[0] * ms,
                "quantity",
                "int",
                f"2.5th percentile of that over {TIMING_BOOTSTRAP_DRAWS:,} session "
                f"resamples (seed {TIMING_BOOTSTRAP_SEED})",
                "ms",
            )
            add(
                f"timing_population_rise_onset_ms_ci_high_{stem}",
                population_rise.onset_lag_interval[1] * ms,
                "quantity",
                "int",
                "97.5th percentile of that over the same resamples",
                "ms",
            )
            for what, point, (ci_low, ci_high) in (
                ("peak", population_curve.peak_lag, population_curve.peak_lag_interval),
                (
                    "onset",
                    population_curve.onset_lag,
                    population_curve.onset_lag_interval,
                ),
            ):
                text = (
                    f"{'peak lag' if what == 'peak' else 'onset (' + rule + ')'} of the "
                    f"mean curve of {who}, each unit's curve signed by its peak"
                )
                add(
                    f"timing_population_{what}_ms_{stem}",
                    point * ms,
                    "quantity",
                    "int",
                    text,
                    "ms",
                )
                add(
                    f"timing_population_{what}_ms_ci_low_{stem}",
                    ci_low * ms,
                    "quantity",
                    "int",
                    f"2.5th percentile of that over {TIMING_BOOTSTRAP_DRAWS:,} session "
                    f"resamples (seed {TIMING_BOOTSTRAP_SEED})",
                    "ms",
                )
                add(
                    f"timing_population_{what}_ms_ci_high_{stem}",
                    ci_high * ms,
                    "quantity",
                    "int",
                    "97.5th percentile of that over the same resamples",
                    "ms",
                )
            onsets_by_animal = []
            rises_by_animal = []
            for animal in np.unique(animals[chosen]):
                own = chosen & (animals == animal)
                own_curve = sign_aligned_mean(
                    curves[curve_key][:, own], timing.peak[own]
                )
                onsets_by_animal.append(
                    peak_and_onset(own_curve, lag_time).onset_lag[0]
                )
                rise = peak_and_onset(own_curve, lag_time, baseline="window_start")
                rises_by_animal.append(rise.onset_lag[0])
                add(
                    f"timing_population_rise_onset_ms_{stem}_{ANIMAL_WORDS[animal]}",
                    rise.onset_lag[0] * ms,
                    "quantity",
                    "int",
                    f"rise onset of the sign-aligned mean curve of {animal}'s {who}",
                    "ms",
                )
            add(
                f"timing_population_onset_animals_{stem}",
                len(onsets_by_animal),
                "count",
                "count",
                f"animals with at least one of the {who}",
            )
            add(
                f"timing_population_onset_animals_negative_{stem}",
                int(np.sum(np.asarray(onsets_by_animal) < 0)),
                "count",
                "count",
                f"of those animals, how many have the half-maximum onset of their own "
                f"mean curve of {who} before event onset",
            )
            add(
                f"timing_population_rise_onset_animals_negative_{stem}",
                int(np.sum(np.asarray(rises_by_animal) < 0)),
                "count",
                "count",
                f"of those animals, how many have the rise onset of their own mean curve "
                f"of {who} before event onset",
            )

    swr_called, swr_timing = timing_by_class["swr_replay"]
    non_called, non_timing = timing_by_class["non_swr_replay"]
    paired = (
        swr_called & non_called & (np.sign(swr_timing.peak) == np.sign(non_timing.peak))
    )
    difference = (non_timing.peak_lag[paired] - swr_timing.peak_lag[paired]) * ms
    add(
        "timing_paired_units",
        int(paired.sum()),
        "count",
        "count",
        "units called for both SWR-coupled and non-SWR replay whose two peaks have the "
        "same sign",
    )
    if paired.any():
        add(
            "timing_peak_lag_difference_median_ms",
            np.median(difference),
            "quantity",
            "int",
            "over those units, the median of the non-SWR replay peak lag minus the "
            "SWR-coupled replay peak lag",
            "ms",
        )
        add(
            "timing_non_swr_later_fraction",
            float((difference > 0).mean()),
            "fraction",
            "pct",
            "share of those units whose non-SWR replay peak is later than their "
            "SWR-coupled replay peak",
        )
    control = peak_and_onset(curves["matched_control"], lag_time)
    usable = np.isfinite(control.peak_lag)
    add(
        "timing_matched_control_peak_before_onset_fraction",
        float((control.peak_lag[usable] < 0).mean()),
        "fraction",
        "pct",
        "share of all units whose matched quiet-rest control curve peaks before the "
        "control window's onset (what a peak lag drawn by noise gives)",
    )

    if calibration is not None:
        for scenario in calibration["scenarios"]:
            if scenario["true_onset_ms"] != 0.0:
                continue
            direction = "enhanced" if scenario["gain"] > 0 else "suppressed"
            stem = f"{scenario['shape']}_{direction}"
            response = (
                "a 100 ms boxcar"
                if scenario["shape"] == "boxcar"
                else "an alpha function peaking 30 ms after it starts"
            )
            what = (
                f"simulated units ({scenario['n_units']}) whose response, {response} of "
                f"log-rate gain {scenario['gain']:+g}, starts exactly at event onset, "
                "fit by the shipped basis, prior and fitter"
            )
            truth = (
                f"the truth's own half-maximum onset "
                f"({scenario['true_half_max_onset_ms']:g} ms) read by the same rule"
            )
            add(
                f"timing_calibration_unit_onset_bias_ms_{stem}",
                scenario["unit_onset_bias_ms_median"],
                "quantity",
                "int",
                f"median half-maximum onset of {what}, minus {truth}",
                "ms",
            )
            add(
                f"timing_calibration_population_rise_onset_bias_ms_{stem}",
                scenario["population_rise_onset_bias_ms"],
                "quantity",
                "int",
                f"rise onset of the sign-aligned mean curve of {what}, minus {truth}",
                "ms",
            )
            add(
                f"timing_calibration_population_onset_bias_ms_{stem}",
                scenario["population_onset_bias_ms"],
                "quantity",
                "int",
                f"half-maximum onset of the sign-aligned mean curve of {what}, minus "
                f"{truth}",
                "ms",
            )
            if scenario["shape"] == "alpha":
                add(
                    f"timing_calibration_population_peak_bias_ms_{stem}",
                    scenario["population_peak_bias_ms"],
                    "quantity",
                    "int",
                    f"peak lag of the sign-aligned mean curve of {what}, minus the "
                    f"truth's peak ({scenario['true_peak_ms']:g} ms)",
                    "ms",
                )
            add(
                f"timing_calibration_onset_before_zero_fraction_{stem}",
                scenario["fraction_onset_before_zero"],
                "fraction",
                "pct",
                f"share of {what} whose apparent half-maximum onset is before event "
                "onset: the noise floor of the per-unit share",
            )
    return values


def summarize(inputs: Part4Inputs, produced_by: str, commit: str) -> Summary:
    """Part 4's summary from ``inputs``."""
    summary = Summary(part=4, produced_by=produced_by, commit=commit)
    sidecars = read_sidecars(inputs.bundle_dir)
    pending = staleness(sidecars, inputs.event_table, inputs.stop_table)
    population, curves, lag_time = load_population_data(
        inputs.bundle_dir, min_firing_rate=MIN_FIRING_RATE_HZ
    )
    cohort = json.loads(inputs.pfc_cohort.read_text())
    source = (
        f"{rel(inputs.bundle_dir)}/*{STEM}.pkl and their provenance sidecars "
        f"(scripts/pfc/run_pfc_analysis_bandit.py), {len(sidecars)} bundles, "
        "read by Figure 4's loader (analysis/pfc/load_population.py, 1 Hz gate)"
    )
    claims = ("20",)
    make = _value_factory(source, claims, ("paper",), pending)
    figure = _value_factory(source, claims, ("paper", "figure: 4d"), pending)
    summary.add(*cohort_values(sidecars, population, cohort, make))
    summary.add(*estimator_values(sidecars, make))
    summary.add(*prior_binding_values(population, make))
    summary.add(*called_fraction_values(population, figure))
    summary.add(*null_geometry_values(population, sidecars, make))
    summary.add(*overlap_and_effect_values(population, make))
    calibration = (
        json.loads(inputs.timing_calibration.read_text())
        if inputs.timing_calibration.exists()
        else None
    )
    summary.add(*timing_values(population, curves, lag_time, calibration, make))
    summary.add(
        *ml_comparator_values(
            inputs.ml_tree,
            _value_factory(
                f"{rel(inputs.ml_tree)}/*{STEM}.provenance.json",
                claims,
                ("paper",),
                ML_PENDING,
            ),
        )
    )

    data = inputs.bundle_dir
    summary.inputs += [
        files_record(f"*{STEM}.pkl", data.glob(f"*{STEM}.pkl"), ROOT_DIR),
        files_record(f"*{STEM}.*.npy", data.glob(f"*{STEM}.*.npy"), ROOT_DIR),
        files_record(
            f"*{STEM}.provenance.json", data.glob(f"*{STEM}.provenance.json"), ROOT_DIR
        ),
        file_record(inputs.pfc_cohort, ROOT_DIR),
        file_record(inputs.event_table, ROOT_DIR),
        file_record(inputs.stop_table, ROOT_DIR),
        *(
            [file_record(inputs.timing_calibration, ROOT_DIR)]
            if inputs.timing_calibration.exists()
            else []
        ),
        files_record(
            f"*{STEM}.provenance.json",
            inputs.ml_tree.glob(f"*{STEM}.provenance.json"),
            ROOT_DIR,
            note="superseded maximum-likelihood tree, for one comparison",
        ),
    ]
    return summary


def rel(path: Path) -> str:
    """``path`` relative to the repository when it is under it."""
    return (
        str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path)
    )
