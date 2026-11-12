"""The mpfc summary: how mPFC units respond to SWR and non-SWR non-local events.

Every value is read from an output an analysis already wrote; nothing is
refitted here. This module names and describes the values. The statistics are
computed in ``analysis.pfc.population_statistics`` (the unit population) and
``analysis.pfc.timing`` (when the responses peak and start), with the smallest
empirical p-value in ``analysis.pfc.heldout``; here only the size of an input
(its bundles, units, animals) is counted, and a value that a provenance sidecar,
the timing calibration, the marked-family summary or the stop-context fits
already hold is copied out. The inputs (:class:`MpfcInputs`):

- the at-well mPFC unit bundles (``Processed-Data/{epoch}_mPFC_modulation.pkl``
  and their ``*.provenance.json`` sidecars, from
  ``scripts/pfc/run_pfc_analysis_bandit.py``), read through Figure 4's loader
  (:func:`~continuum_swr_replay.analysis.pfc.load_population.load_population_data`,
  no firing-rate cutoff: every unit with a usable fit) for every per-unit value,
  with a 1 Hz cutoff as a check;
- the PFC cohort record the runner writes (``pfc_cohort_mPFC.json``);
- the canonical event and stop tables, only to check that the bundles and the
  follow-up fits below were made from the current ones;
- the timing calibration (``pfc_response_followups/timing_calibration.json``,
  from ``scripts/pfc/calibrate_pfc_kernel_timing.py``);
- the extended-event-set bundles with SWRs without replay
  (``pfc_response_followups/swr_without_replay``, from
  ``scripts/pfc/run_pfc_analysis_bandit.py --event-set with_swr_without_replay``);
- the marked-family summary of the follow-up models and the bundle sidecars
  beside it (``pfc_response_followups/marked_family_well_unknown``, from
  ``scripts/pfc/run_pfc_marked_model_family.py`` and
  ``scripts/pfc/summarize_pfc_marked_family.py``), pending when a sidecar
  records other tables (:func:`marked_family_staleness`);
- the stop-context fits (``pfc_response_followups/stop_context``, from
  ``scripts/pfc/run_pfc_stop_context.py``), pending when a bundle, cached
  session table or table they read has changed (:func:`stop_context_staleness`).

Figure 4 (``scripts/figures/generate_figure04.py``) draws the unit fractions
with the same functions. ``scripts/pfc/compare_pfc_cohorts.py`` calls them for
the called fractions and ``scripts/pfc/measure_pfc_null_geometry.py`` for the
units called without improvement; the cohort, fit-count, prior-binding and
other null-geometry numbers those two reports print are computed in the
scripts. Unit fractions are descriptive and pooled over sessions and animals;
they are not population inference.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc import population_statistics, timing
from continuum_swr_replay.analysis.pfc.heldout import empirical_pvalue_floor
from continuum_swr_replay.analysis.pfc.load_population import load_population_data
from continuum_swr_replay.analysis.pfc.modulation import THREE_CLASS_MODEL
from continuum_swr_replay.analysis.replay.coefficient_statistics import group_range
from continuum_swr_replay.paths import (
    ROOT_DIR,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.values import (
    Summary,
    Tags,
    Value,
    file_record,
    files_record,
    rel,
    value_factory,
)

TAGS = {
    "modulation": Tags(("4.1",)),
    "figure_4d": Tags(("4.1",), ("4d",)),
    "units": Tags(("4.1",), ("4c", "4d")),
}
"""Each group of values: every mPFC value answers question 4.1; the groups
differ in the panels of Figure 4 that show them."""

STEM = "_mPFC_modulation"
"""The canonical (at-well, sigma 3) bundle name; other cohorts and priors carry
a suffix after it."""

ALPHA = 0.05
"""Figure 4's raw empirical-p threshold."""

MIN_FIRING_RATE_HZ = 0.0
"""No firing-rate cutoff (the lead, 2026-10-29): every unit with a usable fit.
A unit with no spike in the fitted bins has no fit and drops out in the loader."""

CHECK_FIRING_RATE_HZ = 1.0
"""The cutoff the main values used before 2026-10-29, kept as a check; the full
table over cutoffs is ``scripts/pfc/part4_rate_threshold_sensitivity.py``'s CSV."""

#: The check keys quoted at :data:`CHECK_FIRING_RATE_HZ`, by their main key.
CHECK_KEYS = (
    "units",
    "swr_replay_called_fraction",
    "non_swr_replay_called_fraction",
    "engagement_called_fraction",
    "overlap_odds_ratio",
    "overlap_rank_correlation",
    "swr_no_replay_called_fraction_all",
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

SWR_WITHOUT_REPLAY_TREE = "pfc_response_followups/swr_without_replay"
"""The extended-event-set bundles (``run_pfc_analysis_bandit.py --event-set
with_swr_without_replay``), relative to the processed-data root."""

STOP_CONTEXT_RESULTS = "pfc_response_followups/stop_context/results.csv"
"""The stop-level context fits (``scripts/pfc/run_pfc_stop_context.py fit``)."""

MARKED_FAMILY_SUMMARY = (
    "pfc_response_followups/marked_family_well_unknown/summary.parquet"
)
"""The marked-family summary (``scripts/pfc/summarize_pfc_marked_family.py``)
of the follow-up models fit with the per-stop baseline prior and the
``well_unknown`` baseline column (bundle contract 3)."""

#: Joint follow-up models and targets the mpfc summary quotes, by key word.
MARKED_MODELS = {
    "content": "replay_content",
    "strength": "event_strength",
    "ripple": "ripple_power",
    "ripplepooled": "ripple_power_pooled",
}
MARKED_TARGETS = {
    "current_arm_fraction": "current_arm",
    "content_weighted_recent_reward": "recent_reward",
    "content_weighted_belief_uncertainty": "uncertainty",
    "log_event_duration_s": "duration",
    "ripple_power_z": "ripple_power",
    "mua_rate_z": "mua_rate",
}
MARKED_CLASSES = {
    "non_local_SWR": "swr_replay",
    "non_local_no_SWR": "non_swr_replay",
    "SWR_no_non_local": "swr_no_replay",
    "pooled": "pooled",
}
#: The (model, target) pairs whose held-out-gain share is quoted, per replay class
#: (the lead, 2026-10-29): replay content's current-arm fraction and recent
#: reward, event strength's multiunit rate and ripple power.
MARKED_HELDOUT_SHARES = {
    ("replay_content", "current_arm_fraction"),
    ("replay_content", "content_weighted_recent_reward"),
    ("event_strength", "mua_rate_z"),
    ("event_strength", "ripple_power_z"),
}
MARKED_HELDOUT_SHARE_CLASSES = ("non_local_SWR", "non_local_no_SWR")
"""The replay classes whose held-out-gain share is quoted for the pairs of
:data:`MARKED_HELDOUT_SHARES` (not the pooled fit, nor SWRs without replay)."""

#: The new class's extra curves, as ``load_population_data`` reads them.
SWR_WITHOUT_REPLAY_CURVES = ("SWR_no_non_local", "non_local_SWR_3class")

#: A session-level event floor for the new class's check: the main values are
#: over every session, and the same values over sessions with at least this
#: many SWRs without replay are the check.
SWR_WITHOUT_REPLAY_MIN_EVENTS = 10

#: The curves the timing values are read from, by the word their keys use: the
#: class (``HELDOUT_P_COLUMNS`` key) a unit must be called for, and the curve
#: ``load_population_data`` returns for it.
TIMING_CLASSES = {
    "swr": ("SWR", "swr"),
    "swr_replay": ("non_local_SWR", "non_local_swr"),
    "non_swr_replay": ("non_local_no_SWR", "non_local_no_swr"),
}

#: Recording-day resamples behind the population timing intervals, and their seed.
TIMING_BOOTSTRAP_DRAWS = 1_000
TIMING_BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class MpfcInputs:
    """The files the mpfc summary reads."""

    bundle_dir: Path
    pfc_cohort: Path
    event_table: Path
    stop_table: Path
    timing_calibration: Path
    swr_without_replay_dir: Path
    marked_family_summary: Path
    stop_context_results: Path


def default_inputs() -> MpfcInputs:
    data = processed_data_dir()
    return MpfcInputs(
        bundle_dir=data,
        pfc_cohort=data / "pfc_cohort_mPFC.json",
        event_table=table_path("event_table", replay_content_dir()),
        stop_table=table_path("stop_table", replay_content_dir()),
        timing_calibration=data / TIMING_CALIBRATION,
        swr_without_replay_dir=data / SWR_WITHOUT_REPLAY_TREE,
        marked_family_summary=data / MARKED_FAMILY_SUMMARY,
        stop_context_results=data / STOP_CONTEXT_RESULTS,
    )


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


def staleness(sidecars: list[dict[str, Any]], current: dict[str, str]) -> str | None:
    """Why the bundles are stale against the current tables, or None.

    A bundle records the SHA-256 of the canonical event table and the stop
    table it was fit on; a bundle fit on other tables describes other events.
    ``current`` holds the current tables' (:func:`table_hashes`).
    """
    stale = {
        name: sum(s.get(name) != sha for s in sidecars) for name, sha in current.items()
    }
    return _stale_note("the mPFC bundles", stale, len(sidecars))


def table_hashes(event_table: Path, stop_table: Path) -> dict[str, str]:
    """The SHA-256 of the current canonical event and stop tables, under the
    names the bundles' provenance records them by."""
    return {
        "canonical_event_table_sha256": _sha256(event_table),
        "stop_table_sha256": _sha256(stop_table),
    }


def _stale_note(what: str, stale: dict[str, int], total: int) -> str | None:
    """A pending note naming each recorded input that differs, or None."""
    if not any(stale.values()):
        return None
    return (
        f"{what} were made from other inputs than the current ones "
        f"({', '.join(f'{name} differs in {k} of {total}' for name, k in stale.items() if k)}); "
        "refit before citing"
    )


def stop_context_staleness(
    results: Path, bundle_dir: Path, current: dict[str, str], n_cohort: int
) -> str | None:
    """Why the stop-context fit is stale against the current inputs, or None.

    The fit's ``results.provenance.json`` records the SHA-256 of every canonical
    bundle and cached session table it read; each session table's provenance
    records the event and stop tables it was assembled from. Any of them that
    differs from the current file (matched by file name) makes the fit stale.

    Raises
    ------
    ValueError
        If the fit does not cover the ``n_cohort`` PFC epochs.
    """
    provenance = json.loads(results.with_name("results.provenance.json").read_text())
    if len(provenance["bundles"]) != n_cohort:
        raise ValueError(
            f"{results}: fit on {len(provenance['bundles'])} sessions, not the "
            f"{n_cohort} of the PFC cohort"
        )
    sessions_dir = results.parent / "sessions"
    tables = {
        Path(path).name: sha for path, sha in provenance["session_tables"].items()
    }
    records = [
        json.loads((sessions_dir / name).with_suffix(".provenance.json").read_text())
        for name in tables
    ]

    def differs(path: Path, sha: str) -> bool:
        return not path.exists() or _sha256(path) != sha

    stale = {
        "bundle_sha256": sum(
            differs(bundle_dir / Path(path).name, sha)
            for path, sha in provenance["bundles"].items()
        ),
        "session_table_sha256": sum(
            differs(sessions_dir / name, sha) for name, sha in tables.items()
        ),
        **{
            name: sum(r["input_provenance"].get(name) != sha for r in records)
            for name, sha in current.items()
        },
    }
    return _stale_note("the stop-context fits", stale, n_cohort)


def marked_family_staleness(
    summary_path: Path, current: dict[str, str], n_cohort: int
) -> str | None:
    """Why the marked-family summary is stale, or None.

    Reads every bundle sidecar (``*_marked_family.provenance.json``) beside the
    summary, wherever it lives: each records the event and stop tables its fit
    read. A sidecar recording other tables, or a bundle written after the
    summary, makes the summary stale.

    Raises
    ------
    ValueError
        If the sidecars do not cover the ``n_cohort`` PFC epochs.
    """
    folder = summary_path.parent
    sidecars = [
        json.loads(path.read_text())
        for path in sorted(folder.glob("*_marked_family.provenance.json"))
    ]
    sessions = {(s["nwb_file_name"], s["epoch_name"]) for s in sidecars}
    if len(sessions) != n_cohort:
        raise ValueError(
            f"{folder}: marked-family bundles of {len(sessions)} sessions, not the "
            f"{n_cohort} of the PFC cohort"
        )
    written = summary_path.stat().st_mtime
    stale = {
        **{
            name: sum(s.get(name) != sha for s in sidecars)
            for name, sha in current.items()
        },
        "bundle written after the summary": sum(
            path.stat().st_mtime > written
            for path in folder.glob("*_marked_family.pkl")
        ),
    }
    return _stale_note("the marked-family bundles", stale, len(sidecars))


def _finite_adder(
    values: list[Value], make: Callable[..., Value]
) -> Callable[..., None]:
    """An ``add(key, value, ...)`` that appends ``make(key, value, ...)`` to
    ``values`` when the value is finite. A statistic over an empty set is
    absent, never invented."""

    def add(key: str, value: float, *args: Any, **kwargs: Any) -> None:
        if np.isfinite(float(value)):
            values.append(make(key, value, *args, **kwargs))

    return add


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
    stop restriction; ``compare_pfc_cohorts.session_cohort_table`` shows the
    same totals.
    """
    if len(sidecars) != cohort["n_pfc_cohort"]:
        raise ValueError(
            f"{len(sidecars)} bundles, but the PFC cohort holds "
            f"{cohort['n_pfc_cohort']} epochs"
        )
    if _uniform(sidecars, "stop_cohort") != "at_well":
        raise ValueError("the canonical bundles are not the at-well cohort")
    stats = population_statistics.cohort_count_statistics(
        pd.DataFrame([s["cohort_counts"] for s in sidecars])
    )
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
            "mPFC units in the population: every unit with a usable fit (no "
            "firing-rate cutoff; a unit with no spike in the fitted bins has none) "
            "and finite modulation curves",
            tags=TAGS["units"],
        ),
        make(
            "bins_fit",
            stats["n_bins_fit"],
            "count",
            "count",
            "immobile 2 ms bins inside a kept stop, the bins the GLM is fit on, "
            "summed over sessions",
        ),
        make(
            "bins_immobile",
            stats["n_bins_immobile"],
            "count",
            "count",
            "immobile 2 ms bins before the stop restriction, summed over sessions",
        ),
        make(
            "bins_fit_fraction",
            stats["bins_fit_fraction"],
            "fraction",
            "pct",
            "share of the immobile bins the stop restriction keeps, pooled",
        ),
        make(
            "nonlocal_events_kept",
            stats["n_nonlocal_kept"],
            "count",
            "count",
            "non-local events whose onset is in a kept stop, summed over sessions",
        ),
        make(
            "nonlocal_events_before",
            stats["n_nonlocal_before"],
            "count",
            "count",
            "the session's non-local events before the stop restriction, summed",
        ),
        make(
            "nonlocal_events_kept_fraction",
            stats["nonlocal_kept_fraction"],
            "fraction",
            "pct",
            "share of the non-local events the stop restriction keeps, pooled",
        ),
        make(
            "swrs_kept_fraction",
            stats["swrs_kept_fraction"],
            "fraction",
            "pct",
            "share of the SWRs the stop restriction keeps, pooled",
        ),
        make(
            "swr_coupled_share_session_median",
            stats["swr_coupled_share_session_median"],
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
    floor = empirical_pvalue_floor(draws)
    fits = population_statistics.fit_count_statistics(
        pd.DataFrame(
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
        )
    )
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
            floor["denominator"],
            "count",
            "count",
            "R + 1: the smallest plus-one empirical p-value is one over it",
        ),
        make(
            "min_pvalue",
            floor["min_pvalue"],
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
            fits["nonconverged_percent"],
            "quantity",
            "sig2",
            "non-converged fits as a percent of model fits",
            "%",
        ),
    ]


def prior_binding_values(
    population: pd.DataFrame, make: Callable[..., Value]
) -> list[Value]:
    """How often the prior moves a curve's peak, per curve.

    Among the well-posed
    units of a curve (unpenalised fit converged and no separated column), the
    share whose penalised peak differs from the unpenalised one by more than
    that peak's standard error (the bundle's ``prior_binds``). The values are
    the lowest and highest share over the five curves;
    ``compare_pfc_cohorts.shrinkage_table`` shows each curve's share.
    """
    # A curve with no well-posed unit has no share and is left out.
    shares = group_range(
        population_statistics.prior_binding_percent(population, CURVES)
        .dropna()
        .to_dict()
    )
    return [
        make(
            "prior_binds_percent_min",
            shares["low"],
            "quantity",
            "sig2",
            f"lowest per-curve percent of well-posed units the prior binds on "
            f"({shares['low_group']})",
            "%",
        ),
        make(
            "prior_binds_percent_max",
            shares["high"],
            "quantity",
            "sig2",
            f"highest per-curve percent of well-posed units the prior binds on "
            f"({shares['high_group']})",
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
    fractions = population_statistics.called_fractions(population, ALPHA)
    values = []
    for word, name in CLASSES.items():
        stats = fractions.loc[name]
        label = CLASS_LABELS[name]
        values += [
            make(
                f"{word}_called_fraction",
                stats["fraction"],
                "fraction",
                "pct",
                f"share of tested units at raw empirical p < {ALPHA:g} for {label}, "
                "pooled (descriptive)",
            ),
            make(
                f"{word}_units_called",
                stats["n_called"],
                "count",
                "count",
                f"units at raw empirical p < {ALPHA:g} for {label}",
            ),
            make(
                f"{word}_units_tested",
                stats["n_tested"],
                "count",
                "count",
                f"units with a finite p-value for {label}",
            ),
        ]
    by_animal = population_statistics.called_fraction_by_animal(
        population, "engagement", ALPHA
    )
    for animal, stats in by_animal.iterrows():
        values.append(
            make(
                f"engagement_called_fraction_{ANIMAL_WORDS[animal]}",
                stats["fraction"],
                "fraction",
                "pct",
                f"share of {animal}'s tested units at raw empirical p < {ALPHA:g} for "
                "the engagement test",
            )
        )
    values.append(
        make(
            "engagement_bh_fraction",
            population_statistics.bh_called_fraction(population, "engagement", ALPHA),
            "fraction",
            "pct",
            "share of the tested units with a Benjamini-Hochberg q below "
            f"{ALPHA:g} for the engagement test, BH applied once over the pooled "
            "plus-one empirical p-values of every tested unit "
            "(analysis.statistics.benjamini_hochberg). For reference only: the units "
            "are not independent, and the p-values sit on a 1/(R + 1) grid",
            tags=TAGS["modulation"],
        )
    )
    best = population_statistics.best_model_fractions(
        population, list(BEST_MODELS.values())
    )
    for word, model in BEST_MODELS.items():
        values.append(
            make(
                f"best_model_{word}_fraction",
                best[model],
                "fraction",
                "pct",
                f"share of units whose best model by held-out deviance is {model}",
                tags=TAGS["figure_4d"],
            )
        )
    return values


# --- the null's geometry -----------------------------------------------------------


def null_geometry_values(
    population: pd.DataFrame,
    sidecars: list[dict[str, Any]],
    make: Callable[..., Value],
) -> list[Value]:
    """Three caveats on the empirical p-values, which
    ``scripts/pfc/measure_pfc_null_geometry.py`` reports per session.

    - Units called at raw p < 0.05 whose own observed held-out improvement (on
      the matchable subset) is not positive.
    - How much of the engagement null's windows land on the shipped control:
      each session's median over draws, then the median and range over
      sessions.
    - Events with no legal within-bout control, which enter neither side:
      per class, summed over sessions.
    """
    flat = population_statistics.called_without_improvement(population, ALPHA)
    engagement = flat.loc["engagement"]
    # A class with no called unit has no share and is left out.
    class_shares = group_range(flat["fraction"].drop("engagement").dropna().to_dict())
    overlap = population_statistics.session_value_statistics(
        [s["engagement_null_overlap"]["median"] for s in sidecars]
    )
    unmatchable = population_statistics.unmatchable_event_statistics(
        pd.DataFrame(
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
    )
    worst_label = f"{unmatchable['worst_session']} / {unmatchable['worst_event_class']}"
    return [
        make(
            "engagement_called_flat_fraction",
            engagement["fraction"],
            "fraction",
            "pct",
            "of the units called by the engagement test, the share whose own "
            "observed held-out improvement is not positive",
        ),
        make(
            "engagement_called_flat",
            engagement["n_flat"],
            "count",
            "count",
            "units called by the engagement test with no positive observed improvement",
        ),
        make(
            "class_called_flat_fraction_min",
            class_shares["low"],
            "fraction",
            "pctnum",
            f"lowest such share over the four event classes ({class_shares['low_group']})",
        ),
        make(
            "class_called_flat_fraction_max",
            class_shares["high"],
            "fraction",
            "pct",
            f"highest such share over the four event classes ({class_shares['high_group']})",
        ),
        make(
            "engagement_null_overlap_median",
            overlap["median"],
            "fraction",
            "pct",
            "median over sessions of each session's median share of an engagement "
            "null draw's windows that land on the shipped matched control",
        ),
        make(
            "engagement_null_overlap_session_min",
            overlap["min"],
            "fraction",
            "pctnum",
            "lowest session median of that share",
        ),
        make(
            "engagement_null_overlap_session_max",
            overlap["max"],
            "fraction",
            "pct",
            "highest session median of that share",
        ),
        make(
            "unmatchable_percent_class_min",
            unmatchable["percent_class_min"],
            "quantity",
            "sig2",
            "lowest percent, over event classes, of a class's events with no legal "
            f"within-bout control, summed over sessions ({unmatchable['class_min']})",
            "%",
        ),
        make(
            "unmatchable_percent_class_max",
            unmatchable["percent_class_max"],
            "quantity",
            "sig2",
            f"highest such percent ({unmatchable['class_max']})",
            "%",
        ),
        make(
            "unmatchable_worst_excluded",
            unmatchable["worst_excluded"],
            "count",
            "count",
            f"events with no legal control in the worst session and class ({worst_label})",
        ),
        make(
            "unmatchable_worst_total",
            unmatchable["worst_total"],
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
    overlap = population_statistics.replay_class_overlap_statistics(population, ALPHA)
    split = population_statistics.replay_split_statistics(population)
    values = [
        make(
            "overlap_units_tested",
            overlap["n_tested"],
            "count",
            "count",
            "units with a finite p-value for both SWR-coupled and non-SWR replay",
        ),
        make(
            "overlap_units_both",
            overlap["n_both"],
            "count",
            "count",
            "of those, units called (raw p < 0.05) for both",
        ),
        make(
            "overlap_units_swr_replay_only",
            overlap["n_swr_only"],
            "count",
            "count",
            "units called for SWR-coupled replay and not for non-SWR replay",
        ),
        make(
            "overlap_units_non_swr_replay_only",
            overlap["n_non_swr_only"],
            "count",
            "count",
            "units called for non-SWR replay and not for SWR-coupled replay",
        ),
        make(
            "overlap_units_neither",
            overlap["n_neither"],
            "count",
            "count",
            "units called for neither",
        ),
        make(
            "overlap_units_both_expected",
            overlap["n_both_expected"],
            "quantity",
            "int",
            "units called for both expected if the two calls were independent "
            "(product of the two called shares times the units tested)",
            "units",
        ),
        make(
            "overlap_odds_ratio",
            overlap["odds_ratio"],
            "quantity",
            "sig2",
            "odds ratio of being called for non-SWR replay given called for SWR-coupled "
            "replay, pooled over units (1 = independent)",
        ),
        make(
            "overlap_odds_ratio_animal_min",
            overlap["odds_ratio_animal_min"],
            "quantity",
            "sig2",
            "lowest per-animal odds ratio",
        ),
        make(
            "overlap_odds_ratio_animal_max",
            overlap["odds_ratio_animal_max"],
            "quantity",
            "sig2",
            "highest per-animal odds ratio",
        ),
        make(
            "overlap_animals_odds_ratio_above_one",
            overlap["n_animals_odds_ratio_above_one"],
            "count",
            "count",
            "animals whose odds ratio is above 1",
        ),
        make(
            "overlap_rank_correlation",
            overlap["rank_correlation"],
            "quantity",
            "dec2",
            "Spearman correlation over units tested for both of the peak log modulation "
            "for SWR-coupled replay and for non-SWR replay",
        ),
        make(
            "overlap_rank_correlation_called",
            overlap["rank_correlation_called"],
            "quantity",
            "dec2",
            "the same over units called for either",
        ),
        make(
            "overlap_same_sign_fraction_both",
            overlap["same_sign_fraction_both"],
            "fraction",
            "pct",
            "of the units called for both, the share whose two peak modulations have the "
            "same sign (both up or both down)",
        ),
        make(
            "split_improves_fraction",
            split["improves_fraction"],
            "fraction",
            "pct",
            "share of units whose held-out deviance improves when the non-local response "
            "is split by SWR status (non_local_SWR + non_local_no_SWR vs. non_local)",
        ),
        make(
            "split_best_model_fraction",
            split["best_model_fraction"],
            "fraction",
            "pct",
            "share of units whose best model by held-out deviance (among constant, "
            "non_local, and the split) is the split by SWR status",
        ),
        make(
            "split_swr_larger_fraction_called",
            overlap["swr_larger_fraction_called"],
            "fraction",
            "pct",
            "of units called for either, the share whose absolute peak modulation is "
            "larger for SWR-coupled replay than for non-SWR replay",
        ),
    ]
    effect_curves = (
        ("swr_replay", "non_local_SWR"),
        ("non_swr_replay", "non_local_no_SWR"),
        ("swr", "SWR"),
        ("matched_control", "matched_control"),
    )
    effects = population_statistics.peak_modulation_statistics(
        population, [column for _, column in effect_curves]
    )
    for word, column in effect_curves:
        stats = effects.loc[column]
        label = {
            "swr_replay": "SWR-coupled replay",
            "non_swr_replay": "non-SWR replay",
            "swr": "SWRs",
            "matched_control": "matched quiet-rest control windows (the noise floor)",
        }[word]
        values += [
            make(
                f"effect_abs_peak_median_{word}",
                stats["abs_peak_median"],
                "quantity",
                "dec2",
                f"median over units of the absolute peak log-rate modulation for {label}",
                "log rate",
            ),
            make(
                f"effect_rate_multiplier_median_{word}",
                stats["rate_multiplier_median"],
                "quantity",
                "sig2",
                f"exp of that median: the median unit's largest rate change for {label}, as "
                "a multiplier (up or down)",
                "x",
            ),
            make(
                f"effect_suppressed_fraction_{word}",
                stats["suppressed_fraction"],
                "fraction",
                "pct",
                f"share of units whose peak modulation for {label} is a decrease",
            ),
        ]
    for word, called_median in (
        ("swr_replay", overlap["abs_peak_median_called_swr"]),
        ("non_swr_replay", overlap["abs_peak_median_called_non_swr"]),
    ):
        values.append(
            make(
                f"effect_abs_peak_median_called_{word}",
                called_median,
                "quantity",
                "dec2",
                f"median absolute peak log-rate modulation of the units called for "
                f"{'SWR-coupled' if word == 'swr_replay' else 'non-SWR'} replay, "
                "among the units tested for both replay classes",
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
      curve, with a 95 % interval from resampling recording days (NWB files),
      and the animals
      whose own mean curve has its onset before event onset.

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
            f"recording-day resamples (seed {TIMING_BOOTSTRAP_SEED})",
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
                        f"over {TIMING_BOOTSTRAP_DRAWS:,} recording-day resamples",
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
                f"mean curve of {who} before event onset",
            )
            add(
                f"timing_population_steepest_animals_negative_{stem}",
                stats["n_animals_steepest_before_event"],
                "count",
                "count",
                f"of those animals, how many have the steepest approach of their own "
                f"mean curve of {who} before event onset",
            )
            add(
                f"timing_population_rise_onset_animals_negative_{stem}",
                stats["n_animals_rise_onset_before_event"],
                "count",
                "count",
                f"of those animals, how many have the rise onset of their own mean curve "
                f"of {who} before event onset",
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
                f"the same for {animal}'s tested units {where}",
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


# --- marked family: content, strength, ripple power --------------------------------


def marked_family_values(
    summary: pd.DataFrame, make: Callable[..., Value]
) -> list[Value]:
    """How the mPFC response changes with what is replayed and how strong the event is.

    Read from the marked-family summary of the follow-up joint models (fit with
    a weak prior on the per-stop baseline columns, every unit with a spike
    kept, no rate gate). For each model, target mark and event class: the
    population slope at its peak (log-rate change of the peri-event response
    per within-session SD of the mark; unit -> session -> animal), the curve's
    mean over 0 to +150 ms, and how many animals' own 0 to +150 ms means share
    its sign. For the marks a sentence quotes (``MARKED_HELDOUT_SHARES``), per
    replay class, the share of units whose held-out deviance improves when the
    mark's kernel is in the model (animal-balanced). No shuffle reference was
    run, so the share expected by chance is unknown and the share is not a test.
    Beside them, the within-session correlation of
    ripple power and multiunit rate in each replay class: under the shared
    ridge prior a small slope on either can carry part of the other's effect,
    with the same sign in every animal (possible leakage). Full tables are in
    the summary's CSV and parquet.
    """
    values: list[Value] = []

    add = _finite_adder(values, make)

    wide = summary[summary["statistic"] != "mark_correlation"].pivot_table(
        index=["model", "event_class", "target"], columns="statistic", values="value"
    )
    for model_word, model in MARKED_MODELS.items():
        for (row_model, event_class, target), row in wide.iterrows():
            if row_model != model or target not in MARKED_TARGETS:
                continue
            stem = (
                f"marked_{model_word}_{MARKED_TARGETS[target]}_"
                f"{MARKED_CLASSES[event_class]}"
            )
            what = (
                f"{target} in model {model}, "
                f"{'all classes pooled' if event_class == 'pooled' else event_class}"
            )
            add(
                f"{stem}_peak",
                row["peak_value"],
                "quantity",
                "sig2",
                f"population slope at its peak: change of the peri-event log rate per "
                f"within-session SD of {what} (unit -> session -> animal)",
                "log rate",
            )
            add(
                f"{stem}_peak_lag_ms",
                row["peak_lag_ms"],
                "quantity",
                "int",
                f"lag of that peak, {what}",
                "ms",
            )
            add(
                f"{stem}_post_onset_mean",
                row["post_onset_mean"],
                "quantity",
                "sig2",
                f"mean of the population slope curve from 0 to +150 ms after onset, {what}",
                "log rate",
            )
            add(
                f"{stem}_post_onset_animals_same_sign",
                row["post_onset_animals_same_sign"],
                "count",
                "count",
                f"animals whose own mean slope from 0 to +150 ms after onset has the "
                f"sign of the population's, {what} (a window fixed in advance; the "
                "count at the peak lag is not reported, since the lag is chosen where "
                "the population curve is largest and the animals agree there by "
                "selection)",
            )
            if (model, target) in MARKED_HELDOUT_SHARES and (
                event_class in MARKED_HELDOUT_SHARE_CLASSES
            ):
                add(
                    f"{stem}_heldout_gain_fraction",
                    row["frac_pos"],
                    "fraction",
                    "pct",
                    f"share of units whose held-out deviance improves when the mark's "
                    f"kernel is in the model, {what} (per session, median over sessions "
                    "within animal, median over animals; over "
                    f"{row['n_usable']:.0f} usable of {row['n_units']:.0f} unit fits, "
                    f"{row['n_excluded_blowup']:.0f} left out as blow-ups, in "
                    f"{row['n_sessions']:.0f} sessions of {row['n_animals']:.0f} "
                    "animals). No chance reference was computed, so the share expected "
                    "by chance is unknown: this is not a test",
                )
    correlations = summary[
        (summary["statistic"] == "mark_correlation")
        & (summary["model"] == "event_strength")
        & (summary["target"] == "ripple_power_z & mua_rate_z")
    ]
    for _, row in correlations.iterrows():
        add(
            f"marked_ripple_mua_correlation_{MARKED_CLASSES[row['event_class']]}",
            row["value"],
            "quantity",
            "dec2",
            "within-session correlation of ripple power and hippocampal multiunit rate "
            f"over {row['event_class']} events (mean over animals of session means): "
            "their slopes may carry each other's effect (ridge leakage)",
        )
    animals = wide["n_animals"].max() if "n_animals" in wide else np.nan
    add(
        "marked_animals",
        animals,
        "count",
        "count",
        "animals in the marked-family summary",
    )
    return values


# --- stop context ---------------------------------------------------------------------


def stop_context_values(
    results: pd.DataFrame, make: Callable[..., Value]
) -> list[Value]:
    """Does the response to replay differ with the stop's reward outcome or a coming
    patch switch?

    Read from the stop-level context fits (``analysis/pfc/stop_context.py``): a
    Poisson model, pooled over sessions, of each unit's spike count in the
    ``[0, +200) ms`` window after each replay onset against its rate in the
    same stop's event-free fitted time (the stop baseline estimated
    conditionally), with free unit terms, the stop's known reward outcome, the
    binary next-trip-switches-patch mark and the count of overlapping events;
    fit for SWR-coupled and non-SWR replay, for units whose canonical peak is an
    increase or a decrease. Main numbers: units called for the class
    (conditional estimator); all units are a check (reward, increases only).
    ``exp(beta)`` is the multiplicative change of the post-onset response
    relative to the stop baseline at rewarded stops (or before a patch-leaving
    trip); intervals are 95 % intervals clustered by recording day (mPFC units
    recur in every epoch of a day; CR1 factor, t with ``G - 1`` degrees of
    freedom); animals counted with the pooled sign out of those estimated;
    leave-one-animal-out range, which needs every leave-out to have an estimate.
    """
    values: list[Value] = []

    add = _finite_adder(values, make)

    rows = results[results["estimator"] == "conditional"]
    classes = {"non_local_SWR": "swr_replay", "non_local_no_SWR": "non_swr_replay"}
    terms = {"rewarded": "reward", "next_trip_switches_patch": "switch"}
    for (event_class, sign, unit_set, term), row in rows.set_index(
        ["event_class", "sign", "unit_set", "term"]
    ).iterrows():
        if event_class not in classes or term not in terms:
            continue
        if unit_set == "all" and not (term == "rewarded" and sign == "increase"):
            continue
        stem = f"stop_context_{terms[term]}_gain_{classes[event_class]}_{sign}"
        if unit_set == "all":
            stem += "_all_units"
        what = (
            f"{'rewarded stops' if term == 'rewarded' else 'stops before a patch-leaving trip'}"
            f", {event_class}, units whose peak is {'an increase' if sign == 'increase' else 'a decrease'}"
            f", {'called' if unit_set == 'called' else 'all'} units"
        )
        add(
            stem,
            row["exp_coef"],
            "quantity",
            "dec2",
            f"multiplicative change of the post-onset response relative to the stop "
            f"baseline at {what}",
            "x",
        )
        add(
            f"{stem}_ci_low",
            row["ci_low"],
            "quantity",
            "dec2",
            f"lower end of the recording-day-clustered 95 % interval of that (t with "
            f"G - 1 degrees of freedom), {what}",
            "x",
        )
        add(
            f"{stem}_ci_high",
            row["ci_high"],
            "quantity",
            "dec2",
            f"upper end of that interval, {what}",
            "x",
        )
        if unit_set == "called" and sign == "increase":
            if row["n_loao_estimated"] != row["n_animals"]:
                raise ValueError(
                    f"{stem}: {row['n_loao_estimated']:.0f} of {row['n_animals']:.0f} "
                    "leave-one-animal-out fits have an estimate"
                )
            add(
                f"{stem}_animals_pooled_sign",
                row["n_animals_pooled_sign"],
                "count",
                "count",
                f"animals whose own estimate has the pooled sign, {what}",
            )
            add(
                f"{stem}_animals_estimated",
                row["n_animals_estimated"],
                "count",
                "count",
                f"animals with an estimate, {what}",
            )
            add(
                f"{stem}_loao_min",
                row["loao_exp_coef_min"],
                "quantity",
                "dec2",
                f"smallest leave-one-animal-out estimate, {what}",
                "x",
            )
            add(
                f"{stem}_loao_max",
                row["loao_exp_coef_max"],
                "quantity",
                "dec2",
                f"largest leave-one-animal-out estimate, {what}",
                "x",
            )
    return values


# --- the 1 Hz check ----------------------------------------------------------------


def rate_cutoff_check_values(
    population: pd.DataFrame,
    extended: tuple[pd.DataFrame, dict, np.ndarray, list] | None,
    make: Callable[..., Value],
) -> list[Value]:
    """A few main values recomputed with a :data:`CHECK_FIRING_RATE_HZ` cutoff.

    The same functions as the main values, on the population restricted to
    units at or above the cutoff (``immobility_avg_firing_rate``); only the keys
    of :data:`CHECK_KEYS` are kept, renamed ``check_one_hz_<key>``.
    """
    gated, _ = population_statistics.units_at_or_above_rate(
        population, CHECK_FIRING_RATE_HZ
    )
    computed = [
        make(
            "units",
            len(gated),
            "count",
            "count",
            "mPFC units in the population (a usable fit and finite curves)",
        )
    ]
    computed += called_fraction_values(gated, make)
    computed += overlap_and_effect_values(gated, make)
    if extended is not None:
        ext_population, ext_curves, ext_lag, ext_sidecars = extended
        ext_gated, ext_gated_curves = population_statistics.units_at_or_above_rate(
            ext_population, CHECK_FIRING_RATE_HZ, ext_curves
        )
        computed += swr_without_replay_values(
            ext_gated, ext_gated_curves, ext_lag, ext_sidecars, make
        )
    values = []
    for value in computed:
        if value.key in CHECK_KEYS:
            values.append(
                replace(
                    value,
                    key=f"check_one_hz_{value.key}",
                    description=f"check, units at or above "
                    f"{CHECK_FIRING_RATE_HZ:g} Hz on the fitted bins: "
                    + value.description,
                )
            )
    return values


def followup_values(
    inputs: MpfcInputs,
    current: dict[str, str],
    n_cohort: int,
) -> tuple[list[Value], list[dict[str, Any]]]:
    """The marked-family and stop-context values, each pending when its inputs
    are stale (:func:`marked_family_staleness`, :func:`stop_context_staleness`),
    and the records of the files read. A missing output gives no values."""
    values: list[Value] = []
    records: list[dict[str, Any]] = []
    if inputs.marked_family_summary.exists():
        values += marked_family_values(
            pd.read_parquet(inputs.marked_family_summary),
            value_factory(
                marked_family_values,
                f"{rel(inputs.marked_family_summary)} "
                "(scripts/pfc/summarize_pfc_marked_family.py over "
                "scripts/pfc/run_pfc_marked_model_family.py --model-set followup "
                "--baseline-penalty-sigma 3 --slope-permutations 0 bundles)",
                TAGS["modulation"],
                marked_family_staleness(
                    inputs.marked_family_summary, current, n_cohort
                ),
            ),
        )
        records.append(file_record(inputs.marked_family_summary, ROOT_DIR))
    if inputs.stop_context_results.exists():
        results = pd.read_csv(inputs.stop_context_results)
        if not results["n_sessions"].eq(n_cohort).all():
            raise ValueError(
                f"{inputs.stop_context_results}: a fit covers "
                f"{sorted(results['n_sessions'].unique())} sessions, not the "
                f"{n_cohort} of the PFC cohort"
            )
        values += stop_context_values(
            results,
            value_factory(
                stop_context_values,
                f"{rel(inputs.stop_context_results)} "
                "(scripts/pfc/run_pfc_stop_context.py fit; conditional estimator)",
                TAGS["modulation"],
                stop_context_staleness(
                    inputs.stop_context_results, inputs.bundle_dir, current, n_cohort
                ),
            ),
        )
        records.append(file_record(inputs.stop_context_results, ROOT_DIR))
    return values, records


def summarize(inputs: MpfcInputs, produced_by: str) -> Summary:
    """The mpfc summary from ``inputs``."""
    summary = Summary(name="mpfc", produced_by=produced_by)
    sidecars = read_sidecars(inputs.bundle_dir)
    current = table_hashes(inputs.event_table, inputs.stop_table)
    pending = staleness(sidecars, current)
    population, curves, lag_time = load_population_data(
        inputs.bundle_dir, min_firing_rate=MIN_FIRING_RATE_HZ
    )
    cohort = json.loads(inputs.pfc_cohort.read_text())
    source = (
        f"{rel(inputs.bundle_dir)}/*{STEM}.pkl and their provenance sidecars "
        f"(scripts/pfc/run_pfc_analysis_bandit.py), {len(sidecars)} bundles, "
        "read by Figure 4's loader (analysis/pfc/load_population.py, no "
        "firing-rate cutoff)"
    )
    main = TAGS["modulation"]
    summary.add(
        *cohort_values(
            sidecars,
            population,
            cohort,
            value_factory(cohort_values, source, main, pending),
        ),
        *estimator_values(
            sidecars, value_factory(estimator_values, source, main, pending)
        ),
        *prior_binding_values(
            population, value_factory(prior_binding_values, source, main, pending)
        ),
        *called_fraction_values(
            population,
            value_factory(called_fraction_values, source, TAGS["figure_4d"], pending),
        ),
        *null_geometry_values(
            population,
            sidecars,
            value_factory(null_geometry_values, source, main, pending),
        ),
        *overlap_and_effect_values(
            population, value_factory(overlap_and_effect_values, source, main, pending)
        ),
    )
    calibration = (
        json.loads(inputs.timing_calibration.read_text())
        if inputs.timing_calibration.exists()
        else None
    )
    summary.add(
        *timing_values(
            population,
            curves,
            lag_time,
            calibration,
            value_factory(timing_values, source, main, pending),
        )
    )
    values, records = followup_values(inputs, current, cohort["n_pfc_cohort"])
    summary.add(*values)
    summary.inputs += records
    extended_dir = inputs.swr_without_replay_dir
    extended_bundles = sorted(extended_dir.glob(f"*{STEM}.pkl"))
    extended_check = None
    if extended_bundles:
        extended_sidecars = read_sidecars(extended_dir)
        incomplete = len(extended_sidecars) < len(sidecars)
        extended_pending = staleness(extended_sidecars, current) or (
            f"the extended-event-set batch holds {len(extended_sidecars)} of "
            f"{len(sidecars)} sessions; refresh when it completes"
            if incomplete
            else None
        )
        extended_population, extended_curves, _ = load_population_data(
            extended_dir,
            min_firing_rate=MIN_FIRING_RATE_HZ,
            extra_curves=SWR_WITHOUT_REPLAY_CURVES,
        )
        extended_check = (
            extended_population,
            extended_curves,
            lag_time,
            extended_sidecars,
        )
        summary.add(
            *swr_without_replay_values(
                extended_population,
                extended_curves,
                lag_time,
                extended_sidecars,
                value_factory(
                    swr_without_replay_values,
                    f"{rel(extended_dir)}/*{STEM}.pkl and sidecars "
                    "(scripts/pfc/run_pfc_analysis_bandit.py --event-set "
                    f"with_swr_without_replay), {len(extended_sidecars)} bundles, "
                    "Figure 4's loader (no firing-rate cutoff)",
                    TAGS["modulation"],
                    extended_pending,
                ),
            )
        )
        summary.inputs += [
            files_record(f"*{STEM}.pkl", extended_bundles, ROOT_DIR),
            files_record(
                f"*{STEM}.*.npy",
                extended_dir.glob(f"*{STEM}.*.npy"),
                ROOT_DIR,
            ),
        ]
    summary.add(
        *rate_cutoff_check_values(
            population,
            extended_check,
            value_factory(rate_cutoff_check_values, source, main, pending),
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
    ]
    return summary
