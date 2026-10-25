#!/usr/bin/env python
"""Fit three replay-CONTENT estimators on one at-well cohort and compare them.

The content question -- which of the nine arms an immobility non-local event
represents -- has been asked with three different likelihoods, and the three do
not agree about how much an event is worth:

``conditional_poisson_count``
    The retired count model. Its conditioning group is the event, but its
    observation is the timebin, so an event that spent 80 bins carries 80 times
    the weight of an event that spent one, and its standard errors assume those
    bins are independent draws.
``conditional_poisson_fraction``
    The shipped model. The response is the event's per-arm allocation
    *fraction*, so every event weighs one, and the reported uncertainty is the
    event-clustered sandwich (``bse_cluster_<term>``) rather than the model SE.
    Its likelihood is a quasi-likelihood on the event scale: the AIC it implies
    is not a count-scale AIC and is labelled as such here.
``dirichlet_multinomial``
    The overdispersion-aware model. It keeps the counts, keeps the softmax mean
    structure, and adds an explicit within-event concentration ``log phi_e = g0
    + g1 * is_swr_e``. Its standard errors are event-level by construction, and
    ``coef_log_phi_is_swr`` is a direct test of whether SWR events spread their
    mass over more arms than non-SWR events.

All three are fit **on exactly the same events**: the cohort rules of the
shipped contrast runner are applied once per session for *both* response
columns and intersected, so a difference between estimators is a difference of
likelihood and never of cohort. The per-(session, rung) event counts are
asserted equal across the three, and the events the intersection dropped are
recorded.

Cross-estimator AIC is only meaningful once the likelihood constants match.
The statsmodels conditional likelihood omits the multinomial coefficient and
the Dirichlet-multinomial keeps it, so ``multinomial_log_constant`` -- computed
from the COUNT matrix in both cases -- is added to the two conditional fits
before ``aic_equalized`` is formed. ``pseudo_r2`` has a different null under
each estimator and is deliberately not tabulated.

Inference is the project's standard: five animals, sign consistency, and an
animal bootstrap (resample the animals present with replacement, statistic =
median over the resampled animals of each animal's per-session median).

Usage:
    uv run python scripts/content/compare_content_estimators.py
    uv run python scripts/content/compare_content_estimators.py --max-per-animal 1 --n-jobs 4
    uv run python scripts/content/compare_content_estimators.py --rungs C2_base
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from continuum_swr_replay.analysis.replay._fit_diagnostics import (  # noqa: E402
    SEPARATION_COEF_ABS,
)
from continuum_swr_replay.analysis.replay.content_response import (  # noqa: E402
    renormalize_fraction_over_candidates,
)

# The cohort this comparison runs on is the shipped contrast runner's own at-well
# cohort, assembled by the same helpers rather than re-derived here, so the two
# cannot drift.
from continuum_swr_replay.analysis.replay.contrast_aggregation import (  # noqa: E402
    MIN_EVENTS,
    assemble_arm,
    contrast_fit_inputs,
    prepare_session,
    select_outcome_stratum,
)
from continuum_swr_replay.analysis.replay.dirichlet_multinomial import (  # noqa: E402
    DEFAULT_MIN_EVENTS_SESSION_PHI,
    DEFAULT_N_JOBS,
    LOG_PHI_INTERCEPT,
    concentration_cols_for_cohort,
    fit_sessions_dirichlet_multinomial,
    multinomial_log_constant,
)
from continuum_swr_replay.analysis.replay.dirichlet_multinomial import (  # noqa: E402
    ESTIMATOR as DIRICHLET_MULTINOMIAL,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (  # noqa: E402
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.replay.fitting import (  # noqa: E402
    CONDITIONAL_POISSON_COUNT,
    CONDITIONAL_POISSON_FRACTION,
    fit_conditional_poisson,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    LEAF_ONLY_CONTRAST_RUNGS,
    LEAF_ONLY_N_CATEGORIES,
    MODEL_HIERARCHY,
    leaf_only_predictors,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    SWR_CONTRAST_CONTENT_ARM_KEY_TERM as ARM_KEY_TERM,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    SWR_CONTRAST_CONTENT_ARM_LADDER as ARM_LADDER,
)
from continuum_swr_replay.paths import figure_dir, replay_content_dir
from continuum_swr_replay.tables.io import write_table  # noqa: E402
from continuum_swr_replay.tables.session_keys import (  # noqa: E402
    SESSION_COLUMNS,
    subset_max_per_animal,
)

# Quasi-separated coefficients legitimately overflow exp() outside the fit. Fit
# quality itself is carried per fit as converged/separated flags, so only the
# benign RuntimeWarning noise is silenced -- not a blanket ignore.
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SC = list(SESSION_COLUMNS)

COEFFICIENTS_BASENAME = "content_estimator_comparison"
SUMMARY_BASENAME = "content_estimator_comparison_summary"
BOOTSTRAP_BASENAME = "content_estimator_bootstrap"
SKIPS_BASENAME = "content_estimator_comparison_skips"
PROVENANCE_ESTIMATOR = "three_estimator_comparison"

FRACTION_RESPONSE = "arm_fraction"
COUNT_RESPONSE = "arm_count_hard"
N_ARM_CATEGORIES = 9

#: Short labels used in the summary/bootstrap column names.
ESTIMATOR_LABELS = {
    CONDITIONAL_POISSON_COUNT: "count",
    CONDITIONAL_POISSON_FRACTION: "fraction",
    DIRICHLET_MULTINOMIAL: "dm",
}
ESTIMATORS = tuple(ESTIMATOR_LABELS)

#: The fraction fit's AIC is a quasi-likelihood number on the event scale; the
#: other two are true count-scale log-likelihoods once the constant is added.
QUASI_SCALE = "event-scale (quasi)"
LIKELIHOOD_SCALE = "count-scale"

#: Which standard error each estimator's rows carry. The fraction fit reports the
#: event-clustered sandwich and the other two report the fitted model SE, so an SE
#: RATIO between them is not one kind of uncertainty over the same kind -- the
#: column name has to say so, or "the fraction model is more conservative than the
#: DM" gets read off a comparison of a sandwich against a likelihood.
SE_TYPE_LABELS = {
    "count": "count_model",
    "fraction": "fraction_cluster",
    "dm": "dm_model",
}


def se_ratio_column(label: str) -> str:
    """Name of the SE-ratio column for estimator ``label``, SE types spelled out."""
    return f"se_ratio_{SE_TYPE_LABELS[label]}_over_{SE_TYPE_LABELS['count']}"


N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
BOOTSTRAP_PERCENTILES = (2.5, 50.0, 97.5)

LOG_PHI_IS_SWR = "log_phi_is_swr"

#: Key term the agreement figure plots for each ``MODEL_HIERARCHY`` rung: the
#: LAST predictor the rung adds to its parent. ``M2`` and ``M3`` add exactly one
#: (``is_prev_arm``, the arm reward fraction); ``M4`` adds four -- the three
#: recency-spline columns and ``is_candidate_arm_never_visited`` -- and the
#: novelty indicator is the interpretable one of those. Every added term is
#: still in the parquet and the summary; this map only chooses figure panels.
#: The contrast rungs take theirs from the runner's own key-term map.
_HIERARCHY_KEY_TERMS = {
    "M2": "is_prev_arm",
    "M3": "reward_fraction_candidate_arm_last10_visits",
    "M4": "is_candidate_arm_never_visited",
}


@dataclass(frozen=True)
class RungSpec:
    """One model to fit with all three estimators.

    Attributes
    ----------
    name : str
        Row label in every output.
    predictors : tuple of str
        Category-varying predictor columns, already reduced by
        :func:`leaf_only_predictors` where ``leaf_only`` is set.
    parent : str or None
        Rung whose ``aic_equalized`` this one is differenced against, per
        session and per estimator. ``None`` for a rung fit only as a parent.
    leaf_only : bool
        Fit over the six leaf arms only (trunks carry structural zeros for the
        belief covariates). The fraction response is renormalized to unit mass
        over the retained leaves; the counts keep their timebin weighting.
    key_term : str or None
        The coefficient the agreement figure plots for this rung.
    """

    name: str
    predictors: tuple[str, ...]
    parent: str | None = None
    leaf_only: bool = False
    key_term: str | None = None

    @property
    def n_categories(self) -> int:
        """Candidate categories per event under this rung's likelihood."""
        return LEAF_ONLY_N_CATEGORIES if self.leaf_only else N_ARM_CATEGORIES


def _hierarchy_rung(name: str, parent: str | None) -> RungSpec:
    """A ``MODEL_HIERARCHY`` rung as a comparison spec (never leaf-only)."""
    return RungSpec(
        name=name,
        predictors=tuple(MODEL_HIERARCHY[name].predictors),
        parent=parent,
        leaf_only=False,
        key_term=_HIERARCHY_KEY_TERMS.get(name),
    )


def _contrast_rung(
    name: str, parent: str | None, *, spec_name: str | None = None
) -> RungSpec:
    """An is_swr-contrast rung as a comparison spec, honouring leaf-only scope."""
    source = spec_name or name
    predictors, _scope = ARM_LADDER[source]
    leaf_only = source in LEAF_ONLY_CONTRAST_RUNGS or name.endswith("_leaf")
    resolved = leaf_only_predictors(predictors) if leaf_only else list(predictors)
    return RungSpec(
        name=name,
        predictors=tuple(resolved),
        parent=parent,
        leaf_only=leaf_only,
        key_term=ARM_KEY_TERM.get(source),
    )


# M2 and the two C2_base fits are parents only: they carry no ``delta_aic``
# row of their own. ``C2_base_leaf`` is the six-leaf refit of the same base --
# a leaf-only rung's AIC lives on a six-category likelihood and cannot be
# differenced against a nine-arm one, so ``CS5_armvalue`` is given a parent on
# its own candidate set instead of a nominally-nested but incomparable number.
COMPARISON_RUNGS: tuple[RungSpec, ...] = (
    _hierarchy_rung("M2", None),
    _hierarchy_rung("M3", "M2"),
    _hierarchy_rung("M4", "M3"),
    _contrast_rung("C2_base", None),
    _contrast_rung("C2_base_leaf", None, spec_name="C2_base"),
    _contrast_rung("CS1_current", "C2_base"),
    _contrast_rung("CS4_armreward", "C2_base"),
    _contrast_rung("CS5_armvalue", "C2_base_leaf"),
)

#: Rung whose Dirichlet-multinomial concentration the log-phi figure shows.
DEFAULT_LOG_PHI_RUNG = "C2_base"


def log_phi_figure_rung(rungs: Sequence[RungSpec]) -> str | None:
    """The rung the concentration figure is drawn for, or ``None`` if unselected.

    ``--rungs`` can legitimately select a set that excludes
    :data:`DEFAULT_LOG_PHI_RUNG`. That is a partial run, not a failure: the
    figure is pinned to one rung because its caption names that rung, and
    silently substituting another would publish a figure whose caption describes
    a different model. Returning ``None`` lets the caller skip the panel and
    still finish writing its tables and provenance.
    """
    return (
        DEFAULT_LOG_PHI_RUNG
        if DEFAULT_LOG_PHI_RUNG in {rung.name for rung in rungs}
        else None
    )


def resolve_rungs(names: Sequence[str] | None) -> tuple[RungSpec, ...]:
    """Select comparison rungs by name, pulling in every parent they need.

    Parameters
    ----------
    names : sequence of str, optional
        Rung names. ``None`` selects :data:`COMPARISON_RUNGS` whole.

    Returns
    -------
    tuple of RungSpec
        In :data:`COMPARISON_RUNGS` order, so a parent is always fit before the
        rung that differences against it.
    """
    catalog = {rung.name: rung for rung in COMPARISON_RUNGS}
    if names is None:
        return COMPARISON_RUNGS
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError(f"unknown rung(s) {unknown}; available: {sorted(catalog)}")
    wanted = set(names)
    for name in list(wanted):
        parent = catalog[name].parent
        if parent is not None:
            wanted.add(parent)
    return tuple(rung for rung in COMPARISON_RUNGS if rung.name in wanted)


@dataclass(frozen=True)
class ComparisonResult:
    """Everything one comparison run produces.

    Attributes
    ----------
    coefficients : pandas.DataFrame
        One row per (session, rung, term, estimator).
    summary : pandas.DataFrame
        One row per (rung, term).
    bootstrap : pandas.DataFrame
        One row per (rung, term, estimator) with the animal-bootstrap
        percentiles.
    skips : pandas.DataFrame
        One row per (session, rung) that produced no comparable triple.
    """

    coefficients: pd.DataFrame
    summary: pd.DataFrame
    bootstrap: pd.DataFrame
    skips: pd.DataFrame


# ── one shared event set per (session, rung) ────────────────────────────────


def _rung_frame(
    cohort: pd.DataFrame, rung: RungSpec, *, fraction_col: str
) -> pd.DataFrame:
    """Restrict and rescale the cohort to what one rung is fit on.

    A leaf-only rung drops the three trunk arms. The fraction response is a
    share of the event's whole nine-arm mass and the conditional likelihood
    weights an event by the total it carries, so it is renormalized to unit mass
    over the six retained leaves; an event with no leaf mass renormalizes to NaN
    and is then dropped from all three estimators by
    :func:`_shared_session_rows`. The counts keep their timebin weighting, which
    is also the ``L_e`` the Dirichlet-multinomial and the multinomial constant
    are stated on.
    """
    if not rung.leaf_only:
        return cohort
    frame = cohort[
        pd.to_numeric(cohort["candidate_arm"], errors="coerce") < LEAF_ONLY_N_CATEGORIES
    ]
    return renormalize_fraction_over_candidates(frame, response_col=fraction_col)


def _shared_session_rows(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    *,
    fraction_col: str,
    count_col: str,
) -> tuple[pd.DataFrame | None, dict]:
    """Freeze one event set that both responses (hence all three fits) can use.

    The shipped runner's own cohort rules are applied twice -- once per response
    column -- and intersected. A difference between the estimators is then a
    difference of likelihood and never of which events they saw.

    Returns
    -------
    (rows_or_None, info)
        ``info`` carries ``n_events_in``, the per-response survivor counts, the
        shared count, and a skip ``reason`` when there is nothing to fit.
    """
    predictors = list(predictors)
    fraction_rows, fraction_prep = prepare_session(
        frame, predictors, response_col=fraction_col
    )
    count_rows, count_prep = prepare_session(frame, predictors, response_col=count_col)
    info = {
        "n_events_in": int(fraction_prep["n_in"]),
        "n_events_fraction": int(fraction_prep["n_surv"]),
        "n_events_count": int(count_prep["n_surv"]),
        "n_events_shared": 0,
        "reason": "",
    }
    if fraction_rows is None or count_rows is None:
        info["reason"] = fraction_prep["reason"] or count_prep["reason"] or "no_rows"
        return None, info

    shared = set(fraction_rows["event_id"]) & set(count_rows["event_id"])
    info["n_events_shared"] = len(shared)
    if len(shared) < MIN_EVENTS:
        info["reason"] = "too_few_shared"
        return None, info
    return frame[frame["event_id"].isin(shared)], info


def _drop_undefined_concentration(
    frame: pd.DataFrame, concentration_cols: Sequence[str]
) -> tuple[pd.DataFrame, int]:
    """Drop events whose concentration covariate is missing, for every estimator.

    The Dirichlet-multinomial drops an event with a NaN ``is_swr`` (it is a
    required column of its pack) while the conditional fits never look at it, so
    leaving such an event in would desynchronize the three event sets. Removing
    it up front keeps them identical.
    """
    if not concentration_cols:
        return frame, 0
    missing = frame[list(concentration_cols)].isna().any(axis=1)
    if not missing.any():
        return frame, 0
    bad = set(frame.loc[missing, "event_id"])
    return frame[~frame["event_id"].isin(bad)], len(bad)


# ── per-rung fitting ────────────────────────────────────────────────────────


def _fit_rung(
    cohort: pd.DataFrame,
    rung: RungSpec,
    *,
    fraction_col: str,
    count_col: str,
    n_jobs: int,
    min_events_session_phi: int,
) -> tuple[list[dict], list[dict]]:
    """Fit one rung with all three estimators on one shared event set per session.

    Returns
    -------
    (rows, skips)
        ``rows`` is one dict per (session, term, estimator); ``skips`` is one
        dict per session that produced no complete triple.

    Raises
    ------
    ValueError
        If the three estimators report different ``n_events`` for a session --
        the shared cohort is the contract this comparison rests on.
    """
    predictors = list(rung.predictors)
    frame = _rung_frame(cohort, rung, fraction_col=fraction_col)
    concentration_cols = concentration_cols_for_cohort(frame)
    frame, n_drop_concentration = _drop_undefined_concentration(
        frame, concentration_cols
    )
    if n_drop_concentration:
        logger.info(
            "%s: dropped %d event(s) with a missing concentration covariate",
            rung.name,
            n_drop_concentration,
        )

    prepared: list[tuple[tuple, pd.DataFrame, dict]] = []
    skips: list[dict] = []
    for keys, group in frame.groupby(SC, sort=False):
        rows, info = _shared_session_rows(
            group,
            predictors,
            fraction_col=fraction_col,
            count_col=count_col,
        )
        key = tuple(keys)
        if rows is None:
            skips.append(_skip_record(key, rung, info, info["reason"]))
            continue
        prepared.append((key, rows, info))

    if not prepared:
        return [], skips

    count_fits = {
        key: fit_conditional_poisson(
            rows,
            predictors=predictors,
            response_col=count_col,
            event_col="event_id",
            n_categories=rung.n_categories,
        )
        for key, rows, _info in prepared
    }
    fraction_fits = {
        key: fit_conditional_poisson(
            rows,
            predictors=predictors,
            response_col=fraction_col,
            event_col="event_id",
            n_categories=rung.n_categories,
        )
        for key, rows, _info in prepared
    }
    animal_position = SC.index("animal")
    dm_fitted, dm_skipped = fit_sessions_dirichlet_multinomial(
        [(key, key[animal_position], rows) for key, rows, _info in prepared],
        predictors,
        min_events=min_events_session_phi,
        n_jobs=n_jobs,
        response_col=count_col,
        event_col="event_id",
        concentration_cols=concentration_cols,
        n_categories=rung.n_categories,
    )
    dm_fits = dict(dm_fitted)
    dm_reasons = dict(dm_skipped)

    rows_out: list[dict] = []
    for key, session_rows, info in prepared:
        fits = {
            CONDITIONAL_POISSON_COUNT: count_fits.get(key),
            CONDITIONAL_POISSON_FRACTION: fraction_fits.get(key),
            DIRICHLET_MULTINOMIAL: dm_fits.get(key),
        }
        absent = [name for name, fit in fits.items() if fit is None]
        if absent:
            reason = "no_fit:" + ",".join(ESTIMATOR_LABELS[name] for name in absent)
            dm_reason = dm_reasons.get(key, "")
            skips.append(
                _skip_record(
                    key,
                    rung,
                    info,
                    f"{reason} ({dm_reason})" if dm_reason else reason,
                )
            )
            continue

        n_events = {name: int(fit["n_events"]) for name, fit in fits.items()}
        if len(set(n_events.values())) != 1:
            raise ValueError(
                f"{rung.name} {key}: the three estimators were fit on different "
                f"event sets ({n_events}); the comparison requires one shared "
                "cohort per (session, rung)"
            )
        log_constant = multinomial_log_constant(
            session_rows[count_col], session_rows["event_id"]
        )
        rows_out.extend(
            _estimator_rows(
                key,
                rung,
                fits,
                info=info,
                log_constant=log_constant,
                n_events=next(iter(n_events.values())),
            )
        )
    n_compared = len({tuple(row[column] for column in SC) for row in rows_out})
    logger.info(
        "%s: %d session(s) compared, %d skipped", rung.name, n_compared, len(skips)
    )
    return rows_out, skips


def _skip_record(key: tuple, rung: RungSpec, info: dict, reason: str) -> dict:
    """One audited (session, rung) that produced no comparable estimator triple."""
    return {
        **dict(zip(SC, key, strict=True)),
        "model": rung.name,
        **{name: value for name, value in info.items() if name != "reason"},
        "reason": reason,
    }


def _estimator_rows(
    key: tuple,
    rung: RungSpec,
    fits: dict[str, dict],
    *,
    info: dict,
    log_constant: float,
    n_events: int,
) -> list[dict]:
    """One row per (term, estimator) for a session whose three fits all landed."""
    rows: list[dict] = []
    for estimator, fit in fits.items():
        equalized = fit["llf"] + (
            0.0 if fit["llf_includes_constants"] else log_constant
        )
        aic_equalized = 2.0 * fit["n_params"] - 2.0 * equalized
        separated_terms = str(fit.get("separated_terms") or "")
        separated = set(separated_terms.split("|")) - {""}
        for name in [k[len("coef_") :] for k in fit if k.startswith("coef_")]:
            clustered = fit.get(f"bse_cluster_{name}")
            is_fraction = estimator == CONDITIONAL_POISSON_FRACTION
            if is_fraction and clustered is None:
                raise ValueError(
                    f"{rung.name} {key}: the conditional-Poisson fraction fit "
                    f"carries no bse_cluster_{name}; the event-clustered "
                    "standard error is what this estimator reports"
                )
            # A non-finite sandwich (a predictor that is constant or nearly so
            # within one session) falls back to that session's model SE with
            # ``se_source == "model"``, exactly as the shipped aggregation's
            # ``session_se`` does, so the row stays in the medians on the same
            # footing the audited tables give it. Rows that fell back are
            # counted in the provenance sidecar.
            use_cluster = is_fraction and np.isfinite(float(clustered))
            se_source = (
                "event_clustered"
                if use_cluster
                else ("model_fallback" if is_fraction else "model")
            )
            rows.append(
                {
                    **dict(zip(SC, key, strict=True)),
                    "model": rung.name,
                    "term": name,
                    "estimator": estimator,
                    "coef": float(fit[f"coef_{name}"]),
                    "se": float(clustered if use_cluster else fit[f"bse_{name}"]),
                    "se_source": se_source,
                    "n_events": int(n_events),
                    "n_events_in": int(info["n_events_in"]),
                    "n_events_shared_dropped": int(info["n_events_in"] - n_events),
                    "n_params": int(fit["n_params"]),
                    "phi_source": fit.get("phi_source"),
                    "llf_equalized": float(equalized),
                    "aic_equalized": float(aic_equalized),
                    "scale": QUASI_SCALE
                    if estimator == CONDITIONAL_POISSON_FRACTION
                    else LIKELIHOOD_SCALE,
                    "leaf_only": bool(rung.leaf_only),
                    "parent": rung.parent,
                    # ``sd_<term>`` and the fit's separated-term names travel
                    # with the coefficient so the identifiability rule can be
                    # reproduced downstream from the table alone. The fitter
                    # emits ``sd_`` for the concentration modifiers too (from
                    # their event-level design) and NaN for the intercept.
                    "sd": float(fit[f"sd_{name}"]) if f"sd_{name}" in fit else np.nan,
                    "separated_terms": separated_terms,
                    "separated_term": name in separated,
                    "converged": bool(fit.get("converged")),
                }
            )
    return rows


def _attach_delta_aic(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Add ``delta_aic_vs_parent`` per (session, rung, estimator).

    The difference is only formed where the rung and its parent were fit on the
    same number of events in that session and estimator; an AIC difference
    across different rows is not a model comparison, so it is left NaN and
    counted (``n_delta_aic_row_mismatch``) rather than reported.
    """
    coefficients = coefficients.copy()
    coefficients["delta_aic_vs_parent"] = np.nan
    coefficients["delta_aic_row_mismatch"] = False
    if coefficients.empty:
        return coefficients

    fit_level = coefficients.drop_duplicates([*SC, "model", "estimator"])[
        [*SC, "model", "estimator", "aic_equalized", "n_events"]
    ]
    parent_level = fit_level.rename(
        columns={
            "model": "parent",
            "aic_equalized": "parent_aic_equalized",
            "n_events": "parent_n_events",
        }
    )
    merged = coefficients.merge(
        parent_level, on=[*SC, "parent", "estimator"], how="left"
    )
    matched = merged["parent_n_events"].eq(merged["n_events"])
    merged.loc[matched, "delta_aic_vs_parent"] = (
        merged.loc[matched, "aic_equalized"]
        - merged.loc[matched, "parent_aic_equalized"]
    )
    merged["delta_aic_row_mismatch"] = merged["parent_aic_equalized"].notna() & ~matched
    return merged.drop(columns=["parent_aic_equalized", "parent_n_events"])


# ── aggregation ─────────────────────────────────────────────────────────────


def unidentified_rows(coefficients: pd.DataFrame) -> pd.Series:
    """Per-row mask: is this term unidentified in this fit?

    The same rule the shipped aggregation applies
    (``contrast_aggregation.aggregate_contrast``): the fitter's own per-term
    separation flag, plus a magnitude backstop read on the EFFECT scale --
    ``|coef| * sd_<term>``, the log-rate swing the predictor actually produces
    over one SD of itself. Judging ``|coef|`` alone is selection on the estimate:
    it drops identified sessions whose predictor has a small SD and keeps
    unidentified ones whose predictor has a large one.

    The Dirichlet-multinomial's concentration MODIFIERS do carry a design column
    (the event-level ``is_swr`` flag, and the event length under the
    length-adjusted design), so the fitter persists their SD and the same
    effect-scale rule applies to them. ``log_phi_intercept`` does not: its design
    column is a constant, its SD is NaN by construction, and it is an unbounded
    log concentration anyway -- a session whose events really are
    near-deterministic has a legitimately large one -- so no magnitude rule is
    applied to it and the fitter's separation flag is the whole criterion there.
    """
    if coefficients.empty:
        return pd.Series(dtype=bool, index=coefficients.index)
    coef = coefficients["coef"].abs()
    sd = pd.to_numeric(coefficients["sd"], errors="coerce")
    # A missing SD makes the product NaN, which compares False -- so a term with
    # no design scale is never dropped for magnitude, only for its own flag.
    runaway = coef * sd >= SEPARATION_COEF_ABS
    # A NaN coefficient is a term the session cannot inform: the
    # conditional-Poisson fitter reports it so (``unidentified_terms``).
    not_estimated = coefficients["coef"].isna()
    return coefficients["separated_term"].astype(bool) | runaway | not_estimated


def _identified(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Rows a median may be taken over, dropped as PAIRED (session, rung, term).

    The identifiability rule fires per (session, rung, term, estimator), but a
    summary row compares estimators to each other: if one estimator's row is
    dropped and another's kept, the two medians would be over different
    sessions and the comparison would no longer be paired. The whole triple
    goes.
    """
    if coefficients.empty:
        return coefficients
    bad = unidentified_rows(coefficients)
    if not bad.any():
        return coefficients
    dropped = coefficients.loc[bad, [*SC, "model", "term"]].drop_duplicates()
    keep = (
        coefficients.merge(
            dropped.assign(_drop=True), on=[*SC, "model", "term"], how="left"
        )["_drop"]
        .isna()
        .to_numpy()
    )
    return coefficients[keep]


def n_unpaired_triples(coefficients: pd.DataFrame) -> int:
    """(session, rung, term) triples the identifiability rule removed."""
    if coefficients.empty:
        return 0
    bad = unidentified_rows(coefficients)
    return int(coefficients.loc[bad, [*SC, "model", "term"]].drop_duplicates().shape[0])


def _per_animal_medians(frame: pd.DataFrame, column: str) -> pd.Series:
    """Each animal's median of ``column`` over its own sessions."""
    return frame.groupby("animal", sort=True)[column].median().dropna()


def _animal_balanced_median(frame: pd.DataFrame, column: str) -> float:
    """Median over animals of each animal's per-session median."""
    per_animal = _per_animal_medians(frame, column)
    return float(per_animal.median()) if len(per_animal) else float("nan")


def _wide_by_estimator(coefficients: pd.DataFrame) -> pd.DataFrame:
    """One row per (rung, term, session) with each estimator's coef/se/ΔAIC."""
    values = ["coef", "se", "delta_aic_vs_parent"]
    wide = coefficients.set_index(["model", "term", *SC, "estimator"])[values].unstack(
        "estimator"
    )
    wide = wide.reindex(columns=pd.MultiIndex.from_product([values, list(ESTIMATORS)]))
    wide.columns = [
        f"{value}_{ESTIMATOR_LABELS[estimator]}" for value, estimator in wide.columns
    ]
    wide = wide.reset_index()
    for estimator in (CONDITIONAL_POISSON_FRACTION, DIRICHLET_MULTINOMIAL):
        label = ESTIMATOR_LABELS[estimator]
        with np.errstate(invalid="ignore", divide="ignore"):
            wide[se_ratio_column(label)] = np.where(
                wide["se_count"].to_numpy(dtype=float) > 0,
                wide[f"se_{label}"].to_numpy(dtype=float)
                / wide["se_count"].to_numpy(dtype=float),
                np.nan,
            )
    return wide


def summarize(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-session table to one row per (rung, term).

    Every median here is animal-balanced -- the median over animals of each
    animal's per-session median -- so the five animals weigh equally and the
    number matches the bootstrap statistic. ``pseudo_r2`` is deliberately
    absent: its null differs by estimator and it is not comparable across them.

    The ``dm_log_phi_is_swr_*`` columns are a property of the RUNG, not of the
    row's term, so they repeat across every term of a rung.
    """
    identified = _identified(coefficients)
    if identified.empty:
        return pd.DataFrame(columns=["model", "term"])
    wide = _wide_by_estimator(identified)
    # Sessions each (rung, term) lost to the identifiability rule: separated,
    # runaway, or not estimated at all (a NaN coefficient: the session cannot
    # inform the term). Their whole estimator triple is out of every median.
    unidentified = (
        coefficients.loc[unidentified_rows(coefficients), [*SC, "model", "term"]]
        .drop_duplicates()
        .groupby(["model", "term"])
        .size()
    )

    log_phi = identified[
        (identified["estimator"] == DIRICHLET_MULTINOMIAL)
        & (identified["term"] == LOG_PHI_IS_SWR)
    ]
    per_rung_log_phi = {
        name: _per_animal_medians(group, "coef")
        for name, group in log_phi.groupby("model", sort=False)
    }

    records = []
    for (model, term), group in wide.groupby(["model", "term"], sort=True):
        record = {
            "model": model,
            "term": term,
            "n_sessions_unidentified": int(unidentified.get((model, term), 0)),
        }
        for estimator in ESTIMATORS:
            label = ESTIMATOR_LABELS[estimator]
            coef = group[f"coef_{label}"]
            per_animal = _per_animal_medians(group.assign(_c=coef), "_c")
            record[f"n_sessions_{label}"] = int(coef.notna().sum())
            record[f"n_animals_{label}"] = int(len(per_animal))
            record[f"median_coef_{label}"] = (
                float(per_animal.median()) if len(per_animal) else np.nan
            )
            record[f"n_animals_pos_{label}"] = int((per_animal > 0).sum())
            record[f"median_delta_aic_{label}"] = _animal_balanced_median(
                group, f"delta_aic_vs_parent_{label}"
            )
            if estimator != CONDITIONAL_POISSON_COUNT:
                record[f"median_{se_ratio_column(label)}"] = _animal_balanced_median(
                    group, se_ratio_column(label)
                )
        animal_log_phi = per_rung_log_phi.get(model, pd.Series(dtype=float))
        record["dm_log_phi_is_swr_animal_median"] = (
            float(animal_log_phi.median()) if len(animal_log_phi) else np.nan
        )
        record["dm_log_phi_is_swr_n_animals"] = int(len(animal_log_phi))
        record["dm_log_phi_is_swr_n_animals_neg"] = int((animal_log_phi < 0).sum())
        record["dm_log_phi_is_swr_n_animals_pos"] = int((animal_log_phi > 0).sum())
        records.append(record)
    return pd.DataFrame(records)


def bootstrap_animals(
    coefficients: pd.DataFrame,
    *,
    n_draws: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    percentiles: Sequence[float] = BOOTSTRAP_PERCENTILES,
) -> pd.DataFrame:
    """Animal bootstrap of the animal-balanced median, per (rung, term, estimator).

    The animals present are resampled with replacement; the statistic on each
    draw is the median over the resampled animals of each animal's per-session
    median coefficient. Sessions within an animal are not independent, so the
    animal is the resampling unit.

    Parameters
    ----------
    coefficients : pandas.DataFrame
        The per-session table (already restricted to identified rows here).
    n_draws : int, optional
        Bootstrap draws.
    seed : int, optional
        RNG seed; the draws are generated in sorted group order, so the result
        is reproducible independent of the input row order.
    percentiles : sequence of float, optional
        Percentiles of the bootstrap distribution to report.

    Returns
    -------
    pandas.DataFrame
        One row per (rung, term, estimator) with ``n_animals``, ``n_sessions``,
        the observed statistic, and one ``p<percentile>`` column each.
    """
    identified = _identified(coefficients)
    percentile_names = [f"p{value:g}".replace(".", "_") for value in percentiles]
    columns = [
        "model",
        "term",
        "estimator",
        "n_animals",
        "n_sessions",
        "statistic",
        *percentile_names,
        "n_bootstrap",
        "seed",
    ]
    if identified.empty:
        return pd.DataFrame(columns=columns)

    rng = np.random.default_rng(seed)
    records = []
    for (model, term, estimator), group in identified.groupby(
        ["model", "term", "estimator"], sort=True
    ):
        per_animal = _per_animal_medians(group, "coef").to_numpy(dtype=float)
        n_animals = len(per_animal)
        record = {
            "model": model,
            "term": term,
            "estimator": estimator,
            "n_animals": int(n_animals),
            "n_sessions": int(group["coef"].notna().sum()),
            "statistic": float(np.median(per_animal)) if n_animals else np.nan,
            "n_bootstrap": int(n_draws),
            "seed": int(seed),
        }
        if n_animals == 0:
            record.update(dict.fromkeys(percentile_names, np.nan))
        else:
            draws = rng.integers(0, n_animals, size=(n_draws, n_animals))
            statistics = np.median(per_animal[draws], axis=1)
            for name, value in zip(
                percentile_names, np.percentile(statistics, percentiles), strict=True
            ):
                record[name] = float(value)
        records.append(record)
    return pd.DataFrame(records, columns=columns)


# ── entry point ─────────────────────────────────────────────────────────────


def run_comparison(
    cohort: pd.DataFrame,
    rungs: Sequence[RungSpec],
    *,
    fraction_col: str = FRACTION_RESPONSE,
    count_col: str = COUNT_RESPONSE,
    n_jobs: int = 1,
    min_events_session_phi: int = DEFAULT_MIN_EVENTS_SESSION_PHI,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> ComparisonResult:
    """Fit every rung with all three estimators and summarize their agreement.

    Parameters
    ----------
    cohort : pandas.DataFrame
        Event-grain arm cohort, one row per (event x candidate arm), carrying
        the session key columns, ``event_id``, ``candidate_arm``, ``is_swr``,
        both response columns, and every rung's predictors.
    rungs : sequence of RungSpec
        Models to fit, parents first.
    fraction_col, count_col : str, optional
        The per-event allocation fraction and the timebin counts.
    n_jobs : int, optional
        Worker processes for the Dirichlet-multinomial session fits.
    min_events_session_phi : int, optional
        Sessions below this many modelled events have their concentration
        intercept pinned to the animal-pooled median.
    n_bootstrap, seed : int, optional
        Animal-bootstrap draws and RNG seed.

    Returns
    -------
    ComparisonResult
    """
    all_rows: list[dict] = []
    all_skips: list[dict] = []
    for rung in rungs:
        rows, skips = _fit_rung(
            cohort,
            rung,
            fraction_col=fraction_col,
            count_col=count_col,
            n_jobs=n_jobs,
            min_events_session_phi=min_events_session_phi,
        )
        all_rows.extend(rows)
        all_skips.extend(skips)

    coefficients = _attach_delta_aic(pd.DataFrame(all_rows))
    duplicated = coefficients.duplicated([*SC, "model", "term", "estimator"])
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} duplicated (session, rung, term, estimator) "
            "row(s); the comparison table must be one row per combination"
        )
    return ComparisonResult(
        coefficients=coefficients,
        summary=summarize(coefficients),
        bootstrap=bootstrap_animals(coefficients, n_draws=n_bootstrap, seed=seed),
        skips=pd.DataFrame(all_skips),
    )


# ── figures ─────────────────────────────────────────────────────────────────


def _animal_colors(animals: Sequence[str]) -> dict:
    """A stable colour per animal, in sorted order."""
    palette = plt.get_cmap("tab10")
    return {animal: palette(i % 10) for i, animal in enumerate(sorted(animals))}


def plot_coefficient_agreement(
    coefficients: pd.DataFrame, rungs: Sequence[RungSpec], path: Path
) -> Path:
    """Per key term, per-session count-vs-fraction and count-vs-DM scatters."""
    identified = _identified(coefficients)
    wide = _wide_by_estimator(identified) if not identified.empty else pd.DataFrame()
    pairs = [
        (rung.name, rung.key_term)
        for rung in rungs
        if rung.key_term is not None
        and not wide.empty
        and not wide[
            (wide["model"] == rung.name) & (wide["term"] == rung.key_term)
        ].empty
    ]
    n_rows = max(len(pairs), 1)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(7.0, 3.0 * n_rows), squeeze=False, constrained_layout=True
    )
    colors = _animal_colors(coefficients["animal"].unique()) if not wide.empty else {}
    others = [
        (CONDITIONAL_POISSON_FRACTION, "conditional Poisson, fractions"),
        (DIRICHLET_MULTINOMIAL, "Dirichlet-multinomial"),
    ]
    for row, (model, term) in enumerate(pairs):
        subset = wide[(wide["model"] == model) & (wide["term"] == term)]
        for column, (estimator, label) in enumerate(others):
            ax = axes[row][column]
            other = ESTIMATOR_LABELS[estimator]
            for animal, group in subset.groupby("animal", sort=True):
                ax.scatter(
                    group["coef_count"],
                    group[f"coef_{other}"],
                    s=14,
                    alpha=0.75,
                    color=colors.get(animal),
                    label=animal if row == 0 and column == 0 else None,
                )
            finite = np.concatenate(
                [
                    subset["coef_count"].to_numpy(dtype=float),
                    subset[f"coef_{other}"].to_numpy(dtype=float),
                ]
            )
            finite = finite[np.isfinite(finite)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                pad = 0.05 * (hi - lo) if hi > lo else 0.5
                limits = (lo - pad, hi + pad)
                ax.plot(limits, limits, color="0.4", lw=0.8, ls="--", zorder=0)
                # Shared limits, so the dashed line reads as the 45-degree
                # identity rather than an arbitrary diagonal.
                ax.set_xlim(limits)
                ax.set_ylim(limits)
                ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("conditional Poisson, counts")
            ax.set_ylabel(label)
            ax.set_title(
                f"{model}\n{textwrap.fill(term, 34)}", fontsize=7, linespacing=1.2
            )
            ax.tick_params(labelsize=7)
    if pairs:
        axes[0][0].legend(fontsize=6, frameon=False, title="animal", title_fontsize=6)
    else:
        for ax in axes.ravel():
            ax.set_axis_off()
        axes[0][0].text(0.5, 0.5, "no identified key-term fits", ha="center")
    fig.suptitle("Per-session content coefficients by estimator", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_log_phi_by_session(
    coefficients: pd.DataFrame,
    bootstrap: pd.DataFrame,
    path: Path,
    *,
    rung: str = DEFAULT_LOG_PHI_RUNG,
) -> Path:
    """Concentration intercept and its is_swr modifier, session by session.

    Raises
    ------
    ValueError
        If ``rung`` has no Dirichlet-multinomial rows. The caption names the
        rung the panels show, so quietly substituting another one would publish
        a figure whose caption describes a different model.
    """
    identified = _identified(coefficients)
    dm = identified[
        (identified["estimator"] == DIRICHLET_MULTINOMIAL)
        & (identified["model"] == rung)
    ]
    if dm.empty:
        available = sorted(
            identified.loc[
                identified["estimator"] == DIRICHLET_MULTINOMIAL, "model"
            ].unique()
        )
        raise ValueError(
            f"no identified Dirichlet-multinomial fits for rung {rung!r}; the "
            f"concentration figure cannot be drawn (rungs with DM fits: "
            f"{available})"
        )

    terms = [
        (LOG_PHI_INTERCEPT, "log φ intercept (non-SWR)"),
        (LOG_PHI_IS_SWR, "log φ shift on SWR events"),
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 5.0), sharex=True, constrained_layout=True
    )
    colors = _animal_colors(coefficients["animal"].unique()) if not dm.empty else {}
    for ax, (term, label) in zip(axes, terms, strict=True):
        subset = dm[dm["term"] == term].sort_values([*SC])
        for animal, group in subset.groupby("animal", sort=True):
            order = np.arange(len(group))
            ax.plot(
                order,
                group["coef"].to_numpy(dtype=float),
                marker="o",
                ms=3.5,
                lw=0.8,
                color=colors.get(animal),
                label=animal,
            )
        band = bootstrap[
            (bootstrap["model"] == rung)
            & (bootstrap["term"] == term)
            & (bootstrap["estimator"] == DIRICHLET_MULTINOMIAL)
        ]
        if not band.empty:
            row = band.iloc[0]
            ax.axhspan(
                float(row["p2_5"]),
                float(row["p97_5"]),
                color="0.75",
                alpha=0.4,
                zorder=0,
            )
            ax.axhline(float(row["p50"]), color="0.3", lw=0.9, zorder=1)
        if term == LOG_PHI_IS_SWR:
            ax.axhline(0.0, color="k", lw=0.6, ls=":", zorder=1)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("session order within animal")
    axes[0].legend(fontsize=6, frameon=False, ncol=5, title="animal", title_fontsize=6)
    fig.suptitle(
        f"Dirichlet-multinomial concentration, {rung} "
        "(band = animal-bootstrap 2.5–97.5%)",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


CAPTION = """# Content estimators compared

**Panels.** `coef_agreement.png` — one row per model rung, showing that rung's
key coefficient for every session that all three estimators fit. The x axis is
the retired conditional Poisson on per-arm timebin counts; the left column's y
axis is the shipped conditional Poisson on per-event allocation fractions
(event-clustered standard errors), the right column's is the
Dirichlet-multinomial on the same counts with an `is_swr` concentration
modifier. The dashed line is the identity. Colour identifies the animal. All
three estimators were fit on one shared per-(session, rung) event set, so a
departure from the identity is a difference of likelihood, not of cohort.
Unidentified fits are excluded, by the same rule the shipped aggregation uses:
the fitter flagged that term as separated, or the effect-scale magnitude
|coefficient| × SD(predictor) reaches 10. That effect-scale backstop applies to
the concentration modifier `log_phi_is_swr` as well, on the SD of the event-level
`is_swr` flag; `log_phi_intercept` is the one term it cannot apply to, because
its design column is a constant with no SD, so there the separation flag is the
whole criterion. The drop is applied to the whole (session, rung, term) triple,
so the three estimators are always compared over the same sessions.

`log_phi_by_session.png` — the Dirichlet-multinomial concentration for the
`C2_base` rung. **Top,** the log concentration intercept, i.e. how tightly a
non-SWR event's timebins cluster on a few arms; **bottom,** the additive shift
of that log concentration on SWR events, where a value below zero means SWR
events are *more* concentrated (more redundant bins) and above zero means their
mass is spread over more arms. Each line is one animal, plotted in session order
within that animal. The grey band is the animal bootstrap's 2.5–97.5 percentile
interval and the grey line its median: the animals present are resampled with
replacement for 2000 draws at **seed 0**, and each draw's statistic is the median
over the resampled animals of each animal's per-session median.
"""


def write_caption(path: Path) -> Path:
    """Write the tracked caption markdown beside the figures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CAPTION)
    return path


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rungs",
        default=None,
        help="comma-separated rung names (default: every comparison rung); "
        f"available: {','.join(rung.name for rung in COMPARISON_RUNGS)}",
    )
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="smoke subset: keep the first N sessions per animal",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="worker processes for the Dirichlet-multinomial session fits",
    )
    parser.add_argument(
        "--min-events-session-phi",
        type=int,
        default=DEFAULT_MIN_EVENTS_SESSION_PHI,
        help="sessions below this many events get a pooled concentration intercept",
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=replay_content_dir() / "content_estimator_comparison",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=figure_dir("content_estimators")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Assemble the shipped at-well cohort, compare the estimators, write outputs."""
    args = _parse_args(argv)
    rungs = resolve_rungs(
        [name.strip() for name in args.rungs.split(",") if name.strip()]
        if args.rungs
        else None
    )

    logger.info("loading the at-well arm cohort")
    contrast_inputs = contrast_fit_inputs()
    ev, _z_reference = assemble_arm(contrast_inputs)
    inputs = record_inputs(contrast_inputs)
    # Only the arm cohort is compared, so the patch cohort is not assembled:
    # this is prepare_cohorts' arm half, for the "all" stratum.
    ev, stratum_audit = select_outcome_stratum(ev, "all")
    if args.max_per_animal is not None:
        ev = subset_max_per_animal(ev, args.max_per_animal)
    logger.info(
        "cohort: %d events over %d sessions",
        ev["event_id"].nunique(),
        len(ev[SC].drop_duplicates()),
    )

    result = run_comparison(
        ev,
        rungs,
        n_jobs=args.n_jobs,
        min_events_session_phi=args.min_events_session_phi,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients_path = write_table(
        result.coefficients, output_dir / COEFFICIENTS_BASENAME
    )
    result.summary.to_csv(output_dir / f"{SUMMARY_BASENAME}.csv", index=False)
    result.bootstrap.to_csv(output_dir / f"{BOOTSTRAP_BASENAME}.csv", index=False)
    result.skips.to_csv(output_dir / f"{SKIPS_BASENAME}.csv", index=False)

    figures_dir = Path(args.figures_dir)
    plot_coefficient_agreement(
        result.coefficients, rungs, figures_dir / "coef_agreement.png"
    )
    # The concentration figure is pinned to one rung, and --rungs can select a
    # set that does not include it. That is a legitimate partial run, not a
    # failure: raising here would abort AFTER the tables were written and BEFORE
    # the provenance sidecar, leaving artifacts no reader could date or trace.
    if log_phi_figure_rung(rungs) is not None:
        plot_log_phi_by_session(
            result.coefficients,
            result.bootstrap,
            figures_dir / "log_phi_by_session.png",
        )
    else:
        logger.info(
            "skipping the concentration figure: it is drawn for %s, which is "
            "not among the selected rungs (%s)",
            DEFAULT_LOG_PHI_RUNG,
            ", ".join(rung.name for rung in rungs),
        )
    write_caption(figures_dir / "content_estimators_caption.md")

    write_fit_provenance(
        coefficients_path,
        inputs,
        PROVENANCE_ESTIMATOR,
        sys.argv,
        extra={
            "rungs": [rung.name for rung in rungs],
            "estimators": list(ESTIMATORS),
            "response_columns": {
                "fraction": FRACTION_RESPONSE,
                "count": COUNT_RESPONSE,
            },
            "outcome_stratum": "all",
            "stratum_audit": stratum_audit,
            "max_per_animal": args.max_per_animal,
            "min_events_session_phi": args.min_events_session_phi,
            "bootstrap": {
                "n_draws": args.n_bootstrap,
                "seed": args.seed,
                "percentiles": list(BOOTSTRAP_PERCENTILES),
                "unit": "animal",
            },
            "aic_equalization": (
                "multinomial_log_constant over the count matrix added to both "
                "conditional-Poisson llf values; the fraction estimator's AIC is "
                f"{QUASI_SCALE}"
            ),
            # (session, rung, term) triples the identifiability rule removed
            # from the medians, the bootstrap and the figures. The drop is
            # paired: if one estimator's row is unidentified, all three go.
            "n_unidentified_triples_dropped": n_unpaired_triples(result.coefficients),
            "n_fraction_rows_model_se_fallback": int(
                (result.coefficients["se_source"] == "model_fallback").sum()
            ),
            # (session, rung, estimator) fits whose parent landed on a
            # different number of events, so their AIC was not differenced.
            "n_fits_delta_aic_row_mismatch": (
                0
                if result.coefficients.empty
                else int(
                    result.coefficients.loc[
                        result.coefficients["delta_aic_row_mismatch"]
                    ]
                    .drop_duplicates([*SC, "model", "estimator"])
                    .shape[0]
                )
            ),
        },
    )
    logger.info("wrote %s and its companions", coefficients_path)
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
