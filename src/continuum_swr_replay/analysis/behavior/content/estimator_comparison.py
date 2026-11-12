"""How three content estimators agree, fit on one shared event set per session.

The content question -- which arm an event represents -- is fit per session with
three likelihoods: the conditional Poisson on the timebin counts, the
conditional Poisson on the per-event fractions, and the Dirichlet-multinomial on
the counts. One row per (session, rung, term, estimator) holds a coefficient,
its standard error and the fit's AIC with the likelihood constants equalized
(:func:`estimator_rows`). :func:`run_comparison` fits the rungs of
:data:`~continuum_swr_replay.analysis.behavior.model_specs.CONTENT_ESTIMATOR_COMPARISON_RUNGS`
on one event set per (session, rung) that all three can use
(:func:`shared_session_rows`, :func:`drop_undefined_concentration`). From that
table:

- :func:`attach_delta_aic` differences each rung's AIC against its parent's,
  within session and estimator, where the two fits can be compared;
- :func:`unidentified_rows` and :func:`identified_rows` drop the terms a session
  cannot identify, as whole estimator triples;
- :func:`summarize_estimators` gives one row per (rung, term) of animal-balanced
  medians;
- :func:`bootstrap_animals` gives the nested bootstrap (animals, then their
  sessions) of those medians.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ...shared_statistics import session_median_statistics
from .._fit_diagnostics import SEPARATION_COEF_ABS
from ..coefficient_statistics import animal_median_bootstrap
from ..contrast_aggregation import MIN_EVENTS, prepare_session
from ..fit_comparability import COMPARABILITY_COLUMNS, why_not_comparable
from ..fitting import (
    CONDITIONAL_POISSON_COUNT,
    CONDITIONAL_POISSON_FRACTION,
    fit_conditional_poisson,
)
from ..model_specs import ComparisonRung, leaf_only_cohort
from .dirichlet_multinomial import (
    DEFAULT_MIN_EVENTS_SESSION_PHI,
    concentration_cols_for_cohort,
    fit_sessions_dirichlet_multinomial,
    multinomial_log_constant,
)
from .dirichlet_multinomial import ESTIMATOR as DIRICHLET_MULTINOMIAL

logger = logging.getLogger(__name__)

SC = list(SESSION_COLUMNS)

#: The two response columns the comparison is fit on: the per-event allocation
#: fraction (the shipped model) and the per-arm timebin counts (the retired count
#: model and the Dirichlet-multinomial).
FRACTION_RESPONSE = "arm_fraction"
COUNT_RESPONSE = "arm_count_hard"

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

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
BOOTSTRAP_PERCENTILES = (2.5, 50.0, 97.5)

LOG_PHI_IS_SWR = "log_phi_is_swr"


def se_ratio_column(label: str) -> str:
    """Name of the SE-ratio column for estimator ``label``, SE types spelled out."""
    return f"se_ratio_{SE_TYPE_LABELS[label]}_over_{SE_TYPE_LABELS['count']}"


def estimator_rows(
    key: tuple,
    *,
    model: str,
    parent: str | None,
    leaf_only: bool,
    fits: dict[str, dict],
    info: dict,
    log_constant: float,
    n_events: int,
) -> list[dict]:
    """One row per (term, estimator) for a session whose three fits all landed.

    ``aic_equalized`` is ``2 n_params - 2 llf`` with the multinomial log
    constant added to an ``llf`` that omits it, so the three estimators' AICs
    are on one footing. The fraction fit reports its event-clustered standard
    error; a non-finite sandwich falls back to the model SE, labelled
    ``model_fallback``.

    Parameters
    ----------
    key : tuple
        The session's ``SESSION_COLUMNS`` values.
    model, parent : str
        The rung and the rung its AIC is differenced against (or None).
    leaf_only : bool
        Whether the rung is fit over the six leaf arms.
    fits : dict
        Estimator name -> that estimator's fit of the session.
    info : dict
        The shared event set's counts; ``n_events_in`` is read.
    log_constant : float
        The multinomial log constant of the session's count matrix.
    n_events : int
        Events the three fits share.

    Returns
    -------
    list of dict

    Raises
    ------
    ValueError
        If the fraction fit carries no ``bse_cluster_<term>``.
    """
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
                    f"{model} {key}: the conditional-Poisson fraction fit "
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
                    "model": model,
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
                    "leaf_only": bool(leaf_only),
                    "parent": parent,
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


def attach_delta_aic(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Add ``delta_aic_vs_parent`` per (session, rung, estimator).

    It is the rung's ``aic_equalized`` minus its parent's, child minus parent:
    negative means the rung is better. The difference is only formed where the
    rung's and its parent's fits of that session and estimator can be compared
    (:func:`~continuum_swr_replay.analysis.behavior.fit_comparability.why_not_comparable`,
    on the :data:`~continuum_swr_replay.analysis.behavior.fit_comparability.COMPARABILITY_COLUMNS`
    the rows carry: ``n_events`` here). An AIC difference across different rows
    is not a model comparison, so it is left NaN and flagged
    (``delta_aic_row_mismatch``) rather than reported.

    Parameters
    ----------
    coefficients : pd.DataFrame, shape (n_rows, n_columns)
        One row per (session, rung, term, estimator), with ``parent`` and
        ``aic_equalized``.

    Returns
    -------
    pd.DataFrame
        ``coefficients`` with ``delta_aic_vs_parent`` and
        ``delta_aic_row_mismatch`` (True where the parent was fit and the two
        fits cannot be compared).
    """
    coefficients = coefficients.copy()
    coefficients["delta_aic_vs_parent"] = np.nan
    coefficients["delta_aic_row_mismatch"] = False
    if coefficients.empty:
        return coefficients

    carried = [c for c in COMPARABILITY_COLUMNS if c in coefficients]
    fit_level = coefficients.drop_duplicates([*SC, "model", "estimator"])[
        [*SC, "model", "estimator", "aic_equalized", *carried]
    ]
    parent_level = fit_level.rename(
        columns={
            "model": "parent",
            "aic_equalized": "aic_equalized_parent",
            **{column: f"{column}_parent" for column in carried},
        }
    )
    merged = coefficients.merge(
        parent_level, on=[*SC, "parent", "estimator"], how="left"
    )
    comparable = ~why_not_comparable(merged, ("", "_parent")).any(axis=1)
    merged.loc[comparable, "delta_aic_vs_parent"] = (
        merged.loc[comparable, "aic_equalized"]
        - merged.loc[comparable, "aic_equalized_parent"]
    )
    merged["delta_aic_row_mismatch"] = (
        merged["aic_equalized_parent"].notna() & ~comparable
    )
    return merged.drop(
        columns=["aic_equalized_parent", *(f"{column}_parent" for column in carried)]
    )


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


def identified_rows(coefficients: pd.DataFrame) -> pd.DataFrame:
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


def _animal_summary(frame: pd.DataFrame, column: str) -> dict[str, float]:
    """``session_median_statistics`` of one column over the rows that have it."""
    return session_median_statistics(frame[column], frame["animal"])


def wide_by_estimator(coefficients: pd.DataFrame) -> pd.DataFrame:
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


def summarize_estimators(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-session table to one row per (rung, term).

    Every median here is animal-balanced -- the median over animals of each
    animal's per-session median -- so the five animals weigh equally and the
    number matches the bootstrap statistic. ``pseudo_r2`` is deliberately
    absent: its null differs by estimator and it is not comparable across them.

    The ``dm_log_phi_is_swr_*`` columns are a property of the RUNG, not of the
    row's term, so they repeat across every term of a rung.
    """
    identified = identified_rows(coefficients)
    if identified.empty:
        return pd.DataFrame(columns=["model", "term"])
    wide = wide_by_estimator(identified)
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
        name: _animal_summary(group, "coef")
        for name, group in log_phi.groupby("model", sort=False)
    }
    no_log_phi = _animal_summary(log_phi.iloc[:0], "coef")

    records = []
    for (model, term), group in wide.groupby(["model", "term"], sort=True):
        record = {
            "model": model,
            "term": term,
            "n_sessions_unidentified": int(unidentified.get((model, term), 0)),
        }
        for estimator in ESTIMATORS:
            label = ESTIMATOR_LABELS[estimator]
            coef = _animal_summary(group, f"coef_{label}")
            record[f"n_sessions_{label}"] = coef["n_sessions"]
            record[f"n_animals_{label}"] = coef["n_animals"]
            record[f"median_coef_{label}"] = coef["animal_median"]
            record[f"n_animals_pos_{label}"] = coef["n_animals_positive"]
            record[f"median_delta_aic_{label}"] = _animal_summary(
                group, f"delta_aic_vs_parent_{label}"
            )["animal_median"]
            if estimator != CONDITIONAL_POISSON_COUNT:
                record[f"median_{se_ratio_column(label)}"] = _animal_summary(
                    group, se_ratio_column(label)
                )["animal_median"]
        animal_log_phi = per_rung_log_phi.get(model, no_log_phi)
        record["dm_log_phi_is_swr_animal_median"] = animal_log_phi["animal_median"]
        record["dm_log_phi_is_swr_n_animals"] = animal_log_phi["n_animals"]
        record["dm_log_phi_is_swr_n_animals_neg"] = animal_log_phi["n_animals_negative"]
        record["dm_log_phi_is_swr_n_animals_pos"] = animal_log_phi["n_animals_positive"]
        records.append(record)
    return pd.DataFrame(records)


def bootstrap_animals(
    coefficients: pd.DataFrame,
    *,
    n_draws: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    percentiles: Sequence[float] = BOOTSTRAP_PERCENTILES,
) -> pd.DataFrame:
    """Nested bootstrap of the animal-balanced median, per (rung, term, estimator).

    The coefficient's
    :func:`~continuum_swr_replay.analysis.behavior.coefficient_statistics.animal_median_bootstrap`
    in each (rung, term, estimator), the groups taken in sorted order from one
    generator.

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
    identified = identified_rows(coefficients)
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
        bootstrap = animal_median_bootstrap(
            group["coef"],
            group["animal"],
            rng=rng,
            n_draws=n_draws,
            percentiles=percentiles,
        )
        records.append(
            {
                "model": model,
                "term": term,
                "estimator": estimator,
                "n_animals": bootstrap["n_animals"],
                "n_sessions": bootstrap["n_sessions"],
                "statistic": bootstrap["animal_median"],
                **{
                    name: float(value)
                    for name, value in zip(
                        percentile_names, bootstrap["percentiles"], strict=True
                    )
                },
                "n_bootstrap": int(n_draws),
                "seed": int(seed),
            }
        )
    return pd.DataFrame(records, columns=columns)


def n_fits_delta_aic_row_mismatch(coefficients: pd.DataFrame) -> int:
    """(session, rung, estimator) fits whose ΔAIC was not formed.

    Their parent landed on a different number of events
    (``delta_aic_row_mismatch`` from :func:`attach_delta_aic`), so the two AICs
    were not differenced.
    """
    if coefficients.empty:
        return 0
    return int(
        coefficients.loc[coefficients["delta_aic_row_mismatch"]]
        .drop_duplicates([*SC, "model", "estimator"])
        .shape[0]
    )


# ── one shared event set per (session, rung) ────────────────────────────────


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


def _rung_frame(
    cohort: pd.DataFrame, rung: ComparisonRung, *, fraction_col: str
) -> pd.DataFrame:
    """Restrict and rescale the cohort to what one rung is fit on.

    A leaf-only rung drops the three trunk arms and renormalizes the fraction
    response to unit mass over the six retained leaves
    (:func:`~continuum_swr_replay.analysis.behavior.model_specs.leaf_only_cohort`);
    an event with no leaf mass renormalizes to NaN and is then dropped from all
    three estimators by :func:`shared_session_rows`. The counts keep their
    timebin weighting, which is also the ``L_e`` the Dirichlet-multinomial and
    the multinomial constant are stated on.
    """
    if not rung.leaf_only:
        return cohort
    frame, _renormalized = leaf_only_cohort(
        cohort, response_col=fraction_col, response="fraction"
    )
    return frame


def shared_session_rows(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    *,
    fraction_col: str,
    count_col: str,
) -> tuple[pd.DataFrame | None, dict]:
    """Freeze one event set that both responses (hence all three fits) can use.

    The shipped contrast runner's cohort rules
    (:func:`~continuum_swr_replay.analysis.behavior.contrast_aggregation.prepare_session`)
    are applied twice -- once per response column -- and intersected. A
    difference between the estimators is then a difference of likelihood and
    never of which events they saw.

    Parameters
    ----------
    frame : pandas.DataFrame
        One session's event-grain arm rows.
    predictors : sequence of str
        The rung's predictor columns.
    fraction_col, count_col : str
        The per-event allocation fraction and the timebin counts.

    Returns
    -------
    (rows_or_None, info)
        The rows of the shared events, or ``None`` when there is nothing to fit.
        ``info`` carries ``n_events_in``, the per-response survivor counts, the
        shared count, and a skip ``reason`` (``too_few_shared`` below
        :data:`~continuum_swr_replay.analysis.behavior.contrast_aggregation.MIN_EVENTS`
        shared events).
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


def drop_undefined_concentration(
    frame: pd.DataFrame, concentration_cols: Sequence[str]
) -> tuple[pd.DataFrame, int]:
    """Drop events whose concentration covariate is missing, for every estimator.

    The Dirichlet-multinomial drops an event with a NaN ``is_swr`` (it is a
    required column of its pack) while the conditional fits never look at it, so
    leaving such an event in would desynchronize the three event sets. Removing
    it up front keeps them identical.

    Parameters
    ----------
    frame : pandas.DataFrame
        Event-grain arm rows with ``event_id``.
    concentration_cols : sequence of str
        The event-level columns entering ``log phi``.

    Returns
    -------
    (frame, n_dropped)
        Every row of each event with a NaN in any concentration column removed,
        and the number of events removed.
    """
    if not concentration_cols:
        return frame, 0
    missing = frame[list(concentration_cols)].isna().any(axis=1)
    if not missing.any():
        return frame, 0
    bad = set(frame.loc[missing, "event_id"])
    return frame[~frame["event_id"].isin(bad)], len(bad)


def _skip_record(key: tuple, rung: ComparisonRung, info: dict, reason: str) -> dict:
    """One audited (session, rung) that produced no comparable estimator triple."""
    return {
        **dict(zip(SC, key, strict=True)),
        "model": rung.name,
        **{name: value for name, value in info.items() if name != "reason"},
        "reason": reason,
    }


def _fit_rung(
    cohort: pd.DataFrame,
    rung: ComparisonRung,
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
    frame, n_drop_concentration = drop_undefined_concentration(
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
        rows, info = shared_session_rows(
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
            estimator_rows(
                key,
                model=rung.name,
                parent=rung.parent,
                leaf_only=rung.leaf_only,
                fits=fits,
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


def run_comparison(
    cohort: pd.DataFrame,
    rungs: Sequence[ComparisonRung],
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
    rungs : sequence of ComparisonRung
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

    Raises
    ------
    ValueError
        If a (session, rung, term, estimator) row is duplicated.
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

    coefficients = attach_delta_aic(pd.DataFrame(all_rows))
    duplicated = coefficients.duplicated([*SC, "model", "term", "estimator"])
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} duplicated (session, rung, term, estimator) "
            "row(s); the comparison table must be one row per combination"
        )
    return ComparisonResult(
        coefficients=coefficients,
        summary=summarize_estimators(coefficients),
        bootstrap=bootstrap_animals(coefficients, n_draws=n_bootstrap, seed=seed),
        skips=pd.DataFrame(all_skips),
    )
