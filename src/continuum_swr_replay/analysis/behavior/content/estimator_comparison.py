"""How three content estimators agree, fit on one shared event set per session.

The content question -- which arm an event represents -- is fit per session with
three likelihoods: the conditional Poisson on the timebin counts, the
conditional Poisson on the per-event fractions, and the Dirichlet-multinomial on
the counts. One row per (session, rung, term, estimator) holds a coefficient,
its standard error and the fit's AIC with the likelihood constants equalized
(:func:`estimator_rows`). From that table:

- :func:`attach_delta_aic` differences each rung's AIC against its parent's,
  within session and estimator, where the two fits can be compared;
- :func:`unidentified_rows` and :func:`identified_rows` drop the terms a session
  cannot identify, as whole estimator triples;
- :func:`summarize_estimators` gives one row per (rung, term) of animal-balanced
  medians;
- :func:`bootstrap_animals` gives the animal bootstrap of those medians.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ...shared_statistics import session_median_statistics
from .._fit_diagnostics import SEPARATION_COEF_ABS
from ..coefficient_statistics import animal_median_bootstrap
from ..fitting import CONDITIONAL_POISSON_COUNT, CONDITIONAL_POISSON_FRACTION
from .dirichlet_multinomial import ESTIMATOR as DIRICHLET_MULTINOMIAL

SC = list(SESSION_COLUMNS)

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
    negative means the rung is better. The difference is only formed where the rung and its parent were fit on the
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
    """Animal bootstrap of the animal-balanced median, per (rung, term, estimator).

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
