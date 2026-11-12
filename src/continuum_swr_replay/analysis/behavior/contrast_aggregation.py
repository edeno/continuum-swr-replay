"""Shared audited aggregation for the is_swr contrast ladders (rate + content).

Both the content contrast ladder (C/CS, CP/CPS) and the rate contrast ladder
(G/S) aggregate per-session conditional/Poisson coefficients the SAME way, so the
two ladders' summary tables mirror each other. This module is the single home for
that logic.

PRIMARY population evidence is at the **animal** level (n=5): ``coef_animal_median``
(median of per-animal medians -- balanced so animals with more sessions do not
dominate) and ``n_animals_pos`` / ``n_animals`` (sign consistency). With only 5
animals a frequentist p is barely meaningful, so the sign consistency IS the claim.
``session_signrank_p`` is a DESCRIPTIVE across-session consistency statistic (sessions
within an animal are not independent), with a BH-FDR across the rungs sharing a
population (``session_signrank_fdr_q``). ``typical_session_se`` is the median
per-session SE, not the aggregate's uncertainty; it is the event-clustered SE
wherever the fitters wrote one (``se_source``). ``pct_sig_session`` is read on
that same SE (a per-session Wald z), so the row's dispersion and its
significance count share one definition of uncertainty. Continuous-predictor coefficients
also carry a standardized (per-1-SD) form (``std_*``); binary predictors are
per-category; and the reward-history basis features are weighted reward fractions
in [0, 1], so their coefficients are reported per unit of the fraction (factor 1.0,
``per_unit``) rather than per SD.

The second half of the module assembles the content contrast ladder's cohorts:
it reads the canonical tables (:func:`contrast_fit_inputs`), attaches the design
columns and the fit-time ``_z`` reference (:func:`assemble_arm`,
:func:`assemble_patch`), freezes the per-SD scales before the outcome stratum
(:func:`prepare_cohorts`), and applies the per-session cohort rules
(:func:`prepare_session`). ``scripts/behavior/content/fit_swr_contrast_content.py`` and
``scripts/behavior/content/compare_content_estimators.py`` both fit on these cohorts, so
the rules live here rather than in either script.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.analysis.events.cohort_selection import select_event_cohort
from continuum_swr_replay.analysis.shared_statistics import (
    benjamini_hochberg,
    session_median_statistics,
    signed_rank_p,
)
from continuum_swr_replay.paths import (
    STANDARDIZATION_REFERENCE,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.tables.covariate_registry import require_covariate_schema
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
    session_key_frame,
    subset_max_per_animal,
)
from continuum_swr_replay.tables.standardization import (
    assert_standardized_matches_reference,
    attach_standardized,
    load_reference,
)

from ._fit_diagnostics import SEPARATION_COEF_ABS
from ._helpers import add_arm_fixed_effects
from .cohort_audit import HAS_QVALS_ROW, qvals_gap_event_ids
from .covariates.history.value_uncertainty import validate_uncertainty_inputs
from .covariates.interactions import (
    LOG1P_STOP_TIME,
    LOG_EVENT_LENGTH,
    attach_interactions,
)
from .fitting import parse_unidentified_terms
from .model_specs import LEAF_ONLY_N_CATEGORIES, REWARD_HISTORY_COLUMNS

# Above this share of unidentified session fits, a term's aggregate is a
# statement about the identified minority and is not emitted at all. The share
# counts both kinds: separated or runaway sessions (``n_dropped``) and sessions
# that cannot inform the term at all (``n_unidentified``).
MAX_UNIDENTIFIED_SHARE = 0.20

# Fewest kept sessions a term needs before its aggregate is emitted.
MIN_SESSIONS_PER_TERM = 5

# Two-sided normal critical value at alpha = 0.05, for the per-session Wald test.
_WALD_Z_TWO_SIDED_05 = 1.959963984540054

# The per-row output schema, in emit order. Kept as a constant so an
# all-gated-out run still returns a correctly-shaped (empty) frame rather than a
# column-less one that KeyErrors on column selection. ``session_signrank_fdr_q``
# is appended after the BH-FDR pass, so it is added by the caller/return, not here.
_AGG_COLUMNS = (
    "level",
    "model",
    "population",
    "term",
    "n_sessions",
    "n_animals",
    "n_dropped",
    "n_unidentified",
    "n_sessions_flagged_nonconverged",
    "n_sessions_flagged_separated",
    "coef_animal_median",
    "n_animals_pos",
    "coef_session_median",
    "typical_session_se",
    "se_source",
    "std_unit",
    "std_factor",
    "std_coef_animal_median",
    "std_coef_session_median",
    "std_typical_session_se",
    "pct_positive_session",
    "pct_sig_session",
    "session_signrank_p",
)

logger = logging.getLogger(__name__)

# Binary predictors (per-category; not standardized). Full term names, including the
# swr_x_ / is_swr: interaction forms used by the content and rate ladders.
BINARY_TERMS = frozenset(
    {
        "is_current_arm",
        "swr_x_is_current_arm",
        "is_last_rewarded_noncurrent_arm",
        "swr_x_is_prev_arm",
        "swr_x_is_last_rewarded_noncurrent_arm",
        "is_current_patch",
        "is_last_rewarded_noncurrent_patch",
        "swr_x_is_current_patch",
        "swr_x_is_last_rewarded_noncurrent_patch",
        # Dirichlet-multinomial concentration terms. ``log_phi_is_swr`` is the
        # coefficient on a 0/1 event flag, so its natural unit is the category
        # (how much less concentrated an SWR event's allocation is), not an SD
        # of the flag; ``log_phi_intercept`` is a constant column with no SD to
        # rescale by at all.
        "log_phi_intercept",
        "log_phi_is_swr",
    }
)

# Predictors reported per unit of the predictor itself, not per SD: the
# reward-history basis features are weighted reward fractions in [0, 1], so a
# per-SD rescaling would make "one SD of a fraction" the unit of an effect whose
# natural unit is "all of the recent outcomes rewarded vs none of them".
PER_UNIT_TERMS = frozenset(
    [*REWARD_HISTORY_COLUMNS, *(f"is_swr:{c}" for c in REWARD_HISTORY_COLUMNS)]
)


def load_per_sd_scale(provenance_path: str | Path, cohort: str) -> dict[str, float]:
    """Per-SD units a fitted contrast table states its coefficients in.

    The fitters freeze one pooled SD per raw-scale covariate, over the row
    support the rungs carrying it are fitted on (the six leaf arms for the arm
    cohort), and persist it in the run's provenance record. Readers must take
    the unit from there rather than recomputing an SD of their own: the same
    covariate measured over all nine candidate arms has a 2.1x larger SD,
    because the three trunks are structural zeros, and a figure recomputing it
    would restate every effect in a unit the audited table does not use.

    Parameters
    ----------
    provenance_path : str or Path
        The runner's provenance JSON (``*_provenance.json``).
    cohort : str
        Key within ``per_sd_scale`` (``"arm_cohort"`` / ``"patch_cohort"``).

    Returns
    -------
    dict of str to float
        Covariate -> frozen SD, ready to pass as ``aggregate_contrast``'s ``sd``.

    Raises
    ------
    FileNotFoundError
        If the provenance record is absent.
    KeyError
        If it carries no ``per_sd_scale`` entry for ``cohort`` -- a record
        predating the frozen units, whose fits are in an unknown one.
    """
    path = Path(provenance_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required for the per-SD unit the fit's coefficients are "
            "stated in; re-run the fitter to write it."
        )
    scale = json.loads(path.read_text()).get("per_sd_scale") or {}
    if cohort not in scale:
        raise KeyError(
            f"{path} records no per_sd_scale[{cohort!r}]. It predates the frozen "
            "per-SD units, so the unit its coefficients are stated in is unknown; "
            "re-run the fitter rather than recomputing an SD here."
        )
    return {cov: float(value) for cov, value in scale[cohort].items()}


def load_fit_time_sd_reference(meta_path: str | Path) -> dict[str, float]:
    """Per-SD units the rate contrast's audited table states its coefficients in.

    ``scripts/behavior/rate/fit_swr_contrast_rate.py`` takes one SD per raw-scale rate
    covariate over the stop cohort its per-session fits name (at-well for the
    canonical bundle), aggregates with it, and records it in the run's meta
    JSON. Readers must take the unit from there: the same covariates measured
    over the full tables differ by up to ~7 %, which would restate every
    per-SD rate effect in a unit the audited table does not use.

    Parameters
    ----------
    meta_path : str or Path
        The rate runner's meta JSON (``swr_contrast_rate_meta*.json``).

    Returns
    -------
    dict of str to float
        Covariate -> SD, ready to pass as ``aggregate_contrast``'s ``sd``.

    Raises
    ------
    FileNotFoundError
        If the meta record is absent.
    KeyError
        If it carries no ``fit_time_sd_reference`` -- the unit the audited
        table's ``std_*`` columns are stated in is then unknown.
    """
    path = Path(meta_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required for the per-SD unit the rate fit's coefficients "
            "are stated in; re-run the fitter (or its --aggregate-only path) to "
            "write it."
        )
    reference = json.loads(path.read_text()).get("fit_time_sd_reference")
    if not reference:
        raise KeyError(
            f"{path} records no fit_time_sd_reference, so the unit its audited "
            "table is stated in is unknown; re-run the fitter rather than "
            "recomputing an SD here."
        )
    return {cov: float(value) for cov, value in reference.items()}


def std_factor(term: str, sd: dict) -> tuple[float, str]:
    """SD multiplier + unit label to convert a raw coef to standardized units.

    Resolved in order:

    1. Binary (:data:`BINARY_TERMS`) -> ``(1.0, "per_category")``.
    2. Reward-history basis feature (:data:`PER_UNIT_TERMS`) ->
       ``(1.0, "per_unit")``, checked before the ``sd`` lookup so a pooled SD
       computed for some other purpose cannot silently rescale a
       fraction-valued coefficient.
    3. Continuous covariate in ``sd`` -> ``(SD, "per_SD")``; the term may be the
       bare covariate, its ``swr_x_<cov>`` product, or its ``is_swr:<cov>``
       product.
    4. Z-scored at fit time -> ``(1.0, "per_SD")``: a ``_z`` column or an SWR
       product of one (the term ends in ``_z``), or a product with a ``_z``
       factor (the term contains ``_z_x_``). Its coefficient is already per SD.

    Parameters
    ----------
    term : str
        The aggregated term.
    sd : dict
        Covariate -> SD the fit's per-SD effects are stated in.

    Returns
    -------
    (float, str)
        The factor and its unit label.

    Raises
    ------
    KeyError
        If ``term`` is none of the above: a raw-scale continuous term with no
        SD would be reported at raw scale under a "per SD" label. Also if its
        SD is zero or NaN.
    """
    if term in BINARY_TERMS:
        return 1.0, "per_category"
    if term in PER_UNIT_TERMS:
        return 1.0, "per_unit"
    cov = sd_covariate(term, sd)
    if cov is not None:
        if sd[cov] > 0:
            return sd[cov], "per_SD"
        raise KeyError(
            f"term {term!r} resolves to {cov!r}, whose SD is {sd[cov]}: a "
            "predictor with no spread in the fit's cohort has no per-SD effect."
        )
    if term.endswith("_z") or "_z_x_" in term:
        return 1.0, "per_SD"
    raise KeyError(
        f"term {term!r} is not binary, not a reward-history basis feature, not "
        "z-scored at fit time (no '_z' suffix or '_z_x_' factor), and resolves "
        "to no SD entry; it would be reported at raw scale under a 'per SD' "
        "label. Pass its SD from the fit's own record, or add it to "
        "BINARY_TERMS / PER_UNIT_TERMS if that is what it is."
    )


def sd_covariate(term: str, sd: dict) -> str | None:
    """The covariate in ``sd`` that ``term`` is, or is an SWR product of; else None."""
    for cov in sd:
        if term in (cov, f"swr_x_{cov}", f"is_swr:{cov}"):
            return cov
    return None


def separated_for_term(fits: pd.DataFrame, term: str) -> pd.Series:
    """Per-fit mask: is ``term`` itself unidentified in this fit?

    Separation is a property of a coefficient, not of a model. A fit whose
    ``reward_history_incomplete`` or self-history coefficient ran away still
    identifies its scientific terms perfectly well, so dropping the session for
    every term is what emptied the audited table. The per-term rule reads the
    ``separated_terms`` column written by the fitters (``"|"``-joined names).

    A product term (``is_swr:<covariate>``, ``swr_x_<covariate>``) is judged on
    its OWN coefficient and nothing else. Its ``bse * sd`` already measures how
    well the interaction itself is pinned down, so inheriting a factor's
    separation would drop sessions in which the product is perfectly well
    estimated: an ``is_swr`` main effect with a large SE is a statement about the
    baseline rate difference between the two streams, not about whether the
    covariate's slope differs between them.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits for one (model, population).
    term : str
        The aggregated term.

    Returns
    -------
    pd.Series of bool
        Indexed like ``fits``. Frames written before the term-level flag carry no
        ``separated_terms`` column; for those the fit-level ``separated`` flag is
        the only information available and is used unchanged.
    """
    if "separated_terms" not in fits.columns:
        if "separated" not in fits.columns:
            return pd.Series(False, index=fits.index)
        return fits["separated"] == True  # noqa: E712 (NaN must not count)
    return fits["separated_terms"].map(
        lambda v: isinstance(v, str) and term in v.split("|")
    )


def unidentified_for_term(fits: pd.DataFrame, term: str) -> pd.Series:
    """Per-fit mask: does the fit report ``term`` as one its session cannot inform?

    Read from the ``unidentified_terms`` column the conditional-Poisson fitter
    writes ("|"-joined ``<term>:<reason>``). Such a fit carries a NaN
    coefficient for the term. Frames without the column (older bundles, other
    fitters) report none.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits.
    term : str
        The aggregated term.

    Returns
    -------
    pd.Series of bool
        Indexed like ``fits``.
    """
    if "unidentified_terms" not in fits.columns:
        return pd.Series(False, index=fits.index)
    return (
        fits["unidentified_terms"]
        .map(lambda v: term in parse_unidentified_terms(v))
        .astype(bool)
    )


@dataclass(frozen=True)
class TermSessionGate:
    """Which session fits a term's aggregate keeps, and whether it is emitted.

    Attributes
    ----------
    kept : pandas.Series of bool
        Indexed like the fits; True for a session whose coefficient enters the
        aggregate.
    n_dropped : int
        Sessions with a coefficient that was excluded as separated or runaway.
    n_unidentified : int
        Sessions the fitter reports cannot inform the term (NaN coefficient).
    """

    kept: pd.Series
    n_dropped: int
    n_unidentified: int

    @property
    def n_kept(self) -> int:
        """Sessions whose coefficient enters the aggregate."""
        return int(self.kept.sum())

    @property
    def n_fitted(self) -> int:
        """Sessions fit for the term: kept, dropped, or not informed."""
        return self.n_kept + self.n_dropped + self.n_unidentified

    @property
    def drop_share(self) -> float:
        """Share of the fitted sessions that are dropped or not informed."""
        return (self.n_dropped + self.n_unidentified) / max(self.n_fitted, 1)

    @property
    def refusal(self) -> str | None:
        """Why the term's aggregate is not emitted, or None when it is.

        ``"too_few_sessions"`` below :data:`MIN_SESSIONS_PER_TERM` kept sessions;
        ``"unidentified_share"`` when :attr:`drop_share` exceeds
        :data:`MAX_UNIDENTIFIED_SHARE` -- the surviving median would describe
        the identified minority, not the corpus.
        """
        if self.n_kept < MIN_SESSIONS_PER_TERM:
            return "too_few_sessions"
        if self.drop_share > MAX_UNIDENTIFIED_SHARE:
            return "unidentified_share"
        return None


def gate_term_sessions(fits: pd.DataFrame, term: str) -> TermSessionGate:
    """Apply the per-term identifiability filter and the emission gates.

    Separated or runaway sessions are counted in ``n_dropped``, sessions that
    cannot inform the term in ``n_unidentified``; both enter
    :attr:`TermSessionGate.refusal`. The runaway backstop reads the fit's own
    ``sd_<term>``, which an ``is_swr:<covariate>`` product also carries.

    Parameters
    ----------
    fits : pandas.DataFrame
        Per-session fits for one (model, population), carrying ``coef_<term>``.
    term : str
        The aggregated term.

    Returns
    -------
    TermSessionGate
    """
    ccol = f"coef_{term}"
    present = fits[ccol].notna()
    # Identifiability filter, applied PER TERM. Separation is measured on the
    # effect scale (SE times the predictor's own SD), which is what
    # non-identification actually shows up in, and the fitters name the terms
    # it holds for. The magnitude backstop catches an estimate that ran away
    # without tripping that flag, and it is read on the SAME effect scale:
    # ``|coef| * sd`` is the log-rate swing the predictor actually produces
    # over one SD of itself, so a large coefficient on a tiny-SD covariate --
    # identified, just steep per unit -- is kept, while a genuinely runaway
    # fit is dropped. Judging ``|coef|`` alone was selection ON the estimate:
    # it dropped identified sessions for having small-SD predictors, and it
    # kept whichever unidentified fits happened to land under the threshold,
    # which is how CS3_lastrew came to report -2.72 next to a typical session
    # SE of 2,428 (the identified sessions give -0.77). Precision is
    # deliberately NOT a criterion: the retired ``SE > 10 * |coef|`` rule
    # dropped identified sessions whose effect was near zero, biasing the
    # surviving median away from zero.
    flagged = separated_for_term(fits, term)
    sdcol = f"sd_{term}"
    if sdcol in fits.columns:
        runaway = fits[ccol].abs() * fits[sdcol] >= SEPARATION_COEF_ABS
    else:
        # Legacy artifacts predate the persisted per-fit predictor SDs; the
        # raw coefficient is the only scale available for them.
        runaway = fits[ccol].abs() >= SEPARATION_COEF_ABS
    kept = present & ~runaway & ~flagged
    # A session that cannot inform the term at all (the fitter reports it in
    # ``unidentified_terms``, with a NaN coefficient) is fitted but not
    # present. It is counted apart from the separation drops, and both enter
    # the refusal gate.
    return TermSessionGate(
        kept=kept,
        n_dropped=int((present & ~kept).sum()),
        n_unidentified=int((unidentified_for_term(fits, term) & ~present).sum()),
    )


def session_se(fits: pd.DataFrame, term: str) -> tuple[pd.Series | None, str]:
    """Per-session SE for ``term``, and which SE it is made of.

    The event-clustered SE (``bse_cluster_<term>``) is preferred wherever the
    fitters wrote a finite one. On a fraction response the model SE describes a
    design in which each event contributed one draw and nothing within it
    varied, so it is the clustered SE that reflects the events' own
    variability. A session whose sandwich is not finite keeps its own model SE
    rather than being dropped or filled from another session.

    Parameters
    ----------
    fits : pd.DataFrame
        The per-session fits already kept for this (model, population, term).
    term : str
        The aggregated term.

    Returns
    -------
    (pandas.Series or None, str)
        The per-session SE (None when neither column is present), and
        ``"cluster"`` when more than half of the sessions contributed a
        clustered SE, else ``"model"``. In practice the mix is all-or-nothing:
        a frame either predates the clustered column or carries it for every
        session.
    """
    model = fits.get(f"bse_{term}")
    cluster = fits.get(f"bse_cluster_{term}")
    if cluster is None:
        return (None, "model") if model is None else (model, "model")
    from_cluster = np.isfinite(cluster.to_numpy(dtype=float))
    chosen = (
        cluster.where(from_cluster)
        if model is None
        else cluster.where(from_cluster, model)
    )
    source = "cluster" if from_cluster.sum() > len(fits) / 2 else "model"
    return chosen, source


def typical_session_se(fits: pd.DataFrame, term: str) -> tuple[float, str]:
    """Median of :func:`session_se`, and which SE that median is made of."""
    se, source = session_se(fits, term)
    return (np.nan if se is None else float(se.median())), source


def pct_significant_sessions(
    coef: pd.Series, se: pd.Series | None, pvals: pd.Series | None
) -> float:
    """Share of sessions whose ``term`` is significant at 0.05, two-sided.

    Read on the SAME standard error ``typical_session_se`` reports, so the
    aggregate's dispersion and its significance count answer to one definition
    of uncertainty. That is the event-clustered SE wherever the fitters wrote
    one; the fitted model ``pval_`` is the fallback for a frame that carries no
    SE at all. The two are not interchangeable: on the shipped content fits
    ``bse_cluster / bse`` runs 0.73-0.94, so the model-p version was
    conservative -- it undercounted significant sessions by using a wider SE
    than the clustered one the same row's ``typical_session_se`` quotes.

    Parameters
    ----------
    coef : pandas.Series
        Per-session coefficients.
    se : pandas.Series or None
        Per-session SE from :func:`session_se`, aligned with ``coef``.
    pvals : pandas.Series or None
        Per-session model p-values, used only when ``se`` is None.

    Returns
    -------
    float
        Share in [0, 1], or NaN when neither a usable SE nor a p-value exists.
    """
    if se is not None:
        z = np.abs(coef.to_numpy(dtype=float) / se.to_numpy(dtype=float))
        usable = np.isfinite(z)
        if usable.any():
            return float((z[usable] > _WALD_Z_TWO_SIDED_05).mean())
    if pvals is None:
        return np.nan
    return float((pvals < 0.05).mean())


def gated_term_statistics(
    fits: pd.DataFrame, term: str
) -> tuple[TermSessionGate, dict[str, float | str]]:
    """One term's coefficient summarized over the sessions the per-term gate keeps.

    The statistics are computed whatever the gate's refusal; the caller reads
    :attr:`TermSessionGate.refusal` and decides whether to publish them.

    Parameters
    ----------
    fits : pandas.DataFrame, shape (n_sessions, n_columns)
        Per-session fits of one rung and population, with ``animal``,
        ``coef_<term>`` and the columns :func:`gate_term_sessions` reads
        (optionally ``bse_<term>``, ``bse_cluster_<term>``, ``pval_<term>``).
    term : str
        The term.

    Returns
    -------
    gate : TermSessionGate
        Which sessions are kept, and how many were left out and why.
    statistics : dict
        Over the kept sessions: the names of
        :func:`~continuum_swr_replay.analysis.shared_statistics.session_median_statistics`,
        and

        - ``signed_rank_p``: the Wilcoxon signed-rank p of the coefficients
          against zero
          (:func:`~continuum_swr_replay.analysis.shared_statistics.signed_rank_p`).
        - ``session_se_median``: the median of :func:`session_se`; NaN when the
          fits carry no standard error.
        - ``se_source``: which standard error that is (:func:`session_se`).
        - ``significant_fraction``: :func:`pct_significant_sessions`, read on
          that same standard error.
    """
    gate = gate_term_sessions(fits, term)
    kept = fits[gate.kept]
    coef = kept[f"coef_{term}"]
    se, se_source = session_se(kept, term)
    return gate, {
        **session_median_statistics(coef, kept["animal"]),
        "signed_rank_p": signed_rank_p(coef),
        "session_se_median": np.nan if se is None else float(se.median()),
        "se_source": se_source,
        "significant_fraction": pct_significant_sessions(
            coef, se, kept.get(f"pval_{term}")
        ),
    }


def aggregate_contrast(
    res: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    level: str,
    sd: dict | None = None,
) -> pd.DataFrame:
    """Aggregate per-session coefficients into one audited row per spec.

    Parameters
    ----------
    res : pd.DataFrame
        Per-session fits: columns ``model``, ``population``, ``animal`` and
        ``coef_<term>`` (optionally ``bse_<term>``, ``pval_<term>``,
        ``sd_<term>``). Which sessions a term keeps, and whether its row is
        emitted at all, is :func:`gate_term_sessions`; the row's summary of
        the kept sessions is :func:`gated_term_statistics`.
    specs : list of (model, population, term)
        One aggregated row per spec (in order).
    level : str
        Value for the ``level`` column ("candidate_arm" / "patch" / "rate").
    sd : dict, optional
        Covariate -> global SD, for the standardized (per-SD) columns.
    """
    sd = sd or {}
    rows = []
    for model, pop, term in specs:
        s_all = res[(res.model == model) & (res.population == pop)]
        if f"coef_{term}" not in s_all.columns:
            continue
        # Resolved only for a term the fits carry: a spec with no coefficient
        # column emits no row, so it needs no unit.
        factor, unit = std_factor(term, sd)
        gate, summary = gated_term_statistics(s_all, term)
        if gate.refusal == "too_few_sessions":
            continue
        if gate.refusal == "unidentified_share":
            # The surviving median describes the identified minority, not the
            # corpus, and would read as if it described the corpus.
            logging.warning(
                "%s/%s/%s: %.0f%% of fitted sessions unidentified (%d of %d: %d "
                "separated or runaway, %d not informed); not emitting an "
                "aggregate row",
                model,
                pop,
                term,
                100 * gate.drop_share,
                gate.n_dropped + gate.n_unidentified,
                gate.n_fitted,
                gate.n_dropped,
                gate.n_unidentified,
            )
            continue
        session_se_median = summary["session_se_median"]
        # Session-level fit-diagnostics counts (informational; NOT drops). These stay
        # FIT-level -- they answer "how many fits had any unidentified term", which is
        # a different question from the per-term drop above. Counted over ALL modeled
        # session-fits for this (model, population). A fit with no convergence
        # flag has not converged; a missing separation flag is not separated.
        # Absent columns (older res without diagnostics) count nothing.
        n_nonconverged = (
            int((~s_all["converged"].eq(True)).sum())
            if "converged" in s_all.columns
            else 0
        )
        n_separated = (
            int((s_all["separated"] == True).sum())  # noqa: E712
            if "separated" in s_all.columns
            else 0
        )
        rows.append(
            {
                "level": level,
                "model": model,
                "population": pop,
                "term": term,
                "n_sessions": gate.n_kept,
                "n_animals": summary["n_animals"],
                "n_dropped": gate.n_dropped,
                "n_unidentified": gate.n_unidentified,
                "n_sessions_flagged_nonconverged": n_nonconverged,
                "n_sessions_flagged_separated": n_separated,
                "coef_animal_median": summary["animal_median"],
                "n_animals_pos": summary["n_animals_positive"],
                "coef_session_median": summary["median"],
                "typical_session_se": session_se_median,
                "se_source": summary["se_source"],
                "std_unit": unit,
                "std_factor": factor,
                "std_coef_animal_median": summary["animal_median"] * factor,
                "std_coef_session_median": summary["median"] * factor,
                "std_typical_session_se": session_se_median * factor,
                "pct_positive_session": summary["positive_fraction"],
                "pct_sig_session": summary["significant_fraction"],
                "session_signrank_p": summary["signed_rank_p"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        # Every spec was gated out (all terms unidentified in too large a share
        # of sessions). Return the full schema so callers that select columns
        # (e.g. ``agg["population"].map(...)``) get an empty frame, not a
        # KeyError -- a mostly-separated corpus is a valid, reportable "nothing
        # identified", not a crash.
        return pd.DataFrame(columns=[*_AGG_COLUMNS, "session_signrank_fdr_q"])
    n_dropped_total = int(df["n_dropped"].sum())
    if n_dropped_total:
        logger.info(
            "%s: dropped %d quasi-separated/NaN session-fits before aggregating",
            level,
            n_dropped_total,
        )
    # BH-FDR of the (descriptive) session signed-rank p, across rungs per population.
    df["session_signrank_fdr_q"] = df.groupby("population")[
        "session_signrank_p"
    ].transform(benjamini_hochberg)
    return df


def specs_from_ladder(ladder: dict, key_term: dict) -> list[tuple[str, str, str]]:
    """Expand a content ladder (rung -> (predictors, scope)) into aggregation specs:
    pooled rungs -> the ``all`` population; split rungs -> all / SWR / nonSWR."""
    specs = []
    for name in ladder:
        term = key_term.get(name)
        if term is None:
            continue
        scope = ladder[name][1]
        for pop in ["all"] if scope == "pooled" else ["all", "SWR", "nonSWR"]:
            specs.append((name, pop, term))
    return specs


# ── Content contrast cohort assembly ──

#: Fewest surviving events a session needs before the content contrast fits it.
MIN_EVENTS = 18


def contrast_fit_inputs(root: Path | None = None) -> dict[str, Path]:
    """The tables every content contrast fit is built from, by name.

    Their hashes go into the provenance sidecar of each artifact the content
    contrast runner writes, and the readers of those artifacts check the same
    mapping, so a regenerated table cannot be read through coefficients that
    predate it. `root` is the replay-content root,
    ``paths.replay_content_dir()`` when None, resolved at the call.
    """
    return {
        name: table_path(name, root)
        for name in (
            "event_arm_table",
            "event_patch_table",
            "stop_table",
            "qvals_table",
        )
    }


# event-level scalars z-scored (so :is_current coefs are per-SD & comparable)
_ZSCALARS = [
    "rpe_postoutcome",
    "relative_value_switch_minus_stay_preoutcome",
    "reward_belief_differential_entropy_leaf_mean_preoutcome",
    "reward_belief_mean_update_l1_all_ports_postoutcome",
    "periswitch_index_trials",
    "trials_since_block_change",
    "patch_switch_fraction_last5_completed_trips",
]


# Fit-time standardization: one support label for every ``_z`` these assemblers
# make.
# The moments are taken over the FULL assembled cohort (before the smoke subset and
# before the outcome stratum), so the strata share one per-SD unit and their
# coefficients stay comparable. The reference is persisted beside each artifact.
_FIT_SUPPORT_LABEL = "fit_cohort_all_rows"

# Event-length control. SWR-coupled events are the longer ones, and a longer
# trajectory visits more arms, so ``swr_x_is_current_arm`` partly reads "longer
# events leave the current arm" unless event length is in the design. The control
# is z(log n_nonlocal_bins) crossed with the indicator it modulates.
# ``LOG_EVENT_LENGTH`` and the within-stop-time ``LOG1P_STOP_TIME`` are declared
# with the products they enter, in ``covariates.interactions``.
_ARM_Z_SOURCES = (*_ZSCALARS, LOG1P_STOP_TIME, LOG_EVENT_LENGTH)
_PATCH_Z_SOURCES = (
    "periswitch_index_trials",
    "patch_switch_fraction_last5_completed_trips",
    LOG1P_STOP_TIME,
    LOG_EVENT_LENGTH,
)

# The interaction products each cohort's design carries, in build order. What
# each one multiplies is declared once, in ``covariates.interactions``.
_ARM_INTERACTIONS = (
    "rpe_postoutcome_z_x_is_current_arm",
    "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
    "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm",
    "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
    "periswitch_index_trials_z_x_is_current_arm",
    "is_periswitch_post_x_is_current_arm",
    "periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm",
    "trials_since_block_change_z_x_is_current_arm",
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm",
    "log1p_time_since_stop_start_s_z_x_is_current_arm",
    "log_n_nonlocal_bins_z_x_is_current_arm",
    "log_n_nonlocal_bins_z_x_is_prev_arm",
    "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
)
# The SWR-specific arm baselines of the pooled rungs, built with the arm fixed
# effects they multiply (``model_specs.SWR_ARMFE``).
_SWR_ARM_FE_PRODUCTS = tuple(f"swr_x_is_candidate_arm_{k}" for k in range(8))


def _add_swr_arm_fixed_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the SWR-specific arm baselines (:data:`_SWR_ARM_FE_PRODUCTS`) in place."""
    return attach_interactions(frame, _SWR_ARM_FE_PRODUCTS)


_PATCH_INTERACTIONS = (
    "periswitch_index_trials_z_x_is_current_patch",
    "log1p_time_since_stop_start_s_z_x_is_current_patch",
    "log_n_nonlocal_bins_z_x_is_current_patch",
    "log_n_nonlocal_bins_z_x_is_prev_patch",
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_patch",
    "swr_x_is_current_patch",
)

# Current outcome strata. Rewarded and omitted events differ in their SWR share, so
# the contrast is refit within each; an event whose outcome is not yet known belongs
# to neither.
OUTCOME_STRATA = ("all", "rewarded", "omitted")


def _standardize(frame: pd.DataFrame, cols) -> tuple[pd.DataFrame, dict]:
    """Attach ``<col>_z`` over the fit cohort and return the reference used."""
    return attach_standardized(frame, list(cols), support_label=_FIT_SUPPORT_LABEL)


def _attach_log_event_length(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ``log_n_nonlocal_bins``; refuse an event shorter than one bin.

    Every materialized non-local event spans at least one decoder bin by
    construction, so a zero/negative/missing count is a broken table, not a
    zero-length event -- and ``log`` would silently emit -inf/NaN for it.
    """
    n_bins = pd.to_numeric(frame["n_nonlocal_bins"], errors="coerce").to_numpy(
        dtype=float
    )
    bad = ~np.isfinite(n_bins) | (n_bins < 1)
    if bad.any():
        raise ValueError(
            f"n_nonlocal_bins must be finite and >= 1 on every event; "
            f"{int(bad.sum())} of {len(n_bins)} rows violate it "
            f"(min={np.nanmin(n_bins)})"
        )
    frame[LOG_EVENT_LENGTH] = np.log(n_bins)
    return frame


def select_outcome_stratum(ev: pd.DataFrame, stratum: str) -> tuple[pd.DataFrame, dict]:
    """Restrict a cohort to one current-outcome stratum.

    Parameters
    ----------
    ev : pandas.DataFrame
        Event-grain cohort carrying ``event_id``, ``is_rewarded_outcome`` and
        ``is_swr`` (one row per event x candidate arm/patch).
    stratum : {"all", "rewarded", "omitted"}
        ``"all"`` keeps every event. ``"rewarded"`` keeps
        ``is_rewarded_outcome == 1``, ``"omitted"`` keeps ``== 0``; an event whose
        outcome is not yet known is in neither.

    Returns
    -------
    (frame, audit)
        The restricted frame and the event-grain counts, including the
        unknown-outcome events split by ``is_swr`` so the exclusion is reported.
    """
    if stratum not in OUTCOME_STRATA:
        raise ValueError(
            f"unknown outcome stratum {stratum!r}; expected one of {OUTCOME_STRATA}"
        )
    events = ev.drop_duplicates("event_id")
    outcome = pd.to_numeric(
        events["is_rewarded_outcome"].astype("object"), errors="coerce"
    )
    unknown = outcome.isna()
    is_swr = events["is_swr"].astype("object").astype(bool)
    if stratum == "all":
        kept = ev
    else:
        wanted = 1.0 if stratum == "rewarded" else 0.0
        keep_ids = set(events.loc[~unknown & (outcome == wanted), "event_id"])
        kept = ev[ev["event_id"].isin(keep_ids)]
    audit = {
        "stratum": stratum,
        "n_events_in": int(len(events)),
        "n_events_kept": int(kept["event_id"].nunique()),
        "n_events_unknown_outcome": int(unknown.sum()),
        "n_events_unknown_outcome_swr": int((unknown & is_swr).sum()),
        "n_events_unknown_outcome_non_swr": int((unknown & ~is_swr).sum()),
    }
    return kept, audit


# Raw-scale per-arm covariates whose coefficients the aggregation restates per SD
# (``aggregate_contrast``'s ``std_factor``). Their SDs are frozen over the FULL
# assembled cohort, like the ``_z`` reference, so a stratified refit reports effects
# in the same unit as the pooled one. The patch ladder's key terms are all binary or
# already standardized, so it has no raw-scale term to rescale.
_ARM_SD_TERMS = (
    "reward_belief_mean_candidate_arm_preoutcome",
    "reward_fraction_candidate_arm_last10_visits",
    "task_reward_probability_candidate_arm",
)
_PATCH_SD_TERMS: tuple[str, ...] = ()

# Row support for the per-arm SD unit. Every one of ``_ARM_SD_TERMS`` is a
# leaf-derived belief/reward quantity that is a STRUCTURAL ZERO on the three trunk
# arms, and the rungs that carry them are fitted on the leaves (see
# ``LEAF_ONLY_CONTRAST_RUNGS``). Taking the SD over all nine candidates therefore
# states the effect per SD of a spread the fitted rows never see -- measured on the
# canonical cohort, the trunk zeros inflate the arm-value SD by 2.1x, which would
# shrink every per-SD value effect by the same factor.
_ARM_SD_SUPPORT_LABEL = "leaf_arms"
_PATCH_SD_SUPPORT_LABEL = _FIT_SUPPORT_LABEL


def _leaf_arm_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The six leaf-arm rows of an arm cohort; the trunks carry structural zeros."""
    if "candidate_arm" not in frame.columns:
        raise KeyError(
            "arm cohort has no 'candidate_arm' column; the per-SD unit cannot be "
            "restricted to the leaf arms it is read on"
        )
    return frame[
        pd.to_numeric(frame["candidate_arm"], errors="coerce") < LEAF_ONLY_N_CATEGORIES
    ]


def sd_reference(frame: pd.DataFrame, cols: Sequence[str]) -> dict[str, float]:
    """Pooled SD of each raw covariate present in ``frame``: the per-SD unit.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows of the cohort the unit describes.
    cols : sequence of str
        Covariates; one absent from ``frame`` is left out, not recorded as NaN.

    Returns
    -------
    dict of str to float
        Covariate -> population SD (``ddof=0``) over its non-missing values,
        in the order of ``cols``.
    """
    return {
        c: float(np.nanstd(frame[c].to_numpy(dtype=float)))
        for c in cols
        if c in frame.columns
    }


def prepare_cohorts(
    ev: pd.DataFrame,
    pt: pd.DataFrame,
    *,
    outcome_stratum: str,
    max_per_animal: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Freeze the cohort-wide per-SD scales, then apply the stratum and smoke subset.

    Order is load-bearing: the SDs (like the ``_z`` reference the assemblers
    return) are estimated on the FULL cohorts, so the ``rewarded`` and ``omitted``
    refits state their per-SD effects in the same unit as the pooled fit instead of
    each rescaling to its own stratum.

    Parameters
    ----------
    ev, pt : pandas.DataFrame
        The fully assembled arm and patch cohorts.
    outcome_stratum : {"all", "rewarded", "omitted"}
        Current-outcome stratum to fit.
    max_per_animal : int, optional
        Smoke subset: keep the first N sessions per animal.

    Returns
    -------
    (ev, pt, scale, stratum_audit)
        The selected cohorts, the frozen per-SD scales (each with the row support it
        was taken over -- ``leaf_arms`` for the arm cohort), and the outcome-stratum
        event counts.
    """
    scale = {
        # Support is recorded per cohort because they differ: the arm SDs are taken
        # over the leaf arms only (the trunks are structural zeros for every term the
        # aggregation rescales), the patch SDs over all rows.
        "arm_cohort_support": _ARM_SD_SUPPORT_LABEL,
        "arm_cohort": sd_reference(_leaf_arm_rows(ev), _ARM_SD_TERMS),
        "patch_cohort_support": _PATCH_SD_SUPPORT_LABEL,
        "patch_cohort": sd_reference(pt, _PATCH_SD_TERMS),
    }
    ev, stratum_audit = select_outcome_stratum(ev, outcome_stratum)
    if outcome_stratum != "all":
        # One stratum, one event set: the patch cohort follows the arm cohort's
        # events rather than re-deriving the outcome on patch rows.
        pt = pt[pt["event_id"].isin(set(ev["event_id"]))]
    if max_per_animal is not None:
        ev = subset_max_per_animal(ev, max_per_animal)
        pt = pt.merge(session_key_frame(ev), on=list(SESSION_COLUMNS))
    return ev, pt, scale, stratum_audit


def _cohort_log(label: str, report: dict) -> None:
    """Log one cohort selection at the grain the content contrast reports it on."""
    logger.info(
        "%s cohort = %s (%s): %d -> %d events (%.1f%% retained); dropped events "
        "by state %s",
        label,
        report["cohort"],
        report["rule"],
        report["n_events_before"],
        report["n_events_after"],
        100 * report["n_events_after"] / max(report["n_events_before"], 1),
        {
            state: count
            for state, count in report["counts_before"].items()
            if state not in report["counts_after"]
        },
    )


def assemble_arm(
    fit_inputs: Mapping[str, Path] | None = None,
    reference_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load the arm cohort and attach its design columns and ``_z`` reference.

    Parameters
    ----------
    fit_inputs : Mapping[str, Path], optional
        Table name -> path; the event-arm, stop and qvals tables are read from
        it. :func:`contrast_fit_inputs`, the canonical fit's, when None.
    reference_path : Path, optional
        The tables' standardization reference, against which the persisted
        ``arm_uncertainty_z`` is validated; the canonical root's
        (``paths.STANDARDIZATION_REFERENCE``) when None.

    Returns
    -------
    (ev, reference)
        The at-well arm cohort with its design columns, and the fit-time
        standardization reference actually used.
    """
    if fit_inputs is None:
        fit_inputs = contrast_fit_inputs()
    if reference_path is None:
        reference_path = replay_content_dir() / STANDARDIZATION_REFERENCE
    ev = read_table(fit_inputs["event_arm_table"])
    require_covariate_schema(ev, table="event_arm")
    require_swr_source(ev, what=str(fit_inputs["event_arm_table"]))
    # Cohort: at-well events, stated here rather than left to fall out of the
    # complete-case filter. Every rung carrying `is_current_arm` already lost the
    # off-well events that way (it is NaN off a well), so for those this is a
    # no-op; the arm-fixed-effects rung carries no location predictor and so was
    # being fit on a LARGER population than the rungs it is the null for. Stating
    # the cohort once puts the whole ladder -- and the rate family, which applies
    # the same rule to its stop table -- on one population.
    ev, arm_cohort = select_event_cohort(ev, cohort="at_well")
    _cohort_log("ARM", arm_cohort)
    if not reference_path.exists():
        raise FileNotFoundError(
            f"{reference_path} is required to validate arm_uncertainty_z; "
            "regenerate the canonical replay-content tables"
        )
    try:
        assert_standardized_matches_reference(
            ev,
            ("reward_belief_variance_candidate_arm_preoutcome",),
            load_reference(reference_path),
            support_label="leaf_arms",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "event_arm_table arm_uncertainty_z has a stale or invalid "
            "standardization reference; regenerate the canonical table and "
            f"reference before fitting: {exc}"
        ) from exc
    add_arm_fixed_effects(ev)
    _add_swr_arm_fixed_effects(ev)
    stop = read_table(fit_inputs["stop_table"])
    require_covariate_schema(stop, table="stop")
    require_swr_source(stop, what=str(fit_inputs["stop_table"]))
    # The two columns it would add are read by no fit here; its checks are kept.
    validate_uncertainty_inputs(stop)
    q = read_table(fit_inputs["qvals_table"])
    require_covariate_schema(q, table="qvals")
    ev = ev.merge(
        stop[
            [
                "stop_id",
                "known_reward_outcome_lag_1",
                "patch_switch_fraction_last5_completed_trips",
            ]
        ],
        on="stop_id",
        how="left",
        # One row per stop on the right: a duplicated stop_id would silently
        # duplicate the event-arm rows it joins to.
        validate="many_to_one",
    )
    ev = _merge_qvals(
        ev,
        q,
        [
            "rpe_postoutcome",
            "relative_value_switch_minus_stay_preoutcome",
            "reward_belief_differential_entropy_leaf_mean_preoutcome",
            "reward_belief_mean_update_l1_all_ports_postoutcome",
            "periswitch_index_trials",
            "is_periswitch_post",
            "trials_since_block_change",
        ],
    )
    return _attach_arm_design_columns(ev)


def _merge_qvals(frame: pd.DataFrame, q: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Left-join ``cols`` of ``qvals_table`` by stop, flagging the rows it matched.

    The flag, :data:`~.cohort_audit.HAS_QVALS_ROW`, is what the cohort audit
    books the qvals gap from: a qvals column can also be NaN on a stop that has
    a row, so the column's NaN alone does not say which.
    """
    out = frame.merge(
        q[["stop_id", *cols]],
        on="stop_id",
        how="left",
        # One row per stop on the right: a duplicated stop_id would silently
        # duplicate the rows it joins to.
        validate="many_to_one",
        indicator=HAS_QVALS_ROW,
    )
    out[HAS_QVALS_ROW] = out[HAS_QVALS_ROW].eq("both").to_numpy()
    return out


def _attach_arm_design_columns(ev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize the arm cohort's scalars and build every interaction product.

    Parameters
    ----------
    ev : pandas.DataFrame
        Arm cohort (one row per event x candidate arm) carrying the raw event
        scalars, ``n_nonlocal_bins``, ``time_since_stop_start_s``, and the arm
        indicators.

    Returns
    -------
    (ev, reference)
        ``ev`` with the ``<col>_z`` columns and interaction products appended, and
        the standardization reference actually used (persisted beside the fit).
    """
    ev[LOG1P_STOP_TIME] = np.log1p(ev["time_since_stop_start_s"].to_numpy(dtype=float))
    _attach_log_event_length(ev)
    ev, z_reference = _standardize(ev, _ARM_Z_SOURCES)
    attach_interactions(ev, _ARM_INTERACTIONS)
    return ev, z_reference


def assemble_patch(
    arm_ev: pd.DataFrame, fit_inputs: Mapping[str, Path] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Load the patch cohort and attach its design columns and ``_z`` reference.

    Parameters
    ----------
    arm_ev : pandas.DataFrame
        The assembled arm cohort, which maps each event to its stop and carries
        the event-level time and length the patch table lacks.
    fit_inputs : Mapping[str, Path], optional
        Table name -> path; the event-patch, stop and qvals tables are read from
        it. :func:`contrast_fit_inputs` when None.

    Returns
    -------
    (pt, reference)
        The at-well patch cohort with its design columns, and the fit-time
        standardization reference actually used.
    """
    if fit_inputs is None:
        fit_inputs = contrast_fit_inputs()
    pt = read_table(fit_inputs["event_patch_table"])
    require_covariate_schema(pt, table="event_patch")
    require_swr_source(pt, what=str(fit_inputs["event_patch_table"]))
    pt, patch_cohort = select_event_cohort(pt, cohort="at_well")
    _cohort_log("PATCH", patch_cohort)
    # patch table lacks stop_id: map event_id -> stop_id from the arm table
    e2s = arm_ev[["event_id", "stop_id"]].drop_duplicates("event_id")
    pt = pt.merge(e2s, on="event_id", how="left")
    # Within-stop time and event length are event-level (constant across
    # arms/patches within an event); carry them for the duration and
    # event-length controls.
    e2t = arm_ev[
        ["event_id", "time_since_stop_start_s", "n_nonlocal_bins"]
    ].drop_duplicates("event_id")
    pt = pt.merge(e2t, on="event_id", how="left")
    stop = read_table(fit_inputs["stop_table"])
    require_covariate_schema(stop, table="stop")
    require_swr_source(stop, what=str(fit_inputs["stop_table"]))
    q = read_table(fit_inputs["qvals_table"])
    require_covariate_schema(q, table="qvals")
    pt = pt.merge(
        stop[["stop_id", "patch_switch_fraction_last5_completed_trips"]],
        on="stop_id",
        how="left",
        validate="many_to_one",
    )
    pt = _merge_qvals(pt, q, ["periswitch_index_trials"])
    return _attach_patch_design_columns(pt)


def _attach_patch_design_columns(pt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize the patch cohort's scalars and build every interaction product.

    The patch cohort is its own standardization scope: the same-named arm-cohort
    entries are computed over a different set of rows and must not be reused here.

    Parameters
    ----------
    pt : pandas.DataFrame
        Patch cohort (one row per event x candidate patch) carrying the raw patch
        scalars, ``n_nonlocal_bins``, ``time_since_stop_start_s``, and the patch
        indicators.

    Returns
    -------
    (pt, reference)
        ``pt`` with the ``<col>_z`` columns and interaction products appended, and
        the standardization reference actually used.
    """
    pt[LOG1P_STOP_TIME] = np.log1p(pt["time_since_stop_start_s"].to_numpy(dtype=float))
    _attach_log_event_length(pt)
    pt, z_reference = _standardize(pt, _PATCH_Z_SOURCES)
    attach_interactions(pt, _PATCH_INTERACTIONS)
    return pt, z_reference


# Event-level columns carried for the cohort-audit drop profile (mechanism A vs B).
_PROFILE_COLS = (
    "time_since_stop_start_s",
    "is_rewarded_outcome",
    "reward_belief_mean_candidate_arm_preoutcome",
    "relative_value_switch_minus_stay_preoutcome",
)


def prepare_session(
    g: pd.DataFrame, predictors: list[str], *, response_col: str
) -> tuple[pd.DataFrame | None, dict]:
    """Apply the event-level cohort rules to one session, before any fit.

    Returns ``(fittable_rows_or_None, prep)``. ``prep`` carries the event-level
    cohort counts (``n_in`` and the mutually exclusive ``n_drop_qvals_gap`` /
    ``n_drop_own_predictor_missing`` / ``n_drop_zero_response``), the surviving
    event count, a skip ``reason``, and a one-row-per-dropped-event ``dropped``
    frame for the profile.

    Separated from the fit itself because the Dirichlet-multinomial fits a whole
    ladder's worth of sessions in one batch, and the cohort rules that decide
    which rows it may see must be the same ones the conditional Poisson applies.

    A session missing a required predictor column is a clean, auditable skip
    (``reason="missing:<cols>"`` → ``n_excluded_session``), never a fitter call
    with absent columns.
    """
    n_in = int(g["event_id"].nunique()) if "event_id" in g.columns else 0
    empty = g.iloc[0:0]
    required = [response_col, "event_id", *predictors]
    missing = [c for c in required if c not in g.columns]
    if missing:
        return None, {
            "n_in": n_in,
            "n_qvals": 0,
            "n_own": 0,
            "n_zero": 0,
            "n_surv": n_in,
            "dropped": empty,
            "reason": f"missing:{','.join(missing)}",
        }

    # Split NaN-dropped events into the qvals gap (mechanism B: a qvals-sourced
    # predictor NaN on a stop with no qvals row, decided per row) vs own-predictor
    # missingness (mechanism A: every other predictor NaN, a qvals column masked
    # on a stop that has a row included), qvals taking priority so the buckets
    # stay disjoint.
    qvals_nan = qvals_gap_event_ids(g, predictors, event_col="event_id")
    # The response is not a predictor. An event whose response is undefined --
    # every candidate NaN, which is what a leaf-only event with no leaf mass
    # renormalizes to -- has no allocation to fit, and that is a response drop,
    # not predictor missingness. Keeping it out of this split books it under
    # n_drop_zero_response below, as the ladder's cohort audit does.
    other_nan = (
        set(g.loc[g[predictors].isna().any(axis=1), "event_id"])
        if predictors
        else set()
    )
    other_nan -= qvals_nan
    n_qvals, n_own = len(qvals_nan), len(other_nan)
    dropped_events = qvals_nan | other_nan

    # Preserve the full candidate set within every retained event (row-wise NaN drops
    # would shrink an event's arm/patch set and desync it from the null's n_categories).
    surv = g[~g["event_id"].isin(dropped_events)]
    # skipna=False: an event missing the response on any candidate has no
    # defined allocation, and fitting the candidates it does have would shrink
    # its candidate set out of step with the null's n_categories. Undefined and
    # empty allocations are the same drop -- there is nothing to fit.
    totals = surv.groupby("event_id")[response_col].agg(
        lambda values: values.sum(skipna=False)
    )
    zero_events = set(totals[totals.isna() | totals.le(0)].index)
    n_zero = len(zero_events)
    surv = surv[~surv["event_id"].isin(zero_events)]
    n_surv = int(surv["event_id"].nunique())

    dropped = _dropped_event_rows(g, dropped_events | zero_events)
    prep = {
        "n_in": n_in,
        "n_qvals": n_qvals,
        "n_own": n_own,
        "n_zero": n_zero,
        "n_surv": n_surv,
        "dropped": dropped,
        "reason": "",
    }
    if n_surv < MIN_EVENTS:
        return None, {**prep, "reason": "too_few"}
    return surv, prep


def _dropped_event_rows(g: pd.DataFrame, event_ids: set) -> pd.DataFrame:
    """One row per dropped event, profile columns only (for drop_profile)."""
    cols = [c for c in _PROFILE_COLS if c in g.columns]
    sub = g[g["event_id"].isin(event_ids)]
    return sub.drop_duplicates("event_id")[cols] if cols else g.iloc[0:0]
