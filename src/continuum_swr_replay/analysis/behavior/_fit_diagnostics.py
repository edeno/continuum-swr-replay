"""Shared fit-diagnostics helpers: convergence, separation, warning capture, flag counts.

These make convergence and quasi-separation first-class, *flagged* outputs of every
fitter (conditional Poisson/logit, per-stop GLM, per-bin GLM, choice logit) without
changing any estimand. A fit that was silently non-converged now carries
``converged=False``; a quasi-separated fit (a coefficient blown up to a large finite
value or an exploded/non-finite SE) carries ``separated=True``.

Separation is judged **per term**: :func:`fit_status` returns the names of the terms
that are not identified, and the fit-level ``separated`` flag is simply "at least one
term". Structural nuisance terms (:data:`NUISANCE_TERMS`) are excluded from the
judgment entirely -- a blown-up intercept or a blown-up structural-missingness
indicator says nothing about whether the scientific terms of the model are identified,
and flagging the whole fit on them discards every identified coefficient it carries.

Nothing here *drops* a fit. The only exclusion in the pipeline remains the per-term
quasi-separation filter in ``contrast_aggregation``; these flags are diagnostics on the
retained cohort.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd
from patsy import PatsyError
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

# Numerical failures we expect from a per-session fit/predict and treat as a
# skipped cell (-> None): rank-deficient/ill-conditioned designs, spline knots
# the held-out fold extrapolates past (patsy), perfect separation, optimizer
# failures. Anything else (e.g. a NameError in a spec formula, an offset shape
# mismatch) is a coding bug and is left to propagate rather than be swallowed as
# "fit failed" for every session. Every fitter that skips a failed fit catches
# exactly these.
EXPECTED_FIT_ERRORS = (
    np.linalg.LinAlgError,
    ValueError,
    RuntimeError,
    PerfectSeparationError,
    PatsyError,
)

# Fallback threshold, used only when the caller cannot supply predictor scales. It
# assumes a well-identified coefficient is O(1), which holds for the 0/1 indicators the
# ladders carried historically but NOT for a continuous basis column: a spline column
# with SD 0.03 reaches |coef| = 10 at an effect of 0.3 log-units, an entirely ordinary
# size. That mis-calibration flagged 148 of 213 sessions at M6 when the recency spline
# landed, none of them actually separated.
SEPARATION_COEF_ABS = 10.0
_SEPARATION_BSE_ABS = 1e3

# Scale-free separation criterion, preferred whenever predictor SDs are available.
# Separation is non-identification, which is a property of the STANDARD ERROR, not of
# the coefficient: the likelihood runs monotonically to a bound and the error diverges
# with it. Measured on this corpus, the standard error expressed per SD of its own
# predictor separates the two cases where the coefficient cannot -- a genuinely
# separated term (`is_last_rewarded_elsewhere`, zero replay mass on the marked arm)
# exceeds 1.0 in 49% of sessions, matching the 47% independently measured as degenerate,
# while an ordinary spline basis column exceeds it in 5.2%. Scaling the *coefficient*
# instead does not work: on that scale genuine separation reads 1.37 and ordinary spline
# sessions reach 5.7.
SEPARATION_EFFECT_SE = 1.0

# Terms whose identifiability is structurally uninteresting and must never decide
# whether a fit is usable. ``Intercept`` absorbs the baseline rate and is not a
# reported effect. ``reward_history_incomplete`` marks the handful of stops with fewer
# than two known outcomes; in most sessions those stops carry no events, so its
# coefficient runs to -inf with an exploded SE. Judging the fit on either one flagged
# 55% of the is_swr-contrast fits and left the audited aggregate empty, while every
# scientific term in those same fits was ordinary.
NUISANCE_TERMS = frozenset({"Intercept", "reward_history_incomplete"})

# Warning categories that report a fit-quality problem (as opposed to benign numeric
# noise). Counted per fit and surfaced as ``n_convergence_warnings``.
FIT_QUALITY_WARNINGS = (ConvergenceWarning, PerfectSeparationWarning)

# The counts of flagged fits, by the names a cohort-audit record and an aggregated
# contrast row carry them under (:func:`fit_flag_counts`).
FLAG_FIELDS = ("n_sessions_flagged_nonconverged", "n_sessions_flagged_separated")


def _resolve_term_names(result, term_names, n_terms: int) -> list[str]:
    """Names for the fitted coefficients, in ``result.params`` order.

    Parameters
    ----------
    result : statsmodels results object
        Used only as a fallback source of names (``result.params.index`` for a
        formula fit).
    term_names : sequence of str or None
        Caller-supplied names. When given they must match the coefficient count.
    n_terms : int
        Number of fitted coefficients.

    Returns
    -------
    list of str
        Length ``n_terms``. Falls back to the params index, then to positional
        ``x0``, ``x1``, ... names for an array fit with no index.
    """
    if term_names is not None:
        names = [str(t) for t in term_names]
        if len(names) != n_terms:
            raise ValueError(
                f"term_names has {len(names)} entries; expected {n_terms} to match "
                "the fitted coefficients"
            )
        return names
    index = getattr(result.params, "index", None)
    if index is not None and len(index) == n_terms:
        return [str(t) for t in index]
    return [f"x{i}" for i in range(n_terms)]


def fit_status(
    result,
    kind: str,
    *,
    caught=None,
    predictor_sd=None,
    term_names: Sequence[str] | None = None,
    nuisance_terms: Collection[str] = (),
) -> dict:
    """Convergence + separation flags for a fitted statsmodels result.

    Parameters
    ----------
    result : statsmodels results object
        A fitted result with ``params`` and ``bse``.
    kind : {"glm", "mle", "conditional"}
        ``"glm"`` for :class:`~statsmodels.genmod.GLMResults` (convergence read from
        ``result.converged``); ``"mle"`` for plain discrete MLE results (e.g.
        ``sm.Logit`` — convergence read from ``result.mle_retvals["converged"]``);
        ``"conditional"`` for ``ConditionalPoisson``/``ConditionalLogit``, whose
        ``ConditionalResultsWrapper`` **discards** the optimizer's ``mle_retvals``
        (statsmodels rebuilds a bare ``ConditionalResults`` and throws the raw fit
        result away), so neither ``.converged`` nor ``.mle_retvals`` exists. For that
        case convergence is recovered from ``caught``: a non-converged conditional fit
        emits exactly one ``ConvergenceWarning`` (verified), so ``converged`` is
        ``True`` when no ``ConvergenceWarning`` was captured, ``False`` when one was,
        and ``None`` when the caller did not capture warnings (unknown).
    caught : list, optional
        The warnings list from :func:`capture_fit_warnings`. Required for
        ``kind="conditional"`` to recover convergence; ignored otherwise.
    predictor_sd : array-like, optional
        Per-coefficient standard deviation of the corresponding design column, in
        the same order as ``result.params``. When given, separation is judged on
        the effect scale (``bse * sd``), which is scale-free; without it the
        legacy ``|coef| >= 10`` rule applies and mis-flags small-scale columns.
    term_names : sequence of str, optional
        Coefficient names in ``result.params`` order, used to name the separated
        terms and to apply ``nuisance_terms``. Defaults to ``result.params.index``
        for a formula fit and to positional ``x0``, ``x1``, ... otherwise.
    nuisance_terms : collection of str, optional
        Terms excluded from the separation judgment entirely. Pass
        :data:`NUISANCE_TERMS` from every fitter whose design carries an intercept
        or the structural-missingness indicator.

    Returns
    -------
    dict
        ``{"converged": bool | None, "separated": bool, "separated_terms": tuple}``.
        ``separated_terms`` names the non-nuisance terms that are not identified, in
        coefficient order; ``separated`` is ``True`` iff that tuple is non-empty.
    """
    if kind == "glm":
        converged = bool(getattr(result, "converged", False))
    elif kind == "mle":
        converged = bool(
            (getattr(result, "mle_retvals", {}) or {}).get("converged", False)
        )
    elif kind == "conditional":
        if caught is None:
            converged = None  # statsmodels does not expose it; caller didn't capture
        else:
            converged = not any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )
    else:
        raise ValueError(f"kind must be 'glm', 'mle', or 'conditional', got {kind!r}")

    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    names = _resolve_term_names(result, term_names, params.size)

    terms = separated_terms(
        params,
        bse,
        predictor_sd=predictor_sd,
        term_names=names,
        nuisance_terms=nuisance_terms,
    )
    return {
        "converged": converged,
        "separated": len(terms) > 0,
        "separated_terms": terms,
    }


def separated_terms(
    params,
    bse,
    *,
    predictor_sd=None,
    term_names: Sequence[str],
    nuisance_terms: Collection[str] = (),
) -> tuple[str, ...]:
    """Names of the terms that are not identified, by the effect-scale rule.

    Shared by every fitter so the threshold lives in one place: a statsmodels
    result goes through :func:`fit_status`, while a fitter that optimizes its own
    likelihood (and so has no statsmodels result object) calls this directly with
    its parameter and standard-error vectors.

    Parameters
    ----------
    params : array-like, shape (n_terms,)
        Fitted coefficients.
    bse : array-like, shape (n_terms,)
        Standard errors, in the same order.
    predictor_sd : array-like, optional
        Per-coefficient standard deviation of the corresponding design column.
        When given, separation is judged on the scale-free effect scale
        (``bse * sd >= SEPARATION_EFFECT_SE``); without it, and for any column
        whose SD is 0 or non-finite, the absolute rule ``|coef| >=
        SEPARATION_COEF_ABS`` applies.
    term_names : sequence of str
        Coefficient names, in ``params`` order.
    nuisance_terms : collection of str, optional
        Terms excluded from the judgment entirely.

    Returns
    -------
    tuple of str
        The non-nuisance terms that are not identified, in coefficient order.
    """
    params = np.asarray(params, dtype=float)
    bse = np.asarray(bse, dtype=float)
    names = [str(t) for t in term_names]
    if len(names) != params.size:
        raise ValueError(
            f"term_names has {len(names)} entries; expected {params.size} to match "
            "the fitted coefficients"
        )

    # Every clause is per term: a fit is only as unidentified as its worst
    # *scientific* term, and naming that term is what lets the aggregation drop the
    # session for it alone instead of for every coefficient the fit carries.
    degenerate = ~np.isfinite(bse) | (bse > _SEPARATION_BSE_ABS)
    if predictor_sd is None:
        unpinned = np.abs(params) >= SEPARATION_COEF_ABS
    else:
        scale = np.asarray(predictor_sd, dtype=float)
        if scale.shape != bse.shape:
            raise ValueError(
                f"predictor_sd has shape {scale.shape}; expected {bse.shape} to match "
                "the fitted coefficients"
            )
        # A zero-SD column cannot be judged on the effect scale; fall back for it.
        usable = np.isfinite(scale) & (scale > 0)
        unpinned = np.where(
            usable,
            bse * scale >= SEPARATION_EFFECT_SE,
            np.abs(params) >= SEPARATION_COEF_ABS,
        )
    excluded = frozenset(nuisance_terms)
    return tuple(
        name
        for name, name_flag in zip(names, degenerate | unpinned, strict=True)
        if name_flag and name not in excluded
    )


@contextlib.contextmanager
def capture_fit_warnings():
    """Record (rather than suppress) warnings raised during a fit.

    Yields the list that :func:`warnings.catch_warnings` fills. ``simplefilter("always")``
    is required: without it a warning that already fired once earlier in the process is
    de-duplicated by Python's default filter and never recorded, so per-fit counts would
    silently read zero after the first occurrence.

    Use :func:`count_fit_quality_warnings` on the yielded list to tally
    convergence/separation warnings.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def count_fit_quality_warnings(caught) -> int:
    """Number of convergence/perfect-separation warnings in a captured-warnings list."""
    return sum(1 for w in caught if issubclass(w.category, FIT_QUALITY_WARNINGS))


def flagged_separated(fits: pd.DataFrame) -> pd.Series:
    """Which fits are flagged separated; a missing flag is not separated.

    Parameters
    ----------
    fits : pd.DataFrame, shape (n_fits, n_columns)
        Per-session fits with the fit-level ``separated`` flag.

    Returns
    -------
    separated : pd.Series, shape (n_fits,)
        Boolean, on the index of ``fits``; ``True`` only where the flag is
        ``True``, so a NaN or an absent value is not separated.
    """
    return fits["separated"].eq(True)


def fit_flag_counts(fits: pd.DataFrame) -> dict[str, int]:
    """The fits flagged as not converged and as separated: the :data:`FLAG_FIELDS`.

    Parameters
    ----------
    fits : pd.DataFrame, shape (n_fits, n_columns)
        Per-session fits, optionally with ``converged`` and ``separated``.

    Returns
    -------
    counts : dict
        - ``n_sessions_flagged_nonconverged``: fits whose ``converged`` is not
          True; a missing flag has not converged.
        - ``n_sessions_flagged_separated``: fits whose ``separated`` is True; a
          missing flag is not separated (:func:`flagged_separated`).

        A column the fits do not carry counts 0. The analytic null rung of the
        conditional-Poisson content ladder
        (``fitting.fit_null_model_conditional``) records ``converged=True`` and
        ``separated=False``; a null rung fitted before that was recorded has no
        flags and counts as not converged.
    """
    nonconverged, separated = FLAG_FIELDS
    return {
        nonconverged: (
            int((~fits["converged"].eq(True)).sum()) if "converged" in fits else 0
        ),
        separated: int(flagged_separated(fits).sum()) if "separated" in fits else 0,
    }
