"""Shared fit-diagnostics helpers: convergence, separation, warning capture.

These make convergence and quasi-separation first-class, *flagged* outputs of every
fitter (conditional Poisson/logit, per-stop GLM, per-bin GLM, choice logit) without
changing any estimand. A fit that was silently non-converged now carries
``converged=False``; a quasi-separated fit (a coefficient blown up to a large finite
value or an exploded/non-finite SE) carries ``separated=True``.

Nothing here *drops* a fit. The only exclusion in the pipeline remains the pre-existing
``|coef| >= 10`` quasi-separation filter in ``contrast_aggregation`` (per session, per
term); these flags are diagnostics on the retained cohort.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationWarning,
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

# Warning categories that report a fit-quality problem (as opposed to benign numeric
# noise). Counted per fit and surfaced as ``n_convergence_warnings``.
FIT_QUALITY_WARNINGS = (ConvergenceWarning, PerfectSeparationWarning)


def fit_status(result, kind: str, *, caught=None, predictor_sd=None) -> dict:
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

    Returns
    -------
    dict
        ``{"converged": bool | None, "separated": bool}``.
    """
    if kind == "glm":
        converged = bool(getattr(result, "converged", False))
    elif kind == "mle":
        converged = bool((getattr(result, "mle_retvals", {}) or {}).get("converged", False))
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
    degenerate = bool(np.any(~np.isfinite(bse)) or np.any(bse > _SEPARATION_BSE_ABS))
    if predictor_sd is None:
        unpinned = bool(np.any(np.abs(params) >= SEPARATION_COEF_ABS))
    else:
        scale = np.asarray(predictor_sd, dtype=float)
        if scale.shape != bse.shape:
            raise ValueError(
                f"predictor_sd has shape {scale.shape}; expected {bse.shape} to match "
                "the fitted coefficients"
            )
        # A zero-SD column cannot be judged on the effect scale; fall back for it.
        usable = np.isfinite(scale) & (scale > 0)
        unpinned = bool(
            np.any(bse[usable] * scale[usable] >= SEPARATION_EFFECT_SE)
            or np.any(np.abs(params[~usable]) >= SEPARATION_COEF_ABS)
        )
    return {"converged": converged, "separated": degenerate or unpinned}


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
