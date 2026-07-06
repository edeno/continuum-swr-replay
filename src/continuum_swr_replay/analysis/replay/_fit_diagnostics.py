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

# Matches the quasi-separation filter in contrast_aggregation.py (|coef| >= 10). A
# well-identified coefficient in these models is O(1); a coefficient at/above 10 (or an
# SE that has exploded or gone non-finite) signals (quasi-)separation.
SEPARATION_COEF_ABS = 10.0
_SEPARATION_BSE_ABS = 1e3

# Warning categories that report a fit-quality problem (as opposed to benign numeric
# noise). Counted per fit and surfaced as ``n_convergence_warnings``.
FIT_QUALITY_WARNINGS = (ConvergenceWarning, PerfectSeparationWarning)


def fit_status(result, kind: str) -> dict:
    """Convergence + separation flags for a fitted statsmodels result.

    Parameters
    ----------
    result : statsmodels results object
        A fitted result with ``params`` and ``bse``.
    kind : {"glm", "mle"}
        ``"glm"`` for :class:`~statsmodels.genmod.GLMResults` (convergence read from
        ``result.converged``); ``"mle"`` for discrete MLE results
        (ConditionalPoisson/ConditionalLogit/Logit — convergence read from
        ``result.mle_retvals["converged"]``).

    Returns
    -------
    dict
        ``{"converged": bool, "separated": bool}``.
    """
    if kind == "glm":
        converged = bool(getattr(result, "converged", False))
    elif kind == "mle":
        converged = bool((getattr(result, "mle_retvals", {}) or {}).get("converged", False))
    else:
        raise ValueError(f"kind must be 'glm' or 'mle', got {kind!r}")

    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    separated = bool(
        np.any(np.abs(params) >= SEPARATION_COEF_ABS)
        or np.any(~np.isfinite(bse))
        or np.any(bse > _SEPARATION_BSE_ABS)
    )
    return {"converged": converged, "separated": separated}


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
