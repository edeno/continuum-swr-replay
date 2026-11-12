"""Which sessions of a rung pair can be compared by an information criterion.

AIC, BIC and deviance differences between two rungs mean something only when
the two fits used the same rows on the same likelihood scale. This module owns
that rule; every AIC or BIC difference over sessions applies it, in the rate and
content ladders alike
(:func:`~continuum_swr_replay.analysis.behavior.coefficient_statistics.aic_difference_statistics`,
:func:`~continuum_swr_replay.analysis.behavior.content.model_comparison.compare_nested_models`).
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "COHORT_SIZE_COLUMNS",
    "COHORT_SIZE_REASON",
    "COMPARABILITY_COLUMNS",
    "INVALIDATING_METADATA_COLUMNS",
    "why_not_comparable",
]

#: Metadata columns whose disagreement invalidates a per-session comparison, and
#: why. ``llf_includes_constants``: the Dirichlet-multinomial likelihood carries
#: the response's normalizing constants and a statsmodels conditional likelihood
#: does not. ``time_clock``: a per-bin rung anchored on the arrival clock is fit
#: on the arrival window, a strict subset of the frozen cohort's rows, so a
#: cross-clock delta measures the window rather than the rungs. The row-count
#: guard below would catch that too, but the clock says WHY.
INVALIDATING_METADATA_COLUMNS = {
    "llf_includes_constants": (
        "compares a likelihood that includes the response's normalizing "
        "constants against one that does not — equalize the constants (see "
        "dirichlet_multinomial.multinomial_log_constant) before differencing "
        "information criteria across estimators"
    ),
    "time_clock": (
        "compares rungs anchored on different time clocks, which are fit to "
        "different rows (an arrival-clock rung is fit on the arrival window) — "
        "compare an arrival rung with its own clock's null instead"
    ),
    "stop_cohort": (
        "compares rungs fit on different stop populations (at-well stops "
        "against all stops or in-transit stops), which are different sets of "
        "rows and different science — compare a rung with a null fit on its "
        "own cohort instead"
    ),
}

#: Cohort-size columns: rungs fit to different numbers of events, bins or stops
#: were fit to different rows, and AIC, BIC and deviance all scale with the rows,
#: so the rung fit to fewer wins mechanically. Which one is the unit depends on
#: the family: events (conditional arm/patch), bins (per-bin conditional
#: intensity), stops (per-stop rate).
COHORT_SIZE_COLUMNS = ("n_events", "n_bins", "n_stops")

#: The columns a session's two fits must agree on to be compared
#: (:func:`why_not_comparable`).
COMPARABILITY_COLUMNS = (*COHORT_SIZE_COLUMNS, *INVALIDATING_METADATA_COLUMNS)

#: Why a session whose fits differ on a cohort-size column is not compared.
COHORT_SIZE_REASON = (
    "have mismatched cohort sizes between parent and child — freeze one cohort "
    "for the ladder before fitting"
)


def why_not_comparable(paired: pd.DataFrame, suffixes: tuple[str, str]) -> pd.DataFrame:
    """Which sessions of a rung pair can be compared, and why the others cannot.

    Parameters
    ----------
    paired : pd.DataFrame, shape (n_sessions, n_columns)
        The two rungs' fits merged on session, each rung's columns carried with
        its suffix. A column of :data:`COMPARABILITY_COLUMNS` is checked when
        both rungs carry it.
    suffixes : tuple of str
        The two rungs' column suffixes, for example ``("_parent", "_child")``.

    Returns
    -------
    differs : pd.DataFrame of bool, shape (n_sessions, n_checked)
        On ``paired``'s index, one column per checked column of
        :data:`COMPARABILITY_COLUMNS`: True where the session's two fits differ
        on it. A session can be compared when no column is True. A
        cohort-size column (:data:`COHORT_SIZE_COLUMNS`) differs wherever its two
        values differ, a missing one included: rows of unknown number are not
        known to be the same rows. A column of
        :data:`INVALIDATING_METADATA_COLUMNS` differs only where both fits state
        a value: a fit that records nothing is unknown, not different.
    """
    first, second = suffixes
    differing = {}
    for column in COMPARABILITY_COLUMNS:
        a, b = f"{column}{first}", f"{column}{second}"
        if a not in paired or b not in paired:
            continue
        differs = paired[a] != paired[b]
        if column not in COHORT_SIZE_COLUMNS:
            differs &= paired[a].notna() & paired[b].notna()
        differing[column] = differs.astype(bool)
    return pd.DataFrame(differing, index=paired.index, dtype=bool)
