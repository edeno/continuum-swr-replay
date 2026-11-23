"""The estimands of the visit rate model, per session and over sessions.

Each declared estimand of the model (Q1, Q2, Q7a; ``declared_tests``) is a
linear functional of model A's coefficients (``visit_rate.declared_contrasts``),
so its variance in a session is exact: ``c' V c`` with ``V`` the fit's
Bayesian covariance (``regression.contrasts``). A session without a kept fit,
or with a coefficient the contrast uses left out, has no value (NaN).

Over sessions, each estimand is summarized for each kind and for the
difference between the kinds (the SWR session value less the No SWR one, in
the sessions where both have a value) by ``shared_statistics.estimand_summary``:
session -> recording day -> animal, the animal-balanced estimate with its
nested bootstrap interval, the animals with its sign and the day-level test.
The estimates are on the log scale (a log rate ratio).

Model D's curve over lags 1 to 10 is described, not tested: per session with
its simultaneous band (``bands.simultaneous_band``), and the animal-balanced
curve across sessions with its band (``bands.across_session_band``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.bands import (
    Band,
    across_session_band,
    simultaneous_band,
)
from continuum_swr_replay.analysis.regression.contrasts import evaluate_contrast
from continuum_swr_replay.analysis.regression.newton import CONVERGED
from continuum_swr_replay.analysis.regression.record import FitRecord
from continuum_swr_replay.analysis.shared_statistics import estimand_summary
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ..declared_tests import DIFFERENCE, KINDS
from .visit_rate import LAGS, VisitRateFit, declared_contrasts, lag_curve_rows

__all__ = [
    "lag_curve_band",
    "lag_curves",
    "session_estimands",
    "summarize_estimands",
]


def _kept(record: FitRecord) -> bool:
    """A record whose fit is kept (converged, no ``no_fit`` cause)."""
    return record.status == CONVERGED and not record.no_fit


def session_estimands(fit: VisitRateFit) -> pd.DataFrame:
    """Q1, Q2 and Q7a in each session of model A's fit, with their variance.

    Parameters
    ----------
    fit : VisitRateFit
        Model A's fit of one kind (``visit_rate.fit_visit_rate``).

    Returns
    -------
    pd.DataFrame
        One row per session and estimand: ``animal``, ``session_date``,
        ``epoch``, ``kind``, ``estimand_id``, ``estimate``, ``variance`` and
        ``se``; NaN for a session without a kept fit or with a coefficient
        the contrast uses left out.
    """
    if fit.model != "A":
        raise ValueError(f"the declared estimands are model A's, not {fit.model}'s")
    c = declared_contrasts(fit.design, fit.settings)
    rows = []
    for record in fit.records:
        beta = np.asarray(record.coef, dtype=float)
        vcov = np.asarray(record.vcov, dtype=float)
        for estimand_id, contrast in c.items():
            value = evaluate_contrast(contrast, beta, vcov) if _kept(record) else None
            estimate = np.nan if value is None else value.estimate
            variance = np.nan if value is None else value.variance
            rows.append(
                {
                    "animal": record.animal,
                    "session_date": record.session_date,
                    "epoch": record.epoch,
                    "kind": fit.kind,
                    "estimand_id": estimand_id,
                    "estimate": estimate,
                    "variance": variance,
                    "se": float(np.sqrt(variance)),
                }
            )
    return pd.DataFrame(rows)


def _summary(values: pd.DataFrame, n_draws: int, seed: int) -> dict[str, float]:
    """``estimand_summary`` of one estimand's session values (one per
    session): day = recording day, session = epoch within it."""
    values = values.reset_index(drop=True)
    return estimand_summary(
        values["estimate"],
        values["animal"],
        values["session_date"],
        values["epoch"],
        n_draws=n_draws,
        seed=seed,
    )


def summarize_estimands(
    session_values: pd.DataFrame, *, n_draws: int, seed: int
) -> pd.DataFrame:
    """Each estimand over sessions, for each kind and for their difference.

    Parameters
    ----------
    session_values : pd.DataFrame
        :func:`session_estimands` of both kinds, concatenated.
    n_draws, seed : int
        The nested bootstrap's (``estimand_summary``), the same for every row.

    Returns
    -------
    pd.DataFrame
        One row per estimand and tested (``SWR``, ``No SWR``, ``SWR minus No
        SWR``): ``estimand_id``, ``tested`` and the entries of
        ``estimand_summary`` (on the log scale). The difference is over the
        sessions where both kinds have a value.

    Raises
    ------
    ValueError
        If a kind is missing, or a session has more than one value of an
        estimand and kind.
    """
    missing = set(KINDS) - set(session_values["kind"])
    if missing:
        raise ValueError(f"no session values of {sorted(missing)}")
    keys = [*SESSION_COLUMNS, "estimand_id", "kind"]
    if session_values.duplicated(keys).any():
        raise ValueError("a session has more than one value of an estimand and kind")
    out = []
    for estimand_id, values in session_values.groupby("estimand_id", sort=False):
        for kind in KINDS:
            summary = _summary(values[values["kind"] == kind], n_draws, seed)
            out.append({"estimand_id": estimand_id, "tested": kind, **summary})
        wide = values.pivot(
            index=list(SESSION_COLUMNS), columns="kind", values="estimate"
        ).reset_index()
        wide["estimate"] = wide[KINDS[0]] - wide[KINDS[1]]
        summary = _summary(wide, n_draws, seed)
        out.append({"estimand_id": estimand_id, "tested": DIFFERENCE, **summary})
    return pd.DataFrame(out)


def lag_curves(fit: VisitRateFit, *, seed: int) -> pd.DataFrame:
    """Model D's curve over lags in each session, with its simultaneous band.

    Parameters
    ----------
    fit : VisitRateFit
        Model D's fit of one kind.
    seed : int
        The band's seed (``bands.simultaneous_band``), the same in every
        session.

    Returns
    -------
    pd.DataFrame
        One row per session and lag: the session, ``kind``, ``lag``,
        ``estimate``, ``se``, ``low``, ``high`` and ``critical``. A session
        without a kept fit, or with the curve left out, has NaN.
    """
    if fit.model != "D":
        raise ValueError(f"the lag curve is model D's, not {fit.model}'s")
    C = lag_curve_rows(fit.design)
    rows = []
    for record in fit.records:
        beta = np.asarray(record.coef, dtype=float)
        used = np.any(C != 0.0, axis=0)
        if _kept(record) and np.isfinite(beta[used]).all():
            band = simultaneous_band(C, beta, np.asarray(record.vcov), seed=seed)
            values = (band.estimate, band.se, band.low, band.high, band.critical)
        else:
            values = (*(np.full(len(LAGS), np.nan),) * 4, np.nan)
        estimate, se, low, high, critical = values
        for j, lag in enumerate(LAGS):
            rows.append(
                {
                    "animal": record.animal,
                    "session_date": record.session_date,
                    "epoch": record.epoch,
                    "kind": fit.kind,
                    "lag": lag,
                    "estimate": estimate[j],
                    "se": se[j],
                    "low": low[j],
                    "high": high[j],
                    "critical": critical,
                }
            )
    return pd.DataFrame(rows)


def lag_curve_band(
    curves: pd.DataFrame, *, n_draws: int, seed: int, lags: Sequence[int] = LAGS
) -> Band:
    """The animal-balanced lag curve across sessions, with its band.

    Parameters
    ----------
    curves : pd.DataFrame
        :func:`lag_curves` of one kind.
    n_draws, seed : int
        As ``bands.across_session_band`` takes them.
    lags : sequence of int
        The lags of the grid (all by default).

    Returns
    -------
    Band
    """
    if curves["kind"].nunique() != 1:
        raise ValueError("the curves of one kind")
    wide = curves.pivot(
        index=list(SESSION_COLUMNS), columns="lag", values="estimate"
    ).reset_index()
    return across_session_band(
        wide[list(lags)].to_numpy(dtype=float),
        wide["animal"].to_numpy(),
        wide["session_date"].astype(str).to_numpy(),
        n_draws=n_draws,
        seed=seed,
    )
