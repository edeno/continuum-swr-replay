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

At ``reward_level="animal"`` (``visit_rate.fit_animal_rewards``) Q2 is one
coefficient per animal and kind, and the cohort's value is the mean of the
animals' coefficients with the t interval over the animals
(:func:`summarize_animal_rewards`); the difference between the kinds is taken
animal by animal. Taking the animals' mean with a t interval as the
animal-level value is the plan author's call (2026-11-23 brief), not a
decision of the lead.

Model D's curve over lags 1 to 10 is described, not tested: per session with
its simultaneous band (``bands.simultaneous_band``), and the animal-balanced
curve across sessions with its band (``bands.across_session_band``).

Q7a from a fit with the third clock (``f_m``) is the drop before leaving
conditional on the time until the end of the immobile stretch: in a visit's
last stretch the two clocks are equal, so part of a drop before leaving can
move to ``f_m``. :func:`third_clock_q7a` sets Q7a of the fits with and
without ``f_m`` side by side.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

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
from continuum_swr_replay.analysis.shared_statistics import (
    animal_mean_interval,
    estimand_summary,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ..declared_tests import DIFFERENCE, KINDS
from .visit_rate import (
    LAGS,
    STRETCH,
    AnimalRewardFit,
    VisitRateFit,
    declared_contrasts,
    lag_curve_rows,
)

__all__ = [
    "animal_reward_values",
    "summarize_animal_rewards",
    "lag_curve_band",
    "lag_curves",
    "session_estimands",
    "summarize_estimands",
    "third_clock_q7a",
    "ThirdClockQ7a",
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
        the contrast uses left out. With the third clock in the model, Q7a is
        conditional on it (module docstring).
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


def animal_reward_values(fit: AnimalRewardFit) -> pd.DataFrame:
    """Each animal's reward coefficient of one kind, one row per animal.

    Returns
    -------
    pd.DataFrame
        ``animal``, ``kind``, ``coefficient``, ``se``, ``status``,
        ``n_iter``, ``n_sessions`` and ``n_events``; a coefficient that did
        not converge is NaN.
    """
    return pd.DataFrame(
        [
            {
                "animal": a.animal,
                "kind": a.kind,
                "coefficient": a.coefficient if a.status == CONVERGED else np.nan,
                "se": a.se if a.status == CONVERGED else np.nan,
                "status": a.status,
                "n_iter": a.n_iter,
                "n_sessions": a.n_sessions,
                "n_events": a.n_events,
            }
            for a in fit.animals
        ]
    )


def summarize_animal_rewards(values: pd.DataFrame) -> pd.DataFrame:
    """Q2 at the animal level: the mean of the animals' coefficients with the
    t interval over the animals, for each kind and for their difference.

    The plan author's call for an animal-level value (module docstring): the
    estimate is the mean over the animals with a value, its interval
    ``mean +/- t(0.975, A - 1) SD / sqrt(A)`` over them
    (``shared_statistics.animal_mean_interval``). The difference SWR minus No
    SWR is taken animal by animal, over the animals with both.

    Which animals have a value reads the outcome: an animal with no event on
    one side in all its sessions has no finite coefficient
    (``visit_rate.fit_animal_rewards``), so with a large true effect the most
    extreme animals are the ones left out and the mean can be biased toward
    zero. It did not occur in any cell of the validation runs.

    Parameters
    ----------
    values : pd.DataFrame
        One row per animal and kind: ``animal``, ``kind`` and
        ``coefficient`` (:func:`animal_reward_values` of both kinds).

    Returns
    -------
    pd.DataFrame
        One row per tested (``SWR``, ``No SWR``, ``SWR minus No SWR``):
        ``tested`` and the entries of ``animal_mean_interval``.

    Raises
    ------
    ValueError
        If a kind is missing or an animal has two values of a kind.
    """
    missing = set(KINDS) - set(values["kind"])
    if missing:
        raise ValueError(f"no animal values of {sorted(missing)}")
    if values.duplicated(["animal", "kind"]).any():
        raise ValueError("an animal has more than one value of a kind")
    wide = values.pivot(index="animal", columns="kind", values="coefficient")
    out = [
        {"tested": kind, **animal_mean_interval(wide[kind].dropna())} for kind in KINDS
    ]
    difference = wide[KINDS[0]] - wide[KINDS[1]]
    out.append({"tested": DIFFERENCE, **animal_mean_interval(difference.dropna())})
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


class ThirdClockQ7a(NamedTuple):
    """Q7a of model A with and without the third clock (:func:`third_clock_q7a`).

    Attributes
    ----------
    sessions : pd.DataFrame
        One row per session: the session, ``kind``, ``with_stretch``,
        ``without_stretch`` (Q7a of each fit) and ``difference`` (with less
        without); NaN where either fit has no value.
    summary : pd.DataFrame
        One row per ``value`` (``with_stretch``, ``without_stretch``,
        ``difference``): ``estimand_summary`` of the session values.
    """

    sessions: pd.DataFrame
    summary: pd.DataFrame


def third_clock_q7a(
    with_stretch: VisitRateFit,
    without_stretch: VisitRateFit,
    *,
    n_draws: int,
    seed: int,
) -> ThirdClockQ7a:
    """Q7a of model A's fits with and without ``f_m``, per session and over
    sessions, stored beside the third clock's rule.

    The rule (``visit_rate.third_clock_decision``) reads the concurvity only;
    the difference here shows how much of the drop before leaving moves to
    ``f_m`` when it is in the model.

    Parameters
    ----------
    with_stretch, without_stretch : VisitRateFit
        Model A's fits of one kind, with and without the stretch clock, on
        the same sessions.
    n_draws, seed : int
        The nested bootstrap's (``estimand_summary``).

    Returns
    -------
    ThirdClockQ7a

    Raises
    ------
    ValueError
        If a fit is not model A's, the fits differ in kind or sessions, or
        the stretch clock is not in the first fit alone.
    """
    if with_stretch.model != "A" or without_stretch.model != "A":
        raise ValueError("Q7a is model A's")
    if with_stretch.kind != without_stretch.kind:
        raise ValueError("the two fits are of different kinds")
    if not (
        with_stretch.settings.include_stretch_clock
        and not without_stretch.settings.include_stretch_clock
    ):
        raise ValueError(f"the first fit holds {STRETCH}, the second does not")
    keys = list(SESSION_COLUMNS)
    values = {}
    for name, fit in (
        ("with_stretch", with_stretch),
        ("without_stretch", without_stretch),
    ):
        q7a = session_estimands(fit)
        values[name] = q7a[q7a["estimand_id"] == "Q7a"].set_index(keys)["estimate"]
    if not values["with_stretch"].index.equals(values["without_stretch"].index):
        raise ValueError("the two fits are of different sessions")
    sessions = pd.DataFrame(values)
    sessions["difference"] = sessions["with_stretch"] - sessions["without_stretch"]
    sessions = sessions.reset_index()
    sessions.insert(len(keys), "kind", with_stretch.kind)
    summary = [
        {
            "value": name,
            **_summary(sessions.rename(columns={name: "estimate"}), n_draws, seed),
        }
        for name in ("with_stretch", "without_stretch", "difference")
    ]
    return ThirdClockQ7a(sessions, pd.DataFrame(summary))
