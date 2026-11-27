"""The estimands of the visit rate model, per unit of the fit and over units.

A fit of the model gives one record per unit of the fit (``fit_by``: a
recording day or an animal, or a session for the validation and comparison;
``visit_rate.VisitRateSettings.fit_by``). Each estimand of model A's form is a
linear functional of the fit's coefficients (``visit_rate.declared_contrasts``:
the declared Q1, Q2, Q2-onset and Q7a of the by-outcome form, or the secondary
reward coefficient of the coefficient form), mapped onto each record's
coefficients by name (``terms.Design.contrast_on``, which refuses a contrast
that reads a baseline), so its variance in a unit is exact: ``c' V c`` with
``V`` the fit's Bayesian covariance (``regression.contrasts``). A unit without
a kept fit, or with a coefficient the contrast uses left out, has no value
(NaN).

The support of a window contrast (Q1, Q2, Q2-onset; ``visit_rate.window_support``,
the author's rule) is recorded beside each value: the visits and at-risk
seconds of each outcome in the windows it reads, and ``supported``. A value
whose window lacks support is kept in the unit table, marked, and left out of
the reduction over units, which counts it (``n_units_unsupported``).

Over units, each estimand is summarized for each kind and for the difference
between the kinds (the SWR unit value less the No SWR one, unit by unit, over
the units where both kinds have a supported value) by
``regression.units.unit_summary`` at the records' ``fit_by`` (design note
``docs/design/2026-11-24-unit-of-fit.md``, section 7). Values of records of
different units are refused. The estimates are on the log scale (a log rate
ratio).

The difference curve of the by-outcome form (the log rate ratio after a reward
against after an omission over the time since arrival) is shown, not tested:
per unit with its simultaneous band (``bands.simultaneous_band``), and across
units with its band (``bands.across_session_band``, one row per unit). So is
model D's curve over lags 1 to 10.
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
from continuum_swr_replay.analysis.regression.units import unit_summary
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
    FitBy,
    fit_by_columns,
)

from ..declared_tests import DIFFERENCE, KINDS
from .visit_rate import (
    LAGS,
    WINDOW_SUPPORT,
    VisitRateFit,
    declared_contrasts,
    difference_curve_rows,
    difference_grid,
    lag_curve_rows,
)

__all__ = [
    "difference_curve_band",
    "difference_curves",
    "lag_curve_band",
    "lag_curves",
    "summarize_estimands",
    "unit_estimands",
]

#: The key columns of every unit table: the record's key (``""`` where the
#: unit has no such part, ``FitRecord``) and the unit of the fit.
_KEY = (*SESSION_COLUMNS, "fit_by")


def _records(fit: VisitRateFit) -> list[FitRecord]:
    """The fit's records, each of the fit's unit.

    Raises
    ------
    ValueError
        If a record is of another unit than ``fit.settings.fit_by``.
    """
    other = sorted({r.fit_by for r in fit.records} - {fit.settings.fit_by})
    if other:
        raise ValueError(
            f"records with fit_by {other} in a fit by {fit.settings.fit_by!r}"
        )
    return list(fit.records)


def _key(record: FitRecord) -> dict[str, str]:
    return {
        "animal": record.animal,
        "session_date": record.session_date,
        "epoch": record.epoch,
        "fit_by": record.fit_by,
    }


def _kept(record: FitRecord) -> bool:
    """A record whose fit is kept (converged, no ``no_fit`` cause)."""
    return record.status == CONVERGED and not record.no_fit


def _support_columns(record: FitRecord, estimand_id: str) -> dict:
    """The window support beside one value: ``supported`` and, for each
    outcome the contrast reads, its fewest visits and at-risk seconds over
    the windows it reads (NaN where it reads none)."""
    entry = record.cohort_counts.get(WINDOW_SUPPORT, {}).get(estimand_id)
    out = {"supported": True if entry is None else bool(entry["supported"])}
    for level, name in ((0, "omitted"), (1, "rewarded")):
        reads = (
            []
            if entry is None
            else [r for r in entry["reads"] if r["outcome"] == level]
        )
        out[f"visits_{name}"] = min((r["visits"] for r in reads), default=np.nan)
        out[f"at_risk_s_{name}"] = min((r["at_risk_s"] for r in reads), default=np.nan)
    return out


def unit_estimands(fit: VisitRateFit) -> pd.DataFrame:
    """Each estimand of model A's form in each unit of the fit, with its
    variance and, for a window contrast, its support.

    Parameters
    ----------
    fit : VisitRateFit
        Model A's fit of one kind (``visit_rate.fit_visit_rate``).

    Returns
    -------
    pd.DataFrame
        One row per unit and estimand: ``animal``, ``session_date``,
        ``epoch`` (``""`` for a part the unit lacks), ``fit_by``, ``kind``,
        ``estimand_id``, ``estimate``, ``variance``, ``se``; ``supported``
        (False for a window contrast without support; True otherwise) and
        ``visits_omitted``, ``visits_rewarded``, ``at_risk_s_omitted``,
        ``at_risk_s_rewarded`` (``visit_rate.window_support``; NaN where not
        read). NaN for a unit without a kept fit or with a coefficient the
        contrast uses left out.
    """
    if fit.model != "A":
        raise ValueError(f"the declared estimands are model A's, not {fit.model}'s")
    c = declared_contrasts(fit.design, fit.settings)
    rows = []
    for record in _records(fit):
        beta = np.asarray(record.coef, dtype=float)
        vcov = np.asarray(record.vcov, dtype=float)
        for estimand_id, contrast in c.items():
            value = None
            if _kept(record):
                on_fit = fit.design.contrast_on(contrast, record.coef_names)
                value = evaluate_contrast(on_fit, beta, vcov)
            estimate = np.nan if value is None else value.estimate
            variance = np.nan if value is None else value.variance
            rows.append(
                {
                    **_key(record),
                    "kind": fit.kind,
                    "estimand_id": estimand_id,
                    "estimate": estimate,
                    "variance": variance,
                    "se": float(np.sqrt(variance)),
                    **_support_columns(record, estimand_id),
                }
            )
    return pd.DataFrame(rows)


def _fit_by(values: pd.DataFrame) -> FitBy:
    """The one unit of a table's values.

    Raises
    ------
    ValueError
        If the values are of more than one unit of the fit.
    """
    units = sorted(set(values["fit_by"]))
    if len(units) != 1:
        raise ValueError(f"values of fit_by {units}: one unit of the fit at a time")
    return units[0]


def _indexed(values: pd.DataFrame, fit_by: FitBy, column: str) -> pd.Series:
    """One value per unit, indexed by the unit's key (``unit_summary``)."""
    columns = list(fit_by_columns(fit_by))
    index = (
        pd.Index(values[columns[0]].to_numpy(), name=columns[0])
        if len(columns) == 1
        else pd.MultiIndex.from_frame(values[columns])
    )
    return pd.Series(values[column].to_numpy(dtype=float), index=index)


def _summary(
    values: pd.DataFrame, fit_by: FitBy, n_draws: int, seed: int
) -> dict[str, float | str]:
    """``unit_summary`` of one estimand's supported values (one per unit),
    with the units whose value lacks support counted beside it. With no
    supported finite value the summary is NaN, with no unit."""
    supported = values["supported"].astype(bool).to_numpy()
    kept = _indexed(values[supported], fit_by, "estimate")
    if np.isfinite(kept.to_numpy()).any():
        summary = unit_summary(kept, fit_by, n_draws=n_draws, seed=seed)
    else:
        summary = dict.fromkeys(_SUMMARY_NUMBERS, np.nan) | {
            "n_animals": 0,
            "n_animals_same_sign": 0,
            "n_draws": n_draws,
            "seed": seed,
            "fit_by": fit_by,
            "n_units": 0,
        }
    return {**summary, "n_units_unsupported": int((~supported).sum())}


#: The entries of a summary that are NaN when no unit has a value.
_SUMMARY_NUMBERS = ("estimate", "low", "high", "sd", "n_days", "n_sessions", "day_p")


def summarize_estimands(
    unit_values: pd.DataFrame, *, n_draws: int, seed: int
) -> pd.DataFrame:
    """Each estimand over the units, for each kind and for their difference.

    Parameters
    ----------
    unit_values : pd.DataFrame
        :func:`unit_estimands` of both kinds, concatenated, of one unit of the
        fit.
    n_draws, seed : int
        The bootstrap's (``units.unit_summary``), the same for every row.

    Returns
    -------
    pd.DataFrame
        One row per estimand and tested (``SWR``, ``No SWR``, ``SWR minus No
        SWR``): ``estimand_id``, ``tested``, the entries of
        ``unit_summary`` (on the log scale; with ``fit_by`` and ``n_units``)
        and ``n_units_unsupported`` (the units whose value lacks window
        support, left out). The difference is taken unit by unit, over the
        units where both kinds have a supported value.

    Raises
    ------
    ValueError
        If a kind is missing, the values are of more than one unit of the fit,
        or a unit has more than one value of an estimand and kind.
    """
    missing = set(KINDS) - set(unit_values["kind"])
    if missing:
        raise ValueError(f"no unit values of {sorted(missing)}")
    fit_by = _fit_by(unit_values)
    keys = [*_KEY, "estimand_id", "kind"]
    if unit_values.duplicated(keys).any():
        raise ValueError("a unit has more than one value of an estimand and kind")
    out = []
    for estimand_id, values in unit_values.groupby("estimand_id", sort=False):
        for kind in KINDS:
            summary = _summary(values[values["kind"] == kind], fit_by, n_draws, seed)
            out.append({"estimand_id": estimand_id, "tested": kind, **summary})
        values = values.assign(
            estimate=values["estimate"].where(values["supported"].astype(bool))
        )
        wide = values.pivot(index=list(_KEY), columns="kind", values="estimate")
        wide = wide.reset_index()
        wide["estimate"] = wide[KINDS[0]] - wide[KINDS[1]]
        wide["supported"] = True
        summary = _summary(wide, fit_by, n_draws, seed)
        out.append({"estimand_id": estimand_id, "tested": DIFFERENCE, **summary})
    return pd.DataFrame(out)


def _curve_rows(
    fit: VisitRateFit, C: np.ndarray, grid: np.ndarray, name: str, seed: int
) -> pd.DataFrame:
    """One row per unit and grid point of the curve ``C @ beta`` with its
    simultaneous band; NaN for a unit without a kept fit or with the curve
    left out."""
    rows = []
    for record in _records(fit):
        C_fit = np.stack([fit.design.contrast_on(c, record.coef_names) for c in C])
        beta = np.asarray(record.coef, dtype=float)
        used = np.any(C_fit != 0.0, axis=0)
        if _kept(record) and np.isfinite(beta[used]).all():
            band = simultaneous_band(C_fit, beta, np.asarray(record.vcov), seed=seed)
            values = (band.estimate, band.se, band.low, band.high, band.critical)
        else:
            values = (*(np.full(grid.size, np.nan),) * 4, np.nan)
        estimate, se, low, high, critical = values
        for j, point in enumerate(grid):
            rows.append(
                {
                    **_key(record),
                    "kind": fit.kind,
                    name: point,
                    "estimate": estimate[j],
                    "se": se[j],
                    "low": low[j],
                    "high": high[j],
                    "critical": critical,
                }
            )
    return pd.DataFrame(rows)


def lag_curves(fit: VisitRateFit, *, seed: int) -> pd.DataFrame:
    """Model D's curve over lags in each unit of the fit, with its
    simultaneous band.

    Parameters
    ----------
    fit : VisitRateFit
        Model D's fit of one kind.
    seed : int
        The band's seed (``bands.simultaneous_band``), the same in every
        unit.

    Returns
    -------
    pd.DataFrame
        One row per unit and lag: the unit's key, ``fit_by``, ``kind``,
        ``lag``, ``estimate``, ``se``, ``low``, ``high`` and ``critical``. A
        unit without a kept fit, or with the curve left out, has NaN.
    """
    if fit.model != "D":
        raise ValueError(f"the lag curve is model D's, not {fit.model}'s")
    grid = np.asarray(LAGS)
    return _curve_rows(fit, lag_curve_rows(fit.design), grid, "lag", seed)


def difference_curves(fit: VisitRateFit, *, seed: int) -> pd.DataFrame:
    """The by-outcome form's difference curve in each unit of the fit, with
    its simultaneous band: the outcome coefficient plus the arrival curve
    after a reward less the one after an omission, the log rate ratio after a
    reward at each time since arrival (``visit_rate.difference_curve_rows``,
    on ``visit_rate.difference_grid``). Shown, not tested.

    Returns
    -------
    pd.DataFrame
        One row per unit and grid point: the unit's key, ``fit_by``,
        ``kind``, ``time_since_arrival_s``, ``estimate``, ``se``, ``low``,
        ``high`` and ``critical``; NaN for a unit without a kept fit or with
        a coefficient the curve uses left out.
    """
    if fit.model != "A" or fit.settings.reward_form != "by_outcome":
        raise ValueError("the difference curve is model A's by-outcome form's")
    grid = difference_grid(fit.settings)
    C = difference_curve_rows(fit.design, grid)
    return _curve_rows(fit, C, grid, "time_since_arrival_s", seed)


def _band(
    curves: pd.DataFrame, point: str, points: Sequence, n_draws: int, seed: int
) -> Band:
    """The across-unit band of one kind's curves: one row per unit, with the
    animal and the day (the recording day for a session or a day; the animal
    itself for an animal) as ``across_session_band`` reads them."""
    if curves["kind"].nunique() != 1:
        raise ValueError("the curves of one kind")
    fit_by = _fit_by(curves)
    wide = curves.pivot(index=list(_KEY), columns=point, values="estimate")
    wide = wide.reset_index()
    day = wide["animal"] if fit_by == "animal" else wide["session_date"]
    return across_session_band(
        wide[list(points)].to_numpy(dtype=float),
        wide["animal"].to_numpy(),
        day.astype(str).to_numpy(),
        n_draws=n_draws,
        seed=seed,
    )


def lag_curve_band(
    curves: pd.DataFrame, *, n_draws: int, seed: int, lags: Sequence[int] = LAGS
) -> Band:
    """The lag curve across units, with its band.

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
    return _band(curves, "lag", lags, n_draws, seed)


def difference_curve_band(curves: pd.DataFrame, *, n_draws: int, seed: int) -> Band:
    """The difference curve across units, with its band
    (``bands.across_session_band``, one row per unit), on the grid of
    :func:`difference_curves`.

    Parameters
    ----------
    curves : pd.DataFrame
        :func:`difference_curves` of one kind.
    n_draws, seed : int
        As ``bands.across_session_band`` takes them.
    """
    grid = sorted(curves["time_since_arrival_s"].unique())
    return _band(curves, "time_since_arrival_s", grid, n_draws, seed)
