"""The reduction of one estimate per unit of a fit, and the values of a fit's records.

A fit of a unit (:func:`.sessions.fit_sessions`, ``fit_by``) gives one record
per unit: a session, a recording day or an animal. :func:`unit_values` puts
one value per record in a series indexed by the unit's key;
:func:`unit_summary` reduces it by the rule of the unit
(``docs/design/2026-11-24-unit-of-fit.md``, section 7), with the same keys at
every unit; :func:`unit_estimate` is its estimate alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.shared_statistics import (
    animal_mean_interval,
    estimand_summary,
)
from continuum_swr_replay.tables.session_keys import FitBy, fit_by_columns

from .record import FitRecord

__all__ = ["unit_estimate", "unit_summary", "unit_values"]


def unit_values(values: np.ndarray, records: Sequence[FitRecord]) -> pd.Series:
    """One value per record, indexed by the record's unit key.

    Parameters
    ----------
    values : np.ndarray, shape (n_records,)
    records : sequence of FitRecord
        One unit each, all of one ``fit_by``.

    Returns
    -------
    pd.Series
        Indexed by a MultiIndex of the unit's key columns
        (``tables.session_keys.FIT_BY_COLUMNS``).
    """
    fit_by = records[0].fit_by
    index = pd.MultiIndex.from_tuples(
        [r.unit_key for r in records], names=fit_by_columns(fit_by)
    )
    return pd.Series(np.asarray(values, dtype=float), index=index)


def unit_estimate(values: pd.Series, fit_by: FitBy) -> float:
    """:func:`unit_summary`'s estimate alone: it does not read the bootstrap's
    draws, so none are drawn. NaN when no unit has a finite value."""
    if not np.isfinite(values.to_numpy(dtype=float)).any():
        return float("nan")
    return float(unit_summary(values, fit_by, n_draws=0, seed=0)["estimate"])


def unit_summary(
    values: pd.Series, fit_by: FitBy, *, n_draws: int, seed: int
) -> dict[str, float | str]:
    """The summary of a scalar estimand over the units of a fit.

    One rule for each unit (``docs/design/2026-11-24-unit-of-fit.md``,
    section 7):

    - ``"session"``: ``shared_statistics.estimand_summary`` (session -> day -> animal, the
      nested bootstrap, the t correction over the animals, the day-level test);
    - ``"day"``: ``shared_statistics.estimand_summary`` with each recording day as its own
      session (day -> animal);
    - ``"animal"``: :func:`animal_mean_interval`, the mean over the animals with
      the t interval on ``A - 1`` degrees of freedom; no day-level test.

    A kind difference (SWR less No SWR) is taken unit by unit, over the units
    with both kinds, before it is given here.

    Parameters
    ----------
    values : pd.Series, shape (n_units,)
        One estimate per unit, indexed by the unit's key: a MultiIndex of
        ``(animal, session_date, epoch)`` for a session, of ``(animal,
        session_date)`` for a day, an Index of the animal for an animal
        (``tables.session_keys.FIT_BY_COLUMNS``). A value that is not finite is
        left out.
    fit_by : {"session", "day", "animal"}
    n_draws, seed : int
        The bootstrap's (not read at ``"animal"``, but returned).

    Returns
    -------
    summary : dict
        The keys of ``shared_statistics.estimand_summary``, and ``fit_by`` and ``n_units``
        (the units with a finite value). At ``"day"`` ``n_sessions`` is NaN
        (a day stands for its sessions). At ``"animal"``: ``n_days``,
        ``n_sessions`` and ``day_p`` are NaN; ``sd`` is
        ``SD / sqrt(A) / sqrt(A / (A - 1))``, so that the interval is
        ``estimate +/- t(0.975, A - 1) * sqrt(A / (A - 1)) * sd`` as at the
        other units (``shared_statistics.bootstrap_interval``). ``n_animals_same_sign``
        follows one rule at every unit (``shared_statistics._n_same_sign``).

    Raises
    ------
    ValueError
        If ``fit_by`` is not a unit, the index does not have the unit's key
        levels, no unit has a finite value, or a unit has more than one value.
    """
    columns = fit_by_columns(fit_by)
    if values.index.nlevels != len(columns):
        raise ValueError(
            f"one value per {fit_by}: the index must have the levels {columns}, "
            f"not {values.index.nlevels}"
        )
    if fit_by == "animal":
        found = animal_mean_interval(values)
        if found["n_animals"] == 0:
            raise ValueError("no animal has a finite value")
        n = found["n_animals"]
        return {
            "estimate": found["estimate"],
            "low": found["low"],
            "high": found["high"],
            "sd": found["sd"],
            "n_animals": n,
            "n_days": float("nan"),
            "n_sessions": float("nan"),
            "n_animals_same_sign": found["n_animals_same_sign"],
            "day_p": float("nan"),
            "n_draws": n_draws,
            "seed": seed,
            "fit_by": fit_by,
            "n_units": n,
        }
    index = values.index
    animal, day = (pd.Series(index.get_level_values(k), index=index) for k in (0, 1))
    session = (
        pd.Series(index.get_level_values(2), index=index)
        if fit_by == "session"
        else day
    )
    summary = estimand_summary(values, animal, day, session, n_draws=n_draws, seed=seed)
    n_units = summary["n_sessions"] if fit_by == "session" else summary["n_days"]
    if fit_by == "day":
        summary["n_sessions"] = float("nan")
    return {**summary, "fit_by": fit_by, "n_units": n_units}
