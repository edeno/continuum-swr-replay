"""The declared content estimands of the role models, per session and summarized.

:func:`session_estimands` reads each declared estimand of a fit
(:func:`.role_content.fit_role_content`) in every session, with its exact
variance ``c' V c`` (``regression.contrasts.evaluate_contrast``, ``V`` the
fit's Bayesian covariance), and says whether the session's value is kept: a
converged fit, kept (no ``no_fit`` cause), with every coefficient the contrast
reads fitted. :func:`estimand_summaries` reduces them with
``shared_statistics.estimand_summary`` for each kind and for the difference
of the two kinds' session values (SWR minus No SWR, a session with both).

A record's ``variance`` and ``se`` are on the model's scale: the fit's
Bayesian covariance, with each event one categorical draw (or its weight,
``role_content.EVENT_WEIGHTS``). They are not sampling errors of the estimate
across data sets (under the share response they can be several times its
spread), and no interval is built from them: an estimand's interval is the
project's, over sessions, recording days and animals
(``shared_statistics.estimand_summary``).

:func:`penalized_edf` reports how much the weak penalty decides: each
penalized simple coefficient's effective degrees of freedom in each session,
beside the session's number of events, and :func:`penalized_edf_summary`
summarizes it by kind.

The model-free references (:func:`role_shares`, :func:`within_arm_slopes`,
:func:`current_share_by_time_until_leaving`) describe the cohort rows without a
model: what the model's estimates are compared with.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.contrasts import evaluate_contrast
from continuum_swr_replay.analysis.regression.newton import CONVERGED
from continuum_swr_replay.analysis.shared_statistics import (
    animal_day_session_values,
    estimand_summary,
)

from .. import declared_tests as dt
from . import role_content as rc

__all__ = [
    "LEAVING_BINS_S",
    "current_share_by_time_until_leaving",
    "estimand_summaries",
    "penalized_edf",
    "penalized_edf_summary",
    "role_shares",
    "session_estimands",
    "within_arm_slopes",
]

_KEYS = list(rc.SESSION_COLUMNS)


def session_estimands(fit: rc.RoleContentFit) -> pd.DataFrame:
    """Each declared estimand of the fit in each session.

    Returns
    -------
    pd.DataFrame
        One row per (session, estimand): ``animal``, ``session_date``,
        ``epoch``, ``model``, ``variant``, ``kind``, ``estimand_id``,
        ``estimate``, ``variance`` and ``se`` (on the model's scale, not
        sampling errors: module docstring), ``status``, ``no_fit``, ``kept``
        (the value may enter a summary) and ``left_out`` (the coefficients the
        contrast reads that the session left out, comma-separated).
    """
    rows = []
    names = fit.design.coef_names
    for record in fit.records:
        beta = np.asarray(record.coef, dtype=float)
        vcov = np.asarray(record.vcov, dtype=float)
        fitted = record.status == CONVERGED and not record.no_fit
        for eid, c in fit.contrasts.items():
            reads = [names[j] for j in np.flatnonzero(c)]
            missing = [n for n in reads if n in record.left_out]
            value = evaluate_contrast(c, beta, vcov)
            kept = fitted and not missing and np.isfinite(value.estimate)
            rows.append(
                {
                    "animal": record.animal,
                    "session_date": record.session_date,
                    "epoch": record.epoch,
                    "model": fit.model,
                    "variant": fit.variant,
                    "kind": fit.kind,
                    "estimand_id": eid,
                    "estimate": value.estimate,
                    "variance": value.variance,
                    "se": value.se,
                    "status": record.status,
                    "no_fit": record.no_fit,
                    "kept": bool(kept),
                    "left_out": ",".join(missing),
                }
            )
    return pd.DataFrame(rows)


def penalized_edf(fit: rc.RoleContentFit) -> pd.DataFrame:
    """Each penalized simple coefficient's edf in each session of a fit.

    A penalized simple coefficient is a role level or a role slope with a
    fixed penalty. Its edf (the record's ``edf`` by coefficient name) is near 1
    when the data decide its value and lower as the penalty shrinks it; for
    one coefficient alone it is the shrinkage factor of its estimate, on the
    model's scale. The trunk level is left out: it is a function of the arm,
    fitted at 0 by construction, so its edf of 0 says nothing of the data.

    Returns
    -------
    pd.DataFrame
        One row per (session, coefficient): ``animal``, ``session_date``,
        ``epoch``, ``model``, ``variant``, ``kind``, ``prior_sd`` (the
        fit's), ``coefficient``,
        ``tested`` (a declared estimand reads it), ``edf`` (NaN where the
        session left it out or has no kept fit), ``n_events`` (the events in
        the session's fit).
    """
    design = fit.design
    tested = set(rc.tested_coefficients(design, fit.contrasts))
    names = [
        t.name
        for t in design.terms
        if t.kind in ("level", "slope", "binary")
        and t.block is not None
        and t.name != rc.TRUNK_LEVEL
    ]
    rows = []
    for record in fit.records:
        fitted = record.status == CONVERGED and not record.no_fit
        for name in names:
            edf = record.edf.get(name, np.nan) if fitted else np.nan
            if name in record.left_out:
                edf = np.nan
            rows.append(
                {
                    "animal": record.animal,
                    "session_date": record.session_date,
                    "epoch": record.epoch,
                    "model": fit.model,
                    "variant": fit.variant,
                    "kind": fit.kind,
                    "prior_sd": fit.settings.prior_sd,
                    "coefficient": name,
                    "tested": name in tested,
                    "edf": float(edf),
                    "n_events": int(record.n_units),
                }
            )
    return pd.DataFrame(rows)


def penalized_edf_summary(per_session: pd.DataFrame) -> pd.DataFrame:
    """The edf of each penalized coefficient over sessions, by kind.

    Parameters
    ----------
    per_session : pd.DataFrame
        :func:`penalized_edf` of any fits.

    Returns
    -------
    pd.DataFrame
        One row per (model, variant, kind, prior SD, coefficient): ``tested``,
        ``n_sessions`` (with an edf), the ``edf`` quantiles ``edf_min``,
        ``edf_q10``, ``edf_median``, ``edf_max``, and the median, smallest
        and largest ``n_events`` of those sessions.
    """
    kept = per_session.loc[np.isfinite(per_session["edf"])]
    grouped = kept.groupby(
        ["model", "variant", "kind", "prior_sd", "coefficient"], sort=False
    )
    return grouped.agg(
        tested=("tested", "first"),
        n_sessions=("edf", "size"),
        edf_min=("edf", "min"),
        edf_q10=("edf", lambda e: float(np.quantile(e, 0.1))),
        edf_median=("edf", "median"),
        edf_max=("edf", "max"),
        n_events_median=("n_events", "median"),
        n_events_min=("n_events", "min"),
        n_events_max=("n_events", "max"),
    ).reset_index()


def estimand_summaries(
    per_session: pd.DataFrame, *, n_draws: int, seed: int
) -> pd.DataFrame:
    """Each estimand summarized for each kind and for the kind difference.

    Parameters
    ----------
    per_session : pd.DataFrame
        :func:`session_estimands` of the fits of both kinds (any models and
        variants); only the kept values are read.
    n_draws, seed : int
        The nested bootstrap's (``estimand_summary``).

    Returns
    -------
    pd.DataFrame
        One row per (model, variant, estimand, tested), ``tested`` one of
        ``declared_tests.TESTED``: ``estimand_summary``'s entries, and
        ``n_sessions_left_out`` (sessions with a value that was not kept, for
        a kind; for the difference, sessions with a kept value of one kind
        only). A tested quantity with no kept session has NaN entries.
    """
    kept = per_session.loc[per_session["kept"]]
    out = []
    for (model, variant, eid), group in per_session.groupby(
        ["model", "variant", "estimand_id"], sort=False
    ):
        values = {}
        for kind in dt.KINDS:
            of_kind = kept[
                (kept["model"] == model)
                & (kept["variant"] == variant)
                & (kept["estimand_id"] == eid)
                & (kept["kind"] == kind)
            ]
            values[kind] = of_kind.set_index(_KEYS)["estimate"]
        both = values[dt.KINDS[0]].index.intersection(values[dt.KINDS[1]].index)
        values[dt.DIFFERENCE] = (
            values[dt.KINDS[0]].loc[both] - values[dt.KINDS[1]].loc[both]
        )
        for tested in dt.TESTED:
            series = values[tested]
            if tested == dt.DIFFERENCE:
                left_out = len(
                    values[dt.KINDS[0]].index.symmetric_difference(
                        values[dt.KINDS[1]].index
                    )
                )
            else:
                left_out = int(((group["kind"] == tested) & ~group["kept"]).sum())
            entry = {
                "model": model,
                "variant": variant,
                "estimand_id": eid,
                "tested": tested,
                "n_sessions_left_out": left_out,
            }
            if len(series):
                index = series.index.to_frame(index=False)
                labels = [pd.Series(index[c].to_numpy()) for c in _KEYS]
                entry |= estimand_summary(
                    pd.Series(series.to_numpy()), *labels, n_draws=n_draws, seed=seed
                )
            out.append(entry)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Model-free references
# ---------------------------------------------------------------------------

#: The bins of time until the final poke-out (s) of the model-free current-arm
#: share.
LEAVING_BINS_S: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, np.inf)


def _animal_values(frame: pd.DataFrame, value: str, by: Sequence[str]) -> pd.DataFrame:
    """Each animal's value of a per-session column, session -> day -> animal,
    within each group of ``by``."""
    out = []
    for group, part in frame.groupby(list(by), sort=True):
        per_animal = animal_day_session_values(
            part[value], part["animal"], part["session_date"], part["epoch"]
        )
        group = group if isinstance(group, tuple) else (group,)
        for animal, v in per_animal.items():
            out.append(dict(zip(by, group, strict=True)) | {"animal": animal, value: v})
    return pd.DataFrame(out)


def role_shares(rows: pd.DataFrame) -> pd.DataFrame:
    """Per kind, role and animal: the mean share of an event on the role's
    arms (``total``) and per arm of the role (``per_arm``).

    Each event's share on a role is the sum over its rows of the role; an
    event without a row of the role (no previous arm) has none. The mean is
    over a session's events, then session -> day -> animal.

    Returns
    -------
    pd.DataFrame
        ``kind``, ``role``, ``animal``, ``total``, ``per_arm``.
    """
    per_event = (
        rows.groupby([*_KEYS, rc.KIND, rc.UNIT, rc.ROLE], sort=False)[rc.RESPONSE]
        .agg(["sum", "size"])
        .reset_index()
    )
    per_event["per_arm"] = per_event["sum"] / per_event["size"]
    per_session = (
        per_event.groupby([*_KEYS, rc.KIND, rc.ROLE])[["sum", "per_arm"]]
        .mean()
        .reset_index()
        .rename(columns={"sum": "total"})
    )
    total = _animal_values(per_session, "total", [rc.KIND, rc.ROLE])
    per_arm = _animal_values(per_session, "per_arm", [rc.KIND, rc.ROLE])
    return total.merge(per_arm, on=[rc.KIND, rc.ROLE, "animal"])


def within_arm_slopes(
    rows: pd.DataFrame,
    covariates: Iterable[str] = (rc.REWARD_FRACTION, rc.VALUE),
    *,
    min_rows: int = 10,
) -> pd.DataFrame:
    """Per kind, leaf role, covariate and animal: the within-session,
    within-arm least-squares slope of share on the covariate (per pooled SD).

    In each (session, kind, role, arm) with at least ``min_rows`` rows and a
    covariate that varies, the slope of ``arm_fraction`` on the covariate
    divided by its pooled SD (population, over the leaf rows of the kind); a
    session's value is the mean over its arms; then session -> day -> animal.

    Returns
    -------
    pd.DataFrame
        ``kind``, ``role``, ``covariate``, ``animal``, ``slope``.
    """
    out = []
    leaf = rows.loc[rows[rc.ROLE].isin(rc.LEAF_ROLES)]
    for covariate in covariates:
        sd = leaf.groupby(rc.KIND)[covariate].std(ddof=0)
        x = leaf[covariate] / leaf[rc.KIND].map(sd)
        frame = leaf.assign(_x=x.to_numpy())
        slopes = []
        for key, part in frame.groupby([*_KEYS, rc.KIND, rc.ROLE, rc.ARM], sort=False):
            xv, yv = part["_x"].to_numpy(), part[rc.RESPONSE].to_numpy()
            if len(part) < min_rows or np.ptp(xv) == 0:
                continue
            xc = xv - xv.mean()
            slopes.append((*key, float(xc @ (yv - yv.mean()) / (xc @ xc))))
        per_arm = pd.DataFrame(
            slopes, columns=[*_KEYS, rc.KIND, rc.ROLE, rc.ARM, "slope"]
        )
        per_session = (
            per_arm.groupby([*_KEYS, rc.KIND, rc.ROLE])["slope"].mean().reset_index()
        )
        animal = _animal_values(per_session, "slope", [rc.KIND, rc.ROLE])
        animal.insert(2, "covariate", covariate)
        out.append(animal)
    return pd.concat(out, ignore_index=True)


def current_share_by_time_until_leaving(
    rows: pd.DataFrame, bins: Sequence[float] = LEAVING_BINS_S
) -> pd.DataFrame:
    """Per kind, bin of time until the final poke-out and animal: the mean
    share on the current arm (a session's mean over its events in the bin,
    then session -> day -> animal).

    Returns
    -------
    pd.DataFrame
        ``kind``, ``bin`` (an interval label), ``animal``, ``share``.
    """
    current = rows.loc[rows[rc.ROLE] == "current"].copy()
    current["bin"] = pd.cut(current[rc.LEAVING], list(bins), right=False).astype(str)
    per_session = (
        current.groupby([*_KEYS, rc.KIND, "bin"])[rc.RESPONSE]
        .mean()
        .reset_index()
        .rename(columns={rc.RESPONSE: "share"})
    )
    return _animal_values(per_session, "share", [rc.KIND, "bin"])
