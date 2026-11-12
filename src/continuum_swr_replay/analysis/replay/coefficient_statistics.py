"""Across-session summaries of what the per-session fits wrote.

Every model ladder is fit one session at a time, so a reported number is a
summary over sessions of a per-session number: a coefficient, a difference of
two coefficients, a difference of two rungs' AIC, a separation flag. The
functions here compute those summaries from a table of per-session fits (one
row per session, with the session columns of
:data:`~continuum_swr_replay.tables.session_keys.SESSION_COLUMNS`) and return
named numbers.

:func:`~continuum_swr_replay.analysis.statistics.session_median_statistics` is
the one summary every other function is built on: the median over sessions
with the sessions' signs, and the median over animals of each animal's own
median with the animals' signs. "Animals positive" counts the animals whose own
median is above zero. A signed-rank p treats the sessions as independent, which
sessions of one animal are not: it describes consistency across sessions and is
not a test on the animals.

Which sessions a coefficient is read over is the per-term gate of
:func:`~continuum_swr_replay.analysis.replay.contrast_aggregation.gate_term_sessions`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    gate_term_sessions,
    pct_significant_sessions,
    separated_for_term,
    session_se,
    unidentified_for_term,
)
from continuum_swr_replay.analysis.statistics import (
    benjamini_hochberg,
    session_median_statistics,
    signed_rank_p,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS


def fit_rows(fits: pd.DataFrame, **labels: str) -> pd.DataFrame:
    """The fits that carry the given labels.

    Parameters
    ----------
    fits : pd.DataFrame, shape (n_fits, n_columns)
        Per-session fits of several rungs or populations.
    **labels : str
        A label column and the value to keep, for example
        ``model="M0_const", outcome="swr"``.

    Returns
    -------
    rows : pd.DataFrame
        The rows of ``fits`` whose every named column equals its label, indexed
        as ``fits``.
    """
    keep = pd.Series(True, index=fits.index)
    for column, label in labels.items():
        keep &= fits[column] == label
    return fits[keep]


def require_one_fit_per_session(fits: pd.DataFrame) -> None:
    """Refuse fits that hold two rows for one session.

    A summary over sessions reads one fit per session. Fits of several rungs
    or populations passed where one is read would count a session once per
    rung or population.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits with the session columns.

    Raises
    ------
    ValueError
        If two rows of ``fits`` are of the same session.
    """
    if fits.duplicated(list(SESSION_COLUMNS)).any():
        raise ValueError(
            "more than one fit for a session: pass the fits of one rung and population"
        )


def _paired_difference(
    fits_a: pd.DataFrame,
    fits_b: pd.DataFrame,
    column: str,
    same: Sequence[str] = (),
) -> tuple[pd.Series, pd.Series]:
    """One column of two sets of fits, differenced within session.

    Parameters
    ----------
    fits_a, fits_b : pd.DataFrame
        Per-session fits with the session columns, ``column`` and the ``same``
        columns; at most one row per session.
    column : str
        The column differenced.
    same : sequence of str
        Columns that must be equal in a session's two fits for it to count.

    Returns
    -------
    difference : pd.Series, shape (n_paired_sessions,)
        ``column`` of ``fits_a`` minus that of ``fits_b``, over the sessions in
        both (missing where either is missing).
    animal : pd.Series, shape (n_paired_sessions,)
        Each of those sessions' animal, indexed as ``difference``.

    Raises
    ------
    pandas.errors.MergeError
        If either set of fits holds two rows for one session.
    """
    session = list(SESSION_COLUMNS)
    columns = [*session, column, *same]
    both = fits_a[columns].merge(
        fits_b[columns], on=session, suffixes=("_a", "_b"), validate="1:1"
    )
    for name in same:
        both = both[both[f"{name}_a"] == both[f"{name}_b"]]
    return both[f"{column}_a"] - both[f"{column}_b"], both["animal"]


def coefficient_statistics(fits: pd.DataFrame, column: str) -> dict[str, float]:
    """One fitted coefficient across the sessions that have it.

    Parameters
    ----------
    fits : pd.DataFrame, shape (n_sessions, n_columns)
        One row per session, with the session columns and ``column``. A
        session whose ``column`` is missing is left out.
    column : str
        The coefficient's column.

    Returns
    -------
    statistics : dict
        The names of
        :func:`~continuum_swr_replay.analysis.statistics.session_median_statistics`,
        and:

        - ``exp_median``: ``exp`` of the median over sessions (a rate ratio for
          a log-rate coefficient, an odds ratio for a log-odds one).
        - ``exp_animal_median``: ``exp`` of the median over animals.
        - ``signed_rank_p``: the Wilcoxon signed-rank p of the sessions'
          coefficients against zero
          (:func:`~continuum_swr_replay.analysis.statistics.signed_rank_p`: a
          coefficient of exactly zero is left out of the test, and the p is NaN
          below six sessions with a non-zero coefficient).
    """
    require_one_fit_per_session(fits)
    known = fits.dropna(subset=[column])
    values = known[column]
    statistics = session_median_statistics(values, known["animal"])
    return {
        **statistics,
        "exp_median": float(np.exp(statistics["median"])),
        "exp_animal_median": float(np.exp(statistics["animal_median"])),
        "signed_rank_p": signed_rank_p(values),
    }


def coefficient_family_statistics(
    fits: pd.DataFrame, terms: Sequence[str]
) -> pd.DataFrame:
    """Several terms' coefficients across sessions, their p-values corrected
    together.

    No session is left out for separation: this is the ungated summary of one
    rung's fits (:func:`gated_coefficient_statistics` is the gated one).

    Parameters
    ----------
    fits : pd.DataFrame, shape (n_sessions, n_columns)
        One row per session, with the session columns, ``coef_<term>`` and
        optionally ``bse_<term>`` and ``unidentified_terms``.
    terms : sequence of str
        The terms. A term without a ``coef_<term>`` column has no row.

    Returns
    -------
    statistics : pd.DataFrame, shape (n_terms_with_a_column, n_statistics)
        Indexed by ``term``, in the order of ``terms``. The names of
        :func:`coefficient_statistics` over the sessions with a coefficient,
        and:

        - ``quartile_25``, ``quartile_75``: the sessions' 25th and 75th
          percentiles (linear interpolation).
        - ``bse_median``: the median of ``bse_<term>`` over those sessions; NaN
          when the fits carry no ``bse_<term>``.
        - ``n_unidentified``: sessions that, by the fit's own record, cannot
          inform the term (``unidentified_terms``); 0 when the fits do not
          record it.
        - ``signed_rank_p_adjusted``: ``signed_rank_p`` Benjamini-Hochberg
          corrected over the terms that have one
          (:func:`~continuum_swr_replay.analysis.statistics.benjamini_hochberg`);
          NaN where ``signed_rank_p`` is.
    """
    rows = {}
    for term in terms:
        column = f"coef_{term}"
        if column not in fits.columns:
            continue
        known = fits[column].dropna()
        bse = fits.get(f"bse_{term}")
        rows[term] = {
            **coefficient_statistics(fits, column),
            "quartile_25": float(known.quantile(0.25)),
            "quartile_75": float(known.quantile(0.75)),
            "bse_median": float("nan")
            if bse is None
            else float(bse[known.index].median()),
            "n_unidentified": int(unidentified_for_term(fits, term).sum()),
        }
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("term")
    if not table.empty:
        table["signed_rank_p_adjusted"] = benjamini_hochberg(table["signed_rank_p"])
    return table


def gated_coefficient_statistics(fits: pd.DataFrame, term: str) -> dict[str, float]:
    """One term's coefficient across the sessions the per-term gate keeps.

    The statistics are returned however many sessions the gate leaves out. The
    gate's own refusal
    (:attr:`~continuum_swr_replay.analysis.replay.contrast_aggregation.TermSessionGate.refusal`:
    too few kept sessions, or too large a share left out), at which the audited
    aggregates publish no row, is returned as ``gate_refuses`` for the caller
    to act on; ``n_dropped`` and ``n_unidentified`` say how many sessions the
    kept ones stand for.

    Parameters
    ----------
    fits : pd.DataFrame, shape (n_sessions, n_columns)
        Per-session fits of one rung and population, with the session columns,
        ``coef_<term>`` and the columns
        :func:`~continuum_swr_replay.analysis.replay.contrast_aggregation.gate_term_sessions`
        reads.
    term : str
        The term.

    Returns
    -------
    statistics : dict
        The names of :func:`coefficient_statistics` over the kept sessions, and:

        - ``n_dropped``: sessions with a coefficient that the gate leaves out
          as separated or runaway.
        - ``n_unidentified``: sessions with no coefficient because, by the
          fit's own record, the session cannot inform the term.
        - ``gate_refuses``: whether the gate refuses the term.
        - ``session_se_median``: the median over sessions of the session's
          standard error of the coefficient (event-clustered where the fit
          wrote one); NaN when the fits carry no standard error.
        - ``significant_fraction``: of the kept sessions with a usable standard
          error (a finite ratio of coefficient to standard error), the share
          whose coefficient differs from zero at 0.05, two-sided, on that
          standard error. When no kept session has one, the share of all kept
          sessions whose own p-value (``pval_<term>``) is below 0.05; NaN when
          the fits carry neither.
    """
    gate = gate_term_sessions(fits, term)
    kept = fits[gate.kept]
    column = f"coef_{term}"
    se, _ = session_se(kept, term)
    return {
        **coefficient_statistics(kept, column),
        "n_dropped": gate.n_dropped,
        "n_unidentified": gate.n_unidentified,
        "gate_refuses": gate.refusal is not None,
        "session_se_median": float(se.median()) if se is not None else float("nan"),
        "significant_fraction": float(
            pct_significant_sessions(kept[column], se, kept.get(f"pval_{term}"))
        ),
    }


def gated_exp_median_ratio(
    fits_a: pd.DataFrame, fits_b: pd.DataFrame, term: str
) -> float:
    """``exp`` of one term's median coefficient in one set of fits, over the other's.

    Parameters
    ----------
    fits_a, fits_b : pd.DataFrame
        Per-session fits of one rung, each of one population
        (:func:`gated_coefficient_statistics`).
    term : str
        The term.

    Returns
    -------
    ratio : float
        ``exp_median`` of ``fits_a`` divided by ``exp_median`` of ``fits_b``,
        each over its own gated sessions.
    """
    return (
        gated_coefficient_statistics(fits_a, term)["exp_median"]
        / gated_coefficient_statistics(fits_b, term)["exp_median"]
    )


def gated_coefficient_difference_statistics(
    fits_a: pd.DataFrame, fits_b: pd.DataFrame, term: str
) -> dict[str, float]:
    """One term's coefficient in two fits of the same sessions, differenced
    within session.

    A session enters when the per-term gate keeps the term in both of its fits.
    As in :func:`gated_coefficient_statistics`, the gate's refusal is returned,
    not applied.

    Parameters
    ----------
    fits_a, fits_b : pd.DataFrame
        Per-session fits of one rung, each of one population, with the session
        columns and ``coef_<term>``; at most one row per session.
    term : str
        The term.

    Returns
    -------
    statistics : dict
        :func:`~continuum_swr_replay.analysis.statistics.session_median_statistics`
        of the sessions' ``a - b`` differences, and for each of the two sets of
        fits (``a``, ``b``):

        - ``n_dropped_<fits>``: its sessions with a coefficient that the gate
          leaves out as separated or runaway.
        - ``n_unidentified_<fits>``: its sessions with no coefficient because,
          by the fit's own record, the session cannot inform the term.

        and ``gate_refuses``: whether the gate refuses the term in either set.
    """
    gate_a, gate_b = gate_term_sessions(fits_a, term), gate_term_sessions(fits_b, term)
    difference, animal = _paired_difference(
        fits_a[gate_a.kept], fits_b[gate_b.kept], f"coef_{term}"
    )
    return {
        **session_median_statistics(difference, animal),
        "n_dropped_a": gate_a.n_dropped,
        "n_unidentified_a": gate_a.n_unidentified,
        "n_dropped_b": gate_b.n_dropped,
        "n_unidentified_b": gate_b.n_unidentified,
        "gate_refuses": gate_a.refusal is not None or gate_b.refusal is not None,
    }


def aic_difference_statistics(
    child: pd.DataFrame, parent: pd.DataFrame, same: Sequence[str] = ()
) -> dict[str, float]:
    """A child rung's AIC minus its parent's, within session.

    Parameters
    ----------
    child, parent : pd.DataFrame
        Per-session fits of the two rungs for one population, with the session
        columns, ``aic`` and the ``same`` columns; at most one row per session.
    same : sequence of str
        Columns that must be equal in a session's two fits for it to count
        (``n_stops``: both rungs were fit on the same stops).

    Returns
    -------
    statistics : dict
        :func:`~continuum_swr_replay.analysis.statistics.session_median_statistics`
        of ``AIC(child) - AIC(parent)`` over the sessions with both fits and a difference that is not missing. An
        infinite AIC is not missing: its session is kept, with an infinite
        difference, unless both AICs are infinite. Negative means the child is
        better, so ``negative_fraction`` is the share of sessions in which the
        child has the lower AIC and ``n_animals_negative`` the animals whose
        median step favours it.
    """
    difference, animal = _paired_difference(child, parent, "aic", same)
    difference = difference.dropna()
    return session_median_statistics(difference, animal[difference.index])


def separated_fraction(fits: pd.DataFrame) -> float:
    """The share of fits flagged separated.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits with the fit-level ``separated`` flag.

    Returns
    -------
    fraction : float
        In [0, 1]. A missing flag counts as not separated.
    """
    return float((fits["separated"] == True).mean())  # noqa: E712 (NaN must not count)


def known_fraction(fits: pd.DataFrame, column: str) -> float:
    """The share of fits that have a value in one column.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits.
    column : str
        A column a fit may lack a value in (a held-out score, which a fit
        carries only when its cross-validation ran).

    Returns
    -------
    fraction : float
        In [0, 1]: the fits whose value is not missing, over all the fits. An
        infinite value is a value.
    """
    return float(fits[column].notna().mean())


def fits_carrying(fits: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
    """The fits that have a coefficient for every one of some terms.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits with ``coef_<term>`` for each of ``terms``.
    terms : sequence of str
        The terms.

    Returns
    -------
    carrying : pd.DataFrame
        The rows of ``fits`` in which no ``coef_<term>`` is missing.
    """
    return fits.dropna(subset=[f"coef_{term}" for term in terms])


def term_separation_statistics(
    fits: pd.DataFrame, terms: Sequence[str]
) -> dict[str, float]:
    """How many fits flag any of ``terms`` separated.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits with ``separated_terms``, the ``"|"``-joined names of
        the terms each fit flags
        (:func:`~continuum_swr_replay.analysis.replay.contrast_aggregation.separated_for_term`).
    terms : sequence of str
        The terms.

    Returns
    -------
    statistics : dict
        ``n_fits``, the fits; ``n_flagged``, those flagging at least one of
        ``terms``; ``flagged_fraction``, their share.
    """
    flagged = pd.Series(False, index=fits.index)
    for term in terms:
        flagged |= separated_for_term(fits, term)
    return {
        "n_fits": len(fits),
        "n_flagged": int(flagged.sum()),
        "flagged_fraction": float(flagged.mean()),
    }


def rung_separation_range(fits: pd.DataFrame, term: str) -> dict[str, int]:
    """The fewest and most fits, separated fits and term-flagging fits over the
    rungs that carry a term.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits of several rungs, with ``model``, ``separated``,
        ``separated_terms`` and ``coef_<term>``.
    term : str
        The term. A rung carries it when any of its fits has a coefficient
        for it; every fit of such a rung is counted.

    Returns
    -------
    statistics : dict
        ``n_rungs``, the rungs that carry the term. For each of ``n_fits`` (a
        rung's fits), ``n_separated`` (its fits flagged separated) and
        ``n_flagged`` (its fits flagging ``term``), the lowest and highest
        over those rungs: ``<name>_min`` and ``<name>_max``.
    """
    carrying = fits.loc[fits[f"coef_{term}"].notna(), "model"].unique()
    rungs = fits[fits["model"].isin(carrying)]
    by_rung = rungs["model"]
    per_rung = pd.DataFrame(
        {
            "n_fits": rungs.groupby(by_rung).size(),
            "n_separated": (rungs["separated"] == True).groupby(by_rung).sum(),  # noqa: E712
            "n_flagged": separated_for_term(rungs, term).groupby(by_rung).sum(),
        }
    )
    return {
        "n_rungs": len(per_rung),
        **{
            f"{column}_{end}": int(getattr(per_rung[column], end)())
            for column in per_rung
            for end in ("min", "max")
        },
    }


def audited_drop_statistics(audited: pd.DataFrame) -> dict[str, float | str]:
    """How many sessions the per-term gate cost the rows of an audited aggregate.

    Parameters
    ----------
    audited : pd.DataFrame
        Rows of an audited aggregate
        (:func:`~continuum_swr_replay.analysis.replay.contrast_aggregation.aggregate_contrast`):
        one per rung and term, with ``model``, ``term``, ``n_sessions`` (kept)
        and ``n_dropped`` (left out as separated or runaway).

    Returns
    -------
    statistics : dict
        - ``n_terms_losing_sessions``: rows with at least one dropped session.
        - ``most_dropped``: the most sessions any row drops.
        - ``most_dropped_of``: that row's kept and dropped sessions together,
          the sessions with a coefficient for the term (a session the fit
          records as unable to inform it has none and is not counted).
        - ``most_dropped_model``, ``most_dropped_term``: that row's rung and
          term (the first such row when several tie).
    """
    worst = audited.loc[audited["n_dropped"].idxmax()]
    return {
        "n_terms_losing_sessions": int((audited["n_dropped"] > 0).sum()),
        "most_dropped": int(worst["n_dropped"]),
        "most_dropped_of": int(worst["n_dropped"] + worst["n_sessions"]),
        "most_dropped_model": worst["model"],
        "most_dropped_term": worst["term"],
    }


def group_range(
    values: Mapping[str, float], excluded: Mapping[str, bool] | None = None
) -> dict[str, float | str]:
    """The lowest and highest of one number over groups, and the groups that
    hold them.

    Parameters
    ----------
    values : mapping of str to float
        The number, by group (a population, a rung).
    excluded : mapping of str to bool, optional
        By group, whether to leave it out of the range (a time-course peak on
        the edge of its interval bounds the peak rather than locating it).

    Returns
    -------
    statistics : dict
        ``low`` and ``high``, the lowest and highest over the groups kept;
        ``low_group`` and ``high_group``, the groups that hold them (the first
        in the order of ``values`` when several tie); ``n_excluded``, the groups
        left out.

    Raises
    ------
    ValueError
        If no group is kept, or a kept group's number is missing (the lowest
        and highest would then depend on the order of the groups).
    """
    kept = {
        group: value
        for group, value in values.items()
        if not (excluded and excluded[group])
    }
    if not kept:
        raise ValueError("no group is left to take a range over")
    missing = [group for group, value in kept.items() if np.isnan(value)]
    if missing:
        raise ValueError(f"no number to take a range over for {missing}")
    low, high = min(kept, key=kept.__getitem__), max(kept, key=kept.__getitem__)
    return {
        "low": kept[low],
        "low_group": low,
        "high": kept[high],
        "high_group": high,
        "n_excluded": len(values) - len(kept),
    }
