"""Statistics that know nothing of the task: any analysis package may use them.

- :func:`signed_rank_p`: the Wilcoxon signed-rank p of a set of differences
  against zero.
- :func:`benjamini_hochberg`: Benjamini-Hochberg adjusted p-values of a family
  of tests.
- :func:`akaike_information_criterion`, :func:`bayesian_information_criterion`
  and :func:`mcfadden_pseudo_r2`: a fit's information criteria and pseudo-R²
  from its log-likelihood.
- :func:`session_median_statistics`: the animal-balanced summary of one number
  per session: the median over sessions with the sessions' signs, and the
  median over animals of each animal's own median with the animals' signs.
- :func:`animal_session_values`: each animal's value of a per-row number (an
  event's, a unit's), row -> session -> animal: a median of session medians or
  a mean of session means.
- :func:`nested_bootstrap_draw`: one replicate of the nested bootstrap
  (animals, then sessions within each drawn animal, then the units or events
  within each drawn session), the resampling behind every interval, and
  :func:`nested_bootstrap_draws`, many at once; with :func:`nested_codes` and
  :func:`weighted_median`.
- :func:`bootstrap_interval`: the 95 % interval from those draws, a
  small-sample t correction over the animals.
- :func:`session_mean_curve`, :func:`animal_balanced_curve` and
  :func:`animal_curve_sem`: a curve (one value per lag) taken unit -> session
  -> animal, the mean over animals of each animal's mean over its sessions, and
  its standard error over animals.

A signed-rank p over sessions treats the sessions as independent, which
sessions of one animal are not: it describes consistency across sessions and is
not a test on the animals.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

MIN_SIGNED_RANK_DIFFERENCES = 6
"""The fewest non-zero differences a signed-rank p is computed for.

With n differences the smallest two-sided exact p is 2 / 2 ** n, 0.0625 at
n = 5, so below 6 the test cannot reject at 0.05."""


def signed_rank_p(differences: ArrayLike) -> float:
    """The two-sided Wilcoxon signed-rank p of differences against zero.

    Parameters
    ----------
    differences : array_like, shape (n_differences,)
        Paired differences, or values tested against zero. A missing or
        infinite value is left out; a zero is not a difference (scipy's
        default ``zero_method="wilcox"``).

    Returns
    -------
    p : float
        :func:`scipy.stats.wilcoxon` with its defaults on the finite values;
        NaN when fewer than :data:`MIN_SIGNED_RANK_DIFFERENCES` of them are
        non-zero.
    """
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if np.count_nonzero(values) < MIN_SIGNED_RANK_DIFFERENCES:
        return float("nan")
    return float(wilcoxon(values).pvalue)


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values (the step-up false discovery rate).

    A missing or infinite p is not a test: it is left out of the family and its
    adjusted value is NaN.

    Parameters
    ----------
    p_values : array_like, shape (n_tests,)
        The family's p-values.

    Returns
    -------
    adjusted : numpy.ndarray, shape (n_tests,)
        In the order of ``p_values``. For the ``m`` finite p-values sorted
        ascending, the k-th is ``p / (k / m)``, then the smallest of that over
        it and every larger p (the step-up), at most 1. The arithmetic is that
        of statsmodels' ``multipletests(method="fdr_bh")``.
    """
    p = np.asarray(p_values, dtype=float)
    tested = np.isfinite(p)
    adjusted = np.full(p.shape, np.nan)
    m = int(tested.sum())
    order = np.argsort(p[tested])
    stepped = p[tested][order] / (np.arange(1, m + 1) / m)
    stepped = np.minimum(np.minimum.accumulate(stepped[::-1])[::-1], 1.0)
    in_family = np.empty(m)
    in_family[order] = stepped
    adjusted[tested] = in_family
    return adjusted


def akaike_information_criterion(llf: ArrayLike, n_params: ArrayLike) -> ArrayLike:
    """AIC of a fit: ``2 k - 2 llf``.

    Parameters
    ----------
    llf : float or array_like
        The fit's maximized log-likelihood (one per fit for an array).
    n_params : int or array_like
        Its number of estimated parameters, ``k``.

    Returns
    -------
    aic : float or numpy.ndarray
        Same shape as the broadcast inputs.
    """
    return 2 * n_params - 2 * llf


def bayesian_information_criterion(
    llf: ArrayLike, n_params: ArrayLike, n_observations: ArrayLike
) -> ArrayLike:
    """BIC of a fit: ``k log(n) - 2 llf``.

    Parameters
    ----------
    llf : float or array_like
        The fit's maximized log-likelihood.
    n_params : int or array_like
        Its number of estimated parameters, ``k``.
    n_observations : int or array_like
        The ``n`` of the penalty: the likelihood's independent units, which
        the caller names (events for a fit conditioned on events).

    Returns
    -------
    bic : float or numpy.ndarray
        Same shape as the broadcast inputs; NaN when ``n`` is 0.
    """
    return n_params * np.log(n_observations) - 2 * llf


def mcfadden_pseudo_r2(llf: float, llf_null: float) -> float:
    """McFadden's pseudo-R² of a fit against its null: ``1 - llf / llf_null``.

    Parameters
    ----------
    llf, llf_null : float
        The fit's and its null model's log-likelihoods, on one scale.

    Returns
    -------
    pseudo_r2 : float
        NaN when the null log-likelihood is not finite or is zero.
    """
    return 1.0 - llf / llf_null if np.isfinite(llf_null) and llf_null != 0 else np.nan


def animal_medians(values: pd.Series, animal: pd.Series) -> pd.Series:
    """Each animal's median of a per-session (or per-event) number.

    Parameters
    ----------
    values : pd.Series, shape (n_rows,)
        One number per row. A missing value is left out.
    animal : pd.Series, shape (n_rows,)
        Each row's animal, indexed as ``values``. An animal (a category of a
        categorical) with no row that has a value is not in the result.

    Returns
    -------
    medians : pd.Series, shape (n_animals,)
        Indexed by animal in sorted order.

    Raises
    ------
    ValueError
        If ``animal`` is not indexed as ``values``.
    """
    if not values.index.equals(animal.index):
        raise ValueError("values and animal are not indexed alike")
    has_value = values.notna()
    return values[has_value].groupby(animal[has_value], observed=True).median()


def animal_session_values(
    values: pd.Series,
    animal: pd.Series,
    session: pd.Series | pd.DataFrame,
    statistic: str,
) -> pd.Series:
    """Each animal's value of a per-row number, row -> session -> animal.

    The ``statistic`` over each session's rows, then the same statistic over
    each animal's sessions, so every session of an animal weighs the same
    however many rows it holds.

    Parameters
    ----------
    values : pd.Series, shape (n_rows,)
        One number per row (an event, a unit). A missing value is left out; a
        session with no value is not one of its animal's sessions.
    animal : pd.Series, shape (n_rows,)
        Each row's animal, indexed as ``values``.
    session : pd.Series or pd.DataFrame, shape (n_rows,) or (n_rows, n_columns)
        Each row's session, within its animal (a label, or the columns that
        together name it), indexed as ``values``.
    statistic : {"median", "mean"}
        The reduction at both levels: a median of session medians, or a mean
        of session means.

    Returns
    -------
    per_animal : pd.Series, shape (n_animals,)
        Indexed by animal in sorted order; an animal with no value is not in
        it.

    Raises
    ------
    ValueError
        If ``statistic`` is neither ``"median"`` nor ``"mean"``, or ``animal``
        or ``session`` is not indexed as ``values``.
    """
    if statistic not in ("median", "mean"):
        raise ValueError(f"statistic {statistic!r} is neither 'median' nor 'mean'")
    session = session.to_frame() if isinstance(session, pd.Series) else session
    if not (values.index.equals(animal.index) and values.index.equals(session.index)):
        raise ValueError("values, animal and session are not indexed alike")
    has_value = values.notna()
    keys = [animal[has_value], *(session.loc[has_value, c] for c in session.columns)]
    per_session = values[has_value].groupby(keys, observed=True).agg(statistic)
    return per_session.groupby(level=0, observed=True).agg(statistic)


def session_median_statistics(values: pd.Series, animal: pd.Series) -> dict[str, float]:
    """The median of a per-session number, with the sessions' and animals' signs.

    Parameters
    ----------
    values : pd.Series, shape (n_sessions,)
        One number per session. A missing value is left out: its session
        counts in nothing. An infinite value is a value.
    animal : pd.Series, shape (n_sessions,)
        Each session's animal, indexed as ``values``. An animal (a category of
        a categorical) with no session that has a value is not an animal of
        the summary.

    Returns
    -------
    statistics : dict
        - ``n_sessions``: the sessions with a value.
        - ``median``: the median over sessions.
        - ``n_positive``, ``n_negative``: sessions above and below zero.
        - ``positive_fraction``, ``negative_fraction``: their shares of the
          sessions.
        - ``n_animals``: the animals with a session.
        - ``animal_median``: the median over animals of each animal's median
          over its sessions.
        - ``n_animals_positive``, ``n_animals_negative``: animals whose own
          median is above and below zero.
        - ``n_animals_same_sign``: animals whose own median has the sign of
          ``median``. Zero has its own sign: when ``median`` is exactly zero,
          these are the animals whose own median is exactly zero.

    Raises
    ------
    ValueError
        If ``animal`` is not indexed as ``values`` (a session without an
        animal would be left out of the animals' medians).
    """
    per_animal = animal_medians(values, animal)
    known = values.dropna()
    median = float(known.median())
    return {
        "n_sessions": len(known),
        "median": median,
        "n_positive": int((known > 0).sum()),
        "n_negative": int((known < 0).sum()),
        "positive_fraction": float((known > 0).mean()),
        "negative_fraction": float((known < 0).mean()),
        "n_animals": len(per_animal),
        "animal_median": float(per_animal.median()),
        "n_animals_positive": int((per_animal > 0).sum()),
        "n_animals_negative": int((per_animal < 0).sum()),
        "n_animals_same_sign": int((np.sign(per_animal) == np.sign(median)).sum()),
    }


def nested_codes(
    animal: ArrayLike, session: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Each row's animal and session as codes ``0..n-1``, for
    :func:`nested_bootstrap_draw`.

    Parameters
    ----------
    animal, session : array_like, shape (n_rows,)
        Each row's animal and session labels. A session label names one
        session across animals.

    Returns
    -------
    animal_codes, session_codes : NDArray of int, shape (n_rows,)
        Codes in sorted label order (:func:`numpy.unique`).

    Raises
    ------
    ValueError
        If a session belongs to more than one animal.
    """
    _, animal_codes = np.unique(np.asarray(animal), return_inverse=True)
    _, session_codes = np.unique(np.asarray(session), return_inverse=True)
    pairs = pd.DataFrame({"a": animal_codes, "s": session_codes}).drop_duplicates()
    if pairs.s.duplicated().any():
        raise ValueError("a session belongs to more than one animal")
    return animal_codes.astype(np.int64), session_codes.astype(np.int64)


class NestedDraw(NamedTuple):
    """Replicates of the nested bootstrap: one of :func:`nested_bootstrap_draw`,
    or ``n_draws`` of :func:`nested_bootstrap_draws` (each array then has a
    leading axis of length ``n_draws``).

    Attributes
    ----------
    animals : NDArray of int, shape (n_animals,)
        How often each animal is drawn, by animal code.
    sessions : NDArray of int, shape (n_sessions,)
        How often each session is drawn, by session code.
    rows : NDArray of int, shape (n_rows,)
        How often each row (a unit, an event, or a session when the sessions
        are the rows) is drawn.
    """

    animals: NDArray[np.int64]
    sessions: NDArray[np.int64]
    rows: NDArray[np.int64]


def nested_bootstrap_draw(
    animal: NDArray[np.int64], session: NDArray[np.int64], rng: np.random.Generator
) -> NestedDraw:
    """One replicate of the nested bootstrap: how often each animal, session
    and row is drawn.

    Animals are drawn with replacement (as many as there are); each draw of an
    animal draws its sessions with replacement (as many as it has); each draw of
    a session draws its rows with replacement (as many as it has). Drawing an
    animal or a session ``k`` times and resampling its members each time is
    drawn as one multinomial of ``k`` times as many members, so the ``k``
    copies' members are pooled: exact for a statistic pooled over the rows and
    for a mean of means; a median within a drawn animal or session is taken
    over the pooled members of its copies.

    Parameters
    ----------
    animal, session : NDArray of int, shape (n_rows,)
        Codes ``0..n-1`` (:func:`nested_codes`); every session belongs to one
        animal.
    rng : numpy.random.Generator
        The draws' source.

    Returns
    -------
    NestedDraw
        :func:`nested_bootstrap_draws` with one draw, its leading axis dropped.
    """
    draws = nested_bootstrap_draws(animal, session, rng, 1)
    return NestedDraw(draws.animals[0], draws.sessions[0], draws.rows[0])


def nested_bootstrap_draws(
    animal: NDArray[np.int64],
    session: NDArray[np.int64],
    rng: np.random.Generator,
    n_draws: int,
) -> NestedDraw:
    """``n_draws`` replicates of the nested bootstrap at once.

    Each replicate is drawn as in :func:`nested_bootstrap_draw`; the generator
    is used level by level for all replicates together (the animals of every
    replicate, then each animal's sessions, then each session's rows), so one
    draw here is :func:`nested_bootstrap_draw`'s with the same generator, and
    ``n_draws`` here are not the numbers of ``n_draws`` calls of it (they have
    the same distribution).

    Parameters
    ----------
    animal, session : NDArray of int, shape (n_rows,)
        Codes ``0..n-1`` (:func:`nested_codes`); every session belongs to one
        animal.
    rng : numpy.random.Generator
        The draws' source.
    n_draws : int
        Replicates.

    Returns
    -------
    NestedDraw
        Arrays of shape (n_draws, n_animals), (n_draws, n_sessions) and
        (n_draws, n_rows).
    """
    animal = np.asarray(animal, dtype=np.int64)
    session = np.asarray(session, dtype=np.int64)
    n_animals = int(animal.max()) + 1
    session_animal = np.zeros(int(session.max()) + 1, dtype=np.int64)
    session_animal[session] = animal
    animal_draws = rng.multinomial(
        n_animals, np.full(n_animals, 1.0 / n_animals), size=n_draws
    )
    session_draws = np.zeros((n_draws, session_animal.size), dtype=np.int64)
    # A multinomial of zero draws takes nothing from the generator, so a
    # replicate that does not draw an animal (or a session) uses none of it.
    for a in np.flatnonzero(animal_draws.any(axis=0)):
        mine = np.flatnonzero(session_animal == a)
        session_draws[:, mine] = rng.multinomial(
            animal_draws[:, a] * mine.size, np.full(mine.size, 1.0 / mine.size)
        )
    counts = np.zeros((n_draws, animal.size), dtype=np.int64)
    order = np.argsort(session, kind="stable")
    bounds = np.searchsorted(session[order], np.arange(session_animal.size + 1))
    # A session of one row draws it as often as the session is drawn (a
    # one-category multinomial takes nothing from the generator).
    single = np.diff(bounds) == 1
    counts[:, order[bounds[:-1][single]]] = session_draws[:, single]
    for s in np.flatnonzero((session_draws > 0).any(axis=0) & ~single):
        members = order[bounds[s] : bounds[s + 1]]
        counts[:, members] = rng.multinomial(
            session_draws[:, s] * members.size,
            np.full(members.size, 1.0 / members.size),
        )
    return NestedDraw(animal_draws, session_draws, counts)


class BootstrapInterval(NamedTuple):
    """A 95 % interval of :func:`bootstrap_interval`.

    Attributes
    ----------
    low, high : float
        The interval's bounds.
    sd : float
        The standard deviation of the finite draws (``ddof=1``).
    """

    low: float
    high: float
    sd: float


def bootstrap_interval(
    estimate: float,
    draws: ArrayLike,
    n_top: int,
    bounds: tuple[float, float] | None = None,
) -> BootstrapInterval:
    """The 95 % interval of a statistic from its nested bootstrap draws.

    ``estimate +/- t(0.975, n_top - 1) * sqrt(n_top / (n_top - 1)) * SD(draws)``,
    symmetric about the statistic on the data. ``n_top`` is the number of
    top-level units the draws resample, the animals.

    The percentiles of the draws are not the interval: with five animals the
    nested percentile bootstrap under-covers. Saravanan, Berman & Sober (2020,
    section 3.1.3) find its false-positive rate above the nominal level with
    few subjects; in a simulation of five animals of forty sessions each
    (normal values, true value zero) a nominal 95 % percentile interval covered
    84 to 97 % for a mean of animal means and 85 to 98 % for a median pooled
    over sessions, and this interval 95 to 99.8 % and 96 to 99.8 %. The draws'
    SD of a mean over ``n`` animals is ``sqrt((n - 1) / n)`` times the standard
    error ``s / sqrt(n)``, so the factor ``sqrt(n / (n - 1))`` restores it and
    the interval of a mean is the t-interval over the animals.

    Parameters
    ----------
    estimate : float
        The statistic on the data.
    draws : array_like, shape (n_draws,)
        The statistic on each bootstrap replicate
        (:func:`nested_bootstrap_draw`). A non-finite draw is left out.
    n_top : int
        The number of animals the draws resample.
    bounds : tuple of float, optional
        The range the statistic can take (for a lag, the first and last lag).
        A bound beyond it is clipped to it: the symmetric interval can run
        past a range the statistic cannot leave.

    Returns
    -------
    BootstrapInterval
        ``low`` and ``high`` are NaN when ``estimate`` is not finite, when
        ``n_top`` is below 2 (no t quantile) or when fewer than two draws are
        finite; ``sd`` is NaN with fewer than two finite draws.
    """
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    sd = float(values.std(ddof=1)) if values.size >= 2 else float("nan")
    if n_top < 2 or not np.isfinite(estimate) or not np.isfinite(sd):
        return BootstrapInterval(float("nan"), float("nan"), sd)
    half = float(student_t.ppf(0.975, n_top - 1)) * np.sqrt(n_top / (n_top - 1)) * sd
    low, high = float(estimate - half), float(estimate + half)
    if bounds is not None:
        low, high = max(low, bounds[0]), min(high, bounds[1])
    return BootstrapInterval(low, high, sd)


def weighted_median(values_sorted: NDArray, weights: NDArray) -> float:
    """The median of ``values_sorted`` with each value repeated ``weights`` times.

    With whole weights this is ``np.median`` of the repeated sample (the mean
    of the two middle values when the total is even).

    Parameters
    ----------
    values_sorted : NDArray, shape (n,)
        Ascending.
    weights : NDArray, shape (n,)
        Non-negative counts.

    Returns
    -------
    float
        NaN when the weights sum to 0.
    """
    cumulative = np.cumsum(weights)
    total = cumulative[-1] if cumulative.size else 0
    if total <= 0:
        return float("nan")
    # 1-based positions of the two middle draws of the repeated sample
    low = np.searchsorted(cumulative, (total + 1) // 2, side="left")
    high = np.searchsorted(cumulative, total // 2 + 1, side="left")
    return float((values_sorted[low] + values_sorted[high]) / 2.0)


# --- curves over sessions and animals ----------------------------------------------


def session_mean_curve(unit_curves: NDArray) -> NDArray:
    """One session's curve: the mean over its units whose curve is finite.

    Parameters
    ----------
    unit_curves : NDArray, shape (n_lags, n_units)
        The session's per-unit curves.

    Returns
    -------
    NDArray, shape (n_lags,)
        The mean over the units whose curve is finite at every lag; a unit
        with a non-finite value at any lag is left out whole.

    Raises
    ------
    ValueError
        If no unit's curve is finite at every lag.
    """
    finite = np.isfinite(unit_curves).all(axis=0)
    if not finite.any():
        raise ValueError("no unit's curve is finite at every lag")
    return unit_curves[:, finite].mean(axis=1)


def animal_balanced_curve(
    by_session: dict[str, np.ndarray], animal_of: dict[str, str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Unit-mean session curves -> per-animal means -> mean over animals.

    Parameters
    ----------
    by_session : dict
        ``session -> array``; the last axis is lag (a curve, or draws x lags).
    animal_of : dict
        ``session -> animal``.

    Returns
    -------
    population : NDArray
        Mean over animals of each animal's mean over its sessions.
    per_animal : dict
        ``animal -> that animal's mean over its sessions``.
    """
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for session, curve in by_session.items():
        grouped[animal_of[session]].append(np.asarray(curve, dtype=float))
    per_animal = {
        animal: np.mean(np.stack(group), axis=0)
        for animal, group in sorted(grouped.items())
    }
    return np.mean(np.stack(list(per_animal.values())), axis=0), per_animal


def animal_curve_sem(per_animal: dict[str, np.ndarray]) -> np.ndarray:
    """Standard error over animals of the animals' mean curves, at each lag.

    Parameters
    ----------
    per_animal : dict
        ``animal -> curve``, shape (n_lags,), as :func:`animal_balanced_curve`
        returns them.

    Returns
    -------
    NDArray, shape (n_lags,)
        The sample standard deviation over animals (``ddof=1``) over the
        square root of the number of animals; NaN with one animal.
    """
    stack = np.stack(list(per_animal.values()))  # (n_animals, n_lags)
    if len(stack) < 2:
        return np.full(stack.shape[1], np.nan)
    return stack.std(axis=0, ddof=1) / np.sqrt(len(stack))
