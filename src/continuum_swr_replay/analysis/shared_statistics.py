"""Statistics that know nothing of the task: any analysis package may use them.

- :func:`signed_rank_p`: the Wilcoxon signed-rank p of a set of differences
  against zero.
- :func:`benjamini_hochberg`: Benjamini-Hochberg adjusted p-values of a family
  of tests.
- :func:`session_median_statistics`: the animal-balanced summary of one number
  per session: the median over sessions with the sessions' signs, and the
  median over animals of each animal's own median with the animals' signs.
- :func:`nested_bootstrap_draw`: one replicate of the nested bootstrap
  (animals, then sessions within each drawn animal, then the units or events
  within each drawn session), the resampling behind every interval; with
  :func:`nested_codes` and :func:`weighted_median`.

A signed-rank p over sessions treats the sessions as independent, which
sessions of one animal are not: it describes consistency across sessions and is
not a test on the animals.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
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
    """One replicate of :func:`nested_bootstrap_draw`.

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
    """
    animal = np.asarray(animal, dtype=np.int64)
    session = np.asarray(session, dtype=np.int64)
    n_animals = int(animal.max()) + 1
    session_animal = np.zeros(int(session.max()) + 1, dtype=np.int64)
    session_animal[session] = animal
    animal_draws = rng.multinomial(n_animals, np.full(n_animals, 1.0 / n_animals))
    session_draws = np.zeros(session_animal.size, dtype=np.int64)
    for a in np.flatnonzero(animal_draws):
        mine = np.flatnonzero(session_animal == a)
        session_draws[mine] = rng.multinomial(
            animal_draws[a] * mine.size, np.full(mine.size, 1.0 / mine.size)
        )
    counts = np.zeros(animal.size, dtype=np.int64)
    order = np.argsort(session, kind="stable")
    bounds = np.searchsorted(session[order], np.arange(session_animal.size + 1))
    # A session of one row draws it as often as the session is drawn (a
    # one-category multinomial takes nothing from the generator).
    single = np.diff(bounds) == 1
    counts[order[bounds[:-1][single]]] = session_draws[single]
    for s in np.flatnonzero((session_draws > 0) & ~single):
        members = order[bounds[s] : bounds[s + 1]]
        counts[members] = rng.multinomial(
            session_draws[s] * members.size, np.full(members.size, 1.0 / members.size)
        )
    return NestedDraw(animal_draws, session_draws, counts)


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
