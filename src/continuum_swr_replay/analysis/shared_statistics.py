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
- :func:`animal_balanced_bootstrap`: a per-session number (or curve) taken
  session -> recording day -> animal as a mean of means, with its nested
  bootstrap draws (animals, then their sessions; the days are not resampled).
- :func:`animal_day_session_values`, :func:`day_level_test` and
  :func:`estimand_summary`: each animal's value row -> session -> day ->
  animal; the signed-rank p over recording days; and a scalar estimand's
  summary over sessions (the animal-balanced estimate, its interval, the
  animals with its sign and the day-level p).
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


class AnimalBalancedDraws(NamedTuple):
    """A per-session number taken session -> day -> animal, and its bootstrap
    draws (:func:`animal_balanced_bootstrap`).

    Attributes
    ----------
    estimate : NDArray, shape (*value_shape)
        The mean over animals of each animal's mean over its days of each
        day's mean over its sessions.
    draws : NDArray, shape (n_draws, *value_shape)
        The same statistic on each nested bootstrap replicate.
    n_animals : int
        The animals with a session (the draws' top level).
    """

    estimate: NDArray[np.float64]
    draws: NDArray[np.float64]
    n_animals: int


def _animal_balanced_weights(
    animal: NDArray[np.int64],
    day: NDArray[np.int64],
    animal_counts: NDArray[np.int64],
    session_counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Each session's weight in the day -> animal mean of each replicate.

    A session's weight is ``A_a / n_animals / n_a * R_s / sum_{s' of d} R_s'``,
    with ``A_a`` and ``R_s`` how often its animal and the session itself are
    drawn and ``n_a`` the number of its animal's days with a drawn session: a
    day's mean is over its drawn sessions, and a day with none is absent. An
    animal drawn ``k`` times draws ``k`` times as many sessions, pooled
    (:func:`nested_bootstrap_draws`), so its days' means are taken over the
    pooled sessions.

    Parameters
    ----------
    animal, day : NDArray of int, shape (n_sessions,)
        Codes; every day belongs to one animal.
    animal_counts : NDArray of int, shape (n_draws, n_animals)
    session_counts : NDArray of int, shape (n_draws, n_sessions)

    Returns
    -------
    NDArray, shape (n_draws, n_sessions)
    """
    n_animals = animal_counts.shape[1]
    n_days = int(day.max()) + 1
    day_animal = np.zeros(n_days, dtype=np.int64)
    day_animal[day] = animal
    day_total = np.stack([np.bincount(day, r, n_days) for r in session_counts])
    days_drawn = np.stack(
        [np.bincount(day_animal, t > 0, n_animals) for t in day_total]
    )
    within_day = np.divide(
        session_counts,
        day_total[:, day],
        out=np.zeros(session_counts.shape),
        where=session_counts > 0,
    )
    per_day = np.divide(
        1.0,
        days_drawn[:, animal],
        out=np.zeros(session_counts.shape),
        where=days_drawn[:, animal] > 0,
    )
    return animal_counts[:, animal] / n_animals * per_day * within_day


def _balanced_estimate_and_draws(
    flat: NDArray[np.float64],
    animal: NDArray[np.int64],
    day: NDArray[np.int64],
    n_draws: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The day -> animal mean of means of ``flat`` and its nested bootstrap
    draws, both from :func:`_animal_balanced_weights`.

    Parameters
    ----------
    flat : NDArray, shape (n_sessions, n_points)
        Finite values.
    animal, day : NDArray of int, shape (n_sessions,)
        Codes (:func:`_animal_day_codes`).
    n_draws, seed : int
        The draws' count and seed.

    Returns
    -------
    estimate : NDArray, shape (1, n_points)
    draws : NDArray, shape (n_draws, n_points)
    """
    n_sessions = flat.shape[0]
    n_animals = int(animal.max()) + 1
    estimate = (
        _animal_balanced_weights(
            animal,
            day,
            np.ones((1, n_animals), dtype=np.int64),
            np.ones((1, n_sessions), dtype=np.int64),
        )
        @ flat
    )
    counts = nested_bootstrap_draws(
        animal,
        np.arange(n_sessions, dtype=np.int64),
        np.random.default_rng(seed),
        n_draws,
    )
    draws = (
        _animal_balanced_weights(animal, day, counts.animals, counts.sessions) @ flat
    )
    return estimate, draws


def animal_balanced_bootstrap(
    values: ArrayLike,
    animal: ArrayLike,
    day: ArrayLike,
    *,
    n_draws: int,
    seed: int,
) -> AnimalBalancedDraws:
    """A per-session number taken session -> day -> animal, with its draws.

    The estimate is the mean over animals of each animal's mean over its
    recording days of each day's mean over its sessions. The draws are the
    project's nested bootstrap of animals, then their sessions
    (:func:`nested_bootstrap_draws`, each session one row); the days are not
    resampled. In each draw the estimate is recomputed on the sessions drawn:
    a day's mean over its drawn sessions, an animal's mean over its days with
    a drawn session, the mean over the drawn animals.
    The draws depend on the seed and on the sorted order of the animal labels
    (the animals' codes, :func:`nested_codes`), so relabeling the animals
    changes the draws, not the estimate.

    Parameters
    ----------
    values : array_like, shape (n_sessions,) or (n_sessions, n_points)
        One number, or one curve on a common grid, per session. A session
        with a value that is not finite (anywhere on its curve) is left out.
    animal, day : array_like, shape (n_sessions,)
        Each session's animal and recording day. A day label names a day
        within its animal: the same label in two animals is two days.
    n_draws : int
        Bootstrap replicates.
    seed : int
        The draws' seed (``numpy.random.default_rng(seed)``).

    Returns
    -------
    AnimalBalancedDraws

    Raises
    ------
    ValueError
        If no session has a finite value, or a label has missing values.
    """
    _check_labels({"animal": animal, "day": day})
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values).all(axis=tuple(range(1, values.ndim)))
    if not finite.any():
        raise ValueError("no session has a finite value")
    values = values[finite]
    animal_codes, day_codes = _animal_day_codes(
        np.asarray(animal)[finite], np.asarray(day)[finite]
    )
    estimate, draws = _balanced_estimate_and_draws(
        values.reshape(values.shape[0], -1), animal_codes, day_codes, n_draws, seed
    )
    return AnimalBalancedDraws(
        estimate.reshape(values.shape[1:]),
        draws.reshape((n_draws, *values.shape[1:])),
        int(animal_codes.max()) + 1,
    )


def _animal_day_codes(
    animal: NDArray, day: NDArray
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Codes of each session's animal and of its (animal, day) pair."""
    pairs = pd.MultiIndex.from_arrays([animal, day])
    day_labels = pd.factorize(pairs, sort=True)[0]
    return nested_codes(animal, day_labels)


def _day_and_animal_means(
    values: NDArray[np.float64], animal: NDArray[np.int64], day: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each day's mean over its sessions and each animal's mean over its days.

    The one reduction session -> day -> animal behind :func:`estimand_summary`,
    :func:`animal_day_session_values` and :func:`day_level_test`.

    Parameters
    ----------
    values : NDArray, shape (n_sessions,)
        Finite values.
    animal, day : NDArray of int, shape (n_sessions,)
        Codes (:func:`_animal_day_codes`).

    Returns
    -------
    day_values : NDArray, shape (n_days,)
        By day code.
    animal_values : NDArray, shape (n_animals,)
        By animal code.
    """
    day_values = np.bincount(day, values) / np.bincount(day)
    day_animal = np.zeros(day_values.size, dtype=np.int64)
    day_animal[day] = animal
    return day_values, np.bincount(day_animal, day_values) / np.bincount(day_animal)


RANK_SIGN_DIGITS = 12
"""The significant digits, relative to the largest absolute value, that the
day-level test and the animals' signs read (:func:`_snapped`)."""


def _snapped(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Values rounded to :data:`RANK_SIGN_DIGITS` significant digits of the
    largest absolute value among them.

    The rule behind every rank and sign the session -> day -> animal reduction
    reports: the day values :func:`day_level_test` ranks, and the animals'
    values :func:`estimand_summary` compares with the estimate's sign. Means
    that are equal in exact arithmetic can come out a bit apart, depending on
    the order of a sum; that makes or breaks a tie of the signed-rank test (and
    switches scipy between its exact and its approximate p) or gives a zero a
    sign. A difference below ``10 ** -RANK_SIGN_DIGITS`` of the largest value
    is below anything the data can mean and far above a sum's rounding error.
    A value that snaps to zero is zero: a zero difference for the day test,
    and an animal without the estimate's sign. The estimate, its interval and
    its draws are not snapped.

    Parameters
    ----------
    values : NDArray, shape (n,)
        Finite values.

    Returns
    -------
    NDArray, shape (n,)
    """
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale == 0.0:
        return values
    unit = 10.0 ** (np.floor(np.log10(scale)) - (RANK_SIGN_DIGITS - 1))
    return np.round(values / unit) * unit + 0.0


def _check_labels(labels: dict[str, ArrayLike]) -> None:
    """Refuse a label column with missing values, naming it and how many."""
    for name, label in labels.items():
        n_missing = int(pd.isna(np.asarray(label, dtype=object)).sum())
        if n_missing:
            raise ValueError(f"the {name} labels have {n_missing} missing values")


def _finite_rows(
    values: pd.Series, labels: dict[str, pd.Series]
) -> tuple[pd.Series, list[pd.Series]]:
    """The rows of ``values`` that are finite, and the labels of those rows.

    Raises
    ------
    ValueError
        If a label is not indexed as ``values``, or has missing values.
    """
    if not all(values.index.equals(s.index) for s in labels.values()):
        names = ", ".join(["values", *labels][:-1]) + f" and {list(labels)[-1]}"
        raise ValueError(f"{names} are not indexed alike")
    _check_labels(labels)
    finite = np.isfinite(values.to_numpy(dtype=float))
    return values[finite], [s[finite] for s in labels.values()]


def animal_day_session_values(
    values: pd.Series,
    animal: pd.Series,
    day: pd.Series,
    session: pd.Series,
) -> pd.Series:
    """Each animal's value of a per-row number, row -> session -> day -> animal.

    The three-level analog of :func:`animal_session_values`: the mean over
    each session's rows, then over each recording day's sessions, then over
    each animal's days, so every day of an animal weighs the same however many
    sessions it holds.

    Parameters
    ----------
    values : pd.Series, shape (n_rows,)
        One number per row (a session, or a unit or event of one). A value
        that is not finite (missing or infinite) is left out.
    animal, day, session : pd.Series, shape (n_rows,)
        Each row's animal, recording day within its animal, and session within
        its day, indexed as ``values``.

    Returns
    -------
    per_animal : pd.Series, shape (n_animals,)
        Indexed by animal in sorted order; an animal with no value is not in
        it.

    Raises
    ------
    ValueError
        If the labels are not indexed as ``values`` or have missing values.
    """
    values, (animal, day, session) = _finite_rows(
        values, {"animal": animal, "day": day, "session": session}
    )
    sessions = pd.MultiIndex.from_arrays([animal, day, session])
    session_codes = pd.factorize(sessions, sort=True)[0]
    first = np.unique(session_codes, return_index=True)[1]
    session_values = np.bincount(session_codes, values.to_numpy(dtype=float))
    session_values /= np.bincount(session_codes)
    session_animal = animal.to_numpy()[first]
    _, per_animal = _day_and_animal_means(
        session_values, *_animal_day_codes(session_animal, day.to_numpy()[first])
    )
    return pd.Series(
        per_animal, index=pd.Index(np.unique(session_animal), name=animal.name)
    )


def day_level_test(values: pd.Series, animal: pd.Series, day: pd.Series) -> float:
    """The signed-rank p over recording days of a per-session number.

    Each recording day's value is the mean of its sessions' values; the test
    is :func:`signed_rank_p` of those day values against zero, read to
    :data:`RANK_SIGN_DIGITS` significant digits (:func:`_snapped`).

    Parameters
    ----------
    values : pd.Series, shape (n_sessions,)
        One number per session. A value that is not finite (missing or
        infinite) is left out.
    animal, day : pd.Series, shape (n_sessions,)
        Each session's animal and recording day within its animal, indexed as
        ``values``.

    Returns
    -------
    p : float
        NaN when fewer than :data:`MIN_SIGNED_RANK_DIFFERENCES` day values are
        non-zero.

    Raises
    ------
    ValueError
        If the labels are not indexed as ``values`` or have missing values.
    """
    values, (animal, day) = _finite_rows(values, {"animal": animal, "day": day})
    day_values, _ = _day_and_animal_means(
        values.to_numpy(dtype=float),
        *_animal_day_codes(animal.to_numpy(), day.to_numpy()),
    )
    return signed_rank_p(_snapped(day_values))


def estimand_summary(
    values: pd.Series,
    animal: pd.Series,
    day: pd.Series,
    session: pd.Series,
    *,
    n_draws: int,
    seed: int,
) -> dict[str, float]:
    """The summary of a scalar estimand over sessions.

    The estimate is the animal-balanced value: session -> recording day ->
    animal, a mean of means, with its nested bootstrap draws (animals, then
    their sessions), as :func:`animal_balanced_bootstrap` gives them. Its 95 %
    interval is :func:`bootstrap_interval` over those draws. Beside it stand
    the animals whose own value (:func:`animal_day_session_values`) has the
    estimate's sign and the recording-day-level test (:func:`day_level_test`),
    both read from the same day and animal means. No claim rests on a p over
    five animals.

    Parameters
    ----------
    values : pd.Series, shape (n_sessions,)
        One estimate per session (for a kind difference, the difference of the
        two kinds' session estimates). A value that is not finite (missing or
        infinite) is left out.
    animal, day, session : pd.Series, shape (n_sessions,)
        Each session's animal, recording day within its animal and session
        within its day, indexed as ``values``; one row per session.
    n_draws : int
        Bootstrap replicates.
    seed : int
        Their seed.

    Returns
    -------
    summary : dict
        - ``estimate``: the animal-balanced value.
        - ``low``, ``high``, ``sd``: :func:`bootstrap_interval`'s.
        - ``n_animals``, ``n_days``, ``n_sessions``: those with a finite
          value; every entry is computed on those sessions only.
        - ``n_animals_same_sign``: animals whose own value has the sign of
          ``estimate``; an animal at zero has no sign (:func:`_snapped`).
        - ``day_p``: the day-level signed-rank p (:func:`day_level_test`).
        - ``n_draws``, ``seed``: the bootstrap's.

    Raises
    ------
    ValueError
        If no session has a finite value, a session has more than one value,
        or the labels are not indexed as ``values`` or have missing values.
    """
    values, (animal, day, session) = _finite_rows(
        values, {"animal": animal, "day": day, "session": session}
    )
    if values.empty:
        raise ValueError("no session has a finite value")
    if pd.MultiIndex.from_arrays([animal, day, session]).duplicated().any():
        raise ValueError("a session has more than one value")
    session_values = values.to_numpy(dtype=float)
    animal_codes, day_codes = _animal_day_codes(animal.to_numpy(), day.to_numpy())
    estimate, draws = _balanced_estimate_and_draws(
        session_values[:, None], animal_codes, day_codes, n_draws, seed
    )
    day_values, animal_values = _day_and_animal_means(
        session_values, animal_codes, day_codes
    )
    estimate = float(estimate[0, 0])
    interval = bootstrap_interval(estimate, draws[:, 0], animal_values.size)
    animal_signs = np.sign(_snapped(animal_values))
    return {
        "estimate": estimate,
        "low": interval.low,
        "high": interval.high,
        "sd": interval.sd,
        "n_animals": animal_values.size,
        "n_days": day_values.size,
        "n_sessions": session_values.size,
        "n_animals_same_sign": int(
            ((animal_signs != 0) & (animal_signs == np.sign(estimate))).sum()
        ),
        "day_p": signed_rank_p(_snapped(day_values)),
        "n_draws": n_draws,
        "seed": seed,
    }


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
