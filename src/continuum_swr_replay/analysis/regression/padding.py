"""Rows padded to a few bucket sizes, so that a jitted kernel compiles once per
bucket rather than once per session.

A padded row has a design row of zeros, a response of 0, an offset of 0
(exposure 1) and a weight of 0; the real rows come first and keep their order.
A kernel multiplies each row's term of the log-likelihood, its gradient and its
information by the row's weight, so a padded row adds exactly 0. Its terms are
finite (``eta = 0``, ``mu = exp(0)``): an offset of minus infinity would give
``0 * -inf = nan``.

For the grouped softmax every padded row goes to one extra group after the real
ones, and the number of groups is padded too; the extra groups beyond it have
no rows. A group with a response total of 0 adds 0, provided a kernel guards
its ``n_g log(sum exp)`` term, which is ``0 * -inf`` for a group with no rows.

Every quantity that depends on the number of rows (a degrees-of-freedom count,
a mean, a dispersion) must use ``n_real``, not the padded count.

Everything here is NumPy; nothing imports JAX. The likelihoods
(``likelihoods.Likelihood.pad``) pad with these functions before they call a
kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GROUP_BUCKET",
    "PaddedGroups",
    "PaddedRows",
    "ROW_BUCKET",
    "SOFTMAX_ROW_BUCKET",
    "bucket_size",
    "group_starts",
    "pad_groups",
    "pad_rows",
    "pad_values",
]

ROW_BUCKET = 4096
"""Rows of a Poisson, Bernoulli or Gaussian fit are padded to a multiple of
this (at most 11 % more rows on a 37,000-row session)."""

SOFTMAX_ROW_BUCKET = 1024
"""Rows of a grouped-softmax fit are padded to a multiple of this."""

GROUP_BUCKET = 64
"""Groups of a grouped-softmax fit, the padding group included, are padded to
a multiple of this."""


def bucket_size(n: int, step: int) -> int:
    """The smallest multiple of `step` that is at least `n`.

    Raises
    ------
    ValueError
        When `n` is below 1 or `step` is below 1.
    """
    if n < 1 or step < 1:
        raise ValueError(f"need n >= 1 and step >= 1, not n={n}, step={step}")
    return -(-n // step) * step


def group_starts(groups: np.ndarray) -> np.ndarray:
    """Index of each group's first row, shape ``(n_groups,)``.

    Raises
    ------
    ValueError
        If ``groups`` is empty or not sorted (a group's rows must be
        contiguous).
    """
    groups = np.asarray(groups)
    if groups.size == 0:
        raise ValueError("no rows")
    if np.any(groups[1:] < groups[:-1]):
        raise ValueError("group ids must be sorted so each group's rows are contiguous")
    return np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])


def pad_values(values: np.ndarray, n_rows: int) -> np.ndarray:
    """`values` with zero rows appended to `n_rows` rows, as float64.

    Parameters
    ----------
    values : np.ndarray, shape (n, ...)
        One entry (or row) per real row, ``n <= n_rows``.
    n_rows : int
        The padded count.

    Returns
    -------
    np.ndarray, shape (n_rows, ...)
    """
    values = np.asarray(values, dtype=float)
    out = np.zeros((n_rows, *values.shape[1:]))
    out[: values.shape[0]] = values
    return out


@dataclass(frozen=True)
class PaddedRows:
    """A session's rows padded to a bucket.

    Attributes
    ----------
    X : np.ndarray, shape (n_bucket, n_columns)
    y : np.ndarray, shape (n_bucket,)
    offset : np.ndarray, shape (n_bucket,)
        The Poisson offset, 0 on padded rows; zeros when none was given.
    weight : np.ndarray, shape (n_bucket,)
        1 on a real row, 0 on a padded one.
    n_real : int
        The real rows, which are the first `n_real`.
    """

    X: np.ndarray
    y: np.ndarray
    offset: np.ndarray
    weight: np.ndarray
    n_real: int


def pad_rows(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray | None = None,
    step: int = ROW_BUCKET,
) -> PaddedRows:
    """Pad a session's rows to the next multiple of `step`.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, n_columns)
    y : np.ndarray, shape (n_rows,)
    offset : np.ndarray, shape (n_rows,), optional
        The Poisson offset, log exposure.
    step : int, optional

    Returns
    -------
    PaddedRows

    Raises
    ------
    ValueError
        When the inputs do not have one row each, or there are no rows.
    """
    n_rows = np.shape(y)[0]
    offset = np.zeros(n_rows) if offset is None else offset
    if np.shape(X)[0] != n_rows or np.shape(offset) != (n_rows,):
        raise ValueError(
            f"X {np.shape(X)}, y {np.shape(y)} and offset {np.shape(offset)} "
            "need one row each"
        )
    n_bucket = bucket_size(n_rows, step)
    return PaddedRows(
        X=pad_values(X, n_bucket),
        y=pad_values(y, n_bucket),
        offset=pad_values(offset, n_bucket),
        weight=pad_values(np.ones(n_rows), n_bucket),
        n_real=n_rows,
    )


@dataclass(frozen=True)
class PaddedGroups:
    """A grouped-softmax session's rows and groups padded to buckets.

    Attributes
    ----------
    X : np.ndarray, shape (n_bucket, n_columns)
    y : np.ndarray, shape (n_bucket,)
    group : np.ndarray of int, shape (n_bucket,)
        Each row's group, numbered 0 to ``n_real_groups - 1`` in the order of
        the input; every padded row is in group ``n_real_groups``. Sorted.
    weight : np.ndarray, shape (n_bucket,)
        1 on a real row, 0 on a padded one.
    n_real : int
        The real rows, which are the first `n_real`.
    n_groups : int
        The groups a kernel sums over (``num_segments``): the real ones, the
        padding group and the empty groups after it.
    n_real_groups : int
    """

    X: np.ndarray
    y: np.ndarray
    group: np.ndarray
    weight: np.ndarray
    n_real: int
    n_groups: int
    n_real_groups: int


def pad_groups(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    row_step: int = SOFTMAX_ROW_BUCKET,
    group_step: int = GROUP_BUCKET,
) -> PaddedGroups:
    """Pad a grouped-softmax session's rows and groups to buckets.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, n_columns)
    y : np.ndarray, shape (n_rows,)
    groups : np.ndarray, shape (n_rows,)
        Each row's group id, sorted so that a group's rows are contiguous (the
        ``aux`` of ``likelihoods.GroupedSoftmax``).
    row_step, group_step : int, optional

    Returns
    -------
    PaddedGroups

    Raises
    ------
    ValueError
        When the inputs do not have one row each, there are no rows, or the
        group ids are not sorted.
    """
    padded = pad_rows(X, y, step=row_step)
    if np.shape(groups) != (padded.n_real,):
        raise ValueError(
            f"group ids of shape {np.shape(groups)} for {padded.n_real} rows"
        )
    starts = group_starts(groups)
    n_real_groups = starts.size
    group = np.full(padded.y.size, n_real_groups)
    group[: padded.n_real] = np.repeat(
        np.arange(n_real_groups), np.diff(starts, append=padded.n_real)
    )
    return PaddedGroups(
        X=padded.X,
        y=padded.y,
        group=group,
        weight=padded.weight,
        n_real=padded.n_real,
        n_groups=bucket_size(n_real_groups + 1, group_step),
        n_real_groups=n_real_groups,
    )
