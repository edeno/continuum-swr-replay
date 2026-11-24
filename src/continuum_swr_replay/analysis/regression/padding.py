"""Rows padded to a few bucket sizes, so that a jitted kernel compiles once per
bucket rather than once per session.

A bucket (:func:`bucket_size`) is the next power of two, at least
``SMALLEST_ROW_BUCKET``, below a step, and the next multiple of the step from
the step up: rows of the Poisson, Bernoulli and Gaussian families go to 256,
512, 1,024, 2,048, then multiples of 4,096 (``ROW_BUCKET``); rows of the
grouped softmax to 256, 512, then multiples of 1,024
(``SOFTMAX_ROW_BUCKET``); its groups to multiples of 64 (``GROUP_BUCKET``).

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
kernel: :func:`pad_rows` and :func:`pad_groups` return the :class:`Rows` the
kernels take, and a family only sets its ``aux``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "GROUP_BUCKET",
    "Groups",
    "ROW_BUCKET",
    "Rows",
    "SMALLEST_ROW_BUCKET",
    "SOFTMAX_ROW_BUCKET",
    "bucket_size",
    "group_sizes",
    "group_starts",
    "pad_groups",
    "pad_rows",
    "pad_values",
]

ROW_BUCKET = 4096
"""Rows of a Poisson, Bernoulli or Gaussian fit from this many up are padded to
a multiple of this (at most 11 % more rows on a 37,000-row session)."""

SOFTMAX_ROW_BUCKET = 1024
"""Rows of a grouped-softmax fit from this many up are padded to a multiple of
this."""

SMALLEST_ROW_BUCKET = 256
"""Fewer rows than a row step are padded to the next power of two, and to at
least this. A Bernoulli fit of 40 rows took the same time padded to 64, 128 or
256 rows, so a smaller bucket would add compiled shapes and gain nothing."""

GROUP_BUCKET = 64
"""Groups of a grouped-softmax fit, the padding group included, are padded to
a multiple of this."""


def bucket_size(n: int, step: int, smallest: int) -> int:
    """The padded size of `n`: below `step`, the next power of two that is at
    least `n` and `smallest` (and at most `step`); from `step` up, the next
    multiple of `step`.

    Raises
    ------
    ValueError
        When `n`, `step` or `smallest` is below 1.
    """
    if n < 1 or step < 1 or smallest < 1:
        raise ValueError(
            f"need n, step and smallest >= 1, not n={n}, step={step}, "
            f"smallest={smallest}"
        )
    if n < step:
        return min(step, max(smallest, 1 << (n - 1).bit_length()))
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


def group_sizes(starts: np.ndarray, n_rows: int) -> np.ndarray:
    """Each group's number of rows, shape ``(n_groups,)``, from the index of
    its first row (:func:`group_starts`) and the number of rows."""
    return np.diff(starts, append=n_rows)


def pad_values(
    values: np.ndarray, n_real: int, n_rows: int, name: str = "values"
) -> np.ndarray:
    """`values` with zero rows appended to `n_rows` rows, as float64.

    Parameters
    ----------
    values : np.ndarray, shape (n_real, ...)
        One entry (or row) per real row.
    n_real : int
        The real rows, which `values` must have.
    n_rows : int
        The padded count, at least `n_real`.
    name : str, optional
        What `values` is, for the error.

    Returns
    -------
    np.ndarray, shape (n_rows, ...)

    Raises
    ------
    ValueError
        When `values` does not have `n_real` rows.
    """
    if np.ndim(values) == 0 or np.shape(values)[0] != n_real:
        raise ValueError(
            f"{name} of shape {np.shape(values)} for {n_real} rows: it needs one "
            "value per row"
        )
    values = np.asarray(values, dtype=float)
    out = np.zeros((n_rows, *values.shape[1:]))
    out[: values.shape[0]] = values
    return out


class Rows(NamedTuple):
    """A session's rows padded for a kernel (:func:`pad_rows`,
    :func:`pad_groups`, ``Likelihood.pad``).

    Attributes
    ----------
    X : np.ndarray, shape (n_bucket, p)
    y : np.ndarray, shape (n_bucket,)
    aux : object
        The family's extra data as its kernels take it: the padded offset, the
        scale, :class:`Groups`, or None.
    weight : np.ndarray, shape (n_bucket,)
        1 on a real row, 0 on a padded one.
    n_real : int
        The real rows, which are the first `n_real`.
    """

    X: np.ndarray
    y: np.ndarray
    aux: Any
    weight: np.ndarray
    n_real: int


@dataclass(frozen=True)
class Groups:
    """The grouped softmax's ``aux`` in a kernel (``likelihoods`` registers
    it as a JAX pytree: ``index`` its data, ``n_groups`` static).

    Attributes
    ----------
    index : array of int, shape (n_bucket,)
        Each row's group, numbered 0 to ``n_real_groups - 1`` in the order of
        the input; every padded row is in group ``n_real_groups``. Sorted.
    n_groups : int
        The groups a kernel sums over (``num_segments``): the real ones, the
        padding group and the empty groups after it; static, since
        ``segment_sum`` needs it when traced.
    """

    index: Any
    n_groups: int


def pad_rows(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray | None = None,
    step: int = ROW_BUCKET,
    smallest: int = SMALLEST_ROW_BUCKET,
) -> Rows:
    """Pad a session's rows to their bucket (:func:`bucket_size`).

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, n_columns)
    y : np.ndarray, shape (n_rows,)
    offset : np.ndarray, shape (n_rows,), optional
        The Poisson offset, log exposure.
    step, smallest : int, optional
        The bucket's step and its smallest size.

    Returns
    -------
    Rows
        ``aux`` the offset padded with 0, or None when none was given.

    Raises
    ------
    ValueError
        When the inputs do not have one row each. No rows give one bucket
        of padded rows.
    """
    n_rows = np.shape(y)[0]
    offset_shape = (n_rows,) if offset is None else np.shape(offset)
    if np.shape(X)[0] != n_rows or offset_shape != (n_rows,):
        raise ValueError(
            f"X {np.shape(X)}, y {np.shape(y)} and offset {offset_shape} "
            "need one row each"
        )
    # No rows pad to one bucket of padded rows, whose every sum is 0.
    n_bucket = bucket_size(max(n_rows, 1), step, smallest)
    return Rows(
        X=pad_values(X, n_rows, n_bucket, "X"),
        y=pad_values(y, n_rows, n_bucket, "y"),
        aux=None if offset is None else pad_values(offset, n_rows, n_bucket, "offset"),
        weight=pad_values(np.ones(n_rows), n_rows, n_bucket, "weight"),
        n_real=n_rows,
    )


def pad_groups(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    row_step: int = SOFTMAX_ROW_BUCKET,
    group_step: int = GROUP_BUCKET,
) -> Rows:
    """Pad a grouped-softmax session's rows and groups to buckets.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, n_columns)
    y : np.ndarray, shape (n_rows,)
    groups : np.ndarray, shape (n_rows,)
        Each row's group id, sorted so that a group's rows are contiguous (the
        ``aux`` of ``likelihoods.GroupedSoftmax``).
    row_step, group_step : int, optional
        The steps of the rows' bucket (whose smallest is ``SMALLEST_ROW_BUCKET``)
        and of the groups' (a multiple of `group_step`, whatever the count).

    Returns
    -------
    Rows
        ``aux`` the rows' :class:`Groups`.

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
        np.arange(n_real_groups), group_sizes(starts, padded.n_real)
    )
    n_groups = bucket_size(n_real_groups + 1, group_step, group_step)
    return padded._replace(aux=Groups(group, n_groups))
