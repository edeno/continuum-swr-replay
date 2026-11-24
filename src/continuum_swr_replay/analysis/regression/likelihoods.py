"""Log-likelihoods of the shared fitter and their derivatives on the eta scale.

Each family is a small stateless class. ``eta`` is the linear predictor
``X @ beta``, shape ``(n_rows,)``; ``aux`` is the family's extra data (below),
passed to every method so that one object serves every session and, for the
Gaussian, every scale. Two families are equal when they are of the same class,
so that a family can be a static argument of a jitted kernel: the kernels of
one class are compiled once and shared by its objects. A family therefore
keeps no parameter on the object; anything that changes its result travels in
``aux``. A family object with an attribute of its own is refused
(``TypeError``) when a kernel would key on it.

The methods that take NumPy arrays:

- ``check(y, aux)``: raise ``ValueError``, naming the problem, when the
  response or ``aux`` is outside what the family takes (the fit calls it once,
  at its entry, after checking that ``y`` is finite).
- ``identifying_design(X, aux)``: the design whose column space the likelihood
  identifies (the design check reads it).
- ``pad(X, y, aux)``: the rows padded to their bucket (``padding``), as the
  kernels take them (:class:`Rows`).
- ``weighted_design(X, y, aux, eta)``: the design weighted by the information,
  ``X~`` with ``X~' X~`` the information, shape ``(n_rows, p)`` (what
  ``collinearity`` measures on). It runs in float64
  (``runtime.in_float64``): it pads the rows and runs one jitted kernel.

The Newton and REML kernels compute from the family's JAX functions:
``_jax_loglik`` (the log-likelihood), ``_jax_score`` (``dl/deta``, shape
``(n_rows,)``; the coefficient gradient is ``X.T @ score``),
``_jax_information`` (``-d2l/dbeta2``, shape ``(p, p)``),
``_jax_dinformation`` (its derivative when ``eta`` moves along a direction
``d``, shape ``(n_rows,)``), ``_jax_d2information`` (the second derivative
along ``d1`` and ``d2``), ``_jax_weighted_design`` and
``_jax_information_traces`` (the traces of the information's derivatives
against a matrix, which REML's derivatives read). Each takes the padded rows
and their ``weight`` (1 on a real row, 0 on a padded one) and multiplies every
row's term by it, so a padded row adds exactly 0. A likelihood made for a test
follows the protocol by overriding these.

The information is the observed information. For the three diagonal families it
is ``X' diag(w) X`` with weights ``w(eta)`` that do not depend on ``y``; their
directional derivatives are ``X' diag(w' d) X`` and ``X' diag(w'' d1 d2) X``.

=================  =====================================  ============  ==========  ==========
family             aux                                    w             w'          w''
=================  =====================================  ============  ==========  ==========
Poisson, log link  offset = log(exposure), (n_rows,)      mu            mu          mu
Bernoulli, logit   None                                   mu (1 - mu)   w (1-2 mu)  w (1-6 w)
Gaussian           phi, the scale (float)                 1 / phi       0           0
=================  =====================================  ============  ==========  ==========

The grouped softmax (one group per event, one row per candidate) takes the
group id of each row as ``aux``, sorted so that a group's rows are contiguous;
its kernels sum over groups by ``jax.ops.segment_sum``.
Constants are left out where they do not depend on ``beta``: the Poisson
``-log y!`` and the multinomial coefficient. The Gaussian keeps its
normalizing constant, which depends on the scale (and on the number of real
rows, not the padded count).

A family's ``name`` is how a fit record names it. Its ``quadratic`` is True
when its log-likelihood is quadratic in ``eta`` (the Gaussian): one full Newton
step from any start is then the exact optimum, and the fit takes it without a
cap on its length.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from .padding import group_sizes, group_starts, pad_groups, pad_rows, pad_values
from .runtime import float64_kernel, in_float64

__all__ = [
    "Bernoulli",
    "Gaussian",
    "GroupedSoftmax",
    "Groups",
    "Likelihood",
    "Poisson",
    "Rows",
    "draw",
    "group_starts",
]


class Rows(NamedTuple):
    """A session's rows padded for a kernel (``Likelihood.pad``).

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
    """The grouped softmax's ``aux`` in a kernel.

    Attributes
    ----------
    index : array of int, shape (n_bucket,)
        Each row's group, 0 to ``n_groups - 1``, sorted; the padded rows are in
        one group after the real ones.
    n_groups : int
        The groups summed over, padded (``padding.pad_groups``); static, since
        ``segment_sum`` needs it when traced.
    """

    index: Any
    n_groups: int


jax.tree_util.register_dataclass(
    Groups, data_fields=["index"], meta_fields=["n_groups"]
)


class Likelihood(Protocol):
    """The interface the Newton fit, REML and the collinearity measures read."""

    #: The family's name in a fit record.
    name: str
    quadratic: bool

    def check(self, y: np.ndarray, aux: object) -> None: ...

    def identifying_design(self, X: np.ndarray, aux: object) -> np.ndarray: ...

    def pad(self, X: np.ndarray, y: np.ndarray, aux: object) -> Rows: ...

    def weighted_design(
        self, X: np.ndarray, y: np.ndarray, aux: object, eta: np.ndarray
    ) -> np.ndarray: ...

    def _jax_loglik(self, eta: Any, y: Any, aux: Any, weight: Any) -> Any: ...

    def _jax_score(self, eta: Any, y: Any, aux: Any, weight: Any) -> Any: ...

    def _jax_information(
        self, X: Any, eta: Any, y: Any, aux: Any, weight: Any
    ) -> Any: ...

    def _jax_dinformation(
        self, X: Any, eta: Any, y: Any, aux: Any, weight: Any, d: Any
    ) -> Any: ...

    def _jax_d2information(
        self, X: Any, eta: Any, y: Any, aux: Any, weight: Any, d1: Any, d2: Any
    ) -> Any: ...

    def _jax_weighted_design(
        self, X: Any, eta: Any, y: Any, aux: Any, weight: Any
    ) -> Any: ...

    def _jax_information_traces(
        self, X: Any, eta: Any, y: Any, aux: Any, weight: Any, M: Any, D: Any, E: Any
    ) -> tuple[Any, Any]: ...


# The kernel of the public method: one program per family and padded shape.


@float64_kernel(static_argnames=("lik",))
def _weighted_design_kernel(lik, X, eta, y, aux, weight):
    return lik._jax_weighted_design(X, eta, y, aux, weight)


def _one_axis(values: object, name: str) -> np.ndarray:
    """`values` as float64, refused (``ValueError``) unless it has one axis: a
    column of shape ``(n, 1)`` would broadcast against the padded rows."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"{name} of shape {values.shape}: one value per row, on one axis"
        )
    return values


def _gram(X: Any, weights: Any) -> Any:
    """``X' diag(weights) X``, shape ``(p, p)``, for weights of either sign.

    Written ``X.T @ (weights[:, None] * X)``: XLA folds that transpose into
    the product. ``(X.T * weights) @ X`` made it copy the transposed design,
    which took about half of a Newton iteration at the rate model's size.
    """
    return X.T @ (weights[:, None] * X)


def _upper_pairs(m: int) -> tuple[Any, Any, Any]:
    """The pairs ``j <= k`` of ``m`` items, as two index arrays, and whether
    each is a diagonal pair."""
    j, k = jnp.triu_indices(m)
    return j, k, j == k


def _mirrored(upper: Any, j: Any, k: Any, m: int) -> Any:
    """The symmetric ``(m, m)`` matrix whose entries ``j <= k`` are ``upper``."""
    return jnp.zeros((m, m), upper.dtype).at[j, k].set(upper).at[k, j].set(upper)


@functools.cache
def _can_keep_state(cls: type) -> bool:
    """Whether a family class defines ``__init__`` or a
    ``functools.cached_property`` (itself or a base), either of which can put
    an attribute on an object after it is made."""
    return cls.__init__ is not object.__init__ or any(
        isinstance(value, functools.cached_property)
        for klass in cls.__mro__
        for value in vars(klass).values()
    )


class _Family:
    """The public method of every family, on its JAX functions."""

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self._stateless()

    def __hash__(self) -> int:
        self._stateless()
        return hash(type(self))

    def _stateless(self) -> bool:
        """True; raise ``TypeError`` when the object keeps an attribute of its
        own, which the kernels, keyed on the class, would not see, or when its
        class could put one there (:func:`_can_keep_state`). Checked at every
        hash and every public call."""
        if _can_keep_state(type(self)):
            raise TypeError(
                f"{type(self).__name__} can keep state on its objects (it defines "
                "__init__ or a cached property): a family is keyed on its class in "
                "the compiled kernels, so whatever changes its result goes in aux"
            )
        attributes = sorted(vars(self))
        if attributes:
            raise TypeError(
                f"{type(self).__name__} keeps {attributes} on the object: a family "
                "is keyed on its class in the compiled kernels, so whatever changes "
                "its result goes in aux"
            )
        return True

    @in_float64
    def weighted_design(
        self, X: np.ndarray, y: np.ndarray, aux: object, eta: np.ndarray
    ) -> np.ndarray:
        """The design weighted by the information at ``eta``.

        Parameters
        ----------
        X : np.ndarray, shape (n_rows, p)
        y : np.ndarray, shape (n_rows,)
            The response (the softmax's group totals weight its groups).
        aux : object
            The family's aux; the Gaussian ignores it.
        eta : np.ndarray, shape (n_rows,)
            ``X @ beta`` at the fit.

        Returns
        -------
        np.ndarray, shape (n_rows, p)
            ``X~``, with ``X~' X~`` the information (the Gaussian's at unit
            scale, so ``X~ = X``).
        """
        self._stateless()
        r = self.pad(X, _one_axis(y, "y"), aux)
        eta = pad_values(_one_axis(eta, "eta"), r.n_real, r.y.size, "eta")
        weighted = _weighted_design_kernel(self, r.X, eta, r.y, r.aux, r.weight)
        return np.array(weighted)[: r.n_real]

    def _jax_information_traces(self, X, eta, y, aux, weight, M, D, E):
        """The traces of the information's derivatives against ``M``.

        ``M`` is a symmetric ``(p, p)`` matrix, ``D`` has shape ``(n_rows, k)``
        and ``E`` shape ``(n_rows, m)``, each column a direction of ``eta``.
        Returns ``tr(M _jax_dinformation(D[:, a]))`` for each column ``a``,
        shape ``(k,)``, and ``tr(M _jax_d2information(E[:, j], E[:, l]))`` for
        each pair,
        shape ``(m, m)``, symmetric. Here each derivative is made and
        multiplied by ``M``, one at a time (``jax.lax.map``): one
        ``(p, p)`` product with the design per column and per pair ``j <= l``.
        """

        def against_M(information):
            return jnp.sum(M * information)

        def first(a):
            d = jax.lax.dynamic_index_in_dim(D, a, axis=1, keepdims=False)
            return against_M(self._jax_dinformation(X, eta, y, aux, weight, d))

        m = E.shape[1]
        rows, columns, _ = _upper_pairs(m)

        def second(pair):
            j, k = pair
            e_j = jax.lax.dynamic_index_in_dim(E, j, axis=1, keepdims=False)
            e_k = jax.lax.dynamic_index_in_dim(E, k, axis=1, keepdims=False)
            return against_M(self._jax_d2information(X, eta, y, aux, weight, e_j, e_k))

        upper = jax.lax.map(second, (rows, columns))
        return (
            jax.lax.map(first, jnp.arange(D.shape[1])),
            _mirrored(upper, rows, columns, m),
        )


class _DiagonalFamily(_Family):
    """A family whose information is ``X' diag(w(eta)) X``."""

    def identifying_design(self, X: np.ndarray, aux: object) -> np.ndarray:
        """The design whose column space the likelihood identifies: ``X``."""
        return X

    def _jax_weights(self, eta: Any, aux: Any) -> tuple[Any, Any, Any]:
        """``(w, dw/deta, d2w/deta2)``, each shape ``(n_rows,)``."""
        raise NotImplementedError

    def _jax_information(self, X, eta, y, aux, weight):
        return _gram(X, weight * self._jax_weights(eta, aux)[0])

    def _jax_dinformation(self, X, eta, y, aux, weight, d):
        return _gram(X, weight * self._jax_weights(eta, aux)[1] * d)

    def _jax_d2information(self, X, eta, y, aux, weight, d1, d2):
        return _gram(X, weight * self._jax_weights(eta, aux)[2] * d1 * d2)

    def _jax_weighted_design(self, X, eta, y, aux, weight):
        return X * jnp.sqrt(weight * self._jax_weights(eta, aux)[0])[:, None]

    def _jax_information_traces(self, X, eta, y, aux, weight, M, D, E):
        """The traces of :meth:`_Family._jax_information_traces`, without
        making the derivatives: with ``h_i = x_i' M x_i``,
        ``tr(M X' diag(c) X) = sum_i c_i h_i``, so the traces are
        ``(weight w' h) @ D`` and ``E' diag(weight w'' h) E``. One product of
        the design with ``M`` in all; read from ``_jax_weights``, as the
        directional derivatives are. The second is mirrored from its upper
        triangle, so that it is exactly symmetric."""
        _, w1, w2 = self._jax_weights(eta, aux)
        h = jnp.sum((X @ M) * X, axis=1)
        second = _gram(E, weight * w2 * h)
        return (weight * w1 * h) @ D, jnp.triu(second) + jnp.triu(second, 1).T


class Poisson(_DiagonalFamily):
    """Poisson counts with a log link and an offset ``aux = log(exposure)``.

    ``l = sum y (eta + o) - exp(eta + o)``.
    """

    name = "poisson"
    quadratic = False

    def check(self, y: np.ndarray, aux: np.ndarray) -> None:
        """Counts at least zero; an offset per row, finite (a row of zero
        exposure, offset minus infinity, carries no likelihood: drop it)."""
        if np.any(y < 0):
            raise ValueError(f"Poisson response below zero on {np.sum(y < 0)} rows")
        offset = np.asarray(aux, dtype=float)
        if offset.shape != y.shape:
            raise ValueError(f"offset of shape {offset.shape} for {y.size} rows")
        if np.any(offset == -np.inf):
            raise ValueError(
                f"zero exposure (offset -inf) on {np.sum(offset == -np.inf)} rows: "
                "drop those rows, which carry no likelihood"
            )
        if not np.all(np.isfinite(offset)):
            raise ValueError(
                f"offset not finite on {np.sum(~np.isfinite(offset))} rows"
            )

    def pad(self, X: np.ndarray, y: np.ndarray, aux: np.ndarray) -> Rows:
        """The rows padded to their bucket (``padding.pad_rows``), the offset
        0 on a padded row; a single offset is every row's."""
        if aux is None:
            raise TypeError("the Poisson offset is None: give log(exposure)")
        offset = np.asarray(aux, dtype=float)
        if offset.ndim == 0:  # one offset for every row
            offset = np.full(np.shape(y), float(offset))
        padded = pad_rows(X, y, _one_axis(offset, "offset"))
        return Rows(padded.X, padded.y, padded.offset, padded.weight, padded.n_real)

    def _jax_loglik(self, eta, y, aux, weight):
        log_mu = eta + aux
        return jnp.sum(weight * (y * log_mu - jnp.exp(log_mu)))

    def _jax_score(self, eta, y, aux, weight):
        return weight * (y - jnp.exp(eta + aux))

    def _jax_weights(self, eta, aux):
        mu = jnp.exp(eta + aux)
        return mu, mu, mu


class Bernoulli(_DiagonalFamily):
    """0/1 responses with a logit link; ``aux`` is unused.

    ``l = sum y eta - log(1 + exp(eta))``.
    """

    name = "bernoulli"
    quadratic = False

    def check(self, y: np.ndarray, aux: object) -> None:
        """0/1 outcomes only; the family does not take proportions."""
        outside = (y != 0) & (y != 1)
        if np.any(outside):
            raise ValueError(
                f"Bernoulli response outside {{0, 1}} on {np.sum(outside)} rows "
                "(it takes 0/1 outcomes, not proportions)"
            )

    def pad(self, X: np.ndarray, y: np.ndarray, aux: object) -> Rows:
        """The rows padded to their bucket (``padding.pad_rows``); no aux."""
        padded = pad_rows(X, y)
        return Rows(padded.X, padded.y, None, padded.weight, padded.n_real)

    def _jax_loglik(self, eta, y, aux, weight):
        return jnp.sum(weight * (y * eta - jnp.logaddexp(0.0, eta)))

    def _jax_score(self, eta, y, aux, weight):
        return weight * (y - jax.nn.sigmoid(eta))

    def _jax_weights(self, eta, aux):
        mu = jax.nn.sigmoid(eta)
        w = mu * (1.0 - mu)
        return w, w * (1.0 - 2.0 * mu), w * (1.0 - 6.0 * w)


class Gaussian(_DiagonalFamily):
    """Normal responses with an identity link and scale ``aux = phi``.

    ``l = -sum (y - eta)^2 / (2 phi) - n/2 log(2 pi phi)``, ``n`` the real rows.
    """

    name = "gaussian"
    quadratic = True

    def check(self, y: np.ndarray, aux: float) -> None:
        """A finite, positive scale."""
        if not (np.ndim(aux) == 0 and np.isfinite(aux) and aux > 0):
            raise ValueError(
                f"the Gaussian scale must be finite and positive, not {aux}"
            )

    def pad(self, X: np.ndarray, y: np.ndarray, aux: float | None) -> Rows:
        """The rows padded to their bucket (``padding.pad_rows``); the scale
        as a float64 (None stays None: the weighted design does not read it)."""
        padded = pad_rows(X, y)
        scale = None if aux is None else np.float64(aux)
        return Rows(padded.X, padded.y, scale, padded.weight, padded.n_real)

    def _jax_loglik(self, eta, y, aux, weight):
        residual = y - eta
        return -((weight * residual) @ residual) / (2.0 * aux) - 0.5 * jnp.sum(
            weight
        ) * jnp.log(2.0 * jnp.pi * aux)

    def _jax_score(self, eta, y, aux, weight):
        return weight * (y - eta) / aux

    def _jax_weights(self, eta, aux):
        zero = jnp.zeros_like(eta)
        return jnp.full_like(eta, 1.0 / aux), zero, zero

    def _jax_weighted_design(self, X, eta, y, aux, weight):
        return X * weight[:, None]


class GroupedSoftmax(_Family):
    """A softmax over each group's rows; ``aux`` is the sorted group id per row.

    For group ``g`` with response ``y_gk`` (a fraction summing to one, or a
    count), ``n_g = sum_k y_gk`` and ``p_g = softmax(eta_g)``::

        l = sum_g sum_k y_gk log p_gk
        dl/deta_gk = y_gk - n_g p_gk
        information = sum_g n_g X_g' G_g X_g,   G = diag(p) - p p'

    Within a group, with ``v = G d = p (d - p'd)``::

        D_d G      = diag(v) - v p' - p v'
        D_d2 D_d1 G = diag(q) - q p' - p q' - v1 v2' - v2 v1',   q = (D_d2 G) d1

    It is the conditional Poisson likelihood of a fixed group total. A constant
    within a group cancels, so a group-level column is not identified and must
    not be in ``X``. In a kernel the per-group sums are segment sums over
    :class:`Groups`; a group with no rows (``n_g = 0``, its ``log(sum exp)``
    minus infinity) is read by no row and adds 0.
    """

    name = "grouped_softmax"
    quadratic = False

    def check(self, y: np.ndarray, aux: np.ndarray) -> None:
        """Finite, sorted group ids, one per row; shares (or counts) at least
        zero; every group with a positive total."""
        groups = np.asarray(aux)
        if groups.shape != y.shape:
            raise ValueError(f"group ids of shape {groups.shape} for {y.size} rows")
        if groups.dtype.kind == "f" and not np.all(np.isfinite(groups)):
            raise ValueError(
                f"group ids not finite on {np.sum(~np.isfinite(groups))} rows"
            )
        if np.any(y < 0):
            raise ValueError(f"softmax shares negative on {np.sum(y < 0)} rows")
        empty = np.add.reduceat(y, group_starts(groups)) <= 0
        if np.any(empty):
            raise ValueError(
                f"{np.sum(empty)} group(s) whose responses sum to zero: drop them, "
                "they carry no likelihood"
            )

    def identifying_design(self, X: np.ndarray, aux: np.ndarray) -> np.ndarray:
        """``X`` less each group's column means, shape ``(n_rows, p)``: the
        likelihood sees only variation within a group, so a combination of
        columns constant within every group is not identified."""
        starts = group_starts(aux)
        sizes = group_sizes(starts, X.shape[0])
        means = np.add.reduceat(X, starts, axis=0) / sizes[:, None]
        return X - np.repeat(means, sizes, axis=0)

    def pad(self, X: np.ndarray, y: np.ndarray, aux: np.ndarray) -> Rows:
        """The rows and groups padded (``padding.pad_groups``)."""
        padded = pad_groups(X, y, aux)
        groups = Groups(padded.group, padded.n_groups)
        return Rows(padded.X, padded.y, groups, padded.weight, padded.n_real)

    @staticmethod
    def _per_group(values: Any, groups: Groups) -> Any:
        """Sums over each group's rows, shape ``(n_groups, ...)``."""
        return jax.ops.segment_sum(
            values, groups.index, num_segments=groups.n_groups, indices_are_sorted=True
        )

    def _jax_parts(self, eta, y, groups, weight):
        """``(shifted, norm, p, total)``: ``eta`` less its group's largest,
        ``sum exp(shifted)`` and ``n_g`` per group, ``p`` per row."""
        top = jax.ops.segment_max(
            eta, groups.index, num_segments=groups.n_groups, indices_are_sorted=True
        )
        shifted = eta - top[groups.index]
        exp_shifted = jnp.exp(shifted)
        norm = self._per_group(exp_shifted, groups)
        p = exp_shifted / norm[groups.index]
        return shifted, norm, p, self._per_group(weight * y, groups)

    def _jax_loglik(self, eta, y, aux, weight):
        shifted, norm, _, _ = self._jax_parts(eta, y, aux, weight)
        # y log p summed by row: a group with no rows (norm 0, log -inf) is
        # read by no row, so it adds nothing (a per-group sum would give
        # n_g log(norm) = 0 * -inf there).
        return jnp.sum(weight * y * (shifted - jnp.log(norm)[aux.index]))

    def _jax_score(self, eta, y, aux, weight):
        _, _, p, total = self._jax_parts(eta, y, aux, weight)
        return weight * (y - total[aux.index] * p)

    def _sandwich(self, X, groups, weight, total, diagonal, pairs):
        """``sum_g n_g X_g' (diag(a) - sum_(u, v) u v') X_g`` for a per-row
        ``a`` and per-row vector pairs ``(u, v)``, shape ``(p, p)``."""
        out = _gram(X, weight * total[groups.index] * diagonal)
        for u, v in pairs:
            Xu = self._per_group((weight * u)[:, None] * X, groups)
            Xv = self._per_group((weight * v)[:, None] * X, groups)
            out = out - Xu.T @ (total[:, None] * Xv)
        return out

    def _g_times(self, p, d, groups, weight):
        """``G d = p (d - p'd)`` per group, shape ``(n_rows,)``."""
        return p * (d - self._per_group(weight * p * d, groups)[groups.index])

    def _jax_information(self, X, eta, y, aux, weight):
        _, _, p, total = self._jax_parts(eta, y, aux, weight)
        return self._sandwich(X, aux, weight, total, p, [(p, p)])

    def _jax_dinformation(self, X, eta, y, aux, weight, d):
        _, _, p, total = self._jax_parts(eta, y, aux, weight)
        v = self._g_times(p, d, aux, weight)
        return self._sandwich(X, aux, weight, total, v, [(v, p), (p, v)])

    def _jax_d2information(self, X, eta, y, aux, weight, d1, d2):
        _, _, p, total = self._jax_parts(eta, y, aux, weight)
        v1 = self._g_times(p, d1, aux, weight)
        v2 = self._g_times(p, d2, aux, weight)
        # q = (D_d2 G) d1 = v2 * d1 - v2 (p'd1) - p (v2'd1), per group.
        p_d1 = self._per_group(weight * p * d1, aux)[aux.index]
        v2_d1 = self._per_group(weight * v2 * d1, aux)[aux.index]
        q = v2 * (d1 - p_d1) - p * v2_d1
        return self._sandwich(
            X, aux, weight, total, q, [(q, p), (p, q), (v1, v2), (v2, v1)]
        )

    def _jax_weighted_design(self, X, eta, y, aux, weight):
        """Per group ``sqrt(n_g) diag(sqrt(p_g)) (X_g - 1 p_g' X_g)``."""
        _, _, p, total = self._jax_parts(eta, y, aux, weight)
        means = self._per_group((weight * p)[:, None] * X, aux)
        centered = X - means[aux.index]
        return centered * jnp.sqrt(weight * total[aux.index] * p)[:, None]


def draw(
    lik: Likelihood,
    eta: np.ndarray,
    aux: object,
    rng: np.random.Generator,
    *,
    n_trials: int | np.ndarray = 1,
    concentration: float | np.ndarray | None = None,
) -> np.ndarray:
    """Outcomes drawn from a family at the linear predictor ``eta``.

    The draws use only ``rng``, a NumPy generator, and none of the fitter's
    code, so the same seed gives the same outcomes whatever the fitter's
    solver is. ``aux`` is the family's, as for its other methods:

    - Poisson: counts of mean ``exp(eta + aux)``, ``aux`` the offset.
    - Bernoulli: 0/1 with probability ``expit(eta)``; ``aux`` unused.
    - Gaussian: ``eta`` plus Normal noise of variance ``aux``, the scale.
    - Grouped softmax: ``aux`` the sorted group id of each row; the outcome
      is each row's share of its group, so the shares of a group sum to one,
      as the content response does. With ``p = softmax(eta)`` within a group,
      the mean share of every noise model is ``p``. The noise model is set
      by ``n_trials`` and ``concentration``:

      1. categorical (``n_trials = 1``, no concentration): one row of each
         group is chosen, with probability ``p``, and has share 1. The
         variance of a share is ``p (1 - p)``.
      2. multinomial (``n_trials = n``, an integer or one per group, no
         concentration): ``n`` independent choices per group, the share the
         count over ``n``. Variance ``p (1 - p) / n``: that of (1) over
         ``n``.
      3. Dirichlet-multinomial (``n_trials = n`` and ``concentration = a``,
         a number or one per group): the group's probabilities are first
         drawn from ``Dirichlet(a p)``, then ``n`` choices from them.
         Variance ``p (1 - p) (n + a) / (n (1 + a))``: that of (1) times
         ``(n + a) / (n (1 + a))``, between ``1 / n`` (large ``a``, no
         overdispersion) and 1 (small ``a``); for large ``n`` it tends to
         ``1 / (1 + a)``.

      Each choice is the Gumbel-max draw: the row of the largest
      ``log p + Gumbel`` noise.

    Parameters
    ----------
    lik : Likelihood
    eta : np.ndarray, shape (n_rows,)
    aux : object
    rng : np.random.Generator
    n_trials : int or np.ndarray of int, shape (n_groups,)
        The choices per group under the grouped softmax; not read otherwise.
    concentration : float or np.ndarray, shape (n_groups,), or None
        The Dirichlet concentration of model 3 under the grouped softmax;
        None for models 1 and 2. Not read otherwise.

    Returns
    -------
    np.ndarray, shape (n_rows,)
        Float outcomes.

    Raises
    ------
    ValueError
        If the family has no draw (named by ``lik.name``); under the grouped
        softmax, if ``n_trials`` is not a whole number of at least one or a
        concentration is not finite and positive, or either has the wrong
        number of groups, or a group has no finite ``eta``.
    """
    eta = np.asarray(eta, dtype=float)
    if lik.name == "poisson":
        return rng.poisson(np.exp(eta + np.asarray(aux, dtype=float))).astype(float)
    if lik.name == "bernoulli":
        return (rng.random(eta.size) < expit(eta)).astype(float)
    if lik.name == "gaussian":
        return eta + np.sqrt(aux) * rng.standard_normal(eta.size)
    if lik.name == "grouped_softmax":
        return _softmax_draw(eta, np.asarray(aux), rng, n_trials, concentration)
    raise ValueError(f"no draw for the likelihood {lik.name!r}")


def _per_group(value: object, n_groups: int, name: str) -> np.ndarray:
    """``value`` as one number per group, shape ``(n_groups,)``."""
    value = np.asarray(value, dtype=float)
    if value.ndim > 1 or value.size not in (1, n_groups):
        raise ValueError(f"{name}: one value, or one per group ({n_groups})")
    return np.broadcast_to(value.ravel(), (n_groups,))


def _softmax_draw(
    eta: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    n_trials: int | np.ndarray,
    concentration: float | np.ndarray | None,
) -> np.ndarray:
    """The grouped softmax's shares (:func:`draw`), shape ``(n_rows,)``."""
    starts = group_starts(groups)
    sizes = group_sizes(starts, eta.size)
    n = _per_group(n_trials, starts.size, "n_trials")
    if np.any(n < 1) or np.any(n != np.round(n)):
        raise ValueError(f"n_trials must be whole numbers of at least one, not {n}")
    n = n.astype(int)
    empty = np.flatnonzero(np.maximum.reduceat(eta, starts) == -np.inf)
    if empty.size:
        raise ValueError(
            f"group {empty[0]} has no finite eta (every share zero): no share to draw"
        )
    logits = eta
    if concentration is not None:
        a = _per_group(concentration, starts.size, "concentration")
        if not np.all(np.isfinite(a) & (a > 0)):
            raise ValueError("a concentration must be finite and positive")
        weight = np.exp(eta - np.repeat(np.maximum.reduceat(eta, starts), sizes))
        p = weight / np.repeat(np.add.reduceat(weight, starts), sizes)
        gamma = rng.gamma(np.repeat(a, sizes) * p)
        with np.errstate(divide="ignore"):
            logits = np.log(gamma / np.repeat(np.add.reduceat(gamma, starts), sizes))
    chosen = np.zeros(eta.size)
    for trial in range(int(n.max())):
        noisy = logits + rng.gumbel(size=eta.size)
        top = noisy == np.repeat(np.maximum.reduceat(noisy, starts), sizes)
        # One row per group: the first at the maximum (a tie has probability
        # zero unless every weight of a group underflowed to zero).
        before = np.cumsum(top) - top
        first = top & (before == np.repeat(before[starts], sizes))
        chosen += first & np.repeat(trial < n, sizes)
    return chosen / np.repeat(n, sizes)
