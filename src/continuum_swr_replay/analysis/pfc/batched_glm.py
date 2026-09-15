"""Batched Poisson GLM fitter: every unit of a session in one IRLS.

The PFC ladder fits nine Poisson GLMs per unit on designs of 270k-700k rows,
and every unit in a session shares the *same* design -- only the response
column differs. Fitting them one at a time with statsmodels spends almost all
of its time rebuilding the same weighted cross-products, once per unit.

This module fits them together. One IRLS (equivalently, Newton with the
Fisher information) is run for all units at once: per iteration

.. code-block:: text

    eta_n = Q c_n                         (n_time,) per unit
    mu_n  = exp(eta_n)
    W_n   = family weights = mu_n         (log link, Poisson variance)
    H_n   = Q' diag(W_n) Q                (rank, rank) per unit
    r_n   = Q' (W_n eta_n + (y_n - mu_n))
    c_n  <- H_n^+ r_n

``H`` is accumulated over chunks of the time axis, so the
``(n_time, rank, n_units)`` intermediate is never materialized.

The fit runs in an orthonormal basis ``Q`` of the design's column space
(``X = U S V'``, ``Q = U[:, S > tol]`` with statsmodels' own rank tolerance),
not in the design's own columns, for two reasons. Numerically, the normal
equations square the design's condition number, and a spline design on a
peri-event basis is not well conditioned; ``Q'WQ`` carries only the spread of
the *weights*. Structurally, it makes the rank-deficient case exact rather
than a tolerance question: an event class with no events leaves its spline
columns all-zero, those directions are simply absent from ``Q``, and mapping
back through ``V S^-1`` returns the minimum-norm parameters -- the same ones
statsmodels' pinv-based WLS gives, with exact zeros in the absent columns.

**The convergence rule is statsmodels 0.14.5's, not an approximation of it.**
``GLM.fit(method="IRLS")`` defaults to ``maxiter=100, tol=1e-8`` and
``_fit_irls`` starts from ``family.starting_mu(y) = (y + mean(y)) / 2`` (no
starting parameters), then stops the first time
``np.allclose(deviance_prev, deviance, atol=1e-8, rtol=0)`` -- i.e. when the
deviance moves by less than ``1e-8`` in absolute terms. Both are mirrored here
exactly, unit by unit: a unit that meets the rule is frozen at that iteration's
parameters while the rest of the batch keeps going, so batching cannot move a
unit's answer by letting it iterate longer than statsmodels would have.

Two consequences of that mirroring are worth stating because they are easy to
get wrong:

- ``rank`` is ``numpy.linalg.matrix_rank`` of the design, which is what
  statsmodels uses for ``df_model = rank - 1`` and ``df_resid = nobs - rank``.
  ``aic`` is therefore ``-2 llf + 2 rank``, not ``-2 llf + 2 n_features``;
  they differ exactly when the design is rank-deficient.
- a unit with no spikes at all on the fitted bins has ``starting_mu = 0``, so
  statsmodels' first deviance is ``nan`` and it raises. Here that unit's
  results come back all-``nan`` (the rest of the batch is unaffected), which
  the caller's NaN-parameter check turns into the same failure it always was.

**Where the mirroring stops, and what to do about it.** statsmodels does not
solve the normal equations at all: each step is a least-squares solve on the
weighted design ``W^{1/2} X`` with a singular-value cutoff, which tolerates
condition numbers no Gram-matrix method can. The two agree to ~1e-12 for as
long as the fitted means stay in range, and part company once a fit is driven
into separation -- a spline column with no spikes anywhere in its support
sends its coefficient to ``-inf``, and the fitted means then spread over so
many orders of magnitude that the information matrix's condition number
exceeds what a Gram-matrix solve can carry (statsmodels' own ``_clean``
clipping makes it worse still, giving weights of ``eps**2 / mu`` on the bins
the model has given up on). Past that point neither fitter's answer means
anything, but they are different meaningless answers.
:attr:`BatchedGLMResults.needs_reference_fit` names the units that entered
that regime (or failed to converge), so a caller that must match statsmodels
exactly can refit just those units with statsmodels and pay the per-unit cost
only for them -- :func:`continuum_swr_replay.analysis.pfc.glm.run_session`
does, as does :func:`~continuum_swr_replay.analysis.pfc.glm.process_neuron`
for a single neuron.

Float64 throughout: the convergence test is an absolute ``1e-8`` on a deviance
of order ``1e5``, which float32 cannot even represent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.gpu import require_configured_device

# Float64 is a correctness requirement, not a preference: the IRLS stops on an
# absolute deviance change of 1e-8 against deviances of order 1e5, which is
# below float32's resolution there -- in float32 the loop would either never
# converge or stop at an arbitrary iteration.
jax.config.update("jax_enable_x64", True)

#: statsmodels' ``GLM.fit`` defaults, mirrored so both fitters stop at the
#: same point (``statsmodels.genmod.generalized_linear_model.GLM.fit``:
#: ``maxiter=100, tol=1e-8``, ``atol = tol``, ``rtol = 0``).
DEFAULT_MAXITER = 100
DEFAULT_TOL = 1e-8

#: Eigenvalue cutoff for the symmetric pseudo-inverse solve, relative to the
#: largest eigenvalue of ``Q' W Q``. Because ``Q`` is orthonormal, that ratio
#: reads the spread of the *weights* over the design's directions, not the
#: design's own conditioning: a direction the weights have abandoned is cut,
#: which is what keeps a separating fit from taking an unbounded step, and no
#: direction a healthy fit uses comes anywhere near it.
DEFAULT_RCOND = 1e-12

#: Widest spread of fitted means, ``min(mu) / max(mu)``, this fitter will
#: vouch for. The information matrix's condition number is bounded by that
#: spread, and the normal equations lose twice as many digits as the
#: least-squares solve statsmodels uses, so past ~1e-8 the two stop agreeing to
#: the tolerances the audit is read at. A healthy peri-event fit spans a factor
#: of tens; a spread of 1e-8 is a coefficient of -18 or beyond, i.e. a design
#: column whose support holds no spikes and whose MLE is at -infinity.
_MIN_FITTED_MEAN_RATIO = 1e-8

#: ``statsmodels.genmod.families.family.Family._clean`` clips at this before
#: dividing, and the log link's ``deriv`` clips ``mu`` the same way. Mirrored
#: so an underflowing fitted mean is handled identically.
_FLOAT_EPS = float(np.finfo(float).eps)

#: Elements per chunked intermediate (~64 MB in float64). The time axis is
#: split so that neither the ``(chunk, n_units)`` working arrays nor the
#: ``(chunk, rank**2)`` cross-product block gets large.
_TARGET_CHUNK_ELEMENTS = 8_000_000


@dataclass(frozen=True)
class UnitGLMResult:
    """One unit's fit, exposing the statsmodels ``GLMResults`` attributes used.

    Attributes
    ----------
    params : NDArray, shape (n_features,)
    llf : float
        Full Poisson log-likelihood, ``-log(y!)`` term included, as
        statsmodels' ``llf``.
    deviance : float
    aic : float
        ``-2 llf + 2 rank``, statsmodels' ``-2 llf + 2 (df_model + 1)``.
    converged : bool
    n_iter : int
        IRLS steps taken, as statsmodels' ``fit_history["iteration"]``.
    rank : int
        Rank of the design (shared by every unit).
    nobs : int
    n_features : int
    needs_reference_fit : bool
        The fit left the range where this fitter and statsmodels agree -- it
        did not converge, or its fitted means spread further than the normal
        equations can carry. See the module docstring.
    bse : NDArray, shape (n_features,)
        Standard errors of ``params``, as statsmodels' ``bse``: the square
        root of the diagonal of ``(X' W X)^+`` at the fitted weights, with
        Poisson's fixed unit scale. Exactly zero in a direction the design
        does not span, where the parameter is itself exactly zero.
    """

    params: NDArray
    llf: float
    deviance: float
    aic: float
    converged: bool
    n_iter: int
    rank: int
    nobs: int
    n_features: int
    needs_reference_fit: bool = False
    bse: NDArray | None = None

    @property
    def df_model(self) -> float:
        """``rank - 1``, as statsmodels defines it."""
        return float(self.rank) - 1.0

    @property
    def df_resid(self) -> float:
        """``nobs - rank``, as statsmodels defines it."""
        return float(self.nobs) - float(self.rank)


@dataclass(frozen=True)
class BatchedGLMResults:
    """Every unit's fit of one design.

    Attributes
    ----------
    params : NDArray, shape (n_units, n_features)
    llf, deviance, aic : NDArray, shape (n_units,)
    converged : NDArray of bool, shape (n_units,)
    n_iter : NDArray of int, shape (n_units,)
    rank, nobs, n_features : int
        Properties of the design, shared by every unit.
    needs_reference_fit : NDArray of bool, shape (n_units,)
        Units whose fit left the range where this fitter and statsmodels
        agree. A caller that must match statsmodels refits these with it.
    bse : NDArray, shape (n_units, n_features)
        Per-unit standard errors of ``params``, as statsmodels' ``bse``.
    """

    params: NDArray
    llf: NDArray
    deviance: NDArray
    aic: NDArray
    converged: NDArray
    n_iter: NDArray
    rank: int
    nobs: int
    n_features: int
    needs_reference_fit: NDArray | None = None
    bse: NDArray | None = None

    def __post_init__(self) -> None:
        if self.needs_reference_fit is None:
            object.__setattr__(
                self, "needs_reference_fit", np.zeros(len(self.params), dtype=bool)
            )
        if self.bse is None:
            object.__setattr__(self, "bse", np.full_like(self.params, np.nan))

    @property
    def n_units(self) -> int:
        return int(self.params.shape[0])

    @property
    def df_model(self) -> float:
        return float(self.rank) - 1.0

    @property
    def df_resid(self) -> float:
        return float(self.nobs) - float(self.rank)

    def for_unit(self, index: int) -> UnitGLMResult:
        """The single-unit view ``process_neuron`` and its helpers read.

        Parameters
        ----------
        index : int
            Unit index into the batch.

        Returns
        -------
        UnitGLMResult
        """
        return UnitGLMResult(
            params=np.asarray(self.params[index]),
            llf=float(self.llf[index]),
            deviance=float(self.deviance[index]),
            aic=float(self.aic[index]),
            converged=bool(self.converged[index]),
            n_iter=int(self.n_iter[index]),
            rank=self.rank,
            nobs=self.nobs,
            n_features=self.n_features,
            needs_reference_fit=bool(self.needs_reference_fit[index]),
            bse=np.asarray(self.bse[index]),
        )


def jax_backend() -> str:
    """Platform JAX will run on, for a provenance record (``"gpu"``/``"cpu"``)."""
    return str(jax.devices()[0].platform)


def _chunk_rows(n_units: int, rank: int) -> int:
    """Rows per chunk of the time axis, from the widest intermediate.

    The two are the ``(chunk, n_units)`` working arrays and the
    ``(chunk, rank**2)`` cross-product block.
    """
    widest = max(n_units, rank * rank, 1)
    return int(np.clip(_TARGET_CHUNK_ELEMENTS // widest, 1024, 1 << 17))


def _to_chunks(array: NDArray, chunk_rows: int) -> jnp.ndarray:
    """Pad the leading axis to a multiple of ``chunk_rows`` and split it.

    Padded rows are zeros in both the design and the counts; the design's zero
    rows make their cross-product and score contributions exactly zero, and a
    validity mask keeps them out of the deviance.
    """
    n_rows = array.shape[0]
    n_pad = (-n_rows) % chunk_rows
    if n_pad:
        pad_width = [(0, n_pad)] + [(0, 0)] * (array.ndim - 1)
        array = np.pad(array, pad_width)
    return jnp.asarray(array).reshape((-1, chunk_rows) + array.shape[1:])


def _valid_chunks(n_rows: int, chunk_rows: int) -> jnp.ndarray:
    """Boolean mask of real (non-padding) rows, chunked like the data."""
    n_pad = (-n_rows) % chunk_rows
    valid = np.ones(n_rows + n_pad, dtype=bool)
    if n_pad:
        valid[n_rows:] = False
    return jnp.asarray(valid).reshape(-1, chunk_rows)


@partial(jax.jit, static_argnames=("start",))
def _sweep(
    design_chunks: jnp.ndarray,
    count_chunks: jnp.ndarray,
    valid_chunks: jnp.ndarray,
    params: jnp.ndarray,
    mean_count: jnp.ndarray,
    start: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One pass over the time axis: information, score side, deviance, llf.

    Parameters
    ----------
    design_chunks : jnp.ndarray, shape (n_chunks, chunk_rows, rank)
        The orthonormal basis of the design's column space, chunked.
    count_chunks : jnp.ndarray, shape (n_chunks, chunk_rows, n_units)
    valid_chunks : jnp.ndarray of bool, shape (n_chunks, chunk_rows)
    params : jnp.ndarray, shape (n_units, rank)
        Current coefficients in that basis; ignored when ``start``.
    mean_count : jnp.ndarray, shape (n_units,)
        Per-unit mean count, for ``family.starting_mu``.
    start : bool
        Use statsmodels' starting mean ``(y + mean(y)) / 2`` instead of
        ``exp(Q c)``. Static, so each variant compiles once.

    Returns
    -------
    information : jnp.ndarray, shape (n_units, rank, rank)
    rhs : jnp.ndarray, shape (rank, n_units)
        ``Q' (W eta + (y - mu))``, the weighted-least-squares right-hand side.
    deviance, llf_kernel : jnp.ndarray, shape (n_units,)
        ``llf_kernel`` is ``sum(y log mu - mu)``; the ``-log(y!)`` term is a
        constant of the data and is added once by the caller.
    smallest_mu, largest_mu : jnp.ndarray, shape (n_units,)
        Extremes of the fitted mean over the sampled bins. Their ratio bounds
        the information matrix's condition number, which is what decides
        whether this fitter can vouch for the unit.
    """
    rank = design_chunks.shape[-1]
    n_units = count_chunks.shape[-1]

    def body(carry, block):
        information, rhs, deviance, llf_kernel, smallest_mu, largest_mu = carry
        design, count, valid = block
        if start:
            mu = 0.5 * (count + mean_count)
            eta = jnp.log(mu)
        else:
            eta = design @ params.T
            mu = jnp.exp(eta)
        # ``Family._clean``/``Log.deriv`` clip at FLOAT_EPS before dividing, so
        # the IRLS weight is mu everywhere mu >= eps and degrades the same way
        # statsmodels' does below it.
        mu_clean = jnp.clip(mu, _FLOAT_EPS, jnp.inf)
        weight = mu_clean**2 / mu
        # W z with z = eta + (y - mu) / mu_clean, i.e. the WLS right-hand side.
        weighted_response = weight * eta + (mu_clean / mu) * (count - mu)

        keep = valid[:, None]
        weight = jnp.where(keep, weight, 0.0)
        weighted_response = jnp.where(keep, weighted_response, 0.0)

        rhs = rhs + design.T @ weighted_response
        cross = (design[:, :, None] * design[:, None, :]).reshape(
            design.shape[0], rank * rank
        )
        information = information + weight.T @ cross

        # statsmodels' Poisson deviance residual, clip included.
        ratio = jnp.clip(count / mu, _FLOAT_EPS, jnp.inf)
        resid_dev = 2.0 * (count * jnp.log(ratio) - (count - mu))
        deviance = deviance + jnp.sum(jnp.where(keep, resid_dev, 0.0), axis=0)
        llf_kernel = llf_kernel + jnp.sum(
            jnp.where(keep, count * jnp.log(mu) - mu, 0.0), axis=0
        )
        smallest_mu = jnp.minimum(
            smallest_mu, jnp.min(jnp.where(keep, mu, jnp.inf), axis=0)
        )
        largest_mu = jnp.maximum(
            largest_mu, jnp.max(jnp.where(keep, mu, -jnp.inf), axis=0)
        )
        return (
            information,
            rhs,
            deviance,
            llf_kernel,
            smallest_mu,
            largest_mu,
        ), None

    init = (
        jnp.zeros((n_units, rank * rank)),
        jnp.zeros((rank, n_units)),
        jnp.zeros(n_units),
        jnp.zeros(n_units),
        jnp.full(n_units, jnp.inf),
        jnp.full(n_units, -jnp.inf),
    )
    (information, rhs, deviance, llf_kernel, smallest_mu, largest_mu), _ = jax.lax.scan(
        body, init, (design_chunks, count_chunks, valid_chunks)
    )
    return (
        information.reshape(n_units, rank, rank),
        rhs,
        deviance,
        llf_kernel,
        smallest_mu,
        largest_mu,
    )


@jax.jit
def _solve(information: jnp.ndarray, rhs: jnp.ndarray, rcond: float) -> jnp.ndarray:
    """Per-unit symmetric pseudo-inverse solve of ``H c = r``.

    ``H = Q' W Q`` with ``Q`` orthonormal, so its eigenvalues are the squared
    singular values of the weighted basis and the relative cutoff reads the
    spread of the weights alone. Cutting there is what a least-squares solve on
    the weighted design does, and it is what stops a direction the weights have
    abandoned from producing an unbounded step.

    Parameters
    ----------
    information : jnp.ndarray, shape (n_units, rank, rank)
    rhs : jnp.ndarray, shape (rank, n_units)
    rcond : float
        Eigenvalue cutoff relative to the largest eigenvalue.

    Returns
    -------
    jnp.ndarray, shape (n_units, rank)
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(information)
    largest = jnp.max(jnp.abs(eigenvalues), axis=1, keepdims=True)
    keep = eigenvalues > rcond * largest
    inverse = jnp.where(keep, 1.0 / jnp.where(keep, eigenvalues, 1.0), 0.0)

    projected = jnp.einsum("nji,nj->ni", eigenvectors, rhs.T)
    return jnp.einsum("nij,nj->ni", eigenvectors, inverse * projected)


def _standard_errors(information: jnp.ndarray, back: NDArray, rcond: float) -> NDArray:
    """Per-unit ``sqrt(diag(cov(beta)))`` in the design's own columns.

    The fit runs in an orthonormal basis ``Q`` with information ``H = Q' W Q``,
    and ``beta = B c`` with ``B = V S^-1`` (``back``). So
    ``cov(beta) = B H^+ B'`` -- Poisson's scale is fixed at one, which is why
    no dispersion estimate enters -- and only its diagonal is wanted. ``H^+``
    is taken on the same eigenvalue cutoff the IRLS step uses, so a direction
    the weights have abandoned contributes nothing here either, and a column
    the design does not span gets an exactly zero standard error beside its
    exactly zero parameter.

    Parameters
    ----------
    information : jnp.ndarray, shape (n_units, rank, rank)
        ``H`` at the fitted parameters.
    back : NDArray, shape (n_features, rank)
        ``V S^-1``, the map out of the orthonormal basis.
    rcond : float
        Eigenvalue cutoff relative to the largest, as in :func:`_solve`.

    Returns
    -------
    NDArray, shape (n_units, n_features)
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(information)
    largest = jnp.max(jnp.abs(eigenvalues), axis=1, keepdims=True)
    keep = eigenvalues > rcond * largest
    inverse = jnp.where(keep, 1.0 / jnp.where(keep, eigenvalues, 1.0), 0.0)
    # ``B V`` per unit, then the diagonal of ``(B V) diag(1/lambda) (B V)'``.
    rotated = jnp.einsum("pr,nrk->npk", jnp.asarray(back), eigenvectors)
    variance = jnp.einsum("npk,nk->np", rotated**2, inverse)
    return np.sqrt(np.asarray(variance))


def orthonormal_design_basis(
    design: NDArray,
) -> tuple[NDArray, NDArray, int]:
    """An orthonormal basis of the design's column space, and the map back.

    ``X = U S V'``; the directions with ``S`` above statsmodels' own rank
    tolerance (``numpy.linalg.matrix_rank``'s: ``S.max() * max(shape) * eps``)
    span the column space. Fitting ``eta = Q c`` in that basis and mapping back
    with ``beta = V S^-1 c`` gives the minimum-norm parameters -- the same
    solution statsmodels' pinv-based WLS returns, and exactly zero in any
    all-zero design column.

    Parameters
    ----------
    design : NDArray, shape (n_time, n_features)

    Returns
    -------
    basis : NDArray, shape (n_time, rank)
        Orthonormal columns spanning the design's column space.
    back : NDArray, shape (n_features, rank)
        ``beta = back @ c``.
    rank : int
        Number of kept directions -- ``numpy.linalg.matrix_rank(design)``.
    """
    left, singular_values, right = np.linalg.svd(design, full_matrices=False)
    tolerance = (
        singular_values.max(initial=0.0) * max(design.shape) * np.finfo(float).eps
    )
    kept = singular_values > tolerance
    return (
        np.ascontiguousarray(left[:, kept]),
        np.ascontiguousarray(right[kept].T / singular_values[kept]),
        int(kept.sum()),
    )


def fit_poisson_glms(
    design: NDArray,
    counts: NDArray,
    *,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
    rcond: float = DEFAULT_RCOND,
    chunk_rows: int | None = None,
) -> BatchedGLMResults:
    """Fit one Poisson GLM per column of ``counts`` against a shared design.

    Reproduces ``statsmodels.api.GLM(y, X, family=Poisson()).fit()`` unit by
    unit: same log link, same starting mean, same IRLS step, same convergence
    rule, same pseudo-inverse treatment of a rank-deficient design.

    Parameters
    ----------
    design : NDArray, shape (n_time, n_features)
        Design matrix, already restricted to the bins being fit. Shared by
        every unit.
    counts : NDArray, shape (n_time, n_units) or (n_time,)
        Spike counts per bin and unit. A 1-D array is one unit.
    maxiter : int
        Maximum IRLS steps per unit.
    tol : float
        Absolute deviance change below which a unit has converged.
    rcond : float
        Eigenvalue cutoff for the pseudo-inverse solve, relative to the largest
        eigenvalue of ``Q' W Q``.
    chunk_rows : int, optional
        Rows of the time axis per chunk. Defaults to a size that keeps the
        working arrays around 64 MB.

    Returns
    -------
    BatchedGLMResults
        Units whose fit is not defined -- no spikes at all on these bins, so
        statsmodels' first deviance is ``nan`` and it raises -- come back with
        ``nan`` parameters, deviance, llf and aic, and ``converged`` False.
        Units that did not converge, or whose fitted means spread further
        than the normal equations can carry, are named in
        ``needs_reference_fit``.

    Raises
    ------
    ValueError
        If the design and counts disagree on the number of rows.
    UnconfiguredDeviceError
        If no JAX device has been chosen and CUDA is still reachable. The
        check is here, at the one function that runs the solver, because a
        caller that reaches it without going through a runner would otherwise
        get JAX's default on a shared host: every visible card, most of each
        preallocated.
    """
    require_configured_device()
    design_array = np.ascontiguousarray(np.asarray(design, dtype=float))
    counts_array = np.asarray(counts, dtype=float)
    if counts_array.ndim == 1:
        counts_array = counts_array[:, None]
    counts_array = np.ascontiguousarray(counts_array)

    if design_array.ndim != 2:
        raise ValueError(f"design must be 2-D, got shape {design_array.shape}")
    if counts_array.shape[0] != design_array.shape[0]:
        raise ValueError(
            f"design has {design_array.shape[0]} rows but counts has "
            f"{counts_array.shape[0]}"
        )

    n_time, n_features = design_array.shape
    n_units = counts_array.shape[1]

    if n_time == 0:
        # statsmodels raises here (the mean of an empty response is nan, so the
        # first deviance is nan). Report the same failure per unit instead, and
        # let the caller's NaN check raise as it always has.
        nan = np.full(n_units, np.nan)
        return BatchedGLMResults(
            params=np.full((n_units, n_features), np.nan),
            llf=nan,
            deviance=nan.copy(),
            aic=nan.copy(),
            converged=np.zeros(n_units, dtype=bool),
            n_iter=np.zeros(n_units, dtype=int),
            rank=0,
            nobs=0,
            n_features=n_features,
            needs_reference_fit=np.zeros(n_units, dtype=bool),
        )

    # The fit runs in an orthonormal basis of the design's column space; its
    # width is the design's rank, on statsmodels' own rank tolerance, so
    # ``df_model``/``df_resid``/``aic`` are counted exactly as statsmodels
    # counts them -- and computed once for a design every unit shares.
    basis, back, rank = orthonormal_design_basis(design_array)

    if chunk_rows is None:
        chunk_rows = _chunk_rows(n_units, rank)
    design_chunks = _to_chunks(basis, chunk_rows)
    count_chunks = _to_chunks(counts_array, chunk_rows)
    valid_chunks = _valid_chunks(n_time, chunk_rows)

    mean_count = jnp.asarray(counts_array.mean(axis=0))
    # ``sum_i log(y_i!)``, the part of the Poisson log-likelihood that does not
    # depend on the parameters, so it is computed once rather than per sweep.
    log_factorial = jax.scipy.special.gammaln(jnp.asarray(counts_array) + 1.0).sum(
        axis=0
    )

    params = jnp.zeros((n_units, rank))
    information, rhs, deviance, _, _, _ = _sweep(
        design_chunks, count_chunks, valid_chunks, params, mean_count, True
    )
    # ``_fit_irls`` raises when the deviance at the starting mean is nan; the
    # per-unit analogue is to mark the unit failed and leave the batch alone.
    failed = np.isnan(np.asarray(deviance))
    active = ~failed
    converged = np.zeros(n_units, dtype=bool)
    n_iter = np.zeros(n_units, dtype=int)
    deviance_prev = np.asarray(deviance)
    llf_kernel = np.full(n_units, np.nan)
    deviance_at_params = np.full(n_units, np.nan)

    active_device = jnp.asarray(active)
    params = jnp.where(active_device[:, None], _solve(information, rhs, rcond), params)

    overspread = np.zeros(n_units, dtype=bool)
    for iteration in range(1, maxiter + 1):
        information, rhs, deviance, llf, smallest_mu, largest_mu = _sweep(
            design_chunks, count_chunks, valid_chunks, params, mean_count, False
        )
        # A non-finite largest fitted mean is an overflow of ``exp(X beta)``,
        # not a spread the ratio test can see -- ``nan`` compares false against
        # everything -- so such a unit would otherwise stay active to
        # ``maxiter``, holding the whole batch at its iteration count.
        largest_host = np.asarray(largest_mu)
        overspread |= active & (
            ~np.isfinite(largest_host)
            | (np.asarray(smallest_mu) < _MIN_FITTED_MEAN_RATIO * largest_host)
        )
        deviance_host = np.asarray(deviance)
        deviance_at_params = np.where(active, deviance_host, deviance_at_params)
        llf_kernel = np.where(active, np.asarray(llf), llf_kernel)
        n_iter = np.where(active, iteration, n_iter)

        # ``_check_convergence``: np.allclose(prev, current, atol=tol, rtol=0).
        just_converged = active & (np.abs(deviance_host - deviance_prev) <= tol)
        converged |= just_converged
        # A unit whose fitted means have spread past what this solver can carry
        # is stopped here rather than iterated to ``maxiter``: its answer is
        # going to be replaced by a reference fit either way, and leaving it in
        # would hold the whole batch at the iteration count of its worst
        # member.
        active = active & ~just_converged & ~overspread
        if not active.any() or iteration == maxiter:
            break

        deviance_prev = np.where(active, deviance_host, deviance_prev)
        active_device = jnp.asarray(active)
        params = jnp.where(
            active_device[:, None], _solve(information, rhs, rcond), params
        )

    # Back out of the orthonormal basis: ``beta = V S^-1 c``, the minimum-norm
    # parameters in the design's own columns.
    params_host = np.asarray(params) @ back.T
    llf_host = llf_kernel - np.asarray(log_factorial)
    aic = -2.0 * llf_host + 2.0 * rank
    # A unit with no spikes has no fit to speak of, and statsmodels refuses it
    # rather than returning one, so it is not a candidate for a reference fit.
    needs_reference_fit = (overspread | ~converged) & ~failed

    standard_errors = _standard_errors(information, back, rcond)

    if failed.any():
        params_host = params_host.copy()
        standard_errors = standard_errors.copy()
        standard_errors[failed] = np.nan
        params_host[failed] = np.nan
        deviance_at_params = np.where(failed, np.nan, deviance_at_params)
        llf_host = np.where(failed, np.nan, llf_host)
        aic = np.where(failed, np.nan, aic)

    return BatchedGLMResults(
        params=params_host,
        llf=llf_host,
        deviance=deviance_at_params,
        aic=aic,
        converged=converged,
        n_iter=n_iter,
        rank=rank,
        nobs=n_time,
        n_features=n_features,
        needs_reference_fit=needs_reference_fit,
        bse=standard_errors,
    )
