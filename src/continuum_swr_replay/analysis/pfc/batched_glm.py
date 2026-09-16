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

**The penalty, and why separation stops being a special case.** Left
unpenalised, a peri-event spline fit separates whenever one basis column's
support holds no spikes: that coefficient's maximum-likelihood estimate is at
``-inf``, the fitted means spread over so many orders of magnitude that no
Gram-matrix solve can carry the information matrix, and the number that comes
back means nothing. ``penalty`` replaces that limit with a Gaussian prior on
the coefficients -- a per-column ``lambda_j = 1 / sigma_j**2``, zero on the
intercept -- so the objective is strictly concave and every fit is finite.

The prior is stated in the design's **own** columns, where each spline
coefficient is a log-rate modulation and ``sigma`` is therefore readable as a
rate multiplier. The fit runs in the orthonormal basis, so the penalty is
carried across: with ``X = Q S V'`` and ``beta = V S^-1 c = B c``,

.. code-block:: text

    (1/2) sum_j lambda_j beta_j**2 = (1/2) c' (B' Lambda B) c =: (1/2) c' P c

``P`` is added to the information and its gradient ``P c`` to the score, which
in the IRLS form is exactly ``c <- (H + P)^+ r`` with the same working-response
right-hand side ``r``: the Newton step of the penalised log-likelihood is a
ridge-weighted least squares on the same working response. Nothing else about
the iteration changes, so ``penalty=None`` (or a vector of zeros) reproduces
the unpenalised fit bit for bit.

Convergence is on the **penalised** deviance ``deviance + c' P c`` -- the
quantity the iteration actually maximises -- on the same absolute ``1e-8``
rule. The reported ``deviance`` and ``llf`` stay unpenalised, because they are
what a held-out score and a deviance fraction are read from. ``bse`` becomes
the penalised posterior-mode standard error, ``sqrt(diag((X' W X + Lambda)^+))``
(see :func:`_standard_errors`), which is narrower than the unpenalised one
wherever the prior binds.

Against statsmodels, the unpenalised fit agrees to ~1e-12 for as long as the
fitted means stay in range. statsmodels does not solve the normal equations at
all -- each step is a least-squares solve on the weighted design ``W^{1/2} X``
with a singular-value cutoff -- so where an unpenalised fit is driven into
separation the two part company, and neither answer means anything. That is
the regime the penalty removes.

Float64 throughout: the convergence test is an absolute ``1e-8`` on a deviance
of order ``1e5``, which float32 cannot even represent.

Held-out scoring uses a separate prediction-only engine. It keeps the shared
response on one device, batches designs and folds, runs the per-unit stopping
logic in a compiled ``while_loop``, and skips covariance, log-likelihood and
AIC work that cannot affect a prediction score. Each tall training design is
reduced by QR before a feature-sized SVD supplies the rank-aware minimum-norm
map. Fixed-size design microbatches give repeated null draws stable compiled
shapes and a bounded large-array working set.
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

#: Prior standard deviation of a penalised coefficient, on the log-rate scale.
#: Read what it means for a *curve* with :func:`implied_prior_sd`, not off this
#: number: the coefficients are B-spline weights, so the prior on the kernel at
#: one lag is wider than ``sigma`` by ``sqrt(sum_b B_b(lag)**2)`` (~0.69 on the
#: shipped basis) and the prior on a *sustained* modulation is narrower by
#: ``sqrt(n_basis)``. At 3.0 those are about 2.1 and 0.9 -- a rate multiplier
#: between x0.02 and x57 at one lag, and between x0.17 and x5.9 held across the
#: reported window.
DEFAULT_PENALTY_SIGMA = 3.0

#: Eigenvalue cutoff for the symmetric pseudo-inverse solve, relative to the
#: largest eigenvalue of ``Q' W Q``. Because ``Q`` is orthonormal, that ratio
#: reads the spread of the *weights* over the design's directions, not the
#: design's own conditioning: a direction the weights have abandoned is cut,
#: which is what keeps a separating fit from taking an unbounded step, and no
#: direction a healthy fit uses comes anywhere near it.
DEFAULT_RCOND = 1e-12

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
        The penalised deviance settled inside ``tol`` before ``maxiter``.
    n_iter : int
        IRLS steps taken, as statsmodels' ``fit_history["iteration"]``.
    rank : int
        Rank of the design (shared by every unit).
    nobs : int
    n_features : int
    bse : NDArray, shape (n_features,)
        Standard errors of ``params``: the square root of the diagonal of
        ``(X' W X + Lambda)^+`` at the fitted weights, with Poisson's fixed
        unit scale -- the penalised posterior-mode standard error, which is
        statsmodels' ``bse`` exactly when the penalty is zero. Exactly zero in
        a direction the design does not span, where the parameter is itself
        exactly zero.
    covariance : NDArray, shape (n_features, n_features)
        ``(X' W X + Lambda)^+`` itself. Carried because the uncertainty of a
        *curve* is not a function of the diagonal: the standard error of
        ``basis @ params`` at one lag needs the off-diagonal terms too.
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
    bse: NDArray | None = None
    covariance: NDArray | None = None

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
    bse : NDArray, shape (n_units, n_features)
        Per-unit penalised posterior-mode standard errors of ``params``.
    covariance : NDArray, shape (n_units, n_features, n_features)
        Per-unit ``(X' W X + Lambda)^+``, for the standard error of a curve
        read off the parameters.
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
    bse: NDArray | None = None
    covariance: NDArray | None = None

    def __post_init__(self) -> None:
        if self.bse is None:
            object.__setattr__(self, "bse", np.full_like(self.params, np.nan))
        if self.covariance is None:
            n_units, n_features = self.params.shape
            object.__setattr__(
                self,
                "covariance",
                np.full((n_units, n_features, n_features), np.nan),
            )

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
            bse=np.asarray(self.bse[index]),
            covariance=np.asarray(self.covariance[index]),
        )


@dataclass(frozen=True)
class PreparedHeldoutResponse:
    """One session's response and folds, resident on the configured device.

    Held-out scoring fits many changing event designs against the same spike
    counts and the same fold assignment.  Keeping these two arrays on-device
    avoids transferring and rechunking the response for every class, draw and
    fold.
    """

    counts: jax.Array
    folds: jax.Array
    n_folds: int

    @property
    def n_time(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.counts.shape[1])


@dataclass(frozen=True)
class BatchedHeldoutResults:
    """Prediction-only held-out scores for a batch of equal-width designs."""

    per_fold: NDArray
    total: NDArray
    converged: NDArray
    n_iter: NDArray
    rank: NDArray


def prepare_heldout_response(
    counts: NDArray, folds: NDArray
) -> PreparedHeldoutResponse:
    """Validate and transfer a session response once for repeated scoring."""
    require_configured_device()
    counts_array = np.asarray(counts, dtype=float)
    if counts_array.ndim == 1:
        counts_array = counts_array[:, None]
    folds_array = np.asarray(folds, dtype=np.int64)
    if counts_array.ndim != 2 or folds_array.ndim != 1:
        raise ValueError(
            "counts must be 1-D or 2-D and folds 1-D; got "
            f"{counts_array.shape} and {folds_array.shape}"
        )
    if counts_array.shape[0] != folds_array.shape[0]:
        raise ValueError(
            f"counts has {counts_array.shape[0]} rows but folds has "
            f"{folds_array.shape[0]}"
        )
    labels = np.unique(folds_array)
    if labels.size < 2 or not np.array_equal(labels, np.arange(labels.size)):
        raise ValueError(
            "folds must use every consecutive label from 0 and contain at "
            f"least two folds; got {labels.tolist()}"
        )
    return PreparedHeldoutResponse(
        counts=jax.device_put(np.ascontiguousarray(counts_array)),
        folds=jax.device_put(np.ascontiguousarray(folds_array)),
        n_folds=int(labels.size),
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
        constant of the data and is added once by the caller. Both are
        unpenalised: the penalty is a function of the parameters alone and is
        added by the caller, which holds them.
    """
    rank = design_chunks.shape[-1]
    n_units = count_chunks.shape[-1]

    def body(carry, block):
        information, rhs, deviance, llf_kernel = carry
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
        return (information, rhs, deviance, llf_kernel), None

    init = (
        jnp.zeros((n_units, rank * rank)),
        jnp.zeros((rank, n_units)),
        jnp.zeros(n_units),
        jnp.zeros(n_units),
    )
    (information, rhs, deviance, llf_kernel), _ = jax.lax.scan(
        body, init, (design_chunks, count_chunks, valid_chunks)
    )
    return information.reshape(n_units, rank, rank), rhs, deviance, llf_kernel


@jax.jit
def _solve(information: jnp.ndarray, rhs: jnp.ndarray, rcond: float) -> jnp.ndarray:
    """Per-unit symmetric pseudo-inverse solve of ``(H + P) c = r``.

    ``H = Q' W Q`` with ``Q`` orthonormal, so its eigenvalues are the squared
    singular values of the weighted basis and the relative cutoff reads the
    spread of the weights alone. Cutting there is what a least-squares solve on
    the weighted design does, and it is what stops a direction the weights have
    abandoned from producing an unbounded step. With a penalty, the caller
    passes ``H + P``: adding ``P`` to the information and its gradient ``P c``
    to the score leaves the IRLS step at ``c <- (H + P)^+ r`` with the same
    working-response right-hand side.

    Parameters
    ----------
    information : jnp.ndarray, shape (n_units, rank, rank)
        ``H``, or ``H + P`` when the fit is penalised.
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


def _parameter_covariance(
    information: jnp.ndarray, back: NDArray, rcond: float
) -> NDArray:
    """Per-unit ``cov(beta)`` in the design's own columns.

    The fit runs in an orthonormal basis ``Q`` with information ``H = Q' W Q``,
    and ``beta = B c`` with ``B = V S^-1`` (``back``). So
    ``cov(beta) = B (H + P)^+ B'`` -- Poisson's scale is fixed at one, which is
    why no dispersion estimate enters. On a design of full rank that is exactly ``(X' W X + Lambda)^-1``, the penalised
    posterior-mode covariance: ``B (H + B' Lambda B)^-1 B' =
    (A H A' + Lambda)^-1`` with ``A = B^-T = V S``, and ``A H A' = X' W X``.
    The pseudo-inverse is taken on the same eigenvalue cutoff the IRLS step
    uses, so a direction the weights have abandoned contributes nothing here
    either, and a column the design does not span gets an exactly zero standard
    error beside its exactly zero parameter.

    Parameters
    ----------
    information : jnp.ndarray, shape (n_units, rank, rank)
        ``H + P`` at the fitted parameters (``H`` alone when unpenalised).
    back : NDArray, shape (n_features, rank)
        ``V S^-1``, the map out of the orthonormal basis.
    rcond : float
        Eigenvalue cutoff relative to the largest, as in :func:`_solve`.

    Returns
    -------
    NDArray, shape (n_units, n_features, n_features)
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(information)
    largest = jnp.max(jnp.abs(eigenvalues), axis=1, keepdims=True)
    keep = eigenvalues > rcond * largest
    inverse = jnp.where(keep, 1.0 / jnp.where(keep, eigenvalues, 1.0), 0.0)
    # ``B V`` per unit, then ``(B V) diag(1/lambda) (B V)'``.
    rotated = jnp.einsum("pr,nrk->npk", jnp.asarray(back), eigenvectors)
    return np.asarray(jnp.einsum("npk,nk,nqk->npq", rotated, inverse, rotated))


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


def make_ridge_penalty(
    n_features: int,
    sigma: float,
    *,
    n_unpenalized: int = 1,
) -> NDArray:
    """Per-column ``lambda = 1 / sigma**2``, with the leading columns free.

    The prior is on the event-basis **coefficients**, and those are B-spline
    weights rather than the modulation itself, so ``sigma`` has two different
    readings on the curve and both matter:

    - **at one lag**, the kernel is ``sum_b beta_b B_b(lag)``, so its prior
      standard deviation is ``sigma * sqrt(sum_b B_b(lag)**2)`` -- about
      ``0.69 * sigma`` on the shipped basis, since a cubic B-spline partition of
      unity puts roughly that much squared weight at any lag;
    - **held across the window**, a sustained modulation of height ``h`` is
      ``beta = h`` in every coefficient, which the prior weights as
      ``exp(-h**2 n_basis / 2 sigma**2)``: its prior standard deviation is
      ``sigma / sqrt(n_basis)``, a factor of ``sqrt(11) = 3.3`` **tighter** than
      ``sigma`` on the shipped basis.

    A ``sigma`` chosen to look loose per coefficient is therefore tighter than
    it looks on a sustained response, which is why the default is
    :data:`DEFAULT_PENALTY_SIGMA` rather than the per-coefficient figure the
    peak modulations alone would suggest. :func:`implied_prior_sd` computes
    both readings for a given basis; quote it rather than restating numbers.

    The intercept carries the unit's baseline rate, which the prior has nothing
    to say about, so it is left unpenalised.

    Parameters
    ----------
    n_features : int
        Columns of the design.
    sigma : float
        Prior standard deviation of a penalised coefficient, on the log-rate
        scale. Must be positive. See :data:`DEFAULT_PENALTY_SIGMA`.
    n_unpenalized : int
        Leading columns to leave free -- 1 for the designs here, whose first
        column is the intercept.

    Returns
    -------
    NDArray, shape (n_features,)
        ``0`` on the free columns, ``1 / sigma**2`` on the rest.

    Raises
    ------
    ValueError
        If ``sigma`` is not positive, or ``n_unpenalized`` is out of range.
    """
    if not sigma > 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    if not 0 <= n_unpenalized <= n_features:
        raise ValueError(
            f"n_unpenalized must be in [0, {n_features}], got {n_unpenalized!r}"
        )
    penalty = np.full(n_features, 1.0 / float(sigma) ** 2)
    penalty[:n_unpenalized] = 0.0
    return penalty


def implied_prior_sd(
    sigma: float, spline_basis: NDArray, keep: NDArray | None = None
) -> dict[str, float]:
    """What a per-coefficient ``sigma`` says about the modulation curve.

    Parameters
    ----------
    sigma : float
        Prior standard deviation of one basis coefficient, on the log-rate
        scale.
    spline_basis : NDArray, shape (n_lags, n_basis)
        The basis the coefficients weight.
    keep : NDArray of bool, shape (n_lags,), optional
        Lags to summarise over -- the reported window, when it is narrower than
        the fit window. Defaults to every lag, which is **not** what a reader
        usually wants: the basis sums to one only between the first and last
        internal knot, so over the full fit window ``per_lag_min`` is the value
        at an edge the curves are never reported at and reads as a far tighter
        prior than any reported lag carries. Pass the reported window's mask
        (``|lag| <= REPORT_HALF_WINDOW_S``) for the figure a bundle should
        record.

    Returns
    -------
    dict
        ``per_lag`` : prior sd of the kernel at one lag (root mean square over
        the kept lags), with ``per_lag_min`` / ``per_lag_max`` over the same
        lags; ``sustained`` : prior sd of a modulation held across the window,
        ``sigma / sqrt(n_basis)``; and the 95 % rate-multiplier bounds each
        implies (``per_lag_multiplier_low`` / ``_high``, ``sustained_*``).

    Raises
    ------
    ValueError
        If ``sigma`` is not positive.
    """
    if not sigma > 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    basis = np.asarray(spline_basis, dtype=float)
    if keep is not None:
        basis = basis[np.asarray(keep, dtype=bool)]
    squared = (basis**2).sum(axis=1)
    per_lag = float(sigma * np.sqrt(squared.mean()))
    sustained = float(sigma / np.sqrt(basis.shape[1]))
    return {
        "per_lag": per_lag,
        "per_lag_min": float(sigma * np.sqrt(squared.min())),
        "per_lag_max": float(sigma * np.sqrt(squared.max())),
        "sustained": sustained,
        "per_lag_multiplier_low": float(np.exp(-1.96 * per_lag)),
        "per_lag_multiplier_high": float(np.exp(1.96 * per_lag)),
        "sustained_multiplier_low": float(np.exp(-1.96 * sustained)),
        "sustained_multiplier_high": float(np.exp(1.96 * sustained)),
    }


def _penalty_matrix(back: NDArray, penalty: NDArray | None) -> NDArray | None:
    """``P = B' Lambda B``, the prior's quadratic form in the fit's basis.

    Parameters
    ----------
    back : NDArray, shape (n_features, rank)
        ``B = V S^-1``, the map ``beta = B c``.
    penalty : NDArray, shape (n_features,), optional
        Per-column ``lambda``. ``None`` or all-zero means no penalty, reported
        as ``None`` so the unpenalised path adds nothing at all.

    Returns
    -------
    NDArray, shape (rank, rank), or None

    Raises
    ------
    ValueError
        If ``penalty`` is not one non-negative value per design column.
    """
    if penalty is None:
        return None
    penalty = np.asarray(penalty, dtype=float)
    if penalty.shape != (back.shape[0],):
        raise ValueError(
            f"penalty must have one value per design column "
            f"({back.shape[0]},), got {penalty.shape}"
        )
    if np.any(penalty < 0) or not np.all(np.isfinite(penalty)):
        raise ValueError("penalty must be finite and non-negative")
    if not penalty.any():
        return None
    return back.T @ (penalty[:, None] * back)


@jax.jit
def _penalty_quadratic(params: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
    """``c' P c`` per unit -- the penalty on the deviance scale."""
    return jnp.einsum("nr,rs,ns->n", params, matrix, params)


@jax.jit
def _poisson_deviance_from_params(
    design: jnp.ndarray, counts: jnp.ndarray, params: jnp.ndarray
) -> jnp.ndarray:
    """Held-out Poisson deviance without materialising predictions on host."""
    fitted_mean = jnp.exp(design @ params.T)
    ratio = jnp.where(counts > 0, counts / fitted_mean, 1.0)
    terms = counts * jnp.log(ratio) - (counts - fitted_mean)
    return 2.0 * terms.sum(axis=0)


def poisson_deviance_from_params(
    design: NDArray, counts: NDArray, params: NDArray
) -> NDArray:
    """Predict and score a batch on the configured JAX device.

    Held-out scoring used to form ``design @ params.T`` in NumPy.  On a real
    PFC session that is a tens-of-thousands by hundreds matrix, repeated once
    per fold for roughly a thousand null fits; doing the prediction on the
    host left the A100 idle between short fitting bursts.  This wrapper keeps
    the same float64 formula while transferring only the final per-unit score
    back to the host.

    Parameters
    ----------
    design : NDArray, shape (n_time, n_features)
    counts : NDArray, shape (n_time, n_units) or (n_time,)
    params : NDArray, shape (n_units, n_features)

    Returns
    -------
    NDArray, shape (n_units,)
        Unpenalised Poisson deviance on these rows.
    """
    require_configured_device()
    design = np.asarray(design, dtype=float)
    counts = np.asarray(counts, dtype=float)
    params = np.asarray(params, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    if design.ndim != 2 or params.ndim != 2 or counts.ndim != 2:
        raise ValueError(
            "design, counts and params must be two-dimensional after "
            f"normalizing counts; got {design.shape}, {counts.shape}, {params.shape}"
        )
    if design.shape[0] != counts.shape[0]:
        raise ValueError(
            f"design has {design.shape[0]} rows but counts has {counts.shape[0]}"
        )
    if params.shape != (counts.shape[1], design.shape[1]):
        raise ValueError(
            "params must have shape (n_units, n_features); got "
            f"{params.shape}, expected {(counts.shape[1], design.shape[1])}"
        )
    return np.asarray(
        _poisson_deviance_from_params(
            jnp.asarray(design), jnp.asarray(counts), jnp.asarray(params)
        )
    )


def _heldout_chunk_rows(n_models: int, n_units: int, rank: int) -> int:
    """Rows per chunk for a microbatch of design x fold fits."""
    widest = n_models * max(n_units, rank * rank, 1)
    return int(np.clip(_TARGET_CHUNK_ELEMENTS // widest, 256, 1 << 15))


def _pad_axis_one(array: jax.Array, n_pad: int) -> jax.Array:
    """Pad the time axis (axis 1) of a batched array with zeros."""
    if not n_pad:
        return array
    return jnp.pad(array, ((0, 0), (0, n_pad)) + ((0, 0),) * (array.ndim - 2))


def _sweep_prediction_batch(
    design_chunks: jax.Array,
    count_chunks: jax.Array,
    valid_chunks: jax.Array,
    params: jax.Array,
    mean_count: jax.Array,
    *,
    start: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One IRLS sweep for many design x fold fits, without likelihood work.

    The leading model dimension is a flattened ``design x fold`` batch.  The
    spike counts are shared, so they stay ``(chunk, time, unit)`` instead of
    being copied once per model.
    """
    n_models, _, _, rank = design_chunks.shape
    n_units = count_chunks.shape[-1]

    def body(carry, block):
        information, rhs, deviance = carry
        design, count, valid = block
        if start:
            mu = 0.5 * (count[None, :, :] + mean_count[:, None, :])
            eta = jnp.log(mu)
        else:
            eta = jnp.einsum("mlp,mnp->mln", design, params)
            mu = jnp.exp(eta)
        mu_clean = jnp.clip(mu, _FLOAT_EPS, jnp.inf)
        weight = mu_clean**2 / mu
        weighted_response = weight * eta + (mu_clean / mu) * (count[None, :, :] - mu)

        keep = valid[:, :, None]
        weight = jnp.where(keep, weight, 0.0)
        weighted_response = jnp.where(keep, weighted_response, 0.0)
        rhs = rhs + jnp.einsum("mlp,mln->mpn", design, weighted_response)
        information = information + jnp.einsum(
            "mln,mlp,mlq->mnpq", weight, design, design
        )

        ratio = jnp.clip(count[None, :, :] / mu, _FLOAT_EPS, jnp.inf)
        residual = 2.0 * (count[None, :, :] * jnp.log(ratio) - (count[None, :, :] - mu))
        deviance = deviance + jnp.sum(jnp.where(keep, residual, 0.0), axis=1)
        return (information, rhs, deviance), None

    init = (
        jnp.zeros((n_models, n_units, rank, rank), dtype=count_chunks.dtype),
        jnp.zeros((n_models, rank, n_units), dtype=count_chunks.dtype),
        jnp.zeros((n_models, n_units), dtype=count_chunks.dtype),
    )
    (information, rhs, deviance), _ = jax.lax.scan(
        body,
        init,
        (
            design_chunks.transpose(1, 0, 2, 3),
            count_chunks,
            valid_chunks.transpose(1, 0, 2),
        ),
    )
    return information, rhs, deviance


def _solve_prediction_batch(
    information: jax.Array, rhs: jax.Array, rcond: float
) -> jax.Array:
    """Symmetric pseudo-inverse solve over model and unit batch axes."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(information)
    largest = jnp.max(jnp.abs(eigenvalues), axis=-1, keepdims=True)
    keep = eigenvalues > rcond * largest
    inverse = jnp.where(keep, 1.0 / jnp.where(keep, eigenvalues, 1.0), 0.0)
    projected = jnp.einsum("mnji,mnj->mni", eigenvectors, rhs.transpose(0, 2, 1))
    return jnp.einsum("mnij,mnj->mni", eigenvectors, inverse * projected)


def _score_prediction_batch(
    designs: jax.Array,
    counts: jax.Array,
    folds: jax.Array,
    params: jax.Array,
    design_valid: jax.Array,
    *,
    n_folds: int,
    chunk_rows: int,
) -> jax.Array:
    """Held-out Poisson deviance without materializing full predictions."""
    n_designs, n_time, _ = designs.shape
    n_units = counts.shape[1]
    n_pad = (-n_time) % chunk_rows
    designs = _pad_axis_one(designs, n_pad)
    counts = jnp.pad(counts, ((0, n_pad), (0, 0)))
    folds = jnp.pad(folds, (0, n_pad), constant_values=-1)
    n_chunks = counts.shape[0] // chunk_rows
    design_chunks = designs.reshape(n_designs, n_chunks, chunk_rows, -1)
    count_chunks = counts.reshape(n_chunks, chunk_rows, n_units)
    fold_chunks = folds.reshape(n_chunks, chunk_rows)

    def body(total, block):
        design, count, fold = block
        eta = jnp.einsum("blp,bknp->bkln", design, params)
        fitted_mean = jnp.exp(eta)
        ratio = jnp.where(
            count[None, None, :, :] > 0,
            count[None, None, :, :] / fitted_mean,
            1.0,
        )
        terms = count[None, None, :, :] * jnp.log(ratio) - (
            count[None, None, :, :] - fitted_mean
        )
        score_fold = jnp.arange(n_folds)[:, None] == fold[None, :]
        keep = design_valid[:, None, None, None] & score_fold[None, :, :, None]
        return total + 2.0 * jnp.sum(jnp.where(keep, terms, 0.0), axis=2), None

    initial = jnp.zeros((n_designs, n_folds, n_units), dtype=counts.dtype)
    total, _ = jax.lax.scan(
        body,
        initial,
        (design_chunks.transpose(1, 0, 2, 3), count_chunks, fold_chunks),
    )
    return total


@partial(
    jax.jit,
    static_argnames=("n_folds", "maxiter", "chunk_rows"),
)
def _heldout_prediction_batch(
    designs: jax.Array,
    counts: jax.Array,
    folds: jax.Array,
    penalties: jax.Array,
    design_valid: jax.Array,
    *,
    n_folds: int,
    maxiter: int,
    tol: float,
    rcond: float,
    chunk_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fit and score a microbatch with folds and IRLS entirely on-device."""
    n_designs, n_time, n_features = designs.shape
    n_units = counts.shape[1]
    fold_ids = jnp.arange(n_folds)
    training_by_fold = folds[None, :] != fold_ids[:, None]
    training = jnp.broadcast_to(
        training_by_fold[None, :, :], (n_designs, n_folds, n_time)
    ).reshape(n_designs * n_folds, n_time)
    flat_design = (designs[:, None, :, :] * training_by_fold[None, :, :, None]).reshape(
        n_designs * n_folds, n_time, n_features
    )

    # Stable tall-skinny decomposition: QR reduces the large matrix to a
    # feature-sized R; only that small matrix needs an SVD for rank and the
    # minimum-norm map back to the original scientific coefficients.
    reduced = jnp.linalg.qr(flat_design, mode="r")
    _, singular_values, right = jnp.linalg.svd(reduced, full_matrices=False)
    n_training = training.sum(axis=1)
    tolerance = (
        singular_values[:, :1]
        * jnp.maximum(n_training[:, None], n_features)
        * jnp.finfo(designs.dtype).eps
    )
    kept = singular_values > tolerance
    inverse = jnp.where(kept, 1.0 / jnp.where(kept, singular_values, 1.0), 0.0)
    back = right.transpose(0, 2, 1) * inverse[:, None, :]
    basis = jnp.einsum("mtp,mpr->mtr", flat_design, back)
    rank = kept.sum(axis=1)

    penalty_by_fit = jnp.broadcast_to(
        penalties[:, None, :], (n_designs, n_folds, n_features)
    ).reshape(n_designs * n_folds, n_features)
    penalty_matrix = jnp.einsum("mpr,mp,mps->mrs", back, penalty_by_fit, back)

    n_pad = (-n_time) % chunk_rows
    basis = _pad_axis_one(basis, n_pad)
    training = jnp.pad(training, ((0, 0), (0, n_pad)))
    count_padded = jnp.pad(counts, ((0, n_pad), (0, 0)))
    n_chunks = count_padded.shape[0] // chunk_rows
    basis_chunks = basis.reshape(n_designs * n_folds, n_chunks, chunk_rows, n_features)
    valid_chunks = training.reshape(n_designs * n_folds, n_chunks, chunk_rows)
    count_chunks = count_padded.reshape(n_chunks, chunk_rows, n_units)
    mean_count = (
        jnp.einsum("mt,tn->mn", training[:, :n_time], counts) / n_training[:, None]
    )

    information, rhs, initial_deviance = _sweep_prediction_batch(
        basis_chunks,
        count_chunks,
        valid_chunks,
        jnp.zeros((n_designs * n_folds, n_units, n_features), dtype=counts.dtype),
        mean_count,
        start=True,
    )
    failed = jnp.isnan(initial_deviance)
    active = ~failed
    params = jnp.where(
        active[:, :, None],
        _solve_prediction_batch(
            information + penalty_matrix[:, None, :, :], rhs, rcond
        ),
        0.0,
    )
    penalized_previous = initial_deviance
    converged = jnp.zeros_like(active)
    diverged = jnp.zeros_like(active)
    n_iter = jnp.zeros_like(active, dtype=jnp.int32)

    def condition(carry):
        iteration, _, active_now, _, _, _, _ = carry
        return (iteration < maxiter) & jnp.any(active_now)

    def body(carry):
        (
            iteration,
            params_now,
            active_now,
            converged_now,
            diverged_now,
            iters,
            previous,
        ) = carry
        information_now, rhs_now, deviance_now = _sweep_prediction_batch(
            basis_chunks,
            count_chunks,
            valid_chunks,
            params_now,
            mean_count,
            start=False,
        )
        penalty_value = jnp.einsum(
            "mnr,mrs,mns->mn", params_now, penalty_matrix, params_now
        )
        penalized = deviance_now + penalty_value
        next_iteration = iteration + 1
        iters = jnp.where(active_now, next_iteration, iters)
        just_converged = active_now & (jnp.abs(penalized - previous) <= tol)
        converged_now = converged_now | just_converged
        diverged_now = diverged_now | (active_now & ~jnp.isfinite(penalized))
        next_active = active_now & ~just_converged & ~diverged_now
        previous = jnp.where(next_active, penalized, previous)
        stepped = _solve_prediction_batch(
            information_now + penalty_matrix[:, None, :, :], rhs_now, rcond
        )
        should_step = next_active & (next_iteration < maxiter)
        params_now = jnp.where(should_step[:, :, None], stepped, params_now)
        return (
            next_iteration,
            params_now,
            next_active,
            converged_now,
            diverged_now,
            iters,
            previous,
        )

    _, params, _, converged, diverged, n_iter, _ = jax.lax.while_loop(
        condition,
        body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            params,
            active,
            converged,
            diverged,
            n_iter,
            penalized_previous,
        ),
    )
    usable = converged & ~failed & ~diverged
    beta = jnp.einsum("mpr,mnr->mnp", back, params).reshape(
        n_designs, n_folds, n_units, n_features
    )
    usable = usable.reshape(n_designs, n_folds, n_units)
    n_iter = n_iter.reshape(n_designs, n_folds, n_units)
    usable = usable & jnp.all(jnp.isfinite(beta), axis=-1)
    score = _score_prediction_batch(
        designs,
        counts,
        folds,
        beta,
        design_valid,
        n_folds=n_folds,
        chunk_rows=chunk_rows,
    )
    usable = usable & jnp.isfinite(score) & design_valid[:, None, None]
    return (
        jnp.where(usable, score, jnp.nan),
        usable,
        n_iter,
        rank.reshape(n_designs, n_folds),
    )


def score_poisson_designs_heldout(
    designs: NDArray,
    response: PreparedHeldoutResponse,
    penalties: NDArray | None = None,
    *,
    design_batch_size: int = 4,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
    rcond: float = DEFAULT_RCOND,
) -> BatchedHeldoutResults:
    """Fit and score equal-width designs in stable-shape GPU microbatches.

    This prediction-only path deliberately omits covariance, log-factorials,
    likelihood and AIC.  It batches all folds and several changing null
    designs, keeps the response resident on-device, and returns only the small
    per-fold scores and convergence diagnostics to the host.
    """
    require_configured_device()
    design_array = np.asarray(designs, dtype=float)
    if design_array.ndim == 2:
        design_array = design_array[None, :, :]
    if design_array.ndim != 3:
        raise ValueError(
            "designs must have shape (n_designs, n_time, n_features), got "
            f"{design_array.shape}"
        )
    n_designs, n_time, n_features = design_array.shape
    if n_designs < 1 or design_batch_size < 1:
        raise ValueError(
            "at least one design and a positive design_batch_size are required"
        )
    if n_time != response.n_time:
        raise ValueError(
            f"designs have {n_time} rows but the prepared response has "
            f"{response.n_time}"
        )
    if penalties is None:
        penalty_array = np.zeros((n_designs, n_features), dtype=float)
    else:
        penalty_array = np.asarray(penalties, dtype=float)
        if penalty_array.ndim == 1:
            penalty_array = np.broadcast_to(
                penalty_array, (n_designs, n_features)
            ).copy()
        if penalty_array.shape != (n_designs, n_features):
            raise ValueError(
                "penalties must have one value per design column, per design; "
                f"expected {(n_designs, n_features)}, got {penalty_array.shape}"
            )
        if np.any(penalty_array < 0) or not np.all(np.isfinite(penalty_array)):
            raise ValueError("penalties must be finite and non-negative")

    per_fold_parts = []
    converged_parts = []
    n_iter_parts = []
    rank_parts = []
    n_models = design_batch_size * response.n_folds
    chunk_rows = _heldout_chunk_rows(n_models, response.n_units, n_features)
    for start in range(0, n_designs, design_batch_size):
        stop = min(start + design_batch_size, n_designs)
        n_actual = stop - start
        design_batch = np.zeros((design_batch_size, n_time, n_features), dtype=float)
        penalty_batch = np.zeros((design_batch_size, n_features), dtype=float)
        valid = np.zeros(design_batch_size, dtype=bool)
        design_batch[:n_actual] = design_array[start:stop]
        penalty_batch[:n_actual] = penalty_array[start:stop]
        valid[:n_actual] = True
        scored, converged, n_iter, rank = _heldout_prediction_batch(
            jax.device_put(design_batch),
            response.counts,
            response.folds,
            jax.device_put(penalty_batch),
            jax.device_put(valid),
            n_folds=response.n_folds,
            maxiter=maxiter,
            tol=tol,
            rcond=rcond,
            chunk_rows=chunk_rows,
        )
        scored_host, converged_host, n_iter_host, rank_host = jax.device_get(
            (scored, converged, n_iter, rank)
        )
        per_fold_parts.append(np.asarray(scored_host[:n_actual]))
        converged_parts.append(np.asarray(converged_host[:n_actual]))
        n_iter_parts.append(np.asarray(n_iter_host[:n_actual]))
        rank_parts.append(np.asarray(rank_host[:n_actual]))

    per_fold = np.concatenate(per_fold_parts, axis=0)
    return BatchedHeldoutResults(
        per_fold=per_fold,
        total=per_fold.sum(axis=1),
        converged=np.concatenate(converged_parts, axis=0),
        n_iter=np.concatenate(n_iter_parts, axis=0),
        rank=np.concatenate(rank_parts, axis=0),
    )


def fit_poisson_glms(
    design: NDArray,
    counts: NDArray,
    *,
    penalty: NDArray | None = None,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
    rcond: float = DEFAULT_RCOND,
    chunk_rows: int | None = None,
    compute_covariance: bool = True,
) -> BatchedGLMResults:
    """Fit one Poisson GLM per column of ``counts`` against a shared design.

    Unpenalised, this reproduces ``statsmodels.api.GLM(y, X,
    family=Poisson()).fit()`` unit by unit: same log link, same starting mean,
    same IRLS step, same convergence rule, same pseudo-inverse treatment of a
    rank-deficient design. With ``penalty``, it is the posterior mode of the
    same model under a Gaussian prior on the named coefficients; see the module
    docstring for how the prior is carried into the orthonormal basis.

    Parameters
    ----------
    design : NDArray, shape (n_time, n_features)
        Design matrix, already restricted to the bins being fit. Shared by
        every unit.
    counts : NDArray, shape (n_time, n_units) or (n_time,)
        Spike counts per bin and unit. A 1-D array is one unit.
    penalty : NDArray, shape (n_features,), optional
        Per-column ``lambda = 1 / sigma**2`` of a Gaussian prior on the
        coefficients **in the design's own columns**, penalising
        ``(1/2) sum_j lambda_j beta_j**2``. Zero on a column leaves it free;
        ``None`` (the default) or an all-zero vector is the unpenalised fit,
        bit for bit. See :func:`make_ridge_penalty`.
    maxiter : int
        Maximum IRLS steps per unit.
    tol : float
        Absolute change of the *penalised* deviance
        (``deviance + beta' Lambda beta``) below which a unit has converged.
        Equal to the deviance change when the penalty is zero.
    rcond : float
        Eigenvalue cutoff for the pseudo-inverse solve, relative to the largest
        eigenvalue of ``Q' W Q``.
    chunk_rows : int, optional
        Rows of the time axis per chunk. Defaults to a size that keeps the
        working arrays around 64 MB.
    compute_covariance : bool
        Whether to compute the inverse-information covariance and standard
        errors after convergence. Held-out prediction needs parameters only
        and disables this extra eigendecomposition; full-data fits keep it.

    Returns
    -------
    BatchedGLMResults
        Units whose fit is not defined -- no spikes at all on these bins, so
        statsmodels' first deviance is ``nan`` and it raises, or an iteration
        that ran away to a non-finite deviance -- come back with ``nan``
        parameters, deviance, llf and aic, and ``converged`` False. The rest of
        the batch is unaffected, so a caller reads each unit x model cell on
        its own.

    Raises
    ------
    ValueError
        If the design and counts disagree on the number of rows, or if
        ``penalty`` is not one finite non-negative value per design column.
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
        )

    # The fit runs in an orthonormal basis of the design's column space; its
    # width is the design's rank, on statsmodels' own rank tolerance, so
    # ``df_model``/``df_resid``/``aic`` are counted exactly as statsmodels
    # counts them -- and computed once for a design every unit shares.
    basis, back, rank = orthonormal_design_basis(design_array)
    # The prior is stated in the design's own columns; ``P = B' Lambda B`` is
    # the same quadratic form written in the basis the fit runs in.
    penalty_matrix = _penalty_matrix(back, penalty)
    penalty_device = None if penalty_matrix is None else jnp.asarray(penalty_matrix)

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
    information, rhs, deviance, _ = _sweep(
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

    def step(information: jnp.ndarray) -> jnp.ndarray:
        """The IRLS step, ``(H + P)^+ r``, for the units still active."""
        if penalty_device is not None:
            information = information + penalty_device
        return _solve(information, rhs, rcond)

    def penalty_of(params: jnp.ndarray) -> NDArray:
        """``beta' Lambda beta`` per unit, on the deviance scale."""
        if penalty_device is None:
            return np.zeros(n_units)
        return np.asarray(_penalty_quadratic(params, penalty_device))

    active_device = jnp.asarray(active)
    params = jnp.where(active_device[:, None], step(information), params)

    diverged = np.zeros(n_units, dtype=bool)
    for iteration in range(1, maxiter + 1):
        information, rhs, deviance, llf = _sweep(
            design_chunks, count_chunks, valid_chunks, params, mean_count, False
        )
        deviance_host = np.asarray(deviance)
        # The iteration maximises the penalised log-likelihood, so that is what
        # has to settle; the reported deviance stays the unpenalised one.
        penalized_host = deviance_host + penalty_of(params)
        deviance_at_params = np.where(active, deviance_host, deviance_at_params)
        llf_kernel = np.where(active, np.asarray(llf), llf_kernel)
        n_iter = np.where(active, iteration, n_iter)

        # ``_check_convergence``: np.allclose(prev, current, atol=tol, rtol=0).
        just_converged = active & (np.abs(penalized_host - deviance_prev) <= tol)
        converged |= just_converged
        # An iteration that ran away to a non-finite deviance has no fit to
        # report and can never meet the convergence rule, so it is stopped here
        # rather than held to ``maxiter`` at the whole batch's expense.
        diverged |= active & ~np.isfinite(penalized_host)
        active = active & ~just_converged & ~diverged
        if not active.any() or iteration == maxiter:
            break

        deviance_prev = np.where(active, penalized_host, deviance_prev)
        active_device = jnp.asarray(active)
        params = jnp.where(active_device[:, None], step(information), params)

    # Back out of the orthonormal basis: ``beta = V S^-1 c``, the minimum-norm
    # parameters in the design's own columns.
    params_host = np.asarray(params) @ back.T
    llf_host = llf_kernel - np.asarray(log_factorial)
    aic = -2.0 * llf_host + 2.0 * rank
    # A unit whose iteration ran away has no usable fit; it is reported like
    # one with no spikes at all rather than as a finite near-miss.
    failed = failed | diverged

    covariance = None
    standard_errors = None
    if compute_covariance:
        covariance = _parameter_covariance(
            information if penalty_device is None else information + penalty_device,
            back,
            rcond,
        )
        # A variance can come back a hair below zero on a direction the weights
        # have all but abandoned; the parameter there is zero and so is its error.
        standard_errors = np.sqrt(np.clip(np.einsum("npp->np", covariance), 0.0, None))

    if failed.any():
        params_host = params_host.copy()
        if standard_errors is not None and covariance is not None:
            standard_errors = standard_errors.copy()
            covariance = covariance.copy()
            standard_errors[failed] = np.nan
            covariance[failed] = np.nan
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
        bse=standard_errors,
        covariance=covariance,
    )
