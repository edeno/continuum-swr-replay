#!/usr/bin/env python
"""Fit a one-term Poisson model with neurospatial's REML and write the fixture.

Run with neurospatial's own interpreter (this repository does not install or
import neurospatial)::

    /cumulus/edeno/neurospatial/.venv/bin/python scripts/checks/compare_regression_to_neurospatial.py

The data (:func:`make_data`, NumPy and SciPy only, so the suite can rebuild them
from the same seed): three units' spike counts in 100 position bins over
[0, 1] with unequal occupancy, each a Poisson draw around its own smooth place
field. The term is a cubic P-spline on 12 basis functions with a second-order
difference penalty, written in the penalty's eigenbasis because neurospatial's
fitter takes a diagonal penalty: columns ``B U`` and weights ``d`` from
``P = U diag(d) U'``, the two null weights set to zero (the constant and the
line are unpenalized). The three units share the penalty, as the shared
fitter's sessions do.

neurospatial (``encoding/_glm_numpy.py``) minimizes, summed over units,

    -sum(n eta - o exp(eta)) + lam/2 sum(d gamma^2) - rank/2 log(lam) + 1/2 log|H + lam diag(d) + 1e-10 I|

over ``log lam`` in [-8, 20] with SciPy's bounded scalar minimizer at
``xatol = 1e-3``. The shared fitter's score differs by terms that do not
depend on ``lam``: the offset term ``sum n log o`` of the likelihood,
``-1/2 log|S|_+`` at unit strength and ``-M_p/2 log(2 pi)`` per unit, and by
the 1e-10 ridge on the Hessian. The fixture records

- ``select_penalty_by_reml``'s own result (its ``xatol`` limits ``log lam`` to
  about 1e-3);
- the same objective minimized with ``xatol = 1e-10``, and the curves (the log
  rate per bin and unit) at that minimum, from neurospatial's Newton fit;
- the objective on a grid of ``log lam``, for the comparison of the scores.

Writes ``tests/analysis/regression/data/neurospatial/one_term_poisson.json``
(``--out`` to change it), which
``tests/analysis/regression/test_reml.py::test_one_penalty_matches_neurospatial_fixture``
reads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

REPO = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO
    / "tests"
    / "analysis"
    / "regression"
    / "data"
    / "neurospatial"
    / "one_term_poisson.json"
)
NEUROSPATIAL = Path("/cumulus/edeno/neurospatial")

SEED = 11
N_BINS = 100
N_UNITS = 3
N_BASIS = 12
DEGREE = 3
#: neurospatial's inner fit: Newton steps and relative objective decrease.
MAX_ITER = 200
FIT_TOL = 1e-13
#: The tight minimization of neurospatial's objective, and the score grid.
TIGHT_XATOL = 1e-10
SCORE_GRID = np.arange(-2.0, 10.5, 1.0)


def make_data() -> dict[str, np.ndarray]:
    """The three units' counts, the occupancy and the basis, from ``SEED``.

    Returns
    -------
    dict
        ``x`` (bin centers, shape ``(N_BINS,)``), ``occupancy`` (s, shape
        ``(N_BINS,)``), ``counts`` (shape ``(N_BINS, N_UNITS)``), ``basis``
        (the B-splines at ``x``, shape ``(N_BINS, N_BASIS)``), ``rotation``
        (``U``, shape ``(N_BASIS, N_BASIS)``) and ``weights`` (``d``, shape
        ``(N_BASIS,)``, the first two zero), with ``U diag(d) U'`` the
        second-order difference penalty.
    """
    rng = np.random.default_rng(SEED)
    x = (np.arange(N_BINS) + 0.5) / N_BINS
    occupancy = rng.uniform(0.2, 2.0, size=N_BINS)
    centers = np.array([0.25, 0.5, 0.8])
    widths = np.array([0.08, 0.15, 0.1])
    peaks = np.array([6.0, 3.0, 10.0])
    rate = 0.5 + peaks * np.exp(-0.5 * ((x[:, None] - centers) / widths) ** 2)
    counts = rng.poisson(occupancy[:, None] * rate).astype(float)

    n_intervals = N_BASIS - DEGREE
    knots = np.arange(-DEGREE, n_intervals + DEGREE + 1) / n_intervals
    basis = BSpline.design_matrix(x, knots, DEGREE, extrapolate=False).toarray()
    difference = np.diff(np.eye(N_BASIS), n=2, axis=0)
    weights, rotation = np.linalg.eigh(difference.T @ difference)
    weights[:2] = 0.0
    return {
        "x": x,
        "occupancy": occupancy,
        "counts": counts,
        "basis": basis,
        "rotation": rotation,
        "weights": weights,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=FIXTURE)
    args = parser.parse_args(argv)

    import scipy
    import scipy.optimize
    from neurospatial.encoding import _glm, _glm_numpy

    data = make_data()
    B = data["basis"] @ data["rotation"]
    d = data["weights"]
    rank = N_BASIS - 2
    constant_base = data["rotation"].T @ np.ones(N_BASIS)
    np.testing.assert_allclose(B @ constant_base, 1.0, rtol=0, atol=1e-12)
    counts, occupancy = data["counts"], data["occupancy"]

    penalty, objective, at_boundary = _glm_numpy.select_penalty_by_reml(
        counts,
        occupancy,
        B,
        d,
        rank,
        constant_base=constant_base,
        max_iter=MAX_ITER,
        tol=FIT_TOL,
    )
    extra = (counts, occupancy, B, d, rank, constant_base, MAX_ITER, FIT_TOL)
    tight = scipy.optimize.minimize_scalar(
        _glm_numpy._reml_objective_numpy,
        bounds=_glm._LOG_PENALTY_BOUNDS,
        method="bounded",
        args=extra,
        options={"xatol": TIGHT_XATOL},
    )
    coeffs, eta, _mu, n_iter, _step, converged = _glm_numpy._newton_fit_numpy(
        counts,
        occupancy,
        B,
        np.exp(tight.x) * d,
        constant_base,
        MAX_ITER,
        FIT_TOL,
    )
    if not converged:
        raise RuntimeError("neurospatial's inner fit did not converge at the minimum")
    grid_scores = [
        float(_glm_numpy._reml_objective_numpy(r, *extra)) for r in SCORE_GRID
    ]
    commit = subprocess.run(
        ["git", "-C", str(NEUROSPATIAL), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    fixture = {
        "neurospatial_commit": commit,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "seed": SEED,
        "counts_sum": float(counts.sum()),
        "log_penalty_bounds": list(_glm._LOG_PENALTY_BOUNDS),
        "select_xatol": _glm._REML_XATOL,
        "hessian_jitter": _glm._HESSIAN_JITTER,
        "fit_tol": FIT_TOL,
        "select_log_penalty": float(np.log(penalty)),
        "select_objective": float(objective),
        "select_at_boundary": bool(at_boundary),
        "tight_xatol": TIGHT_XATOL,
        "tight_log_penalty": float(tight.x),
        "tight_objective": float(tight.fun),
        "tight_n_newton_steps": int(n_iter),
        "log_rate": eta.tolist(),
        "coefficients": coeffs.tolist(),
        "score_grid_log_penalty": SCORE_GRID.tolist(),
        "score_grid_objective": grid_scores,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1) + "\n")
    print(
        f"neurospatial {commit}: select_penalty_by_reml log lambda "
        f"{fixture['select_log_penalty']:.6f} (objective {objective:.8f}); "
        f"xatol {TIGHT_XATOL:g}: {tight.x:.10f} (objective {tight.fun:.10f}); "
        f"wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
