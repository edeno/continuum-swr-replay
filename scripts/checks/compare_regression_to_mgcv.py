#!/usr/bin/env python
"""Compare the shared regression fitter with mgcv on small synthetic data sets.

Four data sets, each built from a fixed seed (:func:`make_data`):

- ``poisson_one_smooth``: Poisson counts in 15 ms bins of synthetic stops, a
  smooth of the time since the stop began and a 0/1 covariate (the stop was
  rewarded), with a log(exposure) offset;
- ``poisson_two_smooths``: the same with a second smooth, of the time until the
  stop ends;
- ``bernoulli_one_smooth``: 0/1 outcomes with one smooth;
- ``gaussian_one_smooth``: a continuous outcome with one smooth (its scale
  profiled).

Every smooth is a cubic P-spline on 15 basis functions over [0, 4] with a
second-order difference penalty and a sum-to-zero constraint. mgcv fits them
with ``gam(..., bs = "ps", method = "REML")`` and the same knots
(``compare_regression_to_mgcv.R``, run with the ``csr-mgcv`` environment's
Rscript); the shared fitter selects the penalties with
:func:`~continuum_swr_replay.analysis.regression.reml.select_penalties`. The
script prints, for each data set, the agreement in the log penalty, the edf,
the REML score, the centered curve and its standard error on a grid, the 0/1
coefficient and its standard error, and the Gaussian scale, and writes mgcv's
outputs to ``tests/analysis/regression/data/mgcv/<name>.json``, the fixtures
``tests/analysis/regression/test_reml.py`` compares with (the suite does not
need R: it rebuilds the data from the same seeds).

mgcv scales each penalty matrix before it estimates the strength, so its
``sp`` multiplies its own matrix: the strength on the shared fitter's matrix is
``sp * tr(S_mgcv) / tr(S)``. The shared fitter's score leaves out the constants
the likelihood leaves out; for the Poisson, ``score + sum(log y!)`` is mgcv's
``gcv.ubre``.

Usage:
    python scripts/checks/compare_regression_to_mgcv.py
    python scripts/checks/compare_regression_to_mgcv.py --out-dir /tmp/mgcv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    Likelihood,
    Poisson,
)
from continuum_swr_replay.analysis.regression.pspline import (
    PSplineBasis,
    sum_to_zero_constraint,
)
from continuum_swr_replay.analysis.regression.reml import (
    PenaltyBlock,
    SessionData,
    select_penalties,
)

REPO = Path(__file__).resolve().parents[2]
R_SCRIPT = Path(__file__).with_suffix(".R")
FIXTURE_DIR = REPO / "tests" / "analysis" / "regression" / "data" / "mgcv"
RSCRIPT = Path("/home/edeno/miniconda3/envs/csr-mgcv/bin/Rscript")

N_BASIS = 15
LO, HI = 0.0, 4.0
GRID = np.round(np.arange(0.05, 3.951, 0.05), 2)
BIN_S = 0.015
N_STOPS = 350

#: Data set name -> its seed.
DATA_SETS: dict[str, int] = {
    "poisson_one_smooth": 1,
    "poisson_two_smooths": 3,
    "bernoulli_one_smooth": 5,
    "gaussian_one_smooth": 7,
}
#: The smooths of each data set, in the order of mgcv's terms.
SMOOTHS: dict[str, tuple[str, ...]] = {
    "poisson_one_smooth": ("s",),
    "poisson_two_smooths": ("s", "u"),
    "bernoulli_one_smooth": ("s",),
    "gaussian_one_smooth": ("s",),
}


def _stop_bins(rng: np.random.Generator) -> pd.DataFrame:
    """15 ms bins of ``N_STOPS`` synthetic stops.

    Stop durations are lognormal (sigma 1, median 1 s when rewarded, 0.3 s
    when not, half rewarded); bins run from each stop's start, the last one
    partial.

    Returns
    -------
    pd.DataFrame, shape (n_bins, 4)
        ``s`` (time since the stop began, s), ``u`` (time until it ends, s),
        ``expo`` (the bin's length, s) and ``rew`` (0/1), for the bins with
        ``s`` and ``u`` below 4 s.
    """
    rewarded = rng.random(N_STOPS) < 0.5
    duration = np.where(rewarded, 1.0, 0.3) * np.exp(rng.normal(size=N_STOPS))
    duration = np.maximum(duration, BIN_S)
    n_bins = np.ceil(duration / BIN_S).astype(int)
    stop = np.repeat(np.arange(N_STOPS), n_bins)
    k = np.arange(n_bins.sum()) - np.repeat(np.cumsum(n_bins) - n_bins, n_bins)
    start = k * BIN_S
    end = np.minimum(start + BIN_S, duration[stop])
    s = 0.5 * (start + end)
    u = duration[stop] - s
    keep = (s < HI) & (u < HI)
    return pd.DataFrame(
        {
            "s": s[keep],
            "u": u[keep],
            "expo": (end - start)[keep],
            "rew": rewarded[stop][keep].astype(float),
        }
    )


def make_data(name: str) -> pd.DataFrame:
    """The data set ``name``, built from its seed in ``DATA_SETS``.

    The Poisson sets have about 300 events with a log rate that rises fourfold
    over the first second of a stop (``log 4 (1 - exp(-s / 0.5))``), a reward
    log rate ratio of 0.26 and, for two smooths, a halving over the last 0.3 s
    before the stop ends. The Bernoulli set has 800 rows with ``logit p =
    1.2 sin(1.5 s) - 0.3``, the Gaussian 400 rows with mean ``sin(2 s)`` and
    noise SD 0.5, ``s`` uniform on [0, 4).

    Returns
    -------
    pd.DataFrame, shape (n_rows, 5)
        Columns ``s``, ``u``, ``expo``, ``rew`` and ``y``.
    """
    rng = np.random.default_rng(DATA_SETS[name])
    if name.startswith("poisson"):
        frame = _stop_bins(rng)
        s, u = frame["s"].to_numpy(), frame["u"].to_numpy()
        log_rate = np.log(4.0) * (1.0 - np.exp(-s / 0.5)) + 0.26 * frame["rew"]
        if name == "poisson_two_smooths":
            log_rate += np.log(0.5) / (1.0 + np.exp(-(0.3 - u) / 0.03))
        expected = frame["expo"] * np.exp(log_rate)
        frame["y"] = rng.poisson(300.0 * expected / expected.sum()).astype(float)
        return frame
    n_rows = 800 if name == "bernoulli_one_smooth" else 400
    s = rng.uniform(LO, HI, size=n_rows)
    if name == "bernoulli_one_smooth":
        p = 1.0 / (1.0 + np.exp(-(1.2 * np.sin(1.5 * s) - 0.3)))
        y = (rng.uniform(size=n_rows) < p).astype(float)
    else:
        y = np.sin(2.0 * s) + rng.normal(scale=0.5, size=n_rows)
    zeros = np.zeros(n_rows)
    return pd.DataFrame({"s": s, "u": s, "expo": zeros + 1.0, "rew": zeros, "y": y})


def fit_ours(name: str, frame: pd.DataFrame) -> dict:
    """The shared fitter's REML fit of data set ``name``.

    Returns
    -------
    dict
        ``log_penalty`` and ``edf`` (shape ``(n_smooths,)``), ``trace_S`` (the
        trace of each smooth's penalty matrix), ``reml`` (the score plus the
        constants mgcv's score keeps: ``sum log y!`` for the Poisson),
        ``curve_<v>`` and ``se_curve_<v>`` on ``GRID`` for each smooth ``v``,
        ``coef_rew`` and ``se_rew`` for the Poisson sets, ``scale``, and the
        search's ``converged`` and ``in_null_space``.
    """
    basis = PSplineBasis(LO, HI, N_BASIS)
    y = frame["y"].to_numpy(dtype=float)
    columns = [np.ones((len(frame), 1))]
    poisson = name.startswith("poisson")
    if poisson:
        columns.append(frame[["rew"]].to_numpy(dtype=float))
    blocks, constraints = [], []
    start = len(columns)
    for variable in SMOOTHS[name]:
        B = basis.design(frame[variable].to_numpy(dtype=float))
        Z = sum_to_zero_constraint(B)
        constraints.append(Z)
        columns.append(B @ Z)
        blocks.append(
            PenaltyBlock(
                np.arange(start, start + Z.shape[1]), Z.T @ basis.penalty() @ Z
            )
        )
        start += Z.shape[1]
    X = np.hstack(columns)
    lik: Likelihood
    if poisson:
        lik, data = Poisson(), SessionData(X, y, np.log(frame["expo"].to_numpy()))
    elif name == "bernoulli_one_smooth":
        lik, data = Bernoulli(), SessionData(X, y)
    else:
        lik, data = Gaussian(), SessionData(X, y)
    result = select_penalties(lik, [data], blocks)
    session = result.sessions[0]
    out = {
        "log_penalty": result.rho,
        "edf": session.edf,
        "trace_S": np.array([np.trace(block.S) for block in blocks]),
        "reml": result.score + (float(gammaln(y + 1.0).sum()) if poisson else 0.0),
        "scale": session.scale,
        "converged": result.search.converged,
        "in_null_space": session.in_null_space,
    }
    grid_basis = basis.design(GRID)
    for variable, block, Z in zip(SMOOTHS[name], blocks, constraints, strict=True):
        C = grid_basis @ Z
        cols = block.columns
        out[f"curve_{variable}"] = C @ session.fit.beta[cols]
        covariance = session.covariance[np.ix_(cols, cols)]
        out[f"se_curve_{variable}"] = np.sqrt(
            np.einsum("ij,jk,ik->i", C, covariance, C)
        )
    if poisson:
        out["coef_rew"] = float(session.fit.beta[1])
        out["se_rew"] = float(np.sqrt(session.covariance[1, 1]))
    return out


def agreement(name: str, ours: dict, mgcv: dict) -> dict[str, float]:
    """The differences between the shared fitter's fit and mgcv's.

    Returns
    -------
    dict[str, float]
        ``log_penalty`` (largest absolute difference of the log strengths on
        the shared fitter's penalty matrices), ``edf`` (largest absolute),
        ``reml`` (absolute), ``curve`` and ``se_curve`` (largest absolute over
        the grid and the smooths), and, where they exist, ``coef_rew``,
        ``se_rew`` and ``scale`` (relative).
    """
    sp = np.atleast_1d(mgcv["sp"])
    trace_mgcv = np.atleast_1d(mgcv["trace_S"])
    mgcv_log_penalty = np.log(sp * trace_mgcv / ours["trace_S"])
    out = {
        "log_penalty": float(np.max(np.abs(ours["log_penalty"] - mgcv_log_penalty))),
        "edf": float(np.max(np.abs(ours["edf"] - np.atleast_1d(mgcv["edf"])))),
        "reml": abs(ours["reml"] - mgcv["reml"]),
        "curve": max(
            float(np.max(np.abs(ours[f"curve_{v}"] - np.asarray(mgcv[f"curve_{v}"]))))
            for v in SMOOTHS[name]
        ),
        "se_curve": max(
            float(
                np.max(
                    np.abs(ours[f"se_curve_{v}"] - np.asarray(mgcv[f"se_curve_{v}"]))
                )
            )
            for v in SMOOTHS[name]
        ),
    }
    if "coef_rew" in mgcv:
        out["coef_rew"] = abs(ours["coef_rew"] / mgcv["coef_rew"] - 1.0)
        out["se_rew"] = abs(ours["se_rew"] / mgcv["se_rew"] - 1.0)
    if name == "gaussian_one_smooth":
        out["scale"] = abs(ours["scale"] / mgcv["scale"] - 1.0)
    return out


def run_mgcv(name: str, frame: pd.DataFrame, rscript: Path, work: Path) -> dict:
    """mgcv's fit of data set ``name``, read from the R script's JSON."""
    data_path, grid_path = work / f"{name}.csv", work / "grid.csv"
    out_path = work / f"{name}.json"
    frame.to_csv(data_path, index=False, float_format="%.17g")
    pd.DataFrame({"x": GRID}).to_csv(grid_path, index=False, float_format="%.17g")
    subprocess.run(
        [
            str(rscript),
            str(R_SCRIPT),
            str(data_path),
            name,
            str(N_BASIS),
            str(LO),
            str(HI),
            str(grid_path),
            str(out_path),
        ],
        check=True,
    )
    return json.loads(out_path.read_text())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIXTURE_DIR,
        help="Where mgcv's outputs are written (default: the test fixtures).",
    )
    parser.add_argument("--rscript", type=Path, default=RSCRIPT)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        for name in DATA_SETS:
            frame = make_data(name)
            mgcv = run_mgcv(name, frame, args.rscript, Path(work))
            (args.out_dir / f"{name}.json").write_text(
                json.dumps(mgcv, indent=1) + "\n"
            )
            ours = fit_ours(name, frame)
            diff = agreement(name, ours, mgcv)
            print(
                f"{name}: {len(frame)} rows, y total {frame['y'].sum():.0f}; "
                f"mgcv converged {mgcv['converged']} in {mgcv['outer_iterations']} "
                f"outer steps; ours converged {ours['converged']}"
            )
            print(
                f"  log penalty ours {np.round(ours['log_penalty'], 6)}, "
                f"edf ours {np.round(ours['edf'], 6)}, "
                f"mgcv {np.round(np.atleast_1d(mgcv['edf']), 6)}"
            )
            print("  " + ", ".join(f"{k} {v:.1e}" for k, v in diff.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
