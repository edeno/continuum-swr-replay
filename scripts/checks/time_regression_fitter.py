#!/usr/bin/env python
"""Time the shared regression fitter on fixed synthetic workloads: wall time and
peak memory, the first call apart from the calls after it.

Every case calls a public entry of ``continuum_swr_replay.analysis.regression``
on inputs built from a fixed seed, so the same cases time any fitter behind
that interface (the NumPy solver today, a JAX one later):

1. ``fit_<family>``: one session at fixed log penalties (``reml.refit_session``):
   ``rate`` (37,000 bins of visits, two P-splines of the clocks since arrival
   and until departure, 43 and 33 basis functions, a weakly penalized 0/1
   slope, an offset; 76 columns), ``content`` (400 events of nine candidate
   arms under the grouped softmax: the arm category, a P-spline of a clock,
   seven slopes; 30 columns), ``bernoulli`` and ``gaussian`` (20,000 rows, a
   P-spline of 20 basis functions and three slopes).
2. ``fit_sessions_30``: ``sessions.fit_sessions`` over 30 rate sessions (5
   animals x 2 days x 3 sessions of about 110 events in visits of real length,
   one P-spline of the clock since arrival, 43 basis functions over 0 to 20 s).
3. ``calibration_30``: ``calibration.log_penalty_for_edf`` over the same 30
   sessions, to a median edf of 6 (about 18 passes of the loop).
4. ``select_cost``: ``reml.select_penalties`` on the case of "Cost" in
   ``docs/design/regression_fitter.md``: ``--cost-sessions`` sessions (242 by
   default) of 40,000 bins, two smooths, 56 coefficients (4.3 GB of design at
   242 sessions).
5. ``pytest_regression``: ``pytest tests/analysis/regression``, once with an
   empty JAX compilation cache and once with the cache that run filled.

JAX runs on ``JAX_PLATFORMS``, the CPU when it is unset. Each case runs in its
own process, given ``--threads`` threads (the BLAS
variables and the process's CPU set, the first that many of ``--cores``; JAX's
CPU thread pool follows the CPU set). A process builds the inputs (not timed),
makes the first call (for a JAX fitter it compiles) and then ``--reps`` more;
the report gives the first call, the median and range of the others, and the
first less the median. That difference is a compile time only for a JAX
fitter; for the NumPy fitter it is warm-up and noise (caches, page faults).
Peak memory is the whole process's maximum resident set (``getrusage`` of the
child, what ``/usr/bin/time -v`` reports): the interpreter, the imports and the
case's inputs are in it, not only the fitter's own arrays. The load average is
recorded before and after each case: on a shared machine the times are under
load and say so. The output's ``meta`` names the commit and whether a tracked
file differed from it (``tree_modified``), so a time is tied to its code.

Usage (from the repository root):

    python scripts/checks/time_regression_fitter.py --out timing.json
    python scripts/checks/time_regression_fitter.py --case fit_rate \\
        --case fit_content --threads 1 --threads 6 --cores 10-15
    python scripts/checks/time_regression_fitter.py --case select_cost \\
        --cost-sessions 24
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.calibration import log_penalty_for_edf
from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.reml import (
    SessionData,
    refit_session,
    select_penalties,
)
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.analysis.regression.terms import (
    Category,
    Offset,
    Slope,
    Smooth,
    build_design,
    log_penalty_from_prior_sd,
)
from continuum_swr_replay.runs.device import default_jax_platforms
from continuum_swr_replay.runs.provenance import git_commit, git_dirty

REPO = Path(__file__).resolve().parents[2]

#: Case name -> what it times, for the table.
CASES = {
    "fit_rate": "one rate session, 37,000 x 76, refit_session",
    "fit_content": "one content session, 400 events x 9 arms, refit_session",
    "fit_bernoulli": "one Bernoulli session, 20,000 x 23, refit_session",
    "fit_gaussian": "one Gaussian session, 20,000 x 23, refit_session",
    "fit_sessions_30": "fit_sessions over 30 rate sessions",
    "calibration_30": "log_penalty_for_edf over 30 rate sessions",
    "select_cost": "select_penalties, sessions of 40,000 x 56, two smooths",
    "pytest_regression": "pytest tests/analysis/regression, cold then warm cache",
}
BIN_S = 0.015
#: The weak penalty of the simple terms: a Normal prior of SD 2.
WEAK = log_penalty_from_prior_sd(2.0)


# --------------------------------------------------------------------------
# Inputs


def visit_bins(
    rng: np.random.Generator,
    n_visits: int,
    max_rows: int | None = None,
    base_rate: float = 0.3,
) -> pd.DataFrame:
    """Bins of ``n_visits`` visits of lognormal length (median 2 s, a few long
    stays), or of the first visits whose bins fit in ``max_rows``: visit id,
    exposure, the clocks since arrival and until departure, the last outcome
    (0/1) and a Poisson count from a rate (``base_rate`` per s at arrival) that
    rises over the first 2 s and falls before departure."""
    duration = np.minimum(2.0 * np.exp(0.9 * rng.normal(size=n_visits)), 60.0)
    if max_rows is not None:
        rows = np.cumsum(np.maximum(1, np.ceil(duration / BIN_S)))
        duration = duration[rows <= max_rows]
        n_visits = duration.size
    n_bins = np.maximum(1, np.ceil(duration / BIN_S)).astype(int)
    visit = np.repeat(np.arange(n_visits), n_bins)
    start = np.repeat(np.cumsum(n_bins) - n_bins, n_bins)
    since = (np.arange(visit.size) - start) * BIN_S
    until = np.repeat(duration, n_bins) - since
    exposure = np.minimum(BIN_S, until)
    until = np.maximum(until - exposure, 0.0)
    rewarded = (rng.uniform(size=n_visits) < 0.6).astype(float)[visit]
    log_rate = (
        np.log(base_rate)
        + np.log(3.0) * (1.0 - np.exp(-since / 0.7))
        - 0.8 * np.exp(-until / 0.3)
        + 0.3 * rewarded
    )
    return pd.DataFrame(
        {
            "well_visit_id": visit,
            "exposure": exposure,
            "count": rng.poisson(exposure * np.exp(log_rate)).astype(float),
            "time_since_s": since,
            "time_until_s": until,
            "rewarded": rewarded,
        }
    )


def _session(rows: pd.DataFrame, key: tuple[str, str, str]) -> pd.DataFrame:
    rows = rows.copy()
    rows["animal"], rows["session_date"], rows["epoch"] = key
    return rows


def rate_inputs(n_rows: int, seed: int, smooths: tuple[int, int] = (43, 33)):
    """One rate session of at most ``n_rows`` bins and its design (two smooths,
    the weakly penalized ``rewarded``, the offset, an intercept)."""

    rng = np.random.default_rng(seed)
    rows = visit_bins(rng, n_rows, max_rows=n_rows)
    design = build_design(
        [
            Smooth("time_since_s", 0.0, 20.0, smooths[0], 2.0),
            Smooth("time_until_s", 0.0, 3.0, smooths[1], 2.0),
            *([Slope("rewarded", WEAK)] if smooths == (43, 33) else []),
            Offset("exposure"),
        ],
        rows,
        intercept=True,
    )
    built = design.build(rows)
    data = SessionData(built.X, rows["count"].to_numpy(), built.offset)
    return Poisson(), data, list(design.blocks), design.log_penalties


def content_inputs(seed: int = 0, n_events: int = 400, n_arms: int = 9):
    """One content session (the arm category, a smooth of a clock, seven
    slopes) under the grouped softmax."""

    rng = np.random.default_rng(seed)
    event = np.repeat(np.arange(n_events), n_arms)
    arm = np.tile(np.arange(n_arms), n_events)
    rows = pd.DataFrame({"event": event, "arm": arm})
    rows["clock"] = rng.uniform(0.0, 20.0, rows.shape[0])
    slopes = [f"x{j}" for j in range(7)]
    for name in slopes:
        rows[name] = rng.normal(size=rows.shape[0])
    eta = (
        rng.normal(0.0, 0.5, n_arms)[arm]
        + 0.8 * np.sin(rows["clock"].to_numpy() / 3.0)
        + rows[slopes].to_numpy() @ rng.normal(0.0, 0.3, len(slopes))
    )
    weight = np.exp(eta).reshape(n_events, n_arms)
    share = np.concatenate([rng.dirichlet(20.0 * w / w.sum()) for w in weight])
    design = build_design(
        [
            Category("arm", tuple(range(n_arms)), reference=0, penalty=WEAK),
            Smooth("clock", 0.0, 20.0, 16, 2.0),
            *(Slope(name, WEAK) for name in slopes),
        ],
        rows,
        intercept=False,
    )
    data = SessionData(design.build(rows).X, np.maximum(share, 1e-12), event)
    return GroupedSoftmax(), data, list(design.blocks), design.log_penalties


def diagonal_inputs(family: str, seed: int = 0, n_rows: int = 20_000):
    """One Bernoulli or Gaussian session: a smooth of 20 basis functions and
    three weakly penalized slopes, with an intercept."""

    rng = np.random.default_rng(seed)
    rows = pd.DataFrame(
        {
            "clock": rng.uniform(0.0, 20.0, n_rows),
            **{f"x{j}": rng.normal(size=n_rows) for j in range(3)},
        }
    )
    eta = np.sin(rows["clock"].to_numpy() / 3.0) + 0.3 * rows["x0"].to_numpy()
    design = build_design(
        [
            Smooth("clock", 0.0, 20.0, 20, 2.0),
            *(Slope(f"x{j}", WEAK) for j in range(3)),
        ],
        rows,
        intercept=True,
    )
    if family == "bernoulli":
        lik = Bernoulli()
        y = (rng.uniform(size=n_rows) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        lik = Gaussian()
        y = eta + rng.normal(scale=0.8, size=n_rows)
    data = SessionData(design.build(rows).X, y)
    return lik, data, list(design.blocks), design.log_penalties


def thirty_sessions(seed: int = 7):
    """30 rate sessions (5 animals x 2 days x 3 sessions; about 110 events
    each) and the design of one smooth of the clock since arrival."""

    rng = np.random.default_rng(seed)
    sessions = {}
    for animal in range(5):
        for day in range(2):
            for epoch in range(3):
                key = (f"a{animal}", f"2026010{day + 1}", f"0{2 * epoch + 2}_r1")
                sessions[key] = _session(visit_bins(rng, 181, base_rate=0.075), key)
    pooled = pd.concat(sessions.values(), ignore_index=True)

    def design_at(log_penalty: float):
        return build_design(
            [Smooth("time_since_s", 0.0, 20.0, 43, log_penalty), Offset("exposure")],
            pooled,
            intercept=True,
        )

    return sessions, design_at


def _refit_case(inputs: Callable[[], tuple]) -> Callable[[], Callable[[], object]]:
    def prepare() -> Callable[[], object]:
        lik, data, blocks, rho = inputs()

        def run() -> object:
            refit = refit_session(lik, data, blocks, rho)
            assert refit.fitted, refit.reason
            return refit

        return run

    return prepare


def _fit_sessions_case() -> Callable[[], object]:
    sessions, design_at = thirty_sessions()
    design = design_at(2.0)

    def run() -> object:
        return fit_sessions(
            design,
            sessions,
            Poisson(),
            model="timing",
            kind="timing",
            response="count",
            unit="well_visit_id",
            table_sha256={},
        )

    return run


def _calibration_case() -> Callable[[], object]:
    sessions, design_at = thirty_sessions()
    design = design_at(0.0)

    def run() -> object:
        return log_penalty_for_edf(
            design,
            sessions,
            Poisson(),
            term="s(time_since_s)",
            target_edf=6.0,
            response="count",
            unit="well_visit_id",
        )

    return run


def _select_case(n_sessions: int) -> Callable[[], Callable[[], object]]:
    def prepare() -> Callable[[], object]:
        built = [
            rate_inputs(40_000, 100 + s, smooths=(30, 27)) for s in range(n_sessions)
        ]
        lik, blocks = built[0][0], built[0][2]
        datas = [b[1] for b in built]

        def run() -> object:
            return select_penalties(lik, datas, blocks)

        return run

    return prepare


PREPARE: dict[str, Callable[..., Callable[[], object]]] = {
    "fit_rate": _refit_case(lambda: rate_inputs(37_000, 0)),
    "fit_content": _refit_case(content_inputs),
    "fit_bernoulli": _refit_case(lambda: diagonal_inputs("bernoulli")),
    "fit_gaussian": _refit_case(lambda: diagonal_inputs("gaussian")),
    "fit_sessions_30": _fit_sessions_case,
    "calibration_30": _calibration_case,
}


# --------------------------------------------------------------------------
# One case, in its own process


def child(case: str, reps: int, cost_sessions: int) -> dict:
    """Build the case's inputs, time its first call and ``reps`` more."""
    started = time.perf_counter()
    prepare = _select_case(cost_sessions) if case == "select_cost" else PREPARE[case]
    run = prepare()
    built = time.perf_counter() - started
    times = []
    for _ in range(1 + reps):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    warm = np.array(times[1:]) if reps else np.array([np.nan])
    return {
        "build_s": built,
        "first_s": times[0],
        "warm_median_s": float(np.median(warm)),
        "warm_min_s": float(np.min(warm)),
        "warm_max_s": float(np.max(warm)),
        "first_less_median_s": times[0] - float(np.median(warm)),
    }


def _environment(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            name: str(threads)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return env


def _spawn(
    command: list[str], env: dict, cores: list[int]
) -> tuple[str, float, int, int]:
    """Run ``command`` on ``cores``; its stdout, wall time, exit code and peak
    resident memory in kB."""
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        env=env,
        cwd=REPO,
        stdout=subprocess.PIPE,
        text=True,
        preexec_fn=lambda: os.sched_setaffinity(0, cores),
    )
    out = process.stdout.read()
    _, status, usage = os.wait4(process.pid, 0)
    process.returncode = os.waitstatus_to_exitcode(status)
    return out, time.perf_counter() - start, process.returncode, usage.ru_maxrss


def time_case(
    case: str, threads: int, cores: list[int], reps: int, cost_sessions: int
) -> dict:
    """One case at ``threads`` threads, in a new process."""
    env = _environment(threads)
    pinned = cores[:threads]
    load_before = os.getloadavg()
    record = {"case": case, "threads": threads, "cores": pinned}
    if case == "pytest_regression":
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/analysis/regression",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
        with tempfile.TemporaryDirectory() as cache:
            env["JAX_COMPILATION_CACHE_DIR"] = cache
            for label in ("cold", "warm"):
                out, wall, code, peak = _spawn(command, env, pinned)
                record[f"{label}_s"] = wall
                record[f"{label}_peak_rss_mb"] = peak / 1024
                record[f"{label}_result"] = (
                    out.strip().splitlines()[-1] if out.strip() else ""
                )
                if code:
                    record["error"] = f"pytest exited {code}"
    else:
        command = [
            sys.executable,
            __file__,
            "--child",
            case,
            "--reps",
            str(reps),
            "--cost-sessions",
            str(cost_sessions),
        ]
        out, wall, code, peak = _spawn(command, env, pinned)
        if code:
            record["error"] = f"exited {code}"
        else:
            record.update(json.loads(out.strip().splitlines()[-1]))
        record["process_s"] = wall
        record["peak_rss_mb"] = peak / 1024
    record["load_before"] = list(load_before)
    record["load_after"] = list(os.getloadavg())
    return record


def _cores(text: str | None) -> list[int]:
    if text is None:
        return sorted(os.sched_getaffinity(0))
    cores = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        cores.extend(range(int(lo), int(hi or lo) + 1))
    return cores


def table(records: Sequence[dict]) -> str:
    """The records as a markdown table."""
    lines = [
        "| Case | Threads | First call (s) | Median of the rest (s) | Range (s) | First less median (s) | Peak RSS (MB) | Load before, after (1 min) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        load = f"{r['load_before'][0]:.0f}, {r['load_after'][0]:.0f}"
        if r["case"] == "pytest_regression":
            lines.append(
                f"| {r['case']} | {r['threads']} | cold {r['cold_s']:.1f} | warm {r['warm_s']:.1f} | | "
                f"{r['cold_s'] - r['warm_s']:.1f} | {max(r['cold_peak_rss_mb'], r['warm_peak_rss_mb']):.0f} | {load} |"
            )
        elif "error" in r:
            lines.append(
                f"| {r['case']} | {r['threads']} | {r['error']} | | | | | {load} |"
            )
        else:
            lines.append(
                f"| {r['case']} | {r['threads']} | {r['first_s']:.3f} | {r['warm_median_s']:.3f} | "
                f"{r['warm_min_s']:.3f} to {r['warm_max_s']:.3f} | {r['first_less_median_s']:.3f} | "
                f"{r['peak_rss_mb']:.0f} | {load} |"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--case",
        action="append",
        choices=list(CASES),
        help="Repeatable; default: every case but select_cost.",
    )
    parser.add_argument(
        "--threads", action="append", type=int, help="Repeatable; default: 1."
    )
    parser.add_argument(
        "--cores",
        help="The cores to pin to, as 0-5,8; default: this process's CPU set.",
    )
    parser.add_argument(
        "--reps", type=int, default=5, help="Calls after the first (default 5)."
    )
    parser.add_argument(
        "--cost-sessions",
        type=int,
        default=242,
        help="Sessions of select_cost (default 242).",
    )
    parser.add_argument("--out", type=Path, help="Write the records as JSON here.")
    parser.add_argument("--child", choices=list(CASES), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        print(json.dumps(child(args.child, args.reps, args.cost_sessions)))
        return 0
    cases = args.case or [c for c in CASES if c != "select_cost"]
    cores = _cores(args.cores)
    records = []
    for threads in args.threads or [1]:
        if threads > len(cores):
            parser.error(f"{threads} threads on {len(cores)} cores")
        for case in cases:
            record = time_case(case, threads, cores, args.reps, args.cost_sessions)
            print(json.dumps(record), flush=True)
            records.append(record)
    meta = {
        "commit": git_commit(REPO),
        "tree_modified": git_dirty(REPO),
        "n_cpus": os.cpu_count(),
        "cost_sessions": args.cost_sessions,
        "reps": args.reps,
    }
    print(table(records))
    print(f"commit {meta['commit']}, tree modified: {meta['tree_modified']}")
    if args.out:
        args.out.write_text(
            json.dumps({"meta": meta, "records": records}, indent=1) + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
