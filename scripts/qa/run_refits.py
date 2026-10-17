#!/usr/bin/env python
"""Rerun the CPU model refits and the figures that read them, in dependency order.

The run list (``RUNS``) is every refit the full pipeline refresh ran (R6), plus
the figure and summary scripts that read those fits (R7). Each run is one
command with a CPU budget and the runs it needs first. Selecting a run also
selects every run that depends on it, so ``--only C01`` refits the content
contrasts too: their outputs sit in the folder ``C01`` rewrites.

Every run gets a thread cap (OMP, MKL, OpenBLAS, numba and XLA), CPU-only JAX,
and its own log under ``--log-dir``, with a JSON record of the command, commit,
data root, times and exit code. A run whose dependency failed is not started.
After the fits, ``check_fit_provenance.py`` checks both table roots.

Not here: the decodes, the tables, and the PFC unit GLM, which runs on GPUs
for 14-17 h per card under its own launcher.

Usage
-----
    uv run --no-sync python scripts/qa/run_refits.py --dry-run
    uv run --no-sync python scripts/qa/run_refits.py --only content --cpus 0-23
    CSR_PROCESSED_DATA_DIR=/path/to/root uv run --no-sync python \\
        scripts/qa/run_refits.py --only R01,R05
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Run:
    """One command: its id, group, arguments, CPU budget, thread cap and needs."""

    id: str
    group: str
    argv: tuple[str, ...]
    cores: int
    threads: int = 1
    deps: tuple[str, ...] = field(default=())


def _run(id_, group, cmd, cores, deps=(), threads=1):
    return Run(id_, group, tuple(cmd.split()), cores, threads, tuple(deps))


_KC = "scripts/content/fit_swr_contrast_content.py"
_S6 = "scripts/pfc/run_stage6_learning.py"
_K_ALL = ("K01", "K02", "K03", "K04", "K05", "K06", "K07")

# Cores: a run's worker count, or its thread cap for a run without workers.
# fmt: off
RUNS: tuple[Run, ...] = (
    _run("R01", "rate", "scripts/rate/fit_per_bin_rate_models.py --stop-cohort at_well --workers 16", 16),
    _run("R02", "rate", "scripts/rate/fit_per_bin_rate_models.py --stop-cohort all_stops --workers 16", 16),
    _run("R03", "rate", "scripts/rate/fit_per_bin_rate_models.py --stop-cohort in_transit --workers 10", 10),
    _run("R04", "rate", "scripts/rate/fit_per_stop_rate_models.py --stop-cohort at_well", 1),
    _run("R05", "rate", "scripts/rate/fit_swr_contrast_rate.py --stop-cohort at_well --workers 16", 16),
    _run("R06", "rate", "scripts/rate/fit_swr_contrast_rate.py --stop-cohort all_stops --workers 16", 16),
    _run("R07", "rate", "scripts/rate/fit_swr_contrast_rate.py --aggregate-only --stop-cohort at_well", 1, ["R05"]),
    _run("R08", "rate", "scripts/rate/fit_swr_contrast_rate.py --aggregate-only --stop-cohort all_stops", 1, ["R06"]),
    _run("R09", "rate", "scripts/rate/fit_uncertainty_rate.py --stop-cohort at_well", 1),
    _run("R10", "rate", "scripts/rate/fit_swr_power_model.py", 1),
    # No worker flag: its parallelism is BLAS threads.
    _run("R11", "rate", "scripts/rate/fit_cross_history_rate.py --write", 8, threads=8),
    _run("C01", "content", "scripts/content/fit_replay_content_models.py", 2, threads=2),
    _run("C02", "content", "scripts/content/fit_replay_content_models.py --leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu", 2, threads=2),
    _run("C03", "content", "scripts/content/fit_replay_content_models.py --hierarchy patch", 1),
    _run("C04", "content", "scripts/content/fit_replay_content_models.py --swr-subset T", 2, threads=2),
    _run("C05", "content", "scripts/content/fit_replay_content_models.py --swr-subset F", 2, threads=2),
    _run("C06", "content", "scripts/content/fit_replay_content_models.py --hierarchy patch --swr-subset T", 1),
    _run("C07", "content", "scripts/content/fit_replay_content_models.py --hierarchy patch --swr-subset F", 1),
    _run("C08", "content", "scripts/content/fit_replay_content_models.py --estimator dirichlet_multinomial --n-jobs 16", 16),
    # The contrasts write into models/, which C01 replaces whole, so they follow it.
    _run("K01", "content", f"{_KC} --outcome-stratum all", 2, ["C01"], threads=2),
    _run("K02", "content", f"{_KC} --outcome-stratum rewarded", 2, ["C01"], threads=2),
    _run("K03", "content", f"{_KC} --outcome-stratum omitted", 2, ["C01"], threads=2),
    _run("K04", "content", f"{_KC} --estimator dirichlet_multinomial --n-jobs 16 --outcome-stratum all", 16, ["C01"]),
    _run("K05", "content", f"{_KC} --estimator dirichlet_multinomial --n-jobs 16 --concentration-length --outcome-stratum all", 16, ["C01"]),
    _run("K06", "content", f"{_KC} --estimator dirichlet_multinomial --n-jobs 8 --outcome-stratum rewarded", 8, ["C01"]),
    _run("K07", "content", f"{_KC} --estimator dirichlet_multinomial --n-jobs 8 --outcome-stratum omitted", 8, ["C01"]),
    _run("X01", "content", "scripts/content/compare_content_estimators.py --n-jobs 16", 16),
    _run("X02", "content", "scripts/content/analyze_uncertainty_content.py", 1, ["C02"]),
    _run("X03", "content", "scripts/content/compare_event_source_models.py --event-source non_local", 1),
    _run("X04", "content", "scripts/content/compare_event_source_models.py --event-source SWR", 1),
    _run("X05", "content", "scripts/content/compare_event_source_models.py --event-source MUA", 1),
    _run("F01", "contfrag", "scripts/content/fit_replay_content_models.py --event-source SWR --brain-area HPC", 2, threads=2),
    _run("F02", "contfrag", "scripts/content/fit_replay_content_models.py --event-source SWR --hierarchy patch --brain-area HPC", 2, threads=2),
    _run("F03", "contfrag", "scripts/content/fit_replay_content_models.py --event-source MUA --brain-area HPC", 2, threads=2),
    _run("F04", "contfrag", "scripts/content/fit_replay_content_models.py --event-source MUA --hierarchy patch --brain-area HPC", 2, threads=2),
    _run("P01", "pfc", "scripts/pfc/run_stage5b_retrospective.py", 2, threads=2),
    _run("P02", "pfc", "scripts/pfc/run_stage5b_retrospective.py --swr-only", 2, threads=2),
    _run("P03", "pfc", "scripts/pfc/run_stage5b_retrospective.py --non-swr-only", 2, threads=2),
    _run("S01", "pfc", f"{_S6} --rebuild-summary", 2, threads=2),
    _run("S02", "pfc", f"{_S6} --stop-cohort all_stops --rebuild-summary", 2, threads=2),
    _run("S03", "pfc", f"{_S6} --content-stratum swr", 2, ["S01"], threads=2),
    _run("S04", "pfc", f"{_S6} --content-stratum non_swr", 2, ["S01"], threads=2),
    _run("S05", "pfc", f"{_S6} --content-stratum mua", 2, ["S01"], threads=2),
    _run("S06", "pfc", f"{_S6} --stop-cohort all_stops --content-stratum swr", 2, ["S02"], threads=2),
    _run("S07", "pfc", f"{_S6} --stop-cohort all_stops --content-stratum non_swr", 2, ["S02"], threads=2),
    _run("S08", "pfc", f"{_S6} --stop-cohort all_stops --content-stratum mua", 2, ["S02"], threads=2),
    # Figures and generated docs that read the fits (R7).
    _run("G01", "figures", "scripts/rate/plot_per_bin_rate_models.py", 1, ["R01"]),
    _run("G02", "figures", "scripts/rate/plot_per_bin_rate_summary.py", 1, ["R01"]),
    _run("G03", "figures", "scripts/rate/plot_model_checking.py", 1, ["R01"]),
    _run("G04", "figures", "scripts/rate/summarize_per_bin_rate_models.py", 1, ["R01"]),
    _run("G05", "figures", "scripts/rate/plot_uncertainty_rate.py", 1, ["R09"]),
    _run("G06", "figures", "scripts/figures/plot_swr_contrast_coefs.py", 1, ["R07", *_K_ALL]),
    _run("G07", "figures", "scripts/figures/generate_uncertainty_content_figure.py", 1, ["X02"]),
    _run("G08", "figures", "scripts/content/generate_replay_content_figures.py", 1, ["C01", *_K_ALL]),
    # Reads the PFC unit GLM bundles, which this driver does not make.
    _run("G09", "figures", "scripts/figures/generate_figure04.py", 1),
)
# fmt: on

BY_ID = {r.id: r for r in RUNS}
GROUPS = sorted({r.group for r in RUNS})
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def select(names: list[str] | None, with_dependents: bool = True) -> list[Run]:
    """The runs named by id or group (all when ``names`` is None), in list order.

    With ``with_dependents``, every run that needs a selected run, directly or
    through another, is selected too.
    """
    if not names:
        chosen = set(BY_ID)
    else:
        chosen = set()
        for name in names:
            if name in BY_ID:
                chosen.add(name)
            elif name in GROUPS:
                chosen |= {r.id for r in RUNS if r.group == name}
            else:
                raise SystemExit(f"unknown run or group {name!r}; groups: {GROUPS}")
    if with_dependents:
        grew = True
        while grew:
            more = {r.id for r in RUNS if set(r.deps) & chosen} - chosen
            chosen |= more
            grew = bool(more)
    return [r for r in RUNS if r.id in chosen]


def run_env(run: Run) -> dict[str, str]:
    """The environment for one run: CPU-only JAX and its thread cap."""
    env = dict(os.environ)
    env.update({var: str(run.threads) for var in THREAD_VARS})
    env["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen={'true' if run.threads > 1 else 'false'} "
        f"intra_op_parallelism_threads={run.threads}"
    )
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", "--no-optional-locks", *args], cwd=REPO, capture_output=True, text=True
    )
    return out.stdout.strip()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def schedule(
    runs: list[Run],
    budget: int,
    log_dir: Path,
    prefix: list[str],
    poll_s: float = 5.0,
) -> dict[str, int | None]:
    """Run ``runs`` under a core budget; return each run's exit code (None: not started).

    A run starts when all its selected dependencies exited 0 and its cores fit
    in what is free; a run larger than the budget starts when nothing else runs.
    A dependency outside ``runs`` is taken as already done.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ids = {r.id for r in runs}
    pending = list(runs)
    active: dict[str, tuple[subprocess.Popen, Run, float, str]] = {}
    codes: dict[str, int | None] = {r.id: None for r in runs}
    blocked: set[str] = set()
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

    while pending or active:
        for rid, (proc, run, t0, started) in list(active.items()):
            if proc.poll() is None:
                continue
            codes[rid] = proc.returncode
            del active[rid]
            record = {
                "id": rid,
                "argv": [*prefix, *run.argv],
                "cores": run.cores,
                "threads": run.threads,
                "commit": commit,
                "tree_dirty": dirty,
                "processed_data_dir": str(PROCESSED_DATA_DIR),
                "host": socket.gethostname(),
                "start_utc": started,
                "end_utc": _now(),
                "wall_seconds": round(time.monotonic() - t0, 1),
                "exit_code": proc.returncode,
                "log": str(log_dir / f"{rid}.log"),
            }
            (log_dir / f"{rid}.json").write_text(json.dumps(record, indent=1) + "\n")
            print(f"{_now()} {rid} exit {proc.returncode}", flush=True)
        for run in list(pending):
            needs = [d for d in run.deps if d in ids]
            if any(d in blocked or codes[d] not in (None, 0) for d in needs):
                pending.remove(run)
                blocked.add(run.id)
                print(f"{_now()} {run.id} not started: a dependency failed", flush=True)
                continue
            if any(codes[d] != 0 for d in needs):  # still running
                continue
            used = sum(r.cores for _, r, _, _ in active.values())
            if active and used + run.cores > budget:
                continue
            log = open(log_dir / f"{run.id}.log", "w")
            proc = subprocess.Popen(
                [*prefix, *run.argv],
                cwd=REPO,
                env=run_env(run),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.close()
            active[run.id] = (proc, run, time.monotonic(), _now())
            pending.remove(run)
            print(f"{_now()} {run.id} start ({run.cores} cores)", flush=True)
        if pending and not active:
            # Dependencies come earlier in RUNS, so this cannot happen.
            raise RuntimeError(f"nothing can start: {[r.id for r in pending]}")
        if active:
            time.sleep(poll_s)
    return codes


def check_provenance(log_dir: Path) -> int:
    """``check_fit_provenance.py`` on both table roots; the worst exit code."""
    from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

    worst = 0
    for name, extra in (
        ("replay_content", []),
        (
            "contfrag_replay_content",
            ["--root", str(PROCESSED_DATA_DIR / "contfrag_replay_content")],
        ),
    ):
        with open(log_dir / f"check_fit_provenance_{name}.log", "w") as log:
            code = subprocess.run(
                [sys.executable, "scripts/qa/check_fit_provenance.py", *extra],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        print(f"{_now()} check_fit_provenance {name}: exit {code}", flush=True)
        worst = max(worst, code)
    return worst


def _cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--only", help=f"comma-separated run ids or groups ({', '.join(GROUPS)})"
    )
    parser.add_argument(
        "--no-dependents",
        action="store_true",
        help="do not add the runs that depend on the selected ones",
    )
    parser.add_argument(
        "--cpus", help="CPU list to pin every run to, e.g. 0-23 (default: no pinning)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        help="cores to fill (default: the --cpus count, else all CPUs)",
    )
    parser.add_argument("--log-dir", type=Path, help="default logs/refits_<UTC time>/")
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip check_fit_provenance.py after the fits",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the selected runs and stop"
    )
    args = parser.parse_args(argv)

    runs = select(args.only.split(",") if args.only else None, not args.no_dependents)
    if args.dry_run:
        for r in runs:
            deps = f"  after {','.join(r.deps)}" if r.deps else ""
            print(f"{r.id}  {r.cores:>2} cores  {' '.join(r.argv)}{deps}")
        return 0
    prefix = ["uv", "run", "--no-sync", "python"]
    if args.cpus:
        prefix = ["taskset", "-c", args.cpus, *prefix]
    budget = args.budget or (
        len(_cpu_list(args.cpus)) if args.cpus else os.cpu_count() or 1
    )
    log_dir = (
        args.log_dir or REPO / "logs" / f"refits_{datetime.now(UTC):%Y%m%dT%H%M%S}"
    )
    codes = schedule(runs, budget, log_dir, prefix)
    failed = sorted(rid for rid, code in codes.items() if code != 0)
    print(
        f"{len(codes) - len(failed)}/{len(codes)} runs exited 0; not ok: {failed or 'none'}"
    )
    worst = (
        0
        if args.no_check or not any(r.group != "figures" for r in runs)
        else check_provenance(log_dir)
    )
    return 1 if failed or worst else 0


if __name__ == "__main__":
    sys.exit(main())
