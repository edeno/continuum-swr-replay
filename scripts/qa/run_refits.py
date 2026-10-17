#!/usr/bin/env python
"""Rerun the CPU model fits and the figures that read them, in dependency order.

Every run answers one of the paper's questions (``QUESTIONS``; the full map is
``docs/analysis_map.md``). A run's name says what it is:
``<question>.<analysis>[.<variant>]``, for example ``rate.swr_contrast.at_well``.
Each run is one command with a CPU budget and the runs it needs first.
Selecting a run also selects every run that depends on it, so
``--only content.arm_ladder.all_arms`` refits the content contrasts too: their outputs
sit in the folder that run rewrites.

Every run gets a thread cap (OMP, MKL, OpenBLAS, numba and XLA), CPU-only JAX,
and its own log under ``--log-dir``, with a JSON record of the command, commit,
data root, times and exit code. A run whose dependency failed is not started.
After the fits, ``check_fit_provenance.py`` checks both table roots.

Not here: the decodes, the tables, the Q1 checks (state and spike evidence,
the detectors) and the PFC unit GLM, which runs on GPUs for 14-17 h per card
under its own launcher.

Usage
-----
    uv run --no-sync python scripts/qa/run_refits.py --dry-run
    uv run --no-sync python scripts/qa/run_refits.py --only content --cpus 0-23
    CSR_PROCESSED_DATA_DIR=/path/to/root uv run --no-sync python \\
        scripts/qa/run_refits.py --only rate.per_bin,rate.swr_contrast
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

#: The paper's questions this driver's runs answer (Q1, are the events real,
#: has no CPU refit here). Keys are the first part of a run's name.
QUESTIONS = {
    "rate": "Q2. How often do non-local events occur, and what drives the rate, for SWR against non-SWR events?",
    "content": "Q3. What do the events represent (arm or patch), what shapes that, and does it differ for SWR events?",
    "behavior": "Q4. Does content relate to what the animal does next?",
    "pfc": "Q5. Does PFC respond to the events, SWR or not?",
}


@dataclass(frozen=True)
class Run:
    """One command: its name, arguments, CPU budget, thread cap and needs."""

    name: str
    argv: tuple[str, ...]
    cores: int
    threads: int = 1
    deps: tuple[str, ...] = field(default=())
    figure: bool = False

    @property
    def question(self) -> str:
        return self.name.split(".")[0]


def _run(name, cmd, cores, deps=(), threads=1, figure=False):
    return Run(name, tuple(cmd.split()), cores, threads, tuple(deps), figure)


def _fig(name, cmd, deps=()):
    return _run(name, cmd, 1, deps, figure=True)


_PB = "scripts/rate/fit_per_bin_rate_models.py"
_RC = "scripts/rate/fit_swr_contrast_rate.py"
_CL = "scripts/content/fit_replay_content_models.py"
_KC = "scripts/content/fit_swr_contrast_content.py"
_ES = "scripts/content/compare_event_source_models.py"
_S5 = "scripts/behavior/run_retrospective_alignment.py"
_S6 = "scripts/behavior/run_stay_leave.py"
_DM = "--estimator dirichlet_multinomial"
_SWR_CONTRAST = (
    "content.swr_contrast.all",
    "content.swr_contrast.rewarded",
    "content.swr_contrast.omitted",
)
_BREADTH = (
    "content.breadth_contrast.all",
    "content.breadth_contrast.all_length_adjusted",
    "content.breadth_contrast.rewarded",
    "content.breadth_contrast.omitted",
)

# Cores: a run's worker count, or its thread cap for a run without workers.
# The at_well cohort is the shipped one; all_stops and in_transit are sensitivities.
# fmt: off
RUNS: tuple[Run, ...] = (
    # Q2 rate. The per-bin conditional intensity (Family B) is the main model; the
    # per-stop Poisson (Family A) is the literature-comparable one.
    _run("rate.per_bin.at_well", f"{_PB} --stop-cohort at_well --workers 16", 16),
    _run("rate.per_bin.all_stops", f"{_PB} --stop-cohort all_stops --workers 16", 16),
    _run("rate.per_bin.in_transit", f"{_PB} --stop-cohort in_transit --workers 10", 10),
    _run("rate.per_stop.at_well", "scripts/rate/fit_per_stop_rate_models.py --stop-cohort at_well", 1),
    _run("rate.swr_contrast.at_well", f"{_RC} --stop-cohort at_well --workers 16", 16),
    _run("rate.swr_contrast.all_stops", f"{_RC} --stop-cohort all_stops --workers 16", 16),
    _run("rate.uncertainty.at_well", "scripts/rate/fit_uncertainty_rate.py --stop-cohort at_well", 1),
    _run("rate.ripple_power", "scripts/rate/fit_swr_power_model.py", 1),
    # No worker flag: its parallelism is BLAS threads.
    _run("rate.cross_history", "scripts/rate/fit_cross_history_rate.py --write", 8, threads=8),
    _fig("rate.fig.per_bin_ladder", "scripts/rate/plot_per_bin_rate_models.py", ["rate.per_bin.at_well"]),
    _fig("rate.fig.per_bin_summary", "scripts/rate/plot_per_bin_rate_summary.py", ["rate.per_bin.at_well"]),
    _fig("rate.fig.model_checking", "scripts/rate/plot_model_checking.py", ["rate.per_bin.at_well"]),
    _fig("rate.doc.per_bin_results", "scripts/rate/summarize_per_bin_rate_models.py", ["rate.per_bin.at_well"]),
    _fig("rate.fig.uncertainty", "scripts/rate/plot_uncertainty_rate.py", ["rate.uncertainty.at_well"]),
    # Q3 content. The arm ladder asks what shapes content; the SWR contrast asks how
    # SWR content differs (paired within session); the breadth contrast is the
    # Dirichlet-multinomial fit of how widely an event spreads over arms.
    _run("content.arm_ladder.all_arms", _CL, 2, threads=2),
    _run("content.arm_ladder.leaf_arms", f"{_CL} --leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu", 2, threads=2),
    # The contrasts write into models/, which content.arm_ladder.all_arms replaces whole.
    _run("content.swr_contrast.all", f"{_KC} --outcome-stratum all", 2, ["content.arm_ladder.all_arms"], threads=2),
    _run("content.swr_contrast.rewarded", f"{_KC} --outcome-stratum rewarded", 2, ["content.arm_ladder.all_arms"], threads=2),
    _run("content.swr_contrast.omitted", f"{_KC} --outcome-stratum omitted", 2, ["content.arm_ladder.all_arms"], threads=2),
    _run("content.breadth_contrast.all", f"{_KC} {_DM} --n-jobs 16 --outcome-stratum all", 16, ["content.arm_ladder.all_arms"]),
    _run("content.breadth_contrast.all_length_adjusted", f"{_KC} {_DM} --n-jobs 16 --concentration-length --outcome-stratum all", 16, ["content.arm_ladder.all_arms"]),
    _run("content.breadth_contrast.rewarded", f"{_KC} {_DM} --n-jobs 8 --outcome-stratum rewarded", 8, ["content.arm_ladder.all_arms"]),
    _run("content.breadth_contrast.omitted", f"{_KC} {_DM} --n-jobs 8 --outcome-stratum omitted", 8, ["content.arm_ladder.all_arms"]),
    _run("content.estimator_comparison", "scripts/content/compare_content_estimators.py --n-jobs 16", 16),
    _run("content.uncertainty", "scripts/content/analyze_uncertainty_content.py", 1, ["content.arm_ladder.leaf_arms"]),
    # Content on events found by other detectors (ContFrag SWR- and MUA-anchored).
    _run("content.event_source.swr_arm", f"{_CL} --event-source SWR --brain-area HPC", 2, threads=2),
    _run("content.event_source.swr_patch", f"{_CL} --event-source SWR --hierarchy patch --brain-area HPC", 2, threads=2),
    _run("content.event_source.mua_arm", f"{_CL} --event-source MUA --brain-area HPC", 2, threads=2),
    _run("content.event_source.mua_patch", f"{_CL} --event-source MUA --hierarchy patch --brain-area HPC", 2, threads=2),
    _fig("content.fig.swr_contrast_coefs", "scripts/figures/plot_swr_contrast_coefs.py",
         ["rate.swr_contrast.at_well", *_SWR_CONTRAST, *_BREADTH]),
    _fig("content.fig.uncertainty", "scripts/figures/generate_uncertainty_content_figure.py", ["content.uncertainty"]),
    _fig("content.fig.ladder", "scripts/content/generate_replay_content_figures.py",
         ["content.arm_ladder.all_arms", *_SWR_CONTRAST, *_BREADTH]),
    # Q4 behaviour. One suite per event source: the patch, per-stop rate and patch-switch
    # ladders on one common cohort, the choice prediction and switch-aligned content.
    _run("behavior.event_source_suite.non_local", f"{_ES} --event-source non_local", 1),
    _run("behavior.event_source_suite.swr", f"{_ES} --event-source SWR", 1),
    _run("behavior.event_source_suite.mua", f"{_ES} --event-source MUA", 1),
    # Retrospective content alignment (Stage 5B) and content x outcome -> stay/leave (Stage 6).
    _run("behavior.retrospective.all", _S5, 2, threads=2),
    _run("behavior.retrospective.swr", f"{_S5} --swr-only", 2, threads=2),
    _run("behavior.retrospective.non_swr", f"{_S5} --non-swr-only", 2, threads=2),
    _run("behavior.stay_leave.at_well", f"{_S6} --rebuild-summary", 2, threads=2),
    _run("behavior.stay_leave.all_stops", f"{_S6} --stop-cohort all_stops --rebuild-summary", 2, threads=2),
    _run("behavior.stay_leave.at_well_swr", f"{_S6} --content-stratum swr", 2, ["behavior.stay_leave.at_well"], threads=2),
    _run("behavior.stay_leave.at_well_non_swr", f"{_S6} --content-stratum non_swr", 2, ["behavior.stay_leave.at_well"], threads=2),
    _run("behavior.stay_leave.at_well_mua", f"{_S6} --content-stratum mua", 2, ["behavior.stay_leave.at_well"], threads=2),
    _run("behavior.stay_leave.all_stops_swr", f"{_S6} --stop-cohort all_stops --content-stratum swr", 2, ["behavior.stay_leave.all_stops"], threads=2),
    _run("behavior.stay_leave.all_stops_non_swr", f"{_S6} --stop-cohort all_stops --content-stratum non_swr", 2, ["behavior.stay_leave.all_stops"], threads=2),
    _run("behavior.stay_leave.all_stops_mua", f"{_S6} --stop-cohort all_stops --content-stratum mua", 2, ["behavior.stay_leave.all_stops"], threads=2),
    # Q5 PFC. Figure 4 reads the PFC unit GLM bundles, which this driver does not make.
    _fig("pfc.fig.figure04", "scripts/figures/generate_figure04.py"),
)
# fmt: on

BY_NAME = {r.name: r for r in RUNS}
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def select(names: list[str] | None, with_dependents: bool = True) -> list[Run]:
    """The runs named (all when ``names`` is None), in list order.

    A name selects the run of that name and every run whose name extends it, so
    ``content`` selects all of Q3 and ``content.swr_contrast`` its three strata.
    With ``with_dependents``, every run that needs a selected run, directly or
    through another, is selected too.
    """
    if not names:
        chosen = set(BY_NAME)
    else:
        chosen = set()
        for name in names:
            hits = {
                r.name for r in RUNS if r.name == name or r.name.startswith(name + ".")
            }
            if not hits:
                raise SystemExit(
                    f"no run is named {name!r} or starts with it; see --dry-run"
                )
            chosen |= hits
    if with_dependents:
        grew = True
        while grew:
            more = {r.name for r in RUNS if set(r.deps) & chosen} - chosen
            chosen |= more
            grew = bool(more)
    return [r for r in RUNS if r.name in chosen]


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
    names = {r.name for r in runs}
    pending = list(runs)
    active: dict[str, tuple[subprocess.Popen, Run, float, str]] = {}
    codes: dict[str, int | None] = {r.name: None for r in runs}
    blocked: set[str] = set()
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

    while pending or active:
        for name, (proc, run, t0, started) in list(active.items()):
            if proc.poll() is None:
                continue
            codes[name] = proc.returncode
            del active[name]
            record = {
                "name": name,
                "question": QUESTIONS.get(run.question, run.question),
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
                "log": str(log_dir / f"{name}.log"),
            }
            (log_dir / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
            print(f"{_now()} {name} exit {proc.returncode}", flush=True)
        for run in list(pending):
            needs = [d for d in run.deps if d in names]
            if any(d in blocked or codes[d] not in (None, 0) for d in needs):
                pending.remove(run)
                blocked.add(run.name)
                print(
                    f"{_now()} {run.name} not started: a dependency failed", flush=True
                )
                continue
            if any(codes[d] != 0 for d in needs):  # still running
                continue
            used = sum(r.cores for _, r, _, _ in active.values())
            if active and used + run.cores > budget:
                continue
            log = open(log_dir / f"{run.name}.log", "w")
            proc = subprocess.Popen(
                [*prefix, *run.argv],
                cwd=REPO,
                env=run_env(run),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.close()
            active[run.name] = (proc, run, time.monotonic(), _now())
            pending.remove(run)
            print(f"{_now()} {run.name} start ({run.cores} cores)", flush=True)
        if pending and not active:
            # Dependencies come earlier in RUNS, so this cannot happen.
            raise RuntimeError(f"nothing can start: {[r.name for r in pending]}")
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


def print_plan(runs: list[Run]) -> None:
    """The selected runs, grouped under their question."""
    for key, text in QUESTIONS.items():
        group = [r for r in runs if r.question == key]
        if not group:
            continue
        print(text)
        for r in group:
            deps = f"   after {', '.join(r.deps)}" if r.deps else ""
            print(f"  {r.name:<46} {r.cores:>2} cores  {' '.join(r.argv)}{deps}")
        print()


def _cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--only",
        help="comma-separated run names or name prefixes (rate, content, behavior, pfc, "
        "content.swr_contrast, ...)",
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
        print_plan(runs)
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
    failed = sorted(name for name, code in codes.items() if code != 0)
    print(
        f"{len(codes) - len(failed)}/{len(codes)} runs exited 0; not ok: {failed or 'none'}"
    )
    fits = any(not r.figure for r in runs)
    worst = check_provenance(log_dir) if fits and not args.no_check else 0
    return 1 if failed or worst else 0


if __name__ == "__main__":
    sys.exit(main())
