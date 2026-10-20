#!/usr/bin/env python
"""Rerun the CPU model fits and the figures that read them, in dependency order.

Every run answers one of the paper's questions (``QUESTIONS``; the full map is
``docs/analysis_map.md``). A run's name says what it is:
``<question>.<analysis>[.<variant>]``, for example ``rate.swr_contrast.at_well``.
Each run is one command, the number of CPUs it gets and the runs it needs first.
Selecting a run also selects every run that depends on it, so
``--only content.arm_ladder.all_arms`` refits the content contrasts too: their
outputs sit in the folder that run rewrites.

Each run is pinned to its own CPUs, taken from ``--cpus``, so its BLAS, numba
and JAX thread pools (which size themselves from the CPUs a process may use)
cannot spill onto another run's. OMP, MKL, OpenBLAS and numba threads are also
capped at the run's ``threads``. Each run writes a log and a JSON record (the
command, its CPUs and environment, the commit, the data root, times and exit
code) under ``--log-dir``; a run whose dependency failed gets a "blocked"
record. Then ``check_fit_provenance.py`` checks every artifact the runs wrote.

Not here: the decodes, the tables, the Q1 checks, and the PFC unit GLM (Q5),
which runs on GPUs for 14-17 h per card under its own launcher; Figure 4 reads
it and is drawn after it.

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
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The questions this driver's runs answer; keys are the first part of a run's
#: name. Q1 (are the events real), Q3a (what the events are like) and Q5 (PFC)
#: have no CPU fit here.
QUESTIONS = {
    "rate": "Q2. How often do non-local events occur, and what drives the rate, for SWR against non-SWR events?",
    "content": "Q3b. What do the events represent (arm or patch), what shapes that, and does it differ for SWR events?",
    "behavior": "Q4. Does content relate to what the animal does next?",
}


@dataclass(frozen=True)
class Run:
    """One command: its name, arguments, CPUs, thread cap and needs."""

    name: str
    argv: tuple[str, ...]
    cores: int
    threads: int = 1
    deps: tuple[str, ...] = field(default=())
    figure: bool = False
    optional: bool = False

    @property
    def question(self) -> str:
        return self.name.split(".")[0]


def _run(name, cmd, cores, deps=(), threads=1, figure=False, optional=False):
    return Run(name, tuple(cmd.split()), cores, threads, tuple(deps), figure, optional)


def _fig(name, cmd, deps=()):
    return _run(name, cmd, 1, deps, figure=True)


_PB = "scripts/rate/fit_per_bin_rate_models.py"
_PS = "scripts/rate/fit_per_stop_rate_models.py"
_RC = "scripts/rate/fit_swr_contrast_rate.py"
_CL = "scripts/content/fit_replay_content_models.py"
_KC = "scripts/content/fit_swr_contrast_content.py"
_ES = "scripts/content/compare_event_source_models.py"
_S5 = "scripts/behavior/run_retrospective_alignment.py"
_S6 = "scripts/behavior/run_stay_leave.py"
_DM = "--estimator dirichlet_multinomial"
_ARM = "content.arm_ladder.all_arms"

# Cores: a run's worker count, or its thread cap for a run without workers.
# The at_well cohort is the shipped one; all_stops and in_transit are sensitivities.
# fmt: off
RUNS: tuple[Run, ...] = (
    # Q2 rate. The per-bin conditional intensity (Family B) is the main model; the
    # per-stop Poisson (Family A) is the literature-comparable one.
    _run("rate.per_bin.at_well", f"{_PB} --stop-cohort at_well --workers 16", 16),
    _run("rate.per_bin.all_stops", f"{_PB} --stop-cohort all_stops --workers 16", 16),
    # Opt-in: thin (about 50 sessions) and cited nowhere.
    _run("rate.per_bin.in_transit", f"{_PB} --stop-cohort in_transit --workers 10", 10, optional=True),
    _run("rate.per_stop.at_well", f"{_PS} --stop-cohort at_well", 1),
    _run("rate.per_stop.all_stops", f"{_PS} --stop-cohort all_stops", 1),
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
    # Q3b content. The arm ladder asks what shapes content; the SWR contrast asks how
    # SWR content differs (paired within session); the breadth contrast is the
    # Dirichlet-multinomial fit of how widely an event spreads over arms.
    _run(_ARM, _CL, 2, threads=2),
    _run("content.arm_ladder.leaf_arms", f"{_CL} --leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu", 2, threads=2),
    # The same ladders fit on SWR events alone and on non-SWR events alone, the patch
    # ladder, and the arm ladder under the Dirichlet-multinomial estimator.
    _run("content.arm_ladder.swr_only", f"{_CL} --swr-subset T", 2, threads=2),
    _run("content.arm_ladder.non_swr_only", f"{_CL} --swr-subset F", 2, threads=2),
    _run("content.arm_ladder.dirichlet_multinomial", f"{_CL} {_DM} --n-jobs 16", 16),
    _run("content.patch_ladder.all_events", f"{_CL} --hierarchy patch", 1),
    _run("content.patch_ladder.swr_only", f"{_CL} --hierarchy patch --swr-subset T", 1),
    _run("content.patch_ladder.non_swr_only", f"{_CL} --hierarchy patch --swr-subset F", 1),
    # The contrasts write into models/, which the arm ladder replaces whole.
    _run("content.swr_contrast.all", f"{_KC} --outcome-stratum all", 2, [_ARM], threads=2),
    _run("content.swr_contrast.rewarded", f"{_KC} --outcome-stratum rewarded", 2, [_ARM], threads=2),
    _run("content.swr_contrast.omitted", f"{_KC} --outcome-stratum omitted", 2, [_ARM], threads=2),
    _run("content.breadth_contrast.all", f"{_KC} {_DM} --n-jobs 16 --outcome-stratum all", 16, [_ARM]),
    _run("content.breadth_contrast.all_length_adjusted", f"{_KC} {_DM} --n-jobs 16 --concentration-length --outcome-stratum all", 16, [_ARM]),
    _run("content.breadth_contrast.rewarded", f"{_KC} {_DM} --n-jobs 8 --outcome-stratum rewarded", 8, [_ARM]),
    _run("content.breadth_contrast.omitted", f"{_KC} {_DM} --n-jobs 8 --outcome-stratum omitted", 8, [_ARM]),
    _run("content.estimator_comparison", "scripts/content/compare_content_estimators.py --n-jobs 16", 16),
    _run("content.uncertainty", "scripts/content/analyze_uncertainty_content.py", 1, ["content.arm_ladder.leaf_arms"]),
    # Content of events found by SWR- or MUA-anchored detection (ContFrag).
    _run("content.event_source.swr_arm", f"{_CL} --event-source SWR --brain-area HPC", 2, threads=2),
    _run("content.event_source.swr_patch", f"{_CL} --event-source SWR --hierarchy patch --brain-area HPC", 2, threads=2),
    _run("content.event_source.mua_arm", f"{_CL} --event-source MUA --brain-area HPC", 2, threads=2),
    _run("content.event_source.mua_patch", f"{_CL} --event-source MUA --hierarchy patch --brain-area HPC", 2, threads=2),
    # Draws the Q2 rate contrast and the Q3b content contrast side by side.
    _fig("content.fig.swr_contrast_coefs", "scripts/figures/plot_swr_contrast_coefs.py",
         ["rate.swr_contrast.at_well", "content.swr_contrast.all"]),
    _fig("content.fig.uncertainty", "scripts/figures/generate_uncertainty_content_figure.py", ["content.uncertainty"]),
    _fig("content.fig.ladder", "scripts/content/generate_replay_content_figures.py", [_ARM]),
    _fig("content.fig.pipeline_data_flow", "scripts/figures/generate_supp_pipeline_data_flow.py", [_ARM]),
    # Q4 behaviour. One suite per event source: the choice prediction (claim 19d), plus
    # patch, per-stop rate, patch-switch and switch-aligned blocks on one common cohort.
    _run("behavior.event_source_suite.non_local", f"{_ES} --event-source non_local", 1),
    _run("behavior.event_source_suite.swr", f"{_ES} --event-source SWR", 1),
    _run("behavior.event_source_suite.mua", f"{_ES} --event-source MUA", 1),
    # Content during a stop against the arm visited next, and content x outcome -> stay or leave.
    _run("behavior.retrospective.all", _S5, 2, threads=2),
    _run("behavior.retrospective.swr", f"{_S5} --swr-only", 2, threads=2),
    _run("behavior.retrospective.non_swr", f"{_S5} --non-swr-only", 2, threads=2),
    _run("behavior.stay_leave.at_well", f"{_S6} --rebuild-summary", 2, threads=2),
    # Opt-in: the all-stops stay/leave sensitivity is cited nowhere.
    _run("behavior.stay_leave.all_stops", f"{_S6} --stop-cohort all_stops --rebuild-summary", 2, threads=2, optional=True),
    _run("behavior.stay_leave.at_well_swr", f"{_S6} --content-stratum swr", 2, ["behavior.stay_leave.at_well"], threads=2),
    _run("behavior.stay_leave.at_well_non_swr", f"{_S6} --content-stratum non_swr", 2, ["behavior.stay_leave.at_well"], threads=2),
    _run("behavior.stay_leave.at_well_mua", f"{_S6} --content-stratum mua", 2, ["behavior.stay_leave.at_well"], threads=2),
    _run("behavior.stay_leave.all_stops_swr", f"{_S6} --stop-cohort all_stops --content-stratum swr", 2, ["behavior.stay_leave.all_stops"], threads=2, optional=True),
    _run("behavior.stay_leave.all_stops_non_swr", f"{_S6} --stop-cohort all_stops --content-stratum non_swr", 2, ["behavior.stay_leave.all_stops"], threads=2, optional=True),
    _run("behavior.stay_leave.all_stops_mua", f"{_S6} --stop-cohort all_stops --content-stratum mua", 2, ["behavior.stay_leave.all_stops"], threads=2, optional=True),
)
# fmt: on

BY_NAME = {r.name: r for r in RUNS}
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
#: Fit roots whose provenance sidecars are checked, relative to the data root.
FIT_ROOTS = ("replay_content", "contfrag_replay_content")
_SIDECAR = ".provenance.json"
_SKIPPED_DIRS = ("_backup", "_archive", "_pre_", "_staging", "_golden")


def select(
    names: list[str] | None,
    with_dependents: bool = True,
    include_optional: bool = False,
) -> list[Run]:
    """The runs named (all when ``names`` is None), in list order.

    A name selects the run of that name and every run whose name extends it, so
    ``content`` selects all of Q3b and ``content.swr_contrast`` its three strata.
    With ``with_dependents``, every run that needs a selected run, directly or
    through another, is selected too.

    An optional run is left out unless ``include_optional``, or it is named in
    full, or it depends on an optional run that is selected.
    """

    def wanted(run: Run, name: str | None) -> bool:
        return not run.optional or include_optional or run.name == name

    if not names:
        chosen = {r.name for r in RUNS if wanted(r, None)}
    else:
        chosen = set()
        for name in names:
            matches = [
                r for r in RUNS if r.name == name or r.name.startswith(name + ".")
            ]
            if not matches:
                raise SystemExit(
                    f"no run is named {name!r} or starts with it; see --dry-run"
                )
            chosen |= {r.name for r in matches if wanted(r, name)}
    if with_dependents:
        grew = True
        while grew:
            more = {
                r.name
                for r in RUNS
                if r.name not in chosen
                and (needed := set(r.deps) & chosen)
                and (wanted(r, None) or any(BY_NAME[d].optional for d in needed))
            }
            chosen |= more
            grew = bool(more)
    return [r for r in RUNS if r.name in chosen]


def run_env(run: Run) -> dict[str, str]:
    """The environment for one run: CPU-only JAX and its thread cap."""
    env = dict(os.environ)
    env.update({var: str(run.threads) for var in THREAD_VARS})
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def schedule(
    runs: list[Run],
    cpus: list[int],
    log_dir: Path,
    prefix: list[str],
    poll_s: float = 5.0,
) -> dict[str, int | None]:
    """Run ``runs`` on ``cpus``; return each run's exit code (None: not started).

    A run starts when all its selected dependencies exited 0 and ``run.cores``
    of the CPUs are free; it is pinned to those CPUs. A run that needs more
    CPUs than there are starts on all of them once nothing else runs. A
    dependency outside ``runs`` is taken as already done. On any exception the
    runs still going are terminated.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    names = {r.name for r in runs}
    pending = list(runs)
    active: dict[str, tuple[subprocess.Popen, Run, list[int], float, str]] = {}
    free = list(cpus)
    codes: dict[str, int | None] = {r.name: None for r in runs}
    blocked: set[str] = set()
    from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
    from continuum_swr_replay.runs.provenance import git

    # check=False: a refit batch records an empty commit rather than not running
    commit = git(REPO, "rev-parse", "HEAD", check=False)
    dirty = bool(
        git(REPO, "status", "--porcelain", "--untracked-files=no", check=False)
    )

    def record(run: Run, **fields) -> None:
        body = {
            "name": run.name,
            "question": QUESTIONS.get(run.question, run.question),
            "argv": [*prefix, *run.argv],
            "threads": run.threads,
            "env": {var: str(run.threads) for var in THREAD_VARS},
            "commit": commit,
            "tree_dirty": dirty,
            "processed_data_dir": str(PROCESSED_DATA_DIR),
            "host": socket.gethostname(),
            **fields,
        }
        path = log_dir / f"{run.name}.json"
        path.write_text(json.dumps(body, indent=1) + "\n")

    try:
        while pending or active:
            for name, (proc, run, mine, t0, started) in list(active.items()):
                if proc.poll() is None:
                    continue
                codes[name] = proc.returncode
                del active[name]
                free.extend(mine)
                record(
                    run,
                    status="done",
                    cpus=mine,
                    start_utc=started,
                    end_utc=_now(),
                    wall_seconds=round(time.monotonic() - t0, 1),
                    exit_code=proc.returncode,
                    log=str(log_dir / f"{name}.log"),
                )
                print(f"{_now()} {name} exit {proc.returncode}", flush=True)
            for run in list(pending):
                needs = [d for d in run.deps if d in names]
                if any(d in blocked or codes[d] not in (None, 0) for d in needs):
                    pending.remove(run)
                    blocked.add(run.name)
                    failed = [
                        d for d in needs if d in blocked or codes[d] not in (None, 0)
                    ]
                    record(run, status="blocked", blocked_by=failed, exit_code=None)
                    print(
                        f"{_now()} {run.name} not started: {failed} failed", flush=True
                    )
                    continue
                if any(codes[d] != 0 for d in needs):  # still running
                    continue
                if run.cores <= len(free):
                    mine = free[: run.cores]
                elif not active and len(free) == len(cpus):
                    mine = list(free)
                else:
                    continue
                free = [c for c in free if c not in mine]
                log = open(log_dir / f"{run.name}.log", "w")
                proc = subprocess.Popen(
                    [*prefix, *run.argv],
                    cwd=REPO,
                    env=run_env(run),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=lambda cpus_=tuple(mine): os.sched_setaffinity(0, cpus_),
                )
                log.close()
                active[run.name] = (proc, run, mine, time.monotonic(), _now())
                pending.remove(run)
                print(f"{_now()} {run.name} start on {len(mine)} CPUs", flush=True)
            if pending and not active:
                # Dependencies come earlier in RUNS, so this cannot happen.
                raise RuntimeError(f"nothing can start: {[r.name for r in pending]}")
            if active:
                time.sleep(poll_s)
    finally:
        for proc, *_ in active.values():
            proc.terminate()
    return codes


def sidecar_times(root: Path) -> dict[Path, float]:
    """Every fit-provenance sidecar under ``root`` with its modification time.

    Backup, archive, staging and golden-baseline folders are left out.
    """
    times = {}
    for sidecar in root.rglob(f"*{_SIDECAR}"):
        if not any(
            p.startswith(_SKIPPED_DIRS) for p in sidecar.relative_to(root).parts
        ):
            times[sidecar] = sidecar.stat().st_mtime
    return times


def written_artifacts(root: Path, before: dict[Path, float]) -> list[Path]:
    """The fit artifacts under ``root`` whose sidecar is new or changed since ``before``.

    Comparing against a snapshot, not the driver's clock, keeps this right when
    the file server's clock differs. One artifact per sidecar: the ``.pkl`` or
    ``.parquet`` file it describes (other files beside it, such as a
    ``.z_reference.json``, have no sidecar of their own).
    """
    artifacts = []
    for sidecar, mtime in sorted(sidecar_times(root).items()):
        if before.get(sidecar) == mtime:
            continue
        stem = sidecar.name[: -len(_SIDECAR)]
        found = [sidecar.with_name(stem + ext) for ext in (".pkl", ".parquet")]
        found = [f for f in found if f.exists()] or found[:1]
        artifacts += found
    return artifacts


def check_provenance(log_dir: Path, before: dict[str, dict[Path, float]]) -> int:
    """``check_fit_provenance.py`` on what the runs wrote; the worst exit code.

    Only the artifacts whose sidecar changed from ``before`` (per fit root) are checked, so a bundle no run
    makes any more (left on disk) cannot fail a run that did not touch it.
    """
    from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

    worst = 0
    for name in FIT_ROOTS:
        root = Path(PROCESSED_DATA_DIR) / name
        artifacts = (
            written_artifacts(root, before.get(name, {})) if root.is_dir() else []
        )
        if not artifacts:
            continue
        with open(log_dir / f"check_fit_provenance_{name}.log", "w") as log:
            code = subprocess.run(
                [
                    sys.executable,
                    "scripts/qa/check_fit_provenance.py",
                    "--root",
                    str(root),
                    *map(str, artifacts),
                ],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        print(
            f"{_now()} check_fit_provenance {name}: {len(artifacts)} artifacts, exit {code}",
            flush=True,
        )
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
            deps += "   (optional)" if r.optional else ""
            print(f"  {r.name:<46} {r.cores:>2} CPUs  {' '.join(r.argv)}{deps}")
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
        help="comma-separated run names or name prefixes (rate, content, behavior, "
        "content.swr_contrast, ...)",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="also run the opt-in sensitivity runs (marked 'optional' in --dry-run)",
    )
    parser.add_argument(
        "--no-dependents",
        action="store_true",
        help="do not add the runs that depend on the selected ones",
    )
    parser.add_argument(
        "--cpus",
        help="CPUs to share out among the runs, e.g. 0-23 "
        "(default: every CPU this process may use)",
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

    runs = select(
        args.only.split(",") if args.only else None,
        not args.no_dependents,
        args.include_optional,
    )
    if args.dry_run:
        print_plan(runs)
        return 0
    allowed = sorted(os.sched_getaffinity(0))
    cpus = _cpu_list(args.cpus) if args.cpus else allowed
    if not set(cpus) <= set(allowed):
        parser.error(f"--cpus {args.cpus} is not within the CPUs this process may use")
    log_dir = (
        args.log_dir or REPO / "logs" / f"refits_{datetime.now(UTC):%Y%m%dT%H%M%S}"
    )
    from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

    roots = {name: Path(PROCESSED_DATA_DIR) / name for name in FIT_ROOTS}
    before = {n: sidecar_times(r) for n, r in roots.items() if r.is_dir()}
    # A SIGTERM or SIGHUP leaves through schedule's cleanup, which stops the runs.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, _frame: sys.exit(128 + signum))
    codes = schedule(runs, cpus, log_dir, ["uv", "run", "--no-sync", "python"])
    failed = sorted(name for name, code in codes.items() if code != 0)
    print(
        f"{len(codes) - len(failed)}/{len(codes)} runs exited 0; not ok: {failed or 'none'}"
    )
    fits = any(not r.figure for r in runs)
    worst = check_provenance(log_dir, before) if fits and not args.no_check else 0
    return 1 if failed or worst else 0


if __name__ == "__main__":
    sys.exit(main())
