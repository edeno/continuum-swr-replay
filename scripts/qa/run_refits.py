#!/usr/bin/env python
"""Rerun the CPU model fits and the figures that read them, in dependency order.

Every run answers one of the paper's questions (``QUESTIONS``; the full map is
``docs/analysis_map.md``). A run's name says what it is:
``<question>.<analysis>[.<variant>]``, for example ``rate.swr_contrast.at_well``.
Each run is one command, the number of CPUs it gets and the runs it needs first.
Selecting a run also selects every run that depends on it, so
``--only content.arm_ladder.all_arms`` refits the content contrasts too. They
are ordered after it, though their outputs now sit in ``models_swr_contrast``,
which that run does not rewrite.

With ``--with-tables`` the canonical tables are rebuilt first, as ``tables.*``
runs, in the order their code requires:

1. ``tables.stop_bootstrap``: ``prepare_replay_content_data.py --stop-only``,
   the stop table ``qvals_table`` is aligned to;
2. ``tables.qvals``: ``build_qvals_table.py --write``, which reads that stop
   table (and Alison's DataJoint schemas);
3. ``tables.replay_content``: ``prepare_replay_content_data.py``, the event,
   event x arm, event x patch, sequence and stop tables, which reads
   ``qvals_table``. It is the stop table's canonical producer (the bootstrap's
   table plus the SWR split); ``regenerate_stop_table.py`` is not a run, as it
   assumes the stop bouts of an existing event table;
4. ``tables.timebin``: ``build_timebin_tables.py``, the event-onset and
   stop-timebin tables, from the decodes alone;
5. ``tables.contfrag.swr`` and ``tables.contfrag.mua``:
   ``prepare_contfrag_replay_content_data.py``, which reads ``qvals_table``.

Each fit needs the table runs it reads. With ``--only``, ``--with-tables`` adds
only the table runs the selected runs need. The tables are written where the
fits read them, under ``PROCESSED_DATA_DIR``. The bootstrap refuses to replace
a complete stop table (one with the SWR split), so a rebuild runs on a data root
that has none (``CSR_PROCESSED_DATA_DIR``); in a root that has one it fails at
once and blocks every run after it.

Prerequisites, not run here (GPUs or hours each), in this order:

1. ``scripts/detection/run_artifact_detection.py``: the artifact mask;
2. ``scripts/detection/run_swr_detection.py``: the SWR inventories, masked with
   that artifact run;
3. pin both runs, ``ARTIFACT_INTERVALS_RUN_ID`` and ``SWR_INVENTORY_RUN_ID`` in
   ``analysis/events/swr_source.py``;
4. ``scripts/decoding/run_decoding_bandit.py``: the HPC non-local decodes;
5. ``scripts/decoding/run_decoding_contfrag.py``: the ContFrag decodes, for
   ``tables.contfrag.*``.

Before a table run starts, a preflight checks they are on disk: both pinned run
folders, the inventory masked with the pinned artifact run
(``decode_masks.require_hpc_decode_sources``), and each analysed epoch's decode
files. It reads files only, never the database. ``--dry-run`` lists the
prerequisites and, when table runs are selected, what the preflight found.

Each run is pinned to its own CPUs, taken from ``--cpus``, so its BLAS, numba
and JAX thread pools (which size themselves from the CPUs a process may use)
cannot spill onto another run's. OMP, MKL, OpenBLAS and numba threads are also
capped at the run's ``threads``. Each run writes a log and a JSON record (the
command, its CPUs and environment, the commit, the data root, times and exit
code) under ``--log-dir``; a run whose dependency failed gets a "blocked"
record. Then ``check_fit_provenance.py`` checks every artifact the runs wrote.

Not here: the prerequisites above, the Q1 checks, and the PFC unit GLM (Q5),
which runs on GPUs for 14-17 h per card under its own launcher; Figure 4 reads
it and is drawn after it.

Usage
-----
    uv run --no-sync python scripts/qa/run_refits.py --dry-run
    uv run --no-sync python scripts/qa/run_refits.py --with-tables --dry-run
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
#: name. Q1 (are the events real) and Q5 (PFC) have no CPU fit here; of Q3a
#: (what the events are like) only the SWR-coupling model is one. ``tables``
#: answers none: it is the opt-in table rebuild every fit reads.
TABLES = "tables"
QUESTIONS = {
    TABLES: "Tables (opt-in, --with-tables): the canonical tables the fits read.",
    "rate": "Q2. How often do non-local events occur, and what drives the rate, for SWR against non-SWR events?",
    "coupling": "Q3a. What are the events like: location, speed, extent, certainty, duration, SWR coupling?",
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
_PR = "scripts/tables/prepare_replay_content_data.py"
_CF = "scripts/tables/prepare_contfrag_replay_content_data.py"
_QV = "tables.qvals"
# The table runs a fit reads: the replay-content tables (with qvals_table, which
# they need), the event-onset table, and the ContFrag SWR or MUA tables.
_RT = ("tables.replay_content",)
_RTB = (*_RT, "tables.timebin")
_CFS = ("tables.contfrag.swr",)
_CFM = ("tables.contfrag.mua",)

# Cores: a run's worker count, or its thread cap for a run without workers.
# The at_well cohort is the shipped one; all_stops and in_transit are sensitivities.
# fmt: off
RUNS: tuple[Run, ...] = (
    # Tables, opt-in (--with-tables). Each builder runs serially over the sessions.
    _run("tables.stop_bootstrap", f"{_PR} --stop-only", 1),
    _run(_QV, "scripts/behavior/build_qvals_table.py --write", 1, ["tables.stop_bootstrap"]),
    _run(_RT[0], _PR, 1, [_QV]),
    _run("tables.timebin", "scripts/tables/build_timebin_tables.py", 1),
    _run(_CFS[0], f"{_CF} --event-source SWR", 1, [_QV]),
    _run(_CFM[0], f"{_CF} --event-source MUA", 1, [_QV]),
    # Q2 rate. The per-bin conditional intensity (Family B) is the main model; the
    # per-stop Poisson (Family A) is the literature-comparable one.
    _run("rate.per_bin.at_well", f"{_PB} --stop-cohort at_well --workers 16", 16, _RTB),
    _run("rate.per_bin.all_stops", f"{_PB} --stop-cohort all_stops --workers 16", 16, _RTB),
    # Opt-in: thin (about 50 sessions) and cited nowhere.
    _run("rate.per_bin.in_transit", f"{_PB} --stop-cohort in_transit --workers 10", 10, _RTB, optional=True),
    _run("rate.per_stop.at_well", f"{_PS} --stop-cohort at_well", 1, _RT),
    _run("rate.per_stop.all_stops", f"{_PS} --stop-cohort all_stops", 1, _RT),
    _run("rate.swr_contrast.at_well", f"{_RC} --stop-cohort at_well --workers 16", 16, _RTB),
    _run("rate.swr_contrast.all_stops", f"{_RC} --stop-cohort all_stops --workers 16", 16, _RTB),
    _run("rate.uncertainty.at_well", "scripts/rate/fit_uncertainty_rate.py --stop-cohort at_well", 1, _RTB),
    _run("coupling.ripple_power", "scripts/coupling/fit_swr_power_model.py", 1, _RT),
    # No worker flag: its parallelism is BLAS threads.
    _run("rate.cross_history", "scripts/rate/fit_cross_history_rate.py --write", 8, _RTB, threads=8),
    _fig("rate.fig.per_bin_ladder", "scripts/rate/plot_per_bin_rate_models.py", ["rate.per_bin.at_well"]),
    _fig("rate.fig.per_bin_summary", "scripts/rate/plot_per_bin_rate_summary.py", ["rate.per_bin.at_well"]),
    _fig("rate.fig.model_checking", "scripts/rate/plot_model_checking.py", ["rate.per_bin.at_well"]),
    _fig("rate.doc.per_bin_results", "scripts/rate/summarize_per_bin_rate_models.py", ["rate.per_bin.at_well"]),
    _fig("rate.fig.uncertainty", "scripts/rate/plot_uncertainty_rate.py", ["rate.uncertainty.at_well"]),
    # Q3b content. The arm ladder asks what shapes content; the SWR contrast asks how
    # SWR content differs (paired within session); the breadth contrast is the
    # Dirichlet-multinomial fit of how widely an event spreads over arms.
    _run(_ARM, _CL, 2, _RT, threads=2),
    _run("content.arm_ladder.leaf_arms", f"{_CL} --leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu", 2, _RT, threads=2),
    # The same ladders fit on SWR events alone and on non-SWR events alone, the patch
    # ladder, and the arm ladder under the Dirichlet-multinomial estimator.
    _run("content.arm_ladder.swr_only", f"{_CL} --swr-subset T", 2, _RT, threads=2),
    _run("content.arm_ladder.non_swr_only", f"{_CL} --swr-subset F", 2, _RT, threads=2),
    _run("content.arm_ladder.dirichlet_multinomial", f"{_CL} {_DM} --n-jobs 16", 16, _RT),
    _run("content.patch_ladder.all_events", f"{_CL} --hierarchy patch", 1, _RT),
    _run("content.patch_ladder.swr_only", f"{_CL} --hierarchy patch --swr-subset T", 1, _RT),
    _run("content.patch_ladder.non_swr_only", f"{_CL} --hierarchy patch --swr-subset F", 1, _RT),
    # The contrasts write into models_swr_contrast/, not the models/ the arm ladder
    # replaces whole, so running after it is ordering only, not protection.
    _run("content.swr_contrast.all", f"{_KC} --outcome-stratum all", 2, [_ARM, *_RT], threads=2),
    _run("content.swr_contrast.rewarded", f"{_KC} --outcome-stratum rewarded", 2, [_ARM, *_RT], threads=2),
    _run("content.swr_contrast.omitted", f"{_KC} --outcome-stratum omitted", 2, [_ARM, *_RT], threads=2),
    _run("content.breadth_contrast.all", f"{_KC} {_DM} --n-jobs 16 --outcome-stratum all", 16, [_ARM, *_RT]),
    _run("content.breadth_contrast.all_length_adjusted", f"{_KC} {_DM} --n-jobs 16 --concentration-length --outcome-stratum all", 16, [_ARM, *_RT]),
    _run("content.breadth_contrast.rewarded", f"{_KC} {_DM} --n-jobs 8 --outcome-stratum rewarded", 8, [_ARM, *_RT]),
    _run("content.breadth_contrast.omitted", f"{_KC} {_DM} --n-jobs 8 --outcome-stratum omitted", 8, [_ARM, *_RT]),
    _run("content.estimator_comparison", "scripts/content/compare_content_estimators.py --n-jobs 16", 16, _RT),
    _run("content.uncertainty", "scripts/content/analyze_uncertainty_content.py", 1, ["content.arm_ladder.leaf_arms", *_RT]),
    # Content of events found by SWR- or MUA-anchored detection (ContFrag).
    _run("content.event_source.swr_arm", f"{_CL} --event-source SWR --brain-area HPC", 2, _CFS, threads=2),
    _run("content.event_source.swr_patch", f"{_CL} --event-source SWR --hierarchy patch --brain-area HPC", 2, _CFS, threads=2),
    _run("content.event_source.mua_arm", f"{_CL} --event-source MUA --brain-area HPC", 2, _CFM, threads=2),
    _run("content.event_source.mua_patch", f"{_CL} --event-source MUA --hierarchy patch --brain-area HPC", 2, _CFM, threads=2),
    # Draws the Q2 rate contrast and the Q3b content contrast side by side.
    _fig("content.fig.swr_contrast_coefs", "scripts/figures/plot_swr_contrast_coefs.py",
         ["rate.swr_contrast.at_well", "content.swr_contrast.all"]),
    _fig("content.fig.uncertainty", "scripts/figures/generate_uncertainty_content_figure.py", ["content.uncertainty"]),
    _fig("content.fig.ladder", "scripts/content/generate_replay_content_figures.py", [_ARM]),
    _fig("content.fig.pipeline_data_flow", "scripts/figures/generate_supp_pipeline_data_flow.py", [_ARM]),
    # Q4 behaviour. One suite per event source: the choice prediction (claim 19d), plus
    # patch, per-stop rate, patch-switch and switch-aligned blocks on one common cohort.
    _run("behavior.event_source_suite.non_local", f"{_ES} --event-source non_local", 1, _RT),
    _run("behavior.event_source_suite.swr", f"{_ES} --event-source SWR", 1, _CFS),
    _run("behavior.event_source_suite.mua", f"{_ES} --event-source MUA", 1, _CFM),
    # Content during a stop against the arm visited next, and content x outcome -> stay or leave.
    _run("behavior.retrospective.all", _S5, 2, _RT, threads=2),
    _run("behavior.retrospective.swr", f"{_S5} --swr-only", 2, _RT, threads=2),
    _run("behavior.retrospective.non_swr", f"{_S5} --non-swr-only", 2, _RT, threads=2),
    _run("behavior.stay_leave.at_well", f"{_S6} --rebuild-summary", 2, _RT, threads=2),
    # Opt-in: the all-stops stay/leave sensitivity is cited nowhere.
    _run("behavior.stay_leave.all_stops", f"{_S6} --stop-cohort all_stops --rebuild-summary", 2, _RT, threads=2, optional=True),
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
_SIDECAR = ".provenance.json"
_SKIPPED_DIRS = ("_backup", "_archive", "_pre_", "_staging", "_golden")


def select(
    names: list[str] | None,
    with_dependents: bool = True,
    include_optional: bool = False,
    with_tables: bool = False,
) -> list[Run]:
    """The runs named (all when ``names`` is None), in list order.

    A name selects the run of that name and every run whose name extends it, so
    ``content`` selects all of Q3b and ``content.swr_contrast`` its three strata.
    With ``with_dependents``, every run that needs a selected run, directly or
    through another, is selected too.

    An optional run is left out unless ``include_optional``, or it is named in
    full, or it depends on an optional run that is selected.

    The table runs (``tables.*``) are left out unless named, or
    ``with_tables``: then every table run the selected runs need, directly or
    through another table run, is selected too.
    """

    def wanted(run: Run, name: str | None) -> bool:
        return not run.optional or include_optional or run.name == name

    if not names:
        chosen = {
            r.name
            for r in RUNS
            if wanted(r, None) and (with_tables or r.question != TABLES)
        }
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
    if with_tables:
        grew = True
        while grew:
            more = {
                d
                for r in RUNS
                if r.name in chosen
                for d in r.deps
                if d not in chosen and BY_NAME[d].question == TABLES
            }
            chosen |= more
            grew = bool(more)
    return [r for r in RUNS if r.name in chosen]


#: What the table runs read and nothing here makes, in the order it is made:
#: (script or step, what it makes).
PREREQUISITES = (
    ("scripts/detection/run_artifact_detection.py", "the artifact mask"),
    (
        "scripts/detection/run_swr_detection.py",
        "the SWR inventories, masked with that artifact run",
    ),
    (
        "pin ARTIFACT_INTERVALS_RUN_ID and SWR_INVENTORY_RUN_ID",
        "in analysis/events/swr_source.py",
    ),
    ("scripts/decoding/run_decoding_bandit.py", "the HPC non-local decodes"),
    (
        "scripts/decoding/run_decoding_contfrag.py",
        "the ContFrag decodes, for tables.contfrag.*",
    ),
)


def missing_prerequisites(runs: list[Run]) -> list[str]:
    """What the selected table runs need from :data:`PREREQUISITES` that is
    not on disk; empty when no table run is selected.

    Checks both pinned run folders, that the SWR inventory was masked with the
    pinned artifact run (``decode_masks.require_hpc_decode_sources``), and that
    every analysed epoch (from the epoch register) has its HPC decode files,
    and its ContFrag ones when a ContFrag table run is selected. Reads files
    only, never the database; imported here so the scheduler loads no package
    code.
    """
    names = {r.name for r in runs if r.question == TABLES}
    if not names:
        return []
    from continuum_swr_replay.analysis.events import swr_source
    from continuum_swr_replay.analysis.hippocampus.decode_masks import (
        require_hpc_decode_sources,
    )
    from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids
    from continuum_swr_replay.decoding.files import decode_paths
    from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

    problems = [
        f"no pinned run folder {folder}"
        for folder in (
            swr_source.artifact_intervals_dir(),
            swr_source.swr_inventory_dir(),
        )
        if folder is not None and not folder.is_dir()
    ]
    try:
        require_hpc_decode_sources()
    except ValueError as error:
        problems.append(str(error))
    decodes = {"HPC": ("", {}, ("data_over_time", "environment", "results"))}
    if names & {*_CFS, *_CFM}:
        decodes["ContFrag"] = (
            "_contfrag",
            {"threshold_percent": None},
            ("data_over_time", "environment"),
        )
    epoch_ids = analysed_epoch_ids()
    for kind, (suffix, naming, needed) in decodes.items():
        missing = [
            path.name
            for epoch_id in epoch_ids
            for name, path in decode_paths(
                epoch_id + suffix, PROCESSED_DATA_DIR, **naming
            )
            ._asdict()
            .items()
            if name in needed and not path.exists()
        ]
        if missing:
            problems.append(
                f"{len(missing)} {kind} decode files of the {len(epoch_ids)} "
                f"analysed epochs are not in {PROCESSED_DATA_DIR}: "
                + ", ".join(missing[:5])
                + (", ..." if len(missing) > 5 else "")
            )
    return problems


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


def fit_roots() -> dict[str, Path]:
    """The fit roots whose provenance sidecars are checked, by name.

    Imported here, not at the top, so the scheduler loads no package code.
    """
    from continuum_swr_replay.paths import (
        CONTFRAG_REPLAY_CONTENT_DIR,
        REPLAY_CONTENT_DIR,
    )

    return {
        "replay_content": REPLAY_CONTENT_DIR,
        "contfrag_replay_content": CONTFRAG_REPLAY_CONTENT_DIR,
    }


def check_provenance(log_dir: Path, before: dict[str, dict[Path, float]]) -> int:
    """``check_fit_provenance.py`` on what the runs wrote; the worst exit code.

    Only the artifacts whose sidecar changed from ``before`` (per fit root) are checked, so a bundle no run
    makes any more (left on disk) cannot fail a run that did not touch it.
    """
    worst = 0
    for name, root in fit_roots().items():
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
    """The prerequisites, then the selected runs grouped under their question.

    A table run a run needs is shown only when it is selected.
    """
    print("Prerequisites, not run here, in this order:")
    for i, (step, makes) in enumerate(PREREQUISITES, start=1):
        print(f"  {i}. {step:<54} {makes}")
    print()
    selected = {r.name for r in runs}
    for key, text in QUESTIONS.items():
        group = [r for r in runs if r.question == key]
        if not group:
            continue
        print(text)
        for r in group:
            shown = [
                d for d in r.deps if d in selected or BY_NAME[d].question != TABLES
            ]
            deps = f"   after {', '.join(shown)}" if shown else ""
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
        "--with-tables",
        action="store_true",
        help="first rebuild the tables the selected runs read (the tables.* runs; "
        "they read Spyglass, and need a data root with no complete stop table)",
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
        args.with_tables,
    )
    problems = missing_prerequisites(runs)
    if args.dry_run:
        print_plan(runs)
        if any(r.question == TABLES for r in runs):
            print(f"Preflight: {f'{len(problems)} problems' if problems else 'ok'}")
            for problem in problems:
                print(f"  {problem}")
        return 0
    if problems:
        print("The table runs' prerequisites are not on disk:", *problems, sep="\n  ")
        return 1
    allowed = sorted(os.sched_getaffinity(0))
    cpus = _cpu_list(args.cpus) if args.cpus else allowed
    if not set(cpus) <= set(allowed):
        parser.error(f"--cpus {args.cpus} is not within the CPUs this process may use")
    log_dir = (
        args.log_dir or REPO / "logs" / f"refits_{datetime.now(UTC):%Y%m%dT%H%M%S}"
    )
    roots = fit_roots()
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
