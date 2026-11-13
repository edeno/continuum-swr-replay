#!/usr/bin/env python
"""Objective AST mutation check for value-critical builder logic.

Measures whether the test suite would catch a covariate-value bug. For each
target module it mechanically enumerates value-critical mutations -- comparison
operators, ``searchsorted`` ``side=`` literals, and ``+``/``-`` binops -- applies
one at a time to the source file, runs a named fast test subset with ``pytest
-x``, and records whether the suite catches it (killed) or not (survived).
Survivors are coverage gaps: a wrong covariate value that would ship silently.

The check is deterministic and dependency-free (stdlib ``ast`` + ``subprocess``).
Each mutation restores the source in a ``finally``; a run that dies mid-mutation
leaves a modified file that ``git checkout`` restores.

Usage:
    uv run python scripts/checks/mutation_check.py            # default builders + tests
    uv run python scripts/checks/mutation_check.py --targets src/.../reward_history.py \
        --tests tests/analysis/behavior/covariates/test_covariates.py
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_DEFAULT_TARGETS = (
    "src/continuum_swr_replay/analysis/behavior/covariates/candidate_arm.py",
    "src/continuum_swr_replay/analysis/behavior/covariates/history/_common.py",
    "src/continuum_swr_replay/analysis/behavior/covariates/history/reward_history.py",
    "src/continuum_swr_replay/analysis/behavior/covariates/history/visit_recency.py",
    "src/continuum_swr_replay/analysis/behavior/covariates/history/value_uncertainty.py",
    "src/continuum_swr_replay/analysis/behavior/covariates/history/value_block.py",
    "src/continuum_swr_replay/analysis/behavior/covariates/history/diagnostics.py",
)
_DEFAULT_TESTS = (
    "tests/analysis/behavior/covariates/test_covariates.py",
    "tests/analysis/behavior/covariates/test_uncertainty_covariates.py",
    "tests/analysis/replay_models/test_covariate_timing.py",
    "tests/analysis/behavior/covariates/history/test_builder_value_floor.py",
    "tests/analysis/behavior/covariates/test_recent_trial_reward_rate.py",
    "tests/analysis/behavior/covariates/test_recent_reward_rate.py",
)
_CMP_SWAP = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}


def _enumerate(tree: ast.AST) -> list[tuple[int, str, tuple]]:
    """Every value-critical mutation site, in a stable walk order."""
    muts: list[tuple[int, str, tuple]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            swap = _CMP_SWAP.get(type(node.ops[0]))
            if swap is not None:
                muts.append(
                    (
                        node.lineno,
                        f"{type(node.ops[0]).__name__}->{swap.__name__}",
                        ("cmp", node, swap),
                    )
                )
        elif isinstance(node, ast.Constant) and node.value in ("right", "left"):
            other = "left" if node.value == "right" else "right"
            muts.append(
                (node.lineno, f'side "{node.value}"->"{other}"', ("str", node, other))
            )
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            swap = ast.Sub if isinstance(node.op, ast.Add) else ast.Add
            muts.append(
                (
                    node.lineno,
                    f"{type(node.op).__name__}->{swap.__name__}",
                    ("binop", node, swap),
                )
            )
    return muts


def _apply(kind: str, node: ast.AST, payload) -> None:
    if kind == "cmp":
        node.ops[0] = payload()
    elif kind == "binop":
        node.op = payload()
    elif kind == "str":
        node.value = payload


def _run_tests(tests: tuple[str, ...]) -> int:
    cmd = ["python", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests]
    return subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, timeout=1200
    ).returncode


def check(targets: tuple[str, ...], tests: tuple[str, ...]) -> int:
    if _run_tests(tests) != 0:
        print("BASELINE NOT GREEN -- aborting", file=sys.stderr)
        return 2

    survivors: list[str] = []
    total = 0
    for rel in targets:
        target = REPO / rel
        src = target.read_text()
        muts = _enumerate(ast.parse(src))
        print(f"### {target.name}: {len(muts)} mutations")
        for i, (lineno, desc, _) in enumerate(muts):
            tree = ast.parse(src)
            _l, _d, (kind, node, payload) = _enumerate(tree)[i]
            _apply(kind, node, payload)
            try:
                mutated = ast.unparse(ast.fix_missing_locations(tree))
            except Exception as exc:  # noqa: BLE001
                print(f"  L{lineno} {desc}: UNPARSE-ERROR {exc}")
                continue
            t0 = time.time()
            try:
                target.write_text(mutated)
                killed = _run_tests(tests) != 0
            finally:
                target.write_text(src)
            total += 1
            status = "KILLED" if killed else "SURVIVED"
            if not killed:
                survivors.append(f"{target.name}:{lineno}  {desc}")
            print(
                f"  [{i + 1}/{len(muts)}] L{lineno} {desc:26s} {status} ({time.time() - t0:.0f}s)"
            )

    print("\n===== SURVIVORS (coverage gaps) =====")
    for s in survivors:
        print(f"  {s}")
    killed = total - len(survivors)
    print(
        f"\nSCORE: killed {killed}/{total} ({100 * killed / max(total, 1):.0f}%), "
        f"survivors {len(survivors)}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=list(_DEFAULT_TARGETS))
    parser.add_argument("--tests", nargs="+", default=list(_DEFAULT_TESTS))
    args = parser.parse_args(argv)
    sys.exit(check(tuple(args.targets), tuple(args.tests)))


if __name__ == "__main__":
    main()
