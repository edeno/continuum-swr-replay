#!/usr/bin/env python
"""Generate the symbol-anchored regression pipeline code map.

The map lists every module that carries data from the decoded session to a
fitted regression, grouped by pipeline stage, with the public functions and
classes each module defines. It is derived mechanically from the source with
``ast`` -- no line numbers, so it does not rot the moment a function moves.

Usage:
    uv run python scripts/qa/write_pipeline_map.py            # write the map
    uv run python scripts/qa/write_pipeline_map.py --check    # fail on drift
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from itertools import takewhile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "docs" / "regression_pipeline_map.md"

# Directories walked, in the order their modules are discovered.
ROOTS: tuple[str, ...] = (
    "src/continuum_swr_replay/analysis/replay",
    "src/continuum_swr_replay/analysis/hippocampus",
    "src/continuum_swr_replay/analysis/behavior",
    "scripts/decoding",
    "scripts/content",
    "scripts/rate",
    "scripts/qa",
)

# Pipeline stages, in flow order. Each module is assigned to the first stage
# whose path patterns match it, so more specific patterns come first within a
# stage and earlier stages win ties. Patterns match the repo-relative POSIX
# path with ``Path.match``-style ``fnmatch`` semantics on the whole string.
STAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Tables",
        "Event, arm, patch, stop and time-bin tables built from the decoded "
        "session, plus the behavior-model table they join against.",
        (
            "*/analysis/replay/tables*.py",
            "*/analysis/replay/_tables_common.py",
            "*/analysis/replay/timebin_tables.py",
            "*/analysis/replay/table_contracts.py",
            "*/analysis/replay/cohort_selection.py",
            "*/analysis/replay/prepare_data.py",
            "*/analysis/hippocampus/*.py",
            "*/analysis/behavior/*.py",
        ),
    ),
    (
        "Covariates",
        "Predictor construction: the timing clocks, the history blocks, the "
        "candidate-arm block, standardization, and the name registry.",
        (
            "*/analysis/replay/covariates/*.py",
            "*/analysis/replay/covariates/history/*.py",
            "*/analysis/replay/covariate_registry.py",
            "*/analysis/replay/covariate_timing.py",
            "*/analysis/replay/standardization.py",
            "*/analysis/replay/spatial*.py",
            "*/analysis/replay/uncertainty*.py",
            "*/analysis/replay/content_response.py",
        ),
    ),
    (
        "Specs",
        "The single source of truth for every model ladder.",
        ("*/analysis/replay/model_specs.py",),
    ),
    (
        "Fitters",
        "Estimators and the per-session fit machinery each runner calls.",
        (
            "*/analysis/replay/fitting.py",
            "*/analysis/replay/_fit_diagnostics.py",
            "*/analysis/replay/_helpers.py",
            "*/analysis/replay/per_bin_rate_model.py",
            "*/analysis/replay/per_stop_rate_model.py",
            "*/analysis/replay/dirichlet_multinomial.py",
            "*/analysis/replay/choice_prediction.py",
            "*/analysis/replay/shuffles.py",
            "*/analysis/replay/simulation.py",
        ),
    ),
    (
        "Runners",
        "Command-line entry points that fit or plot a ladder.",
        (
            "scripts/decoding/*.py",
            "scripts/content/*.py",
            "scripts/rate/*.py",
        ),
    ),
    (
        "Aggregation",
        "Across-session and across-animal summaries, contrasts and model "
        "comparison over the per-session fits.",
        (
            "*/analysis/replay/contrast_aggregation.py",
            "*/analysis/replay/summaries.py",
            "*/analysis/replay/cohort_audit.py",
            "*/analysis/replay/model_comparison.py",
            "*/analysis/replay/model_checking.py",
            "*/analysis/replay/learning.py",
            "*/analysis/replay/retrospective.py",
            "*/analysis/replay/switch_analysis.py",
            "*/analysis/replay/trajectory.py",
        ),
    ),
    (
        "Readers and QA",
        "Artifact I/O, provenance gating, and the diagnostics that check the "
        "tables and the fits.",
        (
            "*/analysis/replay/posterior_io.py",
            "*/analysis/replay/fit_provenance.py",
            "*/analysis/replay/versions.py",
            "scripts/qa/*.py",
        ),
    ),
)

_UNASSIGNED = "Unassigned"
_UNASSIGNED_BLURB = (
    "Modules under a mapped root that no stage pattern claims. A module "
    "landing here means `write_pipeline_map.py` needs a stage rule, not that "
    "the module is unused."
)


def _iter_modules() -> list[Path]:
    """Every mapped source module, repo-relative, deduplicated and sorted."""
    seen: dict[str, Path] = {}
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            if "__pycache__" in path.parts or path.name == "__init__.py":
                continue
            rel = path.relative_to(REPO)
            seen.setdefault(rel.as_posix(), rel)
    return [seen[key] for key in sorted(seen)]


def _stage_of(rel: Path) -> str:
    text = rel.as_posix()
    for name, _blurb, patterns in STAGES:
        for pattern in patterns:
            if Path(text).match(pattern):
                return name
    return _UNASSIGNED


def _public_symbols(path: Path) -> list[str]:
    """Module-level public function and class names, in definition order."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    ]


def _summary(path: Path) -> str:
    """Opening sentence of the module docstring, unwrapped, or an empty string.

    Docstring summaries are hard-wrapped, so the first *physical* line is often
    half a sentence. Join the lines of the opening paragraph (up to the first
    blank line) and then cut at the first sentence end, so the map reads as
    prose rather than as truncated fragments.
    """
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    if not doc:
        return ""
    paragraph = " ".join(
        line.strip()
        for line in takewhile(lambda line: line.strip(), doc.strip().splitlines())
    )
    head, sentence_end, _ = paragraph.partition(". ")
    return f"{head}." if sentence_end else paragraph


def render() -> str:
    lines = [
        "# Regression pipeline code map",
        "",
        "Every module that carries data from a decoded session to a fitted",
        "regression, grouped by pipeline stage, with the public functions and",
        "classes it defines (in definition order).",
        "",
        "**Generated file — do not edit by hand.** Regenerate with",
        "`uv run python scripts/qa/write_pipeline_map.py`;",
        "`--check` fails when the tree and this file disagree.",
        "",
        "Names are the anchor, not line numbers: grep the symbol. Private",
        "symbols (`_name`) are omitted, but private *modules* are listed;",
        "`__init__.py` re-export surfaces are omitted.",
        "",
    ]

    by_stage: dict[str, list[Path]] = {}
    for rel in _iter_modules():
        by_stage.setdefault(_stage_of(rel), []).append(rel)

    stage_order = [(name, blurb) for name, blurb, _ in STAGES]
    if _UNASSIGNED in by_stage:
        stage_order.append((_UNASSIGNED, _UNASSIGNED_BLURB))

    for name, blurb in stage_order:
        modules = by_stage.get(name, [])
        if not modules:
            continue
        lines += [f"## {name}", "", blurb, ""]
        for rel in modules:
            symbols = _public_symbols(REPO / rel)
            summary = _summary(REPO / rel)
            lines.append(f"### `{rel.as_posix()}`")
            lines.append("")
            if summary:
                lines += [summary, ""]
            lines.append(
                ", ".join(f"`{s}`" for s in symbols)
                if symbols
                else "_No module-level public symbols._"
            )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if the committed map differs from the tree.",
    )
    args = parser.parse_args()

    generated = render()
    if not args.check:
        OUTPUT.write_text(generated, encoding="utf-8")
        print(f"wrote {OUTPUT.relative_to(REPO)}")
        return 0

    committed = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
    if committed == generated:
        print(f"{OUTPUT.relative_to(REPO)} is current")
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile=f"{OUTPUT.relative_to(REPO)} (committed)",
            tofile=f"{OUTPUT.relative_to(REPO)} (regenerated)",
        )
    )
    print(
        "\nregression pipeline map is stale -- rerun "
        "`uv run python scripts/qa/write_pipeline_map.py`",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
