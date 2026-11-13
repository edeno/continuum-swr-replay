#!/usr/bin/env python
"""Generate the symbol-anchored regression pipeline code map.

The map lists every module that carries data from the decoded session to a
fitted regression, grouped by pipeline stage, with the public functions and
classes each module defines. It is derived mechanically from the source with
``ast`` -- no line numbers, so it does not rot the moment a function moves.

Usage:
    uv run python scripts/checks/write_pipeline_map.py            # write the map
    uv run python scripts/checks/write_pipeline_map.py --check    # fail on drift
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from collections.abc import Sequence
from itertools import takewhile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "docs" / "regression_pipeline_map.md"

# Directories walked, in the order their modules are discovered.
ROOTS: tuple[str, ...] = (
    "src/continuum_swr_replay/analysis/behavior",
    "src/continuum_swr_replay/analysis/event_tables",
    "src/continuum_swr_replay/analysis/replay_models",
    "src/continuum_swr_replay/analysis/hippocampus",
    "src/continuum_swr_replay/analysis/events",
    "src/continuum_swr_replay/tables",
    "src/continuum_swr_replay/decoding",
    "scripts/behavior",
    "scripts/decoding",
    "scripts/tables",
    "scripts/detection",
    "scripts/replay_models",
    "scripts/workflows",
    "scripts/checks",
    "scripts/pfc",
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
            "*/analysis/event_tables/event_table.py",
            "*/analysis/event_tables/stop_table.py",
            "*/analysis/event_tables/arm_patch_tables.py",
            "*/analysis/event_tables/retrospective_metrics.py",
            "*/analysis/event_tables/_common.py",
            "*/analysis/event_tables/bouts.py",
            "*/tables/event_ids.py",
            "*/tables/session_keys.py",
            "*/analysis/event_tables/timebin_tables.py",
            "*/analysis/event_tables/table_contracts.py",
            "*/analysis/events/cohort_selection.py",
            "*/analysis/event_tables/prepare_data.py",
            "*/analysis/hippocampus/*.py",
            "*/analysis/detection/artifacts/*.py",
            "*/analysis/events/*.py",
            "*/analysis/behavior/covariates/qvals_table.py",
            "*/analysis/behavior/covariates/_spatial_bandit_rl.py",
        ),
    ),
    (
        "Covariates",
        "Predictor construction: the timing clocks, the history blocks, the "
        "candidate-arm block, standardization, and the name registry.",
        (
            "*/analysis/behavior/covariates/*.py",
            "*/analysis/behavior/covariates/history/*.py",
            "*/tables/covariate_registry.py",
            "*/analysis/replay_models/covariate_timing.py",
            "*/tables/standardization.py",
            "*/analysis/behavior/spatial*.py",
            "*/analysis/event_tables/uncertainty.py",
            "*/analysis/replay_models/content/uncertainty_content.py",
            "*/analysis/event_tables/content_response.py",
        ),
    ),
    (
        "Specs",
        "The single source of truth for every model ladder, and the design "
        "constants the ladders share with the covariate builders.",
        (
            "*/analysis/replay_models/model_specs.py",
            "*/analysis/behavior/design_constants.py",
        ),
    ),
    (
        "Fitters",
        "Estimators and the per-session fit machinery each runner calls.",
        (
            "*/analysis/replay_models/fitting.py",
            "*/analysis/replay_models/_fit_diagnostics.py",
            "*/analysis/behavior/history_basis.py",
            "*/analysis/replay_models/per_bin_design.py",
            "*/analysis/replay_models/rate/per_bin_rate_model.py",
            "*/analysis/replay_models/rate/per_stop_rate_model.py",
            "*/analysis/replay_models/content/dirichlet_multinomial.py",
            "*/analysis/replay_models/choice/choice_prediction.py",
            "*/analysis/replay_models/choice/shuffles.py",
            "*/analysis/replay_models/content/simulation.py",
            "*/analysis/replay_models/rate/duration_history_simulation.py",
            "*/analysis/replay_models/rate/duration_aware_history.py",
            "*/analysis/replay_models/rate/ridge_history_cv.py",
        ),
    ),
    (
        "Runners",
        "Command-line entry points that fit or plot a ladder.",
        (
            "scripts/behavior/*.py",
            "scripts/decoding/*.py",
            "scripts/tables/*.py",
            "scripts/detection/run_artifact_detection.py",
            "scripts/detection/run_carey_detection.py",
            "scripts/replay_models/*.py",
        ),
    ),
    (
        "Aggregation",
        "Across-session and across-animal summaries, contrasts and model "
        "comparison over the per-session fits.",
        (
            "*/analysis/replay_models/contrast_aggregation.py",
            "*/analysis/event_tables/summaries.py",
            "*/analysis/replay_models/cohort_audit.py",
            "*/analysis/replay_models/content/model_comparison.py",
            "*/analysis/replay_models/content/model_checking.py",
            "*/analysis/replay_models/choice/stay_leave.py",
            "*/analysis/replay_models/choice/retrospective.py",
            "*/analysis/replay_models/choice/switch_analysis.py",
            "*/analysis/event_tables/trajectory.py",
        ),
    ),
    (
        "Readers and QA",
        "Artifact I/O, provenance gating, and the diagnostics that check the "
        "tables and the fits.",
        (
            "*/decoding/results.py",
            "*/decoding/files.py",
            "*/decoding/identifiers.py",
            "*/runs/fit_provenance.py",
            "*/analysis/replay_models/content/fit_outputs.py",
            "*/tables/versions.py",
            "*/tables/io.py",
            "*/tables/parity.py",
            "*/analysis/event_evidence/state_evidence/*.py",
            "scripts/detection/*.py",
            "scripts/workflows/*.py",
            "scripts/checks/*.py",
            "scripts/pfc/*.py",
        ),
    ),
)

_UNASSIGNED = "Unassigned"
_UNASSIGNED_BLURB = (
    "Modules under a mapped root that no stage pattern claims. A module "
    "landing here means `write_pipeline_map.py` needs a stage rule, not that "
    "the module is unused."
)

# Packages under a mapped root that carry no data from a decoded session to a
# fitted regression: the replay-detection comparison lives beside the
# pipeline, not on it; the decode's own settings act before a session is
# decoded. ``scripts/pfc`` is mapped for its audit scripts (which lived in
# ``scripts/qa`` until the 2026-10 move); its batch runner was never mapped.
EXCLUDED_PREFIXES: tuple[str, ...] = (
    "src/continuum_swr_replay/decoding/config.py",
    "src/continuum_swr_replay/decoding/transitions.py",
    "src/continuum_swr_replay/analysis/detection/memoryless_sequence/",
    "src/continuum_swr_replay/analysis/detection/swr_detectors.py",
    "src/continuum_swr_replay/analysis/inventory_comparison/",
    "scripts/detection/run_memoryless_detection.py",
    "scripts/detection/run_swr_detection.py",
    "scripts/detection/sweep_memoryless_detection.py",
    "scripts/pfc/run_pfc_analysis_bandit.py",
)


def _iter_modules() -> list[Path]:
    """Every mapped source module, repo-relative, deduplicated and sorted."""
    seen: dict[str, Path] = {}
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            if "__pycache__" in path.parts or path.name == "__init__.py":
                continue
            rel = path.relative_to(REPO)
            if rel.as_posix().startswith(EXCLUDED_PREFIXES):
                continue
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
        "`uv run python scripts/checks/write_pipeline_map.py`;",
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if the committed map differs from the tree.",
    )
    args = parser.parse_args(argv)

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
        "`uv run python scripts/checks/write_pipeline_map.py`",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
