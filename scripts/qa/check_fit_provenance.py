#!/usr/bin/env python
"""Report whether every fitted artifact is still current for the tables it read.

Each fit runner writes a ``<artifact>.provenance.json`` sidecar recording the
SHA-256 of every table it read. Readers gate on that sidecar, but only for the
artifact they are about to open, and only when someone runs them. This script
sweeps the whole ``models*/`` tree at once so "which fits are stale?" has an
answer that does not depend on running a plotter and waiting for it to refuse.

The inputs checked are the ones each sidecar itself records -- the content
ladder records one table, the SWR-contrast runner four -- so a runner that
gains or loses an input needs no change here.

Usage
-----
    uv run python scripts/qa/check_fit_provenance.py
    uv run python scripts/qa/check_fit_provenance.py --json
    uv run python scripts/qa/check_fit_provenance.py path/to/artifact.pkl

Exit code is 1 when any artifact is stale or has no sidecar, 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from continuum_swr_replay.analysis.replay.fit_provenance import (
    check_fit_provenance,
    fit_provenance_path,
)
from continuum_swr_replay.paths import (
    CANONICAL_TABLE_BASENAMES,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.tables.io import parquet_path

_SIDECAR_SUFFIX = ".provenance.json"


def artifact_for_sidecar(sidecar: Path) -> Path:
    """Return the artifact a sidecar describes."""
    return sidecar.with_name(sidecar.name[: -len(_SIDECAR_SUFFIX)] + ".pkl")


def find_sidecars(root: Path) -> list[Path]:
    """Every fit-provenance sidecar under ``root``'s model directories.

    Backup and archive directories are skipped: they hold deliberately frozen
    copies whose inputs have moved on by design, so reporting them as stale
    would bury the artifacts that are actually read.
    """
    skipped = ("_backup", "_archive", "_pre_")
    return sorted(
        path
        for path in root.glob(f"models*/**/*{_SIDECAR_SUFFIX}")
        if not any(part.startswith(skipped) for part in path.relative_to(root).parts)
    )


def resolve_input(name: str, sidecar: Path, root: Path) -> Path | None:
    """Locate the table a sidecar recorded under ``name``.

    Sidecars record input names, not paths. A canonical table
    (:data:`~continuum_swr_replay.paths.CANONICAL_TABLE_BASENAMES`) is looked up
    at the ``replay_content`` root only, so a copy or a symlink at an old place
    beside the sidecar cannot stand in for it. Any other input is an
    aggregate-only run's fit (a pickle) or a non-canonical table (parquet, such
    as a ContFrag table), looked up beside the sidecar, then at the root, the
    parquet first in each. Returns the first candidate that exists.
    """
    if name in CANONICAL_TABLE_BASENAMES:
        candidates: tuple[Path, ...] = (table_path(name, root),)
    else:
        candidates = tuple(
            folder / f"{name}{suffix}"
            for folder in (sidecar.parent, root)
            for suffix in (".parquet", ".pkl")
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def check_artifact(artifact: Path, sidecar: Path, root: Path) -> dict[str, object]:
    """Classify one artifact against the inputs its sidecar recorded."""
    result: dict[str, object] = {
        "artifact": str(artifact),
        "status": "fresh",
        "stale_inputs": [],
        "inherited_inputs": [],
        "unresolved_inputs": [],
        "inputs": {},
    }
    if not sidecar.exists():
        result["status"] = "missing_sidecar"
        return result
    # A sidecar outliving its artifact certifies nothing: the fit it describes
    # is gone, so anything that reads it reads a file that is not there.
    if not artifact.exists() and not parquet_path(artifact).exists():
        result["status"] = "missing_artifact"
        return result
    recorded = json.loads(sidecar.read_text()).get("inputs", {})
    inputs: dict[str, Path] = {}
    for name in sorted(recorded):
        path = resolve_input(name, sidecar, root)
        if path is None:
            result["unresolved_inputs"].append(name)  # type: ignore[union-attr]
        else:
            inputs[name] = path
    result["inputs"] = {name: str(path) for name, path in inputs.items()}
    if result["unresolved_inputs"]:
        result["status"] = "missing_input"
        return result
    stale = check_fit_provenance(artifact, inputs)
    if stale:
        result["status"] = "stale"
        result["stale_inputs"] = stale
    return result


def propagate_stale(results: list[dict[str, object]]) -> None:
    """Mark an artifact stale when one of its inputs is another stale artifact.

    An aggregate-only run reads a previous fit rather than a table, and its own
    hash check passes as long as that fit's bytes have not moved. Certifying it
    fresh while the fit it summarizes is stale would report the freshest thing
    about the staleest number, so staleness is carried forward until nothing
    changes.
    """
    status = {str(r["artifact"]): str(r["status"]) for r in results}
    changed = True
    while changed:
        changed = False
        for result in results:
            if result["status"] != "fresh":
                continue
            upstream = sorted(
                name
                for name, path in result["inputs"].items()  # type: ignore[union-attr]
                if status.get(path, "fresh") != "fresh"
            )
            if upstream:
                result["status"] = "stale"
                result["inherited_inputs"] = upstream
                status[str(result["artifact"])] = "stale"
                changed = True


def format_line(result: dict[str, object], root: Path) -> str:
    """One human-readable line per artifact."""
    artifact = Path(str(result["artifact"]))
    try:
        label = str(artifact.relative_to(root))
    except ValueError:
        label = str(artifact)
    status = str(result["status"])
    if status == "fresh":
        return f"fresh          {label}"
    if status == "missing_sidecar":
        return f"no sidecar     {label}"
    if status == "missing_artifact":
        return f"no artifact    {label}"
    if status == "missing_input":
        names = ", ".join(result["unresolved_inputs"])  # type: ignore[arg-type]
        return f"missing input  {label}  ({names})"
    # "changed" is this artifact's own input bytes; "inherited" is an upstream
    # fit that went stale, which this artifact's own hashes cannot see.
    if result["stale_inputs"]:
        names = ", ".join(result["stale_inputs"])  # type: ignore[arg-type]
        return f"STALE          {label}  (changed: {names})"
    names = ", ".join(result["inherited_inputs"])  # type: ignore[arg-type]
    return f"STALE          {label}  (inherited: {names})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="*",
        type=Path,
        help="Artifacts to check; default is every sidecar under --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=replay_content_dir(),
        help="Directory holding the models*/ trees.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the results as JSON instead."
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if args.artifacts:
        pairs = [(path, fit_provenance_path(path)) for path in args.artifacts]
    else:
        pairs = [(artifact_for_sidecar(s), s) for s in find_sidecars(root)]

    results = [check_artifact(a, s, root) for a, s in pairs]
    propagate_stale(results)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            print(format_line(result, root))
        counts: dict[str, int] = {}
        for result in results:
            status = str(result["status"])
            counts[status] = counts.get(status, 0) + 1
        summary = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        print(f"\n{len(results)} artifact(s): {summary or 'none'}")
    return int(any(result["status"] != "fresh" for result in results))


if __name__ == "__main__":
    sys.exit(main())
