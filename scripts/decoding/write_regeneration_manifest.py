#!/usr/bin/env python
"""Validate candidate covariate bundles and write a release manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from continuum_swr_replay.analysis.behavior.qvals_table import (
    QVALS_STANDARDIZED_COLS,
)
from continuum_swr_replay.analysis.events.analysed_time import SWR_SOURCE_ATTR
from continuum_swr_replay.analysis.events.swr_inventory import swr_source_name
from continuum_swr_replay.analysis.replay._tables_common import (
    EVENT_ARM_STANDARDIZED_COLS,
)
from continuum_swr_replay.analysis.replay.table_contracts import (
    validate_event_arm_table,
    validate_event_patch_table,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids
from continuum_swr_replay.paths import MODEL_DIRS, table_path
from continuum_swr_replay.runs.io import sha256_file
from continuum_swr_replay.runs.provenance import git, git_commit, git_dirty
from continuum_swr_replay.tables.io import SWR_SOURCE_TABLES, read_attrs, read_table
from continuum_swr_replay.tables.parity import diff_table
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
)
from continuum_swr_replay.tables.session_keys import (
    n_sessions as _sessions,
)
from continuum_swr_replay.tables.session_keys import (
    session_key_set as _session_keys,
)
from continuum_swr_replay.tables.standardization import (
    assert_standardized_matches_reference,
    load_reference,
)
from continuum_swr_replay.tables.versions import MANIFEST_SCHEMA

MODEL_VALIDATION_BUNDLES = {
    "replay": ("base_arm", "base_patch", "belief_arm", "belief_patch"),
    "contfrag": ("mua_arm", "mua_patch", "swr_arm", "swr_patch"),
}


def _inventory_file(path: Path, root: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".parquet":
        metadata = pq.ParquetFile(path).metadata
        schema = pq.read_schema(path)
        record.update(
            {
                "rows": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
                "columns": schema.names,
                "arrow_schema": str(schema),
            }
        )
    return record


def _json_session_keys(keys: set[tuple[object, ...]]) -> list[dict[str, object]]:
    return [
        dict(zip(SESSION_COLUMNS, key, strict=True))
        for key in sorted(keys, key=lambda key: tuple(map(str, key)))
    ]


def _require_one_swr_source(roots: Mapping[str, Path]) -> str:
    """Every SWR-bearing candidate table carries the configured SWR source's stamp.

    Every parquet table under `roots` whose covariate-schema table is one of
    ``tables.io.SWR_SOURCE_TABLES`` must be stamped ``swr_source_name()``:
    a release mixing SWR sources, or built from an earlier one, is refused.

    Returns
    -------
    str
        The source the release records.

    Raises
    ------
    ValueError
        If a table is unstamped or carries another source's stamp.
    """
    expected = swr_source_name()
    wrong = {
        str(path): attrs.get(SWR_SOURCE_ATTR)
        for root in roots.values()
        for path in sorted(root.rglob("*.parquet"))
        if (attrs := read_attrs(path)).get("covariate_schema_table")
        in SWR_SOURCE_TABLES
        and attrs.get(SWR_SOURCE_ATTR) != expected
    }
    if wrong:
        raise ValueError(
            f"{len(wrong)} candidate table(s) were not built from the configured "
            f"SWR source {expected!r}: {dict(list(wrong.items())[:10])}"
        )
    return expected


def _validate_model_bundle(
    directory: Path,
    *,
    expected_session_keys: set[tuple[object, ...]],
) -> dict[str, object]:
    """Validate one atomically published, common-cohort model ladder."""
    required_paths = {
        "all_models": directory / "all_models.pkl",
        "cohort_audit": directory / "all_models_cohort_audit.csv",
        "comparisons": directory / "model_comparison.pkl",
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"{directory} is missing model outputs: {missing}")

    results = pd.read_pickle(required_paths["all_models"])
    required_columns = {
        *SESSION_COLUMNS,
        "model",
        "n_events",
        "aic",
        "bic",
        "llf",
        "pseudo_r2",
    }
    missing_columns = sorted(required_columns.difference(results.columns))
    if missing_columns:
        raise ValueError(f"{directory} is missing model fields: {missing_columns}")
    if results.empty:
        raise ValueError(f"{directory} contains no model results")
    key_columns = [*SESSION_COLUMNS, "model"]
    if results.duplicated(key_columns).any():
        raise ValueError(f"{directory} has duplicate session/model rows")

    models = sorted(results["model"].astype(str).unique())
    model_files = sorted(
        path.stem
        for path in directory.glob("*.pkl")
        if path.name not in {"all_models.pkl", "model_comparison.pkl"}
    )
    if model_files != models:
        raise ValueError(
            f"{directory} model files do not match all_models: "
            f"files={model_files}, rows={models}"
        )

    session_sets: dict[str, set[tuple[object, ...]]] = {}
    for model in models:
        subset = results.loc[results["model"].astype(str).eq(model)]
        session_sets[model] = _session_keys(subset)
        saved = pd.read_pickle(directory / f"{model}.pkl")
        if _session_keys(saved) != session_sets[model] or len(saved) != len(subset):
            raise ValueError(f"{directory}/{model}.pkl disagrees with all_models.pkl")
    reference_sessions = session_sets[models[0]]
    if any(keys != reference_sessions for keys in session_sets.values()):
        raise ValueError(f"{directory} models do not share an exact session block")
    unexpected = reference_sessions.difference(expected_session_keys)
    if unexpected:
        raise ValueError(f"{directory} contains unexpected sessions: {unexpected}")

    n_events = pd.to_numeric(results["n_events"], errors="coerce")
    if (
        not np.isfinite(n_events).all()
        or (n_events <= 0).any()
        or not np.equal(n_events, np.floor(n_events)).all()
    ):
        raise ValueError(f"{directory} has invalid n_events")
    if (
        results.assign(_n_events=n_events)
        .groupby(list(SESSION_COLUMNS), observed=True)["_n_events"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(f"{directory} models do not share exact event counts")
    for column in ("aic", "bic", "llf", "pseudo_r2"):
        values = pd.to_numeric(results[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{directory} has non-finite {column}")
    if "converged" not in results:
        raise ValueError(
            f"{directory} model results lack a 'converged' column; a dropped or "
            "renamed column must not silently bypass the convergence gate"
        )
    failed = results["converged"].notna() & ~results["converged"].eq(True)  # noqa: E712
    if failed.any():
        raise ValueError(f"{directory} contains non-converged published fits")

    audit = pd.read_csv(required_paths["cohort_audit"])
    if set(audit["model"].astype(str)) != set(models):
        raise ValueError(f"{directory} cohort audit model set is incomplete")
    expected_n_sessions = len(reference_sessions)
    if not audit["n_sessions_modeled"].eq(expected_n_sessions).all():
        raise ValueError(f"{directory} cohort audit has wrong session counts")
    expected_events = {
        model: int(
            results.loc[results["model"].astype(str).eq(model), "n_events"].sum()
        )
        for model in models
    }
    observed_events = audit.set_index(audit["model"].astype(str))["n_modeled"]
    if any(int(observed_events[model]) != expected_events[model] for model in models):
        raise ValueError(f"{directory} cohort audit has wrong modeled-event counts")

    comparisons = pd.read_pickle(required_paths["comparisons"])
    if not comparisons.empty:
        if not set(comparisons["parent"]).union(comparisons["child"]).issubset(models):
            raise ValueError(f"{directory} comparisons reference absent models")
        if not comparisons["n_sessions"].eq(expected_n_sessions).all():
            raise ValueError(f"{directory} comparisons use a different session block")

    excluded = expected_session_keys.difference(reference_sessions)
    return {
        "models": models,
        "sessions_expected": len(expected_session_keys),
        "sessions_modeled": expected_n_sessions,
        "sessions_excluded": _json_session_keys(excluded),
        "events_modeled_per_model": expected_events,
        "comparison_edges": len(comparisons),
        "all_fit_statistics_finite": True,
        "all_reported_convergence_flags_pass": True,
    }


def _validate_model_bundles(
    replay_root: Path,
    contfrag_root: Path,
    *,
    replay_sessions: set[tuple[object, ...]],
    contfrag_sessions: dict[str, set[tuple[object, ...]]],
) -> dict[str, object]:
    audits: dict[str, object] = {}
    for label, root in (("replay", replay_root), ("contfrag", contfrag_root)):
        validation_root = root / "models_validation"
        expected_names = MODEL_VALIDATION_BUNDLES[label]
        observed_names = (
            sorted(path.name for path in validation_root.iterdir() if path.is_dir())
            if validation_root.is_dir()
            else []
        )
        if observed_names != sorted(expected_names):
            raise ValueError(
                f"{validation_root} has incomplete validation bundles: "
                f"expected={sorted(expected_names)}, observed={observed_names}"
            )
        for name in expected_names:
            if label == "replay":
                expected_keys = replay_sessions
            else:
                expected_keys = contfrag_sessions[name.split("_", maxsplit=1)[0]]
            audits[f"{label}/{name}"] = _validate_model_bundle(
                validation_root / name,
                expected_session_keys=expected_keys,
            )
    return audits


def _validate_fraction_sum(
    frame: pd.DataFrame, column: str, *, tolerance: float = 1e-12
) -> None:
    sums = frame.groupby("event_id", observed=True)[column].sum(min_count=1)
    if sums.isna().any() or not np.allclose(sums, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(f"{column} does not sum to one within every event")


def _validate_bounds(
    frame: pd.DataFrame, column: str, lower: float, upper: float
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite) or (finite < lower).any() or (finite > upper).any():
        raise ValueError(
            f"{column} violates [{lower}, {upper}] or has no finite support"
        )


def _validate_replay(root: Path, expected_sessions: int) -> dict[str, object]:
    event = read_table(table_path("event_table", root, fmt=""))
    arm = read_table(table_path("event_arm_table", root, fmt=""))
    patch = read_table(table_path("event_patch_table", root, fmt=""))
    stop = read_table(table_path("stop_table", root, fmt=""))
    sequence = read_table(table_path("event_sequence_table", root, fmt=""))
    qvals = read_table(table_path("qvals_table", root, fmt=""))
    validate_event_arm_table(arm, event_table=event)
    validate_event_patch_table(patch, event_table=event)
    # stop_id anchors every stop-level join; a null or a cross-session collision
    # silently mis-joins. (validate_event_arm_table covers event_id.)
    if stop["stop_id"].isna().any() or stop["stop_id"].duplicated().any():
        raise ValueError("Replay stop_id is missing or duplicated")
    # event_sequence_table is otherwise unread by the gate: every sequence row
    # must resolve to an assembled event (the silent event_id mis-join
    # CLAUDE.md flags), with no fully duplicated row.
    sequence_orphans = set(sequence["event_id"].unique()) - set(event["event_id"])
    if sequence_orphans:
        raise ValueError(
            f"event_sequence_table references {len(sequence_orphans)} event_id(s) "
            "absent from event_table"
        )
    if sequence.duplicated().any():
        raise ValueError("event_sequence_table contains fully duplicated rows")
    if _sessions(event) != expected_sessions or _sessions(stop) != expected_sessions:
        raise ValueError("Replay event/stop tables do not have the expected sessions")
    if len(arm) != 9 * len(event) or len(patch) != 3 * len(event):
        raise ValueError("Replay candidate multiplicity failed")
    if int(stop["n_selected_events"].sum()) != len(event):
        raise ValueError("Replay stop/event count conservation failed")
    _validate_fraction_sum(arm, "arm_fraction")
    _validate_fraction_sum(arm, "arm_fraction_posterior")
    for column, lower, upper in (
        ("reward_fraction_candidate_arm_last10_visits", 0.0, 1.0),
        ("task_reward_probability_candidate_arm", 0.0, 1.0),
        ("reward_belief_mean_candidate_arm_preoutcome", 0.0, 1.0),
        ("reward_belief_variance_candidate_arm_preoutcome", 0.0, 0.25),
        ("distance_to_candidate_arm_cm", 0.0, np.inf),
    ):
        _validate_bounds(arm, column, lower, upper)
    assert_standardized_matches_reference(
        arm,
        EVENT_ARM_STANDARDIZED_COLS,
        load_reference(root / "standardization_reference.json"),
        support_label="leaf_arms",
    )
    assert_standardized_matches_reference(
        qvals,
        QVALS_STANDARDIZED_COLS,
        load_reference(
            root
            / MODEL_DIRS["per_bin_rate"].name
            / "qvals_standardization_reference.json"
        ),
        support_label="all_rows",
    )
    if qvals["stop_id"].isna().any() or qvals["stop_id"].duplicated().any():
        raise ValueError("qvals stop_id is missing or duplicated")
    merged = stop.loc[:, [*SESSION_COLUMNS, "stop_id", "stop_start"]].merge(
        qvals.loc[:, ["stop_id"]],
        on="stop_id",
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    internal_gaps = 0
    for _, group in merged.sort_values("stop_start").groupby(
        list(SESSION_COLUMNS), observed=True
    ):
        covered = group["_merge"].eq("both").to_numpy()
        if not covered.any():
            internal_gaps += len(covered)
            continue
        internal_gaps += int((~covered[: np.flatnonzero(covered).max() + 1]).sum())
    if internal_gaps:
        raise ValueError(f"qvals contains {internal_gaps} internal stop gaps")
    mapped_stop = stop.loc[stop["stop_id"].isin(qvals["stop_id"])]
    return {
        "events": len(event),
        "event_arm_rows": len(arm),
        "event_patch_rows": len(patch),
        "event_sequence_rows": len(sequence),
        "stops": len(stop),
        "sessions": _sessions(event),
        "qvals_rows": len(qvals),
        "qvals_sessions": _sessions(mapped_stop),
        "qvals_trailing_unmapped_stops": len(stop) - len(qvals),
        "qvals_internal_gaps": internal_gaps,
    }


def _validate_contfrag(
    root: Path, source: str, expected_sessions: int
) -> dict[str, object]:
    event = read_table(root / f"event_table_{source}_HPC")
    arm = read_table(root / f"event_arm_table_{source}_HPC")
    patch = read_table(root / f"event_patch_table_{source}_HPC")
    stop = read_table(root / f"stop_table_{source}_HPC")
    validate_event_arm_table(arm, event_table=event)
    validate_event_patch_table(patch, event_table=event)
    if _sessions(event) != expected_sessions or _sessions(stop) != expected_sessions:
        raise ValueError(f"ContFrag {source} is missing sessions")
    if len(arm) != 9 * len(event) or len(patch) != 3 * len(event):
        raise ValueError(f"ContFrag {source} candidate multiplicity failed")
    if int(stop["n_selected_events"].sum()) != len(event):
        raise ValueError(f"ContFrag {source} stop/event conservation failed")
    _validate_fraction_sum(arm, "arm_fraction")
    _validate_fraction_sum(arm, "arm_fraction_posterior")
    return {
        "events": len(event),
        "event_arm_rows": len(arm),
        "event_patch_rows": len(patch),
        "stops": len(stop),
        "sessions": _sessions(event),
    }


# Candidate table -> key columns for the baseline parity diff.
_BASELINE_PARITY_TABLES = {
    "event_table": ("event_id",),
    "event_arm_table": ("event_id", "candidate_arm"),
    "stop_table": ("stop_id",),
}


def _verify_baseline(
    candidate_root: Path,
    baseline_root: Path,
    *,
    expected_drift_columns: set[str],
    atol: float = 1e-9,
) -> dict[str, object]:
    """Diff each candidate table against the accepted baseline on its key.

    A regeneration is expected to change a *declared* set of columns and
    nothing else. This raises on structural drift (differing key set, row
    count, or schema) and on numeric/categorical drift in any column not listed
    in ``expected_drift_columns``, so a candidate that is internally
    self-consistent but silently regressed from the accepted baseline cannot
    pass. Returns the full per-table, per-column drift report for the manifest.
    """
    report: dict[str, object] = {}
    for name, key in _BASELINE_PARITY_TABLES.items():
        report[name] = diff_table(
            read_table(candidate_root / name),
            read_table(baseline_root / name),
            keys=list(key),
            label=f"baseline parity: {name}",
            expected_drift_columns=frozenset(expected_drift_columns),
            atol=atol,
        )
    return report


def write_manifest(
    *,
    replay_root: Path,
    contfrag_root: Path,
    output: Path,
    baseline_replay: Path | None,
    baseline_contfrag: Path | None,
    expected_sessions: int,
    expected_drift_columns: set[str] | None = None,
) -> dict[str, object]:
    """Run promotion-blocking checks and persist one content-addressed manifest."""
    replay_event = read_table(table_path("event_table", replay_root, fmt=""))
    contfrag_events = {
        source: read_table(contfrag_root / f"event_table_{source}_HPC")
        for source in ("swr", "mua")
    }
    replay_audit = _validate_replay(replay_root, expected_sessions)
    contfrag_audit = {
        source: _validate_contfrag(contfrag_root, source, expected_sessions)
        for source in ("swr", "mua")
    }
    model_audit = _validate_model_bundles(
        replay_root,
        contfrag_root,
        replay_sessions=_session_keys(replay_event),
        contfrag_sessions={
            source: _session_keys(frame) for source, frame in contfrag_events.items()
        },
    )
    # Promotion-blocking regression gate: the candidate must match the accepted
    # baseline everywhere except an explicitly declared set of changed columns.
    # Without this, a self-consistent but silently regressed candidate would
    # validate. Runs before any manifest is written, so a drift raises here.
    baseline_parity: dict[str, object] | None = None
    if baseline_replay is not None:
        baseline_parity = _verify_baseline(
            replay_root,
            baseline_replay,
            expected_drift_columns=expected_drift_columns or set(),
        )
    roots = {"replay": replay_root, "contfrag": contfrag_root}
    swr_source = _require_one_swr_source(roots)
    files = {
        label: [
            _inventory_file(path, root)
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.resolve() != output.resolve()
            and (
                path.suffix in {".parquet", ".json"}
                or (
                    "models_validation" in path.parts
                    and path.suffix in {".pkl", ".csv"}
                )
            )
        ]
        for label, root in roots.items()
    }
    # the work tree the manifest describes is the one it runs in, as uv.lock below
    repo = Path.cwd()
    tracked_dirty = git_dirty(repo)
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    manifest = {
        "manifest_schema": MANIFEST_SCHEMA,
        "status": "validated_candidate_not_promoted",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source": {
            "git_commit": git_commit(repo),
            "tracked_dirty": tracked_dirty,
            "untracked_paths_present": sorted(untracked),
            "uv_lock_sha256": sha256_file(Path("uv.lock")),
        },
        "candidate_roots": {label: str(path) for label, path in roots.items()},
        # Every SWR-bearing table was checked to carry this stamp above.
        "swr_source": swr_source,
        "preserved_baselines": {
            "replay": str(baseline_replay) if baseline_replay else None,
            "contfrag": str(baseline_contfrag) if baseline_contfrag else None,
        },
        "baseline_parity": {
            "verified": baseline_parity is not None,
            "expected_drift_columns": sorted(expected_drift_columns or set()),
            # Per-table drift vs baseline on the primary key. Empty dicts mean a
            # column-for-column match; listed columns were declared-expected to
            # change (undeclared drift would have raised above).
            "drift": baseline_parity,
        },
        "methods": {
            "replay": "full 218-session v4-cache rebuild; five disjoint animal shards; pooled standardization",
            "qvals": "live 218-session model/timing rebuild against candidate stop table",
            "contfrag": "all non-posterior quantities rebuilt; immutable posterior aggregates transferred by exact source-event/candidate keys",
            "carried_forward": "event_onset_table and stop_timebin_table after exact stop/session key checks",
            "baseline_parity": (
                "event/event_arm/stop diffed against the accepted baseline on "
                "their primary key; structural drift or undeclared column drift "
                "raises"
                if baseline_parity is not None
                else "not run (no --baseline-replay provided)"
            ),
        },
        "audits": {
            "replay": replay_audit,
            "contfrag": contfrag_audit,
            "model_validation": model_audit,
        },
        "files": files,
    }
    if tracked_dirty:
        raise ValueError("tracked working tree is dirty; refusing release manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--contfrag-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-replay", type=Path)
    parser.add_argument("--baseline-contfrag", type=Path)
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=len(analysed_epoch_ids()),
        help="Epochs (sessions) the tables must hold; default: the epoch register's "
        "analysed epochs (HPC cohort).",
    )
    parser.add_argument(
        "--expected-drift-columns",
        nargs="*",
        default=[],
        help="Columns the candidate is expected to change vs --baseline-replay; "
        "any other drift fails the parity gate.",
    )
    args = parser.parse_args()
    write_manifest(
        replay_root=args.replay_root.resolve(),
        contfrag_root=args.contfrag_root.resolve(),
        output=args.output.resolve(),
        baseline_replay=(
            args.baseline_replay.resolve() if args.baseline_replay else None
        ),
        baseline_contfrag=(
            args.baseline_contfrag.resolve() if args.baseline_contfrag else None
        ),
        expected_sessions=args.expected_sessions,
        expected_drift_columns=set(args.expected_drift_columns),
    )


if __name__ == "__main__":
    main()
