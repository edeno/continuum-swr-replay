# Shared Contracts

[← back to PLAN.md](PLAN.md)

Cross-phase contracts with one canonical home. Phases link here by anchor.

- [session_keys module](#session_keys) — introduced phase 2, referenced by phases 3–5.
- [publish_canonical](#publish_canonical) — introduced phase 3, referenced by phases 4–5.

---

## session_keys

New module `src/continuum_swr_replay/session_keys.py` (package-root level, so both `analysis/**` and `scripts/**` import it without reaching across subpackages). It is the single home for the natural session identity.

```python
"""The natural key for one recording session: (animal, session_date, epoch).

This tuple identifies a session everywhere in the pipeline -- it is the join
key for every table and the group key for every per-session fit. Defining it
once here, backed by the covariate registry's row-key authority, keeps a rename
(as `day` -> `session_date` once was) a one-line change instead of a hunt
through twenty files.
"""
from __future__ import annotations

import pandas as pd
from continuum_swr_replay.analysis.replay.covariate_registry import row_keys

# The canonical session columns, in join order. Sourced from the registry so
# there is exactly one authority: the first three segments of any table's row key.
SESSION_COLUMNS: tuple[str, ...] = row_keys(table="event")[:3]  # ("animal", "session_date", "epoch")

# The hippocampus event table names its date column `day`, not `session_date`.
# That column name is load-bearing for that grain; this variant serves it
# without forcing a rename. See analysis/hippocampus/event_table.py.
SESSION_COLUMNS_DAY: tuple[str, ...] = ("animal", "day", "epoch")


def session_key_set(frame: pd.DataFrame, *, columns=SESSION_COLUMNS) -> set[tuple]:
    """The set of distinct session keys present in `frame` (order-independent)."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"frame missing session columns {missing}")
    return set(frame.loc[:, list(columns)].itertuples(index=False, name=None))


def n_sessions(frame: pd.DataFrame, *, columns=SESSION_COLUMNS) -> int:
    """Number of distinct sessions in `frame`."""
    return len(session_key_set(frame, columns=columns))


def session_key_frame(frame: pd.DataFrame, *, columns=SESSION_COLUMNS) -> pd.DataFrame:
    """The distinct session-key rows of `frame`, columns in canonical order."""
    return frame.loc[:, list(columns)].drop_duplicates().reset_index(drop=True)
```

**Cross-phase invariants (do not weaken):**

- `SESSION_COLUMNS` is *derived* from `row_keys(table="event")[:3]`, never hardcoded, so the registry stays the single source of truth.
- The helper return types are fixed: `set[tuple]`, `int`, `DataFrame`. The ~20 existing call sites use divergent types (set, list, tuple, int-count); each migrates to the matching helper, never to a new fourth shape.
- `event_table.py` (grain `day`) uses `SESSION_COLUMNS_DAY`. No phase renames that column to `session_date`.

## publish_canonical

New function in `src/continuum_swr_replay/table_io.py` — the single door onto the canonical tables. It composes the three steps every canonical writer does today (stamp → validate → atomic write) so an unstamped or unvalidated canonical table cannot be persisted.

```python
def publish_canonical(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    table: str,
    basename: str | None = None,
) -> Path:
    """Stamp, validate, and atomically write a canonical table.

    The one supported way to persist a canonical table. `table` is the schema
    key ("event", "event_arm", "stop", "qvals", ...). The frame is stamped with
    the covariate schema, its registered row keys are checked unique/nonmissing,
    and it is written atomically (temp + os.replace, umask-corrected). A writer
    that bypasses this and calls write_table directly persists an unstamped
    table that require_covariate_schema will later reject at fit time -- so the
    invariant is enforced here, at the write, not downstream.
    """
    from continuum_swr_replay.analysis.replay.covariate_registry import (
        stamp_covariate_schema,
    )
    stamped = stamp_covariate_schema(frame, table=table)
    target = path if basename is None else Path(path) / basename
    return write_table(stamped, target)
```

**Cross-phase invariants (do not weaken):**

- `stamp_covariate_schema` already validates row-key uniqueness/vocabulary (it calls `assert_canonical_vocabulary` + `assert_unique_scientific_rows`), so `publish_canonical` needs no separate validation call — but it must not be reducible back to a bare `write_table` at any call site.
- `write_table` stays public (phase-0 baseline scripts and the migration tool use it for non-canonical frames), but every *canonical* writer listed in overview.md's "Canonical-table writers" uses `publish_canonical`.
- The import of `stamp_covariate_schema` is function-local to avoid a module-level cycle (`table_io` is imported by `covariate_registry`'s consumers).
