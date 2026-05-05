# Test Suite Quick Start

```bash
# Install dev deps (pytest, hypothesis, ruff, mypy)
uv sync --extra dev
```

## Fast offline slice (no VPN, no DataJoint)

`test_basic.py` is dependency-free and finishes in well under a
second. It is the right thing to run when reviewing a branch on a
machine without the lab VPN:

```bash
uv run pytest tests/unit/test_basic.py
```

`tests/unit/test_contfrag_paths.py` and
`tests/unit/test_contfrag_hash_safety.py` are also strictly offline —
they load only the named helper functions via `ast` so the package's
`data_loaders/__init__.py → bandit_task → spyglass` import chain is
never triggered.

## Full suite (requires a working DataJoint setup)

`uv run pytest` collects the entire `tests/` tree. Many test modules
import `continuum_swr_replay.*` packages whose `__init__.py` files
pull in Spyglass, which in turn opens a DataJoint connection at
import time. Without the lab VPN this either hangs at collection or
fails with a connection error.

When DataJoint is reachable:

```bash
uv run pytest                                # full unit + property + visualization + analysis
uv run pytest tests/unit/test_replay_tables.py
uv run pytest --ignore=tests/integration     # skip the live-data integration suite
```

`tests/integration/` is gated behind the `integration` pytest marker
and requires saved NWB / Spyglass tables in the configured
`PROCESSED_DATA_DIR`.

## What's actually here

- `tests/unit/` — function-level tests (~30 files: spikes, LFP, task
  variables, replay tables, fitting, decoding, ContFrag paths, hash
  safety, etc.).
- `tests/integration/test_mock_session_fixture.py` — exercises the
  selective-load preset surface against a mock session fixture.
- `tests/property/test_mathematical_invariants.py` — `hypothesis`-based
  property tests. `hypothesis` is included in the `dev` extra.

## What's not here

- No coverage tooling — `pytest-cov` is not in `dev`. Add it
  explicitly if you want coverage:

  ```bash
  uv add --dev pytest-cov
  uv run pytest --cov=src/continuum_swr_replay --cov-report=html
  ```

See [README.md](README.md) for fixture inventory and conventions.
