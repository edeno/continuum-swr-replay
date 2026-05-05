# Test Suite Quick Start

```bash
# Install dev deps (pytest, hypothesis, ruff, mypy)
uv sync --extra dev

# Run the full unit + property suite
uv run pytest

# Or just the fastest sanity slice
uv run pytest tests/unit/test_basic.py
```

`test_basic.py` is the smallest entry point — about 45 tests covering
NumPy/Pandas/NetworkX invariants and other dependency-free checks.

## What's actually here

- `tests/unit/` — function-level tests (spikes, LFP, task variables,
  replay tables, fitting, decoding, etc.). Roughly 30 files.
- `tests/integration/test_mock_session_fixture.py` — exercises the
  selective-load preset surface against a mock session fixture.
- `tests/property/test_mathematical_invariants.py` — `hypothesis`-based
  property tests. `hypothesis` is already pulled in by the `dev` extra.

## What's not here

- No live-DataJoint integration suite — anything that needs the lab VPN
  is staged behind the `integration` marker and not collected by
  default.
- No coverage tooling — `pytest-cov` is not in `dev`. Add it explicitly
  if you want coverage:

  ```bash
  uv add --dev pytest-cov
  uv run pytest --cov=src/continuum_swr_replay --cov-report=html
  ```

See [README.md](README.md) for fixture inventory and conventions.
