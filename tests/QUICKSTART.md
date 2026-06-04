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

After the lazy-import work landed, the rest of the unit / property /
visualization / analysis suites also collect and run without a
working DataJoint connection — the package roots
(`continuum_swr_replay/__init__.py`,
`continuum_swr_replay/data_loaders/__init__.py`,
`continuum_swr_replay/analysis/hippocampus/__init__.py`) re-export
loader functions through PEP 562 ``__getattr__``, so importing a
leaf submodule no longer pulls Spyglass at import time. The lazy
contract is pinned by `tests/unit/test_lazy_imports.py`.

```bash
uv run pytest                              # full suite, offline-friendly
uv run pytest tests/integration/           # mock-fixture tests, also offline
uv run pytest tests/unit/test_replay_tables.py
```

## When you DO need DataJoint

Only the loader functions themselves require the lab VPN:
``load_data``, ``get_epoch_info``, ``get_training_timepoints``,
``get_labels``, ``get_theta``, ``get_ripple``, ``get_hpc_marks``,
``get_pfc_spike_times``, ``get_electrode_group_info``, and the
``run_session`` driver in ``analysis/pfc/glm.py``. Tests that
actually call them (none of the current `tests/` tree does) would
need DataJoint reachable. The ``integration`` pytest marker is
registered in ``pyproject.toml`` for future DB-gated tests but is
not currently applied to any test.

## What's actually here

- `tests/unit/` — function-level tests (~30 files: spikes, LFP, task
  variables, replay tables, fitting, decoding, ContFrag paths, hash
  safety, etc.).
- `tests/integration/test_mock_session_fixture.py` — exercises the
  selective-load preset surface against a synthetic mock session
  fixture. Does NOT call the real ``load_data``.
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
