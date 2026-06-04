# Test Suite

The full suite is offline-friendly (no DataJoint / VPN required) and runs in
under a minute on the dev machine.

## Quick start

```bash
uv sync --extra dev
uv run pytest                              # full suite (~37 s)
uv run pytest -m "not slow"                # fast subset (~28 s, skips 3 statistical tests)
uv run pytest tests/unit/test_spikes.py    # one file
uv run pytest -k "coincident"              # by name pattern
uv run pytest -vv                          # verbose, with full assertion diffs
uv run pytest --lf                         # rerun only last-failed tests
```

The ``slow`` marker is applied to the handful of Monte Carlo /
Tier-B statistical tests (SE coverage, calibration sweeps) that run
40+ replicates of a simulated estimator fit. CI runs everything; the
``-m "not slow"`` form is for tight local cycles.

The package roots (`continuum_swr_replay/__init__.py`,
`continuum_swr_replay/data_loaders/__init__.py`,
`continuum_swr_replay/analysis/hippocampus/__init__.py`) re-export loader
functions through PEP 562 `__getattr__`, and modules with Spyglass
dependencies defer those imports into the functions that actually need them.
The lazy contract is pinned by [`tests/unit/test_lazy_imports.py`](unit/test_lazy_imports.py)
— a regression there reintroduces the import-time DataJoint hang on no-VPN
machines.

The Spyglass-touching loader functions (`load_data`, `get_epoch_info`,
`get_training_timepoints`, `get_labels`, `get_theta`, `get_ripple`,
`get_hpc_marks`, `get_pfc_spike_times`, `get_electrode_group_info`, and
`analysis.pfc.glm.run_session`) require the lab VPN at *call* time. No tests
in this suite invoke them — the offline slice tests the pure helpers and
analysis layer.

## Layout

```
tests/
├── conftest.py              # shared fixtures (track graphs, mock spikes, etc.)
├── unit/                    # function-level tests, the bulk of the suite
├── analysis/                # tests grouped by analysis module
│   ├── hippocampus/         # event analysis, quality metrics, triggered analysis
│   └── pfc/                 # GLM, GLM pipeline, load_population
├── visualization/           # plotting helpers (figure_panels, palettes, etc.)
├── property/                # hypothesis-based invariants
└── benchmarks/              # microbenchmarks (run manually)
```

`tests/integration/` is reserved for future DB-gated tests. The `integration`
pytest marker is registered in `pyproject.toml` but currently unused.

## Fixtures

Shared fixtures in [`conftest.py`](conftest.py):

- **Track graphs**: `simple_linear_track` (3-node), `simplified_star_track`
  (7-node star — *not* the real spider topology; see
  [CLAUDE.md](../CLAUDE.md) for the production graph), `track_with_shortcut`.
- **Position**: `mock_position_data` (10 s @ 500 Hz), `stationary_position_data`.
- **Spikes**: `mock_spike_times`, `mock_spike_waveform_features`,
  `coincident_spike_times`.
- **LFP / events**: `mock_ripple_times`, `mock_ripple_lfp`, `mock_trials`.
- **Composite**: `mock_session_data` — synthetic session matching the shape
  of `load_data()`. Note: it's a *mock*, not a snapshot of real loader
  output — schema regressions in the live loader are invisible to it.

## Property tests

[`tests/property/`](property/) uses
[hypothesis](https://hypothesis.readthedocs.io/) for invariants that have to
hold across all valid inputs (e.g. `path_progress` always in `[0, 1]` and
monotone). Restricted to invariants that exercise project code — generic
NumPy/Pandas/NetworkX smoke tests don't belong here.

## Coverage

`pytest-cov` is in the dev extras. To run with coverage:

```bash
uv run pytest --cov=src/continuum_swr_replay --cov-report=term-missing
uv run pytest --cov=src/continuum_swr_replay --cov-report=html
# open htmlcov/index.html
```

Note: 0% coverage on `data_loaders/bandit_task.py` and 0–14% on most
visualization modules is *expected*. `bandit_task.py` is the Spyglass entry
point and is uncoverable offline; its pure orchestration helpers route into
leaf modules (`spikes.py`, `lfp.py`, `position.py`) whose helpers *are*
tested. Visualization modules are a deliberate non-target — plotting code is
low-ROI to test, and scientific correctness is verified through analysis
tests upstream of the plotting layer.

## Conventions

- **Test the project, not the library.** Tests of `np.isnan`, `df.loc[a:b]`,
  `nx.shortest_path` etc. without project code in the call chain don't belong
  here.
- **Pin behavior, not implementation.** Source-grep substring assertions
  (`assert "helper(" in src.read_text()`) and tautologies (test reproduces
  the function's formula to compute the expected value) are anti-patterns —
  they fail on cosmetic refactors and pass on real breakage.
- **No "should not crash" tests.** If the meaningful assertion is exception
  absence, say so structurally (`pytest.raises` against the relevant input)
  or pin a real downstream property.
- **Magic numbers need a why.** Hard-pinned floats (`assert x == 0.8`) need
  either a documented source (paper, parameter file) or a relaxed
  range/tolerance assertion tied to a downstream requirement.
- **Multi-assertion tests are fine.** "One assertion per test" gets in the
  way of pinning related properties of the same call (e.g. shape *and* value
  *and* dtype). Group naturally; split when the assertions diverge.

## Useful patterns

```python
# Float comparisons
assert result == pytest.approx(expected, abs=1e-6)

# Array comparisons
np.testing.assert_array_almost_equal(result, expected, decimal=6)

# DataFrame comparisons
pd.testing.assert_frame_equal(result, expected)

# Exception checks (match the discriminating part of the message, not the whole text)
with pytest.raises(ValueError, match="event_source must be"):
    fn(bad_input)

# Parametrize when the assertion shape is the same
@pytest.mark.parametrize("filename,animal", [
    ("chimi20200212_.nwb", "chimi"),
    ("j1620210706_.nwb", "j16"),
])
def test_parse(filename, animal):
    assert parse_nwb_filename(filename)[0] == animal
```

## Adding regression tests for un-importable scripts

`scripts/prepare_replay_content_data.py` and similar prep scripts can't be
unit-tested directly — they pull from Spyglass and run for hours. The pattern
used in [`test_prep_event_sequence_always_written.py`](unit/test_prep_event_sequence_always_written.py)
and [`test_contfrag_event_source_dispatch.py`](unit/test_contfrag_event_source_dispatch.py)
parses the script via `ast`, optionally `exec`s a single function definition
into an isolated namespace, then exercises it with synthetic inputs. Brittle
by design (any restructuring of the script breaks the test) but the only way
to catch regressions in code paths that only run during multi-hour pipeline
jobs.
