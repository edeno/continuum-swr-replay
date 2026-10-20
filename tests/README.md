# Test Suite

The last full run took about 31 min for 5,128 tests on the dev server. A plain
`uv run pytest` needs no lab VPN, DataJoint connection or `Processed-Data`: the
tests that do are marked `real_data` and are skipped unless you ask for them.

Two synthetic tests are the exception until the code-organization plan's
Task 12: they import a module that imports Spyglass, so on a host with no
DataJoint connection they fail with a connection error (the rest of the suite
still collects and runs):

- `unit/test_run_state_evidence.py::test_the_loader_builds_the_decodes_masks_from_the_swr_source`
- `unit/test_table_builders_read_the_register.py::test_hpc_raw_stats_run_over_the_registers_analysed_epochs`

## Quick start

```bash
uv sync --extra dev
uv run pytest -q -p no:cacheprovider       # full suite; real-data tests skipped
uv run pytest -m real_data                 # only the real-data tests
uv run pytest -m "not slow"                # skip the slow tests
uv run pytest tests/unit/test_spikes.py    # one file
uv run pytest -k "coincident"              # by name pattern
uv run pytest -vv                          # verbose, with full assertion diffs
uv run pytest --lf                         # rerun only last-failed tests
```

`pytest-xdist` is not installed, so there is no `-n`. A full check is the two
runs together: the default run, then `-m real_data` on a host with the data.

## Markers

The markers are registered in `pyproject.toml` (`[tool.pytest.ini_options]`).

- **`real_data`**: the test reads `Processed-Data/` (`PROCESSED_DATA_DIR`),
  `/stelmo`, a raw NWB file or the Spyglass/DataJoint database.
  - The root [`conftest.py`](conftest.py) skips it unless the `-m` expression
    names `real_data`. The skip reason is `real_data: opt in with -m real_data`.
  - Under `-m real_data` the test runs. Missing or stale data fails it rather
    than skipping it: opting in means the data must be there.
  - Every other test runs with no artifact run configured
    (`continuum_swr_replay.analysis.events.swr_source.ARTIFACT_INTERVALS_RUN_ID`
    is None). A `real_data` test gets the
    committed run back, so the canonical tables pass their SWR-source check.
    That restore is a function-scoped fixture: a module- or session-scoped
    fixture in a `real_data` module still sees None, and one that reads a
    table through `swr_source.with_swr_source` must set the run itself.
  - A test that reads only a small fixture committed under `tests/` is not real
    data. In a file that mixes the two, mark the real-data tests, not the file.
  - The hook is pinned by [`unit/test_markers.py`](unit/test_markers.py).
- **`integration`**: every tracked file in `tests/integration/` sets it with
  `pytestmark`. These tests run a whole runner or pipeline path, mostly on
  synthetic inputs. Deselect them with `-m "not integration"`.
- **`slow`**: Monte Carlo and statistical tests (SE coverage, calibration
  sweeps) and the end-to-end tests. Deselect them with `-m "not slow"`.

Markers combine as usual: `-m "real_data and not slow"`.

The package roots (`continuum_swr_replay/__init__.py`,
`continuum_swr_replay/data_loaders/__init__.py` and the `analysis` package
`__init__`s) re-export through PEP 562 `__getattr__`, and modules with Spyglass
dependencies defer those imports into the functions that need them. The lazy
contract is pinned by [`tests/unit/test_lazy_imports.py`](unit/test_lazy_imports.py).
A regression there brings back the import-time DataJoint hang on machines
without the VPN.

The Spyglass-touching loader functions (`load_data`, `get_epoch_info`,
`get_training_timepoints`, `get_labels`, `get_theta`, `get_ripple`,
`get_hpc_marks`, `get_pfc_spike_times`, `get_electrode_group_info` and
`analysis.pfc.glm.run_session`) need the lab VPN when they are called. Only
`real_data` tests call them for real; the other tests replace them with
`monkeypatch` or test the pure helpers.

## Layout

```
tests/
├── conftest.py              # shared fixtures and the real_data hook
├── unit/                    # the bulk of the suite, still mostly flat
├── analysis/                # tests that mirror src/continuum_swr_replay/analysis/
│   ├── hippocampus/
│   └── pfc/
├── visualization/           # plotting helpers (figure_panels, palettes, etc.)
├── integration/             # whole-runner and pipeline tests (marker: integration)
├── property/                # hypothesis-based invariants
├── baselines/               # figure-baseline manifest (see its README)
└── benchmarks/              # microbenchmarks (run manually)
```

**Where a test goes.** New tests mirror `src`: the tests of
`src/continuum_swr_replay/analysis/replay/<module>.py` go in
`tests/analysis/replay/test_<module>.py`, and those of `data_loaders/` in
`tests/data_loaders/`. Tests of `scripts/` modules go under `tests/scripts/`.
Existing files in `tests/unit/` move to the mirrored place when their module is
next touched. There is no big-bang move.

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

`scripts/decoding/prepare_contfrag_replay_content_data.py` and similar prep scripts can't
be unit-tested directly — they pull from Spyglass and run for hours. The pattern
used in [`test_prep_event_sequence_always_written.py`](unit/test_prep_event_sequence_always_written.py)
and [`test_contfrag_event_source_dispatch.py`](unit/test_contfrag_event_source_dispatch.py)
parses the script via `ast`, optionally `exec`s a single function definition
into an isolated namespace, then exercises it with synthetic inputs. Brittle
by design (any restructuring of the script breaks the test) but the only way
to catch regressions in code paths that only run during multi-hour pipeline
jobs.
