# Test Suite

The last full run took about 31 min for 5,128 tests on the dev server. A plain
`uv run pytest` needs no lab VPN, DataJoint connection or `Processed-Data`: the
tests that do are marked `real_data` and are skipped unless you ask for them.

## Quick start

```bash
uv sync --extra dev
uv run pytest -q -p no:cacheprovider       # full suite; real-data tests skipped
uv run pytest -m real_data                 # only the real-data tests
uv run pytest -m "not slow"                # skip the slow tests
uv run pytest tests/data_loaders/test_spikes.py    # one file
uv run pytest -k "coincident"              # by name pattern
uv run pytest -vv                          # verbose, with full assertion diffs
uv run pytest --lf                         # rerun only last-failed tests
```

`pytest-xdist` is not installed, so there is no `-n`. A full check is the two
runs together: the default run, then `-m real_data` on a host with the data.

JAX keeps what it compiles in a cache on disk,
`${XDG_CACHE_HOME:-~/.cache}/continuum-swr-replay/jax-compilation-cache`, so a
second run compiles little (set in [`conftest.py`](conftest.py)). It helps
most in the simulated decoder tests (`integration/test_*_simulated.py`), which
spend much of a first run compiling. Set `JAX_COMPILATION_CACHE_DIR` to use
another folder, or `JAX_ENABLE_COMPILATION_CACHE=false` to turn it off. JAX
never prunes the folder; delete it to reclaim the space, or when runs warn
that a cache entry cannot be read (two runs at once can damage one, and JAX
does not replace it).

## Markers

The markers are registered in `pyproject.toml` (`[tool.pytest.ini_options]`).

- **`real_data`**: the test reads `Processed-Data/` (`paths.processed_data_dir()`),
  `/stelmo`, a raw NWB file or the Spyglass/DataJoint database.
  - The root [`conftest.py`](conftest.py) skips it unless the `-m` expression
    names `real_data`. The skip reason is `real_data: opt in with -m real_data`.
  - Under `-m real_data` the test runs. Missing or stale data fails it rather
    than skipping it: opting in means the data must be there.
  - Every other test runs with no data root and no artifact run configured.
    Every data root is resolved when used, from `$CSR_PROCESSED_DATA_DIR`
    (`paths.processed_data_dir()`), and the root `conftest.py` sets that
    variable, before any test module is imported, to a sentinel: a regular file,
    so a test that forgets to set a root reads or writes nothing (a read raises
    `NotADirectoryError`, a write `NotADirectoryError` or `FileExistsError`,
    `exists()` is False). A test that needs a root takes the
    `processed_data_root` fixture (an empty `Processed-Data` under its
    `tmp_path`, via `monkeypatch.setenv`) or `replay_content_root` (its
    `replay_content/`), and lays its inputs out as `Processed-Data` is; the
    SWR-source fixtures in `support/_swr_inventory_fixtures.py` write their run
    folders where `paths` puts them under a root and point the variable at it.
    That one variable is the patch point: no test patches a module's root.
    And `continuum_swr_replay.analysis.event_definition.swr_source.ARTIFACT_INTERVALS_RUN_ID`
    is None. A `real_data` test gets the live root and the committed run back,
    so the canonical tables pass their SWR-source check. Those restores are
    function-scoped fixtures: a module- or session-scoped fixture in a
    `real_data` module still sees the sentinel and None; one that reads real
    data takes the `live_processed_data_dir` fixture as its root, and one that
    reads a table through `analyzed_time.with_swr_source` must set the run itself.
  - A test that reads only a small fixture committed under `tests/` is not real
    data. In a file that mixes the two, mark the real-data tests, not the file.
  - The hook is pinned by [`repository/test_markers.py`](repository/test_markers.py).
- **`integration`**: every tracked file in `tests/integration/` sets it with
  `pytestmark`. These tests run a whole runner or pipeline path, mostly on
  synthetic inputs. Deselect them with `-m "not integration"`.
- **`slow`**: Monte Carlo and statistical tests (SE coverage, calibration
  sweeps) and the end-to-end tests. Deselect them with `-m "not slow"`.

Markers combine as usual: `-m "real_data and not slow"`.

The package roots (`continuum_swr_replay/__init__.py`,
`continuum_swr_replay/data_loaders/__init__.py` and the
`analysis.pfc` `__init__`) re-export through PEP 562 `__getattr__`
(the other `analysis` package `__init__`s import nothing), and modules with Spyglass
dependencies defer those imports into the functions that need them. The lazy
contract is pinned by [`tests/repository/test_lazy_imports.py`](repository/test_lazy_imports.py).
A regression there brings back the import-time DataJoint hang on machines
without the VPN.

The Spyglass-touching loader functions (`load_data`, `get_epoch_info`,
`get_training_timepoints`, `get_labels`, `get_theta`, `get_ripple`,
`get_hpc_marks`, `get_pfc_spike_times`, `get_electrode_group_info` and
`analysis.pfc.session.run_session`) need the lab VPN when they are called. Only
`real_data` tests call them for real; the other tests replace them with
`monkeypatch` or test the pure helpers.

## Layout

```
tests/
├── conftest.py              # shared fixtures and the real_data hook
├── support/                 # shared helpers (_imports.py, the _*_fixtures.py) and data/
├── repository/              # rules of the repository, not of one module (layering, docs, no DB writes)
├── analysis/                # tests that mirror src/continuum_swr_replay/analysis/
├── data_loaders/  decoding/  reported/  runs/  tables/  visualization/
│                            # tests that mirror the other src/continuum_swr_replay/ packages
├── test_*.py                # tests of the package's top-level modules (intervals.py, paths.py, ...)
├── scripts/                 # tests that mirror scripts/
├── integration/             # whole-runner and pipeline tests (marker: integration)
├── property/                # hypothesis-based invariants
├── baselines/               # figure-baseline manifest (see its README)
└── benchmarks/              # microbenchmarks (run manually)
```

**Where a test goes.** Tests mirror `src` and `scripts`: the tests of
`src/continuum_swr_replay/analysis/replay_models/rate/<module>.py` go in
`tests/analysis/replay_models/rate/test_<module>.py`, those of `data_loaders/` in
`tests/data_loaders/`, and those of `scripts/<dir>/<script>.py` in
`tests/scripts/<dir>/`. A test goes beside the module it tests, which is the one it
imports for that purpose; a test of a rule several modules share (every SWR consumer
reads one source) goes beside the rule's module (`tests/analysis/event_definition/`). A test of
the repository itself rather than of a module goes in `tests/repository/`. Shared
helpers are in `tests/support/`, imported as `tests.support._name`.

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

Note: 0–14% coverage on most visualization modules is *expected*.
`bandit_task.py` imports Spyglass only inside its loaders; its orchestration
is tested with faked loaders (`tests/data_loaders/test_position.py`), and its Spyglass reads
need `-m real_data`. Visualization modules are a deliberate non-target — plotting code is
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
- **Bit-for-bit equality only for the same computation.** Two runs of the
  same operations on the same arrays are equal bit for bit. Two routes to the
  same numbers are not, on every machine: the linear-algebra library (OpenBLAS)
  picks its kernel from the CPU, and under its AVX2 kernels a row of a matrix
  product can differ in the last digit with the row's place in the matrix, so a
  design built on every row and then cut differs from one built on the rows
  kept. Compare such routes with a tolerance a wrong result cannot pass
  (`assert_allclose(..., rtol=1e-10)`), and do not let an output's order rest
  on which of two values equal but for rounding is larger. To run a test as an
  AVX2-only CI runner does, from a machine with AVX-512:

  ```bash
  OPENBLAS_CORETYPE=Haswell \
  NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL" \
  uv run --frozen pytest tests/analysis/regression
  ```

  This moves NumPy and SciPy only; JAX compiles for the CPU it runs on.

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

## Tests that extract a function by `ast`

Import what you test. A package module imports without Spyglass, so its
functions are tested by a plain import, with the Spyglass-backed steps replaced
by `monkeypatch`
([`test_prep_event_sequence_always_written.py`](../tests/scripts/tables/test_prep_event_sequence_always_written.py)
runs the replay table driver's `run_prepare_data` that way).

A few older tests still parse a file with `ast` and `exec` one function into an
isolated namespace, or inspect a function's syntax tree, instead of importing it.
They break on any restructuring of the file:

- [`test_run_quality_metrics_empty_calib.py`](../tests/scripts/checks/test_run_quality_metrics_empty_calib.py)
  (`scripts/checks/run_quality_metrics.py`);
- [`test_figure_script_fixes.py`](../tests/scripts/figures/test_figure_script_fixes.py)
  (`scripts/figures/plot_swr_contrast_coefs.py`, its plotting functions);
- [`test_prep_zero_event_session_keeps_stop_rows.py`](../tests/scripts/tables/test_prep_zero_event_session_keeps_stop_rows.py)
  (the syntax of `build_session_tables` in `analysis/event_tables/prepare_data.py`
  and `build_contfrag_session_tables` in `analysis/contfrag/prepare_data.py`);
- [`test_contfrag_hash_safety.py`](../tests/analysis/contfrag/test_contfrag_hash_safety.py)
  (`analysis/contfrag/tables.py` and `tables/event_ids.py`, both importable now).

Do not add another; convert one of these to an import when you touch it.
