# Test Suite for Continuum SWR Replay Pipeline

This directory contains a comprehensive test suite for the neuroscience data analysis pipeline.

## Quick Start

```bash
# Install test dependencies
uv sync --extra dev
```

### Fast offline slice (no VPN, no DataJoint)

These three modules import only standard scientific Python (numpy /
pandas / networkx / hypothesis) and finish in well under a second.
They're the right starting point on any machine:

```bash
uv run pytest tests/unit/test_basic.py
uv run pytest tests/unit/test_contfrag_paths.py
uv run pytest tests/unit/test_contfrag_hash_safety.py
```

### Full suite (requires DataJoint reachability)

`uv run pytest` collects the whole tree. Many modules import
`continuum_swr_replay.*` packages whose `__init__.py` files trigger
Spyglass, which opens a DataJoint connection at import time. Without
the lab VPN this either hangs at collection or fails with a
connection error.

```bash
uv run pytest                              # full suite (needs DataJoint)
uv run pytest tests/unit/test_spikes.py    # one file
uv run pytest -k "coincident"              # by name pattern
uv run pytest -v                           # verbose
uv run pytest --ignore=tests/integration   # skip the integration directory
```

## Test Organization

```
tests/
├── conftest.py                              # Shared fixtures
├── unit/                                    # ~30 function-level test files
│   ├── test_basic.py                        # Dependency-free smoke tests
│   ├── test_spikes.py                       # Spike processing
│   ├── test_lfp.py                          # LFP processing
│   ├── test_task_variables.py               # Task variable derivation
│   ├── test_replay_tables.py                # Replay-content tables
│   └── …                                    # See ``ls tests/unit`` for full list
├── integration/
│   └── test_mock_session_fixture.py         # Mock-session preset surface tests
└── property/
    └── test_mathematical_invariants.py      # Hypothesis-based property tests
```

``tests/integration/test_mock_session_fixture.py`` pins the shape /
invariants of the synthetic ``mock_session_data`` fixture and does
**not** call the real ``load_data`` pipeline. However, several
tests in that file do
``from continuum_swr_replay.types import DataLoadingPreset``, which
triggers the package's root ``__init__.py`` →
``data_loaders → bandit_task → spyglass``, so the file still needs
a working DataJoint setup at import time. The ``integration``
pytest marker is registered in ``pyproject.toml`` but no test is
currently decorated with it, so nothing is auto-deselected. There
is no live-data integration suite at the moment.

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Test individual functions in isolation with known inputs and expected outputs.

**Coverage:**
- **Spike Processing** (`test_spikes.py`):
  - Coincident spike detection and removal
  - Temporal filtering to position bounds
  - Spike time sorting and uniqueness
  - Feature-spike count matching

- **Task Variables** (`test_task_variables.py`):
  - Path progress calculation (0-1 normalization)
  - Distance to goal computation
  - Edge cases: same start/end nodes, disconnected graphs
  - Monotonicity and boundedness

- **LFP Processing** (`test_lfp.py`):
  - Z-score normalization
  - Division by zero handling (constant signals)
  - Ripple consensus trace computation
  - Event detection threshold crossings

**Run unit tests:**
```bash
uv run pytest tests/unit/
```

### 2. Integration Tests (`tests/integration/`)

Currently a single file, ``test_mock_session_fixture.py``, which
pins the shape and invariants of the synthetic ``mock_session_data``
fixture. It does **not** call the real ``load_data`` pipeline (that
function reads from Spyglass / DataJoint and is not available
without the lab VPN). A regression in the live loader would be
invisible here. A separate, DB-gated test file would be the natural
home for true end-to-end loader tests.

Note that running this file still requires a working DataJoint
setup: several tests do
``from continuum_swr_replay.types import DataLoadingPreset``, which
triggers ``continuum_swr_replay/__init__.py`` and pulls in
``data_loaders → bandit_task → spyglass`` at import time even though
no test invokes ``load_data``.

**Run:**

```bash
uv run pytest tests/integration/
```

### 3. Property-Based Tests (`tests/property/`)

Test mathematical invariants that should hold for all inputs using [hypothesis](https://hypothesis.readthedocs.io/).

**Coverage:**
- Path progress always in [0, 1]
- Z-score mean ≈ 0, std ≈ 1
- Graph distances satisfy triangle inequality
- Probability distributions sum to 1
- Spike filtering preserves order

``hypothesis`` is already pulled in by the ``dev`` extra
(``uv sync --extra dev``); no separate install step is needed.

**Run property tests:**

```bash
uv run pytest tests/property/
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

### Track Graph Fixtures
- `simple_linear_track` - 3-node linear track for basic tests
- `simplified_star_track` - Simplified 6-well bandit task structure
- `track_with_shortcut` - Track with alternative paths

### Data Fixtures
- `mock_position_data` - 10 seconds of position data at 500 Hz
- `stationary_position_data` - Zero velocity data for replay testing
- `mock_spike_times` - Multi-brain-area spike data
- `mock_spike_waveform_features` - HPC waveform features
- `coincident_spike_times` - Known artifact spikes for testing removal
- `mock_ripple_times` - Detected ripple events
- `mock_ripple_lfp` - Ripple-band filtered LFP
- `mock_trials` - Trial structure data

### Composite Fixtures
- `mock_session_data` - Complete session matching `load_data()` output

### Parametrized Fixtures
- `n_tetrodes` - Parametrize over different tetrode counts (0-4)
- `include_features` - Parametrize with/without waveform features

## Writing New Tests

### Unit Test Example

```python
def test_new_function():
    """Test description following scientific computing best practices."""
    # Arrange: Set up test data
    input_data = np.array([1.0, 2.0, 3.0])

    # Act: Call function under test
    result = my_function(input_data)

    # Assert: Check expected behavior
    expected = np.array([2.0, 4.0, 6.0])
    np.testing.assert_array_almost_equal(result, expected)
```

### Using Fixtures

```python
def test_with_fixture(mock_position_data):
    """Test using shared fixture."""
    assert len(mock_position_data) > 0
    assert mock_position_data.index.is_monotonic_increasing
```

### Parametrized Tests

```python
@pytest.mark.parametrize("n_tetrodes", [1, 2, 4, 8])
def test_multiple_tetrodes(n_tetrodes):
    """Test with different numbers of tetrodes."""
    spike_times = [np.array([1.0, 2.0]) for _ in range(n_tetrodes)]
    # ... test logic ...
```

### Property-Based Tests

```python
from hypothesis import given
from hypothesis import strategies as st

@given(
    n_points=st.integers(min_value=1, max_value=1000),
    position=st.floats(min_value=0, max_value=100),
)
def test_property(n_points, position):
    """Test mathematical property for random inputs."""
    result = compute_something(n_points, position)
    assert result >= 0  # Property: always non-negative
```

## Test Coverage Goals

### Priority 1 (Critical) - Implemented ✅
- [x] Coincident spike detection
- [x] Path progress calculation
- [x] Z-score normalization
- [x] Division by zero edge cases
- [x] Empty array handling
- [x] Temporal alignment
- [x] Data structure validation

### Priority 2 (High) - Recommended
- [ ] Event detection algorithms (HSE, ripple detection)
- [ ] Posterior statistics (MAP, HPD, ahead/behind)
- [ ] Decoder fitting and prediction
- [ ] Trial construction logic
- [ ] Well visit detection

### Priority 3 (Medium) - Nice to Have
- [ ] Visualization functions
- [ ] Result loading and saving
- [ ] Figurl generation
- [ ] Command-line argument parsing
- [ ] Logging configuration

## Testing Best Practices

### 1. Test Names Should Be Descriptive
```python
# Good
def test_coincident_spikes_removed_when_above_threshold():
    ...

# Bad
def test_spikes():
    ...
```

### 2. One Assertion per Test (When Possible)
```python
def test_path_progress_at_start_is_zero():
    progress = compute_progress(...)
    assert progress[0] == 0.0

def test_path_progress_at_end_is_one():
    progress = compute_progress(...)
    assert progress[-1] == 1.0
```

### 3. Test Edge Cases
- Empty arrays
- Single element arrays
- Division by zero
- NaN/Inf values
- Negative values
- Very large values

### 4. Use Appropriate Assertions
```python
# For arrays
np.testing.assert_array_equal(result, expected)
np.testing.assert_array_almost_equal(result, expected, decimal=6)

# For DataFrames
pd.testing.assert_frame_equal(result, expected)
pd.testing.assert_series_equal(result, expected)

# For floats
assert result == pytest.approx(expected, abs=1e-6)

# For exceptions
with pytest.raises(ValueError, match="specific error message"):
    function_that_should_raise()
```

### 5. Mock External Dependencies
For tests that would require:
- Database access (Spyglass/DataJoint)
- NWB file I/O
- GPU/CUDA operations
- Long-running computations

Use mocks or fixtures instead:
```python
@pytest.fixture
def mock_nwb_data(monkeypatch):
    """Mock NWB file access."""
    def mock_load(*args, **kwargs):
        return {"spike_times": [...]}

    monkeypatch.setattr("module.load_function", mock_load)
    return mock_load
```

## Continuous Integration

To run tests automatically on every commit, add to `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync --extra dev
      - name: Run tests
        run: uv run pytest
```

To collect coverage in CI, add ``pytest-cov`` to the ``dev`` extra
first; it is intentionally not part of the default dev set.

## Debugging Failed Tests

### Run with verbose output
```bash
uv run pytest -vv
```

### Run with print statements visible
```bash
uv run pytest -s
```

### Run only failed tests
```bash
uv run pytest --lf
```

### Drop into debugger on failure
```bash
uv run pytest --pdb
```

### Show local variables on failure
```bash
uv run pytest -l
```

## Coverage Reports

``pytest-cov`` is not in the default ``dev`` extra. Add it explicitly
when you want coverage:

```bash
uv add --dev pytest-cov
uv run pytest --cov=src/continuum_swr_replay --cov-report=html
```

Then open ``htmlcov/index.html`` in a browser.

Aspirational coverage targets (not currently enforced):

- **Critical modules** (spikes, task_variables, lfp): >80%
- **Analysis modules** (event_detection, posterior_analysis): >60%
- **Visualization modules**: >40%
- **Overall project**: >50%

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [hypothesis documentation](https://hypothesis.readthedocs.io/)
- [NumPy testing utilities](https://numpy.org/doc/stable/reference/routines.testing.html)
- [Pandas testing utilities](https://pandas.pydata.org/docs/reference/testing.html)

## Contributing Tests

When adding new functionality:

1. **Write tests first** (TDD approach) or alongside implementation
2. **Test edge cases** - don't just test the happy path
3. **Use descriptive names** - tests serve as documentation
4. **Keep tests independent** - each test should run in isolation
5. **Make tests fast** - use small datasets, avoid file I/O when possible
6. **Document complex tests** - explain what's being tested and why

## Questions?

For questions about testing this codebase, see:
- [CLAUDE.md](../CLAUDE.md) for project-specific conventions
- [pytest documentation](https://docs.pytest.org/) for general testing practices
- Open an issue on GitHub for test-specific questions
