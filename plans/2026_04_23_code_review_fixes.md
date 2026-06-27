# Code-Review Fix Plan (2026-04-23)

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Close the confirmed findings from the repo-wide review (`scratch/REVIEW_CONSOLIDATED.md` + `scratch/REVIEW_VERIFICATION.md`) in priority order, gating on tests, without regressing the paper's main figures.

**Architecture:** TDD for every code change. Write a failing test, watch it fail, make the minimal change, watch it pass, commit. Group work into waves by severity so the highest-impact fixes land first and each wave leaves the repo in a shippable state. Skip the one verified false positive (original C-HO1 / `get_overlap_times`).

**Tech Stack:** Python 3.11, uv, pytest, ruff, mypy, numpy/pandas/xarray, matplotlib, Spyglass (DataJoint + NWB), `non_local_detector`, statsmodels, JAX.

**Pre-flight for the engineer:**

1. Read [CLAUDE.md](../CLAUDE.md) end to end — especially the **junction place bins** section under "Spatial Bandit Task Structure".
2. Read [scratch/REVIEW_CONSOLIDATED.md](../scratch/REVIEW_CONSOLIDATED.md) for full context on each finding.
3. Read [scratch/REVIEW_VERIFICATION.md](../scratch/REVIEW_VERIFICATION.md) for severity corrections — the verifier downgraded five findings and upgraded one.
4. This plan assumes you can run `uv run pytest` from the repo root. GPU tests are fine to skip locally with `-m "not gpu"`.
5. Do **not** re-test findings the verifier flagged as false positives — `get_overlap_times` / original C-HO1 is skipped on purpose.
6. Commit after every task. Never batch multiple tasks into one commit.

**Directory conventions:**
- All tests live in `tests/unit/`, `tests/integration/`, `tests/property/`, `tests/baselines/`, `tests/benchmarks/`.
- All source lives in `src/continuum_swr_replay/`.
- Scripts live in `scripts/`.
- Use `uv run pytest <path>` — never `pytest` directly.

---

## Wave 1 — Figure-script hygiene (cheap, high-value)

Rationale: the duplicate `main()` doubles every Figure 1 run, the SWR-split legends are ambiguous about which events they trigger on, and `plot_event_anatomy.py` is a hazard at import time. None require scientific judgment.

### Task 1: Delete duplicate `main()` in Figure 1

**Files:**
- Modify: `scripts/generate_figure01.py:564-565`

**Step 1: Verify the duplicate**

Run: `tail -3 scripts/generate_figure01.py`
Expected output shows `main()` on the last two lines.

**Step 2: Delete line 565**

Use `Edit` to remove the second `main()` call (the last line of the file).

**Step 3: Sanity check**

Run: `uv run python -c "import ast; ast.parse(open('scripts/generate_figure01.py').read())"`
Expected: no output (file parses).

**Step 4: Commit**

```bash
git add scripts/generate_figure01.py
git commit -m "fix: drop duplicate main() in generate_figure01

The last two lines of the script were main() followed by main(), so every
invocation ran the full EM fit and save twice. Paste accident."
```

---

### Task 2: Disambiguate triggered-average legend qualifiers

**Files:**
- Modify: `src/continuum_swr_replay/visualization/figure_panels.py:221`
- Modify: `src/continuum_swr_replay/visualization/figure_panels.py:227`

Rationale: the current legends "SWR Overlap" / "No SWR Overlap" don't say *what* is overlapping with SWRs. Readers can't tell whether the panel triggers on SWR events or on non-local events that overlap SWRs. Adding the "non-local" qualifier removes the ambiguity.

**Step 1: Confirm**

Run: `grep -n "SWR Overlap\|No SWR Overlap" src/continuum_swr_replay/visualization/figure_panels.py`
Expected: exactly two matches.

**Step 2: Replace**

- `"No SWR Overlap"` → `"non-SWR non-local"`
- `"SWR Overlap"` → `"SWR-coincident non-local"`

**Step 3: Also fix `plot_replay_speed_by_swr` and `plot_replay_position_by_swr` legends**

Run: `grep -n '"With SWR"\|"Without SWR"' src/continuum_swr_replay/visualization/figure_panels.py`
Expected: 4 matches around lines 356, 368, 421, 434.

Replace:
- `"With SWR"` → `"SWR-coincident non-local"`
- `"Without SWR"` → `"non-SWR non-local"`

**Step 4: Commit**

```bash
git add src/continuum_swr_replay/visualization/figure_panels.py
git commit -m "fix: disambiguate SWR-split legends with the 'non-local' qualifier

'SWR Overlap'/'No SWR Overlap' and 'With SWR'/'Without SWR' didn't say
what was overlapping. The panels trigger on non-local events split by
SWR coincidence; spell that out."
```

---

### Task 3: Quarantine `plot_event_anatomy.py`

Rationale: the script references `long.pkl`/`events.pkl` which no longer exist, has a hardcoded absolute output path, and has no `__main__` guard, so any test collection that imports it crashes.

**Files:**
- Move: `scripts/plot_event_anatomy.py` → `scripts/archive/plot_event_anatomy.py`

**Step 1: Confirm it's broken as-is**

Run: `uv run python -c "import ast; tree = ast.parse(open('scripts/plot_event_anatomy.py').read()); print([n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign))][:3])`
You should see `Expr` / `For` nodes at module scope → no `__main__` guard.

**Step 2: Move into archive**

```bash
mkdir -p scripts/archive
git mv scripts/plot_event_anatomy.py scripts/archive/plot_event_anatomy.py
```

**Step 3: Confirm no import paths broken**

Run: `grep -rn "plot_event_anatomy" --include="*.py" .`
Expected: no hits outside the moved file itself.

**Step 4: Commit**

```bash
git add -A scripts/
git commit -m "chore: archive stale plot_event_anatomy.py

Script references deleted files (long.pkl, events.pkl), has a hardcoded
absolute output path, and no __main__ guard, so importing it crashes.
Move to scripts/archive/ until it is either updated to the canonical
event_table.pkl/event_arm_table.pkl files or deleted."
```

---

## Wave 2 — Reconcile the two "contained" SWR-overlap definitions (upgraded to Critical-adjacent)

Rationale: `classify_swr_overlap` (event_detection.py) and `compute_swr_overlap_class_counts` (data_aggregation.py) disagree when a non-local event spans two consecutive SWRs. Both feed Figure 2.

### Task 4: Write a failing test exposing the discrepancy

**Files:**
- Create: `tests/unit/test_swr_overlap_consistency.py`

**Step 1: Write the test**

```python
"""Regression test: the two 'contained' definitions must agree.

classify_swr_overlap (event_detection.py) and
compute_swr_overlap_class_counts (data_aggregation.py) both label
non-local events by their overlap with SWRs. They disagreed for events
spanning multiple consecutive SWRs. Ensure they now agree on all
canonical geometries.
"""
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.event_detection import (
    classify_swr_overlap,
)
from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
    compute_swr_overlap_class_counts,
)


def _non_local(start, end, event_number):
    return pd.DataFrame(
        {"start_time": [start], "end_time": [end]},
        index=pd.Index([event_number], name="event_number"),
    )


def _swrs(intervals):
    return pd.DataFrame(intervals, columns=["start_time", "end_time"])


@pytest.mark.parametrize(
    "event,swrs,expected",
    [
        # Single SWR fully containing the event
        ((1.0, 2.0), [(0.5, 2.5)], "contained"),
        # Event fully containing a single SWR
        ((0.5, 2.5), [(1.0, 2.0)], "extends"),
        # Two consecutive SWRs with full coverage, no single one contains
        ((1.0, 2.0), [(0.9, 1.5), (1.5, 2.1)], "contained"),
        # Partial overlap
        ((1.0, 2.0), [(1.5, 2.5)], "overlaps"),
        # Disjoint
        ((1.0, 2.0), [(3.0, 4.0)], "no_overlap"),
    ],
)
def test_contained_definitions_agree(event, swrs, expected):
    non_local = _non_local(*event, event_number=1)
    swr = _swrs(swrs)

    geometric = classify_swr_overlap(non_local, swr).iloc[0]["overlap_class"]
    counts = compute_swr_overlap_class_counts(non_local, swr)

    # Both functions must classify this single event identically.
    assert geometric == expected, f"classify_swr_overlap said {geometric}"
    assert counts.loc[counts["overlap_class"] == expected, "n_events"].iloc[0] == 1
```

**Step 2: Run and watch it fail**

Run: `uv run pytest tests/unit/test_swr_overlap_consistency.py -v`
Expected: at least the "two consecutive SWRs" case fails because `compute_swr_overlap_class_counts` returns `extends` while `classify_swr_overlap` returns `contained` (or vice-versa).

**Step 3: Commit the failing test**

```bash
git add tests/unit/test_swr_overlap_consistency.py
git commit -m "test: pin contained/extends consistency across overlap classifiers"
```

---

### Task 5: Unify the two definitions

**Files:**
- Decide: which definition is the paper's primary? Write a 2-line comment at the top of the unchanged function explaining the choice.
- Modify: whichever function you change.

**Step 1: Pick the authoritative definition**

Consult the analysis plans under `plans/` and the Figure 2 caption draft for which notion of "contained" the paper uses. If unclear, go with **`classify_swr_overlap`** (geometric "any single SWR fully spans the event") since it is the simpler, more interpretable criterion and matches what readers typically imagine.

**Step 2: Make the other function delegate**

Change `compute_swr_overlap_class_counts` in `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:437-443` to call `classify_swr_overlap(non_local, swr)` on a per-event basis and then aggregate counts, rather than re-deriving "contained" from `has_overlap.all()`.

Concrete replacement (paste into the function, adjusting variable names to match the surrounding code):

```python
def compute_swr_overlap_class_counts(
    non_local_events: pd.DataFrame,
    swr_events: pd.DataFrame,
) -> pd.DataFrame:
    """Count non-local events by overlap class.

    Uses ``classify_swr_overlap`` as the single source of truth for the
    contained / extends / overlaps / no_overlap classification.
    """
    from continuum_swr_replay.analysis.hippocampus.event_detection import (
        classify_swr_overlap,
    )

    classified = classify_swr_overlap(non_local_events, swr_events)
    counts = (
        classified["overlap_class"]
        .value_counts()
        .rename_axis("overlap_class")
        .reset_index(name="n_events")
    )
    return counts
```

**Step 3: Run the tests**

Run: `uv run pytest tests/unit/test_swr_overlap_consistency.py -v`
Expected: all parametrized cases pass.

Also run the full hippocampus suite to catch fallout:

Run: `uv run pytest tests/unit/test_hippocampus* tests/unit/test_event_detection* -v`

**Step 4: Commit**

```bash
git add src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py
git commit -m "fix: unify 'contained' SWR overlap definition

compute_swr_overlap_class_counts and classify_swr_overlap used different
criteria for 'contained' (all-timebin-overlap vs. single-SWR-spanning).
For events spanning two consecutive SWRs they disagreed. Make the
counts function delegate to classify_swr_overlap so Figure 2's example
panel and population bar chart use the same definition."
```

---

## Wave 3 — Scientific arithmetic fixes

### Task 6: `arm_fraction` denominator excludes junction bins (M-R1)

**Files:**
- Test: `tests/unit/test_replay_tables.py` (add to existing file)
- Modify: `src/continuum_swr_replay/analysis/replay/tables.py:900-908`

**Step 1: Write the failing test**

Append to `tests/unit/test_replay_tables.py`:

```python
def test_arm_fraction_sums_to_one_per_event_when_real_bins_only(
    synthetic_session_with_junction_bins,
):
    """arm_fraction must sum to 1 per event after the junction-bin fix.

    Prior behaviour: denominator was n_nonlocal_bins (includes junction
    rows); arm_count_hard numerator excluded them; sum < 1 whenever
    junction bins existed.
    """
    from continuum_swr_replay.analysis.replay.tables import build_event_arm_table

    long = build_event_arm_table(
        event_table=synthetic_session_with_junction_bins.event_table,
        posterior=synthetic_session_with_junction_bins.posterior,
        environment=synthetic_session_with_junction_bins.environment,
    )
    # For every event where any real-arm bins exist, the per-event sum
    # must be 1.0 (or NaN for all-junction events).
    per_event_sum = long.groupby("event_number")["arm_fraction"].sum(min_count=1)
    assert (
        np.isnan(per_event_sum) | np.isclose(per_event_sum, 1.0)
    ).all(), f"arm_fraction doesn't sum to 1: {per_event_sum}"
```

If `synthetic_session_with_junction_bins` doesn't exist, look at existing fixtures in `tests/unit/test_replay_tables.py` — there is a `make_synthetic_env` helper. Use it to construct an event where the MAP lands on a junction bin for ≥1 timebin.

**Step 2: Run and watch it fail**

Run: `uv run pytest tests/unit/test_replay_tables.py::test_arm_fraction_sums_to_one_per_event_when_real_bins_only -v`
Expected: FAIL (sum is < 1 for junction-bin events).

**Step 3: Fix the denominator**

Edit `tables.py:900-908`:

```python
# arm_fraction: share of the event's real-arm time attributed to each
# arm. Denominator excludes junction-bin timebins (MAP on edge_id=-1)
# so arm_count_hard is normalized over the same bins that produced it.
real_bins_long = long["n_nonlocal_bins"].to_numpy() - long["n_junction_bins"].to_numpy()
with np.errstate(divide="ignore", invalid="ignore"):
    frac = np.where(
        real_bins_long > 0,
        long["arm_count_hard"].to_numpy() / np.maximum(real_bins_long, 1),
        np.nan,
    )
long["arm_fraction"] = frac
```

If `n_junction_bins` is not yet materialized at the per-event level in `tables.py`, add it — `summaries.py` already computes `n_junction_bins_per_event`; surface it in the event table first (likely a small change around `build_event_table`).

**Step 4: Run all replay tests**

Run: `uv run pytest tests/unit/test_replay_tables.py tests/unit/test_replay_summaries.py -v`
Expected: the new test passes and no regressions.

**Step 5: Commit**

```bash
git add tests/unit/test_replay_tables.py src/continuum_swr_replay/analysis/replay/tables.py
git commit -m "fix(replay): exclude junction bins from arm_fraction denominator

arm_count_hard (numerator) already excludes junction bins via real_mask.
n_nonlocal_bins (denominator) did not, so sum(arm_fraction) < 1 whenever
junction bins existed, deflating patch/arm-fraction figures and biasing
the null log-likelihood in fit_conditional_poisson. Denominator now
uses (n_nonlocal_bins - n_junction_bins)."
```

---

### Task 7: `wilcoxon` on NaN-dropped series (M-R3)

**Files:**
- Test: `tests/unit/test_model_comparison.py` (created here — this also seeds Wave 5's Task 17)
- Modify: `src/continuum_swr_replay/analysis/replay/model_comparison.py:148`

**Step 1: Create the test file**

```python
"""Tests for analysis/replay/model_comparison.py.

These cover the behaviour that the paper's statistical tables rely on:
AIC/BIC differencing, LRT sign handling, mismatch invalidation, and
Wilcoxon-on-NaN defaults.
"""
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.model_comparison import (
    compare_nested_models,
    summarize_comparison,
)


def test_summarize_comparison_ignores_nan_sessions_in_wilcoxon():
    """scipy.stats.wilcoxon's default nan_policy='propagate' returns NaN
    if any value is NaN. summarize_comparison must drop NaN sessions
    before calling wilcoxon so the omnibus p-value is valid whenever at
    least two sessions are comparable."""
    comparisons = pd.DataFrame(
        {
            "session": ["a", "b", "c", "d"],
            "delta_aic": [-1.0, -2.0, np.nan, -0.5],
            "lr_stat": [5.0, 4.0, np.nan, 2.0],
            "df_diff": [1, 1, 1, 1],
        }
    )
    result = summarize_comparison(comparisons)
    assert not np.isnan(result["wilcoxon_p"]), (
        "wilcoxon_p was NaN; summarize_comparison did not drop NaN sessions"
    )
    assert result["n_sessions"] == 3
```

**Step 2: Run and watch it fail**

Run: `uv run pytest tests/unit/test_model_comparison.py -v`
Expected: FAIL (wilcoxon_p is NaN).

**Step 3: Fix**

Edit `model_comparison.py:148`:

```python
# Drop NaN sessions (from mismatched n_events upstream) so wilcoxon's
# default 'propagate' policy does not return NaN.
_, p = wilcoxon(daic.dropna())
```

**Step 4: Verify**

Run: `uv run pytest tests/unit/test_model_comparison.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit/test_model_comparison.py src/continuum_swr_replay/analysis/replay/model_comparison.py
git commit -m "fix(replay): drop NaN sessions before wilcoxon in model comparison

scipy.stats.wilcoxon with default nan_policy='propagate' returns NaN
for any NaN in delta_aic. When the mismatch guard set some sessions
to NaN, the omnibus p-value was silently invalid. Drop NaN first."
```

---

### Task 8: `MUA_zscore` missing reindex (M-HO1)

**Files:**
- Test: `tests/unit/test_hippocampus_data_aggregation.py` (create if absent)
- Modify: `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:266`
- Modify: `src/continuum_swr_replay/analysis/contfrag/data_aggregation.py:140`

**Step 1: Write the test**

```python
"""MUA_zscore must be time-aligned to results.time like other signals."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr


def test_mua_zscore_is_reindexed_to_results_time():
    """Regression: MUA_zscore was assigned raw without the
    reindex(method='nearest', tolerance=0.004) that theta/SWR use.
    A 2 ms offset is ~10% of SWR peak width."""
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        assemble_time_aligned_signals,  # or whatever the helper is named
    )

    # Build a results DataArray on a 500 Hz grid
    time = np.arange(0, 1.0, 0.002)
    results = xr.Dataset({"time": ("time", time)})

    # Simulate MUA sampled at a shifted 500 Hz grid (1 ms offset)
    mua_time = time + 0.001
    mua = pd.DataFrame({"multiunit_rate_zscore": np.arange(len(mua_time))}, index=mua_time)

    out = assemble_time_aligned_signals(
        results=results,
        data={"multiunit_rate_zscore": {"HPC": mua}},
        brain_area="HPC",
    )
    # If reindex is applied, NaN values for any out-of-tolerance samples
    # are acceptable. The key invariant: the length matches results.time.
    assert len(out["MUA_zscore"]) == len(time)
    # And no raw-length mismatch silently truncated or padded:
    assert out.index.equals(pd.Index(time, name="time"))
```

Adapt to whatever helper actually assembles the signals. If there is no such helper and the inline code at `:266` does it all, add the reindex inline (and test via an integration-style fixture).

**Step 2: Watch it fail**

**Step 3: Add the reindex**

Edit `hippocampus/data_aggregation.py:266` and the analogous line in `contfrag/data_aggregation.py:140` to use the same reindex pattern as theta/SWR:

```python
mua_series = (
    data["multiunit_rate_zscore"][brain_area]
    .squeeze()
    .reindex(results.time.values, method="nearest", tolerance=0.004)
)
```

**Step 4: Verify + commit**

```bash
git add tests/ src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py src/continuum_swr_replay/analysis/contfrag/data_aggregation.py
git commit -m "fix: reindex MUA_zscore to results.time like other signals

theta_phase, theta_power_zscore, and SWR_zscore were reindexed with
tolerance=4 ms but MUA_zscore was inserted raw. A 1-sample (2 ms)
offset is ~10% of SWR peak width, corrupting peri-event analyses."
```

---

### Task 9: `fit_conditional_logit` zero-count filter (M-R4)

**Files:**
- Test: `tests/unit/test_fitting.py` (extend)
- Modify: `src/continuum_swr_replay/analysis/replay/fitting.py:244`

**Step 1: Failing test**

```python
def test_fit_conditional_logit_filters_zero_count_events():
    """fit_conditional_poisson filters event_totals > 0;
    fit_conditional_logit must do the same or zero-count events get
    an arbitrary 'dominant' arm via idxmax."""
    import pandas as pd
    from continuum_swr_replay.analysis.replay.fitting import (
        fit_conditional_logit,
    )

    long_df = pd.DataFrame(
        {
            "event_number": [1, 1, 1, 2, 2, 2],
            "arm": [0, 1, 2, 0, 1, 2],
            "arm_count_hard": [0, 0, 0, 3, 1, 0],  # event 1 is all zeros
            "is_prev": [1, 0, 0, 1, 0, 0],
        }
    )
    result = fit_conditional_logit(long_df, predictors=["is_prev"])
    assert result["n_events"] == 1, "zero-count event was not filtered"
```

**Step 2: Watch fail, fix, verify, commit** — mirror the filter from `fit_conditional_poisson:83-84`.

---

### Task 10: `build_choice_dataset` NaN `current_arm` → 9-arm set (M-R5)

**Files:**
- Test: `tests/unit/test_choice_prediction.py` (create — also seeds Wave 6's Task 22)
- Modify: `src/continuum_swr_replay/analysis/replay/choice_prediction.py:244`

**Step 1: Failing test**

```python
def test_build_choice_dataset_drops_nan_current_arm_stops():
    """arm != NaN is True for all arms (IEEE 754), so the exclusion
    step keeps all 9 arms when current_arm is NaN, inflating the
    baseline probability for those stops."""
    import numpy as np
    import pandas as pd
    from continuum_swr_replay.analysis.replay.choice_prediction import (
        build_choice_dataset,
    )

    stops = pd.DataFrame(
        {
            "stop_id": [1, 2],
            "current_arm": [0, np.nan],
            "next_arm": [3, 2],
        }
    )
    event_arm = pd.DataFrame({"arm": list(range(9)) * 2, "arm_fraction": [1 / 9] * 18})
    ds = build_choice_dataset(stops, event_arm)
    # Stop 2 should be excluded or down to 8 arms; currently it has 9.
    assert (ds.groupby("stop_id").size() == 8).all()
```

**Step 2-5: fix, verify, commit** — filter `current_arm.notna()` before the exclusion at `:244`, and document the choice in a short comment.

---

### Task 11: PFC GLM single-fit consistency (M-HO6)

**Files:**
- Modify: `src/continuum_swr_replay/analysis/pfc/glm.py:769-815`

**Rationale:** modulation time courses come from `fit_regularized(alpha=1e-20)`; AIC comes from the unregularized `results`. They are numerically ≈ identical but logically two fits. Collapse to one.

**Step 1: Decide which fit to keep**

Keep the unregularized one (simpler, already used for LR/AIC). Remove the `fit_regularized(alpha=1e-20)` branch unless it is required to sidestep a real IRLS convergence problem — if so, document *why* with a comment citing the failure mode.

**Step 2: Run the whole PFC test suite after the change**

Run: `uv run pytest tests/analysis/pfc/ -v`
Expected: PASS.

**Step 3: Commit**

```bash
git add src/continuum_swr_replay/analysis/pfc/glm.py
git commit -m "refactor(pfc-glm): compute modulation and AIC from the same fit

Previously LR/AIC used unregularized results while modulation time
courses used fit_regularized(alpha=1e-20). Numerically near-identical
but logically inconsistent; if IRLS diverges for one and not the other,
neuron_stats and modulation arrays silently disagree."
```

---

## Wave 4 — Data-loader robustness

### Task 12: `turn_direction` edge-ID mapping (downgraded from Critical but still latent)

**Files:**
- Test: `tests/unit/test_task_variables.py` (extend)
- Modify: `src/continuum_swr_replay/data_loaders/task_variables.py:345, 523`

**Step 1: Failing test**

Construct a `TrackGraph` where NetworkX edge iteration order differs from `linear_edge_order` — e.g., insert edges in `[ (6,9), (0,6), (1,6), ... ]` order but assign `edge_id = [6, 0, 1, ...]` attributes. Then compute `turn_direction` for a canned trajectory and compare to hand-computed expectation.

**Step 2: Watch fail**

**Step 3: Fix — index via the `edge_id` attribute**

Replace:
```python
track_edges = np.array(track_graph.edges)  # :345, :523
```
with:
```python
edge_list = list(track_graph.edges(data=True))
track_edge_by_id = {data["edge_id"]: (u, v) for u, v, data in edge_list}
def _edge_for_segment(sid: int) -> tuple[int, int]:
    return track_edge_by_id[sid]
```
then call `_edge_for_segment(segment_id)` where the code currently does `track_edges[segment_id]`.

**Step 4: Verify + commit**

---

### Task 13: Theta `lfp_band_sampling_rate` restriction (M-DL3)

**Files:**
- Modify: `src/continuum_swr_replay/data_loaders/lfp.py:99-106`

**Step 1: Check the current ripple restriction for the right value**

Run: `grep -n "lfp_band_sampling_rate" src/continuum_swr_replay/data_loaders/lfp.py`
Find the ripple-branch value (likely `1000`).

**Step 2: Add it to `theta_lfp_key`** so the two query patterns match.

**Step 3: Smoke-test against a session**

Run: `uv run python -c "from continuum_swr_replay.data_loaders.lfp import get_lfp_data; d = get_lfp_data('chimi20200212_.nwb', '02_r1'); print(d.keys())"`
Expected: works (or the same behaviour you got before — no runtime regression).

**Step 4: Commit**

---

### Task 14: j16 animal-name extraction (m/pfc minor)

**Files:**
- Modify: `src/continuum_swr_replay/analysis/pfc/load_population.py:56`

**Step 1: Failing test**

```python
def test_load_population_animal_name_j16():
    """str.split('2')[0] returns 'j1' for j1620210706. Use parse_nwb_filename."""
    import pandas as pd
    from continuum_swr_replay.analysis.pfc.load_population import _extract_animal
    assert _extract_animal("j1620210706_.nwb") == "j16"
```

Adapt the test to whatever the actual code structure is — if `_extract_animal` does not exist, create it and route `load_population_data` through it.

**Step 2: Fix** — use `parse_nwb_filename` from `data_loaders.utils`.

**Step 3: Verify + commit**

---

### Task 15: Move DLC time-slice inside `get_position_info` (M-DL2)

**Files:**
- Modify: `src/continuum_swr_replay/data_loaders/position.py:257-272`
- Review: `src/continuum_swr_replay/data_loaders/bandit_task.py:329-337` (may allow removing the `dropna`)

**Step 1: Slice `time` to `valid_interval_times` immediately after slicing `position_info`** so the invariant is self-contained.

**Step 2: Remove the redundant NaN-drop** in `bandit_task.py` if the slice fully replaces it — or leave a short comment explaining why both are needed.

**Step 3: Verify with a full `load_data` smoke test**

Run: `uv run python -c "from continuum_swr_replay.data_loaders.load_data_bandit import load_data; d = load_data('chimi20200212_.nwb', '02_r1'); print(len(d['position_info']))"`
Expected: same length as before the change.

**Step 4: Commit**

---

## Wave 5 — Critical test coverage gaps

Order matters: write the tests for `shuffles`, `model_comparison`, `spatial_mapping` before any further refactors to those modules.

### Task 16: `tests/unit/test_shuffles.py` (C-T2)

**Files:**
- Create: `tests/unit/test_shuffles.py`

**Step 1: Write the test suite**

```python
"""Tests for analysis/replay/shuffles.py.

Shuffles underpin every significance test in the paper. Pin the
invariants: marginal preservation, multi-session guards, non-identity
shift.
"""
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.shuffles import (
    circular_time_shuffle,
    next_arm_circular_shift,
    stop_label_shuffle,
)


class TestCircularTimeShuffle:
    def test_preserves_per_event_arm_marginal(self):
        """Circular shift preserves the empirical distribution of arm
        visits within each event — only the sequential ordering
        changes."""
        rng = np.random.default_rng(42)
        arms = rng.integers(0, 9, size=500)
        event_ids = np.repeat(np.arange(10), 50)
        shuffled = circular_time_shuffle(
            arms, event_ids=event_ids, rng=np.random.default_rng(0)
        )
        # For each event, the sorted counts of arm labels are invariant.
        for eid in np.unique(event_ids):
            orig = np.bincount(arms[event_ids == eid], minlength=9)
            shuf = np.bincount(shuffled[event_ids == eid], minlength=9)
            np.testing.assert_array_equal(orig, shuf)

    def test_single_bin_event_passes_through(self):
        """n=1 events cannot be shifted; output == input."""
        out = circular_time_shuffle(
            np.array([3]), event_ids=np.array([0]), rng=np.random.default_rng(0)
        )
        assert out.tolist() == [3]

    def test_shift_is_never_zero(self):
        """Docstring contract: shift in {1, ..., n-1}."""
        # Run many times and verify the output differs from input for
        # events of length >= 2.
        rng_master = np.random.default_rng(0)
        for _ in range(100):
            arms = np.array([0, 1, 2, 3])
            event_ids = np.zeros(4, dtype=int)
            out = circular_time_shuffle(arms, event_ids=event_ids, rng=rng_master)
            assert not np.array_equal(out, arms)


class TestStopLabelShuffle:
    def test_raises_on_multi_session_input(self):
        """Guard against the most common misuse."""
        stops = pd.DataFrame(
            {"animal": ["a", "a", "b"], "day": [1, 1, 1], "epoch": [1, 1, 1]}
        )
        with pytest.raises(ValueError, match="session"):
            stop_label_shuffle(stops, rng=np.random.default_rng(0))


class TestNextArmCircularShift:
    def test_never_returns_identity(self):
        events = pd.DataFrame(
            {"time": np.arange(10, dtype=float), "next_arm": np.arange(10) % 9}
        )
        out = next_arm_circular_shift(events, rng=np.random.default_rng(7))
        assert not np.array_equal(out["next_arm"], events["next_arm"])
```

Adapt signatures to the real API in `shuffles.py`.

**Step 2-5: watch pass (these are already-correct invariants); commit**

---

### Task 17: Complete `tests/unit/test_model_comparison.py` (C-T3)

Already seeded by Task 7. Now add:

- test that `compare_nested_models` produces correct ΔAIC with hand-computed LLs
- test that sessions with mismatched `n_events` get NaN comparison columns
- test that `summarize_comparison` returns `{"n_sessions": 0}` for empty input
- test that a negative LR stat (child fits worse) is handled — document whether the clip is "silently treat as p=1" or "warn + clip"

Same task structure: red → green → commit.

---

### Task 18: `tests/unit/test_spatial_mapping.py` (C-T4)

**Files:**
- Create: `tests/unit/test_spatial_mapping.py`

Cover:

- `map_estimate_to_arm`: MAP estimate exactly at a junction-bin position maps to the nearest real-arm bin, not NaN.
- All-junction input: returns NaN (no snapping to real arms when no real-arm data exists).
- `arm_to_well`: trunk arms (6, 7, 8) return NaN; leaf arms return their well.
- `map_estimate_to_patch`: one test per patch verifying the mapping matches `TRACK_SEGMENT_TO_PATCH` after junction exclusion.

Structure: one test per invariant, ~8-10 tests.

---

### Task 19: Baseline manifest + log gitignore (C-T5)

**Files:**
- Create: `tests/baselines/manifest.json`
- Modify: `.gitignore`
- Delete: `tests/baselines/figure02.log`, `tests/baselines/figure03.log`
- Possibly: quarantine `tests/baselines/figure03*` until regen (document in README).

**Step 1: Add the log pattern to `.gitignore`**

Append `tests/baselines/*.log`.

**Step 2: Delete the committed logs**

```bash
git rm tests/baselines/figure02.log tests/baselines/figure03.log
```

**Step 3: Write the manifest**

```json
{
  "baselines": {
    "figure02.pdf": {
      "git_sha": "<fill in from `git rev-parse HEAD` at generation time>",
      "data_md5": "<fill in from md5sum of the input NWB>",
      "generated_date": "2025-11-19",
      "status": "valid"
    },
    "figure03.pdf": {
      "git_sha": null,
      "data_md5": null,
      "generated_date": "2025-11-19",
      "status": "quarantined — regenerated from corrupted NWB; do not use for regression"
    }
  }
}
```

**Step 4: Update `tests/baselines/README.md`** to reference the manifest and explain that regression tests must check the manifest status before comparing.

**Step 5: Commit**

```bash
git add tests/baselines/ .gitignore
git commit -m "chore(tests): pin baselines to git SHA + data hash; quarantine figure03

Baselines previously recorded only a calendar date, so 'passes baseline'
meant 'matches the same wrong answer as last time'. Add manifest.json
with git_sha + data_md5; mark figure03 as quarantined (generated from
corrupted NWB). Remove committed .log artifacts containing DB host and
username."
```

---

### Task 20: De-hollow the integration tests (C-T1)

**Files:**
- Modify: `tests/integration/test_data_loading.py`

**Step 1: Decide the approach**

Two options:
- (a) Rename to `test_mock_session_fixture.py` and mark `@pytest.mark.skip(reason="requires Spyglass DB")` for any tests that should exercise real loading.
- (b) Mock DataJoint at the query boundary (not the returned DataFrames), so the loader's column-naming and time-alignment logic is actually exercised.

Recommend (a) for now (fast), and file a separate TODO for (b).

**Step 2: Apply the chosen option**

For (a): rename file, mark every test class, update `TestDataLoadingIntegration` class name to `TestMockSessionFixture`, and add at the top of the file a comment explaining that real integration tests are gated on DB access.

**Step 3: Run pytest to make sure the renamed tests still pass / skip correctly**

Run: `uv run pytest tests/integration/ -v`

**Step 4: Commit**

---

## Wave 6 — Medium-risk test coverage

### Task 21: `tests/unit/test_rate_model.py`

Seed-based tests verifying `fit_per_stop_rate_model` recovers known rates with the `log(immobility_time)` offset, on simulated stop tables.

### Task 22: Extend `tests/unit/test_choice_prediction.py`

Add beyond the one test from Task 10: `predict_arm_fractions` residual has the right sign, `build_residualized_choice_dataset` preserves stop_id ordering, etc.

Each task: red → green → commit pattern.

---

## Wave 7 — Scripts & visualization cleanup

### Task 23: De-duplicate `_decoding_common.py` infrastructure (M-SV1)

**Files:** `scripts/run_pfc_analysis_bandit.py:73-144`, `scripts/_decoding_common.py`

Replace the reimplemented `SkipChecksumFilter`, `setup_logger`, `results_exist` with imports. Add a small `pfc_results_exist` helper if the required-file set differs.

### Task 24: Fail-loud Dask pipeline (M-SV3)

**Files:** `scripts/run_decoding_bandit.py:392-393`

Collect failed identifiers into a shared list; print a summary line with failure count; exit non-zero if any failures. Do **not** re-raise inside `run_decode` (Dask will then fail-fast the whole run, which is worse).

### Task 25: `save_figure(None, ...)` guard

**Files:** `src/continuum_swr_replay/visualization/figure_panels.py:{103,256,316,378,445}`

Add `if figure_name is not None:` wrappers around each `save_figure(figure_name, ...)` call in the standalone branches.

### Task 26: Centralize figure color constants

**Files:** `scripts/generate_figure04.py:281`, `scripts/generate_replay_content_figures.py:132`, `src/continuum_swr_replay/visualization/replay_content_plots.py:54`

Replace hardcoded `"#e6ab02"` / `"#1b9e77"` with `COLORS["Non-Local"]` etc., or add a new named entry to `parameters.COLORS`.

---

## Wave 8 — Quality, tooling, cleanup

### Task 27: Pin `non_local_detector` and `spyglass-neuro`

**Files:** `pyproject.toml:28-29`

Replace `git+https://…git` with `git+https://…git@<sha-or-tag>`. Use `git ls-remote` on each repo to find the current HEAD commit. Rerun `uv sync` to regenerate the lockfile. Commit both files.

### Task 28: Remove `F401` from `bandit_task.py` ignore

**Files:** `pyproject.toml:86`

Restrict to `["E402"]` only. Add `# noqa: F401` on the specific TYPE_CHECKING import lines. Run `uv run ruff check src/` to confirm no new warnings.

### Task 29: Remove stale `test_parallel_loading.pyc`

**Files:** `tests/integration/__pycache__/test_parallel_loading.*.pyc`

```bash
find tests/integration -name "test_parallel_loading*.pyc" -delete
```

Commit after confirming `pytest --collect-only` is clean.

### Task 30: Fix `conftest.py::bandit_track_graph` topology (M-T3)

**Files:** `tests/conftest.py:41-68`

Either rename to `simplified_star_track` and leave topology alone, **or** replace with the canonical 10-node / 9-edge spider matching `TRACK_SEGMENT_TO_PATCH`. Recommend the rename path — many tests depend on the simplified fixture for reasons unrelated to topology.

If renaming, run `grep -rn bandit_track_graph tests/` and update every consumer.

### Task 31: `pyproject.toml` description

**Files:** `pyproject.toml:4`

Replace `"Add your description here"` with a one-sentence scientific description, e.g.:
> `"Analysis of continuum hippocampal replay during Sharp Wave Ripples in a spatial bandit task."`

### Task 32: Remove dead `plot_swr_nl_venn_2circle`

**Files:** `src/continuum_swr_replay/visualization/figure_panels.py:1024`

Delete the function — commit `6d87d5c` already removed the Venn diagram from Figure 3.

---

## Out of scope / explicitly skipped

- **Original C-HO1 (`get_overlap_times` index bug)** — the verifier confirmed `get_labels` re-enumerates from 1, so `label - 1` correctly maps to positional index. No change needed.
- **M-R2 (switch_analysis session filter)** — verifier downgraded to Minor; `stop_id` hash already prevents data corruption. File as a TODO for performance only.
- **M-SV5 (hardcoded wilbur session for Fig 2a/3a/4a)** — requires scientific judgment from the author about candidate sessions; open a separate issue with the author.
- **Original C-HO2 downgraded to M** — now lives as Wave 4 extension; the within-session theta z-score is a QC-only artifact since no main figure uses it. If a main figure starts using it, upgrade back to Critical.

---

## Verification at the end of each wave

After every wave, run:

```bash
uv run ruff check src/ scripts/ tests/
uv run mypy src/
uv run pytest -x
```

All three must be clean before moving to the next wave. If pytest is slow, run `uv run pytest -x -m "not slow and not gpu"` during development and the full suite before merging.

## Branch / PR strategy

- One branch per wave. Wave 1 ships as a PR even before Wave 2 is written — it's all surface fixes.
- Waves 3, 5 should each be their own PR (scientific arithmetic and test coverage; reviewers want to look at these carefully).
- Waves 4, 6, 7, 8 can be bundled if the diff is small.

Never combine a Critical fix with a Minor cleanup in the same PR — review ergonomics win.
