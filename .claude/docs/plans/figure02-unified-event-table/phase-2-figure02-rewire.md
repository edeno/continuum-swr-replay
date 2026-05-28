# Phase 2 — Rewire Figure 2 to consume the classified event table

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/hippocampus/event_table.py` — the module produced by [phase-1](phase-1-classified-event-table.md). Read its docstring and the column documentation; this phase consumes that contract.
- [scripts/generate_figure02.py:102-269](../../../../scripts/generate_figure02.py#L102-L269) — orchestrator. Panels a/b/d rewire here; panel c stays on `get_event_types` with a parity test.
- [src/continuum_swr_replay/analysis/hippocampus/event_analysis.py:232-365](../../../../src/continuum_swr_replay/analysis/hippocampus/event_analysis.py#L232-L365) — `compute_event_timing_offsets`. Deleted in this phase.
- [src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:540-609](../../../../src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py#L540-L609) — `compute_swr_overlap_class_counts`. Reduces to a table query.
- [src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:420-537](../../../../src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py#L420-L537) — `get_event_types`. Left in place; phase 2 adds a parity test.
- [src/continuum_swr_replay/analysis/hippocampus/event_detection.py:512-572](../../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L512-L572) — `select_typical_event_by_class`. Reused; its input shape (a frame with `overlap_class`, `duration`, and a way to compute `peak_prob`) is satisfied by the new table.
- [src/continuum_swr_replay/visualization/figure_panels.py:109-260](../../../../src/continuum_swr_replay/visualization/figure_panels.py#L109-L260) — `plot_triggered_averages`. Untouched; phase 2 verifies that its event-types-driven input is consistent with the new table.
- [src/continuum_swr_replay/visualization/figure_panels.py:933-1025](../../../../src/continuum_swr_replay/visualization/figure_panels.py#L933-L1025) — `plot_event_timing_offsets`. Untouched; phase 2 changes only what's passed in.

**Contracts referenced:** the `build_classified_nonlocal_event_table` column set from phase 1.

## Tasks

### 1. Build the per-session classified table once and feed panels a/b/d

After phase 1, the cached `event_table.pkl` carries classified columns for the replay-content population, but it is filtered by `_in_event_mask` and keeps legacy replay timing in `event_start` / `event_end`. Figure 2 needs the unfiltered classified table with full label-window timing, so **compute the classified table once at the top of `generate_figure` from the aggregated `data_over_time` frame** and feed it to panels a, b, and d. Panel c continues to use `get_event_types` (with a parity test against the table — see Task 5). Do not load `event_table.pkl` for Figure 2 counts or examples; its population and timing contract are replay-specific.

Add at the top of `generate_figure` in [scripts/generate_figure02.py:142](../../../../scripts/generate_figure02.py#L142):

```python
from continuum_swr_replay.analysis.hippocampus.event_table import (
    build_classified_nonlocal_event_table,
)

# After load_data_over_time(...)
classified_events_all = build_classified_nonlocal_event_table(data_over_time)
```

This is computed once (over the full multi-session aggregate) and sliced by session for panel a. Cost: one groupby per session inside the builder, all sessions in one pass — same order of magnitude as the prepare-pipeline cost that already runs offline. No new caching needed.

Replace `pick_example_events` at [scripts/generate_figure02.py:102-129](../../../../scripts/generate_figure02.py#L102-L129):

```python
def pick_example_events(
    classified_events_all: pd.DataFrame,
    nwb_file_name: str,
    epoch_name: str,
    peak_threshold: float = 0.9,
) -> dict:
    """Deterministic three-example selection for panel a.

    Filters the pre-built classified table to the example session,
    restricts to immobile (endpoints rule, to match panel b) and
    peak_nonlocal_prob >= ``peak_threshold``, then runs
    :func:`select_typical_event_by_class` per overlap class.
    """
    animal, day = parse_nwb_filename(nwb_file_name)
    session = classified_events_all[
        (classified_events_all["animal"] == animal)
        & (classified_events_all["day"] == day)
        & (classified_events_all["epoch"] == epoch_name)
        & classified_events_all["endpoints_speed_below_thresh"]
    ]
    # select_typical_event_by_class expects `overlap_class`, `duration`,
    # and (after Task 1a below) a `peak_prob` column. Rename in-place.
    session = session.rename(columns={
        "peak_nonlocal_prob": "peak_prob",
        "event_duration": "duration",
        "event_start": "start_time",
        "event_end": "end_time",
    })
    return select_typical_event_by_class(
        session, peak_threshold=peak_threshold,
    )
```

Import `parse_nwb_filename` from `continuum_swr_replay.data_loaders.utils`; the project already uses that helper for robust NWB filename parsing.

The output dict shape (one `pd.Series` per class) is unchanged. Downstream consumers in `generate_figure` keep working except that `event` field names are `start_time`/`end_time`/`duration` (already true after the rename above).

### 1a. Modify `select_typical_event_by_class` signature

Current signature at [event_detection.py:512-516](../../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L512-L516):

```python
def select_typical_event_by_class(
    classified_events: pd.DataFrame,
    non_local_prob: xr.DataArray,
    peak_threshold: float = 0.9,
) -> dict[str, pd.Series]:
```

Change to:

```python
def select_typical_event_by_class(
    classified_events: pd.DataFrame,
    non_local_prob: xr.DataArray | None = None,
    peak_threshold: float = 0.9,
) -> dict[str, pd.Series]:
    """...

    If ``non_local_prob`` is None, ``classified_events`` must contain a
    ``peak_prob`` column (precomputed by upstream code). If
    ``non_local_prob`` is provided, ``peak_prob`` is computed per event
    from ``classified_events[["start_time", "end_time"]]`` against the
    array (the legacy behavior, preserved for the original
    one-session-at-a-time call site).

    Raises
    ------
    ValueError
        If both ``non_local_prob`` is None and ``classified_events``
        lacks a ``peak_prob`` column.
    """
```

Implementation: at the top of the function, branch on `"peak_prob" in classified_events.columns`. If present, skip the `peak_prob` recomputation block at [event_detection.py:561-564](../../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L561-L564) and use the column directly. If absent, require `non_local_prob is not None` and compute as today. Add one test under `tests/analysis/hippocampus/test_event_classification.py` that calls the function with a pre-populated `peak_prob` column and verifies the result is identical to calling it with the matching `non_local_prob` xarray.

The new orchestrator call passes only the precomputed-column path; the old single-session callers (if any) keep working unchanged.

### 3. Replace `compute_swr_overlap_class_counts`

In [data_aggregation.py:540-609](../../../../src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py#L540-L609), replace the body with a one-liner over the classified table:

```python
def compute_swr_overlap_class_counts(
    classified_events: pd.DataFrame,
) -> dict[str, int]:
    """Count immobile non-local events by SWR-overlap class.

    Operates on the output of
    :func:`build_classified_nonlocal_event_table`, restricted to
    immobile events (``endpoints_speed_below_thresh``).
    """
    zero = {"swr_contained": 0, "swr_partial": 0, "non_swr": 0, "total": 0}
    if classified_events.empty:
        return zero
    # ``.loc`` + ``astype(bool)`` so an empty/object-typed mask column
    # (the builder's zero-row frame leaves it object-dtype) still indexes
    # cleanly and preserves the column set.
    immobile = classified_events.loc[
        classified_events["endpoints_speed_below_thresh"].astype(bool)
    ]
    if immobile.empty:
        return zero
    counts = immobile["overlap_class"].value_counts().to_dict()
    return {
        "swr_contained": int(counts.get("swr_contained", 0)),
        "swr_partial": int(counts.get("swr_partial", 0)),
        "non_swr": int(counts.get("non_swr", 0)),
        "total": int(len(immobile)),
    }
```

**Why the empty guard is mandatory, not defensive padding.** The phase-1 contract allows `build_classified_nonlocal_event_table` to return `pd.DataFrame(columns=[...])` with zero rows (phase-1 task 1, "Empty frame (correct columns, zero rows) if no events"). On such a frame the `endpoints_speed_below_thresh` column is object-dtype, and boolean indexing through `classified_events[<object Series>]` is fragile — guard with an explicit `.empty` check and coerce the mask with `.astype(bool)` via `.loc`. Add a unit test `test_swr_overlap_class_counts_empty` that passes `build_classified_nonlocal_event_table` of an event-free `dot` (or a bare `pd.DataFrame(columns=...)`) and asserts the all-zeros dict — this both proves the guard and locks the contract.

The function's call site in [generate_figure02.py:167](../../../../scripts/generate_figure02.py#L167) changes from `compute_swr_overlap_class_counts(data_over_time)` to `compute_swr_overlap_class_counts(classified_events_all)`, where `classified_events_all` is built once from the aggregated `data_over_time` at the top of `generate_figure`.

Delete `_intervals_from_labels` from `data_aggregation.py` if it has no remaining callers after this change (grep the codebase). The public helper lives in `analysis/hippocampus/event_table.py` as `intervals_from_labels`.

**Rewrite the existing consistency test — the signature change breaks it.** [tests/unit/test_swr_overlap_consistency.py](../../../../tests/unit/test_swr_overlap_consistency.py) calls `compute_swr_overlap_class_counts(dot)` with a raw `data_over_time`-shaped frame at [test_swr_overlap_consistency.py:86](../../../../tests/unit/test_swr_overlap_consistency.py#L86) (inside `_counts_from_timebins`). After the signature change it must be passed a *classified table*, not raw labels, or the "all tests pass" gate fails. The test's original purpose — proving the event-level classifier (`classify_swr_overlap`) and the timebin-derived counts agree — partly collapses, because `compute_swr_overlap_class_counts` now *is* a thin `value_counts` over `build_classified_nonlocal_event_table`'s `overlap_class`, which itself calls `classify_swr_overlap_geometry`. Two options:

- **Preferred — rewrite to exercise the new contract.** Change `_counts_from_timebins` to build the classified table (`build_classified_nonlocal_event_table(dot)`) and pass it to `compute_swr_overlap_class_counts`. Keep all the parametrized geometry cases ([test_swr_overlap_consistency.py:90-119](../../../../tests/unit/test_swr_overlap_consistency.py#L90-L119)); they now assert that the table-driven counts match the direct `classify_swr_overlap` counts. The existing fixture must also be extended to satisfy `build_classified_nonlocal_event_table`'s required columns: add `Non-Local_Continuous_prob` and `Non-Local_Fragmented_prob` to `_build_data_over_time`, for example `Non-Local_Continuous_prob = (non_local_event > 0).astype(float)` and `Non-Local_Fragmented_prob = np.zeros(len(time))`. Note that the fixture builds `dot` with `head_speed = np.zeros(...)` ([test_swr_overlap_consistency.py:67](../../../../tests/unit/test_swr_overlap_consistency.py#L67)), so every event passes `endpoints_speed_below_thresh` and the immobility filter inside the new `compute_swr_overlap_class_counts` is a no-op for this fixture — the counts stay comparable.
- **Alternative — remove it** and rely on the phase-1 classification tests (`test_classified_event_table.py`) plus the phase-2 `get_event_types` parity test (Task 5) for coverage. Only do this if the rewrite above is genuinely redundant with those; otherwise prefer the rewrite so the geometry cases keep a home.

Pick the rewrite. Record the choice in the PR description.

### 4. Delete `compute_event_timing_offsets`

In [event_analysis.py:232-365](../../../../src/continuum_swr_replay/analysis/hippocampus/event_analysis.py#L232-L365), delete the function entirely.

**Remove its public-API export entries too — this is mandatory, not optional.** `compute_event_timing_offsets` is exposed through the package's lazy-export machinery in two places in [analysis/hippocampus/__init__.py](../../../../src/continuum_swr_replay/analysis/hippocampus/__init__.py): the `_LAZY_EXPORTS` dict at [__init__.py:83](../../../../src/continuum_swr_replay/analysis/hippocampus/__init__.py#L83) and the mirrored `TYPE_CHECKING` import at [__init__.py:147](../../../../src/continuum_swr_replay/analysis/hippocampus/__init__.py#L147). If you delete the function but leave these entries, `from continuum_swr_replay.analysis.hippocampus import compute_event_timing_offsets` still *resolves* to a name in `__all__`, then fails at runtime inside `__getattr__` ([__init__.py:125-132](../../../../src/continuum_swr_replay/analysis/hippocampus/__init__.py#L125-L132)) with `AttributeError` when `getattr(module, name)` can't find the deleted symbol — a stale public-API path. Delete both entries. (`__all__` is derived from `_LAZY_EXPORTS.keys()`, so removing the dict entry drops it from `__all__` automatically.)

**Grep for other importers before deleting.** Run `grep -rn "compute_event_timing_offsets" src/ scripts/ tests/` and confirm the only references are (a) the definition, (b) the two `__init__.py` export lines, (c) the `generate_figure02.py` call site replaced below, and (d) the test deleted in the paragraph at the end of this task. If any other caller exists, surface it before proceeding.

Replace its call site in [generate_figure02.py:175-181](../../../../scripts/generate_figure02.py#L175-L181) with:

```python
overlap_pairs = classified_events_all[
    classified_events_all["endpoints_speed_below_thresh"]
    & (classified_events_all["overlap_class"] != "non_swr")
].copy()
overlap_pairs["start_diff"] = (
    overlap_pairs["event_start"] - overlap_pairs["matched_swr_start"]
)
overlap_pairs["end_diff"] = (
    overlap_pairs["event_end"] - overlap_pairs["matched_swr_end"]
)
offsets = overlap_pairs[["animal", "day", "epoch", "start_diff", "end_diff"]]
```

The shape passed to `plot_event_timing_offsets` is the same (`start_diff`, `end_diff` columns), so the plotting helper's *body* at [figure_panels.py:933-1025](../../../../src/continuum_swr_replay/visualization/figure_panels.py#L933-L1025) is unchanged.

**But update its docstring — it references the deleted function.** `plot_event_timing_offsets`'s docstring at [figure_panels.py:942-943](../../../../src/continuum_swr_replay/visualization/figure_panels.py#L942-L943) says it takes "the per-overlap `start_diff` and `end_diff` DataFrame from `:func:`compute_event_timing_offsets`". After Task 4 deletes that function, this `:func:` cross-reference points at nothing (and would break a Sphinx build if docs are generated). Replace the reference with a description of the new source: the `start_diff`/`end_diff` columns are now computed inline in `generate_figure` by subtracting `matched_swr_start`/`matched_swr_end` from the event's `event_start`/`event_end` (drawn from `build_classified_nonlocal_event_table`'s columns), restricted to immobile, SWR-overlapping events. Keep the rest of the docstring (axis semantics, parameters) as is.

Note: this is **two real population changes** for panel d, both flowing from the unification:

1. **Immobility filter.** The old function counted every overlapping `(SWR, NL)` pair regardless of speed; the new code uses only immobile NL events. Aligns panel d with panels b/c (overview.md open question 2).
2. **One row per overlapping event, not per `(SWR, NL)` pair.** The classified table's `matched_swr_*` picks one SWR per event (containing SWR for `swr_contained`; earliest overlapping SWR for `swr_partial`). The old `compute_event_timing_offsets` cross-joined every overlapping pair, so events in `overlap_subclass in {"multi_swr_no_gap", "multi_swr_gap"}` contributed multiple rows. After the rewire, each event contributes exactly one row. Multi-SWR overlap subclasses are rare (SWR detectors enforce minimum inter-event separation), so the histogram change is small but not zero — the diff in Task 9 quantifies it.

Both changes go into the panel-d caption update (Task 8) and the PR-description before/after table (Task 9).

Delete the existing test [tests/unit/test_event_timing_offsets_empty.py](../../../../tests/unit/test_event_timing_offsets_empty.py) since the function it tests is gone. If the empty-frame behavior is still worth testing for the new column-subtraction path, add an equivalent test under `tests/analysis/hippocampus/test_classified_event_table.py` asserting `classified_events_all` with no overlapping events returns the expected schema after the filter.

### 5. Verify panel c is consistent

Panel c calls `plot_triggered_averages` with `event_types["non_local_event_immobile_overlap_SWR_event"]` and `..._no_overlap_SWR_event"` — these are produced by `get_event_types` ([data_aggregation.py:420-537](../../../../src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py#L420-L537)), which uses `event_strict_overlap_keys` (strict positive-duration). After phase 1, the new `overlap_class != "non_swr"` rule is also strict positive-duration. **These should agree row-for-row.**

Add a parity test:

```python
# tests/analysis/hippocampus/test_classified_event_table.py
def test_event_types_matches_classified_overlap(dot_fixture):
    """get_event_types overlap split must equal classified-table is_swr."""
    classified = build_classified_nonlocal_event_table(dot_fixture)
    event_types = get_event_types(dot_fixture, event_col="non_local_event")
    overlap_keys = set(
        event_types["non_local_event_immobile_overlap_SWR_event"]
        .reset_index()
        .apply(lambda r: (r["animal"], r["day"], r["epoch"], r["non_local_event"]), axis=1)
    )
    classified_overlap_keys = set(
        classified[
            classified["endpoints_speed_below_thresh"]
            & (classified["overlap_class"] != "non_swr")
        ]
        .apply(lambda r: (r["animal"], r["day"], r["epoch"], r["non_local_event"]), axis=1)
    )
    assert overlap_keys == classified_overlap_keys, (
        f"Drift: event_types-only={overlap_keys - classified_overlap_keys}, "
        f"classified-only={classified_overlap_keys - overlap_keys}"
    )
```

If this test fails, two helpers that should compute the same thing disagree — investigate the source of drift before merging the phase. Otherwise, panel c stays on `get_event_types` and the rewire for panel c is a no-op beyond the parity test.

### 6. Add the drawn-vs-classified consistency assertion in `generate_figure`

**Why a naive class assertion would be tautological.** After Task 1, `pick_example_events` selects each example by filtering `classified_events_all` to `overlap_class == cls`. So `chosen[cls]["overlap_class"] == cls` *by construction* — re-checking it against any column of `classified_events_all` (the same source) can never fail. The figure now has a **single** classification source (label-derived `SWR_event` intervals), so there is no longer a "panel a path vs panel b path" to drift apart. Asserting on it is pointless.

**What is still a real, reader-visible risk.** Panel a's top row (`plot_ripple_consensus_trace_zscore`, [plots.py:606-625](../../../../src/continuum_swr_replay/visualization/plots.py#L606-L625)) **shades the SWR-active span from `data["ripple_times"]`** — the *threshold-crossing* ripple intervals — via `_intervals_to_mask`. But the column header (`SWR-contained` / `SWR-partial` / `non-SWR`) comes from the *label-derived* classification in `classified_events_all`. These are two discretizations of the same ripples and can disagree by ±1 sample at a boundary. If they disagree for a chosen example, the reader sees a shaded SWR span that contradicts the column label. That is the failure mode worth guarding — and it crosses a genuine boundary (`ripple_times` vs labels).

**Add the pandas import.** `scripts/generate_figure02.py` does not currently import pandas (see its import block at [generate_figure02.py:42-51](../../../../scripts/generate_figure02.py#L42-L51)). The snippet below uses `pd.DataFrame`, so add `import pandas as pd` to the module's imports.

In [generate_figure02.py:154-164](../../../../scripts/generate_figure02.py#L154-L164), after `pick_example_events` returns and the missing-class check passes, add:

```python
from continuum_swr_replay.analysis.hippocampus import classify_swr_overlap

# Cross-check the LABEL-derived class (what the column header says) against
# the THRESHOLD-CROSSING ripple_times (what panel a actually shades). These
# are independent discretizations of the same SWRs; disagreement means the
# drawn SWR span would contradict the column label for the reader.
chosen_df = pd.DataFrame(
    [
        {
            "start_time": ev["start_time"],
            "end_time": ev["end_time"],
            "expected_class": cls,
        }
        for cls, ev in chosen.items()
    ]
)
drawn = classify_swr_overlap(chosen_df, data["ripple_times"])
mismatch = drawn[drawn["overlap_class"] != drawn["expected_class"]]
if not mismatch.empty:
    rows = mismatch[
        ["expected_class", "overlap_class", "start_time", "end_time"]
    ]
    raise AssertionError(
        "Panel-a example(s) classify differently against the threshold-"
        "crossing ripple_times that panel a shades than the label-derived "
        "column header:\n"
        f"{rows.to_string(index=False)}\n"
        "The drawn SWR span would contradict the column header. Pick a "
        "different example for the affected class(es), or widen the "
        "selection's peak/duration criteria so a non-boundary event is "
        "chosen."
    )
```

Two snippet pitfalls deliberately avoided (both would otherwise fail at runtime):

1. **No `itertuples`.** A column named `_col` (leading underscore) is not a valid Python identifier, so `DataFrame.itertuples()` silently renames it to a positional `_1`, and `row._col` raises `AttributeError`. The comparison here is vectorized (`drawn["overlap_class"] != drawn["expected_class"]`), and the label column is named `expected_class` — a valid identifier — so neither trap applies.
2. **`classify_swr_overlap` preserves extra columns.** It does `out = non_local_events.copy(); out["overlap_class"] = classes` ([event_detection.py:334-336](../../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L334-L336)), so the `expected_class` column survives into `drawn`. It reads `start_time`/`end_time` from the input — both valid identifiers, present after the Task-1 rename. `data["ripple_times"]` is the threshold-crossing frame.

**Also keep a cheap hash-collision guard.** Independently, confirm each chosen `event_id` resolves to exactly one row in `classified_events_all` (guards against a `hash_event_id` collision silently aliasing two events):

```python
for cls, ev in chosen.items():
    n = (classified_events_all["event_id"] == ev["event_id"]).sum()
    if n != 1:
        raise AssertionError(
            f"event_id {ev['event_id']!r} (column {cls!r}) matches {n} rows "
            f"in classified_events_all (expected 1) — hash collision."
        )
```

**Verify the consistency assertion can fire.** Add a unit test under `tests/analysis/hippocampus/test_classified_event_table.py` that constructs a `chosen`-style frame and a `ripple_times` frame deliberately offset by one sample at the boundary (so the label-derived class is `swr_contained` but the threshold-crossing class is `swr_partial`), runs the assertion block (extract it into a small helper `_assert_drawn_matches_label(chosen, ripple_times)` so it's unit-testable), and confirms it raises `AssertionError`. This proves the assertion is not tautological.

### 7. Update orchestrator logging

In `generate_figure`, replace the existing logging with structured counts derived from the classified table:

```python
n_total = len(classified_events_all)
n_immobile = classified_events_all["endpoints_speed_below_thresh"].sum()
class_counts_log = (
    classified_events_all.loc[classified_events_all["endpoints_speed_below_thresh"]]
    ["overlap_class"].value_counts()
)
logging.info(
    "Classified %d non-local events; %d pass endpoints-immobility filter; "
    "by class: contained=%d, partial=%d, non_swr=%d",
    n_total, int(n_immobile),
    int(class_counts_log.get("swr_contained", 0)),
    int(class_counts_log.get("swr_partial", 0)),
    int(class_counts_log.get("non_swr", 0)),
)
```

Also: if `select_typical_event_by_class` falls back below the `peak_threshold=0.9` cut for any class (its current behavior at [event_detection.py:565-567](../../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L565-L567)), log a WARNING with the class name and the actual peak prob of the chosen event.

### 8. Documentation updates

- Update the docstring at the top of `scripts/generate_figure02.py` to describe the unified table flow (one paragraph).
- Update the docstrings of the three deleted/shrunken helpers (or their replacements) to point at `build_classified_nonlocal_event_table`.
- Update the panel-d figure caption to say:

  > "Per-overlap NL/SWR timing offsets. Restricted to immobile non-local events (endpoints below 4 cm/s). For events overlapping multiple SWRs (rare; `multi_swr_*` subclasses), the matched SWR is the containing SWR when one exists, otherwise the earliest overlapping SWR — one row per event, not one row per `(SWR, event)` pair."

  This documents both population changes — the immobility filter (was: all events; now: immobile only) AND the one-row-per-event matching (was: one row per overlapping `(SWR, event)` pair; now: one per event). Capture in the [paper/figures/figure02/](../../../../paper/figures/figure02/) directory's caption-source if one exists; otherwise note in the PR description so the manuscript-side caption gets updated when the figure is re-imported.

### 9. Regenerate Figure 2 and diff

```bash
uv run python scripts/generate_figure02.py
```

Open the new `figure02.pdf` and compare against the pre-phase-2 version. Document in the PR description:

- Panel a: example events may have changed if any flipped class under the new rule. Note which session/event-id was chosen per column and the peak prob.
- Panel b: counts may shift because the old helper derived SWR intervals from the already-immobile non-local rows, effectively clipping SWR intervals to the selected NL-event rows; the new table derives SWR intervals from the full session labels. Endpoint immobility itself should not be new for panel b. Report the before/after counts in each class and include the number of events whose class changed under full-session SWR intervals.
- Panel c: should be visually identical (Task 5 parity test guarantees same event population).
- Panel d: histograms shift because panel d gains the immobility filter. Report the before/after total overlapping-pair count.
- Panels e and f: untouched by this phase; should be bit-identical.

If any panel changes unexpectedly (e.g., panel c moves despite passing the parity test), **stop and investigate** before merging.

## Deliberately not in this phase

- **Restructuring `select_typical_event_by_class`** beyond letting it accept a precomputed `peak_prob` column. Its core selection logic (median duration within class, fall back to full subset) is unchanged.
- **Removing `get_event_types`.** It's still used by panel c and by other figures; phase 2 only verifies parity with the new table, doesn't replace.
- **Rerunning regression-fitting scripts** that consume `event_table.pkl`. Same out-of-scope as phase 1.
- **Updating Figure 3 panel a's example-selection logic** even though it shares `plot_event_example_stack`. The plotting helper is unchanged; the selector lives elsewhere and isn't part of Figure 2.
- **Pixel-diffing Figure 2.** Use the per-panel checks in Task 9 instead — pixel diffs are noisy from font-rendering and constrained-layout decisions.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/hippocampus/test_classified_event_table.py::test_event_types_matches_classified_overlap` | `get_event_types` overlap split equals `classified[overlap_class != "non_swr"]` for immobile events. |
| `tests/analysis/hippocampus/test_classified_event_table.py::test_compute_swr_overlap_class_counts` | New (table-driven) `compute_swr_overlap_class_counts` returns counts from `classified[ endpoints_speed_below_thresh ]["overlap_class"]` with zeros filled for missing classes. |
| `tests/analysis/hippocampus/test_classified_event_table.py::test_panel_b_full_swr_intervals_not_clipped_to_nl_rows` | Fixture where an SWR extends beyond an immobile NL event verifies the new count path uses full-session SWR intervals, not SWR bounds clipped to selected NL rows. |
| `tests/analysis/hippocampus/test_classified_event_table.py::test_timing_offsets_via_columns` | Column-subtraction (`event_start - matched_swr_start`) on a fixture matches the values the old `compute_event_timing_offsets` would have produced for overlapping pairs. |
| `tests/analysis/hippocampus/test_classified_event_table.py::test_timing_offsets_empty_session` | Session with no overlapping pairs returns the expected (empty) `(start_diff, end_diff)` schema. |
| Manual: regenerate `figure02.pdf` | All six panels render; logging shows expected event counts and class breakdown; example-class assertion does not fire. |
| Manual: PR description carries before/after panel-b counts and panel-d histogram totals. | Population deltas are documented and reviewed. |

## Fixtures

- Same synthetic `dot`-shaped fixture from phase 1; extend it with the `head_speed` patterns needed for the immobility-filter parity tests (boundary-cases where endpoints differ from max-during-event).
- Manual: the default real session `wilbur20210329_.nwb` epoch `02_r1` for the figure regeneration. Already covered by `generate_figure02.py`'s CLI defaults; no new fixture infrastructure needed.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep (specifically: `get_event_types` is not refactored; `select_typical_event_by_class`'s core logic is not rewritten; regression-fitting scripts are not rerun).
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — the parity test in Task 5 actually exercises both code paths on the same fixture and asserts set equality (not just `len(...) ==`).
- The example-class assertion in `generate_figure` fires on a deliberately-constructed bad input (verify by temporarily injecting a wrong class and confirming it raises).
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed: `compute_event_timing_offsets` is gone; `_intervals_from_labels` is gone (if no remaining callers); the inline cross-join in `compute_swr_overlap_class_counts` is gone.
- User-facing documentation listed as tasks is updated, not deferred: orchestrator docstring, helper docstrings, panel-d caption note in the PR description.
- The figure regen output is committed (or at least its panel-by-panel comparison is captured in the PR description).
