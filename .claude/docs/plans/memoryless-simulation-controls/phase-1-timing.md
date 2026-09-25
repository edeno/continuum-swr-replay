# Phase 1 — Split the sweep into run and score; onset and offset timing

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md#reported-events)

**Inputs to read first:**

- `scripts/qa/sweep_memoryless_detection.py:242-390` — `memoryless_outcome`, to split.
- `scripts/qa/sweep_memoryless_detection.py:405-516` — `non_local_intervals` and
  `state_space_outcome`, to split.
- `scripts/qa/sweep_memoryless_detection.py:620-858` — `summarize`; the `found` frame at
  its top is the paired population the timing rows use.
- `scripts/figures/generate_memoryless_comparison.py:851-935` — `plot_duration_bias`
  (removed) and `plot_absorption` (kept, becomes panel j).
- `scripts/figures/generate_memoryless_comparison.py:975-1044` — `make_figure`.
- `scripts/figures/generate_memoryless_comparison.py:1064-1340` — `check_summary_claims`,
  `write_caption`.

**Contracts referenced:** [Reported events](shared-contracts.md#reported-events),
[Summary CSV layouts](shared-contracts.md#summary-csv-layouts).

## Tasks

- **Capture the baseline**: copy `docs/design/replay_detection_comparison/detection_sweep_summary.csv`
  aside before any change.
- **Split the sliding window** into `run_sliding_window(recording, injected, window,
  cell_key) -> SlidingWindowRun` (decode, detect, shuffle every accepted candidate,
  build one `ReportedEvent` per accepted candidate from its kept windows) and
  `score_sliding_window(run, bounds, true_position_cm, path_cm, position) -> dict`
  holding the current column logic. `memoryless_outcome` becomes the composition.
- **Split the state-space detector** into `run_state_space(recording, injected, window)
  -> StateSpaceRun` (predict, both probabilities, canonical and continuous-only
  intervals via `non_local_intervals`, non-local posterior renormalized) and
  `score_state_space(run, bounds, true_position_cm, path_cm) -> dict`.
  `state_space_outcome` becomes the composition. The existing
  `RuntimeError` on missing probability mass stays in the run step.
- **Verify the refactor**: rerun the sweep; the new summary must equal the baseline in
  every non-provenance row.
- **Timing rows** in `summarize`: from the paired `found` frame, per `duration_s`, the
  25th percentile, median and 75th percentile of `sliding_start_error_s`,
  `sliding_end_error_s`, `state_space_start_error_s`, `state_space_end_error_s`, in ms,
  as `summary == "timing_by_duration"`.
- **Figure row 4**: replace `plot_duration_bias` with `plot_onset(ax, summary)` and
  `plot_offset(ax, summary)` (median line with interquartile band per detector, zero
  line, x = true duration, y = ms); `plot_absorption` moves to the third panel. The
  grid becomes 12 columns so row 4 splits into three equal panels (`grid[3, 0:4]`,
  `grid[3, 4:8]`, `grid[3, 8:12]`); earlier rows keep their widths by doubling their
  column spans. Remove `plot_duration_bias`.
- **Caption**: replace the (h) paragraph with (h, i): signed onset and offset medians,
  the sliding window's bounds being window midpoints whose 80 ms windows reach 40 ms
  past each bound, and the 40 ms example restated as onset plus offset. Rename the old
  (i) to (j). Add a `check_summary_claims` assertion for each factual sentence (for
  example, that the sliding window's median onset is earlier than the state-space's at
  every duration, if the text says so).
- **Spec**: in spec 02's sweep section, replace the reported-duration paragraph's
  framing with the onset/offset rows (keep the duration-ratio table, which the summary
  still carries).

## Deliberately not in this phase

- New injected conditions (phase 3) and any new scoring function in `scoring.py`
  (phase 2).
- `build_recording(cells=...)` (phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| Baseline diff (script) | Every non-provenance row of the regenerated summary equals the baseline |
| `compare_memoryless_to_reference.py --n-seeds 1` (slow, manual) | Still runs against the split functions |
| Figure build | Builds; caption checks pass; PNG inspected |
| Full unit + integration suite | Passes, including the pipeline map |

## Fixtures

The sweep's own simulated recording; no new fixtures.

## Review

Before merging, dispatch `code-reviewer` (read-only) against the diff. Confirm:
- Every task is implemented as specified; the summary is unchanged apart from the new rows.
- The "Deliberately not in this phase" list is honored.
- Tests pass; no test or docstring references this plan.
- `plot_duration_bias` is removed, not left unused.
