# Shared contracts

[← back to PLAN.md](PLAN.md)

- [Reported events](#reported-events)
- [Truth labels](#truth-labels)
- [Detector outputs](#detector-outputs)
- [Summary CSV layouts](#summary-csv-layouts)

## Reported events

Defined in `scripts/qa/sweep_memoryless_detection.py`; the controls and held-out scripts
import them.

```python
@dataclass(frozen=True)
class ReportedEvent:
    start_s: float              # first window midpoint / first 2 ms bin
    end_s: float                # last window midpoint / last 2 ms bin
    decoded_time_s: np.ndarray  # when each decoded position was read
    decoded_cm: np.ndarray      # the decoded positions (kept windows / non-local MAP)

@dataclass(frozen=True)
class SlidingWindowRun:
    time: np.ndarray            # the likelihood's grid (one row shorter than the input)
    midpoint_s: np.ndarray      # (n_windows,)
    posterior: np.ndarray       # (n_windows, n_bins), the content estimate for held-out
    candidates: CandidateSequences
    accepted: list[int]         # candidate indices meeting duration and dispersion
    events: dict[int, ReportedEvent]      # one per accepted candidate
    shuffles: dict[int, ShuffleResult]    # one per accepted candidate

@dataclass(frozen=True)
class StateSpaceRun:
    time: np.ndarray
    p_continuous: np.ndarray    # (n_time,) Non-Local Continuous
    p_fragmented: np.ndarray    # (n_time,) Non-Local Fragmented
    canonical: list[ReportedEvent]        # continuous + fragmented > 0.8 for >= 15 ms
    continuous_only: list[ReportedEvent]  # continuous alone > 0.8 for >= 15 ms
    non_local_posterior: np.ndarray       # (n_time, n_bins) p(x | non-local), rows sum to 1
```

Invariants (do not weaken): an event's decoded positions are the ones the existing
scoring reads (sliding window: kept windows' bin centers at their midpoints; state-space:
the non-local MAP position per bin). `accepted` is `np.flatnonzero(candidates.is_replay)`,
every accepted candidate in the decoded window, and the shuffle runs on each of them;
the injected-replay scoring then keeps those overlapping the injection. A candidate's
shuffle is seeded by the cell key and its first window, so shuffling candidates away
from the injection does not change the overlapping ones' results.

## Truth labels

Strings, one per injected condition or scored event (design C2a):

| label | non-local representation | continuous replay |
|---|---|---|
| `continuous_replay` | yes | yes |
| `non_local` | yes | no |
| `local` | no | no |
| `no_content` | no | no |

Held-out *false support* is support on `local` or `no_content`.

## Detector outputs

The four scored outputs, as column/`detector` values everywhere:
`sliding_accepted`, `sliding_published`, `state_space_canonical`,
`state_space_continuous`.

## Summary CSV layouts

All long format, with provenance rows (`summary == "provenance"`, `measure` in
`produced_by`, `produced_at`, `git_commit`, `n_*`; value NaN, the text in `note`).

- `detection_sweep_summary.csv` (existing): columns
  `summary, measure, path_cm, duration_s, distance_bin_cm, value, note`. Phase 1 adds
  `summary == "timing_by_duration"` rows with `measure` in
  `{sliding,state_space}_{start,end}_error_ms_{q25,median,q75}` and `duration_s` set.
- `detection_controls_summary.csv` (phase 3): columns
  `summary, measure, condition, level, detector, value, note`. `condition` in
  `no_replay, placeless_burst, jumping_burst, jump, chain_continuing, chain_elsewhere,
  isolated_part`; `level` is the grid value (burst or part duration in s, jump in cm,
  gap in s; NaN for no replay).
- `detection_heldout_summary.csv` (phase 4): columns
  `summary, measure, fold, estimator, event_class, truth, value, note`. `fold` in
  `A, B, C`; `estimator` in `state_space, sliding_window`; `event_class` in
  `shared, ours_only, comparator_only`; `truth` a [truth label](#truth-labels).
