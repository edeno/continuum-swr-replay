# Designs

[← back to PLAN.md](PLAN.md)

- [General injector](#general-injector)
- [Builders](#builders)
- [Multi-part scoring](#multi-part-scoring)
- [Held-out rule](#held-out-rule)
- [Folds](#folds)

## General injector

`memoryless_sequence/simulation.py`. `inject_replay` (line 87) keeps its signature and
becomes a builder plus a call to:

```python
def inject_activity(
    spike_times: list[NDArray], activity: InjectedActivity, seed: int
) -> list[NDArray]:
    """Replace every cell's spikes in the activity's span with draws from its rates."""
    at = activity.time_s
    step_s = float(at[1] - at[0])
    rate = activity.rate_hz                      # (n_samples, n_neurons)
    if (rate * step_s).max(initial=0.0) >= 1.0: raise ValueError(...)  # as today
    fired = np.random.default_rng(seed).random(rate.shape) < rate * step_s
    out = []
    for neuron, times in enumerate(spike_times):
        kept = times[(times < at[0]) | (times > at[-1])]
        out.append(np.sort(np.concatenate([kept, at[fired[:, neuron]]])))
    return out

@dataclass(frozen=True)
class InjectedActivity:
    time_s: NDArray          # (n_samples,) evenly spaced
    position_cm: NDArray     # (n_samples,) represented place; NaN where silent or place-less
    rate_hz: NDArray         # (n_samples, n_neurons)
    parts: tuple[tuple[float, float], ...]   # (start_s, end_s) of each spatial part
    truth: str               # a truth label (shared-contracts)
```

The random draw must stay `default_rng(seed).random((n_samples, n_neurons))` with
`n_samples = max(2, int(duration_s * sampling_frequency))` and
`step_s = duration_s / (n_samples - 1)`, so `inject_replay` is bit-identical.

## Builders

All return `InjectedActivity`; rates follow `FIELD_STD_CM` Gaussians of
`place_field_means` at the represented position, zero where it is NaN.

- `field_rates(position_cm, place_field_means, peak_rate_hz)` — the shared rate map.
- `piecewise_replay(segments, at_s, sampling_frequency, place_field_means, peak_rate_hz)`
  — `segments` is a sequence of `(duration_s, start_cm, path_cm)` or `("gap", duration_s)`;
  constant speed within a segment; NaN position in a gap. A jump is two segments with
  no gap where the second's start is not the first's end; a chain is segment, gap,
  segment. `parts` are the segments' spans; truth `continuous_replay`.
- `placeless_burst(duration_s, at_s, ..., peak_rate_hz, track_cm)` — every cell at the
  rate `peak_rate_hz * sum_c g(track_cm / 2 - mu_c) / n_neurons`, the population rate
  of a representation at mid-track, so the population spike count matches a replay of
  the same peak rate. Position NaN; truth `no_content`; one part (its span).
- `jumping_burst(duration_s, at_s, ..., peak_rate_hz, track_cm, hold_s=0.015, seed)` —
  position drawn uniformly on `[0, track_cm]` and held for `hold_s`; truth `non_local`.

## Multi-part scoring

`memoryless_sequence/scoring.py`.

- `score_against_injection` gains NaN-in-gap support: the position error is the median
  over every decoded time where the true path is finite (outside the injection the
  truth is the path's nearer end, as before); an event needs at least one decoded time
  inside the injection where the truth is finite, or it does not match. `path_cm` for a multi-part event is the ground its parts
  cover together, `max - min` over every part's positions. Single-part behavior is
  unchanged (its truth is never NaN).
- `recovered_parts(event_bounds, matches, part_bounds) -> NDArray[bool]` — part `j` is
  recovered when some matching event overlaps at least half of its duration.
- `segmentation(event_bounds, matches, part_bounds) -> str`:

```python
def segmentation(event_bounds, matches, part_bounds) -> str:
    covering = [  # for each matching event, the parts it recovers
        {j for j, (a, b) in enumerate(part_bounds)
         if overlap(e, (a, b)) >= 0.5 * (b - a)}
        for e, m in zip(event_bounds, matches) if m
    ]
    n = len(part_bounds)
    if any(len(c) == n for c in covering): return "spanning"
    recovered = set().union(*covering) if covering else set()
    if len(recovered) < n: return "parts_missing"
    singles = [c for c in covering if c]
    if len(singles) == n and all(len(c) == 1 for c in singles): return "one_per_part"
    return "other"
```

- Onset/offset errors for multi-part events: per part when `one_per_part`; against the
  first part's start and last part's end when `spanning`; otherwise not reported.

## Held-out rule

`src/continuum_swr_replay/analysis/inventory_comparison/heldout.py`. Pure functions on
the 2 ms grid; fields are spikes per sample.

```python
def expected_log_likelihood(content, counts, fields) -> tuple[float, float]:
    """(spike term, silence term) of the evaluation spikes under a content posterior.

    content : (n_time, n_bins), rows sum to one
    counts  : (n_time, n_eval_cells) evaluation spikes per bin
    fields  : (n_eval_cells, n_bins) per-sample rates, floored
    spike   = sum_t sum_x p_t(x) sum_c k_tc log f_c(x) = sum(content * (counts @ log f))
    silence = -sum_t sum_x p_t(x) sum_c f_c(x)         = -sum(content @ f.sum(0))
    """

def heldout_score(content, counts, fields, animal_bin, permutations) -> HeldoutScore:
    # observed = spike + silence under content
    # animal null: content replaced by a point mass at the animal's bin each time bin
    # shuffle null: fields[perm] for each permutation (evaluation cells only)
    # gain_vs_animal = (observed - animal) / max(counts.sum(), 1)
    # p_shuffle = (1 + sum(null >= observed)) / (1 + n_permutations)
    # coverage = sum(content @ fields.sum(0))  (expected evaluation spikes)
    # supported = p_shuffle <= 0.05 and gain_vs_animal > 0
    # informative = coverage >= coverage_floor (5; 2 and 10 as sensitivity)
```

Evaluation fields get the reference's floor (`with_reference_rate_floor`), so a spike
from an evaluation cell nearly silent at a place costs a bounded log penalty: spec 05
requires the score to state how extreme values are handled, and an unfloored field
(clipped at 1e-15 per sample) would let one spike dominate an event.

Content estimators:

- `state_space`: the fold's detector posterior over the non-local states, summed over
  the two states per position and renormalized per time bin.
- `sliding_window`: the fold's sliding-window posterior (floored model, 80 ms windows),
  each 2 ms bin taking the window whose midpoint is nearest.

## Folds

Cells ordered by field position (`place_field_means`, 25 cells).

```python
def interleaved_folds(n_cells):   # A, B
    even, odd = np.arange(0, n_cells, 2), np.arange(1, n_cells, 2)
    return {"A": (odd, even), "B": (even, odd)}   # (discovery, evaluation)

def poor_coverage_fold(n_cells):  # C: evaluation = even cells in the lower half only
    even = np.arange(0, n_cells, 2)
    evaluation = even[even < n_cells // 2]
    discovery = np.setdiff1d(np.arange(n_cells), evaluation)
    return {"C": (discovery, evaluation)}
```

Per fold: `build_recording(cells=discovery)` fits the detector on discovery cells; the
evaluation fields are fitted with `fit_sorted_spikes_kde_encoding_model` on the same
time, position, environment and training mask the detector used, `position_std=6.0`.
