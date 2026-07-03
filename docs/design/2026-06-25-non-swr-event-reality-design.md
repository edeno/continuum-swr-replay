# Design: Are non-SWR non-local events real? — spike-evidence reality analysis

Status: design (engine prototyped on one session; ready to scale)
Date: 2026-06-25
Owner: replay-trajectory-metrics branch

## Goal

Establish that hippocampal **non-local decoding events that do not overlap a
sharp-wave ripple (SWR)** are *real neural events* — population activity that
genuinely represents a remote location — and **not decoding artifacts**. This
is the linchpin (Goal 2 of `paper/claims_table.md`): the prevalence,
characterization, and method-superiority claims all inherit it.

The central skeptic's objection is circularity: the same state-space model
both *labels* an event non-local and *produces* the posterior we interpret.
The answer is to interrogate the **likelihood alone** — the spike evidence,
with the prior removed — and show it supports the decoded position.

## Artifact worries this addresses

| ID | Worry | Discriminating test |
|----|-------|---------------------|
| W1 | Theta- / movement-driven (theta sequences, not immobility replay) | theta-z + speed: SWR vs non-SWR vs running |
| W2a | No spikes at the event → ground-process / continuity-prior coasting | per-event spike count, MUA-z, # active tetrodes vs baseline |
| W2b | Spikes present but non-spatial (interneuron) → posterior carried by prior | prior-excluded likelihood ratio (centerpiece) |
| W3 | Local coding mislabeled non-local because the model's local state is too tight (place-field extent) | likelihood-peak location; multi-bin local-kernel re-decode |
| W4 | No strict distance defines "non-local" | far (remote) events, once W2 is satisfied |

## Core idea

The decoder reports `posterior ∝ likelihood × prior`; its peak is the MAP
(the "decoded position"). Artifacts (W2a, W2b) produce a confident *posterior*
via the *prior* while the *likelihood* stays uninformative. So we recompute
the **prior-excluded spike likelihood** over track positions for each event
and test whether it supports the decoded (remote) position over the animal's
own position, **above a matched non-event baseline**.

Key feasibility fact (verified): the likelihood is recomputable from the
**saved encoding model + the event's marks** — no re-decode. The fitted
detector stores everything `predict_clusterless_kde_log_likelihood` needs.

## Components

1. **Session event table** (`build_session_events`)
   - Input: `{epoch}_HPC_80_data_over_time.pkl` (per-timebin, decode time axis).
   - Non-local events: contiguous spans of `(Non-Local_Continuous_prob +
     Non-Local_Fragmented_prob) > 0.8` **and** `head_speed < 4 cm/s`, ≥3 bins.
   - Per event: `is_swr` (any `SWR_event > 0` in span), `mean_abs_ahead_behind`
     (track-based, from `ahead_behind_distance`) → close (<40 cm) / remote
     (>100 cm), n bins. (The 40/100 cm cutoffs are provisional; they can be
     tied to place-field width to match claim #5's boundary. The 40–100 cm
     middle band is deliberately excluded from both groups for clarity.)
   - **Matched non-event baseline:** contiguous spans of `nl_prob < 0.2` and
     `head_speed < 4`, duration capped to the event range (3–40 bins).
     Currently matched on immobility + duration only; **position/well-matching
     is an available robustness refinement** (do NOT match on firing rate — it
     is part of what we are testing).
   - Groups: nonSWR-remote, nonSWR-close, SWR-remote, baseline(non-event).

2. **Encoding-model loader** (`load_encoding`)
   - Load `{epoch}_HPC_detector.pkl`; take `encoding_model_[('', 0)]`.
   - `env = enc['environment']; env._is_fitted = True` (the fit flag is not
     persisted through save/load, but `place_bin_centers_`,
     `is_track_interior_`, `track_graph`, `distance_between_nodes_` are intact).
   - Build `encargs` from the standard keys; **override `waveform_std` to a
     scalar** (`float(enc['waveform_std'][0])`) so the predict function expands
     it per-electrode — required because some tetrodes have 3 (not 4) waveform
     features and a non-scalar std crashes `kde_distance`.

3. **Mark loader** (`load_marks`, cached)
   - `load_data(nwb, epoch)['spike_times']['HPC']` and
     `['spike_waveform_features']['HPC']` (lists per electrode). Cache to disk
     per session (load_data is slow). Marks are absolute-time and align with
     the decode axis (the position-tail drift does not affect spike times).

4. **Likelihood recompute** (`event_likelihood`)
   - For an event window `[t0, t1]`: slice each electrode's spikes/features to
     the window; call `predict_clusterless_kde_log_likelihood(time, time,
     animal_position[:,None], spike_times, waveform_features, **encargs,
     is_local=False)` → `L` of shape `(n_time, n_interior_bins)`.
   - Also return `n_spikes`, `n_tetrodes` (electrodes with ≥1 spike).

5. **Metrics** (`event_metrics`) — all bin lookups via the native
   `env.get_distances_to_interior_bins(pos).argmin()` (track distance):
   - **ΔLL(MAP vs animal)** = mean over timebins of `logL_t(MAP_t) −
     logL_t(animal_t)`. Real → > 0; prior-driven → ≈ 0.
   - **ΔLL per spike** = `sum_t(logL(MAP) − logL(animal)) / n_spikes`.
     Calibrates per-spike fidelity (real-but-small vs uninformative).
   - **Spike count / # tetrodes** (W2a).
   - **GPI-peak distance** (#8b) = track distance from the event's mean MAP to
     `argmax(summed_ground_process_intensity)`, compared to a random-position
     reference.

6. **Aggregation** (`aggregate`)
   - Per session: median of each metric per group.
   - Across sessions: per-animal medians and **within-animal direction
     consistency** (n/5) — the primary inferential summary (avoid
     event-level pseudoreplication).

7. **Figures** (`render`)
   - Per-timebin-normalized likelihood **heatmaps** (example event per group),
     MAP + animal overlaid.
   - **Metric distributions** by group, including the baseline (ΔLL vs animal,
     ΔLL per spike). Baseline is the reference line, not zero.
   - **GPI-position** panel: GPI(x) with peak/min; event-MAP distribution.

## Comparison groups and the null

- The decisive null is the **matched non-event immobility baseline**, not a
  random-remote position. Prototype lesson: the random-remote null is muddy
  because the *local* position is itself spike-supported, so it also beats a
  random remote bin. The non-event baseline gives ΔLL ≈ 0 and per-spike ≈ 0 —
  the clean reference.
- SWR-remote is the **positive control** (strong, high-spike events).
- **close vs remote** is the gate: if close events fail (group with baseline),
  headline claims restrict to the remote subset.

## Prototype result (chimi20200213_02_r1, one session)

Validated end-to-end. Medians:

| group | ΔLL(MAP vs animal) | ΔLL per spike | dist to GPI peak |
|-------|-------------------|---------------|------------------|
| nonSWR-remote | 0.57 | 0.85 | 106 cm |
| nonSWR-close | 0.73 | 0.36 | 97 cm |
| SWR-remote | 6.97 | 1.45 | 70 cm |
| baseline (non-event) | **−0.04** | **−0.05** | 84 cm |
| random position → GPI peak | — | — | 71 cm |

Reading: events (incl. close) are clearly above the non-event baseline (≈0);
non-SWR is real but lower per-spike than SWR; MAPs are not closer to the GPI
peak than chance (content-specific). Honest caveat: a nonSWR-remote cluster
near the GPI *min* (418 cm) — passes the spike tests, likely a real track
location; spot-check which arm/well.

## Scaling plan

1. Run the engine on a **handful of sessions across all 5 animals** (smoke
   test → representative set). Confirm: per-spike-above-baseline holds; close
   events stay above baseline; the GPI result holds.
2. If clean, run the full set; aggregate per-animal with within-animal
   consistency. Sampling per group per session is fine (cap ~30–50 events) —
   log any cap (no silent truncation).
3. Decide close-subset scope from the aggregated gate result.

## Saved-likelihood workflow (decode-pinned, all-events)

The per-event likelihood recompute is only tractable under the 30/group sample
cap. For a fully decode-pinned, **all-events** result, the decode now persists
the likelihood so the analysis can read it instead of recomputing:

1. **Decode** (`run_decoding_bandit.py`, HPC path): `estimate_parameters(...,
   return_outputs="log_likelihood")` adds a `log_likelihood` (n_time x
   n_state_bins) data_var to `results.nc`. DONE (active on the next decode).
2. **Reality analysis** (TODO, gated on a first re-decoded session): when
   `results.nc` contains `log_likelihood`, read the non-local-state interior-bin
   slice directly (no marks reload, no recompute), drop the sample cap, and run
   **all events**. Keep the per-event recompute (+ cap) as the fallback for
   decodes without a saved likelihood.
3. **Validation** (required before relying on the reader): re-decode one
   session with the change, then confirm (a) the reader extracts the correct
   non-local likelihood and (b) the saved likelihood matches the per-event
   recompute (saved == recomputed ΔLL) — a direct end-to-end consistency check.

Until step 2/3 are done, the 30/group recompute remains the working path;
the all-events result waits on the (deferred) re-decode.

## Out of scope (separate work)

- W3 definitive: multi-bin local-kernel **re-decode** (GPU subset).
- Model selection / label stability (#12), false-positive null (#13).
- The kind/function and method-superiority goals (Goals 3–4).

## Pitfalls / decisions (locked from the prototype)

- `waveform_std` **must** be passed as a scalar (3- vs 4-feature tetrodes).
- `env._is_fitted = True` after load (unpersisted flag; data intact).
- All MAP/animal → bin mapping uses **track distance**
  (`get_distances_to_interior_bins`), not linear-coordinate `argmin`.
- 2 ms bins are spike-sparse → heatmaps are illustrative; the **ΔLL metrics**
  (accumulated over the event) are the quantitative backbone.
- Inference at the **animal** level (within-animal consistency), not pooled
  events.
- Reproducibility: fixed RNG seeds (per-event seeded by start index for the
  random reference); explicit thresholds (0.8 non-local, 4 cm/s immobility,
  40/100 cm close/remote).

## Reusable script target

Promote the prototype (`scratchpad/proto4.py`) into a documented script,
e.g. `scripts/qa/analyze_event_spike_evidence.py`, with the components above,
a `--session`/`--animal` CLI, cached marks, and the three figures written to
`paper/figures/reality/`.
