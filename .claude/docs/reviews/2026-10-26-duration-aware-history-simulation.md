# Duration-aware event history: a simulation of the per-bin self- and cross-history design

2026-10-26. **Simulation only. No real data was read.**

## Question

Non-local events are intervals: median about 50 ms (non-SWR) and 90 ms (SWR-coupled). One
decoder's events never overlap. The per-bin rate design
(`per_bin_design.aggregate_to_timebins`, the `Mself` / `Mselfcross` rungs fitted by
`scripts/rate/fit_cross_history_rate.py`) treats each event as an instant onset: every 15 ms
bin of a stop is at risk, including the bins inside an ongoing event, and self- and
cross-history count past ONSETS (10 lags, 4 raised cosines).

On real data, two patterns stand out:

- the shortest-lag cross-history coefficient is hugely negative (separation-like);
- the self-history "burst" peaks at 105–120 ms, with rate ratio (RR) 1.56–1.98 for the non-local streams.

Are these duration artifacts, or real refractoriness, rhythm and coupling? And does a
duration-aware design recover the truth?

## What was built (branch `worktree-agent-ace075bd61a8ee966`)

- `aggregate_to_timebins(at_risk_events=..., offset_history=...)` (opt-in; the defaults keep
  the shipped design):
  - **At-risk time.** A bin's exposure becomes the time no event covers. Bins wholly inside an
    event drop out, after the lags are built.
  - **Offset history.** `offset_lag{j}` / `offhist_rcos_k` and the cross twins
    `cross_offset_lag{j}` / `cross_offhist_rcos_k` count event ENDS. They use the same 15 ms
    lags and raised cosines as the onset blocks.
  - The default design is exactly equal to the previous code's on simulated data (checked
    with `pd.testing.assert_frame_equal(check_exact=True)` against the HEAD module).
- `PerBinRateModelSpec.self_history_anchors` / `cross_history_anchors` (default `("onset",)`).
  All 13 shipped per-bin rungs keep their formulas, columns and history windows.
- The opt-in ladder `MODEL_CROSS_HISTORY_DURATION_RATE` (`Mself_dur`, `Mselfcross_dur`),
  mirrored in `docs/models_table.md`.
- `analysis/replay/duration_history_simulation.py`: the simulator, plus `fit_history_direction`.
  The latter mirrors the runner's `_fit_direction` with the real `aggregate_to_timebins` →
  `freeze_per_bin_cohort` → `fit_per_bin_rate_model`.
- Tests:
  - `tests/unit/test_duration_aware_history.py`: at-risk exposure, offset columns, specs.
  - `tests/unit/test_duration_history_simulation.py`: simulator, recovery per scenario,
    identifiability.
  - All seeded; 32 tests in about 60 s.
- Outputs, next to this report:
  - `run_simulation.py` → `results.json`, `kernels_no_swr.png`, `kernels_swr.png`,
    `identifiability.png`;
  - `run_null_replicates.py` → `null_replicates.json`.

  Every number below comes from these JSON files, or from test docstrings where marked.

### How the duration-aware design is fit

Each design is fit with the at-risk time over both streams' events. The fits are:

- **Self block:** none, onset-only, offset-only, both.
- **Cross block:** onset, offset or both, added to the best self rung.

All rungs of a direction share one frozen cohort.

The reported contrasts:

- **Fixed-block AIC gains, chosen in advance:**
  - Δself = no history → self block with both anchors;
  - Δcross = self rung → + cross block with both anchors.
- **Anchor contrasts:** AIC(onset-only) − AIC(offset-only). Positive means offset-locked.
- **Kernels:** read only at lags with ≥ 10 expected onsets ("support").

## Simulated scenarios

Shared settings:

- **Rates.** While no event is ongoing, the onset hazard is 0.4 Hz (SWR-coupled) and 0.6 Hz
  (non-SWR).
- **Durations.** Lognormal, medians 90 / 50 ms, log-SD 0.45, minimum 12 ms.
- **Stops.** Lognormal, median 3 s, log-SD 0.4, clipped to 0.5–10 s; the design truncates at
  TMAX = 4 s.
- **Reward-basis columns.** Uniform noise with no effect on the rate.
- **Kernels.** Gaussian bumps in the log hazard.

| scenario | truth |
|---|---|
| (a) independent | no dependence beyond non-overlap |
| (b) fragmentation | after ANY event's END, both streams ×6, peak 40 ms (sd 10 ms) |
| (c) onset rhythm | each stream ×4 at 110 ms (sd 15 ms) after its own previous ONSET |
| (d) cross coupling | non-SWR ×5 at 50 ms (sd 12 ms) after an SWR-coupled END; nothing the other way |

The fits come in three sizes:

- **Pooled:** one 6,000-stop session per scenario, about 7,200–7,800 SWR-coupled and
  10,900–11,800 non-SWR events. Measures bias.
- **Sessions:** 10 sessions of 350 stops per scenario, summarised as the real runner
  summarises sessions.
- **Identifiability:** 4,000 stops.

## What the current (shipped) design infers

### Pooled

| scenario / response | Δself | Δcross | self-onset kernel | cross-onset kernel |
|---|---|---|---|---|
| (a) SWR-coupled | 289.5 | 228.6 | RR 0.00 at 15 ms, rising to 1.03 at 150 ms | 0.04 at 15 ms |
| (a) non-SWR | 307.1 | 459.3 | 0.03 at 15 ms → 1.09 | 0.01 at 15 ms |
| (b) SWR-coupled | 321.5 | 394.9 | **peak 1.78 at 120 ms** | peak 2.09 at 90 ms |
| (b) non-SWR | 622.4 | 517.3 | **peak 2.20 at 90 ms** | peak 1.75 at 135 ms |
| (c) SWR-coupled | 298.5 | 211.4 | peak 1.51 at 105 ms (truth 3.39 at 105) | ≤ 0.88 |
| (c) non-SWR | 692.5 | 439.7 | peak 2.35 at 105 ms | ≤ 0.97 |
| (d) SWR-coupled (reverse, truth none) | 266.0 | **225.8** | ≤ 0.98 | 0.03 at 15 ms |
| (d) non-SWR | 341.7 | 495.0 | ≤ 1.13 | 0.00–0.03 at 15–45 ms, **peak 1.76 at 135 ms** (truth: 50 ms after the END) |

### Sessions (10 × 350 stops)

The median per-lag rate ratio across sessions is read as `self_history_rate_ratios` reads it.

- **(a) Independent events.**
  - Cross-history improves AIC in 90 % (SWR-coupled response) and 100 % (non-SWR) of
    sessions; the median gain is 6.0 / 18.0.
  - The shortest-lag cross coefficient `cross_hist_rcos_0` is negative in 100 % of sessions
    (median −3.8 / −20.1). In half of the non-SWR sessions it is flagged separated.
- **(b) Fragmentation alone.**
  - The self-history "burst" peaks at **1.38 at 120 ms** (SWR-coupled) and **1.99 at 90 ms**
    (non-SWR).
  - Real data: 1.56–1.98 at 105–120 ms.
  - Cross Δ = 21.6 / 28.5 in 100 % of sessions, with `cross_hist_rcos_0` median −5.3 / −73.2.
- **(c) Onset rhythm, no coupling.** Spurious cross gain: median 8.0 / 23.9 in 100 % of
  sessions.
- **(d) Coupling one way.** The reverse direction gains 10.0 in 100 % of sessions, though the
  truth has no coupling that way.

### Reading

In the onset-only design, non-overlap alone produces three things:

- a deep short-lag self "refractoriness";
- a deep, often separated, negative short-lag cross coefficient;
- a cross-history AIC gain in both directions in almost every session.

None of these is a dependence in the hazard.

End-locked fragmentation produces a self-history burst at 90–120 ms after the previous
onset. It is close to the real peak in both lag and size.

A true end-locked coupling is placed about one event duration later (135–150 ms after the
onset) and is preceded by an apparent inhibition.

The current design cannot tell (b) from (c): both give an onset-locked peak near 105–120 ms.

## What the duration-aware design infers

### Pooled

| scenario / response | Δself | Δcross | self offset-vs-onset | cross offset-vs-onset | recovered kernel (truth) |
|---|---|---|---|---|---|
| (a) SWR-coupled | −10.9 | 1.5 | −0.1 | 7.8 | — |
| (a) non-SWR | −4.4 | −2.8 | −1.7 | 6.3 | — |
| (b) SWR-coupled | 404.7 | 410.0 | **247.7** | **175.3** | self offset 4.48, cross offset 3.44, both at 30 ms (truth 4.01 at 45 ms, 2.90 at 30) |
| (b) non-SWR | 661.8 | 459.6 | **283.6** | **240.0** | self offset 4.24, cross offset 3.88 at 30 ms |
| (c) SWR-coupled | 85.2 | −5.6 | **−54.1** | −2.1 | self onset 2.46 at 105 ms (truth 3.39 at 105) |
| (c) non-SWR | 423.1 | −12.3 | **−132.4** | −2.0 | self onset 2.56 at 105 ms |
| (d) SWR-coupled (reverse) | −12.1 | **−0.2** | 0.7 | 7.5 | — |
| (d) non-SWR | −10.9 | **432.7** | 1.9 | **245.6** | cross offset 3.44 at 45 ms (truth 3.80 at 45) |

### Null replicates

`null_replicates.json`: scenario (a), 10 seeds × 2 streams, 6,000 stops each.

- Δself is ≤ −2.7 and Δcross ≤ −4.7 in all 20 fits.
- No rung beats the history-free rung by more than 1.87 AIC.
- AIC's best self rung is not `none` in 4 of 20 fits. Those wins are small, so "best rung"
  is not evidence; use the fixed-block gains.

### Sessions (10 × 350 stops)

- **(a)**
  - Δcross median −9.3 / −8.7, positive in 10 % / 0 % of sessions.
  - Δself median −8.3 / −10.8, never positive.
- **(b)**
  - Offset beats onset per session: median 7.6 / 16.8 (self) and 10.1 / 12.4 (cross).
  - Δcross is positive in 100 % of sessions. That is correct here: the fragmentation also
    follows the other stream's ends.
- **(c)**
  - Δcross median −9.9 / −9.9.
  - Onset beats offset only weakly per session: median −4.8 / −5.9.
- **(d)**
  - Non-SWR response: Δcross median 15.0 (100 % positive), offset beats onset by a median
    of 16.3.
  - Reverse: Δcross median −5.7 (0 % positive).

### Reading

The at-risk time removes the spurious refractoriness, inhibition and coupling: in (a) no
block gains, in every one of 20 null replicates.

- **(b)** is read as end-locked.
- **(c)** is read as onset-locked, at the right lag (105 ms). The peak is attenuated: 2.5 vs
  a true 3.4 for a 15 ms-wide bump through four raised cosines.
- **(d)** is read in the right direction, anchor and lag (45 ms after the SWR-coupled END),
  with no coupling the other way.

Fast-test configuration: 4× base rate and 300 stops, the values pinned in
`test_duration_history_simulation.py`. Across seeds 0–4, as stated in the test module's
comment:

- Null cross gains are ≤ 9.0.
- Every true effect gains ≥ 41.

## Identifiability limits

1. **Onset-locked history shorter than the previous event's duration is not estimable.**
   Those bins are not at risk: support at onset lags 15–45 ms is 0–3 expected onsets for the
   SWR-coupled stream (pooled (c)). There the onset kernel takes arbitrary values; the pooled
   (c) fit gives RR 24.7 at 15 ms with 0 expected onsets. The "refractory period" of the
   current design is the event itself, not an estimable quantity.
2. **A dependence within one bin (< 15 ms) of an end is not resolved.** An offset in the
   current bin is not used: it would leak a short event's own onset. In (b) the offset kernel
   peaks one bin early (30 ms vs the truth's 45 ms) in every fit, pooled and in all ten test
   fits.
3. **With near-constant durations the anchors cannot be told apart.** An offset is then an
   onset shifted by a fixed lag. The `identifiability` part of `results.json` (4,000 stops,
   non-SWR response, both streams' log-SD set to σ):

   | σ | canonical corr. (self onset vs offset block) | (b) offset-vs-onset | (c) offset-vs-onset |
   |---|---|---|---|
   | 0 | 0.94 | +23.0 | **+16.1 (wrong sign)** |
   | 0.1 | 0.93 | +66.7 | −17.0 |
   | 0.2 | 0.89 | +88.9 | −21.5 |
   | 0.45 | 0.77 | +225.3 | −98.4 |

   - With σ = 0, the onset rhythm (c) is attributed to the offset, and both blocks are
     retained up to σ = 0.2.
   - In the fast-test configuration (seed 0), constant durations give canonical correlation
     0.938 and an onset-vs-offset contrast of only −6.0, against 0.694 and −70.1 at σ = 0.45
     (test docstring).
   - Fragmentation (b) stays separable even at σ = 0, helped by durations that are not whole
     bins and by a narrow kernel. That margin is fragile.
   - **Decision rule.** If the real streams' log-SD is ≳ 0.2 and the canonical correlation is
     ≲ 0.9, the anchors are separable at the pooled scale. Below that, report the history
     only as the joint onset+offset effect.
4. **Per-session power is low at realistic rates.** At 350 stops a lag carries < 10
   expected onsets: support ≈ 37–98 per lag at 6,000 stops, so roughly 2–6 at 350. The
   anchor contrasts per session are single digits ((b) 7.6–16.8, (c) −4.8 to −5.9). Inference
   must aggregate across sessions (median, sign consistency, per animal), or pool.
5. **Not modelled.**
   - Stop-to-stop rate heterogeneity, which creates positive history (clustering) in any
     design.
   - A within-stop time course: the spline is fit but the true rate is flat.
   - The real duration distributions, including their correlation with gaps.
   - The kernels are log-additive, matching the GLM's model class; a different true form
     would add misfit.

## Recommendation for the real self- and cross-history refit

1. **Do not interpret the shipped self- and cross-history.** This covers the short-lag
   "refractoriness", the 105–120 ms burst, the negative shortest-lag cross coefficient and the
   cross ΔAIC, all as fitted by `fit_cross_history_rate.py` and the `*_histcos` rungs. In
   simulation, non-overlap plus end-locked fragmentation reproduces each of them with no
   rhythm and no coupling. Keep the cross-history values out of the paper (as `STATUS.md`
   already says) until the refit.
2. **Measure on the real event table first:**
   - each stream's duration log-SD;
   - the distribution of gaps from an event's end to the next onset. Fragmentation shows
     directly as an excess of short gaps.
   - Then compute `onset_offset_collinearity` on a real design, to place the data on the
     identifiability table above.
3. **Refit with `MODEL_CROSS_HISTORY_DURATION_RATE`.**
   - Give the onset rows `t_end_since_stop_onset` (`event_end_s − stop_start`), joined from
     `event_table` by `(session, stop_id, event_start_s)` as `attach_is_swr_to_onsets` joins
     `is_swr`.
   - Pass every non-local event of the session (both streams) as `at_risk_events`, with
     `offset_history=True`. `aggregate_to_timebins` raises if two events overlap.
   - Keep the excluded-time handling and the history window (same 10 lags).
4. **Report:**
   - the fixed-block gains, Δself (both anchors) and Δcross (both anchors);
   - the anchor contrasts AIC(onset-only) − AIC(offset-only);
   - the kernels only at lags with ≥ 10 expected onsets (`lag_support`).

   Aggregate across sessions with the existing median / %positive / per-animal sign counts.
   Do not use the AIC-best rung as evidence: it picked a history rung in 4 of 20 null fits.
5. **Wording.** If the anchors separate, describe the burst as end-locked fragmentation or
   onset-locked rhythm accordingly. If they do not (point 3 of the limits), describe it as
   "history after a previous event" without a mechanism.

## Code notes

- `docs/regression_pipeline_map.md` was already stale on this branch: entries were missing
  for `brain_state_reference`, `gamma_power` and two detection scripts. Regenerating it for
  the new module brought those in too.
- Commits: `f3db33c0` (design option, specs, simulator, tests), `92f8a4f7` (fixed cross
  block, canonical correlation, identifiability test), and the commit adding this report and
  its outputs.
