# Duration-aware self- and cross-history refit (real data, at-well stops)

2026-10-27. Branch `worktree-agent-ace075bd61a8ee966`. All numbers below are read from the JSON files
in this folder, named at each table. Nothing here is in `paper/reported/part3.json` or the claims
table.

## Plain-language answer

Once event duration is handled — a bin inside an ongoing event is no longer counted as at risk, and
history is measured from the previous event's end as well as its onset:

- **Refractory period: no, not beyond the event itself.** The shipped design's short-lag dip (rate
  ratio 0.00–0.17 at 15–30 ms after an onset, pooled within animal) is the event's own
  duration. After an event *ends*, the next onset is not suppressed. For detected SWRs it is most
  likely right away (model-free hazard 5.6× the late rate in the first 15 ms after an SWR ends). For
  non-local events it rises within 20–50 ms. Whether anything is suppressed in the first tens of ms
  after an *onset* cannot be measured: those bins are almost always inside the event.
- **Burst peak: yes, there is real clustering, but it is mostly tied to the previous event's end.**
  The dominant history is an excess of new onsets in the first ~100 ms after an event ends:
  - peak 2.5–3.4× at 30–55 ms for the non-local streams (model-free, pooled);
  - 2.6–4.5× in the duration-aware fits, in 5/5 animals for every population.

  This is the same timing as fragmentation: one underlying event detected as two pieces, or events
  following in quick succession. A smaller onset-locked component at about 90–150 ms survives
  alongside it (1.3–2.3× in the fits that hold both anchors). It is CI > 1 in 5/5 animals for
  detected SWRs, non-local (all) and non-local with SWR, but only 2/5 for non-local without SWR.

  So the shipped "burst at 105–120 ms after an onset" is mainly the end-locked succession seen
  through event duration, plus a weaker onset-locked rhythm.
- **Cross-coupling: not as a directed coupling.** The shipped design's hugely negative shortest-lag
  cross coefficient disappears. It was non-overlap: the other stream cannot start while this one is
  ongoing.

  What remains is a symmetric, end-locked succession: the other stream's onset is about 2.5×
  likelier 30–60 ms after an event ends, in both directions and 5/5 animals (pooled within animal).
  That fits one event splitting across the SWR boundary, or fast succession of events, not one
  stream driving the other.

  Per session the cross block does not pay for its parameters:
  - median ΔAIC −6.1 and −5.5 (the two directions);
  - 0/5 and 1/5 animals with a positive median.

  It shows up only when sessions are pooled.

**Main caveat.** Unmodeled rate variation between stops (and within sessions) inflates any history
term, especially when sessions are pooled. Two things argue against heterogeneity as the whole story:

- the kernels are timed: they peak and then return to about 1 by 100–150 ms;
- the model-free hazard falls below the late rate at 150–250 ms.

The size of the pooled effects is still an upper bound.

## What was run

- **Durations** (`event_durations.json`).
  - Non-local events with SWR: median 94 ms, log-SD 0.67; without SWR: 48 ms, 0.70 (0.61–0.76 per
    animal).
  - Kay SWRs: 82 ms, 0.52.
  - Non-local events never overlap (shortest gap from end to next onset: 4 ms).
  - These spreads are wider than the 2026-10-26 simulation used. Its scenarios were rerun at the
    real spread (`real_spread_check.json`, `real_spread_null.json`). The design still recovers each
    truth; under no dependence the cross block never gained AIC, and the self block gained in 1 of 20
    fits.
- **Refit** (`scripts/rate/fit_duration_aware_history_rate.py`, at-well cohort, 242 sessions; output
  `models_per_bin_rate/duration_aware_history_models.pkl` + `.provenance.json`, git `1fc814f8`; 259 s
  on 16 CPU workers).
  - At-risk time per population: Kay SWR intervals for detected SWRs; every non-local event for the
    three non-local populations. The decoder sample in which an event starts is counted at risk.
  - Rungs (`MODEL_CROSS_HISTORY_DURATION_RATE`), all with the time spline and reward basis:
    - `Mnohist_dur`: no history;
    - `Mself_onset_dur` / `Mself_offset_dur` / `Mself_dur`: self history from onsets, ends, or both;
    - `Mselfcross_*_dur`: as `Mself_dur` plus cross history from the other non-local stream's
      onsets, ends, or both.
- **All-stops cohort: not used.** An all-stops refit was run and then deleted, together with its
  sidecar, on the lead's decision (at-well only). The shipped `cross_history_models.pkl` (2026-10-11)
  was fit on all stops, so it is a different cohort and is not compared here.

### Why per-session kernels are not read

A real session has about 94–188 onsets per population, which leaves at most about 1.2 expected onsets
per history lag. `spotcheck_sessions.png` shows three sessions:

- With ≤ 1 expected onset per lag (j16, the wilbur non-SWR fit), per-session kernels swing by orders
  of magnitude.
- The richest session (chimi, up to 12 per lag) gives smooth kernels. Its non-SWR kernels are flat
  and elevated at every lag, which is a heterogeneity signature.

So:

- **Per session:** only AIC contrasts are aggregated (a block likelihood comparison is valid however
  sparse the session).
- **Kernels:** fit pooled within animal, with session fixed effects (`fit_pooled_kernels.py`).
- **Measurable lag:** pooled support ≥ 3 expected onsets and Wald SE ≤ 0.5 (log scale). Unmeasurable
  lags are not summarized or drawn.

## Session-level AIC contrasts (`session_contrasts.json`, at-well)

How to read the table:

- ΔAIC = AIC(simpler) − AIC(richer), so positive favors the richer rung.
- Each cell shows: session median / % of sessions positive / animals whose median is positive (of 5)
  / animals significant by summed likelihood ratio (of 5).
- "offset vs onset" is AIC(onset only) − AIC(offset only), so positive favors the end anchor; it is
  non-nested and has no likelihood-ratio test.

| population | self, both anchors | self, offset vs onset | offset adds to onset | onset adds to offset | cross, both | cross, offset vs onset |
|---|---|---|---|---|---|---|
| SWR | +11.0 / 81 % / 5 / 4 | +2.4 / 61 % / 5 | −0.2 / 48 % / 1 / 4 | −3.0 / 28 % / 0 / 4 | — | — |
| non-local | +7.2 / 71 % / 4 / 3 | +2.5 / 65 % / 5 | −0.0 / 50 % / 2 / 3 | −2.8 / 25 % / 0 / 3 | — | — |
| w/ SWR | +1.3 / 55 % / 2 / 3 | +2.2 / 67 % / 4 | −1.1 / 40 % / 1 / 3 | −3.5 / 18 % / 0 / 3 | −5.5 / 29 % / 1 / 2 | +0.4 / 56 % / 3 |
| no SWR | −5.2 / 22 % / 0 / 3 | +0.8 / 65 % / 5 | −3.8 / 21 % / 0 / 3 | −4.5 / 12 % / 0 / 1 | −6.1 / 17 % / 0 / 2 | +1.7 / 69 % / 5 |

**Fit failures** (non-finite IRLS weights; per rung, of 242 sessions):

- `Mnohist_dur` and the offset-only self rung: 0 in every population.
- `Mself_onset_dur`: 0–2.
- `Mself_dur`: 0–4.
- Cross rungs: 5–9.

Failures come only from rungs with an onset block, as the simulation predicted: an onset lag shorter
than the previous event cannot be estimated. A history term is flagged separated in 3–149 sessions per
rung. Separation is most common with the cross blocks and with the sparse non-local-without-SWR
stream. Exact counts are in `session_contrasts.json` (`fits`).

## Pooled within animal (`pooled_kernels_summary.json`, at-well; 5 animals each, no failed fits)

### Self history

Kernel columns give the median over animals: peak rate ratio @ lag (measurable lags only) / animals
whose peak CI is above 1.

| population | ΔAIC self both (animals +, sig) | offset vs onset | offset adds to onset | onset adds to offset | shipped onset kernel | end-only kernel | both: onset part | both: end part |
|---|---|---|---|---|---|---|---|---|
| SWR | +945 (5, 5) | +146 (4/5 +) | +190 (5, 5) | +44 (5, 5) | 3.44 @ 105 ms / 5; first lags 0.00, 0.01, 0.17 | 7.93 @ 15 ms / 5 | 2.28 @ 90 ms / 5 | 4.53 @ 15 ms / 5 |
| non-local | +943 (5, 5) | +117 (5/5) | +238 (5, 5) | +61 (5, 5) | 2.12 @ 120 ms / 5; 0.04, 0.07, 0.31 | 3.37 @ 45 ms / 5 | 1.59 @ 120 ms / 5 | 2.63 @ 30 ms / 5 |
| w/ SWR | +531 (5, 5) | +131 (5/5) | +156 (5, 5) | +26 (5, 5) | 2.51 @ 120 ms / 5; 0.04, 0.06, 0.38 | 4.83 @ 30 ms / 5 | 1.83 @ 105 ms / 5 | 2.96 @ 30 ms / 5 |
| no SWR | +306 (5, 5) | +37 (5/5) | +40 (5, 5) | +18 (4, 4) | 2.67 @ 120 ms / 5; 0.06, 0.17, 0.68 | 3.56 @ 45 ms / 5 | 1.34 @ 150 ms / 2 | 2.92 @ 45 ms / 5 |

### Cross history (non-local streams)

| response | ΔAIC cross both (animals +, sig) | offset vs onset | shipped cross-onset kernel | other's end, alone | both: other's onset | both: other's end |
|---|---|---|---|---|---|---|
| w/ SWR | +144 (5, 5) | +17 (5/5) | 2.11 @ 105 ms / 5; first lags 0.03, 0.07, 0.39 | 3.03 @ 45 ms / 5 | 1.42 @ 120 ms / 3 | 2.90 @ 45 ms / 5 |
| no SWR | +104 (5, 5) | +21 (5/5) | 1.66 @ 120 ms / 4; 0.00, 0.00, 0.06 | 2.70 @ 45 ms / 5 | 1.65 @ 105 ms / 4 | 2.28 @ 45 ms / 5 |

Figures: `pooled_kernels_self.png`, `pooled_kernels_cross.png` (one line per animal, 95 % CI,
measurable lags only).

**Session-level vs pooled.** The pooled fits hold one kernel per animal across its sessions, so
sessions too sparse to pay for the parameters on their own add up. That is why "no SWR, self" is
negative per session yet +306 pooled. The pooled gains also absorb any rate variation the session
fixed effect and shared spline leave over.

## Model-free check (`hazard.json`, `hazard_by_animal.png`, at-well)

For each event, this measures the gap to the next onset of the same detector in its stop (censored at
the stop's end), as a hazard per 15 ms bin divided by the late (> 300 ms) rate. Pooled values:

| next onset (previous event) | first 15 ms after the end | peak (time since end) | lowest at 120–300 ms |
|---|---|---|---|
| SWR (SWR) | 5.6 | 5.6 @ 7.5 ms | 0.68 @ 202 ms |
| w/ SWR (any non-local) | 1.3 | 3.4 @ 37.5 ms | 0.54 @ 202 ms |
| no SWR (any non-local) | 0.8 | 2.5 @ 52.5 ms | 0.76 @ 218 ms |
| w/ SWR (w/ SWR) | 2.3 | 4.5 @ 37.5 ms | 0.44 @ 202 ms |
| no SWR (no SWR) | 0.9 | 2.5 @ 52.5 ms | 0.93 @ 188 ms |
| no SWR (w/ SWR) — cross | 0.6 | 2.5 @ 52.5 ms | 0.37 @ 218 ms |
| w/ SWR (no SWR) — cross | 0.5 | 2.5 @ 52.5 ms | 0.55 @ 218 ms |

Against time since the previous *onset* (at risk only after it ended), the peaks are 5.6 @ 82.5 ms
(SWR), 3.1 @ 112.5 ms (with SWR) and 2.4 @ 82.5 ms (no SWR).

These shapes match the fits: an end-locked excess within ~100 ms, then a dip below the late rate. The
5/5 animals agree in direction (per-animal peaks in `hazard.json`).

## Next

The lead decides on claim 19c's burst/refractory text. A wording consistent with this review:

- no refractory period beyond the event;
- clustering mostly locked to the previous event's end (fragmentation-like succession);
- a weaker onset-locked component at ~90–150 ms;
- the cross "coupling" is a symmetric end-locked succession, not directed.

Numbers must go through a `part3` summary before they are quoted. Not run: a stop-level random effect,
to bound the heterogeneity caveat.
