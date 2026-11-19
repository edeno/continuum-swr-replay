# Design (for the lead) — Q8 context as a stop-level analysis pooled across sessions

[← back to PLAN.md](PLAN.md) · [ledger](ledger.md)

**Status:** design only; not built. Waiting for the lead.

## Why the marked family cannot answer it

Q8 asks whether the mPFC response to replay differs with the stop's reward outcome and in the trips before a patch switch. Both are properties of the stop, not of the event, and replay is concentrated after reward and away from switches. In the 75 PFC sessions (well-anchored events, event table), at least 10 events at each reward level occur in 23 sessions for non-SWR replay and 7 for SWR-coupled replay; at least 10 at each next-trip-switch level in 15 and 8 (ledger, 2026-10-27). A within-session marked slope is therefore not estimable in most sessions. The fix is to estimate the context effect once, from all sessions together, with stops (not events) as the place the context lives.

## What is measured (the response)

For every unit `u` and every replay event `e` (one class at a time; SWR-coupled and non-SWR replay separately):

- `y_ue` — the unit's spike count in the post-onset window `[0, +200) ms` of event `e` (the window the canonical kernels peak in; phase 1 values `timing_population_peak_ms_*`).
- `b_us` — the unit's baseline rate in the event's stop `s`: spikes per second over the stop's fitted immobile bins that lie outside every event's `[-250, +250] ms` window. It is computed per stop, so a rate difference between stops (the v1 lesson: context "effects" that were stationary rate differences) sits in the baseline, not in the response.
- The event-level response is `y_ue` against `b_us × 0.2 s`. A stop with no event-free time is left out (counted).

## Model

One Poisson GLM per class and per response sign, across all sessions:

```
y_ue ~ Poisson(mu_ue),
log mu_ue = log(b_us × 0.2 s) + alpha_u + beta_reward × rewarded_s + beta_switch × f(trips_to_switch_s) + gamma × overlap_e
```

- `alpha_u` — one free term per unit: that unit's typical event response (log gain over its stop baseline). It absorbs who responds and how much, so `beta` is a within-unit contrast between stops.
- `rewarded_s` — the stop's known outcome (0/1; unknown left out, counted). `f(trips_to_switch_s)` — the ordinal 0–3 mark as three indicators against "3 or more" (or the binary `next_trip_switches_patch` as the marginal companion), defined as in `analysis/pfc/marked_model_family.py:stop_patch_switch_marks`.
- `overlap_e` — the number of other events of any class whose onset falls in `e`'s post window (clustered events add spikes that are not `e`'s response).
- Fit separately for units whose canonical peak for that class is an increase and a decrease (sign from the canonical bundles' `peak_log_modulation`), so a gain on suppressed units is not averaged against enhanced ones. Optionally restricted to units called for the class (raw p < 0.05), as the timing values are.
- `exp(beta)` is a multiplicative change of the post-onset response relative to the stop baseline: >1 means a stronger increase (or a weaker decrease, for suppressed units).

## Unit of replication and how sessions and animals combine

- Events are not independent (same unit, same stop), and units are not (same session). The unit of replication for inference is the **animal**; sessions are nested in animals.
- Point estimate: the pooled GLM above over all sessions.
- Uncertainty: standard errors clustered by session (sandwich), and a **leave-one-animal-out** estimate for each animal.
- Consistency: the same model fit per animal (5 estimates), reported with the count of animals whose estimate has the pooled sign; no p-value across units.
- Weighting: units and events weight themselves through the Poisson likelihood; an animal with more sessions contributes more to the pooled estimate, which is why the per-animal fits and the leave-one-out are reported beside it.

## Cost

CPU only. The counts come from the canonical session assembly (`analysis/pfc/session.py`: fit mask, event sets, spike binning — no fitting), about 2 min per session to prepare, so about 2.5 h for 75 sessions; the GLMs are small (`statsmodels` with unit dummies, or `batched_glm.fit_poisson_glms` with the unit intercepts unpenalized).

## Decisions for the lead

1. Post window `[0, +200) ms` and baseline = the stop's event-free fitted time — or another window / a pre-event baseline per event?
2. Fit per response sign using the canonical peak sign, and all units or called units only?
3. Switch context: the ordinal `trips_to_patch_switch` (three indicators) or only the binary next-trip-switches?
4. Should SWRs without replay get the same analysis (they have a stop and a reward outcome too)?
