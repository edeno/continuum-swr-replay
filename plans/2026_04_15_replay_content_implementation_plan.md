# Replay content analysis — implementation plan

Execution plan for the analysis specified in
[2026_04_14_replay_content_analysis_plan.md](2026_04_14_replay_content_analysis_plan.md)
and [2026_04_14_replay_content_analysis_tasks.md](2026_04_14_replay_content_analysis_tasks.md).

## Ground rules

- **Single canonical schema at end — no shim.** Rename in place; do not build
  a parallel pipeline. Final state: `replay_arm`, `is_swr_flag`, `SWR_event`,
  `prev_trial_from_well`, `next_trial_to_well`, `last_rewarded_well` appear
  nowhere in `src/`, `scripts/`, or `tests/`.
- **Paper figures are not regenerated during this work.** The scripts behind
  them (including [run_replay_content_glm.py](../scripts/run_replay_content_glm.py))
  **are** refactored, but no figure files are re-saved until explicitly
  requested.
- **Conditional Poisson correctness is audited first.** Everything downstream
  assumes it's right; verify before stacking on top.
- **MUA via existing `hse_times`.** Derive `is_mua` / `is_mua_stop` from
  [`get_multiunit_hse`](../src/continuum_swr_replay/data_loaders/lfp.py#L194)
  output; do not add a new detector.
- **Behavior model deferred.** Stages 7B, 10, 11 are out of scope for this
  plan.
- **Conditional posterior: Option 1.** Load
  `{epoch_identifier}_results.nc` on demand in `tables.py` and compute the
  conditional non-local posterior via the existing
  [`get_conditional_non_local_posterior`](../src/continuum_swr_replay/analysis/hippocampus/posterior_analysis.py#L81).
  No new cached artifacts; no changes to the decoding script or
  `data_over_time.pkl` schema. Revisit caching only if iteration speed
  becomes painful.

## Verification already done

- Conditional non-local posterior **is** computed during decoding and is
  recoverable from `{epoch_identifier}_results.nc` via the acausal posterior.
- It is **not** saved into `data_over_time.pkl` — that pickle carries only
  `map_estimate`, `ahead_behind_distance`, `hpd`, and scalar-per-timebin
  state probabilities.
- MUA events are available per brain area as `hse_times[brain_area]` from
  [`get_multiunit_hse`](../src/continuum_swr_replay/data_loaders/lfp.py#L194).

---

## Phase A — Conditional Poisson correctness audit

Verify the existing fitter is correct before anything is stacked on it.

1. Read-through of
   [fitting.py](../src/continuum_swr_replay/analysis/hippocampus/replay_content/fitting.py)
   against the plan equation:
   `log E[c_ea] = log(n_nonlocal_bins_e) + X_ea β`, grouped by event, no
   per-event intercept absorbed into coefficients, correct within-event
   normalization.
2. Confirm standard-error computation: Hessian-based, cluster-robust by
   event, cluster-robust by session.
3. Audit
   [simulation.py](../src/continuum_swr_replay/analysis/hippocampus/replay_content/simulation.py)
   coverage. Add missing synthetic cases:
   - pure-proximity null; mixed proximity + future; post-hoc `is_next`
     confound
   - interaction terms
   - variable event lengths, including `n_nonlocal_bins == 1` and
     single-arm events
   - all three response variables (`arm_count_hard`, `arm_count_soft`,
     `arm_posterior_mass`) recover coefficients within 2 SE
   - M0 predicts uniform 1/N
4. **Independent cross-check:** fit the same synthetic data with
   `statsmodels.GLM(Poisson)` + event fixed effects; coefficients must
   match within numerical tolerance. Catches any conditional-likelihood
   bug that simulation tests alone would miss.
5. Document audit outcome in a `fitting.py` module docstring listing
   what was checked and what was fixed.

**Gate:** all synthetic recovery tests green; cross-check matches. Any
bug found is fixed before Phase B.

---

## Phase B — Canonical tables + in-place rename

Single rename-and-refactor pass. No parallel pipeline, no shim.

1. **New module** `src/continuum_swr_replay/analysis/replay/` at the
   package root (task-agnostic; eventually extractable). Moves:
   - `spatial_mapping.py`, `simulation.py`, `fitting.py`, `model_specs.py`,
     `model_comparison.py`, `model_checking.py` → `analysis/replay/`
     (kept intact).
   - `event_collapsing.py`, `predictors.py`, `data_preparation.py` →
     rewritten as `tables.py` / `summaries.py` / `uncertainty.py` /
     `spatial.py` / `shuffles.py`. Old files deleted in the same commit.
2. **Canonical column names from day one.** Every internal column is
   renamed exactly as in the plan:
   - `is_swr` (not `is_swr_flag`, not `SWR_event`)
   - `current_arm`, `prev_arm`, `next_arm`, `last_rewarded_arm` (not
     `*_trial_*_well`)
   - `arm_count_hard` / `arm_count_soft` / `arm_posterior_mass` (not
     per-arm count columns)
   - `n_nonlocal_bins`, `nonlocal_weight_sum`, `posterior_mass_total`
   - `is_current`, `is_prev`, `is_next`, `is_last_rewarded`,
     `is_same_patch`, `is_mua`
3. **Four canonical tables** built by `tables.py`:
   - `event_table` — one row per non-local event; all three summaries,
     all three length measures, HPD widths, context flags, `is_swr`,
     `is_mua`.
   - `event_arm_table` — one row per event × 9 arms.
   - `event_sequence_table` — one row per non-local timebin;
     `nonlocal_prob`, `decoded_arm_map`, `posterior_over_arms`.
   - `stop_table` — one row per stopping period, **including zero-event
     stops** (required for Stage 4 Part 1).
4. **`is_mua` / `is_mua_stop`** derived from `hse_times[brain_area]` by
   interval overlap.
5. **Full-posterior-dependent variables** computed from the conditional
   posterior loaded from `results.nc`: `arm_posterior_mass`,
   `posterior_mass_total`, `posterior_peak_mean`,
   `posterior_entropy_mean`, `posterior_hpd_width_{50,80,95}`.
6. **Junction-bin handling** is explicit at the posterior-to-arm step.
   Use the nearest-neighbor-over-real-bins pattern from
   [`map_estimate_to_arm`](../src/continuum_swr_replay/analysis/hippocampus/replay_content/spatial_mapping.py).
   Record `n_junction_bins` in `event_table`.
7. **In-place port of
   [run_replay_content_glm.py](../scripts/run_replay_content_glm.py)**
   onto the canonical tables. Same scientific content (M0–M7), new
   column names, new input pickles. Figures not regenerated during this
   work, but a regeneration run from the refactored script must be
   possible on demand.
8. **[prepare_replay_content_data.py](../scripts/prepare_replay_content_data.py)**
   becomes a thin driver over `tables.py`, emitting `event_table.pkl`,
   `event_arm_table.pkl`, `event_sequence_table.pkl`, `stop_table.pkl`.
   Old `long_count.pkl` / `long_binary.pkl` removed.
9. **Invariant tests** (plan Section 8):
   - `sum(arm_count_hard) == n_nonlocal_bins`
   - `arm_fraction` sums to 1 when `n_nonlocal_bins > 0`
   - `nonlocal_weight_sum == sum(nonlocal_prob)` per event
   - `posterior_peak_mean ∈ [0, 1]`
   - `posterior_hpd_width_50 ≤ _80 ≤ _95`
   - `nonlocal_prob ∈ [0, 1]`
   - `is_mua`, `is_swr` boolean and populated for all events
10. **Final grep guard:** `replay_arm`, `is_swr_flag`, `SWR_event`,
    `prev_trial_from_well`, `next_trial_to_well`, `last_rewarded_well`
    return zero hits in `src/`, `scripts/`, and `tests/`.
11. **Correctness check:** refactored M0–M7 coefficients on canonical
    tables match pre-refactor M0–M7 coefficients within tolerance. Any
    material disagreement is a bug.

**Gate:** invariant tests green; final grep returns zero hits;
pre/post M0–M7 coefficients match within tolerance.

---

## Phase C — Stage 0 QC gate

- `path_continuity` real vs. within-event circular time-shuffle, paired
  per session.
- 4×4 indicator co-occurrence for `is_current`, `is_prev`, `is_next`,
  `is_last_rewarded`; flag any pair with >30% co-occurrence.
- `is_swr × is_mua` 2×2 cross-tab.
- `posterior_hpd_width_80` distribution split by SWR / MUA / both / neither.
- All three nulls (within-event circular time shuffle, within-session
  stop-label shuffle, within-session `next_arm` circular shift)
  implemented as reusable functions in `shuffles.py`; every downstream
  stage calls one.
- New `scripts/qc_replay_tables.py` producing the QC plot bundle.

**Gate:** plan's Stage 0 acceptance criteria:

- median `path_continuity` exceeds shuffled median by > 0.05 in ≥80%
  of sessions
- ≥60% of events have `n_nonlocal_bins > 2`
- invariants hold for all events

Failing any of these → investigate the decoder, not the modeling layer.

---

## Phase D — Stage 2 rate model + Stage 4 hurdle

Built on canonical tables. **Per-session fits primary**; pooled
cluster-robust secondary.

- **Stage 2 (R0–R3).** Poisson GLM on `n_events` per stop, offset
  `log(immobility_time)`. R0 intercept-only; R1 adds `is_after_reward`,
  `is_after_omission`, `is_after_switch`; R2 adds
  `trials_since_block_change`, `block_number`; R3 splits R1 by SWR-only
  / MUA-only / neither. Per-session coefficients → signed-rank across
  sessions.
- **Stage 4.** Part 1 logistic on `has_replay_s` at stop level (needs
  zero-event stops from Phase B). Part 2 reuses the audited conditional
  Poisson with M2 predictors. Output table: per-context-variable
  summary of "affects occurrence, content, or both."

**Gate:** fits converge; per-session outputs saved. Findings catalogued
but paper not updated now.

---

## Phase E — Stage 5 choice + Stage 6 learning

- **Stage 5A (predictive).** Arm-level conditional logistic on
  `chosen_next_arm`; baseline = `nominal_reward_prob`, `visit_recency`,
  `recent_reward_rate`, `cumulative_reward`, `dist_to_arm`; add
  `arm_fraction`. Stop-level `is_stay` logistic as secondary. **No
  `_next_` variables.** Leave-one-session-out CV.
- **Stage 5B (retrospective).** Chosen vs. unchosen `arm_fraction`,
  `ends_at_next`, `contains_path_to_next`; null = within-session
  `next_arm` circular shift. All outputs labeled post-hoc.
- **Stage 6.** `is_stay` after reward / omission vs. replay content;
  k-trial change in visit rate to best arm; SWR and MUA as moderators.

**Gate:** does replay content add held-out log-likelihood above
behavior-only baseline? If no, skip Stage 3 interactions / 7A / 8 in
Phase F.

---

## Phase F — Interactions + observable value + sequence

- **Stage 3 M5–M7** — time / SWR / MUA / context interactions. Deferred
  per the plan until after 5/6, so interaction choices are motivated by
  findings.
- **Stage 7A observable value** — `nominal_reward_prob`,
  `recent_reward_rate`, `cumulative_reward`, `visit_recency`;
  head-to-head against M2 spatial labels.
- **Stage 8 sequence** — `forward_score`, `reverse_score`,
  `contains_path_to_*`, `ends_at_*`, first-order transitions,
  input-driven transition model S2 (without latent value).

**Gate at end of F:** report findings. Stages 7B / 9 / 10 / 11 remain
deferred until a behavior model exists.

---

## Phase G — Robustness pass

Once a final set of analyses is chosen across D–F:

- Repeat with `arm_count_soft` (offset `nonlocal_weight_sum`) and
  `arm_posterior_mass` (offset `posterior_mass_total`).
- Threshold sweep θ ∈ {0.5, 0.7, 0.8, 0.9}.
- Inclusion sweep: `n_nonlocal_bins > 5`, SWR-only, MUA-only,
  excluding high-junction events.
- Bootstrap across sessions; bootstrap across animals where n permits.
- Flag any coefficient whose sign flips across variants.

---

## Out of scope for this plan

- Stage 7B — latent value (requires behavior model).
- Stage 9 — fine spatial / graph-segment resolution (only if arm-level
  leaves ambiguity).
- Stage 10 — computational predictors (requires behavior model).
- Stage 11 — full sequence computational model.
- Any paper figure regeneration.

---

## Commit sequence

1. Phase A: audit + expanded simulation + statsmodels cross-check +
   any fixes to `fitting.py`.
2. Phase B: canonical tables + in-place rename + invariant tests +
   grep guard + pre/post M0–M7 coefficient match.
3. Phase C: QC gate + shuffles + QC plots.
4. Stop, reassess, then D → E → F → G one gate at a time.

## Open items

- **Iteration cost of Option 1.** If Phases C / D / E / F trigger
  frequent `tables.py` re-runs, revisit caching (per-session conditional
  posterior as `.zarr`, or extend `data_over_time.pkl` to carry
  per-arm posteriors). Not a Phase B concern.
- **Package extraction.** If `analysis/replay/` stays task-agnostic
  through Phases B–F, it becomes a natural standalone package. No
  action until then.
