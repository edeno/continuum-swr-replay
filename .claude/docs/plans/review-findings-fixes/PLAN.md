# Review-findings fix plan

> **Ownership reconciliation (2026-07-29):** this file remains the historical
> owner of its decoder-identifier, ripple-query, and trajectory findings. F6's
> switch-conditional choice decision is superseded by
> [`regression-integrity-fixes` Phase 11](../regression-integrity-fixes/phase-11-decision-models.md),
> which retains same-arm and zero-replay decisions at decision grain. F4b/PFC
> regeneration is owned by [`pfc-fig4-recut`](../pfc-fig4-recut/PLAN.md). Do not
> implement either item from this file in parallel.

Fixes for the verified findings from the branch review (findings 1, 2, 4b, 5, 6, 7a,
7b, 8, 9). Finding 3 (per-sorting `curation_id`) is **out of scope** by decision.

All findings were independently verified against source before planning. Each fix
below records the concrete change, the blast radius, and a verification step.

## Decisions locked
- **F1 (training mask):** fix code now; **defer the mPFC/OFC re-decode** (HPC is
  already correct and carries the central claims).
- **F5 (`n_reversals`):** rebuild on the existing graph-based `ahead_behind_distance`
  column (see F5 detail); recompute the 16b "2 vs 7" and reassess.
- **F6 (choice model): SUPERSEDED** by regression-integrity Phase 11; retain this
  older decision only as audit history.
- **F9:** fix line-77 eager import; **skip** the line-361 exclusion-audit item.

## Verification discipline (applies to every finding)
Capture the affected reported numbers **before** editing, recompute **after**, diff.
F2, F8, and the F1 default path must show **zero** numeric change (correctness/
robustness only). F5/F7a/F7b will move numbers — report the deltas against
`paper/claims_table.md`, do not assert improvement.

---

## Batch 1 — cheap, no reported number should change

### F2 — `training_type` absent from artifact identifier
- **Change (write/skip side):** append `_{training_type}` to `epoch_identifier`
  **only when `training_type != "no_ripple"`**, mirroring the existing
  `ripple_detector_name` / `non_stationary` suffix pattern.
  - `scripts/decoding/run_decoding_bandit.py` (run_decode identifier)
  - `scripts/decoding/run_decoding_contfrag.py` (run_decode identifier)
- **Change (read side — added after code review):** `load_results` and
  `load_contfrag_results` reconstructed the identifier WITHOUT the suffix, so a
  `--training_type all` cache hit would `results_exist("X_all")==True` then
  `load_results` open the `no_ripple` `X_results.nc` and re-save it under `X_all`
  — silent overwrite. Threaded `training_type` (default `"no_ripple"`) through both
  loaders + the two cache-hit call sites so all three paths agree.
  - `src/continuum_swr_replay/analysis/hippocampus/result_loading.py`
  - `src/continuum_swr_replay/analysis/contfrag/result_loading.py`
- **Blast radius:** none — default (`no_ripple`) artifacts keep their names; all
  other `load_results` callers (glm.py, figure/content scripts) use the default and
  are unaffected, so `results_exist`
  (`src/continuum_swr_replay/decoding_common.py:153`) can no longer alias regimes on
  either the skip or the load path.

### F1 — sorted-spike decoders ignore `--training_type`
- **Change:** remove the `if brain_area == "HPC"` gate so
  `estimate_params["is_training"] = is_training` is set for all regions.
  - `scripts/decoding/run_decoding_bandit.py:169-170`
  - `scripts/decoding/run_decoding_contfrag.py:172-173`
  - Installed API accepts it: `SortedSpikesDetector.estimate_parameters(..., is_training=None)`;
    defaults to `np.ones(n_time)` (all bins) when omitted.
- **Scope:** code only. **No mPFC/OFC re-decode now** — those decodes feed
  `scripts/decoding/prepare_contfrag_replay_content_data.py --brain-area mPFC/OFC`,
  not the central hippocampal claims. Re-decode only if/when they enter the paper.
- **Verify:** re-decode ONE mPFC session, confirm the encoding model now excludes
  ripple bins (place fields / non-local rate differ from the all-bins version).
- **Blast radius:** HPC unaffected (already passed the mask). Central results unchanged.

### F8 — ripple query uses nonexistent field
- **Change:** rename restriction key `ripple_params` → `ripple_param_name`
  (`src/continuum_swr_replay/data_loaders/lfp.py:168`). Add the `pos_merge_id`
  (PositionOutput) key if any session has >1 position row.
  - Root cause: DataJoint silently drops unknown dict keys, so the param dimension
    has been **unrestricted** and only survived via `fetch1`'s single-row rule.
- **Verify:** DONE against live DataJoint. All 219 `RippleTimesV1` rows under the
  loader's filter (`group_name=CA1`, `filter_name=ripple_150_250`,
  `filter_sampling_rate=1000`) have `ripple_param_name='default'`, exactly one row
  per (session, interval). Old restriction matched exactly 1 row everywhere (no
  latent `fetch1` multiplicity), and adding `ripple_param_name='default'` selects
  that same row (spot-checked chimi20200214 02_r1: dataframes identical, 344 ripples).
  The 7 other param names in the table exist only under other filters and never
  collided. **Provably behavior-neutral on the full dataset.**
- **Blast radius:** none (verified); fail-fast insurance only.

### F9 (line 77) — eager DataJoint import defeats offline paths
- **Change:** move `from continuum_swr_replay.data_loaders import load_data` from
  module top-level into the function that calls it
  (`src/continuum_swr_replay/analysis/replay/prepare_data.py:77`). PEP-562 lazy
  loading means resolving that symbol at import imports `bandit_task` →
  `spyglass.common` → opens DataJoint, before `--from-cache` / `--help` / tests can act.
- **Skip:** line-361 exclusion-audit item (the "partial tables written" premise was
  refuted — writes are batched after the loop behind the invariant gate; only a
  non-persisted skip log remains, which is a nice-to-have).
- **Verify:** `python scripts/.../prepare_data.py --help` and a `--from-cache` run
  succeed with no DB connection (e.g. under an offline/no-DJ env).

---

## Batch 2 — trajectory suite (one recompute from existing `.nc`, no GPU)

All three live in `src/continuum_swr_replay/analysis/replay/trajectory.py` and
recompute from stored decodes. Regenerate the trajectory suite once after all three;
spot-check ~20 flipped-label events per fix against the posterior.

### F5 — `n_reversals` rebuilt on `ahead_behind_distance` (branch-safe)
- **Root cause:** current `_n_reversals` (`trajectory.py:365-372`) counts sign changes
  of `diff(linear_position)`; on the branched track a branch crossing is a
  discontinuous jump that fakes a reversal. Because SWR trajectories cross more
  branches, the artifact likely inflates the SWR "7" more than the non-SWR "2".
- **Change:** count sign changes of `diff(ahead_behind_distance)` over each event's
  time bins.
  - `ahead_behind_distance` is `sign × graph-geodesic-distance(animal, MAP)`
    (`.venv/.../non_local_detector/analysis/distance1D.py:get_ahead_behind_distance`):
    magnitude is a true track-graph shortest path (no linearization discontinuity),
    sign is +ahead/−behind by head orientation.
  - Already a per-timebin input column of `trajectory.py` (line 64) and already
    consumed per-event by `ahead_behind_summary` (line 155) — **zero new plumbing**,
    and consistent with the existing `max_ahead`/`max_behind`/`ab_slope` metrics.
  - Drop the `node_lin` / `linear_position` path for reversals.
  - Edge cases handled correctly by the signed geodesic coordinate:
    outbound-across-a-junction → 0 reversals; pass-through the animal
    (behind→ahead) → 0 reversals (a monotone sign change, not a turn);
    out-and-back → 1 reversal.
- **Design knob — jitter hysteresis:** the raw per-bin series is noisier than the old
  reduced node sequence, so require each monotonic run to exceed a small amplitude
  (~1 place bin ≈ 2 cm, or a few cm) before counting a direction change. Report
  sensitivity across a couple of thresholds — this determines whether the corrected
  "2 vs 7" is stable.
- **Sign caveat (note, don't fix):** sign uses head orientation, defined during
  immobility but can wobble when the decoded position sits ~perpendicular to the head
  axis. Same dependence the existing ahead/behind metrics already ride on. Spot-check
  a few mid-event sign flips to confirm real motion, not head jitter.
- **Reassess:** if the corrected count is fragile / non-significant, fall back to
  dropping `n_reversals` from claim 16b and leaning on `pause_fraction` (1.0 vs 0.93)
  and `max_step_distance` (2 vs 12 cm).
- **DONE / sweep result (218 sessions, inclusive-SWR preview):** old linear metric
  reproduces the reported numbers (median 2 vs 7, mean 3.75 vs 10.88). Raw ab at
  delta=0 is jitter-dominated (3 vs 8). **Production delta = place_bin_size (2 cm):
  median 0 vs 3, mean 1.38 vs 6.41, ratio 4.64x, 5/5 animals** — the dissociation
  *sharpens* (old ratio 2.9x) and non-SWR reversals fall to a median of 0,
  reinforcing the stationary-focal story. Monotone in delta (4.6x/6.2x/8.7x), 5/5 at
  every setting. Metric KEPT. Authoritative strict-`is_swr` numbers to come from the
  regen. **F7b well_radius default = 4 x place_bin_size (~8 cm).**

### F7a — multimodal: pairwise-adjacency → connectivity
- **Change:** replace the all-pairs adjacency test in `_is_multimodal`
  (`trajectory.py:219-226`) with a connected-components test over the occupied arms
  (induced subgraph via the `arm_adjacency` dict); multimodal iff `n_components > 1`.
  A contiguous chain A–B–C becomes unimodal (correct).
- **Blast radius:** feeds claim 17 (Supp, already flagged "weak, 3/5"). Expected to
  lower `multimodal_fraction` ~0.383 → ~0.237 — firms up an already-hedged negative.

### F7b — well endpoint: any leaf arm → actual well proximity
- **Change:** make `_terminates_at_well` (`trajectory.py:395-397`) check the endpoint's
  position along the arm (terminal decoded bin's `node_lin` / graph distance to the
  arm's degree-1 well node), not just `end_arm ∈ {0..5}`. Threshold from the ~2 cm bin
  size (or a small fixed distance); show sensitivity.
- **Blast radius:** feeds the wells-vs-paths framing (#13/#14).

---

## Retired Batch 3 findings — historical context only

### F6 — choice model excludes stays

`choice_prediction.py:405` removed the current arm as a candidate and dropped
stays, so the historical model estimated choice conditional on switching. Do
not relabel or extend that model here. Its replacement, claim adjudication, and
same-arm/zero-replay accounting are owned by
[`regression-integrity-fixes` Phase 11](../regression-integrity-fixes/phase-11-decision-models.md).

### F4b — stale PFC pickles + no schema guard

Commit `c86a240` removed an invalid non-nested LR comparison while old PFC
pickles could retain it. Do not inspect, regenerate, stamp, or load-guard those
artifacts from this plan. Canonical regeneration, provenance, stale-column
rejection, and Figure-4 cutover are owned by
[`pfc-fig4-recut`](../pfc-fig4-recut/PLAN.md).

---

## Follow-up (surfaced in code review) — DONE: wilbur20210404 group_name exclusion
Not one of the numbered findings; surfaced while reviewing F8. `get_ripple` hardcodes
`group_name="CA1"`, but `wilbur20210404_.nwb` (5 run epochs) has ripples only under
`sb_wilbur_ripple` (a different extraction), so `load_data` failed and those epochs were
**silently dropped** (0 decode outputs; confirmed vs live DB — 218/229 sessions have CA1,
the 11 missing are all wilbur20210404). Decision: **exclude + document** (not fall back to
a different ripple definition). Implemented:
- `get_ripple` (`data_loaders/lfp.py`): explicit row-count check → clear `ValueError`
  naming the session + available `group_name`(s) + doc pointer (instead of opaque `fetch1`).
  Verified vs live DB: good session unaffected (senor20201102 loads, 133 ripples);
  wilbur20210404 raises the documented error.
- `NWB_FILES` docstring (`data_loaders/constants.py`): listed under "Epochs with missing
  Spyglass pipeline entries."
- `docs/known_issues/wilbur20210404_missing_ca1_ripples.md`: full write-up + un-exclude recipe.

## Suggested order
1. Batch 1 (F2, F1-code, F8, F9-line77) — no reported number moves; validates workflow.
2. Batch 2 (F5, F7a, F7b) — one trajectory recompute; check 16b/17 survival.
3. Stop. F6 and F4b are historical routing notes with separate owners; no Batch 3
   work remains in this plan.
