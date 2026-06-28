# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

- `src/continuum_swr_replay/data_loaders/trials.py` — trials / well-visits are
  built **only from DIO events** (pump = reward, beam-break = poke). Reward
  *outcomes* are here; nominal `p(R)` and contingency blocks are **not** (they
  live in the StateScript log, which the pipeline never reads). Phase 1 adds the
  StateScript-derived contingency table alongside this; the DIO-based trial
  construction is **unchanged**.
- Spyglass `StateScriptFile` (`nwb_file_name`, `epoch`, `file_object_id`;
  `.fetch_nwb()[0]["file"].content` → the full runtime log) — the source of the
  contingency data. Confirmed: runtime lines `"<ts> contingency = N"` and
  `"<ts> leafProbs1..6 = <pct>"` carry the block index and per-port `p(R)`.
- `trodes_to_nwb/convert_statescript.py` (branch `statescript`,
  <https://github.com/LorenFrankLab/trodes_to_nwb/blob/statescript/src/trodes_to_nwb/convert_statescript.py>)
  — a **generic** StateScript parser to **ADAPT, not import** (the repo must not
  depend on `trodes_to_nwb`). Copy the small line-parse logic
  (`<ts> <name> = <int>`) and the DIO-offset alignment approach
  (`calculate_time_offset` → `timestamp_sync`) into a local exploratory module,
  with attribution. It does not know contingency/leafProbs semantics — Phase 1
  supplies that on top.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:757` `lagged_trial_rewards`
  (style/placement model for new per-stop feature helpers);
  `_helpers.py:880` `aggregate_to_timebins` (`stop_covariates` param at `:889`)
  — how per-stop covariates enter the per-bin design (Phase 2).
- `src/continuum_swr_replay/analysis/replay/model_specs.py:529`
  `PerBinRateModelSpec` (`formula()` at `:580`, `reward_rate` toggle at `:586`);
  `:525` `STOP_COVARIATES` — Phase 2 generalizes with `extra_covariates`.
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:39`
  `_design_columns`; `:58` `fit_per_bin_rate_model` — the fit path Phase 2 uses.
- `src/continuum_swr_replay/analysis/replay/learning.py:383`
  `fit_is_stay_logistic_per_session` — existing choice/`is_stay` (patch-level)
  conventions to match.
- `Processed-Data/replay_content/stop_table.pkl` — per-stop table consumed
  read-only; Phase 1 produces a companion `contingency_table.pkl` keyed by
  `stop_id` rather than regenerating it.

## Scope and dependency policy

### Goals
- Parse the StateScript contingency structure (per-port nominal `p(R)`, block
  index, change times) and align it to trials/stops (Phase 1).
- Test whether non-local replay **rate** is differentially modulated by
  behavioral uncertainty / value across **SWR-coupled vs non-SWR** non-local
  events, using both model-free signals and the task-grounded value/RPE/block
  covariates, controlling for the within-stop time course, self-history, reward
  kernel, and reward rate (Phase 2).

### Non-Goals
- **Replay direction (forward/reverse).** Content question; this is rate-only.
- **The full SpatialBanditTask RL model.** `leaky_q_value_diff` + nominal `p(R)`
  are the value stand-ins; fitting a latent value/belief model is future work.
- **Reward magnitude.** Reward is binary (100 µL or omission) — no magnitude axis.
- **Regenerating the canonical `stop_table`.** Phase 1 persists a companion
  table; folding the columns into the `stop_table` builder is a later cleanup.
- **Promotion into `MODEL_PER_BIN_RATE` / `all_models.pkl`.** Phase 2 runs as a
  standalone analysis; promotion (and the full 45-min regen) waits on a result.

### Dependency policy
**No new dependencies, and do NOT import `trodes_to_nwb`.** Phase 1 adapts a
minimal StateScript line-parser + DIO-offset alignment from
`convert_statescript.py` (attribution comment), implemented locally. Phase 2
uses the already-present statsmodels/pandas/numpy + the merged Family B.

### Temporary / exploratory status
This whole plan is **exploratory and temporary**. A canonical parsed-events
table (with the contingency already extracted) is expected to become available
later and will **supersede** Phase 1's hand-rolled parser and its
`contingency_table.pkl`. Build Phase 1 as a clearly-marked exploratory module
(scratch-tier, attribution + a "supersede when the parsed-events table lands"
note), not a permanent `data_loaders/` fixture. Phase 2's analysis should depend
only on the *column schema* of the contingency table
([shared-contracts.md](shared-contracts.md)), so swapping the source later is a
one-line change.

## Metrics
- **Phase 1 correctness:** parsed contingencies match the documented schedule —
  per-port `p(R) ∈ {0, 0.2, 0.5, 0.8, 1.0}`; block lengths 60 trials (×3) /
  80 (×2 for one animal); the special first session = 100 trials @ 1.0 then
  100 @ 0.5; block-change times land between trials (no mid-trial change); every
  trial/stop maps to exactly one block + a `p(R)` for each of the 6 ports.
  Spot-check ≥3 sessions across ≥2 animals against the raw log.
- **Phase 2 success (exploratory):** report, per uncertainty/value covariate,
  the per-population RR + per-animal sign + the **paired** `non_local_no_swr`
  vs `swr`/`non_local_swr` Wilcoxon, controlling for reward-rate/time/history.
  The pre-registered hypothesis: uncertainty signals elevate `non_local_no_swr`
  rate more than `swr` (the dissociation already seen in reward-rate: 1.28 vs
  0.89, p=2e-6; switch: 0.76 vs 1.16, p=6e-8). A reported null is a valid result.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| StateScript clock ≠ NWB/neural clock | Reuse `convert_statescript.calculate_time_offset` with DIO pump/poke reference times (already loaded) to produce `timestamp_sync`; validate by checking parsed reward events line up with DIO pump times. |
| Contingency changes are **uncued** → objective `trials_since_block_change` is a weak direct regressor (animal infers the change) | Treat block vars as a ground-truth **anchor + response-latency** layer; keep the model-free *subjective* signals as the primary regressors. |
| First session (uniform `p(R)`) is exposure/training, not the bandit | Exclude it from Phase 2 value/uncertainty fits; Phase 1 still parses it (it validates the parser: `contingency=0`, all `leafProbs=100`). |
| Parsed `p(R)` is nominal, not experienced | Carry both: nominal (from contingency) and the model-free `leaky_q` (experienced); the Methods stress they differ. |
| Hand-rolled parser drifts from the eventual canonical contingency table | Phase 2 binds to the column *schema* only; mark Phase 1 temporary and validate it against the documented schedule so a later swap is verifiable. |

## Rollout Strategy
All-at-once, additive. Phase 1 ships a new companion table + loader (nothing
existing changes). Phase 2 is a standalone analysis script + a generic
`extra_covariates` spec field (default `()` → shipped ladder unchanged). No
canonical `stop_table`/`all_models.pkl` regeneration is forced.

## Open Questions
1. **Parser sourcing — RESOLVED.** Do not import `trodes_to_nwb`; adapt a
   minimal line-parser + DIO-offset alignment locally (attribution). The whole
   parser is temporary, to be replaced by the canonical parsed-events table when
   available.
2. **Companion table grain — per-trial or per-stop?** Per-stop keyed by
   `stop_id` (what Phase 2 needs), derived from a per-trial contingency mapping.
   Settled in [shared-contracts.md](shared-contracts.md).
3. **`leaky_q` learning rate `q_alpha` / `switch_window`** — free parameters;
   default `0.3` / `5`, report sensitivity. The real RL model would fit them.

## Estimated Effort
- Phase 1: ~200–300 LOC (adapter + contingency reconstruction + alignment) +
  ~120 LOC tests. Dominated by validation against the known schedule.
- Phase 2: ~150 LOC (covariate helper + `extra_covariates` + runner) + ~100 LOC
  tests, building on the merged Family B infra.
