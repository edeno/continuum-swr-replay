# Phase 1b — Construct marks, centered interactions, rare-level eligibility

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Why: correlated marks split and shrink each other under the joint fit (current-arm and current-patch content at r = 0.66; extent, reversals and pauses at 0.5–0.7; raw interaction products at 0.81–0.95 with their own main effect), and a fraction times a rare binary produces a few enormous event weights that destabilize held-out prediction. This phase changes what the marks are; it does not change how they are fit.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/marked_model_family.py:55-131](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L55) — mark name constants and `MARK_DESCRIPTIONS`.
- [src/continuum_swr_replay/analysis/pfc/marked_model_family.py:135-190](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L135) — `MarkedFamilySpec` (`required_varying_nuisance_marks` at `:169`).
- [src/continuum_swr_replay/analysis/pfc/marked_model_family.py:195-390](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L195) — `RECOMMENDED_MODEL_SPECS`, `CORE_MODEL_NAMES`, `INTERACTION_MODEL_NAMES`.
- [src/continuum_swr_replay/analysis/pfc/marked_model_family.py:609-799](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L609) — `build_recommended_event_mark_table`; the raw products at `:792-797`.
- [src/continuum_swr_replay/analysis/pfc/marked_model_family.py:980-1080](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L980) — `build_marked_family_ladder`, where standardization happens and where the products must now be formed.
- [src/continuum_swr_replay/analysis/pfc/marked_glm.py:785-915](../../../../src/continuum_swr_replay/analysis/pfc/marked_glm.py#L785) — `standardize_event_marks` (`allow_missing`, `allow_constant`, `min_marked_events`).
- `.claude/docs/reviews/2026-09-19-pfc-marked-family-corpus.md` — the correlation and blow-up numbers this phase is answering.

**Contracts referenced:**

- [Construct mark names](shared-contracts.md#construct-mark-names) — the new names and what they replace.

## Tasks

- **Distance mark.** In `build_recommended_event_mark_table`, add `CONTENT_DISTANCE = "content_distance_cm"` from `event_table.mean_abs_ahead_behind_cm` (require it in `event_required`, validate finite and ≥ 0; no log transform: the mark is standardized within class, and its distribution on the corpus is roughly uniform over 0–230 cm). Add its `MARK_DESCRIPTIONS` sentence.

- **Re-specify the models.** Edit `RECOMMENDED_MODEL_SPECS`:
  - `spatial_role`: targets `(CONTENT_DISTANCE, PREVIOUS_ARM_FRACTION)`; adjustments `(EVENT_DURATION_MARK, LEAF_CONTENT_FRACTION, DECODE_ENTROPY)`. `current_arm_fraction` and `current_patch_fraction` leave this model (the single-mark runner keeps `current_arm_fraction`).
  - `trajectory_geometry`: targets `(LOG_SPATIAL_EXTENT, RETROSPECTIVE_BIAS)`; adjustments `(EVENT_DURATION_MARK, DECODE_ENTROPY, LOG_REVERSALS, PAUSE_FRACTION)`.
  - Everywhere `CURRENT_ARM_FRACTION` is an adjustment (`belief_content`, `reward_memory`, `novelty_content`, `within_stop_timing`, `switch_context`, `reward_context`), replace it with `CONTENT_DISTANCE`.
  - Interactions: `exploration_content_interaction` targets `(DISTANCE_X_SWITCH, UNCERTAINTY_X_SWITCH)`, `reward_content_interaction` `(DISTANCE_X_REWARD, UNCERTAINTY_X_REWARD)`, `coupling_content_interaction` `(DISTANCE_X_RIPPLE, UNCERTAINTY_X_RIPPLE)`, each with both constituents as adjustments, as now.
  - Update `test_recommended_specs_share_compiled_widths_within_each_tier` deliberately (it pins the inventory): the compact tier stays at six marked predictors per class and the wide tier at nine; if a count changes, change the test and say why in its docstring.

- **Centered interaction products.** Give `MarkedFamilySpec` a field `interactions: Mapping[str, tuple[str, str]] = MappingProxyType({})` mapping an interaction mark name to its two constituents; `__post_init__` requires both constituents to be in `all_marks`. Remove the six raw product columns from `build_recommended_event_mark_table` (`:792-797`) and their constants. In `build_marked_family_ladder`, after `standardize_event_marks` has produced `<a>_z` and `<b>_z` for the constituents, form each interaction as

  ```python
  frame[f"{name}"] = frame[f"{a}_z"] * frame[f"{b}_z"]
  frame.loc[~(frame[f"{a}_available"] & frame[f"{b}_available"]), name] = np.nan
  ```

  and then standardize the interaction marks with a second `standardize_event_marks` call (`allow_missing` for all of them, same `min_marked_events`), so the product is centered and unit-SD within class like any other mark. Record both scale passes in `ladder.scales`.

- **Rare-level eligibility.** In `build_marked_family_ladder`, before standardizing, for every mark named in `spec.required_varying_nuisance_marks` (the binary modifiers) and every target that is binary (at most two distinct finite values), require at least `MIN_LEVEL_COUNT = 10` finite events at each level in each class; otherwise raise `ModelIneligibleError` naming the model, class, mark and the level count. Add `MIN_LEVEL_COUNT` next to `MIN_MARKED_EVENTS_PER_CLASS` in `marked_glm.py` and record it in the fit contract.

- **`scripts/pfc/measure_pfc_mark_context.py` and the family sidecar's `event_mark_geometry_by_class`** need no code change; the geometry diagnostic will show the new correlations. Add one QA assertion to phase 1c's audit instead (see that phase).

- **Docs.** Regenerate the model table in `docs/design/pfc_marked_model_family.md` ("Recommended models") from the new specs; add the distance mark to "Derived content marks"; replace the sentence on raw products with the centered-product rule and the reference-level consequence (the main effect is now the response at the modifier's class mean); state the rare-level rule next to the ten-event rule.

## Deliberately not in this phase

- No baseline columns (phase 1a) and no refit (phase 1c).
- No marginal specs (phase 2), although the new `interactions` field is what phase 2's generated specs will reuse.
- `current_arm_fraction` stays defined and stays the single-mark runner's mark; only the family stops using it as a target.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_distance_mark_is_the_mean_absolute_represented_distance` | on the adversarial fixture, `content_distance_cm` equals the event table's `mean_abs_ahead_behind_cm` row for row; a negative value raises. |
| `test_interaction_is_the_centered_product_of_standardized_constituents` | for a two-class fixture with known marks: the interaction column before its second standardization equals `z_a * z_b` computed by hand; after it, mean 0 and SD 1 within class; NaN where either constituent is unavailable. |
| `test_interaction_has_low_correlation_with_its_constituents` | on a fixture where `b` is 0/1 with 70 % ones, `|corr(z_a*z_b standardized, z_a)| < 0.2` and likewise for `z_b` (the raw product on the same fixture exceeds 0.8 — assert that too, so the test documents what changed). |
| `test_binary_modifier_needs_ten_events_at_each_level` | nine switch events among 100 → `ModelIneligibleError` naming `switch_patch_context`, the class and "9"; ten → fits. |
| `test_binary_target_needs_ten_events_at_each_level` | same rule applied to a binary target (`rewarded_context`). |
| `test_specs_carry_their_interaction_constituents` | every interaction model's `interactions` map names constituents that are in its adjustments; every core model's targets are in `MARK_DESCRIPTIONS`. |
| `test_recommended_specs_share_compiled_widths_within_each_tier` (edited) | the new inventory: six compact / three wide / three interactions with the stated widths. |
| `test_measure_pfc_mark_context_reports_the_distance_mark` | `scripts/pfc/measure_pfc_mark_context.py` gains `content_distance_cm` alongside `arm_fraction` in its correlation table on the synthetic frame. |

## Fixtures

The adversarial `_tables()` fixture in `tests/analysis/pfc/test_marked_model_family.py:55` gains a `mean_abs_ahead_behind_cm` column; the `_eligibility_inputs` helper (`:554`) gains a 100-event variant with a controllable number of switch events.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the six raw-product columns and their constants are gone.
- User-facing documentation listed as tasks is updated, not deferred.
