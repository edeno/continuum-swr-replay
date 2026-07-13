# Phase 4 — Per-animal inference, Fig 4 regen, claim reconciliation

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Turn the regenerated artifacts into the corrected Figure 4: aggregate inference
at the animal level instead of pooling ~11k neurons, add the matched-control
engagement panel, retitle to the engagement claim, and reconcile the claims
table, figure-claims plan, and the review doc.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/load_population.py:138-222](../../../../src/continuum_swr_replay/analysis/pfc/load_population.py#L138) — population assembly: `animal` column at line 139 and legacy pooled nominal-p FDR at lines 161-213. Replace headline use with Phase-2b empirical-null results.
- [scripts/figures/generate_figure04.py:240-360](../../../../scripts/figures/generate_figure04.py#L240) — panel d `% significant` computed as `df[col] < 0.05` pooled (line 261); title (line 2); example-neuron event construction (lines 330-357, already fixed in Phase 1); output dir `paper/figures/figure04` (line 474), `save_figure` (line 493).
- [paper/claims_table.md:57-58](../../../../paper/claims_table.md#L57) — Goal 3c / claim #20 row. [paper/figure_claims_plan.md:42](../../../../paper/figure_claims_plan.md#L42) — Fig 4 line. [.claude/docs/reviews/05_pfc_cross_area_review.md](../../reviews/05_pfc_cross_area_review.md) — the review to mark resolved.

**Contracts referenced:**

- [Provenance sidecar schema](shared-contracts.md#provenance-sidecar-schema) — the loader guard consumes it here.

## Tasks

- **Add the loader stale-artifact guard** in `load_population_data`. For each consumed pickle, require the sibling `.provenance.json` to exist and have `is_swr_definition == "canonical_event_table"`; raise a clear `ValueError` naming the session otherwise (points the user at Phase 3's rerun). Also raise if any consumed pickle still carries the removed non-nested LR column `"SWR + non_local_no_SWR vs. non_local: pval"`. (Review Next Check #5.)
- **Load the matched-control and Phase-2b inference fields** alongside existing modulation arrays. Correct the prespecified empirical-null contrast family; retain nominal LR/FDR fields only as descriptive diagnostics.
- **Add per-animal aggregation.** New function `aggregate_by_animal(df, tests) -> pd.DataFrame` in `load_population.py`: group by the `animal` column, and per animal per test report `n_neurons`, `n_significant` (FDR q < 0.05), and `frac_significant`. Return a tidy (animal × test) frame. Keep the existing pooled columns for reference, but the figure consumes the per-animal frame. This implements decision 3 (per-animal-then-combine): the headline is 5 per-animal fractions plus a combined animal-level statistic (e.g. proportion of animals with frac_significant above a stated threshold, and a simple sign/rank test across the 5 animals — pick and document one; do not report a single pooled p-value as the primary result).
- **Rewrite Fig 4 for engagement framing:**
  - Retitle ([generate_figure04.py:2](../../../../scripts/figures/generate_figure04.py#L2) and any on-figure title) from "respond to replay regardless of SWR detection" to an engagement claim, e.g. "Non-SWR replay engages mPFC neurons".
  - Panel d: replace the pooled `% significant` (line 261) with **per-animal** empirical-null effect/fraction summaries (5 points / small multiples or a strip over the bars), and add the **`non_local_no_SWR vs. matched_control`** engagement test as the headline. Include held-out deviance and pre-event placebo results in the caption/supplement; state animal imbalance explicitly.
  - Keep panels a–c (example neuron, population heatmaps, population average) but drop any language/annotation implying SWR≈non-SWR equivalence.
  - Regenerate to `paper/figures/figure04/` via `save_figure` (line 493).
- **Reconcile the claim + plan text** (docs are part of this phase, not deferred):
  - [paper/claims_table.md:58](../../../../paper/claims_table.md#L58) row 20 — update status from "have for engagement; control/equivalence recut pending" to reflect the matched control + animal-level inference now implemented; keep it an *engagement* claim.
  - [paper/figure_claims_plan.md:42](../../../../paper/figure_claims_plan.md#L42) — change the Fig 4 title line to the engagement framing; note the matched-control panel.
  - [.claude/docs/reviews/05_pfc_cross_area_review.md](../../reviews/05_pfc_cross_area_review.md) — append a resolution note mapping each of the five Next Checks to the phase/PR that closed it.

## Deliberately not in this phase

- OFC (Non-Goal). Do not add an OFC glob or OFC panels.
- A powered TOST / equivalence test (Non-Goal — engagement only).
- Re-running the batch (Phase 3 produced the artifacts this phase reads).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_loader_rejects_missing_provenance` | A pickle with no sibling `.provenance.json` → `ValueError` naming the session. |
| `test_loader_rejects_stale_lr_column` | A pickle containing the removed non-nested LR column → `ValueError`. |
| `test_aggregate_by_animal_shape` | `aggregate_by_animal` returns one row per (animal, test) with `n_neurons`, `n_significant`, `frac_significant`; sums of `n_neurons` per test equal the filtered population size. |
| `test_aggregate_by_animal_matches_pooled` | Summing `n_significant` across animals per test equals the pooled significant count (guards a grouping bug). |
| `test_matched_control_empirical_null_registered` | corrected empirical-null field, held-out score, and pre-event placebo are present for valid neurons. |
| `generate_figure04.py` smoke (slow) | Runs against regenerated artifacts; writes `figure04.pdf/.png`; title contains "engages" not "regardless"; panel d has 5 per-animal points. |

## Fixtures

- A small synthetic `PROCESSED_DATA_DIR` with 2–3 sessions across 2 animals, each
  with the 5 `.npy` files, a pickle, and a valid `.provenance.json` — plus one
  session missing its sidecar and one carrying the stale LR column, for the guard
  tests. Build in `tests/analysis/pfc/conftest.py`; reuse the existing
  `load_population` test fixtures and extend.
- The real regenerated `PROCESSED_DATA_DIR` for the figure smoke test.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` **and** `scientific-ux-reviewer` (the figure is user-facing) against the diff. Confirm:
- The figure title and captions carry the engagement claim; no "regardless"/equivalence language survives anywhere (grep the figure script + captions).
- Panel d reports per-animal inference, not a single pooled percentage, and the matched-control engagement test is the headline.
- The loader guard actually blocks stale artifacts (both guard tests fail-if-removed).
- `claims_table.md` row 20, `figure_claims_plan.md`, and the review doc are all updated in this PR.
- Docstrings/tests/figure titles don't reference this plan or "Phase 4".
