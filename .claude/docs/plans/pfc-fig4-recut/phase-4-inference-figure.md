# Phase 4 — Per-animal inference, Fig 4 regen, claim reconciliation

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Turn the regenerated artifacts into the corrected Figure 4: aggregate inference
at the animal level instead of pooling ~11k neurons, add the matched-control
engagement panel, retitle to the engagement claim, and emit a machine-readable
claim-20 adjudication for the global release publisher. This phase is also the sole
producer of `pfc_completion.json`, which is installed only after all PFC data,
inference, staged-figure, and test gates below succeed. It does not mutate shared
Markdown or live paper-figure paths before global promotion.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/load_population.py:138-222](../../../../src/continuum_swr_replay/analysis/pfc/load_population.py#L138) — population assembly: `animal` column at line 139 and legacy pooled nominal-p FDR at lines 161-213. Replace headline use with Phase-2b empirical-null results.
- [scripts/figures/generate_figure04.py:240-360](../../../../scripts/figures/generate_figure04.py#L240) — panel d `% significant` computed as `df[col] < 0.05` pooled (line 261); title (line 2); example-neuron event construction (lines 330-357, already fixed in Phase 1); output dir `paper/figures/figure04` (line 474), `save_figure` (line 493).
- [paper/claims_table.md:57-58](../../../../paper/claims_table.md#L57) — Goal 3c / claim #20 row. [paper/figure_claims_plan.md:42](../../../../paper/figure_claims_plan.md#L42) — Fig 4 line. [.claude/docs/reviews/05_pfc_cross_area_review.md](../../reviews/05_pfc_cross_area_review.md) — the review to mark resolved.
- Phase 3's machine-readable batch audit plus the exact frozen Phase-13
  canonical-table-generation manifest — the source/session coverage evidence that
  completion must re-read and hash rather than trust by filename.

**Contracts referenced:**

- [Provenance sidecar schema](shared-contracts.md#provenance-sidecar-schema) — the loader guard consumes it here.
- [PFC completion manifest](shared-contracts.md#pfc-completion-manifest) — this
  phase is its sole producer; the regression release gate consumes it.

## Tasks

### Pre-freeze implementation PR

- Add the loader stale-artifact guard in `load_population_data`. Require the requested `pfc_run_id`,
  frozen generation, per-session input/event hashes, and canonical event tag; reject missing sidecars
  and the retired non-nested LR column.
- Load matched-control and Phase-2b inference fields. Preserve nominal LR/FDR only as explicitly
  descriptive diagnostics.
- Add `aggregate_by_animal(df, tests)`: neuron→session summaries, then equally weighted sessions
  within animal. Report all five animal effects for
  `non_local_no_SWR vs matched_control` and the prespecified one-sided exact binomial sign test;
  discard exact-zero ties from its denominator but display them and the resulting discrete
  resolution. Never substitute pooled-neuron inference.
- Rewrite `generate_figure04.py` for engagement framing and the per-animal matched-control panel.
  Its production interface requires explicit `--release-candidate`, `--pfc-run-id`, and
  `--output-root`; it writes only
  `<regression_root>/release_outputs/<release_id>.staging/paper/figures/figure04/`.
  Its normal post-release read path uses Phase 7's
  `resolve_release_component("pfc")` through global `CURRENT.json`; no PFC-specific pointer or glob
  fallback exists.
- Add deterministic rendering of `claim_20_adjudication.json` with status
  `supported`, `not_supported`, or `unavailable`, exact estimand/effect/interval/animal values, and
  prespecified text fields for claim row 20, the Figure-4 plan row, and the review-resolution note.
  Phase 13 mechanically composes shared Markdown from this record; neither phase improvises wording
  after results.
- Add `scripts/qa/finalize_pfc_completion.py`. It independently re-reads the frozen candidate,
  Phase-3 batch audit, every expected artifact/sidecar, the complete fit-diagnostics table,
  population summaries, staged Figure-4 PDF/PNG, and `claim_20_adjudication.json`. It verifies the
  candidate's clean git identity, exact session/animal accounting, every attempted fit status,
  rank/convergence/headline eligibility, source/control/null contracts, zero stale LR columns,
  output hashes, and smoke/test results. It writes the completion schema last and atomically
  finalizes only the PFC run.
- Land all loader, aggregation, figure, adjudication, finalizer, resolver-adapter, and synthetic-test
  code before the shared clean code/config freeze. This PR contains no production Figure 4 or claims
  edit.

### Post-freeze production record — no source/config changes

- Against the explicit frozen candidate and completed Phase-3 staging run, execute the already-landed
  population aggregation and Figure-4 generator. Write PFC summaries inside the PFC run and the
  Figure-4 PDF/PNG plus `claim_20_adjudication.json` only in the global release-output staging tree.
- Run loader and figure smokes, targeted tests, and the frozen finalizer. A statistically valid null
  writes `status="not_supported"`; a failed validity gate leaves PFC incomplete, and Phase 13 records
  the optional component as unavailable.
- On success, atomically finalize `<pfc_root>/runs/<pfc_run_id>/`. Phase 13 validates its exact
  completion path/hash, composes shared Markdown in release-output staging, and promotes the global
  release. Post-promotion publication copies the already-hashed Figure/claim-derived bytes; it does
  not rerun PFC or follow a second pointer.

## Deliberately not in this phase

- OFC (Non-Goal). Do not add an OFC glob or OFC panels.
- A powered TOST / equivalence test (Non-Goal — engagement only).
- Re-running the batch (Phase 3 produced the artifacts this phase reads).
- Writing directly to `paper/figures/figure04/` or editing shared claims/review Markdown before the
  global release is promoted.
- Creating a partial or provisional completion marker. Phase-3 artifacts and
  audit output are explicitly incomplete until every Phase-4 gate passes.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_loader_rejects_missing_provenance` | A pickle with no sibling `.provenance.json` → `ValueError` naming the session. |
| `test_loader_rejects_source_generation_mismatch` | a wrong run, canonical-generation, session-input, or event-table hash fails before population aggregation |
| `test_loader_rejects_stale_lr_column` | A pickle containing the removed non-nested LR column → `ValueError`. |
| `test_aggregate_by_animal_shape` | session output is one row per `(session_id,test)` and animal output one per `(animal,test)`, with counts/effects and all five production animals |
| `test_animal_effect_is_session_balanced` | duplicating neurons within one session does not change its animal effect; duplicating a session changes only that animal's declared session summary |
| `test_exact_engagement_sign_test` | the one-sided exact sign statistic uses the five animal-level matched-control effects and reports its discrete resolution |
| `test_matched_control_empirical_null_registered` | corrected empirical-null field, held-out score, and pre-event placebo are present for valid neurons. |
| `test_figure_candidate_and_current_resolution` | explicit candidate writes only release-output staging; default reads `components.pfc` through global current; neither path uses a PFC pointer/glob |
| `generate_figure04.py` smoke (slow) | Runs against regenerated artifacts; staged `figure04.pdf/.png` title contains "engages" not "regardless"; panel d has 5 per-animal points. |
| `test_completion_not_written_on_any_failed_gate` | missing session/animal/output, fit-status/rank/convergence gap, hash drift, stale LR field, loader/figure failure, or failed targeted test leaves the final PFC run path absent |
| `test_completion_write_is_atomic_and_schema_exact` | successful finalization fsyncs a complete schema-v1 manifest and atomically renames the run; interrupted/temp writes remain staged and never become release-resolvable |
| `test_completion_accepts_valid_null_adjudication` | a fully valid null engagement result completes with `status="not_supported"` rather than being suppressed |
| `test_completion_excludes_mutable_shared_docs` | completion hashes PFC-run/release-output artifacts and adjudication only; later shared-Markdown composition cannot invalidate it |

## Fixtures

- A small synthetic `PROCESSED_DATA_DIR` with 2–3 sessions across 2 animals, each
  with the 5 `.npy` files, a pickle, and a valid `.provenance.json` — plus one
  session missing its sidecar and one carrying the stale LR column, for the guard
  tests. Build in `tests/analysis/pfc/conftest.py`; reuse the existing
  `load_population` test fixtures and extend.
- Tiny staged-generation/batch-audit/completion fixtures with one source-hash
  mismatch, incomplete animal/session coverage, a failed figure gate, and an
  interrupted temporary write. Use the declared two-animal set for unit fixtures;
  the production finalizer requires the real five-animal declaration.
- The real regenerated PFC run plus explicit release-output staging root for the figure smoke test.

## Review

Before the implementation PR, dispatch `scientific-code-reviewer` and
`scientific-ux-reviewer` against the code/synthetic outputs. Review the post-freeze production record
separately; it contains generated hashes and adjudication but no source/config changes. Confirm:
- The figure title and captions carry the engagement claim; no "regardless"/equivalence language survives anywhere (grep the figure script + captions).
- Panel d reports per-animal inference, not a single pooled percentage, and the matched-control engagement test is the headline.
- The loader guard actually blocks stale artifacts (both guard tests fail-if-removed).
- `claim_20_adjudication.json` contains the prespecified row-20/Figure-4/review text inputs; Phase 13's
  staged shared documents and post-promotion byte copies match them exactly.
- the finalized run's `pfc_completion.json` is absent until every required Phase-3/4 artifact,
  source hash, animal/session/fit-status coverage check, staged figure/adjudication, smoke, and targeted test
  passes; it is then written atomically with the exact declared schema.
- The completion manifest's source-generation hash matches the generation Phase
  13 will place in the regression release, Phase 4—not Phase 3 or Phase 13—is the only producer, and
  no independent PFC current pointer exists.
- Completion never hashes mutable live-repository files, and Figure 4 becomes live only through the
  one global release publisher.
- Docstrings/tests/figure titles don't reference this plan or "Phase 4".
