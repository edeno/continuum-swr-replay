# Phase 3 — Coherent Candidate Review and Scientific Impact Verdict

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#generation-coherent-candidate-review)

**Inputs to read first:**

- The completed Phase 2 `docs/audits/replay-tables/<date>-<shortsha>/{final_summary.json,final_findings.csv,manual_qc.csv,final_report.md}` — no unresolved hard/identity/manual concern may be ignored; a legacy `POSTERIOR_PARTICIPATION_UNATTESTED` row proceeds only with this phase recorded as its resolution owner.
- `.claude/docs/plans/regression-integrity-fixes/phase-7-run-artifacts.md` — source-attested session bundles, expected-session accounting, atomic noncanonical runs, component resolution, and the prohibition on legacy fallback.
- `.claude/docs/plans/regression-integrity-fixes/phase-9a-cohort-identity.md` and `phase-9b-blocked-comparisons.md` — exact key tables, `AnalysisIdentity`, strict allowed-row fitting, and shared folds/transforms.
- `.claude/docs/plans/regression-integrity-fixes/phase-10a-estimability.md` and `phase-10b-dependence-inference.md` — fit/term validity, robust uncertainty, animal-balanced summaries, and multiplicity.
- `.claude/docs/plans/regression-integrity-fixes/phase-11-decision-models.md` — replacement for the invalid stop-grain choice and Stage-6 analyses.
- `.claude/docs/plans/regression-integrity-fixes/phase-12-swr-power.md` — identified power estimands; these do not replace claim #24's timepoint-density analysis.
- `.claude/docs/plans/regression-integrity-fixes/phase-13-refit-reporting.md` — sole production rebuild/refit/promotion/paper-cutover owner; this PR supplies a nonpromoting table gate.
- `src/continuum_swr_replay/analysis/replay/tables.py:215-1751` and `prepare_data.py:199-506` — table calculations/assembly under review; no formula edits are allowed here.
- `src/continuum_swr_replay/analysis/replay/posterior_io.py:1-90`, `summaries.py:59-317`, and `uncertainty.py:28-204` — source posterior loading and production rollups that the independent spot-check must not call to compute expectations.
- `scripts/content/analyze_content_occupancy.py:1-166` — claim #14's mixed canonical-arm/raw-dot components.
- `paper/claims_table.md:25,42-57,61-64` and `paper/figure_claims_plan.md:110-125` — atomic claims, current numbers, unresolved estimator identities, and mixed-owner rows.

**Contracts referenced:**

- [Review boundary and ownership](shared-contracts.md#review-boundary-and-ownership) — consume regression-integrity infrastructure; do not duplicate it.
- [Artifact isolation and provenance](shared-contracts.md#artifact-isolation-and-provenance) — one source-attested, generation-coherent, noncanonical run and unchanged legacy artifacts.
- [Comparison policy](shared-contracts.md#comparison-policy) — exact keyed generation comparison and expected-change discipline.
- [Claim-impact and verdict contract](shared-contracts.md#claim-impact-and-verdict-contract) — corrected estimators, cross-owner completion, deterministic precedence, and final states.

**Designs referenced:** [prerequisite ownership](designs.md#prerequisite-ownership), [staged rebuild](designs.md#staged-rebuild), [keyed comparison](designs.md#keyed-baselinecandidate-comparison), and [corrected claim-impact review](designs.md#corrected-claim-impact-review).

## Tasks

- Add a hard prerequisite gate before creating a review ID:
  - load Phase 2's complete findings/manual dispositions and reject unresolved hard/identity/fail/concern rows;
  - validate completion/schema manifests for regression-integrity Phases 7, 9a, 9b, 10a, 10b, 11, and 12;
  - require the exact Phase-7 expected-session/source-audit manifest and a clean/frozen code/config identity;
  - verify every required producer and consumer resolves through Phase-7 `run_artifacts` and no direct fixed-path fallback remains;
  - load a locked claim-spec registry created before candidate results. A missing prerequisite produces a machine-readable `INCONCLUSIVE` reason; this phase does not implement a substitute.

- Create `config/replay_table_claim_specs.json` with one row per atomic table-dependent estimand. Each row declares claim/location/status/owner, source builder/table/columns, corrected analysis identity/comparison set/grain, corrected estimator/model/term/output, direction/order/equivalence rule, animal coverage, multiplicity family, and completion-manifest requirement. Initial coverage includes #2, #13, the table component of #14, #15-#19, #21, and #22; #24 is a cross-owner row pointing to the SWR-continuum timepoint-density analysis. Validate exact uniqueness and refuse unknown model/term/output identities.

- Resolve currently ambiguous claims before opening results, not in response to them:
  - #15-#18 need author-locked formula, transform, duration control, clustering unit, corrected uncertainty, and saved output identity; there is currently no named runner that reproduces the cited GEE numbers, so absent approval they remain `unresolved`;
  - #13 needs a locked event-distance column, units, event/time-bin weighting, and place-field-width boundary before its two-regime claim is recalculated;
  - #19b must map the prose `recent_reward_rate` to the exact corrected arm predictor rather than compare stale wording to a different term;
  - #19f's cited `rew_x_current`, `value_x_current`, `switch_x_current`, time-control, and graded-value/previous-goal effects must each exist in an approved permanent corrected spec. The current permanent SWR-contrast ladder is not assumed to reproduce the removed exploratory analysis;
  - #24 maps to Local-vs-Non-Local timepoint SWR-z density and its existing owner, not either legacy or identified event-level SWR-power regression.
  An unresolved mapping is visible in the registry and forces the overall verdict to `INCONCLUSIVE`.

- Create `src/continuum_swr_replay/analysis/replay/table_generation_comparison.py` as a table-specific consumer of Phase-7 run components and Phase-9 typed keys. Implement separate baseline/candidate validation; natural-key outer comparison; schema/dtype, exact, NaN-mask, and numerical deltas; per-session/animal cohort summaries; expected-change IDs; and candidate bundle/standardization/auxiliary identities. Do not implement another run manifest, path resolver, cohort hasher, fitter, or fold builder.

- Create `src/continuum_swr_replay/analysis/replay/table_claim_impact.py` to resolve locked claim specs against corrected Phase-9-12 outputs and external owner completion manifests, compute natural/intersection cohort deltas, and render the machine-readable/Markdown impact matrix. It consumes existing `AnalysisIdentity`, strict fitter outputs, fit/term status, robust intervals, animal summaries, and FDR fields; it does not calculate legacy p-values or infer missing specs from output filenames.

- Create `scripts/qa/review_replay_table_candidate.py` as a thin state-machine CLI over Phase-7 `run_artifacts` plus the two table-specific modules. Its stages are `init`, `synthetic`, `stress`, `posterior-spotcheck`, `source-gate`, `candidate`, `compare`, `claim-impact`, and `verdict`, with `run --through`. Every transition verifies code/source/component/canonical hashes. Phase-7 finalization atomically renames a complete `.staging` directory; the driver never advances `CURRENT.json`, and any drift/failure requires a new review/candidate ID.

- Add a checked-in stress manifest beside the QA script with the eight fixed session IDs/reasons in the [design](designs.md#staged-rebuild). Verify their actual current properties at runtime rather than hard-coding event counts as test expectations.

- Run synthetic and stress gates through the same production builders and Phase-7 component API:
  - reuse all Phase-1 adversarial tests, including no-stop/zero-event, multi-bout label rejection, MUA/trajectory availability, malformed posterior, unmapped MAP, posterior-junction mass, and arbitrary indices;
  - construct session-scoped stress components for replay tables plus any behavior/Q/onset dependency required by their columns;
  - compare every raw/unstandardized field and identity. Do not compare per-session all-NaN z placeholders with pooled canonical z. As a wiring diagnostic only, apply the frozen legacy pooled reference to stress raw columns; full candidate z is computed from the full candidate population and compared later.

- Add an independent, stratified NetCDF source-parity check that closes Phase 2's intentional gap:
  - freeze exact event/window keys before opening results, using the stress manifest and Phase-2 QC selections; no outcome-driven replacement sampling;
  - lazily open only required NetCDF slices with xarray, validate dimension/time/position coordinates against raw `dot`/environment, and independently collapse position bins to arms without importing production posterior helpers;
  - compare source-derived sequence arm vectors/MAP, event peak/entropy/HPD/multimodality, and event-arm posterior mass/fractions to the candidate under shared tolerances;
  - a missing/changed selected key or unreadable selected posterior yields `INCONCLUSIVE`.

- Require Phase-7's complete 218-session source audit and expected-session accounting. Do not add a weaker five-session Spyglass/cache freshness check. Canonical before/after hashes are captured independently at every stage.

- Build/finalize one complete noncanonical candidate generation using Phase-7 producers after one-session/stress resource review. It includes, under the same source/table generation hash:
  - behavior timebase/session crosswalk and behavior/Q components;
  - all five replay tables plus pooled standardization reference and the shared ContFrag patch-projection cross-check;
  - event-onset and stop-timebin components;
  - Phase-11 decision tables;
  - corrected model runs required by locked claim specs.
  If a corrected event/stop population changes onset ranks, q-value joins, or decisions, regenerate those dependent components; never pair candidate tables with frozen legacy auxiliaries. Component finalization is directory-atomic and expected cells/sessions must be complete or explicit failures.

- Compare the read-only current materialized generation to the candidate by natural keys. Treat the baseline as legacy/invalid evidence, not an acceptance target. Explicitly adjudicate retired HMM columns, the newly explicit patch natural key, probability overshoot, full/retained windows, junction examples, MUA availability, stop exposure/gaps, trunk fallback, pooled z reference, and every Phase-2 finding. No unexpected delta is waived because a coefficient looks stable.

- For each atomic model claim, use only the existing corrected APIs:
  1. materialize each generation's natural exact key table with Phase-9a `AnalysisIdentity`;
  2. create a typed cross-generation intersection-key manifest;
  3. run the corrected strict fitter on natural and intersection keys, failing on any internal drop;
  4. use Phase-9b shared folds/training-only transforms and Phase-10 fit/term/robust/animal/FDR outputs;
  5. keep legacy-generation results diagnostic and unavailable when a coherent legacy auxiliary does not exist—never borrow a candidate Q/onset component.

- Route claims to corrected owners:
  - deterministic table summaries for #2/#13/#21/#22;
  - split table/raw owner results for #14;
  - locked descriptive/inferential specs for #15-#18;
  - corrected content/rate/uncertainty outputs for #19a/#19b/#19c/#19f/#19g;
  - Phase-11 decision-grain outputs for #19d/#19e; the old stop-selected choice/Stage-6 fits are noncertifying diagnostics only;
  - the SWR-continuum owner's completion manifest for #24. Phase-12 power may appear as its own registered result, never as #24.

- Produce `claim_impact.csv` first, render `claim_impact.md`, then produce `verdict.md` and a machine-readable verdict. Include source/run/component hashes, Phase-1/2 findings/manual review, posterior spot-check, table/auxiliary/cohort deltas, Phase-9-12 identities/validity/inference, owner completion, and every atomic wording action. Apply the shared deterministic precedence exactly: hard/identity/provenance → `FAIL`; missing prerequisite/oracle/spec/coherent component/corrected estimator/owner → `INCONCLUSIVE`; correct tables plus any unsupported/reversed/wording-changing claim → `PASS—PAPER REVISION REQUIRED`; only complete supported evidence → `PASS`.

- Add `docs/replay_table_candidate_review.md` documenting prerequisites, run-component layout, stages/restarts, posterior spot-check, natural/intersection cohorts, claim ownership, verdict precedence, and Phase-13 handoff. Commit only sanitized table-review manifests/comparisons/claim matrix/verdict under `docs/audits/replay-tables/<date>-<shortsha>/`; large run components/refits/logs remain in the configured noncanonical run root.

## Deliberately not in this phase

- Adding generic path flags, cache modes, run manifests, cohort/fold implementations, fit diagnostics, robust inference, decision models, power formulas, or promotion logic already owned by regression-integrity Phases 7-12.
- Editing `tables.py`, direct table-helper formulas, model specs, thresholds, exposure/window/fallback/junction policies, or expected claim directions. A defect returns to a reviewed table/model-owner PR and a new candidate ID.
- Certifying a claim with legacy stop-grain choice/Stage-6, unidentified power OLS, session-Wilcoxon/model-based SE, p-value parity, or an exploratory term absent from the locked corrected registry.
- Freezing legacy q-values/event-onsets for a changed candidate, silently dropping sessions/posteriors/cells, or treating a count-only 218-session check as identity.
- Advancing `CURRENT.json`, copying candidate files into canonical paths, editing manuscript numbers/figures, or promoting artifacts. Phase 13 and the named claim owners retain those responsibilities.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_table_review_prerequisites.py` | Missing/incompatible Phase-7/9/10/11/12 or Phase-2 evidence yields typed `INCONCLUSIVE`, never a private fallback |
| `test_table_review_run_state.py` | Phase-7 `.staging` finalizes directory-atomically, state/hash drift requires a new ID, and `CURRENT.json` never changes |
| `test_table_generation_comparison.py` | Natural-key schema/key/NaN/exact/numeric/cohort deltas and expected-change IDs are complete and row-order invariant |
| `test_stress_standardization_policy.py` | Raw stress values compare exactly; per-session z placeholders are excluded; frozen-reference diagnostic and full pooled reference are distinct |
| `test_posterior_source_spotcheck.py` | Literal xarray slices independently reproduce arm vectors/MAP and event/arm posterior summaries; production helpers are not imported |
| `test_generation_coherence.py` | Replay/Q/onset/time-bin/decision/model components share source/table generation; stale cross-generation auxiliary fails resolution |
| `test_cross_generation_cohorts.py` | Phase-9 typed natural/intersection key manifests reach strict fitters unchanged; same-N/different-key and internal drops fail |
| `test_claim_spec_resolution.py` | Unknown/missing #15-#18/#19f terms, stale #19b wording, legacy choice/stay, or power-as-#24 substitution remain unresolved |
| `test_claim_impact.py` | Corrected fit/term/robust/animal/FDR/cohort fields and cross-owner completion map to atomic support states exactly |
| `test_verdict_precedence.py` | `FAIL`, `INCONCLUSIVE`, `PASS—PAPER REVISION REQUIRED`, and `PASS` precedence covers correct-table/unsupported-claim and obsolete-estimator cases |
| Stress/candidate run (`external`, `slow`) | Eight fixed sessions then all expected sessions finalize complete coherent noncanonical components with unchanged legacy hashes |
| Independent result review | One posterior slice, one table delta, one cohort manifest, and one claim row are traced manually to source/output evidence |

Run code gates before expensive execution:

```bash
uv run pytest -q \
  tests/unit/test_table_review_prerequisites.py \
  tests/unit/test_table_review_run_state.py \
  tests/unit/test_table_generation_comparison.py \
  tests/unit/test_stress_standardization_policy.py \
  tests/unit/test_posterior_source_spotcheck.py \
  tests/unit/test_generation_coherence.py \
  tests/unit/test_cross_generation_cohorts.py \
  tests/unit/test_claim_spec_resolution.py \
  tests/unit/test_claim_impact.py \
  tests/unit/test_verdict_precedence.py
uv run python scripts/qa/review_replay_table_candidate.py init \
  --review-root scratch/tables-review --review-id <review_id> \
  --phase2-audit docs/audits/replay-tables/<audit_id> \
  --expected-session-manifest <phase7_expected_sessions>
uv run python scripts/qa/review_replay_table_candidate.py run --review-id <review_id> --through stress
uv run pytest -q -m slow tests/integration/test_replay_tables_stress_candidate.py
```

After reviewer/resource gates:

```bash
uv run python scripts/qa/review_replay_table_candidate.py posterior-spotcheck --review-id <review_id>
uv run python scripts/qa/review_replay_table_candidate.py source-gate --review-id <review_id>
uv run python scripts/qa/review_replay_table_candidate.py candidate --review-id <review_id>
uv run python scripts/qa/review_replay_table_candidate.py compare --review-id <review_id>
uv run python scripts/qa/review_replay_table_candidate.py claim-impact --review-id <review_id>
uv run python scripts/qa/review_replay_table_candidate.py verdict --review-id <review_id>
uv run pytest -q -m "not slow"
uv run ruff check src/continuum_swr_replay/analysis/replay/table_generation_comparison.py src/continuum_swr_replay/analysis/replay/table_claim_impact.py scripts/qa/review_replay_table_candidate.py
```

## Fixtures

- Reuse the Phase-1 hand oracle and Phase-2 literal audit bundle; expected values never come from production builders/audit registries.
- A tiny Phase-7 run root with an existing current release, incomplete `.staging`, finalized noncurrent candidate, coherent/mixed-generation components, expected-cell failures, and immutable pointers/hashes.
- Synthetic baseline/candidate tables with row reordering, expected schema removal/addition, patch natural key, added/dropped rows, NaN/numeric changes, one invalid cross-table identity, and distinct pooled z references.
- A tiny xarray NetCDF/environment/raw-dot fixture with literal posterior position bins, junction mass, one-hot/diffuse/tied rows, and hand-computed event/arm summaries; import sentinels forbid production posterior helpers.
- The fixed eight-session stress CSV and Phase-7 exact expected-session manifest; real data remain external read-only inputs.
- Synthetic corrected model/owner outputs covering valid/invalid fit terms, natural/intersection cohort changes, 5/5→4/5, equivalence-supported/unresolved, ordering reversal, FDR crossing, missing owner, and legacy-only estimator.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- A scientific reviewer confirms every claim spec names a corrected estimand/owner before results, legacy estimators are noncertifying, and no missing #15-#18/#19f identity was improvised.
- A provenance reviewer independently verifies one Phase-7 component chain, canonical before/after hashes, one NetCDF slice, one keyed table delta, one natural/intersection cohort, and one claim-impact row.
