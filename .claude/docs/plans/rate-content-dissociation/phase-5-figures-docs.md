# Phase 5 - Figures, docs, and claim ledger

[<- back to PLAN.md](PLAN.md) . [overview](overview.md)

Turn the audited analyses into paper-facing outputs. This phase should be
boring: it reads already-validated tables, refuses stale inputs, and produces
figures/docs that state exactly what the rate, content, and choice models ask
and answer.

**Questions:** What is the final supported claim set? Which coefficient tables
and artifacts support each claim? Are rate/content/choice effects shown on
their own scales without overinterpretation?

**Inputs to read first:**

- Phase 1 audit report and provenance manifests.
- Phase 2 summary table.
- Phase 3 dissociation report.
- Phase 4 choice report, if behavior claims passed the gate.
- `docs/models_table.md` - current model-family documentation.

## Tasks

- Add a figure/report runner, e.g.
  `scripts/figures/generate_rate_content_dissociation.py`.
  - Fail if any source artifact has `stale_flag=True`.
  - Fail if required provenance manifests are missing.
  - Read only Phase 2-4 summary/report tables, not raw model pickles.
  - Treat promoted continuous-SWR outputs as valid only if they carry the same
    provenance, availability, and stale-flag fields as the binary-SWR outputs.
- Produce a compact multipanel figure:
  - rate reward gate: SWR-overlap vs non-SWR immediate reward and delayed
    reward-history kernel;
  - content locality/task-state panels: current/current-patch and selected HMM
    terms;
  - rate-content classification panel;
  - optional choice-prediction panel if Phase 4 passes.
- Add `docs/replay_rate_content_synthesis.md`.
  - State the question each model family asks.
  - State supported findings, unsupported findings, and blocked/stale items.
  - Include links to source artifacts and provenance.
- Update `docs/models_table.md` only where the audited outputs change the
  interpretation or artifact status.
- Add a claim ledger:
  - `Processed-Data/replay_content/model_summaries/rate_content_claim_ledger.csv`;
  - each row maps a claim to terms, source artifacts, hashes, and status.

## Gate

The phase is complete only when:

- figures regenerate from a clean command;
- no figure panel uses stale or missing-provenance artifacts;
- every textual claim maps to a claim-ledger row;
- behavior claims are omitted or clearly marked unsupported if Phase 4 failed.

## Deliberately not in this phase

- No new model fitting except optional smoke checks needed by figure code.
- No manual coefficient transcription into docs.
- No unsupported causal phrasing.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_claim_ledger_requires_sources` | Every claim row has source terms and artifact hashes. |
| `test_figure_runner_rejects_stale` | Figure generation fails on a stale source artifact. |
| `test_docs_claims_have_ledger_rows` | Supported claims in the synthesis doc map to ledger entries. |
| figure smoke | Runner emits the expected figure and markdown using audited summary fixtures. |

## Fixtures

- Minimal Phase 2-4 summary tables with two supported claims and one blocked
  stale claim.
- Tiny provenance manifests for figure-runner tests.

## Review

Confirm the final doc distinguishes "reward changes replay rate" from "reward
changes represented content," and distinguishes both from "replay content
predicts choice." The final claim set should be narrower than the exploratory
coefficient readout if any gate fails.
