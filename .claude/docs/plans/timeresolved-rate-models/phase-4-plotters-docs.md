# Phase 4 — Scratch scripts → plotters; two-families docs note

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [scripts/_reward_glm_plot.py](../../../../scripts/_reward_glm_plot.py), [scripts/_timeresolved_kernel_glm.py](../../../../scripts/_timeresolved_kernel_glm.py), [scripts/_selfhistory_glm.py](../../../../scripts/_selfhistory_glm.py), [scripts/_timeresolved_psth.py](../../../../scripts/_timeresolved_psth.py) — the scripts to reduce; note exactly which blocks are fitting (now redundant with the orchestrator) vs plotting (kept).
- [scripts/fit_timeresolved_models.py](../../../../scripts/fit_timeresolved_models.py) — produced in Phase 3; its `models_timeresolved/all_models.pkl` is the new data source for the plotters.
- [paper/claims_table.md](../../../../paper/claims_table.md) — `#19c` already cites the time-resolved results + figure names; the docs note cross-links it.

**Contracts referenced:**

- [tidy schema](shared-contracts.md#tidy-output-schema) — the plotters read this; within-animal medians + `wilcoxon` are computed at read time here.

## Tasks

- Repoint each remaining plotter to read `models_timeresolved/all_models.pkl` (per-session coefficients) instead of re-fitting: `_reward_glm_plot.py` (→ the reward-kernel/RR figures), `_selfhistory_glm.py` (→ the autocorrelation figure). `_timeresolved_psth.py` stays as-is (it is a pure descriptive PSTH off the onset/dense tables, not a model fit — keep it, just confirm it uses the Phase-1 helpers). Delete the now-dead fitting bodies in the model scripts (name them in the PR: the `fit_pop`/per-session `smf.glm` loops).
- Decide and apply final names: either keep the `_`-prefixed scratch names or promote the surviving plotters to un-prefixed `plot_timeresolved_*.py`. Default: promote, since they are now reproducible figure-generators, not scratch. Update any references.
- Remove the fully-superseded scripts: `_reward_glm.py` and `_timeresolved_kernel_glm.py`'s statistical content now lives in the orchestrator — fold their unique figures into the plotters and delete the originals (list each deleted file in the PR).
- Write `docs/rate_models.md`: the two-families reference. Cover: Family A (`MODEL_RATE`, per-stop Poisson, `log(immobility)` offset — *the literature-comparable layer*, cite for the Ambrose/Sosa reward-rate comparison and the detected-SWR positive control); Family B (`MODEL_TIMERESOLVED`, per-bin, spline + reward kernel + self-history — *the confound-free layer*, cite for the dwell/time/history/clustering-controlled dissociation). State the result holds across both (per-stop omission 94%/5-5 → time-resolved lag-1 p≈6.5e-8). Note the deferred `reward_lag1×t` interaction (overview Open Question 3) and that the duration distribution is bimodal (reward creates the long-consummatory mode) so "duration is a behavioral variable, not just exposure" (the noted next layer).
- Link `docs/rate_models.md` from `CLAUDE.md`'s decoding/analysis section and from `paper/claims_table.md` near `#19c`.

## Deliberately not in this phase

- No new modeling (no interaction model, no hazard/departure model, no compositional analysis — all noted as future work in `docs/rate_models.md`, not built).
- No change to Family A or to `#19c`'s claims (the doc *describes*, it does not re-derive).

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run python scripts/plot_timeresolved_kernel.py` (or renamed) | Regenerates `paper/figures/reward_kernel_timeresolved.png` + `selfhistory_autocorr.png` from `all_models.pkl`; figures visually match the Phase-3 outputs (same RR kernel, same autocorr shape). |
| grep check | No surviving script contains a `smf.glm(... reward_lag ...)` fit loop — fitting lives only in `timeresolved_model.py` / `fit_timeresolved_models.py`. |
| `docs/rate_models.md` review | Names both families, both estimators, the cite-for-what mapping, and the parity statement; links resolve. |

## Fixtures

- Real-data: `models_timeresolved/all_models.pkl` from Phase 3 (the plotters read it). No new fixtures.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Plotters read the tidy output; **no fitting code remains in any script** (grep clean) — the old `fit_pop`/`smf.glm` bodies are deleted, not commented out.
- Deleted scripts are actually removed; surviving plotters renamed consistently and references updated.
- `docs/rate_models.md` is accurate and linked; figures reproduce.
- "Deliberately not in this phase" honored — no new models snuck in.
- Docstrings/names free of plan references (no "Phase 4", no `.claude/docs/plans/...`).
