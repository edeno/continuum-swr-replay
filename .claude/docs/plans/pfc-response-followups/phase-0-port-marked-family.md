# Phase 0 — Port the content-marked family onto this branch

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md)

Why: the marked family (single-mark kernels and the twelve-model family) lives on `feature/pfc-marked-event-kernels`, 694 commits behind, written against `analysis/pfc/glm.py` and other modules the 2026-10 reorganization split or moved. Phase 3 builds on it. This phase changes no science: it brings the code over, points it at the current modules and the current session assembly, and makes its tests pass. No GPU.

**Inputs to read first:**

- `git log --stat 1e434151..feature/pfc-marked-event-kernels` — the 15 commits. Port the files of `c80d55c4` through `656fac63`; **not** `66471fd3` (it changes the canonical Figure 4 population; overview Open Questions 1).
- [docs/README.md](../../../../docs/README.md) "Code layout" (line 170) and "Moved modules (2026-10)" (line 192) — the old → new module map; no compatibility shims exist. Line 17–19: where design specs and dated audits live now (`docs/design/` and `.claude/docs/reviews/`).
- [src/continuum_swr_replay/analysis/pfc/session.py:1162](../../../../src/continuum_swr_replay/analysis/pfc/session.py#L1162) `_load_aligned_session`, [`:1220`](../../../../src/continuum_swr_replay/analysis/pfc/session.py#L1220) `_session_fit_population`, [`:1283`](../../../../src/continuum_swr_replay/analysis/pfc/session.py#L1283) `_session_events` — the current assembly `run_session` (`:811`) uses: decode/position alignment, the analysed-time rule on the fit mask (bins whose ±250 ms basis window reaches excluded time are not fit), canonical events, Kay SWR intervals, artifact-dropped events, matched control.
- On the branch: `src/continuum_swr_replay/analysis/pfc/marked_glm.py:307` `prepare_marked_session_context` — its own copy of the old assembly (no excluded-time rule, no artifact drops); it is what must be rewired.
- `tests/unit/test_import_layering.py` — the layering rules the port must satisfy.

## Tasks

- **Bring the files over** with `git checkout feature/pfc-marked-event-kernels -- <path>` (one commit for the raw checkout is fine, so the next commit's diff shows the fixes):
  - `src/continuum_swr_replay/analysis/pfc/{marked_glm,marked_model_family,runner_support}.py`
  - `scripts/pfc/{run_pfc_marked_kernels,run_pfc_marked_model_family,summarize_pfc_marked_family}.py`, `scripts/qa/measure_pfc_mark_context.py`
  - `tests/analysis/pfc/{test_marked_glm,test_marked_model_family,test_runner_support}.py`, `tests/unit/{test_run_pfc_marked_kernels,test_run_pfc_marked_model_family,test_summarize_pfc_marked_family,test_measure_pfc_mark_context}.py`
  - `docs/design/{pfc_marked_event_kernels,pfc_marked_model_family}.md`
  - the dated audits `docs/audits/2026-09-18-pfc-marked-family-smoke.md`, `2026-09-19-pfc-marked-family-corpus.md`, `-summary.md`, `-summary.csv` → into `.claude/docs/reviews/` (the current home of dated audits); fix links that point at them.
  - `.claude/docs/plans/pfc-marked-family-v2/` (all files) as reference; add one line to its `PLAN.md` status saying the follow-up plan `.claude/docs/plans/pfc-response-followups/` executes phases 1a, 1b (in part), 1c, 2 and 3, and drops 4 and 5.
- **Fix imports** to the current layout (from `docs/README.md` "Moved modules"): `analysis.pfc.glm` → `analysis.pfc.design` (`convolve_event_basis`, `event_basis_functions`), `analysis.pfc.event_sets` (`build_session_event_sets`, `build_session_fit_mask`), `intervals.get_spikecount_per_time_bin`; `data_loaders.utils.parse_nwb_filename` → `decoding.identifiers`; `session_keys` → `tables.session_keys`; `table_io.read_table/resolved_table_path` → `tables.io`, `sha256_file` → `runs.io`; `decoding_common.setup_logger` → `runs.batch`; `parameters.PROCESSED_DATA_DIR` → `paths.processed_data_dir()` **read at call time** (module-level path constants such as `DEFAULT_EVENT_ARM_TABLE_PATH` become functions or `None` defaults resolved inside the function, so the test sentinel root works); `canonical_events.DEFAULT_EVENT_TABLE_PATH` → `paths.table_path("event_table")`; event-arm table → `paths.table_path("event_arm_table")`. Check `analysis.replay.fit_provenance`, `analysis.pfc.gpu`, `analysis.pfc.cohort` signatures against the runners' calls (`load_session_stop_cohort` now returns `(stops, report)`).
- **Rewire `prepare_marked_session_context`** to the current assembly. Replace its body's load/align/mask/event steps with calls to `session._load_aligned_session`, `session._session_fit_population` and `session._session_events` (same arguments `run_session` passes, `stop_cohort="at_well"`, `fit_lags = symmetric_lags(DEFAULT_BASIS_HALF_WINDOW_S, sampling_frequency)`), then attach the marks to `event_times[c]` for `c in MARKED_EVENT_CLASSES` as before. Record the excluded-time and artifact-drop counts in `cohort_info`, and the SWR-source hashes (`canonical_event_provenance` with `swr_events_path/sha256`) in `input_provenance`. Keep the deferred imports (they open Spyglass at import time). Leave the matched control out of the marked designs (the family never used it) but keep it out of nothing else.
- **Real-data equivalence check** (opt-in `@pytest.mark.real_data`, CPU, no fit): for one small session (pick the mPFC session with the fewest units from the canonical sidecars), `prepare_marked_session_context(...)` and `run_session(..., scoring_out=ctx)` stopped before fitting give the same `fit_mask` (exactly) and the same `start_time` arrays for `non_local_SWR` and `non_local_no_SWR` (exactly). If stopping `run_session` before the fit is awkward, compare against `_session_fit_population` + `_session_events` called directly.
- **Runners and docs.** Make both runners' `--help` work and their default output directory `paths.processed_data_dir() / "pfc_marked_model_family"` (unchanged name). Register the three runners and `measure_pfc_mark_context.py` in `scripts/README.md`, the two design docs in `docs/README.md`, and the mark-context script in `docs/regression_pipeline_map.md`, as the branch did (`git show 656fac63:scripts/README.md` etc. for the wording; re-apply by hand, do not check out those files).
- **Not ported, and why** (one line each in the commit message): `66471fd3` (Figure 4 population change); the `setup_logger` mode `"w"` → `"a"` change of `4d69b833` (shared infrastructure; the follow-up runs write to their own log directories).
- `uv run ruff check` / `ruff format` on the ported files; `uv run pytest tests/analysis/pfc tests/unit -k "marked or runner_support or mark_context or import_layering"` green, then the whole `tests/analysis/pfc` and `tests/unit` directories.

## Deliberately not in this phase

- No baseline terms, new marks, new specs, marginal tier or permutation draws (phase 3a).
- No GPU fit and no rerun of any corpus (phase 3b).
- No change to `run_pfc_analysis_bandit.py`, `load_population.py`, `generate_figure04.py` or Figure 4 tests.

## Validation slice

| Test | Asserts |
| --- | --- |
| ported `tests/analysis/pfc/test_marked_glm.py`, `test_marked_model_family.py`, `test_runner_support.py` | pass unchanged in what they assert (only imports and fixtures adapted) |
| ported `tests/unit/test_run_pfc_marked_*.py`, `test_summarize_pfc_marked_family.py`, `test_measure_pfc_mark_context.py` | pass |
| `tests/unit/test_import_layering.py` | no violation |
| `test_marked_context_matches_canonical_assembly` (`real_data`) | fit mask and the two classes' onsets equal the canonical assembly's exactly on one session |
| `--help` of the three runners | exits 0 |

## Fixtures

The ported tests' own synthetic fixtures. Real data: one small mPFC session, opt-in.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the old private assembly inside `prepare_marked_session_context` is gone.
- User-facing documentation listed as tasks is updated, not deferred.
