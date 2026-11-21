# Phase 10 — Figure 4, drawn to the standards of Figures 1 to 3 and 5

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Branch: `figure04` from master after Phase 9.

**Inputs to read first:**

- The design, section 8 (first panel plan, a to i, with the question each answers) and section 1 (the
  figure's question in the lead's words).
- The finished figures as the standard: `scripts/figures/generate_figure02.py` (`place_letters` `:1266`,
  `panel_extent` `:1255`, the letter rows and columns at `:1424-1440`), `scripts/figures/generate_figure03.py`
  (`letter_positions` `:544`), `scripts/figures/generate_figure05.py` (`letter_positions` `:688`,
  `assemble_figure` `:720`); `src/continuum_swr_replay/visualization/simulated_run_panels.py:453`
  (`check_placement`); `visualization/palettes.py:42` (`COLORS`, including `"No SWR"`);
  `visualization/figure_utilities.py` (the publication helpers); `docs/development.md` convention 10 (panel
  letters).
- `docs/figure-pipeline.md:19-37` (main figures; Figure 4 "has no script yet"),
  `scripts/figures/generate_all_figures.py:4`, `src/continuum_swr_replay/reported/questions.py:93`
  (`FIGURES`).
- Phase 9's `models_convention/curves.parquet` and `family.parquet`, and the analysis functions that build
  them (the figure calls those functions, not the files' internals, as the summary does).
- Phase 3's `descriptive/visit_clock_curves.parquet` and Phase 6a's `descriptive/role_content.parquet` (the
  model-free points drawn under the model curves).

**Contracts referenced:** [C6](shared-contracts.md#c6-kinds-contrasts-and-the-two-meanings-of-direct) (names and
colors of the kinds), [C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures).

## Tasks

- **Design pass first** (design 8: "this plan is the starting point for the figure's own design pass"):
  draft the layout from the panel plan, one panel per question, SWR and No SWR drawn separately in every
  panel. Panel c's second half (rate against time until a small movement) is drawn from the model only if
  Phase 5a kept `f_m`; otherwise from the model-free curves, labeled as such. Show the draft to the lead
  before polishing; the panel list may change.
- **Script** `scripts/figures/generate_figure04.py` (new), following Figures 2, 3 and 5: read inputs through
  the analysis functions; one `draw_<panel>` function per panel; letters by a `place_letters` of the same
  rule (rows and columns of letters, convention 10); `check_placement` against overlap; `COLORS` for the
  kinds; curves with their simultaneous bands; scalar estimands as estimate and interval per kind with the
  difference, animals as points; the q-value where a panel shows a declared test. Panel a (an example visit
  with events of both kinds against the clocks) follows the posterior conventions if it shows a decode (the
  animal's position in magenta, the linearized track strip, `bone_r`).
- **Outputs**: `paper/figures/figure04/` (PDF and PNG, as the other figures); track the files (the
  `.gitignore` already excepts all five main figures, `docs/figure-pipeline.md:33`).
- **Register**: `FIGURES["4"]` in `reported/questions.py:93` with its panel letters; tag the declared values
  with their panels (`3.11` to `3.17` questions get `figures=("4x",)`); add Figure 4 to
  `generate_all_figures.py`; update `docs/figure-pipeline.md`.
- **Critique**: run the `figure-critic` skill on the draft and the final; fix what it finds within the
  panel plan.
- **Caption source**: if the other figures' captions are generated or kept in a file, add Figure 4's with
  value markers for every number.

## Deliberately not in this phase

- New statistics. A panel that needs a number not in the summary goes back to its family phase.
- The mPFC figure (Figure 5, Phase 8).
- Supplementary figures for the robustness estimators (separate, if the lead asks).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/scripts/figures/test_figure04.py::test_draws_on_synthetic_inputs` | the figure builds from small synthetic curves and family tables; every panel letter is placed; `check_placement` passes |
| `test_figure04.py::test_panels_read_the_summary_functions` | each scalar drawn equals the value the behavior summary holds for it (same function, same input) |
| `tests/reported/test_reported_questions.py` | Figure 4 and its panels are known; the declared values name existing panels |

## Fixtures

Synthetic curves and family tables built in the test.

## Verification and runtime

- Look at every panel at print size; compare each curve with its model-free points; confirm the colors and
  names match Figures 2, 3 and 5.
- Runtime is seconds; no long job.

## Rebuilt

`paper/figures/figure04/`; `paper/reported/questions.md` (the figure tags) via the emitter.

## Exit criteria

The lead has seen the draft and the final; the figure is tracked, registered and drawn by
`generate_all_figures.py`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
