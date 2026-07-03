# Documentation index

Every documentation artifact has **one home by type**. This index states the scheme so
it stops drifting; add new docs to the matching home.

## Where each doc TYPE lives

| type | home | what it is |
|---|---|---|
| catalog | `docs/models_table.md` | every regression model in R notation (mirrors `model_specs.py`) |
| definitions | `docs/covariate_dictionary.md`, `docs/behavior_model_covariates.md` | canonical covariate definitions; the `qvals_table` data dictionary |
| methods | `docs/rate_models.md` | how the model families work (not results) |
| results | `docs/per_bin_rate_results.md` (auto-generated), `docs/replay_rate_content_swr_synthesis.md` (living note) | fitted effects + integrative findings |
| reference | `docs/task_description.md`, `docs/cached_trials_for_testing.md`, `docs/known_issues/` | task/data background |
| **design spec** (scientific/methodological) | **`docs/design/`** | analysis designs — human-discoverable for collaborators/reviewers |
| implementation plan (agent process) | `.claude/docs/plans/` | phased execution plans |
| review (code/methods output) | `.claude/docs/reviews/` | point-in-time review reports |
| exploratory notes + figures | `.claude/research/` | scratch analyses; `*_outputs/` figures |
| model spec (code) | `src/continuum_swr_replay/analysis/replay/model_specs.py` | single source of truth for all model ladders |

Rule of thumb: **published/scientific → `docs/`; agent-process/scratch → `.claude/`.**

## `docs/` contents

- [models_table.md](models_table.md) — *catalog*: every model in R formula notation.
- [covariate_dictionary.md](covariate_dictionary.md) — *definitions*: covariate semantics across families.
- [behavior_model_covariates.md](behavior_model_covariates.md) — *definitions*: `qvals_table.pkl` data dictionary.
- [rate_models.md](rate_models.md) — *methods*: the two event-rate families (A per-stop, B per-bin).
- [per_bin_rate_results.md](per_bin_rate_results.md) — *results*: fitted Family-B effects (auto-generated).
- [replay_rate_content_swr_synthesis.md](replay_rate_content_swr_synthesis.md) — *results*: integrative rate × content × SWR synthesis (living note).
- [task_description.md](task_description.md) — *reference*: behavioral task / recording setup.
- [cached_trials_for_testing.md](cached_trials_for_testing.md) — *reference*: test-fixture trial cache.
- [design/](design/) — *design specs*: per-analysis design documents.
- [known_issues/](known_issues/) — *reference*: documented data/pipeline issues.
