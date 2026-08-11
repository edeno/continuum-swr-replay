# Designs

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

## Contract module and failure surface

Create `src/continuum_swr_replay/analysis/replay/table_contracts.py` as the canonical production boundary. The module owns ordered builder schemas and validation; it does not calculate scientific table values.

Suggested small API:

```python
@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype_family: Literal["int", "float", "bool", "string", "category"]
    nullable: bool

class ReplayTableContractError(ValueError):
    check_id: str

def empty_table(schema: Sequence[ColumnSpec]) -> pd.DataFrame: ...
def validate_dot_event_contract(dot, *, required_columns=()) -> None: ...
def validate_conditional_posterior_contract(
    posterior,
    dot,
    edge_id,
    position_coordinates,
    *,
    time_atol=1e-6,
    position_atol=1e-6,
    normalization_atol=1e-5,
) -> None: ...
def retained_event_labels(dot, *, speed_threshold, nonlocal_threshold) -> np.ndarray: ...
def validate_event_population_match(
    event_table,
    dot,
    *,
    speed_threshold,
    nonlocal_threshold,
    helper_name,
) -> None: ...
def validate_event_table_contract(event_table) -> None: ...
def validate_stop_table_contract(stop_table) -> None: ...
def validate_event_arm_table_contract(event_arm_table, event_table) -> None: ...
def validate_event_patch_table_contract(event_patch_table, event_table) -> None: ...
def validate_event_sequence_table_contract(
    event_sequence_table,
    event_table,
    *,
    posterior_sessions,
) -> None: ...
def validate_session_replay_tables(
    *, dot, event_table, stop_table, event_arm_table, event_sequence_table,
    posterior_available: bool,
) -> None: ...
def validate_assembled_replay_tables(
    *, event_table, stop_table, event_arm_table, event_patch_table,
    event_sequence_table, expected_session_manifest,
    standardization_reference,
) -> None: ...
```

The five builder schema constants are tuples of `ColumnSpec`, not untyped lists of names. Define assembled schemas separately: the assembled event schema adds retrospective columns, and the assembled stop schema adds `n_non_local_SWR` / `n_non_local_no_SWR`; pooled standardization preserves the declared arm schema. Existing `_event_*_columns()` helpers either become thin name projections from builder schemas or are removed after all internal callers migrate. `validate_session_replay_tables` checks raw-derived builder output before assembly joins and receives posterior availability explicitly. `validate_assembled_replay_tables` requires the additive columns, full expected-session/retained-count/posterior-participation manifest, and pooled standardization reference, so empty frames cannot erase session identity or make missing sequence data look legitimate. The validator currently in `analysis/hippocampus/data_aggregation.py` moves here; all repository imports migrate in the same PR and the old symbol is removed. Do not leave a compatibility wrapper unless authors explicitly reverse the no-alias decision.

Every exception includes a stable `check_id`, the failing session/table, counts, and the first few natural keys. Validators operate on copies only when sorting for diagnostics; they never return repaired inputs. Output validators run on empty results too, so empty schema/dtype drift cannot bypass the boundary.

## Independent slow reference oracle

### Independence boundary

Create `tests/oracles/replay_table_reference.py`. It may import only NumPy, pandas, xarray, NetworkX, Python standard-library types, and hand-authored fixture constants. It must not import any production replay table, contract, schema, hashing, bout, trial-context, mapping, summary, uncertainty, or trajectory code.

The fixture supplies explicit mappings rather than asking production code to infer them:

- `map_arm_by_time`: expected real arm for every finite MAP row and missing for an unmapped/NaN MAP row; finite positions near a junction still map to the nearest real arm under the pinned convention;
- `position_arm`: expected real arm or junction for each posterior position bin;
- literal session key and label columns;
- literal trial/context values only for tests that intentionally exercise them.

This avoids making the “reference” result depend on the same environment mapping helper under review. Production builders still receive a small real test environment; the fixture author independently shows that its bins map to the literal arrays.

Suggested result object:

```python
@dataclass(frozen=True)
class ReferenceReplayResponses:
    event_core: pd.DataFrame
    stop_core: pd.DataFrame
    event_arm_responses: pd.DataFrame
    event_patch_responses: pd.DataFrame
    event_sequence_core: pd.DataFrame
```

The reference scope is intentionally narrow: scientific response arithmetic, window boundaries, detector overlap, stop assignment, and alternative alignment. It excludes hashes, pooled z-scores, contingency joins, graph distances, trial-history covariates, retrospective columns, and trajectory rollups. Those are checked by separate literal unit fixtures and the full-corpus audit.

### Reference algorithm

Use obvious Python loops, not vectorized equivalents of the production implementation:

1. For each raw row, calculate `w_t = continuous + fragmented` and the strict retained predicate from the [input contract](shared-contracts.md#dot-and-event-population).
2. Enumerate positive labels in sorted numeric order. For each label, collect both its full-label row positions and retained row positions. Emit an event only when the retained set is nonempty.
3. Derive the full-label interval directly from first/last raw timestamps. Enumerate each positive SWR label's interval independently; count overlap only when `min(event_end, swr_end) - max(event_start, swr_start) > 0`. Verify boundary-touch cases with literal timestamps.
4. Enumerate low-speed bouts by transitions in the row-wise predicate `head_speed < threshold`, preserving the current row-contiguity convention. Record first/last timestamps, including zero-duration single-row bouts. Require every retained label's bins to have one unique bout index, assign it there, and independently count unique retained labels per bout.
5. For each event, calculate filtered start/end/duration, the upper-middle observed sample `times[len(times)//2]`, the median of per-bin `time - stop_start` offsets, bin count, summed `w_t`, MUA presence, unmapped/NaN-MAP count, and unique real MAP arms with scalar loops. Calculate the full-label midpoint by the same upper-middle-sample rule; do not substitute an arithmetic midpoint.
6. For each retained posterior row and each arm `0..8`, explicitly sum position-bin mass whose literal `position_arm` equals that arm. Remove junction mass and normalize across real arms. If real-arm mass is zero, apply the currently pinned uniform-row convention and tag the case so Phase 2/3 can revisit it.
7. Accumulate per arm: hard MAP count, `w_t`-weighted MAP count, and `w_t * arm_posterior[a]`. Use lowest-index argmax for sequence MAP. Accumulate patches from a literal arm-to-patch dictionary.
8. Emit sequence rows in raw-time order with `t_in_event` reset at each label.
9. Compare reference and production subsets after natural-key uniqueness checks and keyed sorting. Use exact equality for keys/counts/flags/literal timestamps and the shared deterministic floating tolerance for calculated sums.

Each case has a handwritten expectation table alongside the input fixture. The oracle result is compared both to those literals and to production; otherwise a bug in the oracle could merely agree with itself.

### Adversarial case matrix

The minimal suite includes:

- every mask clause independently false, strict speed/probability ties, NaN probability/speed, and a label with only some retained bins;
- fragmented-only, continuous-only, and mixed non-local probability;
- no SWR, contained SWR, partial overlap, two overlapping SWRs, and boundary touch;
- zero-event, no-stop, and single-sample stops; a label near a bout boundary; an invalid retained label spanning two bouts; and a row-contiguous bout with a timestamp gap;
- one-hot, diffuse, permuted-position, all-junction-mass, NaN, negative, and non-normalized posterior inputs;
- nonconsecutive labels, shuffled event-table rows, nondefault/duplicate DataFrame index labels, duplicated IDs, and mixed sessions;
- NaN/unmapped MAP rows (finite MAP values always choose the nearest real arm), tied arm posterior, unknown arm, missing/duplicate arm and patch alternatives;
- odd/even and gapped retained-bin sequences proving upper-middle midpoint and exact median `time_since_stop` behavior;
- missing trials/contingency, all-NaN value groups, and exact NaN propagation through interactions.

## Read-only full-corpus audit

### Engine and command surface

Create `src/continuum_swr_replay/analysis/replay/table_audit.py` as a pure read-only engine, `src/continuum_swr_replay/visualization/table_audit.py` for headless QC rendering, and `scripts/qa/audit_replay_tables.py` as a thin CLI.

Core API:

```python
@dataclass(frozen=True)
class AuditInputs:
    ...
    @classmethod
    def from_directory(
        cls,
        input_dir: Path,
        source_dir: Path | None,
        expected_session_manifest: Path,
    ): ...

@dataclass(frozen=True)
class AuditResult:
    summary: Mapping[str, object]
    manifest: pd.DataFrame
    checks: pd.DataFrame
    automated_findings: pd.DataFrame
    failures: pd.DataFrame
    session_metrics: pd.DataFrame

def run_table_audit(
    inputs,
    *,
    expected_session_manifest: pd.DataFrame,
    source_checks: bool = True,
    max_failure_examples: int = 100,
) -> AuditResult: ...

def select_qc_windows(result, *, per_animal=8, seed=20260808) -> pd.DataFrame: ...
def write_audit_bundle(result, output_dir: Path) -> None: ...
def render_qc_catalog(*, audit_dir, input_dir, source_dir, overwrite=False): ...
```

Subcommands and exit codes:

- `audit`: load canonical tables, inventory the exact frozen session manifest, run automated checks, select QC windows, and atomically write a new immutable automated-audit directory. With `--max-per-animal`, deterministically sort by `(animal, day, epoch, session_id)`, take the first `N` rows per animal from the frozen parent manifest, write that active subset, and run identity checks against the subset rather than all 218 rows;
- `render-qc`: verify the automated hashes, render selected windows, and create `manual_qc.csv` with image hashes; it does not modify automated outputs and refuses an existing image/manual tree;
- `audit` also creates `finding_dispositions.csv`, a review ledger with exactly one blank disposition row per immutable automated finding; reviewers may edit only this ledger and the reviewer/verdict/issue/notes/disposition fields in `manual_qc.csv`;
- `finalize`: require one row per QC selection and one disposition-ledger row per automated finding, recheck sources/images, convert every manual `concern|fail` into a finding, join the review ledger to immutable automated findings, and write new `final_findings.csv`, `final_summary.json`, `final_report.md`, and `canonical_after.sha256`; it never rewrites `summary.json`, `automated_findings.csv`, or `report.md`. Open hard/identity rows are preserved and produce `FAIL`; they are never erased merely to permit finalization;
- exit `0` for an automated hard/identity pass or a complete manual sheet with no `fail` and every `concern` dispositioned, `1` for a scientific/manual failure or undispositioned concern, and `2` for IO/schema/configuration failure.

The engine carries an independent, versioned expected schema/check registry. It does not call producer-private `_event_*_columns()` or calculate expected identities through production helpers. Calling Phase 1 contracts in addition is useful defense in depth, but those results have distinct check IDs and cannot replace the independent checks.

Open all inputs read-only. Refuse an output equal to or nested anywhere under the canonical replay-content directory; require ignored `scratch/tables-review/<run_id>/audit`. Refuse an existing nonempty run directory; there is no overwrite mode for immutable automated results. Write into a temporary sibling, fsync/close, then atomically rename. SHA-256 before/after the five canonical tables and tractable small auxiliaries. For every raw dot/environment/result/cache file record only size and `mtime_ns`; Phase 2 neither content-hashes nor opens posterior results.

The expected-session manifest is a frozen, reviewed table of all 218 exact `(animal, day, epoch, session_id)` identities plus expected dot/environment/result/cache participation and `posterior_status = available|unavailable|unattested`, with an evidence token/hash for either attested state. `available` requires a contemporaneous successful-load record from a source/cache build attestation; `unavailable` requires a contemporaneous explicit absent/failed-load record. File presence and materialized sequence rows are not attestations. Legacy sessions without such a record are `unattested`: Phase 2 checks their materialized posterior algebra but emits `POSTERIOR_PARTICIPATION_UNATTESTED` as a scientific-semantic finding, marks source participation `not_evaluable`, and returns an overall `INCONCLUSIVE` report until a source-attested candidate resolves it. At audit time, independent raw-dot reconciliation adds only `n_expected_stops` and `n_expected_retained_events`; exact sequence participation combines retained-bin counts with `available|unavailable` only.

For a pilot, derive `active_session_manifest.csv` from the immutable parent with the selection rule above. Record the parent hash, selector, active-subset hash, and active count; `expected_sessions` refers to the active subset. The full run uses an identical active and parent manifest. An unexpected session cannot replace an omitted expected one in either mode.

### Versioned artifact schemas

Schemas, primary keys, dtype/nullability, and enum values live in a versioned audit schema module and are repeated as literal expectations in tests rather than imported from that module.

- `summary.json` has exactly `schema_version`, `run_id`, `created_at_utc`, `git_commit`, `git_diff_sha256`, `command_sha256`, `parent_session_manifest_sha256`, `active_session_manifest_sha256`, `session_selector`, `input_manifest_sha256`, `expected_sessions`, `observed_sessions`, `table_rows`, `thresholds`, `check_counts`, `automated_status`, `elapsed_sec`, and `peak_rss_mb`.
- `final_summary.json` has exactly `schema_version`, `run_id`, `finalized_at_utc`, `automated_summary_sha256`, `automated_findings_sha256`, `finding_dispositions_sha256`, `manual_qc_sha256`, `final_findings_sha256`, `canonical_after_sha256`, `automated_status`, `manual_qc_status`, `final_status`, `check_counts`, `finding_counts`, and `reviewer_count`.
- `manifest.csv` is normalized with primary key `(session_id, artifact_role)` and columns `artifact_role,animal,day,epoch,session_id,path_token,expected,exists,size_bytes,mtime_ns,sha256,hash_policy,table_rows,status,detail`. Global artifacts use `session_id="__global__"`; `hash_policy` is `sha256|stat_only`.
- `active_session_manifest.csv` has exact columns `animal,day,epoch,session_id,expect_dot,expect_environment,expect_result,expect_cache,posterior_status,posterior_evidence_token,posterior_evidence_sha256`; its rows are the active subset used for every session-identity check.
- `checks.csv` and capped `failures.parquet` use the exact columns below. `failures.parquet` contains examples only and never represents completeness.
- Immutable uncapped `automated_findings.csv` has primary key `finding_id` and columns `finding_id,check_id,severity,phase,run_id,scope,table,columns,animal,day,epoch,event_id,stop_id,selection_id,n_affected,fraction,observed,expected,evidence,downstream_claims,initial_status,initial_owner,initial_resolution`.
- Editable `finding_dispositions.csv` has primary key `finding_id` and exact columns `finding_id,disposition,final_status,final_owner,final_resolution,reviewer,reviewed_at_utc,notes`; its key set must equal `automated_findings.csv`. Allowed final-status values are `open|expected_change|resolved|accepted_diagnostic`; hard/identity rows may be `resolved` only with a resolution PR/run and reviewer.
- Finalize-created `final_findings.csv` has primary key `finding_id` and the automated columns plus `finding_source,final_status,final_owner,final_resolution,disposition,disposition_reviewer,disposition_at_utc,disposition_notes`. `finding_source` is `automated|manual`; each manual `concern|fail` is represented once and linked by `selection_id`. Automated columns are copied byte-for-value and joined to disposition fields by `finding_id`, never edited in place.
- `session_metrics.csv` has primary key `(animal,day,epoch)` and exact columns `animal,day,epoch,session_id,n_events,n_arm_rows,n_patch_rows,n_sequence_bins,n_stops,n_zero_event_stops,n_zero_duration_stops,events_per_stop,swr_fraction,mua_fraction,junction_fraction,trunk_fallback_fraction,probability_overshoot_fraction,timestamp_gap_stop_fraction,posterior_missing_fraction,value_missing_fraction,next_arm_mismatch_fraction,event_duration_p05,event_duration_median,event_duration_p95,event_bins_p05,event_bins_median,event_bins_p95,posterior_peak_p05,posterior_peak_median,posterior_peak_p95,events_per_stop_p05,events_per_stop_median,events_per_stop_p95`.
- `qc_selection.csv` has primary key `selection_id` and exact columns `selection_id,animal,day,epoch,session_id,grain,event_id,stop_id,anchor_time,window_start,window_end,stratum,selection_metric,selection_value,session_tertile,eligibility_status,fallback_reason,seed`.
- `manual_qc.csv` has primary key `selection_id` and exact columns `selection_id,image_set_sha256,reviewer,reviewed_at_utc,verdict,issue_codes,notes,disposition,disposition_reviewer,disposition_at_utc`.

### Automated registry

Each check emits a corpus row and, when meaningful, per-session rows. `checks.csv` has exact columns:

```text
check_id,severity,scope,table,animal,day,epoch,n_evaluated,n_failed,
failure_fraction,max_abs_error,tolerance,status,detail
```

The capped example table preserves total failure counts and uses:

```text
check_id,table,animal,day,epoch,event_id,stop_id,arm,patch,t_in_event,
row_index,observed,expected,abs_error,detail
```

Registry groups:

1. **Input/schema/provenance:** expected session manifest; source presence; exact ordered assembled columns/dtype families; no silent extras; non-null IDs; one session per ID; unique natural/surrogate keys. The assembled schema includes retrospective outputs and pooled z columns in addition to builder schemas. Retired `q_arm_hmm`, `q_arm_hmm_z`, and `swr_x_q_arm_hmm` are a visible hard drift, not an implicit allowlist.
2. **Referential/cardinality:** 9 arms/event, 3 patches/event, valid foreign keys, exact sequence counts, stop/event reconciliation, and SWR/no-SWR split. Participation follows manifest-derived expectations: stops iff bouts exist, event/arm/patch iff retained events exist, and sequence iff posterior plus retained bins exist.
3. **Raw-dot reconciliation:** stream one of the 218 roughly 19-GB-total dot files at a time; independently recalculate strict retained labels, filtered/full boundaries, weights, MUA, SWR overlap, bouts, label counts, event/stop assignment, and sequence time/weight alignment. Never open posterior NetCDF here.
4. **Materialized event/sequence identities:** time ordering/durations, arm-posterior row sums and argmax from the materialized sequence, event posterior rollups/ranges, SWR matched-field missingness, and event/label containment. These prove internal algebra only; without opening NetCDF, Phase 2 does not call posterior vectors/MAP, event peak/entropy/HPD, posterior mass/fractions, or multimodality source-correct. A stratified independent NetCDF parity check is mandatory in Phase 3.
5. **Arm algebra:** response conservation, event broadcasts, indicators, distances/ranks, contingency missingness, demeaning scope, all interactions, and z-reference recomputation. Use `sum(min_count=1)` semantics where all-missing must stay missing.
6. **Patch recomputation:** exact projection from arm rows, leaf-only means, NaN-preserving sums, indicators, session keys, and interactions.
7. **Stop algebra:** bout order/exposure, counts/flags, reward lags/rates, context complements, arm/patch maps, and SWR split.
8. **Retrospective/standardization:** independently recompute materialized retrospective columns from arm/sequence rows and pooled standardized columns from the saved reference.

### Diagnostic summaries

“Sensible answers” are summarized, not hard-coded as expected biological signs. Produce one row per session with counts, zero-event/zero-duration stops, events per stop, SWR/MUA/junction fractions, posterior/value/contingency missingness, next-arm mismatch, event duration/bin/peak quantiles, and stop count/duration quantiles. For each model response and predictor, report by animal and session tertile: finite fraction, robust quantiles, median/MAD outliers (`|robust-z| > 5`, IQR fallback when MAD is zero), and temporal drift.

Explicit diagnostic comparisons include:

- full-label versus filtered event-window deltas and events whose retained bins touch multiple bouts;
- MUA classification under filtered and full windows;
- current-arm trunk fallback rate and its association with `is_current` alternatives;
- probability overshoot, `nonlocal_weight_sum > n_nonlocal_bins`, all-junction/unmapped bins, and zero real-arm mass;
- zero/single-sample stops, timestamp gaps inside row-contiguous bouts, and alternative exposure totals;
- contingency/value/uncertainty missingness and model complete-case attrition by animal/session;
- current source schema versus materialized schema, with the known retired-HMM-column drift surfaced explicitly.

### Deterministic QC catalog

Select eight base windows per animal (40 total), plus at most ten deduplicated automated-failure examples. Sort each animal's sessions chronologically and assign `early|middle|late` tertiles with `np.array_split`; target tertiles for strata 1-8 cycle `early,middle,late,early,middle,late,early,middle`. For a requested target quantile, exclude nonfinite metric values, compute `np.quantile(values, q, method="linear")`, rank by absolute distance to that scalar, then by the full natural key. Search the target tertile first, then the next chronological tertile, then the remaining tertile; record fallback. If a selected key was already used, take the next ranked eligible key.

Per animal, use these exact strata and metrics:

1. `typical_non_swr`: `is_swr == False`, event duration nearest its median;
2. `typical_swr_contained`: contained overlap class, event duration nearest its median;
3. `boundary_partial_swr`: partial overlap class, smallest positive `overlap_duration / label_event_duration`, then shortest label duration;
4. `low_posterior_peak`: finite `posterior_peak_mean`, nearest its 10th percentile;
5. `junction_or_mapping_anomaly`: if any event has `n_junction_bins > 0`, nearest its 90th percentile among positive values; otherwise choose the largest positive probability overshoot; otherwise a trunk-fallback event; otherwise the lowest unused posterior peak and record `no_positive_junction_or_mapping_example`;
6. `long_path`: finite `path_length`, nearest its 90th percentile, with `n_patches_visited` then natural key as tie breaks;
7. `zero_event_stop`: `n_non_local == 0`, stop duration nearest its median;
8. `high_event_count_stop`: positive `n_non_local`, nearest its 90th percentile, then longest duration.

If a stratum has no eligible rows after its explicit fallback, select the median-duration unused event/stop of the same grain and record `no_eligible_stratum_example`; this keeps exactly eight rows per animal without pretending the target anomaly exists. Failure exemplars are sorted by `hard` before `identity`, then `check_id` and natural key; take the first ten unique source windows not already selected. The seed is metadata only after deterministic ranking. Event windows are `[anchor-3, anchor+3]`. Stop windows start as `[stop_start-2, stop_end+2]`; if longer than 12 seconds, replace with `[stop_midpoint-6, stop_midpoint+6]`; finally clip to raw source bounds and record clipping.

For each selection render three images:

- `source`: raw time, speed and threshold, continuous/fragmented/summed non-local probability and threshold, non-local/SWR/MUA labels, map estimate, and trial/bout boundaries;
- `content`: the corresponding materialized event/sequence arm posterior and decoded content on the track graph;
- `design`: the event/arm/patch/stop rows that downstream models receive, including missingness and interactions.

Existing `visualization/session_tables.py:3096-3355` may provide a comparison rendering surface, but the raw/source panel and annotations are independently built and canonical table plots are not treated as an oracle. The QC path remains offline and never opens posterior NetCDF.

`manual_qc.csv` requires exactly:

```text
selection_id,image_set_sha256,reviewer,reviewed_at_utc,verdict,issue_codes,notes,
disposition,disposition_reviewer,disposition_at_utc
```

Verdicts are `pass|concern|fail`; issue codes are `event_boundary`, `swr_class`, `stop_assignment`, `arm_content`, `trial_context`, `value_alignment`, `missingness`, or `other`. `finalize` refuses an incomplete sheet or a `concern` without a disposition and second reviewer/timestamp. A completed `fail` row does not suppress output: finalize converts it into a manual row in `final_findings.csv`, writes a final `FAIL` report, and exits `1`. A disposition never erases the original verdict or Phase 3's entry gate.

### Audit bundle

Write:

```text
scratch/tables-review/<run_id>/audit/
  summary.json
  manifest.csv
  active_session_manifest.csv
  checks.csv
  automated_findings.csv
  finding_dispositions.csv
  failures.parquet
  session_metrics.csv
  report.md
  manual_qc_template.csv
  qc_selection.csv
  manual_qc.csv
  final_findings.csv
  final_summary.json
  final_report.md
  canonical_after.sha256
  qc/index.html
  qc/images/<selection_id>__{source,content,design}.png
```

Commit only sanitized `summary.json`, `final_summary.json`, `manifest.csv`, `active_session_manifest.csv`, `automated_findings.csv`, `finding_dispositions.csv`, `final_findings.csv`, `qc_selection.csv`, `manual_qc.csv`, and `final_report.md` under `docs/audits/replay-tables/<date>-<shortsha>/`. Full failures, images, and source paths remain ignored review artifacts.

## Generation-coherent candidate review

### Prerequisite ownership

Phase 3 is a table-specific consumer of `.claude/docs/plans/regression-integrity-fixes`, not a second pipeline. Its entry manifest requires completed, schema-compatible outputs from:

- Phase 7: source-valid session bundles, expected-session manifest, atomic `run_artifacts` staging/finalization, component resolver, and no legacy fallback;
- Phases 9a/9b: `AnalysisIdentity`, `freeze_comparison_cohort`, strict allowed-row fit mode, persisted exact key tables, and shared blocked folds;
- Phases 10a/10b: rank/separation/convergence gates, robust covariance/bootstrap, animal-balanced summaries, and multiplicity fields;
- Phase 11: decision-grain choice and stay/leave replacements;
- Phase 12: identified SWR-power models (not a surrogate for claim #24's timepoint-density analysis).

If any required API or completion manifest is absent, table structure can still be reported, but the overall paper-facing verdict is `INCONCLUSIVE`. Phase 3 does not add `--input-dir` flags, private manifests, private cohort lists, or legacy estimators to work around a missing prerequisite.

### Run layout and state machine

Create/finalize the candidate only through the Phase-7 context-managed run API. The table review stores references and reports, not copied pseudo-canonical files:

```text
<review_root>/<review_id>/
  table_review_manifest.json
  candidate_run_id.txt
  legacy_baseline_manifest.json
  canonical_before.sha256
  canonical_after.sha256
  logs/
  synthetic/
  stress/<session_id>/
  posterior_spotcheck/
  comparisons/{invariants.json,keyed_deltas.parquet,cohort_delta.csv}
  claims/{claim_impact.csv,claim_impact.md}
  verdict.md

<phase7_run_root>/<candidate_run_id>/
  manifest.json
  source_input_audit.parquet
  components/
    behavior_timebase_and_q/
    replay_tables/
    event_onset_and_stop_timebins/
    decision_tables/
    corrected_model_runs/
```

The Phase-7 API writes a complete component directory into `.staging`, validates declared files/cells/hashes, and renames the directory atomically before explicit run-ID resolution. It never advances `CURRENT.json`. The review state advances only through `initialized -> synthetic_passed -> stress_passed -> posterior_spotcheck_passed -> source_valid -> candidate_finalized -> compared -> corrected_refits_resolved -> owners_complete -> verdict`. Code/source/component drift after a transition requires a new review and candidate run ID.

### Staged rebuild

1. **Synthetic:** run all Phase 1 adversarial oracles, including empty/no-event/no-stop sessions, malformed posterior, threshold/boundary ties, multi-bout labels, zero-duration stops, unmapped MAP, posterior-junction mass, and arm ties.
2. **Stress slice:** use Phase-7 session-scoped candidate runs for the eight fixed sessions below. Compare all raw/unstandardized builder and auxiliary values, schemas, keys, and identities. Per-session arm `_z` placeholders are not compared to pooled canonical z-scores. As a wiring diagnostic only, apply the frozen legacy pooled reference to the stress raw columns and compare; the full candidate later computes its own pooled reference.
   - `chimi20200213_06_r3` — high event count, many events in one stop, long bout;
   - `j1620210707_16_r8` — very long/zero-duration bouts and belief gaps;
   - `senor20201103_10_r5` — low SWR fraction;
   - `chimi20200228_06_r3` — high SWR fraction;
   - `wilbur20210331_10_r5` — few events and many zero-event stops;
   - `wilbur20210329_16_r8` — high Q missingness;
   - `chimi20200224_04_r2` — sparse last-rewarded-elsewhere support;
   - `peanut20201202_10_r5` — smallest-animal-cohort coverage.
3. **Independent posterior source parity:** predeclare events/windows from the stress manifest and Phase-2 QC strata before opening results. Lazily open only the required NetCDF time/position slices; independently verify dimensions/coordinates, position-to-arm collapse, sequence vectors/MAP, event peak/entropy/HPD/multimodality, and event-arm posterior mass/fractions against the candidate. This closes the source-correctness gap that Phase 2 can test only algebraically. Any missing selected posterior or selection changed by the candidate yields `INCONCLUSIVE`, not replacement sampling.
4. **Source validity:** require Phase 7's complete all-session source audit and expected-session accounting. Do not replace it with a five-session Spyglass/cache spot check.
5. **Full coherent candidate:** regenerate/finalize behavior timebase/Q, all five replay tables and pooled z reference, event-onset/stop-timebin, and decision components under one source/table generation. If an event/stop change invalidates onset ranks or Q joins, regenerate the dependent component; never carry a legacy auxiliary across generations. Measure one-session/stress wall time, RSS, and disk before full compute.

### Keyed baseline/candidate comparison

Treat the current materialized generation as a read-only legacy baseline whose known provenance/model defects remain visible; it is not an acceptance target. For each table:

1. Validate baseline and candidate separately against their declared schemas; record baseline failures even when candidate corrects them.
2. Compare exact session participation from manifests, schema/dtypes, and natural/surrogate key uniqueness.
3. Outer-join on natural keys and report added/dropped/common rows by session/animal.
4. On common rows, compare exact columns, NaN masks, then numeric values under shared tolerances.
5. Require a predeclared `expected_change_id` and independent oracle for every accepted candidate delta; report downstream cohort impact.
6. Re-run all candidate bundle/standardization/auxiliary identities. Coefficient agreement cannot compensate for a failed table or generation identity.

The retired HMM columns, probability overshoot, full/retained-window relation, zero real junction examples, MUA policy, exposure, and trunk fallback each require explicit finding dispositions. Stress-stage z values are excluded as described above; full pooled z values and reference hashes are compared after candidate finalization.

## Corrected claim-impact review

### Exact claim specification and ownership

Build rows from a checked-in `ClaimSpec` registry before candidate results exist. A prose row with several effects becomes several atomic specs. Each spec names the exact table columns/builder, corrected Phase-9 comparison set and row-key grain, corrected Phase-10 estimator/term/uncertainty, owner completion manifest, direction/order/equivalence rule, and output identity.

Initial routing:

- #2/#21 and #22: deterministic candidate event-table prevalence/classification/boundary summaries, per animal and session, with no model substitution;
- #13: event-table distance-regime summaries using a predeclared distance column, unit, weighting, and place-field-width boundary; unresolved boundary/weighting cannot be chosen from the candidate histogram;
- #14: split the canonical event-arm occupancy component from the raw-dot distance-quadrant component; the latter requires its separate owner/completion status;
- #15-#18: deterministic table summaries plus an author-locked formula/transform/duration-control/cluster/output spec. No named implementation currently reproduces the claimed duration-controlled GEE values; if that spec is not reviewed before execution, these rows remain `unresolved` rather than letting the executor invent a model;
- #19a/#19b/#19g: corrected content selection/estimation sets and robust inference from Phases 9-10. Reconcile the paper's stale `recent_reward_rate` wording with the exact current predictor (`reward_rate_arm_k10` or its corrected successor) before execution;
- #19c: corrected per-stop/per-bin/SWR-rate estimators, exact cohorts/folds, and robust uncertainty from Phases 9-10;
- #19d/#19e: Phase-11 decision-grain outputs only. Old stop-selected choice and Stage-6 GEE are historical diagnostics and cannot support a claim;
- #19f: every cited `rew_x_current`, `value_x_current`, `switch_x_current`, time-control, and graded-value/previous-goal estimand must resolve to an approved permanent corrected spec. `fit_swr_contrast_content.py`'s current ladder is not assumed to reproduce the removed exploratory terms; missing terms make the atomic row `unresolved`;
- #24: consume the SWR-continuum owner's Local-vs-Non-Local timepoint-density completion manifest. Do not use the event-level SWR-power OLS (legacy or Phase-12 identified replacement) as a proxy.

Inventory all code consumers during implementation; add any additional direct table estimand to the registry before opening candidate results. Mixed/raw claims remain explicit cross-owner rows so an overall `PASS` cannot silently ignore them.

### Natural and intersection cohorts

Resolve each generation as a complete run/component view. Candidate Q/onset/decision inputs come from the candidate generation; legacy diagnostics use only their own coherent baseline components when available. Never combine candidate tables with legacy q-values/onsets.

For each atomic model spec:

1. use Phase-9a `AnalysisIdentity` to materialize both generations' natural exact key tables;
2. create the cross-generation natural-key intersection manifest;
3. pass natural and intersection keys into the Phase-9a strict fitter, which raises on any internal drop;
4. reuse Phase-9b blocked folds and training-only transforms;
5. require Phase-10 fit/term validity and robust/animal-balanced outputs.

Legacy results are labeled diagnostic because their source/estimator provenance is known invalid. Corrected candidate support—not similarity to legacy coefficients—governs the claim state. If a legacy coherent auxiliary/model component does not exist, report the baseline comparison unavailable rather than borrowing a candidate component.

### Claim-impact matrix and verdict

Generate `claim_impact.csv` first and render Markdown from it. Each row carries all fields in the [claim-impact contract](shared-contracts.md#claim-impact-and-verdict-contract), including owner/status and natural/intersection identities. The final report lists:

- Phase-1/2 oracle, source, manual-QC, and finding dispositions;
- Phase-7 source/run/component hashes and canonical before/after hashes;
- expected/unexpected table, auxiliary, cohort, and numerical deltas;
- Phase-9-12 cohort/fold/fit/inference completion and invalid cells;
- every atomic claim state, owner manifest, and wording/withdrawal action;
- exactly one verdict under the deterministic precedence in the shared contract.

No code path advances a current pointer, promotes files, edits paper values, or treats legacy-estimator agreement as support. Phase 13 consumes this gate during its clean production run and owns any authorized cutover.
