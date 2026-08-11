# Shared Contracts

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

These contracts are normative across all three phases. An implementation may make a contract stricter only in a separately reviewed change with new adversarial fixtures. It may not sort, deduplicate, clip, fill, coerce, or relax a tolerance to make observed data pass.

## Review boundary and ownership

The reviewed production surface is `src/continuum_swr_replay/analysis/replay/tables.py` plus the direct helpers that determine its rows or values: `_helpers.py`, `covariates.py`, `summaries.py`, `uncertainty.py`, `trajectory.py`, `spatial_mapping.py`, and the classified-event merge in `analysis/hippocampus/event_table.py`. `prepare_data.py` is included only at its builder assembly, validation, cache, standardization, and write boundaries.

The three phase PRs have strict mutation authority:

| Phase | May change | Must not change |
| --- | --- | --- |
| 1 | Contracts, builder boundary validation, unambiguous structural/alignment defects with hand-oracle regressions, tests, developer docs | Event detector, scientific thresholds, intentional formulas/window conventions, canonical artifacts, model runners, paper outputs |
| 2 | Read-only audit/QC code, tests, audit docs, sanitized audit report | `tables.py`, table artifacts, prepare-data execution, models, figures, claims |
| 3 | Table-specific adapters to the Phase-7 run API, generation comparison, claim specs/reporting, tests, noncanonical review outputs | Competing path/run/cohort/fold/fit infrastructure, table/model formulas, canonical artifacts, artifact promotion, manuscript/figure numbers |

Any scientific formula defect discovered after Phase 1 is fixed in a separate table-fix PR with a finding ID and independent regression oracle. All applicable phase gates rerun from a new run ID. Phase 3 requires regression-integrity Phases 7-12 and consumes their `run_artifacts`, `AnalysisIdentity`, frozen-cohort/fold, fit-validity, robust-inference, decision, and power contracts. Production compute/promotion and manuscript cutover remain owned by `.claude/docs/plans/regression-integrity-fixes/phase-13-refit-reporting.md`.

## Scientific grains and keys

Row order and pandas index are never scientific keys. Before any join or comparison, validate key uniqueness and then sort a copy by the declared key.

| Table | Grain | Natural key | Surrogate/key constraints |
| --- | --- | --- | --- |
| `event_table` | One retained positive `non_local_event` label in one session | `(animal, day, epoch, non_local_event)` | `event_id` non-null and one-to-one with the natural key; `stop_id` maps to exactly one stop in the same session |
| `event_sequence_table` | One retained event time bin | `(animal, day, epoch, non_local_event, time)` | `(event_id, t_in_event)` unique; `t_in_event = 0..n-1` in increasing `time` order |
| `stop_table` | One contiguous row-defined immobility bout | `(animal, day, epoch, stop_idx)` | `stop_id` non-null and one-to-one with the natural key; `stop_idx = 0..n-1` within session |
| `event_arm_table` | One candidate arm for one event | `(animal, day, epoch, non_local_event, arm)` | Exactly one each of arms `0..8` per event; `(event_id, arm)` unique |
| `event_patch_table` | One candidate patch for one event | `(animal, day, epoch, non_local_event, patch)` | Exactly one each of patches `1..3` per event; `(event_id, patch)` unique; Phase 1 adds/broadcasts `non_local_event` explicitly |

The session key is exactly `(animal: str, day: integer, epoch: str)`. Builders that consume raw time-series (`event`, `sequence`, `stop`, and `arm`) accept one and only one non-null session per call. `build_event_patch_table` is a pure keyed projection and may accept several already-validated sessions; it must carry full natural/session keys and validate each event independently. Hash collisions, reused session-local IDs, mixed raw-session inputs, duplicate natural keys, and duplicate surrogate IDs are hard failures.

## Input contract

### `dot` and event population

Builder-specific required columns are validated before any `.iloc[0]`, groupby, or merge. At minimum, event builders require `animal`, `day`, `epoch`, `non_local_event`, `SWR_event`, `Non-Local_Continuous_prob`, `Non-Local_Fragmented_prob`, `map_estimate`, and `head_speed`; columns needed for MUA, trajectory, or trial context are required when that feature is requested rather than silently fabricated.

The `dot` index is time in seconds and must be numeric, finite, unique, and strictly increasing. Public builders do not silently sort it. A `dot["time"]` column must be absent or numerically identical to the index within the coordinate tolerance; the classified helper may not silently promote a competing timebase. Event and detector label columns are integer-like, nonnegative, and have one contiguous run per positive label. A positive label reused in disjoint runs is invalid. State probabilities and speed are numeric; infinities are invalid. The explicit policy for NaN in each optional signal is documented and tested rather than inherited from pandas comparison behavior.

For defaults `speed_threshold = 4 cm/s` and `nonlocal_threshold = 0.8`, a time bin is retained iff:

```text
non_local_event > 0
and head_speed < speed_threshold
and (Non-Local_Continuous_prob + Non-Local_Fragmented_prob) > nonlocal_threshold
```

Both comparisons are strict: speed exactly at the threshold and non-local probability exactly at the threshold are excluded. Event membership is the set of positive labels with at least one retained bin. Every retained label's bins must belong to exactly one row-defined immobility bout; a label with retained bins in multiple bouts fails before either event or stop counts are emitted. Thresholds, session, and that retained label set must match across event, sequence, arm, stop, and patch assembly. No helper may accept an event table generated with a different population merely because hashes happen to join.

The two-window contract is explicit:

- `event_start`, `event_end`, `event_duration`, `event_midpoint_time`, `n_nonlocal_bins`, `nonlocal_weight_sum`, `is_mua`, posterior summaries, arm responses, and sequence rows use retained bins.
- `label_event_start`, `label_event_end`, `label_event_duration`, `label_event_midpoint_time`, speed summaries, `peak_nonlocal_prob`, and SWR overlap fields use the complete positive label window.
- `is_swr` requires strictly positive temporal overlap between the full label interval and a positive SWR-label interval. Boundary touch without positive duration is not overlap.

Phase 1 pins this current convention. Alternative MUA/window semantics are diagnostics until an author-approved table-fix decision.

`MUA_event` is required for canonical event/stop construction; missing classification is not evidence of no MUA and may not become `False/0`. Trajectory inputs follow an all-or-none contract: if none of the declared trajectory source columns are present, all trajectory outputs are explicitly unavailable/NaN for a legacy input; if any is present, the complete declared source set is required. Strict candidate runs require that complete set.

### Trials, graph, environment, and contingency

- Empty trials are valid and produce the documented missing history/context values. Nonempty trials must contain the builder-required columns, have finite time bounds with `start_time <= end_time`, and have unambiguous arm/well mappings. Trial row order is not scientific: helpers stable-sort a copy by the documented chronological cutoff/start key, preserve the caller frame, and reject ambiguous tied records rather than inheriting input order. Sorted and shuffled copies must yield identical keyed values.
- The track graph has the expected bandit arm identifiers, unique edge IDs, required well nodes, and finite nonnegative edge distances. Arm/well/patch mappings are total for the declared real arms; unknown arms fail.
- `env.place_bin_centers_nodes_df_` supplies one `edge_id` per posterior position bin. Real edge IDs are within `0..8`; the declared junction sentinel is handled explicitly. Position coordinates must align with environment linear-position coordinates within `atol=1e-6`.
- Contingency/value rows are keyed unambiguously to `stop_id` and arm. Duplicate matches fail. A missing matched stop remains missing across all nine alternatives; it is not converted to zero. Retired HMM value columns are not part of the current candidate schema.

### Conditional posterior

When supplied, the posterior must have dimensions exactly `("time", "position")`, shape `(len(dot), len(edge_id))`, finite nonnegative entries, a time coordinate aligned to `dot.index` within `atol=1e-6`, and position coordinates aligned to the environment within `atol=1e-6`. Each row sums to one within `atol=1e-5`; dimensions are never guessed or transposed. A missing posterior is allowed only on builder paths whose documented schema uses NaN posterior summaries and omits sequence rows. A present but malformed posterior is a hard failure.

## Output contracts

All builders return the exact documented builder column order and dtype family on both nonempty and empty results. Assembly has separate exact schemas: retrospective columns extend the event table and SWR-split columns extend the stop table; those are never treated as arbitrary extras. Canonical identifiers/counts use stable integer or nullable-integer types, flags use boolean or explicitly nullable boolean types, categories use a declared categorical/string representation, and continuous quantities use float. Inputs are not mutated.

### Event table

- The event natural key and `event_id` are unique.
- `event_start <= event_end`, `event_duration = event_end - event_start`, and the filtered interval lies within the full label interval.
- `event_midpoint_time` and `label_event_midpoint_time` are the upper-middle observed samples (`times[len(times)//2]`), not arithmetic interval centers. `time_since_stop` is the median of retained-bin offsets from their one containing stop start.
- `n_nonlocal_bins` is the exact number of retained bins; `nonlocal_weight_sum` is their summed non-local state probability. Counts are nonnegative integers and finite sums are not clipped.
- `stop_id` is non-null and identifies the unique stop containing `event_start`; the full event-to-stop relationship is independently reconciled against retained raw bins.
- `is_swr == (n_swrs_overlapping > 0)`. Matched SWR fields are present iff required by the declared overlap class and missing otherwise.
- Posterior HPD widths are ordered `50 <= 80 <= 95` when present. Entropy, peak, and posterior mass obey their mathematically defined ranges subject only to the declared input tolerance.

### Sequence table

- Each row corresponds exactly to one retained raw time bin for a posterior-available session, with the same session/event identity, time, and `nonlocal_prob`.
- Within event, `time` is strictly increasing and `t_in_event` starts at zero and increments by one.
- `arm_posterior_0..8` are finite, nonnegative, and sum to one within the deterministic comparison tolerance after the declared junction policy. `decoded_arm_map` is the lowest-index argmax.
- Sequence event IDs are a subset of event-table IDs globally because posterior-unavailable sessions may lack sequence rows. For a posterior-available session, the event set and retained-bin counts are exact.

### Stop table and SWR split

- Every row-defined low-speed bout is present, including zero-event and single-sample bouts. `stop_start <= stop_end` and `immobility_time = stop_end - stop_start` under the currently pinned formula.
- Stops are ordered, nonoverlapping by the current row-contiguity rule, and indexed `0..n-1` per session.
- `n_events == n_non_local`, `has_replay == (n_non_local > 0)`, `is_swr_stop == (n_SWR > 0)`, and `is_mua_stop == (n_MUA > 0)` exactly.
- Distinct label counts are nonnegative integers. Absent MUA input follows one explicit tested policy; null classification is never coerced to false during a join.
- After `add_nonlocal_swr_split`, each stop satisfies `n_non_local_SWR + n_non_local_no_SWR == n_non_local`. The global sums reconcile exactly with event-table `is_swr`; missing keys, duplicate event IDs, null `is_swr`, and cross-session stop IDs raise.
- Reward lags are binary or missing as defined, and `reward_rate_trial_k10` equals the declared mean of its source trials. All context/arm/patch fields are evaluated at the documented bout-start time without forward fill.

### Event-arm table

- There are exactly nine rows per event, in declared arm order, regardless of the source DataFrame index.
- Per event, `sum(arm_count_hard) == n_nonlocal_bins - n_junction_bins`.
- Per event, `sum(arm_count_soft)` equals the sum of non-local weights over retained bins whose MAP maps to a real arm.
- When posterior is present, `sum(arm_posterior_mass) == posterior_mass_total == nonlocal_weight_sum` under the currently pinned real-arm renormalization convention. Posterior fractions sum to one for a positive denominator; unavailable responses and zero denominators preserve the documented NaN mask.
- Event-level fields broadcast exactly by event key. Each indicator equals its defining equality; `is_last_rewarded_elsewhere` has disjoint support from `is_current`; every interaction equals the product of its named factors with NaN propagation.
- Distances are finite/nonnegative where defined, ranks use the declared minimum-rank tie rule, trial-history predictors use only eligible prior trials, and standardized columns match the persisted pooled reference.
- `visit_recency_demeaned` is centered within arm inside each single-session builder call, hence within `(animal, day, epoch, arm)` after concatenation. Phase 2 recomputes that exact scope and fails if a later global-only transform replaces it.

### Event-patch table

- There are exactly three rows per event and no orphan, duplicate, missing, or unknown arm rows in the input projection.
- `patch_count_hard` is the exact sum of mapped arm counts and conserves each event's real-arm hard count.
- Cumulative reward uses `sum(min_count=1)` so all-missing stays missing. Value means use leaf arms only; visit recency uses the documented minimum.
- Current/previous/last-rewarded indicators and every SWR/switch interaction are recomputed exactly. Session keys and `is_swr` are required, total merges; absence is not interpreted as all-false.
- The identical patch projection consumed by the ContFrag preparation path is included as a cross-check, without expanding this audit to ContFrag event semantics.

## Cross-table bundle contract

After per-session construction and again after concatenation/standardization, use an exact expected-session manifest carrying source presence, retained-event count, immobility-bout count, and posterior availability:

1. Session participation is conditional and explicit: every manifest session is accounted for; stop rows exist iff raw data contain at least one immobility bout; event/arm/patch rows exist iff at least one label is retained; sequence rows exist iff a posterior loaded successfully and at least one bin is retained. A valid no-stop or zero-event session remains present in the manifest even when all corresponding table frames are empty.
2. Every event appears once in the event table, nine times in the arm table, three times in the patch table, and once per retained bin in the sequence table when posterior is available.
3. Every event `stop_id` exists in the stop table in the same session. Grouped event counts equal stop `n_non_local`; no event is double-counted across bouts.
4. `sum(stop.n_non_local) == len(event_table)` globally and per session. SWR/no-SWR stop splits equal event-table classifications globally and per stop.
5. Hard arm counts, patch counts, sequence row counts, soft weights, and posterior mass reconcile to their event-level denominators using the definitions above.
6. All cross-table broadcast columns are equal on natural keys, including NaN masks and dtype family.
7. All joins declare `validate=` cardinality. No many-to-many merge or silent dropped key is acceptable.
8. Pooled standardization is performed only after full concatenation, its reference is saved and hashed, and recomputation matches the stored columns.

Schema-complete empty event/arm/patch/sequence tables and a populated zero-event stop table are valid. An entirely empty run is a configuration failure, not a table bundle.

## Comparison policy

Comparison settings are written into every audit/rebuild manifest before reading candidate outcomes:

| Quantity | Policy |
| --- | --- |
| Ordered schema and dtype family | Exact |
| Natural/surrogate keys, session sets, counts, labels, ordinals, flags, categories | Exact |
| NaN mask | Exact before any numerical comparison |
| Infinities | Forbidden |
| Deterministic calculations from table values | `atol=1e-12, rtol=1e-10` |
| Raw float32/source-to-table comparisons | `atol=1e-6, rtol=1e-7` |
| Time and position coordinate alignment | `atol=1e-6, rtol=0` |
| Conditional-posterior row normalization at input | `atol=1e-5, rtol=0` |

The baseline/candidate comparator ignores row order only after checking uniqueness. It reports added/dropped keys, schema/dtype changes, NaN-mask changes, exact changes, and numeric deltas separately. A known intended change needs a predeclared `expected_change_id`, rationale, affected columns/keys, and oracle. There is no blanket schema or floating-point allowlist.

## Finding and severity contract

Every finding has: `finding_id`, phase/run ID, severity, check ID, scientific grain, affected table/columns, first keys, count/fraction, observed/expected, independent evidence, downstream claims/models, status, owner, and resolution PR/run.

- **Hard:** invalid key, schema, input, provenance, path isolation, session population, or finite/range contract. Blocks all scientific sign-off.
- **Identity:** exact algebraic/referential/cross-table failure. Blocks all scientific sign-off.
- **Scientific-semantic:** implementation is internally consistent but the intended definition is unresolved or contradicted by an independent oracle. Produces `INCONCLUSIVE` until decided and rerun.
- **Diagnostic:** unusual distribution, missingness, or robust outlier with no violated identity. Requires review and documented disposition; it does not fail solely for being biologically surprising.
- **Claim-impact:** table is correct but a downstream estimand changes enough to require scientific/paper review. It cannot turn a hard/identity failure into a pass.

## Artifact isolation and provenance

- Canonical inputs are the five pickles, `standardization_reference.json`, q-values/contingency, event-onset/time-bin auxiliary tables, raw `data_over_time`, environment, posterior results, and source-attested session bundle selected for the run.
- Before work, record canonical `(absolute path, size, mtime_ns, sha256)` for tractable canonical files. For the 271 GB posterior corpus, record path/size/mtime and hash the explicit session files only when read by the staged rebuild; never hash or scan them in Phase 2 merely for completeness.
- Phase 1/2 generated data live under ignored `scratch/tables-review/<run_id>/`. Phase 3 creates/finalizes a noncanonical run only through regression-integrity Phase 7's staging API and never advances `CURRENT.json`; staging is finalized at directory level before it becomes resolvable by explicit run ID. Inputs are opened read-only where the library supports it.
- The manifest records schema version, run ID, git commit, dirty diff hash, lock/environment hash, host/Python/package versions, thresholds, session list, source/cache hashes, CLI arguments, seed, comparison tolerances, start/end, status, and output hashes.
- Phase 3 accepts only a Phase-7 source-audit/session-bundle manifest with complete expected-session accounting and exact source/cache identity. It never substitutes a presence-only cache, ad hoc fresh-fetch spot check, or direct Spyglass fallback.
- Candidate replay tables, behavior/Q tables, event-onset/time-bin tables, and decision tables resolve from one generation-coherent run manifest. If a corrected event/stop population invalidates an auxiliary, regenerate and finalize that dependent component in the same candidate; never pair it with a frozen legacy auxiliary.
- Recompute canonical hashes after every stage and before verdict. Any size/mtime/hash change is `FAIL`, even if content appears equivalent.

## Claim-impact and verdict contract

One claim-impact row represents one atomic estimand, not a paragraph containing several effects. Required fields are: claim ID/location/status; source builder/table/columns; runner/model/term/population/output; baseline and candidate event/stop/session/animal counts; added/dropped and common cohorts; baseline/candidate estimate, SE/CI, animal signs, session-sign median, FDR; absolute, relative, and log-rate-ratio delta; expected-change ID; support verdict; wording action.

Allowed support states are `unchanged`, `strengthened`, `weakened-but-supported`, `reversed`, `unsupported`, and `unresolved`.

Predeclared review triggers:

- A headline directional effect must retain its sign on the common cohort and the stated across-animal support. At least 4/5 animal medians is the minimum for a general five-animal direction; a claim that says 5/5 must remain 5/5 or be rewritten.
- Candidate-versus-baseline common-cohort coefficient change within `max(0.10 log units, 20% of |baseline|)` is labeled numerically stable. A larger change forces claim review; it is not automatically a table failure.
- Rate-ratio comparisons occur in log space and preserve any claimed ordering, especially the immediate reward-gradient components of claim #19c.
- Negative claims require a scientifically predeclared equivalence bound and CI. `p > 0.05` is not support.
- Complete-case loss, convergence/separation, estimability, CV direction, animal/session sign changes, and FDR-family crossings are explicit fields and review gates.
- Bounds and tolerances cannot be chosen after observing candidate results.

Final verdicts:

- **`PASS`** — provenance, contracts, independent oracles, full audit, required refits, and canonical hash guards all pass; there is no unexplained drift and all claims remain supported.
- **`PASS—PAPER REVISION REQUIRED`** — tables are correct, but one or more supported claims, numbers, cohorts, or 5/5 statements change; promotion is blocked until authors approve and update paper/docs/figures.
- **`FAIL`** — any unresolved hard/identity/provenance breach or unexplained cohort/value drift remains.
- **`INCONCLUSIVE`** — an independent oracle, frozen input, upstream/cache determination, scientific definition, or required refit is missing.

Sensitivity agreement alone can never produce `PASS`.

Verdict precedence is deterministic: (1) any unresolved provenance/hard/identity breach yields `FAIL`; (2) otherwise any missing independent oracle, source-attested generation, required Phase-7-12 API, exact claim specification, coherent auxiliary, corrected estimator, or owner completion manifest yields `INCONCLUSIVE`; (3) otherwise correct tables plus any reversed/unsupported/wording-changing atomic claim yield `PASS—PAPER REVISION REQUIRED`; (4) only correct tables with every directly dependent claim supported and every mixed/raw claim owner complete yield `PASS`. A table can therefore be correct while the overall paper-facing verdict requires revision; obsolete legacy estimator agreement cannot improve the verdict.
