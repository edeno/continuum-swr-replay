# Shared Contracts

[← back to PLAN.md](PLAN.md)

The following contracts span multiple phases. Each is defined once here; phases link in.

- [Provenance sidecar schema](#provenance-sidecar-schema) — written by Phase 1 (`run_session`), populated at scale by Phase 3 (batch runner), consumed by Phase 4 (loader guard).
- [PFC completion manifest](#pfc-completion-manifest) — Phase 3 supplies the
  source-bound batch evidence; Phase 4 is the sole producer after every output and
  validation gate passes.
- [Matched-control interface](#matched-control-interface) — defined by Phase 2, consumed by Phase 3's rerun.
- [PFC fit-validity contract](#pfc-fit-validity) — defined by Phase 0, consumed by Phases 2b–4.
- [Temporal-null and blocked-score contract](#pfc-temporal-inference) — defined by Phase 2b, consumed by Phases 3–4.

---

## Provenance sidecar schema

Every `run_session` call writes, alongside the modulation artifacts, a JSON
sidecar recording how the events were defined. **Do not weaken**: Phase 4's
loader raises if a consumed pickle has no sidecar or if the `is_swr_definition`
tag is not `"canonical_event_table"`.

**File name:** for save base `{epoch_identifier}_modulation`, the sidecar is
`{epoch_identifier}_modulation.provenance.json` (same directory, `PROCESSED_DATA_DIR`).

**Schema:**

```python
{
    "pfc_run_id": str,                  # identical across the production batch
    "nwb_file_name": str,              # e.g. "wilbur20210329_.nwb"
    "epoch_name": str,                 # e.g. "02_r1"
    "brain_area": str,                 # "mPFC"
    "canonical_table_generation_sha256": str,
    "session_input_manifest_path": str,
    "session_input_manifest_sha256": str,
    "canonical_event_table_path": str,
    "canonical_event_table_sha256": str,
    "is_swr_definition": str,          # MUST be "canonical_event_table"
    "immobility_threshold": float,     # cm/s
    "immobility_inequality": str,      # MUST be "<" (see Open Question 1)
    "non_local_threshold": float,
    "n_events_total": int,             # non-local events (immobile)
    "n_events_swr": int,               # is_swr == True
    "n_events_non_swr": int,           # is_swr == False
    "n_matched_control": int,          # Phase 2 onward; 0 before
    "git_hash": str,                   # from the runner's git_hash arg
    "code_version": str,               # analysis.pfc.__version__ or module git hash
    "fit_schema_version": int,          # Phase-0 validity schema
    "nuisance_basis": dict,             # Phase-2b history/drift basis
    "temporal_null": dict,              # method, seed, requested/valid permutations
    "blocked_score_hash": str,          # common contiguous held-out blocks
}
```

Written with `json.dump(..., indent=2)`. The counts come from the canonical
event table restricted to immobility (Phase 1); `n_matched_control` is filled
once Phase 2 lands (0 in Phase 1's sidecar).

**Amended 2026-09-14 (shipped shape).** The sidecar carries four more
matched-control keys than the block above lists, all written from `cohort_info`:

```python
    "matched_control_seed": int,        # session-keyed, from matched_control_seed()
    "n_matched_control_unplaced": int,  # source events with no control window
    "n_matched_control_onset_outside_fitted_bout": int,
    "n_matched_control_bout_saturated": int,
```

`n_matched_control + n_matched_control_unplaced == n_events_total`, and the last
two sum to `n_matched_control_unplaced`. They are kept apart because they mean
different things: an onset outside every fitted bout is a cohort fact (an event
is kept by its onset bin, which can lie outside the fit mask), while a saturated
bout is an event-density fact (the bout is too short, or too full of real
events, to hold one more window of that duration). A session with many of the
second has a control population thinner than its events exactly where events
cluster, which is the case where the engagement test loses power. Production sidecars must bind to
the frozen staged canonical-table generation and per-session input manifests
created by regression-integrity Phase 13. A sidecar whose source hash differs
from that generation is stale even when its artifact files are present.

---

## PFC completion manifest {#pfc-completion-manifest}

`pfc_completion.json` is the single cross-plan gate consumed by
`regression-integrity-fixes` Phases 12–13. Phase 3 never writes this file: it
produces the source-bound artifacts and batch-audit evidence. Phase 4 is the sole
producer and may emit it only after the population loader, all inference checks,
the staged Figure-4/adjudication smoke test, and the full completion
audit succeed.

**Immutable path:** `<pfc_root>/runs/<pfc_run_id>/pfc_completion.json`. Phase 3 writes every
artifact beneath `<pfc_root>/runs/<pfc_run_id>.staging/`; Phase 4 writes the completion file there
last and atomically renames the whole directory to its final path. The regression release gate is
given that exact path and hash. Recursive search, newest-file fallback, mutation of a finalized run,
and a PFC-specific `CURRENT`/canonical pointer are forbidden.

**Schema:**

```python
{
    "schema_version": 1,
    "completed": True,
    "pfc_run_id": str,
    "created_at_utc": str,
    "git_sha": str,
    "dirty_diff_sha256": str | None,
    "canonical_table_generation": {
        "manifest_path": str,
        "sha256": str,
        "state": "frozen_staged",
    },
    "coverage": {
        "expected_session_ids": list[str],
        "completed_session_ids": list[str],
        "excluded_sessions": list[dict],  # session_id + reviewed reason
        "expected_animals": list[str],
        "observed_animals": list[str],
    },
    "sessions": [
        {
            "session_id": str,
            "animal": str,
            "session_input_manifest_path": str,
            "session_input_manifest_sha256": str,
            "canonical_event_table_sha256": str,
            "provenance_path": str,
            "provenance_sha256": str,
        }
    ],
    "contracts": {
        "is_swr_definition": "canonical_event_table",
        "immobility_inequality": "<",
        "fit_schema_version": int,
        "spike_bin_edges": "half_open_with_final_right_edge",
        "matched_control": True,
        "nuisance_basis_sha256": str,
        "temporal_null_method": str,
        "blocked_score_hashes": list[str],
        "population_unit": "neuron_to_session_to_animal",
    },
    "outputs": [
        {
            "root": str,  # "pfc_run" or "release_output"
            "path": str,  # normalized path relative to that root
            "size": int,
            "sha256": str,
        }
    ],
    "validation": {
        "batch_audit_path": str,
        "batch_audit_sha256": str,
        "missing_required_artifacts": 0,
        "source_hash_mismatches": 0,
        "forbidden_stale_lr_columns": 0,
        "fit_diagnostics_path": str,
        "fit_diagnostics_sha256": str,
        "expected_fit_cells": int,
        "accounted_fit_cells": int,
        "fit_status_counts": {
            "ok": int,
            "rank_deficient": int,
            "nonconverged": int,
            "separated": int,
            "insufficient_data": int,
            "fit_error": int,
        },
        "missing_fit_status": 0,
        "headline_invalid_fit_cells": 0,
        "all_attempted_cells_accounted": True,
        "five_animal_coverage": True,
        "loader_smoke_passed": True,
        "figure_smoke_passed": True,
        "targeted_tests_passed": True,
    },
    "claim_20_adjudication": {
        "status": str,  # "supported", "not_supported", or "unavailable"
        "estimand": "non-SWR replay engagement beyond matched immobile control",
    },
}
```

All PFC-scoped rows in the Phase-13 `expected_sessions.parquet` snapshot must appear exactly once in
either `completed_session_ids` or `excluded_sessions`; exclusions must resolve to a pre-run,
PFC-scoped reviewed entry in the shared `config/regression_session_exclusions.yaml`, not a private
PFC allowlist. Every completed session must have a unique session record whose
source/event/provenance hashes match both its sidecar and the frozen Phase-13
generation. The completion `git_sha` and every session-sidecar `git_hash` must equal the clean,
frozen analysis-code/config commit recorded by the Phase-13 candidate, and
`dirty_diff_sha256` must be null. Generated figures, reports, and claim-text outputs are hashed
separately and do not authorize a different analysis-code identity. `observed_animals` must equal
the declared five-animal set. A null
engagement result may still produce `completed=true` with
`claim_20_adjudication.status="not_supported"`; completion records validity, not
statistical significance.

The completion manifest never hashes mutable live-repository files. Figure-4 PDF/PNG and the
machine-readable `claim_20_adjudication.json` are written to the global release-output staging tree
under repository-relative paths and recorded with `root="release_output"`; PFC population/session
artifacts use `root="pfc_run"`. Shared Markdown claims documents are deliberately excluded because
Phase 13 composes them later from all row-scoped adjudications. The global release validator rechecks
the PFC-declared release-output bytes before finalizing that tree.

The producer is `scripts/qa/finalize_pfc_completion.py`, added in Phase 4. It
constructs the manifest only from freshly re-read files, writes canonical
sorted-key JSON plus a trailing newline to a uniquely named temporary sibling
inside the staging run, flushes and `fsync`s it, re-reads and validates the complete schema and
hashes, then atomically renames the staging run directory to its immutable final path and `fsync`s
the runs directory. Any exception removes only the temporary completion sibling and leaves the
staging run incomplete/non-resolvable; it never changes a finalized run or global pointer. The
script must not accept `--allow-incomplete`, synthesize missing evidence, or reuse another run's
completion file.

---

## Matched-control interface

Phase 2 adds one public helper to `analysis/pfc/glm.py`. Its signature and
output columns are frozen here because Phase 3's rerun depends on them and the
divergence guard tests them.

**Amended 2026-09-14 (shipped shape).** Two departures from the signature
below, both forced by what the placement rule needs. First, the helper takes the
whole *mapping* of event classes `build_session_event_sets` returns
(`non_local_no_SWR`, `non_local_SWR`, `non_local`, `SWR`), not one event frame:
`non_local` supplies the source events, but a control must avoid the windows of
every class — a bare `SWR` belongs to no non-local class, so a helper given only
the non-local frame could place a control on top of one. It also takes
`max_attempts` (default 50). Second it returns a `MatchedControlDraw` named
tuple — `times` (the DataFrame the signature below promises), plus
`n_onset_outside_fitted_bout` and `n_bout_saturated`, the two causes of the
shortfall, with `n_unplaced` as their total. Callers use `.times` where the
signature below says the return value is the frame. `is_immobile` is documented
as the *fit mask*, not raw immobility, so the control samples the bins the
models are fit on.

**Amended 2026-09-16 after the exact-null smoke.** Legal onset bins are
enumerated directly. Control onsets are distinct, but their auxiliary
duration-matching windows may overlap one another. The GLM design contains
only onsets; treating duration windows as occupied time made exact class-count
matching impossible on the gate sessions (j16 had zero exact draws for every
class). Every window still lies in its source event's bout and avoids every
real event window. The placement method is versioned in each sidecar so the
two policies cannot be pooled.

```python
def make_matched_control_times(
    event_times: pd.DataFrame,      # canonical events (immobile), cols start_time/end_time
    is_immobile: pd.Series,         # boolean, indexed by position_info.index (time in s)
    rng: np.random.Generator,       # seeded; caller passes np.random.default_rng(seed)
) -> pd.DataFrame:
    """Time-shifted-onset matched non-event control.

    For each real event, draw a control onset within the *same* immobility
    bout, offset from the real onset, such that:
      - the control window [start, start + duration) lies entirely in immobile bins,
      - it does not overlap ANY real event window,
      - control onsets are distinct, and per-event duration is preserved.

    Returns a DataFrame with columns ``start_time``, ``end_time`` (same schema
    as ``event_times``), one row per real event. Rows that cannot be placed
    after a bounded number of attempts are dropped (logged), so the caller must
    treat the returned count as ``<= len(event_times)``.
    """
```

**Invariants the Phase 2 unit test pins (do not weaken):**

- Every returned window is fully within immobile bins.
- Zero overlap between any control window and any real event window.
- Control count ≤ real count; each control duration equals its source event's duration.
- Deterministic given the same `rng` seed.
- Control onset bins are distinct; duration windows are not required to be
  mutually disjoint because they are not part of the fitted design.

The matched-control onsets enter `_make_design_matrices` as an additional event
class `"matched_control"`, producing a `matched_control` design matrix parallel
to `non_local_no_SWR`. Phase 2 adds the LR test pair
`("non_local_no_SWR", "matched_control")` — the key engagement test: does non-SWR
replay modulate firing *beyond* a rate/duration-matched non-event baseline.

---

## PFC fit-validity contract {#pfc-fit-validity}

Every neuron × model attempt records `fit_status`, failure reason, design rank/columns,
scaled condition number, convergence, and event counts. Rank-deficient,
nonconverged, or non-finite models cannot contribute LR/AIC/CV/modulation outputs;
those fields are NaN with an explicit unavailable reason. Spike bins are half-open
`[time[i], time[i] + dt)` with one count per left edge and a final right edge
`time[-1] + median(diff(time))`.

## Temporal-null and blocked-score contract {#pfc-temporal-inference}

All candidate event models share spike-history and slow-time nuisance columns. Primary
event inference uses seeded circular time shifts or whole-immobility-block permutations
that preserve spike autocorrelation and event counts. Predictive comparison uses
contiguous held-out immobility blocks with identical assignments across models.
Manifest fields record seed, number of valid permutations, nuisance-basis definition,
block boundaries/hash, held-out deviance, and calibration status. Nominal chi-square LR
and in-sample AIC may be retained only as descriptive diagnostics.
