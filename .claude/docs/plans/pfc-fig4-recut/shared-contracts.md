# Shared Contracts

[← back to PLAN.md](PLAN.md)

Two contracts span multiple phases. Each is defined once here; phases link in.

- [Provenance sidecar schema](#provenance-sidecar-schema) — written by Phase 1 (`run_session`), populated at scale by Phase 3 (batch runner), consumed by Phase 4 (loader guard).
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
    "nwb_file_name": str,              # e.g. "wilbur20210329_.nwb"
    "epoch_name": str,                 # e.g. "02_r1"
    "brain_area": str,                 # "mPFC"
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

---

## Matched-control interface

Phase 2 adds one public helper to `analysis/pfc/glm.py`. Its signature and
output columns are frozen here because Phase 3's rerun depends on them and the
divergence guard tests them.

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
      - control count == real event count, and per-event duration is preserved.

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
