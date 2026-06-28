# Shared contracts

[← back to PLAN.md](PLAN.md)

The single coupling between the two phases is the **contingency table** Phase 1
produces and Phase 2 consumes. Phase 2 binds to this *column schema only* — not
to how it was parsed — so the eventual canonical parsed-events table can replace
Phase 1 by reproducing these columns.

## contingency-table-schema

`Processed-Data/replay_content/contingency_table.pkl` — one row per stop, keyed
by `stop_id` (left-joinable to `stop_table`). Nominal reward probabilities are
fractions in `[0, 1]` (the log stores percent; divide by 100). "Port" = one of
the 6 reward wells; at a stop the animal is at `current_arm ∈ {0..5}`, which
indexes the port.

| column | type | semantics |
| --- | --- | --- |
| `stop_id` | int64 | join key to `stop_table` |
| `block_number` | int | 0-indexed contingency block within the session |
| `trials_since_block_change` | int | trials since the current block began (0 on the first trial of a block) |
| `p_reward_port0..5` | float | nominal `p(R)` of each of the 6 ports in the active block |
| `p_reward_current` | float | `p_reward_port{current_arm}` — value of the port the animal is at |
| `p_reward_best` | float | `max` nominal `p(R)` over the 6 ports in the block |
| `is_best_patch` | bool | current patch == the block's best patch (highest two-port average) |
| `rpe` | float | `reward_outcome − p_reward_current` (per-stop reward prediction error; nominal) |
| `value_diff_patch` | float | best-patch value − second-best-patch value (per block); small = high global decision conflict |

Invariants (assert in Phase 1, do not weaken):
- `p_reward_*` ∈ `{0.0, 0.2, 0.5, 0.8, 1.0}` (the documented contingency values;
  1.0 only in the first session).
- exactly one `block_number` per `(session, trial)`; `block_number` increments at
  the logged `contingency` change times.
- every `stop_id` in `stop_table` for a parsed session maps to exactly one row
  (or is explicitly dropped — e.g. the excluded first session).
- `rpe` uses the same binary `reward_outcome` already in `stop_table`.

## uncertainty-covariates (Phase 2, fit-time; not persisted here)

Model-free signals derived at fit time from existing `stop_table` columns (no
new data), defined in full in [phase-2-uncertainty-rate.md](phase-2-uncertainty-rate.md):
`omission_streak`, `recent_switch_rate`, `wsls_deviation`, `leaky_q_value_diff`.
They are the *experienced/subjective* complement to the *nominal* values above;
both enter the Phase-2 rate model so nominal and experienced value are
dissociable.
