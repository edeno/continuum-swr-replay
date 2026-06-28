# Phase 1 — Temporary StateScript contingency parser

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Parse the per-port nominal `p(R)`, block index, and change times out of the
Spyglass `StateScriptFile` log and align them to trials/stops, producing the
[contingency-table](shared-contracts.md#contingency-table-schema). **Exploratory
and temporary** — a hand-rolled adapter to unblock the Phase-2 analysis,
explicitly marked to be replaced by the canonical parsed-events table when it
becomes available. Do **not** import `trodes_to_nwb`; adapt its approach.

**Inputs to read first:**

- Spyglass `StateScriptFile` access pattern (verified): `(StateScriptFile & {nwb_file_name, epoch}).fetch_nwb()[0]["file"].content` → full runtime log string. Runtime lines of interest: `"<ts_ms> contingency = <int>"` and `"<ts_ms> leafProbs1..6 = <pct>"`.
- `trodes_to_nwb/convert_statescript.py` (branch `statescript`) — **reference to ADAPT, not import**: copy the `<ts> <name> = <int>` line-parse and the DIO-offset alignment idea (`calculate_time_offset` → a constant offset to a reference event series). Add an attribution comment with the commit hash.
- `src/continuum_swr_replay/data_loaders/events.py:203` — `pump_events` DIO loader (reward pump activations, in neural/session time) = the reference series for clock alignment.
- `Processed-Data/replay_content/stop_table.pkl` — `stop_id`, `stop_start`, `current_arm`, `current_patch`, `reward_outcome` (the stops to annotate); the `TRACK_SEGMENT_TO_PATCH` mapping in `data_loaders/constants.py` for patch grouping.

**Contracts referenced:**

- [contingency-table-schema](shared-contracts.md#contingency-table-schema) — produce exactly these columns + invariants.

## Tasks

- Create an exploratory module `scripts/_statescript_contingency.py` (scratch-tier,
  `_`-prefixed; header comment: adapted from `trodes_to_nwb@statescript` +
  "TEMPORARY — replace with the canonical parsed-events table when available").
  Minimal line parser (no `trodes_to_nwb` import):

  ```python
  import re
  _ASSIGN = re.compile(r"^\s*(\d+)\s+([A-Za-z]\w*)\s*=\s*(-?\d+)\s*$")

  def parse_assignments(content: str) -> pd.DataFrame:
      """(ts_ms, name, value) for every `<ts> name = int` runtime line."""
      rows = [(int(m[1]), m[2], int(m[3]))
              for ln in content.splitlines() if (m := _ASSIGN.match(ln))]
      return pd.DataFrame(rows, columns=["ts_ms", "name", "value"])
  ```

- **Reconstruct the block timeline.** From the assignments: each `contingency = k`
  marks a (re)statement of the active block; the `leafProbs1..6 = pct` lines that
  follow set the six port probabilities (percent → `/100`). Build a per-block
  record: `(block_number, t_start_ms, p_reward_port0..5)`. `block_number` is the
  0-indexed order of *distinct* contingency epochs (the log restates values
  often — collapse consecutive identical (contingency, leafProbs) states; a new
  block begins when the prob vector changes). Note the documented schedule (60-
  trial blocks ×3, or 80 ×2 for one animal; first session 100@1.0 then 100@0.5)
  as a cross-check, not as the parse rule.

- **Align the StateScript ms-clock to session/neural time.** *Verified empirically
  (chimi20200212, epoch 2):* StateScript timestamps are **milliseconds** at the
  **same clock rate** as the neural clock (StateScript span 2638.13 s vs DIO pump
  span 2638.34 s over ~44 min — match to ~0.2 s → no drift/scale), while the
  neural/DIO clock (`pump_events`) is **Unix epoch seconds** (PTP, ≈1.58e9). So
  alignment is a single **constant additive offset**: `t_sec_neural =
  ts_ms/1000 + offset`. Solve `offset` by matching StateScript pump events to DIO
  `Pump%` ON times (this session: 149 pump-ONs) — adapt `calculate_time_offset`
  (try candidate offsets from event pairings, pick the one minimizing residual,
  assert residual < ~50 ms over all matched events; also assert the StateScript
  span matches the DIO span as a unit/rate sanity check). StateScript reward/pump
  events appear as `"<ts> UP <port>"` / `"<ts> DOWN <port>"` DIO lines and
  `countRewards#` / `totalRewards` increments — use the pump-port UP edges (or
  reward increments) as the reference series. Apply `offset` to each block's
  `t_start_ms` → `t_start_sec`.

- **Map stops → block + nominal value.** For each stop (`stop_start`), find the
  active block (last `t_start_sec ≤ stop_start`); look up `p_reward_port0..5`;
  set `p_reward_current = p_reward_port{current_arm}`; `p_reward_best = max`;
  `is_best_patch` via the best-patch (highest two-port average, using
  `TRACK_SEGMENT_TO_PATCH`); `rpe = reward_outcome − p_reward_current`;
  `value_diff_patch = best − second-best patch value`; `trials_since_block_change`
  = stop's rank within its block (stops are port visits ≈ trials — verify the
  correspondence on one session). Write
  `Processed-Data/replay_content/contingency_table.pkl` keyed by `stop_id`.

- **Runner + provenance.** Iterate sessions via the existing session discovery
  (`scripts/prepare_replay_content_data.find_sessions`); log per-session parse
  success and any sessions where alignment fails (skip, don't crash). Print a
  summary (n sessions parsed, value histogram, block counts).

## Deliberately not in this phase

- **No Phase-2 analysis / covariates / fitting** — this phase only produces the
  contingency table.
- **No `stop_table` builder integration / regen** — write the companion
  `contingency_table.pkl`; folding columns into the canonical builder is a future
  cleanup (and the canonical parsed-events table may make it moot).
- **No `trodes_to_nwb` dependency** — adapt the ~20 lines that matter.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_statescript::test_parse_assignments` | On a synthetic log snippet, extracts the right `(ts, name, value)` rows; ignores `#`-comments and non-assignment lines. |
| `test_statescript::test_block_reconstruction` | A synthetic assignment stream with two contingency epochs → 2 blocks with correct `p_reward_port*` (percent→fraction) and start times; consecutive identical restatements collapse to one block. |
| `test_statescript::test_value_invariants` | Reconstructed `p_reward_*` ∈ {0,0.2,0.5,0.8,1.0}; `value_diff_patch ≥ 0`; `rpe` matches `reward_outcome − p_reward_current`. |
| real-session parse (slow) | ≥3 sessions across ≥2 animals parse; first session shows `p=1.0` then `0.5`; block lengths ≈ 60 trials; alignment residual below threshold; every `stop_id` mapped. Spot-check one block's `p(R)` vector against the raw log by eye. |

Mark the real-session parse slow.

## Fixtures

- **Synthetic**: a hand-written StateScript snippet (a few `contingency=` /
  `leafProbs#=` / pump lines with timestamps) covering two blocks and a percent→
  fraction conversion — drives the unit tests deterministically.
- **Real-data**: one Spyglass `StateScriptFile` content (fetched in the slow
  test) for end-to-end parse + alignment.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- No `trodes_to_nwb` import; the adapted lines carry attribution + the commit hash; the module is clearly marked temporary/exploratory.
- The produced table matches [contingency-table-schema](shared-contracts.md#contingency-table-schema) exactly (names, dtypes, invariants), so Phase 2 binds to the schema, not the parser.
- Block reconstruction collapses restated-but-identical contingencies; alignment is validated against DIO pump times, not assumed.
- Validation slice passes; real-session parse marked slow; value invariants asserted.
- Docstrings/test/module names don't reference this plan.
