# Review fixes, reported values (Parts 2-4): ledger

Branch `review-fixes-reported` (worktree `continuum-swr-replay-fix-reported`), off master 28e66ac9.
Started 2026-10-29.

## Baseline

- Summaries for Parts 2, 3 and 4 rebuilt at 28e66ac9 reproduce the committed `paper/reported/part{2,3,4}.json`
  exactly (0 values differ). Part 2 needs the ignored data folders `paper/figures/reality_all`,
  `paper/figures/content_occupancy` and `.claude/research/refresh_followup`; they are symlinked from the main
  checkout into this worktree (untracked, never committed).

## Findings while fixing

- Pooled duration-aware fit j16 / non_local_no_swr / `Mselfcross_dur` (and `Mselfcross_onset_dur`, shipped
  `Mselfcross`) did not converge. It feeds `history_cross_end_*_non_swr_from_swr`.
- wilbur / non_local_no_swr: the 15 ms end-locked lag has SE 0.503 > `HISTORY_MAX_SE` (not measurable).
- Per-session duration-aware fits: many are flagged unconverged (e.g. `Mselfcross_dur` non_local_no_swr 77 of 233).

## Log
- afa38b9a gamma pass records the event-table SHA-256 (per epoch and manifest); reuse gated on it.
  Timing: one epoch 65 s wall incl. start-up (`j1620210708_08_r4_HPC`), output identical to both existing
  passes; earlier manifests sum to 3.1 h and 4.0 h serial CPU, so 16 workers is well under 3 h. Full rerun
  chosen (no fallback on event ids needed).
- dca667a8 Part 3 history fixes (B); 8a02c6f1 Q8 day clustering and runner (C); 6883847f bootstrap strata and
  counts (D); 8e1a56f1 Part 2 staleness and no-drop (A, E, G); 696e0157 Part 4 staleness, docs, tests (A, E, F).
- Runs at 696e0157 with `CSR_PROCESSED_DATA_DIR=/cumulus/edeno/continuum-swr-replay/Processed-Data` (so manifests
  name the main checkout's paths, not this worktree's symlink). Backups `<dir>_pre_review_fix_20261029` of
  event_gamma, event_gamma_power, event_property_bootstrap, ripple_class_structure,
  pfc_response_followups/stop_context.
  - bootstrap: 1 min 51 s wall, 24 CPU s; 0 NaN replicates and 0 replicates skipping a stratum for every
    property; class medians and raw differences unchanged; matched differences move by at most 0.23 (hpd, cm),
    intervals by at most 0.64 (speed high, cm/s), from the whole-sample strata.
  - tmux `gamma_rerun` (event_gamma then event_gamma_power, 16 workers), `q8_day_cluster` (fit only; the 75
    session caches record the current tables), `ripple_class_analyze`.
  - A first launch from the worktree path was stopped after ~2 min and relaunched with the env var (partial
    outputs are rewritten by the relaunch).
  - Q8 fit (`q8_day_cluster`): 5.5 min wall (330 s), commit 696e0157 clean. 11 recording-day clusters (75 epochs),
    ci_df 10, every leave-out estimated (5 of 5). Point estimates identical to the epoch-clustered fit; most
    intervals widen (e.g. non-SWR replay, called increase units, reward: 1.007-1.106 -> 0.998-1.117, now
    crossing 1).
  - ripple class analyze: 6 min 40 s wall; distribution.csv, step_fits.csv, labels.csv byte-identical to the
    previous `-dirty` run; manifest now at clean 696e0157. Its spike_evidence path names this worktree's
    symlink (`paper/figures/reality_all` -> main checkout); the sha256 beside it identifies the files.
- Per-session duration-aware cross AIC: only 33 / 36 of 242 sessions have both rungs converged and not
  separated (non_local_no_swr: j16 1/58, wilbur 1/45). Values computed over these are made pending when under
  half the sessions are usable (`HISTORY_MIN_USABLE_SESSION_SHARE = 0.5`), applied after the gamma run ends.
  - gamma (`gamma_rerun`): event_gamma 13 min 18 s, event_gamma_power 13 min 17 s wall, 16 workers, 242/242 ok,
    manifests at clean 696e0157 with the event-table SHA-256. event_gamma_power byte-equal to its backup (plus the
    hash); event_gamma equal on every old column, and now also carries the power columns of the later code.
- b13d7c9a per-session cross-AIC pending rule; docs/regression_pipeline_map.md regenerated (new public names).
- Tests: `uv run pytest tests/unit tests/analysis -q -x` at b13d7c9a: 5956 passed, 15 skipped (49 min 40 s under
  load average ~295 from other jobs; 9 min 20 s earlier on a quieter machine, where the only failure was the stale
  pipeline map).
- ef6efd1e summaries (commit recorded b13d7c9a): Part 2 331 values (10 pending, unchanged set), Part 3 917 (47
  pending, +6), Part 4 606 (2 pending, unchanged). 2f28834f emit: 23 numbers changed in docs (changes.md,
  2026-10-30 section); the overview's stop-context sentence now says the intervals cluster by recording day.

## Changed values (from the summaries)

- Part 2: every `*_matched_difference{,_low,_high}` (whole-sample strata); medians, raw differences and their
  intervals unchanged. The gamma values are not pending (both passes record the current table).
- Part 3: `history_fits` 20 -> 19. Newly pending: `history_cross_end_rr_non_swr_from_swr` (2.28 -> 2.42, j16
  dropped), `history_cross_end_animals_non_swr_from_swr` (5 -> 4), `history_cross_session_gain_{non_swr_from_swr,
  swr_from_non_swr}` (-6.07 -> -3.36, -5.45 -> -4.66), `history_cross_session_animals_positive_*` (0 -> 1, 1 -> 0).
- Part 4: all 20 quoted stop-context interval ends (`stop_context_*_ci_{low,high}`); point estimates, animal counts
  and leave-out ranges unchanged. `stop_context_reward_gain_non_swr_replay_increase_ci_low` now 0.998 (< 1).

## For the lead

- The claims table and overview still say the per-session cross block "does not pay for its parameters ... so it
  is not a directed coupling"; the values behind that are now pending. Prose left unchanged.
- The pooled j16 non-SWR `Mselfcross_dur` fit did not converge; not refit (needs a pooled rerun of that cell).
- Most per-session duration-aware fits are flagged unconverged or separated; a refit with more IRLS iterations
  would be needed before the per-session cross AIC can be quoted.
