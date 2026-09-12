# Rate ladders across the three stop cohorts

Generated 2026-09-12 by `scripts/qa/compare_fits.py --cohort-contrast`.

Command: `uv run python scripts/qa/compare_fits.py --cohort-contrast`

The same ladder, the same estimator, three stop populations. `at_well` is the canonical cohort the content family is also fit on; the other two are sensitivity bundles and are never pooled with it. No likelihood-scale quantity is differenced across cohorts — the ΔAIC steps below are each taken WITHIN one cohort, between two rungs that share rows.

## What moved

Immediate reward component (`rr_immediate`, the 0 -> 1 contrast on bump 0) on the non-SWR non-local population (M7r_time_histcos_rew): `at_well` 2.28x (5/5 animals, 218 sessions); `all_stops` 1.23x (4/5 animals, 218 sessions); `in_transit` 1.76x (3/5 animals, 32 sessions).

The published functional on the same population and rung (`rr_lag1`, the reconstructed kernel at lag 1 — a different quantity, not a restatement): `at_well` 1.086x session-pooled / 1.089x animal-balanced (5/5 animals); `all_stops` 0.991x session-pooled / 1.022x animal-balanced (3/5 animals); `in_transit` 0.934x session-pooled / 0.979x animal-balanced (2/5 animals).

SWR-coupled minus non-SWR immediate coefficient, paired within session: `at_well` 69.4% of 216 sessions, 5/5 animals, p=1.1e-11; `all_stops` 79.2% of 216 sessions, 5/5 animals, p=7.8e-22; `in_transit` 40.0% of 10 sessions, 2/4 animals, p=9.2e-01.

Modelled non-local cohort: `at_well` 83,549 stops over 218 sessions; `all_stops` 106,058 stops over 218 sessions; `in_transit` 11,637 stops over 48 sessions.

## Cohorts

| stop_cohort | fit_supplied | rule |
| --- | --- | --- |
| at_well | yes | well-anchored stops (`occupancy_state` at_well or mixed) — canonical |
| all_stops | yes | every immobility stop — sensitivity bundle |
| in_transit | yes | in-transit stops only — sensitivity bundle |


What each artifact records about itself: `at_well`: rows say ['at_well']; `all_stops`: rows say ['all_stops']; `in_transit`: rows say ['in_transit'].

**The in-transit column is thin.** Only **50 of 218** sessions clear the per-bin minimum-sample gates on any population (swr 18, non_local 48, non_local_swr 10, non_local_no_swr 32), against 218 on the canonical cohort. An in-transit cell that shows no effect is an absence of evidence, not evidence of absence, and its paired tests have no power.

## Inputs

| role | path | sha256 |
| --- | --- | --- |
| per-bin (at_well) | /cumulus/edeno/continuum-swr-replay/Processed-Data/replay_content/models_per_bin_rate/all_models.pkl | 0dc2ce9f00fb |
| is_swr-contrast rate (at_well) | /cumulus/edeno/continuum-swr-replay/Processed-Data/replay_content/models_per_bin_rate/swr_contrast_rate.pkl | a3b5f647e5fb |
| per-bin (all_stops) | /cumulus/edeno/continuum-swr-replay/Processed-Data/replay_content/models_per_bin_rate/all_models_allstops.pkl | 141ba3f20c10 |
| is_swr-contrast rate (all_stops) | /cumulus/edeno/continuum-swr-replay/Processed-Data/replay_content/models_per_bin_rate/swr_contrast_rate_allstops.pkl | 396ed95852a4 |
| per-bin (in_transit) | /cumulus/edeno/continuum-swr-replay/Processed-Data/replay_content/models_per_bin_rate/all_models_intransit.pkl | 816c1ed11e03 |


### Provenance of the artifacts read

| artifact | role | git_hash | stop_cohort | written_at |
| --- | --- | --- | --- | --- |
| all_models.pkl | per-bin (at_well) | aee8b258ccc3067a89d4 | at_well | 2026-09-12T18:02:17 |
| swr_contrast_rate.pkl | is_swr-contrast rate (at_well) | aee8b258ccc3067a89d4 | at_well | 2026-09-12T17:40:55 |
| all_models_allstops.pkl | per-bin (all_stops) | 9e6808fc3e3af4854222 | all_stops | 2026-09-12T18:39:32 |
| swr_contrast_rate_allstops.pkl | is_swr-contrast rate (all_stops) | aee8b258ccc3067a89d4 | all_stops | 2026-09-12T18:09:31 |
| all_models_intransit.pkl | per-bin (in_transit) | aee8b258ccc3067a89d4 | in_transit | 2026-09-12T17:48:24 |
| swr_contrast_rate_audited.pkl | is_swr-contrast audited (at_well) | aee8b258ccc3067a89d4 | at_well | 2026-09-12T18:21:19 |
| swr_contrast_rate_audited_allstops.pkl | is_swr-contrast audited (all_stops) | aee8b258ccc3067a89d4 | all_stops | 2026-09-12T18:20:11 |


The hash is taken when the artifact is WRITTEN, not when its run started, so a long fit that spanned a commit records the later one — which is why the rows above do not all agree even though they describe one sitting. A `-dirty` suffix means the tree carried uncommitted edits at that moment; here those edits were to this audit script and to nothing the fit runners import. The audited is_swr-contrast bundles are the one place where the recorded hash PREDATES the code that wrote them: they were re-derived with `--aggregate-only` after the commit that gave the audited table its `stop_cohort` label and fixed the input name its freshness check resolves, and the hash records the tree that re-derivation ran from. The per-session fits beneath them, which carry every number this audit reads, are unaffected.

## Cohort size (per-bin ladder)

| outcome | stop_cohort | n_sessions | n_stops | n_bins | n_event_onsets |
| --- | --- | --- | --- | --- | --- |
| swr | at_well | 218 | 83549 | 6777953 | 29884 |
| swr | all_stops | 218 | 106058 | 7315746 | 31371 |
| swr | in_transit | 18 | 7620 | 187016 | 752 |
| non_local | at_well | 218 | 83549 | 6777953 | 46878 |
| non_local | all_stops | 218 | 106058 | 7315746 | 51907 |
| non_local | in_transit | 48 | 11637 | 293264 | 3861 |
| non_local_swr | at_well | 218 | 83549 | 6777953 | 20275 |
| non_local_swr | all_stops | 218 | 106058 | 7315746 | 20994 |
| non_local_swr | in_transit | 10 | 5363 | 134294 | 427 |
| non_local_no_swr | at_well | 218 | 83549 | 6777953 | 26603 |
| non_local_no_swr | all_stops | 218 | 106058 | 7315746 | 30913 |
| non_local_no_swr | in_transit | 32 | 8756 | 227001 | 2988 |


The `in_transit` rows above are a whole-corpus total over the few sessions that reach the ladder, not over 218. **The in-transit column is thin.** Only **50 of 218** sessions clear the per-bin minimum-sample gates on any population (swr 18, non_local 48, non_local_swr 10, non_local_no_swr 32), against 218 on the canonical cohort. An in-transit cell that shows no effect is an absence of evidence, not evidence of absence, and its paired tests have no power.

## Reward rate ratios, `M7r_time_histcos_rew`

Two different quantities, never to be quoted as one. **`rr_immediate`** is `exp` of the animal-balanced median `coef_reward_history_rcos_0` — the full 0 → 1 contrast on the first raised-cosine bump, i.e. one coefficient. **`rr_lag1_*`** is `exp` of the reconstructed reward kernel at lag 1, the blend `[0.325, 0.066, 0]` over the three bumps, which is the published functional `docs/rate_models.md` and claims row 19c quote; it is given session-pooled (`rr_lag1_session`, how the rate docs report it) and animal-balanced (`rr_lag1_animal`, how the audited aggregates report it). `animals_pos_*` counts animals whose own median is positive for each.

| outcome | stop_cohort | n_sessions | rr_immediate | animals_pos_immediate | rr_lag1_session | rr_lag1_animal | animals_pos_lag1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| swr | at_well | 218 | 4.912 | 5/5 | 1.228 | 1.300 | 4/5 |
| swr | all_stops | 218 | 4.147 | 5/5 | 1.224 | 1.241 | 5/5 |
| swr | in_transit | 17 | 0.952 | 2/4 | 0.913 | 0.924 | 1/4 |
| non_local | at_well | 217 | 2.955 | 5/5 | 1.214 | 1.200 | 5/5 |
| non_local | all_stops | 218 | 2.582 | 5/5 | 1.140 | 1.153 | 5/5 |
| non_local | in_transit | 48 | 2.677 | 4/5 | 1.064 | 1.091 | 3/5 |
| non_local_swr | at_well | 216 | 6.860 | 5/5 | 1.408 | 1.405 | 5/5 |
| non_local_swr | all_stops | 216 | 7.242 | 5/5 | 1.443 | 1.427 | 5/5 |
| non_local_swr | in_transit | 10 | 3.404 | 4/4 | 1.276 | 1.836 | 4/4 |
| non_local_no_swr | at_well | 218 | 2.280 | 5/5 | 1.086 | 1.089 | 5/5 |
| non_local_no_swr | all_stops | 218 | 1.230 | 4/5 | 0.991 | 1.022 | 3/5 |
| non_local_no_swr | in_transit | 32 | 1.764 | 3/5 | 0.934 | 0.979 | 2/5 |


## SWR-coupled vs non-SWR, paired within session

| stop_cohort | comparison | n_paired_sessions | pct_sessions_positive | animals_positive | signrank_p |
| --- | --- | --- | --- | --- | --- |
| at_well | non_local_swr − non_local_no_swr | 216 | 69.444 | 5/5 | 1.1e-11 |
| all_stops | non_local_swr − non_local_no_swr | 216 | 79.167 | 5/5 | 7.8e-22 |
| in_transit | non_local_swr − non_local_no_swr | 10 | 40.000 | 2/4 | 9.2e-01 |


## Nested reward steps, within cohort (negative = child better)

| step | outcome | stop_cohort | n_sessions | median_delta_aic | pct_sessions_child_better |
| --- | --- | --- | --- | --- | --- |
| M7r_time_histcos_rew − M6r_time_histcos | swr | at_well | 218 | 1.58 | 38.07 |
| M7r_time_histcos_rew − M6r_time_histcos | swr | all_stops | 218 | 1.64 | 38.07 |
| M7r_time_histcos_rew − M6r_time_histcos | swr | in_transit | 17 | 1.44 | 35.29 |
| M7r_time_histcos_rew − M6r_time_histcos | non_local | at_well | 217 | 1.19 | 43.32 |
| M7r_time_histcos_rew − M6r_time_histcos | non_local | all_stops | 218 | 1.09 | 41.74 |
| M7r_time_histcos_rew − M6r_time_histcos | non_local | in_transit | 47 | 0.69 | 42.55 |
| M7r_time_histcos_rew − M6r_time_histcos | non_local_swr | at_well | 215 | 2.41 | 33.95 |
| M7r_time_histcos_rew − M6r_time_histcos | non_local_swr | all_stops | 215 | 2.17 | 34.88 |
| M7r_time_histcos_rew − M6r_time_histcos | non_local_swr | in_transit | 10 | -1.14 | 50.00 |
| M7r_time_histcos_rew − M6r_time_histcos | non_local_no_swr | at_well | 218 | 2.28 | 33.49 |
| M7r_time_histcos_rew − M6r_time_histcos | non_local_no_swr | all_stops | 218 | 2.38 | 33.49 |
| M7r_time_histcos_rew − M6r_time_histcos | non_local_no_swr | in_transit | 32 | -0.32 | 53.12 |
| M1_rew − M0_const | swr | at_well | 218 | -3.23 | 65.60 |
| M1_rew − M0_const | swr | all_stops | 218 | -3.01 | 66.97 |
| M1_rew − M0_const | swr | in_transit | 18 | 1.51 | 38.89 |
| M1_rew − M0_const | non_local | at_well | 218 | -8.18 | 85.32 |
| M1_rew − M0_const | non_local | all_stops | 218 | -7.84 | 80.28 |
| M1_rew − M0_const | non_local | in_transit | 48 | 0.52 | 45.83 |
| M1_rew − M0_const | non_local_swr | at_well | 218 | -3.96 | 74.31 |
| M1_rew − M0_const | non_local_swr | all_stops | 218 | -5.42 | 82.57 |
| M1_rew − M0_const | non_local_swr | in_transit | 10 | -3.32 | 50.00 |
| M1_rew − M0_const | non_local_no_swr | at_well | 218 | -0.75 | 58.26 |
| M1_rew − M0_const | non_local_no_swr | all_stops | 218 | 0.14 | 48.17 |
| M1_rew − M0_const | non_local_no_swr | in_transit | 32 | 0.02 | 50.00 |


## Burst peak of the smooth time course (`M5_time`)

| outcome | stop_cohort | n_sessions | support_s | burst_peak_s | peak_frac_support |
| --- | --- | --- | --- | --- | --- |
| swr | at_well | 209 | 1.0–3.0 | 2.630 | 0.815 |
| swr | all_stops | 212 | 1.0–3.0 | 2.385 | 0.692 |
| swr | in_transit | 14 | 1.0–3.0 | 2.965 | 0.982 |
| non_local | at_well | 217 | 1.0–3.0 | 2.815 | 0.907 |
| non_local | all_stops | 217 | 1.0–3.0 | 2.790 | 0.895 |
| non_local | in_transit | 41 | 1.0–3.0 | 1.465 | 0.233 |
| non_local_swr | at_well | 168 | 1.0–3.0 | 2.305 | 0.652 |
| non_local_swr | all_stops | 179 | 1.0–3.0 | 2.230 | 0.615 |
| non_local_swr | in_transit | 9 | 1.0–3.0 | 2.865 | 0.933 |
| non_local_no_swr | at_well | 211 | 1.0–3.0 | 2.680 | 0.840 |
| non_local_no_swr | all_stops | 217 | 1.0–3.0 | 3.000* | 1.000 |
| non_local_no_swr | in_transit | 29 | 1.0–3.0 | 1.700 | 0.350 |


`*` marks a peak sitting on the edge of the spline's supported interior (`peak_frac_support` 0 or 1): the median curve is still rising or falling where the summary stops reading it, so the value bounds the burst rather than locating it.

## Arrival-window retention

| outcome | stop_cohort | n_event_bins_frozen | n_event_bins_arrival_window | retained_share_arrival_events |
| --- | --- | --- | --- | --- |
| swr | at_well | 29884 | 28170 | 0.943 |
| swr | all_stops | 31371 | 28612 | 0.912 |
| swr | in_transit | 1487 | 442 | 0.297 |
| non_local | at_well | 46878 | 42482 | 0.906 |
| non_local | all_stops | 51907 | 44096 | 0.850 |
| non_local | in_transit | 5029 | 1614 | 0.321 |
| non_local_swr | at_well | 20275 | 18827 | 0.929 |
| non_local_swr | all_stops | 20994 | 18946 | 0.902 |
| non_local_swr | in_transit | 719 | 119 | 0.166 |
| non_local_no_swr | at_well | 26603 | 23655 | 0.889 |
| non_local_no_swr | all_stops | 30913 | 25150 | 0.814 |
| non_local_no_swr | in_transit | 4310 | 1495 | 0.347 |


## is_swr-contrast rate bundle, per rung and population

Primary fitting cohort only, so `G5_rpe`, `G7_update`, `S7_rpe`, `S9_update` are absent: they need a post-outcome covariate and are fit on the gated cohort with their own null, which exists for one AIC comparison and is not a second copy of the ladder.

`rr_immediate` is not the same statement on every rung. On a `G` rung it is that population's immediate reward slope, fit on that population's own rows. On an `S` rung that splits the reward block by `is_swr` it is the **non-SWR reference-level** slope, and `rr_swr_x_immediate` (`exp` of the animal-balanced `coef_is_swr:reward_history_rcos_0`) is the multiplier that turns it into the SWR one — so those two cells are a contrast, not two estimates of one thing. `rr_swr_x_immediate` is blank wherever the rung carries no such interaction: on every `G` rung, and on the ladder null `S0_rate`, where `is_swr` enters only as an intercept shift and the reward slope is shared by both populations. On `G8_rewxtime`, `S10_rewxtime` it is blank: the reward block is interacted with the time spline there, so the main effect is the slope at the point where every spline column is zero — off the fitted window, a boundary extrapolation rather than a reward contrast.

| model | population | stop_cohort | n_sessions | n_stops | n_bins | rr_immediate | animals_positive | rr_swr_x_immediate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0_base | non_local | at_well | 218 | 83176 | 6763387 | 2.826 | 5/5 | — |
| G0_base | non_local | all_stops | 217 | 103651 | 7203966 | 2.688 | 5/5 | — |
| G0_base | non_local_no_swr | at_well | 218 | 83176 | 6763387 | 1.973 | 5/5 | — |
| G0_base | non_local_no_swr | all_stops | 218 | 104018 | 7234793 | 1.188 | 5/5 | — |
| G0_base | non_local_swr | at_well | 216 | 82475 | 6709348 | 6.814 | 5/5 | — |
| G0_base | non_local_swr | all_stops | 216 | 103132 | 7176704 | 7.082 | 5/5 | — |
| G1_periswitch | non_local | at_well | 218 | 83176 | 6763387 | 3.177 | 5/5 | — |
| G1_periswitch | non_local | all_stops | 218 | 104018 | 7234793 | 2.808 | 5/5 | — |
| G1_periswitch | non_local_no_swr | at_well | 218 | 83176 | 6763387 | 2.850 | 5/5 | — |
| G1_periswitch | non_local_no_swr | all_stops | 218 | 104018 | 7234793 | 1.237 | 5/5 | — |
| G1_periswitch | non_local_swr | at_well | 217 | 82808 | 6735139 | 6.346 | 5/5 | — |
| G1_periswitch | non_local_swr | all_stops | 217 | 103527 | 7204455 | 8.076 | 5/5 | — |
| G2_block | non_local | at_well | 218 | 83176 | 6763387 | 3.325 | 5/5 | — |
| G2_block | non_local | all_stops | 218 | 104018 | 7234793 | 2.622 | 5/5 | — |
| G2_block | non_local_no_swr | at_well | 218 | 83176 | 6763387 | 2.633 | 5/5 | — |
| G2_block | non_local_no_swr | all_stops | 218 | 104018 | 7234793 | 1.352 | 3/5 | — |
| G2_block | non_local_swr | at_well | 216 | 82475 | 6709348 | 6.342 | 5/5 | — |
| G2_block | non_local_swr | all_stops | 216 | 103132 | 7176704 | 6.292 | 5/5 | — |
| G3_switchrate | non_local | at_well | 218 | 83176 | 6763387 | 3.265 | 5/5 | — |
| G3_switchrate | non_local | all_stops | 218 | 104018 | 7234793 | 2.707 | 5/5 | — |
| G3_switchrate | non_local_no_swr | at_well | 218 | 83176 | 6763387 | 2.321 | 5/5 | — |
| G3_switchrate | non_local_no_swr | all_stops | 218 | 104018 | 7234793 | 1.333 | 5/5 | — |
| G3_switchrate | non_local_swr | at_well | 216 | 82475 | 6709348 | 7.262 | 5/5 | — |
| G3_switchrate | non_local_swr | all_stops | 216 | 103132 | 7176704 | 6.479 | 5/5 | — |
| G4_value | non_local | at_well | 218 | 83176 | 6763387 | 3.378 | 5/5 | — |
| G4_value | non_local | all_stops | 218 | 104018 | 7234793 | 2.831 | 5/5 | — |
| G4_value | non_local_no_swr | at_well | 218 | 83176 | 6763387 | 1.929 | 5/5 | — |
| G4_value | non_local_no_swr | all_stops | 218 | 104018 | 7234793 | 1.245 | 5/5 | — |
| G4_value | non_local_swr | at_well | 216 | 82328 | 6700887 | 8.451 | 5/5 | — |
| G4_value | non_local_swr | all_stops | 217 | 103527 | 7204455 | 6.835 | 5/5 | — |
| G6_uncert | non_local | at_well | 218 | 83176 | 6763387 | 2.732 | 5/5 | — |
| G6_uncert | non_local | all_stops | 218 | 104018 | 7234793 | 2.396 | 5/5 | — |
| G6_uncert | non_local_no_swr | at_well | 218 | 83176 | 6763387 | 2.162 | 5/5 | — |
| G6_uncert | non_local_no_swr | all_stops | 218 | 104018 | 7234793 | 1.353 | 5/5 | — |
| G6_uncert | non_local_swr | at_well | 216 | 82475 | 6709348 | 6.791 | 5/5 | — |
| G6_uncert | non_local_swr | all_stops | 216 | 103132 | 7176704 | 6.329 | 5/5 | — |
| G8_rewxtime | non_local | at_well | 218 | 83176 | 6763387 | — | — | — |
| G8_rewxtime | non_local | all_stops | 218 | 104018 | 7234793 | — | — | — |
| G8_rewxtime | non_local_no_swr | at_well | 218 | 83176 | 6763387 | — | — | — |
| G8_rewxtime | non_local_no_swr | all_stops | 218 | 104018 | 7234793 | — | — | — |
| G8_rewxtime | non_local_swr | at_well | 218 | 83176 | 6763387 | — | — | — |
| G8_rewxtime | non_local_swr | all_stops | 218 | 104018 | 7234793 | — | — | — |
| S0_rate | contrast | at_well | 218 | 83176 | 13526774 | 2.883 | 5/5 | — |
| S0_rate | contrast | all_stops | 218 | 104018 | 14469586 | 2.565 | 5/5 | — |
| S1_rew | contrast | at_well | 218 | 83176 | 13526774 | 2.527 | 5/5 | 1.502 |
| S1_rew | contrast | all_stops | 218 | 104018 | 14469586 | 2.053 | 5/5 | 2.086 |
| S2_base | contrast | at_well | 213 | 81262 | 13235878 | 2.320 | 5/5 | 3.458 |
| S2_base | contrast | all_stops | 212 | 101646 | 14123520 | 1.120 | 5/5 | 6.392 |
| S3_periswitch | contrast | at_well | 208 | 80206 | 12969228 | 2.613 | 5/5 | 2.826 |
| S3_periswitch | contrast | all_stops | 212 | 102040 | 14133430 | 1.236 | 5/5 | 4.349 |
| S4_block | contrast | at_well | 212 | 81119 | 13184794 | 2.730 | 5/5 | 2.042 |
| S4_block | contrast | all_stops | 212 | 101381 | 14133500 | 1.278 | 3/5 | 4.627 |
| S5_switchrate | contrast | at_well | 211 | 80841 | 13148190 | 2.369 | 5/5 | 2.994 |
| S5_switchrate | contrast | all_stops | 211 | 101488 | 14097946 | 1.329 | 5/5 | 5.019 |
| S6_value | contrast | at_well | 212 | 81324 | 13214596 | 2.046 | 5/5 | 2.574 |
| S6_value | contrast | all_stops | 215 | 102979 | 14299558 | 1.245 | 5/5 | 7.333 |
| S8_uncert | contrast | at_well | 215 | 82202 | 13360376 | 2.288 | 5/5 | 2.378 |
| S8_uncert | contrast | all_stops | 216 | 103388 | 14352264 | 1.353 | 4/5 | 6.223 |
| S10_rewxtime | contrast | at_well | 214 | 81920 | 13310338 | — | — | — |
| S10_rewxtime | contrast | all_stops | 215 | 102985 | 14299984 | — | — | — |

