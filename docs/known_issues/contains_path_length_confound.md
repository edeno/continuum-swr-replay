# `contains_path_to_next` is confounded with route length and event length

**Status:** fixed in code; Stage 5B outputs must be regenerated
**Found:** 2026-08-10, following the replay-tables review
**Severity:** severe for raw pooled rates and for the previous unconditioned null

## What

`contains_path_to_next` is all-or-nothing: every arm in
`arm_path_well_to_well(current_arm, next_arm)` must appear in the event's
MAP-arm trace ([`retrospective.py`](../../src/continuum_swr_replay/analysis/replay/retrospective.py),
`_contains_path`). On the spider track the required set has a different size
depending on the trip:

| trip | arms the replay must cover |
| --- | --- |
| same well | 1 (`{i}`) |
| within-patch | 2 (`{i, j}`) |
| cross-patch | 4 (`{i, j, trunk_i, trunk_j}`) |

So the criterion is twice as hard for a switch as for a stay, before any
neuroscience enters. The metric therefore measures route geometry at least as
much as it measures replay content.

## How large (65,913 events; 63,305 with a defined value)

| required arms | events | share | contains rate |
| --- | --- | --- | --- |
| 1 | 3,216 | 5.1 % | **41.6 %** |
| 2 (within-patch) | 55,251 | 87.3 % | **10.3 %** |
| 4 (cross-patch) | 4,838 | 7.6 % | **4.3 %** |

Pooling the 1- and 2-arm rows gives 12.0 % versus 4.3 % cross-patch — a **2.76×
difference produced by the criterion alone**. Comparing the table's own
within-patch row (2 arms, 10.3 %) against cross-patch gives **2.4×**. Either
way the gap is geometry, not content.

It also interacts strongly with event length. Contains rate by **global**
quartile of `n_nonlocal_bins` (quartiles taken over all events, not within
each path length — computing them within-length shifts a couple of cells by
under a point):

| event-length quartile | 1 arm | 2 arms | 4 arms |
| --- | --- | --- | --- |
| Q1 (shortest) | 36.2 % | 2.3 % | 0.4 % |
| Q2 | 42.8 % | 5.9 % | 2.6 % |
| Q3 | 40.9 % | 11.9 % | 3.2 % |
| Q4 (longest) | 46.9 % | 21.6 % | 11.7 % |

A 9.3× swing across event length for 2-arm paths and **30.7× for 4-arm paths**,
versus only 1.3× when a single arm is required. The metric is close to its floor
for most events: the **median event covers 1 unique arm** (mean 2.16, p90 5),
while the criterion demands 2 or 4.

## Why it matters

Any comparison of raw `contains_path_to_next` between switch trials (cross-patch,
4 arms) and stay trials (within-patch, 2 arms) is largely a geometry artifact.
The same applies to comparisons across conditions that differ in event duration —
SWR versus non-SWR events, for instance, differ in length, so they would differ
in contains rate with identical content.

`contains_path_from_prev` has the same structure (40.3 % / 10.2 % / 3.1 % by
1 / 2 / 4 required arms) but *less* exposure: it is not in
`retrospective._DEFAULT_METRICS`, so the consumer below does not report it.

## Current exposure

**One consumer, and it ships outputs.** `scripts/behavior/run_retrospective_alignment.py`
computes `contains_path_to_next` for every session — it is one of
`retrospective._DEFAULT_METRICS` — and its results are on disk as
`replay_content/stage5b_retrospective{,_swr_only,_non_swr_only}.pkl`, 218 rows
each, carrying `real_mean_`, `shuffled_mean_` and
`real_minus_shuffled_contains_path_to_next` (real mean 0.116, real − shuffled
−0.028).

The shipped files predate the calibrated null and must not be interpreted as
corrected results. The earlier event-row circular shift changed the association
between replay content and current location, route length, and event duration.
On this track those variables strongly constrain both the future label and the
probability of satisfying an all-or-nothing path criterion, so preserving only
the session-wide marginal distribution was insufficient.

The revised Stage 5B null time-orders the at-well events in each `stop_id` and
uses each contiguous run of one `current_arm` as its permutation unit. This is
visit context rather than immobility-bout context: a long bout can cross a trial
boundary or return to an earlier arm, and those runs may have different future
labels. Every event in one contiguous visit context receives the same shuffled
future label.

The three comparison families are calibrated separately rather than forcing
every metric onto the path metric's restrictive cohort:

- arm fraction and arm endpoint condition on `current_arm` and permute
  `next_arm`;
- patch fraction and patch endpoint condition on `current_patch` and permute
  `next_patch` (with its paired `next_arm` label);
- path coverage conditions on `current_arm`, exact required-route length
  (1, 2, or 4 arms), and an octave bin of the visit's median
  `n_nonlocal_bins`, then permutes `next_arm`.

Labels move only within strata with at least two visit contexts and variation
in the relevant future target. Singleton and deterministic strata are excluded
rather than returned as degenerate zero-difference evidence. Output columns
report a distinct denominator and retained/excluded event, visit, and stratum
counts for each family.

On this topology, fixing both `current_arm` and exact route length makes the
same-well and within-patch path targets deterministic. The path null is thus
identifiable only in movable strata—effectively cross-patch routes in the
canonical task. This restriction applies only to `contains_path_to_next`; the
arm and patch families retain their broader independently calibrated cohorts.

The metric itself is still set membership and therefore still has a floor
effect. The new null calibrates the reported real-minus-shuffled contrast; it
does not make the raw rate comparable across route or duration classes.

No model spec or figure reads the `event_table` column itself.

## If you use it

- Stratify by required-path length, or restrict to one trip type. Never pool a
  raw rate across trip types; the calibrated Stage 5B shuffle is the minimum
  bar for a null contrast.
- Condition on event length (`n_nonlocal_bins`) as well.
- Prefer a graded alternative to the all-or-nothing rule — the fraction of
  required arms covered — which is far less floor-bound and much weaker in its
  dependence on route length.
- `ends_at_next` does not share the confound: it tests only the final MAP arm, so
  the criterion is one arm regardless of trip type.
- Inspect the family-specific Stage 5B exclusion and denominator columns. For
  path coverage, exact current-location and route conditioning makes some route
  classes deterministic and scientifically non-identifiable under label
  permutation; that does not remove them from the arm or patch families.

## Reproducing

Reads the shipped `replay_content/event_table.parquet` only; no Spyglass access and
no rebuild. Recompute the required-arm count per event with
`arm_path_well_to_well(current_arm, next_arm)` and cross-tabulate against
`contains_path_to_next` and **global** quartiles of `n_nonlocal_bins` (taken
over all events, then crossed with path length — not quartiles computed
within each path length).
