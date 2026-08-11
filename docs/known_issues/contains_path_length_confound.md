# `contains_path_to_next` is confounded with route length and event length

**Status:** confirmed confound, measured, latent (no current consumer)
**Found:** 2026-08-10, following the replay-tables review
**Severity:** would be severe if the raw rate were compared across trip types;
harmless today because nothing reads the column

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

Within-patch 12.0 % versus cross-patch 4.3 % — a **2.8× difference produced by
the criterion alone**.

It also interacts strongly with event length. Contains rate by quartile of
`n_nonlocal_bins`:

| event-length quartile | 1 arm | 2 arms | 4 arms |
| --- | --- | --- | --- |
| Q1 (shortest) | 36.2 % | 2.3 % | 0.4 % |
| Q2 | 42.8 % | 5.9 % | 2.6 % |
| Q3 | 40.9 % | 11.9 % | 3.2 % |
| Q4 (longest) | 46.9 % | 21.6 % | 11.7 % |

A 9.4× swing across event length for 2-arm paths and **29× for 4-arm paths**,
versus only 1.3× when a single arm is required. The metric is close to its floor
for most events: the **median event covers 1 unique arm** (mean 2.16, p90 5),
while the criterion demands 2 or 4.

## Why it matters

Any comparison of raw `contains_path_to_next` between switch trials (cross-patch,
4 arms) and stay trials (within-patch, 2 arms) is largely a geometry artifact.
The same applies to comparisons across conditions that differ in event duration —
SWR versus non-SWR events, for instance, differ in length, so they would differ
in contains rate with identical content.

`contains_path_from_prev` has the same structure and the same exposure.

## Current exposure

**None.** The column is exported from `analysis.replay` and listed as
response-only in `covariate_timing.py`, but no script, model spec, or figure
reads it. Nothing published depends on it. This is recorded so it stays that way,
or is used correctly if it is picked up.

## If you use it

- Stratify by required-path length, or restrict to one trip type. Never pool.
- Condition on event length (`n_nonlocal_bins`) as well.
- Prefer a graded alternative to the all-or-nothing rule — the fraction of
  required arms covered — which is far less floor-bound and much weaker in its
  dependence on route length.
- `ends_at_next` does not share the confound: it tests only the final MAP arm, so
  the criterion is one arm regardless of trip type.

## Reproducing

Reads the shipped `replay_content/event_table.pkl` only; no Spyglass access and
no rebuild. Recompute the required-arm count per event with
`arm_path_well_to_well(current_arm, next_arm)` and cross-tabulate against
`contains_path_to_next` and quartiles of `n_nonlocal_bins`.
