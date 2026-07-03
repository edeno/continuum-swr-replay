# Fig 2b robustness: is non-SWR non-local dominance driven by certain days/animals?

**Date:** 2026-06-17
**Owner:** analysis (with Claude)
**Status:** COMPLETE — verdict: **H2 (broadly robust) confirmed; the fear (H1) is refuted.**

## The claim under scrutiny

Figure 2b is a stacked bar of **immobile** non-local events split into three
SWR-overlap classes: `swr_contained`, `swr_partial`, `non_swr`. The headline
read is that **non-SWR non-local events dominate** (the largest slice).

**Fear:** the aggregate plurality of `non_swr` is an artifact of a minority of
animals or sessions — i.e. a few days/animals contribute most of the non-SWR
events, and the rest of the dataset does not show the pattern.

## Data / provenance (verified)

- Panel b source: `build_classified_nonlocal_event_table(data_over_time)` →
  filter `endpoints_speed_below_thresh` → tally `overlap_class`
  (`compute_swr_overlap_class_counts`, immobile-only).
  - `event_table.py:206`, `data_aggregation.py:540`.
- `data_over_time` = concat of 218 `*_HPC_80_data_over_time.pkl` files
  (NonLocal 4-state, 0.80 threshold). Animals: chimi(45), j16(58),
  peanut(13), senor(58), wilbur(44). One row per timebin; carries
  `animal/day/epoch`.
- Each classified row = one non-local event; `event_id` is a stable hash of
  `(animal, day, epoch, non_local_event)` → globally unique. So per-animal /
  per-session decomposition is well-defined.
- `overlap_class`: strict positive-duration overlap with SWR_event timebins.
  `non_swr` = no positive-duration overlap.

## Operational definition of "dominates"

I will track several, because "dominates" is ambiguous:
- **D1 (plurality):** `non_swr` is the single largest class.
- **D2 (majority):** `non_swr` > 50% of immobile events.
- **D3 (vs SWR-any):** `non_swr` count vs (`swr_contained` + `swr_partial`).

The non-SWR *fraction* `f_nonswr = non_swr / total_immobile` is the core
per-unit statistic, computed at population, per-animal, and per-session level.

## Competing hypotheses

- **H1 — Concentrated artifact (the fear).** Non-SWR dominance is carried by a
  minority of animals or sessions; remove them and the plurality flips.
  *Prediction:* `f_nonswr` is high in 1–2 animals, low in others; a large share
  of total non_swr events comes from a few sessions; leave-one-animal-out
  flips D1 for at least one animal removed.
  *Prior confidence: 20%.*

- **H2 — Broadly robust.** Non-SWR plurality holds within most/all animals and
  the majority of individual sessions. Aggregate reflects a per-unit pattern.
  *Prediction:* every animal shows D1; median session `f_nonswr` is well above
  the next class; leave-one-animal-out never flips D1.
  *Prior confidence: 55%.*

- **H3 — Count-imbalance / mixing.** Some sessions contribute far more events
  and happen to be non-SWR heavy, so the *pooled* count over-weights them even
  if the typical session is balanced (Simpson-flavored).
  *Prediction:* event counts per session are heavy-tailed; `f_nonswr`
  correlates with session event count; session-mean `f_nonswr` (unweighted) is
  meaningfully lower than event-weighted pooled `f_nonswr`.
  *Prior confidence: 20%.*

- **H4 — Threshold/detection covariate (mechanistic, secondary).** `f_nonswr`
  tracks something measurable (e.g. ripple rate, number of SWRs, immobile time,
  peak-prob distribution) rather than animal identity per se.
  *Prior confidence: 5% as the primary driver; worth a look as explanation.*

## Analysis plan (structured)

1. **Reproduce** the pooled panel-b counts exactly (sanity vs the figure).
2. **Per-animal** breakdown: counts + `f_nonswr` + D1/D2/D3 within each animal.
3. **Per-session** distribution of `f_nonswr` (weighted vs unweighted means;
   fraction of sessions satisfying D1/D2).
4. **Concentration:** share of total non_swr from top animal / top-k sessions;
   Gini over sessions.
5. **Leave-one-out:** drop each animal in turn; does D1 survive?
6. **H3 check:** correlate session `f_nonswr` with event count.
7. Spot-check a couple of extreme sessions to make sure counts aren't a bug
   (e.g. degenerate 23s-MUA event flagged earlier).

## Findings log

Script: `scripts/_fig2b_robustness_decompose.py`; outputs in
`.claude/research/fig2b_outputs/` (per_animal.csv, per_session.csv, summary.txt).
Run 2026-06-17 over 218 sessions / 5 animals / 65,913 immobile non-local events
(event_id duplicates = 0, so no cross-session collision).

**[0] Pooled (reproduces panel b):** contained 12,964 / partial 13,449 /
non_swr 39,500 → **f_nonswr = 0.599**. non_swr is both plurality AND majority.

**[1] Per-animal — non-SWR wins in ALL 5 animals:**
| animal | n | sessions | f_nonswr | plurality | majority |
|--------|---|----------|----------|-----------|----------|
| chimi  | 22,400 | 45 | 0.593 | ✔ | ✔ |
| j16    | 14,211 | 58 | 0.608 | ✔ | ✔ |
| peanut |  4,251 | 13 | 0.679 | ✔ | ✔ |
| senor  | 17,890 | 58 | 0.593 | ✔ | ✔ |
| wilbur |  7,161 | 44 | 0.571 | ✔ | ✔ |

f_nonswr range a tight [0.571, 0.679] — no animal is an outlier carrying it.

**[2] Per-session:** non_swr is the plurality in **95.9% (209/218)** of sessions
and the majority in 76.1%. Median session f_nonswr = 0.584, IQR [0.505, 0.648].
Unweighted session-mean f_nonswr = 0.578 vs event-weighted pooled 0.599 — only a
2.1-pt gap, so the pooled bar is not an artifact of a few huge sessions (H3 weak).

**[3] Concentration:** top animal (chimi) holds 33.6% of non_swr events but also
holds the most sessions/events; no single session holds >4.1%; top-10 sessions
hold 24.3%. Gini of non_swr across sessions = 0.383 ≈ the Gini of total
events/session (0.337) → non_swr is spread the same way events are, not clumped.

**[4] Leave-one-animal-out:** pooled non_swr plurality survives every drop;
f_nonswr stays in [0.594, 0.603]. Removing any single animal moves it <1 pt.

**[5] H3 directionality:** corr(session f_nonswr, log event count) = +0.268;
f_nonswr by size tercile small/mid/large = 0.554 / 0.603 / 0.620. Larger sessions
are *modestly* more non-SWR heavy (explains the 2-pt weighted gap), but even the
smallest tercile is non-SWR plurality. So count-imbalance is a minor secondary
effect, not the driver.

**Exception sessions (9/218 where non_swr is not plurality):** all are *near-ties*
where non_swr is the 2nd class (never collapsed) and SWR-partial or
SWR-contained edges it out by a few events. They cluster within single days:
all 5 j16 exceptions are day 20210706; chimi's 4 are spread over 3 late days.
This is day-level variation in SWR overlap, not a sign the pooled pattern is
spurious.

### Verdict (calibrated)
- **H1 (concentrated artifact — the fear): REFUTED. Confidence 95%.** Holds in
  all 5 animals, 96% of sessions, survives leave-one-animal-out, no dominant
  session/animal.
- **H2 (broadly robust): CONFIRMED. Confidence 90%.** The non-SWR plurality is a
  per-animal, per-session property, not a pooling artifact.
- **H3 (count-imbalance): minor real effect, ~2 pts. Confidence 70%** it is
  present but secondary; does not threaten the claim.
- **H4 (detection covariate): not the primary driver** (animal identity explains
  almost none of the spread); session-size is the only covariate probed and it
  is weak. Untested: ripple rate / immobile-time per session — a possible
  follow-up if a reviewer pushes on *why* f_nonswr varies session-to-session.

## Follow-up: does non-local amount vary by date per animal? (SWR-over-learning)

Scripts: `scripts/_fig2b_nonlocal_over_dates.py` (+ `_fig2b_dates_plot.py`).
Outputs: `per_session_dated.csv`, `per_animal_date_trends.csv`,
`date_trends_summary.txt`, `dates_trends.png`. Rates use **immobile minutes** as
the denominator (events/SWRs are immobility phenomena) so session length / the
shrinking immobile time over days does not inflate counts. Trend = Spearman of
the metric vs ordinal date within each animal.

**Premise check — SWR rate decreases over learning:** directionally TRUE in
**5/5 animals** (Spearman rho all negative, range [-0.57, -0.04]); significant in
2/5 (j16 -0.48, wilbur -0.57). Modest but consistent. Confidence 70% the decline
is real; it is weak in chimi/peanut/senor.

**Does non-local AMOUNT vary by date?** YES, strongly in some animals, and the
direction is **animal-specific, not universal**:
- chimi: non-local rate **collapses** 62.6 → 18.1 /immob-min (rho -0.91*), event
  count 1898 → 167; immobile time also drops 30 → 9 min (rho -0.64*). chimi's
  early days are a major event reservoir (it alone = 33.6% of all non_swr events).
- j16: non-local rate **rises** 22.5 → 26.1 (rho +0.56*).
- peanut / senor / wilbur: weak decline, not significant for rate.

So "non-local amount varies by date" is true but driven largely by chimi's early
high-activity sessions; there is no single population-wide trend in the rate.

**Non-SWR FRACTION (the Fig 2b quantity) vs date:** rises over learning in
**4/5 animals** (j16 +0.65*, wilbur +0.69*, senor +0.29*, peanut +0.13 ns),
with **chimi the lone exception (-0.52*, falls to 0.44)**. Directionally this is
what the premise predicts: as SWRs wane, a larger share of non-local activity is
non-SWR. Confidence 65% this is a learning effect rather than recording drift —
chimi's opposite sign keeps it from being clean.

**Internal consistency — explains the 9 Fig 2b exceptions:** the only sessions
where non_swr was NOT the plurality were j16's **first day** (low f_nonswr 0.40)
and chimi's **late days** (f_nonswr falls to 0.44). Those are exactly the tails
of these within-animal date trends, not random noise.

**Caveats:** immobile time per session shrinks over days in most animals
(animals settle into the task) — controlled for by rating per immobile-minute,
but the raw counts would mislead. chimi day-1 (20200213, 1898 events, 30 immob
min) is an outlier session and drives the "rate decreases" tally; the effect is
not uniform. Untested driver of session-to-session f_nonswr spread: SWR
detection threshold stability across days.

## Follow-up: are the LONG non-SWR events concentrated in certain animals?

Script: `scripts/_fig2b_nonswr_durations.py`. Outputs:
`nonswr_durations_per_animal.csv`, `nonswr_longest_events.csv`,
`nonswr_durations.png`. On the 39,500 immobile non-SWR events.

- **Non-SWR events are SHORT.** median 50 ms, p90 142 ms, p99 428 ms, **max 2.69 s**.
  (SWR-overlapping events are if anything slightly *longer* in the body —
  median 88 ms — because SWRs cap them; the non-SWR set just has a fatter rare
  tail.) The earlier-flagged 23 s degenerate event was an **MUA** event, not in
  this immobile non-local set — nothing here exceeds 2.7 s.
- **The long tail is tiny:** only 34 events (0.09%) exceed 1 s; 6 exceed 2 s.
- **j16 disproportionately carries the long tail.** j16 is 22% of all non-SWR
  events but **47% of the >1 s events and 67% (4/6) of the >2 s events**.
  chimi is the opposite — 34% of non-SWR events but only 12% of >1 s and its
  events are the most short-peaked. peanut barely contributes any tail.
  - >0.5 s (272 ev): j16 37% / chimi 25% / senor 25% / wilbur 10% / peanut 3%
  - >1 s (34 ev): j16 47% / senor 21% / wilbur 18% / chimi 12% / peanut 3%
  - >2 s (6 ev): j16 67% / chimi 33%
- The longest events look **real, not artifactual**: peak_nonlocal_prob ≈ 1.0,
  max_speed < 4 cm/s (genuinely immobile), n_swrs_overlapping = 0. j16's longest
  cluster on day 20210709 epoch 10_r5 (4 of the top 11).

**Verdict:** YES — j16 (and to a lesser extent senor/wilbur) carries the long
non-SWR tail; chimi has many but short ones. But the tail is a negligible
fraction (<0.1% > 1 s) of the 39.5k events, so it does **not** drive the Fig 2b
bar — that bar is built from the ~50 ms median events, which every animal has.
Confidence 90%.

## Self-critique / watch-list

- Pooled counts weight by event; a robust population claim should also hold
  unweighted across sessions. Report both.
- `f_nonswr` for tiny sessions (few events) is noisy — weight or annotate n.
- Don't conflate "every animal shows plurality" with "every session" — report
  the session-level fraction too.
- peanut has only 13 sessions — watch its leverage.
- Confirm event_id uniqueness holds (no cross-session collisions inflating one
  class).
