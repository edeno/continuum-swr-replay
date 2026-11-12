# SWR tetrode selection (rule R-T), and why there is no event veto

**Status (2026-10-09):** the project lead chose rule R-T on 2026-10-09. Implemented in
`src/continuum_swr_replay/analysis/hippocampus/ripple_tetrodes.py` and applied by
`scripts/detection/run_swr_detection.py` (`detection_channels`) once per epoch, before any of
the four detectors (Task 17b of the full pipeline refresh,
[2026-10-03-full-pipeline-refresh-design.md](2026-10-03-full-pipeline-refresh-design.md)).
**Not yet applied:** the configured SWR inventory (`swr_source.SWR_INVENTORY_RUN_ID`) predates
the rule; the next SWR runner run applies it. The reference veto R-rip was studied and is not
used. Companion note: [2026-10-09-artifact-detection.md](2026-10-09-artifact-detection.md) (the
artifact mask).

**Where the numbers come from.** Unless a number names a later run, it is a dated
measurement of 2026-10-09 or earlier, made by the studies below; no summary in
`paper/reported/` computes it, and nothing in the repository recomputes it. The studies'
working files were scratch analyses and are not in the repository: this note is their
record, and it names them by these labels.

- **C1:** the events at the ports, and the proxies we have for licking;
- **C1b:** the field's noise-channel veto and R-rip, and the j16 residual;
- **C1c:** the j16 tetrodes, and rule R-T with its effect (R-T was written down before any
  effect was measured);
- **C2:** R-T held out, its sensitivity, and the field precedents.

## Names

Neither validation set is ground truth.

- **The reference SWR set** is rule-defined: 9,208 events that meet Kay ∩ Karlsson, a Kay
  duration of 30–200 ms, speed < 4 cm/s over ± 250 ms, a sharp wave, and MUA ≥ 2× the epoch
  mean.
- **Model-reviewed labels** come from one AI reviewer, with no human and no second rater
  (55 SWRs in the pooled review; 36 events in C1c's blind spot-check).
- **"Lacks SWR features"** (C1's label for a detected event): no sharp wave, MUA < 2×, and either
  ripple on few CA1 channels or a spectral peak outside the reference set's 1–99 % range. It is
  an evaluation label, never a rule.

## C1: are the events detected at the ports SWRs?

We have no EMG and no lick sensor; a beam break says only that the nose is in the port. So C1
tested the **detected events**, not the licking. Kay events (ripple_detection 2.0, all 218
epochs, outside the prototype artifact mask), by period:

| Period | Kay events | MUA ratio (median) | Sharp wave \|dev\| (median) | Ripple on ≥ 3 z channels (median share) | Lacks SWR features |
|---|---|---|---|---|---|
| At a rewarded port | 30,320 | 2.66 | 1.14 | 0.77 | 15.5 % |
| At an unrewarded port | 4,002 | 1.19 | 0.56 | 0.36 | 49.6 % |
| Immobile, away from the ports | 616 | 3.32 | 1.14 | 0.82 | 8.8 % |
| Other | 4,398 | 2.55 | 1.09 | 0.77 | 16.1 % |

By animal:

| Animal | Rewarded port | Unrewarded port | Away |
|---|---|---|---|
| chimi | 1.4 % | 15.8 % | 0 % (n = 91) |
| j16 | 40.4 % | 63.7 % | 14.9 % (n = 350) |
| peanut | 1.4 % | 11.4 % | 0 % (n = 7) |
| senor | 0.8 % | 3.7 % | 0 % (n = 39) |
| wilbur | 2.8 % | 7.7 % | 1.6 % (n = 129) |

- The events at the ports are not broadband, EMG-like bursts: in every period ≥ 99.2 % have
  in-distribution spike marks, ≥ 96 % a peak frequency inside the reference set's range, and
  ≥ 99.1 % no broadband rise. Where they differ, they are weaker: less MUA, a smaller sharp wave,
  ripple on fewer channels.
- At the ports the reference electrode carries more 250–400 Hz power than away (128 of 153
  epochs on rewarded visits, 143 of 153 on unrewarded ones), and unreferenced CA1 shares more of
  it. The field's 300–600 Hz EMG band is not available: the stored LFP is low-passed at 400 Hz.
  Orofacial muscle reaching the reference is a hypothesis we cannot test.
- **j16 dominates the pooled numbers:** 2,981 of the 4,002 unrewarded-port events, and a high
  featureless share in every period. C1b and C1c asked why.

## The j16 finding: three tetrodes that do not ripple with CA1

C1c found:

- Electrode groups **0, 10 and 22** are among the three loudest tetrodes in 95 %, 88 % and 92 %
  of j16's featureless Kay events, on all 8 days (chance is 3/22 = 14 %).
- During reference-set SWRs their envelope rises to 1.9–2.0 robust z, against 5.1–8.5 for the
  other 19 j16 tetrodes. Their median envelope z while the other tetrodes ripple at ≥ 5 z is
  0.29–0.36, against 3.8–6.7.
- They carry 5.4–6.6× the ripple-band power of the other j16 tetrodes, every day. Kay sums raw
  squared envelopes, so these three weigh about as much as the other 19 together.
- Their high-frequency power rises at the ports relative to away; every other j16 tetrode's
  falls.
- Nothing in the metadata flags them: hippocampal tetrodes in both hemispheres, no bad-channel
  flag, the same single reference as the other 19. **Why they behave this way is unknown.**

## Rule R-T

Written down before any recomputed consensus or event count was looked at. It
reads only the stored ripple band, head speed and the artifact mask: no spikes, MUA, sharp wave
or event label. Per epoch, for each CA1 band tetrode `c` (the stored band holds one channel per
tetrode in all 218 epochs: 22, or 13 for wilbur):

1. `env_c = |Hilbert(stored ripple band)|`; `z_c` is its robust z (median, MADn) over the
   epoch's finite samples.
2. `LOO_c(t) = sqrt(gaussian_smooth(Σ over the other tetrodes of env², σ = 4 ms))`, z-scored
   by its mean and SD over the **reference samples**: finite, speed < 4 cm/s, and outside the
   artifact mask (the one mask of the companion note).
3. `coupling_c` = the median of `z_c` over the reference samples where `LOO_c` ≥ 5.
4. **Drop `c` when `coupling_c < 1.0`.**

**Constants**, each with its evidence:

| Constant | Value | Evidence |
|---|---|---|
| Cut | 1.0 robust z | A mechanism constant: a rise of less than one robust SD while the rest of CA1 ripples. j16 groups 0, 10, 22 score −0.12 to 0.81 in all 58 epochs; every other j16 tetrode ≥ 1.78 |
| LOO threshold | 5 z | "The others ripple" must mean strong ripples. 3.75 drops 61 rippling tetrode-epochs; 5 drops 2 weakly rippling ones; 6.25 drops none (C2) |
| σ | 4 ms | Kay's own smoothing. 3 ms drops 1 weakly rippling tetrode-epoch, 5 ms drops 4 (C2) |

**Fail closed.** The epoch fails, with its reason and every tetrode's coupling in the
manifest, when:

- the rule would drop more than a third of the tetrodes;
- a tetrode's coupling is undefined: it has no reference sample with `LOO_c` ≥ 5, or its
  envelope is flat (MADn 0);
- the band holds two channels of one tetrode (checked before the rule runs, so no coupling is
  recorded). A sibling channel would count as "another tetrode" in `LOO_c`, so a rogue
  tetrode with two channels would look coupled to itself, and the one-third cap would count
  channels. Every epoch on record has one band channel per tetrode; a repopulate could change
  that.

In C1c no epoch of the 218 came near the first two: at most 4 of 22 tetrodes dropped
(senor20201107_08_r4), and every tetrode-epoch had at least 701 high-LOO reference samples.

**In the runner.**

- It runs once per epoch, after the artifact mask is read and before any detector. Kay,
  Karlsson and Yu read the kept tetrodes' band; Zugaro chooses its channel only among them.
- The manifest records, per epoch, every band electrode with its group, its reference, its
  coupling and its count of high-LOO reference samples, and the dropped electrode groups and
  ids (`tetrode_selection`). The run's settings record the constants, so a rerun under other
  constants is refused. When a tetrode is dropped, the Kay version check lists
  "tetrode selection (R-T)" among the direct Kay's differences from the canonical Spyglass
  events, which read every tetrode.
- The rule holds two `(n_time, n_tetrodes)` arrays besides the band, and runs the Hilbert
  transform one column at a time: on the longest epoch tested (j1620210706_02_r1, 2.5 M
  samples × 22) its own peak is about 1 GiB, and the worker's peak (6.9 GiB) is set later, by
  the detectors and the Kay check on Spyglass's own inputs.
- **Reference electrodes are never dropped.** They are not band channels: each band electrode
  is referenced to one (j16 references all 22 to a single corpus-callosum electrode). Dropping a
  band tetrode removes its column; each kept electrode keeps its own reference.
  `electrode_ids` and `reference_ids` in the record are subset together, so they stay paired;
  `tetrode_selection` keeps every band electrode's reference.

## Effect (C1c, all 218 epochs)

Which tetrodes it drops:

| Animal | Epochs with a drop | Tetrodes dropped |
|---|---|---|
| j16 | 58 / 58 | groups 0, 10, 22 in every epoch |
| chimi | 5 / 45 | group 17, Feb 27–28 only (a good SWR tetrode on the other days) |
| senor | 39 / 58 | mostly group 1 (28 epochs) and group 0; both near-silent (0.02 Hz marks), on the S-affected electrodes 0–7 |
| wilbur | 5 / 44 | group 2, Apr 2–3 |
| peanut | 0 / 13 | none |

Before and after, Kay with the inventory's settings. These numbers are
C1c's: the prototype mask, R-rip applied, and Kay's default normalization.

| Animal | Kay events | Lacks SWR features | Reference-set SWRs detected (of n) | Model-reviewed SWRs detected | Rewarded port: lacking | Unrewarded port: lacking |
|---|---|---|---|---|---|---|
| **j16** | 16,114 → 11,092 | **43.2 % → 1.9 %** | 3,767 → **4,229** (4,327) | 5 → 6 (7) | 40.1 % → 0.8 % | 63.6 % → 3.7 % |
| chimi | 9,386 → 9,361 | 1.4 % → 0.5 % | 1,915 → 1,924 (2,168) | 23 → 23 (24) | 1.2 % → 0.4 % | 14.3 % → 3.6 % |
| peanut | 1,773 → 1,773 | 1.9 % → 1.9 % | 569 → 569 (651) | 6 → 6 | unchanged | unchanged |
| senor | 7,270 → 7,284 | 0.9 % → 0.9 % | 1,293 → 1,300 (1,458) | 4 → 4 (9) | 0.8 % → 0.7 % | 3.7 % → 3.4 % |
| wilbur | 4,773 → 4,771 | 3.9 % → 4.0 % | 562 → 561 (604) | 9 → 9 | 2.8 % → 2.9 % | 7.6 % → 7.4 % |
| all | 39,316 → 34,281 | 18.8 % → 1.6 % | 8,106 → 8,583 (9,208) | 47 → 48 (55) | 15.4 % → 1.0 % | 49.5 % → 4.8 % |

- In j16, 8,937 events disappear (77 % of them lacking features) and 4,124 new ones appear
  (4 % lacking). Removing the three loud tetrodes lowers the consensus SD, so smaller SWRs
  cross the threshold.
- 28 reference-set SWRs are lost and 505 gained, over all animals.

**Blind spot-check** of the j16 change (36 events, labels given before unblinding, by the model
reviewer):

| Stratum | SWR | Unclear | No event (background crossing) |
|---|---|---|---|
| Lost events (random 12) | 1 | 0 | 11 |
| New events (random 12) | 7 | 5 | 0 |
| Kept events (random 6) | 6 | 0 | 0 |
| Reference-set SWRs lost (random 6 of 25) | 3 | 2 | 1 |

**Held out and alternatives** (C2):

- With j16 held out, cut 1.0 drops 174 of 174 culprit tetrode-epochs and 0 of 1,045 rippling
  j16 tetrode-epochs. With each other animal held out it drops one weakly rippling tetrode-epoch
  in senor (group 0) and one in wilbur (group 2); C1c measured the wilbur drop's cost at
  1 reference-set SWR.
- Rule P's analogue (1/1.5 of the rippling floor, 0.61) keeps 9 of 174 culprit tetrode-epochs,
  whose epochs return to 9–50 % featureless events. A relative form (0.25 × the epoch's median
  coupling) drops 172 of 174 and no rippling one; it is not better enough to replace the stated
  rule.
- The asymmetry favours dropping: one weak tetrode of 13–22 costs about one reference-set SWR,
  while one kept culprit restores up to half of an epoch's events as featureless.
- **Precedent** (read from source in C2): PREP and EEGLAB `clean_channels` drop a channel whose
  correlation with the others is low; IBL / SpikeInterface `detect_bad_channels` drops a channel
  unlike the population median. R-T is the same test, conditioned on the moments the others
  ripple.

## Acceptance against C1c (Task 17b)

The runner, with a scratch artifact run at the implementing commit, on 13 epochs: the 5 pilots,
j16 on four more days, chimi Feb 27 and Feb 28 (one epoch each) and two senor S epochs.

- **Dropped groups** equal C1c's in all 13: j16 0, 10 and 22 on 5 days; chimi 17
  on Feb 27 and 28; senor 0 (20201101_14_r7) and 1 (20201107_12_r6); none elsewhere (wilbur's
  13 tetrodes included).
- **Coupling.** With C1c's (prototype) mask as the reference exclusion, every tetrode's coupling
  equals C1c's exactly. With the one mask it moves by at most 0.016, except in
  chimi20200227_02_r1 (up to +1.15, upward, no drop changed), where the prototype masked a 0.2 s
  spike-only core that the one mask does not.
- **Kay events.** Kay on C1c's kept tetrodes with Kay's default normalization reproduces C1c's
  selected events exactly. Every difference between the runner's Kay events and C1c's selected
  run has one cause, and none is unexplained: the runner normalizes over the unmasked samples
  (17a), which adds events in the 4 epochs whose mask holds an LFP or S core (+152, +9, +112,
  +131; none lost), and the two masks differ by one event (chimi20200227_02_r1). R-rip would have
  vetoed none of C1c's selected events in these 13 epochs.
- **LOO 6.25** (sensitivity): one tetrode-epoch differs from LOO 5. senor20201101_14_r7 group 0
  scores 0.99 at 5 (dropped) and 1.15 at 6.25 (kept); C2 also names senor group 0 as weakly
  rippling.

## The reference veto (C1b) and the field's form: not used

**The field's form** (buzcode `bz_FindRipples`, FMAToolbox `FindRipples`, read from source):
drop an event when a noise channel's ripple-band power z-score also crosses a threshold. With
the reference electrode as the noise channel:

- at the field's default threshold (5 SD) it loses 251 of 9,208 reference-set SWRs (2.7 %),
  because large SWRs reach our white-matter reference;
- at a threshold safe for the reference set (14.1 SD) it removes 10 Kay events, half of them
  SWR-like.

**R-rip** (C1b): drop a detected event when the reference electrode's 150–250 Hz envelope is
more than **11.36** times the CA1 median envelope at the event's peak (± 25 ms), the noise-channel
check scaled by CA1 at the same moment. θ = 1.5 × its largest value over the calibration SWR
sets; it lost 0 reference-set SWRs in all 7 held-out splits, and 8 of 8 of its removed events
reviewed blind were not SWRs.

**Why neither is used.**

- The field's form is either unsafe for large SWRs or removes nothing.
- At 11.36, R-rip removes 117 of C1c's Kay events before R-T (C1b reported 116 at 11.4), 103 of
  them j16's. After R-T it would remove **5** over all 218 epochs (3 chimi, 2 senor; none in
  j16), counted from C1c's per-epoch events. The lead dropped it (2026-10-09): one rule fewer,
  and R-T handles what it was for.
- The j16 residual that R-rip could not reach (43.1 % featureless with R-rip alone) is what R-T
  removed.

## Visit onset: numbers only, no rule

- **In every animal,** events in the first second after the pump (rewarded) or the poke-in
  (unrewarded) lack SWR features more often (C1b, share lacking): rewarded 0–0.5 s, chimi 59 %,
  j16 85 %, peanut 35 %, senor 22 %, wilbur 20 %; rewarded 1–2 s, 0–4 % except j16 (79 %).
  Before R-T, j16's window lasted about 2 s, and on unrewarded visits it never ended.
- The median R-rip there is 2–7, against about 1.1–2 later in the visit. This is when licking
  starts; that is a hypothesis.
- A visit-onset window cannot be calibrated on the reference set, which needs < 4 cm/s over
  ± 250 ms: it holds 4 reference-set SWRs in the first second of visits and 560 in the first 2 s.
  In the first second, 85 % of Kay events exceed 4 cm/s within ± 250 ms.
- **After R-T,** the featureless share at the ports is **1.0 %** on rewarded visits and **4.8 %**
  on unrewarded visits over all animals (j16 0.8 % and 3.7 %).
- A padded speed rule (every sample < 4 cm/s over the event ± 0.25 s) would keep 81 % of Kay
  events and only 37–79 % at unrewarded ports, and lowers j16's featureless share only from
  43 % to 33 %.
- **No visit-onset or padded-speed rule is added.** The canonical whole-event immobility rule
  (every sample < 4 cm/s, `analysis/events/immobility.py`) is unchanged.

## Limits

- **Licking was not measured.** There is no EMG, lick sensor or broadband raw data in the
  project loaders; the 250–400 Hz proxies are bounded by the 400 Hz low-pass and overlap the
  ripple band's upper edge.
- **One rater.** Every label (the pooled review, C1c's 36-event spot-check, C1b's vetoed events)
  is the model reviewer's, with no human and no second rater. "Lacks SWR features" is a
  heuristic label.
- **Why j16's three tetrodes are loud and uncoupled is unknown.** The hypothesis that they sit
  away from the pyramidal layer, or pick up a non-hippocampal signal, is untested.
- **R-T is calibrated on one bad-tetrode animal.** The cut sits 1.23× above every culprit
  tetrode-epoch (at most 0.81), but the weakest rippling tetrode-epoch scores 0.92, below it, so
  the rule drops 2 weakly rippling tetrode-epochs (senor group 0, wilbur group 2), about 1
  reference-set SWR. More animals with bad tetrodes would test it.
- senor's dropped groups 0 and 1 are near-silent tetrodes on the electrodes the S noise affects;
  R-T removes them from detection only, not from decoding.
