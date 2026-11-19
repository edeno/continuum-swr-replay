# Decoder discrete-transition prior (decision R10: setting P30-B)

**Status (2026-10-04):** decided by the project lead after validation on the five pilot
epochs. Implemented in `src/continuum_swr_replay/decoding/transitions.py`, the only place
the values live. Part of the full pipeline refresh
([2026-10-03-full-pipeline-refresh-design.md](2026-10-03-full-pipeline-refresh-design.md));
the re-decode runs from the commit that adds this pin.

## Why

- **The decoders never set their discrete transitions.** `run_decoding_bandit.py` and
  `run_decoding_contfrag.py` passed `discrete_transition_type=None` and nothing else, so they
  took `non_local_detector`'s defaults.
- **Those defaults changed after the April decodes.** In April 2026 the library froze the
  No-Spike row by default, cut the Local/No-Spike stickiness from 1e6 to 1e5 and the
  non-local stay from 0.98 to 0.96 (commits `cb6d25a`, `def82da`, `a1a26b0`). On identical
  inputs the five pilot epochs went from 1,021 non-local events under April's settings to
  1,349 under today's defaults (+32 %).
- **Nothing in our code recorded the values,** so a library upgrade could move the decode
  again without any trace.
- **The project lead required** every value to be justifiable from prior knowledge, with a
  source, a stated rationale and a sensitivity check.

## Decision

| # | Decision | Why |
|---|---|---|
| R10 | The NonLocal decode uses setting **P30-B**. Each learned row has a fixed-count Dirichlet prior whose mode (the M-step's zero-data estimate) gives dwell times from the literature (Local 3 s, each non-local state 100 ms), worth 30 visits per state. The No-Spike row is frozen, and its exits follow base rates. The ContFrag decode keeps the values it has always used, now written out. Both runners refuse `--non_stationary`. | The project lead's choice (2026-10-04), after the pilot validation below, under the requirement that every value be justifiable from prior knowledge: each value has a source or a stated base-rate assumption. Against P30 (the same prior with even No-Spike exits), P30-B halves the extra events, which are mostly No-Spike-entered and weakly supported, and restores much of April's immobile regime |

## How the library uses these values

- **States**, in the library's order: 0 Local, 1 No-Spike, 2 Non-Local Continuous,
  3 Non-Local Fragmented. The decode works in 2 ms bins (`SAMPLING_FREQUENCY` = 500 Hz).
- **Starting matrix.** `discrete_transition_type=DiscreteStationaryCustom(values=M)` is EM's
  starting matrix M.
- **Learned rows.** Each EM M-step re-estimates every learned row as its expected transition
  counts plus α − 1, normalized, with α = concentration + diag(stickiness)
  (`discrete_state_transitions.py`: `get_transition_prior`,
  `estimate_stationary_state_transition_from_counts`).
- **Fixed counts.** With `discrete_transition_prior_weight` 0, these pseudo-counts are fixed;
  they are not scaled to the amount of data.
- **Frozen rows** are reset to their row of M after every M-step.
- **Prior strength.** The M-step adds Σ(α − 1) pseudo-counts to each learned row, and with no
  data its estimate is the Dirichlet's mode, the normalized α − 1. The Dirichlet's effective
  sample size is Σα (Morita et al. 2008).

## Parameters

With each state's stay p = 1 − bin/dwell, and e = `PRIOR_VISITS` / 3 = 10 pseudo-exits to
each destination:

| constant | value | meaning | source |
|---|---|---|---|
| `DECODE_SAMPLE_S` (`parameters`) | 2 ms | decode bin (`1 / SAMPLING_FREQUENCY`) | the decode's 500 Hz grid. The epochs' median step is 2.0000935 ms (R9), a 5e-5 relative difference that is immaterial here |
| `LOCAL_DWELL_S` | 3.0 s, so p_L = 0.99933 | expected stay in Local: the interval between awake events during stops | Davidson, Kloosterman & Wilson 2009, *Neuron*, [doi:10.1016/j.neuron.2009.07.027](https://doi.org/10.1016/j.neuron.2009.07.027): candidate events at a "mean rate during STOP of 0.36, 0.40, 0.32, 0.40 events/s in rats 1–4", an interval of 2.5–3.1 s |
| `NONLOCAL_DWELL_S` | 0.1 s, so p_NL = 0.98 | expected stay in each non-local state | Buzsáki 2015, *Hippocampus*, [doi:10.1002/hipo.22488](https://doi.org/10.1002/hipo.22488): SPW-Rs are transient ("mode: 50 ms in the rat"), and their duration "varies between 30 and 150 ms". Sullivan et al. 2011, *J Neurosci*, [doi:10.1523/JNEUROSCI.0294-11.2011](https://doi.org/10.1523/JNEUROSCI.0294-11.2011): ripple SPW sinks average 51.8 ms. Davidson 2009: replay "event durations ranged from 40–1018 ms". 100 ms covers single ripples and longer replays |
| `NO_SPIKE_DWELL_S` | 0.1 s, so p_NS = 0.98 | expected stay in No-Spike (frozen row) | English et al. 2014, *J Neurosci*, [doi:10.1523/JNEUROSCI.2600-14.2014](https://doi.org/10.1523/JNEUROSCI.2600-14.2014): "Ripple events were followed by a prominent afterhyperpolarization and spike suppression", with firing still reduced 150 ms after the ripple peak. Buzsáki 2015: the "silent period after the SPW‐R" is "associated with decreased multiple unit neuronal activity" |
| `PRIOR_VISITS` | 30 | prior strength: e = 10 pseudo-exits to each destination, so the M-step adds Σ(α − 1) = 30 × dwell/bin pseudo-counts to each learned row: 45,000 (Local) and 1,500 (each non-local state). The Dirichlet's effective sample size Σα is 45,004 and 1,504 | Morita, Thall & Müller 2008, *Biometrics*, [doi:10.1111/j.1541-0420.2007.00888.x](https://doi.org/10.1111/j.1541-0420.2007.00888.x), for the effective sample size Σα ("ESS = 50 = Σ α̃j"). Sensitivity over 10–100 visits is below |
| `NO_SPIKE_EXIT_TO_LOCAL` | 0.96 | share of No-Spike exits that go to Local; 0.02 goes to each non-local state | base rates: 0.36 events/s × 0.1 s puts non-local content at about 3.6 % of immobile time (Davidson 2009). No source reports which state follows a silence in awake animals, so the next state follows the stationary distribution |

Arguments passed to `NonLocalClusterlessDetector` / `NonLocalSortedSpikesDetector`:

| argument | value |
|---|---|
| `discrete_transition_type` | `DiscreteStationaryCustom(values=M)`, M below |
| `discrete_transition_concentration` | 1 + e = 11.0 |
| `discrete_transition_stickiness` | e(4p − 1)/(1 − p) per state = [44,960, 1,460, 1,460, 1,460] (the No-Spike entry has no effect, because that row is frozen) |
| `discrete_transition_prior_weight` | 0.0 (fixed counts) |
| `frozen_discrete_transition_rows` | (1,), the No-Spike row |
| `local_position_std` | 2.5 cm (unchanged, `LOCAL_POSITION_STD`) |

The starting matrix M, row = from, column = to:

| from \ to | Local | No-Spike | NL-Cont | NL-Frag | expected stay |
|---|---|---|---|---|---|
| Local | 0.999333 | 0.000222 | 0.000222 | 0.000222 | 3 s |
| No-Spike (frozen) | 0.0192 | 0.98 | 0.0004 | 0.0004 | 100 ms |
| NL-Cont | 0.00667 | 0.00667 | 0.98 | 0.00667 | 100 ms |
| NL-Frag | 0.00667 | 0.00667 | 0.00667 | 0.98 | 100 ms |

Each learned row's prior mode (the M-step's zero-data estimate, its normalized α − 1) equals
its row of M, so the prior pulls EM toward these dwell times with the strength of 30 visits.
The Dirichlet's mean, α/Σα, has slightly shorter stays: 2.7 s for Local (0.99927) and 91 ms
for each non-local state (0.978). `tests/decoding/test_decoding_transitions.py` pins all of the
above.

Against the P30-B pilot decodes (`Processed-Data/_ablation_prior_P30B_20261004/`), the
pinned values are identical bit for bit in float32, which the HMM computes in, and agree to
within 13 ulps in float64, because the pilot typed the non-local literals (0.98, 0.02) where
the code derives them; compare against the ablation pickles with a relative tolerance.

### ContFrag

The ContFrag classifier keeps the values it has always decoded with: the library's ContFrag
defaults, now passed explicitly by `contfrag_transition_settings()`. They are a 100 ms
starting stay with a near-flat prior (concentration 1.1), i.e. weakly informative.

| argument | value |
|---|---|
| `discrete_transition_type` | `DiscreteStationaryDiagonal(diagonal_values=[0.98, 0.98])`: a 100 ms expected stay in Continuous and in Fragmented |
| `discrete_transition_concentration` | 1.1 (0.1 pseudo-counts on every transition) |
| `discrete_transition_stickiness` | [0.0, 0.0] |
| `discrete_transition_prior_weight` | 0.0 |
| `frozen_discrete_transition_rows` | None (both rows learned) |

### Non-stationary transitions are disabled

Both runners, `run_decoding_bandit.py` and `run_decoding_contfrag.py`, refuse
`--non_stationary` at the command line, before any cluster or GPU work, and their
`perform_decoding(non_stationary=True)` raises. The library's non-stationary transitions
have two problems:

- **They cannot be saved.** The fitted detector does not pickle
  ([LorenFrankLab/non_local_detector#50](https://github.com/LorenFrankLab/non_local_detector/issues/50)).
- **The prior is added at every time bin** rather than once
  ([#51](https://github.com/LorenFrankLab/non_local_detector/issues/51)). The NonLocal
  stickiness above would then freeze every row, and ContFrag's concentration of 1.1 would add
  0.1 pseudo-counts to every transition at every bin.

The flag comes back when both are fixed.

## Validation (five pilot epochs)

Pilot epochs (HPC): peanut20201202_10_r5, j1620210707_06_r3, chimi20200225_02_r1,
wilbur20210329_12_r6 and senor20201101_06_r3. Every setting decoded the same inputs; only
the discrete transitions differed. Every number in this section is a dated measurement of
2026-10-04 on these five epochs, not computed by a summary; the validation's working files
were scratch analyses and are not in the repository.

- **Events.** 245 / 236 / 264 / 195 / 223 = **1,163**. For comparison, April's discrete
  settings on the same inputs give 1,021, and today's library defaults give 1,349.
- **Held-out spike support.** The primary test is the location test with running-only
  encoding and split tetrodes; validity comes only from this test, never from SWR overlap.
  - Against April's settings, P30-B adds 148 events. They pass at 0.68, against 0.80 for
    shared events (adjusted logit OR for the non-SWR additions 0.78, 95 % CI 0.51–1.21).
  - It drops 57 April events, which pass at only 0.30.
  - Only 7 % of the additions start right after No-Spike; under P30 with even No-Spike
    exits it was 51 %.
- **Immobile regime.** The median non-local run is 74 ms, with 0.55 entries/s. April's
  settings give 98 ms and 0.46/s; today's defaults 26 ms and 1.07/s.
- **Held-out evidence per state.** In all four settings tested (April's settings, W1-L, P30
  and P30-B), held-out spikes support Local, both immobile and running, and immobile
  Non-Local Continuous and Fragmented.
  - In P30-B's immobile bouts the evidence is above 0 in 82 % of Local bouts. It is above 0
    in 73 % of Non-Local Continuous bouts (remote MAP against the animal's position) and in
    67 % of Non-Local Fragmented bouts.
  - The evidence does not separate W1-L, P30 and P30-B.
  - April's settings have one distinct flaw: 15 % of their immobile No-Spike bouts contain
    held-out spikes, against 2 % for P30-B.
- **EM.** No EM iteration decreased the log-likelihood in any of the 15 B-family pilot fits
  (P10-B, P30-B and P100-B on the five epochs). The final E-step, stored after the last
  M-step, fell slightly in three of them: P30-B senor −126, P10-B j16 −28 and P10-B wilbur
  −104 nats, on totals of 5 to 11 million.
- **No takeover.** This was checked for P10-B, P30-B and P100-B. Values below are in that
  order, with April's settings for comparison.
  - **Immobile bouts ≥ 1 s:** 612 per setting. Of these, 8 / 8 / 8 are ≥ 50 % confidently
    non-local (P(non-local) > 0.8), and 3 / 3 / 3 are ≥ 90 %. April's settings: 13 and 5.
  - **Running bouts ≥ 1 s:** 1,002. Of these, 8 / 8 / 8 are ≥ 50 % non-local, and one in
    each setting is ≥ 90 %. The time-weighted non-local share per epoch is at most 0.172,
    0.169 and 0.162.
  - **Events:** the longest is 0.726, 0.736 and 0.730 s; none is longer than 1 s. April's
    settings have 3 events over 1 s, the longest 1.58 s.
- **Simulation (same recordings as the earlier variants).**
  - Noise-extension recall is 169 of 242 true events (100 of 166 short ones), with
    12 false calls and no splits.
  - The silence probe gives 31 false calls. For comparison: P30 gives 55; W1-L 27; today's
    defaults 21; April's settings 1.
- **Sensitivity to prior strength.** P30-B's prior was also decoded at 10 and 100 visits
  (P10-B, P100-B). Values are P10-B / P30-B / P100-B.
  - **Events:** 1,163 / 1,163 / 1,158. Per epoch (peanut, j16, chimi, wilbur, senor):
    243, 237, 265, 193, 225; 245, 236, 264, 195, 223; and 241, 239, 260, 197, 221.
  - **Against April's settings** (shared / setting-only / April-only): 1,012 / 151 / 59;
    1,015 / 148 / 57; and 1,012 / 146 / 55.
  - **Held-out pass rate:** all events 0.78 / 0.78 / 0.77; setting-only 0.67 / 0.68 / 0.69;
    April-only 0.29 / 0.30 / 0.31.
  - **Logit OR** for the non-SWR additions: 0.77 / 0.78 / 0.88, all p > 0.2.
  - **Immobile regime:** median non-local run 74 / 74 / 78 ms, and 0.55 / 0.55 / 0.53
    entries per s.
  - **No-Spike occupancy:** 9.0 / 8.9 / 8.6 %.
  - **Additions entered from No-Spike:** 8 / 7 / 8 %.

  So a 10x range of prior strength (10 to 100 visits) changes the event count by at most
  0.5 %, and each epoch's count by at most 4 events from P30-B's count. The P family with
  even No-Spike exits was also insensitive: 1,381 / 1,390 / 1,416 events at
  N = 10 / 30 / 100, a 2.5 % range (−0.6 % to +1.9 % from P30). There, too, the No-Spike
  exit row, not N, was the lever.

## Known limits

- **Theta sequences are not captured.** A 100 ms non-local stay blends theta-scale
  excursions during running into the non-local states. The project lead accepted this when
  setting the non-local stay to 100 ms.
- **No-Spike occupancy** is 8.9 % of the epoch, against 4.7 % under April's settings. No
  setting tried brought it back down.
- **The added events are slightly weaker than shared ones** (pass 0.68 vs 0.80; the CI of the
  OR includes 1). In chimi, wilbur and senor the additions pass 0.19–0.35 below shared
  events. Wilbur ends with fewer events than under April's settings (195 vs 206).
- **No awake source for the post-silence destination.** `NO_SPIKE_EXIT_TO_LOCAL` is a
  base-rate assumption.
- **Running Non-Local Continuous is not supported by held-out spikes.** Under running-only
  place fields, P30-B's held-out spikes favor the animal's own position in 65 % of running
  Non-Local Continuous bouts. Our event analyses exclude these bouts through the whole-event
  immobility rule (every sample below 4 cm/s).
- **Validated on HPC clusterless decodes only.** `create_detector` serves every brain area in
  `decoding/config.py:nonlocal_detector_classes`, so mPFC and OFC sorted-spike decodes, which
  are not part of the refresh, would also take P30-B.
- **Five epochs.** Pooled event-level rates treat events as independent.

## Sources

- Buzsáki G (2015). Hippocampal sharp wave-ripple: a cognitive biomarker for episodic memory
  and planning. *Hippocampus* 25:1073–1188. doi:10.1002/hipo.22488
- Davidson TJ, Kloosterman F, Wilson MA (2009). Hippocampal replay of extended experience.
  *Neuron* 63:497–507. doi:10.1016/j.neuron.2009.07.027
- English DF, Peyrache A, Stark E, et al. (2014). Excitation and inhibition compete to
  control spiking during hippocampal ripples: intracellular study in behaving mice.
  *J Neurosci* 34:16509–16517. doi:10.1523/JNEUROSCI.2600-14.2014
- Morita S, Thall PF, Müller P (2008). Determining the effective sample size of a parametric
  prior. *Biometrics* 64:595–602. doi:10.1111/j.1541-0420.2007.00888.x
- Sullivan D, Csicsvari J, Mizuseki K, et al. (2011). Relationships between hippocampal sharp
  waves, ripples, and fast gamma oscillation: influence of dentate and entorhinal cortical
  activity. *J Neurosci* 31:8605–8616. doi:10.1523/JNEUROSCI.0294-11.2011
- `non_local_detector` 0.6.10.dev330+g2d730c3a8 (locked in `uv.lock`); issues
  [#50](https://github.com/LorenFrankLab/non_local_detector/issues/50) and
  [#51](https://github.com/LorenFrankLab/non_local_detector/issues/51).
