# Figure narrative: claims → comparison → reasoning

Working plan for the SWR / non-SWR non-local-event story. Each row states
the **claim**, the **comparison** the panel makes (and the *control* that
licenses it), and the **reasoning** for why that comparison establishes the
claim. "Status" ties each to a current panel or marks it a GAP.

Top-level claims, in order:

- **A.** Non-local events exist that do **not** overlap detected SWRs — a
  large population that SWR-triggered analysis never sees.
- **B.** These non-SWR events are **real**, not decoding artifacts.
- **C.** Non-SWR events are a **different kind** of event than SWR ones
  (dynamics + content), and serve a **different function**.
- **D.** Our state-space method **detects non-local events that the
  traditional "detect-SWR → Bayesian-decode" pipeline misses.**

---

## Current figure inventory (what is actually plotted today)

**Fig 1 — the method (switching state-space non-local detector).**
A: 4-state transition diagram. B: observation model, 3 scenarios
(Local/Non-Local/No-Spike wins). C: continuous-dynamics kernels per state.
D: simulated example (raster, non-local posterior, state probs, collapsed
non-local prob + threshold, speed).

**Fig 2 — "the model finds SWR-coincident replay (and more)."**
a: three example events, one per overlap class (SWR-contained / SWR-partial /
non-SWR). b: stacked bar — proportions of immobile non-local events by
overlap class. c: event-triggered averages of non-local prob + SWR z-score,
split by overlap status. d: NL↔SWR timing-offset histograms (NL-start − SWR-
start, NL-end − SWR-end) on overlapping events. e: median Local-state
decoding error across sessions (decoder-accuracy positive control).

**Fig 3 — "non-local events exist outside detected SWRs."**
a: real-data non-SWR non-local example (flat SWR z, clear posterior replay).
b: SWR z-score density, Local vs Non-Local timepoints (non-local lives at
*low* SWR power, not just below threshold). c: replay speed by SWR status.
d: replay position by SWR status ("non-SWR represents reward zones").

**Fig 4 — "PFC neurons respond to replay regardless of SWR detection."**
a: single mPFC neuron, overlaid SWR + non-SWR GLM modulation + PSTHs.
b: population heatmaps, SWR vs non-SWR modulation (shared sort order).
c: population average modulation (mean ± SEM). d: % neurons significantly
modulated + best-AIC model distribution.

---

## Claim A — non-SWR non-local events exist and are common

| panel | comparison (control) | reasoning | status |
|---|---|---|---|
| Fig 2a | three real events, one per overlap class | grounds the reader; a non-SWR event whose posterior is a clean trajectory shows the phenomenon is not a label artifact | **have** |
| Fig 2b | proportions across overlap classes | quantifies how much of the non-local population carries no SWR (the part SWR-triggering misses) | **have** |
| Fig 2c | NL-prob + SWR-z triggered avg, by overlap | shows non-SWR events have a real non-local-prob rise with no SWR-z rise — the two signals dissociate at event onset | **have** |
| NEW | **2×2: SWR present/absent × non-local-state present/absent** (rate per min) | makes the *decoupling* explicit and seeds Claim D: most non-local events sit in the no-SWR cell; many SWRs carry no non-local content | **GAP** |

Reasoning note: A is about *prevalence + dissociation*, not yet reality. The
2×2 is the cleanest single statement of "what SWR-triggering misses" and is
the hinge to Claim D.

---

## Claim B — the events are real, not artifacts (LINCHPIN)

Key insight from analysis: non-local events split into a **close subset**
(~28% of non-SWR; decoded ~23 cm away, same arm as the animal, flagged
non-local by *dynamics state* not distance) and a **remote subset** (~72%;
~1.8 m away, different arm). The close subset is the artifact-suspect one and
must be handled explicitly. "Non-local" = a dynamics-state label, not a
distance threshold — state and spatial-remoteness can dissociate.

The reality argument is built around four artifact worries (W1–W4); the
**centerpiece is the prior-excluded likelihood**, not the posterior. See
`claims_table.md` Goal-2 (rows 6–13) for the authoritative per-claim list —
this section tracks figure/status. **Status: the spike-evidence engine is
built, reviewed, and confirmed 5/5 animals (cross-animal smoke).**

| worry → test | comparison (control) | reasoning | status |
|---|---|---|---|
| W1 → theta/speed | theta-z + speed: SWR / non-SWR / running | both event types at low theta+speed, unlike running — not theta sequences | data in hand |
| W2a → spikes | per-event spike count / MUA-z / #tetrodes vs **matched non-event baseline** | real coordinated spikes present, not a spike vacuum | **prototyped; 5/5 smoke** |
| W2b+W3 → **likelihood (CENTERPIECE)** | ΔLL = logL(MAP) − logL(animal) and per-spike vs **matched non-event baseline (≈0)**; recomputed from the saved encoding model + marks, native track-distance bins | spikes specifically support the decoded location above baseline → real, not prior-coasting; per-spike rules out interneurons *without sorting* | **prototyped + 5/5 smoke** (events ≫ baseline≈0, all 5 animals) |
| W2b → GPI | event-MAP track-distance to GPI peak vs random | MAPs not closer to the GPI peak than chance → content-specific, not a ground-process default | **prototyped; 5/5 smoke** |
| W3 definitive → local-kernel re-decode | re-decode a subset with the multi-bin local kernel — do close events absorb into Local? | model-based test the local/non-local boundary isn't set too tight | **GAP (GPU subset, pending re-decode)** |
| W4 → far events | remote subset survives W2 | distance proves non-locality given spikes | follows from W2 |

Fig 2e (Local-state decode error) and Fig 3b (SWR-z density) are *supporting*,
not the reality core — there's no ground truth for the non-local location, so
reality rests on the spike-evidence tests above. The old edge-effect-control,
MUA/ripple-baseline, and spike-shuffle panels are **subsumed** by the
likelihood engine + the GPI test (tetrode-holdout and interneuron-exclusion
likewise dropped — flat-vs-peaked likelihood handles them). **Decode-save /
all-events:** the decode now persists `log_likelihood`; the full decode-pinned
all-events run (reading it, no recompute) is **deferred to the re-decode** (the
30-cap recompute sweep was killed). Spec:
`docs/design/2026-06-25-non-swr-event-reality-design.md`.

Open decision (B): headline claims cover **all** events or the **remote**
subset? Default remote-clean; close is shown separately and clears the W2 part
of the gate in 5/5 (smoke), with the W3-definitive local-kernel test pending.

---

## Claim C — non-SWR events are a different kind / function

Caveats baked into the reasoning: the dissociation must be **duration-
controlled** (SWR events are longer) and **consistent within animals**
(pooled p-values can be subset-driven — e.g. `multimodal_fraction` was only
3/5). Content/behavior claims carry an **autocorrelation caveat**
(current ≈ next/last-rewarded arm in this within-patch-alternation task).

| panel | comparison (control) | reasoning | status |
|---|---|---|---|
| Fig 3c | replay speed by SWR status | non-SWR replay is slower; duration-controlled GEE ×4.91, **5/5 animals** → a genuine dynamics difference, not a size artifact | **have — annotate with dur-control + 5/5** |
| Fig 3d | replay position by SWR status ("reward zones") | non-SWR content over-represents current + last-rewarded arms | **have — but revisit:** confounded with the close subset and arm autocorrelation; re-cut on remote subset + per-arm content model |
| NEW | spatial extent / fragmentation, **duration-controlled + within-animal** | the clean "two-axis" dissociation: size (SWR bigger) + dynamics (SWR faster/jumpier); report `spatial_extent`, drop `crosses_trunk`/`n_unique_arms` as redundant (binarized extent) | **GAP (analysis done)** |
| Fig 4a–d | PFC modulation, SWR vs non-SWR replay | PFC neurons respond to *both* event types with parallel structure → non-SWR events are functionally engaged at the network level (also a reality check for B) | **have** |
| NEW | behavioral modulation: event content vs behavioral state, SWR vs non-SWR | establishes *different function* (e.g. SWR content modulated peri-switch/reward; non-SWR flat or value-locked) | **GAP — needs the clean, duration-controlled version** |

---

## Claim D — our method beats SWR-triggered decoding

| panel | comparison (control) | reasoning | status |
|---|---|---|---|
| Fig 2 title / 2b | "finds SWR replay and more" | proportions already imply a sensitivity gain | **partial** |
| Fig 3 | events exist outside SWR | demonstrates the missed population qualitatively | **partial** |
| NEW | **head-to-head on the same data:** traditional (detect SWR → decode in window) vs state-space detector — events captured by both / unique-to-ours (non-SWR) / SWRs-with-no-replay | the payoff: quantifies sensitivity gain (~the no-SWR fraction) and false positives of SWR-triggering; only credible *after* B and C show the extra events are real and distinct | **GAP** |

The arc in one line: they exist and are common (A) → they are real (B) →
they are a distinct kind with a distinct function (C) → therefore SWR-
triggering misses something real and important and our method does not (D).

---

## Build order / status

- **Reality core (B) — DONE (prototype + 5/5 smoke).** The likelihood engine
  (#7/#8/#8b) is built, reviewed/approved, and reproduces 5/5 animals. The full
  decode-pinned all-events run is **deferred to the re-decode** (decode now
  saves `log_likelihood`). The **W3-definitive local-kernel re-decode (#9) is
  pending**.
- **Still GAP:** 2×2 decoupling (A, #4), threshold sweep (#3), model selection
  (#12), FP null (#13), head-to-head (D, #20), extent axis (C, #15), behavioral
  content recut (C, #16) on the remote subset.
- The edge-effect-control and MUA/ripple-baseline panels in the old plan are
  **subsumed** by the likelihood engine + the GPI test — not separate work.

## Open decisions

- Close vs remote: scope of headline claims (default remote-clean).
- Whether D needs a literal traditional-pipeline re-implementation or a
  derived contingency suffices.
- Does Fig 3d's "reward zones" survive the remote-subset + autocorrelation
  controls, or get reframed as "recent/valued locations"?

---

## Revisions — missing-claims pass

`claims_table.md` is now the authoritative numbered list (20 rows, with a
Main/Supplement placement column). This pass added six claims and fixed
several justifications. Summary:

**New claims (M1–M6):**
- **M1 (table row 10) — spike-driven, multi-tetrode.** Reality is now claimed
  at the per-event level: each event is driven by real spikes from multiple
  tetrodes, not the No-Spike/continuity process. This is the core answer to
  the firing-rate objection and the single most important addition. New
  analysis (per-event spike count + tetrode participation); nothing existing
  computes it. **Main.**
- **M2 (row 6) — two subpopulations (near + remote).** Promote the close/remote
  split from a buried control to a stated result; the remote set is the
  unambiguous population and fixes the denominator. `distance_from_animal`
  exists; split it by SWR. **Main.**
- **M3 (row 7) — not theta sequences.** Pre-empt "non-SWR replay = known theta
  sequences." `event_context` (SWR/theta/neither) + low theta-z during events.
  **Supp.**
- **M4 (row 13) — detector specificity / low false-positive rate.** Shuffle /
  no-replay-expected nulls. Note `nonlocal_during_movement` is treated by the
  code as real theta, so it is NOT the FP control. **Supp.**
- **M5 (row 3) — threshold robustness.** Sweep non-local + SWR-z thresholds;
  the ~60% and the decoupling must survive. **Supp.**
- **M6 (row 19) — content predicts choice.** Promote the existing
  `behavior_prediction` analysis (all / SWR / non-SWR → next well) to a Main
  functional claim — strengthens the thin "different function" goal.

**Justification fixes (existing rows):**
- Row 8 (decoder accurate): movement-regime accuracy does not transfer to the
  immobile/sparse regime — condition calibration on immobility + firing rate.
- Rows 9↔11 reconcile: "genuinely low SWR power" vs "rides ripple-z 0.54" —
  one coherent sub-threshold-but-real story.
- Row 14 (speed): do NOT claim firing-rate-independence — reframe as "slower,
  consistent with fewer participating cells, and still spike-driven (M1)."
- Row 16 (content): restate "reward zones" as current-arm bias;
  value/recency unidentifiable under task collinearity (`predictor_collapse`
  already quantifies it).
- Row 17 (PFC): add a matched non-event immobile control + an equivalence
  test; shared sort order is not a test.
- Stop citing posterior-peak 0.99 as evidence of reality (a spike-vacuum
  posterior can be sharp).

**Main vs Supplement:** Main carries the forward argument (existence, the
spike-driven reality core, the kind/function dissociation, the method payoff);
Supplement carries robustness + controls (threshold sweep, not-theta,
population-activity baseline, FP null, extent axis, collapse quantification,
recruitment model). The edge-effect gate (row 12) must exist before any
"all events" number is published, but can live in Supp once clean.

## Revisions — feedback pass 2

**Dropped as unsupported** (moved to "Considered and dropped" in the table):
- *Not sub-threshold SWRs* — contradicted by the ripple-power bump at
  non-local onset. Reframed into a method claim (events lie on a continuum
  with SWRs; an LFP threshold misses real sub-threshold events). Dissolves the
  old low-power↔rides-ripple contradiction.
- *Recruited differently by behavior* — not shown ("flat" is a null).
- *Content predicts upcoming choice* — evidence points to no; previously
  invalidated by task autocorrelation.

**Goal 3 reframed:** with content-predicts-choice and behavioral-recruitment
gone, and PFC responding to *both* event types, the evidence supports
"different **kind**, functionally **engaged**" — not "different function."
The PFC similarity is reframed as a relevance/reality argument (non-SWR events
drive cortex like SWR events → not epiphenomenal).

**Model cross-validation = model selection by held-out likelihood** (new
table row 8): held-out LL favoring the 4-state model over ContFrag / reduced
models justifies the event-defining architecture and answers the circularity
objection. (Note: existing `cross_validate_conditional_poisson` is for the
content GLM, not the decoder — decoder CV is new work.)

**Goal 4 expanded** to: (17) detect events with no SWR; (18) more accurate
event boundaries than SWR (Fig 2d) → better cross-area alignment; (19) SWRs
sometimes carry local content (specificity); (20, candidate) detection spans
the ripple-power continuum; (21, candidate) content-targeted not
LFP-proxy-triggered.

## Revisions — feedback pass 3 (reality block by worry)

The Goal-2 block is now organized around four artifact worries, each with its
discriminating test (claims_table rows 6–13):
- **W1 theta/movement** → theta/speed contrast {SWR, non-SWR, running}; data
  in hand, no filter tightening needed.
- **W2a no-spike/ground-process** → per-event spike count + MUA-z + #active
  tetrodes vs baseline (the simple version suffices).
- **W2b non-spatial spikes + W3 field extent → CENTERPIECE:** the *likelihood*
  (prior-excluded), not the posterior, must be peaked at the decoded location.
  A flat likelihood = interneuron/prior-coasting artifact; peaked = earned by
  spatial spikes. Rules out interneurons *without sorting*, and prior-coasting
  via likelihood-peak-location vs the GPI/prior prediction.
- **W3 definitive** → re-decode a subset with the multi-bin local kernel (not
  used in the current decode) — do close events absorb into Local?
- **W4 distance** → far (remote) events prove non-locality, given W2 survives.

Dropped from the reality controls: **tetrode-holdout** and
**interneuron-exclusion** — both subsumed by the likelihood test (non-spatial
spikes → flat likelihood). The edge-effect gate is now rows 8–9 (likelihood
location + local-kernel re-decode); the remote subset is the clean fallback.

Decoder-accuracy (row 11) is downgraded: with no ground truth for the
non-local location, it shows only that the decoder isn't broken — reality
rests on the spike-driven tests (7–8). CV (row 12) needs label-stability
across model variants, not held-out LL alone.

**Immediate gate — RESOLVED:** the per-position likelihood *is* recoverable
from the saved encoding model + event marks (verified, prototyped end-to-end,
no re-decode for the recompute path). The decode now also persists
`log_likelihood` for the decode-pinned all-events read. The engine is built
(`scripts/detection/analyze_event_spike_evidence.py`, reviewed) and confirmed 5/5 animals.
