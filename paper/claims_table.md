# Claims → comparison → what it shows → placement

One row per claim. **Comparison** is the plot/contrast that supports it;
**What it shows** is the conclusion that comparison licenses; **Placement**
is Main figure vs Supplement + status (have / recut / GAP). Rows tagged
**[M#]** were added in the missing-claims pass; the Goal-2 (reality) block is
organized around the four artifact worries (W1–W4) and the discriminating
test for each. See "Considered and dropped" at the bottom.
Current status reflects the saved figure inventory after the figure audit:
some standalone panels exist but are stale relative to the argument.

The paper's spine is **descriptive — characterize a class of non-local events
that occur outside SWRs.** The four goals: (1) the events exist and are
common; (2) they are real, not artifacts; (3) what they are — split into
**3a** what the activity does (location, dynamics, certainty, duration), **3b**
how it relates to behavior, **3c** how it relates to mPFC; (4) content-targeted
detection captures what SWR-triggered workflows miss. The SWR contrasts
(slower, more compact) are *descriptive*, not a categorical "different kind"
claim.

| # | Claim | Comparison (plot) | What it shows | Placement |
|---|---|---|---|---|
| **Goal 1 — the events exist and are common** | | | | |
| 1 | Non-local events occur without an SWR | Example decodes, one per overlap class (posterior + LFP + ripple-band + MUA + position) | A non-SWR event can have a coherent trajectory posterior; examples motivate the phenomenon, but event reality rests on #6–#7 | Main Fig 2a · have |
| 2 | A large fraction of non-local events carry no SWR | Stacked bar: proportion of immobile non-local events by overlap class | Quantifies what SWR-triggering misses: 39,500/65,913 immobile non-local events are non-SWR (~60%), non-SWR the majority/plurality in 5/5 animals (~57–68%). Raw prevalence — the *clean* fraction is remote-only pending #8; quote post-reality | Main Fig 2b · have |
| 3 | **[M5]** The non-SWR fraction is robust to detector thresholds | Non-SWR fraction + close/remote split across a grid of non-local (0.8) and SWR-z thresholds | The prevalence claim is not an artifact of threshold tuning (gates the point estimates in #2 / #13) | Supp · GAP |
| 4 | SWR-detection and non-local-detection are separate axes | Venn / 2×2: SWR ± × non-local-state ± (events and timepoints); Fig 2b is the non-local-conditioned view; the continuous version is #24 | Many non-local events fall outside SWRs, and many SWRs lack non-local content; sets up Goal 4 | Main Fig 3 Venn · have, standardize with Fig 2b |
| **Goal 2 — the events are real** (organized by artifact worry) | | | | |
| 5 | **(W1)** Non-SWR events are not theta / movement-driven | theta-z and speed contrast across **SWR events / non-SWR events / running** | Both event types sit at low theta and low speed, unlike running — not theta sequences or movement artifacts | Supp · have (data in hand) |
| 6 | **(W2a · [M1])** Events are spike-driven by multiple tetrodes (not a no-spike / ground-process event) | Per-event spike count, MUA-z, # active tetrodes: non-SWR vs SWR vs **matched non-event immobility baseline** (falls out of the same mark extraction as #7) | Coordinated spikes are present during events — not a spike vacuum | Main reality fig · preliminary; 5/5 animals (one session each); full-dataset run pending |
| 7 | **(W2b + W3 · CENTERPIECE)** The decoded displacement is supported by spatially informative spikes, not just the prior | Recompute the **prior-excluded spike likelihood** per event (native `predict_clusterless_kde_log_likelihood` + saved encoding model, without re-running the decoder; native track-distance bins; the decode now also persists `log_likelihood` to `results.nc` so the future full-dataset run *reads* it instead of recomputing). Metrics: **ΔLL = logL(decoded MAP) − logL(animal)** and **ΔLL per spike**, each vs a **matched non-event immobility baseline (≈0)** | Initial validation (5 animals, one session each) supports that spikes carry location-specific evidence above matched baseline. Non-SWR events are lower-fidelity than SWR events; argues against a purely flat/non-spatial spike explanation, but the full-dataset run is still pending | Main reality fig · validated on one session/animal (5/5); full-dataset run pending (after the decoder is re-run) |
| 7b | **(W2b)** The decode is content-specific, not GPI/baseline-driven | Track-distance from event MAP to the **GPI peak** vs a random-position reference; distribution of event MAP positions over GPI(x) | In the initial validation set, non-SWR MAPs are spread across the track and not closer to the GPI peak than chance (~100 cm vs ~71 cm random) → not a ground-process default | Supp · validated on one session/animal (5/5); full-dataset run pending |
| 7c | The matched non-event immobility baseline is the key null for spike evidence | Event ΔLL and ΔLL/spike vs matched immobile non-event windows, same event-level scoring | The matched baseline sits near zero, avoiding the muddier random-remote null where the animal's local position can itself be spike-supported | Methods note (folds into #7) · validated on one session/animal (5/5) |
| 8 | **(W3 definitive)** Close events are not place-field-extent leakage | Re-decode a subset **with the multi-bin local kernel** (not used in current decode) — do close events absorb into Local or persist? | Model-based test that the local/non-local boundary is not mis-set too tight; decides whether the close subset counts | Supp · GAP (GPU subset) |
| 9 | **(W4)** Remote events are the clean non-local population | Remote subset (#13), shown spike-driven via #6–#7 | Distance settles the "how far is non-local" objection for remote events; close-event claims remain conditional on #8 | Main · follows from #13 + #6/#7 |
| 10 | The decoder is accurate in general and out-of-sample (NOT a positive control for the events) | Local-state decoding error + held-out (cross-validated) decode | The decoder is not broken; but there is no ground truth for the non-local location, so reality rests on #6–#7, not here | Supp · have (in-sample); GAP (held-out + immobile) |
| 11 | **(CV)** Held-out likelihood and label stability favor the 4-state model over ContFrag / reduced | Model selection by cross-validated LL **and event-label agreement** across model variants | The event-defining architecture is justified by the data, not arbitrary (label stability, not LL alone) | Supp · GAP |
| 12 | **[M4]** The detector has a low false-positive rate / is specific | Non-local incidence on shuffled/thinned spikes and in no-replay-expected windows | Events are not spurious detector firing (non-local-during-movement is treated as real theta, NOT the FP control). Also the **specificity counterpart to the sensitivity claim #21** | Supp · GAP |
| **Goal 3 — what these events are** (characterization) | | | | |
| **Goal 3a — what the non-local activity does** (location, dynamics, certainty, duration) | | | | |
| *Framing — one contrast:* non-SWR events are **focal, *stationary* point-representations of reward-well locations**; SWR events are **extended, *moving* trajectory replays along paths**. Every 3a row is a facet of this single dissociation: location (wells vs paths, #13/#14); **movement/stationarity** (non-SWR pause-fraction ≈ 1.0 / near-zero displacement vs SWR moving — *the through-line; data in the trajectory suite, not yet its own row*); extent (compact ~8 cm vs extended ~43 cm, #16); speed (slow vs fast ×4.9, #15); certainty (sharp/concentrated, peak 0.93/HPD 33 cm vs diffuse, #17); duration (short ~52 ms vs ~88 ms, #18). Duration-controlled and 5/5 within animals where tested. | | | | |
| 13 | **[M2]** Non-local events split into close and remote distance regimes | Distribution of decoded distance-from-animal, split by SWR status, with a close/remote boundary set by place-field width | Where they replay relative to the animal: two regimes; the remote set is the clean non-local population and fixes the headline denominator (#2). Current plot shows the regimes but not the boundary or SWR split cleanly | Main Fig 3 distance_from_animal · raw plot exists; recut needed |
| 14 | Non-SWR events represent reward-well *locations* (predominantly remote wells); SWR replays *paths* between wells | Content occupancy, **both axes track distance**: content→animal (`ahead_behind`) vs content→nearest reward-well (track-graph `get_distances_to_interior_bins`); quadrant breakdown (far >20 cm from animal × at-well <20 cm), all 218 sessions, per-animal | Where they replay: non-SWR content is **well-dominated** (72% at wells = ~42% *remote* reward well + ~30% own well; 27% paths); SWR is **path-dominated** (~76% paths, ~17% at wells). The **~42% remote reward well is the clean, artifact-immune claim** (far from animal → cannot be field-extent leakage); content is **not** the animal's position (non-local) and is **not** value-selective (spread across wells, current≈last-rewarded collinear — see Supp `predictor_collapse`). The **~30% own-well (close subset) is conditional on #8**. 5/5 animals | Main Fig 3 (content_occupancy) · have (track-distance, all sessions); own-well gated on #8 |
| 15 | Non-SWR replay is slower than SWR replay | Replay speed by SWR status, duration-controlled, within-animal (GEE ×4.91, 5/5) | A dynamics property: slower — *consistent with fewer participating cells* and still spike-driven (#6); a descriptive contrast with SWR, not firing-rate-independence | Main Fig 3c · have; annotate duration control + 5/5 |
| 16 | Non-SWR replay is spatially compact | Spatial extent / path length by SWR status, duration-controlled, within-animal | A dynamics/size property; the clearest descriptive contrast with SWR (with speed, #15); `crosses_trunk`/`n_unique_arms` are redundant and should be dropped | Promote to Main Fig 3 · ECDF exists; main-panel recut needed |
| 16b | Non-SWR replay is **stationary** — the content barely moves; SWR sweeps | pause-fraction, max step-distance, #reversals by SWR status, duration-controlled, within-animal | **The dynamic through-line:** non-SWR content is near-stationary — pause-fraction **1.0** vs 0.93, max step **2 cm** vs 12 cm, **2** reversals vs 7 (all 5/5, duration-controlled, GEE p<1e-100). It is a fixed focal point, not a trajectory — this is what unifies the focal/reward-site picture (with extent #16, speed #15) and explains the sustained own-well events (#13/#14). (`step_distance_median` is 0/0 — too sparse to use) | Main Fig 3 · have (pause_fraction, step_distance_max, n_reversals); panel GAP |
| 17 | Non-SWR replay has a sharper, more spatially-concentrated posterior than SWR | Posterior peak / HPD coverage / multimodality by SWR status, duration-controlled, within-animal | A certainty/spread property: non-SWR posteriors are more focal (higher peak, smaller HPD) — *consistent with smaller, fewer-cell events* (note the size/firing-rate relationship); multimodality difference is weak (3/5, not robust) | Supp · analysis exists; panel GAP |
| 18 | Non-SWR events are shorter than SWR events | Event duration by SWR status, within-animal | A basic descriptive property (non-SWR ~52 ms vs SWR ~88 ms median); duration is also the key confound controlled in #15–#17 | Supp · data in hand; panel GAP |
| **Goal 3b — how the non-local activity relates to behavior** | | | | |
| | **Framing — what 3b actually shows.** Non-SWR content **over-represents the current arm** (near the animal but spike-supported/displaced, #19a — *not* an artifact: close events clear the reality baseline, Goal 2) and **avoids the off-current last-rewarded arm** (#19a, sparse). It is **not value-selective** (#19b). Occurrence reveals a **dwell-robust dissociation with a specific locus**: a lagged reward kernel splits the effect into an **immediate/consummatory boost** (lag-1) that **dissociates** — non-SWR events are far less reward-boosted than SWRs (#19c, paired non-local split, 94%, 5/5 animals) — and a slower **recent-reward (RPE) suppression** (lags 2–10) that is **shared** across SWR and non-SWR. So non-SWR events are a **less consummatory/dopamine-gated mode** but obey the same RPE baseline. There is also a **modest, non-distance-controlled** choice correlation (#19d) and **no stay/leave** signal (#19e). The strongest, cleanest 3b results are the current-arm preference, the not-value-selective negative, and the **consummatory-gating dissociation** (#19c); the last-rewarded avoidance is sparse and the choice correlation is uncontrolled — keep those hedged. Distance from the animal is a *graded* axis (#13), not a reason to exclude near events. | | | |
| 19a | Non-SWR content over-represents the **current arm** (near the animal but spike-supported and displaced) and **suppresses the off-current last-rewarded arm** | Arm conditional-Poisson **M5, non-SWR-only** fit (non-SWR-only fit), hard arm assignment: `is_current`, `is_last_rewarded_elsewhere` (disjoint support); per-session medians | `is_current = +0.98` (178+/16−). "Non-local" is a *dynamics* label and "same arm" ≠ "at the animal" (a leaf arm runs well→junction, so content can be displaced down the current arm). These near events are **real**, not leakage: the spike-evidence engine (Goal 2) puts close non-SWR ΔLL(MAP vs animal) ≈ **0.73 above** the non-event baseline ≈ 0 (5/5 animals) — the spikes support a *displaced* position over the animal's own. The off-current reward effect is avoidance: `is_last_rewarded_elsewhere = −1.66` (p=7e-12, **sparse ~2%**) — reward-history *avoidance*, not value *selection* (#19b). Residual W3 place-field-extent leakage is settled by the **#8 local-kernel re-run of the decoder**, **not** by excluding near events | Main/Supp Fig 3 · have; #8 re-run of the decoder pending |
| 19b | Non-SWR content is **not value-selective** | Same M5 non-SWR fit: `recent_reward_rate` | **Null** (median +0.003, 97+/97−, p=0.31) — the positive reward-rate effect is **SWR-flavored**, absent in non-SWR. Confirms occupancy #14 (content at reward *sites*, not the high-value ones) | Supp · have |
| 19c | Non-SWR non-local occurrence is **less reward-gated than SWRs — specifically in the immediate/consummatory component**, while the slower reward-memory baseline is shared | Per-stop rate Poisson (`models_per_stop_rate`), log-immobility offset, **four outcomes** (detected SWRs `n_SWR`, non-local, non-local-with-SWR, non-local-no-SWR); models **R2** (`is_after_omission` + `recent_trial_reward_rate`) and the **lagged reward kernel Rlag** (`reward_lag1..10`); per-session medians + **paired within-session** | **Positive control passes:** detected SWRs reproduce the canonical reward-gating — omission **RR 0.36**, recent-reward **RR 0.62** (the RPE-like sign of Kleinman & Foster) — replicating Ambrose 2016 / Sosa 2020 with their own Poisson-events/sec estimator. **Two-component kernel (Rlag, 217 sessions):** (i) a strong **lag-1 immediate-reward boost** that **dissociates** — gradient non-local-w/-SWR **12.6×** > detected-SWR **3.1×** > **non-SWR non-local 2.35×** (concordant with the paired omission dissociation on the non-local split: less-gated in **94%, 5/5 animals**, p=1.5e-32) = the **consummatory/dopamine-coupled** drive (Sosa & Giocomo 2021), to which non-SWR events are least coupled; (ii) a distributed **lags-2–10 suppression** cumulating to **~0.6**, **shared** across all populations = the **RPE / relative-reward baseline** (Kleinman & Foster 2024; Ambrose adaptive coding). So non-SWR events are a **less consummatory/dopamine-gated reactivation mode** but obey the **same recent-reward (RPE) baseline** as ripples — breaking from canonical SWR reward-gating only in the immediate component (pairs with #24; no direct literature precedent for non-SWR reward-dependence). **Time-resolved confirmation (per-bin Poisson, within-stop 4/15 ms bins):** controlling for dwell, the within-stop quiescence time course (spline), reward history, **and** event clustering, the immediate-reward dissociation holds — lag-1 RR **1.42 (SWR) / 3.10 (NL-w-SWR) / 1.08 (non-SWR)**, p=6.5e-08, 4/5 animals; the time course shows SWR-coupled events **sharply peaking ~3 s post-reward then decaying** while non-SWR events are **flatter/sustained**. Events are **non-Poisson** — refractory <60 ms then bursty ~100 ms (SWR most bursty). **Pooling all non-local masks the dissociation** (pooled reward RR 1.41 ≈ SWR, driven by the SWR-coupled subset) → the split is required. `is_after_switch` is a dwell artifact. **Duration is a behavioral variable, not just exposure**: the per-second analysis isolates the neural effect; total-dose / compositional / departure-hazard views are a noted next layer. Figures: `rate_timecourse_vs_pump`, `reward_kernel_per_bin_rate`, `selfhistory_autocorr`. Both model families documented in [`docs/rate_models.md`](../docs/rate_models.md). | Main/Supp · have (control + kernel + time-resolved + autocorr) |
| 19d | Non-SWR content **modestly correlates with the upcoming choice** (non-SWR-specific) | Conditional-logit choice model, SWR-split `replay_frac` over a behavioral baseline (`event_source_comparison`) | `replay_frac_non_swr = +0.34` (p=0.019, 110+/73−) above behaviour; `replay_frac_swr` **null** (−0.14, p=0.086). **Caveat:** model drops `dist_to_arm` (quasi-separation) → **not distance-controlled**; report as a correlation, may partly reflect current/nearest-arm bias | Supp · have |
| 19e | Non-SWR content does **not** predict stay/leave | Per-stop logistic + pooled GEE, `is_stay ~ is_after_omission × frac_on_<role>`, all strata (`stage6_is_stay`) | **Null everywhere** — no role survives FDR in any stratum. Content does not gate the stay/leave decision after a reward outcome | Supp · have (negative) |
| **Goal 3c — how the non-local activity relates to mPFC** | | | | |
| 20 | Non-SWR replay significantly modulates mPFC activity | mPFC GLM modulation for non-SWR replay + matched non-event immobile control; SWR comparison shown as reference, equivalence tested only if powered | Cross-area relationship: Fig 4 supports mPFC *engagement* by non-SWR replay, not equivalence to SWR replay. Makes the events consequential/relevant | Main Fig 4 · have for engagement; control/equivalence recut pending |
| **Goal 4 — content-targeted detection captures events SWR-triggered workflows miss** | | | | |
| 21 | We detect non-local events that do not overlap any SWR | The non-SWR population (#2, #13) | Sensitivity: replay-like events invisible to SWR-triggered detection (quote only post-gate) | Main · have |
| 22 | Content-based event boundaries differ from SWR boundaries | NL vs SWR start/end offsets | Overlapping non-local events are usually close to SWR boundaries (median start +6 ms, median end −6 ms) but have long tails. Supports "differ," not "more accurate" without an external yardstick | Main Fig 2d · have; do not claim better boundaries yet |
| 23 | Content-based detection is specific to trajectory content; LFP detection is content-agnostic | Fraction of SWRs with local (non-replay) MAP content (`swr_not_nonlocal`, Venn) | Our method targets the representation; SWR detection does not (drop "over-counts/false-positive" — local SWRs are biology). Pairs with the detector specificity check #12 | Main (Venn / gallery) · partly |
| 24 | Detection spans the ripple-power continuum | **SWR-z density, Local vs Non-Local immobile timepoints (Fig 3b)** — mark the SWR-detection threshold and quantify the below-threshold non-local mass | Non-local content spans the SWR-power range; many events sit **below the SWR threshold** (= what SWR-triggering misses), with a right tail of ripple-coincident events — a continuum, not independence. Bridges Goal 1 (#4) → Goal 4 (#21) | Main Fig 3b · have; add threshold line & below-threshold % |
| 25 | **(candidate)** Content-targeted, not LFP-proxy-triggered | Conceptual + #21–24 | Decodes the representation directly, so it captures replay when the LFP marker is weak/absent | framing · candidate |

---

**Narrative order:** exist + common (1–4) → **real**, by artifact worry (5–12)
→ **what these events are** (13–20: 3a *what it does* #13–18, 3b *behavior*
#19, 3c *mPFC* #20) → method sensitivity/content-targeting (21–25). Reality
precedes characterization (you earn the right to describe by first showing the
events aren't artifacts). The close/remote distinction (#13) is *used* in the
reality block (#9, the gate) and *fully characterized* in Goal 3a — introduce
it where first needed.

**The reality logic (Goal 2):** four worries, their tests.
- **W1 theta/movement** → #5 (theta/speed contrast vs running).
- **W2a no spikes** → #6 (spike count / MUA / #tetrodes vs baseline).
- **W2b non-spatial spikes + W3 field extent** → #7, the **centerpiece**:
  the prior-excluded *likelihood ratio* `logL(MAP) − logL(animal)` (and per
  spike) must exceed the matched non-event baseline. + #7b (not GPI-driven)
  + #7c (matched baseline as the key null).
- **W3 definitive** → #8 (a local-kernel re-run of the decoder).
- **W4 distance** → #9 (remote events are the clean population).

**Reality-analysis status — built, code-reviewed, validated across animals.**
The #7 spike-likelihood analysis recomputes the prior-excluded likelihood from
the saved encoding model and spikes (without re-running the decoder), over
native track-distance bins. **Initial result (one session per animal, all 5):
ΔLL and per-spike ΔLL are positive for every event group including the
near-animal (close) subset, and ≈0 for the matched non-event baseline, in 5/5
animals** (ordering SWR > non-SWR-remote > non-SWR-close > baseline≈0); event
MAP positions are **not** at the ground-process-intensity peak. **Lesson:** the
*matched non-event immobility baseline* is the right null — a random-remote-
position null is muddier, because the animal's own position is itself
spike-supported and so also beats a random remote bin.

**Status caveats.** The reality core (#6, #7, #7b, #7c) is validated on one
session per animal but **not yet run on all events** — the full-dataset version
is deferred until the decoder is re-run (the re-run now saves the per-timebin
likelihood, so the full analysis becomes a direct read rather than a per-event
recompute). Goal 2 is design-complete and confirmed in 5/5 animals but cannot
be stated at full strength until then. **Open question:** #7 + #8 decide whether
the **close** (near-animal) subset counts; if #8 fails, the headline claims (and
the #2 denominator) restrict to the **remote** subset (#13/#9). The feasibility
test (#7) is cleared and the close subset passes the spike-evidence test in all
5 animals; the **definitive place-field-extent test (#8, a local-kernel re-run
of the decoder) is still pending**.

**Main vs Supplement:** Main = the spike-driven reality core (#6, #7), the
distance/content description (#13/#14), the dynamics (#15/#16), mPFC engagement
(#20), and the method sensitivity/content-targeting payoff plus the
ripple-continuum bridge (#24). Supplement = controls and robustness (#5, #8,
#10, #11, #12, threshold sweep #3), the certainty/duration properties
(#17/#18), content-collapse (#14-supp), and the behavior axis (#19a–#19e:
content anchoring/suppression, occurrence-vs-reward, modest choice correlation).

---

## Considered and dropped (unsupported)

- **"Non-SWR events are not sub-threshold SWRs."** Contradicted by the
  ripple-power bump at non-local onset. Reframed as continuum claim #24.
- **"SWR and non-SWR events are recruited differently by behavior."** Not
  shown as a *gross* recruitment difference; the peri-switch "flat" non-SWR
  profile is a null. Superseded by the per-effect 3b decomposition (#19a–#19e),
  where non-SWR and SWR *do* differ on specific axes (switch-rate sign, choice
  correlation) but share others (omission/reward suppression).
- **"Replay content predicts upcoming choice."** Largely *no*; previously
  invalidated by task autocorrelation. The one surviving thread is the **modest,
  non-distance-controlled non-SWR `replay_frac` correlation (#19d)** — reported
  as a correlation, not clean prediction. **"Different function" /
  "different kind" is not claimed** — the events are *characterized* (Goal 3),
  and the SWR contrasts (slower, more compact) are descriptive, not
  categorical.
- **Tetrode-holdout & interneuron-exclusion (reality controls).** Folded into
  the likelihood test (#7): a peaked prior-excluded likelihood argues against
  a purely flat/non-spatial spike explanation without requiring sorting or
  tetrode holdout as a main-text control.
