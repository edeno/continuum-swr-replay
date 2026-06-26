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
| 6 | **(W2a · [M1])** Events are spike-driven by multiple tetrodes (not a no-spike / ground-process event) | Per-event spike count, MUA-z, # active tetrodes: non-SWR vs SWR vs **matched non-event immobility baseline** (falls out of the same mark extraction as #7) | Coordinated spikes are present during events — not a spike vacuum | Main reality fig · prototyped; 5/5 animals (smoke); all-events run pending |
| 7 | **(W2b + W3 · CENTERPIECE)** The decoded displacement is supported by spatially informative spikes, not just the prior | Recompute the **prior-excluded spike likelihood** per event (native `predict_clusterless_kde_log_likelihood` + saved encoding model, no re-decode; native track-distance bins; the decode now also persists `log_likelihood` to `results.nc` so the future all-events run *reads* it instead of recomputing). Metrics: **ΔLL = logL(decoded MAP) − logL(animal)** and **ΔLL per spike**, each vs a **matched non-event immobility baseline (≈0)** | Prototype + 5-animal smoke supports that spikes carry location-specific evidence above matched baseline. Non-SWR events are lower-fidelity than SWR events; argues against a purely flat/non-spatial spike explanation, but the all-events run is still pending | Main reality fig · prototype + 5/5 smoke; decode-pinned all-events run pending |
| 7b | **(W2b)** The decode is content-specific, not GPI/baseline-driven | Track-distance from event MAP to the **GPI peak** vs a random-position reference; distribution of event MAP positions over GPI(x) | In the prototype/smoke set, non-SWR MAPs are spread across the track and not closer to the GPI peak than chance (~100 cm vs ~71 cm random) → not a ground-process default | Supp · prototype + 5/5 smoke; all-events run pending |
| 7c | The matched non-event immobility baseline is the key null for spike evidence | Event ΔLL and ΔLL/spike vs matched immobile non-event windows, same event-level scoring | The matched baseline sits near zero, avoiding the muddier random-remote null where the animal's local position can itself be spike-supported | Methods note (folds into #7) · prototype + 5/5 smoke |
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
| | **Framing — what 3b actually shows.** Non-SWR content **over-represents the current arm** (near the animal but spike-supported/displaced, #19a — *not* an artifact: close events clear the reality baseline, Goal 2) and **avoids the off-current last-rewarded arm** (#19a, sparse). It is **not value-selective** (#19b). Occurrence reveals a **dwell-robust dissociation**: non-SWR events are **far less omission-gated than SWRs** (#19c, paired within-session, 94% of sessions, 5/5 animals) — less coupled to the post-reward consummatory state. There is also a **modest, non-distance-controlled** choice correlation (#19d) and **no stay/leave** signal (#19e). The strongest, cleanest 3b results are the current-arm preference, the not-value-selective negative, and the **omission-gating dissociation** (#19c); the last-rewarded avoidance is sparse and the choice correlation is uncontrolled — keep those hedged. Distance from the animal is a *graded* axis (#13), not a reason to exclude near events. | | | |
| 19a | Non-SWR content over-represents the **current arm** (near the animal but spike-supported and displaced) and **suppresses the off-current last-rewarded arm** | Arm conditional-Poisson **M5, non-SWR-only** fit (`models_swr_F`), hard arm assignment: `is_current`, `is_last_rewarded_elsewhere` (disjoint support); per-session medians | `is_current = +0.98` (178+/16−). "Non-local" is a *dynamics* label and "same arm" ≠ "at the animal" (a leaf arm runs well→junction, so content can be displaced down the current arm). These near events are **real**, not leakage: the spike-evidence engine (Goal 2) puts close non-SWR ΔLL(MAP vs animal) ≈ **0.73 above** the non-event baseline ≈ 0 (5/5 animals) — the spikes support a *displaced* position over the animal's own. The off-current reward effect is avoidance: `is_last_rewarded_elsewhere = −1.66` (p=7e-12, **sparse ~2%**) — reward-history *avoidance*, not value *selection* (#19b). Residual W3 place-field-extent leakage is settled by the **#8 local-kernel re-decode**, **not** by excluding near events | Main/Supp Fig 3 · have; #8 re-decode pending |
| 19b | Non-SWR content is **not value-selective** | Same M5 non-SWR fit: `recent_reward_rate` | **Null** (median +0.003, 97+/97−, p=0.31) — the positive reward-rate effect is **SWR-flavored**, absent in non-SWR. Confirms occupancy #14 (content at reward *sites*, not the high-value ones) | Supp · have |
| 19c | Non-SWR occurrence is **far less omission-gated than SWR occurrence** (dwell-robust dissociation) | Per-stop rate Poisson, **SWR-split outcome** (`models_rate`, `outcome=non_swr`/`swr`), log-immobility offset; **paired within-session** SWR vs non-SWR `is_after_omission` coefficient | Omission suppresses both, but SWR ~6× (RR **0.16**, 214/218−) — the expected SWR reward/quiescence gating, a positive control — vs non-SWR ~2× (RR **0.50**, 196/218−). The **paired difference** Δ(SWR−nonSWR) = **−1.15**, SWR-more-suppressed in **94% of 218 sessions, 5/5 animals**, p≈1e-32. Both outcomes share the same stops/durations/offset, so the *difference* is **dwell-robust**: non-SWR events are genuinely **less coupled to the post-reward consummatory state** than SWRs — supporting that they are not sub-threshold/reward-driven SWRs (pairs with continuum #24). Caveats: each type's *absolute* omission effect is still dwell-tinged; the slow `recent_trial_reward_rate` suppression (RR 0.68) is shared & weaker; the `is_after_switch` term (RR 1.29) is a dwell artifact (raw counts fall, 0.58 vs 0.62); per-stop CV is flat (sparse counts) so inference is on cross-session sign, not CV | Main/Supp · have (committed); dissociation panel GAP |
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
- **W3 definitive** → #8 (local-kernel re-decode).
- **W4 distance** → #9 (remote events are the clean population).

**Reality-analysis status — engine built, approved, cross-animal smoke:** the
#7 engine is formalized in `scripts/analyze_event_spike_evidence.py`
(scientific-code-reviewer approved). Likelihood recomputed from the saved
encoding model + marks (no re-decode), native track-distance bins. **Prototype
result (cross-animal smoke, 1 session/animal): ΔLL and per-spike are positive
for all event groups incl. close, and ≈0 for the matched non-event baseline,
in 5/5 animals** (ordering SWR > nonSWR-remote > nonSWR-close > baseline≈0);
MAPs are **not** at the GPI peak. **Lesson:** the *matched non-event baseline*
is the right null — the random-remote null is muddier (the local position is
itself spike-supported, so it also beats a random remote). Figures:
`paper/figures/reality_prototype/`. Spec:
`docs/superpowers/specs/2026-06-25-non-swr-event-reality-design.md`.

**Critical path / status caveats.** The entire reality core (#6, #7, #7b, #7c)
is **prototype + 5/5-smoke, not yet full-scale** — the decode-pinned all-events
run is **deferred to the re-decode** (the 30-cap recompute sweep was killed;
the decode now persists `log_likelihood` so that run becomes a *read*). Goal 2
is design-complete and 5/5-confirmed but cannot be stated at full strength
until then. **Gate:** #7 + #8 decide whether the **close** subset counts; if
#8 fails, headline claims (and the #2 denominator) restrict to the **remote**
subset (#13/#9). The #7 feasibility gate is **cleared** and the close subset
clears the W2 part in all 5 animals; the **W3-definitive gate (#8, local-kernel
re-decode) is still pending**.

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
