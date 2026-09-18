# PFC ridge/held-out implementation review

Date: 2026-09-16; updated 2026-09-17 after the independent review and R = 1,000 gate

Branch reviewed: `pfc-ridge-heldout` release candidate

Scope: scientific correctness, inference validity, failure handling, memory and runtime

## Outcome

The library, the runner integration and the batch are all complete. The pre-production gates
below (the corrected three-session GPU gate and the full R = 1,000 endurance gate on the largest
510-unit session) were followed by the production batch: **71/71 sessions at `db01df0`**, launched
2026-09-16 and finished 2026-09-17 on five idle A100s (cards 3, 4, 6, 7 and 9), 13 h 52 m to
16 h 59 m of wall clock per worker and at most 15.5 GiB of host memory; every worker exited 0. The
maximum-likelihood tree it replaced is kept whole at `Processed-Data/_pre_ridge_backup_20260915`.
Non-converged fits fell from 14.8 % of shipped model fits to 0.28 % (432 of 155,952, over 48
neurons). The before/after audit is
[docs/audits/2026-09-18-pfc-ridge-heldout.md](../../../docs/audits/2026-09-18-pfc-ridge-heldout.md),
Figure 4 is regenerated from these bundles and claim 20 is requoted from that audit.

Three properties of the null were measured afterwards rather than assumed
(`scripts/qa/measure_pfc_null_geometry.py`, folded into that audit): the engagement null's draws
land on the shipped control they are entered beside on a median 11.0 % of their windows (5.4–52.9 %
across sessions), not the ~3 % assumed here; 7.0–10.0 % of the units called at raw p < .05 have a
non-positive observed held-out improvement; and 0.3–1.4 % of a class's events corpus-wide have no
legal within-bout control, with one session/class excluding 24.7 %. None of them changes the draw;
all three are terms on which the p-values should be read.

The initial review found and fixed eight inference defects:

| defect | consequence before review | resolution |
|---|---|---|
| A finite fit with `converged=False` was scored out of fold | an iteration-cap result could contribute a plausible held-out deviance | every affected unit × fold is now `NaN`; a unit is reported only when every fold converged and scored finitely |
| Monte Carlo p-values used `b/R` | zero p-values were possible with only 40 draws | use the finite-sample `(b + 1)/(R + 1)` correction; the minimum at R = 40 is 1/41 |
| Incompletely placed control draws were retained | a thinner regressor entered the null and could make it anti-conservative | a draw is usable only when it places every requested window; production now requires all requested draws per unit and records the effective count (20 remains only a shortened-diagnostic floor) |
| Exact-draw validation was combined with mutually disjoint control duration windows | the first corrected smoke produced 0/40 exact draws for every j16 class and for both key non-local classes in chimi, so their empirical p-values were unavailable | enumerate legal same-bout, real-event-free onset bins directly and require distinct onsets; auxiliary duration windows may overlap because duration constrains placement but never enters the onset-only GLM design; version the placement policy in provenance |
| The observed statistic still used every real event when some events had no legal within-bout match | the replacement gate startup found two genuinely unmatchable events in both j16 and chimi; their null could never reach the all-event count, so the p-value would remain unavailable (and accepting a thinner draw would change the estimand) | select one deterministic, jointly matchable subset from event timing/duration alone; compute both the observed statistic and every null draw on that fixed subset, keep the all-event held-out difference separately, and version/record the tested population in provenance |
| Held-out empirical p-values were BH-corrected across thousands of dependent units | the old zero p-values became spuriously tiny q-values; units were also treated as independent replicates | the phase-0b fraction is raw `p < .05`, explicitly descriptive; population inference remains session → animal in Phase 4 |
| Short gaps could put two runs touched by one event kernel in different folds | one onset could inform both training and held-out bins | runs whose ±250 ms supports overlap are grouped before fold assignment |
| The population loader trusted column names without checking held-out provenance | a pre-review `b/R` bundle could silently reach Figure 4 after the formula changed | held-out loads now require the complete canonical ridge/blocked-deviance contract in every used artifact's sidecar |

The independent release review found four further launch blockers. Figure 4 still executed with
R = 40 while its prose advertised R = 1,000; the figure now imports the runner's canonical
`DEFAULT_CONTROL_DRAWS` and a rendering regression requires the `R=1000, min p=0.001` title. The
population loader could also pool retained R = 40 bundles with new R = 1,000 bundles; it now
requires the canonical draw count, fold count, penalty, p-value method, null policy, placement
method and usable-draw requirement. Finally, requested R was not effective R: p-values could be
reported from only 20 finite fits. Production now requires all 1,000 usable draws per finite
p-value, stores the per-unit counts in the table and sidecar, and cross-validates the two on load.
The prepared-response fast path also accepted redundant same-shaped raw counts and folds, which
could silently score another dataset or split. The API now accepts either raw inputs or one prepared
response, never both, so the ambiguous identity path is structurally unavailable.

The unpenalised comparison used to audit prior binding is also gated on convergence and on
model-level separation. A failed maximum-likelihood fit no longer supplies a peak, covariance or
`prior_binds` call.

## Performance and memory

The host has ten NVIDIA A100 80GB PCIe devices (indices 0--9), as confirmed by the user. A
repository-managed claim on idle GPU 3 confirmed that JAX used the intended single device with a
10% memory cap. Production uses only cards that are idle at launch.

| benchmark | before | after | result |
|---|---:|---:|---:|
| Event-basis construction, 300,000 bins, 500 onsets, 251 lags, 11 basis columns | 0.186 s | 0.024 s | 7.7× faster |
| Fold assignment, 698,265 bins, 800 runs / 436 joined blocks | 0.451 s block rescans | 0.007 s linear fill | 64.6× faster |
| Synthetic session score, 30,000 bins, 3 units, K = 3, R = 4 | 18.59 s | 13.48 s | 27% faster |
| Steady CPU fit, 30,000 × 24 response, 12-column design | 0.572 s with covariance | 0.522 s prediction-only | covariance was about 9.7% overhead |
| A100 fit, 100,000 × 128 response, 12-column design | 2.78 s compile + first fit | about 0.13 s steady | single-GPU execution and compilation confirmed |
| Held-out prediction/deviance, 64,000 bins × 344 units, 12-column design | 0.445 s NumPy/host | 0.096 s steady A100 | 4.65× faster; only the 344 final scores return to host |

The sparse onset accumulator remains value-equivalent to the former lagged-matrix product to
`1e-12`. At 700,000 bins, the old 251-lag float64 matrix would occupy about 1.34 GiB per event
class; the 11-column output is about 58.7 MiB. Null designs are streamed in four-design
microbatches. The engagement model reuses the current class design and the fixed baseline block
without retaining an R-sized design cache, so the speedup does not create multi-gigabyte host
growth.

The canonical K = 5, R = 1,000 workload is intrinsically large: 45 observed model-fold fits plus
about 25,000 null model-fold fits per session, before the full-data fits. JAX compilation is reused
for repeated shapes (the measured CPU first call was 2.29 s versus 0.52 s steady state). On the
A100 synthetic workload, covariance added about 1.5 ms (roughly 1.2%) in a counterbalanced warm
benchmark; disabling it remains a material CPU saving and avoids an unnecessary calculation. The
prediction-only path keeps the shared response and folds device-resident, transfers only each
bounded design microbatch, and returns only scores and fit diagnostics.

Other removed work:

- held-out fits no longer compute inverse-information covariance matrices they never consume;
- each session's spikes are binned once for the batched fit, scoring context and serial summaries;
- the engagement null reuses the already drawn times and class design instead of drawing and
  convolving them again.

The fixed-subset correction does not add an observed-model refit for fully matchable classes: it
reuses the all-event held-out deviances already computed by the model ladder. Only a class that
actually excludes an unmatchable event is rescored on its tested subset (one additional K-fold fit),
and the engagement model is likewise rescored only when its tested class was reduced. The upfront
selection is four timing-only placement passes per session and does not inspect or copy spike data.

The device audit also found that an explicit `--gpu N` attempted to claim the card but ignored a
failed claim. It now refuses a card held by a sibling worker, preventing two explicitly pinned
processes from colliding. The host has ten physical A100s (indices 0--9), as the user subsequently
confirmed; the production worker pool is selected from whichever of those ten are idle at launch.

The first post-review real-session gate was deliberately retained as a conservative baseline after
the placement failure was found. It used GPUs 3/4/6 and peaked at about 1.5/2.5/4.6 GiB of device
memory. Wall/RSS were 3:41/4.16 GiB for j16, 8:15/4.98 GiB for chimi and 37:46/9.06 GiB for wilbur.
The short j16 time is not a valid full-cost estimate: all of its incomplete null draws were correctly
skipped. Wilbur exercised the full 40-draw path for all four classes; its low, bursty GPU utilisation
motivated moving held-out prediction and deviance onto JAX. A replacement three-session gate under
the versioned placement policy and device-side scoring remains required before production launch.

The replacement gate under `df757b3` was stopped during startup rather than spending another full
GPU-hour on a known-invalid estimand. Direct legal-onset enumeration still placed only 275/277 j16
baseline controls and 208/210 chimi controls (wilbur placed 217/217). Because controls no longer
exclude one another except at their onset bins, those four shortfalls are properties of the source
events' own bouts, not rejection-sampling failures. The corrected implementation now carries source
indices out of placement, freezes a matchable subset independently of spikes, and reports both the
all-event and subset-matched observed differences.

The full corrected gate at `00765b2` then passed on GPUs 3, 4 and 6. Every class produced all 40
exact draws on the same fixed event subset used by its observed statistic, and all sidecars record
`fixed_jointly_matchable_subset`, the plus-one p-value correction, K = 5, R = 40 and sigma = 3.

| session | wall | peak RSS | usable units | excluded matched-test events | exact draws |
|---|---:|---:|---:|---:|---:|
| j1620210710_02_r1 | 12:00 | 4.25 GiB | 73/74 | 1 non-SWR, 1 SWR-coupled; 2 in each parent class | 40/40, every class |
| chimi20200223_06_r3 | 13:38 | 5.06 GiB | 166/166 | 2 non-SWR; therefore 2 in non-local | 40/40, every class |
| wilbur20210329_02_r1 | 27:25 | 8.89 GiB | 342/344 | none | 40/40, every class |

Wilbur is the comparable full-workload performance case: device-side held-out scoring reduced it
from 37:46 to 27:25 (27% less wall time). At that gate GPU use remained bursty and host CPU stayed
near 83%, identifying the host-orchestrated IRLS loop and its per-iteration convergence
synchronisation as the next ceiling. The follow-up below moves that loop onto the device. The three
smoke artifacts loaded
together as 581 usable units with five aligned 201-sample modulation matrices; all five empirical
p-value columns were finite for every retained unit. The estimator audit completed successfully.

### Follow-up GPU optimization and production draw count

The held-out scorer now has a prediction-only path: spike counts and folds are transferred once,
up to four equal-width designs and all five folds are fit together, per-unit convergence stays inside one
compiled JAX loop, and a tall QR followed by an SVD of its small triangular factor replaces the
repeated tall SVD. Null designs are generated and released in microbatches, so the large design
working set does not grow with R. On the saved 64,000-bin x 128-unit benchmark, warm scoring fell
from 0.471 s/design to 0.097 s for one design and 0.048 s/design in a four-design batch. The maximum
score difference from
the reference was 1.82e-12; convergence flags and iteration counts were identical.

The end-to-end j16 gate at the same K = 5, R = 40, seeds and fixed event subsets completed in 5:23
instead of 12:00. Its null phase took 1:49 instead of 8:26 (4.6x faster), and peak host RSS was
4.66 GiB instead of 4.25 GiB. The 74 x 131 result table passed the numerical gate (absolute 1e-8,
relative 1e-10), including identical p-values and model choices, and all five modulation arrays were
byte-identical. Linear extrapolation of the measured null phase puts R = 1,000 at about 49 minutes
for this session. That cost is suitable for the ten-device production batch and gives a minimum
plus-one p-value of 1/1,001 rather than 1/41, so production now defaults to R = 1,000. The R = 40
artifacts above remain pre-production correctness/performance gates, not the final inferential
resolution.

The final endurance gate used `wilbur20210331_06_r3`, the largest profiled session (510 units,
290,322 fit bins), at the exact production K = 5, R = 1,000 and sigma = 3 settings on GPU 9. It
completed in 1:33:07 wall time; the null phase was 1:27:22. Across 4,976 one-second null-phase
samples the A100 averaged 95.05% compute utilization, reached 100%, and used at most 9,124 MiB of
its 80 GiB framebuffer. Peak host RSS was 8.40 GiB and no thermal-limit event occurred. An attempted
eight-design microbatch was rejected before this gate because a real R = 20 replay increased the
null phase from 104 s to 119 s; four remains the measured production choice.

All 506 estimable units had exactly 1,000 usable draws for all five null tests; the four
non-estimable units had zero usable draws and `NaN` p-values. Every finite p-value was exactly on
the k/1,001 grid, all four class window records reported 1,000/1,000 exact draws, and sidecar counts
matched the table exactly. Relative to the R = 20 equivalence run, all R-independent scalar and
fold results were identical, all model/fit decisions were identical, and all five modulation arrays
were byte-identical. The canonical population loader accepted the bundle and retained the same 506
estimable units.

## Validation

- Prediction-only scorer and null-microbatch gate: 132 passed, 1 slow test deselected; this includes
  exact convergence/iteration agreement, rank-deficient and silent-unit behavior, and a direct
  2/2/1 bounded-batch regression.
- Broad PFC/runner/cohort-audit gate after optimization: 354 passed, 2 slow tests deselected.
- Repository-wide non-slow gate: 2,831 tests passed and one stale generated pipeline-map check
  failed because `null_window_table` was absent from the map. Regenerating the map fixed that branch
  hygiene issue; its dedicated check then passed.
- Focused correctness/performance gate: 136 passed, including the 40-draw null calibration.
- Follow-up control/inference/runner/loader gate after the real-smoke finding: 110 passed in 8 min
  13 s; the dense-bout regression fails under the disjoint-window policy and passes only when all
  event-free onset matches are retained.
- Post-refactor integration gate: 149 passed, 1 slow test deselected.
- Full PFC and runner/loader/QA gate: 361 passed in 10 min 54 s; six pre-existing warnings.
- Repository-wide gate: 2,846 passed in 19 min 27 s. Its sole failure was a stale-module artifact
  from moving the p-value method constant while that same Python process was running; the exact
  real-session test passed in a fresh process (1 passed in 2 min 10 s).
- Final focused gates after the provenance, linear-fold and GPU-claim fixes: 42 passed and 28 passed.
- Fixed-subset runner, loader and cohort-audit regressions: 81 passed; the two direct
  placement/count-matching regressions passed separately. The full focused scientific suite ran all
  120 tests (including the slow real-session and null-calibration gates); 119 passed immediately and
  the sole failure was a stale fixture expectation that every event class excluded two events. The
  diagnostics correctly showed that the non-SWR subclass excluded zero; the expectation was fixed
  and the corrected runner test passed in the 81-test gate.
- Single-device A100 compile and steady-state fit benchmark: passed on GPU 3.
- Corrected real-data GPU gate: all three processes exited 0; all 12 class/session nulls produced
  40 exact draws; the combined population-loader and estimator-audit smokes passed.
- Ruff and `git diff --check`: clean.
- Independent-review regressions and expanded PFC gate: 257 passed, 1 slow test deselected.
- Final repository-wide non-slow gate: 2,839 passed, 27 slow tests deselected; only known warnings.
- Full R = 1,000 largest-session profile and artifact audit: passed as described above.

The pre-production gates are satisfied. The production batch may launch on whichever of the ten
cards are idle at launch. Existing bundles without `empirical_pvalue_method = plus_one`,
`control_null_event_policy = fixed_jointly_matchable_subset` and
the exact canonical R = 1,000 / K = 5 / sigma = 3 contract are deliberately rejected by the skip
check and population loader.
