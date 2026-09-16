# PFC ridge/held-out implementation review

Date: 2026-09-16

Branch reviewed: `pfc-ridge-heldout` through `df757b3`, plus the follow-up fixed-subset fix in this working tree

Scope: scientific correctness, inference validity, failure handling, memory and runtime

## Outcome

Task A and the Task B integration are implemented. The corrected three-session smoke must be
repeated, and Task C has not run: no ridge production batch, final cohort audit, regenerated
Figure 4 or updated claim should be treated as complete yet.

The review found and fixed eight inference defects:

| defect | consequence before review | resolution |
|---|---|---|
| A finite fit with `converged=False` was scored out of fold | an iteration-cap result could contribute a plausible held-out deviance | every affected unit × fold is now `NaN`; a unit is reported only when every fold converged and scored finitely |
| Monte Carlo p-values used `b/R` | zero p-values were possible with only 40 draws | use the finite-sample `(b + 1)/(R + 1)` correction; the minimum at R = 40 is 1/41 |
| Incompletely placed control draws were retained | a thinner regressor entered the null and could make it anti-conservative | a draw is usable only when it places every requested window; fewer than 20 usable draws yields `NaN` |
| Exact-draw validation was combined with mutually disjoint control duration windows | the first corrected smoke produced 0/40 exact draws for every j16 class and for both key non-local classes in chimi, so their empirical p-values were unavailable | enumerate legal same-bout, real-event-free onset bins directly and require distinct onsets; auxiliary duration windows may overlap because duration constrains placement but never enters the onset-only GLM design; version the placement policy in provenance |
| The observed statistic still used every real event when some events had no legal within-bout match | the replacement gate startup found two genuinely unmatchable events in both j16 and chimi; their null could never reach the all-event count, so the p-value would remain unavailable (and accepting a thinner draw would change the estimand) | select one deterministic, jointly matchable subset from event timing/duration alone; compute both the observed statistic and every null draw on that fixed subset, keep the all-event held-out difference separately, and version/record the tested population in provenance |
| Held-out empirical p-values were BH-corrected across thousands of dependent units | the old zero p-values became spuriously tiny q-values; units were also treated as independent replicates | the phase-0b fraction is raw `p < .05`, explicitly descriptive; population inference remains session → animal in Phase 4 |
| Short gaps could put two runs touched by one event kernel in different folds | one onset could inform both training and held-out bins | runs whose ±250 ms supports overlap are grouped before fold assignment |
| The population loader trusted column names without checking held-out provenance | a pre-review `b/R` bundle could silently reach Figure 4 after the formula changed | held-out loads now require the ridge/blocked-deviance estimator plus `empirical_pvalue_method = plus_one` and the 20-draw floor in every used artifact's sidecar |

The unpenalised comparison used to audit prior binding is also gated on convergence and on
model-level separation. A failed maximum-likelihood fit no longer supplies a peak, covariance or
`prior_binds` call.

## Performance and memory

The restricted review shell did not expose NVIDIA device nodes, but a host-level probe found ten
NVIDIA A100 80GB PCIe devices (indices 0--9). A repository-managed claim on idle GPU 3 confirmed
that JAX used the intended single device with a 10% memory cap. The user expected eight devices;
the host inventory should be reconciled before choosing the production worker pool, but it does
not block a single-device smoke.

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
class; the 11-column output is about 58.7 MiB. Null designs are streamed one draw at a time. The
engagement model reuses the current class design and the fixed baseline block without retaining an
R-sized design cache, so the speedup does not create multi-gigabyte host growth.

The canonical K = 5, R = 40 workload is still intrinsically large: 45 observed model-fold fits plus
about 1,010 null model-fold fits per session, before the full-data fits. JAX compilation is reused
for repeated shapes (the measured CPU first call was 2.29 s versus 0.52 s steady state). On the
A100 synthetic workload, covariance added about 1.5 ms (roughly 1.2%) in a counterbalanced warm
benchmark; disabling it remains a material CPU saving and avoids an unnecessary calculation. The
production smoke must still confirm memory and end-to-end wall time on three real sessions. In
particular, every null fit currently transfers its fold's design and response through the public
NumPy-to-JAX fitter boundary; the synthetic kernel timing does not establish whether those repeated
host/device transfers dominate the canonical 1,055-fit session workload.

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
all-event and subset-matched observed differences. A fresh full gate of that implementation remains
required.

## Validation

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
- Ruff and `git diff --check`: clean.

The production batch must not launch until the corrected three-session GPU smoke, the real-session
placement/fold diagnostics and the intended eight-device worker pool are resolved. Existing
bundles without `empirical_pvalue_method = plus_one` and `min_usable_null_draws = 20` are
deliberately rejected by the skip check and population loader.
