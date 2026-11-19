# mPFC marked-model family: three-session GPU smoke (2026-09-18)

Branch `feature/pfc-marked-event-kernels` at `df884f7` plus this audit.
Runner `scripts/pfc/run_pfc_marked_model_family.py`, defaults (σ = 3, K = 5,
at-well cohort, nine core models; the small session also ran the three
interaction models). One tmux job per idle A100, output under
`Processed-Data/pfc_marked_model_family_smoke_20260918/`, logs under
`logs/pfc_marked_family_smoke_20260918/`. Sessions were chosen by unit count
from the canonical Figure 4 bundles: the smallest, a median one, the largest.

## Outcome

| session | units | models run | bundles | ineligible | wall time | exit |
|---|---|---|---|---|---|---|
| j1620210710 02_r1 | 74 | 9 core + 3 interaction | 11 | 1 (`novelty_content`) | 17 min | 0 |
| senor20201101 08_r4 | 276 | 9 core | 9 | 0 | 28.5 min | 0 |
| wilbur20210331 10_r5 | 511 | 9 core | 8 | 1 (`switch_context`) | 39 min | 0 |

Device memory at full utilization: 34.1 GB on both larger sessions, 20.3 GB on
the smallest, on 80 GB cards. Per-model wall time on the median session:
spatial role 209 s, belief content 288 s, reward memory 277 s, novelty 176 s,
trajectory geometry 303 s, within-stop timing 86 s, switch context 91 s,
reward context 92 s, graded coupling 184 s. On the largest session the
wide-tier models took 410 s each and the compact ones 126–283 s. Extrapolated
to the 71-session mPFC corpus at these rates, the nine core models are about
25 GPU-hours, so five idle cards clear it in an evening.

What the run exercised, and how it behaved:

- **Ineligibility markers.** `novelty_content` on the small session and
  `switch_context` on the largest had a target with no variation among
  SWR-coupled events. Each wrote a sidecar-only marker with the reason and
  model name, and the runner continued. `reward_context`, which carries the
  switch mark only as an adjustment, fit with a zero block on the largest
  session, as the design doc specifies.
- **Convergence gate.** Three of 511 full-data unit fits on the largest
  session did not converge in every model, the same three units each time.
  The warning was logged, the fraction is far below the 10 % gate, and those
  units carry NaN curves in every bundle.
- **NaN accounting.** On the small session one unit has a NaN held-out delta
  in every comparison (a fold did not converge) while its full-data curves are
  finite; the two populations are reported separately, as intended.
- **Resumability.** Rerunning the finished small session's exact command
  exited in 9 s after finding all twelve bundles current, without reloading
  the session. A rerun does, however, truncate that session's log file to the
  single "skipping" line: the shared `setup_logger` opens the file in write
  mode. Every PFC runner shares that behavior; see follow-ups.
- **Held-out deltas.** For the spatial-role model on the small session the
  per-unit held-out deviance change is negative for most units (median about
  −10, 73 units) with a right tail to about +11. A 22-parameter block fit to
  no effect should cost held-out deviance, so this is the expected shape.

## The typical kernel is not the Figure 4 kernel

The family's "typical event" curve for each class was compared with the
canonical Figure 4 curve for the same session, units, and lag axis (the
canonical curve is the joint two-class ridge model's per-class kernel). On the
small session the per-unit correlation is high (median 0.94) but the peaks
are systematically larger. An isolation refit on the same session locates the
difference:

| design (σ = 3) | width | peak ratio to canonical, median [IQR] | max abs difference, median |
|---|---|---|---|
| two class kernels only | 23 | 1.000 | 0.0000 (bit-identical) |
| plus log-duration kernels | 45 | 1.15 (no-SWR), 1.05 (SWR) | 0.09, 0.16 |
| full `spatial_role` | 155 | 1.23 [1.00, 1.60], 1.26 [0.99, 1.61] | 0.46, 0.57 |
| full `spatial_role`, marks permuted within class | 155 | 1.22 [0.96, 1.57], 1.26 [0.99, 1.58] | 0.36, 0.63 |

The base-only refit reproduces the canonical bundle exactly, so the fit mask,
event sets, spike counts, and estimator agree end to end. The inflation is
unchanged when every mark is permuted within class, so it is not content: it
is what adding 132 near-collinear columns under the σ = 3 prior does to the
base block. About a third of same-class events have a neighbor within
0.5 s, so mark columns overlap base columns, and blocks fit to noise let the
base block escape part of its shrinkage. With σ = 1 the full design's base
peaks are within a few percent of the base-only σ = 1 fit (ratios 0.89 vs
0.84 and 0.78 vs 0.77 against canonical), so the effect scales with how loose
the prior is relative to the design width.

Consequences:

- A family bundle's `typical` curve must not be quoted as, or compared with,
  the Figure 4 modulation. It is the same estimand only in the base-only
  design.
- The target slope curves are estimated in the same regime, so their
  amplitudes are read only alongside the held-out delta, never alone; this is
  what the design doc already requires.
- A width-aware prior (empirical-Bayes σ per design, or a σ that scales with
  block count) is the natural fix and is left as a follow-up; nothing here
  changes a shipped number, since no production family bundle exists.

## Choosing the marked blocks' prior

At σ = 3 the held-out deltas explode on many units: |Δ| > 1000 for 12 of 74
units in `belief_content` on the small session, 58 of 276 in
`trajectory_geometry` on the median one, and 177 of 511 in `spatial_role` on
the largest, with single-unit values up to 10^7. Those smoke bundles are kept
only as evidence, under `Processed-Data/pfc_marked_model_family_smoke_20260918_sigma3_superseded/`.

The marked blocks were then given their own prior σ_mark with the base
kernels held at σ = 3, and σ_mark was scanned on the three sessions for
`spatial_role` and `belief_content` (fold seed 7, K = 5). Total held-out
deviance of the full model, and the median ratio of the base-kernel peak to
the canonical Figure 4 peak:

| σ_mark | j16 spatial (dev, ratio) | senor spatial | wilbur spatial | j16 belief | senor belief | wilbur belief |
|---|---|---|---|---|---|---|
| base only | 1,354,660 | 2,969,793 | — | 1,354,660 | 2,969,793 | — |
| 0.15 | 1,354,464 | 2,970,052 | — | 1,354,580 | 2,970,277 | — |
| 0.25 | 1,354,919 | 2,971,565 | — | 1,355,142 | 2,972,505 | — |
| 0.35 | 1,355,544 | 2,973,588 | — | 1,355,897 | 2,975,460 | — |
| 0.5 | 1,356,531 (1.02) | 2,976,912 (1.02) | 9,411,294 (1.01) | 1,357,111 (1.03) | 2,980,331 (1.03) | 9,411,810 (1.01) |
| 1.0 | 1,359,397 (1.09) | 2,987,368 (1.10) | 9,431,062 (1.05) | 1,360,838 (1.11) | 2,996,364 (1.17) | 9,432,545 (1.09) |
| 2.0 | 1,363,251 (1.17) | 3,003,916 (1.27) | 9,503,146 (1.20) | 1,435,174 (1.22) | 3,144,697 (1.39) | 9,521,379 (1.27) |
| 3.0 | 1,365,801 (1.23) | 3,019,412 (1.37) | 9,880,102 (1.31) | 5.2 × 10^10 (1.27) | 9.0 × 10^7 (1.55) | 5.9 × 10^7 (1.39) |

Two things follow. The marks carry no out-of-fold predictive value in
aggregate on these sessions: every marked model is at best level with the
base-only model and total deviance improves monotonically as σ_mark shrinks,
so there is no interior optimum and the per-unit deltas will be mostly
negative with a right tail. And σ = 3 was simply too loose for the width.
`DEFAULT_MARK_PENALTY_SIGMA = 0.5` is therefore a stated operating point:
it restores the canonical base kernels within 2 % on all three sessions,
removes the blow-ups, and still admits a per-SD modulation of roughly ×0.7
to ×1.4 (implied per-lag prior SD 0.35 log-rate). It is exposed as
`--mark-penalty-sigma` and recorded in the fit contract, so a corpus at
another value is a separate, non-skippable set of bundles.

## Follow-ups

1. A per-unit empirical-Bayes σ_mark would replace the fixed operating point;
   the sensitivity table above is the baseline for it.
2. `setup_logger` now appends (changed with this audit), so a skipped rerun
   no longer erases the fit log.
3. The 71-session corpus runs at σ_mark = 0.5 with the nine core models and
   the three interactions.
