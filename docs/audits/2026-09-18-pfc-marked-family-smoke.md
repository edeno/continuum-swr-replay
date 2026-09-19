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

Device memory at full utilisation: 34.1 GB on both larger sessions, 20.3 GB on
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
  mode. Every PFC runner shares that behaviour; see follow-ups.
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
base block. About a third of same-class events have a neighbour within
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

## Follow-ups

1. Empirical-Bayes or width-scaled σ for the family designs, then a repeat of
   the isolation table.
2. `setup_logger` should append, not truncate, so a skipped rerun does not
   erase the fit log.
3. Run the 71-session corpus only after (1) is decided; the runner and
   provenance are ready as they stand.
