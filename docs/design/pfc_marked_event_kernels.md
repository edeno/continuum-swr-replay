# Content-marked mPFC replay kernels

## Question and estimand

The existing mPFC temporal GLM asks how firing changes around the average
replay event. This extension asks whether that temporal response changes with
the decoded content of each event.

The primary mark is `current_arm_fraction`: the fraction of an event's
hard-decoded non-local bins assigned to the arm occupied by the animal during
the stop. A positive marked kernel means that the response becomes more
positive (or less negative) as replay allocates more content to the current
arm. A negative marked kernel means that the response becomes stronger as
content shifts away from the current arm. This is a current-arm allocation
measure, not a direct spatial-distance measure.

`current_arm_fraction_posterior`, the corresponding fraction of posterior arm
mass, is available as a sensitivity analysis.

## Model

For each event class—non-SWR replay and SWR-coupled replay—the design contains:

1. An ordinary cubic-spline event kernel.
2. A duration-marked kernel, scaled by log event duration, as a nuisance term.
3. A current-arm-content-marked kernel.

Every mark is centered and divided by its population SD separately within the
session and replay class (`ddof=0`). The intercept plus all six kernels are fit
jointly. Consequently, the ordinary kernel is the response to an event with
typical content and typical duration in that class, while the content kernel
is the lag-by-lag change in log firing rate per within-session/class SD of
current-arm content.

Since bundle contract 2 the design also carries unpenalised per-stop baseline
columns after the intercept -- a well indicator per kept well other than the
reference well, `rewarded_outcome` and `outcome_unknown`
(`marked_glm.build_baseline_columns`; `--no-baseline` drops them) -- so the
ordinary kernel is the response at the reference well after an unrewarded,
known outcome, and stationary rate differences between stops are not read as
event-locked effects.

The fit uses the same 500 Hz axis, ±250 ms spline support, ±200 ms reporting
window, at-well stop cohort, ridge-Poisson estimator, and support-separated
blocked folds as the canonical PFC analysis. The session is assembled by the
canonical PFC assembly (`analysis/pfc/session.py`), so the fit mask (with the
analysed-time rule: no fitted bin's ±250 ms basis window reaches excluded
time) and the replay-class onsets are exactly the ones Figure 4 is fit on.

## Nested held-out comparisons

There is one shared full model containing both content kernels. To evaluate a
class, its reduced model drops only that class's content kernel and retains:

- both ordinary replay kernels;
- both duration kernels; and
- the other replay class's content kernel.

The reported held-out improvement is reduced deviance minus full deviance, so
a positive number means the class-specific content kernel improved prediction
of spikes that were not used for fitting. Both descriptive content kernels
come from the same joint full-data fit.

## Label-randomization reference

The content mark is permuted among events of the same class and session.
Onsets, durations, class membership, spike counts, folds, and the other
class's content marks do not move. Each shuffled full model is refit and
rescored on the same folds. The plus-one tail fraction and the number of
finite shuffled fits are saved for audit.

This is an unrestricted label-randomization reference, not a calibrated
conditional p-value. Free permutation assumes that content labels are
exchangeable over the nuisance and contextual variables that stay attached to
each event. That strong assumption is not generally satisfied here: content
is associated to varying degrees with duration, occupied arm, stop context,
and session time. In the current canonical table, content/log-duration
correlation has median `-0.129` across 436 session × class groups (138 have
`|r| > 0.2`), and content/session-time correlation exceeds `|r| = 0.2` in 71
groups. Occupied-arm identity explains a median 6.8% of content-mark variance.
These are computed by `scripts/pfc/measure_pfc_mark_context.py` over the
at-well events (event `occupancy_state` of `at_well`, the events whose content
mark the kernels fit) of all canonical sessions, not only the mPFC
sessions; "session time" is the event midpoint. The numbers above were measured
2026-09-18 on the 218-session tables of that date.
The reference therefore describes where the observed held-out gain falls
relative to freely reassigned labels; it does not test a context-independent
content effect.

The tail fraction never filters neurons, events, curves, denominators, or
population summaries. The primary outputs are the continuous content-slope
curves and held-out deviance improvements for all units with estimable fits.
A future conditional test would require a separately validated restricted or
conditional-randomization scheme and correlated-nuisance type-I simulations.

Production requires every requested shuffle to be finite before reporting the
tail fraction. The default is 100 shuffles, giving resolution 1/101.
Users can request 1,000 with `--n-shuffles 1000`, but the profile below shows
why that is a costly sensitivity run rather than the default for this wider
model.

## Undefined current arm

The canonical well-anchored stop cohort includes both `at_well` and `mixed`
stops. Some events in mixed stops have no defined occupancy arm, so a
current-arm content value does not exist. Across the current tables there are
60,308 events in well-anchored stops, of which 59,699 have a current-arm mark
and 609 do not. The 609 unavailable events are principally event-level
`in_transit` or `mixed` states within mixed stops.

These events are not dropped. They remain in the ordinary replay kernel and
the duration nuisance kernel. Their content-mark regressor is zero, and mark
permutations operate only among events with a defined mark. Every sidecar
records total and marked event counts by class. All 436 session × replay-class
groups currently have at least 33 marked events and nonzero mark SD, so every
group can identify a content slope. A class is required to carry at least
`MIN_MARKED_EVENTS_PER_CLASS` = 10 marked events (defined in `marked_glm.py`);
below that the session refuses to fit rather than reporting a slope no group
size can support.

## Correctness and performance

The implementation keeps the legacy PFC path unchanged. Regression tests
cover:

- exact equality between unit-weighted and ordinary event regressors;
- weighted convolution against a slow event-by-event reference (`1e-12`);
- one-to-one event/content alignment, timing and SWR-class agreement;
- missing-current-arm retention without moving missingness during shuffles;
- exact nesting of the reduced designs inside the joint full design;
- batched sparse permutation blocks against the scalar construction
  (`1e-12`); and
- synthetic recovery of a planted content kernel and positive held-out gain.

The design is built once. Reduced models delete columns from that matrix rather
than recomputing convolutions. Permuted mark blocks are accumulated in batches
directly onto fitted rows, avoiding a full-session convolution for every draw.
Counts and the prepared held-out response remain resident
on the accelerator while equal-width designs are scored in stable batches.

Every performance figure in the rest of this section -- the stage timings, the
1.28× largest-session ratio, and the rejected warm start -- is a **one-off
measurement taken during the development session**, recorded for capacity
planning. None of it is reproducible from the repository: no
`pfc_marked_kernels` bundle has been written yet, so no artifact, sidecar or
log on disk backs these numbers and nothing re-derives them. Treat them as an
order-of-magnitude note, not as a result.

A real-session gate on an NVIDIA A100 80 GB (Wilbur 2021-03-30, epoch 06_r3) used
430 units, 269,662 fitted bins, 125 replay events, 67 full-model columns, and
five folds:

- session preparation: about 108 s;
- marked design construction: about 0.4 s;
- observed held-out scoring: about 65 s;
- full-data joint fit: about 7 s;
- total zero-shuffle run: 3 min 17 s, peak host RSS about 8.4 GB;
- total 20-shuffle run: 9 min 33 s, with GPU utilization sampled at 100% and
about 34 GB device memory.

The largest observed session is about 1.28 times the profiled bins × units and
should receive its own memory gate before a full-corpus production run. Ridge
scale is exposed as a runner option, but population conclusions should also be
checked for sensitivity to that choice before manuscript use.

The incremental 20-draw cost projects to roughly 31 minutes for 100 draws and
5.2 hours for 1,000 on this relatively large session. Session sizes vary, so
these are capacity-planning figures rather than universal runtimes. The main
remaining bottleneck is exact refitting: every permutation requires five
penalized Poisson fits over the full fitted time axis, and cost grows roughly
with time bins × units × folds × shuffles × design-rank².

A reduced-model warm start was profiled but rejected. It improved wall time by
about 22%, yet individual null deviances differed from the cold-start reference
by up to `1.2e-7`, outside the project's predeclared `1e-8` held-out comparison
tolerance. Observed statistics and all empirical tail decisions matched, but
the stricter numerical contract takes precedence.

## Runner and artifacts

Run one process per GPU with disjoint session files:

```bash
python scripts/pfc/run_pfc_marked_kernels.py \
  --gpu 0 --sessions worker0.sessions --n-shuffles 100
```

One bundle per session × mark (`current_arm_fraction`, or
`current_arm_fraction_posterior` as a sensitivity), under
`Processed-Data/pfc_marked_kernels/{epoch_identifier}_{mark}_marked_kernels.*`. **No bundle
exists on disk yet.** Each bundle contains:

- a per-neuron statistics pickle (`.pkl`): the held-out deviance of the joint full model
  and of each class's reduced model, the class's held-out improvement, the label-shuffle
  tail fraction and usable-shuffle count, and the convergence flags;
- typical-event and content-slope arrays for both replay classes
  (`.{class}.typical.npy`, `.{class}.mark_slope.npy`, class `non_local_SWR` or
  `non_local_no_SWR`);
- the reported lag axis (`.lag_time.npy`);
- compressed per-class label-shuffle reference improvements
  (`.mark_label_shuffle_reference.npz`); and
- a provenance sidecar (`.provenance.json`) containing input hashes, definitions, scaling
  references, event counts, design-rank and fold-size diagnostics, device
  identity, usable shuffle counts, and stage timings. Per-unit convergence
  flags are not in the sidecar; they are columns of the per-neuron statistics
  table.
