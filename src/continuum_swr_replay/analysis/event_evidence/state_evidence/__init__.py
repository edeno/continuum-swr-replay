"""Held-out spike evidence for each decoder state.

The decoder's four states each make a claim about the spikes:

- **Local**: the content is the animal's own position;
- **Non-Local Continuous / Fragmented**: the content is a remote position;
- **No-Spike**: the population is silent.

Each claim is judged by spikes that did not choose the location. The epoch's
tetrodes are split into two halves; the selecting half S chooses (its maximum
a posteriori position, MAP_S) and the scoring half C scores; then the roles
swap. Ten seeded partitions in both directions give 20 runs, and every value is
the mean over the runs of a window total, in nats.

Units (per epoch; regimes per 2 ms bin, :func:`regimes`)
--------------------------------------------------------
- **Bouts** (:func:`state_bouts`): maximal runs of P(state) > 0.8 with every bin
  in one regime, at least 5 bins; at most 1500 per state x regime (a seeded
  sample). A bout longer than 50 bins is scored on one random 50-bin
  sub-window, seeded by the bout's bounds.
- **Matched windows** (:func:`matched_windows`): one random window per scored
  bout window, same regime in every bin and same length, from a seeded
  sequence per (regime, length).
- **Calibration windows** (:func:`calibration_windows`): 2000 random 10-bin
  windows per regime, independent of the decode; each carries the mean
  posterior of every state.
- **Event windows** (:func:`event_windows`, a runner's other label set): one
  window per canonical non-local event, every bin of it, labelled by its
  ``is_swr``, each with one matched window drawn as a bout's is.

Evidence (:func:`window_evidence`; ``_mark`` = spike-mark part, the primary;
``_full`` = the full clusterless log-likelihood, the secondary)
------------------------------------------------------------------------------
The spike-mark part of logL_C(x) is the sum over C's spikes of
log lambda_e(x, mark); the full part also subtracts C's ground intensity
sum_e g_e(x) in every bin. ``x`` runs over the real place bins only: the
junction placeholders (``edge_id == -1``) are exactly the bins outside the
track interior and never enter (:func:`interior_bin_mask`).

- ``loc_*``: logL_C(animal_t) - mean over 20 partners of
  logL_C(partner's animal position). Partners are other windows of the same
  pool whose centre-bin animal position is at least 20 cm away by track
  distance; never the window itself (:func:`null_partners`).
- ``nl_anim_*``: logL_C(MAP_S) - logL_C(animal_t), with MAP_S the argmax of S's
  window-summed log-likelihood.
- ``nl_null_*``: logL_C(MAP_S) - mean over 20 partners of logL_C(partner's
  MAP_S), partners being any other windows of the same pool.
- ``nlr_anim_*``: logL_C(S's best bin >= 20 cm from the animal) -
  logL_C(animal_t). Its negative is the only Local test in which S chooses.
- ``ns_logratio``: log P(C's counts | silence) - log P(C's counts | local
  coding), Poisson per tetrode-bin; silence is the decoder's ``no_spike_rate``
  x dt, local coding g_e(animal_t). With C silent it is E_C minus a negligible
  term. ``ns_count`` is C's spike count, ``ns_expected`` E_C (the spikes local
  coding expects of C), ``ns_silent_frac`` the share of runs with C silent and
  ``ns_pred_silent`` exp(-E_C).
- ``cond_*`` (summed over runs, not averaged): runs with S silent, runs with
  both halves silent, and exp(-E_C) summed over the S-silent runs; the
  decoder-free test of coordinated silence, P(C silent | S silent) against
  local coding's exp(-E_C) (:func:`coordinated_silence`).
- ``*_ps``: per held-out spike, runs with C silent excluded.
- ``mapS_dist_cm`` / ``mapSr_dist_cm``: track distance from the animal's
  window-centre position to MAP_S / S's best remote bin.

Encodings: what scores a window, and whose spikes it was fit on
---------------------------------------------------------------
Every window is scored under three encodings, the column prefixes:

- ``runcv_`` (primary): the running-only encoding, 2-fold time-held-out. The
  decode axis is cut into alternating 30 s blocks (:func:`time_folds`); fold
  f's model is the decode's detector refit on the running bins (head speed
  >= 4 cm/s) of fold f's blocks, and a window is scored by the model of the
  fold that does not hold its centre bin (:func:`window_folds`,
  :func:`held_out_evidence`). A window's own spikes are not in the model that
  scores it, but for a window straddling a block edge (``cv_insample_share``
  of its bins; under 0.1 % of windows in the prototype). The library pads
  every training run by one bin (:func:`training_spike_bins`), so a window's
  edge bin next to the other fold's running bins can be in too.
- ``run_`` (secondary): the running-only encoding fit on every running bin,
  so it includes running windows' own spikes.
- ``can_`` (secondary): the decode's own encoding, fit on every non-ripple
  bin, so it includes non-ripple windows' own spikes.

Running bins within two bins of a non-local event bin are left out of every
running-only fit (:func:`running_only_masks`), so no event spike enters one even through the library's
padding (one bin width either side, with its float rounding).

What is and is not held out (properties, not defects)
-----------------------------------------------------
- Only the remote location is held out by the tetrode split: S alone chooses
  MAP_S. The state labels come from the all-tetrode decode, so C's spikes
  helped make them. Labels from a decode that saw S alone (Task 13's half
  decodes, since removed) hold them out too, but for the canonical fit's
  discrete transitions and initial conditions; such labels are scored by their own C only
  (``window_evidence(..., both_directions=False)``), and a table holding
  several splits names each in :data:`~.summary.SPLIT_COLUMN`.
- Run-averaged Local and No-Spike values equal half the all-tetrode
  (in-sample) values: no half chooses the animal's position, and every
  tetrode scores in exactly half of the runs.
- One spike costs -log(no_spike_rate x dt) nats under No-Spike, tens of nats
  at the decoder's rate (the runner's README gives the value), so a confident
  No-Spike label forces both halves silent; only the size of E_C is
  informative there.
- With few spikes, S's silence term pulls MAP_S toward low-rate bins, so
  MAP_S's distance from the animal is not a Local test.

Inference (:mod:`.summary`)
---------------------------
Epochs are nested in animals, and an epoch's windows are not independent:
each epoch enters as one value, and the animal is the unit of replication.

- An epoch's value is its median over windows for the log-likelihood ratios,
  which are heavy-tailed (in one pilot chimi epoch three spike bursts score
  -1641, -1407 and -1016 nats), and its mean for expected counts, counts,
  shares and distances, which are bounded. The rule is set in one place,
  ``EPOCH_SUMMARY``; the other summary is reported as secondary. A calibration
  curve's value is the epoch's own slope.
- The primary inference is at the animal level: the mean of the animal
  summaries (each the mean of the animal's epoch values), its 95 % interval
  from the nested bootstrap (animals, then epochs within each) with a
  small-sample t correction over the A animals, and its t-test with A - 1
  degrees of freedom, with k of A animals and k of N epochs.
- Secondary: the epoch-weighted mean with its interval from the same
  replicates by the same rule, and the sign-flip p over epochs, which ignores
  the animals.
- A calibration verdict needs at least 10 windows in the top and in the
  bottom probability bins, pooled; otherwise it reads "too few windows".
- Window-level statistics are descriptive. The ``_window_level`` CIs and
  p-values ignore the nesting, are not for inference and are computed only on
  request (``window_level=True``).

Modules: :mod:`.terms` (per-tetrode terms, encodings, decode files),
:mod:`.windows` (regimes, bouts, matched and calibration windows, tetrode
halves, partners), :mod:`.evidence` (window evidence) and :mod:`.summary`;
:mod:`.io` reads the runners' code version and the decode's masks (their JSON
records are written with ``runs.io``).
"""

from continuum_swr_replay.analysis.event_evidence.state_evidence.evidence import (
    held_out_evidence,
    window_evidence,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.io import (
    code_version,
    saved_decode_masks,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.summary import (
    CALIBRATION_METRICS,
    ENCODING_HELD_OUT,
    ENCODING_LABELS,
    EPOCH_SUMMARY,
    MIN_BIN_WINDOWS,
    PRIMARY_ENCODING,
    PRIMARY_METRICS,
    PROBABILITY_BIN_LABELS,
    SECONDARY_METRICS,
    SPLIT_COLUMN,
    AnimalTTest,
    animal_t_test,
    bout_animal_summaries,
    bout_statistics,
    calibration_statistics,
    coordinated_silence,
    epoch_slopes,
    epoch_summaries,
    epoch_summary_kind,
    nested_mean_intervals,
    ols_slope_ci,
    paired_bouts,
    rate_calibration,
    sign_flip_p,
    split_non_local_by_swr,
    stratified_bootstrap_ci,
    summarize_state_evidence,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    CV_BLOCK_S,
    EVENT_MARGIN_BINS,
    UNDERFLOW_OCCUPANCY_FLOOR,
    TetrodeTerms,
    encoding_comparison,
    encoding_inputs,
    event_likelihood,
    interior_bin_mask,
    load_detector,
    load_encoding,
    nearest_interior_bins,
    per_tetrode_terms,
    running_only_masks,
    spikes_in_bins,
    subnormal_occupancy_bin_mask,
    time_folds,
    training_spike_bins,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    EVENT_STATE,
    HALVES_SEED,
    REGIMES,
    STATES,
    SWR_LABELS,
    analysed_regimes,
    build_windows,
    calibration_windows,
    event_windows,
    keyed_rng,
    matched_windows,
    null_partners,
    regimes,
    state_bouts,
    tetrode_halves,
    window_folds,
)

__all__ = [
    "AnimalTTest",
    "CALIBRATION_METRICS",
    "CV_BLOCK_S",
    "ENCODING_HELD_OUT",
    "ENCODING_LABELS",
    "EVENT_MARGIN_BINS",
    "EVENT_STATE",
    "EPOCH_SUMMARY",
    "HALVES_SEED",
    "MIN_BIN_WINDOWS",
    "PRIMARY_ENCODING",
    "PRIMARY_METRICS",
    "PROBABILITY_BIN_LABELS",
    "REGIMES",
    "SECONDARY_METRICS",
    "SPLIT_COLUMN",
    "STATES",
    "SWR_LABELS",
    "UNDERFLOW_OCCUPANCY_FLOOR",
    "TetrodeTerms",
    "analysed_regimes",
    "animal_t_test",
    "bout_animal_summaries",
    "bout_statistics",
    "build_windows",
    "calibration_statistics",
    "calibration_windows",
    "code_version",
    "coordinated_silence",
    "encoding_comparison",
    "encoding_inputs",
    "epoch_slopes",
    "epoch_summaries",
    "epoch_summary_kind",
    "event_likelihood",
    "event_windows",
    "held_out_evidence",
    "interior_bin_mask",
    "keyed_rng",
    "load_detector",
    "load_encoding",
    "matched_windows",
    "nearest_interior_bins",
    "nested_mean_intervals",
    "null_partners",
    "ols_slope_ci",
    "paired_bouts",
    "per_tetrode_terms",
    "rate_calibration",
    "regimes",
    "running_only_masks",
    "saved_decode_masks",
    "sign_flip_p",
    "spikes_in_bins",
    "split_non_local_by_swr",
    "state_bouts",
    "stratified_bootstrap_ci",
    "summarize_state_evidence",
    "tetrode_halves",
    "time_folds",
    "training_spike_bins",
    "subnormal_occupancy_bin_mask",
    "window_evidence",
    "window_folds",
]
