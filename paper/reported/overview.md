# What the reported numbers say, question by question

Every number on this page is filled by `scripts/reported/emit_reported_values.py` from the summary files
`paper/reported/part{n}.json`; none is typed by hand. The full list of values, with what each one is, is in
[values_index.md](values_index.md). The interpretations are the assistant's reading of those values
(2026-10-26), for the lead to check; "pending" marks values that may not be quoted as findings yet.

**A caution that applies throughout.**
- Unless a line says otherwise, a property contrast is a plain median, not controlled for event duration, and
  non-SWR events are about half as long as SWR events.

## Part 1. The model: how it works and what it captures

**Do the decoder's states mean what they claim?** The held-out tetrode test asks whether spikes that did not
choose a location still support it.
- At rest, all three states pass: Local <!--v:part1.heldout_local_immobile_estimate-->2.8<!--/v--> nats per window
  (<!--v:part1.heldout_local_immobile_animals_positive-->5<!--/v-->/<!--v:part1.state_evidence_animals-->5<!--/v--> animals,
  <!--v:part1.heldout_local_immobile_epochs_positive-->242<!--/v-->/<!--v:part1.heldout_local_immobile_epochs-->242<!--/v--> epochs), No-Spike
  <!--v:part1.heldout_no_spike_immobile_estimate-->7.7<!--/v--> (<!--v:part1.heldout_no_spike_immobile_animals_positive-->5<!--/v-->/<!--v:part1.state_evidence_animals-->5<!--/v-->),
  Non-Local Continuous <!--v:part1.heldout_continuous_immobile_estimate-->6.1<!--/v-->
  (95 % interval <!--v:part1.heldout_continuous_immobile_ci_low-->4.2<!--/v--> to <!--v:part1.heldout_continuous_immobile_ci_high-->7.9<!--/v-->;
  <!--v:part1.heldout_continuous_immobile_animals_positive-->5<!--/v-->/<!--v:part1.state_evidence_animals-->5<!--/v--> animals,
  <!--v:part1.heldout_continuous_immobile_epochs_positive-->239<!--/v-->/<!--v:part1.heldout_continuous_immobile_epochs-->242<!--/v--> epochs).
- Two do not: Non-Local Continuous **while running** is <!--v:part1.heldout_continuous_running_estimate-->−1.2<!--/v-->
  (<!--v:part1.heldout_continuous_running_animals_positive-->0<!--/v-->/<!--v:part1.state_evidence_animals-->5<!--/v--> animals), and Fragmented
  against a mismatched location is <!--v:part1.heldout_fragmented_immobile_mismatched_estimate-->0.12<!--/v-->
  (interval <!--v:part1.heldout_fragmented_immobile_mismatched_ci_low-->−0.56<!--/v--> to <!--v:part1.heldout_fragmented_immobile_mismatched_ci_high-->0.79<!--/v-->,
  <!--v:part1.heldout_fragmented_immobile_mismatched_animals_positive-->3<!--/v-->/<!--v:part1.state_evidence_animals-->5<!--/v-->).
- *Reading:* at rest the states are real; the paper should limit its events to immobility, and should not claim
  Fragmented events point to a specific place.

**What does it find on replays of known shape, against the Widloski detector?** On <!--v:part1.sweep_replays-->4,536<!--/v-->
simulated replays:
- ours reports <!--v:part1.sweep_state_space_fraction-->91<!--/v--> %, the sliding window <!--v:part1.sweep_sliding_fraction-->18<!--/v--> %; on
  the target both are built for, <!--v:part1.sweep_domain_state_space_fraction-->99<!--/v--> % against <!--v:part1.sweep_domain_sliding_fraction-->63<!--/v--> %;
- the sliding window found <!--v:part1.sweep_sliding_only_replays-->0<!--/v--> replays ours missed, and <!--v:part1.sweep_sliding_not_the_replay_fraction-->23<!--/v--> %
  of its accepted events are not the replay put in;
- ours places event edges within <!--v:part1.sweep_state_space_edge_error_max_ms-->5<!--/v--> ms; the sliding window starts
  <!--v:part1.sweep_sliding_start_early_min_ms-->18<!--/v-->–<!--v:part1.sweep_sliding_start_early_max_ms-->28<!--/v--> ms early and ends
  <!--v:part1.sweep_sliding_end_late_min_ms-->21<!--/v-->–<!--v:part1.sweep_sliding_end_late_max_ms-->65<!--/v--> ms late.
- With no replay, neither reported an event in <!--v:part1.controls_no_replay_minutes-->1.7<!--/v--> min of simulated rest, which
  bounds the false-event rate only loosely (below <!--v:part1.controls_no_replay_rate_bound-->1.8<!--/v--> per minute).
- *Reading:* on simulation, ours is more sensitive and times events better. There is no real-data false-positive
  rate (claim 12), threshold sweep (3), wider-Local-kernel re-decode (8) or model comparison (11) yet.

## Part 2. The events: what they are, and how they compare with SWRs and Widloski events

**How common are non-local events without an SWR?** Of <!--v:part2.nonlocal_events-->75,675<!--/v--> immobile non-local events,
<!--v:part2.nonlocal_events_non_swr-->42,974<!--/v--> (<!--v:part2.non_swr_fraction-->57 %<!--/v-->) overlap no SWR. Non-SWR is the largest overlap class in
<!--v:part2.animals_non_swr_plurality-->5<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals and more than half the events in
<!--v:part2.animals_non_swr_majority-->4<!--/v-->/<!--v:part2.animals-->5<!--/v--> (<!--v:part2.non_swr_fraction_animal_min@pctnum-->47<!--/v-->–<!--v:part2.non_swr_fraction_animal_max-->65 %<!--/v--> by animal).

**The other side: do detected SWRs carry non-local content?** (claim 23; Kay for the main figure)
- <!--v:part2.kay_no_nonlocal_fraction-->17 %<!--/v--> of Kay SWRs overlap no non-local event
  (<!--v:part2.kay_no_nonlocal_fraction_animal_min@pctnum-->8<!--/v-->–<!--v:part2.kay_no_nonlocal_fraction_animal_max-->28 %<!--/v--> by animal).
  Of those, <!--v:part2.kay_no_nonlocal_mostly_local_fraction-->74 %<!--/v--> are Local in most of their time bins,
  <!--v:part2.kay_no_nonlocal_mostly_no_spike_fraction-->1 %<!--/v--> No-Spike, and <!--v:part2.kay_no_nonlocal_mixed_fraction-->26 %<!--/v--> neither.
- Kay SWRs that do overlap a non-local event are non-local for a median <!--v:part2.kay_overlap_nonlocal_share_median-->88 %<!--/v-->
  of their time bins, and <!--v:part2.kay_overlap_any_local_fraction-->32 %<!--/v--> of them hold at least one Local bin.
- The other definitions (supplement) show the same two sides. Overlapping no non-local event: MUA
  <!--v:part2.mua_no_nonlocal_fraction-->15 %<!--/v-->, Carey <!--v:part2.carey_no_nonlocal_fraction-->20 %<!--/v-->, Zugaro <!--v:part2.zugaro_no_nonlocal_fraction-->35 %<!--/v-->,
  Karlsson <!--v:part2.karlsson_no_nonlocal_fraction-->38 %<!--/v-->, Yu <!--v:part2.yu_no_nonlocal_fraction-->56 %<!--/v-->. Widloski:
  <!--v:part2.widloski_no_nonlocal_fraction-->3 %<!--/v-->.
- *Reading:* SWR detection and content detection measure different things. Many SWRs carry only local content,
  and even the overlapping ones are partly local. The Widloski detector, which requires a moving sequence, almost
  never fires without a non-local event.

**What does an analysis that starts from SWRs leave out?** Over <!--v:part2.rest_hours-->40<!--/v--> h of analysed rest:
- starting from Kay SWRs: <!--v:part2.nonlocal_events_outside_kay-->42,962<!--/v--> non-local events (<!--v:part2.nonlocal_events_outside_kay_fraction-->57 %<!--/v-->; about
  <!--v:part2.nonlocal_events_outside_kay_per_hour-->1100<!--/v--> per hour of rest) and <!--v:part2.nonlocal_time_outside_kay_s-->3485<!--/v--> s of non-local time
  (<!--v:part2.nonlocal_time_outside_kay_fraction-->53 %<!--/v-->). For scale, Kay's SWRs span <!--v:part2.kay_time_s-->4127<!--/v--> s and all non-local events
  <!--v:part2.nonlocal_time_s-->6562<!--/v--> s;
- starting from any of the five SWR definitions: <!--v:part2.nonlocal_time_outside_swr_fraction-->31 %<!--/v--> of non-local time;
- starting from any SWR definition or MUA: <!--v:part2.nonlocal_events_outside_trigger-->19,221<!--/v--> events (<!--v:part2.nonlocal_events_outside_trigger_fraction-->25 %<!--/v-->;
  about <!--v:part2.nonlocal_events_outside_trigger_per_hour-->480<!--/v--> per hour) and <!--v:part2.nonlocal_time_outside_trigger_fraction-->28 %<!--/v--> of the time;
- starting from Widloski events: <!--v:part2.nonlocal_events_outside_widloski_fraction-->94 %<!--/v--> of events and
  <!--v:part2.nonlocal_time_outside_widloski_fraction-->93 %<!--/v--> of the time.

**Are these immobility theta events?** No, in general. Non-SWR events sit at quiet-rest theta
(<!--v:part2.theta_median_non_swr-->−0.62<!--/v--> z against <!--v:part2.theta_rest_reference-->−0.60<!--/v--> z at rest and <!--v:part2.theta_running_reference-->0.42<!--/v--> z while running;
below running in <!--v:part2.theta_animals_below_running_non_swr-->5<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals); <!--v:part2.theta_like_fraction_non_swr-->13 %<!--/v--> reach their epoch's
running theta (SWR events <!--v:part2.theta_like_fraction_swr-->11 %<!--/v-->).
- Ripple power: non-SWR events <!--v:part2.ripple_median_non_swr-->0.50<!--/v--> z, above quiet rest (<!--v:part2.ripple_rest_reference-->−0.60<!--/v--> z) in
  <!--v:part2.ripple_animals_above_rest_non_swr-->5<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals; SWR events <!--v:part2.ripple_median_swr-->2.35<!--/v--> z.
- Slow gamma (25–55 Hz): non-SWR <!--v:part2.slow_gamma_median_non_swr-->−0.10<!--/v--> z, SWR <!--v:part2.slow_gamma_median_swr-->0.36<!--/v--> z, quiet rest
  <!--v:part2.slow_gamma_rest_reference-->−0.28<!--/v--> z. Fast gamma (65–100 Hz): <!--v:part2.fast_gamma_median_non_swr-->−0.21<!--/v-->, <!--v:part2.fast_gamma_median_swr-->0.11<!--/v-->
  and <!--v:part2.fast_gamma_rest_reference-->−0.43<!--/v--> z. Spiking and the ripple leak broadband power into both bands, so part of
  these rises may be leakage rather than oscillation.
- Quiet rest matched in duration to the events reaches running theta in <!--v:part2.theta_like_fraction_rest_matched_non_swr-->…<!--/v--> (matched to non-SWR
  events) and <!--v:part2.theta_like_fraction_rest_matched_swr-->…<!--/v--> (matched to SWR events) of windows; events exceed it in <!--v:part2.theta_like_animals_above_rest_matched_non_swr-->…<!--/v-->/<!--v:part2.animals-->5<!--/v-->
  (non-SWR) and <!--v:part2.theta_like_animals_above_rest_matched_swr-->…<!--/v-->/<!--v:part2.animals-->5<!--/v--> (SWR) animals. *Reading:* the theta-like share of events is what quiet rest of the same
  length shows; it does not mark theta events.

**Do event boundaries agree?** For <!--v:part2.swr_overlapping_events-->32,701<!--/v--> overlapping events, the median start offset is
<!--v:part2.start_offset_median_ms-->4<!--/v--> ms and the median end offset <!--v:part2.end_offset_median_ms-->−4<!--/v--> ms (event minus SWR):
close on average.

**Are the events backed by spikes?** (every event of every epoch)
- Per spike, the decoded location beats the animal's by <!--v:part2.non_swr_close_dll_per_spike_median-->0.47<!--/v--> nats (non-SWR,
  close), <!--v:part2.non_swr_remote_dll_per_spike_median-->0.69<!--/v--> (non-SWR, remote) and <!--v:part2.swr_remote_dll_per_spike_median-->1.17<!--/v-->
  (SWR, remote); matched quiet windows score <!--v:part2.baseline_dll_per_spike_median-->0.00<!--/v-->.
- Non-SWR events have fewer spikes (<!--v:part2.non_swr_remote_spikes_median-->34<!--/v--> against <!--v:part2.swr_remote_spikes_median-->116<!--/v-->)
  on fewer tetrodes (<!--v:part2.non_swr_remote_tetrodes_median-->10<!--/v--> against <!--v:part2.swr_remote_tetrodes_median-->15<!--/v-->).
- The held-out version (pending: untracked script) gives <!--v:part2.heldout_non_swr_per_spike-->0.20<!--/v--> nats per held-out spike
  without SWR and <!--v:part2.heldout_swr_per_spike-->0.60<!--/v--> with.
- *Reading:* non-SWR events are real but carry less evidence per spike and involve fewer cells. Close events stay
  open until the wider-Local-kernel test (claim 8).

**What are the events like?** Non-SWR against SWR, medians; in brackets the 95 % interval of the SWR minus non-SWR
difference (nested bootstrap: animals, then epochs, then events):
- duration <!--v:part2.duration_median_non_swr-->48<!--/v--> against <!--v:part2.duration_median_swr-->94<!--/v--> ms [<!--v:part2.duration_difference_low-->…<!--/v--> to <!--v:part2.duration_difference_high-->…<!--/v-->];
- spatial extent <!--v:part2.extent_median_non_swr-->8.0<!--/v--> against <!--v:part2.extent_median_swr-->50<!--/v--> cm [<!--v:part2.extent_difference_low-->…<!--/v--> to <!--v:part2.extent_difference_high-->…<!--/v-->];
- replay speed <!--v:part2.speed_median_non_swr-->320<!--/v--> against <!--v:part2.speed_median_swr-->490<!--/v--> cm/s [<!--v:part2.speed_difference_low-->…<!--/v--> to <!--v:part2.speed_difference_high-->…<!--/v-->];
- pause fraction <!--v:part2.pause_median_non_swr-->1.00<!--/v--> against <!--v:part2.pause_median_swr-->0.93<!--/v--> [<!--v:part2.pause_difference_low-->…<!--/v--> to <!--v:part2.pause_difference_high-->…<!--/v-->], largest step
  <!--v:part2.max_step_median_non_swr-->2.0<!--/v--> against <!--v:part2.max_step_median_swr-->14<!--/v--> cm [<!--v:part2.max_step_difference_low-->…<!--/v--> to <!--v:part2.max_step_difference_high-->…<!--/v-->], reversals
  <!--v:part2.reversals_median_non_swr-->0<!--/v--> against <!--v:part2.reversals_median_swr-->3<!--/v--> [<!--v:part2.reversals_difference_low-->…<!--/v--> to <!--v:part2.reversals_difference_high-->…<!--/v-->];
- arms visited <!--v:part2.arms_median_non_swr-->…<!--/v--> against <!--v:part2.arms_median_swr-->…<!--/v--> [<!--v:part2.arms_difference_low-->…<!--/v--> to <!--v:part2.arms_difference_high-->…<!--/v-->], fewer in
  <!--v:part2.arms_animals_in_direction-->…<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals; start distance (track distance from the animal to the first decoded position)
  <!--v:part2.start_distance_median_non_swr-->…<!--/v--> against <!--v:part2.start_distance_median_swr-->…<!--/v--> cm [<!--v:part2.start_distance_difference_low-->…<!--/v--> to <!--v:part2.start_distance_difference_high-->…<!--/v-->], shorter
  for non-SWR events in <!--v:part2.start_distance_animals_in_direction-->…<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals;
- posterior peak <!--v:part2.peak_median_non_swr-->0.90<!--/v--> against <!--v:part2.peak_median_swr-->0.82<!--/v--> [<!--v:part2.peak_difference_low-->…<!--/v--> to <!--v:part2.peak_difference_high-->…<!--/v-->] (<!--v:part2.peak_animals_in_direction-->4<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals),
  HPD <!--v:part2.hpd_median_non_swr-->44<!--/v--> against <!--v:part2.hpd_median_swr-->80<!--/v--> cm [<!--v:part2.hpd_difference_low-->…<!--/v--> to <!--v:part2.hpd_difference_high-->…<!--/v-->] (<!--v:part2.hpd_animals_in_direction-->4<!--/v-->/<!--v:part2.animals-->5<!--/v-->).
- At matched duration (SWR minus non-SWR within ten duration-quantile strata, averaged; 95 % interval): extent
  <!--v:part2.extent_matched_difference-->…<!--/v--> cm [<!--v:part2.extent_matched_difference_low-->…<!--/v--> to <!--v:part2.extent_matched_difference_high-->…<!--/v-->], speed
  <!--v:part2.speed_matched_difference-->…<!--/v--> cm/s [<!--v:part2.speed_matched_difference_low-->…<!--/v--> to <!--v:part2.speed_matched_difference_high-->…<!--/v-->], pause fraction
  <!--v:part2.pause_matched_difference-->…<!--/v--> [<!--v:part2.pause_matched_difference_low-->…<!--/v--> to <!--v:part2.pause_matched_difference_high-->…<!--/v-->], largest step
  <!--v:part2.max_step_matched_difference-->…<!--/v--> cm [<!--v:part2.max_step_matched_difference_low-->…<!--/v--> to <!--v:part2.max_step_matched_difference_high-->…<!--/v-->], reversals
  <!--v:part2.reversals_matched_difference-->…<!--/v--> [<!--v:part2.reversals_matched_difference_low-->…<!--/v--> to <!--v:part2.reversals_matched_difference_high-->…<!--/v-->], arms visited
  <!--v:part2.arms_matched_difference-->…<!--/v--> [<!--v:part2.arms_matched_difference_low-->…<!--/v--> to <!--v:part2.arms_matched_difference_high-->…<!--/v-->], posterior peak
  <!--v:part2.peak_matched_difference-->…<!--/v--> [<!--v:part2.peak_matched_difference_low-->…<!--/v--> to <!--v:part2.peak_matched_difference_high-->…<!--/v-->], HPD
  <!--v:part2.hpd_matched_difference-->…<!--/v--> cm [<!--v:part2.hpd_matched_difference_low-->…<!--/v--> to <!--v:part2.hpd_matched_difference_high-->…<!--/v-->], start distance
  <!--v:part2.start_distance_matched_difference-->…<!--/v--> cm [<!--v:part2.start_distance_matched_difference_low-->…<!--/v--> to <!--v:part2.start_distance_matched_difference_high-->…<!--/v-->].
- Where content sits: non-SWR <!--v:part2.non_swr_content_remote_well_fraction-->45 %<!--/v--> at a remote well,
  <!--v:part2.non_swr_content_own_well_fraction-->13 %<!--/v--> at the animal's own well and <!--v:part2.non_swr_content_remote_path_fraction-->41 %<!--/v--> on remote
  paths; SWR <!--v:part2.swr_content_remote_path_fraction-->73 %<!--/v--> on remote paths and <!--v:part2.swr_content_remote_well_fraction-->21 %<!--/v--> at a remote well.
- *Reading:* non-SWR events are short, compact, nearly stationary and sharper, and sit at wells more often than
  SWR events. Duration does not explain the contrasts: at matched duration extent, speed, largest step, reversals,
  arms visited, pause fraction and posterior peak still differ, with intervals that exclude zero. HPD and start
  distance do not differ clearly, raw or matched.

**One class or two?** (claim 24; `scripts/detection/measure_ripple_class_structure.py`) The events' peak ripple power
has one mode in <!--v:part2.ripple_peak_animals_unimodal-->5<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals, lying at
<!--v:part2.ripple_peak_mode_animal_min-->0.90<!--/v-->–<!--v:part2.ripple_peak_mode_animal_max-->1.99<!--/v--> z, so the SWR threshold cuts one mode rather than falling
between two. On the Kay criterion around each event, no property but spike evidence jumps at the SWR threshold in
more than <!--v:part2.step_other_properties_criterion_animals_max-->1<!--/v--> animal(s); spike evidence per spike rises there in
<!--v:part2.step_evidence_criterion_animals_same_sign-->5<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals (a step in <!--v:part2.step_evidence_criterion_animals-->3<!--/v-->), by a median
<!--v:part2.step_evidence_criterion_share_median-->+0.19<!--/v--> of the SWR minus non-SWR difference. *Reading:* one continuum of ripple power cut by
the detector's threshold, with spike evidence per spike the one hint of a boundary.

## Part 3. Behavior: how the events relate to the task, SWR against non-SWR

**How often do events occur, and does reward drive the rate?**
- Baseline rates are similar: <!--v:part3.rate_baseline_non_local_no_swr-->0.22<!--/v--> non-SWR against <!--v:part3.rate_baseline_non_local_swr-->0.21<!--/v-->
  SWR-coupled events per second of rest (within-session ratio <!--v:part3.rate_baseline_ratio_swr_coupled_to_no_swr-->0.98<!--/v-->,
  p <!--v:part3.rate_baseline_ratio_swr_coupled_to_no_swr_p-->0.30<!--/v-->).
- One reward later, the rate rises <!--v:part3.rate_kernel_lag_one_non_local_swr-->1.51<!--/v-->× for SWR-coupled events and
  <!--v:part3.rate_kernel_lag_one_non_local_no_swr-->1.01<!--/v-->× for non-SWR events. The immediate reward coefficient is larger for
  SWR-coupled events in <!--v:part3.rate_paired_positive_fraction-->77 %<!--/v--> of <!--v:part3.rate_paired_sessions-->240<!--/v--> sessions
  (p <!--v:part3.rate_paired_p-->1.1e-19<!--/v-->).
- In the per-stop model, adding reward history improves the fit for SWR-coupled events
  (median AIC(with) − AIC(without), negative = better: <!--v:part3.per_stop_delta_aic_session_non_local_swr-->−3.1<!--/v-->) but not for non-SWR events
  (<!--v:part3.per_stop_delta_aic_session_non_local_no_swr-->+0.88<!--/v-->).
- *Reading:* reward gates SWR-coupled events; non-SWR events occur at a similar rate but barely respond to reward.

**What do events represent?** (content, log weights per arm; fraction estimator unless named; q is the false-discovery-adjusted p)
- Current arm: <!--v:part3.base_current_all-->+0.73<!--/v--> (<!--v:part3.base_current_all_animals_positive-->5<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals); fitted separately,
  SWR <!--v:part3.base_current_swr-->+0.88<!--/v--> and non-SWR <!--v:part3.base_current_non_swr-->+0.53<!--/v-->. *Open:* the pooled SWR × current-arm contrast is
  negative (total <!--v:part3.swr_current_total-->−0.41<!--/v-->, direct <!--v:part3.swr_current_direct-->−0.071<!--/v-->), the opposite sign to the separate fits.
  The two models disagree, and this needs checking before 19a is cited.
- Breadth: SWR events spread over arms <!--v:part3.dm_breadth_ratio-->5.6<!--/v-->× more (Dirichlet-multinomial; <!--v:part3.dm_breadth_length_adjusted_ratio-->3.9<!--/v-->×
  after event length; <!--v:part3.dm_breadth_length_share-->21 %<!--/v--> of the contrast runs through length).
- Recently rewarded arms: non-SWR content is not selective (<!--v:part3.armreward_non_swr-->+0.027<!--/v-->, q <!--v:part3.armreward_non_swr_q-->0.35<!--/v-->);
  SWR content avoids them (<!--v:part3.armreward_swr-->−0.26<!--/v-->, <!--v:part3.armreward_swr_animals_positive-->0<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals positive).
- Uncertain arms: content favours them, <!--v:part3.armuncert_all-->+0.31<!--/v--> (<!--v:part3.armuncert_all_animals_positive-->5<!--/v-->/<!--v:part2.animals-->5<!--/v-->, q <!--v:part3.armuncert_all_q-->2.7e-27<!--/v-->),
  for both non-SWR (<!--v:part3.armuncert_non_swr-->+0.18<!--/v-->) and SWR (<!--v:part3.armuncert_swr-->+0.37<!--/v-->). Adding uncertainty helps the leaf model
  (median AIC improvement, positive = better: <!--v:part3.leaf_step_uncertainty-->+1.3<!--/v-->, <!--v:part3.leaf_step_uncertainty_animals-->5<!--/v-->/<!--v:part2.animals-->5<!--/v--> animals); adding value does not
  (<!--v:part3.leaf_step_value-->−2.4<!--/v-->, <!--v:part3.leaf_step_value_animals-->0<!--/v-->/<!--v:part2.animals-->5<!--/v-->).
- Arm value: <!--v:part3.armvalue_all-->+0.078<!--/v--> (<!--v:part3.armvalue_all_animals_positive-->4<!--/v-->/<!--v:part2.animals-->5<!--/v-->, q <!--v:part3.armvalue_all_q-->0.11<!--/v-->). The SWR × value contrast is
  near zero once event length is in (direct <!--v:part3.swr_armvalue_direct-->−0.014<!--/v-->) and negative without it (total <!--v:part3.swr_armvalue_total-->−0.25<!--/v-->).
- *Reading:* the robust content facts are current-arm bias, preference for uncertain arms, broader SWR content, and SWR
  avoidance of recently rewarded arms. Value effects are weak or run through event length.

**Does content relate to what the animal does next?** (pending: the choice fit has no provenance sidecar)
- Next choice: over the behavioural baseline, non-SWR content <!--v:part3.choice_non_swr-->+0.040<!--/v--> (p <!--v:part3.choice_non_swr_p-->0.84<!--/v-->),
  SWR content <!--v:part3.choice_swr-->+0.041<!--/v--> (p <!--v:part3.choice_swr_p-->0.66<!--/v-->). Without the baseline it is <!--v:part3.choice_raw-->−0.65<!--/v--> (p <!--v:part3.choice_raw_p-->2.3e-13<!--/v-->),
  so the raw association (content pointing away from the next choice) is carried by behaviour, not by content.
- Stay or leave: the smallest q over roles is <!--v:part3.stay_min_q_signrank-->0.44<!--/v--> (signed-rank) and <!--v:part3.stay_min_q_gee-->0.52<!--/v--> (GEE).
- *Reading:* content predicts neither the next choice nor staying, beyond what behaviour already predicts.

## Part 4. Other brain areas: mPFC

- <!--v:part4.units-->11,604<!--/v--> mPFC units in <!--v:part4.sessions-->75<!--/v--> sessions. Against duration-matched quiet controls, non-SWR replay
  modulates <!--v:part4.engagement_called_fraction-->23 %<!--/v--> of units at raw p < 0.05 (<!--v:part4.engagement_units_called-->2,718<!--/v--> units;
  <!--v:part4.engagement_bh_fraction-->10 %<!--/v--> after Benjamini-Hochberg, for reference only). By animal it ranges from
  <!--v:part4.engagement_called_fraction_jsixteen-->20 %<!--/v--> to <!--v:part4.engagement_called_fraction_chimi-->44 %<!--/v-->.
- Descriptive shares at raw p < 0.05: SWR <!--v:part4.swr_called_fraction-->31 %<!--/v-->, all non-local events <!--v:part4.nonlocal_called_fraction-->40 %<!--/v-->,
  SWR-coupled replay <!--v:part4.swr_replay_called_fraction-->33 %<!--/v-->, non-SWR replay <!--v:part4.non_swr_replay_called_fraction-->23 %<!--/v-->.
- Best model by held-out deviance: constant <!--v:part4.best_model_constant_fraction-->48 %<!--/v-->, non-local <!--v:part4.best_model_nonlocal_fraction-->33 %<!--/v-->,
  SWR <!--v:part4.best_model_swr_fraction-->14 %<!--/v-->, both <!--v:part4.best_model_joint_fraction-->5 %<!--/v-->.
- **Same units?** Of <!--v:part4.overlap_units_tested-->11,604<!--/v--> units, <!--v:part4.overlap_units_both-->1,443<!--/v--> respond to both SWR-coupled and non-SWR
  replay (raw p < 0.05), against <!--v:part4.overlap_units_both_expected-->878<!--/v--> expected if the two were independent (odds ratio
  <!--v:part4.overlap_odds_ratio-->3.3<!--/v-->; <!--v:part4.overlap_odds_ratio_animal_min-->2.2<!--/v-->–<!--v:part4.overlap_odds_ratio_animal_max-->5.5<!--/v--> by animal, above 1 in
  <!--v:part4.overlap_animals_odds_ratio_above_one-->5<!--/v-->/5). Another <!--v:part4.overlap_units_swr_replay_only-->2,407<!--/v--> respond only to SWR-coupled replay and
  <!--v:part4.overlap_units_non_swr_replay_only-->1,203<!--/v--> only to non-SWR replay.
- **Alike?** Across units the two peak modulations correlate <!--v:part4.overlap_rank_correlation-->0.35<!--/v--> (Spearman;
  <!--v:part4.overlap_rank_correlation_called-->0.46<!--/v--> among units called for either), and <!--v:part4.overlap_same_sign_fraction_both-->84 %<!--/v--> of the units called for
  both change in the same direction. Splitting the non-local response by SWR status improves held-out prediction in
  only <!--v:part4.split_improves_fraction-->14 %<!--/v--> of units (best model for <!--v:part4.split_best_model_fraction-->9 %<!--/v-->).
- **How much?** The median unit's largest rate change is <!--v:part4.effect_rate_multiplier_median_swr_replay-->2.0<!--/v-->× for SWR-coupled replay and
  <!--v:part4.effect_rate_multiplier_median_non_swr_replay-->1.9<!--/v-->× for non-SWR replay, against <!--v:part4.effect_rate_multiplier_median_matched_control-->1.5<!--/v-->× for matched
  quiet-rest control windows (the noise floor of a peak taken over lags). Among called units:
  <!--v:part4.effect_abs_peak_median_called_swr_replay-->0.79<!--/v--> and <!--v:part4.effect_abs_peak_median_called_non_swr_replay-->0.70<!--/v--> log units. In
  <!--v:part4.split_swr_larger_fraction_called-->65 %<!--/v--> of the units called for either, the SWR-coupled response is the larger.
- The share of peaks that are decreases is about the same for events (<!--v:part4.effect_suppressed_fraction_non_swr_replay-->63 %<!--/v-->) and for the
  control windows (<!--v:part4.effect_suppressed_fraction_matched_control-->61 %<!--/v-->), so the peak measure does not show whether replay mostly
  suppresses or excites.
- *Reading:* a sizeable minority of mPFC units respond to non-SWR replay beyond matched rest. These are unit-level,
  uncorrected fractions; the population-level SWR against non-SWR test is still to do, and only mPFC is fitted.
