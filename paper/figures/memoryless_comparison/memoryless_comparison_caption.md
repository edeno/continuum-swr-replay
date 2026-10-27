# memoryless_comparison_caption

**Two replay detectors on replays whose shape is known.** The sliding-window
detector decodes each 80 ms window on its own, with no model of how one window
follows the next, and keeps or drops windows on the spread of their posterior
and on the jump to the next window. The state-space detector infers the whole
event at once and carries the animal's own place as a competing hypothesis.
Throughout, both detectors read one encoding model, fitted once; the
sliding-window detector decodes it as the reference does, with 1e-3 Hz added to
every place field, and is held to the published definition as adapted here, its
significance test included: each accepted sequence is decoded again 100 times
with the place fields swapped between cells, and must beat those draws on both
its mean posterior spread and its mean jump. The adaptations: a window's spread
is a root-mean-square distance along the track about a place bin, cut at 10 cm,
where the paper's open-field statistic is a two-dimensional image moment that a
10 cm cutoff does not carry over to exactly; centers are place bins, so the
pairs of windows whose jump the original sets aside as zero are read from the
spikes instead; windows step 4 ms rather than 5 ms; and the immobility cutoff
is 4 rather than 5 cm/s. The figure compares the events two complete pipelines
report, each with its own acceptance criteria, so a difference between them
cannot be put down to how either finds candidates alone.

**(a-f) One recording with labeled events.** The simulation carries a
continuous outbound replay, a fragmented replay, a continuous inbound replay
and a silent stretch; a remote stationary event, a 30 cm replay, an 80 ms
replay and a still event with one window decoded far away were written into the
spike trains here and are marked with a star.

**(a)** The whole recording. Green is where the animal is; dots are the decoded
place of each window, kept or rejected; yellow marks a labeled event; bars
below the axis are the events each detector reports and the candidates the
sliding-window detector rejected.

**(b-f)** Five events, with the posterior in grey; bars below the axis are the
events each detector reports, and a sequence that fails the significance test
is drawn as rejected. The sliding-window detector finds the continuous replay
and gives it 5.8 times its true length, because 81% of the event it reports is
the animal's own place, absorbed by the merge rule. It also accepts the 80 ms
replay, at 1.6 times its length with 0% local coding in it: the window alone
puts a floor under any duration it can report. It passes the significance test
(p = 0.010 on spread and 0.030 on jump). It assembles a candidate from the 30
cm replay and another from the remote stationary event and declines both for
too little dispersion (6 and 0 cm against the 12 cm it requires). The
state-space detector reports all four, with every bound within 42 ms of the
truth. The outbound continuous replay fails it (p = 0.010 on spread and 0.079
on jump), 3 draws from passing, and the inbound continuous replay passes it (p
= 0.010 on spread and 0.020 on jump). Most of each reported continuous replay
is the animal's own place either way: the test asks whether a sequence is
spatially coherent, not whether it is somewhere other than the animal. **(f)**
A still event at the animal's own place, with spikes placed by hand so that one
80 ms window holds only two spikes of a cell 149 cm away (star). The still
parts, 120 ms either side, come from the two cells whose fields lie nearest the
animal, and 99% of the reported event's kept windows decode within 15 cm of it;
the event is reported 396 ms long. The windows either side of the far one are
too broad to keep, but the one after it centres near it, so it passes the jump
rule; as a run of one window it is dropped before the merge, and the still
parts either side merge across it. Its dispersion still counts toward the
merged event's, as in the reference: 14.4 cm with it and 0.5 cm without,
against the 12 cm required, so a still event is called replay on the strength
of one window. Put through the significance test it fails it (p = 0.41 on
spread and 0.32 on jump). The state-space detector reports no non-local event
over it. On the first recorded pilot epoch this pattern decided <!--v:part1.pilot_calls_decided_by_outside_window-->2<!--/v--> of its <!--v:part1.pilot_replay_calls-->4<!--/v-->
replay calls, both of which passed the test; the event table now carries the
dispersion without such windows beside the published one.

**(g-k) <!--v:part1.sweep_replays-->4,536<!--/v--> replays of chosen shape.** Made-up replays were written into the
spike trains of a simulated linear-track recording, one at a time, each with a
path length, duration, firing rate and distance from the animal's own place
that we chose. Because the right answer is known, what each detector reports
can be scored against it. A detector counts as reporting a replay only when
what it returns is that replay: on its path throughout, and not covering far
more ground than the path does.

**(g, h)** Percent of those <!--v:part1.sweep_replays-->4,536<!--/v--> replays reported, by the path the replay
travels and how long it lasts; both detectors are scored the same way against
the same known answer. The comparison that isolates detection uses a shared
target: the replays the published definition can accept at all, a path over
about 42 cm and a duration over 100 ms, <!--v:part1.sweep_domain_fraction|pctnum-->25<!--/v-->% of the cells, with both detectors
held to that definition's duration and dispersion criteria. There the
sliding-window detector reports <!--v:part1.sweep_domain_sliding_before_test_fraction|pctnum-->68<!--/v-->% before its significance test and the
state-space detector <!--v:part1.sweep_domain_state_space_published_fraction|pctnum-->68<!--/v-->%. The significance test, which only the sliding-window
detector is put through, brings it to <!--v:part1.sweep_domain_sliding_fraction|pctnum-->63<!--/v-->%; the state-space detector's own
acceptance criteria, in place of the published ones, bring it to <!--v:part1.sweep_domain_state_space_fraction|pctnum-->99<!--/v-->%. Over the
whole grid, each under its own criteria, the two read <!--v:part1.sweep_sliding_fraction|pctnum-->18<!--/v-->% and <!--v:part1.sweep_state_space_fraction|pctnum-->91<!--/v-->%; that gap
combines differences in detection with differences in what each counts as a
replay. Over the whole grid the sliding-window detector reports <!--v:part1.sweep_sliding_before_test_fraction|pctnum-->21<!--/v-->% before its
significance test, and <!--v:part1.sweep_sliding_spread_only_fraction|pctnum-->20<!--/v-->% with the spread half alone, the only half the
detection script's own filter could fail; the paper's analyses read both
halves, from a corrected script. Of the replays it reported before the test,
<!--v:part1.sweep_spread_half_pass_fraction|pctnum-->97<!--/v-->% pass the spread half and <!--v:part1.sweep_jump_half_pass_fraction|pctnum-->87<!--/v-->% the jump half; a short replay covers its path
in few windows, so its own jumps are large, and the jump half passes <!--v:part1.sweep_jump_half_pass_shortest_fraction|pctnum-->31<!--/v-->% of 40
ms replays against <!--v:part1.sweep_jump_half_pass_longest_fraction|pctnum-->98<!--/v-->% of 500 ms ones. The test keeps the reference's
conventions where they carry over, counting only draws strictly below the
observed value and passing each half at p ≤ 0.05, and translates its zero-jump
rule to binned centers. <!--v:part1.sweep_state_space_only_fraction|pctnum-->73<!--/v-->% were reported by the state-space detector and not
by the sliding-window detector, counting the <!--v:part1.sweep_failed_test_fraction|pctnum-->4<!--/v-->% it reported and then failed on
the significance test; <!--v:part1.sweep_sliding_only_replays-->0<!--/v--> of the <!--v:part1.sweep_replays-->4,536<!--/v--> went the other way (<!--v:part1.sweep_sliding_only_before_test_replays-->5<!--/v--> before the significance
test).

The two detectors reach their limits on different quantities. The state-space
detector is limited by duration, which is spike count, which is the data. The
sliding-window detector is limited by path length, through its own dispersion
criterion: a replay must spread more than 12 cm about its own centre, and a 30
cm path cannot, however well it is decoded. Held to that same criterion our
state-space detector also reports nothing below 45 cm, so this is the published
replay definition acting, not the window-by-window inference.

**(i, j)** Where each detector puts a replay's start (i) and end (j): reported
minus true, in ms, median line and interquartile band by true duration, over
the replays both detectors reported, the sliding window's before its
significance test, which never moves a sequence's bounds. The sliding window's
bounds are its first and last window midpoints, and its 80 ms windows reach 40
ms beyond each. The state-space detector's median start and end sit within <!--v:part1.sweep_state_space_edge_error_max_ms-->5<!--/v--> ms
of the truth at every duration. The sliding window starts a median <!--v:part1.sweep_sliding_start_early_min_ms-->18<!--/v--> to <!--v:part1.sweep_sliding_start_early_max_ms-->28<!--/v--> ms
early and ends <!--v:part1.sweep_sliding_end_late_min_ms-->21<!--/v--> to <!--v:part1.sweep_sliding_end_late_max_ms-->65<!--/v--> ms late, and its late ends carry a long tail, the upper
quartile reaching <!--v:part1.sweep_sliding_end_late_upper_quartile_max_ms-->231<!--/v--> ms; panel (k) shows how much of a reported event is the
animal's own place. A 40 ms replay's reported start is a median <!--v:part1.sweep_shortest_start_early_ms-->28<!--/v--> ms early and
its end a median <!--v:part1.sweep_shortest_end_late_ms-->65<!--/v--> ms late, and its median reported duration is about <!--v:part1.sweep_shortest_reported_duration_ms-->132<!--/v--> ms,
the floor panel (c) shows for one event.

**(k)** Reported duration over actual duration against how far the replay sits
from the animal's own place. Numbers give the share of the reported event's 80
ms windows that decode the animal's place rather than a replayed one. That
absorption of local coding reaches about 80 cm from the animal and is what
drives the largest over-reporting, as in panel (b); beyond it the window floor
of panels (i, j) still leaves the reported duration too long. Over the whole
sweep, <!--v:part1.sweep_sliding_not_the_replay_fraction|pctnum-->23<!--/v-->% of the sliding-window detector's accepted events are not the replay
that was put in.

**(l-o) Activity that is not one clean replay**, written into the same
recording and scored by the same rule; our detector is read both ways, its
non-local events (continuous and fragmented states together) and the continuous
state alone. **(l)** Share of bursts with an event reported on them. A
place-less burst, every cell firing at one rate unrelated to its field, carries
no spatial content, so any event on it is false; a jumping burst represents a
random place every 15 ms, so calling it non-local is right and calling it a
sequence is not. Our detector's non-local events cover <!--v:part1.controls_placeless_nonlocal_fraction|pctnum-->90<!--/v-->% of place-less
bursts, which spend <!--v:part1.controls_placeless_fragmented_share_min|pctnum-->59<!--/v-->% to <!--v:part1.controls_placeless_fragmented_share_max|pctnum-->80<!--/v-->% of their time in the fragmented state; the
continuous state alone covers <!--v:part1.controls_placeless_continuous_fraction|pctnum-->1<!--/v-->%, and the sliding window <!--v:part1.controls_placeless_sliding_fraction|pctnum-->8<!--/v-->% (<!--v:part1.controls_placeless_sliding_before_test_fraction|pctnum-->15<!--/v-->% before its
significance test). These pool the three peak rates, 20, 60 and 160 Hz; at 20
Hz a place-less burst fires about as fast as the local coding it replaces, and
our non-local events still cover <!--v:part1.controls_placeless_nonlocal_lowest_rate_fraction|pctnum-->75<!--/v-->% of them. On jumping bursts our non-local
events cover <!--v:part1.controls_jumping_nonlocal_fraction|pctnum-->97<!--/v-->%, which is right; the continuous state alone covers <!--v:part1.controls_jumping_continuous_fraction|pctnum-->50<!--/v-->% and the
sliding window <!--v:part1.controls_jumping_sliding_fraction|pctnum-->7<!--/v-->% (<!--v:part1.controls_jumping_sliding_before_test_fraction|pctnum-->13<!--/v-->% before its test), each calling a sequence that is not
there. Over <!--v:part1.controls_no_replay_minutes-->1.7<!--/v--> minutes of immobility with no replay, from <!--v:part1.controls_no_replay_recordings-->8<!--/v--> simulated
recordings that pause in the same places at the track's end, no output reported
any event: fewer than <!--v:part1.controls_no_replay_rate_bound-->1.8<!--/v--> per minute, at 95% confidence, and only there.
**(m)** Two 60 cm parts, the second starting the given distance behind where
the first ended, with no gap in time, pooled over 120 and 200 ms parts, their
far end held 45 cm from where the animal rests whatever the jump; bars give how
each detector carved the event (vermilion outline the sliding window under the
published definition, blue our non-local events), and the numbers above them
the share of parts recovered. Our detector reports one event over both parts in
<!--v:part1.controls_jump_ours_spanning_min|pctnum-->73<!--/v-->% to <!--v:part1.controls_jump_ours_spanning_max|pctnum-->85<!--/v-->% of trials at every jump; the sliding window does so in up to <!--v:part1.controls_jump_sliding_spanning_max|pctnum-->79<!--/v-->% of
trials for jumps up to 40 cm and never across 80 cm. **(n, o)** Two 150 ms, 60
cm replays with a silent gap, the second continuing from the first's end (n) or
starting 140 cm away from it (o), both reaching no nearer the animal than 20
cm. Continuing, at a 10 ms gap both detectors report one event over both parts
(<!--v:part1.controls_continuing_sliding_spanning_shortest_fraction|pctnum-->71<!--/v-->% of trials for the sliding window, <!--v:part1.controls_continuing_ours_spanning_shortest_fraction|pctnum-->88<!--/v-->% for ours); at 200 ms neither does,
and they report one event per part in <!--v:part1.controls_continuing_sliding_one_per_part_longest_fraction|pctnum-->54<!--/v-->% and <!--v:part1.controls_continuing_ours_one_per_part_longest_fraction|pctnum-->79<!--/v-->%. Far apart, the sliding
window never reports one event over both, and misses a part in <!--v:part1.controls_elsewhere_sliding_missing_min|pctnum-->38<!--/v-->% to <!--v:part1.controls_elsewhere_sliding_missing_max|pctnum-->58<!--/v-->% of
trials; our detector reports one event per part in <!--v:part1.controls_elsewhere_ours_one_per_part_shortest_fraction|pctnum-->17<!--/v-->% of trials at a 10 ms gap
and <!--v:part1.controls_elsewhere_ours_one_per_part_longest_fraction|pctnum-->83<!--/v-->% at 200 ms. Each part is also decoded alone, with exactly the spikes it
had in its event and the recording's own spikes around it; a part counts as
recovered only where the event's positions over that part sit on its path.
Across jumps and chains, pooled recovery is higher inside the event than alone:
the sliding window recovers <!--v:part1.controls_sliding_recovery_in_event_fraction|pctnum-->72<!--/v-->% of parts inside the event against <!--v:part1.controls_sliding_recovery_alone_fraction|pctnum-->61<!--/v-->% alone,
and ours <!--v:part1.controls_ours_recovery_in_event_fraction|pctnum-->89<!--/v-->% against <!--v:part1.controls_ours_recovery_alone_fraction|pctnum-->86<!--/v-->%. Paired by trial, the gain is not uniform: the sliding
window recovers fewer parts inside the event than alone in <!--v:part1.controls_sliding_fewer_in_event_trials-->15<!--/v--> of <!--v:part1.controls_multipart_trials-->432<!--/v--> trials and
more in <!--v:part1.controls_sliding_more_in_event_trials-->95<!--/v-->, ours fewer in <!--v:part1.controls_ours_fewer_in_event_trials-->2<!--/v--> and more in <!--v:part1.controls_ours_more_in_event_trials-->22<!--/v-->.

**(p, q) Held-out evidence.** The simulated cells are split into discovery and
evaluation cells three ways: two interleaved folds with their roles swapped (A,
B), each half tiling the track, and one whose evaluation cells cover only its
lower half (C, hatched). Both detectors are fitted and run on the discovery
cells alone, over every injected replay, controls trial and no-replay stretch.
Events are counted once each, as our detector bounds them. An event's content,
estimated from the discovery cells by our decoder (bars) or by the sliding
window (dots), is scored on the evaluation cells against the animal's own
position and against 100 permutations of the evaluation fields, spikes and
silence both counting, and has support when it beats both (p ≤ 0.05). **(p)**
Share of events with support, among those along whose content the evaluation
cells are expected to fire at least 5 spikes; numbers give how many. True
replays that only our detector reported have support in <!--v:part1.heldout_ours_only_support_min|pctnum-->93<!--/v-->% to <!--v:part1.heldout_ours_only_support_max|pctnum-->97<!--/v-->% of cases in
folds A and B and <!--v:part1.heldout_ours_only_support_fold_c|pctnum-->87<!--/v-->% in fold C; shared ones in <!--v:part1.heldout_shared_support|pctnum-->100<!--/v-->%. No informative place-less
burst event has support (<!--v:part1.heldout_placeless_informative_events-->19<!--/v--> of them), and at a floor of 2 expected spikes <!--v:part1.heldout_placeless_loose_supported_events-->4<!--/v--> of
<!--v:part1.heldout_placeless_loose_events-->75<!--/v--> do; jumping bursts, which do represent places, have support in <!--v:part1.heldout_jumping_support_min|pctnum-->57<!--/v-->% to <!--v:part1.heldout_jumping_support_max|pctnum-->63<!--/v-->%.
Neither detector reported an event from discovery cells on the no-replay
stretches, so no event of pure local coding reached the test. The two content
estimates agree on <!--v:part1.heldout_estimator_agreement_min|pctnum-->99<!--/v-->% of the events informative under both, or more. The
sliding window reported no event our detector missed, so there is no
comparator-only class to test. **(q)** Share of each kind of replay the
evaluation cells can judge at all: <!--v:part1.heldout_ours_only_informative_min|pctnum-->6<!--/v-->% to <!--v:part1.heldout_ours_only_informative_max|pctnum-->7<!--/v-->% of those only our detector
reported, <!--v:part1.heldout_shared_informative_min|pctnum-->16<!--/v-->% to <!--v:part1.heldout_shared_informative_max|pctnum-->23<!--/v-->% of shared ones, in folds A and B. The rest are too short
or too sparse for a dozen evaluation cells, so an event without support is
mostly one that cannot be judged, not one shown false. Held-out support tests
spatial information, not sequence, and is read with these error rates, not
alone.

**Caveats.** Sorted spikes, a linear track, no artifacts. In (g-k) the animal
rests at one end of the track and pauses nowhere else, so distance from its
place is fixed by where the replay is put and cannot be varied independently.
Replays run at constant speed with Poisson firing from the same place fields
the encoding model was fitted on, so there is no model mismatch and the picture
is optimistic. Every cell of (g-k) holds a replay, so those panels measure how
often each detector reports a replay that is there; (l) is where they are
tested on activity that holds none. The significance test is the reference's
place-cell identity shuffle, which needs sorted cells; recorded data, decoded
clusterless, are tested with an encoding-pair shuffle instead.

Produced by `scripts/figures/generate_memoryless_comparison.py` from
`docs/design/replay_detection_comparison/detection_sweep_summary.csv`.
