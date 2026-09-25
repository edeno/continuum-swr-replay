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

**(a-e) One recording with labeled events.** The simulation carries a
continuous outbound replay, a fragmented replay, a continuous inbound replay
and a silent stretch; a remote stationary event, a 30 cm replay and an 80 ms
replay were written into the spike trains here and are marked with a star.

**(a)** The whole recording. Green is where the animal is; dots are the decoded
place of each window, kept or rejected; yellow marks a labeled event; bars
below the axis are the events each detector reports and the candidates the
sliding-window detector rejected.

**(b-e)** Four events, with the posterior in grey; bars below the axis are the
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
spatially coherent, not whether it is somewhere other than the animal.

**(f-j) 4536 replays of chosen shape.** Made-up replays were written into the
spike trains of a simulated linear-track recording, one at a time, each with a
path length, duration, firing rate and distance from the animal's own place
that we chose. Because the right answer is known, what each detector reports
can be scored against it. A detector counts as reporting a replay only when
what it returns is that replay: on its path throughout, and not covering far
more ground than the path does.

**(f, g)** Percent of those 4536 replays reported, by the path the replay
travels and how long it lasts; both detectors are scored the same way against
the same known answer. The comparison that isolates detection uses a shared
target: the replays the published definition can accept at all, a path over
about 42 cm and a duration over 100 ms, 25% of the cells, with both detectors
held to that definition's duration and dispersion criteria. There the
sliding-window detector reports 68% before its significance test and the
state-space detector 68%. The significance test, which only the sliding-window
detector is put through, brings it to 63%; the state-space detector's own
acceptance criteria, in place of the published ones, bring it to 99%. Over the
whole grid, each under its own criteria, the two read 18% and 91%; that gap
combines differences in detection with differences in what each counts as a
replay. Over the whole grid the sliding-window detector reports 21% before its
significance test, and 20% with the spread half alone, the only half the
detection script's own filter could fail; the paper's analyses read both
halves, from a corrected script. Of the replays it reported before the test,
97% pass the spread half and 87% the jump half; a short replay covers its path
in few windows, so its own jumps are large, and the jump half passes 31% of 40
ms replays against 98% of 500 ms ones. The test keeps the reference's
conventions where they carry over, counting only draws strictly below the
observed value and passing each half at p ≤ 0.05, and translates its zero-jump
rule to binned centers. 73% were reported by the state-space detector and not
by the sliding-window detector, counting the 4% it reported and then failed on
the significance test; 0.00% went the other way (0.11% before the significance
test).

The two detectors reach their limits on different quantities. The state-space
detector is limited by duration, which is spike count, which is the data. The
sliding-window detector is limited by path length, through its own dispersion
criterion: a replay must spread more than 12 cm about its own centre, and a 30
cm path cannot, however well it is decoded. Held to that same criterion our
state-space detector also reports nothing below 45 cm, so this is the published
replay definition acting, not the window-by-window inference.

**(h, i)** Where each detector puts a replay's start (h) and end (i): reported
minus true, in ms, median line and interquartile band by true duration, over
the replays both detectors reported, the sliding window's before its
significance test, which never moves a sequence's bounds. The sliding window's
bounds are its first and last window midpoints, and its 80 ms windows reach 40
ms beyond each. The state-space detector's median start and end sit within 5 ms
of the truth at every duration. The sliding window starts a median 18 to 28 ms
early and ends 21 to 65 ms late, and its late ends carry a long tail, the upper
quartile reaching 231 ms; panel (j) shows how much of a reported event is the
animal's own place. A 40 ms replay's reported start is a median 28 ms early and
its end a median 65 ms late, and its median reported duration is about 132 ms,
the floor panel (c) shows for one event.

**(j)** Reported duration over actual duration against how far the replay sits
from the animal's own place. Numbers give the share of the reported event's 80
ms windows that decode the animal's place rather than a replayed one. That
absorption of local coding reaches about 80 cm from the animal and is what
drives the largest over-reporting, as in panel (b); beyond it the window floor
of panels (h, i) still leaves the reported duration too long. Over the whole
sweep, 23% of the sliding-window detector's accepted events are not the replay
that was put in.

**Caveats.** Sorted spikes, a linear track, no artifacts. In (f-j) the animal
rests at one end of the track and pauses nowhere else, so distance from its
place is fixed by where the replay is put and cannot be varied independently.
Replays run at constant speed with Poisson firing from the same place fields
the encoding model was fitted on, so there is no model mismatch and the picture
is optimistic. Every cell of (f-j) holds a replay; there is no stretch without
one and no incoherent activity, so the sweep measures how often each detector
reports a replay that is there and not how often it reports one that is not,
and more detection here is not by itself better performance. The significance
test is the reference's place-cell identity shuffle, which needs sorted cells;
recorded data, decoded clusterless, are tested with an encoding-pair shuffle
instead.

Produced by `scripts/figures/generate_memoryless_comparison.py` from
`docs/design/replay_detection_comparison/detection_sweep_summary.csv`.
