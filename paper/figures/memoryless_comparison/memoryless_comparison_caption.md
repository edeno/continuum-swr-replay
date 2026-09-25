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

**(k-n) Activity that is not one clean replay**, written into the same
recording and scored by the same rule; our detector is read both ways, its
non-local events (continuous and fragmented states together) and the continuous
state alone. **(k)** Share of bursts with an event reported on them. A
place-less burst, every cell firing at one rate unrelated to its field, carries
no spatial content, so any event on it is false; a jumping burst represents a
random place every 15 ms, so calling it non-local is right and calling it a
sequence is not. Our detector's non-local events cover 90% of place-less
bursts, which spend 59% to 80% of their time in the fragmented state; the
continuous state alone covers 1%, and the sliding window 8% (15% before its
significance test). These pool the three peak rates, 20, 60 and 160 Hz; at 20
Hz a place-less burst fires about as fast as the local coding it replaces, and
our non-local events still cover 75% of them. On jumping bursts our non-local
events cover 97%, which is right; the continuous state alone covers 50% and the
sliding window 7% (12% before its test), each calling a sequence that is not
there. Over 1.7 minutes of immobility with no replay, from 8 simulated
recordings that pause in the same places at the track's end, no output reported
any event: fewer than 1.8 per minute, at 95% confidence, and only there.
**(l)** Two 60 cm parts, the second starting the given distance behind where
the first ended, with no gap in time, pooled over 120 and 200 ms parts, their
far end held 45 cm from where the animal rests whatever the jump; bars give how
each detector carved the event (vermilion outline the sliding window under the
published definition, blue our non-local events), and the numbers above them
the share of parts recovered. Our detector reports one event over both parts in
73% to 85% of trials at every jump; the sliding window does so in up to 79% of
trials for jumps up to 40 cm and never across 80 cm. **(m, n)** Two 150 ms, 60
cm replays with a silent gap, the second continuing from the first's end (m) or
starting 140 cm away from it (n), both reaching no nearer the animal than 20
cm. Continuing, at a 10 ms gap both detectors report one event over both parts
(71% of trials for the sliding window, 88% for ours); at 200 ms neither does,
and they report one event per part in 54% and 79%. Far apart, the sliding
window never reports one event over both, and misses a part in 38% to 58% of
trials; our detector reports one event per part in 17% of trials at a 10 ms gap
and 83% at 200 ms. Each part is also injected alone, at the time and with the
seed it has in its event: across jumps and chains the sliding window recovers
72% of parts inside the event against 62% alone, and ours 89% against 87%, so
neither loses parts to the jump or the chain itself.

**(o, p) Held-out evidence.** The simulated cells are split into discovery and
evaluation cells three ways: two interleaved folds with their roles swapped (A,
B), each half tiling the track, and one whose evaluation cells cover only its
lower half (C, hatched). Both detectors are fitted and run on the discovery
cells alone, over every injected replay, controls trial and no-replay stretch.
Events are counted once each, as our detector bounds them. An event's content,
estimated from the discovery cells by our decoder (bars) or by the sliding
window (dots), is scored on the evaluation cells against the animal's own
position and against 100 permutations of the evaluation fields, spikes and
silence both counting, and has support when it beats both (p ≤ 0.05). **(o)**
Share of events with support, among those along whose content the evaluation
cells are expected to fire at least 5 spikes; numbers give how many. True
replays that only our detector reported have support in 93% to 97% of cases in
folds A and B and 87% in fold C; shared ones in 100%. No informative place-less
burst event has support (19 of them), and at a floor of 2 expected spikes 4 of
75 do; jumping bursts, which do represent places, have support in 57% to 62%.
Neither detector reported an event from discovery cells on the no-replay
stretches, so no event of pure local coding reached the test. The two content
estimates agree on 99% of the events informative under both, or more. The
sliding window reported no event our detector missed, so there is no
comparator-only class to test. **(p)** Share of each kind of replay the
evaluation cells can judge at all: 6% to 7% of those only our detector
reported, 16% to 23% of shared ones, in folds A and B. The rest are too short
or too sparse for a dozen evaluation cells, so an event without support is
mostly one that cannot be judged, not one shown false. Held-out support tests
spatial information, not sequence, and is read with these error rates, not
alone.

**Caveats.** Sorted spikes, a linear track, no artifacts. In (f-j) the animal
rests at one end of the track and pauses nowhere else, so distance from its
place is fixed by where the replay is put and cannot be varied independently.
Replays run at constant speed with Poisson firing from the same place fields
the encoding model was fitted on, so there is no model mismatch and the picture
is optimistic. Every cell of (f-j) holds a replay, so those panels measure how
often each detector reports a replay that is there; (k) is where they are
tested on activity that holds none. The significance test is the reference's
place-cell identity shuffle, which needs sorted cells; recorded data, decoded
clusterless, are tested with an encoding-pair shuffle instead.

Produced by `scripts/figures/generate_memoryless_comparison.py` from
`docs/design/replay_detection_comparison/detection_sweep_summary.csv`.
