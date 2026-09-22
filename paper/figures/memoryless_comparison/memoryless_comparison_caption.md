# memoryless_comparison_caption

**Two replay detectors on replays whose shape is known.** The sliding-window
detector decodes each 80 ms window on its own, with no model of how one window
follows the next, and keeps or drops windows on the spread of their posterior
and on the jump to the next window. The state-space detector infers the whole
event at once and carries the animal's own place as a competing hypothesis.
Throughout, both detectors read one encoding model, fitted once, and the
sliding-window detector is held to the whole published definition, its
significance test included: each accepted sequence is decoded again 100 times
with the place fields swapped between cells, and must beat those draws on both
its mean posterior spread and its mean jump.

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
replay, at 1.3 times its length with 0% local coding in it: the window alone
puts a floor under any duration it can report. The significance test rejects it
(p = 0.079 on spread and 0.020 on jump, against 0.05), 3 draws from passing,
and the published method would not report it. It assembles a candidate from the
30 cm replay and another from the remote stationary event and declines both for
too little dispersion (6 and 2 cm against the 12 cm it requires). The
state-space detector reports all four, with every bound within 42 ms of the
truth. Both continuous replays pass the significance test, though most of each
reported event is the animal's own place: the test asks whether a sequence is
spatially coherent, not whether it is somewhere other than the animal.

**(f-i) 4536 replays of chosen shape.** Made-up replays were written into the
spike trains of a simulated linear-track recording, one at a time, each with a
path length, duration, firing rate and distance from the animal's own place
that we chose. Because the right answer is known, what each detector reports
can be scored against it. A detector counts as reporting a replay only when
what it returns is that replay: on its path throughout, and not covering far
more ground than the path does.

**(f, g)** Percent of those 4536 replays reported, by the path the replay
travels and how long it lasts. Each detector runs as published, with its own
acceptance criteria, and both are scored the same way against the same known
answer. The sliding-window detector reports 11% of the replays under the
published definition: 22% before its significance test, and 14% with the spread
half alone, the only half the detection script's own filter could fail; the
paper's analyses read both halves, from a corrected script. Of the replays it
reported before the test, 66% pass the spread half and 81% the jump half; a
short, fast replay jumps far between windows and fails the second. The
state-space detector reports 91%. 80% were reported by the state-space detector
and missed by the sliding-window detector; 0.00% went the other way.

The two detectors reach their limits on different quantities. The state-space
detector is limited by duration, which is spike count, which is the data. The
sliding-window detector is limited by path length, through its own dispersion
criterion: a replay must spread more than 12 cm about its own centre, and a 30
cm path cannot, however well it is decoded. Held to that same criterion our
state-space detector also reports nothing below 45 cm, so this is the published
replay definition acting, not the window-by-window inference. On the part of
the grid that definition can accept at all — a path over about 42 cm and a
duration over 100 ms, 25% of the cells — the two read 37% (64% before the
significance test) and 99%.

**(h)** Reported duration over actual duration, for replays both detectors
reported, the sliding window's before its significance test, which never moves
a sequence's bounds. A 40 ms replay is reported as about 188 ms — the same
floor that panel (c) shows for one event.

**(i)** The same ratio against how far the replay sits from the animal's own
place. Numbers give the share of the reported event's 80 ms windows that decode
the animal's place rather than a replayed one. That absorption of local coding
reaches about 80 cm from the animal and is what drives the largest
over-reporting, as in panel (b); beyond it the window floor of panel (h) still
leaves the reported duration too long. Over the whole sweep, 36% of the
sliding-window detector's accepted events are not the replay that was put in.

**Caveats.** Sorted spikes, a linear track, no artifacts. In (f-i) the animal
rests at one end of the track and pauses nowhere else, so distance from its
place is fixed by where the replay is put and cannot be varied independently.
Replays run at constant speed with Poisson firing from the same place fields
the encoding model was fitted on, so there is no model mismatch and the picture
is optimistic. The significance test is the reference's place-cell identity
shuffle, which needs sorted cells; recorded data, decoded clusterless, are
tested with the encoding-pair shuffle of spec 02 instead.

Produced by `scripts/figures/generate_memoryless_comparison.py` from
`docs/design/replay_detection_comparison/detection_sweep_summary.csv`.
