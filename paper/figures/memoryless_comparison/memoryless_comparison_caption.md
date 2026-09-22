# memoryless_comparison_caption

**Two replay detectors on replays whose shape is known.** The sliding-window
detector decodes each 80 ms window on its own, with no model of how one window
follows the next, and keeps or drops windows on the spread of their posterior
and on the jump to the next window. The state-space detector infers the whole
event at once and carries the animal's own place as a competing hypothesis.
Throughout, both detectors read one encoding model, fitted once.

**(a-e) One recording with labeled events.** The simulation carries a
continuous outbound replay, a fragmented replay, a continuous inbound replay
and a silent stretch; a remote stationary event, a 30 cm replay and an 80 ms
replay were written into the spike trains here and are marked with a star.

**(a)** The whole recording. Green is where the animal is; dots are the decoded
place of each window, kept or rejected; yellow marks a labeled event; bars
below the axis are the events each detector reports and the candidates the
sliding-window detector rejected.

**(b-e)** Four events, with the posterior in grey. The sliding-window detector
finds the continuous replay and gives it 5.8 times its true length, because 74%
of the event it reports is the animal's own place, absorbed by the merge rule.
It also finds the 80 ms replay and reports it 1.3 times too long, with 0% local
coding in it: the window alone puts a floor under any duration it can report.
It assembles a candidate from the 30 cm replay and another from the remote
stationary event and declines both for too little dispersion (6 and 2 cm
against the 12 cm it requires). The state-space detector reports all four, with
every bound within 42 ms of the truth.

**(f-i) 4536 replays of chosen shape.** Made-up replays were written into the
spike trains of a simulated linear-track recording, one at a time, each with a
path length, duration, firing rate and distance from the animal's own place
that we chose. Because the right answer is known, what each detector reports
can be scored against it. A detector counts as reporting a replay only when
what it returns is that replay: on its path, and not covering far more ground.

**(f, g)** Percent of those 4536 replays reported, by the path the replay
travels and how long it lasts. Each detector runs as published, with its own
acceptance criteria, and both are scored the same way against the same known
answer. The sliding-window detector reports 24% of the replays and the
state-space detector 91%. 67% were reported by the state-space detector and
missed by the sliding-window detector; 0.04% went the other way.

The two detectors reach their limits on different quantities. The state-space
detector is limited by duration, which is spike count, which is the data. The
sliding-window detector is limited by path length, through its own dispersion
criterion: a replay must spread more than 12 cm about its own centre, and a 30
cm path cannot, however well it is decoded. Held to that same criterion our
state-space detector also reports nothing below 45 cm, so this is the published
replay definition acting, not the window-by-window inference. On the part of
the grid that definition can accept at all — a path over about 42 cm and a
duration over 100 ms, 25% of the cells — the two read 70% and 99%.

**(h)** Reported duration over actual duration, for replays both detectors
reported. A 40 ms replay is reported as about 232 ms — the same floor that
panel (c) shows for one event.

**(i)** The same ratio against how far the replay sits from the animal's own
place. Numbers give the share of the reported event's 80 ms windows that decode
the animal's place rather than a replayed one. That absorption of local coding
reaches about 80 cm from the animal and is what drives the largest
over-reporting, as in panel (b); beyond it the window floor of panel (h) still
leaves the reported duration too long. Over the whole sweep, 30% of the
sliding-window detector's accepted events are not the replay that was put in.

**Caveats.** Sorted spikes, a linear track, no artifacts. In (f-i) the animal
rests at one end of the track and pauses nowhere else, so distance from its
place is fixed by where the replay is put and cannot be varied independently.
Replays run at constant speed with Poisson firing from the same place fields
the encoding model was fitted on, so there is no model mismatch and the picture
is optimistic. The significance test the published method requires is not
applied here.

Produced by `scripts/figures/generate_memoryless_comparison.py` from
`docs/design/replay_detection_comparison/detection_sweep_summary.csv`.
