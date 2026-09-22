# memoryless_sweep_caption

**Two replay detectors on replays of known shape.** 4536
made-up replays were written into the spike trains of a simulated linear-track
recording, one at a time, each with a path length, duration, firing rate and
distance from the animal's own place that we chose. Because the right answer is
known, what each detector reports can be scored against it. A detector counts
as reporting a replay only when what it returns is that replay: on its path,
and not covering far more ground. Both detectors read one encoding model, fitted
once.

**(a, b)** Percent of replays reported, by the path the replay travels and how
long it lasts. The sliding-window detector
(24% overall) is limited by path length,
which is one of its criteria: below about 45 cm it reports a few percent at any
duration and any firing rate. The state-space detector
(94%) is limited by duration, which is
spike count, which is the data. No replay was reported by the sliding-window
detector and missed by the state-space detector;
71% went the other way.

**(c)** Reported duration over actual duration, for replays both detectors
reported. A 40 ms replay is reported as
232 ms. The sliding-window
detector cannot report an event shorter than its own window allows.

**(d)** The same ratio against how far the replay sits from the animal's own
place. Numbers give the share of the reported event's 80 ms windows that decode
the animal's place rather than a replayed one. Over-reporting happens only
within about 80 cm of the animal, where local coding is absorbed into the
event. 31% of the
sliding-window detector's accepted events are not the replay that was put in.

**Caveats.** The animal rests at one end of the track and pauses nowhere else,
so distance from its place is fixed by where the replay is put and cannot be
varied independently. Replays run at constant speed with Poisson firing from
the same place fields the encoding model was fitted on, so there is no model
mismatch and the picture is optimistic. Sorted spikes on a linear track.

Produced by `scripts/figures/generate_memoryless_sweep.py` from
`docs/design/replay_detection_comparison/detection_sweep_summary.csv`.
