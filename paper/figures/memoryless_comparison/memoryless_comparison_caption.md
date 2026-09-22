# memoryless_comparison_caption

**Two replay detectors on one simulated recording with known events.** The
simulation carries a continuous outbound replay, a fragmented replay, a
continuous inbound replay and a silent stretch; a remote stationary event, a
30 cm replay and an 80 ms replay were written into the spike trains here and
are marked with a star. Both detectors read one encoding model, fitted once.
The sliding-window detector decodes each 80 ms window on its own, with no model
of how one window follows the next, and keeps or drops windows on the spread of
their posterior and on the jump to the next window.

**(a)** The whole recording. Green is where the animal is; dots are the decoded
place of each window, kept or rejected; yellow marks a labeled event; bars below
the axis are the events each detector reports and the candidates the
sliding-window detector rejected.

**(b-e)** Four events, with the posterior in grey. The sliding-window detector
finds the continuous replay and gives it
5.8 times its true length, because
74% of the event it reports is the
animal's own place, absorbed by the merge rule. It also finds the 80 ms replay
and reports it 1.3 times too long, this time with
no local coding in it at all: the window alone puts a floor under any duration
it can report. It misses the 30 cm replay and declines the remote stationary
event, both for too little dispersion. The state-space detector reports all
four with bounds within about 20 ms.

**(f)** Every candidate sequence the sliding-window detector built, against the
two criteria that accept one. Dispersion is how far the decoded path spreads
about its own centre, so the shaded box means an event lasted long enough and
travelled far enough. 3 of 32 reach it.

**Caveats.** One simulation, sorted spikes, a linear track, no artifacts. The
significance test the published method requires is not applied here.

Produced by `scripts/figures/generate_memoryless_comparison.py`.
