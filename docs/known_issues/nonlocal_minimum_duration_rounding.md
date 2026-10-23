# Non-local events: the 15 ms minimum is 7 samples, not 8

**Status:** open, kept as is. Fix at the next full rebuild (lead, 2026-10-18).
**Affects:** non-local (state-space) events only, in every epoch.

## What

`get_confident_times` (`analysis/events/nonlocal_detection.py`) keeps runs of
non-local probability > 0.8 that last at least `minimum_duration=0.015`. The run
length goes through `ripple_detection.core.minimum_sample_count`. That function
rounds halves up, so 15 ms at 500 Hz (7.5 samples) is meant to give 8 samples.

On the decode grid the median time step reads 2.0000935 ms, not 2 ms, because
the timestamps are quantized at a ~1.6e9 s clock. So 0.015 s is 7.49965
samples, which rounds to 7. The function's 1e-6 tolerance is about 300× too
small to absorb this. This holds in all 242 analysed epochs.

So the shortest non-local event is 7 samples: a 12 ms span between first and
last sample, or a 14 ms `native_duration_s`.

## Size

859 of 75,675 state-space events in the Phase 3 inventory (`afeb897d850f`) are
exactly 7 samples: 1.1 % overall, 0.7–1.5 % per animal.

The SWR and MUA detectors are not affected. They run on 1 kHz LFP, where their
minimums are whole numbers of samples, and none of their events fall below their
minimum (shortest: Kay 21 ms, Karlsson 22, Yu 22, Carey 20, MUA 34 ms). Zugaro
has no minimum.

## Fix (next full rebuild)

Make the minimum 8 samples. Either compute the count in `get_confident_times`
instead of through `segment_boolean_series`, or widen the tolerance in
`ripple_detection.minimum_sample_count` (an upstream change; ask the lead first).

The change drops the 7-sample events and renumbers the non-local labels. So it
needs, as one set:
- `data_over_time` rebuilt for all epochs (not a re-decode);
- the atomic event table set (`event_table`, `event_arm_table`,
  `event_sequence_table`, `stop_table`, and the qvals/stop joins);
- the refits that read them;
- the state-space parts of the detector comparison (inventory and event features).

Until then, describe the rule as "at least 7 samples (12 ms span)", not 15 ms.
