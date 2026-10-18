"""Comparing event inventories from different detectors over one substrate.

Each detector's events brought onto one schema, the time each detector
observed, every overlap between inventories, and the counts built from them,
with per-detector denominators.

- ``intervals``: arithmetic on sets of closed intervals.
- ``schema``: the common inventory table and its builders.
- ``availability``: what each detector observed, and its exposure.
- ``overlap``: overlap pairs, per-event membership, directional agreement,
  time coverage, and removal of events bridging their own missing data.
- ``components``: connected components of the overlap graph.
- ``recount``: one detector's events recounted with nearby fragments merged.
- ``cells``: events no detector of a set found, and time no detector found.
- ``swr_matrix``: each sequence cohort labelled under each SWR definition.
- ``adapters``: each detector's recorded outputs for an epoch, read onto the
  schema with what it observed and the events it set aside.
- ``features``: per-event features read on the decode grid by one rule for
  every inventory: rows, population engagement, LFP and MUA, timing, and the
  stop holding the onset.
- ``evidence``: the common spike-evidence diagnostic, how much better a remote
  place explains each row's spikes than the animal's own place, and its
  summary over each interval's rows.
"""
