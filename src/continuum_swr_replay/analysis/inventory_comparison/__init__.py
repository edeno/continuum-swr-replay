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
"""
