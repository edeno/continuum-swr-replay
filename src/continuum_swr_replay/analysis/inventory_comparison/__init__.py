"""Comparing event inventories from different detectors over one substrate.

Each detector's events brought onto one schema, the time each detector
observed, every overlap between inventories, and the counts built from them,
with per-detector denominators.

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
  every inventory: rows, population engagement, LFP and MUA, timing, the
  stop holding the onset, and the posterior in two namespaces, the
  state-space decode's (``ss_``) and the memoryless windows' (``ms_``).
- ``evidence``: the common spike-evidence diagnostic, how much better a remote
  place explains each row's spikes than the animal's own place, and its
  summary over each interval's rows.
- ``populations``: the comparison's memberships rebuilt from its stored
  tables, and the populations of compared events they define.
- ``selection``: matched baseline windows for the evidence score, drawn from
  the shared exposure outside every event.
- ``kay_version``: the Kay version check, a direct Kay inventory's events
  matched against the canonical Spyglass ones.
- ``inventory_run``: the inventory comparison run's roster, comparisons, decode
  grid and table layout, which the runner and every reader of its runs share.
- ``epoch_comparison``: one epoch's inventories compared, every table the
  inventory comparison run writes for it.
- ``feature_run``: the event features run's tables, and the comparison's and
  memoryless run's tables its readers read beside it.
- ``pooling``: sessions pooled within an animal, the rule of every per-animal
  table of the comparison's reports, and the counts, cells and directional
  agreement tables built by it.
- ``heldout``: held-out evidence for an event, scored on evaluation cells, and
  its summary on simulated data; ``heldout_sweep`` runs that simulation.
"""
