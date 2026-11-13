"""The analysed-time and event substrate every table reads.

One rule decides which time is analysed and which events are kept, and these
modules hold it, below the hippocampus, behavior, event-table, replay-model,
ContFrag, PFC and inventory-comparison analyses that use it:

- ``swr_source``: the paper's one SWR source, the Kay inventory read with
  its run ids, the artifact mask and the source's name.
- ``analysed_time``: excluded time and the read point that applies the
  analysed-time rule to a decode frame (``with_swr_source``,
  ``artifact_event_drops``).
- ``eligibility``: excluded time and which events lie wholly outside it.
- ``immobility``: the whole-event speed rule and labelled event intervals.
- ``nonlocal_detection``: non-local event detection and SWR-overlap
  classification.
- ``classified_table``: the canonical per-non-local-event table
  (``build_classified_nonlocal_event_table``).
- ``epoch_normalizers``: an epoch's run-speed baseline, recomputed outside
  the artifact mask at the read point.
- ``artifact_masks``: the artifact padding table and the artifact mask.
- ``artifact_intervals``: the artifact detector's per-epoch cores file and
  run manifest.
- ``cohort_selection``: the shared stop and event cohort rules and the
  whole-unit complete-case selection.

The package imports only the foundation modules, ``analysis.constants`` and
itself (``tests/repository/test_import_layering.py``). Submodules are imported by
name; nothing is re-exported here.
"""
