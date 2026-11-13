"""The canonical event, arm, patch, stop and time-bin tables and their contracts.

The second of three packages in data-flow order: ``behavior`` ->
``event_tables`` -> ``replay_models``. It imports ``behavior`` and never
``replay_models``.

- ``prepare_data``: the per-session table build (``build_session_tables``) and
  the corpus-level steps around it (``assemble_canonical_tables``,
  ``standardize_event_arm``); ``scripts/tables/`` runs it.
- ``event_table``: one row per non-local event (``build_event_table``) and the
  in-event time bins (``build_event_sequence_table``).
- ``arm_patch_tables``: one row per (event, arm) and (event, patch)
  (``build_event_arm_table``, ``build_event_patch_table``).
- ``stop_table``: one row per stopping period (``build_stop_table``).
- ``timebin_tables``: the within-stop time-bin and event-onset tables.
- ``bouts``: immobility bouts, where a stop begins and ends.
- ``summaries``, ``uncertainty``, ``trajectory``: the per-event posterior
  summaries, decoder-confidence summaries and trajectory rollups.
- ``retrospective_metrics``: the per-event alignment of content with the next
  and previous arm, merged into the event table.
- ``content_response``: which per-event response the content ladders are fit
  on.
- ``table_contracts``: the scientific contracts each table is validated
  against.
- ``_common``: the parts the builders share (column orders, posterior and
  response-matrix alignment, NaN-strict reducers).
"""
