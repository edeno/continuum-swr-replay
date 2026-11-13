"""Is the decode trustworthy: the decoder's accuracy, states and posteriors.

- ``quality_metrics``: local decoding error, state fractions, calibration and
  the per-session quality summary (``compute_session_quality_summary``).
- ``hpc_raw_stats``: raw-data statistics and flags of the clusterless decode,
  and the discrete-state transition matrix.
- ``posterior_analysis``: posterior summaries (MAP speed, HPD, the
  conditional non-local posterior).
- ``data_aggregation``: a decode aggregated onto one time-indexed frame with
  behavior (``get_data_over_time``) and its saved form read back
  (``load_data_over_time``).

Imports ``event_definition`` and the foundation. The loader of a saved decode,
``load_results``, is foundation code, in ``decoding.result_loading``.
Submodules are imported by name; nothing is re-exported here.
"""
