"""Is the decode trustworthy: the decoder's accuracy, states and posteriors.

- ``quality_metrics``: local decoding error, state fractions, calibration and
  the per-session quality summary (``compute_session_quality_summary``).
- ``hpc_raw_stats``: raw-data statistics and flags of the clusterless decode,
  and the discrete-state transition matrix.
- ``posterior_analysis``: posterior summaries (MAP speed, HPD, the
  conditional non-local posterior).

Imports ``event_definition`` and the foundation. The loader of a saved decode,
``load_results``, is foundation code, in ``decoding.results``; the decode frame
the metrics read is written by ``detection.nonlocal_events``.
Submodules are imported by name; nothing is re-exported here.
"""
