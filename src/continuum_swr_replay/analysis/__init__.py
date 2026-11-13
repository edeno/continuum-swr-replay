"""The analyses, one package per subject.

How a non-local event is defined -> ``analysis/event_definition``; how SWRs are detected
-> ``analysis/detection/swr_detectors``; how detectors are compared ->
``analysis/detector_comparison``.

Packages
--------
events
    The analysed-time rule, the SWR source, event eligibility, the classified
    non-local event table and the event cohorts every table reads.
hippocampus
    Decode masks and aggregation, posterior and quality metrics, the SWR
    detectors and inventories, and the artifact detector.
inventory_comparison
    Event inventories from different detectors brought onto one substrate and
    compared.
behavior
    What the animal did and what the behavioral model says: the covariates,
    the behavioral-model values, the design constants and the track lookups.
event_tables
    The canonical event, arm, patch, stop and time-bin tables and their
    contracts.
replay_models
    The Part 3 rate, content, coupling and choice models and the fitting
    machinery they share.
contfrag
    The Continuous-Fragmented classifier analysis and its tables.
pfc
    mPFC and OFC unit analysis.

Modules
-------
constants
    Shared analysis constants and default parameters.
shared_statistics
    Task-agnostic statistics (signed-rank p, Benjamini-Hochberg) any package
    may use.
literature_survey
    Statistics of the published replay studies surveyed in Supplementary
    Table S1.

``behavior``, ``event_tables`` and ``replay_models`` import in that order:
each imports only those before it.
"""

__all__ = []
