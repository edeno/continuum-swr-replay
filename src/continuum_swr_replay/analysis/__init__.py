"""The analyses, one package per question, in pipeline order.

Where is X: how a non-local event is defined -> ``analysis/event_definition``
(the rule, ``nonlocal_event_rule``; the canonical SWR source, ``swr_source``);
how events are detected -> ``analysis/detection`` (SWRs: ``swr_detectors``;
the non-local detector on a decode: ``nonlocal_events``); how detectors are
compared -> ``analysis/detector_comparison``; whether the decode is
trustworthy -> ``analysis/decode_quality``; whether the events are real ->
``analysis/event_evidence``; what they are like -> ``analysis/event_properties``.

Packages
--------
event_definition
    The rule: the analysed-time rule, immobility, the non-local event rule, the
    SWR source, the decode masks, the classified non-local event table and the
    event cohorts every table reads.
detection
    The algorithms that produce inventories: the SWR and Carey detectors, rule
    R-T, the artifact detector, the memoryless sequence detector and the
    paper's non-local detector applied to a decode.
detector_comparison
    The inventories from different detectors brought onto one substrate and
    compared on the same recordings.
decode_quality
    Is the decode trustworthy: decoding error, state fractions, calibration,
    raw-data flags and posterior summaries.
event_evidence
    Are the events real: held-out spike evidence per event and per decoder
    state.
event_properties
    What the events are like, and how SWR and non-SWR events differ.
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

``event_definition`` imports no other analysis package; then
``decode_quality``, ``detection``, ``event_evidence``, and
``event_properties`` and ``detector_comparison`` each import only those
before them. ``behavior``, ``event_tables`` and ``replay_models`` import in
that order. ``tests/repository/test_import_layering.py`` checks both.
"""

__all__ = []
