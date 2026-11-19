"""Are the events real: spike evidence that did not choose the decode.

- ``spike_evidence``: per non-local event, whether the prior-excluded spike
  likelihood favors the decoded location over the animal's.
- ``state_evidence``: held-out spike evidence for each decoder state, its
  windows, terms and summaries.

Imports ``event_definition``, ``detection`` and ``decode_quality``.
Submodules are imported by name; nothing is re-exported here.
"""
