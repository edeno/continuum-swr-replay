"""Reported values: every number the project reports, read from a computed file.

Each part of the paper (``docs/analysis_map.md``) has one summary file,
``paper/reported/part{n}.json``, written by ``scripts/reported/summarize_part.py``
from outputs the analyses already wrote. ``scripts/reported/emit_reported_values.py``
turns the summaries into ``paper/reported_values.tex`` and fills the inline value
markers in the claims table and the results docs. Rounding happens only in
:mod:`continuum_swr_replay.reported.formatting`. Design:
``docs/design/2026-10-26-reported-values-design.md``.
"""
