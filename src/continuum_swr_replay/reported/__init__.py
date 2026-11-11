"""Reported values: every number the project reports, read from a computed file.

Each part of the paper has one summary file, ``paper/reported/part{n}.json``,
written by ``scripts/reported/summarize_part.py`` from outputs the analyses already
wrote, with a record of the code that built it (:mod:`.code_record`).
:mod:`.questions` says which question each value answers and which figure shows
it. ``scripts/reported/emit_reported_values.py`` turns the summaries into
``paper/reported_values.tex`` and ``paper/reported/questions.md``, and fills the
inline value markers in the docs. Rounding happens only in
:mod:`continuum_swr_replay.reported.formatting`.
"""
