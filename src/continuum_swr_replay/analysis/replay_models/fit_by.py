"""The units every behavior model is fitted at: one declaration.

:data:`BEHAVIOR_FIT_BY` is the units of the fit (``tables.session_keys.FitBy``)
at which the rate, content, choice, stay-or-leave and ripple-power models are
fitted and compared: one fit for each recording day and one for each animal,
each session with a baseline of its own (design C). The mPFC models do not
read it. ``"session"`` stays a value of ``FitBy`` (the earlier design, kept for
the validation and for comparison) and the default of ``fit_sessions``; a
model's settings carry one ``fit_by`` for each run.

The decision record (the lead's words with their times, what the lead decided,
and what is the author's decision or reading) is in
``docs/design/2026-11-24-unit-of-fit.md``, section 11.1.
"""

from __future__ import annotations

from continuum_swr_replay.tables.session_keys import FitBy

__all__ = ["BEHAVIOR_FIT_BY"]

#: The units every behavior model is fitted at, in this order (module
#: docstring).
BEHAVIOR_FIT_BY: tuple[FitBy, ...] = ("day", "animal")
