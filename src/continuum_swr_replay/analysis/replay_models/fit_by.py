"""The units every behavior model is fitted at: one declaration.

Decision record (``docs/design/2026-11-24-unit-of-fit.md``, sections 2 and 11.1).
The lead's words, 2026-11-24:

- 02:09 and 02:10: "I'd like to be consistent with what we do with the behavior
  models" / "By behavior models, I mean rate, content, etc"
- 02:14: "Does not include the mpfc models for now."
- 03:13: "I'm inclined to do C and then attempt D as we are doing C, but we will
  see if it is prohibitive"
- 03:33: "Maybe we make it easy in the code to do day or animal grouping"
- 09:47: "I suspect with will end up doing design C for all. And then running
  both the day and animal model and compare"
- 09:49: "So maybe design C with day and animal model and then design D later to
  check."
- 09:55: "I did say maybe, lets just make this the decision now and we can change
  if the evidence is against it"

What is decided: design C. There is one fit for each unit. Each session in the
unit keeps a baseline of its own (the intercept of a rate model, the arm effects
of a content model). The tested terms, the curves included, are shared inside
the unit. The penalties are fixed and declared. The weak penalty stays on every
simple tested coefficient. Every behavior model is fitted at the day unit and at
the animal unit, and the two are compared.

The decision applies to the rate, content, choice, stay-or-leave and
ripple-power models. It does not apply to the mPFC models (02:14). Design D (random
session effects with strengths selected by REML) is a later check on the same
mechanism. The decision changes if the evidence is against it (09:55).

``"session"`` stays a value of :data:`~continuum_swr_replay.tables.session_keys.FitBy`
(the earlier design, kept for the validation and for comparison); it is not a
unit the behavior models are fitted at. A model's settings carry one ``fit_by``
for each run.
"""

from __future__ import annotations

from continuum_swr_replay.tables.session_keys import FitBy

__all__ = ["BEHAVIOR_FIT_BY"]

#: The units every behavior model is fitted at, in this order (module
#: docstring).
BEHAVIOR_FIT_BY: tuple[FitBy, ...] = ("day", "animal")
