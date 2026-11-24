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

What the lead decided (09:47 to 09:55): design C, with a day model and an animal
model, both run and compared; design D later, as a check; the decision changes
if the evidence is against it. It is consistent across the behavior models
("rate, content, etc", 02:10) and does not include the mPFC models for now
(02:14). Design C is one fit for each unit, in which each session keeps a
baseline of its own (the intercept of a rate model, the arm effects of a content
model) and the tested terms are shared.

What the author decided or read, not the lead's words:

- Fixed, declared penalties: the author's decision, on the lead's "Should we
  keep the fixed penalty?" / "Ok decide then" (03:18, 03:20), from the evidence
  in the design note, section 11.2. The weak penalty stays on every simple
  tested coefficient (decided by the lead, 2026-11-23).
- The curves are among the terms shared inside a unit: the author's reading.
  The author's message to the lead just before 09:55 said that design C shares
  the curves inside a unit, that the lead had declined "one curve for each
  animal" on 2026-11-21 for the per-session design, and that the author read
  the lead's messages of that morning as changing that for design C. The lead's
  next message was the 09:55 one above. The rate model's clock curves are the
  point the running scenario may change (design note, section 11.4).
- Which models: that "etc" means the choice, stay-or-leave and ripple-power
  models is the author's reading, stated to the lead several times and not
  contradicted.

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
