# Content estimators compared

**Panels.** `coef_agreement.png` — one row per model rung, showing that rung's
key coefficient for every session that all three estimators fit. The x axis is
the retired conditional Poisson on per-arm timebin counts; the left column's y
axis is the shipped conditional Poisson on per-event allocation fractions
(event-clustered standard errors), the right column's is the
Dirichlet-multinomial on the same counts with an `is_swr` concentration
modifier. The dashed line is the identity. Colour identifies the animal. All
three estimators were fit on one shared per-(session, rung) event set, so a
departure from the identity is a difference of likelihood, not of cohort.
Quasi-separated fits (a term flagged separated, or |coefficient| ≥ 10) are
excluded.

`log_phi_by_session.png` — the Dirichlet-multinomial concentration for the
`C2_base` rung. **Top,** the log concentration intercept, i.e. how tightly a
non-SWR event's timebins cluster on a few arms; **bottom,** the additive shift
of that log concentration on SWR events, where a value below zero means SWR
events are *more* concentrated (more redundant bins) and above zero means their
mass is spread over more arms. Each line is one animal, plotted in session order
within that animal. The grey band is the animal bootstrap's 2.5–97.5 percentile
interval and the grey line its median: the animals present are resampled with
replacement for 2000 draws at **seed 0**, and each draw's statistic is the median
over the resampled animals of each animal's per-session median.
