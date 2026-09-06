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
Unidentified fits are excluded, by the same rule the shipped aggregation uses:
the fitter flagged that term as separated, or the effect-scale magnitude
|coefficient| × SD(predictor) reaches 10. That effect-scale backstop applies to
the concentration modifier `log_phi_is_swr` as well, on the SD of the event-level
`is_swr` flag; `log_phi_intercept` is the one term it cannot apply to, because
its design column is a constant with no SD, so there the separation flag is the
whole criterion. The drop is applied to the whole (session, rung, term) triple,
so the three estimators are always compared over the same sessions.

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
