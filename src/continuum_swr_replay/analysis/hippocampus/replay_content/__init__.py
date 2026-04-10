"""Replay content analysis: what spatial locations does hippocampal replay represent?

This subpackage provides a systematic model hierarchy (M0-M7) for analyzing
replay spatial content during immobility using conditional Poisson regression
(multinomial-count softmax). The response is per-arm timebin counts and the
model conditions on the event, giving correct standard errors for the
within-event allocation question. Includes behavior prediction following
Gillespie et al. 2021.

Modules
-------
spatial_mapping
    Map decoded positions to arms (track segments) and patches.
event_collapsing
    Collapse timebins to events, compute arm counts, construct long-format data.
predictors
    Compute predictors: time_since_stop, visit_recency, cumulative_reward,
    dist_to_arm.
model_specs
    Model hierarchy definitions (M0-M7) for conditional Poisson framework.
data_preparation
    Session processing pipeline producing count-based and binary long formats.
fitting
    Conditional Poisson (primary), conditional logit (robustness), legacy GLM.
model_comparison
    Nested model AIC/BIC/LRT comparisons.
model_checking
    Event-based cross-validation, Pearson residuals, diagnostics.
behavior_prediction
    Gillespie-style choice prediction analysis.
"""
