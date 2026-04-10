"""Replay content analysis: what spatial locations does hippocampal replay represent?

This subpackage provides a systematic model hierarchy (M0-M7) for analyzing
replay spatial content during immobility, from simple proximity models to
full mixed-effects models with temporal dynamics, reward history, and SWR
interactions. Includes behavior prediction following Gillespie et al. 2021.

Modules
-------
spatial_mapping
    Map decoded positions to arms (track segments) and patches.
event_collapsing
    Collapse timebins to events and construct long-format data.
predictors
    Compute new predictors: time_since_stop, visit_recency, cumulative_reward.
model_specs
    Model hierarchy definitions (M0-M7).
data_preparation
    Session processing pipeline for data prep.
fitting
    Per-session GLM and mixed-effects fitting.
model_comparison
    Nested model AIC/BIC/LRT comparisons.
model_checking
    Cross-validation, separation detection, diagnostic aggregation.
behavior_prediction
    Gillespie-style choice prediction analysis.
"""
