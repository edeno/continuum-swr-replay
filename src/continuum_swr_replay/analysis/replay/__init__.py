"""Task-agnostic replay content analysis package.

Houses the conditional Poisson fitter, model hierarchy, spatial mapping,
simulation utilities, and model comparison/checking helpers used across
the replay content analysis pipeline. Independent of brain area —
designed to be reusable for any decoder + track graph combination that
produces non-local events on a graph of arms.
"""
