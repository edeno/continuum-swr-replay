"""The Part 3 replay models and the fitting machinery they share.

The last of three packages in data-flow order: ``behavior`` ->
``event_tables`` -> ``replay_models``. It reads the tables ``event_tables``
builds and the covariates ``behavior`` defines; neither imports it.

Subpackages, none of which imports another:

- ``rate``: the event-rate models, the per-stop (Family A) and per-bin
  (Family B) ladders, their history designs and their statistics.
- ``content``: the content models, which arms the events represent, under the
  three estimators, and their statistics.
- ``coupling``: the per-event SWR-power model.
- ``choice``: next-choice prediction, stay/leave (``stay_leave``),
  switch-aligned content and the retrospective-alignment null.

Modules shared by the subpackages (none of them imports a subpackage):

- ``model_specs``: every model ladder (the single source of truth that
  ``docs/models_table.md`` mirrors) and ``add_arm_fixed_effects``.
- ``per_bin_design``: the (stop x bin) point-process design.
- ``fitting``, ``_fit_diagnostics``: the conditional-Poisson and logit
  fitters and their convergence and separation diagnostics.
- ``common_cohort``, ``cohort_audit``, ``covariate_timing``: one frozen
  complete-case cohort per block, its drop audit, and the pre/post-outcome
  timing check.
- ``contrast_aggregation``, ``coefficient_statistics``, ``fit_comparability``:
  the audited is_swr-contrast aggregation, the across-session coefficient
  statistics, and which rung pairs an information criterion can compare.

The package ``__init__`` imports nothing, so importing a leaf module (e.g.
``replay_models.model_specs``) loads no JAX or statsmodels.
"""
