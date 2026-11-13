"""What the animal did, and what the behavioral model says about it.

The first of three packages in data-flow order: ``behavior`` ->
``event_tables`` -> ``replay_models``. Each imports only those before it, so
nothing here imports ``event_tables`` or ``replay_models``.

- ``covariates``: the behavioral covariates (trial clocks, per-timebin trial
  context, reward and visit history, value and uncertainty, the candidate-arm
  expansion and the SWR interactions), their catalog (``definitions``), and the
  behavioral-model values (``qvals_table``) they join.
- ``design_constants``: the numbers the covariate builders and the model
  ladders share (bin width, history lags, spline knots, the arm fixed-effect
  names), so neither side imports the other.
- ``history_basis``: the raised-cosine history basis and its projection, read
  by the reward-history covariates and the per-bin rate design.
- ``spatial_mapping``: decoded position to arm and patch, and the well, arm
  and patch lookups on the track graph (``single_arm_from_well``,
  ``arm_to_patch``, ``stay_indicator``, ``build_well_to_arm_map``).

The package ``__init__`` imports nothing: import a name from the module that
defines it.
"""
