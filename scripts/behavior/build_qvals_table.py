#!/usr/bin/env python
"""Per-stop RL-model values via DataJoint — canonical, complete, timestamp-aligned.

Builds the per-stop behavioral-model covariate table (`qvals_table.pkl`),
sourcing everything from Alison's shared DataJoint schemas (read-only):

  * VALUES: `alison_rlmodel.BehaviorModelResults` (the `beta_stable_withleaf`
    fit), read from its AnalysisNwbfile — the q_vals model output re-keyed to
    (nwb_file_name, epoch, trial_number_by_epoch).
  * TIMES: `alison_behav.TrialsInfoByEpoch.ByTrial` — poke_in_ts / poke_out_ts /
    reward_on_ts / reward_off_ts per (nwb_file_name, epoch, trial_number_by_epoch),
    stored as MySQL blobs (complete for all 5 animals, incl. peanut).
  * JOIN: the complete timing table is authoritative. Model/timing trial keys
    are outer-audited; model-only rows are rejected and timing-only rows retain
    their true interval with model values unavailable.
  * STOP -> TRIAL: Alison's rule (alison_decoding.add_trial_info_to_clusterless_results):
    within an epoch, trial t spans [poke_out_ts[t-1], poke_out_ts[t]);
    i.e. trial = searchsorted(poke_out_ts, stop_time, side='right'); stops at or
    after the last poke_out are unassigned (her trailing-NaN drop). Stops in a
    timing-only/model-missing trial are omitted rather than shifted to a later
    model row.

Beta values are enriched with the vendored decision-variable pipeline
(`analysis.behavior._spatial_bandit_rl.make_full_beta_bernoulli_dv_df`: DeltaV,
Stay/Switch value, value-update magnitudes, betadist_mu), the belief entropy
`beta_reward_entropy` is derived from the Beta posterior (`betadist_α/β`), and RPE
is derived; the curated COVARIATES are broadcast to stops. See docs/behavior_model_covariates.md for the data dictionary.

Virtual modules reflect the schema from the DB, so none of Alison's module code
(julia / spyglass / os.chdir) is imported. Independent validation compares the
model's realized destination arm with canonical ``occupancy_well_arm`` and its
reward label with canonical ``is_rewarded_outcome`` wherever that stop-start
outcome is already observable.

Usage: uv run python scripts/behavior/build_qvals_table.py [--animal A] [--write]
The build itself is
``continuum_swr_replay.analysis.behavior.qvals_table.run_build_qvals_table``;
this file is its command line.
Output: Processed-Data/replay_content/models_per_bin_rate/qvals_table.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from continuum_swr_replay.analysis.behavior.qvals_table import (
    RC,
    run_build_qvals_table,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_animals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--animal", choices=analysed_animals(), default=None)
    p.add_argument("--write", action="store_true")
    p.add_argument(
        "--replay-content-dir",
        type=Path,
        default=RC,
        help=(
            "Replay-content root containing stop_table and receiving the qvals "
            "artifact. Pass the same explicit staging root used by "
            "prepare_replay_content_data.py."
        ),
    )
    args = p.parse_args()
    run_build_qvals_table(args)


if __name__ == "__main__":
    main()
