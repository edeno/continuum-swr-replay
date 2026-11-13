#!/usr/bin/env python
"""Per-stop RL-model values via DataJoint — canonical, complete, timestamp-aligned.

Builds the per-stop behavioral-model covariate table (`qvals_table`),
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
(`analysis.behavior.covariates._spatial_bandit_rl.make_full_beta_bernoulli_dv_df`: DeltaV,
Stay/Switch value, value-update magnitudes, betadist_mu), the belief entropy
`beta_reward_entropy` is derived from the Beta posterior (`betadist_α/β`), and RPE
is derived; the curated COVARIATES are broadcast to stops. See docs/behavior_model_covariates.md for the data dictionary.

Virtual modules reflect the schema from the DB, so none of Alison's module code
(julia / spyglass / os.chdir) is imported. Independent validation compares the
model's realized destination arm with canonical ``occupancy_well_arm`` and its
reward label with canonical ``is_rewarded_outcome`` wherever that stop-start
outcome is already observable.

Usage: uv run python scripts/behavior/build_qvals_table.py [--animal A] [--write]
This file reads the two schemas and runs the build; the table's steps are in
``continuum_swr_replay.analysis.behavior.covariates.qvals_table``.
Output: Processed-Data/replay_content/qvals_table.parquet and
qvals_standardization_reference.json beside it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import datajoint as dj
import pandas as pd

from continuum_swr_replay.analysis.behavior.covariates.qvals_table import (
    ANIMAL_SUBJECT,
    MODEL_KEY,
    QVALS_STANDARDIZED_COLS,
    build_animal,
    prepare_model_values,
    require_complete_mapping,
    standardize_qvals_table,
    stop_mapping_agreement,
    write_qvals_outputs,
)
from continuum_swr_replay.analysis.event_definition.analysed_time import (
    require_swr_source,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_animals
from continuum_swr_replay.data_loaders.utils import quiet_spyglass
from continuum_swr_replay.paths import replay_content_dir, table_path
from continuum_swr_replay.tables.covariate_registry import require_covariate_schema
from continuum_swr_replay.tables.io import read_table


def _fetch_model_df(rl: Any, subject: str, key: dict) -> pd.DataFrame:
    """Read a BehaviorModelResults dataframe (per-trial, keyed by KEYS)."""
    # Deferred import: pulling spyglass at module top opens a DataJoint
    # connection on import, which would make --help and the tests that import
    # this file require live DB access.
    from spyglass.common import AnalysisNwbfile
    from spyglass.utils.nwb_helper_fn import get_nwb_file

    quiet_spyglass()

    sel = rl.BehaviorModelResults & {"subject_id": subject, **key}
    # Resolve the AnalysisNwbfile by hand rather than via ``fetch_nwb``. The
    # helper only exists on Spyglass mixin tables, and ``rl`` is a virtual
    # module reflected from Alison's schema, whose tables are plain DataJoint.
    recs = sel.fetch(
        "analysis_file_name", "behavior_model_results_object_id", as_dict=True
    )
    if not recs:
        raise ValueError(f"no BehaviorModelResults for {subject!r} with {key}")
    if len(recs) > 1:
        raise ValueError(
            f"{len(recs)} BehaviorModelResults rows for {subject!r} with {key}; "
            "expected exactly one -- the model key no longer identifies a "
            "single fit, so which values get used would be arbitrary"
        )
    rec = recs[0]
    path = AnalysisNwbfile.get_abs_path(rec["analysis_file_name"])
    obj = get_nwb_file(path).objects[rec["behavior_model_results_object_id"]]
    return obj.to_dataframe().reset_index(drop=True)


def _fetch_trial_times(beh: Any, nwb_file_names: list[str]) -> pd.DataFrame:
    """``TrialsInfoByEpoch.ByTrial`` for the given NWB files."""
    return pd.DataFrame(
        (
            beh.TrialsInfoByEpoch.ByTrial
            & [{"nwb_file_name": n} for n in nwb_file_names]
        ).fetch()
    )


def run_build_qvals_table(args: argparse.Namespace) -> None:
    """Build every requested animal's qvals rows, validate, and optionally publish."""
    replay_content_dir = args.replay_content_dir.resolve()
    animals = [args.animal] if args.animal else list(analysed_animals())

    beh = dj.create_virtual_module("alison_behav", "alison_behav")
    rl = dj.create_virtual_module("alison_rlmodel", "alison_rlmodel")
    stop_path = table_path("stop_table", replay_content_dir)
    st_all = read_table(stop_path)
    require_covariate_schema(st_all, table="stop")
    require_swr_source(st_all, what=str(stop_path))

    parts = []
    for a in animals:
        vals, model_cols = prepare_model_values(
            _fetch_model_df(rl, ANIMAL_SUBJECT[a], MODEL_KEY)
        )
        times = _fetch_trial_times(beh, vals["nwb_file_name"].unique().tolist())
        t = build_animal(a, vals, model_cols, times, st_all)
        st = st_all[st_all.animal == a]
        agreement = stop_mapping_agreement(t, st)
        arm_ok = agreement["arm_ok"]
        rew_ok = agreement["rew_ok"]
        rew_share = agreement["rew_share"]
        entropy_cov = agreement["entropy_cov"]
        n_trial_gaps = t.attrs["n_timing_trials_without_model"]
        n_gap_stops = t.attrs["n_stops_missing_model_trial"]
        n_before_gap = t.attrs["n_model_trials_before_timing_gap"]
        n_after_gap = t.attrs["n_model_trials_after_timing_gap"]
        print(
            f"{a:7s} mapped {len(t):6d}/{len(st):6d} stops ({100 * len(t) / len(st):.0f}%)  "
            f"in_poke={100 * t['in_poke'].mean():.0f}%  "
            f"at-well arm=={100 * arm_ok:.1f}%  "
            f"at-well reward=={100 * rew_ok:.1f}% (on {100 * rew_share:.0f}% w/ known outcome)  "
            f"[leaf/reward join exact  entropy {100 * entropy_cov:.0f}%  "
            f"model gaps={n_trial_gaps} trials/{n_gap_stops} stops; "
            f"masked neighbors={n_before_gap} before/{n_after_gap} after]"
        )
        if n_trial_gaps:
            print(
                "  timing-only model trials by session: "
                f"{t.attrs['timing_trials_without_model_by_session']}"
            )
        require_complete_mapping(a, agreement)
        parts.append(t)
    tab = pd.concat(parts, ignore_index=True)
    print(
        f"\ntotal mapped stops: {len(tab)}  unique: {tab.stop_id.nunique()}  cols: {tab.shape[1]}"
    )

    tab, qv_reference = standardize_qvals_table(tab)
    z_cols = [f"{c}_z" for c in QVALS_STANDARDIZED_COLS if c in tab.columns]

    if args.write:
        out = write_qvals_outputs(tab, qv_reference, replay_content_dir)
        print(f"\nwrote {len(tab)} rows ({len(z_cols)} _z cols) -> {out}")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--animal", choices=analysed_animals(), default=None)
    p.add_argument("--write", action="store_true")
    p.add_argument(
        "--replay-content-dir",
        type=Path,
        default=replay_content_dir(),
        help=(
            "Replay-content root containing stop_table and receiving the qvals "
            "artifact. Pass the same explicit staging root used by "
            "prepare_replay_content_data.py."
        ),
    )
    args = p.parse_args(argv)
    run_build_qvals_table(args)


if __name__ == "__main__":
    main()
