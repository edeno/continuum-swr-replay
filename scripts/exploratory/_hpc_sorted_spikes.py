"""Standalone: load sorted CA1 (HPC) spike times for j1620210710 02_r1.

Pulls MountainSort4-curated units from the Spyglass v0 spike-sorting schema.
This is independent of the project's data loader, which uses *clusterless*
marks for HPC and only loads sorted units for mPFC/OFC.

Sort identity (per-epoch sort for epoch 02_r1):
    sorter              = mountainsort4
    sorter_params_name  = franklab_tetrode_hippocampus_30KHz
    preproc_params_name = franklab_tetrode_hippocampus_min_seg
    sort_interval_name  = runs_noPrePostTrialTimes raw data valid times_02_r1_ac
    team_name           = ac_em_xs
    curation_id         = 1
    CA1 tetrodes        = sort groups 0-23 excluding 2 & 12 (corpus callosum)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

NWB_FILE = "j1620210710_.nwb"
RESTRICTION = {
    "nwb_file_name": NWB_FILE,
    "sorter": "mountainsort4",
    "sorter_params_name": "franklab_tetrode_hippocampus_30KHz",
    "preproc_params_name": "franklab_tetrode_hippocampus_min_seg",
    "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times_02_r1_ac",
    "team_name": "ac_em_xs",
    "curation_id": 1,
}


def get_hpc_sorted_spike_times(
    restriction: dict,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Fetch sorted CA1 spike times.

    Returns
    -------
    spike_times : list[np.ndarray]
        One array of spike times (seconds) per curated unit, pooled across
        all CA1 tetrodes.
    unit_info : pd.DataFrame
        One row per unit with sort_group_id, unit id, n_spikes, and rate.
    """
    from spyglass.spikesorting.v0 import CuratedSpikeSorting

    rel = CuratedSpikeSorting & restriction
    n = len(rel)
    if n == 0:
        raise ValueError(f"No CuratedSpikeSorting rows for restriction:\n{restriction}")
    print(f"Matched {n} sort-group rows.")

    nwb_rows = rel.fetch_nwb()

    spike_times: list[np.ndarray] = []
    info_rows: list[dict] = []
    for row in nwb_rows:
        if "units" not in row:
            continue
        sg_id = int(row["sort_group_id"])
        units = row["units"]  # pandas DataFrame indexed by unit id
        for unit_id, st in units["spike_times"].items():
            st = np.asarray(st, dtype=np.float64)
            spike_times.append(st)
            duration = (st[-1] - st[0]) if st.size > 1 else np.nan
            info_rows.append(
                {
                    "sort_group_id": sg_id,
                    "unit_id": unit_id,
                    "n_spikes": st.size,
                    "first_s": st[0] if st.size else np.nan,
                    "last_s": st[-1] if st.size else np.nan,
                    "mean_rate_hz": st.size / duration if duration and duration > 0 else np.nan,
                }
            )

    unit_info = pd.DataFrame(info_rows).sort_values(["sort_group_id", "unit_id"])
    return spike_times, unit_info


if __name__ == "__main__":
    spike_times, unit_info = get_hpc_sorted_spike_times(RESTRICTION)

    print(f"\nTotal units: {len(spike_times)}")
    print(f"Total spikes: {sum(s.size for s in spike_times):,}")
    print(f"Sort groups with units: {sorted(unit_info['sort_group_id'].unique())}")
    print("\nPer-unit summary (head):")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(unit_info.head(20).to_string(index=False))
    print("\nUnits per sort group:")
    print(unit_info.groupby("sort_group_id").size().to_string())
