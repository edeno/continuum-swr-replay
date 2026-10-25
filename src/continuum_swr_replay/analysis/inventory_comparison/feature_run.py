"""The event features run's tables, and what it reads beside them.

What ``scripts/detection/run_event_features.py`` writes for each epoch
(:data:`TABLE_DTYPES`: ``features``, :data:`FEATURE_DTYPES`; ``pair_membership``;
``rejection``) and where (:func:`output_path`), which the event features report
and its example figures read; and the inventory comparison's and the
memoryless run's tables those two read beside a run (:func:`read_inputs`).
Design: ``docs/design/replay_detection_comparison/04-event-features.md``.
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.hippocampus.detection_inputs import (
    memoryless_output_paths as memoryless_outputs,
)
from continuum_swr_replay.analysis.inventory_comparison import inventory_run
from continuum_swr_replay.analysis.inventory_comparison.features import (
    STOP_CONTEXT_DTYPES,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import MATCH_DTYPES
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    INVENTORY_DTYPES,
    typed_table,
)
from continuum_swr_replay.analysis.inventory_comparison.selection import (
    REJECTION_DTYPES,
)
from continuum_swr_replay.tables.io import read_table

__all__ = [
    "CANDIDATE_COLUMNS",
    "FEATURE_DTYPES",
    "MS_DTYPES",
    "TABLE_DTYPES",
    "output_path",
    "output_paths",
    "read_inputs",
]

MS_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "ms_n_windows": "Int64",
        "ms_n_windows_decoded": "Int64",
        "ms_n_windows_with_center": "Int64",
        "ms_spread_cm_mean": np.float64,
        "ms_extent_cm": np.float64,
        "ms_distance_cm_mean": np.float64,
        "ms_frac_windows_kept": np.float64,
    }
)
"""The ``ms_features`` columns; ``ms_n_windows`` is missing, not 0, in an epoch
with no memoryless run."""

FEATURE_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        **INVENTORY_DTYPES,
        "duration_stratum": object,
        "first_row": np.int64,
        "n_rows": np.int64,
        "n_spikes": "Int64",
        "n_active_tetrodes": "Int64",
        "n_tetrodes": np.int64,
        "spike_rate_hz": np.float64,
        "ripple_z_max": np.float64,
        "ripple_z_mean": np.float64,
        "mua_z_max": np.float64,
        "mua_z_mean": np.float64,
        "linear_position_cm": np.float64,
        "track_segment_id": "Int64",
        "patch_id": "Int64",
        "evidence_dll": np.float64,
        "evidence_n_valid_rows": "Int64",
        "evidence_valid_fraction": np.float64,
        "evidence_unscorable": object,
        "previous_gap_s": np.float64,
        "next_gap_s": np.float64,
        **STOP_CONTEXT_DTYPES,
        "time_in_stop_s": np.float64,
        "time_since_arrival_s": np.float64,
        "ss_local_prob_mean": np.float64,
        "ss_no_spike_prob_mean": np.float64,
        "ss_continuous_prob_mean": np.float64,
        "ss_fragmented_prob_mean": np.float64,
        "ss_nonlocal_prob_max": np.float64,
        "ss_frac_rows_nonlocal": np.float64,
        "ss_extent_cm": np.float64,
        "ss_distance_cm_mean": np.float64,
        "ss_hpd_cm_mean": np.float64,
        "ss_n_runs_nonlocal": "Int64",
        "ss_longest_run_nonlocal_s": np.float64,
        "ss_nearest_event_gap_s": np.float64,
        "ss_disagreement": object,
        **MS_DTYPES,
    }
)
"""One row per compared event: the inventory schema's columns, the span's
stratum (a label of ``DURATION_STRATUM_LABELS``, in that order), and every
feature."""

TABLE_DTYPES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "features": FEATURE_DTYPES,
        "pair_membership": MATCH_DTYPES,
        "rejection": REJECTION_DTYPES,
    }
)
"""Every table an epoch writes, with its columns and dtypes."""


def output_path(identifier: str, output_dir: Path, name: str) -> Path:
    """Where an epoch's table `name` is in the run folder `output_dir`."""
    return output_dir / f"{identifier}_{name}.parquet"


def output_paths(identifier: str, output_dir: Path) -> dict[str, Path]:
    """Every table an epoch can write, by name."""
    return {name: output_path(identifier, output_dir, name) for name in TABLE_DTYPES}


CANDIDATE_COLUMNS: Mapping[str, object] = MappingProxyType(
    {
        "animal": object,
        "day": np.int64,
        "epoch": object,
        "ms_event_id": np.int64,
        "is_replay": np.bool_,
        "shuffle_pass": np.bool_,
        "shuffle_testable": np.bool_,
        "n_shuffles": np.int64,
        "artifact_drop_reason": object,
        "whole_event_speed_eligible": np.bool_,
    }
)
"""The columns of the memoryless run's candidate tables the report reads."""


def read_inputs(
    manifest: Mapping, inventory_root: Path, memoryless_root: Path
) -> dict[str, pd.DataFrame]:
    """The comparison's ``cells`` and ``directional`` tables and the memoryless
    run's candidates, for the run's ``ok`` epochs, concatenated.

    The runs are the ones the manifest's ``input_runs`` name. Candidates are
    read for the epochs the memoryless run observed.
    """
    runs = manifest["settings"]["input_runs"]
    inventory_dir = inventory_root / runs["inventory"]
    memoryless_dir = memoryless_root / runs["memoryless"]
    ok = sorted(i for i, r in manifest["epochs"].items() if r["status"] == "ok")
    tables = {
        name: typed_table(
            (
                read_table(inventory_run.output_path(identifier, inventory_dir, name))
                for identifier in ok
            ),
            inventory_run.TABLE_DTYPES[name],
        )
        for name in ("cells", "directional")
    }
    tables["memoryless_events"] = typed_table(
        (
            read_table(
                memoryless_outputs(identifier, memoryless_dir)["events"],
                columns=list(CANDIDATE_COLUMNS),
            )
            for identifier in ok
            if manifest["epochs"][identifier].get("memoryless_available")
        ),
        CANDIDATE_COLUMNS,
    )
    return tables
