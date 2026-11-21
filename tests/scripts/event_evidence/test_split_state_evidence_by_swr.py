"""The held-out split by SWR, run to the end on a tiny synthetic run.

One epoch with three immobile Non-Local Continuous bouts and their matched
windows: one bout touches an SWR, one does not, and a planted one with a huge
score starts before the epoch's only well visit. The decode frame and the well
visits are stand-ins; the run folder, its manifest and the outputs are real
files, so the run writes its statistics and its provenance as on real data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.paths import NL_BOUTS_BY_SWR_STEM
from continuum_swr_replay.tables.io import read_table, write_table
from scripts.event_evidence import split_state_evidence_by_swr as splitter

EPOCH = "chimi20200225_02_r1_HPC"
TIME = 100.0 + 0.002 * np.arange(60)
SWR_BIN = 21
"""The one SWR bin, inside the bout at bins 20 to 24."""
VISIT = (TIME[10], TIME[50])
"""The epoch's one well visit: bins 10 to 49."""


def _windows() -> pd.DataFrame:
    """Bouts at bins 2 (before the visit, planted), 20 (SWR) and 30 (no SWR),
    each with a matched window inside the visit, and one Local bout."""
    bout = [(2, 6, 1000.0), (20, 24, 1.0), (30, 34, 3.0)]
    rows = []
    for s, e, score in bout:
        rows.append(("bout", "Non-Local Continuous", s, s, e, np.nan, score))
    for k, (s, _, _) in enumerate(bout):
        rows.append(
            ("baseline", "Non-Local Continuous", 40 + k, np.nan, np.nan, s, 0.0)
        )
    rows.append(("bout", "Local", 12, 12, 16, np.nan, 0.0))
    windows = pd.DataFrame(
        rows,
        columns=["kind", "state", "s", "bout_s", "bout_e", "matched_bout_s", "score"],
    )
    windows["start_time"] = TIME[windows["s"]]
    windows["regime"] = "immobile"
    windows["epoch_id"] = EPOCH
    windows["animal"] = "chimi"
    windows["n_found"] = 3
    for metric in splitter.METRICS:
        windows[metric] = windows["score"]
    return windows.drop(columns="score")


def test_the_run_writes_its_statistics_over_the_bouts_in_visits_and_its_provenance(
    tmp_path, processed_data_root, monkeypatch
):
    run_dir = tmp_path / "state_evidence"
    write_table(_windows(), run_dir / "per_epoch" / f"{EPOCH}_windows.parquet")
    (run_dir / "manifest.json").write_text(
        json.dumps({"epochs": {EPOCH: {"status": "ok"}}})
    )
    swr = np.zeros(TIME.size)
    swr[SWR_BIN] = 1
    frame = pd.DataFrame({"SWR_event": swr}, index=pd.Index(TIME, name="time"))
    monkeypatch.setattr(splitter, "read_dot_with_swr_source", lambda path: frame)
    monkeypatch.setattr(
        splitter,
        "well_visit_times",
        lambda epoch_id: (np.array([VISIT[0]]), np.array([VISIT[1]])),
    )

    splitter.main([str(run_dir)])

    stats = pd.read_csv(run_dir / f"{NL_BOUTS_BY_SWR_STEM}_stats.csv")
    row = stats[
        (stats["state"] == "Non-Local Continuous | no SWR")
        & (stats["regime"] == "immobile")
        & (stats["metric"] == "runcv_nl_anim_mark_ps")
    ].iloc[0]
    # the planted bout (1000) is left out: only the bout at 30 (3.0) is read
    assert row["n"] == 1
    assert row["mean"] == 3.0
    swr_row = stats[
        (stats["state"] == "Non-Local Continuous | SWR")
        & (stats["regime"] == "immobile")
        & (stats["metric"] == "runcv_nl_anim_mark_ps")
    ].iloc[0]
    assert swr_row["n"] == 1
    assert swr_row["mean"] == 1.0

    provenance = json.loads(
        (run_dir / f"{NL_BOUTS_BY_SWR_STEM}.provenance.json").read_text()
    )
    assert provenance["subset"] == ANALYZED_SUBSET
    assert provenance["epochs"] == [EPOCH]
    counts = {row["state"]: row for row in provenance["bouts"]}
    assert counts["Non-Local Continuous | no SWR"] == {
        "state": "Non-Local Continuous | no SWR",
        "regime": "immobile",
        "n_bouts": 2,
        "n_kept": 1,
        "n_left_out": 1,
        "n_kept_matched_outside_visit": 0,
    }
    assert counts["Non-Local Continuous | SWR"]["n_kept"] == 1

    # every bout and matched window is kept in the labeled table, marked
    labeled = read_table(run_dir / f"{NL_BOUTS_BY_SWR_STEM}.parquet")
    assert len(labeled) == 6
    assert labeled["starts_in_well_visit"].sum() == 4
