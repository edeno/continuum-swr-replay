"""Tests for the StateScript -> contingency-table parser.

Unit tests run on hand-written synthetic log snippets (deterministic). The
real-session parse is marked ``slow`` (needs Spyglass DB access) and checks the
documented schedule + alignment on a few sessions across two animals.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.paths import table_path
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table, table_exists

_SPEC = importlib.util.spec_from_file_location(
    "_statescript_contingency",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "exploratory"
    / "_statescript_contingency.py",
)
ssc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ssc)


# ── Synthetic fixtures ──

# Two contingency blocks with a percent->fraction conversion. The contingency
# line is restated (collapse), and the leafProbs are updated one line at a time
# at a block change (transient intermediate vectors must NOT spawn extra
# blocks). A header ``#`` comment and a non-assignment line are interleaved to
# confirm they are ignored.
_SYNTH_LOG = """\
#int contingency = -1
#int leafProbs1 = 0
1000 contingency = 0
1002 leafProbs1 = 80
1004 leafProbs2 = 50
1006 leafProbs3 = 20
1008 leafProbs4 = 80
1010 leafProbs5 = 20
1012 leafProbs6 = 50
1100 UP 4
1200 DOWN 4
1300 contingency = 0
1302 leafProbs1 = 80
1304 leafProbs2 = 50
1306 leafProbs3 = 20
1308 leafProbs4 = 80
1310 leafProbs5 = 20
1312 leafProbs6 = 50
some non-assignment runtime line
2000 contingency = 1
2002 leafProbs1 = 20
2004 leafProbs2 = 50
2006 leafProbs3 = 80
2008 leafProbs4 = 50
2010 leafProbs5 = 80
2012 leafProbs6 = 20
2100 UP 2
"""


def test_parse_assignments():
    df = ssc.parse_assignments(_SYNTH_LOG)
    # only the timestamped ``name = int`` lines, never the ``#`` declarations
    assert set(df["name"].unique()) >= {"contingency", "leafProbs1", "leafProbs6"}
    assert (df["ts_ms"] >= 1000).all()
    # the ``#int contingency = -1`` header must be excluded
    assert not ((df["name"] == "contingency") & (df["value"] == -1)).any()
    # UP / DOWN / prose lines are not assignments
    assert "UP" not in df["name"].to_numpy()


def test_block_reconstruction():
    blocks = ssc.reconstruct_blocks(ssc.parse_assignments(_SYNTH_LOG))
    # the restated (identical) contingency=0 collapses; two genuine blocks remain
    assert len(blocks) == 2
    assert list(blocks["block_number"]) == [0, 1]
    assert list(blocks["t_start_ms"]) == [1000, 2000]  # the contingency-line ts
    # percent -> fraction
    b0 = blocks.iloc[0]
    assert b0["task_reward_probability_port_0"] == 0.8
    assert b0["task_reward_probability_port_2"] == 0.2
    b1 = blocks.iloc[1]
    assert b1["task_reward_probability_port_0"] == 0.2
    assert b1["task_reward_probability_port_2"] == 0.8


def test_poke_times():
    pokes = ssc.parse_poke_times_ms(_SYNTH_LOG)
    assert list(pokes) == [1100.0, 2100.0]


def _synth_stops():
    """Three stops: two in block 0, one in block 1; arms span both patches."""
    return pd.DataFrame(
        {
            "stop_id": [10, 11, 12],
            "stop_start": [1.5, 1.9, 2.5],  # neural s (offset already applied)
            "occupancy_well_arm": [0, 2, 2],  # port 0 (p=0.8 then 0.2), port 2
            "is_rewarded_outcome": [1.0, 0.0, 1.0],
        }
    )


def test_value_invariants():
    blocks = ssc.reconstruct_blocks(ssc.parse_assignments(_SYNTH_LOG))
    # offset chosen so block starts (1.0 s, 2.0 s) sit before the stops
    table = ssc.annotate_stops(_synth_stops(), blocks, offset=0.0)
    ssc._assert_invariants(table)

    port_cols = [f"task_reward_probability_port_{p}" for p in range(6)]
    vals = table[port_cols].to_numpy()
    assert set(np.round(vals[np.isfinite(vals)], 3)).issubset(ssc.VALID_P_REWARD)
    assert (table["task_reward_probability_patch_mean_gap_best_vs_second"] >= 0).all()

    # This diagnostic is outcome minus the programmed task probability, not
    # the behavioral model's reward prediction error.
    by_id = table.set_index("stop_id")
    assert by_id.loc[10, "task_reward_probability_chosen_arm"] == 0.8  # block 0, port 0
    assert by_id.loc[
        10, "task_reward_probability_error_chosen_arm_postoutcome"
    ] == pytest.approx(1.0 - 0.8)
    assert by_id.loc[11, "task_reward_probability_chosen_arm"] == 0.2  # block 0, port 2
    assert by_id.loc[
        11, "task_reward_probability_error_chosen_arm_postoutcome"
    ] == pytest.approx(0.0 - 0.2)
    assert by_id.loc[12, "task_reward_probability_chosen_arm"] == 0.8  # block 1, port 2
    assert by_id.loc[
        12, "task_reward_probability_error_chosen_arm_postoutcome"
    ] == pytest.approx(1.0 - 0.8)
    # trials_since_block_change resets at the block boundary
    assert by_id.loc[10, "trials_since_block_change"] == 0
    assert by_id.loc[11, "trials_since_block_change"] == 1
    assert by_id.loc[12, "trials_since_block_change"] == 0


def test_task_reward_probability_best_and_best_patch():
    blocks = ssc.reconstruct_blocks(ssc.parse_assignments(_SYNTH_LOG))
    table = ssc.annotate_stops(_synth_stops(), blocks, offset=0.0).set_index("stop_id")
    # block 0 ports: [.8 .5 .2 .8 .2 .5] -> best port = .8
    assert table.loc[10, "task_reward_probability_best_arm"] == 0.8
    # block 0 patch means: P1=(.8+.5)/2=.65, P2=(.2+.8)/2=.5, P3=(.2+.5)/2=.35
    # -> best patch = 1 (ports 0,1). stop 10 at port 0 -> is_best_patch True.
    assert bool(table.loc[10, "is_best_patch"]) is True
    # stop 11 at port 2 (patch 2) -> not best.
    assert bool(table.loc[11, "is_best_patch"]) is False


def test_solve_time_offset_recovers_constant():
    rng = np.random.default_rng(0)
    ss_ms = np.sort(rng.uniform(0, 3_600_000, size=300))  # 0..1 h in ms
    true_offset = 1.5e9
    dio = ss_ms / 1000.0 + true_offset + rng.normal(0, 0.005, size=ss_ms.size)
    offset, residual, n_matched = ssc.solve_time_offset(ss_ms, dio)
    assert abs(offset - true_offset) < 0.01
    assert residual < ssc.MAX_ALIGN_RESIDUAL_S
    assert n_matched > 0.9 * ss_ms.size


def test_stop_before_first_block_is_dropped():
    blocks = ssc.reconstruct_blocks(ssc.parse_assignments(_SYNTH_LOG))
    stops = pd.DataFrame(
        {
            "stop_id": [99],
            "stop_start": [0.2],  # before block 0 start (1.0 s)
            "occupancy_well_arm": [0],
            "is_rewarded_outcome": [1.0],
        }
    )
    table = ssc.annotate_stops(stops, blocks, offset=0.0)
    assert table.empty


# ── Real-data smoke (slow; needs Spyglass) ──


@pytest.mark.slow
@pytest.mark.real_data
def test_real_session_parse():
    stop_table_path = table_path("stop_table")
    if not table_exists(stop_table_path):
        pytest.fail("canonical stop table not materialized")
    stop_table = read_table(stop_table_path)
    try:
        require_covariate_schema(stop_table, table="stop")
    except ValueError as exc:
        pytest.fail(f"canonical stop table requires regeneration: {exc}")
    # >= 3 sessions across >= 2 animals that exist in stop_table.
    targets = [
        ("chimi", 20200213, "02_r1"),
        ("chimi", 20200214, "02_r1"),
        ("senor", 20201027, "02_r1"),
    ]
    n_ok = 0
    for animal, day, epoch in targets:
        stops = stop_table[
            (stop_table["animal"] == animal)
            & (stop_table["session_date"] == day)
            & (stop_table["epoch"] == epoch)
        ]
        if stops.empty:
            continue
        nwb = f"{animal}{day}_.nwb"
        content, poke_ms, dio_on = ssc._load_session_inputs(nwb, epoch)
        table, prov = ssc.build_session_contingency_table(
            content, poke_ms, dio_on, stops
        )
        n_ok += 1
        assert prov["residual_ms"] < ssc.MAX_ALIGN_RESIDUAL_S * 1e3
        # the bulk of stops map (those before block 0 are dropped).
        assert len(table) > 0.5 * len(stops)
    assert n_ok >= 3, "need >=3 real sessions to parse"

    # The first (exposure) session's documented schedule (block 0 all 1.0,
    # block 1 all 0.5) is validated directly from the raw log -- it was not
    # decoded, so it has no rows in stop_table.
    first_content, _, _ = ssc._load_session_inputs("chimi20200212_.nwb", "02_r1")
    first_blocks = ssc.reconstruct_blocks(ssc.parse_assignments(first_content))
    port_cols = [f"task_reward_probability_port_{p}" for p in range(6)]
    assert (first_blocks.iloc[0][port_cols] == 1.0).all()
    assert (first_blocks.iloc[1][port_cols] == 0.5).all()
