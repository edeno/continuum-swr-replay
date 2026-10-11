"""Fixes to the figure scripts found in the R7a figure run (task 26b, items 4-6).

- The event gallery selects the same events in every process.
- The pipeline data-flow figure reads its frame through the SWR source, has a
  default event that has every arm indicator, refuses one that does not with a
  clear error, and fits its panel E headers in their cells.
- Figure 2's render cache is keyed on a version token, so a library change can
  invalidate it.
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from continuum_swr_replay.analysis.hippocampus import swr_source  # noqa: E402
from scripts.figures import generate_supp_pipeline_data_flow as flow  # noqa: E402
from tests.unit._swr_inventory_fixtures import (  # noqa: E402
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Event gallery: a stable sample
# --------------------------------------------------------------------------

_SELECT = textwrap.dedent(
    """
    import json

    import pandas as pd

    from scripts.figures import generate_event_gallery as gallery

    rows = [
        {
            "animal": animal,
            "category": category,
            "event_id": 1000 * a + 100 * c + k,
            "ambiguity": (k * 37) % 41 / 41,
        }
        for a, animal in enumerate(("chimi", "j16", "peanut"))
        for c, category in enumerate(gallery.CATEGORIES)
        for k in range(40)
    ]
    selected = gallery.select_events(pd.DataFrame(rows), n_per_category=5)
    print(json.dumps(sorted(int(i) for i in selected["event_id"])))
    """
)


def _selection_in_a_new_process(hash_seed: str) -> list[int]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    done = subprocess.run(
        [sys.executable, "-c", _SELECT],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_gallery_selects_the_same_events_in_two_processes():
    """``hash()`` of a string is salted per process (``PYTHONHASHSEED``); the
    selection must not depend on it."""
    first = _selection_in_a_new_process("1")
    second = _selection_in_a_new_process("2")

    assert len(first) == 3 * 4 * 5
    assert first == second


# --------------------------------------------------------------------------
# Pipeline data-flow figure
# --------------------------------------------------------------------------


def test_the_data_flow_figure_defaults_to_an_event_with_every_indicator():
    assert flow.build_parser().parse_args([]).event_id == 275


def _event_rows(**indicators) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "event_id": 5548608985892833789,
            "candidate_arm": np.arange(9),
            "arm_count_hard": [20, 21, 0, 14, 3, 0, 0, 0, 0],
            "is_current_arm": 0.0,
            "is_prev_arm": 0.0,
            "is_next_arm": 0.0,
            "is_last_rewarded_arm": 0.0,
        }
    )
    return rows.assign(**indicators)


def test_an_event_with_a_missing_indicator_is_refused_by_name():
    """The old default, event 14, is off-well: its ``is_current_arm`` is NaN
    under the covariate contract, and the cast to int raised
    ``IntCastingNaNError``."""
    rows = _event_rows(is_current_arm=np.nan, is_prev_arm=np.nan)

    with pytest.raises(ValueError, match=r"event 14.*is_current_arm.*is_prev_arm"):
        flow.indicator_table(rows, event_id=14)


def test_an_event_with_every_indicator_gives_its_integer_table():
    rows = _event_rows(is_current_arm=(np.arange(9) == 5).astype(float))

    table = flow.indicator_table(rows, event_id=275)

    assert list(table.columns) == flow.INDICATOR_COLUMNS
    assert (table.dtypes == np.int64).all()
    assert table["is_current_arm"].tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0]


def test_the_panel_e_headers_fit_their_cells():
    import matplotlib.pyplot as plt

    fig, axes = flow.figure_axes()
    table = flow.draw_indicator_table(
        axes["E"], flow.indicator_table(_event_rows(), event_id=275)
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    headers = [cell for (row, _), cell in table.get_celld().items() if row == 0]
    assert len(headers) == len(flow.INDICATOR_COLUMNS)
    for cell in headers:
        text = cell.get_text().get_window_extent(renderer)
        box = cell.get_window_extent(renderer)
        assert box.x0 <= text.x0 and text.x1 <= box.x1, cell.get_text().get_text()
        assert box.y0 <= text.y0 and text.y1 <= box.y1, cell.get_text().get_text()
    # every column is named, in full, across the wrapped lines
    names = [cell.get_text().get_text().replace("\n", "") for cell in headers]
    assert sorted(names) == sorted(flow.INDICATOR_COLUMNS)
    plt.close(fig)


EPOCH_ID = "chimi20200213_02_r1_HPC"
TIME = np.round(100.0 + np.arange(2000) / 500.0, 6)


def test_the_data_flow_figure_reads_its_frame_through_the_source(tmp_path, monkeypatch):
    """An artifact core at 102.15 s (a spike line, padded by 0.1 s and widened
    by 0.3 s) puts [101.75, 102.55] in excluded time."""
    inventory = write_swr_inventory(tmp_path / "swr", {EPOCH_ID: []})
    artifacts = write_artifact_run(
        tmp_path / "artifacts", {EPOCH_ID: [("spk", 102.15, 102.15)]}
    )
    use_swr_source(monkeypatch, inventory, artifacts)
    pd.DataFrame(
        {
            "animal": "chimi",
            "day": 20200213,
            "epoch": "02_r1",
            "nwb_file_name": "chimi20200213_.nwb",
            "head_speed": 0.0,
            "map_estimate": 0.0,
            "non_local_event": 0,
            "SWR_event": 0,
            "MUA_event": 0,
            "Non-Local_Continuous_prob": 0.0,
            "Non-Local_Fragmented_prob": 0.0,
        },
        index=pd.Index(TIME, name="time"),
    ).to_pickle(tmp_path / f"{EPOCH_ID}_80_data_over_time.pkl")
    (tmp_path / f"{EPOCH_ID}_environment.pkl").write_bytes(
        pickle.dumps({"environment": "stand-in"})
    )
    monkeypatch.setattr(flow, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        flow, "map_estimate_to_arm", lambda values, env: np.full(len(values), np.nan)
    )

    dot, _ = flow.load_session("chimi", 20200213, "02_r1")

    ((start, end),) = swr_source.excluded_intervals(EPOCH_ID).to_numpy()
    np.testing.assert_array_equal(
        dot["in_excluded_time"].to_numpy(), (TIME >= start) & (TIME <= end)
    )
    assert dot["in_excluded_time"].any()


# --------------------------------------------------------------------------
# Figure 2's render cache
# --------------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    """Import a script with ``spyglass`` stubbed; forget it afterwards."""
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = type("PositionIntervalMap", (), {})
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)
    data_loaders = importlib.import_module("continuum_swr_replay.data_loaders")
    load_data_was_cached = "load_data" in vars(data_loaders)
    bound = ("continuum_swr_replay.data_loaders.bandit_task",)
    fresh_bound = [name for name in bound if name not in sys.modules]
    imported: list[str] = []

    def load(name: str):
        if name not in sys.modules:
            imported.append(name)
        return importlib.import_module(name)

    yield load

    for name in [*imported, *fresh_bound]:
        sys.modules.pop(name, None)
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            vars(sys.modules[parent]).pop(child, None)
    if not load_data_was_cached:
        vars(data_loaders).pop("load_data", None)


def test_the_figure02_cache_is_keyed_on_a_version_token(offline, tmp_path):
    figure02 = offline("scripts.figures.generate_figure02")

    path = figure02._figure02_cache_path("wilbur20210329_.nwb", "02_r1", tmp_path)

    # the name before the token: a cache written with it is not read again
    v1 = (
        "wilbur20210329_02_r1_figure02_whole_event_speed_v1_"
        f"{swr_source.swr_source_name()}_cache.joblib"
    )
    assert path.parent == tmp_path / "intermediates"
    assert path.name != v1
    assert f"_v{figure02.FIGURE02_CACHE_VERSION}_" in path.name
    assert figure02.FIGURE02_CACHE_VERSION >= 2
    assert swr_source.swr_source_name() in path.name


def _register(path: Path, epochs: list[str]) -> Path:
    pd.DataFrame(
        [["wilbur20210329_.nwb", epoch, "analysed", "", ""] for epoch in epochs],
        columns=["nwb_file_name", "epoch", "status", "reason", "evidence"],
    ).to_csv(path, index=False)
    return path


def test_the_figure02_cache_is_keyed_on_the_analysed_epochs(
    offline, tmp_path, monkeypatch
):
    """Panel e summarises every analysed epoch, so a register change must make
    a new cache without a code change."""
    from continuum_swr_replay.data_loaders import epoch_register

    figure02 = offline("scripts.figures.generate_figure02")

    def cache_name(epochs: list[str]) -> str:
        register = _register(tmp_path / f"register_{len(epochs)}.csv", epochs)
        monkeypatch.setattr(epoch_register, "REGISTER_PATH", register)
        return figure02._figure02_cache_path(
            "wilbur20210329_.nwb", "02_r1", tmp_path
        ).name

    one = cache_name(["02_r1"])
    two = cache_name(["02_r1", "04_r2"])

    assert one != two
    assert cache_name(["02_r1"]) == one
