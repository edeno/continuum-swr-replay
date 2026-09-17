"""Publication-facing inference labels for Figure 4."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import pandas as pd

from continuum_swr_replay.analysis.pfc.heldout import DEFAULT_CONTROL_DRAWS
from continuum_swr_replay.analysis.pfc.load_population import HELDOUT_P_COLUMNS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "generate_figure04",
    Path(__file__).resolve().parents[2] / "scripts" / "figures" / "generate_figure04.py",
)
figure04 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(figure04)


def test_panel_d_uses_the_canonical_control_draw_count():
    assert figure04.DEFAULT_CONTROL_DRAWS == DEFAULT_CONTROL_DRAWS == 1_000

    frame = pd.DataFrame(
        {column: [0.01, 0.5] for column in HELDOUT_P_COLUMNS.values()}
    )
    fig, ax = plt.subplots()
    try:
        figure04.plot_significance(frame, ax)
        assert "R=1000, min p=0.001" in ax.get_title()
    finally:
        plt.close(fig)
