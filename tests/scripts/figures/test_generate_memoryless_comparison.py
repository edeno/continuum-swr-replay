"""The memoryless comparison figure's checks of its caption's claims."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.figures import generate_memoryless_comparison as figure  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def controls() -> pd.DataFrame:
    """The committed controls summary the figure reads."""
    return pd.read_csv(REPO / figure.CONTROLS_SUMMARY_PATH)


def test_committed_controls_summary_passes(controls):
    figure.check_controls_claims(controls)


def test_no_replay_claim_needs_every_output(controls):
    """The caption's no-replay claim covers the four outputs the figure scores:
    a summary in which one of them was not run there is refused."""
    missing = (
        (controls["summary"] == "no_replay")
        & (controls["measure"] == "n_events")
        & (controls["detector"] == "state_space_continuous")
    )
    assert missing.sum() == 1
    with pytest.raises(RuntimeError, match="3 detector outputs"):
        figure.check_controls_claims(controls[~missing])
