"""The uncertainty-content analysis's written summary refuses a missing contrast."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.replay_models.content import analyze_uncertainty_content as mod

PROFILE_COLUMNS = ["metric", "response", "scope", "level", "value"]
CONTRAST_COLUMNS = [
    "metric",
    "response",
    "scope",
    "level",
    "contrast",
    "n_animals_positive",
    "n_animals",
]


def test_the_summary_names_the_contrast_it_has_no_overall_row_for(tmp_path):
    # the leaf-share and absolute-rate rows are there; the event share's
    # overall row is not (only its per-animal row)
    contrasts = pd.DataFrame(
        [
            ("conditional_event_share", "posterior", "unchosen", "animal", 0.1, 1, 1),
            ("conditional_leaf_share", "posterior", "unchosen", "overall", 0.1, 3, 5),
            ("absolute_rate", "posterior", "unchosen", "overall", 0.1, 3, 5),
        ],
        columns=CONTRAST_COLUMNS,
    )
    with pytest.raises(
        ValueError,
        match=r"no overall row for conditional_event_share \(posterior response, "
        r"unchosen wells\)",
    ):
        mod._write_summary(
            tmp_path / "summary.md",
            pd.DataFrame(columns=PROFILE_COLUMNS),
            contrasts,
            {},
            {},
        )
