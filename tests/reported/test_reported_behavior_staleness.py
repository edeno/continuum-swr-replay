"""The behavior summary: values read from a fit made on other tables are pending,
never refused."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.fit_provenance import (
    fit_provenance_path,
    write_fit_provenance,
)
from continuum_swr_replay.paths import table_path
from continuum_swr_replay.reported.behavior import fit_staleness
from continuum_swr_replay.tables.io import write_table


def _write_event_table(root, counts) -> None:
    frame = pd.DataFrame({"event_id": range(len(counts)), "count": counts})
    frame.attrs["covariate_schema_version"] = "v9"
    write_table(frame, table_path("event_table", root))


@pytest.fixture
def fit(replay_content_root):
    """A fit beside the canonical event table it records as its input."""
    _write_event_table(replay_content_root, [1, 0, 4])
    artifact = replay_content_root / "models" / "all_models.pkl"
    artifact.parent.mkdir()
    artifact.write_bytes(b"fitted-model-bytes")
    write_fit_provenance(
        artifact,
        {"event_table": table_path("event_table", replay_content_root)},
        "poisson_glm",
        ["fit.py"],
    )
    return artifact


def test_a_fit_made_on_the_current_tables_is_not_stale(fit):
    assert fit_staleness(fit) is None
    assert fit_staleness() is None


def test_a_fit_made_on_another_table_is_stale_and_named(fit, replay_content_root):
    _write_event_table(replay_content_root, [1, 0, 5])
    note = fit_staleness(fit)
    assert "all_models.pkl was fit on other inputs than the current ones" in note
    assert "(event_table); refit before citing" in note


def test_a_fit_with_no_provenance_record_is_stale(fit):
    other = fit.with_name("other.pkl")
    other.write_bytes(b"x")
    note = fit_staleness(fit, other)
    assert note.endswith("other.pkl has no provenance record; refit before citing")
    assert "all_models.pkl" not in note
    fit_provenance_path(fit).unlink()
    assert fit_staleness(fit, other).count("has no provenance record") == 2


def test_a_fit_whose_record_names_no_input_is_stale(fit):
    # A record that lists nothing cannot say the fit was made on these tables.
    path = fit_provenance_path(fit)
    record = json.loads(path.read_text())
    record["inputs"] = {}
    path.write_text(json.dumps(record))
    note = fit_staleness(fit)
    assert note.endswith(
        "all_models.pkl records no input it was fit on; refit before citing"
    )
