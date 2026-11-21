"""The fit-row record: written and read back."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.likelihoods import Poisson
from continuum_swr_replay.analysis.regression.record import (
    JSON_FIELDS,
    read_records,
    records_frame,
    write_records,
)
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.analysis.regression.terms import (
    Offset,
    Slope,
    Smooth,
    build_design,
)


def test_round_trip(rate_sessions, tmp_path) -> None:
    """Records written and read back are identical, JSON columns included (NaN
    at a column left out survives)."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    design = build_design(
        [Smooth("time_since_s", 0.0, 8.0, 10, 2.0), Slope("x"), Offset("exposure")],
        pooled,
        intercept=True,
    )
    records = fit_sessions(
        design,
        rate_sessions,
        Poisson(),
        model="rate_test",
        kind="SWR",
        response="count",
        unit="well_visit_id",
        table_sha256={"stop_table": "ab" * 32},
    )
    records[0] = records[0].__class__(
        **{
            **records[0].__dict__,
            "coef": [*records[0].coef[:-1], float("nan")],
            "left_out": {"x": "support"},
        }
    )
    path = write_records(records, tmp_path / "fits" / "rate_test.parquet")
    back = read_records(path)
    assert len(back) == len(records)
    pd.testing.assert_frame_equal(records_frame(back), records_frame(records))
    assert np.isnan(back[0].coef[-1]) and back[0].left_out == {"x": "support"}
    assert back[1].basis == json.loads(json.dumps(records[1].basis))
    stored = pd.read_parquet(path)
    assert all(stored[name].map(type).eq(str).all() for name in JSON_FIELDS)
    assert set(JSON_FIELDS) == {
        "cohort_counts",
        "coef_names",
        "coef",
        "vcov",
        "log_penalty",
        "edf",
        "left_out",
        "support",
        "collinearity",
        "basis",
        "standardization",
        "table_sha256",
    }
