"""The fit-row record: written and read back."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.likelihoods import Poisson
from continuum_swr_replay.analysis.regression.record import (
    JSON_FIELDS,
    NoFit,
    describe,
    read_records,
    records_frame,
    write_records,
)
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.analysis.regression.support import Removed
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
        [
            Smooth("time_since_s", 0.0, 8.0, 10, 2.0),
            Slope("x", "none"),
            Offset("exposure"),
        ],
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
            "left_out": {"x": Removed.SUPPORT},
        }
    )
    path = write_records(records, tmp_path / "fits" / "rate_test.parquet")
    back = read_records(path)
    assert len(back) == len(records)
    pd.testing.assert_frame_equal(records_frame(back), records_frame(records))
    assert np.isnan(back[0].coef[-1]) and back[0].left_out == {"x": Removed.SUPPORT}
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
        "left_out_round",
        "support",
        "collinearity",
        "basis",
        "standardization",
        "table_sha256",
    }


def test_each_cause_has_its_words() -> None:
    """The record stores causes as codes from closed sets; the words for a
    person are made from the code (with the round for a fit without a finite
    maximum)."""
    words = {
        Removed.SUPPORT: (
            "fewer than MIN_UNITS = 5 units with a row on a side of the column "
            "(a level: of the level)"
        ),
        Removed.TOO_SMALL: (
            "largest absolute value below SMALLEST_COLUMN_SIZE = 0.01 in the session"
        ),
        Removed.WITH_LEVEL: "left with its level, whose rows were dropped",
        NoFit.NO_ROWS: "no rows left to fit",
        NoFit.NO_EVENTS: "no events of the kind in the rows left to fit",
        NoFit.ALONG_THE_INTERCEPT: (
            "no finite maximum along the intercept: the rows of the reference "
            "level, or of the baseline, hold no event of the kind"
        ),
        NoFit.ALONG_A_PENALTY: (
            "no finite maximum along a penalty block, whose columns cannot be "
            "made inactive"
        ),
        NoFit.TOO_MANY_ROUNDS: (
            "no finite maximum after REFIT_MAX_ROUNDS = 5 rounds of columns made "
            "inactive"
        ),
        NoFit.OTHER_STATUS: "the fit at the log penalties ended with another status",
    }
    assert set(words) | {Removed.NO_FINITE_MAXIMUM} == set(Removed) | set(NoFit)
    for code, text in words.items():
        assert describe(code) == text
    assert describe(Removed.NO_FINITE_MAXIMUM, 2) == "no finite maximum (round 2)"


def test_a_missing_record_file_is_named(tmp_path) -> None:
    """Reading records that are not there names the path, without the
    canonical tables' advice; writing replaces any suffix with .parquet."""
    import pytest

    missing = tmp_path / "fits" / "none.parquet"
    with pytest.raises(FileNotFoundError, match=r"no fit records at .*none\.parquet$"):
        read_records(missing)
    written = write_records([], tmp_path / "fits" / "empty.pq")
    assert written == tmp_path / "fits" / "empty.parquet" and written.exists()
