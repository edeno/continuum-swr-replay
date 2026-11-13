"""A ``cells`` table of the inventory comparison, for the examples' tests."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.detector_comparison import inventory_run


def cells(records: list[tuple[str, str, str, str, int]]) -> pd.DataFrame:
    """A ``cells`` table from (cell, detector, animal, membership, count)."""
    rows = []
    for cell, detector, animal, membership, count in records:
        for number in range(count):
            rows.append(
                {
                    "cell": cell,
                    "animal": animal,
                    "day": 20200101,
                    "epoch": "02_r1",
                    "detector": detector,
                    "event_key": f"{detector}:{animal}20200101_02_r1:{membership}{number}",
                    "n_against": 7,
                    "membership": membership,
                }
            )
    dtypes = inventory_run.TABLE_DTYPES["cells"]
    return pd.DataFrame(rows, columns=list(dtypes)).astype(dict(dtypes))
