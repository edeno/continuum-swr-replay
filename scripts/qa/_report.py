"""Report helpers shared by the QA and PFC audit scripts: the run's command line,
markdown tables.

Not a script. The scripts import it as ``scripts.qa._report``, after putting
the repo root on ``sys.path``.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _invocation(argv: Sequence[str]) -> str:
    """This run's command line, relative to the repo root and shell-quoted.

    The program is this process's ``sys.argv[0]``; ``argv`` is what the
    script's ``main`` parsed, so a ``main([...])`` call reports its own
    arguments.
    """
    script = Path(sys.argv[0])
    try:
        script = script.resolve().relative_to(REPO)
    except ValueError:
        pass
    return shlex.join(["uv", "run", "python", str(script), *argv])


def _cell(value: object, float_format: str) -> str:
    """One markdown cell: missing renders blank, a count renders as a count."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return float_format.format(value)
    return str(value)


def _md(frame: pd.DataFrame, float_format: str = "{:.4g}") -> str:
    """A markdown table, with missing values rendered as an empty cell.

    Cells are rendered by iterating the columns rather than with
    ``Series.map``: mapping over a nullable-integer column coerces its values
    to float, which turns a neuron count into ``10986.0``.
    """
    if frame.empty:
        return "_(no rows)_\n"
    columns = {
        name: [_cell(value, float_format) for value in frame[name]]
        for name in frame.columns
    }
    header = "| " + " | ".join(str(name) for name in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(cells[i] for cells in columns.values()) + " |"
        for i in range(len(frame))
    ]
    return "\n".join([header, rule, *body]) + "\n"
