"""Values the guides quote match the code that defines them.

The guides state settings a reader needs without opening the code: the decoding
settings in ``docs/reproduce.md``, and in ``docs/development.md`` the
analyzed-time rule's margin and speed limit, the coincident-spike filter, and
the cohort's size per animal. Each test reads the number out of the guide's text
and compares it with the constant or file it describes, so a changed setting
fails here until the guide is updated in the same change.
"""

from __future__ import annotations

import calendar
import csv
import importlib
import inspect
import math
import re
from collections import defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REPRODUCE = (REPO / "docs" / "reproduce.md").read_text()
DEVELOPMENT = (REPO / "docs" / "development.md").read_text()
REGISTER = REPO / "src" / "continuum_swr_replay" / "data_loaders" / "epoch_register.csv"


def _settings_rows() -> list[tuple[str, str, str]]:
    """(setting, value, dotted name) rows of the guide's decoding-settings table."""
    section = REPRODUCE.split("### Key Decoding Parameters", 1)[1].split("\n### ", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[2].startswith("`continuum_swr_replay."):
            rows.append((cells[0], cells[1], cells[2].strip("`")))
    return rows


def _resolve(dotted: str) -> object:
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _documented(value: str) -> float | str:
    """The value a table cell states: a square root, a code string, or a number."""
    if root := re.match(r"√([\d.]+)", value):
        return math.sqrt(float(root.group(1)))
    if code := re.fullmatch(r"`([^`]+)`", value):
        return code.group(1)
    return float(re.match(r"[\d.]+", value).group(0))


def test_settings_table_lists_the_decoding_settings():
    names = {dotted.rsplit(".", 1)[1] for _, _, dotted in _settings_rows()}
    assert names >= {
        "PLACE_BIN_SIZE_1D",
        "POSITION_STD",
        "WAVEFORM_STD",
        "LOCAL_POSITION_STD",
        "BLOCK_SIZE",
        "DEFAULT_TRAINING_TYPE",
        "DEFAULT_NON_LOCAL_THRESHOLD",
    }


@pytest.mark.parametrize(("setting", "value", "dotted"), _settings_rows())
def test_settings_table_matches_the_code(setting, value, dotted):
    documented, actual = _documented(value), _resolve(dotted)
    if isinstance(documented, str):
        assert documented == actual, setting
    else:
        assert documented == pytest.approx(actual), setting


def test_analyzed_time_margin_and_speed_limit():
    from continuum_swr_replay.analysis.event_definition.eligibility import (
        ARTIFACT_REACH_S,
    )
    from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

    margin = re.search(r"widened by ([\d.]+) s on each side", DEVELOPMENT)
    speed = re.search(
        r"every sampled speed finite and below ([\d.]+) cm/s", DEVELOPMENT
    )
    assert margin and float(margin.group(1)) == ARTIFACT_REACH_S
    assert speed and float(speed.group(1)) == DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC


def test_coincident_spike_filter():
    from continuum_swr_replay.data_loaders.spikes import detect_coincident_spikes

    stated = re.search(r">(\d+)% of tetrodes firing within (\d+)\s*μs", DEVELOPMENT)
    defaults = inspect.signature(detect_coincident_spikes).parameters
    assert stated
    assert int(stated.group(1)) == round(
        defaults["max_coincident_fraction"].default * 100
    )
    assert int(stated.group(2)) == round(
        defaults["spike_closeness_threshold"].default * 1e6
    )


def _analyzed_by_animal() -> dict[str, tuple[set[str], int, set[str]]]:
    """Per animal: the analyzed epochs' NWB files, epoch count and recording days."""
    by: dict[str, list] = defaultdict(lambda: [set(), 0, set()])
    with REGISTER.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "analyzed":
                continue
            animal, day = re.match(
                r"([a-z]+\d*?)(\d{8})_", row["nwb_file_name"]
            ).groups()
            by[animal][0].add(row["nwb_file_name"])
            by[animal][1] += 1
            by[animal][2].add(day)
    return {a: (files, n, days) for a, (files, n, days) in by.items()}


def _months(days: set[str]) -> str:
    """``Feb 2020``, ``Nov-Dec 2020``: the span of the recording days, as the table writes it."""
    first, last = min(days), max(days)
    name = lambda d: calendar.month_abbr[int(d[4:6])]  # noqa: E731
    if first[:6] == last[:6]:
        return f"{name(first)} {first[:4]}"
    if first[:4] == last[:4]:
        return f"{name(first)}-{name(last)} {first[:4]}"
    return f"{name(first)} {first[:4]}-{name(last)} {last[:4]}"


def test_cohort_totals():
    stated = re.search(r"analyzes (\d+) epochs in (\d+) NWB files", DEVELOPMENT)
    by = _analyzed_by_animal()
    assert stated
    assert int(stated.group(1)) == sum(n for _, n, _ in by.values())
    assert int(stated.group(2)) == sum(len(files) for files, _, _ in by.values())


def test_cohort_table_per_animal():
    table = DEVELOPMENT.split("| Animal | NWB Files | Epochs | Dates |", 1)[1]
    stated = {}
    for line in table.splitlines()[2:]:
        if not line.startswith("|"):
            break
        animal, files, epochs, dates = (c.strip() for c in line.strip("|").split("|"))
        stated[animal] = (int(files), int(epochs), dates)
    actual = {
        a: (len(f), n, _months(d)) for a, (f, n, d) in _analyzed_by_animal().items()
    }
    assert stated == actual
