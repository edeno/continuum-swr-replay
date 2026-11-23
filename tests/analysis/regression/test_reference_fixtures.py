"""The shared fitter against its committed reference outputs.

Each case of ``scripts/checks/_regression_reference.py`` is run again and its
outputs compared with ``data/numpy_reference/<group>.json``, written by
``scripts/checks/write_regression_reference.py`` (the commit and the command
are in each file). The rule: a float within ``1e-11 + 1e-9 |b|`` of its
reference ``b``; statuses, reasons, step counts, the columns left out and every
count equal.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from scripts.checks import _regression_reference as reference

FIXTURE_DIR = Path(__file__).parent / "data" / "numpy_reference"


@functools.cache
def _stored(group: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{group}.json").read_text())


def assert_matches_reference(actual: object, stored: object, where: str = "") -> None:
    """Assert that ``actual`` (plain values, ``reference.plain``) matches the
    ``stored`` reference by the rule of ``reference.differences``, listing
    every mismatch."""
    mismatches = reference.differences(actual, stored, where)
    assert not mismatches, "\n".join(mismatches)


@pytest.mark.parametrize(
    ("group", "name"),
    list(reference.CASES),
    ids=[f"{g}-{n}" for g, n in reference.CASES],
)
def test_the_fitter_matches_its_reference(group: str, name: str) -> None:
    assert_matches_reference(
        reference.run(group, name), _stored(group)["cases"][name], f"{group}.{name}"
    )


@pytest.mark.parametrize("group", list(reference.GROUPS))
def test_every_case_has_one_reference(group: str) -> None:
    """The stored file holds the group's cases, no more and no fewer, and
    names the commit that made it."""
    stored = _stored(group)
    assert list(stored["cases"]) == list(reference.GROUPS[group])
    assert len(stored["commit"]) == 40


def test_the_rule_reads_floats_with_its_tolerance_and_the_rest_exactly() -> None:
    """A float passes within ``1e-11 + 1e-9 |b|`` and fails beyond it; an
    int, a string or a missing value must be equal, of the same type; NaN is
    stored as a string, so it matches only NaN."""
    b = 4.0e-4
    assert_matches_reference(b * (1 + 0.9e-9) + 0.9e-11, b)
    assert reference.differences(b * (1 + 1.1e-9) + 1.1e-11, b)
    assert_matches_reference([[1.0, -2.0], [0.0, 3.0]], [[1.0, -2.0], [1e-12, 3.0]])
    assert reference.differences([[1.0, -2.0], [0.0, 3.0]], [[1.0, -2.0], [0.0, 3.1]])
    assert reference.differences(3, 4) and reference.differences(3.0, 3)
    assert reference.differences("converged", "no finite maximum")
    assert reference.differences([1, 2], [2, 1])
    assert reference.plain(float("nan")) == "NaN"
    assert reference.differences(reference.plain([float("nan"), 1.0]), [0.0, 1.0])
    assert reference.differences({"a": 1.0}, {"a": 1.0, "b": 2.0})


def test_the_rule_wants_a_float_for_a_float_and_an_int_for_an_int() -> None:
    """An int does not match a float reference, nor a float an int one, alone
    or inside a list: a fitter that returned a count as a float, or a
    coefficient as an int, would differ in kind from the reference."""
    for actual, stored in [
        (3, 3.0),
        (3.0, 3),
        ([1, 2], [1.0, 2.0]),
        ([1.0, 2.0], [1, 2]),
        ([[1.0, 2], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]),
        ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2], [3.0, 4.0]]),
        ({"n": [0.0]}, {"n": [0]}),
        (True, 1.0),
        (1.0, True),
    ]:
        assert reference.differences(actual, stored), (actual, stored)
    assert_matches_reference([3, [1, 2], 0.0], [3, [1, 2], 0.0])
