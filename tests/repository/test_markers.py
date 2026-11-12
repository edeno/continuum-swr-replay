"""The ``real_data`` marker: skipped by default, run under ``-m real_data``.

The hook under test is the root ``tests/conftest.py``'s
``pytest_collection_modifyitems``. Each case runs pytest in-process on a tiny
throwaway project whose conftest re-exports that hook, so the real hook decides.
"""

from __future__ import annotations

import pytest

from tests.conftest import REAL_DATA_SKIP_REASON

_EXAMPLE_TESTS = """
import pytest


@pytest.mark.real_data
def test_reads_real_data():
    pass


def test_synthetic():
    pass
"""


@pytest.fixture
def project(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(
        "from tests.conftest import pytest_collection_modifyitems  # noqa: F401\n"
    )
    pytester.makeini("[pytest]\nmarkers =\n    real_data: reads real data\n")
    pytester.makepyfile(test_example=_EXAMPLE_TESTS)
    return pytester


@pytest.mark.parametrize("args", [(), ("-m", "not slow")])
def test_real_data_is_skipped_unless_the_expression_names_it(project, args):
    result = project.runpytest("-p", "no:cacheprovider", "-rs", *args)

    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines([f"*test_example.py*{REAL_DATA_SKIP_REASON}*"])


def test_real_data_runs_under_dash_m_real_data(project):
    result = project.runpytest("-p", "no:cacheprovider", "-m", "real_data")

    result.assert_outcomes(passed=1, deselected=1)


def test_a_wider_expression_naming_real_data_runs_everything(project):
    result = project.runpytest(
        "-p", "no:cacheprovider", "-m", "real_data or not real_data"
    )

    result.assert_outcomes(passed=2)


def test_not_real_data_deselects_the_real_data_tests(project):
    result = project.runpytest("-p", "no:cacheprovider", "-m", "not real_data")

    result.assert_outcomes(passed=1, deselected=1)


def test_real_data_and_not_slow_runs_the_fast_real_data_tests(project):
    result = project.runpytest("-p", "no:cacheprovider", "-m", "real_data and not slow")

    result.assert_outcomes(passed=1, deselected=1)


def test_a_marker_whose_name_contains_real_data_is_not_an_opt_in(project):
    """``real_data_lite`` is another marker; naming it does not opt in."""
    result = project.runpytest(
        "-p", "no:cacheprovider", "-rs", "-m", "not real_data_lite"
    )

    result.assert_outcomes(passed=1, skipped=1)
