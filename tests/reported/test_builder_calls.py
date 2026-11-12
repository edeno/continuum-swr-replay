"""The analysis functions a builder function calls, read from a small builder's source."""

from __future__ import annotations

from pathlib import Path

import pytest

from continuum_swr_replay.reported import emit, values
from continuum_swr_replay.reported.builder_calls import (
    BuilderCalls,
    builder_calls,
    builder_files,
)
from continuum_swr_replay.reported.values import Tags, value_factory

ANALYSIS = {
    "analysis/__init__.py": "",
    "analysis/stats.py": (
        "NAMES = {'a': 1}\n\n"
        "def median_of(x):\n    return x\n\n"
        "def share(x):\n    return x\n\n"
        "class Gate:\n    pass\n"
    ),
    "analysis/grouped/__init__.py": (
        "from continuum_swr_replay.analysis.grouped.inner import spread\n"
    ),
    "analysis/grouped/inner.py": "def spread(x):\n    return x\n",
}

#: A builder package of two modules. ``first_values`` calls an analysis
#: function by name, one through a module, a method of an analysis constant (not
#: a function), a function a package re-exports, a non-analysis function, and a
#: helper that makes no value; ``second_values`` makes values of its own and
#: calls ``first_values``.
BUILDER = {
    "reported/tiny/__init__.py": (
        "from continuum_swr_replay.reported.tiny.parts import first_values\n\n"
        "def second_values(t):\n"
        "    first_values(t)\n"
        "    return t\n"
    ),
    "reported/tiny/parts.py": (
        "from continuum_swr_replay.analysis import stats\n"
        "from continuum_swr_replay.analysis.grouped import spread\n"
        "from continuum_swr_replay.analysis.stats import median_of\n"
        "from continuum_swr_replay.tables.session_keys import n_sessions\n\n"
        "def _helper(t):\n"
        "    return stats.Gate()\n\n"
        "def first_values(t):\n"
        "    median_of(t)\n"
        "    stats.share(t)\n"
        "    stats.NAMES.items()\n"
        "    spread(t)\n"
        "    n_sessions(t)\n"
        "    return _helper(t)\n"
    ),
}


@pytest.fixture
def package(tmp_path) -> Path:
    for name, text in {**ANALYSIS, **BUILDER}.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_a_builder_is_its_module_or_every_module_of_its_package(package):
    root = package / "reported"
    (root / "single.py").write_text("")
    assert builder_files("single", root) == [root / "single.py"]
    assert [p.name for p in builder_files("tiny", root)] == ["__init__.py", "parts.py"]
    with pytest.raises(FileNotFoundError):
        builder_files("absent", root)


def test_each_function_lists_the_analysis_functions_it_and_its_helpers_call(package):
    root = package / "reported"
    calls = builder_calls(
        builder_files("tiny", root),
        "continuum_swr_replay.reported.tiny",
        {"first_values", "second_values"},
        root,
    )
    assert calls["first_values"] == BuilderCalls(
        "reported/tiny/parts.py",
        (
            "analysis.grouped.inner.spread",
            "analysis.stats.Gate",
            "analysis.stats.median_of",
            "analysis.stats.share",
        ),
        (),
    )
    # A value-making builder function it calls is named, not followed into.
    assert calls["second_values"] == BuilderCalls(
        "reported/tiny/__init__.py", (), ("first_values",)
    )
    # Were first_values a helper, its calls would be second_values' own.
    as_helper = builder_calls(
        builder_files("tiny", root),
        "continuum_swr_replay.reported.tiny",
        {"second_values"},
        root,
    )
    assert as_helper["second_values"].analysis == calls["first_values"].analysis
    assert as_helper["second_values"].through == ()


def test_two_modules_defining_one_name_are_an_error(package):
    root = package / "reported"
    (root / "tiny" / "more.py").write_text("def first_values(t):\n    return t\n")
    with pytest.raises(ValueError, match="first_values is defined in"):
        builder_calls(
            builder_files("tiny", root), "continuum_swr_replay.reported.tiny", (), root
        )


def test_the_view_lists_each_builder_function_with_its_calls(package):
    root = package / "reported"
    (root / "tiny").rename(root / "events")
    for path in (root / "events").glob("*.py"):
        path.write_text(path.read_text().replace("reported.tiny", "reported.events"))

    def first_values():
        pass

    def unknown_values():
        pass

    summary = values.Summary(name="events", produced_by="by")
    for function, key in ((first_values, "one"), (unknown_values, "two")):
        make = value_factory(function, "a table", Tags(("2d.3",)))
        summary.add(make(key, 1, "count", "count", "a count"))
    view = emit.questions_view({"events": summary}, root)
    assert view == emit.questions_view({"events": summary}, root)
    section = view.split("## Builder functions and the analysis code they call")[1]
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    assert rows == [
        "| `events.first_values` | `reported/events/parts.py` | "
        "`analysis.grouped.inner.spread`, `analysis.stats.Gate`, "
        "`analysis.stats.median_of`, `analysis.stats.share` |  |",
        "| `events.unknown_values` | not in the builder's source |  |  |",
    ]
