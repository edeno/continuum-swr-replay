"""The repository writes American English, in prose and in names.

``scripts/checks/american_spelling.py`` holds the British spellings it knows and
what it reads; this test fails on any it finds in the tracked files, and pins
how it treats the cases that are easy to get wrong. To fix the spellings it
finds, run ``uv run --frozen python scripts/checks/american_spelling.py --fix``.
"""

from __future__ import annotations

import json

import pytest

from scripts.checks.american_spelling import (
    americanize,
    americanize_notebook,
    americanize_prose,
    british_spellings,
    read_as,
)


@pytest.mark.parametrize(
    ("british", "american"),
    [
        ("the analysed epochs", "the analyzed epochs"),
        ("analysed_epochs()", "analyzed_epochs()"),
        ("n_analysed = 3", "n_analyzed = 3"),
        ("NOT_ANALYSED", "NOT_ANALYZED"),
        ("MpfcAnalysedEpochs", "MpfcAnalyzedEpochs"),
        ("Behavioural baseline", "Behavioral baseline"),
        ("ANIMAL_COLOURS", "ANIMAL_COLORS"),
        ('color="lightgrey"', 'color="lightgray"'),
        ("the bin centred on 0.75", "the bin centered on 0.75"),
        ("penalised and unpenalised fits", "penalized and unpenalized fits"),
        ("labelled, mislabelled, unlabelled", "labeled, mislabeled, unlabeled"),
        ("a catalogued covariate", "a cataloged covariate"),
        ("shrinks towards zero afterwards", "shrinks toward zero afterward"),
    ],
)
def test_british_spellings_are_made_american_with_their_case(british, american):
    assert americanize(british) == american


@pytest.mark.parametrize(
    "text",
    [
        "these analyses read the table",  # the plural noun is American
        "MarkedContentRecentRewardSwrReplay",  # recentreward holds "centre"
        "MarkedModelLadder",  # model + ladder
        "ax.tick_params(labelleft=False)",  # label + left
        "run_session only forwards it",  # the verb
        "plt.table(colColours=palette)",  # matplotlib's keyword
        'plt.get_cmap("Greys")',  # matplotlib's colormap
        'cohort["n_analysed"]',  # the key on disk
    ],
)
def test_american_text_and_kept_names_are_left_alone(text):
    assert americanize(text) == text


def test_the_verb_analyses_is_made_american():
    assert americanize("the epochs the register analyses") == (
        "the epochs the register analyzes"
    )
    assert americanize("whatever_the_register_analyses") == (
        "whatever_the_register_analyzes"
    )


def test_a_record_keeps_its_code_as_written():
    record = "The analysed epochs, `analysed_time.py`.\n```\nanalysed = 1\n```\n"
    assert americanize_prose(record) == (
        "The analyzed epochs, `analysed_time.py`.\n```\nanalysed = 1\n```\n"
    )


def test_a_notebook_changes_only_its_cell_sources():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["color = 'grey'\n", "x = 1"],
                "outputs": [{"text": "grey"}],
            }
        ],
        "metadata": {},
    }
    raw = json.dumps(notebook, indent=1) + "\n"
    fixed = json.loads(americanize_notebook(raw))
    assert fixed["cells"][0]["source"] == ["color = 'gray'\n", "x = 1"]
    assert fixed["cells"][0]["outputs"] == [{"text": "grey"}]


def test_vendored_and_quoted_text_is_not_read():
    assert read_as(".claude/skills/figure-critic/SKILL.md") is None
    assert read_as("reference/widloski_foster_2025/LICENSE") is None
    assert read_as("paper/references.bib") is None
    assert read_as(".claude/docs/reviews/2026-11-11-review.md") == "prose"
    assert read_as("docs/design/2026-10-09-swr-source.md") == "prose"
    assert read_as("docs/development.md") == "whole"


def test_the_tracked_files_have_no_british_spellings():
    found = british_spellings()
    assert not found, (
        f"{len(found)} British spellings (fix with `uv run --frozen python "
        "scripts/checks/american_spelling.py --fix`):\n" + "\n".join(found[:50])
    )
