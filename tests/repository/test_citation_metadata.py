"""The paper's title, authors and version agree wherever they are written by hand.

They are stated in ``CITATION.cff``, in the manuscript (``paper/main.tex``: title
and author block), in the README's heading, and, the version only, in
``pyproject.toml``. Nothing generates one from another, so these tests hold them
together, and check that ``CITATION.cff`` has the fields the Citation File Format
requires and that every ORCID iD is well formed (its check digit), offline.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
MAIN_TEX = (REPO / "paper" / "main.tex").read_text()
README = (REPO / "README.md").read_text()


@pytest.fixture(scope="module")
def citation() -> dict[str, Any]:
    return yaml.safe_load((REPO / "CITATION.cff").read_text())


def _name(author: dict[str, str]) -> str:
    return f"{author['given-names']} {author['family-names']}"


def _plain(tex: str) -> str:
    """A LaTeX title as text: commands, braces and line breaks removed."""
    tex = re.sub(r"\\\\", " ", tex)
    tex = re.sub(r"\\[a-zA-Z]+", " ", tex)
    return " ".join(tex.replace("{", " ").replace("}", " ").split())


def _orcid_ok(orcid: str) -> bool:
    """ISO 7064 MOD 11-2 check digit of an ORCID iD such as 0000-0002-1825-0097."""
    digits = orcid.replace("-", "")
    if not re.fullmatch(r"\d{15}[\dX]", digits):
        return False
    total = 0
    for d in digits[:-1]:
        total = (total + int(d)) * 2
    check = (12 - total % 11) % 11
    return digits[-1] == ("X" if check == 10 else str(check))


def test_required_fields(citation):
    assert citation["cff-version"] == "1.2.0"
    for key in ("message", "title", "authors", "version", "license"):
        assert citation.get(key), key
    assert citation["preferred-citation"]["type"] == "article"


def test_software_authors_are_the_paper_authors(citation):
    software = [_name(a) for a in citation["authors"]]
    paper = [_name(a) for a in citation["preferred-citation"]["authors"]]
    assert software == paper


def test_manuscript_authors_match_the_citation_file(citation):
    manuscript = re.findall(
        r"^\\author(?:\[[^\]]*\])?\{([^}]*)\}", MAIN_TEX, re.MULTILINE
    )
    assert manuscript == [_name(a) for a in citation["preferred-citation"]["authors"]]


def test_title_is_the_same_everywhere(citation):
    title = citation["preferred-citation"]["title"]
    tex_title = re.search(r"\\title\{(.*?)\}\n", MAIN_TEX, re.DOTALL)
    assert tex_title and _plain(tex_title.group(1)) == title
    assert README.splitlines()[0] == f"# {title}"
    assert f"'{title}'" in citation["title"]


def test_version_matches_the_package(citation):
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert citation["version"] == project["version"]


def test_orcids_are_well_formed(citation):
    orcids = [a["orcid"] for a in citation["authors"] if "orcid" in a]
    assert orcids
    for orcid in orcids:
        assert orcid.startswith("https://orcid.org/"), orcid
        assert _orcid_ok(orcid.removeprefix("https://orcid.org/")), orcid


def test_orcid_check_digit_rule():
    assert _orcid_ok("0000-0002-1825-0097")  # ORCID's documented example
    assert _orcid_ok("0000-0002-1694-233X")
    assert not _orcid_ok("0000-0002-1825-0098")
