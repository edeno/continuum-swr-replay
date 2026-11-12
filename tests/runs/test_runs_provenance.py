"""How a run, a measurement and a summary name the commit they ran from
(``runs.provenance.commit_label``, ``run_id``)."""

from __future__ import annotations

from continuum_swr_replay.runs import provenance

COMMIT = "7a8d168c4ab9e0f1d2c3b4a5968778695a4b3c2d"


def test_a_commit_label_is_twelve_characters_and_dirty_when_changed():
    assert provenance.commit_label(COMMIT, dirty=False) == "7a8d168c4ab9"
    assert provenance.commit_label(COMMIT, dirty=True) == "7a8d168c4ab9-dirty"


def test_a_run_id_is_the_one_asked_for_or_the_commit_label(capsys):
    """A dirty checkout is warned of either way: the run cannot be tied to a
    commit."""
    clean = {"commit": COMMIT, "dirty": False}
    dirty = {"commit": COMMIT, "dirty": True}

    assert provenance.run_id(clean, None) == "7a8d168c4ab9"
    assert capsys.readouterr().out == ""
    assert provenance.run_id(clean, "pilot") == "pilot"
    assert provenance.run_id(dirty, None) == "7a8d168c4ab9-dirty"
    assert "uncommitted changes; run 7a8d168c4ab9-dirty" in capsys.readouterr().out
    assert provenance.run_id(dirty, "pilot") == "pilot"
    assert "run pilot cannot be tied" in capsys.readouterr().out


def test_the_described_commit_is_the_checkout_s_label(tmp_path, monkeypatch):
    """The working directory's commit label, as git reports it."""
    answers = {("rev-parse", "HEAD"): COMMIT, ("status", "--porcelain"): " M x.py"}

    def fake_git(repo, *args, check=True):
        return answers[args[:2]]

    monkeypatch.setattr(provenance, "git", fake_git)
    monkeypatch.chdir(tmp_path)

    assert provenance.described_commit() == "7a8d168c4ab9-dirty"
