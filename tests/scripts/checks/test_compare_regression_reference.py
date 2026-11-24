"""The real-slice script writes its fits only to a scratch folder: never inside
the repository, never under the processed-data root."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.checks import compare_regression_reference as script


def test_a_folder_inside_the_repository_is_refused(tmp_path, processed_data_root):
    repo = tmp_path / "repo"
    repo.mkdir()
    message = script.why_not_a_write_folder(repo / "out" / "slice", repo=repo)
    assert message and "repository" in message
    assert script.why_not_a_write_folder(repo, repo=repo)


def test_a_folder_under_the_processed_data_root_is_refused(
    tmp_path, processed_data_root
):
    repo = tmp_path / "repo"
    message = script.why_not_a_write_folder(processed_data_root / "slice", repo=repo)
    assert message and "processed-data root" in message
    assert script.why_not_a_write_folder(processed_data_root, repo=repo)


def test_a_scratch_folder_is_taken(tmp_path, processed_data_root):
    scratch = tmp_path / "scratch" / "slice"
    assert script.why_not_a_write_folder(scratch, repo=tmp_path / "repo") is None


def test_the_script_refuses_before_it_reads_or_writes(
    tmp_path, processed_data_root, monkeypatch, capsys
):
    def must_not_run() -> dict:
        raise AssertionError("the slice was read")

    monkeypatch.setattr(script, "real_slice", must_not_run)
    target = processed_data_root / "slice"
    with pytest.raises(SystemExit) as raised:
        script.main(["--write", str(target)])
    assert raised.value.code == 2
    assert "processed-data root" in capsys.readouterr().err
    assert not target.exists()
    inside = Path(script.REPO) / "never_written_slice"
    with pytest.raises(SystemExit):
        script.main(["--write", str(inside)])
    assert not inside.exists()
