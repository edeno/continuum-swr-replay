"""The digest of a set of files (``runs.io.files_sha256``)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from continuum_swr_replay.runs.io import files_sha256, sha256_file


def _write(folder: Path, name: str, data: bytes) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(data)
    return path


def test_the_digest_is_each_file_s_name_then_its_sha256_in_sorted_order(tmp_path):
    first = _write(tmp_path / "epoch_a", "metrics.parquet", b"1")
    second = _write(tmp_path / "epoch_b", "metrics.parquet", b"22")
    by_hand = hashlib.sha256(
        b"metrics.parquet"
        + hashlib.sha256(b"1").digest()
        + b"metrics.parquet"
        + hashlib.sha256(b"22").digest()
    ).hexdigest()

    assert files_sha256([first, second]) == by_hand
    # the order given does not matter; the sorted path order does
    assert files_sha256([second, first]) == by_hand
    assert files_sha256(iter([str(second), first])) == by_hand


def test_the_digest_changes_with_a_file_s_bytes_its_name_and_the_set(tmp_path):
    first = _write(tmp_path / "a", "x.csv", b"1")
    second = _write(tmp_path / "b", "y.csv", b"2")
    digest = files_sha256([first, second])

    assert files_sha256([first]) != digest
    assert files_sha256([first, _write(tmp_path / "b", "z.csv", b"2")]) != digest
    second.write_bytes(b"3")
    assert files_sha256([first, second]) != digest


def test_one_file_s_digest_is_not_its_own_sha256(tmp_path):
    # the name enters the digest, so a set of one is told from the file itself
    path = _write(tmp_path, "x.csv", b"1")
    assert files_sha256([path]) != sha256_file(path)
    assert files_sha256([]) == hashlib.sha256().hexdigest()
