"""The held-out runner's merge of its hosts' shard manifests and of retries."""

import json

import pytest

from scripts.event_evidence import score_tetrode_holdout as runner


def _manifest(path, shards, epochs, commit="abc"):
    path.write_text(
        json.dumps(
            {
                "produced_by": "runner",
                "provenance": {"commit": commit},
                "inputs": {"event_table": {"sha256": "x"}},
                "settings": {"n_folds": 5, "event_table": f"/{commit}/table"},
                "shards": shards,
                "epochs": epochs,
            }
        )
    )


def test_the_merge_takes_every_shard_and_a_retry_replaces_a_failed_epoch(tmp_path):
    _manifest(
        tmp_path / "manifest_shards0of2.json",
        [0],
        {"e1": {"status": "ok"}, "e2": {"status": "failed", "error": "OOM"}},
    )
    _manifest(tmp_path / "manifest_shards1of2.json", [1], {"e3": {"status": "ok"}})
    _manifest(tmp_path / "retry.json", [0], {"e2": {"status": "ok"}})
    runner.merge_shards(tmp_path, 2, ["retry.json"])
    merged = json.loads((tmp_path / "manifest.json").read_text())
    assert merged["epochs"] == {
        "e1": {"status": "ok"},
        "e2": {"status": "ok", "retried": True},
        "e3": {"status": "ok"},
    }
    assert merged["retries"] == ["retry.json"]


def test_the_merge_refuses_a_missing_shard_another_commit_or_a_needless_retry(
    tmp_path,
):
    _manifest(tmp_path / "manifest_shards0of2.json", [0], {"e1": {"status": "ok"}})
    with pytest.raises(ValueError, match="shards"):
        runner.merge_shards(tmp_path, 2)
    _manifest(
        tmp_path / "manifest_shards1of2.json", [1], {"e2": {"status": "ok"}}, "def"
    )
    with pytest.raises(ValueError, match="different commits"):
        runner.merge_shards(tmp_path, 2)
    _manifest(tmp_path / "manifest_shards1of2.json", [1], {"e2": {"status": "ok"}})
    _manifest(tmp_path / "retry.json", [0], {"e2": {"status": "ok"}})
    with pytest.raises(ValueError, match="not a failed epoch"):
        runner.merge_shards(tmp_path, 2, ["retry.json"])
