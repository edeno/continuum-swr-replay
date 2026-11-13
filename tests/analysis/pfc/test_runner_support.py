"""Tests for the session-selection, seeding and sidecar helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from continuum_swr_replay.analysis.pfc.runner_support import (
    fails_convergence_gate,
    is_nonempty_file,
    json_safe,
    matching_sidecar,
    remove_bundle,
    session_seed,
)
from continuum_swr_replay.runs.fit_provenance import fit_provenance_path


class TestSessionSeed:
    def test_it_is_deterministic_and_stream_specific(self):
        folds = session_seed("folds", "wilbur20210329_.nwb", "02_r1")

        assert folds == session_seed("folds", "wilbur20210329_.nwb", "02_r1")
        assert folds != session_seed("shuffles", "wilbur20210329_.nwb", "02_r1")
        assert folds != session_seed("folds", "wilbur20210329_.nwb", "04_r2")
        assert folds != session_seed("folds", "wilbur20210330_.nwb", "02_r1")

    def test_it_stays_inside_the_unsigned_64_bit_range(self):
        seeds = [
            session_seed(stream, f"rat2021032{index}_.nwb", "02_r1")
            for index in range(5)
            for stream in ("folds", "shuffles")
        ]

        assert all(0 <= seed < 2**64 for seed in seeds)
        assert len(set(seeds)) == len(seeds)

    def test_it_reproduces_the_seeds_a_stored_fit_records(self):
        # ``fold_seed`` and ``control_draw_seed`` as recorded in the sidecar of
        # the stored canonical fit
        # Processed-Data/chimi20200223_02_r1_mPFC_modulation.provenance.json
        # (written by scripts/pfc/run_pfc_analysis_bandit.py at 110c9ae4). A
        # change to the key, the digest or the byte order would make a refit
        # draw other folds and another null than the bundles were scored on.
        assert (
            session_seed("folds", "chimi20200223_.nwb", "02_r1")
            == 13_651_600_409_216_308_528
        )
        assert (
            session_seed("control_draws", "chimi20200223_.nwb", "02_r1")
            == 12_003_865_441_139_279_814
        )


class TestIsNonemptyFile:
    def test_only_a_written_regular_file_counts(self, tmp_path):
        empty = tmp_path / "empty.npy"
        empty.touch()
        written = tmp_path / "written.npy"
        written.write_bytes(b"x")

        assert not is_nonempty_file(tmp_path / "missing.npy")
        assert not is_nonempty_file(empty)
        assert not is_nonempty_file(tmp_path)
        assert is_nonempty_file(written)


class TestJsonSafe:
    def test_non_finite_floats_become_none_at_every_depth(self):
        sanitized = json_safe(
            {
                "delta": float("nan"),
                "nested": {"upper": float("inf"), "lower": -np.inf},
                "series": [1.5, float("nan"), (np.float32("nan"), 2)],
                "kept": {"count": 3, "name": "model", "flag": True, "none": None},
            }
        )

        assert sanitized == {
            "delta": None,
            "nested": {"upper": None, "lower": None},
            "series": [1.5, None, [None, 2]],
            "kept": {"count": 3, "name": "model", "flag": True, "none": None},
        }

    def test_the_result_is_strict_json(self):
        def refuse(token: str) -> None:
            raise AssertionError(f"non-JSON token {token!r}")

        text = json.dumps(json_safe({"delta": float("nan"), "finite": 0.25}))

        assert json.loads(text, parse_constant=refuse) == {
            "delta": None,
            "finite": 0.25,
        }


def test_matching_sidecar_needs_a_mapping_that_records_the_contract(tmp_path):
    artifact = tmp_path / "bundle.pkl"
    sidecar = fit_provenance_path(artifact)
    assert matching_sidecar(artifact, {}) is None  # no sidecar
    sidecar.write_text("")
    assert matching_sidecar(artifact, {}) is None  # empty
    sidecar.write_text("{not json")
    assert matching_sidecar(artifact, {}) is None
    sidecar.write_text("[1, 2]")
    assert matching_sidecar(artifact, {}) is None  # not a mapping
    sidecar.write_text(json.dumps({"version": 3, "status": "complete"}))
    assert matching_sidecar(artifact, {"version": 3}) == {
        "version": 3,
        "status": "complete",
    }
    assert matching_sidecar(artifact, {"version": 2}) is None
    # an unrecorded key reads as None
    assert matching_sidecar(artifact, {"version": 3, "absent": None}) is not None
    assert matching_sidecar(artifact, {"absent": 0}) is None


def test_remove_bundle_unlinks_sidecar_table_and_companions(tmp_path):
    artifact = tmp_path / "bundle.pkl"
    companions = [tmp_path / "bundle.a.npy", tmp_path / "bundle.b.npy"]
    for path in (artifact, fit_provenance_path(artifact), companions[0]):
        path.write_text("x")
    (tmp_path / "other.pkl").write_text("x")

    remove_bundle(artifact, companions)  # one companion was never written

    assert sorted(path.name for path in tmp_path.iterdir()) == ["other.pkl"]


@pytest.mark.parametrize(
    ("n_not_converged", "n_units", "fails"),
    [
        (0, 0, False),
        (0, 7, False),
        (1, 10, False),  # exactly a tenth is still certified
        (2, 10, True),
        (10, 100, False),
        (11, 100, True),
        (1, 9, True),  # a ninth is more than a tenth
        (3, 3, True),
    ],
)
def test_more_than_a_tenth_of_units_not_converged_fails_the_gate(
    n_not_converged, n_units, fails
):
    assert fails_convergence_gate(n_not_converged, n_units) is fails
