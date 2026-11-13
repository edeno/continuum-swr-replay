"""Tests for the PFC runners' seeding and convergence gate."""

from __future__ import annotations

import pytest

from continuum_swr_replay.analysis.pfc.runner_support import (
    fails_convergence_gate,
    session_seed,
)


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
