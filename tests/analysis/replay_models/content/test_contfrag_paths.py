"""Regression tests for the ContFrag prep ↔ fitter filename contract.

These tests pin the path-resolution logic so the wiring between
``analysis.contfrag.prepare_data`` (which writes
``event_arm_table_{source}_{brain_area}``) and
``fit_replay_content_models.py`` (which resolves
``--event-source`` + ``--brain-area`` into matching reads + a
distinct ``models_{source}_{brain_area}/`` output dir, through
``analysis.replay_models.content.fit_outputs.resolve_contfrag_paths``) cannot
silently regress.

The helpers exercised here are pure string formatting, so the tests run
on any machine.
"""

from __future__ import annotations

import pytest

from continuum_swr_replay.analysis.contfrag.prepare_data import (
    contfrag_table_basenames,
)
from continuum_swr_replay.analysis.replay_models.content.fit_outputs import (
    resolve_contfrag_paths,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (
    default_content_model_names,
)


class TestResolveContfragPaths:
    def test_non_local_arm_unsuffixed(self):
        table, subdir = resolve_contfrag_paths("non_local", "HPC", "candidate_arm")
        assert table == "event_arm_table"
        assert subdir == "models"

    def test_non_local_patch_unsuffixed(self):
        table, subdir = resolve_contfrag_paths("non_local", "HPC", "patch")
        assert table == "event_patch_table"
        assert subdir == "models_patch"

    @pytest.mark.parametrize(
        ("source", "brain_area", "expected_table", "expected_subdir"),
        [
            ("SWR", "HPC", "event_arm_table_swr_HPC", "models_swr_HPC"),
            ("SWR", "mPFC", "event_arm_table_swr_mPFC", "models_swr_mPFC"),
            ("MUA", "HPC", "event_arm_table_mua_HPC", "models_mua_HPC"),
            ("MUA", "OFC", "event_arm_table_mua_OFC", "models_mua_OFC"),
        ],
    )
    def test_contfrag_arm_per_source_and_area(
        self, source, brain_area, expected_table, expected_subdir
    ):
        table, subdir = resolve_contfrag_paths(source, brain_area, "candidate_arm")
        assert table == expected_table
        assert subdir == expected_subdir

    def test_swr_and_mua_resolve_to_distinct_subdirs(self):
        # Round-tripping the same brain_area through different sources
        # must not clobber a sibling fit's outputs.
        _, swr_dir = resolve_contfrag_paths("SWR", "HPC", "candidate_arm")
        _, mua_dir = resolve_contfrag_paths("MUA", "HPC", "candidate_arm")
        assert swr_dir != mua_dir

    def test_hpc_and_mpfc_resolve_to_distinct_subdirs(self):
        _, hpc_dir = resolve_contfrag_paths("SWR", "HPC", "candidate_arm")
        _, mpfc_dir = resolve_contfrag_paths("SWR", "mPFC", "candidate_arm")
        assert hpc_dir != mpfc_dir


def test_default_content_ladder_excludes_every_belief_dependent_rung():
    hierarchy = {name: object() for name in ("M0", "M5", "M6", "M7", "Mv", "Mu", "Mvu")}

    assert default_content_model_names(hierarchy) == ("M0", "M5", "M6", "M7")
    assert default_content_model_names(
        {"P0": object(), "P4": object(), "Pv": object()}
    ) == ("P0", "P4")


class TestPrepFitterContract:
    """The prep saver and the fitter resolver must agree on the same
    ``event_arm_table_{source}_{brain_area}`` basename for every
    ContFrag (source, brain_area) combo."""

    @pytest.mark.parametrize("source", ["SWR", "MUA"])
    @pytest.mark.parametrize("brain_area", ["HPC", "mPFC", "OFC"])
    def test_event_arm_table_basenames_match(self, source, brain_area):
        prep_basenames = contfrag_table_basenames(source, brain_area)
        fitter_table, _ = resolve_contfrag_paths(source, brain_area, "candidate_arm")
        assert prep_basenames["event_arm_table"] == fitter_table
