"""Output-directory keying for partial content-ladder runs.

A run that fits only some rungs must not publish into the directory that holds
the full ladder: the canonical ``models/`` bundle is a single cohort-matched
set, and a partial run silently replacing part of it leaves rungs that were fit
on different event sets side by side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from continuum_swr_replay.analysis.replay.model_specs import (
    default_content_model_names,
)
from scripts.content import fit_replay_content_models as mod


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "event_source": "non_local",
        "hierarchy": "arm",
        "brain_area": "HPC",
        "swr_subset": "all",
        "grain": "timebin",
        "input_dir": Path("/tmp/content-in"),
        "output_dir": None,
        "models": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_run_writes_the_canonical_directory() -> None:
    config = mod._resolve_run_config(_args())
    assert config.output_dir == Path("/tmp/content-in/models")
    assert list(config.model_names) == list(
        default_content_model_names(config.hierarchy)
    )


def test_explicit_full_ladder_still_writes_the_canonical_directory() -> None:
    """Naming every default rung is the canonical run, not a subset."""
    names = default_content_model_names(mod.MODEL_HIERARCHY)
    config = mod._resolve_run_config(_args(models=",".join(names)))
    assert config.output_dir == Path("/tmp/content-in/models")


def test_subset_run_is_keyed_by_its_rungs() -> None:
    config = mod._resolve_run_config(_args(models="M4,Mv,Mu,Mvu"))
    assert config.output_dir == Path("/tmp/content-in/models_M4-Mv-Mu-Mvu")
    assert list(config.model_names) == ["M4", "Mv", "Mu", "Mvu"]


def test_patch_subset_run_is_keyed_off_the_patch_directory() -> None:
    config = mod._resolve_run_config(_args(hierarchy="patch", models="P3,Pv"))
    assert config.output_dir == Path("/tmp/content-in/models_patch_P3-Pv")


def test_subset_models_do_not_write_canonical_dir() -> None:
    """An explicit --output-dir cannot aim a subset at the canonical bundle."""
    with pytest.raises(ValueError, match="models"):
        mod._resolve_run_config(
            _args(models="M4,Mv", output_dir=Path("/tmp/content-in/models"))
        )
    with pytest.raises(ValueError, match="models_patch"):
        mod._resolve_run_config(
            _args(
                hierarchy="patch",
                models="P3,Pv",
                output_dir=Path("/tmp/content-in/models_patch"),
            )
        )


def test_subset_run_may_be_sent_to_an_explicit_non_canonical_directory() -> None:
    config = mod._resolve_run_config(
        _args(models="M4,Mv", output_dir=Path("/tmp/staging/models_belief"))
    )
    assert config.output_dir == Path("/tmp/staging/models_belief")
