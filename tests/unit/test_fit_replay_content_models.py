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


def test_subset_may_not_target_another_configurations_canonical_bundle() -> None:
    """The guard covers every canonical bundle, not just this invocation's.

    A patch run pointed at ``models/``, or an SWR-subset run pointed at the
    unrestricted ``models/``, would replace part of a bundle the run does not
    even resolve to.
    """
    with pytest.raises(ValueError, match="canonical models/ bundle"):
        mod._resolve_run_config(
            _args(
                hierarchy="patch",
                models="P3,Pv",
                output_dir=Path("/tmp/content-in/models"),
            )
        )
    with pytest.raises(ValueError, match="canonical models/ bundle"):
        mod._resolve_run_config(
            _args(
                swr_subset="T",
                models="M0,M1",
                output_dir=Path("/tmp/content-in/models"),
            )
        )


def test_canonical_bundle_names_cover_every_suffix_variant() -> None:
    names = mod.canonical_output_subdirs()
    assert {"models", "models_patch"} <= names
    assert {"models_swr_T", "models_patch_swr_F"} <= names
    assert {"models_event_grain", "models_swr_T_event_grain"} <= names
    assert {"models_swr_HPC", "models_patch_mua_OFC"} <= names
    # Every name a full run of some configuration resolves to is in the set.
    for hierarchy in ("arm", "patch"):
        for swr_subset in ("all", "T"):
            for grain in ("timebin", "event"):
                config = mod._resolve_run_config(
                    _args(hierarchy=hierarchy, swr_subset=swr_subset, grain=grain)
                )
                assert config.output_dir.name in names


def test_swr_subset_run_keeps_its_own_suffixed_bundle() -> None:
    config = mod._resolve_run_config(_args(swr_subset="T", grain="event"))
    assert config.output_dir == Path("/tmp/content-in/models_swr_T_event_grain")
