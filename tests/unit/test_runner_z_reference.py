"""F2: the contrast runner persists its fit-time _z reference and it is reproducible.

The is_swr-contrast content runner standardizes its event scalars over the *event
cohort* at fit time (its own scope, distinct from the builder's pooled-global table
_z). The runner is NOT repointed at the builder _z; instead it persists the
run's own means/SDs into the provenance so the standardized coefficients are
reproducible. This locks that in.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.standardization import attach_standardized
from scripts.content import fit_swr_contrast_content as mod


def test_provenance_includes_fit_time_z_reference():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "hmm_rpe": rng.normal(size=200),
        "reward_entropy": rng.normal(2.0, 0.5, size=200),
    })
    ref = mod._z_reference(frame, ["hmm_rpe", "reward_entropy"])
    prov = mod._provenance(argparse.Namespace(max_per_animal=None), 218, ref)

    assert prov["fit_time_z_reference"] == ref
    for c in ("hmm_rpe", "reward_entropy"):
        assert ref[c]["mean"] == float(np.nanmean(frame[c].to_numpy()))
        assert ref[c]["std"] == float(np.nanstd(frame[c].to_numpy()))


def test_full_reference_covers_arm_tsc_and_patch_cohort():
    """The persisted reference must cover EVERY standardization the run computes: arm
    event scalars, the arm/patch within-stop-time control, and the SEPARATE patch
    cohort scalars (a different scope than the same-named arm entries)."""
    rng = np.random.default_rng(0)
    n = 100
    ev = pd.DataFrame({
        **{c: rng.normal(size=n) for c in mod._ZSCALARS},
        "time_since_stop": rng.uniform(0, 3, n),
    })
    pt = pd.DataFrame({
        "trials_since_switch": rng.normal(size=n),
        "recent_switch_rate": rng.uniform(0, 1, n),
        "time_since_stop": rng.uniform(0, 5, n),  # patch cohort: different values
    })
    ref = mod._fit_z_reference(ev, pt)

    assert set(ref["arm_event_scalars"]) == set(mod._ZSCALARS)
    assert "log1p_time_since_stop" in ref["arm_within_stop_time"]
    assert set(ref["patch_cohort"]) == {
        "trials_since_switch", "recent_switch_rate", "log1p_time_since_stop"
    }
    # The patch within-stop-time uses log1p over the PATCH cohort (its own scope).
    import numpy as _np
    assert ref["patch_cohort"]["log1p_time_since_stop"]["mean"] == float(
        _np.nanmean(_np.log1p(pt["time_since_stop"].to_numpy(float)))
    )
    # The reference reproduces the runner's own _z for a patch scalar.
    from continuum_swr_replay.analysis.replay.standardization import attach_standardized
    applied, _ = attach_standardized(
        pt[["trials_since_switch"]].copy(), ["trials_since_switch"],
        reference={"trials_since_switch": ref["patch_cohort"]["trials_since_switch"]},
    )
    np.testing.assert_allclose(
        applied["trials_since_switch_z"].to_numpy(),
        mod._z(pt["trials_since_switch"].to_numpy(float)), atol=1e-12,
    )


def test_reference_reproduces_runner_z():
    """Re-applying the persisted reference reproduces the runner's own _z() output."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"hmm_rpe": rng.normal(3.0, 2.0, size=150)})
    ref = mod._z_reference(frame, ["hmm_rpe"])

    applied, _ = attach_standardized(frame.copy(), ["hmm_rpe"], reference=ref)
    runner_z = mod._z(frame["hmm_rpe"].to_numpy(dtype=float))
    np.testing.assert_allclose(applied["hmm_rpe_z"].to_numpy(), runner_z, atol=1e-12)


def test_z_reference_skips_absent_columns():
    frame = pd.DataFrame({"hmm_rpe": [1.0, 2.0, 3.0]})
    ref = mod._z_reference(frame, ["hmm_rpe", "not_a_column"])
    assert set(ref) == {"hmm_rpe"}
