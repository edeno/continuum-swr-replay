"""Provenance labels on the SWR-contrast content runner must name the beta value construct.

The content value rungs (``C4_armvalue``/``CS5``) use the beta-bernoulli per-arm value
(``q_arm``), not the retired HMM Global-Model value. The saved provenance record must
say so, and must not list per-arm value as still ``deferred``.
"""

from __future__ import annotations

import argparse

from scripts.replay_models.content import fit_swr_contrast_content as mod


def test_q_arm_source_is_beta():
    args = argparse.Namespace(
        max_per_animal=None,
        response="fraction",
        estimator="conditional_poisson",
        min_events_session_phi=40,
        outcome_stratum="all",
    )
    prov = mod._provenance(args, n_sessions=218)

    assert "beta" in prov["q_arm_source"].lower()
    assert "reward_belief_mean_candidate_arm_preoutcome" in prov["q_arm_source"]
    # The stale label must be gone.
    assert "harmonization deferred" not in prov["q_arm_source"]

    # beta per-arm value is implemented (model_specs C4_armvalue = [*BASEc, "reward_belief_mean_candidate_arm_preoutcome"]),
    # so it must no longer appear in the deferred list.
    deferred = " ".join(prov["deferred"]).lower()
    assert "per-arm q_arm" not in deferred
