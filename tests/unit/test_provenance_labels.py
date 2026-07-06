"""Provenance labels on the SWR-contrast content runner must name the HMM value construct.

The content value rungs (``C4_armvalue``/``CS5``) use ``q_arm_hmm`` (the HMM
Global-Model per-arm value), not the beta ``q_arm``. The saved provenance record
must say so, and must not list HMM per-arm value as still ``deferred``.
"""

from __future__ import annotations

import argparse

from scripts.content import fit_swr_contrast_content as mod


def test_q_arm_source_is_hmm():
    prov = mod._provenance(argparse.Namespace(max_per_animal=None), n_sessions=218)

    assert "HMM" in prov["q_arm_source"]
    assert "q_arm_hmm" in prov["q_arm_source"]
    # The stale label must be gone.
    assert "harmonization deferred" not in prov["q_arm_source"]
    assert "beta-model per-arm value" not in prov["q_arm_source"]

    # HMM per-arm value is implemented (model_specs C4_armvalue = [*BASEc, "q_arm_hmm"]),
    # so it must no longer appear in the deferred list.
    deferred = " ".join(prov["deferred"]).lower()
    assert "hmm per-arm q_arm" not in deferred
