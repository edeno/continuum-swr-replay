"""Regression test for ContFrag dominant-state NaN handling.

The per-event Continuous/Fragmented summary used to assign
``dominant="Fragmented"`` and ``continuous_fraction=0.0`` whenever
``cont_mean >= frag_mean`` was False. Because ``NaN >= anything`` and
``anything >= NaN`` are both False, that silently routed two
degenerate cases into a confident "Fragmented / 0% Continuous" label:

  1. **Both** state probabilities all-NaN (degenerate decoder output).
  2. **One** state's probabilities all-NaN (asymmetric missingness):
     e.g. Continuous_prob finite, Fragmented_prob all NaN. The
     comparison ``cont > frag`` returns False bin-wise (NaN
     comparisons), so ``cont_frac`` collapses to 0.0.

The fix lives in
``analysis.contfrag.tables._summarize_event_state``. This test
exercises that production helper directly (no local mirror) so the
test fails if the builder is rewired to bypass it or if the helper's
contract regresses.
"""

from __future__ import annotations

import numpy as np

from continuum_swr_replay.analysis.contfrag.tables import _summarize_event_state


class TestContfragDominantStateNaNGuard:
    def test_all_nan_state_probs_yield_nan_summary(self):
        cont = np.array([np.nan, np.nan, np.nan])
        frag = np.array([np.nan, np.nan, np.nan])
        cm, fm, dom, cf = _summarize_event_state(cont, frag)
        assert np.isnan(cm)
        assert np.isnan(fm)
        assert dom is None, f"dominant_state leaked: {dom!r}"
        assert np.isnan(cf), (
            f"continuous_fraction collapsed to {cf} for an event with "
            "no finite state-probability data — should propagate NaN."
        )

    def test_asymmetric_finite_continuous_only_yields_no_label(self):
        # User repro: Continuous_prob finite, Fragmented_prob all NaN.
        # Pre-fix output silently became dominant="Fragmented",
        # continuous_fraction=0.0 because cont >= NaN is False and
        # cont > NaN is False bin-wise.
        cm, fm, dom, cf = _summarize_event_state(
            np.array([0.7, 0.8]),
            np.array([np.nan, np.nan]),
        )
        assert np.isclose(cm, 0.75), f"cont mean lost on asymmetric input: {cm}"
        assert np.isnan(fm), f"frag mean fabricated on all-NaN input: {fm}"
        assert dom is None, (
            f"dominant_state still labeled as {dom!r} when one state "
            "has no finite bin — comparison is undefined and should "
            "surface as missing."
        )
        assert np.isnan(cf), (
            f"continuous_fraction={cf} on asymmetric all-NaN frag — "
            "should propagate NaN, not collapse to 0.0."
        )

    def test_asymmetric_finite_fragmented_only_yields_no_label(self):
        # Mirror case: Fragmented_prob finite, Continuous_prob all NaN.
        cm, fm, dom, cf = _summarize_event_state(
            np.array([np.nan, np.nan]),
            np.array([0.7, 0.8]),
        )
        assert np.isnan(cm)
        assert np.isclose(fm, 0.75)
        assert dom is None
        assert np.isnan(cf)

    def test_finite_state_probs_pick_continuous_when_dominant(self):
        cm, fm, dom, cf = _summarize_event_state(
            np.array([0.7, 0.8, 0.6]), np.array([0.2, 0.1, 0.2])
        )
        assert dom == "Continuous"
        assert 0.0 < cf <= 1.0

    def test_finite_state_probs_pick_fragmented_when_dominant(self):
        cm, fm, dom, cf = _summarize_event_state(
            np.array([0.2, 0.1, 0.2]), np.array([0.7, 0.8, 0.6])
        )
        assert dom == "Fragmented"
        assert 0.0 <= cf < 1.0

    def test_partial_nan_does_not_trigger_missing_branch(self):
        # If at least one bin per state is finite, the function still
        # produces a real label / fraction (NaN propagation in nanmean
        # is fine for this case).
        cm, fm, dom, cf = _summarize_event_state(
            np.array([0.7, np.nan, 0.6]), np.array([0.2, np.nan, 0.2])
        )
        assert dom == "Continuous"
        assert not np.isnan(cf)

    def test_partial_nan_uses_paired_finite_only_for_cont_frac(self):
        # User repro: cont=[0.7, NaN, 0.6], frag=[0.2, NaN, 0.2].
        # Pre-fix: ``np.nanmean(cont > frag)`` evaluates to nanmean of
        # [True, False, True] = 0.6667 because NaN comparisons collapse
        # to False BEFORE nanmean sees them. Post-fix: only the two
        # paired-finite bins contribute, so cont_frac is 1.0.
        cm, fm, dom, cf = _summarize_event_state(
            np.array([0.7, np.nan, 0.6]),
            np.array([0.2, np.nan, 0.2]),
        )
        assert dom == "Continuous"
        assert np.isclose(cf, 1.0), (
            f"continuous_fraction={cf} — expected 1.0 (Continuous won "
            "in both paired-finite bins). Pre-fix bug returned 0.6667 "
            "by counting the NaN bin as Fragmented."
        )

    def test_partial_nan_asymmetric_pairs_uses_paired_finite_only(self):
        # cont=[0.1, 0.8, 0.6], frag=[0.9, NaN, 0.2]. Paired-finite
        # bins are indices 0 and 2: cont=[0.1, 0.6] vs frag=[0.9, 0.2]
        # → cont > frag is [False, True] → cont_frac = 0.5.
        # Pre-fix would have evaluated ``cont > frag`` over all three
        # bins (with the NaN-frag bin returning False), giving
        # 1/3 ≈ 0.333.
        cm, fm, dom, cf = _summarize_event_state(
            np.array([0.1, 0.8, 0.6]),
            np.array([0.9, np.nan, 0.2]),
        )
        assert dom is not None
        assert np.isclose(cf, 0.5), (
            f"continuous_fraction={cf} — expected 0.5 from the two "
            "paired-finite bins; pre-fix returned ~0.333 by counting "
            "the asymmetric NaN bin as a Fragmented win."
        )

    def test_disjoint_finite_bins_yield_nan_cont_frac(self):
        # cont=[0.7, NaN], frag=[NaN, 0.2]. Both states have a finite
        # bin, but they don't overlap on any row, so the bin-wise
        # comparison is undefined.
        cm, fm, dom, cf = _summarize_event_state(
            np.array([0.7, np.nan]),
            np.array([np.nan, 0.2]),
        )
        # Means still computable per-state.
        assert np.isclose(cm, 0.7)
        assert np.isclose(fm, 0.2)
        # Dominant is still computable from the means even with no
        # paired bin.
        assert dom == "Continuous"
        # cont_frac has no defined comparison set → NaN.
        assert np.isnan(cf), (
            f"continuous_fraction={cf} on disjoint-finite bins; the "
            "per-bin comparison has no overlap and should be NaN."
        )
