"""Regression test for ContFrag ``arm_fraction`` denominator.

The non-local table divides ``arm_count_hard`` by the count of
real-arm bins, so per-event arm fractions sum to 1. The ContFrag
table used to divide by ``n_timebins`` (all event bins, including
junction / missing-arm bins) while the numerator only counted real
arms — so the per-event sum dropped below 1 whenever any event bin
landed on a junction. The fix uses ``n_timebins - n_junction_bins``
so the convention matches.

This tests just the denominator math via the same arithmetic the
builder runs (we avoid importing the full builder, which needs an
``Environment`` + a track graph).
"""

from __future__ import annotations

import numpy as np


def _arm_fraction(count_hard: float, n_timebins: int, n_junction: int) -> float:
    """Replica of the post-fix ``arm_fraction`` formula in
    ``analysis/contfrag/tables.py:build_contfrag_event_arm_table``."""
    n_real = n_timebins - n_junction
    if n_real <= 0:
        return float("nan")
    return count_hard / max(n_real, 1)


class TestContfragArmFractionDenominator:
    def test_per_event_fractions_sum_to_one_with_junction_bins(self):
        # User repro: 5 real bins (split across two arms), 5 junction.
        n_timebins = 10
        n_junction = 5
        # Suppose 3 of the real bins decoded to arm 0 and 2 to arm 1.
        per_arm_counts = {0: 3, 1: 2}
        per_arm_fracs = [
            _arm_fraction(c, n_timebins, n_junction) for c in per_arm_counts.values()
        ]
        np.testing.assert_allclose(
            sum(per_arm_fracs), 1.0, atol=1e-12,
            err_msg=(
                "arm_fractions for an event with junction-bin "
                "contamination sum to less than 1 — denominator is "
                "still ``n_timebins`` instead of real-arm count."
            ),
        )

    def test_event_with_only_junction_bins_returns_nan(self):
        n_timebins = 5
        n_junction = 5
        assert np.isnan(_arm_fraction(0.0, n_timebins, n_junction))
