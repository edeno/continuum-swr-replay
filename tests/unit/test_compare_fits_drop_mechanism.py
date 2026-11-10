"""The refit audit must not report a runaway coefficient as a separation flag.

``aggregate_contrast`` excludes a session from a term's aggregate either because
the fitters named that coefficient in ``separated_terms`` or because it exceeded
the magnitude backstop, read on the effect scale as ``|coef| * sd >= 10`` (raw
``|coef| >= 10`` only for legacy frames with no ``sd_`` column). The audit
reports one total, so the decomposition is what keeps a reader from quoting a
backstop exclusion as a separation diagnosis -- and it must use the same
predicate as the aggregation, or the raise below fires on a correct aggregate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.qa.compare_fits import drop_mechanism_table

TERM = "is_swr:patch_switch_fraction_last5_completed_trips"


def _contrast(
    rows: list[tuple[str, float, str]], population: str, sd: float | None = 1.0
) -> pd.DataFrame:
    """Per-session fits: (animal, coefficient, ``separated_terms``) per session.

    ``sd`` is the term's persisted predictor SD, shared by every session; ``None``
    writes no ``sd_`` column, which is what a legacy artifact looks like.
    """
    fits = pd.DataFrame(
        {
            "animal": [animal for animal, _, _ in rows],
            "model": "S5_switchrate",
            "population": population,
            "cohort": "primary",
            f"coef_{TERM}": [coef for _, coef, _ in rows],
            "separated_terms": [flagged for _, _, flagged in rows],
        }
    )
    if sd is not None:
        fits[f"sd_{TERM}"] = sd
    return fits


def _audited(population: str, n_dropped: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["S5_switchrate"],
            "population": [population],
            "cohort": ["primary"],
            "term": [TERM],
            "n_dropped": [n_dropped],
        }
    )


def test_flag_and_backstop_exclusions_are_counted_apart():
    """One flagged session, one runaway coefficient, one that is both."""
    contrast = _contrast(
        [
            ("chimi", 0.4, TERM),  # flagged, magnitude ordinary
            ("j16", 12.5, ""),  # runaway, never flagged
            ("senor", -18.0, f"is_swr|{TERM}"),  # both
            ("wilbur", 0.2, ""),  # kept
        ],
        "contrast",
    )
    out = drop_mechanism_table(contrast, _audited("contrast", 3))
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row.n_flag_only) == 1
    assert int(row.n_backstop_only) == 1
    assert int(row.n_both) == 1


def test_a_session_flagged_for_another_term_only_is_not_excluded():
    """``separated_terms`` membership is per term, not per fit."""
    contrast = _contrast(
        [
            ("chimi", 0.4, "is_swr:hist_rcos_0"),
            ("j16", 0.3, "reward_history_rcos_1"),
        ],
        "contrast",
    )
    out = drop_mechanism_table(contrast, _audited("contrast", 0))
    row = out.iloc[0]
    assert int(row.n_flag_only) == 0
    assert int(row.n_backstop_only) == 0
    assert int(row.n_both) == 0


def test_a_steep_slope_on_a_narrow_covariate_is_not_a_backstop_exclusion():
    """The backstop is the log-rate swing over one SD, not the raw coefficient.

    ``|coef| = 12`` with an SD of 0.01 is a swing of 0.12: identified, kept by
    ``aggregate_contrast``, and so counted here as no exclusion at all.
    """
    contrast = _contrast([("chimi", 12.0, ""), ("j16", -40.0, "")], "contrast", sd=0.01)
    out = drop_mechanism_table(contrast, _audited("contrast", 0))
    row = out.iloc[0]
    assert int(row.n_flag_only) == 0
    assert int(row.n_backstop_only) == 0
    assert int(row.n_both) == 0


def test_legacy_frame_without_predictor_sds_uses_the_raw_coefficient():
    """With no ``sd_`` column the only available scale is the coefficient itself."""
    contrast = _contrast([("chimi", 12.0, ""), ("j16", 0.5, "")], "contrast", sd=None)
    out = drop_mechanism_table(contrast, _audited("contrast", 1))
    assert int(out.iloc[0].n_backstop_only) == 1


def test_stratified_population_labels_are_translated():
    """The audited aggregate says ``SWR``; the fits say ``non_local_swr``."""
    contrast = _contrast([("chimi", 11.0, ""), ("j16", 0.5, "")], "non_local_swr")
    out = drop_mechanism_table(contrast, _audited("SWR", 1))
    assert int(out.iloc[0].n_backstop_only) == 1


def test_a_decomposition_that_misses_the_audited_total_raises():
    """A silent mismatch would annotate the aggregate with other sessions."""
    contrast = _contrast([("chimi", 0.4, ""), ("j16", 0.5, "")], "contrast")
    with pytest.raises(ValueError, match="recomputed 0 excluded sessions"):
        drop_mechanism_table(contrast, _audited("contrast", 2))
