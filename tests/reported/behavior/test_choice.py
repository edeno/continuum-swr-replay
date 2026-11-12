"""The behavior summary's choice and stay/leave values, on fits small enough
to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported.behavior import choice
from continuum_swr_replay.reported.behavior.choice import stay_leave_values
from tests.support._behavior_fits import ANIMALS, coefficient_fits, values_by_key

NAN = np.nan


def _logistic(rows: dict[str, list[float]], other: list[str]) -> pd.DataFrame:
    """Per-session fits: converged ones with the given ``beta_inter`` per role,
    and one row of role ``current`` per other status."""
    converged = [
        {"role": role, "status": "ok", "beta_inter": beta}
        for role, betas in rows.items()
        for beta in betas
    ]
    failed = [{"role": "current", "status": s, "beta_inter": NAN} for s in other]
    return pd.DataFrame(converged + failed)


def test_stay_leave_values_read_each_stratum_and_each_test():
    bundles = {
        "all": (
            # six sessions, all positive: the exact two-sided p is 2 / 2 ** 6,
            # and the one testable role's adjusted p is the same
            _logistic(
                {"current": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
                ["too_few_stops_3", "target_constant", "not_converged"],
            ),
            # adjusted across two roles: 0.04 * 2 and 0.2
            pd.DataFrame(
                {"role": ["current", "prev"], "status": "ok", "p_inter": [0.2, 0.04]}
            ),
        ),
        "swr": (
            # best, five sessions all positive: p = 2 / 2 ** 5 = 0.0625, adjusted
            # across the two testable roles to 0.125
            _logistic(
                {
                    "prev": [0.5, -0.1, 0.2, -0.3, 0.4],
                    "best": [0.1, 0.2, 0.3, 0.4, 0.5],
                },
                ["fit_error_LinAlgError"],
            ),
            # adjusted across the two converged roles: 0.01 * 2 and 0.5; the
            # failed fit's p is not used
            pd.DataFrame(
                {
                    "role": ["current", "prev", "best"],
                    "status": ["ok", "ok", "fit_error_ValueError"],
                    "p_inter": [0.01, 0.5, 0.0001],
                }
            ),
        ),
    }
    stops = pd.DataFrame(
        {
            "occupancy_state": ["at_well"] * 3
            + ["mixed"] * 2
            + ["in_transit"] * 2
            + ["unknown"],
            "is_stay": [1.0, 0.0, NAN, 0.0, NAN, 1.0, NAN, NAN],
        }
    )
    values = {v.key: v for v in stay_leave_values(bundles, stops, "x")}
    read = {key: value.value for key, value in values.items()}
    assert read["stay_converged_all"] == 6
    assert read["stay_fits_all"] == 9
    assert read["stay_converged_swr"] == 10
    assert read["stay_fits_swr"] == 11
    assert read["stay_min_q_signrank"] == 0.03125
    assert "role current, stratum all" in values["stay_min_q_signrank"].description
    assert read["stay_min_q_gee"] == pytest.approx(0.02)
    assert "role current, stratum swr" in values["stay_min_q_gee"].description
    assert read["stops_total"] == 8
    assert read["stops_well_anchored"] == 5
    assert read["stops_dropped"] == 3
    assert read["stops_dropped_with_stay"] == 1
    assert read["stops_mixed"] == 2
    assert "1 of them with an is_stay" in values["stops_mixed"].description
    assert read["stops_mixed_with_stay"] == 1
    assert read["stops_at_well_with_stay"] == 2


def test_choice_values_read_each_block_s_own_term():
    non_swr = [0.1, 0.2, 0.3, 0.4, 0.5, -0.6]  # median 0.25, five positive
    swr = [-0.1, -0.2, -0.3, -0.4, 0.5, 0.6]  # median -0.15, two positive
    split = coefficient_fits("replay_fraction_non_swr", non_swr, 0.1).merge(
        coefficient_fits("replay_fraction_swr", swr, 0.1)
    )
    fits = {
        "choice_swr_split": {"C_full_swr_split": split},
        "choice": {
            "C_full": coefficient_fits("replay_fraction", [0.3] * 6, 0.1),
            "C_replay_only": coefficient_fits("replay_fraction", [-0.9] * 6, 0.1),
        },
    }
    values = values_by_key(choice.choice_values(fits, "source"))
    assert values["choice_non_swr"] == pytest.approx(0.25)
    assert values["choice_swr"] == pytest.approx(-0.15)
    assert values["choice_full"] == pytest.approx(0.3)
    assert values["choice_raw"] == pytest.approx(-0.9)
    assert values["choice_non_swr_sessions"] == 6
    assert values["choice_non_swr_positive_sessions"] == 5
    assert values["choice_non_swr_negative_sessions"] == 1
    assert values["choice_swr_positive_sessions"] == 2
    assert values["choice_swr_negative_sessions"] == 4
    assert values["choice_full_animals_positive"] == 5
    assert values["choice_raw_animals_positive"] == 0
    # |coefficient| / 0.1 is 1 to 6: all but the first are above 1.96
    assert values["choice_non_swr_significant_fraction"] == pytest.approx(5 / 6)
    # six sessions of one sign: two-sided exact p = 2 / 2 ** 6
    assert values["choice_full_p"] == pytest.approx(0.03125)


def test_choice_values_read_through_a_refusing_gate_are_pending():
    # Twelve sessions. The SWR term is separated in three of them, a quarter,
    # which is more than the per-term gate lets be left out: the audited
    # aggregates would publish no row for it. Its seven values are kept, over
    # the nine sessions left, and marked pending; the other terms' are not.
    swr, non_swr = "replay_fraction_swr", "replay_fraction_non_swr"
    split = pd.DataFrame(
        {
            "animal": list(ANIMALS) * 2,
            "session_date": list(range(12)),
            "epoch": "02_r1",
            f"coef_{non_swr}": 0.2,
            f"sd_{non_swr}": 0.1,
            f"bse_{non_swr}": 0.1,
            f"coef_{swr}": -0.3,
            f"sd_{swr}": 0.1,
            f"bse_{swr}": 0.1,
            "separated_terms": [swr, swr, "", "", "", "", swr, "", "", "", "", ""],
        }
    )
    fits = {
        "choice_swr_split": {"C_full_swr_split": split},
        "choice": {
            "C_full": coefficient_fits("replay_fraction", [0.3] * 6, 0.1),
            "C_replay_only": coefficient_fits("replay_fraction", [-0.9] * 6, 0.1),
        },
    }

    values = {value.key: value for value in choice.choice_values(fits, "source")}

    pending = {key for key, value in values.items() if value.pending is not None}
    assert pending == {
        "choice_swr",
        "choice_swr_p",
        "choice_swr_sessions",
        "choice_swr_positive_sessions",
        "choice_swr_negative_sessions",
        "choice_swr_animals_positive",
        "choice_swr_significant_fraction",
    }
    assert values["choice_swr_sessions"].value == 9
    assert values["choice_non_swr_sessions"].value == 12
    note = values["choice_swr"].pending
    assert "the per-term session gate refuses replay_fraction_swr" in note
    assert "more than 20 % of the sessions" in note
