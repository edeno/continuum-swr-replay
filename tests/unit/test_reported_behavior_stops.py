"""The behavior summary's stop-table, stay/leave and cohort-audit values, on
tables small enough to check by hand."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.reported.behavior import (
    BehaviorInputs,
    cohort_audit_values,
    stay_leave_values,
    stop_table_values,
)

NAN = np.nan


def _stops() -> pd.DataFrame:
    """Eight stops of one session; the first six are well-anchored.

    Spans 3, 5, 8, 1, 4.5 and 0.5 s; stops 2, 5 and 6 hold excluded time, so
    their analysed immobility is shorter. Stops 1 and 4 learn their outcome
    inside them, within the 4 s the design keeps. Stop 3 learns it 5 s in.
    Stop 5 learns it 1 s in, but contains stop 6's arrival (52 s). Stop 2
    contains stop 3's arrival (24 s) and learned its outcome before it began.
    """
    return pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "stop_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "occupancy_state": ["at_well"] * 4
            + ["mixed", "at_well", "in_transit", "in_transit"],
            "is_rewarded_outcome": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, NAN, 1.0],
            "stop_start": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "stop_end": [13.0, 25.0, 38.0, 41.0, 54.5, 60.5, 72.0, 86.0],
            "immobility_duration_s": [3.0, 1.0, 8.0, 1.0, 4.0, 0.25, 2.0, 6.0],
            "outcome_known_time_s": [11.0, 19.0, 35.0, 40.5, 51.0, 59.0, NAN, NAN],
            "last_well_arrival_time_s": [9.0, 18.0, 24.0, 39.0, 48.0, 52.0, NAN, NAN],
            "is_stay": [1.0, 0.0, 1.0, 0.0, NAN, 1.0, NAN, NAN],
            "reward_history_incomplete": [1, 0, 0, 0, 0, 1, 0, 0],
        }
    )


def _onsets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": ["non_local"] * 5 + ["SWR"] * 2 + ["MUA"] * 4,
            "stop_id": [1, 1, 2, 3, 6, 2, 3, 3, 3, 3, 5],
            "t_since_stop_onset": [0.5, 1.5, 3.0, 5.0, 0.2, 1.0, 6.0]
            + [1.0, 2.0, 7.0, 0.1],
        }
    )


def test_stop_table_values_name_each_statistic():
    values = {v.key: v for v in stop_table_values(_stops(), _onsets(), "x")}
    read = {key: value.value for key, value in values.items()}
    # analysed immobility of the well-anchored stops, rewarded: 3, 1 and 8;
    # omitted: 1, 4 and 0.25 (the rewarded in-transit stop is not one of them)
    assert read["stop_duration_mean_rewarded"] == 4.0
    assert read["stop_duration_mean_unrewarded"] == 1.75
    assert read["stop_duration_mean_ratio"] == pytest.approx(16 / 7)
    assert read["stop_duration_median_rewarded"] == 3.0
    assert read["stop_duration_median_unrewarded"] == 1.0
    assert read["stop_duration_median_ratio"] == 3.0
    # spans of the six well-anchored stops: 3, 5, 8, 1, 4.5 and 0.5
    assert read["stop_survival_two_s"] == pytest.approx(4 / 6)
    assert read["stop_survival_four_s"] == 0.5
    assert "2 s after they began" in values["stop_survival_two_s"].description
    assert "4 s after they began" in values["stop_survival_four_s"].description
    assert read["stops_in_transit"] == 2
    assert read["stops_crossing_outcome"] == 4  # stops 1, 3, 4 and 5
    assert read["stops_crossing_outcome_fraction"] == 0.5
    assert read["stops_outcome_shifted"] == 2  # stops 1 and 4
    assert read["stops_outcome_shifted_fraction"] == 0.25
    assert read["stops_spanning_arrival"] == 2  # stops 2 and 5
    assert read["stops_spanning_arrival_crossing"] == 1  # stop 5
    # onsets earlier than 4 s: 4 of 5 non-local, 1 of 2 SWR, 3 of 4 MUA
    assert read["onsets_within_tmax_fraction_non_local"] == 0.8
    assert read["onsets_within_tmax_fraction_swr"] == 0.5
    assert read["onsets_within_tmax_fraction_mua"] == 0.75
    assert read["stops_reward_history_incomplete_fraction"] == 0.25  # stops 1 and 6
    assert read["onsets_reward_history_incomplete_fraction_non_local"] == 0.6
    assert read["onset_median_s_non_local"] == 1.5
    assert {v.function for v in values.values()} == {"stop_table_values"}


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


def test_cohort_audit_values_read_the_non_local_rows_of_each_audit(tmp_path):
    columns = [
        "population",
        "model",
        "n_in",
        "n_drop_own_predictor_missing",
        "n_drop_common_cohort_other",
        "n_modeled",
        "n_excluded_session",
    ]
    per_bin = pd.DataFrame(
        [
            ("swr", "M0_const", 1, 2, 3, 4, 5),
            ("non_local", "M0_const", 900, 11, 12, 800, 13),
            ("non_local", "M5_time", 900, 21, 22, 700, 23),
        ],
        columns=columns,
    )
    per_stop = pd.DataFrame(
        [
            ("non_local", "R1", 500, 31, 32, 300, 33),
            ("non_local", "R0", 500, 41, 42, 400, 43),
            ("swr", "R0", 6, 7, 8, 9, 10),
        ],
        columns=columns,
    )
    inputs = BehaviorInputs(
        **{field.name: tmp_path / field.name for field in fields(BehaviorInputs)}
    )
    inputs = replace(
        inputs,
        per_bin_audit=tmp_path / "per_bin_audit.csv",
        per_stop_audit=tmp_path / "per_stop_audit.csv",
    )
    per_bin.to_csv(inputs.per_bin_audit, index=False)
    per_stop.to_csv(inputs.per_stop_audit, index=False)

    values = {v.key: v for v in cohort_audit_values(inputs)}
    read = {key: value.value for key, value in values.items()}
    assert read == {
        "rate_audit_stops_in": 900,
        "rate_audit_stops_own_drop": 11,
        "rate_audit_stops_common_drop": 12,
        "rate_audit_stops_modeled": 800,
        "rate_audit_stops_modeled_session_excluded": 13,
        # what entered and what was dropped are read from the R0 row
        "per_stop_audit_stops_in": 500,
        "per_stop_audit_stops_own_drop": 41,
        "per_stop_audit_stops_common_drop": 42,
        "per_stop_audit_stops_modeled": 400,
        "per_stop_audit_stops_modeled_session_excluded": 43,
        "per_stop_audit_stops_full": 300,
        "per_stop_audit_stops_full_session_excluded": 33,
    }
    assert "per_bin_audit.csv" in values["rate_audit_stops_in"].source
    assert "per_stop_audit.csv" in values["per_stop_audit_stops_in"].source
