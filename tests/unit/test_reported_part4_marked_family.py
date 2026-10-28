"""Part 4's marked-family values from a synthetic summary table."""

import pandas as pd

from continuum_swr_replay.reported.part4 import _value_factory, marked_family_values


def test_marked_family_values_quote_peak_sign_gain_and_leakage():
    rows = []
    for event_class in ("non_local_SWR", "non_local_no_SWR"):
        for statistic, value in (
            ("peak_value", -0.03),
            ("peak_lag_ms", 62.0),
            ("animals_same_sign", 5.0),
            ("frac_pos", 0.35),
            ("n_animals", 5.0),
        ):
            rows.append(
                (
                    "replay_content",
                    event_class,
                    "current_arm_fraction",
                    statistic,
                    value,
                )
            )
        rows.append(
            (
                "event_strength",
                event_class,
                "ripple_power_z & mua_rate_z",
                "mark_correlation",
                0.79,
            )
        )
    rows.append(("ripple_power_pooled", "pooled", "ripple_power_z", "peak_value", 0.01))
    # A marginal model is not quoted.
    rows.append(
        ("marginal_mua_rate_z", "non_local_SWR", "mua_rate_z", "peak_value", 0.02)
    )
    summary = pd.DataFrame(
        rows, columns=["model", "event_class", "target", "statistic", "value"]
    )
    make = _value_factory("synthetic", ("20",), ("paper",))

    values = {v.key: v.value for v in marked_family_values(summary, make)}

    assert values["marked_content_current_arm_swr_replay_peak"] == -0.03
    assert values["marked_content_current_arm_non_swr_replay_animals_same_sign"] == 5
    assert values["marked_content_current_arm_swr_replay_heldout_gain_fraction"] == 0.35
    assert values["marked_ripple_mua_correlation_swr_replay"] == 0.79
    assert values["marked_ripplepooled_ripple_power_pooled_peak"] == 0.01
    assert not any(key.startswith("marked_marginal") for key in values)
    assert all(not any(ch.isdigit() for ch in key) for key in values)
