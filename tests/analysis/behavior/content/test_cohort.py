"""The content ladder's fit request, cohort selection and cohort-audit record."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.cohort_audit import HAS_QVALS_ROW
from continuum_swr_replay.analysis.behavior.content.cohort import (
    content_cohort_record,
    contrast_fit_cohort,
    prepare_canonical_fit_request,
    select_content_cohort,
    zero_response_event_ids,
)
from continuum_swr_replay.analysis.behavior.model_specs import (
    MODEL_HIERARCHY,
    MODEL_PATCH,
    ModelSpec,
    default_content_model_names,
)
from tests.support._content_tables import (
    canonical_arm_table,
    canonical_patch_table,
    default_ladder_arm_table,
    default_ladder_patch_table,
)

_ARM_LADDER = {"X": ModelSpec("X", ("is_current_arm",), "current-arm bias", None)}
_PATCH_LADDER = {"X": ModelSpec("X", ("is_current_patch",), "current patch", None)}

# A qvals-sourced predictor and the row-gated column that marks the qvals gap.
_QVALS_PREDICTOR = "reward_belief_mean_candidate_arm_preoutcome"
_QVALS_ROW_GATED = "task_reward_probability_candidate_arm"


# ── The fit request ──


def _prepare_arm(table: pd.DataFrame, **kwargs):
    return prepare_canonical_fit_request(
        table,
        hierarchy=_ARM_LADDER,
        model_names=["X"],
        hierarchy_kind="arm",
        table_label="synthetic-arm-table",
        **kwargs,
    )


def test_default_response_is_the_per_event_arm_fraction() -> None:
    _prepared, _models, _preds, response_col, event_col = _prepare_arm(
        canonical_arm_table()
    )
    assert response_col == "arm_fraction"
    assert event_col == "event_id"


def test_hard_count_response_resolves_to_the_arm_counts() -> None:
    _prepared, _models, _preds, response_col, _event = _prepare_arm(
        canonical_arm_table(), response="hard_count"
    )
    assert response_col == "arm_count_hard"


def test_patch_hierarchy_resolves_to_the_patch_fraction_and_counts() -> None:
    for response, expected in (
        ("fraction", "patch_fraction"),
        ("hard_count", "patch_count_hard"),
    ):
        _prepared, _models, _preds, response_col, _event = (
            prepare_canonical_fit_request(
                canonical_patch_table(),
                hierarchy=_PATCH_LADDER,
                model_names=["X"],
                hierarchy_kind="patch",
                table_label="synthetic-patch-table",
                response=response,
            )
        )
        assert response_col == expected


def test_fraction_that_does_not_sum_to_one_is_refused() -> None:
    table = canonical_arm_table(fraction_scale=0.9)
    with pytest.raises(ValueError, match="does not sum to 1.0"):
        _prepare_arm(table)
    # The same table is fine on the counts, which are untouched.
    assert _prepare_arm(table, response="hard_count")[3] == "arm_count_hard"


def test_fraction_off_by_more_than_the_tolerance_is_refused() -> None:
    table = canonical_arm_table()
    table.loc[0, "arm_fraction"] += 1e-8
    with pytest.raises(ValueError, match="does not sum to 1.0"):
        _prepare_arm(table)


def test_fraction_within_the_tolerance_is_accepted() -> None:
    table = canonical_arm_table()
    table.loc[0, "arm_fraction"] += 1e-12
    assert _prepare_arm(table)[3] == "arm_fraction"


def test_missing_fraction_column_names_the_column_and_the_way_out() -> None:
    table = canonical_arm_table().drop(columns=["arm_fraction"])
    with pytest.raises(ValueError, match="arm_fraction.*--response hard_count"):
        _prepare_arm(table)


def test_event_with_no_real_arm_bins_is_allowed_as_all_missing() -> None:
    """An all-junction event carries no fraction anywhere; the cohort drops it."""
    assert _prepare_arm(canonical_arm_table(undecoded_event="e2"))[3] == "arm_fraction"


def test_fraction_defined_on_only_some_candidates_is_refused() -> None:
    with pytest.raises(ValueError, match="only some candidates"):
        _prepare_arm(canonical_arm_table(partial_event="e1"))


def test_complete_default_ladder_passes_preflight_without_dropping_rungs():
    """Every requested rung survives preflight when its real columns are present."""
    requested = default_content_model_names(MODEL_HIERARCHY)

    prepared, model_names, predictors, response_col, event_col = (
        prepare_canonical_fit_request(
            default_ladder_arm_table(),
            hierarchy=MODEL_HIERARCHY,
            model_names=requested,
            hierarchy_kind="arm",
            table_label="synthetic event_arm_table.pkl",
        )
    )

    assert model_names == requested
    assert "is_candidate_arm_never_visited" in predictors
    assert "is_candidate_arm_0" in prepared
    assert response_col == "arm_fraction"
    assert event_col == "event_id"


def test_arm_current_indicator_must_match_occupancy_well_arm():
    frame = default_ladder_arm_table()
    event_mask = frame["event_id"].eq(0)
    frame.loc[event_mask, "is_current_arm"] = (
        frame.loc[event_mask, "candidate_arm"].eq(1).astype(float)
    )

    with pytest.raises(ValueError, match="must match occupancy_well_arm"):
        prepare_canonical_fit_request(
            frame,
            hierarchy=MODEL_HIERARCHY,
            model_names=("M0",),
            hierarchy_kind="arm",
            table_label="corrupt arm table",
        )


def test_unknown_location_events_are_accepted_with_null_at_well():
    """``unknown`` carries ``at_well = NA`` by construction, not by corruption.

    Events before the first completed trip have no observed location, so
    ``at_well`` is null while transit/mixed windows are explicitly False.
    A canonical table is entitled to contain them; rejecting the null would
    abort a whole session for a state this validator lists as valid.
    """
    frame = default_ladder_arm_table()
    event_mask = frame["event_id"].eq(0)
    frame["at_well"] = frame["at_well"].astype("boolean")
    frame.loc[event_mask, "occupancy_state"] = "unknown"
    frame.loc[event_mask, "at_well"] = pd.NA
    for column in ("occupancy_well_id", "occupancy_well_arm", "is_current_arm"):
        frame.loc[event_mask, column] = np.nan

    request = prepare_canonical_fit_request(
        frame,
        hierarchy=MODEL_HIERARCHY,
        model_names=("M0",),
        hierarchy_kind="arm",
        table_label="table with an unknown-location event",
    )

    assert request is not None


@pytest.mark.parametrize(
    ("state", "at_well_value"),
    [("at_well", pd.NA), ("in_transit", pd.NA), ("unknown", False)],
)
def test_at_well_null_pattern_must_track_location_state(state, at_well_value):
    """Null-ness is only permitted for ``unknown``, and required there."""
    frame = default_ladder_arm_table()
    event_mask = frame["event_id"].eq(0)
    frame["at_well"] = frame["at_well"].astype("boolean")
    frame.loc[event_mask, "occupancy_state"] = state
    frame.loc[event_mask, "at_well"] = at_well_value
    if state != "at_well":
        for column in (
            "occupancy_well_id",
            "occupancy_well_arm",
            "is_current_arm",
        ):
            frame.loc[event_mask, column] = np.nan

    with pytest.raises(ValueError, match="inconsistent with location_state"):
        prepare_canonical_fit_request(
            frame,
            hierarchy=MODEL_HIERARCHY,
            model_names=("M0",),
            hierarchy_kind="arm",
            table_label="corrupt arm table",
        )


def test_patch_current_indicator_must_match_mapped_occupancy_well_arm():
    frame = default_ladder_patch_table()
    event_mask = frame["event_id"].eq(0)
    frame.loc[event_mask, "is_current_patch"] = (
        frame.loc[event_mask, "patch"].eq(2).astype(float)
    )

    with pytest.raises(ValueError, match="mapped occupancy_well_arm"):
        prepare_canonical_fit_request(
            frame,
            hierarchy=MODEL_PATCH,
            model_names=("P0",),
            hierarchy_kind="patch",
            table_label="corrupt patch table",
        )


def test_location_reference_must_be_constant_across_candidate_rows():
    """Row-varying references cannot make a corrupt one-hot look valid."""
    frame = default_ladder_arm_table()
    corrupt_row = frame["event_id"].eq(0) & frame["candidate_arm"].eq(1)
    # This row still has is_current=0 and arm=1, so the later row-wise
    # indicator check would pass (1 != 2) unless the event-level reference is
    # first required to be constant.
    frame.loc[corrupt_row, "occupancy_well_arm"] = 2.0

    with pytest.raises(ValueError, match="event context.*changes"):
        prepare_canonical_fit_request(
            frame,
            hierarchy=MODEL_HIERARCHY,
            model_names=("M0",),
            hierarchy_kind="arm",
            table_label="row-varying location table",
        )


@pytest.mark.parametrize("bad_count", [np.nan, np.inf, -1.0, 0.5])
def test_response_counts_must_be_finite_nonnegative_integers(bad_count):
    frame = default_ladder_arm_table()
    frame.loc[0, "arm_count_hard"] = bad_count

    with pytest.raises(ValueError, match="finite nonnegative integer counts"):
        prepare_canonical_fit_request(
            frame,
            hierarchy=MODEL_HIERARCHY,
            model_names=("M0",),
            hierarchy_kind="arm",
            table_label="corrupt response table",
        )


def test_each_session_event_requires_the_exact_candidate_set():
    frame = default_ladder_arm_table()
    frame.loc[frame.index[0], "candidate_arm"] = 1

    with pytest.raises(
        ValueError, match="exactly the expected candidate_arm candidates"
    ):
        prepare_canonical_fit_request(
            frame,
            hierarchy=MODEL_HIERARCHY,
            model_names=("M0",),
            hierarchy_kind="arm",
            table_label="corrupt candidate table",
        )


# ── The cohort ──


def _select(table: pd.DataFrame, **kwargs):
    options = {
        "hierarchy_kind": "arm",
        "union_predictors": ["is_current_arm"],
        "response_col": "arm_fraction",
        "response": "fraction",
        "table_label": "synthetic-arm-table",
    }
    options.update(kwargs)
    return select_content_cohort(table, **options)


def _events(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["event_id"].unique())


def test_swr_subset_keeps_the_events_on_its_side() -> None:
    table = canonical_arm_table(n_events=4)
    table["is_swr"] = table["event_id"].isin(["e0", "e1"])

    assert _events(_select(table, swr_subset="T").long_df) == ["e0", "e1"]
    assert _events(_select(table, swr_subset="F").long_df) == ["e2", "e3"]
    assert _events(_select(table).long_df) == ["e0", "e1", "e2", "e3"]


def test_swr_subset_needs_the_is_swr_column() -> None:
    with pytest.raises(ValueError, match="is_swr.*synthetic-arm-table"):
        _select(canonical_arm_table(), swr_subset="T")


def test_off_well_events_leave_before_the_freeze_and_are_not_counted_in() -> None:
    """At-well is a cohort, not missing data: the audit's n_in never sees them."""
    table = canonical_arm_table(n_events=4)
    table.loc[table["event_id"].eq("e1"), "occupancy_state"] = "in_transit"

    selection = _select(table)

    assert _events(selection.long_df_prefilter) == ["e0", "e2", "e3"]
    assert _events(selection.long_df) == ["e0", "e2", "e3"]
    assert selection.n_events_before_complete_case == 3


def test_a_table_without_occupancy_state_cannot_state_a_cohort() -> None:
    with pytest.raises(ValueError, match="occupancy_state"):
        _select(canonical_arm_table().drop(columns="occupancy_state"))


def test_leaf_only_keeps_six_rows_and_rescales_a_fraction() -> None:
    """Each event put 6/8 and 2/8 of its mass on two leaves: unchanged by the rescale."""
    table = canonical_arm_table(n_events=2)
    table.loc[table["candidate_arm"].eq(6), "arm_fraction"] = 0.2
    table["arm_fraction"] /= table.groupby("event_id")["arm_fraction"].transform("sum")

    fraction = _select(table, leaf_only=True)
    count = _select(
        table, leaf_only=True, response_col="arm_count_hard", response="hard_count"
    )

    assert fraction.response_renormalized is True
    assert fraction.long_df.groupby("event_id").size().eq(6).all()
    sums = fraction.long_df.groupby("event_id")["arm_fraction"].sum()
    assert np.allclose(sums, 1.0)
    e0 = fraction.long_df[fraction.long_df["event_id"].eq("e0")]
    assert e0.set_index("candidate_arm")["arm_fraction"][0] == pytest.approx(0.75)
    assert count.response_renormalized is False
    assert count.long_df.groupby("event_id")["arm_count_hard"].sum().eq(8.0).all()


def test_an_event_missing_or_nonfinite_in_any_rungs_predictor_leaves_every_rung() -> (
    None
):
    table = canonical_arm_table(n_events=4)
    table["other_rung_term"] = 1.0
    table.loc[
        table["event_id"].eq("e1") & table["candidate_arm"].eq(3), "other_rung_term"
    ] = np.nan
    table.loc[
        table["event_id"].eq("e2") & table["candidate_arm"].eq(0), "other_rung_term"
    ] = np.inf

    selection = _select(table, union_predictors=["is_current_arm", "other_rung_term"])

    assert _events(selection.long_df) == ["e0", "e3"]
    # The audit is read off the frame before the freeze, which keeps them.
    assert _events(selection.long_df_prefilter) == ["e0", "e1", "e2", "e3"]
    assert selection.n_events_before_complete_case == 4
    assert selection.long_df.index.equals(pd.RangeIndex(18))


def test_an_event_with_no_response_is_frozen_out() -> None:
    table = canonical_arm_table(n_events=3)
    table.loc[table["event_id"].eq("e1"), "arm_count_hard"] = 0.0

    selection = _select(table, response_col="arm_count_hard", response="hard_count")

    assert _events(selection.long_df) == ["e0", "e2"]


def test_removing_every_event_is_refused() -> None:
    table = canonical_arm_table(n_events=2)
    table.loc[table["candidate_arm"].eq(0), "is_current_arm"] = np.nan
    with pytest.raises(RuntimeError, match="removed all 2 events"):
        _select(table)

    table = canonical_arm_table(n_events=2)
    table["arm_count_hard"] = 0.0
    with pytest.raises(RuntimeError, match="every complete-case event had zero"):
        _select(table, response_col="arm_count_hard", response="hard_count")


def test_a_qvals_predictor_brings_the_qvals_row_flag() -> None:
    """The flag is read off the row-gated column, NaN on a stop with no qvals row."""
    table = canonical_arm_table(n_events=3)
    table[_QVALS_ROW_GATED] = 0.5
    table.loc[table["event_id"].eq("e2"), _QVALS_ROW_GATED] = np.nan
    table[_QVALS_PREDICTOR] = 0.1

    with_value = _select(table, union_predictors=["is_current_arm", _QVALS_PREDICTOR])
    without = _select(table)

    flags = with_value.long_df_prefilter.groupby("event_id")[HAS_QVALS_ROW].first()
    assert flags.to_dict() == {"e0": True, "e1": True, "e2": False}
    assert HAS_QVALS_ROW not in without.long_df_prefilter


def test_zero_response_events_cover_undefined_and_empty_allocations() -> None:
    """Both mechanisms are the same drop: the event has no allocation to fit."""
    frame = pd.DataFrame(
        {
            "event_id": ["a", "a", "b", "b", "c", "c"],
            "arm_fraction": [0.5, 0.5, np.nan, np.nan, 0.0, 0.0],
        }
    )

    assert zero_response_event_ids(
        frame, event_col="event_id", response_col="arm_fraction"
    ) == {"b", "c"}


# ── The cohort-audit record ──


def _audit_frame() -> pd.DataFrame:
    """Eight events in two sessions, each event's fate set by hand.

    Session one (``02_r1``): e0 is fitted; e1 lacks this rung's qvals predictor
    on a stop with no qvals row; e2 lacks it on a stop that has one; e3 lacks
    only another rung's predictor; e4 carries an infinite value; e5 has no
    response. Session two (``04_r2``): e6 and e7 are complete, but the session
    is not fitted.
    """
    table = canonical_arm_table(n_events=8)
    table["epoch"] = np.where(table["event_id"].isin(["e6", "e7"]), "04_r2", "02_r1")
    table[_QVALS_PREDICTOR] = 0.1
    table["other_rung_term"] = 1.0
    table[HAS_QVALS_ROW] = ~table["event_id"].eq("e1")
    table.loc[table["event_id"].isin(["e1", "e2"]), _QVALS_PREDICTOR] = np.nan
    table.loc[table["event_id"].eq("e3"), "other_rung_term"] = np.nan
    table.loc[
        table["event_id"].eq("e4") & table["candidate_arm"].eq(2), _QVALS_PREDICTOR
    ] = np.inf
    table.loc[table["event_id"].eq("e5"), "arm_fraction"] = np.nan
    return table


def _record(prefilter: pd.DataFrame, fits: pd.DataFrame, **kwargs) -> dict:
    options = {
        "long_df_prefilter": prefilter,
        "predictors": [_QVALS_PREDICTOR],
        "union_predictors": [_QVALS_PREDICTOR, "other_rung_term"],
        "response_col": "arm_fraction",
        "event_col": "event_id",
        "n_events_in": int(prefilter["event_id"].nunique()),
        "population": "all",
        "flag_counts": {
            "n_sessions_flagged_nonconverged": 0,
            "n_sessions_flagged_separated": 1,
        },
    }
    options.update(kwargs)
    return content_cohort_record("X", fits, **options)


_ONE_FIT = pd.DataFrame(
    {"animal": ["chimi"], "session_date": ["20200212"], "epoch": ["02_r1"]}
).assign(n_events=1)


def test_every_event_is_booked_once_under_the_first_reason_it_left() -> None:
    record = _record(_audit_frame(), _ONE_FIT)

    assert record["n_in"] == 8
    assert record["n_drop_qvals_gap"] == 1  # e1
    assert record["n_drop_own_predictor_missing"] == 1  # e2
    assert record["n_drop_common_cohort_other"] == 1  # e3
    assert record["n_drop_nonfinite"] == 1  # e4
    assert record["n_drop_zero_response"] == 1  # e5
    assert record["n_excluded_session"] == 2  # e6, e7
    assert record["n_modeled"] == 1  # e0
    assert record["n_sessions_in"] == 2
    assert record["n_sessions_modeled"] == 1
    assert record["n_animals_modeled"] == 1
    assert record["cohort_mode"] == "common"
    assert record["grain"] == "event"
    assert record["n_sessions_flagged_separated"] == 1


def test_another_rungs_qvals_predictor_is_not_this_rungs_qvals_gap() -> None:
    """A rung without the value covariate books e1 and e2 as common-cohort drops."""
    record = _record(_audit_frame(), _ONE_FIT, predictors=["other_rung_term"])

    assert record["n_drop_qvals_gap"] == 0
    # e1, e2 (the other rung's NaN); e3 is this rung's own.
    assert record["n_drop_common_cohort_other"] == 2
    assert record["n_drop_own_predictor_missing"] == 1
    assert record["n_drop_nonfinite"] == 1
    assert record["n_excluded_session"] == 2


def test_an_unfitted_rung_books_every_survivor_as_session_excluded() -> None:
    record = _record(_audit_frame(), pd.DataFrame())

    assert record["n_modeled"] == 0
    assert record["n_excluded_session"] == 3
    assert record["n_sessions_modeled"] == 0
    assert record["n_animals_modeled"] == 0


def test_a_session_the_estimator_refused_is_named_in_the_notes() -> None:
    skips = [
        {"session": "chimi/20200212/04_r2", "reason": "dm_design: rank 1 of 2"},
        {"session": "chimi/20200212/06_r3", "reason": "too_few"},
    ]

    record = _record(_audit_frame(), _ONE_FIT, estimator_skips=skips)

    assert "no fit: dm_design, too_few" in record["drop_notes"]
    assert "estimator skipped chimi/20200212/04_r2" in record["drop_notes"]
    assert "06_r3" not in record["drop_notes"]
    assert record["n_excluded_session"] == 2


def test_a_response_that_sums_below_zero_cannot_be_booked() -> None:
    """The drop buckets book a zero total; a negative one would go unaccounted."""
    prefilter = _audit_frame()
    prefilter.loc[prefilter["event_id"].eq("e0"), "arm_fraction"] = -0.1

    with pytest.raises(RuntimeError, match="does not account"):
        _record(prefilter, _ONE_FIT)


# ── The is_swr-contrast cohort ──


def _contrast_cohort() -> pd.DataFrame:
    """Two events x nine arms. ``s`` is an SWR event with 2/8 of its mass on a
    trunk; ``n`` is not, with 4/8 on a trunk."""
    counts = {
        "s": {0: 3.0, 1: 3.0, 6: 2.0},
        "n": {2: 4.0, 7: 4.0},
    }
    rows = [
        {
            "event_id": event,
            "is_swr": event == "s",
            "candidate_arm": arm,
            "arm_fraction": counts[event].get(arm, 0.0) / 8.0,
            "arm_count_hard": counts[event].get(arm, 0.0),
        }
        for event in ("s", "n")
        for arm in range(9)
    ]
    return pd.DataFrame(rows, index=range(100, 118))


def test_a_contrast_population_is_its_events_by_is_swr() -> None:
    cohort = _contrast_cohort()
    for population, events in (("all", {"s", "n"}), ("SWR", {"s"}), ("nonSWR", {"n"})):
        selected = contrast_fit_cohort(
            cohort,
            "C1_current",
            ["is_current_arm"],
            population,
            response_col="arm_fraction",
            response="fraction",
            n_categories=9,
        )
        assert set(selected.frame["event_id"]) == events
        assert len(selected.frame) == 9 * len(events)
        # A nine-arm rung keeps the table's own fractions.
        assert selected.frame["arm_fraction"].sum() == pytest.approx(len(events))
        assert selected.predictors == ["is_current_arm"]
        assert selected.n_categories == 9
        assert selected.response_renormalized is False


def test_an_unknown_contrast_population_is_refused() -> None:
    with pytest.raises(ValueError, match="population"):
        contrast_fit_cohort(
            _contrast_cohort(),
            "C1_current",
            [],
            "swr",
            response_col="arm_fraction",
            response="fraction",
            n_categories=9,
        )


def test_a_leaf_only_contrast_rung_is_fit_on_the_leaves_at_unit_mass() -> None:
    """``C4_armvalue`` is leaf-only: six candidates, the trunk arm fixed effects
    gone, and each event's fraction rescaled over its leaves."""
    selected = contrast_fit_cohort(
        _contrast_cohort(),
        "C4_armvalue",
        ["is_candidate_arm_0", "is_candidate_arm_6", "is_current_arm"],
        "all",
        response_col="arm_fraction",
        response="fraction",
        n_categories=9,
    )

    assert sorted(selected.frame["candidate_arm"].unique()) == [0, 1, 2, 3, 4, 5]
    assert selected.n_categories == 6
    assert selected.predictors == ["is_candidate_arm_0", "is_current_arm"]
    assert selected.response_renormalized is True
    share = selected.frame.set_index(["event_id", "candidate_arm"])["arm_fraction"]
    # s: 3/8 and 3/8 of 6/8 on the leaves -> 1/2 each; n: 4/8 of 4/8 -> 1.
    assert share[("s", 0)] == pytest.approx(0.5)
    assert share[("s", 1)] == pytest.approx(0.5)
    assert share[("n", 2)] == pytest.approx(1.0)


def test_a_leaf_only_contrast_count_rung_keeps_its_counts() -> None:
    selected = contrast_fit_cohort(
        _contrast_cohort(),
        "C4_armvalue",
        ["is_current_arm"],
        "SWR",
        response_col="arm_count_hard",
        response="hard_count",
        n_categories=9,
    )

    assert selected.response_renormalized is False
    assert selected.frame["arm_count_hard"].tolist() == [3.0, 3.0, 0.0, 0.0, 0.0, 0.0]
