"""The events summary carries a planted out-of-visit event, quiet window and
other-detector event through each of its steps: no step after the selection
reads them, and every output not made over the subset, from the current event
table or from the current well visits makes its values pending.

Every builder is a stand-in recording what it is given and returning one value
named after it; the readers of the stored outputs are stand-ins too, or tiny
files where the summary reads a file itself. The selections themselves
(``select_visit_context``, ``select_events_in``, ``in_analyzed_subset``) and the
staleness checks are the real ones."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detector_comparison import nonlocal_comparison
from continuum_swr_replay.analysis.event_definition import cohort_selection
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_record,
)
from continuum_swr_replay.paths import (
    BRAIN_STATE_REFERENCE_DIR,
    CONTENT_ALONG_ARMS_DIR,
    DECODER_QUALITY_DIR,
    EVENT_GAMMA_DIR,
    EVENT_LIKELIHOOD_ADVANTAGE_DIR,
    EVENT_PARTICIPATION_DIR,
    EVENT_PROPERTY_BOOTSTRAP_DIR,
    EVENT_START_REST_THETA_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    NL_BOUTS_BY_SWR_STEM,
    NONLOCAL_BY_RIPPLE_POWER_DIR,
    RIPPLE_CLASS_STRUCTURE_DIR,
)
from continuum_swr_replay.reported import events as builder
from continuum_swr_replay.reported.values import Tags, pend, value_factory
from tests.reported.test_reported_events_values import _events
from tests.support._session_cache_fixtures import write_session_cache_entries

PLANTED = 999
"""The event that starts outside every visit (on the track, at 50 s)."""
EPOCHS = ["ad1_e1_HPC", "ad1_e2_HPC", "bd1_e1_HPC", "cd1_e1_HPC"]
VISITS = {epoch: (np.array([0.0]), np.array([20.0])) for epoch in EPOCHS}
"""Each epoch's one well visit, 0 to 20 s: the builders' events (10 to 11 s)
start inside it."""
EXPOSURE_DIR = nonlocal_comparison.EXPOSURE_IN_VISITS_DIR

#: Each builder the summary calls, with the argument whose rows are checked.
EVENT_BUILDERS = (
    "event_cohort_values",
    "boundary_offset_values",
    "state_composition_values",
    "property_values",
    "start_distance_values",
    "rest_theta_values",
    "brain_state_values",
    "gamma_z_values",
    "participation_values",
    "likelihood_advantage_values",
    "tetrode_holdout_values",
    "content_occupancy_values",
)
OTHER_BUILDERS = (
    "visit_context_values",
    "property_interval_values",
    "gamma_db_values",
    "other_detector_values",
    "detector_visit_values",
    "spike_evidence_values",
    "spike_evidence_visit_values",
    "heldout_by_swr_values",
    "heldout_bout_values",
    "decoding_error_values",
    "calibration_values",
    "state_fraction_values",
    "class_distribution_values",
    "nonlocal_power_values",
)


def _table() -> pd.DataFrame:
    """The builders' thirteen events, inside a visit, and the planted one."""
    inside = _events().assign(
        event_id=pd.Series(range(13), dtype="uint64"), visit_context="in_visit"
    )
    planted = inside.iloc[[12]].assign(
        event_id=np.uint64(PLANTED),
        visit_context="track_center",
        event_start_s=50.0,
        event_end_s=51.0,
    )
    return pd.concat([inside, planted], ignore_index=True)


def _metrics() -> pd.DataFrame:
    """The spike-evidence run's windows: an event inside a visit, the planted
    one, and a quiet window inside and one outside the visits."""
    return pd.DataFrame(
        {
            "epoch_id": ["ad1_e1_HPC", "cd1_e1_HPC", "ad1_e1_HPC", "ad1_e1_HPC"],
            "group": ["nonSWR-close", "SWR-remote", "baseline", "baseline"],
            "start_time": [10.0, 50.0, 5.0, 30.0],
            "event_id": pd.array([0, PLANTED, pd.NA, pd.NA], dtype="UInt64"),
        }
    )


def _per_event_run() -> pd.DataFrame:
    """A per-event run's rows: each event and its control, the planted one too."""
    ids = [*range(13), PLANTED]
    return pd.DataFrame(
        {
            "event_id": np.repeat(ids, 2).astype("uint64"),
            "kind": ["event", "control"] * len(ids),
        }
    )


class _Run:
    """The inputs laid out under ``root``, the spies, and the manifests the
    summary reads (``changes`` alters one folder's)."""

    def __init__(self, root: Path, monkeypatch, changes=None):
        self.root = root
        self.seen: dict[str, tuple] = {}
        entries = write_session_cache_entries(EPOCHS)
        assert entries
        fresh = {
            "subset": ANALYZED_SUBSET,
            "session_cache": session_cache_record(EPOCHS),
            "event_table": {"sha256": "current"},
            "event_participation": {"sha256": "participation"},
        }
        self.manifests = {}
        for folder in (
            EVENT_START_REST_THETA_DIR,
            EVENT_PROPERTY_BOOTSTRAP_DIR,
            BRAIN_STATE_REFERENCE_DIR,
            EVENT_GAMMA_DIR,
            RIPPLE_CLASS_STRUCTURE_DIR,
            NONLOCAL_BY_RIPPLE_POWER_DIR,
            EXPOSURE_DIR,
            CONTENT_ALONG_ARMS_DIR,
            EVENT_PARTICIPATION_DIR,
            EVENT_LIKELIHOOD_ADVANTAGE_DIR,
            EVENT_TETRODE_HOLDOUT_DIR,
            "state_evidence",
        ):
            (root / folder).mkdir(parents=True, exist_ok=True)
            self.manifests[folder] = json.loads(json.dumps(fresh))
        for folder, (key, value) in (changes or {}).items():
            if value is None:
                self.manifests[folder].pop(key)
            else:
                self.manifests[folder][key] = value
        self._lay_out_files()
        self._patch(monkeypatch)

    def _lay_out_files(self) -> None:
        root = self.root
        pd.DataFrame({"a": [1]}).to_csv(
            root / EVENT_PROPERTY_BOOTSTRAP_DIR / "intervals.csv"
        )
        pd.DataFrame({"a": [1]}).to_csv(
            root / "state_evidence" / f"{NL_BOUTS_BY_SWR_STEM}_stats.csv"
        )
        provenance = {**self.manifests["state_evidence"], "epochs": EPOCHS, "bouts": []}
        (
            root / "state_evidence" / f"{NL_BOUTS_BY_SWR_STEM}.provenance.json"
        ).write_text(json.dumps(provenance))
        (root / DECODER_QUALITY_DIR).mkdir()
        for name in ("sessions", "calibration"):
            pd.DataFrame({"epoch_id": EPOCHS}).to_parquet(
                root / DECODER_QUALITY_DIR / f"{name}.parquet"
            )
        for name in ("distribution", "step_fits"):
            pd.DataFrame({"a": [1]}).to_csv(
                root / RIPPLE_CLASS_STRUCTURE_DIR / f"{name}.csv", index=False
            )
        pd.DataFrame({"epoch_id": EPOCHS}).to_parquet(
            root / NONLOCAL_BY_RIPPLE_POWER_DIR / "counts.parquet"
        )
        pd.DataFrame({"epoch_id": EPOCHS}).to_parquet(
            root / EXPOSURE_DIR / "exposure.parquet"
        )

    def inputs(self) -> builder.EventsInputs:
        root = self.root
        return builder.EventsInputs(
            event_table=root / "event_table.parquet",
            event_features_run=root / "run",
            spike_evidence_dir=root / "spike_evidence",
            heldout_by_swr=root
            / "state_evidence"
            / f"{NL_BOUTS_BY_SWR_STEM}_stats.csv",
            content_along_arms=root / CONTENT_ALONG_ARMS_DIR,
            brain_state_reference=root
            / BRAIN_STATE_REFERENCE_DIR
            / "reference.parquet",
            event_gamma=root / EVENT_GAMMA_DIR,
            class_structure=root / RIPPLE_CLASS_STRUCTURE_DIR,
            nonlocal_by_ripple_power=root / NONLOCAL_BY_RIPPLE_POWER_DIR,
            event_start_rest_theta=root / EVENT_START_REST_THETA_DIR,
            property_bootstrap=root / EVENT_PROPERTY_BOOTSTRAP_DIR,
            decoder_quality=root / DECODER_QUALITY_DIR,
            event_participation=root / EVENT_PARTICIPATION_DIR,
            likelihood_advantage=root / EVENT_LIKELIHOOD_ADVANTAGE_DIR,
            tetrode_holdout=root / EVENT_TETRODE_HOLDOUT_DIR,
            exposure_in_visits=root / EXPOSURE_DIR,
        )

    def _spy(self, name: str):
        def spy(*args):
            self.seen[name] = args
            make = value_factory(spy, "s", Tags(("2a.1",)))
            return [make(name.removesuffix("_values"), 1, "count", "count", name)]

        return spy

    def _patch(self, monkeypatch) -> None:
        seen = self.seen
        for name in (*EVENT_BUILDERS, *OTHER_BUILDERS):
            monkeypatch.setattr(builder, name, self._spy(name))

        def class_steps(manifest, sha256, class_stale, steps, source, what):
            seen["class_step_values_with_staleness"] = (class_stale,)
            make = value_factory(class_steps, "s", Tags(("2a.1",)))
            return pend([make("class_step", 1, "count", "count", "d")], class_stale)

        def left_out(run, inventory, event_ids, visits, exposure, source):
            seen["left_out_values"] = (list(event_ids), visits, exposure)
            make = value_factory(left_out, "s", Tags(("2a.1",)))
            return [make("left_out", 1, "count", "count", "d")]

        def extras(events, folder, n_epochs):
            seen["with_start_and_rest_theta"] = (events,)
            return events, [Path(f"{epoch}.parquet") for epoch in EPOCHS]

        def brain_state(events, inputs, n_epochs):
            seen["with_brain_state"] = (events,)
            levels = pd.DataFrame({"epoch_id": EPOCHS})
            return events, levels, levels

        def along(events, folder, n_epochs, columns):
            seen["content_along_arms_bins"] = (events,)
            return events, []

        def comparison(run, visits):
            seen["read_comparison"] = (visits,)
            return nonlocal_comparison.Comparison(
                pd.DataFrame({"planted": [PLANTED]}),
                pd.DataFrame(),
                self.root / "inventory",
                self.root / "directional.csv",
                pd.DataFrame(),
            )

        checked = cohort_selection.check_visit_context_starts

        def check(table, visits):
            seen["check_visit_context_starts"] = (table, visits)
            checked(table, visits)

        monkeypatch.setattr(
            builder.Summary,
            "add",
            lambda summary, *values: seen.setdefault("added", []).extend(values),
        )
        for name, value in {
            "class_step_values_with_staleness": class_steps,
            "left_out_values": left_out,
            "with_start_and_rest_theta": extras,
            "with_brain_state": brain_state,
            "content_along_arms_bins": along,
            "check_visit_context_starts": check,
            "read_swr_table": lambda path: _table(),
            "read_manifest": lambda run: {
                "epochs": dict.fromkeys(EPOCHS, {"status": "ok"})
            },
            "well_visit_times": lambda identifier: VISITS[identifier],
            "analyzed_epoch_ids": lambda: EPOCHS,
            "file_record": lambda path, root, *args: {"sha256": "current"},
            "files_record": lambda *args: {"sha256": "files"},
            "files_sha256": lambda paths: "participation",
            "ok_table_paths": lambda run, name: [],
            "read_output_manifest": lambda folder: self.manifests[folder.name],
            "spike_evidence_metrics": lambda folder, n_epochs: _metrics(),
            "with_event_ids": lambda metrics, table: metrics,
            "read_complete_run": lambda folder, name, epochs: _per_event_run(),
        }.items():
            monkeypatch.setattr(builder, name, value)
        monkeypatch.setattr(builder.nonlocal_comparison, "read_comparison", comparison)
        monkeypatch.setattr(builder.tetrode_holdout, "window_scores", lambda t: t)

    def summarize(self) -> dict:
        builder.summarize(self.inputs(), "by")
        return {value.key: value for value in self.seen["added"]}


def test_no_step_after_the_selection_reads_the_planted_rows(
    processed_data_root, monkeypatch
):
    run = _Run(processed_data_root, monkeypatch)
    values = run.summarize()
    seen = run.seen

    # the label counts and the check against the visits see every event
    assert len(seen["visit_context_values"][0]) == 14
    assert len(seen["check_visit_context_starts"][0]) == 14
    for name in (*EVENT_BUILDERS, "with_start_and_rest_theta", "with_brain_state"):
        assert PLANTED not in set(seen[name][0]["event_id"]), name
    assert PLANTED not in set(seen["content_along_arms_bins"][0]["event_id"])
    # the per-event runs' tables are read over the selected events, controls too
    for name in (
        "participation_values",
        "likelihood_advantage_values",
        "tetrode_holdout_values",
    ):
        assert len(seen[name][0]) == 2 * 13, name
    # the other detectors' events are selected by the comparison, given the visits
    assert seen["read_comparison"][0] is not None
    assert set(seen["read_comparison"][0]) == set(EPOCHS)
    assert seen["other_detector_values"][0]["planted"].tolist() == [PLANTED]
    # the spike evidence: the event inside a visit and the quiet window inside
    evidence = seen["spike_evidence_values"][0]
    assert evidence["start_time"].tolist() == [10.0, 5.0]
    assert len(seen["spike_evidence_visit_values"][0]) == 4
    assert seen["spike_evidence_visit_values"][1].tolist() == [True, False, True, False]
    # what another inventory leaves out: of the selected events, with the visits
    event_ids, visits, exposure = seen["left_out_values"]
    assert PLANTED not in event_ids and len(event_ids) == 13
    assert set(visits) == set(EPOCHS)
    assert exposure["epoch_id"].tolist() == EPOCHS
    # the held-out bouts are counted when the split is over the subset
    assert "heldout_bout_values" in seen
    assert not [key for key, value in values.items() if value.pending]
    for key in ("participation", "tetrode_holdout"):
        assert "drawn from the whole epoch" in values[key].description
    assert "wherever that event starts" in values["left_out"].description
    assert "partner may start outside a visit" in values["other_detector"].description


def _other_cache() -> dict:
    return {"sha256": "0" * 64}


#: (folder, (manifest key, value or None to drop it)) -> the values made pending
PENDING_CASES = [
    (EVENT_START_REST_THETA_DIR, ("subset", None), {"rest_theta"}),
    (EVENT_START_REST_THETA_DIR, ("session_cache", _other_cache()), {"rest_theta"}),
    (
        EVENT_START_REST_THETA_DIR,
        ("event_table", {"sha256": "old"}),
        {"rest_theta", "start_distance"},
    ),
    (EVENT_PROPERTY_BOOTSTRAP_DIR, ("subset", None), {"property_interval"}),
    (
        EVENT_PROPERTY_BOOTSTRAP_DIR,
        ("event_table", {"sha256": "old"}),
        {"property_interval"},
    ),
    (BRAIN_STATE_REFERENCE_DIR, ("subset", None), {"brain_state", "gamma_z"}),
    (
        BRAIN_STATE_REFERENCE_DIR,
        ("session_cache", _other_cache()),
        {"brain_state", "gamma_z"},
    ),
    (EVENT_GAMMA_DIR, ("subset", None), {"gamma_z", "gamma_db"}),
    (EVENT_GAMMA_DIR, ("session_cache", None), {"gamma_z", "gamma_db"}),
    (EVENT_GAMMA_DIR, ("event_table", {"sha256": "old"}), {"gamma_z", "gamma_db"}),
    (
        RIPPLE_CLASS_STRUCTURE_DIR,
        ("subset", None),
        {"class_distribution", "class_step"},
    ),
    (
        RIPPLE_CLASS_STRUCTURE_DIR,
        ("event_table", {"sha256": "old"}),
        {"class_distribution", "class_step"},
    ),
    (NONLOCAL_BY_RIPPLE_POWER_DIR, ("subset", None), {"nonlocal_power"}),
    (
        NONLOCAL_BY_RIPPLE_POWER_DIR,
        ("session_cache", _other_cache()),
        {"nonlocal_power"},
    ),
    (EXPOSURE_DIR, ("subset", None), {"left_out"}),
    (EXPOSURE_DIR, ("session_cache", _other_cache()), {"left_out"}),
    (
        CONTENT_ALONG_ARMS_DIR,
        ("event_table", {"sha256": "old"}),
        {"content_occupancy"},
    ),
    (EVENT_PARTICIPATION_DIR, ("event_table", {"sha256": "old"}), {"participation"}),
    (
        EVENT_LIKELIHOOD_ADVANTAGE_DIR,
        ("event_table", {"sha256": "old"}),
        {"likelihood_advantage"},
    ),
    (
        EVENT_LIKELIHOOD_ADVANTAGE_DIR,
        ("event_participation", {"sha256": "old"}),
        {"likelihood_advantage"},
    ),
    (
        EVENT_TETRODE_HOLDOUT_DIR,
        ("event_table", {"sha256": "old"}),
        {"tetrode_holdout"},
    ),
    (
        EVENT_TETRODE_HOLDOUT_DIR,
        ("event_participation", {"sha256": "old"}),
        {"tetrode_holdout"},
    ),
    ("state_evidence", ("subset", None), {"heldout_by_swr"}),
    ("state_evidence", ("session_cache", _other_cache()), {"heldout_by_swr"}),
]


@pytest.mark.parametrize(
    ("folder", "change", "pending"),
    PENDING_CASES,
    ids=[f"{folder}-{change[0]}" for folder, change, _ in PENDING_CASES],
)
def test_an_output_off_the_subset_or_stale_makes_its_values_pending(
    processed_data_root, monkeypatch, folder, change, pending
):
    run = _Run(processed_data_root, monkeypatch, {folder: change})
    values = run.summarize()
    assert {key for key, value in values.items() if value.pending} == pending
    if folder == "state_evidence":
        # a split not over the subset has no bout counts to report
        assert "heldout_bout_values" not in run.seen


def test_a_table_whose_labels_disagree_with_the_visits_is_refused(
    processed_data_root, monkeypatch
):
    run = _Run(processed_data_root, monkeypatch)
    # the planted event's epoch has a visit that holds it
    visits_by = dict(VISITS)
    visits_by["cd1_e1_HPC"] = (np.array([0.0]), np.array([60.0]))
    monkeypatch.setattr(
        builder, "well_visit_times", lambda identifier: visits_by[identifier]
    )
    with pytest.raises(ValueError, match="1 of 14 events' visit_context disagrees"):
        run.summarize()
