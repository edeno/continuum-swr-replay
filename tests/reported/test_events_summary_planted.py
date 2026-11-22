"""The events summary carries a planted out-of-visit event, quiet window and
other-detector event through each of its steps: no step after the selection
reads them, and every output not made over the subset, from the current event
table or from the current well visits makes its values pending.

The restriction runs are planted too: the window run's windows (an event and
its control per event, the planted one's too, with the columns the summary
checks for); the held-out event run's events (one per canonical event that
starts inside a visit, and two that start outside one), its canonical events
found again (every canonical event, the planted one too), its time sums (one
row per epoch) and each epoch's time step; the restriction run (a fold fit) and
the per-bin facts run only through their manifests. The restriction run and the
held-out tetrode run are held out in folds of whole trials, as their settings
record.

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
    EVENT_BIN_FACTS_DIR,
    EVENT_GAMMA_DIR,
    EVENT_LIKELIHOOD_ADVANTAGE_DIR,
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
    EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
    EVENT_PARTICIPATION_DIR,
    EVENT_PROPERTY_BOOTSTRAP_DIR,
    EVENT_START_REST_THETA_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    EXPOSURE_IN_VISITS_DIR,
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
#: The restriction builders, by the argument holding the canonical windows
#: (None: none).
RESTRICTION_BUILDERS = {
    "restriction_window_values": 0,
    "restriction_heldout_event_values": None,
    "restriction_agreement_values": None,
    "restriction_outside_swr_values": None,
    "restriction_immobile_values": None,
    "restriction_outside_swr_part_values": None,
}
#: The builders given the held-out events, by the argument holding them.
HELDOUT_BUILDERS = {
    "restriction_heldout_event_values": 0,
    "restriction_agreement_values": 0,
    "restriction_outside_swr_values": 2,
}
N_HELDOUT_INSIDE = 13
"""The held-out events that start inside a visit; two more start outside."""
#: The window value that reads the well visits (through the bin facts), and the
#: outside-SWR value over the time no held-out event holds.
AFTER_VISIT = "restriction_window_after_visit"
OUTSIDE_EVENTS = "restriction_outside_swr_outside_events"
BIN_S = 0.002
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


def _restriction_windows() -> pd.DataFrame:
    """The window run's rows: a per-event run's, with the sums over the bins
    outside SWRs and from the per-bin facts that the summary requires (zero)."""
    columns = (
        *builder.restriction_windows.OUTSIDE_SWR_COLUMNS,
        *builder.restriction_windows.BIN_FACT_COLUMNS,
    )
    return _per_event_run().assign(**dict.fromkeys(columns, 0))


def _heldout_events() -> pd.DataFrame:
    """The held-out event run's events: 13 that start inside a visit and two
    planted that start outside one."""
    inside = [True] * N_HELDOUT_INSIDE + [False, False]
    return pd.DataFrame(
        {
            "heldout_event_id": [f"h{i}" for i in range(len(inside))],
            "starts_in_well_visit": inside,
        }
    )


def _canonical_found() -> pd.DataFrame:
    """The held-out event run's canonical events: every event, the planted one
    too."""
    return pd.DataFrame(
        {"event_id": np.array([*range(13), PLANTED], dtype="uint64"), "found": True}
    )


def _time_sums() -> pd.DataFrame:
    """The held-out event run's time sums: one row per epoch."""
    return pd.DataFrame({"epoch_id": EPOCHS, "n_bins": 1})


def _read_complete_run(folder, name, epochs) -> pd.DataFrame:
    return {
        "nonlocal_restriction_windows": _restriction_windows,
        "heldout_events": _heldout_events,
        "canonical_events_found": _canonical_found,
        "heldout_time_sums": _time_sums,
    }.get(name, _per_event_run)()


DECODE = "d" * 64
"""Each epoch's current decode frame's SHA-256, as the event features run
records it."""


def _restriction_settings(**changes) -> dict:
    """The restriction run's settings: the fold fit on folds of whole trials,
    under the current SWR source, with ``changes``."""
    return {
        "n_folds": 5,
        "fit": "folds",
        "fold_design": "trials",
        "em": {"max_iter": 20},
        "swr_source": builder.swr_source.swr_source_name(),
        **changes,
    }


def _restriction_records(**decodes) -> dict:
    """The restriction run's records, each with the decode frame it read: the
    current one, or ``decodes[epoch]`` (None: a record without
    ``epoch_inputs``)."""
    records = {}
    for epoch in EPOCHS:
        record = {"status": "ok", "bin_s": BIN_S}
        sha256 = decodes.get(epoch, DECODE)
        if sha256 is not None:
            record["epoch_inputs"] = {"data_over_time": {"sha256": sha256}}
        records[epoch] = record
    return records


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
            "settings": {"subset": ANALYZED_SUBSET},
            "tetrode_holdout": {"sha256": "current"},
            "nonlocal_restriction": {"sha256": "current"},
            "bin_facts": {"sha256": "current"},
            "epochs": {epoch: {"status": "ok", "bin_s": BIN_S} for epoch in EPOCHS},
        }
        self.manifests = {}
        for folder in (
            EVENT_START_REST_THETA_DIR,
            EVENT_PROPERTY_BOOTSTRAP_DIR,
            BRAIN_STATE_REFERENCE_DIR,
            EVENT_GAMMA_DIR,
            RIPPLE_CLASS_STRUCTURE_DIR,
            NONLOCAL_BY_RIPPLE_POWER_DIR,
            EXPOSURE_IN_VISITS_DIR,
            CONTENT_ALONG_ARMS_DIR,
            EVENT_PARTICIPATION_DIR,
            EVENT_LIKELIHOOD_ADVANTAGE_DIR,
            EVENT_TETRODE_HOLDOUT_DIR,
            EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
            EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
            EVENT_BIN_FACTS_DIR,
            "state_evidence",
        ):
            (root / folder).mkdir(parents=True, exist_ok=True)
            self.manifests[folder] = json.loads(json.dumps(fresh))
        # the restriction run is the fold fit: its settings record the fit and
        # the current SWR source, its records the current decode frames
        (root / EVENT_NONLOCAL_RESTRICTION_DIR).mkdir(parents=True, exist_ok=True)
        self.manifests[EVENT_NONLOCAL_RESTRICTION_DIR] = {
            **json.loads(json.dumps(fresh)),
            "settings": _restriction_settings(),
            "epochs": _restriction_records(),
        }
        # the held-out tetrode run is held out in folds of whole trials
        self.manifests[EVENT_TETRODE_HOLDOUT_DIR]["settings"] = {
            "n_folds": 5,
            "fold_design": "trials",
        }
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
            root / EXPOSURE_IN_VISITS_DIR / "exposure.parquet"
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
            nonlocal_restriction=root / EVENT_NONLOCAL_RESTRICTION_DIR,
            nonlocal_restriction_windows=root / EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
            nonlocal_restriction_heldout_events=root
            / EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
            bin_facts=root / EVENT_BIN_FACTS_DIR,
            exposure_in_visits=root / EXPOSURE_IN_VISITS_DIR,
        )

    def _spy(self, name: str, group=None, group_key: str | None = None):
        """A builder that records its arguments and returns one value, or, given
        ``group`` (a builder's named groups), that value and ``group_key``'s in
        the group's two fields."""

        def spy(*args):
            self.seen[name] = args
            make = value_factory(spy, "s", Tags(("2a.1",)))
            value = make(name.removesuffix("_values"), 1, "count", "count", name)
            if group is None:
                return [value]
            return group([value], [make(group_key, 1, "count", "count", name)])

        return spy

    def _patch(self, monkeypatch) -> None:
        seen = self.seen
        for name in (*EVENT_BUILDERS, *OTHER_BUILDERS, *RESTRICTION_BUILDERS):
            monkeypatch.setattr(builder, name, self._spy(name))
        # the window values hold counts that read the visits
        monkeypatch.setattr(
            builder,
            "restriction_window_values",
            self._spy("restriction_window_values", builder.WindowValues, AFTER_VISIT),
        )
        # the outside-SWR values hold the time that no event holds
        monkeypatch.setattr(
            builder,
            "restriction_outside_swr_values",
            self._spy(
                "restriction_outside_swr_values",
                builder.OutsideSwrValues,
                OUTSIDE_EVENTS,
            ),
        )

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
            # the event features run, with the decode frames it read
            "read_manifest": lambda run: {
                "epochs": dict.fromkeys(
                    EPOCHS, {"status": "ok", "decode": {"sha256": DECODE}}
                )
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
            "read_complete_run": _read_complete_run,
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
    # the restriction windows too; the time sums and the time steps are whole
    for name, at in RESTRICTION_BUILDERS.items():
        if at is not None:
            windows = seen[name][at]
            assert len(windows) == 2 * 13 and PLANTED not in set(windows["event_id"])
    # the held-out events that start inside a visit, not the two outside
    for name, at in HELDOUT_BUILDERS.items():
        heldout = seen[name][at]
        assert len(heldout) == N_HELDOUT_INSIDE, name
        assert heldout["starts_in_well_visit"].all(), name
    # the canonical events found again: the selected ones only
    found = seen["restriction_agreement_values"][1]
    assert len(found) == 13 and PLANTED not in set(found["event_id"])
    for name in (
        "restriction_outside_swr_values",
        "restriction_immobile_values",
        "restriction_outside_swr_part_values",
    ):
        assert seen[name][0]["epoch_id"].tolist() == EPOCHS
        assert seen[name][1].to_dict() == dict.fromkeys(EPOCHS, BIN_S)
    # the other detectors' events are selected by the comparison, given the visits
    assert _same_visits(seen["read_comparison"][0])
    assert seen["other_detector_values"][0]["planted"].tolist() == [PLANTED]
    # the spike evidence: the event inside a visit and the quiet window inside
    evidence = seen["spike_evidence_values"][0]
    assert evidence["start_time"].tolist() == [10.0, 5.0]
    assert len(seen["spike_evidence_visit_values"][0]) == 4
    assert seen["spike_evidence_visit_values"][1].tolist() == [True, False, True, False]
    # what another inventory leaves out: of the selected events, with the visits
    event_ids, visits, exposure = seen["left_out_values"]
    assert PLANTED not in event_ids and len(event_ids) == 13
    assert _same_visits(visits)
    assert exposure["epoch_id"].tolist() == EPOCHS
    # the held-out bouts are counted when the split is over the subset
    assert "heldout_bout_values" in seen
    assert not [key for key, value in values.items() if value.pending]
    assert "wholly inside one well visit" in values["participation"].description
    assert "wholly inside one well visit" in values["tetrode_holdout"].description
    assert "wherever that event starts" in values["left_out"].description
    assert "partner may start outside a visit" in values["other_detector"].description
    # the restriction values say what they are over, in the summary's words
    subset = builder.CANONICAL_WINDOWS_SUBSET
    for key in ("restriction_window", AFTER_VISIT):
        assert values[key].description.endswith(f"; over {subset}")
    assert values["restriction_outside_swr"].description.endswith(
        f"; over {builder.TIME_SUBSET}, and {builder.HELDOUT_EVENTS_SUBSET}"
    )
    assert values["restriction_heldout_event"].description.endswith(
        f"; over {builder.HELDOUT_EVENTS_SUBSET}"
    )
    assert values["restriction_agreement"].description.endswith(
        "so its partner may start outside a visit"
    )
    # the time no event holds is outside every event, selected or not
    for key in ("restriction_immobile", "restriction_outside_swr_part", OUTSIDE_EVENTS):
        assert values[key].description.endswith(f"; over {builder.TIME_SUBSET}")


def _same_visits(visits) -> bool:
    """Whether ``visits`` are the epochs' visits, as ``well_visit_times`` gave them."""
    return set(visits) == set(EPOCHS) and all(
        np.array_equal(visits[epoch][i], VISITS[epoch][i])
        for epoch in EPOCHS
        for i in (0, 1)
    )


def _other_cache() -> dict:
    return {"sha256": "0" * 64}


#: The restriction values, made pending by what they read: the window run's,
#: the held-out event run's over its events, and over immobile time
WINDOW_VALUES = {"restriction_window", AFTER_VISIT}
HELDOUT_EVENT_VALUES = {
    "restriction_heldout_event",
    "restriction_agreement",
    "restriction_outside_swr",
    OUTSIDE_EVENTS,
}
HELDOUT_RUN_VALUES = HELDOUT_EVENT_VALUES | {
    "restriction_immobile",
    "restriction_outside_swr_part",
}
RESTRICTION_VISIT_VALUES = (WINDOW_VALUES | HELDOUT_RUN_VALUES) - {"restriction_window"}
_NOTE = {"event_table_sha256": "c" * 64, "tetrode_holdout_event_table_sha256": None}
#: The values against the participation run's Local-rest controls
CONTROL_VALUES = {"participation", "tetrode_holdout", "restriction_window", AFTER_VISIT}

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
    (EXPOSURE_IN_VISITS_DIR, ("subset", None), {"left_out"}),
    (EXPOSURE_IN_VISITS_DIR, ("session_cache", _other_cache()), {"left_out"}),
    (
        CONTENT_ALONG_ARMS_DIR,
        ("event_table", {"sha256": "old"}),
        {"content_occupancy"},
    ),
    (EVENT_PARTICIPATION_DIR, ("event_table", {"sha256": "old"}), {"participation"}),
    # a participation run off the subset or the visits: its controls are not
    # Local rest inside the current visits, so the values against them pend
    (EVENT_PARTICIPATION_DIR, ("settings", None), CONTROL_VALUES),
    (EVENT_PARTICIPATION_DIR, ("settings", {"subset": "every event"}), CONTROL_VALUES),
    (EVENT_PARTICIPATION_DIR, ("session_cache", _other_cache()), CONTROL_VALUES),
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
    *(
        (EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR, (name, {"sha256": "old"}), pending)
        for name, pending in (
            ("event_table", WINDOW_VALUES),
            ("event_participation", WINDOW_VALUES),
            ("tetrode_holdout", WINDOW_VALUES),
            ("nonlocal_restriction", WINDOW_VALUES),
            ("bin_facts", WINDOW_VALUES),
        )
    ),
    (
        EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
        ("settings", {builder.restriction_windows.EVENT_TABLE_NOTE: _NOTE}),
        WINDOW_VALUES,
    ),
    # the held-out event run: its events read the event table (the canonical
    # check and the canonical events found again), the restriction run and the
    # bin facts; its time sums over immobile time read the latter two only
    *(
        (
            EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
            (name, {"sha256": "old"}),
            pending,
        )
        for name, pending in (
            ("event_table", HELDOUT_EVENT_VALUES),
            ("nonlocal_restriction", HELDOUT_RUN_VALUES),
            ("bin_facts", HELDOUT_RUN_VALUES),
        )
    ),
    # the visits reach the restriction values through the bin facts only: a run
    # made from another session cache, or before it recorded one, makes the
    # values that read them pending (not the window values that do not)
    (EVENT_BIN_FACTS_DIR, ("session_cache", _other_cache()), RESTRICTION_VISIT_VALUES),
    (EVENT_BIN_FACTS_DIR, ("session_cache", None), RESTRICTION_VISIT_VALUES),
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


def test_nothing_is_pending_when_every_output_is_current(
    processed_data_root, monkeypatch
):
    values = _Run(processed_data_root, monkeypatch).summarize()
    assert values and not [key for key, value in values.items() if value.pending]


#: The restriction values, every one of which reads the restriction run
RESTRICTION_VALUES = WINDOW_VALUES | HELDOUT_RUN_VALUES


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            ("settings", _restriction_settings(swr_source="kay@old+artifacts@none")),
            "its settings record the SWR source 'kay@old+artifacts@none', not the "
            "current",
        ),
        (
            ("epochs", _restriction_records(ad1_e2_HPC="e" * 64)),
            "1 of its epochs record another decode frame than the one event "
            "features run",
        ),
        (
            ("epochs", _restriction_records(ad1_e1_HPC=None, bd1_e1_HPC=None)),
            "2 of its epochs record another decode frame",
        ),
    ],
    ids=["another SWR source", "another decode", "no decode recorded"],
)
def test_a_restriction_run_off_the_current_decodes_or_swr_source_is_pending(
    processed_data_root, monkeypatch, change, message
):
    """A restriction run made from other decode frames or another SWR source
    than the current ones makes every restriction value pending (the window
    run's and the held-out event run's, over events and over time), with a
    note naming the run and what differs; no other value."""
    run = _Run(
        processed_data_root, monkeypatch, {EVENT_NONLOCAL_RESTRICTION_DIR: change}
    )
    values = run.summarize()
    pending = {key: value for key, value in values.items() if value.pending}
    assert set(pending) == RESTRICTION_VALUES
    for value in pending.values():
        assert f"{EVENT_NONLOCAL_RESTRICTION_DIR} was not made from the current " in (
            value.pending
        )
        assert message in value.pending


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


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"n_folds": 5, "fold_design": "trials"}, "is not the fold fit \\("),
        ({"n_folds": 5, "fit": "shared"}, "is not the fold fit \\("),
        ({"n_folds": 5, "fit": "folds"}, "fold_design None, not 'trials'"),
        (
            {"n_folds": 5, "fit": "folds", "fold_design": "contiguous"},
            "fold_design 'contiguous', not 'trials'",
        ),
    ],
    ids=["no fit", "another fit", "no fold design", "contiguous blocks"],
)
def test_a_restriction_run_that_is_not_the_fold_fit_on_trials_is_refused(
    processed_data_root, monkeypatch, settings, message
):
    """A run of the earlier design (its settings record no fit), or a fold fit
    on contiguous blocks of time, cannot give values that describe the fold fit
    on folds of whole trials."""
    run = _Run(
        processed_data_root,
        monkeypatch,
        {EVENT_NONLOCAL_RESTRICTION_DIR: ("settings", settings)},
    )
    with pytest.raises(ValueError, match=message):
        run.summarize()


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"n_folds": 5, "margin_bins": 2}, "fold_design None, not 'trials'"),
        (
            {"n_folds": 5, "fold_design": "contiguous"},
            "fold_design 'contiguous', not 'trials'",
        ),
    ],
    ids=["no fold design", "another design"],
)
def test_a_tetrode_holdout_run_not_on_trial_folds_is_refused(
    processed_data_root, monkeypatch, settings, message
):
    """A held-out tetrode run of an older fold design (none recorded, as the
    first run's ``{n_folds, margin_bins}``, or another design) cannot give
    values that describe encodings held out in folds of whole trials."""
    run = _Run(
        processed_data_root,
        monkeypatch,
        {EVENT_TETRODE_HOLDOUT_DIR: ("settings", settings)},
    )
    with pytest.raises(ValueError, match=message) as raised:
        run.summarize()
    assert "event_tetrode_holdout is not held out in folds" in str(raised.value)
