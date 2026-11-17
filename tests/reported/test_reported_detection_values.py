"""Each value of the detection summary is the statistic its key names.

The sweeps' summaries and the labeled recording's numbers are kept in the
repository, and so is the detection summary built from them
(``paper/reported/detection.json``): the builder, run on those inputs, must
give that summary's values, key by key, so a statistic wired to another key
(the lowest for the highest, one detector for the other) cannot pass. The
held-out tetrode values come from a run that is not in the repository; they are
checked on a table small enough to read.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported import detection

COMMITTED = ROOT_DIR / "paper" / "reported" / "detection.json"


def _committed_values(function: str) -> dict[str, float]:
    values = json.loads(COMMITTED.read_text())["values"]
    return {
        key: value["value"]
        for key, value in values.items()
        if value["function"] == function
    }


def _recording_numbers() -> dict:
    path = ROOT_DIR / detection.SWEEP_DIR / "memoryless_comparison_recording.json"
    numbers: dict = json.loads(path.read_text())["numbers"]
    return numbers


def _sweep_summary(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT_DIR / detection.SWEEP_DIR / f"detection_{name}_summary.csv")


@pytest.mark.parametrize(
    ("build", "n_values"),
    [
        (detection.sweep_values, 69),
        (detection.controls_values, 34),
        (detection.heldout_values, 18),
        (detection.recording_values, 12),
    ],
)
def test_the_committed_inputs_give_the_committed_values(build, n_values):
    """The comparison allows a relative difference of 1e-12, not none: the
    summaries are read with pandas' default CSV float parser, which is not
    exact, so a value can differ in its last digit from the committed one,
    built on another machine."""
    table = {
        detection.sweep_values: lambda: _sweep_summary("sweep"),
        detection.controls_values: lambda: _sweep_summary("controls"),
        detection.heldout_values: lambda: _sweep_summary("heldout"),
        detection.recording_values: _recording_numbers,
    }[build]()
    built = {value.key: value.value for value in build(table, "source")}
    committed = _committed_values(build.__name__)
    assert len(committed) == n_values
    assert set(built) == set(committed)
    for key, value in committed.items():
        assert built[key] == pytest.approx(value, rel=1e-12, abs=0), key


# --- the held-out tetrode test per decoder state --------------------------------


def _bout(
    state: str,
    regime: str,
    metric: str,
    estimate: float,
    animals_positive: int,
    epochs_positive: int,
    n_epochs: int,
    primary: bool = True,
) -> dict:
    """One row of ``bout_stats``: an interval of ``estimate`` - 1 to ``estimate``
    + 2 over 5 animals."""
    return {
        "state": state,
        "regime": regime,
        "metric": metric,
        "primary": primary,
        "n_animals": 5,
        "n_epochs": n_epochs,
        "animal_estimate": estimate,
        "animal_ci_lo": estimate - 1,
        "animal_ci_hi": estimate + 2,
        "animals_pos": f"{animals_positive}/5",
        "epochs_pos": f"{epochs_positive}/{n_epochs}",
    }


BOUTS = [
    _bout("Local", "immobile", "runcv_loc_mark", 10.0, 5, 9, 9),
    # the same state when running is not reported
    _bout("Local", "running", "runcv_loc_mark", 99.0, 1, 1, 9),
    _bout("No-Spike", "immobile", "runcv_ns_logratio", 20.0, 4, 8, 9),
    # continuous bouts are reported in both regimes, against the animal's place
    _bout("Non-Local Continuous", "immobile", "runcv_nl_anim_mark", 30.0, 3, 7, 9),
    _bout("Non-Local Continuous", "running", "runcv_nl_anim_mark", -40.0, 0, 2, 8),
    _bout("Non-Local Continuous", "immobile", "runcv_nl_null_mark", 98.0, 1, 1, 9),
    # fragmented bouts against a mismatched MAP, not against the animal's place
    _bout("Non-Local Fragmented", "immobile", "runcv_nl_null_mark", 50.0, 2, 5, 9),
    _bout("Non-Local Fragmented", "immobile", "runcv_nl_anim_mark", 97.0, 1, 1, 9),
]
MANIFEST = {
    "epochs": {
        "a_02_r1_HPC": {"status": "ok"},
        "a_04_r2_HPC": {"status": "failed"},
        "b_02_r1_HPC": {"status": "ok"},
    }
}


def _state_evidence(bouts: list[dict]) -> dict[str, float]:
    values = detection.state_evidence_values(pd.DataFrame(bouts), MANIFEST, "source")
    return {value.key: value.value for value in values}


def test_state_evidence_values_read_each_states_primary_row():
    values = _state_evidence(BOUTS)
    assert len(values) == 33
    assert values["state_evidence_epochs"] == 3
    assert values["state_evidence_epochs_ok"] == 2
    assert values["state_evidence_animals"] == 5
    assert values["heldout_local_immobile_estimate"] == 10.0
    assert values["heldout_local_immobile_ci_low"] == 9.0
    assert values["heldout_local_immobile_ci_high"] == 12.0
    assert values["heldout_no_spike_immobile_animals_positive"] == 4
    assert values["heldout_no_spike_immobile_epochs_positive"] == 8
    assert values["heldout_no_spike_immobile_epochs"] == 9
    assert values["heldout_continuous_immobile_estimate"] == 30.0
    assert values["heldout_continuous_running_estimate"] == -40.0
    assert values["heldout_continuous_running_animals_positive"] == 0
    assert values["heldout_continuous_running_epochs_positive"] == 2
    assert values["heldout_continuous_running_epochs"] == 8
    assert values["heldout_fragmented_immobile_mismatched_estimate"] == 50.0
    assert values["heldout_fragmented_immobile_mismatched_animals_positive"] == 2


def test_state_evidence_values_refuse_a_state_without_one_primary_row():
    not_primary = [row | {"primary": row["state"] != "No-Spike"} for row in BOUTS]
    with pytest.raises(ValueError, match="no single primary row for No-Spike"):
        _state_evidence(not_primary)
    with pytest.raises(ValueError, match="no single primary row for Local"):
        _state_evidence([*BOUTS, BOUTS[0]])
    with pytest.raises(ValueError, match="no single primary row for Local"):
        _state_evidence(BOUTS[1:])


@pytest.mark.parametrize(
    "change", [{"animals_pos": "4/4"}, {"epochs_pos": "8/8"}, {"n_epochs": 8}]
)
def test_state_evidence_values_refuse_counts_that_disagree(change):
    # the first row counts 5 animals and 9 epochs
    with pytest.raises(ValueError, match="inconsistent counts for Local"):
        _state_evidence([BOUTS[0] | change, *BOUTS[1:]])
