"""The ripple-power covariate: the Kay detector's own z-scored consensus.

``SWR_zscore`` (``ripple_power_z`` in the analyses) is the consensus the Kay
detector thresholds, over the tetrodes rule R-T kept for the configured SWR
inventory, z-scored with the artifact mask left out of the normalization, and
no value (NaN) inside the mask. So a rogue tetrode that R-T dropped does not
move it, masked time does not enter its normalization, and it agrees with the
inventory's events. Offline: synthetic bands, no Spyglass.
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest
from ripple_detection import Kay_ripple_detector

from continuum_swr_replay.analysis.detection.swr_detectors import (
    DETECTOR_SETTINGS,
)
from continuum_swr_replay.analysis.events import swr_source
from continuum_swr_replay.data_loaders.lfp import (
    KAY_MINIMUM_DURATION_S,
    KAY_SMOOTHING_SIGMA_S,
    kay_ripple_power_z,
    with_ripple_power,
)
from tests.support._swr_inventory_fixtures import (
    use_artifact_run,
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

FS = 1000.0
TIME = 100.0 + np.arange(20_000) / FS
"""20 s at 1000 Hz."""
MASK = pd.DataFrame({"start_time": [108.0], "end_time": [109.0]})


def _band(seed: int = 0) -> np.ndarray:
    """Three tetrodes rippling together every 2 s, and a fourth, rogue, that
    bursts on its own; a loud step inside [108, 109] on every channel."""
    rng = np.random.default_rng(seed)
    band = rng.normal(0.0, 1.0, (TIME.size, 4))
    carrier = np.sin(2 * np.pi * 180.0 * TIME)
    for at in np.arange(101.0, 120.0, 2.0):
        burst = 6.0 * carrier * np.exp(-0.5 * ((TIME - at) / 0.02) ** 2)
        band[:, :3] += burst[:, None]
        band[:, 3] += 8.0 * carrier * np.exp(-0.5 * ((TIME - at - 1.0) / 0.02) ** 2)
    band[(TIME >= 108.4) & (TIME <= 108.6)] += (
        500.0 * carrier[(TIME >= 108.4) & (TIME <= 108.6), None]
    )
    return band


def _unobserved() -> np.ndarray:
    return (TIME >= 108.0) & (TIME <= 109.0)


def test_the_smoothing_and_minimum_duration_are_kays():
    assert KAY_SMOOTHING_SIGMA_S == DETECTOR_SETTINGS["kay"]["smoothing_sigma"]
    assert KAY_MINIMUM_DURATION_S == DETECTOR_SETTINGS["kay"]["minimum_duration"]


def test_a_rogue_tetrode_rt_dropped_does_not_move_it():
    band = _band()

    _, kept_only = kay_ripple_power_z(TIME, band[:, :3], FS, _unobserved())
    _, with_rogue = kay_ripple_power_z(TIME, band, FS, _unobserved())

    np.testing.assert_array_equal(np.isnan(kept_only), _unobserved())
    # the rogue's own bursts, 1 s after the shared ones, are in one but not the other
    rogue_burst = np.abs(TIME - 102.0) < 0.01
    assert np.nanmax(with_rogue[rogue_burst]) > 3.0
    assert np.nanmax(kept_only[rogue_burst]) < 2.0


def test_masked_time_is_not_in_the_normalization_and_has_no_value():
    band = _band()
    unobserved = _unobserved()

    consensus, z = kay_ripple_power_z(TIME, band[:, :3], FS, unobserved)

    assert np.isnan(z[unobserved]).all()
    observed = consensus[~unobserved]
    expected = (consensus - observed.mean()) / observed.std()
    np.testing.assert_allclose(z[~unobserved], expected[~unobserved], rtol=1e-12)
    # the step inside the mask would otherwise dominate the scale
    _, unmasked = kay_ripple_power_z(TIME, band[:, :3], FS, np.zeros(TIME.size, bool))
    assert np.nanmax(z[~unobserved]) > 3 * np.nanmax(unmasked[~unobserved])


def test_it_is_the_trace_the_kay_detector_thresholds():
    """The Kay detector's events, normalized over the same sample, peak at the
    covariate's values: the covariate and the SWR events agree."""
    band = _band()[:, :3]
    unobserved = _unobserved()
    events = Kay_ripple_detector(
        TIME,
        band,
        np.zeros(TIME.size),
        FS,
        **dict(DETECTOR_SETTINGS["kay"]),
        normalization_mask=~unobserved,
    )
    _, z = kay_ripple_power_z(TIME, band, FS, unobserved)

    # the runner drops the events in the mask, where the covariate has no value
    events = events[(events.end_time < 108.0) | (events.start_time > 109.0)]
    assert len(events) >= 8
    for event in events.itertuples():
        inside = (TIME >= event.start_time) & (TIME <= event.end_time)
        assert np.max(z[inside]) == pytest.approx(event.max_zscore, rel=1e-9)


def test_on_a_gappy_band_it_is_still_the_trace_kay_thresholds():
    """Kay treats a gap-free block shorter than its minimum duration as
    missing, so it is in neither the normalization nor the trace: the
    covariate leaves it out the same way, and every event still peaks at its
    ``max_zscore``."""
    band = _band()[:, :3].copy()
    unobserved = _unobserved()
    # islands between NaN gaps: 10 ms (shorter than Kay's 15 ms) of very loud
    # signal, which would dominate the scale if it counted, and a 50 ms one,
    # each fenced by missing samples
    for start, n_samples, gain in ((103.400, 10, 40.0), (115.400, 50, 1.0)):
        first = int(np.searchsorted(TIME, start))
        band[first - 20 : first] = np.nan
        band[first : first + n_samples] *= gain
        band[first + n_samples : first + n_samples + 20] = np.nan
    with pytest.warns(UserWarning, match="shorter"):
        events = Kay_ripple_detector(
            TIME,
            band,
            np.zeros(TIME.size),
            FS,
            **dict(DETECTOR_SETTINGS["kay"]),
            normalization_mask=~unobserved,
        )
    with pytest.warns(UserWarning, match="shorter"):
        _, z = kay_ripple_power_z(TIME, band, FS, unobserved)

    short = (TIME >= 103.400) & (TIME < 103.410)
    assert np.isnan(z[short]).all()
    events = events[(events.end_time < 108.0) | (events.start_time > 109.0)]
    assert len(events) >= 8
    for event in events.itertuples():
        inside = (TIME >= event.start_time) & (TIME <= event.end_time)
        assert np.nanmax(z[inside]) == pytest.approx(event.max_zscore, rel=1e-9)


class TestTheSourcesInputs:
    def test_the_kept_tetrodes_and_the_band_come_from_the_inventory(
        self, tmp_path, monkeypatch
    ):
        epoch_id = "m20200101_02_r1_HPC"
        inventory = write_swr_inventory(tmp_path, {epoch_id: [(1.0, 1.05)]})
        manifest = json.loads((inventory / "manifest.json").read_text())
        record = manifest["epochs"][epoch_id]
        record["electrode_ids"] = [4, 12]
        record["sampling"] = {"nominal": 1000.0, "measured": 999.99}
        record["position_span_s"] = [10.0, 900.0]
        record["tetrode_selection"] = {
            "electrode_ids": [0, 4, 12],
            "kept": [False, True, True],
        }
        (inventory / "manifest.json").write_text(json.dumps(manifest))
        artifacts = write_artifact_run(tmp_path, {epoch_id: [("S", 3.0, 4.0)]})
        use_swr_source(monkeypatch, inventory)
        use_artifact_run(monkeypatch, artifacts)

        inputs = swr_source.ripple_power_inputs(epoch_id)

        assert inputs["band_electrode_ids"] == [0, 4, 12]
        assert inputs["kept_electrode_ids"] == [4, 12]
        assert inputs["sampling_frequency"] == 1000.0
        assert inputs["position_span_s"] == (10.0, 900.0)
        np.testing.assert_allclose(
            inputs["artifact_intervals"].to_numpy(), [[2.0, 5.0]]
        )

    def test_an_inventory_made_before_rule_rt_keeps_every_band_tetrode(
        self, tmp_path, monkeypatch
    ):
        epoch_id = "m20200101_02_r1_HPC"
        inventory = write_swr_inventory(tmp_path, {epoch_id: [(1.0, 1.05)]})
        manifest = json.loads((inventory / "manifest.json").read_text())
        manifest["epochs"][epoch_id]["electrode_ids"] = [0, 4, 12]
        manifest["epochs"][epoch_id]["sampling"] = {"nominal": 1000.0}
        (inventory / "manifest.json").write_text(json.dumps(manifest))
        use_swr_source(monkeypatch, inventory)

        inputs = swr_source.ripple_power_inputs(epoch_id)

        assert inputs["band_electrode_ids"] == inputs["kept_electrode_ids"]
        assert inputs["artifact_intervals"].empty


INPUTS = {
    "band_electrode_ids": [0, 4, 12, 16],
    "kept_electrode_ids": [0, 4, 12],
    "sampling_frequency": FS,
    "position_span_s": (float(TIME[1000]), float(TIME[-1001])),
    "artifact_intervals": MASK,
}


def _ripple_data(time: np.ndarray = TIME) -> dict:
    band = pd.DataFrame(_band()[np.isin(TIME, time)], index=pd.Index(time, name="time"))
    legacy = pd.Series(np.zeros(time.size), index=band.index)
    return {
        "ripple_times": pd.DataFrame({"start_time": [], "end_time": []}),
        "ripple_filtered_lfps": band,
        "ripple_lfps": band,
        "ripple_consensus_trace": legacy,
        "zscored_ripple_consensus_trace": legacy,
    }


class TestTheLoader:
    """``load_data`` replaces both consensus traces when it is given the
    source's inputs, on the band within the position span."""

    def test_the_traces_are_the_kept_tetrodes_on_the_band_given(self):
        span = TIME[1000:-1000]

        out = with_ripple_power(
            _ripple_data(span),
            INPUTS,
            band_electrode_ids=np.array([0, 4, 12, 16]),
            position_span_s=(span[0], span[-1]),
        )

        consensus, z = kay_ripple_power_z(
            span, _band()[1000:-1000, :3], FS, (span >= 108.0) & (span <= 109.0)
        )
        np.testing.assert_array_equal(
            out["zscored_ripple_consensus_trace"].to_numpy(), z
        )
        np.testing.assert_array_equal(
            out["ripple_consensus_trace"].to_numpy(), consensus
        )
        assert out["zscored_ripple_consensus_trace"].index.equals(pd.Index(span))
        # the other entries are the loader's
        assert out["ripple_filtered_lfps"].shape[1] == 4

    def test_a_band_other_than_the_inventorys_is_refused(self):
        span = TIME[1000:-1000]
        with pytest.raises(ValueError, match="band"):
            with_ripple_power(
                _ripple_data(span),
                INPUTS | {"band_electrode_ids": [0, 4, 12, 20]},
                band_electrode_ids=np.array([0, 4, 12, 16]),
                position_span_s=(span[0], span[-1]),
            )

    def test_a_span_other_than_the_inventorys_is_refused(self):
        """The normalization reads the band within the position span, so a
        position trimmed differently from the runner's would move it."""
        span = TIME[1000:-1002]
        with pytest.raises(ValueError, match="span"):
            with_ripple_power(
                _ripple_data(span),
                INPUTS,
                band_electrode_ids=np.array([0, 4, 12, 16]),
                position_span_s=(span[0], span[-1]),
            )


@pytest.fixture
def bandit_task():
    """``data_loaders.bandit_task``; it imports Spyglass only inside its loaders."""
    return importlib.import_module("continuum_swr_replay.data_loaders.bandit_task")


@pytest.mark.parametrize("given", [True, False])
def test_load_data_uses_the_sources_covariate_when_given(
    bandit_task, monkeypatch, given
):
    span = TIME[1000:-1000]
    monkeypatch.setattr(
        bandit_task,
        "get_epoch_position_info",
        lambda nwb_file_name, epoch_name: {
            "position_info": pd.DataFrame(
                {"head_speed": np.zeros(span.size)}, index=pd.Index(span, name="time")
            ),
            "track_graph": None,
            "linear_edge_order": None,
            "linear_edge_spacing": None,
        },
    )
    monkeypatch.setattr(bandit_task, "get_ripple", lambda *args: _ripple_data())
    monkeypatch.setattr(
        bandit_task,
        "ripple_band_electrode_ids",
        lambda nwb_file_name, epoch_name: np.array([0, 4, 12, 16]),
    )

    data = bandit_task.load_data(
        "sim20200101_.nwb",
        "02_r1",
        include=frozenset({"zscored_ripple_consensus_trace"}),
        ripple_power=INPUTS if given else None,
    )

    trace = data["zscored_ripple_consensus_trace"]
    assert trace.index.equals(pd.Index(span, name="time"))
    if given:
        _, z = kay_ripple_power_z(
            span, _band()[1000:-1000, :3], FS, (span >= 108.0) & (span <= 109.0)
        )
        np.testing.assert_array_equal(trace.to_numpy(), z)
    else:
        assert (trace == 0.0).all()
