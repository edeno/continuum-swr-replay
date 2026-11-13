"""The common spike-evidence diagnostic, on a real clusterless encoding model.

The fixture fits the library's clusterless KDE model on simulated spikes over a
Y track whose bins lie at most 36 cm apart by track distance, so the tests use
a remote distance of 30 cm in place of the analysis's 100 cm.

Blocked and whole-grid likelihoods are compared at ``rtol=atol=1e-6``: the
library sums in float32, so another call shape may round a value differently
(about 1e-7 relative, a few units of 1e-5 on log-likelihoods of order 100),
while a spike put in the wrong row moves a value by a whole spike's term,
orders of magnitude more.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    clusterless_epoch_log_likelihood,
    interior_bin_distances_cm,
)
from continuum_swr_replay.analysis.detection.swr_detectors import (
    tetrode_spike_counts,
)
from continuum_swr_replay.analysis.detector_comparison.evidence import (
    animal_bins,
    epoch_row_evidence,
    interval_evidence,
    log_likelihood_blocks,
    remote_evidence,
)
from continuum_swr_replay.analysis.detector_comparison.features import (
    RowTraces,
    interval_features,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    encoding_inputs,
    nearest_interior_bins,
)
from tests.support._memoryless_clusterless_epoch import (
    BOUNDARY_SAMPLES,
    SAMPLING_FREQUENCY,
    build_simulated_epoch,
)

REMOTE_CM = 30.0
"""Remote on the fixture's Y track, whose farthest bins are 36 cm apart."""
SAMPLE_INTERVAL_S = 1.0 / SAMPLING_FREQUENCY


@pytest.fixture(scope="module")
def epoch(tmp_path_factory):
    return build_simulated_epoch(tmp_path_factory.mktemp("evidence"))


@pytest.fixture(scope="module")
def encargs(epoch):
    return encoding_inputs(epoch.encoding_model)["encargs"]


@pytest.fixture(scope="module")
def whole_grid(epoch, encargs):
    """One library call on the whole grid, its unreachable last row dropped."""
    return clusterless_epoch_log_likelihood(
        epoch.time,
        epoch.linear_position,
        epoch.spike_times,
        epoch.spike_waveform_features,
        encargs,
    ).log_likelihood


def requested(n_time, runs):
    rows = np.zeros(n_time, dtype=bool)
    for start, stop in runs:
        rows[start:stop] = True
    return rows


def assembled(epoch, encargs, rows, **kwargs):
    """The blocks of the requested rows on the grid, NaN elsewhere."""
    n_bins = int(np.asarray(encargs["environment"].is_track_interior_).sum())
    out = np.full((epoch.time.size, n_bins), np.nan)
    for first, block in log_likelihood_blocks(
        epoch.time,
        rows,
        epoch.linear_position,
        epoch.spike_times,
        epoch.spike_waveform_features,
        encargs,
        **kwargs,
    ):
        assert block.shape[1] == n_bins
        assert np.isnan(out[first : first + block.shape[0]]).all()  # no overlap
        out[first : first + block.shape[0]] = block
    return out


def a_row_beside_a_spike(epoch):
    """The sample just after a spike, the spike less than a tenth of a row before it."""
    spikes = np.concatenate(epoch.spike_times)
    following = np.searchsorted(epoch.time, spikes, side="right")
    inside = (following > 50) & (following < epoch.time.size - 50)
    lead = epoch.time[following[inside]] - spikes[inside]
    best = int(np.argmin(lead))
    assert lead[best] < 0.1 * SAMPLE_INTERVAL_S
    return int(following[inside][best])


def a_row_holding_a_spike(epoch):
    counts = tetrode_spike_counts(epoch.spike_times, epoch.time).sum(axis=1)
    return int(np.flatnonzero(counts[300:] > 0)[0]) + 300


def with_spikes(spike_times, features, tetrode, times, marks):
    """Spikes added to one tetrode, kept in time order."""
    spike_times, features = list(spike_times), list(features)
    joined_times = np.concatenate([spike_times[tetrode], times])
    joined_marks = np.concatenate([features[tetrode], marks])
    order = np.argsort(joined_times, kind="stable")
    spike_times[tetrode] = joined_times[order]
    features[tetrode] = joined_marks[order]
    return spike_times, features


def without_spikes(spike_times, features, start_s, stop_s):
    """Every spike in ``[start_s, stop_s)`` removed."""
    keep = [~((t >= start_s) & (t < stop_s)) for t in spike_times]
    return (
        [t[k] for t, k in zip(spike_times, keep, strict=True)],
        [f[k] for f, k in zip(features, keep, strict=True)],
    )


def row_evidence(epoch, encargs, rows, *, linear_position=None, spikes=None, **kw):
    spike_times, features = spikes or (
        epoch.spike_times,
        epoch.spike_waveform_features,
    )
    return epoch_row_evidence(
        epoch.time,
        rows,
        epoch.linear_position if linear_position is None else linear_position,
        spike_times,
        features,
        encargs,
        remote_cm=REMOTE_CM,
        **kw,
    )


def row_traces(epoch, spike_times):
    n_time = epoch.time.size
    return RowTraces(
        time=epoch.time,
        spike_counts=tetrode_spike_counts(spike_times, epoch.time),
        ripple_z=np.zeros(n_time),
        mua_z=np.zeros(n_time),
        linear_position=epoch.linear_position,
        track_segment_id=np.zeros(n_time, dtype=np.int64),
        patch_id=np.zeros(n_time, dtype=np.int64),
    )


def on_rows(epoch, first, last):
    """One interval whose rows are ``first`` to ``last``."""
    return pd.DataFrame(
        {"event_start_s": [epoch.time[first]], "event_end_s": [epoch.time[last]]}
    )


class TestBlocks:
    def test_the_whole_grid_in_blocks_equals_one_call(self, epoch, encargs, whole_grid):
        rows = requested(epoch.time.size, [(0, epoch.time.size - 1)])
        for block_rows in BOUNDARY_SAMPLES:  # cuts at 100, 200, ... and 140, 280, ...
            got = assembled(epoch, encargs, rows, block_rows=block_rows)
            np.testing.assert_allclose(got[:-1], whole_grid, rtol=1e-6, atol=1e-6)
            assert np.isnan(got[-1]).all()

    def test_runs_cut_at_a_spike_beside_a_spike_and_at_a_boundary(
        self, epoch, encargs, whole_grid
    ):
        cuts = [
            BOUNDARY_SAMPLES[0],  # a spike sits exactly on this sample
            BOUNDARY_SAMPLES[1],
            a_row_holding_a_spike(epoch),
            a_row_beside_a_spike(epoch),
        ]
        last = epoch.time.size - 2
        runs = [(cut, cut + 1) for cut in cuts]  # one row alone
        runs += [(cut - 5, cut) for cut in cuts]  # a run ending before it
        runs += [(cut, cut + 5) for cut in cuts]  # a run starting on it
        runs += [(0, 1), (last, last + 1)]  # the grid's first and last rows
        for run in runs:
            rows = requested(epoch.time.size, [run])
            got = assembled(epoch, encargs, rows)
            np.testing.assert_allclose(
                got[rows], whole_grid[rows[:-1]], rtol=1e-6, atol=1e-6
            )
            assert np.isnan(got[~rows]).all()

    def test_the_spike_on_a_boundary_sample_is_in_the_row_it_starts(
        self, epoch, encargs
    ):
        boundary = BOUNDARY_SAMPLES[0]
        spike_times, features = without_spikes(
            epoch.spike_times,
            epoch.spike_waveform_features,
            epoch.time[boundary - 1],
            epoch.time[boundary + 1],
        )
        silent = np.asarray(encargs["summed_ground_process_intensity"])
        # only the spike exactly on t_boundary is put back
        spike_times, features = with_spikes(
            spike_times,
            features,
            0,
            epoch.time[[boundary]],
            np.full((1, features[0].shape[1]), 100.0),
        )
        rows = requested(epoch.time.size, [(boundary - 1, boundary + 1)])
        blocks = dict(
            log_likelihood_blocks(
                epoch.time,
                rows,
                epoch.linear_position,
                spike_times,
                features,
                encargs,
                block_rows=1,
            )
        )
        np.testing.assert_allclose(blocks[boundary - 1][0], -silent, rtol=1e-6)
        assert not np.allclose(blocks[boundary][0], -silent)

    def test_every_call_has_one_time_shape(self, epoch, encargs, monkeypatch):
        import non_local_detector.likelihoods.clusterless_kde as library

        lengths = []
        call = library.predict_clusterless_kde_log_likelihood

        def counted(time, *args, **kwargs):
            lengths.append(len(time))
            return call(time, *args, **kwargs)

        monkeypatch.setattr(library, "predict_clusterless_kde_log_likelihood", counted)
        # the first, an interior and the last block, the last moved inward
        rows = requested(epoch.time.size, [(5, 20), (250, 260), (590, 599)])
        assembled(epoch, encargs, rows, block_rows=100)
        assert lengths == [103, 103, 103]

    def test_the_last_sample_is_refused(self, epoch, encargs):
        rows = requested(epoch.time.size, [(epoch.time.size - 3, epoch.time.size)])
        with pytest.raises(ValueError, match="last sample"):
            list(
                log_likelihood_blocks(
                    epoch.time,
                    rows,
                    epoch.linear_position,
                    epoch.spike_times,
                    epoch.spike_waveform_features,
                    encargs,
                )
            )


class TestIntervalScores:
    def test_independent_of_blocking_and_of_other_intervals(self, epoch, encargs):
        traces = row_traces(epoch, epoch.spike_times)
        target = on_rows(epoch, 210, 230)
        alone = row_evidence(epoch, encargs, requested(epoch.time.size, [(210, 231)]))
        among_others = row_evidence(
            epoch,
            encargs,
            requested(epoch.time.size, [(30, 90), (200, 400), (500, 560)]),
            block_rows=7,
        )
        a = interval_evidence(traces, alone, target)
        b = interval_evidence(traces, among_others, target)
        np.testing.assert_allclose(
            a["evidence_dll"], b["evidence_dll"], rtol=1e-6, atol=1e-6
        )
        assert a["evidence_n_valid_rows"].iloc[0] == 21

        others = pd.concat([target, on_rows(epoch, 100, 140), on_rows(epoch, 220, 380)])
        c = interval_evidence(traces, among_others, others)
        assert c["evidence_dll"].iloc[0] == b["evidence_dll"].iloc[0]

    def test_an_interval_across_a_block_edge(self, epoch, encargs):
        traces = row_traces(epoch, epoch.spike_times)
        target = on_rows(epoch, 90, 110)  # blocks of 100 rows cut it at row 100
        alone = row_evidence(epoch, encargs, requested(epoch.time.size, [(90, 111)]))
        across = row_evidence(
            epoch,
            encargs,
            requested(epoch.time.size, [(90, 111), (450, 470)]),
            block_rows=100,
        )
        np.testing.assert_allclose(
            interval_evidence(traces, across, target)["evidence_dll"],
            interval_evidence(traces, alone, target)["evidence_dll"],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(across[90:111], alone[90:111], rtol=1e-6, atol=1e-6)

    def test_mean_over_the_rows(self, epoch, encargs):
        evidence = row_evidence(epoch, encargs, requested(epoch.time.size, [(0, 599)]))
        traces = row_traces(epoch, epoch.spike_times)
        got = interval_evidence(traces, evidence, on_rows(epoch, 300, 309))
        valid = np.isfinite(evidence[300:310])
        assert 0 < valid.sum() < 10  # near the junction no bin is remote
        assert got["evidence_dll"].iloc[0] == pytest.approx(
            evidence[300:310][valid].mean()
        )
        assert got["evidence_valid_fraction"].iloc[0] == valid.mean()
        assert got["evidence_unscorable"].iloc[0] == ""


class TestSign:
    @pytest.fixture(scope="class")
    def burst(self, epoch, encargs):
        """Five tetrode-1 encoding marks alone in row 300, and their field's bin."""
        row = 300
        spike_times, features = without_spikes(
            epoch.spike_times,
            epoch.spike_waveform_features,
            epoch.time[row],
            epoch.time[row + 1],
        )
        marks = np.asarray(epoch.encoding_model["encoding_spike_waveform_features"][1])
        times = epoch.time[row] + np.linspace(0.1, 0.9, 5) * SAMPLE_INTERVAL_S
        spikes = with_spikes(spike_times, features, 1, times, marks[:5])
        field = np.median(
            np.asarray(epoch.encoding_model["encoding_positions"][1]).ravel()
        )
        field_bin = int(nearest_interior_bins(encargs["environment"], np.r_[field])[0])
        return row, spikes, field_bin

    def _at(self, epoch, encargs, burst, animal_bin):
        row, spikes, _ = burst
        centers = np.asarray(encargs["environment"].place_bin_centers_).ravel()[
            np.asarray(encargs["environment"].is_track_interior_).ravel()
        ]
        position = epoch.linear_position.copy()
        position[row] = centers[animal_bin]
        rows = requested(epoch.time.size, [(row, row + 1)])
        return row_evidence(
            epoch, encargs, rows, linear_position=position, spikes=spikes
        )[row]

    def test_a_remote_burst_scores_above_zero(self, epoch, encargs, burst):
        distance = interior_bin_distances_cm(encargs["environment"])
        far_bin = int(np.argmax(distance[burst[2]]))
        assert distance[burst[2], far_bin] > REMOTE_CM
        assert self._at(epoch, encargs, burst, far_bin) > 0.0

    def test_a_burst_at_the_animal_scores_below_zero(self, epoch, encargs, burst):
        assert self._at(epoch, encargs, burst, burst[2]) < 0.0

    def test_a_silent_interval_is_scored_with_zero_spikes(self, epoch, encargs):
        first, last = 400, 409
        spike_times, features = without_spikes(
            epoch.spike_times,
            epoch.spike_waveform_features,
            epoch.time[first],
            epoch.time[last + 1],
        )
        rows = requested(epoch.time.size, [(first, last + 1)])
        evidence = row_evidence(epoch, encargs, rows, spikes=(spike_times, features))
        traces = row_traces(epoch, spike_times)
        interval = on_rows(epoch, first, last)
        scored = interval_evidence(traces, evidence, interval)
        assert scored["evidence_unscorable"].iloc[0] == ""
        assert np.isfinite(scored["evidence_dll"].iloc[0])
        assert (
            interval_features(traces, interval, sample_interval_s=SAMPLE_INTERVAL_S)[
                "n_spikes"
            ].iloc[0]
            == 0
        )
        # silence: every row is the ground term alone
        env = encargs["environment"]
        ground = -np.asarray(encargs["summed_ground_process_intensity"])
        expected = remote_evidence(
            np.tile(ground, (last + 1 - first, 1)),
            animal_bins(env, epoch.linear_position[first : last + 1]),
            interior_bin_distances_cm(env) > REMOTE_CM,
            np.ones(ground.size, dtype=bool),
        )
        np.testing.assert_allclose(evidence[first : last + 1], expected, rtol=1e-6)


class TestRemoteEvidence:
    remote = np.array(
        [
            [False, False, True, True],
            [False, False, False, True],
            [True, False, False, False],
            [True, True, False, False],
        ]
    )
    L = np.array([[0.0, 1.0, 5.0, 3.0]])

    def test_max_over_remote_less_the_animal(self):
        got = remote_evidence(self.L, np.array([0]), self.remote, np.ones(4, bool))
        assert got[0] == 5.0

    def test_a_flagged_bin_never_wins(self):
        usable = np.array([True, True, False, True])
        got = remote_evidence(self.L, np.array([0]), self.remote, usable)
        assert got[0] == 3.0

    def test_invalid_rows_are_nan(self):
        usable = np.array([True, True, True, False])
        L = np.tile(self.L, (3, 1))
        got = remote_evidence(L, np.array([-1, 3, 1]), self.remote, usable)
        # no position; the animal's bin flagged; its only remote bin flagged
        assert np.isnan(got).all()


class TestValidity:
    FLAGGED = 0

    @pytest.fixture(scope="class")
    def flagged(self, encargs):
        """The encoding with bin 0's occupancy subnormal-sized, so it is flagged."""
        occupancy = np.array(encargs["occupancy"], copy=True)
        occupancy[self.FLAGGED] = 1e-30
        return {**encargs, "occupancy": occupancy}

    def test_a_flagged_bin_never_wins_the_remote_maximum(self, epoch, flagged):
        env = flagged["environment"]
        whole = clusterless_epoch_log_likelihood(
            epoch.time,
            epoch.linear_position,
            epoch.spike_times,
            epoch.spike_waveform_features,
            flagged,
        ).log_likelihood
        distance = interior_bin_distances_cm(env)
        far = np.flatnonzero(distance[self.FLAGGED] > REMOTE_CM)[0]
        centers = np.asarray(env.place_bin_centers_).ravel()[
            np.asarray(env.is_track_interior_).ravel()
        ]
        position = np.full(epoch.time.size, centers[far])
        rows = requested(epoch.time.size, [(250, 350)])
        evidence = row_evidence(epoch, flagged, rows, linear_position=position)
        remote = distance[far] > REMOTE_CM
        assert remote[self.FLAGGED]
        # the flagged bin, divided by a tiny occupancy, is the largest somewhere
        assert (whole[250:350].argmax(axis=1) == self.FLAGGED).any()
        others = remote.copy()
        others[self.FLAGGED] = False
        expected = whole[250:350][:, others].max(axis=1) - whole[250:350, far]
        np.testing.assert_allclose(evidence[250:350], expected, rtol=1e-6)

    def test_invalid_rows_and_coverage(self, epoch, flagged):
        env = flagged["environment"]
        centers = np.asarray(env.place_bin_centers_).ravel()[
            np.asarray(env.is_track_interior_).ravel()
        ]
        position = np.full(epoch.time.size, 46.0)
        position[301] = np.nan
        position[303] = centers[self.FLAGGED]
        rows = requested(epoch.time.size, [(300, 305)])
        evidence = row_evidence(epoch, flagged, rows, linear_position=position)
        assert np.isnan(evidence[[301, 303]]).all()
        assert np.isfinite(evidence[[300, 302, 304]]).all()

        traces = row_traces(epoch, epoch.spike_times)
        intervals = pd.concat(
            [
                on_rows(epoch, 300, 304),
                on_rows(epoch, 301, 301),
                on_rows(epoch, 303, 303),
                pd.DataFrame(  # between two samples: no row
                    {
                        "event_start_s": [epoch.time[302] + 0.0002],
                        "event_end_s": [epoch.time[302] + 0.0008],
                    }
                ),
            ],
            ignore_index=True,
        )
        got = interval_evidence(traces, evidence, intervals)
        assert got["evidence_n_valid_rows"].iloc[0] == 3
        assert got["evidence_valid_fraction"].iloc[0] == pytest.approx(0.6)
        assert got["evidence_dll"].iloc[0] == pytest.approx(
            evidence[[300, 302, 304]].mean()
        )
        assert list(got["evidence_unscorable"]) == [
            "",
            "no_valid_row",
            "no_valid_row",
            "no_row",
        ]
        assert np.isnan(got["evidence_dll"].iloc[1:]).all()
        assert got["evidence_n_valid_rows"].iloc[1] == 0
        assert pd.isna(got["evidence_n_valid_rows"].iloc[3])
        assert np.isnan(got["evidence_valid_fraction"].iloc[3])

    def test_an_interval_ending_on_the_last_sample_is_scored_without_it(
        self, epoch, encargs
    ):
        last = epoch.time.size - 1
        traces = row_traces(epoch, epoch.spike_times)
        evidence = row_evidence(
            epoch, encargs, requested(epoch.time.size, [(580, last)])
        )
        got = interval_evidence(
            traces,
            evidence,
            pd.concat(
                [
                    on_rows(epoch, 580, last),
                    on_rows(epoch, 580, last - 1),
                    on_rows(epoch, last, last),
                ],
                ignore_index=True,
            ),
        )
        pd.testing.assert_frame_equal(
            got.iloc[[0]].reset_index(drop=True),
            got.iloc[[1]].reset_index(drop=True),
            check_exact=True,
        )
        n_valid = np.isfinite(evidence[580:last]).sum()
        assert n_valid > 0
        assert got["evidence_n_valid_rows"].iloc[0] == n_valid
        assert got["evidence_valid_fraction"].iloc[0] == n_valid / (last - 580)
        assert got["evidence_unscorable"].iloc[2] == "no_row"
        assert pd.isna(got["evidence_n_valid_rows"].iloc[2])

    def test_animal_bins_mark_a_missing_position(self, epoch, encargs):
        bins = animal_bins(encargs["environment"], np.array([2.0, np.nan, 74.0]))
        assert list(bins) == [0, -1, 14]


def test_junction_placeholder_bins_never_enter(epoch, encargs):
    env = copy.copy(encargs["environment"])
    nodes = env.place_bin_centers_nodes_df_.copy()
    placeholder = np.flatnonzero(nodes["edge_id"].to_numpy() == -1)[0]
    nodes.iloc[placeholder, nodes.columns.get_loc("edge_id")] = 0
    env.place_bin_centers_nodes_df_ = nodes
    with pytest.raises(ValueError, match="junction placeholder"):
        row_evidence(
            epoch,
            {**encargs, "environment": env},
            requested(epoch.time.size, [(300, 310)]),
        )
