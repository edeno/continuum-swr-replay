"""The memoryless detector end to end, on ``non_local_detector``'s simulated replays.

The package simulates an animal running a linear track and three kinds of
replay event: a continuous trajectory, a stationary hover, and a fragmented
jump sequence. Only the first is what this detector is built to find, so the
three together check that the pipeline both fires and declines to fire for the
stated reasons.

Nothing here is decoded from our own recordings. The encoding model is fitted
on the simulated run, and each replay is spliced into an otherwise silent
epoch, so a detection can only come from the replay's own spikes.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
)
from non_local_detector.simulate.clusterless_simulation import (
    make_continuous_replay,
    make_fragmented_replay,
    make_hover_replay,
    make_simulated_run_data,
)
from non_local_detector.simulate.simulate import simulate_position

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    CandidateSequences,
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.epoch import (
    run_epoch,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.event_table import (
    SequenceIdentity,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    clusterless_epoch_log_likelihood,
    likelihood_grid,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    ShuffleSeed,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    aggregate_windows,
    normalize_windows,
)
from continuum_swr_replay.decoding.encoding_model import interior_bin_distances_cm

SAMPLING_FREQUENCY = 1000
TRACK_HEIGHT_CM = 175
RUNNING_SPEED_CM_PER_S = 15
N_ELECTRODES = 5
EPOCH_S = 1.6
REPLAY_START_S = 0.4
CONFIG = MemorylessSequenceConfig(decode_sampling_frequency=float(SAMPLING_FREQUENCY))

pytestmark = pytest.mark.integration

SLOW_SPEEDUP = 25.0
"""375 cm/s: below the speed the spread criterion can follow."""

FAST_SPEEDUP = 45.0
"""675 cm/s: above it."""


def spread_limited_speed_cm_per_s(config: MemorylessSequenceConfig) -> float:
    """The fastest trajectory whose window posterior can stay narrow enough.

    A window covering a stretch of track of length ``L`` at a steady speed has
    a posterior spread of about ``L / sqrt(12)``, the standard deviation of a
    uniform distribution. Requiring that to stay under the spread threshold
    caps how far the trajectory may travel inside one window.
    """
    return config.spread_threshold_cm * np.sqrt(12.0) / config.window_duration_s


@dataclass
class Simulation:
    """A fitted encoding model and the grid the detector reads it on."""

    encoding_model: dict
    place_bin_centers_cm: np.ndarray
    bin_distance_cm: np.ndarray


@dataclass
class Detection:
    """What the detector returned for one replay, and what was put in."""

    candidates: CandidateSequences
    midpoint_s: np.ndarray
    replay_time_s: np.ndarray
    true_position_cm: np.ndarray | None


@pytest.fixture(scope="module")
def simulation() -> Simulation:
    run = make_simulated_run_data(sampling_frequency=SAMPLING_FREQUENCY)
    environment = Environment(place_bin_size=CONFIG.place_bin_size_cm).fit_place_grid(
        run.position
    )

    encoding_model = fit_clusterless_kde_encoding_model(
        run.position_time,
        run.position,
        run.spike_times,
        run.spike_waveform_features,
        environment,
        sampling_frequency=SAMPLING_FREQUENCY,
        disable_progress_bar=True,
    )
    # electrodes may differ in feature count, so the library expands a scalar
    encoding_model["waveform_std"] = float(
        np.asarray(encoding_model["waveform_std"]).ravel()[0]
    )
    interior = np.asarray(environment.is_track_interior_).ravel()

    return Simulation(
        encoding_model=encoding_model,
        place_bin_centers_cm=np.asarray(environment.place_bin_centers_).ravel()[
            interior
        ],
        bin_distance_cm=interior_bin_distances_cm(environment),
    )


def as_spike_lists(
    replay_time_s: np.ndarray, multiunits: np.ndarray, start_s: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Turn the simulator's NaN-padded array into per-electrode spike lists."""
    has_spike = np.any(~np.isnan(multiunits), axis=1)
    return (
        [replay_time_s[has_spike[:, e]] + start_s for e in range(N_ELECTRODES)],
        [multiunits[has_spike[:, e], :, e] for e in range(N_ELECTRODES)],
    )


def repeat_pattern(
    replay_time_s: np.ndarray, multiunits: np.ndarray, n_repeats: int
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat a short pattern so it outlasts the duration criterion.

    The hover and fragmented events the package makes are 50 ms and 10 ms, both
    shorter than a sequence may be. Repeating them keeps what each one is --
    one place held, or the same jumps taken again -- while making the duration
    criterion stop deciding the outcome on its own.
    """
    step_s = replay_time_s[-1] - replay_time_s[0] + np.diff(replay_time_s)[0]
    return (
        np.concatenate([replay_time_s + i * step_s for i in range(n_repeats)]),
        np.concatenate([multiunits] * n_repeats, axis=0),
    )


def detect(
    simulation: Simulation,
    replay_time_s: np.ndarray,
    multiunits: np.ndarray,
    true_position_cm: np.ndarray | None = None,
) -> Detection:
    """Splice one replay into a silent epoch and run the detector over it."""
    spike_times, features = as_spike_lists(replay_time_s, multiunits, REPLAY_START_S)
    time = np.arange(round(EPOCH_S * SAMPLING_FREQUENCY)) / SAMPLING_FREQUENCY

    result = clusterless_epoch_log_likelihood(
        time,
        np.full((time.size, 1), TRACK_HEIGHT_CM / 2.0),  # the animal is parked
        spike_times,
        features,
        simulation.encoding_model,
        CONFIG,
    )
    posterior = normalize_windows(
        aggregate_windows(
            result.log_likelihood, CONFIG.window_n_samples, CONFIG.stride_n_samples
        )
    )
    midpoint_s = window_midpoints_s(result.time, CONFIG)

    return Detection(
        candidates=detect_candidate_sequences(
            posterior,
            midpoint_s,
            simulation.bin_distance_cm,
            np.zeros(posterior.shape[0]),  # the animal is still throughout
            CONFIG,
        ),
        midpoint_s=midpoint_s,
        replay_time_s=replay_time_s,
        true_position_cm=true_position_cm,
    )


def continuous(speedup: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A continuous replay and the trajectory the simulator built it from."""
    replay_time_s, multiunits = make_continuous_replay(
        sampling_frequency=SAMPLING_FREQUENCY, replay_speedup=speedup
    )
    true_position_cm = simulate_position(
        replay_time_s, TRACK_HEIGHT_CM, RUNNING_SPEED_CM_PER_S * speedup
    )
    return replay_time_s, multiunits, true_position_cm


@pytest.fixture(scope="module")
def slow_continuous(simulation) -> Detection:
    return detect(simulation, *continuous(SLOW_SPEEDUP))


@pytest.fixture(scope="module")
def hover(simulation) -> Detection:
    return detect(
        simulation,
        *repeat_pattern(*make_hover_replay(sampling_frequency=SAMPLING_FREQUENCY), 4),
    )


@pytest.fixture(scope="module")
def fragmented(simulation) -> Detection:
    return detect(
        simulation,
        *repeat_pattern(
            *make_fragmented_replay(sampling_frequency=SAMPLING_FREQUENCY), 20
        ),
    )


class TestContinuousReplay:
    """The event the detector is built to find."""

    def test_it_is_found_as_one_sequence(self, slow_continuous):
        assert slow_continuous.candidates.runs.shape == (1, 2)

    def test_it_meets_both_replay_criteria(self, slow_continuous):
        candidates = slow_continuous.candidates

        assert candidates.is_replay.tolist() == [True]
        assert candidates.duration_s[0] > CONFIG.minimum_duration_s
        assert candidates.dispersion_cm[0] > CONFIG.dispersion_threshold_cm

    def test_it_covers_the_replay_it_was_given(self, slow_continuous):
        """The event is timestamped by window midpoints, so it may begin up to
        half a window before the first replay spike."""
        candidates = slow_continuous.candidates
        start, stop = candidates.runs[0]
        replay_end_s = REPLAY_START_S + slow_continuous.replay_time_s[-1]

        assert (
            REPLAY_START_S - CONFIG.window_duration_s
            < slow_continuous.midpoint_s[start]
            <= REPLAY_START_S
        )
        # bounded from both sides: it must not run on into the silence after
        # the replay, where the ground process still gives every window a
        # center at one place
        assert (
            replay_end_s - 0.050
            < slow_continuous.midpoint_s[stop]
            <= replay_end_s + CONFIG.window_duration_s
        )

    def test_the_decoded_path_follows_the_simulated_one(
        self, simulation, slow_continuous
    ):
        """A window under a uniform prior estimates the mean position over its
        own support, not the position at its midpoint, so that is what the
        decoded center is compared against."""
        candidates = slow_continuous.candidates
        start, stop = candidates.runs[0]
        kept = np.flatnonzero(candidates.decisions.gates.kept[start : stop + 1]) + start
        decoded_cm = simulation.place_bin_centers_cm[
            candidates.decisions.center_index[kept]
        ]

        window_start_s = (
            slow_continuous.midpoint_s[kept]
            - 0.5 * CONFIG.window_duration_s
            - REPLAY_START_S
        )
        replay_time_s = slow_continuous.replay_time_s
        window_mean_cm = np.array(
            [
                np.mean(
                    np.interp(
                        np.clip(
                            np.linspace(
                                s, s + CONFIG.window_duration_s, CONFIG.window_n_samples
                            ),
                            replay_time_s[0],
                            replay_time_s[-1],
                        ),
                        replay_time_s,
                        slow_continuous.true_position_cm,
                    )
                )
                for s in window_start_s
            ]
        )

        assert np.corrcoef(decoded_cm, window_mean_cm)[0, 1] > 0.99
        assert np.median(np.abs(decoded_cm - window_mean_cm)) < 10.0

    def test_the_decoded_path_crosses_most_of_the_track(
        self, simulation, slow_continuous
    ):
        candidates = slow_continuous.candidates
        start, stop = candidates.runs[0]
        kept = np.flatnonzero(candidates.decisions.gates.kept[start : stop + 1]) + start
        decoded_cm = simulation.place_bin_centers_cm[
            candidates.decisions.center_index[kept]
        ]

        assert np.ptp(decoded_cm) > 0.75 * TRACK_HEIGHT_CM


class TestHoverReplay:
    """One place held. Long enough to be a sequence, too still to be replay."""

    def test_it_is_assembled_as_a_sequence(self, hover):
        assert hover.candidates.runs.shape == (1, 2)
        assert hover.candidates.duration_s[0] > CONFIG.minimum_duration_s

    def test_it_is_rejected_for_going_nowhere(self, hover):
        assert hover.candidates.dispersion_cm[0] < CONFIG.dispersion_threshold_cm
        assert hover.candidates.is_replay.tolist() == [False]

    def test_its_centers_do_not_move(self, hover):
        np.testing.assert_allclose(hover.candidates.dispersion_cm, 0.0, atol=1e-9)


class TestFragmentedReplay:
    """Scattered positions, which an 80 ms window averages into one place."""

    def test_it_is_rejected(self, fragmented):
        assert not fragmented.candidates.is_replay.any()

    def test_the_window_average_makes_it_look_stationary(self, fragmented):
        """Not rejected on jumps: the window is wider than the jumps are fast,
        so its center barely moves. A detector reading this cohort must not be
        described as rejecting fragmented events for jumping too far."""
        candidates = fragmented.candidates

        assert candidates.decisions.gates.jump_below_threshold.mean() > 0.9
        assert candidates.dispersion_cm.max() < CONFIG.dispersion_threshold_cm


class TestReplaySpeedLimit:
    """The spread criterion caps how fast a trajectory may be and still be found."""

    def test_the_slow_replay_is_below_the_limit_and_is_found(self, slow_continuous):
        speed_cm_per_s = RUNNING_SPEED_CM_PER_S * SLOW_SPEEDUP

        assert speed_cm_per_s < spread_limited_speed_cm_per_s(CONFIG)
        assert slow_continuous.candidates.is_replay.tolist() == [True]

    def test_a_replay_above_the_limit_is_not_recovered_whole(self, simulation):
        """It is still decoded; it just no longer survives as one sequence."""
        speed_cm_per_s = RUNNING_SPEED_CM_PER_S * FAST_SPEEDUP
        assert speed_cm_per_s > spread_limited_speed_cm_per_s(CONFIG)

        found = detect(simulation, *continuous(FAST_SPEEDUP)).candidates

        assert not found.is_replay.any()
        assert found.decisions.gates.posterior_usable.all()
        assert found.decisions.gates.spread_below_threshold.any()


class TestSilence:
    @pytest.fixture(scope="class")
    def silent(self, simulation):
        return detect(
            simulation,
            np.arange(10) / SAMPLING_FREQUENCY,
            np.full((10, 4, N_ELECTRODES), np.nan),
        ).candidates

    def test_an_epoch_with_no_spikes_yields_nothing(self, silent):
        assert silent.runs.shape == (0, 2)
        assert not silent.decisions.gates.kept.any()

    def test_silence_is_rejected_by_the_spread_gate_alone(self, silent):
        """Every silent window has a center, all at one place, because the
        ground process gives silence a spatially varying likelihood. Only the
        spread gate stands between that and a stationary run; a more peaked
        ground process could pass it. Recorded so the mechanism is on file."""
        gates = silent.decisions.gates

        assert gates.has_center.all()
        assert not gates.spread_below_threshold.any()


class TestWholeEpoch:
    """``run_epoch`` strings every step together, the shuffle included.

    Its table must be the detector's output, row for row, and the shuffle must
    run on exactly the candidates the published definition would test.
    """

    N_SHUFFLES = 40
    IDENTITY = SequenceIdentity(
        animal="sim",
        day=20200101,
        epoch="02_r1",
        brain_area="HPC",
        epoch_id="sim20200101_02_r1_HPC",
        run_id="integration",
    )
    SEED = ShuffleSeed(run=1, session="sim20200101_02_r1_HPC")

    def run(self, simulation, seed, speed=None):
        replay_time_s, multiunits, _ = continuous(SLOW_SPEEDUP)
        spike_times, features = as_spike_lists(
            replay_time_s, multiunits, REPLAY_START_S
        )
        time = np.arange(round(EPOCH_S * SAMPLING_FREQUENCY)) / SAMPLING_FREQUENCY
        speed = np.zeros(time.size) if speed is None else speed
        return run_epoch(
            time,
            np.full(time.size, TRACK_HEIGHT_CM / 2.0),  # parked, as in detect()
            speed,  # still throughout, unless a test says otherwise
            spike_times,
            features,
            simulation.encoding_model,
            self.IDENTITY,
            swr_intervals=None,
            swr_observed_intervals=None,
            swr_definition="none",
            seed=seed,
            config=CONFIG,
            n_shuffles=self.N_SHUFFLES,
        )

    @pytest.fixture(scope="class")
    def epoch_run(self, simulation):
        return self.run(simulation, self.SEED)

    def test_the_table_is_the_detector_output(self, epoch_run, simulation):
        direct = detect(simulation, *continuous(SLOW_SPEEDUP)).candidates

        np.testing.assert_array_equal(epoch_run.candidates.runs, direct.runs)
        np.testing.assert_array_equal(epoch_run.table.is_replay, direct.is_replay)
        np.testing.assert_allclose(epoch_run.table.dispersion_cm, direct.dispersion_cm)

    def test_every_replay_of_a_still_animal_is_shuffled(self, epoch_run):
        table = epoch_run.table
        tested = table[table.n_shuffles > 0]

        assert table.is_replay.any()
        np.testing.assert_array_equal(table.n_shuffles > 0, table.is_replay)
        assert (tested.n_shuffles == self.N_SHUFFLES).all()

    def test_every_draw_is_kept_against_its_event(self, epoch_run):
        tested = epoch_run.table[epoch_run.table.n_shuffles > 0]
        draws = epoch_run.shuffle_draws

        assert len(draws) == self.N_SHUFFLES * len(tested)
        assert set(draws.ms_event_id) == set(tested.ms_event_id)

    def test_the_spread_half_separates_and_the_jump_half_cannot(
        self, epoch_run, simulation
    ):
        """A property of the encoding-pair shuffle on this simulation, on file
        for the real-data pilot.

        The simulation gives each electrode one stretch of track. A shuffle
        within an electrode keeps that stretch, so a shuffled decode still
        knows roughly where the replay is and its center still sweeps the
        track; what it loses is which mark goes with which place, so the
        spread half separates. The replay's own jumps are no smaller than the
        shuffled sweeps' (the observed mean sits above every draw), so the
        jump half fails and the candidate does not pass. CA1 tetrodes are not
        topographic and should keep less position in their identity than
        these electrodes do; how much less has to be measured.
        """
        footprints = [
            (float(np.min(positions)), float(np.max(positions)))
            for positions in simulation.encoding_model["encoding_positions"]
        ]
        replay = epoch_run.table[epoch_run.table.is_replay].iloc[0]

        # each electrode covers well under half of the 175 cm track
        assert all(high - low < TRACK_HEIGHT_CM / 2 for low, high in footprints)
        assert replay.shuffle_testable
        assert replay.p_spread <= 0.05
        assert replay.p_jump > 0.05
        assert not replay.shuffle_pass

    def test_a_replay_the_animal_moved_during_is_tested_but_not_eligible(
        self, simulation
    ):
        """One fast speed sample inside the replay, off every window midpoint:
        each window's gate reads its midpoint and passes, and the whole-event
        rule, which reads every sample, must still catch it. The shuffle
        still runs, as the reference would test it; eligibility is a column."""
        time = np.arange(round(EPOCH_S * SAMPLING_FREQUENCY)) / SAMPLING_FREQUENCY
        speed = np.zeros(time.size)
        # the windows sit on the detector's grid, derived from this decode grid
        midpoints = window_midpoints_s(likelihood_grid(time, CONFIG)[:-1], CONFIG)
        inside = round(0.5 * SAMPLING_FREQUENCY) + 1  # the replay runs from 0.4 s
        assert time[inside] not in set(midpoints.tolist())
        speed[inside] = 9.0

        moved = self.run(simulation, self.SEED, speed=speed)

        replay = moved.table[moved.table.is_replay]
        assert len(replay) == 1
        assert not replay.whole_event_speed_eligible.iloc[0]
        assert replay.n_shuffles.iloc[0] == self.N_SHUFFLES
        assert not moved.shuffle_draws.empty

    def test_no_seed_runs_no_shuffle(self, simulation):
        unshuffled = self.run(simulation, seed=None)

        assert (unshuffled.table.n_shuffles == 0).all()
        assert unshuffled.shuffle_draws.empty
