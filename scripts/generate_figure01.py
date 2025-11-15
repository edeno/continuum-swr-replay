import matplotlib.pyplot as plt
import numpy as np
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from non_local_detector.visualization import plot_non_local_model

(
    speed,
    position,
    spike_times,
    time,
    event_times,
    sampling_frequency,
    is_event,
    place_fields,
) = make_simulated_data(n_neurons=30)

fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=True, constrained_layout=True)
axes[0].plot(time, position, label="Position")
axes[0].set_ylabel("Position (cm)")
axes[0].fill_between(
    time, is_event * np.max(position), alpha=0.2, label="Event Periods"
)
axes[0].legend()

axes[1].plot(time, speed, label="Speed", color="orange")
axes[1].set_ylabel("Speed (cm/s)")
axes[1].fill_between(time, is_event * np.max(speed), alpha=0.2, label="Event Periods")

axes[2].eventplot(spike_times, lineoffsets=1, linelengths=0.5, color="black")
axes[2].set_ylabel("Spike Times")
axes[2].set_xlabel("Time (s)")
axes[2].set_yticks([])
axes[2].set_title("Spike Raster Plot")
axes[2].fill_between(
    time, is_event * len(spike_times), alpha=0.2, label="Event Periods"
)
axes[2].set_xlim(0, time[-1])


detector = NonLocalSortedSpikesDetector(
    sorted_spikes_algorithm="sorted_spikes_kde",
    sorted_spikes_algorithm_params={
        "position_std": 6.0,
        "block_size": int(2**12),
    },
).fit(time, position, spike_times, is_training=~is_event)

results = detector.predict(
    spike_times=spike_times,
    time=time,
    position=position,
    position_time=time,
    save_log_likelihood_to_results=True,
)

detector.plot_initial_conditions()
detector.plot_discrete_state_transition()
detector.plot_continuous_state_transition()


plot_non_local_model(
    time,
    position,
    spike_times,
    speed,
    detector,
    results,
    figsize=(20, 10),
)
