"""The detection summary: the held-out tetrode test of each decoder state, and the
detector against the Widloski & Foster detector on simulated replays.

Every value is read from an output an analysis already wrote; nothing is
refitted or re-decoded here. This module names and describes the values. A
statistic an input already holds is copied from it; the statistics of the sweeps'
summaries, of the labeled recording and of the pilot's events are computed in
``analysis.detection.memoryless_sequence`` (``sweep_statistics``,
``recording_statistics`` and ``event_table``); here only the size of an input
(its epochs, its animals) is counted. The inputs (:class:`DetectionInputs`):

- the held-out tetrode test per decoder state
  (``Processed-Data/_refresh_20261016/state_evidence``, from
  ``scripts/event_evidence/run_state_evidence.py``): its pooled ``bout_stats.csv``
  and the per-epoch status in ``manifest.json``;
- the three simulation sweeps of the two replay detectors
  (``paper/reported/inputs/detection/detection_{sweep,controls,heldout}_summary.csv``,
  from ``scripts/detection/sweep_memoryless_{detection,controls}.py`` and
  ``scripts/detector_comparison/sweep_memoryless_heldout.py``),
  for the memoryless-comparison caption's panels (g-q);
- the one labeled recording of that caption's panels (a-f)
  (``paper/reported/inputs/detection/memoryless_comparison_recording.json``:
  ``memoryless_sequence.recording_statistics.recording_numbers``, written by
  ``scripts/figures/generate_memoryless_comparison.py``);
- the pilot epoch of the main memoryless run (:data:`MEMORYLESS_PILOT_RUN`, a
  ``replay_detection_comparison/memoryless`` run, from
  ``scripts/detection/run_memoryless_detection.py``), for the caption's
  stray-window note.

The caption is prose with markers; these values are its only numbers. Where
it states a fact rather than a number (one value standing for several folds,
no event of a kind), the builder refuses inputs that contradict it; the figure
script checks the recording's verdicts the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    sweep_statistics,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.event_table import (
    outside_window_call_statistics,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep_controls import (
    MULTIPART_CONDITIONS,
)
from continuum_swr_replay.analysis.detector_comparison.heldout_sweep import (
    FOLDS,
    OUR_EVENT_CLASSES,
    TILING_FOLDS,
)
from continuum_swr_replay.paths import (
    ROOT_DIR,
    STATE_EVIDENCE_DIR,
    STATE_EVIDENCE_RUN,
    memoryless_root,
    processed_data_dir,
)
from continuum_swr_replay.reported.formatting import format_number, milliseconds
from continuum_swr_replay.reported.values import (
    Summary,
    Tags,
    Value,
    file_record,
    rel,
    value_factory,
)
from continuum_swr_replay.runs.io import read_manifest

MEMORYLESS_PILOT_RUN = "e2331f976fd1-seed20260922"
"""The memoryless run whose pilot epoch (chimi20200225_02_r1, the epoch of the
detector's first real-data pilot) the caption's stray-window note reads: the main
cohort run, at the 5 ms step on its own running-only encoding model. The note
first read the October pilot, ``0c2548cacf2d-seed20260922``."""

MEMORYLESS_PILOT_EPOCH = "chimi20200225_02_r1_HPC"

SWEEP_DIR = Path("paper/reported/inputs/detection")

TAGS = {
    "state_evidence": Tags(("2c.3",)),
    "memoryless": Tags(("1.1",), ("memoryless_comparison",)),
    "pilot": Tags(("2a.3",), ("memoryless_comparison",)),
}
"""Each group of values: the questions it answers and the figures that show it.
The memoryless-comparison values are used in that supplementary figure and its
caption; the pilot values come from one real epoch, not the simulations."""


@dataclass(frozen=True)
class DetectionInputs:
    """The files the detection summary reads."""

    state_evidence_dir: Path
    sweep_summary: Path
    controls_summary: Path
    heldout_summary: Path
    recording_numbers: Path
    pilot_events: Path
    pilot_manifest: Path


def default_inputs() -> DetectionInputs:
    pilot = memoryless_root() / MEMORYLESS_PILOT_RUN
    return DetectionInputs(
        state_evidence_dir=processed_data_dir()
        / STATE_EVIDENCE_RUN
        / STATE_EVIDENCE_DIR,
        sweep_summary=ROOT_DIR / SWEEP_DIR / "detection_sweep_summary.csv",
        controls_summary=ROOT_DIR / SWEEP_DIR / "detection_controls_summary.csv",
        heldout_summary=ROOT_DIR / SWEEP_DIR / "detection_heldout_summary.csv",
        recording_numbers=ROOT_DIR / SWEEP_DIR / "memoryless_comparison_recording.json",
        pilot_events=pilot / f"{MEMORYLESS_PILOT_EPOCH}_memoryless_events.parquet",
        pilot_manifest=pilot / "manifest.json",
    )


# --- the held-out tetrode test per decoder state ------------------------------

#: (word, state, regime, metric, what the evidence compares)
_STATE_TESTS = (
    (
        "local_immobile",
        "Local",
        "immobile",
        "runcv_loc_mark",
        "the animal's position against a mismatched one",
    ),
    (
        "no_spike_immobile",
        "No-Spike",
        "immobile",
        "runcv_ns_logratio",
        "silence against local coding",
    ),
    (
        "continuous_immobile",
        "Non-Local Continuous",
        "immobile",
        "runcv_nl_anim_mark",
        "the held-out MAP against the animal's position",
    ),
    (
        "continuous_running",
        "Non-Local Continuous",
        "running",
        "runcv_nl_anim_mark",
        "the held-out MAP against the animal's position",
    ),
    (
        "fragmented_immobile_mismatched",
        "Non-Local Fragmented",
        "immobile",
        "runcv_nl_null_mark",
        "the held-out MAP against a mismatched MAP",
    ),
)


def _k_of(text: str) -> tuple[int, int]:
    """``"3/5"`` as ``(3, 5)``."""
    k, n = (int(x) for x in text.split("/"))
    return k, n


def state_evidence_values(
    bouts: pd.DataFrame, manifest: dict, source: str
) -> list[Value]:
    """Held-out tetrodes support each decoder state.

    Half the tetrodes choose the location, the other half score it; each
    animal's value is the mean of its epochs' medians, and "animals positive"
    counts the animals whose mean is above zero. Bouts are the primary
    (``runcv_``, time-held-out running-only encoding) rows of ``bout_stats``.
    """
    make = value_factory(state_evidence_values, source, TAGS["state_evidence"])
    statuses = pd.Series({k: r["status"] for k, r in manifest["epochs"].items()})
    n_animals = int(bouts["n_animals"].max())
    values = [
        make(
            "state_evidence_epochs",
            len(statuses),
            "count",
            "count",
            "epochs the held-out tetrode run lists",
        ),
        make(
            "state_evidence_epochs_ok",
            (statuses == "ok").sum(),
            "count",
            "count",
            "of those, epochs that passed every check of the run (status ok)",
        ),
        make(
            "state_evidence_animals",
            n_animals,
            "count",
            "count",
            "animals in the held-out tetrode run",
        ),
    ]
    for word, state, regime, metric, compares in _STATE_TESTS:
        rows = bouts[
            (bouts["state"] == state)
            & (bouts["regime"] == regime)
            & (bouts["metric"] == metric)
        ]
        if len(rows) != 1 or not rows["primary"].iloc[0]:
            raise ValueError(f"bout_stats: no single primary row for {state}, {regime}")
        row = rows.iloc[0]
        positive, total = _k_of(row["animals_pos"])
        epochs_positive, n_epochs = _k_of(row["epochs_pos"])
        if total != n_animals or n_epochs != row["n_epochs"]:
            raise ValueError(f"bout_stats: inconsistent counts for {state}, {regime}")
        label = f"{state} bouts, {regime}"
        values += [
            make(
                f"heldout_{word}_estimate",
                row["animal_estimate"],
                "quantity",
                "sig2",
                f"held-out evidence of {label}, {compares}: mean over animals of "
                "each animal's mean epoch median",
                "nats per window",
            ),
            make(
                f"heldout_{word}_ci_low",
                row["animal_ci_lo"],
                "quantity",
                "sig2",
                f"lower bound of its 95 % interval (nested bootstrap of the animals "
                f"and their epochs, small-sample t correction over the {total} "
                "animals)",
                "nats per window",
            ),
            make(
                f"heldout_{word}_ci_high",
                row["animal_ci_hi"],
                "quantity",
                "sig2",
                "upper bound of that interval",
                "nats per window",
            ),
            make(
                f"heldout_{word}_animals_positive",
                positive,
                "count",
                "count",
                f"animals whose mean held-out evidence of {label} is above zero",
            ),
            make(
                f"heldout_{word}_epochs_positive",
                epochs_positive,
                "count",
                "count",
                f"epochs whose median held-out evidence of {label} is above zero",
            ),
            make(
                f"heldout_{word}_epochs",
                n_epochs,
                "count",
                "count",
                f"epochs with {label} scored",
            ),
        ]
    return values


# --- the memoryless comparison: one recording's pilot note ---------------------


def pilot_stray_window_values(
    events: pd.DataFrame, dispersion_threshold_cm: float, source: str
) -> list[Value]:
    """The caption's stray-window note on the pilot epoch.

    A replay call is decided by kept windows outside every piece when its
    dispersion clears the threshold only with them: the pieces' own
    dispersion does not. The note counts the calls, those decided so, and how
    many of those pass the significance test; it asserts nothing about the last
    count.
    """
    make = value_factory(pilot_stray_window_values, source, TAGS["pilot"])
    stats = outside_window_call_statistics(events, dispersion_threshold_cm)
    return [
        make(
            "pilot_replay_calls",
            stats["n_replay_calls"],
            "count",
            "count",
            "sliding-window replay calls (duration and dispersion criteria) on the "
            f"pilot epoch, {MEMORYLESS_PILOT_EPOCH}",
        ),
        make(
            "pilot_calls_decided_by_outside_window",
            stats["n_decided_by_outside_window"],
            "count",
            "count",
            "of those, calls whose dispersion clears the threshold only with kept "
            "windows outside every piece",
        ),
        make(
            "pilot_calls_decided_by_outside_window_passing",
            stats["n_decided_passing_shuffle"],
            "count",
            "count",
            "of those, calls that pass the significance test",
        ),
    ]


# --- the memoryless comparison: one labeled recording (a-f) -------------------


def recording_values(numbers: dict, source: str) -> list[Value]:
    """Panels (a-f): what each detector made of the labeled events of one
    simulated recording (``recording_statistics.recording_numbers``, which the
    figure script writes)."""
    make = value_factory(recording_values, source, TAGS["memoryless"])

    def p(key: str, description: str) -> Value:
        return make(f"recording_{key}", numbers[key], "quantity", "pval", description)

    return [
        make(
            "recording_outbound_duration_ratio",
            numbers["outbound_duration_ratio"],
            "quantity",
            "sig2",
            "the sliding window's reported duration of the continuous outbound "
            "replay over its true duration",
        ),
        make(
            "recording_outbound_local_fraction",
            numbers["outbound_local_fraction"],
            "fraction",
            "pctnum",
            "share of that reported event's kept windows decoding the animal's own "
            "place (within 15 cm)",
        ),
        p("outbound_p_spread", "significance test of the outbound replay: spread half"),
        p("outbound_p_jump", "significance test of the outbound replay: jump half"),
        make(
            "recording_outbound_draws_from_passing",
            numbers["outbound_draws_from_passing"],
            "count",
            "count",
            "how many more shuffle draws below the observed value than a pass allows",
        ),
        p("inbound_p_spread", "significance test of the inbound replay: spread half"),
        p("inbound_p_jump", "significance test of the inbound replay: jump half"),
        make(
            "recording_short_replay_duration_ratio",
            numbers["short_replay_duration_ratio"],
            "quantity",
            "sig2",
            "the sliding window's reported duration of the 80 ms replay over its "
            "true duration",
        ),
        make(
            "recording_short_replay_local_fraction",
            numbers["short_replay_local_fraction"],
            "fraction",
            "pctnum",
            "share of that reported event's kept windows decoding the animal's own "
            "place",
        ),
        p(
            "short_replay_p_spread",
            "significance test of the 80 ms replay: spread half",
        ),
        p("short_replay_p_jump", "significance test of the 80 ms replay: jump half"),
        make(
            "recording_short_path_dispersion_cm",
            numbers["short_path_dispersion_cm"],
            "quantity",
            "int",
            "largest dispersion of a sliding-window candidate over the 30 cm replay",
            "cm",
        ),
        make(
            "recording_remote_stationary_dispersion_cm",
            numbers["remote_stationary_dispersion_cm"],
            "quantity",
            "int",
            "largest dispersion of a candidate over the remote stationary event",
            "cm",
        ),
        make(
            "recording_state_space_bound_error_ms",
            numbers["state_space_bound_error_ms"],
            "quantity",
            "int",
            "largest error in either bound of the state-space detector's reports of "
            "the four zoomed events",
            "ms",
        ),
        make(
            "recording_stray_window_distance_cm",
            numbers["stray_window_distance_cm"],
            "quantity",
            "int",
            "distance from the animal of the far cell whose two spikes fill the "
            "stray window",
            "cm",
        ),
        make(
            "recording_stray_window_local_fraction",
            numbers["stray_window_local_fraction"],
            "fraction",
            "pctnum",
            "share of the stray-window event's kept windows decoding within 15 cm "
            "of the animal",
        ),
        make(
            "recording_stray_window_duration_ms",
            numbers["stray_window_duration_ms"],
            "quantity",
            "int",
            "the stray-window event's reported duration",
            "ms",
        ),
        make(
            "recording_stray_window_dispersion_cm",
            numbers["stray_window_dispersion_cm"],
            "quantity",
            "sig2",
            "its dispersion with the window outside every piece",
            "cm",
        ),
        make(
            "recording_stray_window_pieces_dispersion_cm",
            numbers["stray_window_pieces_dispersion_cm"],
            "quantity",
            "sig2",
            "its dispersion without that window",
            "cm",
        ),
        p(
            "stray_window_p_spread",
            "significance test of the stray-window event: spread half",
        ),
        p(
            "stray_window_p_jump",
            "significance test of the stray-window event: jump half",
        ),
    ]


# --- the memoryless comparison: the replay sweep (g-k) ------------------------


def sweep_values(summary: pd.DataFrame, source: str) -> list[Value]:
    """Panels (g-k): replays of chosen shape, one at a time, scored against the
    known answer. Shares are of the swept replays (grid cells)."""
    make = value_factory(sweep_values, source, TAGS["memoryless"])
    overall = sweep_statistics.sweep_measures(summary, "overall")
    domain = sweep_statistics.sweep_measures(summary, "overall_in_domain")
    grid = sweep_statistics.sweep_grid_statistics(summary)
    jump_pass = sweep_statistics.sweep_jump_pass_statistics(summary)
    # refuses a sweep whose sliding window does not start early and end late
    timing = sweep_statistics.sweep_timing_statistics(summary)
    durations = (timing["shortest_duration_s"], timing["longest_duration_s"])
    jump_durations = (jump_pass["shortest_duration_s"], jump_pass["longest_duration_s"])
    if jump_durations != durations:
        raise ValueError(
            "the shortest and longest replays of the jump half's pass rates and of "
            f"the timing errors are not of the same durations: {jump_durations} and "
            f"{durations} s"
        )
    shortest_ms, longest_ms = (
        format_number(milliseconds(duration), "int") for duration in durations
    )
    shortest = f"shortest ({shortest_ms} ms)"
    longest = f"longest ({longest_ms} ms)"
    over_durations = f"over true durations ({shortest_ms} to {longest_ms} ms)"

    def share(key: str, x: float, description: str) -> Value:
        return make(key, x, "fraction", "pctnum", description)

    def ms(key: str, x: float, description: str) -> Value:
        return make(key, x, "quantity", "int", description, "ms")

    return [
        make(
            "sweep_replays",
            grid["n_replays"],
            "count",
            "count",
            "replays swept (grid cells)",
        ),
        share(
            "sweep_domain_fraction",
            grid["domain_fraction"],
            "share of the swept replays the published definition can accept at all "
            "(its shared target: path over about 42 cm, duration over 100 ms)",
        ),
        share(
            "sweep_domain_sliding_before_test_fraction",
            domain["sliding_matches_before_shuffle"],
            "on the shared target, share the sliding-window detector reports before "
            "its significance test",
        ),
        share(
            "sweep_domain_state_space_published_fraction",
            domain["state_space_replay_matches"],
            "on the shared target, share the state-space detector reports held to "
            "the published duration and dispersion criteria",
        ),
        share(
            "sweep_domain_sliding_fraction",
            domain["sliding_matches"],
            "on the shared target, share the sliding-window detector reports after "
            "its significance test",
        ),
        share(
            "sweep_domain_state_space_fraction",
            domain["state_space_matches"],
            "on the shared target, share the state-space detector reports under its "
            "own acceptance criteria",
        ),
        share(
            "sweep_sliding_fraction",
            overall["sliding_matches"],
            "over the whole grid, share the sliding-window detector reports under "
            "its own criteria, significance test included",
        ),
        share(
            "sweep_state_space_fraction",
            overall["state_space_matches"],
            "over the whole grid, share the state-space detector reports under its "
            "own criteria",
        ),
        share(
            "sweep_sliding_before_test_fraction",
            overall["sliding_matches_before_shuffle"],
            "over the whole grid, share the sliding-window detector reports before "
            "its significance test",
        ),
        share(
            "sweep_sliding_spread_only_fraction",
            overall["sliding_matches_spread_only"],
            "over the whole grid, share the sliding-window detector reports with the "
            "spread half of its significance test alone",
        ),
        share(
            "sweep_spread_half_pass_fraction",
            overall["sliding_shuffle_spread_pass_rate"],
            "of the replays the sliding window reported before its test, share "
            "passing the test's spread half",
        ),
        share(
            "sweep_jump_half_pass_fraction",
            overall["sliding_shuffle_jump_pass_rate"],
            "of the replays the sliding window reported before its test, share "
            "passing the test's jump half",
        ),
        share(
            "sweep_jump_half_pass_shortest_fraction",
            jump_pass["shortest_fraction"],
            f"share of the {shortest} replays passing the jump half",
        ),
        share(
            "sweep_jump_half_pass_longest_fraction",
            jump_pass["longest_fraction"],
            f"share of the {longest} replays passing the jump half",
        ),
        share(
            "sweep_state_space_only_fraction",
            overall["found_by_state_space_only"],
            "share of the swept replays reported by the state-space detector and not "
            "by the sliding-window detector",
        ),
        share(
            "sweep_failed_test_fraction",
            overall["missed_failed_shuffle"],
            "share the sliding-window detector reported and then failed on its "
            "significance test",
        ),
        make(
            "sweep_sliding_only_replays",
            grid["n_sliding_only"],
            "count",
            "count",
            "swept replays the sliding-window detector reported and the state-space "
            "detector did not",
        ),
        make(
            "sweep_sliding_only_before_test_replays",
            grid["n_sliding_only_before_test"],
            "count",
            "count",
            "the same, counting the sliding window's reports before its test",
        ),
        ms(
            "sweep_state_space_edge_error_max_ms",
            timing["state_space_edge_error_max_ms"],
            "largest absolute median start or end error of the state-space "
            f"detector {over_durations}, replays both detectors reported",
        ),
        ms(
            "sweep_sliding_start_early_min_ms",
            timing["sliding_start_early_min_ms"],
            f"smallest median amount the sliding window starts early, {over_durations}",
        ),
        ms(
            "sweep_sliding_start_early_max_ms",
            timing["sliding_start_early_max_ms"],
            f"largest median amount the sliding window starts early, {over_durations}",
        ),
        ms(
            "sweep_sliding_end_late_min_ms",
            timing["sliding_end_late_min_ms"],
            f"smallest median amount the sliding window ends late, {over_durations}",
        ),
        ms(
            "sweep_sliding_end_late_max_ms",
            timing["sliding_end_late_max_ms"],
            f"largest median amount the sliding window ends late, {over_durations}",
        ),
        ms(
            "sweep_sliding_end_late_upper_quartile_max_ms",
            timing["sliding_end_late_upper_quartile_max_ms"],
            "largest upper quartile of the sliding window's end error, "
            f"{over_durations}",
        ),
        ms(
            "sweep_shortest_start_early_ms",
            timing["shortest_start_early_ms"],
            f"median amount the sliding window starts the {shortest} replays early",
        ),
        ms(
            "sweep_shortest_end_late_ms",
            timing["shortest_end_late_ms"],
            f"median amount the sliding window ends the {shortest} replays late",
        ),
        ms(
            "sweep_shortest_reported_duration_ms",
            timing["shortest_reported_duration_ms"],
            f"median duration the sliding window reports for the {shortest} replays",
        ),
        share(
            "sweep_sliding_not_the_replay_fraction",
            overall["sliding_accepted_but_not_the_injection"],
            "share of the sliding-window detector's accepted events that are not the "
            "replay put in",
        ),
    ]


# --- the memoryless comparison: activity that is not one clean replay (l-o) ---

_SLIDING = "sliding_published"
_OURS = "state_space_canonical"


def controls_values(controls: pd.DataFrame, source: str) -> list[Value]:
    """Panels (l-o): bursts with no clean replay, stretches with no replay, and
    events in two parts."""
    make = value_factory(controls_values, source, TAGS["memoryless"])

    def share(key: str, x: float, description: str) -> Value:
        return make(key, x, "fraction", "pctnum", description)

    def called(condition: str, detector: str, measure: str = "called_rate") -> float:
        return sweep_statistics.controls_value(
            controls, "bursts", measure, condition, None, detector
        )

    def trials(condition: str, detector: str) -> float:
        return sweep_statistics.controls_value(
            controls, "parts", "n_trials", condition, None, detector
        )

    def carving(
        conditions: tuple[str, ...], label: str, detector: str
    ) -> dict[str, float]:
        return sweep_statistics.carving_statistics(
            controls, conditions, label, detector
        )

    # refuses a summary in which an output reported an event with no replay
    no_replay = sweep_statistics.no_replay_statistics(controls)
    fragmented = sweep_statistics.burst_share_range(
        controls, "placeless_burst", "fragmented_share", _OURS
    )
    jumps = ("jump_120ms", "jump_200ms")
    spanning_ours = carving(jumps, "spanning", _OURS)
    spanning_sliding = carving(jumps, "spanning", _SLIDING)
    continuing = {
        (label, detector): carving(("chain_continuing",), label, detector)
        for label in ("spanning", "one_per_part")
        for detector in (_SLIDING, _OURS)
    }
    elsewhere_missing = carving(("chain_elsewhere",), "parts_missing", _SLIDING)
    elsewhere_split = carving(("chain_elsewhere",), "one_per_part", _OURS)
    recovery = {
        detector: sweep_statistics.part_recovery_statistics(
            controls, MULTIPART_CONDITIONS, detector
        )
        for detector in (_SLIDING, _OURS)
    }
    if any(
        trials(condition, _SLIDING) != trials(condition, _OURS)
        for condition in MULTIPART_CONDITIONS
    ):
        raise ValueError("the two detectors are scored on different multi-part trials")

    return [
        share(
            "controls_placeless_nonlocal_fraction",
            called("placeless_burst", _OURS),
            "share of place-less bursts our detector's non-local events cover, "
            "pooled over the three peak rates",
        ),
        share(
            "controls_placeless_fragmented_share_min",
            fragmented["min"],
            "lowest share of a place-less burst's time in the fragmented state, over "
            "burst durations",
        ),
        share(
            "controls_placeless_fragmented_share_max",
            fragmented["max"],
            "highest share of a place-less burst's time in the fragmented state, over "
            "burst durations",
        ),
        share(
            "controls_placeless_continuous_fraction",
            called("placeless_burst", "state_space_continuous"),
            "share of place-less bursts our continuous state alone covers",
        ),
        share(
            "controls_placeless_sliding_fraction",
            called("placeless_burst", _SLIDING),
            "share of place-less bursts the sliding window reports under the "
            "published definition",
        ),
        share(
            "controls_placeless_sliding_before_test_fraction",
            called("placeless_burst", "sliding_accepted"),
            "the same before its significance test",
        ),
        share(
            "controls_placeless_nonlocal_lowest_rate_fraction",
            called("placeless_burst", _OURS, "called_rate_at_20hz"),
            "share of place-less bursts at the lowest peak rate, 20 Hz, our non-local "
            "events cover",
        ),
        share(
            "controls_jumping_nonlocal_fraction",
            called("jumping_burst", _OURS),
            "share of jumping bursts our detector's non-local events cover",
        ),
        share(
            "controls_jumping_continuous_fraction",
            called("jumping_burst", "state_space_continuous"),
            "share of jumping bursts our continuous state alone covers",
        ),
        share(
            "controls_jumping_sliding_fraction",
            called("jumping_burst", _SLIDING),
            "share of jumping bursts the sliding window reports under the published "
            "definition",
        ),
        share(
            "controls_jumping_sliding_before_test_fraction",
            called("jumping_burst", "sliding_accepted"),
            "the same before its significance test",
        ),
        make(
            "controls_no_replay_minutes",
            no_replay["minutes"],
            "quantity",
            "sig2",
            "minutes of simulated immobility with no replay; no output reported any "
            "event there",
            "min",
        ),
        make(
            "controls_no_replay_recordings",
            no_replay["n_recordings"],
            "count",
            "count",
            "simulated recordings the no-replay stretches come from",
        ),
        make(
            "controls_no_replay_rate_bound",
            no_replay["rate_bound_per_minute"],
            "quantity",
            "sig2",
            "95 % upper bound on the false event rate with none observed (the rule "
            "of three over those minutes)",
            "events per minute",
        ),
        share(
            "controls_jump_ours_spanning_min",
            spanning_ours["min"],
            "lowest share, over jump sizes, of two-part jump trials our detector "
            "reports as one event over both parts (pooled over 120 and 200 ms parts)",
        ),
        share(
            "controls_jump_ours_spanning_max",
            spanning_ours["max"],
            "highest such share",
        ),
        share(
            "controls_jump_sliding_spanning_max",
            spanning_sliding["max"],
            "highest share, over jump sizes, of jump trials the sliding window "
            "reports as one event over both parts",
        ),
        share(
            "controls_continuing_sliding_spanning_shortest_fraction",
            continuing["spanning", _SLIDING]["at_lowest_level"],
            "continuing chains at the shortest gap: share of trials the sliding "
            "window reports as one event over both parts",
        ),
        share(
            "controls_continuing_ours_spanning_shortest_fraction",
            continuing["spanning", _OURS]["at_lowest_level"],
            "the same for our detector",
        ),
        share(
            "controls_continuing_sliding_one_per_part_longest_fraction",
            continuing["one_per_part", _SLIDING]["at_highest_level"],
            "continuing chains at the longest gap: share of trials the sliding window "
            "reports as one event per part",
        ),
        share(
            "controls_continuing_ours_one_per_part_longest_fraction",
            continuing["one_per_part", _OURS]["at_highest_level"],
            "the same for our detector",
        ),
        share(
            "controls_elsewhere_sliding_missing_min",
            elsewhere_missing["min"],
            "chains whose parts are far apart: lowest share of trials, over gaps, in "
            "which the sliding window misses a part",
        ),
        share(
            "controls_elsewhere_sliding_missing_max",
            elsewhere_missing["max"],
            "highest such share",
        ),
        share(
            "controls_elsewhere_ours_one_per_part_shortest_fraction",
            elsewhere_split["at_lowest_level"],
            "chains whose parts are far apart, shortest gap: share of trials our "
            "detector reports as one event per part",
        ),
        share(
            "controls_elsewhere_ours_one_per_part_longest_fraction",
            elsewhere_split["at_highest_level"],
            "the same at the longest gap",
        ),
        share(
            "controls_sliding_recovery_in_event_fraction",
            recovery[_SLIDING]["recovery_in_event"],
            "share of parts the sliding window recovers inside the event, mean over "
            "jump and chain conditions and levels",
        ),
        share(
            "controls_sliding_recovery_alone_fraction",
            recovery[_SLIDING]["recovery_alone"],
            "the same with each part decoded alone",
        ),
        share(
            "controls_ours_recovery_in_event_fraction",
            recovery[_OURS]["recovery_in_event"],
            "share of parts our detector recovers inside the event, the same mean",
        ),
        share(
            "controls_ours_recovery_alone_fraction",
            recovery[_OURS]["recovery_alone"],
            "the same with each part decoded alone",
        ),
        make(
            "controls_multipart_trials",
            recovery[_SLIDING]["n_trials"],
            "count",
            "count",
            "two-part trials over every jump and chain condition",
        ),
        make(
            "controls_sliding_fewer_in_event_trials",
            recovery[_SLIDING]["n_trials_fewer_in_event"],
            "count",
            "count",
            "trials in which the sliding window recovers fewer parts inside the event "
            "than alone",
        ),
        make(
            "controls_sliding_more_in_event_trials",
            recovery[_SLIDING]["n_trials_more_in_event"],
            "count",
            "count",
            "trials in which it recovers more inside the event than alone",
        ),
        make(
            "controls_ours_fewer_in_event_trials",
            recovery[_OURS]["n_trials_fewer_in_event"],
            "count",
            "count",
            "trials in which our detector recovers fewer parts inside the event than "
            "alone",
        ),
        make(
            "controls_ours_more_in_event_trials",
            recovery[_OURS]["n_trials_more_in_event"],
            "count",
            "count",
            "trials in which it recovers more inside the event than alone",
        ),
    ]


# --- the memoryless comparison: held-out evidence (p, q) ----------------------


def heldout_values(heldout: pd.DataFrame, source: str) -> list[Value]:
    """Panels (p, q): content estimated from discovery cells, scored on held-out
    evaluation cells. Folds A and B tile the track; fold C's evaluation cells
    cover half of it."""
    make = value_factory(heldout_values, source, TAGS["memoryless"])

    def share(key: str, x: float, description: str) -> Value:
        return make(key, x, "fraction", "pctnum", description)

    def over_folds(
        folds: tuple[str, ...], classes: tuple[str, ...], truth: str, **which
    ) -> dict[str, float]:
        return sweep_statistics.heldout_fold_statistics(
            heldout, folds, classes, truth, **which
        )

    def in_fold(fold: str, classes: tuple[str, ...]) -> float:
        return sweep_statistics.pooled_heldout_rate(
            heldout, fold, classes, "continuous_replay"
        )["rate"]

    ours = over_folds(TILING_FOLDS, ("ours_only",), "continuous_replay")
    shared = over_folds(FOLDS, ("shared",), "continuous_replay")
    jumping = over_folds(TILING_FOLDS, OUR_EVENT_CLASSES, "non_local")
    informative = {
        name: over_folds(
            TILING_FOLDS, classes, "continuous_replay", measure="informative_share"
        )
        for name, classes in (("ours", ("ours_only",)), ("shared", ("shared",)))
    }
    # refuses a summary in which an informative place-less burst has support
    placeless = sweep_statistics.placeless_burst_statistics(heldout, FOLDS)
    placeless_loose = over_folds(
        FOLDS, OUR_EVENT_CLASSES, "no_content", suffix="_floor2"
    )
    if shared["rate_spread"] != 0:
        raise ValueError(
            "the caption states one held-out support rate for shared replays; the "
            "folds differ, or one has no informative shared replay (from "
            f"{shared['rate_min']} to {shared['rate_max']})"
        )
    # refuses a summary holding an event only the sliding window reported
    sweep_statistics.heldout_event_classes(heldout)
    return [
        share(
            "heldout_ours_only_support_min",
            ours["rate_min"],
            "replays only our detector reported, informative in the evaluation cells: "
            "lower of folds A and B's share with held-out support",
        ),
        share(
            "heldout_ours_only_support_max",
            ours["rate_max"],
            "the higher of the two",
        ),
        share(
            "heldout_ours_only_support_fold_c",
            in_fold("C", ("ours_only",)),
            "the same share in fold C",
        ),
        share(
            "heldout_shared_support",
            shared["rate_min"],
            "replays both detectors reported: share with held-out support, the same "
            "in every fold",
        ),
        make(
            "heldout_placeless_informative_events",
            placeless["n_events"],
            "count",
            "count",
            "informative place-less burst events over the three folds (none has "
            "held-out support)",
        ),
        make(
            "heldout_placeless_loose_supported_events",
            placeless_loose["n_counted"],
            "count",
            "count",
            "place-less burst events with held-out support at a floor of 2 expected "
            "spikes",
        ),
        make(
            "heldout_placeless_loose_events",
            placeless_loose["n_events"],
            "count",
            "count",
            "place-less burst events informative at that floor",
        ),
        share(
            "heldout_jumping_support_min",
            jumping["rate_min"],
            "jumping bursts: lower of folds A and B's share with held-out support",
        ),
        share(
            "heldout_jumping_support_max", jumping["rate_max"], "the higher of the two"
        ),
        share(
            "heldout_estimator_agreement_min",
            sweep_statistics.lowest_estimator_agreement(heldout),
            "lowest share over folds of the events informative under both content "
            "estimates on which they agree",
        ),
        share(
            "heldout_ours_only_informative_min",
            informative["ours"]["rate_min"],
            "replays only our detector reported: lower of folds A and B's share the "
            "evaluation cells can judge at all",
        ),
        share(
            "heldout_ours_only_informative_max",
            informative["ours"]["rate_max"],
            "the higher of the two",
        ),
        share(
            "heldout_shared_informative_min",
            informative["shared"]["rate_min"],
            "shared replays: lower of folds A and B's share the evaluation cells can "
            "judge at all",
        ),
        share(
            "heldout_shared_informative_max",
            informative["shared"]["rate_max"],
            "the higher of the two",
        ),
    ]


# --- the summary ---------------------------------------------------------------


def summarize(inputs: DetectionInputs, produced_by: str) -> Summary:
    """The detection summary from ``inputs``."""
    summary = Summary(name="detection", produced_by=produced_by)

    bouts_path = inputs.state_evidence_dir / "bout_stats.csv"
    manifest_path = inputs.state_evidence_dir / "manifest.json"
    summary.add(
        *state_evidence_values(
            pd.read_csv(bouts_path),
            read_manifest(inputs.state_evidence_dir),
            f"{rel(bouts_path)} and manifest.json (scripts/event_evidence/run_state_evidence.py; "
            "pooled by the runner's own summarize over the per-epoch records, "
            "2026-10-17)",
        )
    )
    summary.inputs += [
        file_record(bouts_path, ROOT_DIR),
        file_record(manifest_path, ROOT_DIR),
    ]

    pilot_settings = read_manifest(inputs.pilot_manifest.parent)["settings"]
    summary.add(
        *pilot_stray_window_values(
            pd.read_parquet(inputs.pilot_events),
            pilot_settings["config"]["dispersion_threshold_cm"],
            f"{rel(inputs.pilot_events)} (scripts/detection/run_memoryless_detection.py, "
            f"commit {pilot_settings['git_commit'][:12]})",
        )
    )
    summary.inputs += [
        file_record(inputs.pilot_events, ROOT_DIR),
        file_record(inputs.pilot_manifest, ROOT_DIR),
    ]

    recording = json.loads(inputs.recording_numbers.read_text())
    summary.add(
        *recording_values(
            recording["numbers"],
            f"{rel(inputs.recording_numbers)} ({recording['produced_by']}, commit "
            f"{recording['git_commit']})",
        )
    )
    summary.inputs.append(file_record(inputs.recording_numbers, ROOT_DIR))

    for path, build, script in (
        (inputs.sweep_summary, sweep_values, "detection/sweep_memoryless_detection.py"),
        (
            inputs.controls_summary,
            controls_values,
            "detection/sweep_memoryless_controls.py",
        ),
        (
            inputs.heldout_summary,
            heldout_values,
            "detector_comparison/sweep_memoryless_heldout.py",
        ),
    ):
        summary.add(*build(pd.read_csv(path), f"{rel(path)} (scripts/{script})"))
        summary.inputs.append(file_record(path, ROOT_DIR))
    return summary
