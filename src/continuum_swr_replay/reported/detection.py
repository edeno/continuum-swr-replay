"""The detection summary: the held-out tetrode test of each decoder state, and the
detector against the Widloski & Foster detector on simulated replays.

Every value is read from an output an analysis already wrote; nothing is
refitted or re-decoded here. The inputs (:class:`DetectionInputs`):

- the held-out tetrode test per decoder state
  (``Processed-Data/_refresh_20261016/state_evidence``, from
  ``scripts/detection/run_state_evidence.py``): its pooled ``bout_stats.csv``
  and the per-epoch status in ``manifest.json``;
- the three simulation sweeps of the two replay detectors
  (``docs/design/replay_detection_comparison/detection_{sweep,controls,heldout}_summary.csv``,
  from ``scripts/detection/sweep_memoryless_{detection,controls,heldout}.py``),
  for the memoryless-comparison caption's panels (g-q);
- the one labeled recording of that caption's panels (a-f)
  (``docs/design/replay_detection_comparison/memoryless_comparison_recording.json``,
  written by ``scripts/figures/generate_memoryless_comparison.py``);
- the sliding-window detector's first real-data pilot epoch (a
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

import numpy as np
import pandas as pd

from continuum_swr_replay.paths import ROOT_DIR, memoryless_root, processed_data_dir
from continuum_swr_replay.reported.values import Summary, Value, file_record

STATE_EVIDENCE_RUN = "_refresh_20261016"
"""The held-out tetrode run the detection summary reads (242 epochs, all ``ok``)."""

MEMORYLESS_PILOT_RUN = "0c2548cacf2d-seed20260922"
"""The sliding-window detector's first real-data pilot (chimi20200225_02_r1)."""

MEMORYLESS_PILOT_EPOCH = "chimi20200225_02_r1_HPC"

SWEEP_DIR = Path("docs/design/replay_detection_comparison")

MEMORYLESS_FIGURE = ("supplement", "figure: memoryless comparison")
"""Where the sweep values are used: the supplemental figure and its caption."""


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
        state_evidence_dir=processed_data_dir() / STATE_EVIDENCE_RUN / "state_evidence",
        sweep_summary=ROOT_DIR / SWEEP_DIR / "detection_sweep_summary.csv",
        controls_summary=ROOT_DIR / SWEEP_DIR / "detection_controls_summary.csv",
        heldout_summary=ROOT_DIR / SWEEP_DIR / "detection_heldout_summary.csv",
        recording_numbers=ROOT_DIR / SWEEP_DIR / "memoryless_comparison_recording.json",
        pilot_events=pilot / f"{MEMORYLESS_PILOT_EPOCH}_memoryless_events.parquet",
        pilot_manifest=pilot / "manifest.json",
    )


def _value_factory(source: str, questions: tuple[str, ...], used_in: tuple[str, ...]):
    """A :class:`Value` builder with the shared source, claims and use filled in."""

    def make(
        key: str,
        value: float | int,
        kind: str,
        style: str,
        description: str,
        unit: str = "",
        *,
        pending: str | None = None,
    ) -> Value:
        if kind == "count":
            if not np.isclose(float(value), round(float(value)), rtol=0, atol=1e-6):
                raise ValueError(f"{key}: a count must be whole, got {value}")
            value = int(round(float(value)))
        else:
            value = float(value)
        return Value(
            key=key,
            value=value,
            kind=kind,
            style=style,
            unit=unit,
            description=description,
            questions=questions,
            used_in=used_in,
            source=source,
            pending=pending,
        )

    return make


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
    make = _value_factory(source, ("2c.3",), ("paper", "supplement"))
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
                "lower bound of its 95 % t-interval over animals",
                "nats per window",
            ),
            make(
                f"heldout_{word}_ci_high",
                row["animal_ci_hi"],
                "quantity",
                "sig2",
                "upper bound of its 95 % t-interval over animals",
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
    """The caption's stray-window note on the first real-data pilot epoch.

    A replay call is decided by kept windows outside every piece when its
    dispersion clears the threshold only with them: the pieces' own
    dispersion does not.
    """
    make = _value_factory(source, ("2a.3",), MEMORYLESS_FIGURE)
    replay = events[events["is_replay"]]
    decided = replay[
        (replay["n_kept_windows_outside_pieces"] > 0)
        & ~(replay["pieces_dispersion_cm"] > dispersion_threshold_cm)
    ]
    if not decided["shuffle_pass"].all():
        raise ValueError(
            "the caption says every pilot replay call decided by an outside window "
            "passed the significance test; the pilot's events disagree"
        )
    return [
        make(
            "pilot_replay_calls",
            len(replay),
            "count",
            "count",
            "sliding-window replay calls (duration and dispersion criteria) on the "
            "first real-data pilot epoch, chimi20200225_02_r1",
        ),
        make(
            "pilot_calls_decided_by_outside_window",
            len(decided),
            "count",
            "count",
            "of those, calls whose dispersion clears the threshold only with kept "
            "windows outside every piece",
        ),
    ]


# --- the memoryless comparison: one labeled recording (a-f) -------------------


def recording_values(numbers: dict, source: str) -> list[Value]:
    """Panels (a-f): what each detector made of the labeled events of one
    simulated recording, as the figure script computed them."""
    make = _value_factory(source, ("2a.3",), MEMORYLESS_FIGURE)
    if not numbers["outbound_draws_from_passing"] > 1:
        raise ValueError(
            "the caption says the outbound continuous replay fails the test by "
            "several draws; the recording's numbers disagree"
        )

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


def _timing(summary: pd.DataFrame, detector: str, edge: str, statistic: str):
    """True duration (s) and the timing error statistic (ms), by duration."""
    rows = summary[
        (summary["summary"] == "timing_by_duration")
        & (summary["measure"] == f"{detector}_{edge}_error_ms_{statistic}")
    ].sort_values("duration_s")
    return rows["duration_s"].to_numpy(), rows["value"].to_numpy()


def sweep_values(summary: pd.DataFrame, source: str) -> list[Value]:
    """Panels (g-k): replays of chosen shape, one at a time, scored against the
    known answer. Shares are of the swept replays (grid cells)."""
    make = _value_factory(source, ("2a.3",), MEMORYLESS_FIGURE)
    overall = summary[summary["summary"] == "overall"].set_index("measure")["value"]
    domain = summary[summary["summary"] == "overall_in_domain"].set_index("measure")[
        "value"
    ]
    n = overall["n_cells"]
    passes = summary[summary["summary"] == "shuffle_pass_rate_by_duration"]
    jump_pass = (
        passes[passes["measure"] == "sliding_jump_pass"]
        .set_index("duration_s")["value"]
        .sort_index()
    )
    _, sliding_start = _timing(summary, "sliding", "start", "median")
    _, sliding_end = _timing(summary, "sliding", "end", "median")
    _, sliding_end_q75 = _timing(summary, "sliding", "end", "q75")
    state_space_edges = np.concatenate(
        [_timing(summary, "state_space", e, "median")[1] for e in ("start", "end")]
    )
    ratio = summary[
        (summary["summary"] == "median_duration_ratio_by_duration")
        & (summary["measure"] == "sliding_duration_ratio")
    ].sort_values("duration_s")
    shortest = ratio.iloc[0]
    if not (sliding_start < 0).all() or not (sliding_end > 0).all():
        raise ValueError(
            "the caption says the sliding window starts early and ends late at "
            "every duration; the sweep's median timing errors disagree"
        )

    def share(key: str, x: float, description: str) -> Value:
        return make(key, x, "fraction", "pctnum", description)

    def ms(key: str, x: float, description: str) -> Value:
        return make(key, x, "quantity", "int", description, "ms")

    return [
        make("sweep_replays", n, "count", "count", "replays swept (grid cells)"),
        share(
            "sweep_domain_fraction",
            domain["n_cells"] / n,
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
            jump_pass.iloc[0],
            f"share of the shortest ({1000 * jump_pass.index[0]:.0f} ms) replays "
            "passing the jump half",
        ),
        share(
            "sweep_jump_half_pass_longest_fraction",
            jump_pass.iloc[-1],
            f"share of the longest ({1000 * jump_pass.index[-1]:.0f} ms) replays "
            "passing the jump half",
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
            overall["found_by_sliding_window_only"] * n,
            "count",
            "count",
            "swept replays the sliding-window detector reported and the state-space "
            "detector did not",
        ),
        make(
            "sweep_sliding_only_before_test_replays",
            overall["found_by_sliding_window_only_before_shuffle"] * n,
            "count",
            "count",
            "the same, counting the sliding window's reports before its test",
        ),
        ms(
            "sweep_state_space_edge_error_max_ms",
            np.abs(state_space_edges).max(),
            "largest absolute median start or end error of the state-space "
            "detector over true durations, replays both detectors reported",
        ),
        ms(
            "sweep_sliding_start_early_min_ms",
            -sliding_start.max(),
            "smallest median amount the sliding window starts early, over true "
            "durations",
        ),
        ms(
            "sweep_sliding_start_early_max_ms",
            -sliding_start.min(),
            "largest median amount the sliding window starts early, over true durations",
        ),
        ms(
            "sweep_sliding_end_late_min_ms",
            sliding_end.min(),
            "smallest median amount the sliding window ends late, over true durations",
        ),
        ms(
            "sweep_sliding_end_late_max_ms",
            sliding_end.max(),
            "largest median amount the sliding window ends late, over true durations",
        ),
        ms(
            "sweep_sliding_end_late_upper_quartile_max_ms",
            sliding_end_q75.max(),
            "largest upper quartile of the sliding window's end error, over true "
            "durations",
        ),
        ms(
            "sweep_shortest_start_early_ms",
            -sliding_start[0],
            "median amount the sliding window starts the shortest replays early",
        ),
        ms(
            "sweep_shortest_end_late_ms",
            sliding_end[0],
            "median amount the sliding window ends the shortest replays late",
        ),
        ms(
            "sweep_shortest_reported_duration_ms",
            1000 * shortest["duration_s"] * shortest["value"],
            "median duration the sliding window reports for the shortest replays",
        ),
        share(
            "sweep_sliding_not_the_replay_fraction",
            overall["sliding_accepted_but_not_the_injection"],
            "share of the sliding-window detector's accepted events that are not the "
            "replay put in",
        ),
    ]


# --- the memoryless comparison: activity that is not one clean replay (l-o) ---

_MULTIPART = ("jump_120ms", "jump_200ms", "chain_continuing", "chain_elsewhere")
_SLIDING = "sliding_published"
_OURS = "state_space_canonical"


def _controls_value(
    controls: pd.DataFrame,
    summary: str,
    measure: str,
    condition: str,
    level: float | None,
    detector: str,
) -> float:
    """One value of the controls summary, refusing to guess between several."""
    rows = controls[
        (controls["summary"] == summary)
        & (controls["measure"] == measure)
        & (controls["condition"] == condition)
        & (controls["detector"] == detector)
    ]
    rows = (
        rows[rows["level"].isna()]
        if level is None
        else rows[np.isclose(rows["level"], level)]
    )
    if len(rows) != 1:
        raise ValueError(
            f"controls summary: {len(rows)} rows for {summary}, {measure}, "
            f"{condition}, {level}, {detector}; expected one"
        )
    return float(rows["value"].iloc[0])


def _levels(controls: pd.DataFrame, condition: str) -> list[float]:
    rows = controls[
        (controls["summary"] == "parts") & (controls["condition"] == condition)
    ]
    return sorted(rows["level"].dropna().unique())


def controls_values(controls: pd.DataFrame, source: str) -> list[Value]:
    """Panels (l-o): bursts with no clean replay, stretches with no replay, and
    events in two parts."""
    make = _value_factory(source, ("2a.3",), MEMORYLESS_FIGURE)

    def share(key: str, x: float, description: str) -> Value:
        return make(key, x, "fraction", "pctnum", description)

    def called(condition: str, detector: str, measure: str = "called_rate") -> float:
        return _controls_value(controls, "bursts", measure, condition, None, detector)

    def carving(conditions: tuple[str, ...], label: str, detector: str) -> np.ndarray:
        """Share of trials carved this way at each level, pooled over conditions."""
        return np.array(
            [
                np.mean(
                    [
                        _controls_value(
                            controls,
                            "parts",
                            f"segmentation_{label}",
                            c,
                            level,
                            detector,
                        )
                        for c in conditions
                    ]
                )
                for level in _levels(controls, conditions[0])
            ]
        )

    def mean_recovery(detector: str, measure: str) -> float:
        return float(
            np.mean(
                [
                    _controls_value(controls, "parts", measure, c, level, detector)
                    for c in _MULTIPART
                    for level in _levels(controls, c)
                ]
            )
        )

    def trials(detector: str, measure: str) -> float:
        return sum(
            _controls_value(controls, "parts", measure, c, None, detector)
            for c in _MULTIPART
        )

    no_replay = controls[
        (controls["summary"] == "no_replay") & (controls["measure"] == "n_events")
    ]
    if no_replay.empty or (no_replay["value"] != 0).any():
        raise ValueError(
            "the caption says no output reported an event where there was no replay; "
            "the controls summary disagrees"
        )
    minutes = float(
        controls[
            (controls["summary"] == "no_replay")
            & (controls["measure"] == "immobile_minutes")
        ]["value"].iloc[0]
    )
    n_recordings = float(
        controls[
            (controls["summary"] == "provenance")
            & (controls["measure"] == "n_recordings")
        ]["value"].iloc[0]
    )
    fragmented = controls[
        (controls["summary"] == "bursts")
        & (controls["measure"] == "fragmented_share")
        & (controls["condition"] == "placeless_burst")
    ]["value"]
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
    n_trials = {d: trials(d, "n_trials") for d in (_SLIDING, _OURS)}
    if n_trials[_SLIDING] != n_trials[_OURS]:
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
            fragmented.min(),
            "lowest share of a place-less burst's time in the fragmented state, over "
            "burst durations",
        ),
        share(
            "controls_placeless_fragmented_share_max",
            fragmented.max(),
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
            minutes,
            "quantity",
            "sig2",
            "minutes of simulated immobility with no replay; no output reported any "
            "event there",
            "min",
        ),
        make(
            "controls_no_replay_recordings",
            n_recordings,
            "count",
            "count",
            "simulated recordings the no-replay stretches come from",
        ),
        make(
            "controls_no_replay_rate_bound",
            3 / minutes,
            "quantity",
            "sig2",
            "95 % upper bound on the false event rate with none observed (the rule "
            "of three over those minutes)",
            "events per minute",
        ),
        share(
            "controls_jump_ours_spanning_min",
            spanning_ours.min(),
            "lowest share, over jump sizes, of two-part jump trials our detector "
            "reports as one event over both parts (pooled over 120 and 200 ms parts)",
        ),
        share(
            "controls_jump_ours_spanning_max",
            spanning_ours.max(),
            "highest such share",
        ),
        share(
            "controls_jump_sliding_spanning_max",
            spanning_sliding.max(),
            "highest share, over jump sizes, of jump trials the sliding window "
            "reports as one event over both parts",
        ),
        share(
            "controls_continuing_sliding_spanning_shortest_fraction",
            continuing["spanning", _SLIDING][0],
            "continuing chains at the shortest gap: share of trials the sliding "
            "window reports as one event over both parts",
        ),
        share(
            "controls_continuing_ours_spanning_shortest_fraction",
            continuing["spanning", _OURS][0],
            "the same for our detector",
        ),
        share(
            "controls_continuing_sliding_one_per_part_longest_fraction",
            continuing["one_per_part", _SLIDING][-1],
            "continuing chains at the longest gap: share of trials the sliding window "
            "reports as one event per part",
        ),
        share(
            "controls_continuing_ours_one_per_part_longest_fraction",
            continuing["one_per_part", _OURS][-1],
            "the same for our detector",
        ),
        share(
            "controls_elsewhere_sliding_missing_min",
            elsewhere_missing.min(),
            "chains whose parts are far apart: lowest share of trials, over gaps, in "
            "which the sliding window misses a part",
        ),
        share(
            "controls_elsewhere_sliding_missing_max",
            elsewhere_missing.max(),
            "highest such share",
        ),
        share(
            "controls_elsewhere_ours_one_per_part_shortest_fraction",
            elsewhere_split[0],
            "chains whose parts are far apart, shortest gap: share of trials our "
            "detector reports as one event per part",
        ),
        share(
            "controls_elsewhere_ours_one_per_part_longest_fraction",
            elsewhere_split[-1],
            "the same at the longest gap",
        ),
        share(
            "controls_sliding_recovery_in_event_fraction",
            mean_recovery(_SLIDING, "part_recovery"),
            "share of parts the sliding window recovers inside the event, mean over "
            "jump and chain conditions and levels",
        ),
        share(
            "controls_sliding_recovery_alone_fraction",
            mean_recovery(_SLIDING, "part_recovery_alone"),
            "the same with each part decoded alone",
        ),
        share(
            "controls_ours_recovery_in_event_fraction",
            mean_recovery(_OURS, "part_recovery"),
            "share of parts our detector recovers inside the event, the same mean",
        ),
        share(
            "controls_ours_recovery_alone_fraction",
            mean_recovery(_OURS, "part_recovery_alone"),
            "the same with each part decoded alone",
        ),
        make(
            "controls_multipart_trials",
            n_trials[_SLIDING],
            "count",
            "count",
            "two-part trials over every jump and chain condition",
        ),
        make(
            "controls_sliding_fewer_in_event_trials",
            trials(_SLIDING, "n_trials_fewer_in_context"),
            "count",
            "count",
            "trials in which the sliding window recovers fewer parts inside the event "
            "than alone",
        ),
        make(
            "controls_sliding_more_in_event_trials",
            trials(_SLIDING, "n_trials_more_in_context"),
            "count",
            "count",
            "trials in which it recovers more inside the event than alone",
        ),
        make(
            "controls_ours_fewer_in_event_trials",
            trials(_OURS, "n_trials_fewer_in_context"),
            "count",
            "count",
            "trials in which our detector recovers fewer parts inside the event than "
            "alone",
        ),
        make(
            "controls_ours_more_in_event_trials",
            trials(_OURS, "n_trials_more_in_context"),
            "count",
            "count",
            "trials in which it recovers more inside the event than alone",
        ),
    ]


# --- the memoryless comparison: held-out evidence (p, q) ----------------------

_FOLDS = ("A", "B", "C")


def _heldout_rate(
    heldout: pd.DataFrame,
    fold: str,
    classes: tuple[str, ...],
    truth: str,
    measure: str = "support_rate",
    suffix: str = "",
    estimator: str = "state_space",
) -> tuple[float, int]:
    """A held-out rate over our detector's events, pooled over event classes.

    Each event is counted once, as our detector bounds it. A rate over
    informative events is weighted by how many are informative, a share of all
    events by how many there are. NaN, 0 when no event counts.
    """
    rows = heldout[
        (heldout["summary"] == "by_class")
        & (heldout["fold"] == fold)
        & (heldout["estimator"] == estimator)
        & (heldout["detector"] == "state_space")
        & heldout["event_class"].isin(classes)
        & (heldout["truth"] == truth)
    ].pivot_table(index="event_class", columns="measure", values="value")
    weight = "n_events_all" if measure == "informative_share" else f"n_events{suffix}"
    if rows.empty or weight not in rows or rows[weight].sum() == 0:
        return float("nan"), 0
    n = rows[weight].fillna(0)
    rate = float((rows[f"{measure}{suffix}"].fillna(0) * n).sum() / n.sum())
    return rate, int(n.sum())


def heldout_values(heldout: pd.DataFrame, source: str) -> list[Value]:
    """Panels (p, q): content estimated from discovery cells, scored on held-out
    evaluation cells. Folds A and B tile the track; fold C's evaluation cells
    cover half of it."""
    make = _value_factory(source, ("2a.3",), MEMORYLESS_FIGURE)

    def share(key: str, x: float, description: str) -> Value:
        return make(key, x, "fraction", "pctnum", description)

    ours = [
        _heldout_rate(heldout, f, ("ours_only",), "continuous_replay")[0] for f in "AB"
    ]
    shared = [
        _heldout_rate(heldout, f, ("shared",), "continuous_replay")[0] for f in _FOLDS
    ]
    jumping = [
        _heldout_rate(heldout, f, ("ours_only", "shared"), "non_local")[0] for f in "AB"
    ]
    informative = {
        name: [
            _heldout_rate(
                heldout, f, classes, "continuous_replay", "informative_share"
            )[0]
            for f in "AB"
        ]
        for name, classes in (("ours", ("ours_only",)), ("shared", ("shared",)))
    }
    placeless = [
        _heldout_rate(heldout, f, ("ours_only", "shared"), "no_content", estimator=e)
        for f in _FOLDS
        for e in ("state_space", "sliding_window")
    ]
    if any(n and rate > 0 for rate, n in placeless):
        raise ValueError(
            "the caption says no informative place-less burst event has held-out "
            "support; the held-out summary disagrees"
        )
    placeless_ours = [
        _heldout_rate(heldout, f, ("ours_only", "shared"), "no_content") for f in _FOLDS
    ]
    placeless_loose = [
        _heldout_rate(
            heldout, f, ("ours_only", "shared"), "no_content", suffix="_floor2"
        )
        for f in _FOLDS
    ]
    if np.ptp(shared) > 0:
        raise ValueError(
            "the caption states one held-out support rate for shared replays; the "
            f"folds differ ({shared})"
        )
    by_class = heldout[heldout["summary"] == "by_class"]
    if (by_class["event_class"] == "comparator_only").any():
        raise ValueError(
            "the caption says the sliding window reported no event our detector "
            "missed; the held-out summary holds comparator-only events"
        )
    agreement = heldout[
        (heldout["summary"] == "agreement")
        & (heldout["measure"] == "estimator_agreement")
    ]["value"]
    return [
        share(
            "heldout_ours_only_support_min",
            min(ours),
            "replays only our detector reported, informative in the evaluation cells: "
            "lower of folds A and B's share with held-out support",
        ),
        share(
            "heldout_ours_only_support_max",
            max(ours),
            "the higher of the two",
        ),
        share(
            "heldout_ours_only_support_fold_c",
            _heldout_rate(heldout, "C", ("ours_only",), "continuous_replay")[0],
            "the same share in fold C",
        ),
        share(
            "heldout_shared_support",
            shared[0],
            "replays both detectors reported: share with held-out support, the same "
            "in every fold",
        ),
        make(
            "heldout_placeless_informative_events",
            sum(n for _, n in placeless_ours),
            "count",
            "count",
            "informative place-less burst events over the three folds (none has "
            "held-out support)",
        ),
        make(
            "heldout_placeless_loose_supported_events",
            sum(round(rate * n) for rate, n in placeless_loose if n),
            "count",
            "count",
            "place-less burst events with held-out support at a floor of 2 expected "
            "spikes",
        ),
        make(
            "heldout_placeless_loose_events",
            sum(n for _, n in placeless_loose),
            "count",
            "count",
            "place-less burst events informative at that floor",
        ),
        share(
            "heldout_jumping_support_min",
            min(jumping),
            "jumping bursts: lower of folds A and B's share with held-out support",
        ),
        share("heldout_jumping_support_max", max(jumping), "the higher of the two"),
        share(
            "heldout_estimator_agreement_min",
            agreement.min(),
            "lowest share over folds of the events informative under both content "
            "estimates on which they agree",
        ),
        share(
            "heldout_ours_only_informative_min",
            min(informative["ours"]),
            "replays only our detector reported: lower of folds A and B's share the "
            "evaluation cells can judge at all",
        ),
        share(
            "heldout_ours_only_informative_max",
            max(informative["ours"]),
            "the higher of the two",
        ),
        share(
            "heldout_shared_informative_min",
            min(informative["shared"]),
            "shared replays: lower of folds A and B's share the evaluation cells can "
            "judge at all",
        ),
        share(
            "heldout_shared_informative_max",
            max(informative["shared"]),
            "the higher of the two",
        ),
    ]


# --- the summary ---------------------------------------------------------------


def summarize(inputs: DetectionInputs, produced_by: str, commit: str) -> Summary:
    """The detection summary from ``inputs``."""
    summary = Summary(name="detection", produced_by=produced_by, commit=commit)

    bouts_path = inputs.state_evidence_dir / "bout_stats.csv"
    manifest_path = inputs.state_evidence_dir / "manifest.json"
    summary.add(
        *state_evidence_values(
            pd.read_csv(bouts_path),
            json.loads(manifest_path.read_text()),
            f"{rel(bouts_path)} and manifest.json (scripts/detection/run_state_evidence.py; "
            "pooled by the runner's own summarize over the per-epoch records, "
            "2026-10-17)",
        )
    )
    summary.inputs += [
        _record(bouts_path),
        _record(manifest_path),
    ]

    pilot_settings = json.loads(inputs.pilot_manifest.read_text())["settings"]
    summary.add(
        *pilot_stray_window_values(
            pd.read_parquet(inputs.pilot_events),
            pilot_settings["config"]["dispersion_threshold_cm"],
            f"{rel(inputs.pilot_events)} (scripts/detection/run_memoryless_detection.py, "
            f"commit {pilot_settings['git_commit'][:12]})",
        )
    )
    summary.inputs += [
        _record(inputs.pilot_events),
        _record(inputs.pilot_manifest),
    ]

    recording = json.loads(inputs.recording_numbers.read_text())
    summary.add(
        *recording_values(
            recording["numbers"],
            f"{rel(inputs.recording_numbers)} ({recording['produced_by']}, commit "
            f"{recording['git_commit']})",
        )
    )
    summary.inputs.append(_record(inputs.recording_numbers))

    for path, build, script in (
        (inputs.sweep_summary, sweep_values, "sweep_memoryless_detection.py"),
        (inputs.controls_summary, controls_values, "sweep_memoryless_controls.py"),
        (inputs.heldout_summary, heldout_values, "sweep_memoryless_heldout.py"),
    ):
        summary.add(
            *build(pd.read_csv(path), f"{rel(path)} (scripts/detection/{script})")
        )
        summary.inputs.append(_record(path))
    return summary


def _record(path: Path) -> dict:
    """An input's :func:`file_record`, its path shown relative to the repository.

    ``file_record`` resolves symbolic links first, so in a worktree whose
    ``Processed-Data`` links to another checkout's it would show an absolute
    path; this shows the path as the repository reads it.
    """
    return {**file_record(path, ROOT_DIR), "path": rel(path)}


def rel(path: Path) -> str:
    """``path`` relative to the repository when it is under it."""
    return (
        str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path)
    )
