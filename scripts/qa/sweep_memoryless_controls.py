#!/usr/bin/env python
"""What each detector does with activity that is not one clean replay.

The injected-replay sweep asks how often each detector reports a replay that
is there. This asks the other questions: what each reports where there is no
replay, how it carves a replay that jumps partway or comes in a chain, and
where it puts the bounds of each part. Both detectors, the same recording and
encoding model, the same placements and scoring rule as the sweep:

- no replay: the recording's own immobile stretches, decoded as they are. They
  hold only local coding, so every event reported there is a false positive.
  Eight simulated recordings, the sweep's and seven more, for exposure;
- a place-less burst: every cell fires at one rate unrelated to its field, as
  many spikes as a replay of the same duration and peak rate. No spatial
  content, so a detection is a false positive for either target;
- a jumping burst: the represented place redrawn along the track every 15 ms.
  Non-local representation without a sequence: calling it non-local is right,
  calling it a sequence is not;
- a mid-event jump: two 60 cm parts at constant speed, the second starting J cm
  behind where the first ended, with no gap in time; J = 0 is a continuous
  replay of both parts;
- a chain: two 150 ms, 60 cm replays with a silent gap, the second continuing
  from the first's end or far from it.

Each part is 60 cm and longer than 100 ms, so it can meet the published replay
definition on its own, and each is also decoded alone, with exactly the spikes it
had inside its event and the recording's own spikes around it. Recovery inside
the event and alone is compared both pooled and paired by trial, since pooled
recovery can rise while some trials still lose a part to the jump or the chain.
The
simulated animal rests at the track's 170 cm end, and absorbing local coding
depends on how near it an event comes, so every jump and chain keeps its
parts' far end at the same place, whatever the jump or the gap. Our detector is scored two ways throughout, canonical
(continuous plus fragmented) and continuous-only; the sliding window before and
after its significance test.

Usage:
    python scripts/qa/sweep_memoryless_controls.py
    python scripts/qa/sweep_memoryless_controls.py --n-seeds 1 --out /tmp/pilot.parquet
"""

import argparse
import sys
import time as clock
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.scoring import (  # noqa: E402
    SEGMENTATION_LABELS,
    bound_errors,
    part_agreement,
    recovered_parts,
    score_against_injection,
    segmentation,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (  # noqa: E402
    Gap,
    InjectedActivity,
    Segment,
    immobile_intervals,
    inject_activity,
    jumping_burst,
    piecewise_replay,
    place_field_means,
    placeless_burst,
    placements,
    without,
)
from continuum_swr_replay.table_io import write_table  # noqa: E402
from scripts.qa import sweep_memoryless_detection as sweep  # noqa: E402

BURST_DURATION_S = (0.08, 0.20, 0.50)
JUMP_CM = (0.0, 20.0, 40.0, 80.0)
SEGMENT_S = (0.12, 0.20)
CHAIN_GAP_S = (0.01, 0.03, 0.06, 0.10, 0.20)
CHAIN_PART_S = 0.15
PART_CM = 60.0

FAR_END_CM = 125.0
"""Where a jump's parts reach nearest the animal, 45 cm from where it rests,
whatever the jump: the second part starts J cm behind the first's end, and the
first starts wherever puts the far end of the two here."""

CHAIN_STARTS_CM = {"chain_continuing": (30.0, 90.0), "chain_elsewhere": (90.0, 10.0)}
"""Where each chain's two parts start; both reach 150 cm at most. Continuing: the
second picks up where the first ended. Elsewhere: the second starts 140 cm
from the first's end."""

DETECTORS = (
    "sliding_accepted",
    "sliding_published",
    "state_space_canonical",
    "state_space_continuous",
)
N_RECORDINGS = 8
"""Recordings for the no-replay control. The simulator's seed changes the
spikes, not the trajectory, so every recording pauses in the same places and
the control reads local coding at the track's end, over eight draws of it."""

OUT_PATH = (
    Path("Processed-Data/replay_detection_comparison") / "memoryless_controls.parquet"
)
SUMMARY_PATH = Path(
    "docs/design/replay_detection_comparison/detection_controls_summary.csv"
)


def usable_intervals(recording: dict) -> np.ndarray:
    """The recording's immobile stretches with its own labeled events cut out."""
    return without(
        immobile_intervals(
            recording["time"],
            recording["speed"],
            sweep.CONFIG.speed_threshold_cm_per_s,
            minimum_duration_s=1.0,
        ),
        recording["event_times"],
        margin_s=sweep.PLACEMENT_MARGIN_S,
    )


def truth_path(activity: InjectedActivity):
    """Where the injected activity says the replay is, at any time.

    Before the first sample the first represented place, after the last the
    last; inside, the represented place at the nearest sample, NaN in a
    silent gap.
    """
    time, position = activity.time_s, activity.position_cm
    finite = np.flatnonzero(np.isfinite(position))
    first, last = position[finite[0]], position[finite[-1]]

    def true_position_cm(at):
        at = np.asarray(at, dtype=np.float64)
        nearest = np.clip(np.searchsorted(time, at), 0, time.size - 1)
        previous = np.clip(nearest - 1, 0, time.size - 1)
        closer = np.where(
            np.abs(time[previous] - at) < np.abs(time[nearest] - at), previous, nearest
        )
        inside = position[closer]
        return np.where(at < time[0], first, np.where(at > time[-1], last, inside))

    return true_position_cm


def covered_cm(activity: InjectedActivity) -> float:
    """The ground the activity's parts cover together: the union of their paths."""
    spans = []
    for start_s, end_s in activity.parts:
        here = (activity.time_s >= start_s) & (activity.time_s <= end_s)
        position = activity.position_cm[here]
        position = position[np.isfinite(position)]
        spans.append((position.min(), position.max()))
    covered, reach = 0.0, -np.inf
    for low, high in sorted(spans):
        covered += max(0.0, high - max(low, reach))
        reach = max(reach, high)
    return float(covered)


def detector_events(sliding: sweep.SlidingWindowRun, state: sweep.StateSpaceRun):
    """The events each of the four scored outputs reports."""
    return {
        "sliding_accepted": [sliding.events[i] for i in sliding.accepted],
        "sliding_published": [
            sliding.events[i] for i in sliding.accepted if sliding.shuffles[i].passes
        ],
        "state_space_canonical": state.canonical,
        "state_space_continuous": state.continuous_only,
    }


def score_trial(
    activity: InjectedActivity, events: dict, state: sweep.StateSpaceRun
) -> list[dict]:
    """One row per scored output: was it called, and how were its parts carved."""
    bounds = activity.bounds
    in_trial = (state.time >= bounds[0]) & (state.time <= bounds[1])
    shares = {
        "continuous_share": float(state.p_continuous[in_trial].mean()),
        "fragmented_share": float(state.p_fragmented[in_trial].mean()),
    }
    rows = []
    for detector, reported in events.items():
        overlapping = [
            event
            for event in reported
            if sweep.overlaps((event.start_s, event.end_s), bounds)
        ]
        row = {"detector": detector, "called": bool(overlapping)} | shares
        if activity.truth == "continuous_replay":
            event_bounds = [(event.start_s, event.end_s) for event in overlapping]
            parts = list(activity.parts)
            truth = truth_path(activity)
            # right about a part: the event is the injection as a whole, and its
            # positions over that part sit on that part's path
            agreement = [
                score_against_injection(
                    event.decoded_time_s,
                    event.decoded_cm,
                    truth,
                    bounds,
                    covered_cm(activity),
                    sweep.MATCH_TOLERANCE_CM,
                ).matches
                & part_agreement(
                    event.decoded_time_s,
                    event.decoded_cm,
                    truth,
                    parts,
                    sweep.MATCH_TOLERANCE_CM,
                )
                for event in overlapping
            ]
            errors = bound_errors(event_bounds, agreement, parts)
            row |= {
                "n_parts": len(parts),
                "n_recovered": int(
                    recovered_parts(event_bounds, agreement, parts).sum()
                ),
                "segmentation": segmentation(event_bounds, agreement, parts),
                "start_errors_s": errors[:, 0].tolist(),
                "end_errors_s": errors[:, 1].tolist(),
            }
        rows.append(row)
    return rows


def jump_segments(segment_s: float, jump_cm: float) -> list[Segment]:
    """Two 60 cm parts, the second starting ``jump_cm`` behind the first's end,
    their far end at ``FAR_END_CM``."""
    first_start = FAR_END_CM - max(PART_CM, 2 * PART_CM - jump_cm)
    return [
        Segment(segment_s, first_start, PART_CM),
        Segment(segment_s, first_start + PART_CM - jump_cm, PART_CM),
    ]


def trials(means: np.ndarray, sampling_frequency: float):
    """Every injected condition: name, grid level, peak rate, builder, and pieces.

    ``pieces`` is the multi-part event's list of segments and gaps, None for a
    burst.
    """
    track_cm = sweep.TRACK_HEIGHT

    def burst(builder, duration_s, rate_hz):
        def make(at_s, seed):
            extra = {"seed": seed} if builder is jumping_burst else {}
            return builder(
                duration_s, at_s, sampling_frequency, means, rate_hz, track_cm, **extra
            )

        return make

    def replay(pieces, rate_hz):
        def make(at_s, seed):
            return piecewise_replay(pieces, at_s, sampling_frequency, means, rate_hz)

        return make

    for rate_hz in sweep.PEAK_RATE_HZ:
        for duration_s in BURST_DURATION_S:
            for name, builder in (
                ("placeless_burst", placeless_burst),
                ("jumping_burst", jumping_burst),
            ):
                yield (
                    name,
                    duration_s,
                    rate_hz,
                    burst(builder, duration_s, rate_hz),
                    None,
                )
        for segment_s in SEGMENT_S:
            for jump_cm in JUMP_CM:
                pieces = jump_segments(segment_s, jump_cm)
                yield (
                    f"jump_{1000 * segment_s:.0f}ms",
                    jump_cm,
                    rate_hz,
                    replay(pieces, rate_hz),
                    pieces,
                )
        for condition, (first, second) in CHAIN_STARTS_CM.items():
            for gap_s in CHAIN_GAP_S:
                pieces = [
                    Segment(CHAIN_PART_S, first, PART_CM),
                    Gap(gap_s),
                    Segment(CHAIN_PART_S, second, PART_CM),
                ]
                yield condition, gap_s, rate_hz, replay(pieces, rate_hz), pieces


def run_detectors(recording: dict, spike_times, activity: InjectedActivity, key: str):
    """Both detectors over the activity and a margin either side, from these spikes."""
    start_s, end_s = activity.bounds
    mask = (recording["time"] >= start_s - sweep.DECODE_MARGIN_S) & (
        recording["time"] <= end_s + sweep.DECODE_MARGIN_S
    )
    window = tuple(recording[k][mask] for k in ("time", "position", "speed"))
    sliding = sweep.run_sliding_window(recording, spike_times, window, key)
    state = sweep.run_state_space(recording, spike_times, window)
    return sliding, state


def part_alone_spikes(
    original: list[np.ndarray],
    injected: list[np.ndarray],
    part_bounds: tuple[float, float],
) -> list[np.ndarray]:
    """One part of an injected event on its own: the event's own spikes inside
    the part, the recording's own spikes everywhere else.

    Paired with the event, so a part's recovery alone and inside the event
    differ only in what surrounds it, not in a new draw of its spikes.
    """
    start_s, end_s = part_bounds
    return [
        np.sort(
            np.concatenate(
                [
                    before[(before < start_s) | (before > end_s)],
                    after[(after >= start_s) & (after <= end_s)],
                ]
            )
        )
        for before, after in zip(original, injected, strict=True)
    ]


def condition_rng(condition: str, level: float, rate_hz: float) -> np.random.Generator:
    """Placements for one condition, apart from every other's, so adding a
    condition moves no other's."""
    return np.random.default_rng(zlib.crc32(f"{condition}-{level}-{rate_hz}".encode()))


def run_trials(recording: dict, n_seeds: int) -> list[dict]:
    """Every injected condition at every seed, placed as the sweep places replays.

    Each part of a multi-part event is also decoded alone, with exactly the
    spikes it had inside the event and the recording's own spikes everywhere
    else, and its recovery alone is kept beside its recovery inside the event.
    """
    means = place_field_means(sweep.N_NEURONS, sweep.TRACK_HEIGHT)
    intervals = usable_intervals(recording)
    rows = []
    for condition, level, rate_hz, make, pieces in trials(
        means, sweep.SAMPLING_FREQUENCY
    ):
        span_s = make(0.0, 0).bounds[1]
        at_s = placements(
            intervals,
            span_s,
            sweep.PLACEMENT_MARGIN_S,
            n_seeds,
            condition_rng(condition, level, rate_hz),
        )
        for seed in range(n_seeds):
            activity = make(float(at_s[seed]), seed)
            key = f"{condition}-{level}-rate{rate_hz}-seed{seed}"
            injected = inject_activity(recording["spike_times"], activity, seed)
            sliding, state = run_detectors(recording, injected, activity, key)
            trial = score_trial(activity, detector_events(sliding, state), state)
            if pieces is not None:
                alone = np.zeros(len(DETECTORS))
                for part, ((part_start, _), segment) in enumerate(
                    zip(
                        activity.parts,
                        [piece for piece in pieces if isinstance(piece, Segment)],
                        strict=True,
                    )
                ):
                    single = piecewise_replay(
                        [segment], part_start, sweep.SAMPLING_FREQUENCY, means, rate_hz
                    )
                    sliding, state = run_detectors(
                        recording,
                        part_alone_spikes(
                            recording["spike_times"], injected, single.parts[0]
                        ),
                        single,
                        f"{key}-part{part}",
                    )
                    for index, row in enumerate(
                        score_trial(single, detector_events(sliding, state), state)
                    ):
                        alone[index] += row["n_recovered"]
                for index, row in enumerate(trial):
                    row["n_recovered_alone"] = int(alone[index])
            for row in trial:
                rows.append(
                    {
                        "condition": condition,
                        "level": level,
                        "peak_rate_hz": rate_hz,
                        "seed": seed,
                        "truth": activity.truth,
                    }
                    | row
                )
    return rows


def run_no_replay(n_recordings: int) -> list[dict]:
    """Every usable immobile stretch of each recording, decoded as it is."""
    rows = []
    for recording_seed in range(n_recordings):
        recording = sweep.build_recording(seed=recording_seed)
        for index, (start_s, end_s) in enumerate(usable_intervals(recording)):
            mask = (recording["time"] >= start_s) & (recording["time"] <= end_s)
            window = tuple(recording[k][mask] for k in ("time", "position", "speed"))
            key = f"no-replay-recording{recording_seed}-stretch{index}"
            sliding = sweep.run_sliding_window(
                recording, recording["spike_times"], window, key
            )
            state = sweep.run_state_space(recording, recording["spike_times"], window)
            for detector, reported in detector_events(sliding, state).items():
                inside = [
                    event
                    for event in reported
                    if start_s <= (event.start_s + event.end_s) / 2 <= end_s
                ]
                rows.append(
                    {
                        "condition": "no_replay",
                        "recording": recording_seed,
                        "stretch": index,
                        "exposure_s": float(end_s - start_s),
                        "detector": detector,
                        "truth": "local",
                        "n_events": len(inside),
                        "event_time_s": float(
                            sum(
                                min(event.end_s, end_s) - max(event.start_s, start_s)
                                for event in inside
                            )
                        ),
                    }
                )
    return rows


def summarize(trial_table: pd.DataFrame, no_replay: pd.DataFrame) -> pd.DataFrame:
    """The aggregates the figure and the spec quote, in one long-format table."""
    rows = []

    def add(summary, measure, condition, level, detector, value):
        rows.append(
            {
                "summary": summary,
                "measure": measure,
                "condition": condition,
                "level": level,
                "detector": detector,
                "value": float(value),
            }
        )

    exposure_min = (
        no_replay.groupby(["recording", "stretch"]).exposure_s.first().sum() / 60
    )
    add("no_replay", "immobile_minutes", "no_replay", np.nan, "", exposure_min)
    for detector, group in no_replay.groupby("detector"):
        add(
            "no_replay", "n_events", "no_replay", np.nan, detector, group.n_events.sum()
        )
        add(
            "no_replay",
            "events_per_minute",
            "no_replay",
            np.nan,
            detector,
            group.n_events.sum() / exposure_min,
        )
        add(
            "no_replay",
            "time_share",
            "no_replay",
            np.nan,
            detector,
            group.event_time_s.sum() / (60 * exposure_min),
        )

    bursts = trial_table[trial_table.condition.str.endswith("burst")]
    for (condition, level, detector), group in bursts.groupby(
        ["condition", "level", "detector"]
    ):
        add("bursts", "called_rate", condition, level, detector, group.called.mean())
        if detector == "state_space_canonical":
            for share in ("continuous_share", "fragmented_share"):
                add("bursts", share, condition, level, detector, group[share].mean())
    for (condition, detector), group in bursts.groupby(["condition", "detector"]):
        add("bursts", "called_rate", condition, np.nan, detector, group.called.mean())
        for rate_hz, at_rate in group.groupby("peak_rate_hz"):
            add(
                "bursts",
                f"called_rate_at_{rate_hz:g}hz",
                condition,
                np.nan,
                detector,
                at_rate.called.mean(),
            )

    multi = trial_table[~trial_table.condition.str.endswith("burst")]
    for (condition, level, detector), group in multi.groupby(
        ["condition", "level", "detector"]
    ):
        add(
            "parts",
            "part_recovery",
            condition,
            level,
            detector,
            (group.n_recovered / group.n_parts).mean(),
        )
        add(
            "parts",
            "part_recovery_alone",
            condition,
            level,
            detector,
            (group.n_recovered_alone / group.n_parts).mean(),
        )
        for label in SEGMENTATION_LABELS:
            add(
                "parts",
                f"segmentation_{label}",
                condition,
                level,
                detector,
                (group.segmentation == label).mean(),
            )
    for (condition, detector), group in multi.groupby(["condition", "detector"]):
        # paired by trial: pooled recovery can rise while some trials lose parts
        change = group.n_recovered - group.n_recovered_alone
        add("parts", "n_trials", condition, np.nan, detector, len(group))
        for measure, count in (
            ("n_trials_fewer_in_context", (change < 0).sum()),
            ("n_trials_more_in_context", (change > 0).sum()),
        ):
            add("parts", measure, condition, np.nan, detector, count)
    table = pd.DataFrame(rows)
    table["note"] = ""
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--n-recordings", type=int, default=N_RECORDINGS)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    started = clock.time()
    recording = sweep.build_recording()
    trial_table = pd.DataFrame(run_trials(recording, args.n_seeds))
    print(f"{len(trial_table)} trial rows in {(clock.time() - started) / 60:.1f} min")
    no_replay = pd.DataFrame(run_no_replay(args.n_recordings))
    print(f"no-replay stretches done at {(clock.time() - started) / 60:.1f} min")

    table = pd.concat([trial_table, no_replay], ignore_index=True)
    table.attrs = {
        "produced_by": "scripts/qa/sweep_memoryless_controls.py",
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": sweep.git_commit(),
        "n_seeds": args.n_seeds,
        "n_recordings": args.n_recordings,
    }
    write_table(table, args.out)
    summary = summarize(trial_table, no_replay)
    for measure in ("produced_by", "produced_at", "git_commit"):
        summary.loc[len(summary)] = {
            "summary": "provenance",
            "measure": measure,
            "value": np.nan,
            "note": str(table.attrs[measure]),
        }
    for measure in ("n_seeds", "n_recordings"):
        summary.loc[len(summary)] = {
            "summary": "provenance",
            "measure": measure,
            "value": float(table.attrs[measure]),
            "note": "",
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(f"\n  raw     -> {args.out}\n  summary -> {args.summary}")
    print(f"  total {(clock.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
