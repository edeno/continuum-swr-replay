"""DIAGNOSTIC, not part of the analysis: is the fold fit reproducible within one
process on this machine?

Two tests of ``tests/integration/test_nonlocal_restriction_simulated.py`` that
require bit-for-bit equality failed on GitHub's runners at commit d2d3df23 and
pass on the development machine. This script repeats what they compare, on the
same simulated recording, and says which repeat differs:

1. the five folds are fit and scored in order, as ``fold_fit_scores`` does;
2. each fold's fitted detector is fingerprinted then and again at the end
   (was a detector changed after its pass?);
3. folds 0 and 1 are scored again from their detectors (the first test);
4. fold 0 is fit again with the same spikes, twice, and with its own spikes
   replaced (the second test);
5. one likelihood and one scoring are repeated back to back.

Usage: fold_fit_repeats.py <label>   (writes diagnostic_<label>.json)
"""

import hashlib
import json
import os
import platform
import sys
import time as clock

import numpy as np

sys.path.insert(0, os.getcwd())

from continuum_swr_replay.analysis.event_evidence import (  # noqa: E402
    nonlocal_restriction as nr,
)
from continuum_swr_replay.analysis.event_evidence import (  # noqa: E402
    tetrode_holdout as th,
)
from tests.integration import test_nonlocal_restriction_simulated as t  # noqa: E402
from tests.support import tetrode_holdout_simulation as sim  # noqa: E402

REPORT: dict = {}


def digest(array) -> str:
    array = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def fingerprint(obj, path="detector", depth=0, seen=None) -> dict:
    """A digest of every array reachable from an object's attributes."""
    seen = set() if seen is None else seen
    out: dict = {}
    if id(obj) in seen or depth > 6:
        return out
    seen.add(id(obj))
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        try:
            out[path] = digest(obj)
        except Exception as error:  # noqa: BLE001
            out[path] = f"unreadable: {error!r}"
    elif isinstance(obj, dict):
        for key, value in obj.items():
            out.update(fingerprint(value, f"{path}[{key!r}]", depth + 1, seen))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            out.update(fingerprint(value, f"{path}[{i}]", depth + 1, seen))
    elif isinstance(obj, (int, float, str, bool, type(None))):
        out[path] = repr(obj)
    elif hasattr(obj, "__dict__") and not type(obj).__module__.startswith("networkx"):
        for key, value in vars(obj).items():
            out.update(fingerprint(value, f"{path}.{key}", depth + 1, seen))
    return out


def compare(label: str, a, b) -> None:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        line = {"shapes": [list(a.shape), list(b.shape)]}
    else:
        differ = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        line = {
            "n_differ": int(differ.sum()),
            "of": int(a.size),
            "max_abs": float(np.abs(a[differ].astype(np.float64) - b[differ]).max())
            if differ.any()
            else 0.0,
        }
    REPORT[label] = line
    print(f"{label}: {line}", flush=True)


def main() -> None:
    label = sys.argv[1]
    with open("/proc/cpuinfo") as handle:
        cpuinfo = handle.read()
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    flags = next(
        (line for line in cpuinfo.splitlines() if line.startswith("flags")), ""
    )
    REPORT["machine"] = {
        "label": label,
        "cpu": model,
        "cpus_usable": len(os.sched_getaffinity(0)),
        "avx2": " avx2" in flags,
        "avx512f": " avx512f" in flags,
        "fma": " fma" in flags,
        "python": platform.python_version(),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
    }
    print(REPORT["machine"], flush=True)

    recording = sim.simulate_recording(
        kinds=("sweep", "stationary", "local_burst"),
        n_laps=4,
        boundary_event=False,
        seed=2,
    )
    REPORT["recording"] = {
        "time": digest(recording.time),
        "position": digest(recording.position),
        "spike_times": [digest(s) for s in recording.spike_times],
        "marks": [digest(m) for m in recording.marks],
        "n_spikes": [int(s.size) for s in recording.spike_times],
    }
    print("recording", REPORT["recording"], flush=True)
    canonical = sim.fit_detector(recording)
    starts, classes = sim.trials(recording)
    folds = th.trial_folds(recording.time, starts, classes, nr.N_FOLDS)
    margin = th.margin_in_bins(th.TRIAL_MARGIN_S, recording.time)
    remote = recording.events[~recording.events.kind.isin(t.NOT_REMOTE)]
    in_swr = t._event_mask(recording, remote)
    is_missing = np.zeros(recording.time.size, dtype=bool)
    is_missing[:150] = True
    for k in (0, 3):
        start = np.flatnonzero(folds == k)[0]
        is_missing[start + 200 : start + 350] = True
    is_training = recording.is_running & ~is_missing
    masks = th.fold_training_masks(is_training, folds, nr.N_FOLDS, margin)
    position = recording.position[:, None]
    REPORT["folds"] = digest(folds)

    def fit(k, spikes=None, marks=None):
        t0 = clock.time()
        detector, record = nr.fit_fold_detector(
            canonical,
            recording.time,
            position,
            recording.spike_times if spikes is None else spikes,
            recording.marks if marks is None else marks,
            masks[k],
            is_missing | (folds == k),
        )
        print(
            f"  fold {k} fit in {clock.time() - t0:.0f} s; trace {record['marginal_log_likelihoods']}",
            flush=True,
        )
        return detector, record

    def likelihood(detector):
        return nr.fold_log_likelihood(
            detector,
            recording.time,
            position,
            recording.spike_times,
            recording.marks,
            is_missing,
        )

    def scores(detector, lik, k):
        return nr.model_scores(
            detector, lik, in_swr, summary_bins=folds == k, log=sim.quiet
        )[0]

    # 1. the five folds in order, as fold_fit_scores runs them
    first = []
    for k in range(nr.N_FOLDS):
        detector, record = fit(k)
        lik = likelihood(detector)
        first.append(
            {
                "detector": detector,
                "transitions": np.array(detector.discrete_state_transitions_),
                "initial": np.array(detector.initial_conditions_),
                "trace": list(record["marginal_log_likelihoods"]),
                "likelihood": np.asarray(lik) if k < 2 else None,
                "likelihood_digest": digest(lik),
                "scores": {
                    name: np.asarray(v) for name, v in scores(detector, lik, k).items()
                },
                "fingerprint": fingerprint(detector),
            }
        )
        del lik
    REPORT["first_pass"] = [
        {
            "transitions": f["transitions"].tolist(),
            "trace": f["trace"],
            "likelihood_digest": f["likelihood_digest"],
            "scores_digest": {name: digest(v) for name, v in f["scores"].items()},
        }
        for f in first
    ]

    # 2. was a detector changed after its pass?
    for k, f in enumerate(first):
        now = fingerprint(f["detector"])
        changed = sorted(
            key
            for key in set(now) | set(f["fingerprint"])
            if now.get(key) != f["fingerprint"].get(key)
        )
        REPORT[f"fold {k} detector attributes changed since its pass"] = changed
        print(
            f"fold {k} detector attributes changed since its pass: {changed}",
            flush=True,
        )

    # 3. the first test: folds 0 and 1 scored again from their detectors
    for k in (0, 1):
        f = first[k]
        lik = likelihood(f["detector"])
        compare(f"fold {k} likelihood, again vs first pass", lik, f["likelihood"])
        again = scores(f["detector"], lik, k)
        mine = folds == k
        for name, values in again.items():
            compare(
                f"fold {k} scores[{name}] own rows, again vs first pass",
                np.asarray(values)[mine],
                f["scores"][name][mine],
            )
        on_first = scores(f["detector"], f["likelihood"], k)
        differing = [
            name
            for name in on_first
            if not np.array_equal(
                np.asarray(on_first[name]), f["scores"][name], equal_nan=True
            )
        ]
        REPORT[
            f"fold {k} scores on the first pass's own likelihood: columns that differ"
        ] = differing
        print(
            f"fold {k} scores on the first pass's own likelihood: columns that differ: {differing}",
            flush=True,
        )
        del lik

    # 4. the second test: fold 0 fit again
    for attempt in (1, 2):
        detector, record = fit(0)
        compare(
            f"fold 0 refit {attempt} (same spikes) transitions vs first pass",
            detector.discrete_state_transitions_,
            first[0]["transitions"],
        )
        compare(
            f"fold 0 refit {attempt} (same spikes) initial conditions vs first pass",
            detector.initial_conditions_,
            first[0]["initial"],
        )
        compare(
            f"fold 0 refit {attempt} (same spikes) trace vs first pass",
            record["marginal_log_likelihoods"],
            first[0]["trace"],
        )
        REPORT[f"fold 0 refit {attempt} (same spikes) transitions"] = np.asarray(
            detector.discrete_state_transitions_
        ).tolist()
    spike_times, marks = t._fold_spikes_replaced(recording, folds == 0)
    detector, record = fit(0, spike_times, marks)
    compare(
        "fold 0 refit (its spikes replaced) transitions vs first pass",
        detector.discrete_state_transitions_,
        first[0]["transitions"],
    )
    compare(
        "fold 0 refit (its spikes replaced) initial conditions vs first pass",
        detector.initial_conditions_,
        first[0]["initial"],
    )
    compare(
        "fold 0 refit (its spikes replaced) trace vs first pass",
        record["marginal_log_likelihoods"],
        first[0]["trace"],
    )
    REPORT["fold 0 refit (its spikes replaced) transitions"] = np.asarray(
        detector.discrete_state_transitions_
    ).tolist()

    # 5. back-to-back repeats
    one, two = likelihood(first[0]["detector"]), likelihood(first[0]["detector"])
    compare("fold 0 likelihood, two calls back to back", one, two)
    a, b = scores(first[0]["detector"], one, 0), scores(first[0]["detector"], one, 0)
    differing = [
        name
        for name in a
        if not np.array_equal(np.asarray(a[name]), np.asarray(b[name]), equal_nan=True)
    ]
    REPORT[
        "fold 0 scores, two calls back to back on one likelihood: columns that differ"
    ] = differing
    print(
        f"fold 0 scores, two calls back to back: columns that differ: {differing}",
        flush=True,
    )
    finite = np.asarray(one)[np.isfinite(np.asarray(one))]
    REPORT["likelihood range"] = [float(finite.min()), float(finite.max())]

    with open(f"diagnostic_{label}.json", "w") as handle:
        json.dump(REPORT, handle, indent=1)
    print(f"wrote diagnostic_{label}.json", flush=True)


if __name__ == "__main__":
    main()
