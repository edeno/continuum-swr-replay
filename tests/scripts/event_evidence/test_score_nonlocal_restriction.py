"""The non-local restriction runner's wiring: what ``run_epoch`` hands the
scores, how it labels and sums the bins, and what it writes.

``epoch_inputs``, the SWR source and ``fold_fit_scores`` are replaced at
their boundaries; the frame, the SWR events and the scores are made by hand.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import nonlocal_restriction as nr
from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.decoding.files import DecodePaths
from continuum_swr_replay.runs.io import output_path
from scripts.event_evidence import score_nonlocal_restriction as runner
from scripts.event_evidence._per_event_run import EpochInputs

EPOCH = "wilbur20210329_02_r1_HPC"
N = 12
TIME = 100.0 + 0.002 * np.arange(N)
# Three source SWRs: bins 1-2 (immobile, labeled in the frame), bins 5-6
# (moving, one bin without a speed; not labeled, as a moving SWR is not) and
# bin 11 (in excluded time; not labeled).
EVENTS = pd.DataFrame({"start_time": TIME[[1, 5, 11]], "end_time": TIME[[2, 6, 11]]})
IN_SWR = np.isin(np.arange(N), [1, 2, 5, 6, 11])
CATEGORY = [
    "immobile_outside_swr",
    "immobile_in_swr",
    "immobile_in_swr",
    "immobile_outside_swr",
    "moving",  # 10 cm/s
    "moving",  # 10 cm/s, inside an SWR
    "moving",  # no speed, inside an SWR
    "immobile_outside_swr",
    "immobile_outside_swr",
    "immobile_outside_swr",
    "immobile_outside_swr",
    "excluded",
]
# Six trials: departures at bins 0, 2, 4, 6, 8 and 10, to wells 2, 3, 2, 3, 2, 3,
# each ending at the next departure and the last at bin 11, so bin 11 is the
# time after the last trial.
TRIALS = pd.DataFrame(
    {
        "movement_start_time": TIME[[0, 2, 4, 6, 8, 10]],
        "end_time": TIME[[2, 4, 6, 8, 10, 11]],
        "to_well": [2, 3, 2, 3, 2, 3],
    }
)
# Seconds of training data at the animal's place: bins 0 and 3 none, bin 7
# 4 s, the other immobile bins 30 s; NaN at the excluded bin.
SAMPLING = np.full(N, 30.0, dtype=np.float32)
SAMPLING[[0, 3]] = 0.0
SAMPLING[7] = 4.0
SAMPLING[11] = np.nan


def _frame(swr_label=None) -> pd.DataFrame:
    speed = np.ones(N)
    speed[[4, 5]] = 10.0
    speed[6] = np.nan
    label = np.zeros(N, dtype=np.int64)
    label[[1, 2]] = 1
    if swr_label is not None:
        label = swr_label
    return pd.DataFrame(
        {
            "head_speed": speed,
            "in_excluded_time": np.arange(N) == 11,
            "head_position_x": np.arange(N, dtype=float),
            "head_position_y": -np.arange(N, dtype=float),
            "linear_position": 2.0 * np.arange(N, dtype=float),
            "SWR_event": label,
        },
        index=pd.Index(TIME, name="time"),
    )


def _scores() -> pd.DataFrame:
    """``none`` -1 everywhere; ``full`` gains 2 nats in each immobile bin
    inside an SWR, 1 in bin 7 (immobile, outside) and 3 in bin 4 (moving);
    ``swr`` -0.5 inside SWRs. Two-sided scores equal the filter scores."""
    none = -np.ones(N)
    full = none.copy()
    full[[1, 2]] += 2.0
    full[7] += 1.0
    full[4] += 3.0
    swr = np.where(IN_SWR, -0.5, none)
    columns = {"none": none, "swr": swr, "full": full}
    return pd.DataFrame(
        {
            "time": TIME,
            **{c: columns[c].astype(np.float32) for c in ("none", "swr", "full")},
            **{
                f"{c}_two_sided": columns[c].astype(np.float32)
                for c in ("none", "swr", "full")
            },
            "in_swr": IN_SWR,
            "block": np.r_[np.zeros(N - 3), np.ones(3)].astype(np.int64),
            "place_sampling_s": SAMPLING,
        }
    )


def _record(**changes) -> dict:
    record = {"continuous_block_deviation": 1e-15, "n_in_swr": int(IN_SWR.sum())}
    for model in nr.MODELS:
        record |= {
            f"filter_identity_deviation_{model}": 1e-7,
            f"marginal_log_likelihood_{model}": -12.0,
            f"marginal_difference_{model}": 0.0,
            f"causal_mass_deviation_{model}": 1e-7,
            f"predicted_mass_deviation_{model}": 1e-7,
        }
    return record | changes


@pytest.fixture
def patched(tmp_path, monkeypatch):
    """Patch the runner's boundaries; return a function that runs the epoch
    with a given frame and record and returns its record and output folder."""
    (tmp_path / "dot.pkl").write_bytes(b"dot")
    (tmp_path / "detector.pkl").write_bytes(b"detector")
    paths = DecodePaths(
        tmp_path / "dot.pkl",
        tmp_path / "results.nc",
        tmp_path / "detector.pkl",
        tmp_path / "environment.pkl",
    )
    monkeypatch.setattr(
        runner.swr_source,
        "epoch_source",
        lambda epoch_id: SimpleNamespace(events=EVENTS),
    )
    monkeypatch.setattr(
        runner.swr_source,
        "swr_source_files",
        lambda epoch_id: {
            "events": {"path": "events.parquet", "sha256": "e"},
            "observed": {"path": "observed.parquet", "sha256": "o"},
        },
    )

    def run(frame, record, fold_design="trials"):
        loaded = {
            "spike_times": [np.array([100.001])],
            "marks": [np.ones((1, 4))],
            "is_training": np.ones(N, dtype=bool),
            "is_missing": np.zeros(N, dtype=bool),
        }
        detector = object()

        def epoch_inputs(epoch_id, data_dir, *, trials):
            assert trials == (fold_design == "trials")
            return EpochInputs(
                paths,
                frame,
                frame.index.to_numpy(),
                loaded | ({"trials": TRIALS} if trials else {}),
                detector,
            )

        monkeypatch.setattr(runner, "epoch_inputs", epoch_inputs)

        def fold_fit_scores(
            d,
            time,
            position,
            spikes,
            marks,
            training,
            missing,
            in_swr,
            *,
            folds,
            margin_bins,
            linear_position,
            estimate_encoding_model,
            log,
        ):
            assert estimate_encoding_model is nr.FOLD_FIT_ESTIMATE_ENCODING_MODEL
            if fold_design == "trials":
                np.testing.assert_array_equal(
                    folds,
                    th.trial_folds(
                        TIME,
                        TIME[[0, 2, 4, 6, 8, 10, 11]],
                        np.r_[TRIALS["to_well"], th.AFTER_LAST_TRIAL],
                        nr.N_FOLDS,
                    ),
                )
                assert margin_bins == 500  # 1 s of 2 ms bins
            else:
                assert folds is None and margin_bins == th.MARGIN_BINS
            np.testing.assert_array_equal(linear_position, 2.0 * np.arange(N))
            assert d is detector
            np.testing.assert_array_equal(time, TIME)
            np.testing.assert_array_equal(
                position, frame[["head_position_x", "head_position_y"]].to_numpy()
            )
            assert spikes is loaded["spike_times"] and marks is loaded["marks"]
            assert training is loaded["is_training"]
            assert missing is loaded["is_missing"]
            np.testing.assert_array_equal(in_swr, IN_SWR)
            return _scores(), {
                "folds": [{"fold": 0, "converged": True, "em_iterations": 4, **record}],
                "fit_s": 1.0,
                "n_in_swr": int(IN_SWR.sum()),
            }

        monkeypatch.setattr(runner.nr, "fold_fit_scores", fold_fit_scores)
        output = tmp_path / "out"
        return runner.run_epoch(EPOCH, tmp_path, output, fold_design), output

    return run


def test_run_epoch_labels_sums_and_records_the_epoch(patched):
    record, output = patched(_frame(), _record())
    assert record["status"] == "ok" and "error" not in record

    bins = pd.read_parquet(output_path(EPOCH, output, runner.BINS_NAME))
    assert list(bins["category"].astype(str)) == CATEGORY
    assert bins["place_sampling_s"].dtype == np.float32
    np.testing.assert_array_equal(bins["in_swr"], IN_SWR)

    sums = pd.read_parquet(output_path(EPOCH, output, runner.OUTPUT_NAME))
    sums = sums.set_index("category")
    assert sums["n_bins"].to_dict() == {
        "excluded": 1,
        "immobile_in_swr": 2,
        "immobile_outside_swr": 6,
        "moving": 3,
    }
    assert sums.loc["immobile_in_swr", "full"] == 2.0
    assert sums.loc["immobile_in_swr", "swr"] == -1.0
    assert sums.loc["immobile_outside_swr", "full"] == -5.0
    assert sums.loc["moving", "full_two_sided"] == 0.0
    assert set(sums["epoch_id"]) == {EPOCH} and set(sums["animal"]) == {"wilbur"}

    # full - none: 4 nats inside SWRs, 1 outside, over the immobile bins
    assert record["share_outside_swr"] == pytest.approx(0.2)
    assert record["share_outside_swr_two_sided"] == pytest.approx(0.2)
    assert record["n_bins"] == N
    assert record["n_bins_by_category"] == sums["n_bins"].to_dict()
    assert record["n_analyzed_without_speed"] == 1
    assert record["n_swr_source_events"] == 3
    assert record["n_in_swr"] == 5
    assert record["n_in_swr_label"] == 2
    assert record["n_immobile_in_swr_label"] == 2
    assert set(record["epoch_inputs"]) == {
        "data_over_time",
        "detector",
        "swr_events",
        "swr_observed",
    }
    # the folds: seven segments, 9 and 3 of the 12 bins in the two folds the
    # scores name; of the 8 immobile bins, 2 had no training data at the
    # animal's place and 3 had 10 s or less
    assert record["fold_design"] == "trials"
    assert record["margin_bins"] == 500
    assert record["n_segments"] == 7  # six trials and the time after them
    assert record["n_groups"] == 7
    assert record["trial_bin_share"] == pytest.approx(11 / 12)
    assert record["fold_share_min"] == 0.0
    assert record["fold_share_max"] == pytest.approx(0.75)
    assert record["immobile_share_place_sampling_0s"] == pytest.approx(2 / 8)
    assert record["immobile_share_place_sampling_le_10s"] == pytest.approx(3 / 8)
    assert record["n_immobile_without_place"] == 0
    json.dumps(record, allow_nan=False)


def test_the_contiguous_design_passes_no_folds(patched):
    record, _ = patched(_frame(), _record(), fold_design="contiguous")
    assert record["status"] == "ok"
    assert record["fold_design"] == "contiguous"
    assert record["margin_bins"] == th.MARGIN_BINS
    assert record["n_segments"] is None and record["n_groups"] is None
    assert record["trial_bin_share"] is None


def test_a_failed_check_is_recorded_with_its_message(patched):
    record, output = patched(_frame(), _record(predicted_mass_deviation_full=1e-3))
    assert record["status"] == "failed_check"
    assert "full: predictive posterior mass" in record["error"]
    # the tables are still written, for diagnosis
    assert output_path(EPOCH, output, runner.BINS_NAME).exists()


def test_a_frame_swr_outside_the_source_raises(patched):
    label = np.zeros(N, dtype=np.int64)
    label[[1, 2]] = 1
    label[8] = 2  # immobile, but in no source SWR
    with pytest.raises(ValueError, match="outside every SWR of the source"):
        patched(_frame(label), _record())


def test_the_fold_fit_scores_and_checks_every_fold(patched):
    """The epoch is scored by ``fold_fit_scores`` and fails its check when any
    fold's pass fails one, naming the fold."""
    record, output = patched(_frame(), _record())
    assert record["status"] == "ok" and record["fit"] == "folds"
    assert record["folds"][0]["fold"] == 0
    assert output_path(EPOCH, output, runner.BINS_NAME).exists()
    record, _ = patched(_frame(), _record(causal_mass_deviation_none=1.0))
    assert record["status"] == "failed_check"
    assert record["error"].startswith("fold 0: none: filtered posterior mass")
    record, _ = patched(_frame(), _record(converged=False))
    assert record["status"] == "failed_check"
    assert record["error"] == "fold 0: EM did not converge in 4 iterations"


def test_the_header_names_the_fit():
    """The header's settings name the fit and its EM settings, as the finished
    fold-fit run's do, so its records still gather and resume, and a run of
    the earlier design (no fit setting) is not gathered with them."""
    assert runner.fit_settings() == {
        "fit": "folds",
        "em": nr.em_settings(nr.FOLD_FIT_ESTIMATE_ENCODING_MODEL),
    }


def test_the_run_writes_to_the_canonical_folder(tmp_path, monkeypatch):
    """The gather reads and writes ``EVENT_NONLOCAL_RESTRICTION_DIR``, with
    ``"fit": "folds"`` and the EM settings after ``n_folds`` in the settings."""
    gathered = {}
    monkeypatch.setattr(
        runner,
        "gather_manifest",
        lambda output, name, header: gathered.update(output=output, header=header),
    )
    runner.main(["--gather", "--data-dir", str(tmp_path)])
    assert gathered["output"] == tmp_path / "event_nonlocal_restriction"
    assert list(gathered["header"]["settings"])[:4] == [
        "n_folds",
        "fit",
        "em",
        "fold_design",
    ]
