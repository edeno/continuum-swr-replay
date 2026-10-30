"""The bundle-tree comparison pairs only the classes both trees fitted."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "pfc"
    / "compare_pfc_marked_family_trees.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "compare_pfc_marked_family_trees", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare)

TARGET = "ripple_power_z"
CLASSES = ("non_local_no_SWR", "non_local_SWR", "SWR_no_non_local")


def _bundle(directory: Path, fitted_classes: tuple[str, ...], offset: float) -> Path:
    """Write one two-unit bundle that carries curves only for `fitted_classes`."""
    directory.mkdir()
    artifact = directory / "rat20200101_01_r1_mPFC_ripple_power_marked_family.pkl"
    stats = {
        f"{event_class}:{TARGET}: heldout_delta_deviance": [1.0 + offset, 2.0]
        for event_class in fitted_classes
    }
    pd.DataFrame(stats).to_pickle(artifact)
    for event_class in fitted_classes:
        curve = np.full((5, 2), offset)
        np.save(artifact.with_suffix(f".{event_class}.{TARGET}.slope.npy"), curve)
        np.save(artifact.with_suffix(f".{event_class}.typical.npy"), curve)
    sidecar = {
        "extra": {
            "nwb_file_name": "rat20200101_.nwb",
            "epoch_name": "01_r1",
            "model_name": "ripple_power",
            "target_classes": list(CLASSES),
            "fitted_event_classes": list(fitted_classes),
            "kept_unit_indices": [0, 1],
            "marked_family_diagnostics": {"scientific_marks": [TARGET]},
        }
    }
    artifact.with_suffix(".provenance.json").write_text(json.dumps(sidecar))
    return artifact


def test_a_class_the_reference_dropped_is_not_compared(tmp_path):
    # The reference session dropped the SWRs without replay (too few events);
    # the candidate fitted them.
    reference = _bundle(tmp_path / "reference", CLASSES[:2], offset=0.0)
    candidate = _bundle(tmp_path / "candidate", CLASSES, offset=0.5)

    rows = compare.compare_pair(reference, candidate, np.array([5.0, 5.0]), 1.0)

    compared = {row["event_class"] for row in rows}
    assert compared == set(CLASSES[:2])
    slope_rows = [row for row in rows if row["target"] == TARGET]
    assert all(row["slope_max_abs_diff"] == 0.5 for row in slope_rows)
