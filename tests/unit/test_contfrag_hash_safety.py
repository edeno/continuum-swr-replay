"""Regression tests for ContFrag event_id / stop_id hash safety.

Without ``event_source`` / ``brain_area`` mixed into the hashes,
concatenating SWR and MUA tables (or HPC and mPFC tables) produces
silent ID collisions on every join keyed by ``event_id`` / ``stop_id``.

These tests must run offline — Spyglass is not reachable on machines
without the lab VPN. ``analysis.replay._helpers`` and
``analysis.contfrag.tables`` cannot be imported directly because both
trigger ``data_loaders/__init__.py → bandit_task → spyglass``.
Instead we load just the named function bodies via ``ast`` so the
test only needs pandas / numpy.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src" / "continuum_swr_replay"


def _load_function(
    rel_module_path: str,
    attr_name: str,
    *,
    extra_top_level_names: tuple[str, ...] = (),
):
    """Compile + exec one named top-level function from ``src/`` by path.

    Avoids triggering the package's ``__init__.py`` import chain so the
    test stays offline. The functions exercised here use only pandas
    and numpy at runtime, which we provide via the exec namespace.
    Pass ``extra_top_level_names`` to also load top-level constants
    or helper functions that the target function closes over.
    """
    src = (SRC_DIR / rel_module_path).read_text()
    tree = ast.parse(src)
    ns: dict = {"pd": pd, "np": np}

    # Pre-load any extra top-level constants / helpers the target needs.
    for extra in extra_top_level_names:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == extra
                for t in node.targets
            ):
                wrapper = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(wrapper)
                exec(compile(wrapper, rel_module_path, "exec"), ns)  # noqa: S102
                break
            if isinstance(node, ast.FunctionDef) and node.name == extra:
                wrapper = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(wrapper)
                exec(compile(wrapper, rel_module_path, "exec"), ns)  # noqa: S102
                break

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == attr_name:
            wrapper = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(wrapper)
            exec(compile(wrapper, rel_module_path, "exec"), ns)  # noqa: S102
            return ns[attr_name]
    raise AttributeError(f"{attr_name!r} not found in {rel_module_path}")


# ``_hash_contfrag_event_id`` calls ``_resolve_event_col``, which in turn
# reads ``_EVENT_SOURCE_TO_COL``. Pull both into the exec namespace.
_hash_contfrag_event_id = _load_function(
    "analysis/contfrag/tables.py",
    "_hash_contfrag_event_id",
    extra_top_level_names=("_EVENT_SOURCE_TO_COL", "_resolve_event_col"),
)
hash_stop_id = _load_function(
    "analysis/replay/_helpers.py", "hash_stop_id"
)


def _two_row_event_frame(event_col: str) -> pd.DataFrame:
    """Build a minimal frame the hash function expects.

    Same (animal, day, epoch, event_label) tuples on both rows so any
    collision under the old signature is visible.
    """
    return pd.DataFrame(
        {
            "animal": ["chimi", "chimi"],
            "day": [20200212, 20200212],
            "epoch": ["02_r1", "02_r1"],
            event_col: [3, 7],
        }
    )


class TestEventIdHashSafety:
    def test_swr_vs_mua_same_label_distinct_ids(self):
        # SWR_event = MUA_event = [3, 7] with identical (animal/day/epoch).
        swr = _two_row_event_frame("SWR_event")
        mua = _two_row_event_frame("MUA_event")
        swr_ids = _hash_contfrag_event_id(swr, event_source="SWR")
        mua_ids = _hash_contfrag_event_id(mua, event_source="MUA")
        # Element-wise: SWR(label) ≠ MUA(label).
        assert not np.any(swr_ids == mua_ids)

    def test_hpc_vs_mpfc_same_source_distinct_ids(self):
        events = _two_row_event_frame("SWR_event")
        hpc = _hash_contfrag_event_id(events, "SWR", brain_area="HPC")
        mpfc = _hash_contfrag_event_id(events, "SWR", brain_area="mPFC")
        assert not np.any(hpc == mpfc)

    def test_same_source_and_area_reproducible(self):
        events = _two_row_event_frame("SWR_event")
        a = _hash_contfrag_event_id(events, "SWR", brain_area="HPC")
        b = _hash_contfrag_event_id(events, "SWR", brain_area="HPC")
        np.testing.assert_array_equal(a, b)


class TestStopIdHashSafety:
    def _stop_idx(self) -> np.ndarray:
        return np.array([0, 1, 2], dtype=np.int64)

    def test_swr_vs_mua_distinct_ids(self):
        idx = self._stop_idx()
        swr = hash_stop_id(
            "chimi", 20200212, "02_r1", idx, brain_area="HPC", event_source="SWR"
        )
        mua = hash_stop_id(
            "chimi", 20200212, "02_r1", idx, brain_area="HPC", event_source="MUA"
        )
        assert not np.any(swr == mua)

    def test_hpc_vs_mpfc_distinct_ids(self):
        idx = self._stop_idx()
        hpc = hash_stop_id(
            "chimi", 20200212, "02_r1", idx, brain_area="HPC", event_source="SWR"
        )
        mpfc = hash_stop_id(
            "chimi", 20200212, "02_r1", idx, brain_area="mPFC", event_source="SWR"
        )
        assert not np.any(hpc == mpfc)

    def test_unsuffixed_call_preserves_legacy_hash(self):
        # NonLocal callers don't pass brain_area / event_source. Their
        # historical IDs must keep working — i.e. omitting the new
        # keywords reproduces the pre-fix hash exactly.
        idx = self._stop_idx()
        keys = pd.DataFrame(
            {
                "animal": "chimi",
                "day": 20200212,
                "epoch": "02_r1",
                "stop_idx": idx,
            }
        )
        legacy = (
            pd.util.hash_pandas_object(keys, index=False) % (2**63)
        ).to_numpy()
        new_unsuffixed = hash_stop_id("chimi", 20200212, "02_r1", idx)
        np.testing.assert_array_equal(legacy, new_unsuffixed)
