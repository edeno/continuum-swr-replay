"""Regression tests for the lazy-import contract.

The package's two top-level ``__init__.py`` files re-export functions
from submodules that pull in ``spyglass.common`` at module top level
(opening a DataJoint connection at import time). Those re-exports are
lazy via PEP 562 ``__getattr__`` so that pure-analysis leaf imports
(e.g. ``continuum_swr_replay.analysis.replay.fitting``) finish without
touching Spyglass — keeping the offline test slice fast.

A regression here would re-introduce the hang that
``timeout 8 uv run python -c "from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson"``
used to hit on no-VPN machines.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _import_does_not_load_spyglass(import_stmt: str) -> bool:
    """Run an isolated subprocess and confirm spyglass is not loaded."""
    code = textwrap.dedent(
        f"""
        import sys
        {import_stmt}
        loaded = [m for m in sys.modules if m == "spyglass" or m.startswith("spyglass.")]
        if loaded:
            print("LOADED:", ",".join(sorted(loaded)))
            sys.exit(1)
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.returncode == 0 and "OK" in result.stdout


class TestLazyImports:
    def test_analysis_replay_fitting_does_not_load_spyglass(self):
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.analysis.replay.fitting "
            "import fit_conditional_poisson"
        )

    def test_types_module_does_not_load_spyglass(self):
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.types import DataLoadingPreset"
        )

    def test_data_loaders_constants_does_not_load_spyglass(self):
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.data_loaders.constants "
            "import TRACK_SEGMENT_TO_PATCH"
        )

    def test_data_loaders_package_constants_does_not_load_spyglass(self):
        # The package re-export of constants must stay eager-but-leaf.
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.data_loaders import NWB_FILES"
        )

    def test_package_root_does_not_load_spyglass(self):
        assert _import_does_not_load_spyglass("import continuum_swr_replay")

    def test_hippocampus_event_analysis_does_not_load_spyglass(self):
        # Importing a leaf hippocampus submodule used to drag in
        # ``data_aggregation`` (which imports
        # ``data_loaders.get_labels`` → spyglass).
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.analysis.hippocampus.event_analysis "
            "import get_event_start_end_times"
        )

    def test_hippocampus_event_detection_does_not_load_spyglass(self):
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.analysis.hippocampus.event_detection "
            "import get_confident_times"
        )

    def test_data_loaders_spikes_pure_helpers_do_not_load_spyglass(self):
        # The spike-filtering / coincident-detection helpers used by
        # tests/unit/test_spikes.py are pure numpy/pandas. The DB-loading
        # functions in the same module (``get_hpc_marks``, etc.) must
        # keep their spyglass imports function-local so the unit suite
        # can import these helpers offline.
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.data_loaders.spikes "
            "import detect_coincident_spikes, filter_spike_times"
        )

    def test_data_loaders_lfp_pure_helpers_do_not_load_spyglass(self):
        # ``get_multiunit_hse`` / ``get_multiunit`` / ``filter_ripple_lfp_time``
        # are pure helpers; only ``get_theta`` / ``get_ripple`` need
        # spyglass at runtime. ``test_lfp.py`` and
        # ``test_multiunit_hse.py`` should be importable offline.
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.data_loaders.lfp "
            "import get_multiunit, get_multiunit_hse, filter_ripple_lfp_time"
        )

    def test_pfc_load_population_does_not_load_spyglass(self):
        # ``_extract_animal`` and the rest of ``load_population`` are
        # pure pandas; only ``run_session`` in ``glm`` needs Spyglass.
        # The eager re-export from ``analysis/pfc/__init__.py``
        # shouldn't drag DataJoint into a leaf import.
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.analysis.pfc.load_population "
            "import _extract_animal"
        )

    def test_pfc_glm_pure_helpers_do_not_load_spyglass(self):
        # ``get_spikecount_per_time_bin`` and friends are pure helpers
        # used by ``tests/unit/test_spike_binning.py``.
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.analysis.pfc.glm "
            "import get_spikecount_per_time_bin"
        )

    def test_data_loaders_events_pure_helpers_do_not_load_spyglass(self):
        # ``convert_dio_events_to_start_stop_times`` is pure pandas;
        # the Spyglass ``DIOEvents`` table is only needed by
        # ``load_dios`` / ``_load_dio_event_type``. Without lazy
        # imports the helper drags ``spyglass.common`` into every
        # offline test that touches DIO conversion.
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.data_loaders.events "
            "import convert_dio_events_to_start_stop_times"
        )

    def test_data_loaders_position_pure_helpers_do_not_load_spyglass(self):
        # ``get_interpolated_position_info`` is pure
        # ``track_linearization`` + pandas; the Spyglass tables and
        # ``DataJointError`` are only needed by ``get_position_info``.
        assert _import_does_not_load_spyglass(
            "from continuum_swr_replay.data_loaders.position "
            "import get_interpolated_position_info"
        )
