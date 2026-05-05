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
