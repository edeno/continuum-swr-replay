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

All listed imports run in a single subprocess (rather than one per
import) because subprocess startup dominates the actual work —
~13 s across 13 separate tests vs ~2 s for one combined run.
Sequential ordering matters: once ``spyglass.*`` lands in
``sys.modules`` it stays, so the first leak surfaces immediately
with the offending import statement.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


# Every import statement here must complete in a fresh interpreter
# without dragging ``spyglass.*`` into ``sys.modules``. Comments after
# each block name the surface the lazy contract protects — update
# them if you move or rename a function.
_LAZY_IMPORTS: list[str] = [
    # Package roots.
    "import continuum_swr_replay",
    "from continuum_swr_replay.types import DataLoadingPreset",
    # ``data_loaders/*`` pure helpers. The Spyglass-touching loaders
    # in the same modules (``get_hpc_marks``, ``get_theta``,
    # ``get_position_info``, ``load_dios``, …) must keep their
    # spyglass imports function-local so the helpers below stay
    # offline-importable.
    "from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH",
    "from continuum_swr_replay.data_loaders import NWB_FILES",
    "from continuum_swr_replay.data_loaders.spikes import detect_coincident_spikes, filter_spike_times",
    "from continuum_swr_replay.data_loaders.lfp import get_multiunit, get_multiunit_hse, filter_ripple_lfp_time",
    "from continuum_swr_replay.data_loaders.events import convert_dio_events_to_start_stop_times",
    "from continuum_swr_replay.data_loaders.position import get_interpolated_position_info",
    # ``analysis/*`` leaf modules. The hippocampus leaves used to drag
    # ``data_aggregation`` (which imports ``get_labels`` → spyglass);
    # the PFC leaves used to be poisoned by the eager re-export from
    # ``analysis/pfc/__init__.py``.
    "from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson",
    "from continuum_swr_replay.analysis.hippocampus.event_analysis import get_event_start_end_times",
    "from continuum_swr_replay.analysis.hippocampus.event_detection import get_confident_times",
    "from continuum_swr_replay.analysis.pfc.load_population import _extract_animal",
    "from continuum_swr_replay.analysis.pfc.glm import get_spikecount_per_time_bin",
]


def test_no_lazy_import_loads_spyglass():
    """Run every contracted lazy import in one fresh interpreter; fail
    on the first one that puts ``spyglass.*`` into ``sys.modules``."""
    code = textwrap.dedent(
        """
        import sys
        imports = {imports!r}
        for stmt in imports:
            exec(stmt, {{}})
            leaked = sorted(
                m for m in sys.modules
                if m == "spyglass" or m.startswith("spyglass.")
            )
            if leaked:
                print(f"LEAKED after {{stmt!r}}: {{','.join(leaked)}}")
                sys.exit(1)
        print("OK")
        """
    ).format(imports=_LAZY_IMPORTS)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"lazy-import contract broken.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout
