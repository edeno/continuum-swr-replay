"""The one naming rule for a decode's files (``decoding_common.decode_paths``).

The decode runners write these names and every reader builds them through the
same function, so the names are pinned here as literal strings: a change to the
rule would otherwise move writer and readers together, silently, away from the
files already on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from continuum_swr_replay.decoding_common import (
    DEFAULT_NON_LOCAL_THRESHOLD_PERCENT,
    data_over_time_suffix,
    decode_paths,
    non_local_threshold_percent,
    results_exist,
)

ROOT = Path("/decodes")


def test_the_canonical_decode_files():
    paths = decode_paths("chimi20200212_02_r1_HPC", ROOT)

    assert [str(path) for path in paths] == [
        "/decodes/chimi20200212_02_r1_HPC_80_data_over_time.pkl",
        "/decodes/chimi20200212_02_r1_HPC_results.nc",
        "/decodes/chimi20200212_02_r1_HPC_detector.pkl",
        "/decodes/chimi20200212_02_r1_HPC_environment.pkl",
    ]


def test_a_contfrag_decode_names_no_threshold():
    paths = decode_paths("j1620210706_04_r2_HPC_contfrag", ROOT, threshold_percent=None)

    assert (
        paths.data_over_time.name == "j1620210706_04_r2_HPC_contfrag_data_over_time.pkl"
    )
    assert paths.results.name == "j1620210706_04_r2_HPC_contfrag_results.nc"
    assert data_over_time_suffix(None) == "_data_over_time.pkl"


def test_a_variant_keeps_its_suffixes_before_the_file_names():
    paths = decode_paths("a20200101_02_r1_HPC_Karlsson_non_stationary_all", ROOT)

    assert paths.data_over_time.name == (
        "a20200101_02_r1_HPC_Karlsson_non_stationary_all_80_data_over_time.pkl"
    )
    assert (
        paths.detector.name
        == "a20200101_02_r1_HPC_Karlsson_non_stationary_all_detector.pkl"
    )


@pytest.mark.parametrize(
    ("threshold", "percent"),
    [(0.8, 80), (0.5, 50), (0.95, 95), (0.05, 5), (0.29, 28), (0.57, 56)],
)
def test_the_threshold_percent_truncates_as_the_decoders_wrote_it(threshold, percent):
    """0.29 * 100 is 28.999999999999996: the decoders have always written ``_28_``."""
    assert non_local_threshold_percent(threshold) == percent


def test_the_default_threshold_names_80():
    assert DEFAULT_NON_LOCAL_THRESHOLD_PERCENT == 80
    assert data_over_time_suffix() == "_80_data_over_time.pkl"
    assert data_over_time_suffix(50) == "_50_data_over_time.pkl"


def test_results_exist_reads_the_three_saved_files(tmp_path):
    epoch_id = "a20200101_02_r1_HPC"
    paths = decode_paths(epoch_id, tmp_path)
    for path in (paths.results, paths.detector):
        path.touch()

    assert not results_exist(tmp_path, epoch_id, overwrite=False)
    paths.environment.touch()
    assert results_exist(tmp_path, epoch_id, overwrite=False)
    assert not results_exist(tmp_path, epoch_id, overwrite=True)
    assert not paths.data_over_time.exists()
