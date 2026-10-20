"""Tests for data_loaders/utils.py functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders.utils import (
    DEFAULT_SWR_DEFINITION,
    format_epoch_identifier,
    get_labels,
    parse_epoch_identifier,
    parse_nwb_filename,
)


class TestGetLabels:
    """Tests for event labeling function."""

    def test_single_event(self):
        """Test labeling a single event."""
        # Create time index from 0 to 10 seconds
        time = pd.Index(np.linspace(0, 10, 101))

        # Single event from 3 to 7 seconds
        events = pd.DataFrame({"start_time": [3.0], "end_time": [7.0]})

        result = get_labels(events, time)

        # Check that event is labeled with 1
        assert result.loc[3.0:7.0, "event_number"].eq(1).all()

        # Check that times outside event are labeled 0
        assert result.loc[0.0:2.9, "event_number"].eq(0).all()
        assert result.loc[7.1:10.0, "event_number"].eq(0).all()

    def test_multiple_non_overlapping_events(self):
        """Test labeling multiple non-overlapping events."""
        time = pd.Index(np.linspace(0, 20, 201))

        # Three events
        events = pd.DataFrame(
            {
                "start_time": [2.0, 8.0, 15.0],
                "end_time": [4.0, 10.0, 17.0],
            }
        )

        result = get_labels(events, time)

        # Check each event has correct label (1, 2, 3)
        assert result.loc[2.0:4.0, "event_number"].eq(1).all()
        assert result.loc[8.0:10.0, "event_number"].eq(2).all()
        assert result.loc[15.0:17.0, "event_number"].eq(3).all()

        # Check gaps are labeled 0
        assert result.loc[0.0:1.9, "event_number"].eq(0).all()
        assert result.loc[4.1:7.9, "event_number"].eq(0).all()
        assert result.loc[10.1:14.9, "event_number"].eq(0).all()
        assert result.loc[17.1:20.0, "event_number"].eq(0).all()

    def test_empty_events(self):
        """Test with no events."""
        time = pd.Index(np.linspace(0, 10, 101))

        # Empty events DataFrame
        events = pd.DataFrame({"start_time": [], "end_time": []})

        result = get_labels(events, time)

        # All labels should be 0
        assert result["event_number"].eq(0).all()

    def test_event_at_time_boundaries(self):
        """Test event that spans entire time range."""
        time = pd.Index(np.linspace(0, 10, 11))

        # Event covers entire time range
        events = pd.DataFrame({"start_time": [0.0], "end_time": [10.0]})

        result = get_labels(events, time)

        # All time points should be labeled 1
        assert result["event_number"].eq(1).all()

    def test_labels_are_sequential(self):
        """Test that event labels are sequential integers starting from 1."""
        time = pd.Index(np.linspace(0, 50, 501))

        # Create 5 events
        events = pd.DataFrame(
            {
                "start_time": [1.0, 11.0, 21.0, 31.0, 41.0],
                "end_time": [3.0, 13.0, 23.0, 33.0, 43.0],
            }
        )

        result = get_labels(events, time)

        # Check that unique labels are [0, 1, 2, 3, 4, 5]
        unique_labels = sorted(result["event_number"].unique())
        assert unique_labels == [0, 1, 2, 3, 4, 5]

    def test_output_structure(self):
        """Test that output has correct structure."""
        time = pd.Index(np.linspace(0, 10, 101))
        events = pd.DataFrame({"start_time": [3.0], "end_time": [7.0]})

        result = get_labels(events, time)

        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(time)
        assert result.columns.tolist() == ["event_number"]
        assert result.index.equals(time)
        assert result["event_number"].dtype == int

    def test_short_events(self):
        """Test labeling very short events."""
        time = pd.Index(np.arange(0, 10, 0.001))  # 1ms resolution

        # Very short events (10ms each)
        events = pd.DataFrame(
            {
                "start_time": [1.0, 5.0],
                "end_time": [1.01, 5.01],
            }
        )

        result = get_labels(events, time)

        # Count labeled time points for each event
        n_event1 = result.loc[1.0:1.01, "event_number"].eq(1).sum()
        n_event2 = result.loc[5.0:5.01, "event_number"].eq(2).sum()

        # Should have at least a few time points labeled
        assert n_event1 > 0
        assert n_event2 > 0

    def test_adjacent_events(self):
        """Test events that are adjacent (end of one = start of next)."""
        time = pd.Index(np.linspace(0, 10, 101))

        # Adjacent events
        events = pd.DataFrame(
            {
                "start_time": [2.0, 5.0],
                "end_time": [5.0, 8.0],
            }
        )

        result = get_labels(events, time)

        # Both events should be labeled
        # Note: t=5.0 may be labeled as either event 1 or 2 (edge case)
        assert result.loc[2.0:4.9, "event_number"].eq(1).all()
        assert result.loc[5.1:8.0, "event_number"].eq(2).all()


class TestParseNwbFilename:
    """Tests for NWB filename parsing.

    The ``j16`` case is the historical-bug anchor: the legacy
    ``str.split("2")`` parser returned ``"j1"`` because the date
    20210706 contains a ``"2"`` early on. ``parse_nwb_filename`` is
    the canonical parser; ``analysis.pfc.load_population._extract_animal``
    is a thin wrapper around it (covered separately in
    ``tests/analysis/pfc/test_load_population.py``).
    """

    @pytest.mark.parametrize(
        "filename,expected_animal,expected_date",
        [
            ("chimi20200212_.nwb", "chimi", 20200212),
            ("peanut20201125_.nwb", "peanut", 20201125),
            ("wilbur20210315_.nwb", "wilbur", 20210315),
            ("senor20201015_.nwb", "senor", 20201015),
            # Historical bug: legacy str.split("2")[0] returned "j1".
            ("j1620210706_.nwb", "j16", 20210706),
        ],
    )
    def test_valid_filenames(self, filename, expected_animal, expected_date):
        animal, date = parse_nwb_filename(filename)
        assert animal == expected_animal
        assert date == expected_date

    @pytest.mark.parametrize(
        "bad_filename",
        [
            "chimi20200212.nwb",  # missing underscore
            "chimi20200212_.txt",  # wrong extension
            "chimi2020021_.nwb",  # date with < 8 digits
            "20200212_.nwb",  # no animal name
        ],
    )
    def test_invalid_format_raises(self, bad_filename):
        with pytest.raises(ValueError, match="Invalid NWB filename format"):
            parse_nwb_filename(bad_filename)


class TestParseEpochIdentifier:
    """The one parser of epoch identifiers and their decode suffixes."""

    def test_the_canonical_decode_carries_kay_swrs(self):
        key = parse_epoch_identifier("chimi20200212_02_r1_HPC")

        assert key.nwb_file_name == "chimi20200212_.nwb"
        assert key.epoch_name == "02_r1"
        assert key.brain_area == "HPC"
        assert key.swr_definition == DEFAULT_SWR_DEFINITION == "Kay"
        assert key.is_canonical

    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            ("_Karlsson", {"swr_definition": "Karlsson"}),
            ("_all", {"training_type": "all"}),
            ("_non_stationary", {"non_stationary": True}),
            ("_contfrag", {"contfrag": True}),
            (
                "_contfrag_Karlsson_non_stationary_all",
                {
                    "contfrag": True,
                    "swr_definition": "Karlsson",
                    "non_stationary": True,
                    "training_type": "all",
                },
            ),
        ],
    )
    def test_each_decode_suffix_is_read_in_the_order_it_is_appended(
        self, suffix, expected
    ):
        """As run_decoding_bandit.py and run_decoding_contfrag.py append them."""
        key = parse_epoch_identifier("chimi20200212_02_r1_HPC" + suffix)

        assert {name: getattr(key, name) for name in expected} == expected
        assert key.brain_area == "HPC"
        assert not key.is_canonical

    @pytest.mark.parametrize(
        "bad",
        [
            "chimi20200212_02",
            "a__c_d",
            "chimi20200212_02_r1_HPC_Kay",
            "x_02_r1_HPC_all_Karlsson",
        ],
    )
    def test_a_name_no_decode_writes_is_refused(self, bad):
        """Including suffixes out of order, and an explicit Kay, which no
        decode appends."""
        with pytest.raises(ValueError):
            parse_epoch_identifier(bad)

    @pytest.mark.parametrize(
        ("epoch_id", "animal", "date", "epoch_name", "brain_area"),
        [
            ("chimi20200212_02_r1_HPC", "chimi", 20200212, "02_r1", "HPC"),
            ("j1620210707_04_r2_HPC", "j16", 20210707, "04_r2", "HPC"),
            ("peanut20201125_10_r5_HPC", "peanut", 20201125, "10_r5", "HPC"),
            ("senor20201031_06_r3_HPC", "senor", 20201031, "06_r3", "HPC"),
            ("wilbur20210329_02_r1_HPC", "wilbur", 20210329, "02_r1", "HPC"),
            ("j1620210706_02_r1_mPFC", "j16", 20210706, "02_r1", "mPFC"),
            ("chimi20200223_08_r4_OFC", "chimi", 20200223, "08_r4", "OFC"),
            ("j1620210710_14_r7_HPC_contfrag", "j16", 20210710, "14_r7", "HPC"),
            ("j1620210710_14_r7_mPFC_contfrag", "j16", 20210710, "14_r7", "mPFC"),
            ("j1620210707_02_r1_HPC_non_stationary", "j16", 20210707, "02_r1", "HPC"),
        ],
    )
    def test_every_animal_keeps_its_id_and_date(
        self, epoch_id, animal, date, epoch_name, brain_area
    ):
        """``j16`` keeps its digits; the date is the stem's last 8, an int."""
        key = parse_epoch_identifier(epoch_id)

        assert (key.animal, key.date, key.epoch_name, key.brain_area) == (
            animal,
            date,
            epoch_name,
            brain_area,
        )
        assert type(key.date) is int
        assert key.nwb_file_name == f"{epoch_id.split('_')[0]}_.nwb"
        assert parse_nwb_filename(key.nwb_file_name) == (key.animal, key.date)

    def test_a_stem_without_a_date_is_refused_when_its_animal_is_read(self):
        """The suffixes parse; the stem is read only for ``animal``/``date``."""
        key = parse_epoch_identifier("x_02_r1_HPC")

        assert key.epoch_name == "02_r1"
        with pytest.raises(ValueError, match="Invalid NWB filename format"):
            _ = key.animal


class TestFormatEpochIdentifier:
    """The inverse of parse_epoch_identifier."""

    def test_it_joins_the_stem_epoch_area_and_suffix(self):
        assert (
            format_epoch_identifier("j1620210706_.nwb", "02_r1", "mPFC", "_contfrag")
            == "j1620210706_02_r1_mPFC_contfrag"
        )
        assert format_epoch_identifier("chimi20200212_.nwb", "02_r1", "HPC") == (
            "chimi20200212_02_r1_HPC"
        )

    def test_only_a_trailing_nwb_extension_is_removed(self):
        assert format_epoch_identifier("a_.nwbx_.nwb", "02_r1", "HPC") == (
            "a_.nwbx_02_r1_HPC"
        )


# Note: get_epoch_info() caching is validated through integration tests.
# Unit tests would require database access and are omitted.
# The @cache decorator is tested implicitly by verifying all integration
# tests pass with caching enabled.
