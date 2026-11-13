"""The epoch identifier, built and read back (``decoding.identifiers``).

Every decode file is named for an identifier ``format_epoch_identifier`` and
``decode_identifier`` build, and every reader of a decode folder reads one back
with ``parse_epoch_identifier``. The two directions share the defaults no
identifier spells out (Kay's SWRs, ``no_ripple`` training), so they are tested
together here, on literal names and on every epoch of the committed register.
"""

from __future__ import annotations

import itertools

import pytest

from continuum_swr_replay.data_loaders.epoch_register import Status, read_register
from continuum_swr_replay.decoding.identifiers import (
    DEFAULT_SWR_DEFINITION,
    DEFAULT_TRAINING_TYPE,
    OTHER_RIPPLE_DETECTORS,
    RUN_EPOCH_PATTERN,
    TrainingType,
    decode_identifier,
    format_epoch_identifier,
    nwb_file_name,
    nwb_stem,
    parse_epoch_identifier,
    parse_epoch_name,
    parse_nwb_filename,
    parse_session_name,
    session_name,
)


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


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "chimi20200212_02_r1_HPC"),
        ({"ripple_detector_name": "Karlsson"}, "chimi20200212_02_r1_HPC_Karlsson"),
        ({"non_stationary": True}, "chimi20200212_02_r1_HPC_non_stationary"),
        ({"training_type": "all"}, "chimi20200212_02_r1_HPC_all"),
        (
            {
                "ripple_detector_name": "Karlsson",
                "non_stationary": True,
                "training_type": "all",
            },
            "chimi20200212_02_r1_HPC_Karlsson_non_stationary_all",
        ),
    ],
)
def test_decode_identifier_appends_each_non_default_setting_in_order(kwargs, expected):
    assert decode_identifier("chimi20200212_02_r1_HPC", **kwargs) == expected


def test_the_defaults_are_the_plain_strings_no_identifier_spells_out():
    """A plain ``str``, not the enum, so an f-string or a comparison reads the
    same everywhere."""
    assert type(DEFAULT_TRAINING_TYPE) is str
    assert DEFAULT_TRAINING_TYPE == "no_ripple" == TrainingType.NO_RIPPLE
    assert DEFAULT_SWR_DEFINITION == "Kay"
    assert DEFAULT_SWR_DEFINITION not in OTHER_RIPPLE_DETECTORS
    assert decode_identifier("chimi20200212_02_r1_HPC") == "chimi20200212_02_r1_HPC"


def test_every_register_decode_identifier_round_trips():
    """For every epoch the register lists (any status), every brain area, and
    every run setting a decode can be named for, the built identifier parses back
    to the settings it was built from, and rebuilding from the parse gives the
    same identifier."""
    register = read_register()
    epochs = [key for key in register.index if key[1] != "*"]
    assert len(epochs) > 200
    settings = itertools.product(
        ["HPC", "mPFC", "OFC"],
        [False, True],
        [DEFAULT_SWR_DEFINITION, *OTHER_RIPPLE_DETECTORS],
        [False, True],
        [t.value for t in TrainingType],
    )
    for (name, epoch), setting in itertools.product(epochs, list(settings)):
        brain_area, contfrag, ripple, non_stationary, training_type = setting
        stem = format_epoch_identifier(
            name, epoch, brain_area, "_contfrag" if contfrag else ""
        )
        identifier = decode_identifier(
            stem,
            ripple_detector_name=ripple,
            non_stationary=non_stationary,
            training_type=training_type,
        )
        key = parse_epoch_identifier(identifier)

        assert (
            key.epoch_id,
            key.nwb_file_name,
            key.epoch_name,
            key.brain_area,
            key.contfrag,
            key.swr_definition,
            key.non_stationary,
            key.training_type,
        ) == (
            identifier,
            name,
            epoch,
            brain_area,
            contfrag,
            ripple,
            non_stationary,
            training_type,
        )
        assert (key.animal, key.date) == parse_nwb_filename(name)
        assert key.is_canonical == (identifier == stem and not contfrag)
        rebuilt = decode_identifier(
            format_epoch_identifier(
                key.nwb_file_name,
                key.epoch_name,
                key.brain_area,
                "_contfrag" if key.contfrag else "",
            ),
            ripple_detector_name=key.swr_definition,
            non_stationary=key.non_stationary,
            training_type=key.training_type,
        )
        assert rebuilt == identifier


class TestSessionNames:
    """The NWB file name, its stem, the session name and the epoch number."""

    @pytest.mark.parametrize(
        ("animal", "date", "name", "stem"),
        [
            ("chimi", 20200212, "chimi20200212_.nwb", "chimi20200212"),
            ("j16", 20210706, "j1620210706_.nwb", "j1620210706"),
            ("wilbur", "20210329", "wilbur20210329_.nwb", "wilbur20210329"),
        ],
    )
    def test_the_nwb_name_and_stem_of_an_animal_and_date(
        self, animal, date, name, stem
    ):
        assert nwb_file_name(animal, date) == name
        assert nwb_stem(name) == stem
        assert parse_nwb_filename(name) == (animal, int(date))

    def test_only_a_trailing_nwb_extension_is_stripped(self):
        assert nwb_stem("a_.nwbx_.nwb") == "a_.nwbx"
        assert nwb_stem("chimi20200212") == "chimi20200212"

    def test_the_session_name_is_the_identifier_without_its_area(self):
        assert session_name("j1620210706_.nwb", "02_r1") == "j1620210706_02_r1"
        assert (
            format_epoch_identifier("j1620210706_.nwb", "02_r1", "HPC")
            == session_name("j1620210706_.nwb", "02_r1") + "_HPC"
        )

    def test_a_session_name_parses_back_to_its_file_and_epoch(self):
        assert parse_session_name("j1620210706_02_r1") == ("j1620210706_.nwb", "02_r1")

    @pytest.mark.parametrize(
        "bad", ["j1620210706_02", "j1620210706_02_r1_HPC", "_02_r1", "a__r1"]
    )
    def test_a_name_that_is_not_a_session_name_is_refused(self, bad):
        with pytest.raises(ValueError, match="is not"):
            parse_session_name(bad)

    @pytest.mark.parametrize(
        ("epoch_name", "parsed"),
        [("02_r1", (2, "r1")), ("16_r10", (16, "r10")), ("01_s1", (1, "s1"))],
    )
    def test_an_epoch_name_splits_into_its_number_and_tag(self, epoch_name, parsed):
        assert parse_epoch_name(epoch_name) == parsed

    @pytest.mark.parametrize("bad", ["02", "02_r1_x", "r1_02", "_r1"])
    def test_an_epoch_name_that_is_not_number_and_tag_is_refused(self, bad):
        with pytest.raises(ValueError):
            parse_epoch_name(bad)

    @pytest.mark.parametrize(
        ("epoch_name", "is_run"),
        [("02_r1", True), ("16_r10", True), ("01_s1", False), ("02_r1_x", False)],
    )
    def test_the_run_epoch_pattern(self, epoch_name, is_run):
        assert bool(RUN_EPOCH_PATTERN.match(epoch_name)) is is_run


def test_every_analysed_epoch_round_trips_through_its_names():
    """For each of the register's analysed epochs, the NWB name rebuilt from its
    animal and date is the register's, the session name and the HPC identifier
    agree, and the identifier parses back to the epoch."""
    register = read_register()
    analysed = register.index[register["status"] == Status.ANALYSED]
    assert len(analysed) == 242
    for name, epoch in analysed:
        animal, date = parse_nwb_filename(name)
        assert nwb_file_name(animal, date) == name
        assert nwb_stem(name) == f"{animal}{date}"
        assert RUN_EPOCH_PATTERN.match(epoch)
        number, tag = parse_epoch_name(epoch)
        assert f"{number:02d}_{tag}" == epoch
        identifier = format_epoch_identifier(name, epoch, "HPC")
        assert identifier == f"{session_name(name, epoch)}_HPC"
        assert parse_session_name(session_name(name, epoch)) == (name, epoch)
        key = parse_epoch_identifier(identifier)
        assert (key.nwb_file_name, key.epoch_name, key.animal, key.date) == (
            name,
            epoch,
            animal,
            date,
        )
