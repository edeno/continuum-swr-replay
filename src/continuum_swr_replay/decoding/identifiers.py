"""The epoch identifier: how a decode is named, built and read back.

An epoch identifier is ``{animal}{date}_{epoch}_{brain_area}`` (the NWB file's
stem, the epoch name, the brain area) followed by a decode's run suffixes, in
this order: ``_contfrag`` for the ContFrag classifier, a ripple detector other
than :data:`DEFAULT_SWR_DEFINITION`, ``_non_stationary``, and a training type
other than :data:`DEFAULT_TRAINING_TYPE`. Every file of a decode is named for
it (``decoding.files.decode_paths``).

Both directions live here, with the defaults they share:

- build: :func:`format_epoch_identifier` (stem, epoch, area and an optional
  ``_contfrag``), then :func:`decode_identifier` (the run suffixes);
- read: :func:`parse_epoch_identifier` into an :class:`EpochKey`, and
  :func:`parse_nwb_filename` for the stem's animal and date.

The names an identifier is made of are built here too: the NWB file name of an
animal and date (:func:`nwb_file_name`), its stem (:func:`nwb_stem`), the
area-less session name ``{stem}_{epoch}`` (:func:`session_name`, read back by
:func:`parse_session_name`), and an epoch
name's number (:func:`parse_epoch_name`, :data:`RUN_EPOCH_PATTERN`).

A leaf module: it imports only the standard library, so an offline reader of
decode folders, or a table builder keying rows by epoch, loads nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TrainingType(str, Enum):
    """Types of training data selection for decoder.

    Attributes
    ----------
    ALL : str
        Use all time points for training (no exclusions)
    NO_RIPPLE : str
        Exclude ripple times from training data

    Notes
    -----
    Inherits from str to allow direct string comparisons and JSON serialization.
    Use .value to get the underlying string value.

    Examples
    --------
    >>> TrainingType.ALL
    <TrainingType.ALL: 'all'>
    >>> TrainingType.ALL.value
    'all'
    >>> TrainingType.ALL == "all"
    True

    """

    ALL = "all"
    NO_RIPPLE = "no_ripple"

    def __str__(self) -> str:
        """Return string value for easy formatting."""
        return self.value


DEFAULT_SWR_DEFINITION = "Kay"
"""A decode run with the default ripple detector adds no suffix to its identifier."""

OTHER_RIPPLE_DETECTORS = ("Karlsson", "Yu")
"""The ripple detectors whose name a decode would append; Kay's it does not.

No decode carries ``_Yu`` yet; it is listed so the day one does, the parser
reads it as a ripple detector and not as a name no decode writes."""

DEFAULT_TRAINING_TYPE = TrainingType.NO_RIPPLE.value
"""The training type a decode does not append to its identifier (``"no_ripple"``)."""

CONTFRAG_SUFFIX = "_contfrag"
"""The suffix a ContFrag classifier decode appends, first of the run suffixes."""


RUN_EPOCH_PATTERN = re.compile(r"^[0-9]+_r[0-9]+$")
"""A run epoch's name, such as ``02_r1``: the epoch number, ``_r``, the run."""


def nwb_file_name(animal: str, date: object) -> str:
    """The NWB file name of `animal`'s recording on `date`: ``{animal}{date}_.nwb``.

    The inverse of :func:`parse_nwb_filename`.

    Parameters
    ----------
    animal : str
        Such as ``"chimi"``.
    date : int or str
        The recording date ``YYYYMMDD``, such as ``20200212``.

    Returns
    -------
    str

    Examples
    --------
    >>> nwb_file_name("j16", 20210706)
    'j1620210706_.nwb'
    """
    return f"{animal}{date}_.nwb"


def nwb_stem(nwb_file_name: str) -> str:
    """`nwb_file_name` without its trailing ``_.nwb``: ``{animal}{date}``.

    Only a trailing ``_.nwb`` is removed; a name without one is returned as is.

    Examples
    --------
    >>> nwb_stem("chimi20200212_.nwb")
    'chimi20200212'
    """
    return nwb_file_name.removesuffix("_.nwb")


def session_name(nwb_file_name: str, epoch_name: object) -> str:
    """The area-less session name ``{animal}{date}_{epoch}``.

    The epoch identifier without its brain area and decode suffixes
    (:func:`format_epoch_identifier`).

    Parameters
    ----------
    nwb_file_name : str
        Such as ``"chimi20200212_.nwb"`` (:func:`nwb_file_name` builds it
        from an animal and a date).
    epoch_name : str
        Such as ``"02_r1"``.

    Returns
    -------
    str

    Examples
    --------
    >>> session_name("chimi20200212_.nwb", "02_r1")
    'chimi20200212_02_r1'
    """
    return f"{nwb_stem(nwb_file_name)}_{epoch_name}"


def parse_session_name(name: str) -> tuple[str, str]:
    """Split a session name ``{animal}{date}_{epoch}`` into its NWB file and epoch.

    The inverse of :func:`session_name`.

    Parameters
    ----------
    name : str
        Such as ``"chimi20200212_02_r1"``.

    Returns
    -------
    tuple[str, str]
        ``(nwb_file_name, epoch_name)``, such as ``("chimi20200212_.nwb", "02_r1")``.

    Raises
    ------
    ValueError
        If `name` is not three non-empty ``_``-separated parts.

    Examples
    --------
    >>> parse_session_name("j1620210706_02_r1")
    ('j1620210706_.nwb', '02_r1')
    """
    parts = name.split("_")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"{name!r} is not {{animal}}{{date}}_{{epoch}}.")
    stem, number, tag = parts
    return f"{stem}_.nwb", f"{number}_{tag}"


def parse_epoch_name(epoch_name: str) -> tuple[int, str]:
    """Split an epoch name ``{number}_{tag}`` into its number and its tag.

    Parameters
    ----------
    epoch_name : str
        Such as ``"02_r1"``.

    Returns
    -------
    tuple[int, str]
        Such as ``(2, "r1")``.

    Raises
    ------
    ValueError
        If `epoch_name` is not two ``_``-separated parts, or its first part is
        not an integer.

    Examples
    --------
    >>> parse_epoch_name("04_r2")
    (4, 'r2')
    """
    number, tag = epoch_name.split("_")
    return int(number), tag


def parse_nwb_filename(filename: str) -> tuple[str, int]:
    """Parse NWB filename to extract animal name and date.

    Parameters
    ----------
    filename : str
        NWB filename in format: {animal}{YYYYMMDD}_.nwb
        Example: "chimi20200212_.nwb"

    Returns
    -------
    tuple[str, int]
        Animal name and date as (animal, date)
        Example: ("chimi", 20200212)

    Raises
    ------
    ValueError
        If filename doesn't match expected format

    Examples
    --------
    >>> parse_nwb_filename("chimi20200212_.nwb")
    ('chimi', 20200212)

    >>> parse_nwb_filename("peanut20201125_.nwb")
    ('peanut', 20201125)

    >>> parse_nwb_filename("j1620210706_.nwb")
    ('j16', 20210706)

    """
    # Pattern: {animal_name}{8_digits}_.nwb
    # Animal name is one or more alphanumeric chars (non-greedy), date is exactly 8 digits
    # The date is always YYYYMMDD format, so we match the last 8 digits before _.nwb
    pattern = r"^([a-zA-Z0-9]+?)(\d{8})_\.nwb$"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(
            f"Invalid NWB filename format: {filename}. "
            f"Expected format: {{animal}}{{YYYYMMDD}}_.nwb"
        )

    animal = match.group(1)
    date = int(match.group(2))

    return animal, date


@dataclass(frozen=True)
class EpochKey:
    """What an epoch identifier says about the decode it names."""

    epoch_id: str
    nwb_file_name: str
    epoch_name: str
    brain_area: str
    swr_definition: str
    contfrag: bool
    non_stationary: bool
    training_type: str

    @property
    def animal(self) -> str:
        """The animal, as :func:`parse_nwb_filename` reads ``nwb_file_name``."""
        return parse_nwb_filename(self.nwb_file_name)[0]

    @property
    def date(self) -> int:
        """The recording date ``YYYYMMDD``, as :func:`parse_nwb_filename` reads it."""
        return parse_nwb_filename(self.nwb_file_name)[1]

    @property
    def is_canonical(self) -> bool:
        """The decode the comparison uses: Kay's SWRs, no_ripple, stationary."""
        return (
            not self.contfrag
            and self.swr_definition == DEFAULT_SWR_DEFINITION
            and not self.non_stationary
            and self.training_type == DEFAULT_TRAINING_TYPE
        )


def format_epoch_identifier(
    nwb_file_name: str, epoch: str, brain_area: str, suffix: str = ""
) -> str:
    """Build ``{animal}{date}_{epoch}_{brain_area}{suffix}``, the epoch identifier.

    The inverse of :func:`parse_epoch_identifier`. Only a trailing ``_.nwb`` is
    removed from `nwb_file_name`.

    Parameters
    ----------
    nwb_file_name : str
        Such as ``"chimi20200212_.nwb"``.
    epoch : str
        Such as ``"02_r1"``.
    brain_area : str
        ``"HPC"``, ``"mPFC"`` or ``"OFC"``.
    suffix : str, optional
        Decode suffixes, each with its leading ``_``, in the order
        :func:`parse_epoch_identifier` reads them, such as ``"_contfrag"`` or
        ``"_non_stationary"``. None by default.

    Returns
    -------
    str

    Examples
    --------
    >>> format_epoch_identifier("chimi20200212_.nwb", "02_r1", "mPFC", "_contfrag")
    'chimi20200212_02_r1_mPFC_contfrag'
    """
    return f"{session_name(nwb_file_name, epoch)}_{brain_area}{suffix}"


def decode_identifier(
    epoch_stem: str,
    *,
    ripple_detector_name: str = DEFAULT_SWR_DEFINITION,
    non_stationary: bool = False,
    training_type: str = DEFAULT_TRAINING_TYPE,
) -> str:
    """A decode's full identifier: `epoch_stem` and its run suffixes.

    The suffixes follow in this order: ``_{ripple_detector_name}`` unless it is
    :data:`DEFAULT_SWR_DEFINITION` (``"Kay"``), ``_non_stationary`` when
    `non_stationary`, and ``_{training_type}`` unless it is
    :data:`DEFAULT_TRAINING_TYPE` (``"no_ripple"``). The canonical decode
    (Kay, stationary, no_ripple) is `epoch_stem` alone.
    :func:`parse_epoch_identifier` reads them back in the same order. Only Kay
    is decodable today (``data_loaders.lfp`` refuses any other ripple
    detector), so no decode currently writes the detector suffix.

    Parameters
    ----------
    epoch_stem : str
        Epoch identifier before the run suffixes, such as
        ``"chimi20200212_02_r1_HPC"`` or ``"chimi20200212_02_r1_HPC_contfrag"``
        (:func:`format_epoch_identifier`).
    ripple_detector_name : str, optional
        The SWR detector the decode's training mask used, by default ``"Kay"``.
    non_stationary : bool, optional
        Whether the decode used non-stationary discrete transitions.
    training_type : str, optional
        The decode's training regime, by default ``"no_ripple"``.

    Returns
    -------
    str
        The identifier every file of the decode is named for
        (``decoding.files.decode_paths``).
    """
    epoch_identifier = epoch_stem
    if ripple_detector_name != DEFAULT_SWR_DEFINITION:
        epoch_identifier += f"_{ripple_detector_name}"
    if non_stationary:
        epoch_identifier += "_non_stationary"
    if training_type != DEFAULT_TRAINING_TYPE:
        epoch_identifier += f"_{training_type}"
    return epoch_identifier


def parse_epoch_identifier(epoch_id: str) -> EpochKey:
    """Split ``{animal}{date}_{epoch}_{brain_area}`` and any decode suffixes.

    Suffixes come in the order :func:`format_epoch_identifier` and
    :func:`decode_identifier` append them: ``_contfrag``, a ripple detector
    other than Kay, ``_non_stationary``, and a training type other than
    ``no_ripple``. Anything else is refused rather than read as, say, a
    ripple detector called ``all``.

    Parameters
    ----------
    epoch_id : str
        Such as ``"j1620210707_02_r1_HPC"`` or ``"chimi20200212_02_r1_mPFC_contfrag"``.

    Returns
    -------
    EpochKey
        ``animal`` and ``date`` are read from the NWB stem only when asked for,
        so a stem :func:`parse_nwb_filename` refuses is refused there.

    Raises
    ------
    ValueError
        If `epoch_id` has fewer than four ``_``-separated parts, an empty one,
        or a suffix no decode appends.

    Examples
    --------
    >>> key = parse_epoch_identifier("j1620210707_02_r1_HPC_contfrag")
    >>> key.animal, key.date, key.epoch_name, key.brain_area, key.contfrag
    ('j16', 20210707, '02_r1', 'HPC', True)
    """
    parts = epoch_id.split("_")
    if len(parts) < 4 or not all(parts):
        raise ValueError(
            f"{epoch_id!r} is not {{animal}}{{date}}_{{epoch}}_{{brain_area}} with "
            f"optional decode suffixes."
        )
    stem, number, replicate, brain_area, *rest = parts
    contfrag = rest[:1] == [CONTFRAG_SUFFIX[1:]]
    rest = rest[1:] if contfrag else rest
    swr_definition = DEFAULT_SWR_DEFINITION
    if rest and rest[0] in OTHER_RIPPLE_DETECTORS:
        swr_definition, rest = rest[0], rest[1:]
    non_stationary = rest[:2] == ["non", "stationary"]
    rest = rest[2:] if non_stationary else rest
    training_type = DEFAULT_TRAINING_TYPE
    if rest[:1] == [TrainingType.ALL.value]:
        training_type, rest = TrainingType.ALL.value, rest[1:]
    if rest:
        raise ValueError(
            f"{epoch_id!r} ends in {'_'.join(rest)!r}, which no decode appends."
        )
    return EpochKey(
        epoch_id=epoch_id,
        nwb_file_name=f"{stem}_.nwb",
        epoch_name=f"{number}_{replicate}",
        brain_area=brain_area,
        swr_definition=swr_definition,
        contfrag=contfrag,
        non_stationary=non_stationary,
        training_type=training_type,
    )
