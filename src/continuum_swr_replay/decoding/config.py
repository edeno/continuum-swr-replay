"""The decoders' encoding-likelihood configuration.

``run_decoding_bandit.py`` (the NonLocal 4-state detector) decodes with
:class:`NonLocalDecodingConfig` and ``run_decoding_contfrag.py`` (the ContFrag
2-state classifier) with :class:`DecodingConfig`. Their values are
``parameters``'.
"""

from dataclasses import dataclass

from continuum_swr_replay.data_types import ClusterlessParams, SortedSpikesParams
from continuum_swr_replay.parameters import (
    BLOCK_SIZE,
    LOCAL_POSITION_STD,
    POSITION_STD,
    WAVEFORM_STD,
)


@dataclass(frozen=True)
class DecodingConfig:
    """Encoding-likelihood configuration shared by every detector family.

    ``position_std``, ``waveform_std``, and ``block_size`` configure the
    encoding-model likelihood (the ``*_algorithm_params`` dict) and apply
    to both the NonLocal 4-state detector and the ContFrag 2-state
    classifier. State-specific detector arguments live on subclasses —
    see :class:`NonLocalDecodingConfig`.
    """

    position_std: float = POSITION_STD
    block_size: int = BLOCK_SIZE
    waveform_std: float = WAVEFORM_STD

    def to_clusterless_params(self) -> ClusterlessParams:
        """Get parameters for clusterless decoding (HPC)."""
        return {
            "position_std": self.position_std,
            "waveform_std": self.waveform_std,
            "block_size": self.block_size,
        }

    def to_sorted_spikes_params(self) -> SortedSpikesParams:
        """Get parameters for sorted spikes decoding (mPFC/OFC)."""
        return {
            "position_std": self.position_std,
            "block_size": self.block_size,
        }


@dataclass(frozen=True)
class NonLocalDecodingConfig(DecodingConfig):
    """Decoding config for the NonLocal 4-state detector.

    Adds ``local_position_std``, a detector-level argument that controls
    the Local state's spatial representation. It is passed directly to the
    detector constructor (not through ``to_*_params``). The ContFrag
    2-state classifier has no Local state, so it uses the base
    :class:`DecodingConfig` instead of this subclass.
    """

    local_position_std: float = LOCAL_POSITION_STD
