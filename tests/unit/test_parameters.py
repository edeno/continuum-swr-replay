"""Unit tests for parameter validation."""

from __future__ import annotations

import pytest

from continuum_swr_replay.parameters import DecodingParameters


class TestDecodingParametersValidation:
    """Test validation in DecodingParameters dataclass."""

    def test_default_parameters_are_valid(self) -> None:
        """Default parameters should pass all validation checks."""
        # Should not raise any exceptions
        params = DecodingParameters()
        assert params.block_size == 1024
        assert params.position_std > 0
        assert params.waveform_std > 0
        assert params.place_bin_size > 0

    def test_valid_block_size_powers_of_two(self) -> None:
        """Block size should accept valid powers of 2."""
        valid_block_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        for block_size in valid_block_sizes:
            params = DecodingParameters(block_size=block_size)
            assert params.block_size == block_size

    def test_invalid_block_size_not_power_of_two(self) -> None:
        """Block size must be a power of 2."""
        invalid_block_sizes = [0, 3, 5, 6, 7, 9, 10, 15, 100, 1000, 1023, 1025]
        for block_size in invalid_block_sizes:
            with pytest.raises(
                ValueError, match=f"block_size must be power of 2, got {block_size}"
            ):
                DecodingParameters(block_size=block_size)

    def test_invalid_block_size_negative(self) -> None:
        """Block size must be positive."""
        with pytest.raises(ValueError, match="block_size must be power of 2, got -1"):
            DecodingParameters(block_size=-1)

    def test_invalid_position_std_zero(self) -> None:
        """Position std must be positive (not zero)."""
        with pytest.raises(ValueError, match="position_std must be positive, got 0"):
            DecodingParameters(position_std=0.0)

    def test_invalid_position_std_negative(self) -> None:
        """Position std must be positive (not negative)."""
        with pytest.raises(
            ValueError, match="position_std must be positive, got -1.0"
        ):
            DecodingParameters(position_std=-1.0)

    def test_invalid_waveform_std_zero(self) -> None:
        """Waveform std must be positive (not zero)."""
        with pytest.raises(ValueError, match="waveform_std must be positive, got 0"):
            DecodingParameters(waveform_std=0.0)

    def test_invalid_waveform_std_negative(self) -> None:
        """Waveform std must be positive (not negative)."""
        with pytest.raises(
            ValueError, match="waveform_std must be positive, got -5.0"
        ):
            DecodingParameters(waveform_std=-5.0)

    def test_invalid_place_bin_size_zero(self) -> None:
        """Place bin size must be positive (not zero)."""
        with pytest.raises(
            ValueError, match="place_bin_size must be positive, got 0"
        ):
            DecodingParameters(place_bin_size=0.0)

    def test_invalid_place_bin_size_negative(self) -> None:
        """Place bin size must be positive (not negative)."""
        with pytest.raises(
            ValueError, match="place_bin_size must be positive, got -2.0"
        ):
            DecodingParameters(place_bin_size=-2.0)

    def test_valid_custom_parameters(self) -> None:
        """Should accept valid custom parameters."""
        params = DecodingParameters(
            place_bin_size=5.0,
            waveform_std=30.0,
            position_std=4.0,
            block_size=512,
        )
        assert params.place_bin_size == 5.0
        assert params.waveform_std == 30.0
        assert params.position_std == 4.0
        assert params.block_size == 512
