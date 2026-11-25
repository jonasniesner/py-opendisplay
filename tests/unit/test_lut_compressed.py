"""Compressed 0xAF LUT slots (presets plan 5.3 F1): bit 7 of lut_slot, raw deflate with a 10-bit window.

Decided 2026-11-25: 1024 B and 16 slots; ``crc16`` covers the stored (compressed) bytes.
"""

from __future__ import annotations

import os
import zlib

import pytest
from test_profile import ALPHA_CONFIG

from opendisplay.models.panel_drive import (
    LUT_DEFLATE_WBITS,
    LUT_MAX_BYTES,
    LUT_SLOT_COMPRESSED,
    PACKET_TYPE_DISPLAY_LUT,
    DisplayLut,
    LengthPrefixedPacket,
    _crc16_ccitt_false,
    find_lut,
)
from opendisplay.panels import TargetClass
from opendisplay.panels.database import check_drive_packets_fit

WAVEFORM = ALPHA_CONFIG.display_luts[1].data  # a real 535 B JD-BWRY table


def _deflate(data: bytes, wbits: int = LUT_DEFLATE_WBITS) -> bytes:
    deflater = zlib.compressobj(9, zlib.DEFLATED, wbits)
    return deflater.compress(data) + deflater.flush()


def test_plain_slot_is_unchanged() -> None:
    lut = ALPHA_CONFIG.display_luts[0]

    assert not lut.compressed
    assert lut.slot_id == lut.lut_slot == 0
    assert lut.waveform == lut.data
    assert DisplayLut.from_body(lut.to_body()) == lut


def test_build_compresses_and_sets_bit_7() -> None:
    lut = DisplayLut.build(0, 1, WAVEFORM)

    assert lut.compressed
    assert lut.lut_slot == LUT_SLOT_COMPRESSED | 1
    assert lut.slot_id == 1
    assert len(lut.data) < len(WAVEFORM)
    assert lut.waveform == WAVEFORM


def test_crc_covers_the_stored_compressed_bytes() -> None:
    lut = DisplayLut.build(0, 1, WAVEFORM)

    assert lut.crc16 == _crc16_ccitt_false(lut.data)
    assert lut.crc16 != _crc16_ccitt_false(WAVEFORM)
    assert lut.crc_valid


def test_body_round_trip() -> None:
    lut = DisplayLut.build(2, 15, WAVEFORM)
    again = DisplayLut.from_body(lut.to_body())

    assert again == lut
    assert again.waveform == WAVEFORM


def test_stream_uses_a_raw_10_bit_window() -> None:
    lut = DisplayLut.build(0, 0, WAVEFORM)

    assert zlib.decompress(lut.data, -10) == WAVEFORM
    with pytest.raises(zlib.error):
        zlib.decompress(lut.data)  # no zlib header: raw deflate


def test_auto_keeps_incompressible_data_plain() -> None:
    noise = os.urandom(800)
    lut = DisplayLut.build(0, 3, noise)

    assert not lut.compressed
    assert lut.data == noise
    assert DisplayLut.build(0, 3, noise, compress=True).compressed
    assert not DisplayLut.build(0, 3, WAVEFORM, compress=False).compressed


def test_1024_byte_waveform_fits_and_1025_does_not() -> None:
    assert DisplayLut.build(0, 0, bytes(LUT_MAX_BYTES)).waveform == bytes(LUT_MAX_BYTES)
    with pytest.raises(ValueError, match="at most 1024"):
        DisplayLut.build(0, 0, bytes(LUT_MAX_BYTES + 1))


@pytest.mark.parametrize("slot_id", [-1, 16, 0x80])
def test_slot_id_must_be_0_to_15(slot_id: int) -> None:
    with pytest.raises(ValueError, match="0-15"):
        DisplayLut.build(0, slot_id, WAVEFORM)


def test_a_bomb_past_1024_bytes_is_refused() -> None:
    stream = _deflate(bytes(4096))
    lut = DisplayLut(0, LUT_SLOT_COMPRESSED, _crc16_ccitt_false(stream), stream)

    with pytest.raises(ValueError, match="past 1024"):
        _ = lut.waveform


def test_truncated_and_garbage_streams_are_refused() -> None:
    stream = _deflate(WAVEFORM)
    truncated = DisplayLut(0, LUT_SLOT_COMPRESSED, 0, stream[:-8])
    garbage = DisplayLut(0, LUT_SLOT_COMPRESSED, 0, b"\xff" * 16)

    with pytest.raises(ValueError, match="truncated"):
        _ = truncated.waveform
    with pytest.raises(ValueError, match="inflate"):
        _ = garbage.waveform


def test_find_lut_matches_plain_and_compressed_slots_by_id() -> None:
    plain = DisplayLut.build(0, 0, WAVEFORM, compress=False)
    packed = DisplayLut.build(0, 1, WAVEFORM)
    other = DisplayLut.build(1, 1, WAVEFORM)
    luts = [plain, packed, other]

    assert find_lut(luts, 0, 0) is plain
    assert find_lut(luts, 0, 1) is packed
    assert find_lut(luts, 1, 1) is other
    assert find_lut(luts, 0, 2) is None


def test_reserved_bits_4_to_6_do_not_change_the_slot_id() -> None:
    lut = DisplayLut(0, LUT_SLOT_COMPRESSED | 0x30 | 5, 0, b"")

    assert lut.slot_id == 5
    assert lut.compressed


def test_fit_check_accepts_good_slots_and_names_bad_ones() -> None:
    def packet(lut: DisplayLut) -> LengthPrefixedPacket:
        return LengthPrefixedPacket(0, PACKET_TYPE_DISPLAY_LUT, lut.to_body())

    good = DisplayLut.build(0, 1, WAVEFORM)
    bad_crc = DisplayLut(0, good.lut_slot, good.crc16 ^ 1, good.data)
    stream = _deflate(bytes(2048))
    bomb = DisplayLut(0, LUT_SLOT_COMPRESSED | 2, _crc16_ccitt_false(stream), stream)
    big = DisplayLut(0, 3, _crc16_ccitt_false(bytes(1025)), bytes(1025))

    assert check_drive_packets_fit([packet(good)], TargetClass.BG22) == []
    problems = check_drive_packets_fit([packet(bad_crc), packet(bomb), packet(big)], TargetClass.BG22)
    assert len(problems) == 3
    assert "CRC" in problems[0] and "past 1024" in problems[1] and "1025 B stored" in problems[2]
