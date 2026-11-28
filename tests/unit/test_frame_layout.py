"""The DisplayConfig frame layout (protocol structs 2.5): parsing, the packer, and its users."""

from __future__ import annotations

import dataclasses
import gzip
import json
from importlib.resources import files
from typing import Any

import numpy as np
import pytest
from epaper_dithering import DitherMode
from PIL import Image
from test_profile import FW3, _config, _display

from opendisplay.device.imaging import prepare_image
from opendisplay.encoding import encode_1bpp, encode_2bpp, encode_4bpp, encode_bitplanes, encode_gray4_bitplanes
from opendisplay.encoding.images import encode_image
from opendisplay.encoding.layout import frame_size, lay_out_plane, resolve_layout, scheme_geometry
from opendisplay.encoding.multipass import encode_multipass
from opendisplay.exceptions import FrameLayoutError
from opendisplay.models.capabilities import DeviceCapabilities
from opendisplay.models.config import DisplayConfig
from opendisplay.models.enums import ColorScheme, FrameScan, RowOrder
from opendisplay.models.frame_layout import IDENTITY, FrameLayout
from opendisplay.models.panel_drive import CalibrationEntry, DisplayCalibration
from opendisplay.models.profile import build_profile
from opendisplay.panels import PanelDatabase, TargetClass
from opendisplay.panels.migration import apply_migration, display_layout_fix, plan_migration
from opendisplay.partial import PartialState, compute_partial_region
from opendisplay.protocol.config_serializer import serialize_display_config

RNG = np.random.default_rng(0x2_5)


def _p(codes: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(codes, dtype=np.uint8), mode="P")


def _rows_2bpp(codes: np.ndarray) -> np.ndarray:
    """Plain row-major MSB-first 2 bpp rows (width a multiple of 4): rows x bytes."""
    c = codes.reshape(codes.shape[0], -1, 4).astype(np.uint8)
    return (c[:, :, 0] << 6) | (c[:, :, 1] << 4) | (c[:, :, 2] << 2) | c[:, :, 3]


# ── reference vectors ────────────────────────────────────────────────────────


def test_gdey0579f52_halves_with_folded_rows() -> None:
    """Panel 78: each controller takes, for i in 0..135, row 271-i then row i of its 396 columns."""
    codes = RNG.integers(0, 4, size=(272, 792), dtype=np.uint8)
    layout = FrameLayout(frame_split=1, part_width=396, row_order=RowOrder.FOLD)

    data = encode_2bpp(_p(codes), layout=layout)

    expected = b""
    for part in range(2):
        rows = _rows_2bpp(codes[:, part * 396 : (part + 1) * 396])
        assert rows.shape == (272, 99)
        for i in range(136):
            expected += rows[271 - i].tobytes() + rows[i].tobytes()
    assert len(expected) == 2 * 26_928
    assert data == expected
    assert frame_size(layout, ColorScheme.BWRY, 792, 272) == 2 * 26_928


def test_crowpanel_579_matches_bb_epaper_write_half() -> None:
    """Panel 51: bbepWriteHalf orientation 0, ``row_buffer[ty][tx + 0 or 49]`` for tx < 50, ty < 272."""
    raster = RNG.integers(0, 2, size=(272, 792), dtype=np.uint8)
    row_buffer = np.packbits(raster, axis=1)
    assert row_buffer.shape == (272, 99)
    expected = bytes(int(row_buffer[ty][tx + offset]) for offset in (0, 49) for tx in range(50) for ty in range(272))
    layout = FrameLayout(frame_split=1, part_width=400, part_stride=392, frame_scan=FrameScan.COLUMN_MAJOR)

    data = encode_1bpp(_p(raster), layout)

    assert len(data) == 2 * 13_600
    assert data == expected


def _retired_half_planes(idx: np.ndarray) -> bytes:
    """The scheme-8 packer this library had: left half of every row, then the right half."""

    def pack(plane: np.ndarray) -> bytes:
        h, w = plane.shape
        p = np.pad(plane, ((0, 0), (0, 1))) if w & 1 else plane
        p = p.reshape(h, -1, 2)
        return ((p[:, :, 0] << 4) | p[:, :, 1]).astype(np.uint8).tobytes()

    mid = idx.shape[1] // 2
    return pack(idx[:, :mid]) + pack(idx[:, mid:])


@pytest.mark.parametrize(("width", "height"), [(4, 2), (184, 30), (1200, 16)])
def test_bwgbry_halves_equal_the_retired_scheme_8(width: int, height: int) -> None:
    indices = RNG.integers(0, 6, size=(height, width), dtype=np.uint8)
    lut = np.array([0, 1, 2, 3, 5, 6], dtype=np.uint8)

    data = encode_image(_p(indices), ColorScheme.BWGBRY, FrameLayout(frame_split=1))

    assert data == _retired_half_planes(lut[indices])


# ── an all-zero layout is today's output ─────────────────────────────────────


@pytest.mark.parametrize("width", [8, 13, 122])
def test_default_layout_is_the_plain_packing_for_every_scheme(width: int) -> None:
    height = 5
    two = RNG.integers(0, 2, size=(height, width), dtype=np.uint8)
    three = RNG.integers(0, 3, size=(height, width), dtype=np.uint8)
    four = RNG.integers(0, 4, size=(height, width), dtype=np.uint8)
    six = RNG.integers(0, 6, size=(height, width), dtype=np.uint8)
    seven = RNG.integers(0, 7, size=(height, width), dtype=np.uint8)
    sixteen = RNG.integers(0, 16, size=(height, width), dtype=np.uint8)

    def nibbles(idx: np.ndarray) -> bytes:
        p = np.pad(idx, ((0, 0), (0, idx.shape[1] & 1))).reshape(height, -1, 2)
        return ((p[:, :, 0] << 4) | p[:, :, 1]).astype(np.uint8).tobytes()

    def crumbs(idx: np.ndarray) -> bytes:
        return _rows_2bpp(np.pad(idx, ((0, 0), (0, -idx.shape[1] % 4)))).tobytes()

    def bits(mask: np.ndarray) -> bytes:
        return np.packbits(mask, axis=1).tobytes()

    assert encode_image(_p(two), ColorScheme.MONO) == bits(two > 0)
    assert encode_image(_p(four), ColorScheme.BWRY) == crumbs(four)
    assert encode_2bpp(_p(four), codes=(0, 1, 3, 2)) == crumbs(np.array([0, 1, 3, 2], np.uint8)[four])
    lut = np.array([0, 1, 2, 3, 5, 6], dtype=np.uint8)
    assert encode_image(_p(six), ColorScheme.BWGBRY) == nibbles(lut[six])
    assert encode_image(_p(sixteen), ColorScheme.GRAYSCALE_16) == nibbles(sixteen)
    assert encode_image(_p(seven), ColorScheme.SEVEN_COLOR) == nibbles(seven)
    assert encode_bitplanes(_p(three), ColorScheme.BWR) == (bits((three == 1) | (three == 2)), bits(three == 2))
    assert encode_bitplanes(_p(three), ColorScheme.BWY) == (bits(three == 1), bits(three == 2))
    gray = np.array((3, 1, 2, 0), dtype=np.uint8)[four]
    assert encode_gray4_bitplanes(_p(four), (3, 1, 2, 0)) == (bits(gray & 1), bits(gray & 2))
    for scheme in (ColorScheme.MONO, ColorScheme.BWR, ColorScheme.BWRY, ColorScheme.BWGBRY, ColorScheme.GRAYSCALE_4):
        assert scheme_geometry(IDENTITY, scheme, width, height).cells == width * height


# ── each field on a small frame ──────────────────────────────────────────────

ROWS = np.array([[0, 1, 2, 3, 3, 2, 1, 0], [1, 1, 2, 2, 3, 3, 0, 0]], dtype=np.uint8)


def _lay(**fields: Any) -> bytes:
    return lay_out_plane(ROWS, 2, FrameLayout(**fields), white=1)


def test_mirror_x_mirror_y_and_lsb_first() -> None:
    assert _lay() == bytes([0x1B, 0xE4, 0x5A, 0xF0])
    assert _lay(frame_scan=FrameScan.MIRROR_X) == bytes([0x1B, 0xE4, 0x0F, 0xA5])
    assert _lay(frame_scan=FrameScan.MIRROR_Y) == bytes([0x5A, 0xF0, 0x1B, 0xE4])
    assert _lay(frame_scan=FrameScan.LSB_FIRST) == bytes([0xE4, 0x1B, 0xA5, 0x0F])


def test_mirror_y_applies_before_fold() -> None:
    codes = np.zeros((5, 8), dtype=np.uint8)
    codes[0] = 1  # only row 0 is set
    rows = lay_out_plane(codes, 1, FrameLayout(row_order=RowOrder.FOLD))
    assert list(rows) == [0x00, 0xFF, 0x00, 0x00, 0x00]  # rows 4, 0, 3, 1, 2
    mirrored = lay_out_plane(codes, 1, FrameLayout(row_order=RowOrder.FOLD, frame_scan=FrameScan.MIRROR_Y))
    assert list(mirrored) == [0xFF, 0x00, 0x00, 0x00, 0x00]  # mirrored first: row 0 is now row 4


def test_interleave_sends_row_pairs_lower_row_first() -> None:
    # per column x: code(x, 1) then code(x, 0)
    stream = [c for x in range(8) for c in (ROWS[1, x], ROWS[0, x])]
    expected = bytes(
        (stream[i] << 6) | (stream[i + 1] << 4) | (stream[i + 2] << 2) | stream[i + 3] for i in range(0, 16, 4)
    )
    assert _lay(row_order=RowOrder.INTERLEAVE) == expected


def test_nibble_pack_and_plane_invert() -> None:
    assert _lay(pixel_pack=1) == bytes([0x01, 0x23, 0x32, 0x10, 0x11, 0x22, 0x33, 0x00])
    assert _lay(plane_invert=1) == bytes(b ^ 0xFF for b in _lay())


def test_columns_past_the_edge_are_white() -> None:
    codes = np.zeros((1, 8), dtype=np.uint8)
    data = lay_out_plane(codes, 2, FrameLayout(frame_split=1, part_width=8, part_stride=4), white=1)
    assert data == bytes([0x00, 0x00, 0x00, 0x55])  # part 1 covers columns 4..11; 8..11 are white


def test_column_major_with_fold() -> None:
    codes = np.array([[1] * 8 + [0] * 8, [0] * 16, [1] * 16], dtype=np.uint8)
    data = lay_out_plane(codes, 1, FrameLayout(frame_scan=FrameScan.COLUMN_MAJOR, row_order=RowOrder.FOLD))
    # rows 2, 0, 1; byte-column 0 then 1
    assert data == bytes([0xFF, 0xFF, 0x00, 0xFF, 0x00, 0x00])


# ── refusals ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fields", "scheme", "match"),
    [
        ({"frame_split": 2}, ColorScheme.MONO, "frame_split 2 is not defined"),
        ({"row_order": 3}, ColorScheme.MONO, "row_order 3 is not defined"),
        ({"pixel_pack": 2}, ColorScheme.BWRY, "pixel_pack 2 is not defined"),
        ({"frame_scan": 0x10}, ColorScheme.MONO, "frame_scan 0x10 sets reserved bits"),
        ({"plane_invert": 0x04}, ColorScheme.BWR, "plane_invert 0x04 sets reserved bits"),
        ({"reserved": b"\x00\x00\x01\x00"}, ColorScheme.MONO, "reserved bytes"),
        ({"frame_split": 1, "part_width": 396}, ColorScheme.MONO, "part_width 396 at 1 bits"),
        ({"frame_split": 1, "part_width": 400, "part_stride": 394}, ColorScheme.BWRY, "part_stride 394 at 2 bits"),
        ({"frame_split": 1}, ColorScheme.BWR, "one-plane schemes"),
        ({"frame_split": 1}, ColorScheme.BWY, "one-plane schemes"),
        ({"frame_split": 1}, ColorScheme.GRAYSCALE_4, "one-plane schemes"),
        ({"row_order": 2, "frame_scan": 1}, ColorScheme.BWRY, "INTERLEAVE cannot be combined"),
        ({"row_order": 2}, ColorScheme.MONO, "INTERLEAVE is for 2-bit"),
        ({"pixel_pack": 1}, ColorScheme.BWGBRY, "NIBBLE is for 2-bit"),
        ({"part_width": 8}, ColorScheme.MONO, "frame_split is NONE"),
        ({"plane_invert": 2}, ColorScheme.MONO, "plane 1"),
    ],
)
def test_refused_layouts(fields: dict[str, Any], scheme: ColorScheme, match: str) -> None:
    with pytest.raises(FrameLayoutError, match=match):
        scheme_geometry(FrameLayout(**fields), scheme, 792, 272)


def test_interleave_needs_an_even_height() -> None:
    with pytest.raises(FrameLayoutError, match="even height"):
        resolve_layout(FrameLayout(row_order=2), 2, 8, 3)


def test_the_retired_scheme_is_refused_by_its_number() -> None:
    with pytest.raises(FrameLayoutError, match="BWGBRY.*frame_split 1"):
        scheme_geometry(IDENTITY, 8, 800, 480)


def test_prepare_image_refuses_before_dithering() -> None:
    caps = DeviceCapabilities(16, 2, ColorScheme.BWR, frame_layout=FrameLayout(frame_split=1))
    with pytest.raises(FrameLayoutError, match="one-plane"):
        prepare_image(Image.new("RGB", (16, 2)), capabilities=caps, compress=False)


# ── DisplayConfig 0x20 ───────────────────────────────────────────────────────


def test_display_config_round_trips_the_layout_bytes() -> None:
    display = _display(frame_split=1, frame_scan=0x0B, part_width=400, part_stride=392, plane_invert=1)
    display = dataclasses.replace(display, pixel_pack=1, row_order=2)
    raw = serialize_display_config(display)
    assert len(raw) == 46
    assert raw[33:46] == bytes([1, 0x0B, 0x90, 0x01, 0x88, 0x01, 1, 1, 2, 0, 0, 0, 0])
    parsed = DisplayConfig.from_bytes(raw)
    assert parsed == display
    assert serialize_display_config(parsed) == raw
    # Undefined values still parse (refused when used) so a bad config can be read and fixed.
    odd = raw[:33] + bytes([9, 0xF0, 0, 0, 0, 0, 0xFC, 7, 7, 1, 2, 3, 4])
    assert serialize_display_config(DisplayConfig.from_bytes(odd)) == odd


def test_a_scheme_8_config_still_parses() -> None:
    raw = serialize_display_config(_display(color_scheme=8))
    assert DisplayConfig.from_bytes(raw).color_scheme == 8


# ── multi-pass ───────────────────────────────────────────────────────────────


def test_multipass_lays_out_each_pass_frame() -> None:
    entries = tuple(
        CalibrationEntry(codes=c, lab_l=lightness, lab_a=0.0, lab_b=0.0)
        for c, lightness in ((0b0000, 0.0), (0b0101, 100.0), (0b1110, 60.0), (0b0111, 40.0))
    )
    cal = DisplayCalibration(instance_number=0, mode_index=1, flags=0, entries=entries)
    indices = RNG.integers(0, 4, size=(6, 16), dtype=np.uint8)
    layout = FrameLayout(frame_split=1, row_order=RowOrder.FOLD, pixel_pack=1)

    data = encode_multipass(indices, cal, 2, layout=layout, white=1)

    expected = b""
    for pass_index in range(2):
        table = np.array([e.code_for_pass(pass_index) for e in entries], dtype=np.uint8)
        expected += lay_out_plane(table[indices], 2, layout, white=1)
    assert data == expected
    assert len(data) == 2 * 2 * (8 * 4 // 8) * 6  # 2 passes x 2 parts x 4 bytes/row x 6 rows


# ── partial updates ──────────────────────────────────────────────────────────


def _partial(**layout: Any) -> str | object:
    config = _config(
        displays=[_display(color_scheme=0, pixel_width=16, pixel_height=2, partial_update_support=1, **layout)]
    )
    old = bytes(32)
    new = Image.frombytes("P", (16, 2), bytes([1]) + bytes(31))
    state = PartialState(etag=1, last_image=old, width=16, height=2, bytes_per_pixel=1)
    return compute_partial_region(new, state, config, ColorScheme.MONO)


@pytest.mark.parametrize("field", ["frame_split", "frame_scan", "row_order", "pixel_pack"])
def test_partial_falls_back_when_the_layout_reorders(field: str, caplog: pytest.LogCaptureFixture) -> None:
    value = {"frame_split": 1, "frame_scan": FrameScan.MIRROR_X, "row_order": 1, "pixel_pack": 1}[field]
    with caplog.at_level("INFO"):
        assert _partial(**{field: value}) == "fallback_full"
    assert "frame layout" in caplog.text


def test_partial_still_runs_with_plane_invert_only() -> None:
    assert not isinstance(_partial(plane_invert=1), str)


# ── migration ────────────────────────────────────────────────────────────────


def _db_with_layout(panel: str, layout: dict[str, int]) -> PanelDatabase:
    package = files("opendisplay.panels")
    doc = json.loads(gzip.decompress(package.joinpath("drive_presets.json.gz").read_bytes()))
    doc["panels"][panel]["frame_layout"] = layout
    return PanelDatabase(doc)


def test_a_scheme_8_config_migrates_to_bwgbry_halves() -> None:
    db = PanelDatabase.load()
    config = _config(displays=[_display(panel_ic_type=0x0001, color_scheme=8)])
    plan = plan_migration(config, build_profile(FW3, config, None), db)

    assert (plan.needed, plan.blocked, plan.color_scheme) == (True, False, 4)
    assert plan.frame_layout == FrameLayout(frame_split=1)
    assert "color scheme 8 becomes 4" in plan.reason

    migrated = apply_migration(config, plan).displays[0]
    assert (migrated.color_scheme, migrated.frame_split, migrated.part_width) == (4, 1, 0)
    assert migrated.frame_layout == FrameLayout(frame_split=1)
    # Encoding the migrated display gives the bytes scheme 8 sent.
    indices = RNG.integers(0, 6, size=(4, 8), dtype=np.uint8)
    lut = np.array([0, 1, 2, 3, 5, 6], dtype=np.uint8)
    assert encode_4bpp(_p(indices), True, migrated.frame_layout) == _retired_half_planes(lut[indices])

    again = apply_migration(config, plan)
    assert not plan_migration(again, build_profile(FW3, again, None), db).needed


def test_the_preset_frame_layout_is_written_and_a_mismatch_needs_migration() -> None:
    wanted = {"frame_split": 1, "part_width": 400, "part_stride": 392, "frame_scan": 1}
    db = _db_with_layout("0x0001", wanted)
    entry = db.entry_for(0x0001, TargetClass.DEFAULT)
    assert entry is not None and entry.frame_layout == FrameLayout.from_mapping(wanted)

    config = _config(displays=[_display(panel_ic_type=0x0001, color_scheme=0)])
    migrated = apply_migration(config, plan_migration(config, build_profile(FW3, config, None), db))
    assert migrated.displays[0].frame_layout.to_mapping() == {**FrameLayout().to_mapping(), **wanted}
    assert not plan_migration(migrated, build_profile(FW3, migrated, None), db).needed

    # Drive data in place but a stale 0x20 layout: the migration is the layout alone.
    stale = dataclasses.replace(migrated, displays=[dataclasses.replace(migrated.displays[0], part_stride=0)])
    plan = plan_migration(stale, build_profile(FW3, stale, None), db)
    assert (plan.needed, plan.blocked, plan.add, plan.remove) == (True, False, (), ())
    assert plan.reason == "the frame layout differs from the panel's preset"
    assert apply_migration(stale, plan).displays[0].part_stride == 392


def test_an_entry_without_frame_layout_leaves_the_stored_layout() -> None:
    db = PanelDatabase.load()
    entry = db.entry_for(0x0001, TargetClass.DEFAULT)
    assert entry is not None and entry.frame_layout is None
    assert display_layout_fix(_display(frame_scan=2), entry) == (None, None)


def test_an_invalid_preset_frame_layout_blocks_the_entry() -> None:
    db = _db_with_layout("0x0001", {"frame_split": 7})
    entry = db.entry_for(0x0001, TargetClass.DEFAULT)
    assert entry is not None and not entry.usable
    assert "frame_split 7" in (entry.blocked_reason or "")
    with pytest.raises(ValueError, match="unknown keys"):
        FrameLayout.from_mapping({"split": 1})


def test_prepare_image_uses_the_config_layout() -> None:
    config = _config(displays=[_display(color_scheme=0, pixel_width=16, pixel_height=4, frame_scan=1)])
    image = Image.new("RGB", (16, 4), "white")
    data, _, dithered = prepare_image(image, config=config, dither_mode=DitherMode.NONE, compress=False)
    assert data == encode_1bpp(dithered, FrameLayout(frame_scan=1))


def test_prior_masking_counts_the_laid_out_cells() -> None:
    from opendisplay.device.api import _frame_cells  # pylint: disable=import-outside-toplevel

    plain = _display(color_scheme=0, pixel_width=792, pixel_height=272)
    crow = dataclasses.replace(plain, frame_split=1, part_width=400, part_stride=392, frame_scan=1)
    assert _frame_cells(plain) == 792 * 272
    assert _frame_cells(crow) == 800 * 272
