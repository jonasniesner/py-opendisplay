"""The DisplayConfig frame layout (protocol structs 2.5): the order a panel takes its frame bytes in.

The device forwards uploaded bytes unchanged, so a panel whose controllers do not take one
linear row-major stream needs the host to produce their order. The input is a 2-D array of
panel codes per plane (after dithering and ``code_map``); per plane:

1. ``frame_split`` cuts the columns into parts, one per controller;
2. ``frame_scan`` mirroring and ``row_order`` order the codes inside each part, and
   ``FrameScan.COLUMN_MAJOR`` emits the packed bytes column by column;
3. ``pixel_pack``, ``FrameScan.LSB_FIRST`` and ``plane_invert`` shape each byte.

An all-zero layout packs exactly as the colour scheme always has: rows top to bottom, MSB
first, each row zero-padded to a byte. No I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from ..exceptions import FrameLayoutError
from ..models.enums import ColorScheme, FrameScan, FrameSplit, PixelPack, PlaneInvert, RowOrder
from ..models.frame_layout import IDENTITY, FrameLayout

__all__ = [
    "IDENTITY",
    "RETIRED_SPLIT_REASON",
    "SCHEME_BITS",
    "SCHEME_PLANES",
    "FrameGeometry",
    "FrameLayout",
    "FrameLayoutError",
    "frame_size",
    "lay_out_plane",
    "lay_out_planes",
    "resolve_layout",
    "scheme_geometry",
]

#: Bits per pixel code and plane count of each scheme a frame is packed for.
SCHEME_BITS: Final[Mapping[int, int]] = {
    ColorScheme.MONO: 1,
    ColorScheme.BWR: 1,
    ColorScheme.BWY: 1,
    ColorScheme.BWRY: 2,
    ColorScheme.BWGBRY: 4,
    ColorScheme.GRAYSCALE_4: 1,
    ColorScheme.GRAYSCALE_16: 4,
    ColorScheme.SEVEN_COLOR: 4,
}
SCHEME_PLANES: Final[Mapping[int, int]] = {
    ColorScheme.BWR: 2,
    ColorScheme.BWY: 2,
    ColorScheme.GRAYSCALE_4: 2,
}

RETIRED_SPLIT_REASON: Final = (
    "color scheme 8 (BWGBRY_SPLIT) is retired: migrate the config to color scheme 4 (BWGBRY) "
    "with frame_split 1 (HALVES)"
)


@dataclass(frozen=True)
class FrameGeometry:
    """A layout resolved against a panel: what one plane of a frame looks like on the wire.

    Attributes:
        parts: 1, or 2 for ``FrameSplit.HALVES``.
        part_width: Columns each part carries.
        part_stride: Panel column where part ``n`` starts is ``n * part_stride``.
        cell_bits: Bits each code takes in a byte (the scheme's, or 4 for ``PixelPack.NIBBLE``).
        bytes_per_part: Bytes of one part of one plane.
    """

    parts: int
    part_width: int
    part_stride: int
    cell_bits: int
    height: int
    bytes_per_part: int

    @property
    def plane_bytes(self) -> int:
        """Bytes of one laid-out plane."""
        return self.parts * self.bytes_per_part

    @property
    def cells(self) -> int:
        """Code cells of one laid-out plane, padding columns included."""
        return self.parts * self.part_width * self.height


def resolve_layout(  # pylint: disable=too-many-branches
    layout: FrameLayout, bits: int, width: int, height: int, *, planes: int = 1
) -> FrameGeometry:
    """Check ``layout`` against a panel and work out its geometry.

    Args:
        layout: The DisplayConfig layout fields.
        bits: Bits per pixel code of the scheme (1, 2 or 4).
        width: Panel columns.
        height: Panel rows.
        planes: Planes the scheme sends (2 for BWR, BWY and GRAY4).

    Raises:
        FrameLayoutError: Any value the protocol does not define, or a combination this panel
            and scheme cannot take; the message names the field.
    """
    layout.check_defined()
    split = FrameSplit(layout.frame_split)
    row_order = RowOrder(layout.row_order)
    scan = FrameScan(layout.frame_scan)
    if layout.plane_invert & PlaneInvert.PLANE_1 and planes < 2:
        raise FrameLayoutError("plane_invert bit 1 names plane 1, which a one-plane scheme does not have")
    cell_bits = bits
    if layout.pixel_pack == PixelPack.NIBBLE:
        if bits != 2:
            raise FrameLayoutError(f"pixel_pack NIBBLE is for 2-bit codes, not {bits}-bit")
        cell_bits = 4
    if row_order == RowOrder.INTERLEAVE:
        if bits != 2:
            raise FrameLayoutError(f"row_order INTERLEAVE is for 2-bit codes, not {bits}-bit")
        if height % 2:
            raise FrameLayoutError(f"row_order INTERLEAVE needs an even height, not {height}")
        if scan & FrameScan.COLUMN_MAJOR:
            raise FrameLayoutError("row_order INTERLEAVE cannot be combined with frame_scan COLUMN_MAJOR")

    if split == FrameSplit.NONE:
        if layout.part_width or layout.part_stride:
            raise FrameLayoutError("part_width and part_stride must be 0 when frame_split is NONE")
        parts, part_width, part_stride = 1, width, 0
    else:
        if planes != 1:
            raise FrameLayoutError(f"frame_split HALVES is for one-plane schemes; this scheme sends {planes} planes")
        parts = 2
        part_width = layout.part_width or width // 2
        part_stride = layout.part_stride or part_width
        if part_width <= 0:
            raise FrameLayoutError(f"part_width resolves to {part_width} on a {width}-column panel")
        if part_width * cell_bits % 8:
            raise FrameLayoutError(
                f"part_width {part_width} at {cell_bits} bits per pixel does not end on a byte boundary"
            )
        if part_stride * bits % 8:
            raise FrameLayoutError(
                f"part_stride {part_stride} at {bits} bits per pixel does not start on a byte boundary"
            )
    if row_order == RowOrder.INTERLEAVE:
        stream_rows, stream_cells = height // 2, 2 * part_width
    else:
        stream_rows, stream_cells = height, part_width
    row_bytes = -(-stream_cells * cell_bits // 8)
    return FrameGeometry(
        parts=parts,
        part_width=part_width,
        part_stride=part_stride,
        cell_bits=cell_bits,
        height=height,
        bytes_per_part=row_bytes * stream_rows,
    )


def _fold_order(height: int) -> np.ndarray:
    """Rows h-1, 0, h-2, 1, ...; an odd height ends on the middle row."""
    order: list[int] = []
    lo, hi = 0, height - 1
    while lo <= hi:
        order.append(hi)
        if lo != hi:
            order.append(lo)
        lo, hi = lo + 1, hi - 1
    return np.asarray(order, dtype=np.intp)


def _pack_rows(stream: np.ndarray, cell_bits: int, lsb_first: bool) -> np.ndarray:
    """Pack each row of codes ``cell_bits`` wide, zero-padding it to a byte; returns rows x bytes."""
    per = 8 // cell_bits
    rows, cells = stream.shape
    pad = (-cells) % per
    if pad:
        stream = np.pad(stream, ((0, 0), (0, pad)))
    grouped = stream.reshape(rows, -1, per).astype(np.uint8)
    slots = np.arange(per) if lsb_first else np.arange(per - 1, -1, -1)
    shifts = (slots * cell_bits).astype(np.uint8)
    mask = np.uint8((1 << cell_bits) - 1) if cell_bits < 8 else np.uint8(0xFF)
    packed = np.zeros(grouped.shape[:2], dtype=np.uint8)
    for k in range(per):
        packed |= (grouped[:, :, k] & mask) << shifts[k]
    return packed


def lay_out_plane(
    codes: np.ndarray,
    bits: int,
    layout: FrameLayout = IDENTITY,
    *,
    white: int = 0,
    plane: int = 0,
    planes: int = 1,
) -> bytes:
    """Lay out one plane of panel codes as wire bytes.

    The packer entry point: every frame the library uploads passes through here.

    Args:
        codes: Panel codes, rows x columns, each ``< 2**bits``.
        bits: Bits per code of the scheme (1, 2 or 4).
        layout: The DisplayConfig layout.
        white: The code sent for columns past the panel's edge (the scheme's white).
        plane: This plane's index, for ``plane_invert``.
        planes: How many planes the scheme sends.

    Raises:
        FrameLayoutError: See ``resolve_layout``.
    """
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2:
        raise ValueError(f"codes must be rows x columns, not shape {codes.shape}")
    height, width = codes.shape
    geometry = resolve_layout(layout, bits, width, height, planes=planes)
    scan = FrameScan(layout.frame_scan)
    row_order = RowOrder(layout.row_order)
    out = []
    for part in range(geometry.parts):
        start = part * geometry.part_stride
        columns = np.arange(start, start + geometry.part_width)
        inside = columns < width
        block = np.full((height, geometry.part_width), white, dtype=np.uint8)
        block[:, inside] = codes[:, columns[inside]]
        if scan & FrameScan.MIRROR_X:
            block = block[:, ::-1]
        if scan & FrameScan.MIRROR_Y:
            block = block[::-1, :]
        if row_order == RowOrder.FOLD:
            block = block[_fold_order(height), :]
        elif row_order == RowOrder.INTERLEAVE:
            pairs = block.reshape(height // 2, 2, geometry.part_width)
            block = np.stack((pairs[:, 1, :], pairs[:, 0, :]), axis=2).reshape(height // 2, -1)
        packed = _pack_rows(block, geometry.cell_bits, bool(scan & FrameScan.LSB_FIRST))
        if scan & FrameScan.COLUMN_MAJOR:
            packed = packed.T
        if layout.plane_invert & (1 << plane):
            packed = packed ^ np.uint8(0xFF)
        out.append(np.ascontiguousarray(packed).tobytes())
    return b"".join(out)


def lay_out_planes(
    planes: Sequence[np.ndarray], bits: int, layout: FrameLayout = IDENTITY, *, whites: Sequence[int] | None = None
) -> bytes:
    """Lay out every plane of a frame and concatenate them, plane 0 first."""
    whites = whites if whites is not None else [0] * len(planes)
    return b"".join(
        lay_out_plane(p, bits, layout, white=whites[i], plane=i, planes=len(planes)) for i, p in enumerate(planes)
    )


def scheme_geometry(layout: FrameLayout, color_scheme: int, width: int, height: int) -> FrameGeometry:
    """``resolve_layout`` for a DisplayConfig colour scheme.

    Raises:
        FrameLayoutError: Scheme 8 (retired), an unknown scheme, or a layout the scheme cannot take.
    """
    if color_scheme == ColorScheme.BWGBRY_SPLIT:
        raise FrameLayoutError(RETIRED_SPLIT_REASON)
    bits = SCHEME_BITS.get(color_scheme)
    if bits is None:
        raise FrameLayoutError(f"color scheme {color_scheme} has no frame packing")
    return resolve_layout(layout, bits, width, height, planes=SCHEME_PLANES.get(color_scheme, 1))


def frame_size(layout: FrameLayout, color_scheme: int, width: int, height: int) -> int:
    """Bytes of one laid-out frame of ``color_scheme`` on a ``width`` x ``height`` panel."""
    geometry = scheme_geometry(layout, color_scheme, width, height)
    return geometry.plane_bytes * SCHEME_PLANES.get(color_scheme, 1)
