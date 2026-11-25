"""Encoding one image for a multi-pass drive mode (firmware 3.0, packet 0xAE with passes > 1).

A multi-pass mode drives the panel several times per upload. Its renderable colours come from
the mode's calibration packet (0xB0): each entry has a CIELAB value and, per pass, the panel
code that pass sends. The image is dithered against those colours, then every pass gets an
ordinary frame of the panel's packing, and the frames are concatenated. BWRY16 codes are 2-bit
and the frames 2 bpp; Spectra 6 codes are 3-bit (protocol structs 2.4,
``OD_DRIVE_CAL_FLAG_CODES_3BIT``) and the frames 4 bpp nibbles, split into half planes on
dual-controller panels (BWGBRY_SPLIT). No I/O.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from epaper_dithering import Palette, PaletteEntry
from PIL import Image

from ..exceptions import ImageEncodingError
from ..models.panel_drive import DisplayCalibration
from .images import encode_2bpp, encode_4bpp

MAX_PASSES: Final = 8  # CalibrationEntry.codes holds 2 bits for each of up to 8 passes (3 bits: 5)
SCHEME_BWRY16: Final = 10  # the multi-pass scheme: 2 bpp per pass, colours from the 0xB0 palette


def mode_palette(calibration: DisplayCalibration) -> Palette:
    """The palette a mode renders, from its 0xB0 entries, in entry order.

    The entries are absolute CIELAB (measured or nominal); the lightest is taken as the paper
    white. ``measured`` follows the packet's "measured on this unit" flag.

    Raises:
        ImageEncodingError: The calibration has no entries.
    """
    if not calibration.entries:
        raise ImageEncodingError(f"The calibration of drive mode {calibration.mode_index} has no colours")
    entries = [PaletteEntry(appearance=(e.lab_l, e.lab_a, e.lab_b)) for e in calibration.entries]
    white = max(range(len(entries)), key=lambda i: calibration.entries[i].lab_l)
    return Palette.from_absolute_lab(entries, white_index=white, measured=calibration.measured)


def encode_multipass(
    indices: np.ndarray, calibration: DisplayCalibration, passes: int, *, half_planes: bool = False
) -> bytes:
    """Encode palette indices (one per pixel, rows x columns) into ``passes`` concatenated frames.

    In pass ``p`` a pixel sends ``calibration.entries[index].code_for_pass(p)``: panel codes, to
    which ``code_map`` is never applied. 2-bit codes give 2 bpp frames; 3-bit codes give 4 bpp
    nibble frames (high nibble first), as left then right half planes when ``half_planes``.

    Raises:
        ImageEncodingError: An index outside the palette, or more passes than the codes hold.
    """
    limit = min(MAX_PASSES, calibration.max_passes)
    if not 1 <= passes <= limit:
        raise ImageEncodingError(
            f"A drive mode with {calibration.code_bits}-bit palette codes has 1 to {limit} passes, not {passes}"
        )
    if indices.size and int(indices.max()) >= len(calibration.entries):
        raise ImageEncodingError(
            f"Palette index {int(indices.max())} is outside the mode's {len(calibration.entries)}-colour palette"
        )
    frames = []
    for pass_index in range(passes):
        table = np.array([e.code_for_pass(pass_index) for e in calibration.entries], dtype=np.uint8)
        codes = Image.fromarray(table[indices], mode="P")
        frames.append(encode_4bpp(codes, half_planes=half_planes) if calibration.code_bits == 3 else encode_2bpp(codes))
    return b"".join(frames)
