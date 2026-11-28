"""Bitplane encoding for multi-color e-paper displays."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..models.enums import ColorScheme
from .layout import IDENTITY, FrameLayout, lay_out_plane


def encode_bitplanes(
    image: Image.Image,
    color_scheme: ColorScheme,
    layout: FrameLayout = IDENTITY,
) -> tuple[bytes, bytes]:
    """Encode image to bitplane format for BWR/BWY displays.

    BWR/BWY displays use two bitplanes:
    - Plane 1 (BW): Black/White layer
    - Plane 2 (R/Y): Red/Yellow accent color layer

    Args:
        image: Dithered palette image
        color_scheme: Must be BWR or BWY
        layout: The display's frame layout, applied to each plane.

    Returns:
        Tuple of (plane1_bytes, plane2_bytes)

    Raises:
        ValueError: If color_scheme is not BWR or BWY
    """
    if color_scheme not in (ColorScheme.BWR, ColorScheme.BWY):
        raise ValueError(f"Bitplane encoding only supports BWR/BWY, got {color_scheme.name}")

    if image.mode != "P":
        raise ValueError(f"Expected palette image, got {image.mode}")

    pixels = np.asarray(image)

    # Palette mapping (matches the website encoder and firmware boot renderer):
    # Index 0 = Black      -> BW=0, R/Y=0
    # Index 1 = White      -> BW=1, R/Y=0
    # Index 2 (BWR = Red)  -> BW=1, R/Y=1  (red sets BOTH the BW and accent plane)
    # Index 2 (BWY = Yellow) -> BW=0, R/Y=1
    # Each row is zero-padded to a byte boundary (8 pixels per byte, MSB first).
    if color_scheme == ColorScheme.BWR:
        plane1_mask = (pixels == 1) | (pixels == 2)  # white and red both set BW
    else:  # ColorScheme.BWY
        plane1_mask = pixels == 1

    return _split_planes([plane1_mask, pixels == 2], layout, whites=(1, 0))


def _split_planes(planes: list[np.ndarray], layout: FrameLayout, whites: tuple[int, int]) -> tuple[bytes, bytes]:
    """Lay out two 1-bit planes and return them separately."""
    first, second = (
        lay_out_plane(p.astype(np.uint8), 1, layout, white=whites[i], plane=i, planes=2) for i, p in enumerate(planes)
    )
    return first, second


def encode_gray4_bitplanes(
    image: Image.Image,
    gray_codes: tuple[int, int, int, int],
    layout: FrameLayout = IDENTITY,
) -> tuple[bytes, bytes]:
    """Encode a 4-gray palette image to two 1-bit controller planes.

    Each pixel's dither level (palette index 0=black..3=white) is mapped through
    the panel's gray-code table to a 2-bit stored code; plane0 carries the code's
    bit0 and plane1 its bit1, matching the firmware's bbepSetPixel4Gray. The
    firmware streams these planes straight to PLANE_0/PLANE_1, so no on-device
    de-interleave is needed.

    Args:
        image: Dithered palette image (mode "P", indices 0..3)
        gray_codes: level -> stored 2-bit code (see display_palettes.get_gray4_codes)
        layout: The display's frame layout, applied to each plane.

    Returns:
        Tuple of (plane0_bytes, plane1_bytes)

    Raises:
        ValueError: If image is not a palette image
    """
    if image.mode != "P":
        raise ValueError(f"Expected palette image, got {image.mode}")

    pixels = np.asarray(image)

    # Map each pixel's dither level (index 0..3) through the panel's gray-code
    # table to a 2-bit stored code, then split into two 1-bit planes.
    codes = np.asarray(gray_codes, dtype=np.uint8)[pixels & 0x03]
    white = gray_codes[3]
    return _split_planes([codes & 0x01, (codes >> 1) & 0x01], layout, whites=(white & 1, (white >> 1) & 1))
