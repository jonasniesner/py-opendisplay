"""Host-side prior frames: drive only the pixels that change (presets plan F5/F6, BWRY first).

A mode whose 0xAE tail declares ``hold_mask`` leaves every pixel sent with one of those codes as
it is. With the frame the panel shows now (the "prior", kept by the host) a new frame can be sent
with the lowest hold code in place of every unchanged pixel, so only changed pixels are driven
and the rest of the panel does not flash.

Works on packed single-pass frames of 1, 2 or 4 bits per pixel. Both frames must come from the
same encoder (same size, packing and code map); the comparison is per code field, so the packing
order does not matter. No I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PriorConflictError(ValueError):
    """A changed pixel needs a code the mode holds, so the mode cannot draw this update."""


@dataclass(frozen=True)
class MaskedFrame:
    """A frame with unchanged pixels replaced by the hold code."""

    frame: bytes
    changed: int
    total: int


def _unpack(data: bytes, bits: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    per = 8 // bits
    shifts = np.arange(8 - bits, -1, -bits, dtype=np.uint8)
    return ((raw[:, None] >> shifts[None, :]) & ((1 << bits) - 1)).reshape(-1) if per > 1 else raw.copy()


def _pack(codes: np.ndarray, bits: int) -> bytes:
    per = 8 // bits
    shifts = np.arange(8 - bits, -1, -bits, dtype=np.uint8)
    grouped = codes.reshape(-1, per).astype(np.uint8)
    return (grouped << shifts[None, :]).sum(axis=1, dtype=np.uint16).astype(np.uint8).tobytes()


def bits_per_pixel(frame_len: int, pixels: int) -> int:
    """The code width of a packed frame of ``frame_len`` bytes for ``pixels`` pixels.

    Raises:
        ValueError: The length matches no supported width.
    """
    for bits in (1, 2, 4, 8):
        if frame_len * 8 == pixels * bits:
            return bits
    raise ValueError(f"a {frame_len} B frame is no packed 1/2/4/8-bit frame of {pixels} pixels")


def mask_unchanged(frame: bytes, prior: bytes, hold_codes: frozenset[int], pixels: int) -> MaskedFrame:
    """Replace every pixel equal to the prior with the lowest hold code.

    Raises:
        ValueError: The frames differ in size, or there are no hold codes.
        PriorConflictError: A changed pixel's new code is a hold code (the mode cannot draw it).
    """
    if not hold_codes:
        raise ValueError("the mode declares no hold codes")
    if len(frame) != len(prior):
        raise ValueError(f"frame is {len(frame)} B but the prior is {len(prior)} B")
    bits = bits_per_pixel(len(frame), pixels)
    new = _unpack(frame, bits)
    old = _unpack(prior, bits)
    changed = new != old
    bad = changed & np.isin(new, sorted(hold_codes))
    if bad.any():
        codes = sorted({int(c) for c in new[bad]})
        raise PriorConflictError(f"{int(bad.sum())} changed pixels need codes {codes}, which this mode only holds")
    out = np.where(changed, new, min(hold_codes))
    return MaskedFrame(frame=_pack(out, bits), changed=int(changed.sum()), total=int(new.size))
