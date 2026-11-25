"""Extension tails of the 0xAE display_mode and 0xB1 display_panel bodies.

Firmware 3.0 reads a 0xAE body up to ``37 + seq_len`` and a 0xB1 body up to ``13 + sleep_len``
and ignores whatever follows. Those trailing bytes ("the tail") carry host information that
was added on 2026-11-25 (presets plan sections 2.9, 2.10 and 5.3):

- 0xAE mode tail: whether a mode is experimental, what one update in it costs from the
  panel's update budget, and which mode must run right before it.
- 0xB1 drive tail: which drive preset (panel key + revision) the drive data came from,
  whether someone hand-edited it, and the panel's update budget.

PROVISIONAL. None of this is in ``opendisplay_structs.h`` yet; the layouts below are this
library's proposal for testing the firmware side against. Every version only appends fields,
so a reader of version N parses the fields it knows from any version >= N and keeps the rest
as ``extra``. Fields after the core may be left off; a missing field reads as ``None``.

0xAE mode tail, ``ext_version`` 1 (2 bytes core + 3 optional)::

    [0] ext_version  u8   1
    [1] mode_flags   u8   bit0 EXPERIMENTAL; bits 1-7 reserved (0)
    [2] budget_cost  u8   units of the panel's budget one update uses; 0 = a stock/scrub
                          mode, which resets the host's count          (optional)
    [3] scrub_before u8   mode_index the host must run right before pass 1 of this mode
                          (multi-pass); 0xFF = none                     (optional)
    [4] hold_mask    u8   bit n set = frame code n leaves a pixel as it is in this mode. With
                          a known prior frame the host sends the lowest such code for every
                          unchanged pixel, so only changed pixels are driven; a changed pixel
                          whose new code is in the mask cannot be drawn by this mode (the host
                          runs a stock full instead). 0x00 or 0xFF = none (optional)

0xB1 drive tail, ``ext_version`` 1 (6 bytes core + 3 optional)::

    [0]   ext_version      u8   1
    [1:3] drive_preset_key u16  panel_ic_type key of the preset the drive data came from
                                (0 = not from a preset)
    [3:5] drive_rev        u16  that preset entry's rev (0 = unknown)
    [5]   drive_flags      u8   bit0 HAND_EDITED (never offer a preset update); bits 1-7 0
    [6]   budget_limit     u8   budget units between stock refreshes; 0xFF = no limit (optional)
    [7]   lut_family       u8   0 none, 1 S6-800, 2 JD-BWRY-535, 3 x4-536          (optional)
    [8]   busy_timeout_s   u8   BUSY ceiling in seconds, wins over busy_timeout_ms when
                                non-zero; 0 = use busy_timeout_ms               (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Final, Self

#: Byte offset of ``seq_len`` in a 0xAE body, and its head size.
MODE_HEAD_SIZE: Final = 37
#: Byte offset of ``sleep_seq_len`` in a 0xB1 body is 12; head size 13.
PANEL_HEAD_SIZE: Final = 13

EXT_VERSION: Final = 1

MODE_FLAG_EXPERIMENTAL: Final = 0x01
DRIVE_FLAG_HAND_EDITED: Final = 0x01

SCRUB_NONE: Final = 0xFF
BUDGET_UNLIMITED: Final = 0xFF


class LutFamily(IntEnum):
    """Which waveform format a panel's LUT slots hold (device-side lint picks its checks by it)."""

    NONE = 0
    S6_800 = 1
    JD_BWRY_535 = 2
    X4_536 = 3


def mode_tail_offset(body: bytes) -> int:
    """Where the tail of a 0xAE body starts (``37 + seq_len``).

    Raises:
        ValueError: The body is shorter than its head or its declared sequence.
    """
    if len(body) < MODE_HEAD_SIZE:
        raise ValueError(f"display_mode body too short: {len(body)} bytes (need at least {MODE_HEAD_SIZE})")
    end = MODE_HEAD_SIZE + int.from_bytes(body[35:37], "little")
    if len(body) < end:
        raise ValueError(f"display_mode body too short: {len(body)} bytes (need at least {end})")
    return end


def panel_tail_offset(body: bytes) -> int:
    """Where the tail of a 0xB1 body starts (``13 + sleep_seq_len``).

    Raises:
        ValueError: The body is shorter than its head or its declared sleep sequence.
    """
    if len(body) < PANEL_HEAD_SIZE:
        raise ValueError(f"display_panel body too short: {len(body)} bytes (need at least {PANEL_HEAD_SIZE})")
    end = PANEL_HEAD_SIZE + body[12]
    if len(body) < end:
        raise ValueError(f"display_panel body too short: {len(body)} bytes (need at least {end})")
    return end


def _u8_field(tail: bytes, offset: int) -> int | None:
    return tail[offset] if len(tail) > offset else None


@dataclass(frozen=True)
class ModeExt:
    """The 0xAE mode tail (provisional layout, see the module docstring)."""

    mode_flags: int = 0
    budget_cost: int | None = None
    scrub_before: int | None = None
    hold_mask: int | None = None
    ext_version: int = EXT_VERSION
    extra: bytes = b""  # fields of a later ext_version, verbatim

    _CORE: ClassVar[int] = 2
    _KNOWN: ClassVar[int] = 5

    @property
    def experimental(self) -> bool:
        """The mode has not been seen right on enough glass; never pick it by default."""
        return bool(self.mode_flags & MODE_FLAG_EXPERIMENTAL)

    @property
    def scrub_mode(self) -> int | None:
        """The mode that must run right before this one, or ``None``."""
        return None if self.scrub_before in (None, SCRUB_NONE) else self.scrub_before

    @property
    def hold_codes(self) -> frozenset[int]:
        """The frame codes that leave a pixel as it is in this mode (empty = none)."""
        if self.hold_mask in (None, 0, 0xFF):
            return frozenset()
        return frozenset(n for n in range(8) if self.hold_mask >> n & 1)

    @classmethod
    def from_tail(cls, tail: bytes) -> Self | None:
        """Parse a mode tail; ``None`` for an empty tail, version 0 or a truncated core."""
        if len(tail) < cls._CORE or tail[0] < 1:
            return None
        return cls(
            ext_version=tail[0],
            mode_flags=tail[1],
            budget_cost=_u8_field(tail, 2),
            scrub_before=_u8_field(tail, 3),
            hold_mask=_u8_field(tail, 4),
            extra=tail[cls._KNOWN :],
        )

    def to_bytes(self) -> bytes:
        """Serialise; optional fields are written up to the last one that is set.

        Raises:
            ValueError: A field is out of range, or an optional field is set after an unset one.
        """
        _check_u8("mode tail", ext_version=self.ext_version, mode_flags=self.mode_flags)
        out = bytearray([self.ext_version, self.mode_flags])
        optional = [self.budget_cost, self.scrub_before, self.hold_mask]
        _append_optional(out, optional, "mode tail")
        if self.extra:
            if len(out) != self._KNOWN:
                raise ValueError("mode tail: extra bytes need every version-1 field set")
            out += self.extra
        return bytes(out)


@dataclass(frozen=True)
class DriveExt:  # pylint: disable=too-many-instance-attributes
    """The 0xB1 drive tail (provisional layout, see the module docstring)."""

    drive_preset_key: int = 0
    drive_rev: int = 0
    drive_flags: int = 0
    budget_limit: int | None = None
    lut_family: int | None = None
    busy_timeout_s: int | None = None
    ext_version: int = EXT_VERSION
    extra: bytes = b""

    _CORE: ClassVar[int] = 6
    _KNOWN: ClassVar[int] = 9

    @property
    def hand_edited(self) -> bool:
        """Someone changed the drive data by hand: never offer a preset update."""
        return bool(self.drive_flags & DRIVE_FLAG_HAND_EDITED)

    @property
    def lut_family_enum(self) -> LutFamily | int | None:
        """``lut_family`` as an enum, a raw int the library does not know, or ``None``."""
        if self.lut_family is None:
            return None
        try:
            return LutFamily(self.lut_family)
        except ValueError:
            return self.lut_family

    @classmethod
    def from_tail(cls, tail: bytes) -> Self | None:
        """Parse a drive tail; ``None`` for an empty tail, version 0 or a truncated core."""
        if len(tail) < cls._CORE or tail[0] < 1:
            return None
        return cls(
            ext_version=tail[0],
            drive_preset_key=int.from_bytes(tail[1:3], "little"),
            drive_rev=int.from_bytes(tail[3:5], "little"),
            drive_flags=tail[5],
            budget_limit=_u8_field(tail, 6),
            lut_family=_u8_field(tail, 7),
            busy_timeout_s=_u8_field(tail, 8),
            extra=tail[cls._KNOWN :],
        )

    def to_bytes(self) -> bytes:
        """Serialise; optional fields are written up to the last one that is set.

        Raises:
            ValueError: A field is out of range, or an optional field is set after an unset one.
        """
        for name, value in (("drive_preset_key", self.drive_preset_key), ("drive_rev", self.drive_rev)):
            if not 0 <= value <= 0xFFFF:
                raise ValueError(f"drive tail: {name} out of range (u16): {value}")
        _check_u8("drive tail", ext_version=self.ext_version, drive_flags=self.drive_flags)
        out = bytearray([self.ext_version])
        out += self.drive_preset_key.to_bytes(2, "little") + self.drive_rev.to_bytes(2, "little")
        out.append(self.drive_flags)
        _append_optional(out, [self.budget_limit, self.lut_family, self.busy_timeout_s], "drive tail")
        if self.extra:
            if len(out) != self._KNOWN:
                raise ValueError("drive tail: extra bytes need every version-1 field set")
            out += self.extra
        return bytes(out)


def _check_u8(what: str, **values: int) -> None:
    for name, value in values.items():
        if not 0 <= value <= 0xFF:
            raise ValueError(f"{what}: {name} out of range (u8): {value}")


def _append_optional(out: bytearray, values: list[int | None], what: str) -> None:
    last = max((i for i, v in enumerate(values) if v is not None), default=-1)
    for i in range(last + 1):
        value = values[i]
        if value is None:
            raise ValueError(f"{what}: optional field {i} is unset but a later one is set")
        _check_u8(what, value=value)
        out.append(value)


def with_mode_tail(body: bytes, ext: ModeExt | None) -> bytes:
    """A 0xAE body with its tail replaced by ``ext`` (``None`` removes the tail)."""
    head = body[: mode_tail_offset(body)]
    return head if ext is None else head + ext.to_bytes()


def with_drive_tail(body: bytes, ext: DriveExt | None) -> bytes:
    """A 0xB1 body with its tail replaced by ``ext`` (``None`` removes the tail)."""
    head = body[: panel_tail_offset(body)]
    return head if ext is None else head + ext.to_bytes()
