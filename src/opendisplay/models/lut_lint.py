"""The device-side table lint and never-send list, mirrored so the host refuses what the device would.

Port of Firmware_Unified ``shared/core/od_epd_lut.c`` (``od_lut_lint``, ``od_lut_never_send``)
and the command scan in ``shared/core/od_epd.c`` (``od_epd_validate``), branch
``fw-refresh-modes``. The device runs these only when the 0xB1 drive tail declares a
``lut_family`` (:class:`~opendisplay.models.drive_ext.LutFamily`); a failing stored slot reads
as absent there, and a mode that uploads an absent slot, a failing inline table or a sequence
with a never-send command is refused at START (``OD_ERR_DRIVE_SEQ_UNSUPPORTED``, 0x07).

PROVISIONAL, like the layouts in :mod:`opendisplay.models.drive_ext`.

Families:

- 0 none (or a value the firmware does not know): only the size, at most 1024 bytes.
- 1 S6-800: 799 or 800 bytes, header bytes 0-7 not all zero, never 8 zero bytes in a row
  (anywhere, the header included). The 7.09in class hangs BUSY on either.
- 2 JD-BWRY: exactly 535 bytes, plus the block check below.
- 3 x4: 535 or 536 bytes, plus the same block check (the firmware runs it for both JD families).

JD block check: 6 blocks of 8 rows x 11 bytes at ``8 + 88k``. Bytes are active-low scrambled
binary (:func:`jd_value`). Row ``r``'s repeat ``M`` is the byte just before the row; its timing
is ``R_A * (TP_A bytes 2-4) + R_B * (TP_B bytes 6-8)`` with ``R_A`` at byte 5 and ``R_B`` at
byte 9. Scanning rows 0-6, the first row whose ``M`` is 0 **or** whose timing is 0 ends the
block; it is an error when that row has timing 0 but ``M != 0`` (a block that runs on into rows
with a repeat and no time never releases BUSY). Row 7 is exempt.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from .drive_ext import LutFamily
from .panel_drive import LUT_SLOT_NONE, DisplayLut, DisplayMode

LUT_MAX_BYTES: Final = 1024

S6_HEADER: Final = 8
S6_ZERO_RUN: Final = 8
JD_BODY: Final = 8
JD_BLOCKS: Final = 6
JD_BLOCK: Final = 88
JD_ROW: Final = 11
JD_LAST_CHECKED_ROW: Final = 6  # row 7 of a block is exempt

#: Commands a sequence of a declared family may never send: OTP/MTP programming and the vendor
#: commands the datasheets warn about. 0x4D and 0xB6 are not here: stock inits send them.
NEVER_SEND: Final = frozenset({0x90, 0x91, 0xA0, 0xA1, 0xA2, 0xA3, 0x8F, 0xFB, 0xFF})
#: The TSC read that may not follow a table upload before the refresh (it reloads the
#: waveform register from OTP on the 7.09in class).
CMD_TSC_READ: Final = 0x40
CMD_REFRESH: Final = 0x12

# Drive sequence escapes ([0x00][esc][args]); a plain step is [n][cmd][n-1 data bytes].
ESC_END: Final = 0x00
ESC_WAVE: Final = 0x02  # SUBST: [cnt][(kind, val) * cnt]
ESC_LUT: Final = 0x04
ESC_WRITE_SCRATCH: Final = 0x0A  # [cmd][off][len]
ESC_LUT_SLICE: Final = 0x0B
ESC_WINDOW: Final = 0x0C  # [cnt][(kind, val) * cnt]
_ESC_ARGS: Final = {
    0x01: 0,  # BUSY
    0x03: 0,  # IMAGE
    ESC_LUT: 0,
    0x05: 2,  # DELAY
    0x06: 2,  # SET_PIN
    0x07: 2,  # IMAGE_SLICE
    0x08: 1,  # CS_SELECT
    0x09: 2,  # READ_SCRATCH
    ESC_WRITE_SCRATCH: 3,
    ESC_LUT_SLICE: 4,
    0x0D: 3,  # FILL
}
_MAX_PAIRS: Final = 32


def jd_value(byte: int) -> int:
    """Decode one JD-BWRY table byte (active-low scrambled binary, 0xFF = 0).

    ``value = b4 + 2 b7 + 4 b6 + 8 b1 + 16 b0 + 32 b3 + 64 b2 + 128 b5`` of the inverted byte.
    """
    i = ~byte & 0xFF
    return (
        ((i >> 4) & 1)
        | (((i >> 7) & 1) << 1)
        | (((i >> 6) & 1) << 2)
        | (((i >> 1) & 1) << 3)
        | ((i & 1) << 4)
        | (((i >> 3) & 1) << 5)
        | (((i >> 2) & 1) << 6)
        | (((i >> 5) & 1) << 7)
    )


def jd_encode(value: int) -> int:
    """The JD-BWRY byte that decodes to ``value`` (inverse of :func:`jd_value`)."""
    return next(b for b in range(256) if jd_value(b) == value)


def _family_len_ok(family: int, size: int) -> bool:
    if family == LutFamily.S6_800:
        return size in (799, 800)
    if family == LutFamily.JD_BWRY_535:
        return size == 535
    if family == LutFamily.X4_536:
        return size in (535, 536)
    return size <= LUT_MAX_BYTES


def _s6_problem(table: bytes) -> str | None:
    if not any(table[:S6_HEADER]):
        return "S6-800 header bytes 0-7 are all zero (hangs BUSY with the custom-LUT bit set)"
    run = 0
    for offset, byte in enumerate(table):
        run = run + 1 if byte == 0 else 0
        if run >= S6_ZERO_RUN:
            return f"S6-800 table has {S6_ZERO_RUN} zero bytes in a row ending at byte {offset} (hangs BUSY)"
    return None


def _jd_problem(table: bytes) -> str | None:
    for block in range(JD_BLOCKS):
        for row in range(JD_LAST_CHECKED_ROW + 1):
            base = JD_BODY + block * JD_BLOCK + row * JD_ROW
            if base + 9 >= len(table):
                return None  # only reached by a table the size check already refuses
            m = jd_value(table[base - 1])
            v = [jd_value(b) for b in table[base : base + 10]]
            timing = v[5] * (v[2] + v[3] + v[4]) + v[9] * (v[6] + v[7] + v[8])
            if m == 0 or timing == 0:
                if m != 0:
                    return f"JD block {block} row {row} runs no frames but has repeat M={m} (never releases BUSY)"
                break
    return None


def lut_lint(family: int, table: bytes) -> str | None:
    """Why the device would refuse ``table`` for ``family``, or ``None`` when it passes.

    ``table`` is the whole raw (inflated) table. Mirrors ``od_lut_lint``.
    """
    if not _family_len_ok(family, len(table)):
        return f"table is {len(table)} bytes, which lut_family {family} does not allow"
    if family == LutFamily.S6_800:
        return _s6_problem(table)
    if family in (LutFamily.JD_BWRY_535, LutFamily.X4_536):
        return _jd_problem(table)
    return None


def never_send(family: int, cmd: int) -> bool:
    """Whether a sequence of ``family`` may never send ``cmd`` (``od_lut_never_send``).

    Family 0 and families the firmware does not know have no list.
    """
    if family == LutFamily.NONE or family >= len(LutFamily):
        return False
    return cmd in NEVER_SEND


@dataclass(frozen=True)
class SequenceStep:
    """One decoded drive-sequence step: the command it sends (``None`` for none) or its escape."""

    offset: int
    cmd: int | None
    escape: int | None  # None for a plain step


def iter_sequence(sequence: bytes) -> Iterator[SequenceStep]:
    """Decode a drive sequence's steps the way ``od_epd.c`` ``step_decode`` does.

    The command of a step is what the never-send scan checks: a plain step's first byte, a
    WRITE_SCRATCH's ``cmd`` argument, and a SUBST/WINDOW step's first pair when that pair is a
    literal (kind 0). Running off the end without END is legal.

    Raises:
        ValueError: A step runs past the end, a SUBST/WINDOW count is 0 or over 32, or an
            escape is unknown (the device refuses each before any panel I/O).
    """
    i = 0
    while i < len(sequence):
        n = sequence[i]
        if n != 0:
            if i + 1 + n > len(sequence):
                raise ValueError(f"sequence step at byte {i} runs past the end")
            yield SequenceStep(i, sequence[i + 1], None)
            i += 1 + n
            continue
        if i + 2 > len(sequence):
            raise ValueError(f"sequence escape at byte {i} is truncated")
        esc = sequence[i + 1]
        if esc == ESC_END:
            yield SequenceStep(i, None, esc)
            return
        if esc in (ESC_WAVE, ESC_WINDOW):
            if i + 3 > len(sequence):
                raise ValueError(f"sequence escape at byte {i} is truncated")
            count = sequence[i + 2]
            size = 3 + 2 * count
            if count == 0 or count > _MAX_PAIRS or i + size > len(sequence):
                raise ValueError(f"sequence escape 0x{esc:02x} at byte {i} has a bad pair count {count}")
            cmd = sequence[i + 4] if sequence[i + 3] == 0 else None
            yield SequenceStep(i, cmd, esc)
            i += size
            continue
        if esc not in _ESC_ARGS:
            raise ValueError(f"sequence escape 0x{esc:02x} at byte {i} is unknown")
        size = 2 + _ESC_ARGS[esc]
        if i + size > len(sequence):
            raise ValueError(f"sequence escape 0x{esc:02x} at byte {i} runs past the end")
        yield SequenceStep(i, sequence[i + 2] if esc == ESC_WRITE_SCRATCH else None, esc)
        i += size


def sequence_uploads_lut(sequence: bytes) -> bool:
    """Whether the sequence has a LUT or LUT_SLICE escape (``seq_has_lut``).

    Raises:
        ValueError: See :func:`iter_sequence`.
    """
    return any(step.escape in (ESC_LUT, ESC_LUT_SLICE) for step in iter_sequence(sequence))


def sequence_lint(family: int, sequence: bytes) -> str | None:
    """Why the device would refuse ``sequence`` under ``family``, or ``None``.

    Mirrors the family part of ``od_epd_validate``: no never-send command in any step, and no
    TSC read (0x40) after a table upload before the next refresh (0x12). Family 0 checks
    nothing but the sequence's structure.
    """
    lut_pending = False
    try:
        for step in iter_sequence(sequence):
            if family != LutFamily.NONE and step.cmd is not None:
                if never_send(family, step.cmd):
                    return f"step at byte {step.offset} sends 0x{step.cmd:02x}, which lut_family {family} never sends"
                if lut_pending and step.cmd == CMD_TSC_READ:
                    return f"step at byte {step.offset} reads the TSC (0x40) after a table upload, before 0x12"
                if step.cmd == CMD_REFRESH:
                    lut_pending = False
            if step.escape in (ESC_LUT, ESC_LUT_SLICE):
                lut_pending = True
    except ValueError as err:
        return str(err)
    return None


_LUT_SLOT_RESERVED: Final = 0x70  # bits 4-6 of the 0xAF slot byte: a set bit matches no slot


def stored_slot_problem(luts: Sequence[DisplayLut], instance_number: int, slot_id: int, family: int) -> str | None:
    """Why the device would read stored slot ``slot_id`` as absent, or ``None`` when it is usable.

    Mirrors ``find_lut`` + ``io_lut_info``: the first 0xAF of the instance whose slot byte has
    that id and no reserved bit, with data, at most 1024 stored bytes, a good CRC, inflating
    (when compressed) to a table that passes the lint. A target without an inflater (BG22,
    Telink) reads every compressed slot as absent; the host cannot see that and does not check it.
    """
    lut = next(
        (
            x
            for x in luts
            if x.instance_number == instance_number and x.slot_id == slot_id and not x.lut_slot & _LUT_SLOT_RESERVED
        ),
        None,
    )
    if lut is None:
        return f"no stored LUT slot {slot_id}"
    if not lut.data or len(lut.data) > LUT_MAX_BYTES:
        return f"stored LUT slot {slot_id} holds {len(lut.data)} bytes"
    if not lut.crc_valid:
        return f"stored LUT slot {slot_id} fails its CRC"
    try:
        table = lut.waveform
    except ValueError as err:
        return str(err)
    problem = lut_lint(family, table)
    return None if problem is None else f"stored LUT slot {slot_id} fails the device lint: {problem}"


def mode_problem(
    mode: DisplayMode,
    *,
    lut_family: int,
    stored_luts: Sequence[DisplayLut] = (),
    inline_luts: Mapping[int, bytes] | None = None,
) -> str | None:
    """Why the device would refuse to START ``mode`` under ``lut_family``, or ``None``.

    The family checks of ``od_epd_check_pass`` / ``od_epd_begin_pass`` for every pass: the
    sequence lint (:func:`sequence_lint`), and, when the sequence uploads a table, the pass's
    slot must be an inline table (``inline_luts``, which shadow stored slots) or a usable
    stored slot. Family 0 runs neither check (configs written before the 0xB1 tail); inline
    tables are linted where the prefix is built.
    """
    if lut_family == LutFamily.NONE:
        return None
    problem = sequence_lint(lut_family, mode.sequence)
    if problem is not None:
        return f"mode '{mode.name}': {problem}"
    if not sequence_uploads_lut(mode.sequence):
        return None
    inline = inline_luts or {}
    for pass_index, slot in enumerate(mode.lut_slots):
        if slot == LUT_SLOT_NONE:
            return f"mode '{mode.name}' pass {pass_index} uploads a table but names no slot"
        if slot in inline:
            continue
        problem = stored_slot_problem(stored_luts, mode.instance_number, slot, lut_family)
        if problem is not None:
            return f"mode '{mode.name}' pass {pass_index}: {problem}"
    return None
