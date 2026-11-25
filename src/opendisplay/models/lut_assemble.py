"""Host-built per-update tables (the S6 ``hosttable`` mode, sent with INLINE_LUT).

An S6-800 table is an 8-byte header, the 88-byte LUTC (frame timing) at byte 8, and 8 level
slots of 88 bytes at ``96 + 88k``, slot ``k`` driving the pixels written with code ``k``. An
update that writes only some codes can drop the slots of the others: each is replaced with a
verbatim copy of the hold slot (the code whose pixels do not move, 4 on these panels), so the
refresh runs only as long as the remaining slots need.

Charge: every slot that stays is copied byte for byte, and every replaced one becomes the hold
slot, which the source table already carries. So whatever charge gate the source table passed
(per-slot DC balance) still holds for every slot of the result; nothing new is synthesised.
The border slot is kept even when no pixel uses its code, because the panel border is driven
through it.
"""

from __future__ import annotations

from collections.abc import Set
from typing import Final

from .drive_ext import LutFamily
from .lut_lint import lut_lint

S6_TABLE: Final = 800
S6_SLOTS_AT: Final = 96
S6_SLOT: Final = 88
S6_CODES: Final = 8
S6_HOLD_CODE: Final = 4
S6_BORDER_CODE: Final = 1


def _s6_slot(table: bytes, code: int) -> bytes:
    start = S6_SLOTS_AT + code * S6_SLOT
    return table[start : start + S6_SLOT]


def assemble_subset_s6(
    table: bytes, codes_present: Set[int], hold_code: int = S6_HOLD_CODE, border_code: int = S6_BORDER_CODE
) -> bytes:
    """``table`` with the slot of every code not in ``codes_present`` replaced by the hold slot.

    The hold and border slots always stay. The result passes the S6-800 device lint or this
    raises (copying slots can join zero runs across a slot boundary).

    Raises:
        ValueError: ``table`` is not 800 bytes, a code is not 0-7, or the result fails the lint.
    """
    if len(table) != S6_TABLE:
        raise ValueError(f"an S6 table is {S6_TABLE} bytes, got {len(table)}")
    for name, code in (("hold_code", hold_code), ("border_code", border_code)):
        if not 0 <= code < S6_CODES:
            raise ValueError(f"{name} must be 0-{S6_CODES - 1}, got {code}")
    bad = sorted(c for c in codes_present if not 0 <= c < S6_CODES)
    if bad:
        raise ValueError(f"pixel codes must be 0-{S6_CODES - 1}, got {bad}")
    keep = set(codes_present) | {hold_code, border_code}
    hold = _s6_slot(table, hold_code)
    out = bytearray(table[:S6_SLOTS_AT])
    for code in range(S6_CODES):
        out += _s6_slot(table, code) if code in keep else hold
    problem = lut_lint(LutFamily.S6_800, bytes(out))
    if problem is not None:
        raise ValueError(f"the assembled table fails the device lint: {problem}")
    return bytes(out)
