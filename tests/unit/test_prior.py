"""Host-side prior masking (panels/prior.py) and the 0xAE hold_mask field."""

import numpy as np
import pytest

from opendisplay.models.drive_ext import ModeExt
from opendisplay.panels.prior import PriorConflictError, _pack, _unpack, bits_per_pixel, mask_unchanged


def frame(codes: list[int]) -> bytes:
    return _pack(np.array(codes, dtype=np.uint8), 2)


def test_hold_mask_round_trip() -> None:
    ext = ModeExt(mode_flags=1, budget_cost=1, scrub_before=0xFF, hold_mask=0b1100)
    assert ext.to_bytes() == bytes([1, 1, 1, 0xFF, 0x0C])
    back = ModeExt.from_tail(ext.to_bytes())
    assert back == ext
    assert back is not None and back.hold_codes == frozenset({2, 3})


def test_old_tail_has_no_hold() -> None:
    ext = ModeExt.from_tail(bytes([1, 1, 1, 0xFF]))
    assert ext is not None and ext.hold_mask is None and ext.hold_codes == frozenset()
    assert ModeExt(hold_mask=0xFF).hold_codes == frozenset()


def test_unchanged_pixels_get_lowest_hold_code() -> None:
    prior = frame([0, 1, 2, 3, 0, 1, 0, 1])
    new = frame([0, 0, 2, 3, 1, 1, 0, 1])
    out = mask_unchanged(new, prior, frozenset({2, 3}), 8)
    assert out.changed == 2
    assert list(_unpack(out.frame, 2)) == [2, 0, 2, 2, 1, 2, 2, 2]


def test_changed_pixel_needing_a_hold_code_conflicts() -> None:
    prior = frame([0, 1, 0, 1])
    new = frame([0, 1, 3, 1])
    with pytest.raises(PriorConflictError):
        mask_unchanged(new, prior, frozenset({2, 3}), 4)


def test_size_checks() -> None:
    assert bits_per_pixel(42 * 384, 168 * 384) == 2
    with pytest.raises(ValueError):
        mask_unchanged(frame([0] * 8), frame([0] * 4), frozenset({2}), 8)
