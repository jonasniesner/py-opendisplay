"""The DisplayConfig frame layout fields (protocol structs 2.5, 0x20 offsets 33-45).

The packer that applies them is ``opendisplay.encoding.layout``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Final

from ..exceptions import FrameLayoutError
from .enums import FrameScan, FrameSplit, PixelPack, PlaneInvert, RowOrder

_RESERVED_SIZE: Final = 4
_FRAME_SCAN_DEFINED: Final = int(FrameScan.COLUMN_MAJOR | FrameScan.MIRROR_X | FrameScan.MIRROR_Y | FrameScan.LSB_FIRST)
_PLANE_INVERT_DEFINED: Final = int(PlaneInvert.PLANE_0 | PlaneInvert.PLANE_1)


@dataclass(frozen=True)
class FrameLayout:  # pylint: disable=too-many-instance-attributes
    """The frame layout fields of DisplayConfig (0x20 offsets 33-45). All zero = the scheme's own packing."""

    frame_split: int = 0
    frame_scan: int = 0
    part_width: int = 0
    part_stride: int = 0
    plane_invert: int = 0
    pixel_pack: int = 0
    row_order: int = 0
    reserved: bytes = bytes(_RESERVED_SIZE)

    #: The keys of a drive preset's ``frame_layout`` object.
    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "frame_split",
        "frame_scan",
        "part_width",
        "part_stride",
        "plane_invert",
        "pixel_pack",
        "row_order",
    )

    @classmethod
    def from_display(cls, display: Any) -> FrameLayout:
        """The layout of a ``DisplayConfig``."""
        return cls(
            frame_split=display.frame_split,
            frame_scan=display.frame_scan,
            part_width=display.part_width,
            part_stride=display.part_stride,
            plane_invert=display.plane_invert,
            pixel_pack=display.pixel_pack,
            row_order=display.row_order,
            reserved=bytes(display.reserved),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> FrameLayout:
        """A drive preset's ``frame_layout`` object: keys are the field names, a missing key is 0.

        Raises:
            ValueError: An unknown key, or a value that is not a non-negative integer.
        """
        unknown = sorted(set(data) - set(cls.FIELD_NAMES))
        if unknown:
            raise ValueError(f"frame_layout has unknown keys {unknown}")
        values: dict[str, int] = {}
        for name in cls.FIELD_NAMES:
            value = data.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"frame_layout.{name} must be a non-negative integer, not {value!r}")
            values[name] = value
        return cls(
            frame_split=values["frame_split"],
            frame_scan=values["frame_scan"],
            part_width=values["part_width"],
            part_stride=values["part_stride"],
            plane_invert=values["plane_invert"],
            pixel_pack=values["pixel_pack"],
            row_order=values["row_order"],
        )

    def to_mapping(self) -> dict[str, int]:
        """The named fields as a ``frame_layout`` object (``reserved`` left out)."""
        return {name: getattr(self, name) for name in self.FIELD_NAMES}

    @property
    def is_default(self) -> bool:
        """Every field and reserved byte is zero: the scheme's own packing."""
        return not any(getattr(self, f.name) for f in fields(self) if f.name != "reserved") and not any(self.reserved)

    @property
    def reorders(self) -> bool:
        """The layout moves pixels (split, scan, row order or packing), so a frame is not row-major.

        ``plane_invert`` alone does not: it changes bits in place.
        """
        return bool(self.frame_split or self.frame_scan or self.row_order or self.pixel_pack)

    def check_defined(self) -> None:
        """Refuse undefined values, reserved bits and non-zero reserved bytes.

        Raises:
            FrameLayoutError: Naming the field.
        """
        for name, enum in (("frame_split", FrameSplit), ("row_order", RowOrder), ("pixel_pack", PixelPack)):
            value = getattr(self, name)
            if value not in {int(m) for m in enum}:
                raise FrameLayoutError(f"{name} {value} is not defined (protocol structs 2.5)")
        if self.frame_scan & ~_FRAME_SCAN_DEFINED:
            raise FrameLayoutError(f"frame_scan 0x{self.frame_scan:02x} sets reserved bits 4-7")
        if self.plane_invert & ~_PLANE_INVERT_DEFINED:
            raise FrameLayoutError(f"plane_invert 0x{self.plane_invert:02x} sets reserved bits 2-7")
        if len(self.reserved) != _RESERVED_SIZE or any(self.reserved):
            raise FrameLayoutError(f"DisplayConfig reserved bytes must be 4 zero bytes, not {self.reserved.hex()}")
        for name in ("part_width", "part_stride"):
            value = getattr(self, name)
            if not 0 <= value <= 0xFFFF:
                raise FrameLayoutError(f"{name} {value} does not fit a uint16")


IDENTITY: Final = FrameLayout()
