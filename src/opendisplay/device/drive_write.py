"""An upload in a named drive mode: CMD_DRIVE_WRITE_START (0x0085), data over 0x71, END 0x72.

One transfer carries every pass of a multi-pass mode back to back (one zlib stream when
compressed); the device advances to the next pass at each frame boundary, and each boundary is
a panel refresh that can land on any data chunk. ``PIPELINED`` is never set. Timeouts follow the
design note (``2026-11-16-v3-3b-drive-modes-design.md``, the website's rule).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ..exceptions import DriveModeSubstitutedError, DriveModeTemperatureError, InvalidResponseError
from ..models.enums import RefreshMode
from ..models.lut_lint import mode_problem
from ..models.panel_drive import DisplayLut, DisplayMode
from ..protocol import CommandCode, validate_ack_response
from ..protocol.commands import (
    build_direct_write_end_command,
    build_direct_write_end_with_etag,
    build_drive_write_start,
    build_inline_lut_prefix,
)
from ..protocol.responses import parse_drive_write_start
from .channel import SecureChannel
from .upload import Uploader

_LOGGER = logging.getLogger(__name__)

_START_HEAD: Final = 11  # [00][85][mode][refresh][flags][rsv:2][uncompressed_size:4]
_ETAG_LEN: Final = 4  # [old_etag:4 BE] after rsv when the START carries flag ETAG
_PASS_REFRESH_S: Final = 45.0  # budget for one pass boundary (a panel refresh, observed up to 35 s)
_MIN_ACK_S: Final = 90.0


@dataclass(frozen=True)
class DriveWriteResult:
    """What a drive write left on the device."""

    #: The panel temperature the device reported, ``None`` when unknown.
    temp_c: int | None
    #: The etag the device now shows (END carried it and the device took it); ``None`` = none.
    etag: int | None


def start_timeout(passes: int, *, inline_data: bool, first_chunk: float) -> float:
    """How long to wait for the START answer.

    A START that carries data can see every pass boundary but the last before it is answered.
    Measured on 2026-11-16: 0.9 s without inline data (OD58D951, BG22, colour16) and 1.1 s with
    it (E1002, ESP32, single pass, so no pass boundary before the answer).
    """
    if not inline_data:
        return first_chunk
    return 10.0 + max(_MIN_ACK_S, (passes - 1) * _PASS_REFRESH_S)


def end_timeout(passes: int) -> float:
    """How long to wait for the END acknowledgement: one refresh per pass."""
    return max(_MIN_ACK_S, passes * _PASS_REFRESH_S)


def drive_prefix(
    mode: DisplayMode,
    *,
    inline_luts: Mapping[int, bytes] | None = None,
    lut_family: int = 0,
    stored_luts: Sequence[DisplayLut] = (),
) -> bytes:
    """Check ``mode`` the way the device will at START and build its INLINE_LUT prefix (or ``b""``).

    The checks (:func:`~opendisplay.models.lut_lint.mode_problem`, and the inline tables' own
    lint in :func:`~opendisplay.protocol.commands.build_inline_lut_prefix`) refuse what a device
    with this ``lut_family`` would refuse, before any I/O.

    Raises:
        ValueError: The device would refuse the mode or a table.
    """
    prefix = build_inline_lut_prefix(inline_luts, lut_family=lut_family) if inline_luts else b""
    problem = mode_problem(mode, lut_family=lut_family, stored_luts=stored_luts, inline_luts=inline_luts)
    if problem is not None:
        raise ValueError(f"The device would refuse this drive: {problem}")
    return prefix


async def drive_write(  # pylint: disable=too-many-arguments,too-many-locals
    uploader: Uploader,
    channel: SecureChannel,
    mode: DisplayMode,
    frames: bytes,
    *,
    stream: bytes | None = None,
    refresh: RefreshMode = RefreshMode.FULL,
    progress_callback: Callable[[int, int], None] | None = None,
    inline_luts: Mapping[int, bytes] | None = None,
    lut_family: int = 0,
    stored_luts: Sequence[DisplayLut] = (),
    old_etag: int | None = None,
    new_etag: int | None = None,
) -> DriveWriteResult:
    """Upload ``frames`` (every pass, concatenated) in ``mode``; no CPU-heavy work here.

    Args:
        stream: The data as one zlib stream, to send compressed; ``None`` sends it as it is.
            With ``inline_luts`` it must be the stream of ``drive_prefix(...) + frames`` (the
            prefix is compressed together with the frames).
        inline_luts: Tables (raw, slot id -> bytes) that shadow the stored 0xAF slots for this
            transfer only (START flag INLINE_LUT).
        lut_family: The 0xB1 tail's ``lut_family`` (0 = undeclared: the device lints nothing
            but inline table sizes).
        stored_luts: The config's 0xAF slots, to check that every slot the mode uploads exists.
        old_etag: START flag ETAG (protocol 2.5): the etag of the frame the panel must still
            show, or the device refuses before drawing anything. ``None`` sends no flag.
        new_etag: Sent with END; the device shows it as its displayed etag afterwards.

    Returns:
        The panel temperature, and the etag the device now shows.

    Raises:
        ValueError: The device would refuse the mode or an inline table; nothing was sent.
        DriveModeSubstitutedError: The device runs another mode; nothing was sent after START.
        DriveModeTemperatureError, DriveSequenceUnsupportedError, PanelBusyError,
        NeedsMigrationError: The device refused the START.
        DriveEtagMismatchError: The panel no longer shows ``old_etag``; nothing was drawn.
        InvalidResponseError: Another refusal, or the device expects another pass count.
    """
    prefix = drive_prefix(mode, inline_luts=inline_luts, lut_family=lut_family, stored_luts=stored_luts)
    compressed = stream is not None
    if stream is not None:
        room = channel.start_payload_limit() - _START_HEAD - (_ETAG_LEN if old_etag is not None else 0)
        start = build_drive_write_start(
            mode.mode_index,
            refresh.value,
            uncompressed_size=len(prefix) + len(frames),
            first_chunk=stream[:room],
            inline_len=len(prefix),
            old_etag=old_etag,
        )
        remaining = stream[room:]
    else:
        start = build_drive_write_start(mode.mode_index, refresh.value, inline_len=len(prefix), old_etag=old_etag)
        remaining = prefix + frames

    await channel.write(start)
    answer = await channel.read(
        start_timeout(mode.passes, inline_data=compressed, first_chunk=Uploader.TIMEOUT_FIRST_CHUNK)
    )
    try:
        actual, passes, temp_c = parse_drive_write_start(answer)
    except DriveModeTemperatureError as err:
        raise DriveModeTemperatureError(mode.name, mode.temp_min_c, mode.temp_max_c) from err
    if actual != mode.mode_index:
        raise DriveModeSubstitutedError(mode.name, actual, temp_c)
    if passes != mode.passes:
        raise InvalidResponseError(
            f"The device expects {passes} passes for '{mode.name}', the config declares {mode.passes}; nothing sent"
        )
    _LOGGER.info("Drive write '%s': %d passes, panel at %s °C", mode.name, passes, temp_c)

    auto_completed = await uploader.send_data_chunks(remaining, progress_callback) if remaining else False
    if not auto_completed:
        end = (
            build_direct_write_end_with_etag(refresh.value, new_etag)
            if new_etag is not None
            else build_direct_write_end_command(refresh.value)
        )
        await channel.write(end)
        validate_ack_response(await channel.read(end_timeout(passes)), CommandCode.DIRECT_WRITE_END)
    await uploader.await_refresh_complete()
    # A firmware auto-END never sets the displayed etag.
    return DriveWriteResult(temp_c, new_etag if not auto_completed else None)
