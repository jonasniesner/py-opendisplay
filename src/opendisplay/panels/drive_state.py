"""What the host remembers about a panel between drives: budget used, band, last mode.

The device enforces neither the update budget nor the temperature band (decision 2026-11-25:
the host tracks them, the device stays stateless). So the host keeps, per device, the state the
mode planner (:mod:`opendisplay.panels.drive_plan`) needs:

- ``fingerprint``: a hash of the drive packets (0xAE, 0xAF, 0xB1, 0xB3) the state was built
  against. A new config (migration, preset update, hand edit) changes it, and the state is then
  treated as lost.
- ``budget_used``: budget units spent since the last stock/scrub drive.
- ``last_mode``, ``last_band``, ``last_temp_c``, ``updated`` (Unix seconds).

Lost, unreadable or mismatched state means "needs a stock full": the planner schedules mode 0
before anything that is not itself a stock drive. Nothing here is a secret, so an unreadable
file is replaced rather than refused (unlike the key store).

Two stores, the key store's pattern: :class:`MemoryDriveStateStore` and
:class:`JsonFileDriveStateStore` (``~/.config/opendisplay/drive_state.json``). Keyed by the
device's advertised name, as the key store is.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from ..models.config import GlobalConfig
from ..models.panel_drive import (
    PACKET_TYPE_DISPLAY_CONTROLLER,
    PACKET_TYPE_DISPLAY_LUT,
    PACKET_TYPE_DISPLAY_MODE,
    PACKET_TYPE_DISPLAY_PANEL,
)

FORMAT: Final = 1
DEFAULT_PATH: Final = Path("~/.config/opendisplay/drive_state.json")

#: The packets a drive depends on. 0xB0 (measured palettes, host-only) is left out: a new
#: calibration does not change what the panel is driven with.
FINGERPRINT_TYPES: Final = (
    PACKET_TYPE_DISPLAY_MODE,
    PACKET_TYPE_DISPLAY_LUT,
    PACKET_TYPE_DISPLAY_PANEL,
    PACKET_TYPE_DISPLAY_CONTROLLER,
)


def drive_fingerprint(config: GlobalConfig) -> str:
    """A hash of the config's drive packets, in stored order (SHA-256, first 16 bytes, hex)."""
    digest = hashlib.sha256()
    for packet in config.length_prefixed_packets:
        if packet.packet_type in FINGERPRINT_TYPES:
            digest.update(bytes([packet.packet_type]) + len(packet.body).to_bytes(2, "little") + packet.body)
    return digest.hexdigest()[:32]


@dataclass(frozen=True)
class DriveState:
    """The host's record of one panel's drive history."""

    fingerprint: str
    budget_used: int = 0
    last_mode: int | None = None
    last_band: int | None = None  # band digit of the last banded drive since the last stock drive
    last_temp_c: int | None = None
    updated: float = 0.0

    def matches(self, config: GlobalConfig) -> bool:
        """Whether this state was recorded against ``config``'s drive packets."""
        return self.fingerprint == drive_fingerprint(config)

    def to_json(self) -> dict[str, Any]:
        """The record as stored."""
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, record: Any) -> DriveState | None:
        """Parse a stored record; ``None`` for one that does not parse (treated as lost)."""
        if not isinstance(record, dict) or not isinstance(record.get("fingerprint"), str):
            return None
        try:
            return cls(
                fingerprint=record["fingerprint"],
                budget_used=int(record.get("budget_used", 0)),
                last_mode=_opt_int(record.get("last_mode")),
                last_band=_opt_int(record.get("last_band")),
                last_temp_c=_opt_int(record.get("last_temp_c")),
                updated=float(record.get("updated", 0.0)),
            )
        except (TypeError, ValueError):
            return None


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


class DriveStateStore(Protocol):
    """The storage a caller provides for drive state."""

    def get(self, name: str) -> DriveState | None:
        """The state of ``name``, or ``None`` when there is none (or it is unreadable)."""

    def put(self, name: str, state: DriveState) -> None:
        """Replace the state of ``name``."""

    def forget(self, name: str) -> None:
        """Drop the state of ``name`` (its next drive starts with a stock full)."""


class _DocumentDriveStateStore:
    """The store over one JSON-shaped document; subclasses load and save it."""

    def _load(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def _save(self, document: dict[str, Any]) -> None:
        raise NotImplementedError

    def _document(self) -> dict[str, Any]:
        document = self._load()
        if not isinstance(document, dict) or document.get("format") != FORMAT:
            return {"format": FORMAT, "devices": {}}
        if not isinstance(document.get("devices"), dict):
            document["devices"] = {}
        return document

    def get(self, name: str) -> DriveState | None:
        """See :class:`DriveStateStore`."""
        return DriveState.from_json(self._document()["devices"].get(name))

    def put(self, name: str, state: DriveState) -> None:
        """See :class:`DriveStateStore`."""
        if not name:
            raise ValueError("device name required")
        document = self._document()
        document["devices"][name] = state.to_json()
        self._save(document)

    def forget(self, name: str) -> None:
        """See :class:`DriveStateStore`."""
        document = self._document()
        if document["devices"].pop(name, None) is not None:
            self._save(document)


class MemoryDriveStateStore(_DocumentDriveStateStore):
    """Drive state in memory only: for tests and callers that persist it themselves."""

    def __init__(self) -> None:
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._data)

    def _save(self, document: dict[str, Any]) -> None:
        self._data = copy.deepcopy(document)


class JsonFileDriveStateStore(_DocumentDriveStateStore):
    """Drive state in a JSON file, replaced atomically. An unreadable file reads as empty."""

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path).expanduser()

    def _load(self) -> dict[str, Any] | None:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return document if isinstance(document, dict) else None

    def _save(self, document: dict[str, Any]) -> None:
        text = json.dumps(document, indent=2, sort_keys=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, self.path)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
