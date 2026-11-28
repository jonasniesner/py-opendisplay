"""Which presets on a device have a newer revision in the bundle, and how to apply a drive update.

The rule of presets plan 2.9, shared by HA, the CLI and the website (its JS port):

- Each of driver board / display / power with ``simple_config_*_index`` set, and the drive data
  via the 0xB1 drive tail's ``drive_preset_key``, is looked up in the bundle.
- A bundled ``rev`` higher than the device's (0 = unknown = older than rev 1) whose
  ``min_firmware`` the device meets means an update is available.
- 0xB1 ``drive_flags`` bit 0 (hand-edited) means the drive data is never offered an update.
- Applying a drive update rewrites only the drive packets (0xAE/0xAF/0xB1 and type-level 0xB0),
  plus the 0x20 frame layout when the entry has one; a 0xB0 measured on this unit (``cal_flags``
  bit 0) is never replaced.
- Experimental modes are written only on opt-in (``include_experimental``).

The drive update is a :class:`~opendisplay.panels.migration.MigrationPlan`, applied with
:func:`~opendisplay.panels.migration.apply_migration`. Simple-config updates are reported here;
applying them needs the simple-config bundle loader, which does not exist yet.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, Protocol

from ..models.config import GlobalConfig
from ..models.profile import DeviceProfile, Generation
from .database import PanelDatabase, target_class_for
from .migration import (
    MigrationPlan,
    _drive_packets,
    apply_migration,
    check_fits,
    display_layout_fix,
    drive_ext_of,
    measured_calibrations,
    preset_packets,
)
from .migration import _apply as _apply_plan


class PresetKind(Enum):
    """Which subconfig a preset revision belongs to."""

    DRIVER = "driver"
    DISPLAY = "display"
    POWER = "power"
    DRIVE = "drive"


@dataclass(frozen=True)
class SimplePresetInfo:
    """What the bundle says about one simple-config preset (board, display or power option)."""

    rev: int
    min_firmware: tuple[int, int, int] | None = None
    changelog: str | None = None


class SimpleConfigCatalog(Protocol):  # pylint: disable=too-few-public-methods
    """Looks up a simple-config preset by kind and its 1-based ``simple_config_*_index``."""

    def simple_preset(self, kind: PresetKind, index: int) -> SimplePresetInfo | None:
        """The bundle's entry, or ``None`` when the bundle has none for this index."""


@dataclass(frozen=True)
class MappingCatalog:
    """A :class:`SimpleConfigCatalog` over a plain mapping ``{kind: {index: info}}`` (tests, small hosts)."""

    entries: Mapping[PresetKind, Mapping[int, SimplePresetInfo]]

    def simple_preset(self, kind: PresetKind, index: int) -> SimplePresetInfo | None:
        """See :class:`SimpleConfigCatalog`."""
        return self.entries.get(kind, {}).get(index)


@dataclass(frozen=True)
class PresetUpdate:
    """One subconfig whose bundled preset is newer than what the device carries.

    Attributes:
        kind: Which subconfig.
        key: The ``simple_config_*_index`` (simple configs) or ``drive_preset_key`` (drive).
        device_rev: The revision the device records; 0 = unknown.
        bundle_rev: The bundle's revision.
        changelog: The bundle entry's changelog line, when it has one.
        applicable: This library can apply it (today: drive updates only).
    """

    kind: PresetKind
    key: int
    device_rev: int
    bundle_rev: int
    changelog: str | None
    applicable: bool


@dataclass(frozen=True)
class PresetUpdatePlan:
    """Every update available for a config, and the drive update ready to apply.

    Attributes:
        updates: The available updates, driver / display / power / drive in that order.
        drive: The drive update as a migration plan, when one is available and not blocked;
            apply it with :func:`apply_preset_update`. ``None`` otherwise.
        notes: Why a subconfig was not offered (hand-edited, firmware too old, unknown preset,
            does not fit), one sentence each.
    """

    updates: tuple[PresetUpdate, ...] = ()
    drive: MigrationPlan | None = None
    notes: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """Whether any update is available."""
        return bool(self.updates)


_SIMPLE_FIELDS: Final = (
    (PresetKind.DRIVER, "simple_config_driver_index", "simple_config_driver_rev"),
    (PresetKind.DISPLAY, "simple_config_display_index", "simple_config_display_rev"),
    (PresetKind.POWER, "simple_config_power_index", "simple_config_power_rev"),
)


def _firmware_tuple(profile: DeviceProfile) -> tuple[int, int, int]:
    fw = profile.firmware
    return int(fw["major"]), int(fw["minor"]), int(fw.get("patch", 0))


def _meets(profile: DeviceProfile, need: tuple[int, int, int] | None) -> bool:
    return need is None or _firmware_tuple(profile) >= need


def _fmt(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def plan_preset_update(
    config: GlobalConfig,
    profile: DeviceProfile,
    db: PanelDatabase,
    *,
    catalog: SimpleConfigCatalog | None = None,
    include_experimental: bool = False,
) -> PresetUpdatePlan:
    """Which presets on ``config`` have a newer revision, and the drive update to apply.

    Args:
        config: The device's config.
        profile: The device's profile (firmware version, target class, limits).
        db: The panel drive database (the bundle's drive presets).
        catalog: The bundle's simple-config presets; without it only the drive data is checked.
        include_experimental: Also write the entry's experimental modes (opt-in).
    """
    updates: list[PresetUpdate] = []
    notes: list[str] = []
    if catalog is not None:
        for kind, index_field, rev_field in _SIMPLE_FIELDS:
            index = int(getattr(config.manufacturer, index_field))
            if not index:
                continue
            info = catalog.simple_preset(kind, index)
            device_rev = int(getattr(config.manufacturer, rev_field))
            if info is None:
                notes.append(f"{kind.value} preset {index} is not in the bundle")
            elif info.rev > device_rev and not _meets(profile, info.min_firmware):
                assert info.min_firmware is not None
                notes.append(f"{kind.value} preset {index} rev {info.rev} needs firmware {_fmt(info.min_firmware)}")
            elif info.rev > device_rev:
                updates.append(PresetUpdate(kind, index, device_rev, info.rev, info.changelog, applicable=False))
    drive_update, drive_plan, drive_notes = _plan_drive(config, profile, db, include_experimental)
    notes += drive_notes
    if drive_update is not None:
        updates.append(drive_update)
    return PresetUpdatePlan(updates=tuple(updates), drive=drive_plan, notes=tuple(notes))


def _plan_drive(  # pylint: disable=too-many-return-statements
    config: GlobalConfig, profile: DeviceProfile, db: PanelDatabase, include_experimental: bool
) -> tuple[PresetUpdate | None, MigrationPlan | None, list[str]]:
    if profile.generation is not Generation.V3 or not config.displays:
        return None, None, []
    present = _drive_packets(config)
    if not present:
        return None, None, ["the config has no drive data; that is a migration, not a preset update"]
    display = config.displays[0]
    ext = drive_ext_of(config, display.instance_number)
    if ext is not None and ext.hand_edited:
        return None, None, ["the drive data was edited by hand; no preset update is offered"]
    # Without a drive tail the data came from the preset the display names (rev 0 = unknown).
    key = ext.drive_preset_key if ext is not None and ext.drive_preset_key else display.panel_ic_type
    device_rev = ext.drive_rev if ext is not None else 0
    target_class = target_class_for(profile.target)
    entry = db.entry_for(key, target_class)
    if entry is None or not entry.usable:
        return None, None, [f"drive preset {key:#06x} is not available for this target"]
    if entry.rev <= device_rev:
        return None, None, []
    if not _meets(profile, entry.min_firmware):
        assert entry.min_firmware is not None
        return None, None, [f"drive preset {key:#06x} rev {entry.rev} needs firmware {_fmt(entry.min_firmware)}"]
    update = PresetUpdate(PresetKind.DRIVE, key, device_rev, entry.rev, entry.changelog, applicable=True)
    keep = measured_calibrations(present)
    try:
        add, skipped = preset_packets(entry, include_experimental=include_experimental, keep=keep)
    except ValueError as err:
        return None, None, [f"drive preset {key:#06x}: {err}"]
    scheme, layout = display_layout_fix(display, entry)
    plan = MigrationPlan(
        needed=True,
        blocked=False,
        reason=f"drive preset {key:#06x} rev {device_rev} -> {entry.rev}",
        panel_ic_type=key,
        target_class=target_class,
        entry=entry,
        add=add,
        remove=tuple(p for p in present if p not in keep),
        kept=keep,
        skipped_experimental=skipped,
        needs_confirmation=entry.verified != "glass",
        color_scheme=scheme,
        frame_layout=layout,
    )
    problems = check_fits(_apply_plan(config, plan), profile, caps=db.caps)
    if problems:
        blocked = dataclasses.replace(
            plan, blocked=True, reason=f"drive preset {key:#06x} does not fit this device: " + "; ".join(problems)
        )
        return dataclasses.replace(update, applicable=False), blocked, [blocked.reason]
    return update, plan, []


def apply_preset_update(config: GlobalConfig, plan: PresetUpdatePlan) -> GlobalConfig:
    """Apply the drive update of ``plan`` (the only kind this library applies today).

    Raises:
        ValueError: There is no drive update, or it is blocked.
    """
    if plan.drive is None:
        raise ValueError("the plan has no drive update to apply")
    return apply_migration(config, plan.drive)
