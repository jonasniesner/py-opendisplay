"""plan_preset_update (presets plan 2.9): newer bundled revisions, measured palettes, experimental modes."""

from __future__ import annotations

import copy
import dataclasses
import gzip
import json
from importlib.resources import files
from typing import Any

import pytest
from test_profile import FW3, IDENT_NEEDS_MIGRATION, _config, _display

from opendisplay.models.config import GlobalConfig
from opendisplay.models.drive_ext import DRIVE_FLAG_HAND_EDITED, DriveExt, ModeExt, with_drive_tail, with_mode_tail
from opendisplay.models.firmware import FirmwareVersion
from opendisplay.models.panel_drive import (
    PACKET_TYPE_DISPLAY_CALIBRATION,
    PACKET_TYPE_DISPLAY_LUT,
    PACKET_TYPE_DISPLAY_MODE,
    PACKET_TYPE_DISPLAY_PANEL,
    LengthPrefixedPacket,
)
from opendisplay.models.profile import build_profile
from opendisplay.panels import (
    MappingCatalog,
    PanelDatabase,
    PresetKind,
    SimplePresetInfo,
    TargetClass,
    apply_migration,
    apply_preset_update,
    plan_migration,
    plan_preset_update,
)
from opendisplay.panels.database import parse_min_firmware
from opendisplay.panels.migration import measured_calibrations, preset_packets, stamp_drive_preset, strip_modes
from opendisplay.protocol.config_parser import parse_config_response
from opendisplay.protocol.config_serializer import serialize_config
from opendisplay.protocol.responses import parse_read_ident

PANEL = 0x001E  # EP29YR: modes 0 "full" and 1 "colour16", two LUT slots, one 0xB0 (mode 1)
IDENT = parse_read_ident(bytes.fromhex("004604010029a00f800003"))  # ESP32, max_config 4000
FW31: FirmwareVersion = {"major": 3, "minor": 1, "patch": 0, "sha": "1" * 40}


def _raw_bundle() -> dict[str, Any]:
    package = files("opendisplay.panels")
    return json.loads(gzip.decompress(package.joinpath("drive_presets.json.gz").read_bytes()))  # type: ignore[no-any-return]


RAW = _raw_bundle()


def _bundle(**fields: Any) -> PanelDatabase:
    """The vendored bundle, with ``fields`` set on PANEL's entry; ``experimental`` flags modes."""
    doc = copy.deepcopy(RAW)
    entry = doc["panels"][f"0x{PANEL:04X}"]
    experimental = fields.pop("experimental", ())
    entry.update(fields)
    for modes in (entry["modes"], entry["targets"]["default"]["modes"]):
        for mode in modes:
            if mode["index"] in experimental:
                mode["experimental"] = True
    return PanelDatabase(doc)


DB_V1 = _bundle(rev=1)
DB_V2 = _bundle(rev=2, changelog="faster colour16", min_firmware="3.0")


def _migrated(db: PanelDatabase = DB_V1) -> GlobalConfig:
    """A 2.x ESP32 config for PANEL, migrated with ``db`` (its 0xB1 names the preset and rev)."""
    config = _config(displays=[_display(panel_ic_type=PANEL)])
    plan = plan_migration(config, build_profile(FW3, config, IDENT_NEEDS_MIGRATION), db, include_experimental=True)
    assert plan.needed and not plan.blocked, plan.reason
    return apply_migration(config, plan)


def _profile(config: GlobalConfig, fw: FirmwareVersion = FW3) -> Any:
    return build_profile(fw, config, IDENT)


def _replace_packets(config: GlobalConfig, packet_type: int, edit: Any) -> GlobalConfig:
    packets = [
        dataclasses.replace(p, body=edit(p.body)) if p.packet_type == packet_type else p
        for p in config.length_prefixed_packets
    ]
    return dataclasses.replace(config, length_prefixed_packets=packets)


def _measured(config: GlobalConfig) -> GlobalConfig:
    """``config`` with its 0xB0 flagged measured on this unit and its colours changed."""

    def edit(body: bytes) -> bytes:
        patched = bytearray(body)
        patched[2] |= 0x01
        patched[6] ^= 0x55  # a different L* than the preset's, so it is recognisably ours
        return bytes(patched)

    return _replace_packets(config, PACKET_TYPE_DISPLAY_CALIBRATION, edit)


# ── the bundle keeps the new per-entry fields ────────────────────────────────


def test_bundle_fields_are_loaded() -> None:
    entry = _bundle(
        rev=4, min_firmware="3.2.1", changelog="x", experimental=[1], experimental_note="one unit"
    ).entry_for(PANEL, TargetClass.DEFAULT)

    assert entry is not None
    assert (entry.rev, entry.min_firmware, entry.changelog) == (4, (3, 2, 1), "x")
    assert entry.experimental_modes == frozenset({1})
    assert entry.experimental_note == "one unit"


def test_vendored_bundle_has_no_revs_yet() -> None:
    entry = PanelDatabase.load().entry_for(PANEL, TargetClass.DEFAULT)

    assert entry is not None
    assert (entry.rev, entry.min_firmware, entry.experimental_modes) == (0, None, frozenset())


@pytest.mark.parametrize(
    ("text", "parsed"), [("3", (3, 0, 0)), ("3.1", (3, 1, 0)), ("v3.1.2", (3, 1, 2)), (None, None), ("next", None)]
)
def test_parse_min_firmware(text: str | None, parsed: tuple[int, int, int] | None) -> None:
    assert parse_min_firmware(text) == parsed


# ── writers record the preset they applied ───────────────────────────────────


def test_migration_stamps_the_drive_tail_when_the_entry_has_a_rev() -> None:
    panel = _migrated().display_panels[0]

    assert panel.ext == DriveExt(drive_preset_key=PANEL, drive_rev=1)


def test_migration_without_a_rev_writes_the_entry_unchanged() -> None:
    db = PanelDatabase.load()
    config = _config(displays=[_display(panel_ic_type=PANEL)])
    plan = plan_migration(config, build_profile(FW3, config, IDENT_NEEDS_MIGRATION), db)

    assert plan.add == db.entry_for(PANEL, TargetClass.DEFAULT).packets  # type: ignore[union-attr]


def test_stamp_keeps_the_presets_own_tail_fields_and_clears_hand_edited() -> None:
    body = bytes(13)
    tailed = with_drive_tail(body, DriveExt(drive_flags=DRIVE_FLAG_HAND_EDITED, budget_limit=20))
    (stamped,) = stamp_drive_preset([LengthPrefixedPacket(0, PACKET_TYPE_DISPLAY_PANEL, tailed)], 7, 9)

    assert stamped.body == with_drive_tail(body, DriveExt(drive_preset_key=7, drive_rev=9, budget_limit=20))


# ── when an update is offered ────────────────────────────────────────────────


def test_newer_rev_offers_a_drive_update() -> None:
    config = _migrated()
    plan = plan_preset_update(config, _profile(config), DB_V2)

    assert plan.available
    (update,) = plan.updates
    assert (update.kind, update.key, update.device_rev, update.bundle_rev) == (PresetKind.DRIVE, PANEL, 1, 2)
    assert update.changelog == "faster colour16"
    assert update.applicable
    assert plan.drive is not None and not plan.drive.blocked


def test_same_rev_offers_nothing() -> None:
    config = _migrated()

    assert not plan_preset_update(config, _profile(config), DB_V1).available


def test_no_drive_tail_counts_as_rev_0_of_the_displays_panel() -> None:
    config = _replace_packets(_migrated(), PACKET_TYPE_DISPLAY_PANEL, lambda b: with_drive_tail(b, None))
    (update,) = plan_preset_update(config, _profile(config), DB_V1).updates

    assert (update.key, update.device_rev, update.bundle_rev) == (PANEL, 0, 1)


def test_hand_edited_drive_data_is_never_offered_an_update() -> None:
    config = _replace_packets(
        _migrated(),
        PACKET_TYPE_DISPLAY_PANEL,
        lambda b: with_drive_tail(b, DriveExt(drive_preset_key=PANEL, drive_rev=1, drive_flags=DRIVE_FLAG_HAND_EDITED)),
    )
    plan = plan_preset_update(config, _profile(config), DB_V2)

    assert not plan.available
    assert plan.drive is None
    assert "by hand" in plan.notes[0]


def test_min_firmware_not_met_offers_nothing() -> None:
    config = _migrated()
    plan = plan_preset_update(config, _profile(config), _bundle(rev=2, min_firmware="3.1"))

    assert not plan.available
    assert "needs firmware 3.1.0" in plan.notes[0]
    assert plan_preset_update(config, _profile(config, FW31), _bundle(rev=2, min_firmware="3.1")).available


def test_a_config_without_drive_data_is_a_migration_not_an_update() -> None:
    config = _config(displays=[_display(panel_ic_type=PANEL)])
    plan = plan_preset_update(config, _profile(config), DB_V2)

    assert not plan.available
    assert "migration" in plan.notes[0]


def test_drive_tail_key_wins_over_the_display_panel_id() -> None:
    config = _replace_packets(
        _migrated(), PACKET_TYPE_DISPLAY_PANEL, lambda b: with_drive_tail(b, DriveExt(drive_preset_key=0x7FFF))
    )
    plan = plan_preset_update(config, _profile(config), DB_V2)

    assert not plan.available
    assert "0x7fff" in plan.notes[0]


# ── applying it ──────────────────────────────────────────────────────────────


def test_apply_rewrites_only_the_drive_packets_and_records_the_new_rev() -> None:
    config = _migrated()
    updated = apply_preset_update(config, plan_preset_update(config, _profile(config), DB_V2))

    assert updated.display_panels[0].ext == DriveExt(drive_preset_key=PANEL, drive_rev=2)
    legacy = dataclasses.replace(config, length_prefixed_packets=[])
    assert serialize_config(dataclasses.replace(updated, length_prefixed_packets=[])) == serialize_config(legacy)
    assert not plan_preset_update(updated, _profile(updated), DB_V2).available


def test_apply_without_a_drive_update_refuses() -> None:
    config = _migrated()
    with pytest.raises(ValueError, match="no drive update"):
        apply_preset_update(config, plan_preset_update(config, _profile(config), DB_V1))


def test_a_measured_palette_is_never_replaced() -> None:
    config = _measured(_migrated())
    ours = measured_calibrations(config.length_prefixed_packets)
    plan = plan_preset_update(config, _profile(config), DB_V2)
    assert plan.drive is not None

    assert plan.drive.kept == ours
    assert not any(p in plan.drive.remove for p in ours)
    assert not any(p.packet_type == PACKET_TYPE_DISPLAY_CALIBRATION for p in plan.drive.add)
    updated = apply_preset_update(config, plan)
    assert [p for p in updated.length_prefixed_packets if p.packet_type == PACKET_TYPE_DISPLAY_CALIBRATION] == list(
        ours
    )


def test_an_unmeasured_palette_is_replaced_by_the_presets() -> None:
    config = _migrated()
    plan = plan_preset_update(config, _profile(config), DB_V2)

    assert plan.drive is not None and plan.drive.kept == ()
    assert any(p.packet_type == PACKET_TYPE_DISPLAY_CALIBRATION for p in plan.drive.remove)
    assert any(p.packet_type == PACKET_TYPE_DISPLAY_CALIBRATION for p in plan.drive.add)


def test_migration_keeps_a_measured_palette_of_the_same_panel() -> None:
    config = _measured(_migrated(PanelDatabase.load()))
    ours = measured_calibrations(config.length_prefixed_packets)
    plan = plan_migration(config, build_profile(FW3, config, IDENT), PanelDatabase.load(), panel_ic_type=PANEL)

    assert plan.needed and plan.kept == ours
    assert ours[0] in apply_migration(config, plan).length_prefixed_packets


def test_migration_to_another_panel_drops_the_measured_palette() -> None:
    config = _measured(_migrated(PanelDatabase.load()))
    plan = plan_migration(config, build_profile(FW3, config, IDENT), PanelDatabase.load(), panel_ic_type=0x0036)

    assert plan.needed and plan.kept == ()
    assert measured_calibrations(apply_migration(config, plan).length_prefixed_packets) == ()


# ── experimental modes ───────────────────────────────────────────────────────


def test_experimental_modes_are_left_out_unless_opted_in() -> None:
    db = _bundle(rev=1, experimental=[1])
    config = _config(displays=[_display(panel_ic_type=PANEL)])
    profile = build_profile(FW3, config, IDENT_NEEDS_MIGRATION)

    plain = apply_migration(config, plan := plan_migration(config, profile, db))
    assert plan.skipped_experimental == (1,)
    assert [m.mode_index for m in plain.display_modes] == [0]
    assert plain.display_panels[0].mode_count == 1
    assert plain.display_calibrations == []  # mode 1's palette goes with it
    used = set(plain.display_modes[0].lut_slots)
    assert all(lut.slot_id in used for lut in plain.display_luts)

    opted = apply_migration(config, plan_migration(config, profile, db, include_experimental=True))
    assert [m.mode_index for m in opted.display_modes] == [0, 1]
    assert opted.display_panels[0].mode_count == 2


def test_experimental_flag_in_the_mode_tail_counts_too() -> None:
    entry = DB_V1.entry_for(PANEL, TargetClass.DEFAULT)
    assert entry is not None
    packets = [
        dataclasses.replace(p, body=with_mode_tail(p.body, ModeExt(mode_flags=1)))
        if p.packet_type == PACKET_TYPE_DISPLAY_MODE and p.body[1] == 1
        else p
        for p in entry.packets
    ]
    entry = dataclasses.replace(entry, packets=tuple(packets))
    out, skipped = preset_packets(entry, include_experimental=False)
    assert skipped == (1,)
    assert [p.body[1] for p in out if p.packet_type == PACKET_TYPE_DISPLAY_MODE] == [0]


def test_an_experimental_mode_0_blocks() -> None:
    config = _config(displays=[_display(panel_ic_type=PANEL)])
    plan = plan_migration(config, build_profile(FW3, config, IDENT_NEEDS_MIGRATION), _bundle(rev=1, experimental=[0]))

    assert plan.blocked
    assert "mode 0" in plan.reason


def test_strip_modes_keeps_a_lut_slot_another_mode_still_uses() -> None:
    entry = DB_V1.entry_for(PANEL, TargetClass.DEFAULT)
    assert entry is not None
    mode0 = next(p for p in entry.packets if p.packet_type == PACKET_TYPE_DISPLAY_MODE and p.body[1] == 0)
    mode1 = next(p for p in entry.packets if p.packet_type == PACKET_TYPE_DISPLAY_MODE and p.body[1] == 1)
    shared = bytearray(mode1.body)
    shared[19:27] = mode0.body[19:27]  # mode 1 now uses mode 0's slots only
    packets = [dataclasses.replace(mode1, body=bytes(shared)) if p is mode1 else p for p in entry.packets]

    out = strip_modes(packets, frozenset({1}))
    luts_before = [p for p in packets if p.packet_type == PACKET_TYPE_DISPLAY_LUT]
    luts_after = [p for p in out if p.packet_type == PACKET_TYPE_DISPLAY_LUT]
    assert luts_after == luts_before


# ── simple-config presets (reported; applying them needs the bundle loader) ──


def test_simple_config_updates_are_reported() -> None:
    config = _migrated()
    config = dataclasses.replace(config, manufacturer=copy.deepcopy(config.manufacturer))
    config.manufacturer.simple_config_driver_index = 3
    config.manufacturer.simple_config_driver_rev = 1
    config.manufacturer.simple_config_power_index = 5
    config.manufacturer.simple_config_power_rev = 2
    config.manufacturer.simple_config_display_index = 9
    catalog = MappingCatalog(
        {
            PresetKind.DRIVER: {3: SimplePresetInfo(rev=2, changelog="new pins")},
            PresetKind.POWER: {5: SimplePresetInfo(rev=2)},
            PresetKind.DISPLAY: {9: SimplePresetInfo(rev=1, min_firmware=(3, 5, 0))},
        }
    )
    plan = plan_preset_update(config, _profile(config), DB_V1, catalog=catalog)

    (update,) = plan.updates
    assert (update.kind, update.key, update.device_rev, update.bundle_rev) == (PresetKind.DRIVER, 3, 1, 2)
    assert not update.applicable
    assert plan.notes == ("display preset 9 rev 1 needs firmware 3.5.0",)


def test_tailed_config_written_by_an_update_round_trips() -> None:
    config = _measured(_migrated())
    updated = apply_preset_update(config, plan_preset_update(config, _profile(config), DB_V2))

    assert serialize_config(parse_config_response(serialize_config(updated))) == serialize_config(updated)
