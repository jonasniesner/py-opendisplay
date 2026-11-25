"""0xAE mode tails and 0xB1 drive tails: kept verbatim everywhere, parsed by the views, built by drive_ext.

Firmware 3.0 ignores bytes after ``seq`` (0xAE) and ``sleep_seq`` (0xB1); presets plan 2.9/2.10
put host information there. Layouts are provisional (not yet in opendisplay_structs.h).
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest
from test_profile import ALPHA_CONFIG, FW3

from opendisplay.models.config import GlobalConfig
from opendisplay.models.config_json import config_from_json, config_to_json
from opendisplay.models.drive_ext import (
    BUDGET_UNLIMITED,
    SCRUB_NONE,
    DriveExt,
    LutFamily,
    ModeExt,
    mode_tail_offset,
    panel_tail_offset,
    with_drive_tail,
    with_mode_tail,
)
from opendisplay.models.ident import FirmwareTarget
from opendisplay.models.panel_drive import (
    PACKET_TYPE_DISPLAY_LUT,
    PACKET_TYPE_DISPLAY_MODE,
    PACKET_TYPE_DISPLAY_PANEL,
    DisplayMode,
    DisplayPanel,
)
from opendisplay.models.profile import build_profile
from opendisplay.panels import PanelDatabase, TargetClass, plan_migration
from opendisplay.panels.database import check_drive_packets_fit
from opendisplay.panels.migration import MigrationPlan, apply_migration
from opendisplay.protocol.config_parser import parse_config_response
from opendisplay.protocol.config_serializer import serialize_config

MODE_TAIL = ModeExt(mode_flags=0x01, budget_cost=2, scrub_before=0).to_bytes()
DRIVE_TAIL = DriveExt(drive_preset_key=0x001C, drive_rev=3, drive_flags=0, budget_limit=10).to_bytes()


def _with_tails(config: GlobalConfig) -> GlobalConfig:
    """``config`` with a tail on its second 0xAE and on its 0xB1 (the first 0xAE stays bare)."""
    packets = []
    seen_mode = False
    for p in config.length_prefixed_packets:
        if p.packet_type == PACKET_TYPE_DISPLAY_MODE and seen_mode:
            p = dataclasses.replace(p, body=p.body + MODE_TAIL)
        elif p.packet_type == PACKET_TYPE_DISPLAY_MODE:
            seen_mode = True
        elif p.packet_type == PACKET_TYPE_DISPLAY_PANEL:
            p = dataclasses.replace(p, body=p.body + DRIVE_TAIL)
        packets.append(p)
    return dataclasses.replace(copy.deepcopy(config), length_prefixed_packets=packets)


TAILED = _with_tails(ALPHA_CONFIG)


# ── round trips: tails are kept byte for byte ────────────────────────────────


def test_tails_survive_write_and_read_back() -> None:
    blob = serialize_config(TAILED)
    again = parse_config_response(blob)

    assert again.length_prefixed_packets == TAILED.length_prefixed_packets
    assert serialize_config(again) == blob


def test_tails_survive_json_export_and_import() -> None:
    """The JSON export is also the ``backup-config`` file, so this covers backup -> restore."""
    exported = json.loads(json.dumps(config_to_json(TAILED)))

    assert serialize_config(config_from_json(exported)) == serialize_config(TAILED)


def test_tails_survive_a_migration_that_does_not_remove_them() -> None:
    lut = next(p for p in TAILED.length_prefixed_packets if p.packet_type == PACKET_TYPE_DISPLAY_LUT)
    plan = MigrationPlan(
        needed=True, blocked=False, reason="test", panel_ic_type=None, target_class=TargetClass.BG22, remove=(lut,)
    )
    migrated = apply_migration(TAILED, plan)

    tailed = [p for p in TAILED.length_prefixed_packets if p.packet_type != PACKET_TYPE_DISPLAY_LUT]
    assert [p for p in migrated.length_prefixed_packets if p.packet_type != PACKET_TYPE_DISPLAY_LUT] == tailed


def test_tailed_drive_data_counts_as_present_and_runnable() -> None:
    """A tail must not make the planner think the stored drive data is broken."""
    assert check_drive_packets_fit(TAILED.length_prefixed_packets, TargetClass.BG22) == []
    profile = build_profile(FW3, TAILED, None)
    assert profile.target == FirmwareTarget.EFR32BG22

    plan = plan_migration(TAILED, profile, PanelDatabase.load())
    assert not plan.needed, plan.reason


def test_a_sequence_longer_than_the_body_is_still_a_problem() -> None:
    mode = next(p for p in ALPHA_CONFIG.length_prefixed_packets if p.packet_type == PACKET_TYPE_DISPLAY_MODE)
    truncated = dataclasses.replace(mode, body=mode.body[:-1])

    assert check_drive_packets_fit([truncated], TargetClass.BG22)


# ── the .tail / .ext views ───────────────────────────────────────────────────


def test_views_expose_the_tail() -> None:
    bare, tailed = TAILED.display_modes
    panel = TAILED.display_panels[0]

    assert bare.tail == b""
    assert bare.ext is None
    assert not bare.experimental
    assert tailed.tail == MODE_TAIL
    assert tailed.ext == ModeExt(mode_flags=1, budget_cost=2, scrub_before=0)
    assert tailed.experimental
    assert tailed.sequence == ALPHA_CONFIG.display_modes[1].sequence
    assert panel.tail == DRIVE_TAIL
    assert panel.ext == DriveExt(drive_preset_key=0x001C, drive_rev=3, budget_limit=10)
    assert panel.sleep_sequence == ALPHA_CONFIG.display_panels[0].sleep_sequence


def test_profile_labels_experimental_modes() -> None:
    modes = build_profile(FW3, TAILED, None).drive_modes

    assert [(m.name, m.experimental) for m in modes] == [("full", False), ("colour16", True)]


def test_tail_offsets() -> None:
    mode = next(p.body for p in TAILED.length_prefixed_packets if p.packet_type == PACKET_TYPE_DISPLAY_MODE)
    panel = next(p.body for p in TAILED.length_prefixed_packets if p.packet_type == PACKET_TYPE_DISPLAY_PANEL)

    assert mode_tail_offset(mode) == len(mode)
    assert panel_tail_offset(panel) == len(panel) - len(DRIVE_TAIL)
    with pytest.raises(ValueError):
        mode_tail_offset(mode[:36])
    with pytest.raises(ValueError):
        panel_tail_offset(panel[:13])


# ── the mode tail ────────────────────────────────────────────────────────────


def test_mode_tail_layout() -> None:
    assert ModeExt(mode_flags=1, budget_cost=3, scrub_before=SCRUB_NONE).to_bytes() == bytes([1, 1, 3, 0xFF])
    assert ModeExt(mode_flags=1).to_bytes() == bytes([1, 1])
    assert ModeExt(budget_cost=0).to_bytes() == bytes([1, 0, 0])


@pytest.mark.parametrize(
    "ext",
    [
        ModeExt(),
        ModeExt(mode_flags=1),
        ModeExt(mode_flags=1, budget_cost=5),
        ModeExt(mode_flags=0, budget_cost=0, scrub_before=4),
        ModeExt(mode_flags=1, budget_cost=1, scrub_before=SCRUB_NONE, hold_mask=0, ext_version=2, extra=b"\x09\x08"),
    ],
)
def test_mode_tail_round_trips(ext: ModeExt) -> None:
    assert ModeExt.from_tail(ext.to_bytes()) == ext


def test_mode_tail_scrub_mode() -> None:
    assert ModeExt(scrub_before=SCRUB_NONE).scrub_mode is None
    assert ModeExt().scrub_mode is None
    assert ModeExt(budget_cost=1, scrub_before=0).scrub_mode == 0


@pytest.mark.parametrize("tail", [b"", b"\x01", b"\x00\x01\x02\x03"])
def test_no_mode_ext_for_empty_short_or_version_0_tail(tail: bytes) -> None:
    assert ModeExt.from_tail(tail) is None


def test_mode_tail_refuses_a_gap_and_bad_values() -> None:
    with pytest.raises(ValueError, match="unset"):
        ModeExt(scrub_before=1).to_bytes()
    with pytest.raises(ValueError, match="u8"):
        ModeExt(budget_cost=256).to_bytes()
    with pytest.raises(ValueError, match="u8"):
        ModeExt(mode_flags=-1).to_bytes()
    with pytest.raises(ValueError, match="extra"):
        ModeExt(budget_cost=1, extra=b"\x00").to_bytes()


def test_with_mode_tail_replaces_and_removes() -> None:
    body = next(p.body for p in ALPHA_CONFIG.length_prefixed_packets if p.packet_type == PACKET_TYPE_DISPLAY_MODE)
    tailed = with_mode_tail(body, ModeExt(mode_flags=1))

    assert tailed == body + b"\x01\x01"
    assert with_mode_tail(tailed, ModeExt(mode_flags=0, budget_cost=4)) == body + b"\x01\x00\x04"
    assert with_mode_tail(tailed, None) == body
    assert DisplayMode.from_body(tailed).experimental


# ── the drive tail ───────────────────────────────────────────────────────────


def test_drive_tail_layout() -> None:
    ext = DriveExt(drive_preset_key=0x0123, drive_rev=0x0405, drive_flags=1)

    assert ext.to_bytes() == bytes.fromhex("012301050401")
    assert ext.hand_edited
    full = DriveExt(drive_preset_key=1, drive_rev=2, budget_limit=BUDGET_UNLIMITED, lut_family=2, busy_timeout_s=120)
    assert full.to_bytes() == bytes.fromhex("010100020000ff0278")
    assert full.lut_family_enum is LutFamily.JD_BWRY_535


@pytest.mark.parametrize(
    "ext",
    [
        DriveExt(),
        DriveExt(drive_preset_key=0xE001, drive_rev=65535, drive_flags=1),
        DriveExt(drive_preset_key=1, drive_rev=1, budget_limit=20),
        DriveExt(drive_preset_key=1, drive_rev=1, budget_limit=0, lut_family=1, busy_timeout_s=0),
        DriveExt(drive_preset_key=1, budget_limit=1, lut_family=9, busy_timeout_s=1, ext_version=3, extra=b"\xaa"),
    ],
)
def test_drive_tail_round_trips(ext: DriveExt) -> None:
    assert DriveExt.from_tail(ext.to_bytes()) == ext


def test_unknown_lut_family_stays_an_int() -> None:
    assert DriveExt(lut_family=9).lut_family_enum == 9
    assert DriveExt().lut_family_enum is None


@pytest.mark.parametrize("tail", [b"", b"\x01\x02\x03\x04\x05", b"\x00\x01\x00\x01\x00\x00"])
def test_no_drive_ext_for_empty_short_or_version_0_tail(tail: bytes) -> None:
    assert DriveExt.from_tail(tail) is None


def test_drive_tail_refuses_bad_values() -> None:
    with pytest.raises(ValueError, match="u16"):
        DriveExt(drive_rev=0x10000).to_bytes()
    with pytest.raises(ValueError, match="u8"):
        DriveExt(drive_flags=256).to_bytes()
    with pytest.raises(ValueError, match="unset"):
        DriveExt(lut_family=1).to_bytes()


def test_with_drive_tail_replaces_and_removes() -> None:
    body = next(p.body for p in ALPHA_CONFIG.length_prefixed_packets if p.packet_type == PACKET_TYPE_DISPLAY_PANEL)
    ext = DriveExt(drive_preset_key=0x1C, drive_rev=1)
    tailed = with_drive_tail(body, ext)

    assert DisplayPanel.from_body(tailed).ext == ext
    assert with_drive_tail(with_drive_tail(tailed, DriveExt(drive_rev=2)), None) == body
