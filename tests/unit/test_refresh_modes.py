"""Experimental refresh modes: device-equivalent lint, INLINE_LUT, drive state, planner, S6 subset tables.

Drive data and tables are copies from Firmware_Unified ``fw-refresh-modes``
(``panels/generated/refresh_modes.json``, ``panels/waves/*.bin``); the lint mirrors that
branch's ``shared/core/od_epd_lut.c`` and ``od_epd.c`` ``od_epd_validate``.
"""

from __future__ import annotations

import dataclasses
import json
import zlib
from collections import Counter
from pathlib import Path

import pytest

from opendisplay.models.config import GlobalConfig
from opendisplay.models.drive_ext import DriveExt, LutFamily, ModeExt, with_drive_tail, with_mode_tail
from opendisplay.models.lut_assemble import assemble_subset_s6
from opendisplay.models.lut_lint import (
    jd_encode,
    jd_value,
    lut_lint,
    mode_problem,
    never_send,
    sequence_lint,
    sequence_uploads_lut,
)
from opendisplay.models.panel_drive import DisplayLut, LengthPrefixedPacket, lut_crc16
from opendisplay.panels.drive_plan import mode_band, plan_drive, record_drive
from opendisplay.panels.drive_state import (
    DriveState,
    JsonFileDriveStateStore,
    MemoryDriveStateStore,
    drive_fingerprint,
)
from opendisplay.protocol.commands import build_drive_write_start, build_inline_lut_prefix

DATA = Path(__file__).parent / "data" / "refresh_modes"
GENERATED = json.loads((DATA / "refresh_modes.json").read_text())
S6_TABLES = sorted(p.name for p in DATA.glob("s6_*.bin"))
JD_TABLES = ["bwry29_BW2_kick3_w10.bin", "wave_colour.bin", "wave_edit.bin"]


def _table(name: str) -> bytes:
    return (DATA / name).read_bytes()


def _base() -> GlobalConfig:
    from opendisplay.protocol.config_parser import parse_config_response

    fixture = Path(__file__).parent.parent / "fixtures" / "configs" / "bg22_alpha_drive_packets.bin"
    return parse_config_response(fixture.read_bytes())


def _config(panel: str) -> GlobalConfig:
    counts: Counter[int] = Counter()
    packets = []
    for p in GENERATED["panels"][panel]["packets"]:
        packets.append(
            LengthPrefixedPacket(number=counts[p["id"]], packet_type=p["id"], body=bytes.fromhex(p["body_hex"]))
        )
        counts[p["id"]] += 1
    return dataclasses.replace(_base(), length_prefixed_packets=packets)


S73 = _config("s73_spectra6")


# ── lint: every shipped table passes, mutations fail ─────────────────────────


@pytest.mark.parametrize("name", S6_TABLES)
def test_shipped_s6_tables_pass(name: str) -> None:
    assert lut_lint(LutFamily.S6_800, _table(name)) is None
    assert lut_lint(LutFamily.S6_800, _table(name)[:799]) is None  # 799 B is allowed too


@pytest.mark.parametrize("name", JD_TABLES)
def test_shipped_jd_tables_pass(name: str) -> None:
    table = _table(name)
    assert lut_lint(LutFamily.JD_BWRY_535, table) is None
    assert lut_lint(LutFamily.X4_536, table) is None
    assert lut_lint(LutFamily.X4_536, table + b"\xff") is None


def test_s6_all_zero_header_fails() -> None:
    table = bytearray(_table("s6_73_F1d6.bin"))
    table[0:8] = bytes(8)
    assert "header" in (lut_lint(LutFamily.S6_800, bytes(table)) or "")


def test_s6_eight_zero_bytes_in_a_row_fail_seven_do_not() -> None:
    table = bytearray(_table("s6_73_F1d6.bin"))
    table[300:307] = bytes(7)
    table[299] = table[307] = 1
    assert lut_lint(LutFamily.S6_800, bytes(table)) is None
    table[307] = 0
    assert "zero bytes" in (lut_lint(LutFamily.S6_800, bytes(table)) or "")


@pytest.mark.parametrize(("family", "size"), [(1, 801), (1, 535), (2, 536), (2, 800), (3, 537), (0, 1025), (9, 1025)])
def test_wrong_sizes_fail(family: int, size: int) -> None:
    assert lut_lint(family, b"\x01" * size) is not None


def test_family_none_and_unknown_check_only_the_size() -> None:
    assert lut_lint(LutFamily.NONE, bytes(1024)) is None
    assert lut_lint(9, bytes(100)) is None


def test_jd_value_decodes_active_low_scrambled_binary() -> None:
    assert jd_value(0xFF) == 0
    assert jd_value(0x00) == 255
    assert jd_value(~(1 << 4) & 0xFF) == 1  # b4 is the 1s bit
    assert jd_value(~(1 << 5) & 0xFF) == 128  # b5 the 128s
    assert all(jd_value(jd_encode(v)) == v for v in range(256))


def test_jd_a_terminator_row_with_a_repeat_fails() -> None:
    table = bytearray(_table("bwry29_BW2_kick3_w10.bin"))
    # Block 0 ends at row 1 (M = 0, no time); M of row 1 is the byte before it, 7 + 11.
    table[7 + 11] = jd_encode(3)
    problem = lut_lint(LutFamily.JD_BWRY_535, bytes(table))
    assert problem is not None and "block 0 row 1" in problem
    assert lut_lint(LutFamily.X4_536, bytes(table)) is not None  # the x4 family runs the same check


def test_jd_row_7_is_exempt_and_m_zero_with_time_ends_the_block() -> None:
    table = bytearray(_table("wave_colour.bin"))  # block 1 runs all of rows 0-6
    assert lut_lint(LutFamily.JD_BWRY_535, bytes(table)) is None
    # Row 0 of block 0 with M = 0 but time: ends the block, so a later zero-time row with M is fine.
    table[7] = jd_encode(0)
    table[7 + 11 * 3] = jd_encode(5)
    assert lut_lint(LutFamily.JD_BWRY_535, bytes(table)) is None


# ── never-send list and the 0x40-after-upload rule ───────────────────────────


@pytest.mark.parametrize("cmd", [0x90, 0x91, 0xA0, 0xA1, 0xA2, 0xA3, 0x8F, 0xFB, 0xFF])
def test_never_send_commands(cmd: int) -> None:
    assert never_send(LutFamily.S6_800, cmd)
    assert not never_send(LutFamily.NONE, cmd)
    assert not never_send(9, cmd)
    assert sequence_lint(LutFamily.JD_BWRY_535, bytes([2, cmd, 0x00])) is not None
    assert sequence_lint(LutFamily.NONE, bytes([2, cmd, 0x00])) is None


def test_stock_commands_4d_and_b6_are_allowed() -> None:
    assert sequence_lint(LutFamily.JD_BWRY_535, bytes([2, 0x4D, 0x78, 2, 0xB6, 0x07])) is None


def test_never_send_is_seen_in_subst_window_and_write_scratch() -> None:
    subst = bytes([0x00, 0x02, 0x01, 0x00, 0xA0])  # SUBST, one literal pair: cmd 0xA0
    window = bytes([0x00, 0x0C, 0x01, 0x00, 0x91])
    scratch = bytes([0x00, 0x0A, 0xFB, 0x00, 0x01])
    for seq in (subst, window, scratch):
        assert sequence_lint(LutFamily.S6_800, seq) is not None
    assert sequence_lint(LutFamily.S6_800, bytes([0x00, 0x02, 0x01, 0x01, 0xA0])) is None  # geometry pair


def test_tsc_read_between_upload_and_refresh_is_refused() -> None:
    lut = bytes([1, 0x20, 0x00, 0x04])
    tsc = bytes([1, 0x40])
    refresh = bytes([1, 0x12])
    family = LutFamily.S6_800
    assert sequence_lint(family, tsc + lut + refresh) is None
    assert sequence_lint(family, lut + refresh + tsc) is None
    assert "0x40" in (sequence_lint(family, lut + tsc + refresh) or "")
    assert sequence_lint(LutFamily.NONE, lut + tsc + refresh) is None
    assert sequence_uploads_lut(lut) and not sequence_uploads_lut(tsc)


def test_a_truncated_or_unknown_escape_is_refused() -> None:
    assert sequence_lint(LutFamily.NONE, bytes([3, 0x20])) is not None
    assert sequence_lint(LutFamily.NONE, bytes([0x00, 0x7E])) is not None
    assert sequence_lint(LutFamily.NONE, bytes([0x00, 0x02, 0x00])) is not None  # SUBST of 0 pairs
    assert sequence_lint(LutFamily.NONE, bytes([0x00, 0x00, 0x00, 0x7E])) is None  # nothing after END


@pytest.mark.parametrize("panel", list(GENERATED["panels"]))
def test_every_generated_mode_passes_the_device_checks(panel: str) -> None:
    config = _config(panel)
    family = config.display_panels[0].ext.lut_family  # type: ignore[union-attr]
    for mode in config.display_modes:
        problem = mode_problem(mode, lut_family=family, stored_luts=config.display_luts)
        if mode.name == "hosttable":
            assert problem is not None and "slot 5" in problem  # it needs INLINE_LUT every time
            inline = {5: _table("s6_73_F1d6.bin")}
            assert mode_problem(mode, lut_family=family, stored_luts=config.display_luts, inline_luts=inline) is None
        else:
            assert problem is None, (mode.name, problem)


def test_a_stored_slot_that_fails_its_crc_or_lint_is_absent() -> None:
    mode = next(m for m in S73.display_modes if m.name == "fast6")
    good = next(lut for lut in S73.display_luts if lut.slot_id == 2)
    bad_crc = dataclasses.replace(good, crc16=good.crc16 ^ 1)
    assert "CRC" in (mode_problem(mode, lut_family=1, stored_luts=[bad_crc]) or "")
    zeros = DisplayLut.build(0, 2, bytes(800))
    assert "lint" in (mode_problem(mode, lut_family=1, stored_luts=[zeros]) or "")
    reserved = dataclasses.replace(good, lut_slot=good.lut_slot | 0x10)
    assert "no stored" in (mode_problem(mode, lut_family=1, stored_luts=[reserved]) or "")
    assert mode_problem(mode, lut_family=0, stored_luts=[]) is None  # undeclared family: no checks


# ── INLINE_LUT ───────────────────────────────────────────────────────────────


def test_inline_prefix_layout() -> None:
    a, b = _table("s6_73_F1d6.bin"), _table("s6_73_F1d7.bin")
    prefix = build_inline_lut_prefix({6: b, 5: a}, lut_family=LutFamily.S6_800)
    first = bytes([5]) + (800).to_bytes(2, "little") + lut_crc16(a).to_bytes(2, "little") + a
    second = bytes([6]) + (800).to_bytes(2, "little") + lut_crc16(b).to_bytes(2, "little") + b
    assert prefix == first + second


@pytest.mark.parametrize(
    "tables",
    [{}, {16: b"\x01" * 10}, {-1: b"\x01"}, {0: b""}, {0: b"\x01" * 1025}, {0: bytes(800)}, {0: b"\x01" * 535}],
)
def test_inline_prefix_refuses_what_the_device_refuses(tables: dict[int, bytes]) -> None:
    with pytest.raises(ValueError):
        build_inline_lut_prefix(tables, lut_family=LutFamily.S6_800)


def test_start_with_inline_lut_carries_the_flag_and_length() -> None:
    assert build_drive_write_start(8, 0, inline_len=810) == bytes([0x00, 0x85, 8, 0, 0x04]) + (810).to_bytes(
        2, "little"
    )
    compressed = build_drive_write_start(8, 0, uncompressed_size=1000, first_chunk=b"\x78", inline_len=810)
    assert compressed[4] == 0x05 and compressed[5:7] == (810).to_bytes(2, "little")
    assert compressed[7:11] == (1000).to_bytes(4, "little") and compressed[11:] == b"\x78"
    with pytest.raises(ValueError):
        build_drive_write_start(8, 0, inline_len=0x10000)


async def test_inline_tables_lead_an_uncompressed_stream(fake_transport) -> None:
    from PIL import Image
    from test_drive_write import IDENT_BG22, SEC_UNCLAIMED, _config_reply, _image, _script_transfer, _version

    import opendisplay
    from opendisplay.device.imaging import prepare_multipass

    base = _base()
    fake = fake_transport([_version(), IDENT_BG22, SEC_UNCLAIMED, _config_reply(base)])
    dev = opendisplay.connect(transport=fake)
    await dev.__aenter__()
    fake.written.clear()
    table = _table("wave_colour.bin")
    image: Image.Image = _image()
    frames, _ = prepare_multipass(image, base, next(m for m in base.display_modes if m.name == "colour16"))
    prefix = build_inline_lut_prefix({0: table})
    _script_transfer(dev, fake, bytes.fromhex("0085010319"), len(prefix) + len(frames))

    await dev.display.show(image, mode="colour16", inline_luts={0: table})

    assert fake.written[0] == bytes([0x00, 0x85, 1, 0, 0x04]) + len(prefix).to_bytes(2, "little")
    data = b"".join(frame[2:] for frame in fake.written[1:-1])
    assert data == prefix + frames
    await dev.__aexit__(None, None, None)


async def test_inline_tables_are_compressed_with_the_frames(fake_transport) -> None:
    from test_drive_write import (
        IDENT_BG22,
        SEC_UNCLAIMED,
        _config_reply,
        _image,
        _script_transfer,
        _streaming,
        _version,
    )

    import opendisplay
    from opendisplay.device.imaging import prepare_multipass
    from opendisplay.encoding import compress_image_data

    config = _streaming(_base())
    fake = fake_transport([_version(), IDENT_BG22, SEC_UNCLAIMED, _config_reply(config)])
    dev = opendisplay.connect(transport=fake)
    await dev.__aenter__()
    fake.written.clear()
    table = _table("wave_colour.bin")
    frames, _ = prepare_multipass(_image(), config, next(m for m in config.display_modes if m.name == "colour16"))
    prefix = build_inline_lut_prefix({0: table})
    stream = compress_image_data(prefix + frames, level=6, window_bits=9)
    _script_transfer(dev, fake, bytes.fromhex("0085010319"), len(stream), inline=200 - 11)

    await dev.display.show(_image(), mode="colour16", inline_luts={0: table})

    start = fake.written[0]
    assert start[4] == 0x05 and int.from_bytes(start[5:7], "little") == len(prefix)
    assert int.from_bytes(start[7:11], "little") == len(prefix) + len(frames)
    sent = start[11:] + b"".join(f[2:] for f in fake.written[1:] if f[:2] == b"\x00\x71")
    assert zlib.decompressobj().decompress(sent) == prefix + frames
    await dev.__aexit__(None, None, None)


async def test_a_table_the_device_would_refuse_is_refused_before_any_io(fake_transport) -> None:
    from test_drive_write import IDENT_BG22, SEC_UNCLAIMED, _config_reply, _image, _version

    import opendisplay

    base = _base()
    panel = next(p for p in base.length_prefixed_packets if p.packet_type == 0xB1)
    tailed = dataclasses.replace(panel, body=with_drive_tail(panel.body, DriveExt(budget_limit=20, lut_family=2)))
    config = dataclasses.replace(
        base, length_prefixed_packets=[tailed if p is panel else p for p in base.length_prefixed_packets]
    )
    fake = fake_transport([_version(), IDENT_BG22, SEC_UNCLAIMED, _config_reply(config)])
    dev = opendisplay.connect(transport=fake)
    await dev.__aenter__()
    fake.written.clear()
    broken = bytearray(_table("wave_colour.bin"))
    broken[7 + 11 * 5] = jd_encode(2)  # block 0's terminator row now has a repeat

    with pytest.raises(ValueError, match="lint"):
        await dev.display.show(_image(), mode="colour16", inline_luts={0: bytes(broken)})
    with pytest.raises(ValueError, match="mode="):
        await dev.display.show(_image(), inline_luts={0: _table("wave_colour.bin")})

    assert fake.written == []
    await dev.__aexit__(None, None, None)


# ── drive state store ────────────────────────────────────────────────────────


def test_fingerprint_follows_drive_packets_not_calibration() -> None:
    base = _base()
    calibration = next(p for p in base.length_prefixed_packets if p.packet_type == 0xB0)
    recal = dataclasses.replace(calibration, body=calibration.body[:-1] + bytes([calibration.body[-1] ^ 1]))
    other = dataclasses.replace(
        base, length_prefixed_packets=[recal if p is calibration else p for p in base.length_prefixed_packets]
    )
    assert drive_fingerprint(other) == drive_fingerprint(base)
    assert drive_fingerprint(S73) != drive_fingerprint(base)


def test_memory_store_round_trip() -> None:
    store = MemoryDriveStateStore()
    state = DriveState(fingerprint="ab", budget_used=3, last_mode=5, last_band=6, last_temp_c=-2, updated=1.5)
    assert store.get("OD1") is None
    store.put("OD1", state)
    assert store.get("OD1") == state
    store.forget("OD1")
    assert store.get("OD1") is None


def test_json_store_round_trip_and_a_corrupt_file_reads_as_lost(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "drive_state.json"
    store = JsonFileDriveStateStore(path)
    state = DriveState(fingerprint="ab", budget_used=1, last_mode=3)
    store.put("OD1", state)
    assert JsonFileDriveStateStore(path).get("OD1") == state
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_text("{not json")
    assert store.get("OD1") is None
    store.put("OD2", state)  # replaces the corrupt file
    assert store.get("OD2") == state
    path.write_text(json.dumps({"format": 1, "devices": {"OD1": {"budget_used": 2}}}))
    assert store.get("OD1") is None  # no fingerprint: lost


# ── planner ──────────────────────────────────────────────────────────────────


def _state(**kwargs: object) -> DriveState:
    return DriveState(fingerprint=drive_fingerprint(S73), **kwargs)  # type: ignore[arg-type]


def _steps(plan: object) -> list[str]:
    return [s.name for s in plan.steps]  # type: ignore[attr-defined]


def test_experimental_modes_need_the_opt_in() -> None:
    with pytest.raises(ValueError, match="experimental"):
        plan_drive(S73, _state(), 20, "fast6")
    assert _steps(plan_drive(S73, _state(), 20, "full")) == ["full"]


def test_bands_are_families_with_a_window_and_a_digit() -> None:
    names = {m.name: mode_band(m) for m in S73.display_modes}
    assert names["fast6"] == ("fast", 6) and names["replica7"] == ("replica", 7)
    assert names["hosttable"] is None and names["pll08"] is None and names["full"] is None
    ee02 = _config("ee02_709_spectra6")
    assert mode_band(next(m for m in ee02.display_modes if m.name == "fast")) is None


def test_cool_and_warm_suffixes_are_bands_1_and_2() -> None:
    """The readable names (faster_cool / faster_warm, partialcool) band like the digit names."""
    renamed = {"fast6": "faster_cool", "fast7": "faster_warm", "partial6": "partialcool", "pll08": "full_quick"}
    modes = {
        renamed.get(m.name, m.name): dataclasses.replace(m, name=renamed.get(m.name, m.name)) for m in S73.display_modes
    }
    assert mode_band(modes["faster_cool"]) == ("faster_", 1)
    assert mode_band(modes["faster_warm"]) == ("faster_", 2)
    assert mode_band(modes["partialcool"]) == ("partial", 1)
    assert mode_band(modes["full_quick"]) is None


@pytest.mark.parametrize(
    ("temp", "chosen", "near"),
    [
        (10, "fast6", False),
        (23, "fast6", False),
        (24, "fast6", True),
        (25, "fast6", True),
        (26, "fast7", False),
        (30, "fast7", False),
        (None, "fast6", True),
        (80, "fast6", True),
    ],
)
def test_band_pick(temp: int | None, chosen: str, near: bool) -> None:
    plan = plan_drive(S73, _state(), temp, "fast6", include_experimental=True)
    assert _steps(plan) == [chosen]
    assert plan.near_edge is near
    assert plan.requested == 5


def test_lost_or_mismatched_state_runs_a_stock_full_first() -> None:
    plan = plan_drive(S73, None, 20, "fast6", include_experimental=True)
    assert _steps(plan) == ["full", "fast6"] and plan.stock_full_first and plan.budget_after == 1
    stale = DriveState(fingerprint="0" * 32, budget_used=0)
    assert _steps(plan_drive(S73, stale, 20, 5, include_experimental=True)) == ["full", "fast6"]
    # A mode that resets the count is its own stock drive.
    assert _steps(plan_drive(S73, None, 20, "pll08", include_experimental=True)) == ["pll08"]
    assert _steps(plan_drive(S73, None, 20, 0)) == ["full"]


def test_a_band_change_runs_a_stock_full_first() -> None:
    plan = plan_drive(S73, _state(last_band=6, budget_used=2), 30, "fast6", include_experimental=True)
    assert _steps(plan) == ["full", "fast7"] and plan.band == 7 and plan.budget_after == 1
    same = plan_drive(S73, _state(last_band=7, budget_used=2), 30, "fast7", include_experimental=True)
    assert _steps(same) == ["fast7"] and same.budget_after == 3


def test_the_budget_runs_a_stock_full_first_when_it_would_overrun() -> None:
    assert _steps(plan_drive(S73, _state(budget_used=9), 20, "fast6", include_experimental=True)) == ["fast6"]
    plan = plan_drive(S73, _state(budget_used=10), 20, "fast6", include_experimental=True)
    assert _steps(plan) == ["full", "fast6"] and plan.budget_after == 1


def test_unlimited_budget_never_forces_a_full() -> None:
    panel = next(p for p in S73.length_prefixed_packets if p.packet_type == 0xB1)
    ext = dataclasses.replace(S73.display_panels[0].ext, budget_limit=0xFF)  # type: ignore[type-var]
    unlimited = dataclasses.replace(panel, body=with_drive_tail(panel.body, ext))
    config = dataclasses.replace(
        S73, length_prefixed_packets=[unlimited if p is panel else p for p in S73.length_prefixed_packets]
    )
    state = DriveState(fingerprint=drive_fingerprint(config), budget_used=200)
    assert _steps(plan_drive(config, state, 20, "fast6", include_experimental=True)) == ["fast6"]


def test_scrub_before_runs_right_before_the_mode() -> None:
    packet = next(p for p in S73.length_prefixed_packets if p.packet_type == 0xAE and p.body[1] == 3)  # replica6
    scrubbed = dataclasses.replace(
        packet, body=with_mode_tail(packet.body, ModeExt(mode_flags=1, budget_cost=1, scrub_before=2))
    )
    config = dataclasses.replace(
        S73, length_prefixed_packets=[scrubbed if p is packet else p for p in S73.length_prefixed_packets]
    )
    state = DriveState(fingerprint=drive_fingerprint(config), budget_used=10)
    plan = plan_drive(config, state, 20, "replica6", include_experimental=True)
    # The scrub (pll08, cost 0) resets the count itself, so no extra stock full.
    assert _steps(plan) == ["pll08", "replica6"] and plan.budget_after == 1
    assert not plan.stock_full_first


def test_unknown_modes_are_named() -> None:
    with pytest.raises(ValueError, match="fast6"):
        plan_drive(S73, None, 20, "sepia")
    with pytest.raises(ValueError):
        plan_drive(S73, None, 20, 42)


def test_record_drive_tracks_budget_band_and_resets() -> None:
    state = record_drive(S73, None, 5, 20, now=1.0)
    assert (state.budget_used, state.last_mode, state.last_band, state.last_temp_c) == (1, 5, 6, 20)
    state = record_drive(S73, state, 8, None, now=2.0)  # hosttable: no band, keeps band 6
    assert (state.budget_used, state.last_band, state.last_temp_c) == (2, 6, None)
    state = record_drive(S73, state, 0, 21, now=3.0)  # stock full resets
    assert (state.budget_used, state.last_band) == (0, None)
    assert state.matches(S73) and not state.matches(_base())
    assert record_drive(_base(), state, 0, 21).budget_used == 0


def test_a_plan_driven_through_the_store() -> None:
    store = MemoryDriveStateStore()
    temps = [20, 20, 30, 30]
    seen = []
    for temp in temps:
        plan = plan_drive(S73, store.get("OD1"), temp, "fast6", include_experimental=True)
        seen.append(_steps(plan))
        state = store.get("OD1")
        for step in plan.steps:
            state = record_drive(S73, state, step.mode_index, temp)
        assert state is not None
        store.put("OD1", state)
    assert seen == [["full", "fast6"], ["fast6"], ["full", "fast7"], ["fast7"]]


# ── S6 subset tables ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", S6_TABLES)
def test_subset_keeps_present_border_and_hold_slots(name: str) -> None:
    table = _table(name)
    out = assemble_subset_s6(table, {0, 6})
    assert len(out) == 800 and out[:96] == table[:96]
    hold = table[96 + 4 * 88 : 96 + 5 * 88]
    for code in range(8):
        slot = out[96 + code * 88 : 96 + (code + 1) * 88]
        assert slot == (table[96 + code * 88 : 96 + (code + 1) * 88] if code in {0, 1, 4, 6} else hold)
    assert lut_lint(LutFamily.S6_800, out) is None
    assert assemble_subset_s6(table, set(range(8))) == table


def test_subset_refuses_bad_input_and_a_result_that_fails_the_lint() -> None:
    table = _table("s6_73_F1d6.bin")
    with pytest.raises(ValueError):
        assemble_subset_s6(table[:799], {0})
    with pytest.raises(ValueError):
        assemble_subset_s6(table, {8})
    with pytest.raises(ValueError):
        assemble_subset_s6(table, {0}, hold_code=9)
    synthetic = bytearray(b"\x01" * 800)
    hold_at = 96 + 4 * 88
    synthetic[hold_at : hold_at + 4] = bytes(4)
    synthetic[hold_at + 84 : hold_at + 88] = bytes(4)
    assert lut_lint(LutFamily.S6_800, bytes(synthetic)) is None
    with pytest.raises(ValueError, match="lint"):
        assemble_subset_s6(bytes(synthetic), {0})  # holds at slots 2 and 3 join into 8 zeros
