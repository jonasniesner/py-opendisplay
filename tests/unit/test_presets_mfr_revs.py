"""ManufacturerData simple-config revisions (presets plan 2.9) survive every round trip.

The three ``simple_config_*_rev`` u16 fields live in the 6 bytes that were ``reserved``. The JSON
import used to zero those bytes and the export to drop them, so a toolbox/JSON round trip would
have wiped the revisions.
"""

from __future__ import annotations

import copy
import json

import pytest
from test_profile import ALPHA_CONFIG

from opendisplay.models.config import ManufacturerData
from opendisplay.models.config_json import config_from_json, config_to_json
from opendisplay.protocol.config_parser import parse_config_response
from opendisplay.protocol.config_serializer import serialize_config, serialize_manufacturer_data


def _mfr(**revs: int) -> ManufacturerData:
    mfr = ManufacturerData(
        manufacturer_id=0x0005,
        board_type=3,
        board_revision=1,
        reserved=bytes(6),
        simple_config_driver_index=7,
        simple_config_display_index=12,
        simple_config_power_index=2,
        simple_config_configured_at=1_764_000_000,
    )
    for name, value in revs.items():
        setattr(mfr, name, value)
    return mfr


def test_revs_are_u16_le_in_the_former_reserved_bytes() -> None:
    mfr = _mfr(simple_config_driver_rev=0x0102, simple_config_display_rev=3, simple_config_power_rev=0xFFFF)

    assert mfr.reserved == bytes.fromhex("02010300ffff")
    assert serialize_manufacturer_data(mfr)[16:22] == mfr.reserved


def test_revs_read_back_from_bytes() -> None:
    raw = serialize_manufacturer_data(_mfr(simple_config_driver_rev=4, simple_config_display_rev=5))
    parsed = ManufacturerData.from_bytes(raw)

    assert (parsed.simple_config_driver_rev, parsed.simple_config_display_rev, parsed.simple_config_power_rev) == (
        4,
        5,
        0,
    )


def test_unknown_rev_is_zero() -> None:
    assert _mfr().simple_config_driver_rev == 0
    assert (
        ManufacturerData(manufacturer_id=0, board_type=0, board_revision=0, reserved=b"").simple_config_power_rev == 0
    )


@pytest.mark.parametrize("value", [-1, 0x10000])
def test_rev_out_of_range_is_refused(value: int) -> None:
    with pytest.raises(ValueError, match="u16"):
        _mfr().simple_config_display_rev = value


def test_json_round_trip_keeps_the_revs() -> None:
    config = copy.deepcopy(ALPHA_CONFIG)
    config.manufacturer.simple_config_driver_rev = 9
    config.manufacturer.simple_config_display_rev = 0x1234
    config.manufacturer.simple_config_power_rev = 1
    exported = json.loads(json.dumps(config_to_json(config)))
    fields = next(p for p in exported["packets"] if p["id"] == "2")["fields"]
    assert fields["simple_config_display_rev"] == str(0x1234)
    assert fields["reserved"] == "0x" + config.manufacturer.reserved.hex()

    back = config_from_json(exported)
    assert back.manufacturer.reserved == config.manufacturer.reserved
    assert serialize_config(back) == serialize_config(config)


def test_json_import_keeps_unknown_reserved_bytes_without_named_fields() -> None:
    """A JSON from a writer that only knows ``reserved`` keeps its bytes."""
    exported = config_to_json(ALPHA_CONFIG)
    fields = next(p for p in exported["packets"] if p["id"] == "2")["fields"]
    for name in ("simple_config_driver_rev", "simple_config_display_rev", "simple_config_power_rev"):
        del fields[name]
    fields["reserved"] = "0xa1b2c3d4e5f6"

    assert config_from_json(exported).manufacturer.reserved == bytes.fromhex("a1b2c3d4e5f6")


def test_named_rev_wins_over_reserved_blob() -> None:
    exported = config_to_json(ALPHA_CONFIG)
    fields = next(p for p in exported["packets"] if p["id"] == "2")["fields"]
    fields["reserved"] = "0x010002000300"
    fields["simple_config_display_rev"] = "42"
    del fields["simple_config_driver_rev"], fields["simple_config_power_rev"]

    mfr = config_from_json(exported).manufacturer
    assert (mfr.simple_config_driver_rev, mfr.simple_config_display_rev, mfr.simple_config_power_rev) == (1, 42, 3)


def test_legacy_json_reserved_scalar_reads_as_zero() -> None:
    """Older exports wrote ``"reserved": "0x0"`` and no rev fields."""
    exported = config_to_json(ALPHA_CONFIG)
    fields = next(p for p in exported["packets"] if p["id"] == "2")["fields"]
    fields["reserved"] = "0x0"
    for name in ("simple_config_driver_rev", "simple_config_display_rev", "simple_config_power_rev"):
        del fields[name]

    assert config_from_json(exported).manufacturer.reserved == bytes(6)


def test_binary_round_trip_keeps_the_revs() -> None:
    blob = serialize_config(ALPHA_CONFIG)
    # Patch the revs into the stored config and re-read it.
    config = parse_config_response(blob)
    config.manufacturer.simple_config_power_rev = 77
    again = parse_config_response(serialize_config(config))

    assert again.manufacturer.simple_config_power_rev == 77
