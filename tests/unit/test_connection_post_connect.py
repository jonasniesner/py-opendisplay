"""Tests for post-connect link readiness (notify retry, settle, fresh GATT)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opendisplay.transport.connection import (
    BLEConnection,
    NOTIFY_SETUP_MAX_ATTEMPTS,
    NOTIFY_SETUP_RETRY_DELAY_S,
    POST_NOTIFY_SETTLE_S,
)


def _make_client_with_char(*, properties: list[str], cccd_value: bytes | None = None) -> MagicMock:
    char = MagicMock()
    char.properties = properties
    if cccd_value is not None:
        cccd = MagicMock()
        cccd.uuid = "00002902-0000-1000-8000-00805f9b34fb"
        char.descriptors = [cccd]
    else:
        char.descriptors = []
    service = MagicMock()
    service.characteristics = [char]
    client = MagicMock()
    client.is_connected = True
    client.services.get_service.return_value = service
    client.start_notify = AsyncMock()
    if cccd_value is not None:
        client.read_gatt_descriptor = AsyncMock(return_value=cccd_value)
    return client


@pytest.mark.asyncio
async def test_fresh_gatt_disables_service_cache() -> None:
    conn = BLEConnection("AA:BB:CC:DD:EE:FF", fresh_gatt=True)
    assert conn.use_services_cache is False
    assert conn.fresh_gatt is True


@pytest.mark.asyncio
@patch("opendisplay.transport.connection.asyncio.sleep", new_callable=AsyncMock)
async def test_setup_notifications_settles_after_notify(mock_sleep: AsyncMock) -> None:
    conn = BLEConnection("AA:BB:CC:DD:EE:FF")
    conn._client = _make_client_with_char(properties=["write", "write-without-response", "notify"])

    await conn._setup_notifications()

    mock_sleep.assert_awaited()
    assert POST_NOTIFY_SETTLE_S in [call.args[0] for call in mock_sleep.await_args_list]


@pytest.mark.asyncio
@patch("opendisplay.transport.connection.asyncio.sleep", new_callable=AsyncMock)
async def test_start_notifications_retries_transient_failures(mock_sleep: AsyncMock) -> None:
    conn = BLEConnection("AA:BB:CC:DD:EE:FF")
    client = _make_client_with_char(properties=["write", "notify"])
    client.start_notify = AsyncMock(side_effect=[OSError("gatt busy"), None])
    conn._client = client
    conn._notification_characteristic = client.services.get_service.return_value.characteristics[0]

    await conn._start_notifications_with_retry()

    assert client.start_notify.await_count == 2
    mock_sleep.assert_awaited_with(NOTIFY_SETUP_RETRY_DELAY_S)


@pytest.mark.asyncio
async def test_start_notifications_raises_after_max_attempts() -> None:
    conn = BLEConnection("AA:BB:CC:DD:EE:FF")
    client = _make_client_with_char(properties=["write", "notify"])
    client.start_notify = AsyncMock(side_effect=OSError("gatt busy"))
    conn._client = client
    conn._notification_characteristic = client.services.get_service.return_value.characteristics[0]

    with pytest.raises(Exception, match="Failed to start notifications"):
        await conn._start_notifications_with_retry()

    assert client.start_notify.await_count == NOTIFY_SETUP_MAX_ATTEMPTS


@pytest.mark.asyncio
@patch("opendisplay.transport.connection.asyncio.sleep", new_callable=AsyncMock)
async def test_wait_for_notify_subscription_polls_cccd(mock_sleep: AsyncMock) -> None:
    conn = BLEConnection("AA:BB:CC:DD:EE:FF")
    client = _make_client_with_char(properties=["write", "notify"], cccd_value=b"\x00\x00")
    conn._client = client
    conn._notification_characteristic = client.services.get_service.return_value.characteristics[0]
    client.read_gatt_descriptor = AsyncMock(side_effect=[b"\x00\x00", b"\x01\x00"])

    await conn._wait_for_notify_subscription()

    assert client.read_gatt_descriptor.await_count == 2
