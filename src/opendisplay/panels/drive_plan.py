"""Which drives to send for one update: band pick, budget, scrub, stock full first.

The device runs whatever mode it is asked for (an ADVISORY temperature window does not stop it,
and it keeps no budget), so the host decides, from the config's modes, the stored
:class:`~opendisplay.panels.drive_state.DriveState` and the latest panel temperature (the
0x0085 ACK's ``temp_c``):

1. **Band pick.** Experimental modes that share a stem and differ only by a trailing band marker
   (a digit: ``fast6`` / ``fast7``; or ``cool`` / ``warm``: ``faster_cool`` / ``faster_warm``),
   each with an ADVISORY temperature window, are one family; the one whose window holds
   ``temp_c`` is driven. ``cool`` is band 1 and ``warm`` band 2, so ``partialcool`` and
   ``faster_cool`` share a band and moving to ``faster_warm`` counts as a band change. A band's edge sits
   half a degree past its last whole degree (``[-40, 24]`` and ``[25, 60]`` meet at 24.5 C).
   Within ``hysteresis_c`` of an edge, or with an unknown temperature, the requested mode is
   kept and the plan says ``near_edge``. An ENFORCED window (``temp_policy`` SUBSTITUTE or
   REJECT) is the device's own rule, so hysteresis never keeps a mode past it.
1a. **Out of window.** When the temperature is known and lies outside the chosen mode's
   enforced window, the host does what the device would: SUBSTITUTE plans mode 0, the stock
   full, instead (``substituted``); REJECT refuses the update. The device checks again at START
   (a stale reading), and its ACK says which mode ran.
2. **Stock full first** (mode 0) when the state is lost or was recorded against other drive
   packets, when the band changed since the last stock drive, or when the budget would run
   over (``budget_used + cost > budget_limit``; 0xB1 tail, 0xFF = unlimited). A mode that
   itself resets the count (mode 0, or a mode tail ``budget_cost`` of 0) needs none of these.
3. **Scrub.** A mode tail's ``scrub_before`` mode runs right before the mode.
4. **Experimental** modes (the requested one, its band sibling or its scrub) are refused unless
   the caller opts in with ``include_experimental=True``.

After each drive succeeds, :func:`record_drive` folds the ACK's mode and ``temp_c`` into the
state. Mode costs: a mode tail's ``budget_cost``; a mode without one costs nothing and resets
nothing, except mode 0, which always resets.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Final

from ..models.config import GlobalConfig
from ..models.drive_ext import BUDGET_UNLIMITED
from ..models.panel_drive import DisplayMode
from .drive_state import DriveState, drive_fingerprint

STOCK_MODE: Final = 0
TEMP_POLICY_ADVISORY: Final = 0
TEMP_POLICY_SUBSTITUTE: Final = 1
TEMP_POLICY_REJECT: Final = 2
DEFAULT_HYSTERESIS_C: Final = 1.0
_BAND_NAME: Final = re.compile(r"^(?P<stem>.*\D)(?P<band>\d)$")
_BAND_WORD: Final = re.compile(r"^(?P<stem>.*?)(?P<band>cool|warm)$", re.IGNORECASE)
_BAND_WORDS: Final = {"cool": 1, "warm": 2}


@dataclass(frozen=True)
class PlannedDrive:
    """One drive of a plan, in order."""

    mode_index: int
    name: str
    reason: str


@dataclass(frozen=True)
class DrivePlan:  # pylint: disable=too-many-instance-attributes
    """What to send for one update; drive ``steps`` in order, recording each."""

    steps: tuple[PlannedDrive, ...]
    requested: int  # mode_index the caller asked for
    chosen: int  # mode_index of the update itself (the last step)
    band: int | None  # band digit of ``chosen``, None when it is not banded
    near_edge: bool  # temperature unknown or within the hysteresis of a band edge
    stock_full_first: bool
    budget_after: int  # budget_used once every step has run
    notes: tuple[str, ...] = ()
    substituted: bool = False  # the temperature is outside the mode's enforced window: stock instead


def _has_window(mode: DisplayMode) -> bool:
    return (mode.temp_min_c, mode.temp_max_c) != (0, 0)


def enforced(mode: DisplayMode) -> bool:
    """Whether the device enforces the mode's window (SUBSTITUTE or REJECT)."""
    return _has_window(mode) and mode.temp_policy in (TEMP_POLICY_SUBSTITUTE, TEMP_POLICY_REJECT)


def mode_band(mode: DisplayMode) -> tuple[str, int] | None:
    """``(stem, band)`` of an experimental mode with a temperature window and a band marker.

    The band is the trailing digit, or 1 for a ``cool`` and 2 for a ``warm`` suffix.
    """
    if not mode.experimental or not _has_window(mode):
        return None
    match = _BAND_NAME.match(mode.name)
    if match is not None:
        return match["stem"].lower(), int(match["band"])
    word = _BAND_WORD.match(mode.name)
    if word is not None:
        return word["stem"].lower(), _BAND_WORDS[word["band"].lower()]
    return None


def mode_cost(mode: DisplayMode) -> tuple[int, bool]:
    """``(budget cost, resets the count)`` of one drive in ``mode``."""
    if mode.mode_index == STOCK_MODE:
        return 0, True
    ext = mode.ext
    if ext is None or ext.budget_cost is None:
        return 0, False
    return ext.budget_cost, ext.budget_cost == 0


def _spend(used: int, run: list[DisplayMode]) -> int:
    for mode in run:
        cost, resets = mode_cost(mode)
        used = 0 if resets else used + cost
    return used


def _modes(config: GlobalConfig, instance: int | None) -> dict[int, DisplayMode]:
    if instance is None:
        if not config.displays:
            raise ValueError("the config has no display")
        instance = config.displays[0].instance_number
    return {m.mode_index: m for m in config.display_modes if m.instance_number == instance}


def _resolve(modes: dict[int, DisplayMode], requested: int | str) -> DisplayMode:
    if isinstance(requested, int):
        if requested not in modes:
            raise ValueError(f"the config declares no mode {requested}")
        return modes[requested]
    found = [m for m in modes.values() if m.name.lower() == requested.lower()]
    if not found:
        names = ", ".join(m.name for m in modes.values()) or "none"
        raise ValueError(f"Unknown drive mode '{requested}'; the device has: {names}")
    return found[0]


def _edge_distance(mode: DisplayMode, temp_c: float) -> float:
    """Distance from ``temp_c`` to the nearer edge of the window (edges at min-0.5 and max+0.5)."""
    return min(abs(temp_c - (mode.temp_min_c - 0.5)), abs(temp_c - (mode.temp_max_c + 0.5)))


def _contains(mode: DisplayMode, temp_c: float) -> bool:
    return mode.temp_min_c - 0.5 <= temp_c < mode.temp_max_c + 0.5


def _pick_band(
    modes: dict[int, DisplayMode], wanted: DisplayMode, temp_c: int | None, hysteresis_c: float
) -> tuple[DisplayMode, bool, list[str]]:
    band = mode_band(wanted)
    if band is None:
        return wanted, False, []
    siblings = sorted(
        (m for m in modes.values() if (b := mode_band(m)) is not None and b[0] == band[0]),
        key=lambda m: m.mode_index,
    )
    if temp_c is None:
        return wanted, True, ["panel temperature unknown: keeping the requested band"]
    near = _edge_distance(wanted, temp_c) <= hysteresis_c
    if _contains(wanted, temp_c) or (near and not enforced(wanted)):
        return wanted, near, []
    for sibling in siblings:
        if _contains(sibling, temp_c):
            near = _edge_distance(sibling, temp_c) <= hysteresis_c
            return sibling, near, [f"{temp_c} C is in the band of '{sibling.name}', not '{wanted.name}'"]
    if enforced(wanted):  # plan_drive says what happens instead
        return wanted, True, []
    return wanted, True, [f"{temp_c} C is outside every '{band[0]}' band: keeping '{wanted.name}'"]


def _check_opt_in(mode: DisplayMode, include_experimental: bool, role: str) -> None:
    if mode.experimental and not include_experimental:
        raise ValueError(f"{role} '{mode.name}' is experimental; pass include_experimental=True to drive it")


def _budget_limit(config: GlobalConfig, instance: int) -> int | None:
    panel = next((p for p in config.display_panels if p.instance_number == instance), None)
    ext = panel.ext if panel is not None else None
    if ext is None or ext.budget_limit is None or ext.budget_limit == BUDGET_UNLIMITED:
        return None
    return ext.budget_limit


def plan_drive(  # pylint: disable=too-many-arguments,too-many-locals
    config: GlobalConfig,
    state: DriveState | None,
    temp_c: int | None,
    requested: int | str,
    *,
    include_experimental: bool = False,
    instance: int | None = None,
    hysteresis_c: float = DEFAULT_HYSTERESIS_C,
) -> DrivePlan:
    """Plan one update in ``requested`` (mode index or name). No I/O.

    Args:
        state: The stored state, ``None`` when there is none (then a stock full comes first).
        temp_c: The latest panel temperature (``None`` = unknown, the ACK's 0x7F).
        instance: The display instance; display 0's by default.

    Raises:
        ValueError: An unknown mode, no mode 0 to scrub with, or an experimental mode without
            ``include_experimental``.
    """
    modes = _modes(config, instance)
    wanted = _resolve(modes, requested)
    _check_opt_in(wanted, include_experimental, "mode")
    chosen, near_edge, notes = _pick_band(modes, wanted, temp_c, hysteresis_c)
    _check_opt_in(chosen, include_experimental, "band mode")
    substituted = False
    if temp_c is not None and enforced(chosen) and not _contains(chosen, temp_c):
        window = f"{chosen.temp_min_c} to {chosen.temp_max_c} C"
        if chosen.temp_policy == TEMP_POLICY_REJECT:
            raise ValueError(f"{temp_c} C is outside the window of '{chosen.name}' ({window}); the display refuses it")
        if STOCK_MODE not in modes:
            raise ValueError(f"{temp_c} C is outside the window of '{chosen.name}' and the config has no mode 0")
        notes = [*notes, f"{temp_c} C is outside the window of '{chosen.name}' ({window}): stock full instead"]
        chosen, near_edge, substituted = modes[STOCK_MODE], False, True

    scrub: DisplayMode | None = None
    scrub_index = chosen.ext.scrub_mode if chosen.ext is not None else None
    if scrub_index is not None:
        if scrub_index not in modes:
            raise ValueError(f"mode '{chosen.name}' scrubs with mode {scrub_index}, which the config does not declare")
        scrub = modes[scrub_index]
        _check_opt_in(scrub, include_experimental, "scrub mode")

    run = [m for m in (scrub, chosen) if m is not None]
    reasons: list[str] = []
    valid = state is not None and state.fingerprint == drive_fingerprint(config)
    if not valid:
        reasons.append("no drive state for this config" if state is None else "the drive packets changed")
    band = mode_band(chosen)
    if valid and state is not None and band is not None and state.last_band not in (None, band[1]):
        reasons.append(f"band changed from {state.last_band} to {band[1]}")
    used = state.budget_used if valid and state is not None else 0
    limit = _budget_limit(config, chosen.instance_number)
    after = _spend(used, run)
    if limit is not None and after > limit:
        reasons.append(f"budget {after} of {limit} after this update")

    first_resets = mode_cost(run[0])[1]
    stock_first = bool(reasons) and not first_resets
    steps: list[PlannedDrive] = []
    if stock_first:
        if STOCK_MODE not in modes:
            raise ValueError("the config declares no mode 0 to run a stock full with")
        stock = modes[STOCK_MODE]
        steps.append(PlannedDrive(stock.mode_index, stock.name, "stock full: " + "; ".join(reasons)))
        used = 0
    if scrub is not None:
        steps.append(PlannedDrive(scrub.mode_index, scrub.name, f"scrub before '{chosen.name}'"))
    steps.append(PlannedDrive(chosen.mode_index, chosen.name, "update"))

    budget = _spend(used, [modes[step.mode_index] for step in steps])
    return DrivePlan(
        steps=tuple(steps),
        requested=wanted.mode_index,
        chosen=chosen.mode_index,
        band=band[1] if band is not None else None,
        near_edge=near_edge,
        stock_full_first=stock_first,
        budget_after=budget,
        notes=tuple(notes),
        substituted=substituted,
    )


def record_drive(  # pylint: disable=too-many-arguments
    config: GlobalConfig,
    state: DriveState | None,
    mode_index: int,
    temp_c: int | None,
    *,
    instance: int | None = None,
    now: float | None = None,
) -> DriveState:
    """The state after a successful drive in ``mode_index`` (the mode the ACK reported).

    A lost or mismatched ``state`` starts over from an empty budget; the band is kept only
    while no stock/scrub drive has reset it.

    Raises:
        ValueError: The config declares no such mode.
    """
    modes = _modes(config, instance)
    if mode_index not in modes:
        raise ValueError(f"the config declares no mode {mode_index}")
    mode = modes[mode_index]
    fingerprint = drive_fingerprint(config)
    valid = state is not None and state.fingerprint == fingerprint
    used = state.budget_used if valid and state is not None else 0
    last_band = state.last_band if valid and state is not None else None
    spend, resets = mode_cost(mode)
    band = mode_band(mode)
    if resets:
        used, last_band = 0, None
    else:
        used += spend
        if band is not None:
            last_band = band[1]
    return DriveState(
        fingerprint=fingerprint,
        budget_used=used,
        last_mode=mode_index,
        last_band=last_band,
        last_temp_c=temp_c,
        updated=time.time() if now is None else now,
    )
