# Presets test implementation (py-opendisplay side): report

Branch `local/presets-test` on a fresh clone of `origin/main` 307b3cb (8.3.1). Local only:
not pushed, no PR. This is our own test implementation of the presets plan (proposal
sections 2.9, 2.10, 5.3 F1/F4, 6), written so the firmware side can be tried against it.
Gabriel's implementation is separate (issue draft:
`v3_worktrees/reports/py_opendisplay_issue_presets_draft.md`).

`uv run pytest`: **2168 passed** (2082 before; 86 new). `uv run prek run --all-files`
(ruff, ruff-format, mypy strict, pylint): passes.

## What was implemented

| # | Item | Where | Tests |
|---|---|---|---|
| 1 | JSON import keeps `ManufacturerData.reserved` (it used to zero it) and export writes it (it used to write `"0x0"`). The three revisions `simple_config_driver_rev` / `_display_rev` / `_power_rev` are read/write properties over `reserved[0:6]`, so binary, JSON and constructors stay compatible. The export writes both the named revs and the raw `reserved`; on import the named fields win. | `models/config.py`, `models/config_json.py` | `test_presets_mfr_revs.py` |
| 2 | `.tail` on `DisplayMode` (bytes after `seq`) and `DisplayPanel` (bytes after `sleep_seq`), plus `.ext` (parsed tail). Round-trip tests: write/read, JSON export/import (= `backup-config` / restore), `apply_migration` when the packet is not removed. **Fix:** `check_drive_packets_fit` demanded `head + seq_len == len(body)`, so a tailed 0xAE/0xB1 counted as broken (re-migrated every time, and DB entries with tails would be refused). It now demands `<=`. | `models/panel_drive.py`, `panels/database.py` | `test_drive_tails.py` |
| 3 | Tail parsers and builders: `ModeExt` (0xAE) and `DriveExt` (0xB1), `with_mode_tail` / `with_drive_tail`. `DisplayMode.experimental`; `ProfileMode.experimental`, so the profile can label those modes. | `models/drive_ext.py`, `models/profile.py` | `test_drive_tails.py` |
| 4 | `plan_preset_update(config, profile, db, *, catalog=None, include_experimental=False)` and `apply_preset_update`. A drive update is a `MigrationPlan` (same `apply_migration`). The DB loader keeps `rev`, `min_firmware`, `changelog`, `modes[].experimental` and `experimental_note`. Shared helpers: `measured_calibrations`, `strip_modes`, `stamp_drive_preset`, `preset_packets`. **`plan_migration` changes:** it keeps a measured 0xB0 (`cal_flags` bit 0) when the panel id stays the same, leaves experimental modes out unless `include_experimental=True`, and stamps the 0xB1 drive tail when the entry has a `rev`. `MigrationPlan` gains `kept` and `skipped_experimental`. | `panels/preset_update.py`, `panels/migration.py`, `panels/database.py` | `test_preset_update.py` |
| 5 | Compressed 0xAF slots: `DisplayLut.compressed` (bit 7), `.slot_id` (bits 0-3), `.waveform` (inflated with raw deflate, wbits -10, capped at 1024 B, refuses truncated/garbage streams), `DisplayLut.build(..., compress=None)` (auto: compress only when smaller), `to_body()`, `find_lut()`. `crc16` covers the stored (compressed) bytes, as decided. `check_drive_packets_fit` now also checks each 0xAF (stored ≤ 1024 B, CRC, inflates ≤ 1024 B). | `models/panel_drive.py`, `panels/database.py` | `test_lut_compressed.py` |

The vendored bundle has no `rev` / `experimental` yet, so the current DB behaves exactly as
before (an entry without `rev` is written unchanged; `test_migration.py` is untouched and passes).

## Provisional byte layouts (not in `opendisplay_structs.h` yet)

All little-endian. A reader of `ext_version` N parses the fields it knows from any version ≥ N
and keeps the rest as `extra`, so later versions may only append fields. Fields after the
core may be left off, and a missing field reads as `None`. An empty tail, `ext_version` 0, or
a truncated core means "no extension".

**ManufacturerData (0x02) bytes 16-21** (former `reserved[6]`): `simple_config_driver_rev`
u16, `simple_config_display_rev` u16, `simple_config_power_rev` u16; 0 = unknown.

**0xAE mode tail** (after `seq`), v1 = 4 bytes, core = 2:

| off | field | |
|---|---|---|
| 0 | `ext_version` u8 | 1 |
| 1 | `mode_flags` u8 | bit0 EXPERIMENTAL, bits 1-7 reserved |
| 2 | `budget_cost` u8 | budget units one update uses; 0 = stock/scrub mode, which resets the count (optional) |
| 3 | `scrub_before` u8 | `mode_index` the host must run right before pass 1; 0xFF = none (optional) |

**0xB1 drive tail** (after `sleep_seq`), v1 = 9 bytes, core = 6:

| off | field | |
|---|---|---|
| 0 | `ext_version` u8 | 1 |
| 1-2 | `drive_preset_key` u16 | panel_ic_type key of the preset (0 = not from a preset) |
| 3-4 | `drive_rev` u16 | that entry's rev (0 = unknown) |
| 5 | `drive_flags` u8 | bit0 HAND_EDITED (never offer an update) |
| 6 | `budget_limit` u8 | budget units between stock refreshes; 0xFF = no limit (optional) |
| 7 | `lut_family` u8 | 0 none, 1 S6-800, 2 JD-BWRY-535, 3 x4-536 (optional) |
| 8 | `busy_timeout_s` u8 | BUSY ceiling in s, wins over `busy_timeout_ms` when non-zero (optional) |

**0xAF `lut_slot`:** bit 7 = compressed (raw deflate, 10-bit window), bits 0-3 = slot id
(0-15, the id `DisplayMode.lut_slots` names), bits 4-6 reserved (ignored when reading). Stored
data ≤ 1024 B, inflated ≤ 1024 B, `crc16` (CCITT-FALSE) over the stored bytes.

## Needs Jonas's decision

1. **`scrub_before` semantics.** The proposal names the field but not its form. I made it the
   `mode_index` of the mode to run first (0xFF = none), so the config says which mode
   scrubs. The alternative is a flag bit in `mode_flags` meaning "stock full first".
2. **`budget_cost` / `budget_limit` units and sentinels.** u8 each; cost 0 = resets the count;
   limit 0xFF = unlimited. Is a limit of 0 meant to be "stock only" (multi-pass has a starting
   value of 0 in 5.3 F4)?
3. **What goes in the 0xB1 tail v1.** I put `budget_limit`, `lut_family` (F2) and
   `busy_timeout_s` (F7) into v1 as optional fields after the 6-byte core, so the firmware
   side can test them. If they should wait for v2, drop bytes 6-8. Nothing else changes.
4. **No drive tail = rev 0 of the display's own panel id.** `plan_preset_update` then uses
   0x20 `panel_ic_type` as the key. The result: once the bundle carries revs, every device
   migrated before the tail existed is offered one drive update. That follows 2.9 ("0 =
   older than rev 1"), but confirm it.
5. **Measured 0xB0 during a migration.** Kept when the panel id stays the same, dropped on a
   remap or when another panel is named. When the new preset no longer has that mode, the
   measured 0xB0 stays as an orphan (host-only, harmless). OK?
6. **JSON precedence.** The named `simple_config_*_rev` fields win over the raw `reserved` hex.
   So someone who edits only `reserved` in an exported file sees the edit ignored.
7. **0xB1 `mode_count`** is rewritten when experimental modes are left out (it counts the
   modes kept). Checked against Firmware_Unified `origin/main` 31e46ea: firmware never reads
   it. It finds modes by `mode_index` (`od_epd_cfg.c` `mode_entry`, `e->sub == mode`) and bounds
   them at one past the highest index (`:349-352`), so a gap (modes 0 and 2 kept) is fine.
   Note: the local `feat/silabs-config-driven-driver` checkout is older and finds modes **by
   position**, where that gap would shift mode 2 into slot 1. Don't test this branch against
   that build. Should `mode_count` be defined as "count" or as "highest index + 1"?

## Blockers outside py-opendisplay (found here)

- **The website toolbox rejects tails.** `v3_worktrees/web/httpdocs/js/od-v3.js:400` and `:404`
  (v3/website c013f1d) use `37 + len !== b.length` and `13 + b[12] !== b.length`. The same
  exact-length check was in py and is fixed here. Before any writer puts a tail on a device,
  the JS check must become `>`, or the toolbox will call tailed drive data broken and
  re-migrate it (dropping the tail).
- The website's config writer was not checked for `ManufacturerData.reserved` (proposal 2.9
  says so too).

## Not done (later items)

- Vendoring from a `presets-v*` release (`panels/SOURCE.md`). There is no release yet; the
  loader already keeps the new fields.
- A simple-config bundle loader. `plan_preset_update` takes a `SimpleConfigCatalog` protocol
  (`MappingCatalog` for tests) and only **reports** driver/display/power updates
  (`applicable=False`); it cannot apply them without the bundle's packet data.
- The CLI: `migrate` has no `--include-experimental` flag, nothing uses `plan_preset_update`
  yet, `drive_write` does not refuse experimental modes, and the setup writers do not fill
  the simple-config revs.
- ~~The host drive-state store, band picking, `INLINE_LUT`, the table assembler~~: done, see
  "Refresh modes" below. Priors are not in the store.
- **TODO (deferred): `display_tracks`.** Not started.
- Vendoring the full lint/charge gate from Firmware_Unified `presets/tools/` (the device-side
  lint is mirrored, the charge gate is not).
- Device-side: nothing here is checked against firmware. The layouts above are what the
  firmware test implementation should parse.

## Refresh modes (second commit): INLINE_LUT, device lint, drive state, planner, S6 subset

Built against Firmware_Unified `v3_worktrees/fw-refresh-modes` (`docs/REFRESH_MODES.md`,
`shared/core/od_epd_lut.c`, `od_epd.c` `od_epd_validate`, `od_epd_cfg.c` `inline_feed` /
`io_lut_info`, `od_cmd_drive.c`). `uv run pytest`: **2249 passed** (2168 before; 81 new, all
in `tests/unit/test_refresh_modes.py`). `prek` passes. No hardware was touched. Test data:
`tests/unit/data/refresh_modes/` holds copies of `panels/waves/*.bin` and
`panels/generated/refresh_modes.json`.

### API

| What | Where |
|---|---|
| `build_drive_write_start(mode, refresh, *, uncompressed_size=None, first_chunk=b"", inline_len=0)`; `DRIVE_WRITE_FLAG_INLINE_LUT = 0x04` | `protocol/commands.py` |
| `build_inline_lut_prefix(tables: Mapping[int, bytes], *, lut_family=0) -> bytes` | `protocol/commands.py` |
| `drive_prefix(mode, *, inline_luts=None, lut_family=0, stored_luts=()) -> bytes` (checks + prefix) | `device/drive_write.py` |
| `drive_write(uploader, channel, mode, frames, *, stream=None, refresh=FULL, progress_callback=None, inline_luts=None, lut_family=0, stored_luts=())` | `device/drive_write.py` |
| `Device.display.show(..., inline_luts: Mapping[int, bytes] \| None = None)` (reads `lut_family` from the 0xB1 tail and the 0xAF slots from the cached config) | `device/api.py` |
| `lut_lint(family, table) -> str \| None`, `never_send(family, cmd)`, `sequence_lint(family, seq) -> str \| None`, `iter_sequence`, `sequence_uploads_lut`, `stored_slot_problem`, `mode_problem(mode, *, lut_family, stored_luts=(), inline_luts=None) -> str \| None`, `jd_value` / `jd_encode` | `models/lut_lint.py` |
| `DriveState`, `DriveStateStore` (protocol: `get` / `put` / `forget`), `MemoryDriveStateStore`, `JsonFileDriveStateStore(path="~/.config/opendisplay/drive_state.json")`, `drive_fingerprint(config) -> str` | `panels/drive_state.py` |
| `plan_drive(config, state, temp_c, requested: int \| str, *, include_experimental=False, instance=None, hysteresis_c=1.0) -> DrivePlan`; `record_drive(config, state, mode_index, temp_c, *, instance=None, now=None) -> DriveState`; `DrivePlan`, `PlannedDrive`, `mode_band`, `mode_cost` | `panels/drive_plan.py` |
| `assemble_subset_s6(table, codes_present, hold_code=4, border_code=1) -> bytes` | `models/lut_assemble.py` |
| `lut_crc16(data)` (public CCITT-FALSE) | `models/panel_drive.py` |

None of it is exported from `opendisplay` / `opendisplay.panels` yet (the public API snapshot is
unchanged); import from the modules.

### INLINE_LUT sender

START flag 0x04; the two reserved bytes are `inline_len` (u16 LE). The stream is
`prefix + frames`, prefix = `[slot][len:2][crc16:2][raw table]` per table in **ascending slot
order**. Uncompressed: the prefix goes first over 0x71. Compressed: the caller compresses
`prefix + frames` as one zlib stream (`show()` does) and `uncompressed_size` counts the prefix.
`drive_write` cannot check that a caller-made stream contains the prefix (it does no CPU-heavy
work); that is documented on the parameter. Checked before any I/O (ValueError): slot 0-15,
unique (mapping keys), 1-1024 B (the device refuses `len == 0`), the family lint, prefix fits a
u16; plus `mode_problem` for the mode (below).

### Device-equivalent lint

Ported from the C, not from the brief, where they differ:

- **JD block rule**: the first row among 0-6 whose `M == 0` **or** whose timing is 0 ends the
  block; it is an error only when that row has timing 0 and `M != 0`. So a row with `M == 0`
  and non-zero timing also ends the block (and later rows are not checked). The brief said
  "the first row whose timing is zero must have M == 0", which would also check rows after an
  `M == 0` row.
- **x4 (family 3) runs the same JD block check**, not only the size check (the C feeds both
  JD families through it). All three shipped JD tables pass as x4 too.
- **S6 8-zero run** counts across the whole table, header included.
- **Family 0 and unknown families** (>= 4): only `len <= 1024`, and no never-send list.
- **Never-send** (family != 0): 0x90, 0x91, 0xA0-0xA3, 0x8F, 0xFB, 0xFF, seen in a plain step's
  first byte, WRITE_SCRATCH's cmd, and a SUBST/WINDOW step's first pair when it is a literal
  (kind 0). FILL's cmd is not scanned (the C does not either). 0x4D / 0xB6 stay allowed.
- **0x40 after a LUT / LUT_SLICE escape before the next 0x12** is refused (family != 0).
- **Slot availability** (`mode_problem`, family != 0): for every pass of a mode whose sequence
  uploads a table, the pass's slot must be inline, or a stored 0xAF of that instance with
  reserved bits 4-6 clear, data 1-1024 B, good CRC, inflating to a table that lints. A pass with
  slot 0xFF under a LUT escape is refused, as on the device.
- Structural sequence errors (step past the end, unknown escape, SUBST/WINDOW count 0 or > 32)
  are reported for every family. Target-specific refusals (WINDOW/FILL on BG22) are not modelled.

Every generated mode of the three panels passes; `hosttable` fails without an inline slot 5
and passes with one.

### Drive state store

JSON document `{"format": 1, "devices": {name: record}}`, keyed by advertised name (as the key
store), file mode 0600, atomic replace. Record: `fingerprint`, `budget_used`, `last_mode`,
`last_band`, `last_temp_c`, `updated`. **Decisions:** the fingerprint is SHA-256 (first 16 B,
hex) over the 0xAE, 0xAF, 0xB1 and 0xB3 packets in stored order; **0xB0 is left out** (a new
measured palette does not change the drive). Unlike the key store, an unreadable or
unknown-format file reads as empty and is replaced on the next `put` (nothing secret in it,
and lost state only costs a stock full).

### Planner decisions

- **Band** = an experimental mode with an ADVISORY policy (0), a window that is not `(0, 0)`
  (how the generator writes "no window"), and a name ending in one digit; the family is the
  name without that digit. `partial6` is banded (it has a window) though it has no sibling.
  EE02 `fast` has no digit, so it is not banded.
- **Band edges** sit half a degree past the integer window ends (`[-40, 24]` / `[25, 60]` meet at
  24.5 C, matching the "band <= 24.5 C" notes). `near_edge` = within `hysteresis_c` (1 C) of the
  chosen window's edge, either edge. Inside the hysteresis on the far side, the requested mode
  is kept. Unknown temperature: requested kept, `near_edge`. A temperature outside every band:
  requested kept, `near_edge`, note.
- **Costs**: mode tail `budget_cost`; **a mode without a tail (e.g. 2.9in `colour16`) costs 0 and
  does not reset**; mode 0 always resets; `budget_cost == 0` resets (pll08 / pll07 / pll04).
- **Stock full first** when the state is missing or its fingerprint differs, when the band
  differs from `last_band`, or when `budget_used` plus the run's cost would exceed `budget_limit`
  (0xFF or no tail = unlimited; a limit of 0 forces a full before every costed update). It is
  skipped when the first drive of the run itself resets (mode 0, a cost-0 mode, or a cost-0 scrub).
- **`last_band`** is set by a banded drive, kept by an unbanded costed drive (e.g. `hosttable`),
  and cleared by any resetting drive. So `fast6` -> `hosttable` -> `fast7` still gets a full.
- **`scrub_before`** mode runs right before the chosen mode (after the stock full if one is
  scheduled); it must exist, and an experimental scrub also needs the opt-in.
- Experimental refusal covers the requested mode, the band sibling actually chosen, and the scrub.
- The caller drives `plan.steps` in order and calls `record_drive` after each with the ACK's
  mode and `temp_c`. Nothing calls the planner yet (no CLI, `show()` does not plan).

### S6 subset tables

`assemble_subset_s6` keeps bytes 0-95 and the slots of `codes_present`, hold (4) and border (1),
and puts a verbatim copy of the hold slot in every other slot. Charge per remaining slot is
unchanged (slots are copied byte for byte, the hold slot already belongs to the source table),
so the source table's charge gate still holds. Refresh length follows the remaining blocks. The
result is re-linted (copying can join zero runs across a slot boundary; there is a test for
that) and raises if it fails. Only 800-B tables are accepted (799 B is refused).

### For Jonas / the firmware side

1. **Inline table count vs the device's LUT cache.** Inline tables take RAM cache entries
   (`OD_EPD_LUT_CACHE`, default 2) and `od_lut_cache_claim` only evicts non-inline entries. More
   inline tables than cache entries are refused by the device, and two inline tables leave no
   entry for inflating a compressed stored slot another pass needs. The host cannot query the
   cache size, so it does not limit the count. Worth a capability bit or a documented floor.
2. **BG22 / Telink**: compressed stored slots read as absent and INLINE_LUT is NACKed
   (`OD_ERR_DRIVE_UNSUPPORTED`). The host lint cannot see the target here and does not model it.
3. **Lint refusals are `OD_ERR_DRIVE_SEQ_UNSUPPORTED` (0x07)** on the wire, so a host that skipped
   the local check sees `DriveSequenceUnsupportedError`, the same as an unknown escape.
4. **The modes without a tail** (stock `full`, `colour16`) are budget-neutral by my decision; if
   `colour16` should cost (or reset), it needs a tail.
