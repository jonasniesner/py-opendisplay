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
- The host drive-state store (budget count, band, priors), band picking from `temp_c`.
- `display_tracks`, `INLINE_LUT`, the table assembler, and vendoring the lint/charge gate from
  Firmware_Unified `presets/tools/`.
- Device-side: nothing here is checked against firmware. The layouts above are what the
  firmware test implementation should parse.
