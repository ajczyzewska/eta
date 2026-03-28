# Implementation Progress

## Phase 1: `deploy.sh` — Complete

**Date:** 2026-03-27
**Branch:** `main`

### What was created/changed

| File | Purpose |
|------|---------|
| [deploy.sh](deploy.sh) | New script: builds `meta.json` + `manifest.json` locally and rsyncs audiobook to mydevil server |

### Key design decisions

- **Python for JSON building**: MP3 enumeration, chapter-title mapping, and all JSON serialisation delegated to embedded Python heredocs — avoids bash string-escaping pitfalls with Unicode titles/authors.
- **bash 3.2 compatibility**: No `mapfile`, no `${var,,}`, no associative arrays — required because macOS ships bash 3.2.
- **Dry-run skips rsync**: `--dry-run` lists files that would be transferred instead of attempting SSH — avoids needing live server connectivity for verification.
- **Manifest from filesystem scan**: `manifest.json` is rebuilt by scanning all `generated_books/*/audio/meta.json` files locally before upload — no server-side script required.
- **Chapter mapping**: R0 → "Wprowadzenie", R1..RN → `chapters.json[0..N-1]`. Ordering follows `chapters.json` (which matches the TTS pipeline's generation order).

### Review process

Three iterations of dual-agent review (senior architect + JavaScript/shell expert):

- **Iteration 1:** 2 BLOCKER, 3 HIGH, 3 MEDIUM, 2 LOW. Fixed: bash 3.2 `${CONFIRM,,}` bug, Python failure guards, slug validation, `echo` → `printf`, SSH failure handling, dead arg removed.
- **Iteration 2:** 1 BLOCKER, 2 HIGH, 4 MEDIUM, 2 LOW. Fixed: `read -r CONFIRM || true`, rsync `||` error guards, `printf|head` pipefail guard, dry-run listing guard, `--tom` integer validation, SSH ConnectTimeout.
- **Iteration 3:** 0 BLOCKER, 0 HIGH, 1 MEDIUM, 3 LOW. Fixed: `shift 2` missing-value guard, `--tom 0` rejected.

### Test results

**18 passing, 0 failing** — baseline fully preserved (no test files modified; deploy.sh has no unit tests, verified via dry-run on existing book data)

---

## Phase 2: `generatebook.sh` — add `--deploy` flag — Complete

**Date:** 2026-03-27
**Branch:** `main`

### What was created/changed

| File | Purpose |
|------|---------|
| [generatebook.sh](generatebook.sh) | Added `--deploy` flag: calls `deploy.sh` after Phase 3 TTS pipeline completes |

### Key design decisions

- **Defaults initialized at top**: `SERIES_NAME=""` and `TOM_NUM=""` added to defaults block — required because `set -eu` would fail if these were referenced (in the deploy call) without prior assignment; they were previously only set inside conditional auto-detection blocks.
- **TTS_ARGS extraction fallback**: If user passed `--tom`/`--series` manually (routing to TTS_ARGS), auto-detection is skipped and those script-level variables stay empty. Added a loop to extract them from TTS_ARGS before the deploy call so `meta.json` always contains the correct series/tom.
- **`DEPLOY_ARGS` array**: Optional args (`--series`, `--tom`) are conditionally added to a `DEPLOY_ARGS=()` array; empty array expansion with `set -eu` is safe for explicitly-initialized indexed arrays (same pattern as existing `TTS_ARGS`).
- **`BOOK_TITLE` at deploy time**: By the time deploy runs, `BOOK_TITLE` has been overwritten to the subtitle (if extracted) — this is correct; the short title is what deploy.sh stores in `meta.json`.

### Review process

Three iterations of dual-agent review (senior architect + shell expert):

- **Iteration 1:** 0 BLOCKER, 1 HIGH, 0 MEDIUM. Fixed: when user passes `--tom`/`--series` via TTS args, those values were not reaching deploy.sh — added TTS_ARGS extraction fallback loop.
- **Iteration 2:** 0 BLOCKER, 0 HIGH. Verified: empty `DEPLOY_ARGS` expansion safe under `set -eu`; `${TTS_ARGS[$next_i]:-}` guards out-of-bounds; `BOOK_TITLE` subtitle reassignment ordering correct.
- **Iteration 3:** Clean — no new findings.

### Test results

**Syntax check: pass** (`bash -n generatebook.sh` — no errors). No project-level unit test runner; no tests modified.

---

---

## Phase 3: nginx config (2 location blocks) — Complete

**Date:** 2026-03-28
**Branch:** `main`

### What was created/changed

| File | Purpose |
|------|---------|
| [nginx/audiobooks.conf](nginx/audiobooks.conf) | nginx location blocks to paste into the mydevil server block; serves `/audiobooks/` and `/player/` |

### Key design decisions

- **Flattened top-level locations (not nested):** nginx does not propagate `alias` into nested `location` blocks — each block that serves files must declare its own `alias`. Three separate top-level location blocks replace the originally-planned nested structure.
- **Regex + capture group for MP3s:** `location ~* ^/audiobooks/([a-z0-9_-]+/[^/]+\.mp3)$` with `alias /srv/audiobooks/$1;` — capture group required for correct path rewriting when using `alias` with a regex location. Character class restricts to slug/filename pattern (blocks path traversal).
- **Exact match for manifest.json:** `location = /audiobooks/manifest.json` (highest nginx priority) with `alias /srv/audiobooks/manifest.json;` — ensures the exact-match rule always wins over the prefix block.
- **`try_files $uri /player/index.html =404`**: dropped `$uri/` — with `autoindex off`, directory hits return 403 instead of falling through to SPA fallback. Two-path form is correct for a SPA with no real subdirectories.
- **`immutable` caching on MP3s:** safe because slugs are write-once (validated in `deploy.sh`); the same URL will never serve different content.

### Review process

Three iterations of dual-agent review (senior architect + web/security expert):

- **Iteration 1:** 1 BLOCKER (alias not re-declared in nested locations — files would 404), 1 HIGH (try_files `=404` terminal — misread as dead code, verified false positive). Fixed: flattened to top-level locations with own `alias` per block.
- **Iteration 2:** 1 HIGH (regex capture `.+` allows path traversal — restricted to `[a-z0-9_-]+/[^/]+`). `try_files =404` re-verified as correct nginx semantics, not dead code.
- **Iteration 3:** 1 MEDIUM (`$uri/` in try_files returns 403 on directory hits with `autoindex off` — dropped). Clean otherwise.

### Test results

**No automated tests** — nginx config is infrastructure-only. File verified by reading final state; logic verified against nginx documentation for alias+regex, location priority, and try_files semantics.

---

## Phase 4: PWA (`pwa/`) — Vite + vanilla TypeScript player — Complete

**Date:** 2026-03-28
**Branch:** `main`

### What was created/changed

| File | Purpose |
|------|---------|
| [pwa/package.json](pwa/package.json) | Vite + TypeScript deps; `build`, `dev`, `preview`, `deploy` scripts |
| [pwa/tsconfig.json](pwa/tsconfig.json) | Strict TypeScript config targeting ES2020 with bundler module resolution |
| [pwa/vite.config.ts](pwa/vite.config.ts) | Vite config with `base: '/player/'` for nginx serving |
| [pwa/manifest.webmanifest](pwa/manifest.webmanifest) | PWA manifest: standalone display, dark theme colours |
| [pwa/index.html](pwa/index.html) | Full-viewport tablet UI: now-playing, progress bar, transport controls, book/chapter lists; all CSS inline |
| [pwa/src/player.ts](pwa/src/player.ts) | Audio engine: `Player` class, shared types (`Book`, `Chapter`, `Manifest`, `PlayerState`, `ResumeState`), localStorage resume, auto-advance |
| [pwa/src/mediaSession.ts](pwa/src/mediaSession.ts) | Media Session API: `setupMediaSession` (handlers registered once) + `updateMediaMetadata` (per chapter) |
| [pwa/src/main.ts](pwa/src/main.ts) | Bootstrap: fetch manifest, wire UI, player state → DOM, seek slider, resume on load |

### Key design decisions

- **Types in `player.ts`, re-exported**: avoids a separate `types.ts` file; `main.ts` and `mediaSession.ts` import from `./player` — no circular deps.
- **Media Session handlers registered once**: `setupMediaSession` is called once at init; only metadata is updated per chapter change via `updateMediaMetadata`. Avoids re-registering handlers on every `timeupdate`.
- **Chapter highlight without full re-render**: `onStateChange` tracks `lastChapterIndex + lastBookSlug`; only calls `updateChapterHighlight()` (updates classList) when chapter changes — avoids rebuilding the entire chapter list DOM on every second of playback.
- **Seek slider uses pointerdown/pointerup/pointercancel**: `isSeeking` flag prevents `onStateChange` from overwriting the slider position during drag; `pointercancel` prevents the flag getting stuck `true` on Android OS interruptions (permission dialogs etc.).
- **`error: boolean` in `PlayerState`**: audio `error` event sets `_hasError = true` (cleared on chapter load); `onStateChange` shows "Błąd odtwarzania" in the spinner element, distinct from the buffering state.
- **Resume bounds validation**: stored `chapterIndex` is compared to actual chapter count (book may have been updated); stored `currentTime` guarded `>= 0`; invalid chapter resets to chapter 0, time 0.
- **`base: '/player/'` in Vite**: all built asset paths are prefixed `/player/` matching nginx `location /player/` block.

### Review process

Three iterations of dual-agent review (senior architect + TypeScript/browser expert):

- **Iteration 1:** 1 BLOCKER (pointercancel missing — `isSeeking` permanently stuck on Android), 1 HIGH (audio `error` event not handled — silent playback failure), 2 MEDIUM (stale resume chapterIndex not validated, empty books list silent fail). Fixed all 4. Discarded 4 false positives (textContent XSS, empty chapters math, playPrev bounds, event listener memory).
- **Iteration 2:** 1 HIGH (no `error` field in `PlayerState` — UI indistinguishable from buffering after error), 1 LOW (resume.currentTime not guarded for negative/NaN). Fixed both. Discarded 2 false positives (pointercancel race with input harmless in practice, manifest as-cast acceptable for same-origin trusted server).
- **Iteration 3:** Clean — 1 false positive discarded (safeTime logic flagged as wrong but is provably correct: resetting time to 0 when chapter no longer exists is intentional).

### Test results

**18 passing, 0 failing** — Python baseline fully preserved (PWA has no unit tests; correctness verified via `tsc --noEmit` + `vite build` clean build, 7.72 kB gzip bundle)
