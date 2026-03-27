# Audiobook Server Deploy + Tablet PWA Player — Implementation Plan

## Context

Replace the manual workflow (generate locally → Google Drive → download on tablet → add to playlist) with:
- Auto-push generated MP3s to mydevil server via SSH after generation
- Tablet-facing PWA on the same server that dynamically loads new books and plays them

**Server**: mydevil.net (Linux, nginx already running, SSH access)
**Tablet**: Android (Chrome, Media Session API fully supported)
**Frontend**: Hosted on same nginx server — no Firebase needed, same origin = no CORS

---

## Architecture

```
[generatebook.sh]
       |
  [deploy.sh] ──rsync over SSH──> [mydevil /srv/audiobooks/{slug}/]
       |                                  |
  generates manifest.json           nginx (HTTPS, same server)
  + meta.json locally               ├── /audiobooks/manifest.json
  before rsync                      ├── /audiobooks/{slug}/R*.mp3
                                    └── /player/  (PWA static files)
                                         |
                                  [Android Chrome Tablet]
                                  Media Session API (headphone controls)
```

---

## Components

### 1. `deploy.sh` (new file)

- Accepts audio directory path as argument
- Reads `../chapters.json` (sibling of audio dir) to get sorted chapter titles
- Writes `meta.json` into the audio dir:
  ```json
  {
    "slug": "rod-debickich-t5",
    "title": "Praca i nadzieja",
    "author": "Stanisław Modrzejewski",
    "series": "Ród Dębickich",
    "tom": 5,
    "generated_at": "ISO timestamp",
    "chapters": [
      { "index": 0, "filename": "R0_...mp3", "label": "Wprowadzenie" },
      { "index": 1, "filename": "R1_...mp3", "label": "Rozdział 1: Cień szabli" }
    ]
  }
  ```
- Scans all `meta.json` files in local audio output directories to build `manifest.json` locally
- Writes `manifest.json` locally, then rsyncs it along with the new book's MP3s + `meta.json`
- No server-side script needed — manifest is computed before upload
- Supports `--dry-run` flag
- SSH host/user configured at top of script (or via `.env`)

### 2. `generatebook.sh` changes — small

- Add `--deploy` flag to argument parser
- At end of Phase 3, call `./deploy.sh "$OUTPUT_DIR/audio"` if `--deploy` is set
- All needed variables already in scope: `OUTPUT_DIR`, `BOOK_TITLE`, `BOOK_AUTHOR`, `SERIES_NAME`, `TOM_NUM`, `BOOK_NAME`

### 3. nginx config additions — 2 location blocks

```nginx
location /audiobooks/ {
    alias /srv/audiobooks/;
    location ~* \.mp3$ {
        add_header Cache-Control "public, max-age=31536000, immutable";
        gzip off;
    }
    location = /audiobooks/manifest.json {
        add_header Cache-Control "no-cache";
    }
}

location /player/ {
    alias /srv/player/;
    try_files $uri $uri/ /player/index.html;
}
```

### 4. Frontend PWA: `pwa/` — Vite + vanilla TypeScript

**Stack**: Vite + vanilla TypeScript. No React — single-purpose player, keeps bundle tiny.

**File structure**:
```
pwa/
  package.json
  tsconfig.json
  vite.config.ts
  index.html
  manifest.webmanifest
  src/
    main.ts          ← bootstrap, fetch manifest, render UI
    player.ts        ← audio element, queue, seek, auto-advance, localStorage resume
    mediaSession.ts  ← Media Session API integration
```

**Deploy**: `npm run build` → `rsync dist/ user@mydevil:/srv/player/`

#### Player (`player.ts`)
- Single `<audio>` element with `preload="auto"` — Chrome buffers aggressively, handles brief WiFi drops
- Auto-advance: `audio.addEventListener('ended', playNext)`
- Resume position persisted to `localStorage` (`{ slug, chapterIndex, currentTime }`)

#### Tablet UI
- Full-viewport, no scroll needed, minimum 80px tap targets
- Layout:
  ```
  ┌─────────────────────────────────────┐
  │  Ród Dębickich — Tom 5              │
  │  Rozdział 3: Żelazny koń            │
  │                                     │
  │  ████████████░░░░░░░░░  12:34/38:00 │
  │                                     │
  │  [⏮ 30s]  [⏸ PAUSE]  [30s ⏭]      │
  │  [◀ PREV CHAPTER]  [NEXT CHAPTER ▶] │
  │                                     │
  │  Books ↕            Chapters ↕      │
  │  • Rod Dębickich T5 • Rozdział 1    │
  │    Rod Dębickich T6 │ Rozdział 2    │
  │                     │▶Rozdział 3    │
  └─────────────────────────────────────┘
  ```
- Loading spinner on `waiting` event, clears on `canplaythrough`

#### Media Session API (`mediaSession.ts`)
```typescript
navigator.mediaSession.metadata = new MediaMetadata({
  title: chapter.label,
  artist: book.author,
  album: `${book.series} — Tom ${book.tom}`,
});
navigator.mediaSession.setActionHandler('play', () => audio.play());
navigator.mediaSession.setActionHandler('pause', () => audio.pause());
navigator.mediaSession.setActionHandler('nexttrack', playNext);
navigator.mediaSession.setActionHandler('previoustrack', playPrev);
navigator.mediaSession.setActionHandler('seekforward', () => audio.currentTime += 30);
navigator.mediaSession.setActionHandler('seekbackward', () => audio.currentTime -= 30);
```

Android sends a `play` action to Media Session when headphones are plugged in (if this was the last active media source). Works after the first manual tap — first play must be a user gesture per browser policy.

**Setup step**: PWA must be added to Android home screen ("Add to Home Screen") for full-screen mode and reliable Media Session behaviour.

---

## Key Risks

| Risk | Mitigation |
|------|-----------|
| Android autoplay policy | First play must be manual tap; headphone resume works via Media Session after that |
| rsync slow on home upload (~15 min) | rsync resumes if interrupted; safe to re-run |
| mydevil disk quota | ~1.1GB/book; script warns if server disk < 2GB free before sync |
| `chapters.json` sorted alphabetically | Deploy script sorts by numeric chapter index before writing `meta.json` |

---

## Effort Estimate

| Component | Hours |
|-----------|-------|
| `deploy.sh` + `generatebook.sh` integration | 2-3 |
| nginx config | 0.5 |
| PWA scaffold + Vite + deploy script | 1 |
| `player.ts` (audio, queue, resume) | 3-4 |
| Tablet UI (HTML/CSS) | 2-3 |
| Media Session API | 1 |
| **Total** | **~10-12 hrs** |

---

## Verification Steps

1. `./deploy.sh --dry-run generated_books/rod-debickich-t5_2026-03-23/audio` → verify rsync dry-run and `meta.json` + `manifest.json` content
2. Full deploy → `curl https://<domain>/audiobooks/manifest.json` → verify books + chapters listed
3. Open `https://<domain>/player/` on Android Chrome → tap book → verify playback
4. Press headphone play/pause → verify Media Session handles it
5. Close Chrome tab, reopen → verify resume at last position
6. Generate a new book with `--deploy` → verify it appears in PWA

---

## Manual / Device Tests (Android Chrome)

| Scenario | Pass criteria |
|----------|--------------|
| First open | App loads, newest book shown |
| Headphone plug-in | After first manual play, plugging headphones in resumes playback |
| Headphone button: play/pause | Single press toggles playback |
| Headphone button: next/prev | Next/prev chapter advances correctly |
| Resume after close | Close Chrome tab, reopen → resumes at last position |
| New book deploy | Run `generatebook.sh --deploy` → open PWA → new book appears |
