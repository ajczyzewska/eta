#!/bin/bash
# deploy.sh — Push generated audiobook to mydevil server
#
# Usage:
#   bash deploy.sh <audio-dir> --name <slug> --title <title> --author <author> [options]
#   bash deploy.sh <audio-dir> --name <slug> --title <title> --author <author> --dry-run
#
# Example:
#   bash deploy.sh generated_books/rod-debickich-t5_2026-03-23/audio \
#     --name rod-debickich-t5 --title "Praca i nadzieja" --author "Stanisław Modrzejewski" \
#     --series "Ród Dębickich" --tom 5

set -euo pipefail

# ─── SSH config (override via .env) ──────────────────────────
SSH_USER="eta"
SSH_HOST="mydevil.net"
REMOTE_AUDIOBOOKS_DIR="/srv/audiobooks"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    # shellcheck disable=SC1091
    set -o allexport
    source "$SCRIPT_DIR/.env"
    set +o allexport
fi

# ─── Defaults ─────────────────────────────────────────────────
AUDIO_DIR=""
BOOK_NAME=""
BOOK_TITLE=""
BOOK_AUTHOR=""
SERIES_NAME=""
TOM_NUM=""
DRY_RUN=false

# ─── Parse arguments ──────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: bash deploy.sh <audio-dir> --name <slug> --title <title> --author <author> [--series <series>] [--tom <N>] [--dry-run]"
    exit 1
fi

AUDIO_DIR="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name|--title|--author|--series|--tom)
            [[ $# -lt 2 ]] && { echo "Error: $1 requires a value"; exit 1; }
            case "$1" in
                --name)   BOOK_NAME="$2"   ;;
                --title)  BOOK_TITLE="$2"  ;;
                --author) BOOK_AUTHOR="$2" ;;
                --series) SERIES_NAME="$2" ;;
                --tom)    TOM_NUM="$2"     ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *)         echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Validate ─────────────────────────────────────────────────
if [[ -z "$AUDIO_DIR" ]]; then
    echo "Error: audio directory is required as first argument"
    exit 1
fi

if [[ ! -d "$AUDIO_DIR" ]]; then
    echo "Error: audio directory not found: $AUDIO_DIR"
    exit 1
fi

if [[ -z "$BOOK_NAME" ]]; then
    echo "Error: --name (slug) is required"
    exit 1
fi

# Slug must be safe for use as a remote path component
if ! echo "$BOOK_NAME" | grep -qE '^[a-z0-9][a-z0-9_-]*$'; then
    echo "Error: --name must be a lowercase alphanumeric slug (letters, digits, hyphens, underscores)"
    exit 1
fi

if [[ -z "$BOOK_TITLE" ]]; then
    echo "Error: --title is required"
    exit 1
fi

if [[ -z "$BOOK_AUTHOR" ]]; then
    echo "Error: --author is required"
    exit 1
fi

if [[ -n "$TOM_NUM" ]] && ! echo "$TOM_NUM" | grep -qE '^[1-9][0-9]*$'; then
    echo "Error: --tom must be a positive integer (got: '$TOM_NUM')"
    exit 1
fi

AUDIO_DIR="$(cd "$AUDIO_DIR" && pwd)"
CHAPTERS_JSON="$(dirname "$AUDIO_DIR")/chapters.json"

if [[ ! -f "$CHAPTERS_JSON" ]]; then
    echo "Error: chapters.json not found at: $CHAPTERS_JSON"
    exit 1
fi

MP3_COUNT=$(find "$AUDIO_DIR" -maxdepth 1 -name "*.mp3" | wc -l | tr -d ' ')
if [[ "$MP3_COUNT" -eq 0 ]]; then
    echo "Error: no MP3 files found in $AUDIO_DIR"
    exit 1
fi

echo "======================================"
echo "  deploy.sh — Audiobook Deploy"
echo "======================================"
echo "  Slug:   $BOOK_NAME"
echo "  Title:  $BOOK_TITLE"
echo "  Author: $BOOK_AUTHOR"
[[ -n "$SERIES_NAME" ]] && echo "  Series: $SERIES_NAME"
[[ -n "$TOM_NUM"     ]] && echo "  Tom:    $TOM_NUM"
echo "  MP3s:   $MP3_COUNT"
echo "  Remote: $SSH_USER@$SSH_HOST:$REMOTE_AUDIOBOOKS_DIR"
[[ "$DRY_RUN" == true ]] && echo "  Mode:   DRY RUN"
echo "======================================"
echo ""

# ─── Check remote disk space ──────────────────────────────────
echo ">>> Checking remote disk space..."
if [[ "$DRY_RUN" == false ]]; then
    if ! REMOTE_FREE_KB=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_USER@$SSH_HOST" \
            "df -k '$REMOTE_AUDIOBOOKS_DIR' 2>/dev/null | awk 'NR==2{print \$4}'" 2>/dev/null); then
        echo "WARNING: could not connect to $SSH_USER@$SSH_HOST to check disk space."
        echo "Proceed anyway? [y/N]"
        read -r CONFIRM || true
        if [ "$(echo "$CONFIRM" | tr '[:upper:]' '[:lower:]')" != "y" ]; then
            echo "Aborted."
            exit 1
        fi
        REMOTE_FREE_KB="0"
    fi
    if ! echo "$REMOTE_FREE_KB" | grep -qE '^[0-9]+$'; then
        echo "WARNING: could not parse disk space from remote (got: '$REMOTE_FREE_KB'). Proceeding."
        REMOTE_FREE_KB="9999999"
    fi
    REMOTE_FREE_MB=$((REMOTE_FREE_KB / 1024))
    if [[ "$REMOTE_FREE_MB" -lt 2048 ]]; then
        echo "WARNING: only ${REMOTE_FREE_MB}MB free on remote server (< 2GB). Proceed anyway? [y/N]"
        read -r CONFIRM || true
        if [ "$(echo "$CONFIRM" | tr '[:upper:]' '[:lower:]')" != "y" ]; then
            echo "Aborted."
            exit 1
        fi
    else
        echo "  Remote free: ${REMOTE_FREE_MB}MB — OK"
    fi
else
    echo "  [dry-run] Skipping disk space check"
fi
echo ""

# ─── Build meta.json via Python ───────────────────────────────
# Python handles: MP3 enumeration, chapter title mapping, JSON serialisation
echo ">>> Building meta.json..."
META_JSON=$(python3 - "$AUDIO_DIR" "$CHAPTERS_JSON" "$BOOK_NAME" "$BOOK_TITLE" "$BOOK_AUTHOR" "$SERIES_NAME" "$TOM_NUM" <<'PYEOF'
import json, os, sys, re
from datetime import datetime, timezone

audio_dir, chapters_json_path, slug, title, author, series, tom = sys.argv[1:]

with open(chapters_json_path, encoding='utf-8') as f:
    chapters_data = json.load(f)

chapter_titles = [ch.get('title', '') for ch in chapters_data]

# Find and sort MP3s by R-index
mp3_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

def r_index(filename):
    m = re.match(r'^R(\d+)_', filename)
    return int(m.group(1)) if m else -1

mp3_files = sorted([f for f in mp3_files if r_index(f) >= 0], key=r_index)

chapters = []
for mp3 in mp3_files:
    idx = r_index(mp3)
    if idx == 0:
        label = 'Wprowadzenie'
    else:
        ch_idx = idx - 1
        label = chapter_titles[ch_idx] if ch_idx < len(chapter_titles) else f'Rozdział {idx}'
    chapters.append({'index': idx, 'filename': mp3, 'label': label})

meta = {
    'slug': slug,
    'title': title,
    'author': author,
    'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'chapters': chapters,
}
if series:
    meta['series'] = series
if tom:
    try:
        meta['tom'] = int(tom)
    except ValueError:
        meta['tom'] = tom

print(json.dumps(meta, ensure_ascii=False, indent=2))
PYEOF
) || { echo "Error: failed to build meta.json"; exit 1; }

META_JSON_PATH="$AUDIO_DIR/meta.json"

if [[ "$DRY_RUN" == false ]]; then
    printf '%s\n' "$META_JSON" > "$META_JSON_PATH"
    echo "  Written: $META_JSON_PATH"
else
    echo "  [dry-run] Would write: $META_JSON_PATH"
    echo "  Content preview:"
    { printf '%s\n' "$META_JSON" | head -30; } || true
fi
echo ""

# ─── Build manifest.json from all local meta.json files ───────
echo ">>> Building manifest.json..."
MANIFEST_PATH="$SCRIPT_DIR/manifest.json"

MANIFEST_JSON=$(SCRIPT_DIR="$SCRIPT_DIR" python3 - <<'PYEOF'
import json, os, sys

script_dir = os.environ['SCRIPT_DIR']
generated_books_dir = os.path.join(script_dir, 'generated_books')
books = []
seen_slugs = set()

if os.path.isdir(generated_books_dir):
    for book_dir in sorted(os.listdir(generated_books_dir)):
        audio_dir = os.path.join(generated_books_dir, book_dir, 'audio')
        meta_path = os.path.join(audio_dir, 'meta.json')
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, encoding='utf-8') as f:
                    meta = json.load(f)
                slug = meta.get('slug', '')
                if slug not in seen_slugs:
                    books.append(meta)
                    seen_slugs.add(slug)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: skipping {meta_path}: {e}", file=sys.stderr)

books.sort(key=lambda b: b.get('generated_at', ''))
print(json.dumps({'books': books}, ensure_ascii=False, indent=2))
PYEOF
) || { echo "Error: failed to build manifest.json"; exit 1; }

BOOK_COUNT=$(printf '%s\n' "$MANIFEST_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['books']))") \
    || { echo "Error: failed to parse manifest.json"; exit 1; }

if [[ "$DRY_RUN" == false ]]; then
    printf '%s\n' "$MANIFEST_JSON" > "$MANIFEST_PATH"
    echo "  Written: $MANIFEST_PATH ($BOOK_COUNT book(s))"
else
    echo "  [dry-run] Would write: $MANIFEST_PATH"
    echo "  Would include $BOOK_COUNT existing book(s) in manifest (current book not counted — meta.json not written in dry-run)"
fi
echo ""

# ─── Rsync MP3s + meta.json to server ─────────────────────────
REMOTE_BOOK_DIR="$REMOTE_AUDIOBOOKS_DIR/$BOOK_NAME"

if [[ "$DRY_RUN" == true ]]; then
    echo ">>> [dry-run] Would sync audio files to: $SSH_USER@$SSH_HOST:$REMOTE_BOOK_DIR/"
    echo "  Files that would be transferred:"
    find "$AUDIO_DIR" -maxdepth 1 \( -name "*.mp3" -o -name "meta.json" \) \
        | sort | while IFS= read -r f; do
        echo "    $(basename "$f")"
    done || true
    echo ""
    echo ">>> [dry-run] Would sync manifest.json to: $SSH_USER@$SSH_HOST:$REMOTE_AUDIOBOOKS_DIR/manifest.json"
    echo ""
else
    echo ">>> Syncing audio files to server..."
    rsync -av --progress \
        --include="*.mp3" \
        --include="meta.json" \
        --exclude="*" \
        "$AUDIO_DIR/" \
        "$SSH_USER@$SSH_HOST:$REMOTE_BOOK_DIR/" \
        || { echo "Error: failed to sync audio files to $SSH_USER@$SSH_HOST:$REMOTE_BOOK_DIR/"; exit 1; }
    echo ""

    echo ">>> Syncing manifest.json..."
    rsync -av --progress \
        "$MANIFEST_PATH" \
        "$SSH_USER@$SSH_HOST:$REMOTE_AUDIOBOOKS_DIR/manifest.json" \
        || { echo "Error: manifest.json sync failed — book files were uploaded but manifest not updated. Re-run deploy to fix."; exit 1; }
    echo ""
fi

echo "======================================"
if [[ "$DRY_RUN" == true ]]; then
    echo "  Dry run complete — no files transferred"
else
    echo "  Deploy complete!"
    echo "  Book available at: https://$SSH_HOST/audiobooks/$BOOK_NAME/"
fi
echo "======================================"
