#!/bin/bash
# generatebook.sh — AI Book Generation → Audiobook Pipeline
#
# Generates a book using Claude Code agent (chapter by chapter),
# converts to chapters JSON, then runs the TTS pipeline.
#
# Usage:
#   sh generatebook.sh --name "moja-ksiazka"
#   sh generatebook.sh --name "moja-ksiazka" --prompts-dir ./custom_prompts/
#   sh generatebook.sh --name "moja-ksiazka" --skip-generate   # skip AI generation, use existing book.md
#   sh generatebook.sh --name "moja-ksiazka" --skip-convert     # skip md→json, use existing chapters.json
#   sh generatebook.sh --name "moja-ksiazka" --speaker voice.wav --chunk-size 250

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────
PROMPTS_DIR="./book_prompts"
BOOK_NAME=""
SKIP_GENERATE=false
SKIP_CONVERT=false
TTS_ARGS=()
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Parse arguments ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            BOOK_NAME="$2"; shift 2 ;;
        --prompts-dir)
            PROMPTS_DIR="$2"; shift 2 ;;
        --skip-generate)
            SKIP_GENERATE=true; shift ;;
        --skip-convert)
            SKIP_CONVERT=true; shift ;;
        --speaker|--output|--chunk-size|--crossfade|--speed|--pause-stretch|--optimize)
            TTS_ARGS+=("$1" "$2"); shift 2 ;;
        --postprocess|--resume|--verbose)
            TTS_ARGS+=("$1"); shift ;;
        -h|--help)
            echo "Usage: sh generatebook.sh --name <book-name> [options]"
            echo ""
            echo "Options:"
            echo "  --name <name>           Book name (required, used for output directory)"
            echo "  --prompts-dir <path>    Prompts directory (default: ./book_prompts/)"
            echo "  --skip-generate         Skip AI generation, use existing book.md"
            echo "  --skip-convert          Skip markdown→JSON conversion, use existing chapters.json"
            echo "  --speaker <wav>         Voice sample WAV file (passed to TTS)"
            echo "  --chunk-size <n>        TTS chunk size (passed to TTS)"
            echo "  --postprocess           Apply audio post-processing (passed to TTS)"
            echo "  -h, --help              Show this help"
            exit 0 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Validate ────────────────────────────────────────────────
if [[ -z "$BOOK_NAME" ]]; then
    echo "Error: --name is required"
    echo "Usage: sh generatebook.sh --name <book-name>"
    exit 1
fi

if ! command -v claude &> /dev/null; then
    echo "Error: 'claude' CLI not found. Install Claude Code: https://docs.anthropic.com/en/docs/claude-code"
    exit 1
fi

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)

if [[ ! -d "$PROMPTS_DIR" ]]; then
    echo "Error: Prompts directory not found: $PROMPTS_DIR"
    exit 1
fi

if [[ ! -f "$PROMPTS_DIR/rules.md" ]]; then
    echo "Error: rules.md not found in $PROMPTS_DIR"
    exit 1
fi

if [[ ! -f "$PROMPTS_DIR/book.md" ]]; then
    echo "Error: book.md not found in $PROMPTS_DIR"
    exit 1
fi

# ─── Setup output directory ──────────────────────────────────
DATE=$(date +%Y-%m-%d)
OUTPUT_DIR="generated_books/${BOOK_NAME}_${DATE}"
mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "  EtA — AI Book Generator"
echo "======================================"
echo "  Book: $BOOK_NAME"
echo "  Prompts: $PROMPTS_DIR"
echo "  Output: $OUTPUT_DIR"
echo "======================================"

# ─── Read outline to determine chapter count ─────────────────
# outline.md should have lines like: "## Rozdział 1: Tytuł" or "## Chapter 1: Title"
OUTLINE_FILE="$PROMPTS_DIR/outline.md"
if [[ -f "$OUTLINE_FILE" ]]; then
    CHAPTER_COUNT=$(grep -c '^## ' "$OUTLINE_FILE" || true)
else
    # Default chapter count from book.md — look for "Liczba rozdziałów:" line
    CHAPTER_COUNT=$(sed -n 's/.*Liczba rozdziałów:[[:space:]]*\([0-9]*\).*/\1/p' "$PROMPTS_DIR/book.md" | head -1)
    CHAPTER_COUNT=${CHAPTER_COUNT:-5}
fi

if [[ "$CHAPTER_COUNT" -lt 1 ]]; then
    echo "Error: Could not determine chapter count from outline.md or book.md"
    exit 1
fi
echo "  Chapters: $CHAPTER_COUNT"
echo ""

# ─── Helper: extract chapter outline for chapter N ───────────
get_chapter_outline() {
    local chapter_num=$1
    if [[ -f "$OUTLINE_FILE" ]]; then
        # Extract section between ## Chapter N and the next ## or EOF
        awk "/^## .*${chapter_num}[^0-9]/{found=1; next} /^## /{if(found) exit} found{print}" "$OUTLINE_FILE"
    fi
}

# ─── PHASE 1: Generate book text ─────────────────────────────
if [[ "$SKIP_GENERATE" == false ]]; then
    echo ">>> Phase 1: Generating book (chapter by chapter)..."
    echo ""

    RULES=$(cat "$PROMPTS_DIR/rules.md")
    BOOK_SPEC=$(cat "$PROMPTS_DIR/book.md")
    STYLE=""
    if [[ -f "$PROMPTS_DIR/style.md" ]]; then
        STYLE=$(cat "$PROMPTS_DIR/style.md")
    fi

    PREV_SUMMARY=""

    for ((i=1; i<=CHAPTER_COUNT; i++)); do
        CHAPTER_FILE="$OUTPUT_DIR/chapter_${i}.md"
        SUMMARY_FILE="$OUTPUT_DIR/summary_${i}.txt"
        REVIEW_FILE="$OUTPUT_DIR/review_${i}.txt"

        # Skip if chapter already exists (for resumability)
        if [[ -f "$CHAPTER_FILE" ]] && [[ -s "$CHAPTER_FILE" ]]; then
            echo "  [skip] Chapter $i already exists: $CHAPTER_FILE"
            if [[ -f "$SUMMARY_FILE" ]]; then
                PREV_SUMMARY=$(cat "$SUMMARY_FILE")
            fi
            continue
        fi

        echo "  [$i/$CHAPTER_COUNT] Generating chapter $i..."

        # Assemble prompt
        CHAPTER_OUTLINE=$(get_chapter_outline "$i")
        PROMPT="$RULES

---

$BOOK_SPEC"

        if [[ -n "$STYLE" ]]; then
            PROMPT="$PROMPT

---

$STYLE"
        fi

        PROMPT="$PROMPT

---

Napisz rozdział $i z $CHAPTER_COUNT."

        if [[ -n "$CHAPTER_OUTLINE" ]]; then
            PROMPT="$PROMPT

Wskazówki do tego rozdziału:
$CHAPTER_OUTLINE"
        fi

        if [[ -n "$PREV_SUMMARY" ]]; then
            PROMPT="$PROMPT

---

Streszczenie poprzedniego rozdziału (dla ciągłości narracji):
$PREV_SUMMARY"
        fi

        # Generate chapter
        claude -p "$PROMPT" --output-format text > "$CHAPTER_FILE" 2>/dev/null

        # ── Structural validation ──
        VALID=true
        if [[ ! -s "$CHAPTER_FILE" ]]; then
            echo "    [WARN] Chapter $i is empty!"
            VALID=false
        fi

        if ! grep -q '^# ' "$CHAPTER_FILE" 2>/dev/null; then
            echo "    [WARN] Chapter $i missing '# ' heading"
            VALID=false
        fi

        WORD_COUNT=$(wc -w < "$CHAPTER_FILE" 2>/dev/null | tr -d ' ')
        if [[ "$WORD_COUNT" -lt 500 ]]; then
            echo "    [WARN] Chapter $i has only $WORD_COUNT words (expected 500+)"
            VALID=false
        fi

        if grep -qiE '(TODO|insert here|\[\.\.\.\]|\[wstaw\])' "$CHAPTER_FILE" 2>/dev/null; then
            echo "    [WARN] Chapter $i contains placeholder text"
            VALID=false
        fi

        # ── Claude review pass ──
        echo "    Reviewing chapter $i..."
        REVIEW_PROMPT="Przeanalizuj poniższy rozdział książki i oceń:
1. Czy zawiera halucynacje lub fakty niespójne z treścią?
2. Czy są problemy z ciągłością narracji?
3. Czy zawiera placeholder lub meta-komentarze?
4. Czy jakość tekstu jest akceptowalna?

Odpowiedz JEDNYM SŁOWEM: PASS lub FAIL, a następnie krótkie uzasadnienie (max 3 zdania).

---

$(cat "$CHAPTER_FILE")"

        claude -p "$REVIEW_PROMPT" --output-format text > "$REVIEW_FILE" 2>/dev/null

        if grep -qi 'FAIL' "$REVIEW_FILE" 2>/dev/null; then
            echo "    [WARN] Review FAILED for chapter $i. See: $REVIEW_FILE"
        else
            echo "    [OK] Review passed for chapter $i"
        fi

        # ── Generate summary for next chapter ──
        echo "    Generating summary..."
        SUMMARY_PROMPT="Streść poniższy rozdział w maksymalnie 200 słowach. Skup się na kluczowych wydarzeniach, postaciach i wątkach fabularnych:

$(cat "$CHAPTER_FILE")"

        claude -p "$SUMMARY_PROMPT" --output-format text > "$SUMMARY_FILE" 2>/dev/null
        PREV_SUMMARY=$(cat "$SUMMARY_FILE")

        echo "    Done. ($WORD_COUNT words)"
        echo ""
    done

    # Concatenate all chapters into book.md
    echo "  Combining chapters into book.md..."
    cat "$OUTPUT_DIR"/chapter_*.md > "$OUTPUT_DIR/book.md"
    echo "  Book saved: $OUTPUT_DIR/book.md"
    echo ""
else
    echo ">>> Phase 1: Skipped (--skip-generate)"
    if [[ ! -f "$OUTPUT_DIR/book.md" ]]; then
        echo "Error: $OUTPUT_DIR/book.md not found. Cannot skip generation without existing book."
        exit 1
    fi
    echo ""
fi

# ─── PHASE 2: Convert markdown → JSON ────────────────────────
CHAPTERS_JSON="$OUTPUT_DIR/chapters.json"

if [[ "$SKIP_CONVERT" == false ]]; then
    echo ">>> Phase 2: Converting markdown to chapters JSON..."
    $PYTHON "$SCRIPT_DIR/md_to_chapters.py" "$OUTPUT_DIR/book.md" --output "$CHAPTERS_JSON"
    echo ""
else
    echo ">>> Phase 2: Skipped (--skip-convert)"
    if [[ ! -f "$CHAPTERS_JSON" ]]; then
        echo "Error: $CHAPTERS_JSON not found. Cannot skip conversion without existing JSON."
        exit 1
    fi
    echo ""
fi

# ─── PHASE 3: Run TTS pipeline ───────────────────────────────
# Extract title from book.md prompts
BOOK_TITLE=$(sed -n 's/.*\*\*Tytuł:\*\*[[:space:]]*//p' "$PROMPTS_DIR/book.md" | head -1)
BOOK_TITLE=${BOOK_TITLE:-$BOOK_NAME}
BOOK_AUTHOR=$(sed -n 's/.*\*\*Autor:\*\*[[:space:]]*//p' "$PROMPTS_DIR/book.md" | head -1)
BOOK_AUTHOR=${BOOK_AUTHOR:-AI}

echo ">>> Phase 3: Running TTS pipeline..."
echo "  Title: $BOOK_TITLE"
echo "  Author: $BOOK_AUTHOR"
echo ""

$PYTHON "$SCRIPT_DIR/epub_to_audiobook.py" \
    --chapters-json "$CHAPTERS_JSON" \
    --title "$BOOK_TITLE" \
    --author "$BOOK_AUTHOR" \
    --output "$OUTPUT_DIR/audio" \
    "${TTS_ARGS[@]}"

echo ""
echo "======================================"
echo "  Done!"
echo "  Book: $OUTPUT_DIR/book.md"
echo "  Audio: $OUTPUT_DIR/audio/"
echo "======================================"
