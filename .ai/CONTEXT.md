# EtA - AI Agent Context

> **Quick Start:** Python CLI tool converting EPUB → MP3 audiobooks using XTTS v2 voice cloning. 819 lines, v1.0.0, by Aga Czyzewska.

## Core Info

**Main:** `epub_to_audiobook.py` | **Entry:** `main()` line 586 | **Lang:** Polish (line 499) | **Output:** MP3 @192kbps

**Stack:** TTS (XTTS v2), ebooklib, BeautifulSoup, pydub, rich, torch | **GPU:** CUDA (10x faster) | **Model:** 2.5GB

**CLI:** `python epub_to_audiobook.py book.epub [--speaker voice.wav] [--optimize auto] [--speed 0.5-2.0] [--chunk-size 200-5000] [--crossfade 0-500] [--resume] [--verbose]`

## Architecture (Key Functions)

| Function | Lines | Purpose |
|----------|-------|---------|
| `main()` | 586-818 | Entry point, orchestration |
| `extract_chapters_from_epub()` | 231-312 | EPUB parsing, HTML→text |
| `generate_chapter_audio()` | 440-558 | TTS generation, crossfade, MP3 export |
| `split_into_chunks()` | 336-401 | Text→chunks (paragraphs→sentences→words) |
| `should_skip_chapter()` | 156-199 | Filter covers/TOC/prefaces/page numbers |
| `is_likely_chapter()` | 202-228 | Validate actual chapters |
| `detect_system_capabilities()` | 58-84 | GPU/CPU/RAM detection |
| `get_optimization_profile()` | 86-154 | auto/speed/balanced/quality profiles |
| `adjust_audio_speed()` | 417-438 | Speed change without pitch shift |

## Processing Flow

```
EPUB → Parse HTML → Filter (skip covers/TOC) → Detect Chapters → Chunk Text (300 chars) →
TTS (XTTS v2) → Speed Adjust → Crossfade (100ms) → MP3 Export → Checkpoints
```

## Content Filtering

**Skip Keywords (case-insensitive):** cover, okładka, copyright, dedication, acknowledgment, foreword, preface, introduction, wstęp, table of contents, spis treści, about author, isbn, publisher, title page

**Chapter Indicators:** rozdział, chapter, część, part + regex: `^(rozdział|chapter|część|czesc|part)?\s*[0-9ivxIVX]+\.?\s*`

**Rules:** Skip <400 chars, skip >30% digits (page numbers)

## Key Constants & Config

```python
CHUNK_SIZE = 300            # Line 59 - Default chars/chunk
MIN_CHUNK_SIZE = 10         # Line 60
OUTPUT_FORMAT = "mp3"       # Line 61
CROSSFADE_DURATION = 100    # Line 62 - ms
language = "pl"             # Hardcoded in _tts_with_retry() and intro generation
```

## Optimization Profiles

- **auto:** Detects hardware → speed (GPU 8GB+), balanced (GPU <8GB or 8+ CPU cores), or quality
- **speed:** 400 char chunks, GPU required
- **balanced:** 300 chars (default)
- **quality:** 200 chars, smoother speech

## Common Modifications

### Change Language
```python
# Line 499
language="en"  # or "es", "de", "fr", etc.
```

### Change Output Format
```python
# Line 53
OUTPUT_FORMAT = "wav"
# Lines 545-548: Update export logic
```

### Add Skip Keywords
```python
# Lines 165-183
skip_keywords = [
    'your_new_keyword',
    # ...existing keywords
]
```

### Adjust Chunk Algorithm
```python
# Lines 336-401 in split_into_chunks()
# Current: paragraphs → sentences → words
```

## Performance

| Mode | Speed/300chars | 300pg book | Memory |
|------|----------------|------------|--------|
| CPU | ~5s | 2-3hr | 2-4GB RAM |
| GPU | ~0.5s | 15-20min | 2-6GB VRAM |

## Common Issues

| Issue | Fix |
|-------|-----|
| "Text exceeds 224 chars" warning | Ignore if audio complete, else `--chunk-size 200` |
| "Model on CPU" (have GPU) | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| "No chapters found" | Use `--verbose` to debug filtering |
| Interrupted processing | Use `--resume` |

## Output Structure

```
book_audio/
├── temp/                           # Auto-removed
│   └── chapter_000_chunk_0000.wav
├── .checkpoint.json                # Auto-removed on completion
├── 01_Chapter_1.mp3
├── 02_Chapter_2.mp3
└── ...
```

## Testing Checklist

- [ ] Basic: `python epub_to_audiobook.py test.epub`
- [ ] GPU: Check `torch.cuda.is_available()`
- [ ] Resume: Ctrl+C then `--resume`
- [ ] Speed: Test `--speed 0.75` and `1.5`
- [ ] Profiles: Test `--optimize speed/quality`
- [ ] Verbose: Check `--verbose` output
- [ ] Audio: Verify transitions/crossfade quality

## Quick Commands

```bash
# Recommended usage
python epub_to_audiobook.py book.epub --optimize auto

# Debug
python epub_to_audiobook.py book.epub --verbose 2>&1 | tee log.txt

# Check GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# View function
sed -n '586,818p' epub_to_audiobook.py  # main()

# List functions
grep -n "^def " epub_to_audiobook.py
```

## Dependencies

Core: TTS≥0.22.0, ebooklib≥0.18, beautifulsoup4≥4.12.0, lxml≥4.9.0, pydub≥0.25.1, rich≥13.0.0, psutil≥5.9.0, torch≥2.0.0, torchaudio≥2.0.0

System: Python 3.9-3.11, CUDA (optional), FFmpeg (required for MP3 export and post-processing)

## Extension Ideas

- Multi-language auto-detect | Parallel processing | Web UI | ID3 tags | MOBI/PDF support | REST API | Voice emotion control | Background music

## Limitations

- Single-threaded | Polish hardcoded (line 499) | MP3-only | No ID3 metadata | Large EPUBs slow on CPU

---

**Full Docs:** README.md (users), QUICKSTART.md (Polish), EXAMPLES.md (Polish) | **Author:** Aga Czyzewska | **License:** MPL 2.0
