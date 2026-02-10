#!/usr/bin/env python3
"""
EPUB to Audiobook Converter using XTTS v2

Converts EPUB files to audiobooks with chapter separation.
Automatically skips covers, editorial notes, publisher info, tables of contents, and page numbers.
Keeps introductions, forewords, and prefaces as book content.
Supports checkpoint system for resuming after interruption.
Uses text chunks (~300 characters) with automatic retry/splitting for XTTS v2 Polish and crossfade for smooth speech.

Usage:
    python epub_to_audiobook.py book.epub
    python epub_to_audiobook.py book.epub --speaker voice.wav
    python epub_to_audiobook.py book.epub --optimize auto  # auto-optimization (recommended)
    python epub_to_audiobook.py book.epub --optimize speed  # maximum speed (GPU)
    python epub_to_audiobook.py book.epub --resume  # resume from checkpoint
    python epub_to_audiobook.py book.epub --chunk-size 250  # larger chunks
    python epub_to_audiobook.py book.epub --crossfade 150  # longer crossfade
    python epub_to_audiobook.py book.epub --crossfade 0  # no crossfade
    python epub_to_audiobook.py book.epub --speed 0.75  # slower speech (75%)
    python epub_to_audiobook.py book.epub --speed 1.5  # faster speech (150%)
    python epub_to_audiobook.py book.epub --verbose  # show skipped elements
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any
import multiprocessing
import psutil

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
try:
    from TTS.api import TTS
except ImportError:
    TTS = None


warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

# Configuration
# XTTS v2 has a hard limit of 400 tokens per inference.
# Larger chunks may exceed this limit, but _tts_with_retry() automatically splits
# chunks that fail, so we can safely use larger sizes for more natural speech.
CHUNK_SIZE = 300  # Maximum characters per chunk - aggressive but safe with _tts_with_retry fallback
MIN_CHUNK_SIZE = 10  # Minimum characters (low threshold since chunks are small)
OUTPUT_FORMAT = "mp3"  # Output format (mp3 or wav)
CROSSFADE_DURATION = 100  # Overlap time in ms (crossfade for smoothness)
# Crossfade gives a much more natural effect than pauses

XTTS_CHUNK_LIMIT = 220  # Safe character limit for Polish XTTS v2


def detect_system_capabilities() -> Dict:
    """
    Detects system capabilities (GPU, CPU, RAM) and returns hardware information.

    Returns:
        Dict with system information: gpu_available, gpu_memory, cpu_cores, ram_gb, gpu_name
    """
    capabilities = {
        'gpu_available': False,
        'gpu_memory': 0,
        'gpu_name': None,
        'cpu_cores': multiprocessing.cpu_count(),
        'ram_gb': psutil.virtual_memory().total / (1024 ** 3)
    }

    try:
        import torch
        if torch.cuda.is_available():
            capabilities['gpu_available'] = True
            capabilities['gpu_name'] = torch.cuda.get_device_name(0)
            # GPU memory in GB
            capabilities['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass

    return capabilities


def get_optimization_profile(capabilities: Dict, user_optimize: str = 'auto') -> Dict:
    """
    Returns optimization profile based on system capabilities.

    Args:
        capabilities: Dict with system information
        user_optimize: 'auto', 'speed', 'quality', 'balanced', or None

    Returns:
        Dict with recommended parameters: chunk_size, batch_size, use_gpu
    """
    profile = {
        'chunk_size': CHUNK_SIZE,
        'batch_size': 1,
        'use_gpu': capabilities['gpu_available'],
        'description': 'Standard settings'
    }

    if user_optimize is None or user_optimize == 'off':
        return profile

    # Auto-detection or profile selection
    if user_optimize == 'auto':
        if capabilities['gpu_available']:
            # GPU available - fast mode
            if capabilities['gpu_memory'] >= 8:
                user_optimize = 'speed'
            else:
                user_optimize = 'balanced'
        else:
            # CPU - balanced or quality mode
            if capabilities['cpu_cores'] >= 8 and capabilities['ram_gb'] >= 16:
                user_optimize = 'balanced'
            else:
                user_optimize = 'quality'

    # SPEED profile - maximum speed (requires GPU)
    if user_optimize == 'speed':
        if capabilities['gpu_available']:
            profile.update({
                'chunk_size': 400,  # Large chunks for speed, retry handles overflow
                'batch_size': 1,  # TTS doesn't support batch processing natively
                'use_gpu': True,
                'description': 'Maximum speed (GPU)'
            })
        else:
            console.print("[yellow]Warning: 'speed' profile requires GPU. Using 'balanced'[/yellow]")
            user_optimize = 'balanced'

    # BALANCED profile - speed/quality balance
    if user_optimize == 'balanced':
        profile.update({
            'chunk_size': 300,  # Standard size with retry fallback
            'batch_size': 1,
            'use_gpu': capabilities['gpu_available'],
            'description': 'Balanced (standard settings)'
        })

    # QUALITY profile - best quality (smaller chunks, smoother transitions)
    if user_optimize == 'quality':
        profile.update({
            'chunk_size': 200,  # Smaller chunks = smoother speech
            'batch_size': 1,
            'use_gpu': capabilities['gpu_available'],
            'description': 'Best quality (smaller chunks, smoother speech)'
        })

    return profile


def should_skip_chapter(title: str, content: str, filename: str) -> bool:
    """
    Checks if a chapter should be skipped.
    Skips covers, editorial notes, publisher info, tables of contents, etc.
    Keeps introductions, forewords, and prefaces as part of book content.
    """
    title_lower = title.lower()
    filename_lower = filename.lower()

    # Keywords to skip
    skip_keywords = [
        'cover', 'okładka', 'okladka',
        'copyright', 'rights', 'prawa autorskie',
        'dedication', 'dedykacja', 'dedykacje',
        'acknowledgment', 'podziękowania', 'podziekowania',
        'about the author', 'o autorze',
        'about author',
        'isbn',
        'publisher', 'wydawca', 'wydawnictwo',
        'nota redakcyjna', 'nota wydawcy', 'nota wydawnicza',
        'od redakcji', 'od wydawcy',
        'editorial note', 'editor\'s note',
        'redakcja', 'korekta', 'redaction',
        'table_of_content', 'table_of_contents', 'toc',
        'title page', 'strona tytułowa',
        'titlepage',
        'half title',
        'frontmatter'
    ]

    # Check title and filename
    for keyword in skip_keywords:
        if keyword in title_lower or keyword in filename_lower:
            return True

    # Skip very short chapters (likely metadata)
    if len(content) < 400:  # Skip very short chapters (likely metadata)
        return True

    # Skip if mostly page numbers (more than 30% digits)
    digit_count = sum(c.isdigit() for c in content)
    if len(content) > 0 and digit_count / len(content) > 0.3:
        return True

    return False


def is_likely_chapter(title: str, content: str) -> bool:
    """
    Checks if this is likely an actual book chapter.
    """
    title_lower = title.lower()

    # Positive chapter indicators
    chapter_indicators = [
        'rozdział', 'rozdzial',
        'chapter',
        'część', 'czesc',
        'part',
    ]

    for indicator in chapter_indicators:
        if indicator in title_lower:
            return True

    # Check if title contains chapter number (e.g. "1.", "Chapter 1", "Rozdział I")
    if re.match(r'^(rozdział|chapter|część|czesc|part)?\s*[0-9ivxIVX]+\.?\s*', title_lower):
        return True

    # If content is long enough (at least 1000 characters), it's probably a chapter
    if len(content) > 1000:
        return True

    return False


def _get_items_in_reading_order(book) -> list:
    """
    Returns EPUB document items in correct reading order.
    Uses spine (reading order) when available and different from raw item order.
    Falls back to get_items() order otherwise.
    """
    # Get all document items indexed by filename
    items_by_name = {}
    raw_order = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            items_by_name[item.get_name()] = item
            raw_order.append(item.get_name())

    # Try to get spine order
    spine_order = []
    try:
        for item_id, _linear in book.spine:
            try:
                item = book.get_item_with_id(item_id)
                if item and item.get_name() in items_by_name:
                    spine_order.append(item.get_name())
            except Exception:
                continue
    except Exception:
        pass

    # Use spine order if it covers the documents, otherwise fall back to raw order
    if spine_order and set(spine_order) & set(raw_order):
        # Spine may not list every item; append any missing items at the end
        ordered_names = spine_order + [n for n in raw_order if n not in spine_order]
        if ordered_names != raw_order:
            console.print("[dim]Using EPUB spine reading order[/dim]")
    else:
        ordered_names = raw_order

    return [items_by_name[name] for name in ordered_names if name in items_by_name]


def extract_chapters_from_epub(epub_path: str, verbose: bool = False) -> tuple:
    """
    Extracts chapters from EPUB file.
    Skips covers, introductions, prefaces, and other elements before main content.

    Returns:
        Tuple: (chapter list, skipped elements list)
    """
    book = epub.read_epub(epub_path)
    all_items = []

    # Collect all documents in reading order
    for item in _get_items_in_reading_order(book):
            content = item.get_content().decode('utf-8')

            # Fix drop cap initials before parsing:
            # <span class="litera_inicjal">W</span>isława -> Wisława
            content = re.sub(
                r'<span[^>]*(?:inicjal|dropcap|drop.cap|lettrine)[^>]*>(\w)</span>',
                r'\1', content, flags=re.IGNORECASE
            )

            soup = BeautifulSoup(content, 'html.parser')

            # Remove non-content elements
            for element in soup.find_all(['script', 'style', 'nav']):
                element.decompose()

            # Remove footnote references (e.g. <sup><a href="...">[1]</a></sup>)
            for element in soup.find_all('sup'):
                element.decompose()

            # Remove images and their captions
            for element in soup.find_all(['figure', 'figcaption', 'img', 'svg', 'picture']):
                element.decompose()
            # Remove elements with caption-like CSS classes
            caption_patterns = re.compile(
                r'(caption|image[-_]?desc|photo[-_]?credit|ilustracja|opis[-_]?zdj|podpis)',
                re.IGNORECASE
            )
            for element in soup.find_all(class_=caption_patterns):
                element.decompose()
            for element in soup.find_all(id=caption_patterns):
                element.decompose()

            # Extract chapter title
            title = None
            for tag in ['h1', 'h2', 'h3', 'title']:
                title_tag = soup.find(tag)
                if title_tag:
                    title = title_tag.get_text().strip()
                    break

            if not title:
                title = item.get_name().replace('.xhtml', '').replace('.html', '')

            # Extract text
            text = soup.get_text(separator=' ')
            text = clean_text(text)

            # Remove page numbers (e.g. "12", "Strona 12", "Page 12")
            text = re.sub(r'\b(strona|page)\s+\d+\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone numbers in lines
            text = clean_text(text)  # Clean again after removing numbers

            # Regex preprocessing
            # Convert cardinal numbers to ordinals for natural TTS reading
            text = convert_numbers_to_ordinals(text)

            # Remove any remaining footnote markers like [1], [2], etc.
            text = re.sub(r'\s*\[\d+\]\s*', ' ', text)
            # Clean up orphaned punctuation after footnote removal
            text = re.sub(r'(["\u201d\u201c])\s+\.', r'\1.', text)
            text = re.sub(r'\.\s+\.', '.', text)
            text = clean_text(text)

            # Add "Cytat:" before long quotes for TTS emphasis
            text = re.sub(
                r'(?<=[.!?…])\s+(„[^"]{50,}?")',
                r' Cytat: \1',
                text
            )

            all_items.append({
                'title': title,
                'content': text,
                'filename': item.get_name()
            })

    # Filter chapters
    chapters = []
    skipped = []
    found_first_chapter = False

    for item in all_items:
        # Skip unwanted elements
        if should_skip_chapter(item['title'], item['content'], item['filename']):
            skipped.append({
                'title': item['title'],
                'reason': 'Skipped (cover, intro, metadata, or too short)'
            })
            continue

        # Check if this is likely a chapter
        if is_likely_chapter(item['title'], item['content']):
            found_first_chapter = True

        # If we haven't found the first chapter yet
        if not found_first_chapter:
            skipped.append({
                'title': item['title'],
                'reason': 'Before first chapter'
            })
            continue

        # Add all chapters after finding the first one
        if len(item['content']) > MIN_CHUNK_SIZE:
            chapters.append({
                'title': sanitize_filename(item['title']),
                'content': item['content']
            })

    return chapters, skipped


def _polish_ordinal(n: int, gender: str = 'm') -> Optional[str]:
    """
    Converts an integer to Polish ordinal form.

    Args:
        n: Number to convert (1-999)
        gender: 'm' (masculine), 'f' (feminine), 'n' (neuter)

    Returns:
        Polish ordinal string or None if number out of range.
    """
    if n < 1 or n > 999:
        return None

    # Base ordinals 1-19
    _ordinals = {
        'm': ['', 'pierwszy', 'drugi', 'trzeci', 'czwarty', 'piąty', 'szósty',
               'siódmy', 'ósmy', 'dziewiąty', 'dziesiąty', 'jedenasty', 'dwunasty',
               'trzynasty', 'czternasty', 'piętnasty', 'szesnasty', 'siedemnasty',
               'osiemnasty', 'dziewiętnasty'],
        'f': ['', 'pierwsza', 'druga', 'trzecia', 'czwarta', 'piąta', 'szósta',
               'siódma', 'ósma', 'dziewiąta', 'dziesiąta', 'jedenasta', 'dwunasta',
               'trzynasta', 'czternasta', 'piętnasta', 'szesnasta', 'siedemnasta',
               'osiemnasta', 'dziewiętnasta'],
        'n': ['', 'pierwsze', 'drugie', 'trzecie', 'czwarte', 'piąte', 'szóste',
               'siódme', 'ósme', 'dziewiąte', 'dziesiąte', 'jedenaste', 'dwunaste',
               'trzynaste', 'czternaste', 'piętnaste', 'szesnaste', 'siedemnaste',
               'osiemnaste', 'dziewiętnaste'],
    }

    # Tens 20-90
    _tens = {
        'm': ['', '', 'dwudziesty', 'trzydziesty', 'czterdziesty', 'pięćdziesiąty',
               'sześćdziesiąty', 'siedemdziesiąty', 'osiemdziesiąty', 'dziewięćdziesiąty'],
        'f': ['', '', 'dwudziesta', 'trzydziesta', 'czterdziesta', 'pięćdziesiąta',
               'sześćdziesiąta', 'siedemdziesiąta', 'osiemdziesiąta', 'dziewięćdziesiąta'],
        'n': ['', '', 'dwudzieste', 'trzydzieste', 'czterdzieste', 'pięćdziesiąte',
               'sześćdziesiąte', 'siedemdziesiąte', 'osiemdziesiąte', 'dziewięćdziesiąte'],
    }

    # Hundreds 100-900
    _hundreds = {
        'm': ['', 'setny', 'dwusetny', 'trzechsetny', 'czterechsetny', 'pięćsetny',
               'sześćsetny', 'siedemsetny', 'osiemsetny', 'dziewięćsetny'],
        'f': ['', 'setna', 'dwusetna', 'trzechsetna', 'czterechsetna', 'pięćsetna',
               'sześćsetna', 'siedemsetna', 'osiemsetna', 'dziewięćsetna'],
        'n': ['', 'setne', 'dwusetne', 'trzechsetne', 'czterechsetne', 'pięćsetne',
               'sześćsetne', 'siedemsetne', 'osiemsetne', 'dziewięćsetne'],
    }

    g = gender if gender in _ordinals else 'm'
    h = n // 100
    rest = n % 100
    t = rest // 10
    u = rest % 10

    if n < 20:
        return _ordinals[g][n]

    parts = []
    if h > 0:
        if rest == 0:
            return _hundreds[g][h]
        # For hundreds + remainder, hundreds use cardinal prefix form
        _hundreds_cardinal = ['', 'sto', 'dwieście', 'trzysta', 'czterysta',
                              'pięćset', 'sześćset', 'siedemset', 'osiemset', 'dziewięćset']
        parts.append(_hundreds_cardinal[h])

    if rest < 20:
        parts.append(_ordinals[g][rest])
    else:
        if u == 0:
            parts.append(_tens[g][t])
        else:
            # Compound ordinals like "dwudziesty pierwszy" - both parts are ordinal
            parts.append(_tens[g][t])
            parts.append(_ordinals[g][u])

    return ' '.join(p for p in parts if p)


def convert_numbers_to_ordinals(text: str) -> str:
    """
    Converts cardinal numbers after specific Polish/English keywords to ordinal forms.
    E.g. "Rozdział 1" -> "Rozdział pierwszy", "Strona 5" -> "Strona piąta".

    Handles gender agreement: masculine for rozdział/chapter, feminine for strona/część/page/part.
    """
    # Masculine keywords
    masculine_keywords = r'(?:rozdział|rozdzial|chapter|tom|akt|punkt|paragraf|ustęp|artykuł|wiersz|psalm|pieśń|sonet|epizod|numer|nr)'
    # Feminine keywords
    feminine_keywords = r'(?:strona|część|czesc|page|part|księga|pieśń|scena|lekcja|sesja)'

    def _replace_with_ordinal(match, gender):
        keyword = match.group(1)
        number_str = match.group(2)
        try:
            n = int(number_str)
        except ValueError:
            return match.group(0)
        ordinal = _polish_ordinal(n, gender)
        if ordinal:
            return f"{keyword} {ordinal}"
        return match.group(0)

    # Replace masculine patterns
    text = re.sub(
        rf'\b({masculine_keywords})\s+(\d+)\b',
        lambda m: _replace_with_ordinal(m, 'm'),
        text,
        flags=re.IGNORECASE
    )

    # Replace feminine patterns
    text = re.sub(
        rf'\b({feminine_keywords})\s+(\d+)\b',
        lambda m: _replace_with_ordinal(m, 'f'),
        text,
        flags=re.IGNORECASE
    )

    return text


def clean_text(text: str) -> str:
    """Cleans text from unnecessary characters and formatting."""
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that may interfere with TTS
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Replace quotes with standard ones
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    return text.strip()


def sanitize_filename(name: str) -> str:
    """Converts title to safe filename."""
    # Remove characters not allowed in filenames
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Replace spaces with underscores
    name = re.sub(r'\s+', '_', name)
    # Limit length
    return name[:50]



def _split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using heuristics that avoid breaking on:
    - Dots inside quotes/titles (e.g. „Walka. Tygodnik...")
    - Common abbreviations (nr., im., gen., itd., itp., etc.)
    - Initials (A. Mickiewicz)
    - Numbers with dots (1. 2. etc.)

    A sentence boundary is a dot/!/? followed by whitespace and an uppercase letter
    or a dash (dialogue), but NOT inside quotes.
    """
    # First, protect text inside Polish quotes „..." from splitting
    protected = {}
    counter = [0]

    def protect_quoted(m):
        key = f"\x00QUOTE{counter[0]}\x00"
        protected[key] = m.group(0)
        counter[0] += 1
        return key

    # Protect „..." and "..." quoted strings
    safe_text = re.sub(r'[„"][^""]*["""]', protect_quoted, text)

    # Split on sentence-ending punctuation followed by space + uppercase letter or dash
    # This avoids splitting on abbreviations (followed by lowercase) or mid-sentence dots
    parts = re.split(r'(?<=[.!?…])\s+(?=[A-ZŻŹĆĄŚĘŁÓŃ–—(„\[])', safe_text)

    # Restore protected quotes
    sentences = []
    for part in parts:
        for key, val in protected.items():
            part = part.replace(key, val)
        part = part.strip()
        if part:
            sentences.append(part)

    return sentences


def split_into_chunks(text: str, max_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits text into sentence-based chunks for natural TTS output.
    Each chunk is one or more complete sentences. Short consecutive sentences
    are combined up to max_size for efficiency. Long sentences are kept whole
    and _tts_with_retry() handles splitting if XTTS can't process them.
    """
    sentences = _split_into_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If current chunk is empty, start with this sentence (regardless of length)
        if not current_chunk:
            current_chunk = sentence
            continue

        # Combine short sentences together up to max_size
        if len(current_chunk) + len(sentence) + 1 <= max_size:
            current_chunk = current_chunk + " " + sentence
        else:
            # Current chunk is full - save it and start new one
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]


def load_checkpoint(checkpoint_path: str) -> dict:
    """Loads checkpoint from file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'completed_chapters': [], 'current_chapter': 0, 'current_chunk': 0}


def save_checkpoint(checkpoint_path: str, data: dict):
    """Saves checkpoint to file."""
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f)


def stretch_pauses(audio_segment: AudioSegment, factor: float = 1.5,
                    silence_thresh: int = -35, min_silence_len: int = 150) -> AudioSegment:
    """
    Extends natural pauses between words/phrases without changing pitch or speech speed.

    Args:
        audio_segment: Audio segment to process
        factor: How much to stretch pauses (1.5 = 50% longer pauses)
        silence_thresh: Volume threshold in dBFS to consider as silence
        min_silence_len: Minimum silence duration in ms to detect
    """
    from pydub.silence import detect_nonsilent

    nonsilent_ranges = detect_nonsilent(
        audio_segment,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    if not nonsilent_ranges:
        return audio_segment

    result = AudioSegment.empty()

    for i, (start, end) in enumerate(nonsilent_ranges):
        # Add silence before this speech segment (stretched)
        if i == 0:
            if start > 0:
                silence_dur = int(start * factor)
                result += AudioSegment.silent(duration=silence_dur,
                                              frame_rate=audio_segment.frame_rate)
        else:
            prev_end = nonsilent_ranges[i - 1][1]
            gap = start - prev_end
            if gap > 0:
                stretched_gap = int(gap * factor)
                result += AudioSegment.silent(duration=stretched_gap,
                                              frame_rate=audio_segment.frame_rate)

        # Add the speech part unchanged
        result += audio_segment[start:end]

    # Trailing silence
    last_end = nonsilent_ranges[-1][1]
    if last_end < len(audio_segment):
        trailing = int((len(audio_segment) - last_end) * factor)
        result += AudioSegment.silent(duration=trailing,
                                      frame_rate=audio_segment.frame_rate)

    return result


def pad_chunk_with_silence(audio: AudioSegment, pad_ms: int = 150) -> AudioSegment:
    """
    Adds silence padding at the start and end of a chunk so that crossfade
    blends silence rather than cutting into speech.

    Args:
        audio: Audio segment to pad
        pad_ms: Duration of silence padding in ms (should be >= crossfade duration)
    """
    silence = AudioSegment.silent(duration=pad_ms, frame_rate=audio.frame_rate)
    return silence + audio + silence


CHAPTER_TITLE_PAUSE_MS = 2500  # silence before and after chapter title


def separate_chapter_heading(text: str) -> tuple:
    """
    Detects chapter heading (e.g. "Rozdział 6 TYTUŁ WIELKIMI LITERAMI") at the
    beginning of text and separates it from the body.

    Returns:
        (heading, rest_of_text) or (None, text) if no heading found.
    """
    text = text.strip()
    prefix_match = re.match(
        r'^((?:rozdział|rozdzial|chapter|część|czesc|part)\s*\d+)\s*',
        text, re.IGNORECASE
    )
    if not prefix_match:
        return None, text

    prefix = prefix_match.group(1)
    rest = text[prefix_match.end():]

    # Collect uppercase title words, stop at single letter followed by lowercase
    # (that's the start of the body text, e.g. "W isława" = "Wisława")
    words = rest.split()
    title_words = []
    for j, w in enumerate(words):
        clean = re.sub(r'[,.\"\"\"\(\)\-]', '', w)
        if not clean:
            title_words.append(w)
            continue
        # Single uppercase letter + next word starts lowercase = sentence start
        if len(clean) == 1 and clean.isupper() and j + 1 < len(words):
            next_clean = re.sub(r'[,.\"\"\"]', '', words[j + 1])
            if next_clean and next_clean[0].islower():
                break
        if clean[0].isupper() and (len(clean) == 1 or clean.isupper()):
            title_words.append(w)
        else:
            break

    if title_words:
        heading = prefix + ' ' + ' '.join(title_words)
    else:
        heading = prefix

    # Convert numbers in heading to ordinals for natural TTS reading
    heading = convert_numbers_to_ordinals(heading)

    body = ' '.join(words[len(title_words):])
    if body:
        return heading.strip(), body.strip()
    return None, text


def adjust_audio_speed(audio_segment: AudioSegment, speed: float) -> AudioSegment:
    """
    Changes audio speed without changing pitch.

    Args:
        audio_segment: Audio segment to process
        speed: Speed factor (0.5 = 50% slower, 2.0 = 2x faster)

    Returns:
        AudioSegment with adjusted speed
    """
    if speed == 1.0:
        return audio_segment

    # Change sample rate to change speed
    # Then restore original sample rate to preserve pitch
    sound_with_altered_frame_rate = audio_segment._spawn(
        audio_segment.raw_data,
        overrides={"frame_rate": int(audio_segment.frame_rate * speed)}
    )
    return sound_with_altered_frame_rate.set_frame_rate(audio_segment.frame_rate)


_tts_sub_counter = 0


def _tts_with_retry(tts, text: str, temp_dir: str, chapter_idx: int, chunk_idx: int, speaker_wav: str) -> List[str]:
    """
    Tries to generate TTS for text. If XTTS fails with 400 token error,
    splits text in half and retries recursively.
    Returns list of generated wav file paths.
    """
    global _tts_sub_counter
    _tts_sub_counter += 1
    chunk_file = os.path.join(temp_dir, f"chapter_{chapter_idx:03d}_chunk_{chunk_idx:04d}_s{_tts_sub_counter:03d}.wav")

    # Add trailing punctuation to prevent XTTS from cutting off the last syllable
    tts_text = text.rstrip()
    if tts_text and tts_text[-1] not in '.!?…':
        tts_text += '.'

    try:
        tts.tts_to_file(
            text=tts_text,
            file_path=chunk_file,
            speaker_wav=speaker_wav,
            language="pl",
            split_sentences=True
        )
        return [chunk_file]
    except Exception as e:
        if "400 tokens" in str(e) and len(text) > 20:
            mid = len(text) // 2
            space_pos = text.rfind(' ', 0, mid)
            if space_pos == -1:
                space_pos = text.find(' ', mid)
            if space_pos == -1:
                console.print(f"[red]Error at chunk {chunk_idx}: {e}[/red]")
                return []
            half1 = text[:space_pos].strip()
            half2 = text[space_pos:].strip()
            console.print(f"[yellow]   Chunk {chunk_idx} too long ({len(text)} chars), splitting...[/yellow]")
            results = []
            if half1:
                results.extend(_tts_with_retry(tts, half1, temp_dir, chapter_idx, chunk_idx, speaker_wav))
            if half2:
                results.extend(_tts_with_retry(tts, half2, temp_dir, chapter_idx, chunk_idx, speaker_wav))
            return results
        else:
            console.print(f"[red]Error at chunk {chunk_idx}: {e}[/red]")
            return []


def generate_chapter_audio(
    tts: TTS,
    chapter: dict,
    chapter_idx: int,
    output_dir: str,
    speaker_wav: str,
    checkpoint_path: str,
    checkpoint: dict,
    chunk_size: int = CHUNK_SIZE,
    crossfade_duration: int = CROSSFADE_DURATION,
    speed: float = 1.0,
    pause_stretch: float = 1.0,
) -> Optional[str]:
    """
    Generates audio for one chapter.

    Returns:
        Path to generated audio file or None if error.
    """
    title = chapter['title']
    content = chapter['content']

    heading, content_body = separate_chapter_heading(content)
    chunks = split_into_chunks(content_body, max_size=chunk_size)

    if not chunks and not heading:
        return None

    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    global _tts_sub_counter
    _tts_sub_counter = 0

    audio_segments = []
    start_chunk = 0

    # Check checkpoint for this chapter
    if checkpoint['current_chapter'] == chapter_idx:
        start_chunk = checkpoint['current_chunk']

    console.print(f"\n[bold cyan]📖 Chapter {chapter_idx + 1}: {title}[/bold cyan]")
    if heading:
        console.print(f"   Heading: \"{heading}\"")
    console.print(f"   Chunks: {len(chunks)}, characters: {sum(len(c) for c in chunks)}")

    # Generate chapter heading with emphasis (only for regex path)
    if heading and start_chunk == 0:
        heading_file = os.path.join(temp_dir, f"chapter_{chapter_idx:03d}_heading.wav")
        try:
            tts.tts_to_file(
                text=heading + '.',
                file_path=heading_file,
                speaker_wav=speaker_wav,
                language="pl",
                split_sentences=True
            )
            heading_audio = AudioSegment.from_wav(heading_file)
            pause = AudioSegment.silent(duration=CHAPTER_TITLE_PAUSE_MS,
                                        frame_rate=heading_audio.frame_rate)
            emphasized = pause + heading_audio + pause
            emphasized.export(heading_file, format="wav")
            audio_segments.append(heading_file)
        except Exception as e:
            console.print(f"[red]Error generating heading: {e}[/red]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Generating...", total=len(chunks))
        progress.update(task, completed=start_chunk)

        for i, chunk in enumerate(chunks):
            if i < start_chunk:
                continue

            generated = _tts_with_retry(tts, chunk, temp_dir, chapter_idx, i, speaker_wav)
            audio_segments.extend(generated)

            # Update checkpoint
            checkpoint['current_chapter'] = chapter_idx
            checkpoint['current_chunk'] = i + 1
            save_checkpoint(checkpoint_path, checkpoint)

            progress.update(task, advance=1)

    # Combine all chunks into one file
    if audio_segments:
        output_file = os.path.join(
            output_dir,
            f"{chapter_idx + 1:02d}_{title}.{OUTPUT_FORMAT}"
        )

        console.print(f"   Combining chunks...")
        combined = None

        for seg_idx, segment_file in enumerate(audio_segments):
            if os.path.exists(segment_file):
                segment = AudioSegment.from_wav(segment_file)

                # Apply speed adjustment if needed
                if speed != 1.0:
                    segment = adjust_audio_speed(segment, speed)

                # Stretch pauses between words/phrases
                if pause_stretch > 1.0:
                    segment = stretch_pauses(segment, factor=pause_stretch)

                if combined is None:
                    combined = segment
                else:
                    if crossfade_duration > 0:
                        segment = pad_chunk_with_silence(segment, pad_ms=crossfade_duration + 50)
                        combined = combined.append(segment, crossfade=crossfade_duration)
                    else:
                        combined += segment

        # Export
        if OUTPUT_FORMAT == "mp3":
            combined.export(output_file, format="mp3", bitrate="192k")
        else:
            combined.export(output_file, format="wav")

        # Clean up temporary files
        for segment_file in audio_segments:
            if os.path.exists(segment_file):
                os.remove(segment_file)

        console.print(f"   [green]✅ Saved: {output_file}[/green]")
        return output_file

    return None


def extract_metadata(epub_path: str) -> dict:
    """
    Extracts metadata from EPUB file (title, author).
    """
    try:
        book = epub.read_epub(epub_path)
        metadata = {
            'title': 'Unknown title',
            'author': 'Unknown author'
        }

        # Extract title
        if book.get_metadata('DC', 'title'):
            metadata['title'] = book.get_metadata('DC', 'title')[0][0]

        # Extract author
        if book.get_metadata('DC', 'creator'):
            metadata['author'] = book.get_metadata('DC', 'creator')[0][0]

        return metadata
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to extract metadata: {e}[/yellow]")
        return {'title': 'Unknown title', 'author': 'Unknown author'}


def main():
    parser = argparse.ArgumentParser(
        description="Converts EPUB to audiobook using XTTS v2"
    )
    parser.add_argument("epub_file", help="Path to EPUB file")
    parser.add_argument(
        "--speaker",
        default=None,
        help="WAV file with voice sample (default: sample-agent.wav)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: book_name_audio)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Maximum chunk size in characters (default: {CHUNK_SIZE}). Keep under 250 for Polish XTTS v2"
    )
    parser.add_argument(
        "--crossfade",
        type=int,
        default=CROSSFADE_DURATION,
        help=f"Crossfade time between chunks in ms (default: {CROSSFADE_DURATION}). Set 0 to disable"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information about skipped chapters"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed (0.5-2.0, default: 1.0). Values < 1.0 slow down speech, > 1.0 speed up"
    )
    parser.add_argument(
        "--pause-stretch",
        type=float,
        default=1.0,
        help="Stretch pauses between words/phrases (default: 1.0). E.g. 1.5 = 50%% longer pauses, 2.0 = double. Does not change pitch or speech speed"
    )
    parser.add_argument(
        "--optimize",
        choices=['auto', 'speed', 'balanced', 'quality', 'off'],
        default='off',
        help="Optimization profile: auto (auto-detect), speed (GPU, max speed), balanced (balance), quality (best quality), off (use manual settings)"
    )
    args = parser.parse_args()

    # Speed parameter validation
    if args.speed < 0.5 or args.speed > 2.0:
        console.print(f"[red]Error: Speed must be in range 0.5-2.0 (provided: {args.speed})[/red]")
        sys.exit(1)

    # Detect system capabilities and apply optimization
    console.print(f"\n[bold yellow]🔍 Detecting system capabilities...[/bold yellow]")
    capabilities = detect_system_capabilities()

    console.print(f"   CPU: {capabilities['cpu_cores']} cores")
    console.print(f"   RAM: {capabilities['ram_gb']:.1f} GB")
    if capabilities['gpu_available']:
        console.print(f"   GPU: {capabilities['gpu_name']} ({capabilities['gpu_memory']:.1f} GB)")
    else:
        console.print(f"   GPU: Not detected (CPU only)")

    # Apply optimization profile
    optimization_profile = get_optimization_profile(capabilities, args.optimize)

    if args.optimize != 'off':
        console.print(f"\n[bold cyan]⚡ Optimization: {optimization_profile['description']}[/bold cyan]")
        # Override chunk_size if using optimization and not manually set by user
        # Check if chunk_size is default value
        if args.chunk_size == CHUNK_SIZE:
            args.chunk_size = optimization_profile['chunk_size']
            console.print(f"   Chunk size: {args.chunk_size} characters")

    # Check EPUB file
    if not os.path.exists(args.epub_file):
        console.print(f"[red]Error: File does not exist: {args.epub_file}[/red]")
        sys.exit(1)

    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        book_name = Path(args.epub_file).stem
        output_dir = f"{book_name}_audio"

    os.makedirs(output_dir, exist_ok=True)

    # Set voice file
    speaker_wav = args.speaker
    if not speaker_wav:
        # Look for default file
        default_speakers = ["sample-agent.wav", "speaker.wav", "voice.wav"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for name in default_speakers:
            path = os.path.join(script_dir, name)
            if os.path.exists(path):
                speaker_wav = path
                break

    if not speaker_wav or not os.path.exists(speaker_wav):
        console.print("[red]Error: Voice file not found. Use --speaker[/red]")
        sys.exit(1)

    # Checkpoint
    checkpoint_path = os.path.join(output_dir, ".checkpoint.json")

    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        console.print(f"[yellow]Resuming from chapter {checkpoint['current_chapter'] + 1}[/yellow]")
    else:
        checkpoint = {'completed_chapters': [], 'current_chapter': 0, 'current_chunk': 0}

    # Extract metadata
    console.print(f"\n[bold yellow]📚 Loading EPUB: {args.epub_file}[/bold yellow]")
    metadata = extract_metadata(args.epub_file)
    console.print(f"   [cyan]Title:[/cyan] {metadata['title']}")
    console.print(f"   [cyan]Author:[/cyan] {metadata['author']}")

    # Extract chapters
    console.print(f"\n[bold yellow]🔍 Analyzing chapters...[/bold yellow]")
    chapters, skipped = extract_chapters_from_epub(args.epub_file, verbose=args.verbose)

    if not chapters:
        console.print("[red]Error: No chapters found in EPUB file[/red]")
        console.print("[yellow]Check if the file contains proper book chapters.[/yellow]")
        sys.exit(1)

    console.print(f"   [green]✅ Found chapters to process: {len(chapters)}[/green]")

    if skipped:
        console.print(f"   [dim]Skipped elements: {len(skipped)}[/dim]")
        if args.verbose:
            console.print(f"\n[bold yellow]Skipped elements:[/bold yellow]")
            for item in skipped:
                console.print(f"   [dim]- {item['title']}: {item['reason']}[/dim]")

    total_chars = sum(len(ch['content']) for ch in chapters)
    console.print(f"   Total characters: {total_chars:,}")

    # Estimated time (~5s per small chunk for XTTS generation)
    chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else CHUNK_SIZE
    estimated_minutes = (total_chars / chunk_size) * 5 / 60
    console.print(f"   Estimated time: ~{estimated_minutes:.0f} minutes")

    # Load TTS model
    console.print(f"\n[bold yellow]🤖 Loading TTS model...[/bold yellow]")
    if TTS is None:
        console.print("[red]Error: TTS package not installed. Run: pip install TTS==0.22.0[/red]")
        console.print("[yellow]Note: Requires Python 3.9-3.11 (not compatible with Python 3.12+)[/yellow]")
        sys.exit(1)
    try:
        import torch
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        # Use GPU if available and recommended by optimization profile
        device = "cpu"
        if optimization_profile['use_gpu'] and torch.cuda.is_available():
            device = "cuda"
            tts = tts.to("cuda")
            console.print("[green]✅ Model loaded on GPU[/green]")
        else:
            tts = tts.to("cpu")
            console.print("[green]✅ Model loaded on CPU[/green]")
            if not capabilities['gpu_available']:
                console.print("[yellow]   Tip: Use GPU for ~10x faster processing[/yellow]")
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        sys.exit(1)

    console.print(f"   Using voice: {speaker_wav}")
    console.print(f"   Output directory: {output_dir}")
    console.print(f"   Chunk size: {args.chunk_size} characters")
    if args.crossfade > 0:
        console.print(f"   Crossfade between chunks: {args.crossfade}ms")
    else:
        console.print(f"   Crossfade disabled - direct chunk concatenation")
    if args.speed != 1.0:
        if args.speed < 1.0:
            console.print(f"   Speech speed: {args.speed}x (slower)")
        else:
            console.print(f"   Speech speed: {args.speed}x (faster)")
    if args.pause_stretch > 1.0:
        console.print(f"   Pause stretch: {args.pause_stretch}x (longer pauses between words)")
    if args.optimize != 'off':
        console.print(f"   [cyan]Optimization profile: {args.optimize}[/cyan]")
    # Generate audio for each chapter
    start_chapter = checkpoint['current_chapter']

    for i, chapter in enumerate(chapters):
        if i < start_chapter:
            continue

        if chapter['title'] in checkpoint['completed_chapters']:
            console.print(f"[dim]Skipping chapter {i + 1} (already completed)[/dim]")
            continue

        result = generate_chapter_audio(
            tts=tts,
            chapter=chapter,
            chapter_idx=i,
            output_dir=output_dir,
            speaker_wav=speaker_wav,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            chunk_size=args.chunk_size,
            crossfade_duration=args.crossfade,
            speed=args.speed,
            pause_stretch=args.pause_stretch,
        )

        if result:
            checkpoint['completed_chapters'].append(chapter['title'])
            checkpoint['current_chapter'] = i + 1
            checkpoint['current_chunk'] = 0
            save_checkpoint(checkpoint_path, checkpoint)

    # Clean up temp
    temp_dir = os.path.join(output_dir, 'temp')
    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
        except:
            pass

    # Remove checkpoint after completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    console.print(f"\n[bold green]🎉 Completed! Audio files in: {output_dir}[/bold green]")


if __name__ == "__main__":
    main()
