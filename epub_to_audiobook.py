#!/usr/bin/env python3
"""
EPUB to Audiobook Converter using XTTS v2

Converts EPUB files to audiobooks with chapter separation.
Automatically skips covers, introductions, prefaces, tables of contents, and page numbers.
Supports checkpoint system for resuming after interruption.
Uses larger text chunks (3000 characters) and crossfade for smooth speech.

Usage:
    python epub_to_audiobook.py book.epub
    python epub_to_audiobook.py book.epub --speaker voice.wav
    python epub_to_audiobook.py book.epub --optimize auto  # auto-optimization (recommended)
    python epub_to_audiobook.py book.epub --optimize speed  # maximum speed (GPU)
    python epub_to_audiobook.py book.epub --resume  # resume from checkpoint
    python epub_to_audiobook.py book.epub --chunk-size 5000  # larger chunks
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
from typing import List, Optional, Dict
import multiprocessing
import psutil

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from TTS.api import TTS

warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

# Configuration
# XTTS v2 officially has a ~224 character limit, but the model handles longer texts via internal streaming
# If you get a limit warning - check if the audio is complete
# If audio is OK - you can ignore the warning
CHUNK_SIZE = 3000  # Maximum characters per chunk (~30s audio for Polish)
MIN_CHUNK_SIZE = 200  # Minimum characters
OUTPUT_FORMAT = "mp3"  # Output format (mp3 or wav)
CROSSFADE_DURATION = 100  # Overlap time in ms (crossfade for smoothness)
# Crossfade gives a much more natural effect than pauses


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
                'chunk_size': 5000,  # Larger chunks
                'batch_size': 1,  # TTS doesn't support batch processing natively
                'use_gpu': True,
                'description': 'Maximum speed (GPU, large chunks)'
            })
        else:
            console.print("[yellow]Warning: 'speed' profile requires GPU. Using 'balanced'[/yellow]")
            user_optimize = 'balanced'

    # BALANCED profile - speed/quality balance
    if user_optimize == 'balanced':
        profile.update({
            'chunk_size': 3000,  # Standard size
            'batch_size': 1,
            'use_gpu': capabilities['gpu_available'],
            'description': 'Balanced (standard settings)'
        })

    # QUALITY profile - best quality (smaller chunks, smoother transitions)
    if user_optimize == 'quality':
        profile.update({
            'chunk_size': 2000,  # Smaller chunks = smoother speech
            'batch_size': 1,
            'use_gpu': capabilities['gpu_available'],
            'description': 'Best quality (smaller chunks, smoother speech)'
        })

    return profile


def should_skip_chapter(title: str, content: str, filename: str) -> bool:
    """
    Checks if a chapter should be skipped.
    Skips covers, introductions, prefaces, tables of contents, etc.
    """
    title_lower = title.lower()
    filename_lower = filename.lower()

    # Keywords to skip
    skip_keywords = [
        'cover', 'okładka', 'okladka',
        'copyright', 'rights', 'prawa autorskie',
        'dedication', 'dedykacja', 'dedykacje',
        'acknowledgment', 'podziękowania', 'podziekowania',
        'foreword', 'przedmowa',
        'preface', 'wstęp', 'wstep',
        'introduction', 'wprowadzenie',
        'table of contents', 'spis treści', 'spis tresci',
        'contents',
        'about the author', 'o autorze',
        'about author',
        'isbn',
        'publisher', 'wydawca', 'wydawnictwo',
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
    if len(content) < MIN_CHUNK_SIZE * 2:  # Minimum 400 characters
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


def extract_chapters_from_epub(epub_path: str, verbose: bool = False) -> tuple:
    """
    Extracts chapters from EPUB file.
    Skips covers, introductions, prefaces, and other elements before main content.

    Returns:
        Tuple: (chapter list, skipped elements list)
    """
    book = epub.read_epub(epub_path)
    all_items = []

    # Collect all documents
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')

            # Remove non-content elements
            for element in soup.find_all(['script', 'style', 'nav']):
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


def split_into_chunks(text: str, max_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits text into larger chunks for speech fluency.
    Tries to split on paragraphs, if not possible, on sentences.
    Similar to chunking in Whisper - we use overlapping boundaries for smoothness.
    """
    # First try to split on paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph fits in current chunk
        if len(current_chunk) + len(para) + 2 <= max_size:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If paragraph is too long, split it into sentences
            if len(para) > max_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= max_size:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())

                        # If single sentence is too long, split it
                        if len(sentence) > max_size:
                            words = sentence.split()
                            current_chunk = ""
                            for word in words:
                                if len(current_chunk) + len(word) + 1 <= max_size:
                                    if current_chunk:
                                        current_chunk += " " + word
                                    else:
                                        current_chunk = word
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                    current_chunk = word
                        else:
                            current_chunk = sentence
            else:
                current_chunk = para

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
    speed: float = 1.0
) -> Optional[str]:
    """
    Generates audio for one chapter.

    Returns:
        Path to generated audio file or None if error.
    """
    title = chapter['title']
    content = chapter['content']
    chunks = split_into_chunks(content, max_size=chunk_size)

    if not chunks:
        return None

    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    audio_segments = []
    start_chunk = 0

    # Check checkpoint for this chapter
    if checkpoint['current_chapter'] == chapter_idx:
        start_chunk = checkpoint['current_chunk']

    console.print(f"\n[bold cyan]📖 Chapter {chapter_idx + 1}: {title}[/bold cyan]")
    console.print(f"   Chunks: {len(chunks)}, characters: {len(content)}")

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

            chunk_file = os.path.join(temp_dir, f"chapter_{chapter_idx:03d}_chunk_{i:04d}.wav")

            try:
                tts.tts_to_file(
                    text=chunk,
                    file_path=chunk_file,
                    speaker_wav=speaker_wav,
                    language="pl"
                )
                audio_segments.append(chunk_file)

                # Update checkpoint
                checkpoint['current_chapter'] = chapter_idx
                checkpoint['current_chunk'] = i + 1
                save_checkpoint(checkpoint_path, checkpoint)

            except Exception as e:
                console.print(f"[red]Error at chunk {i}: {e}[/red]")
                continue

            progress.update(task, advance=1)

    # Combine all chunks into one file
    if audio_segments:
        output_file = os.path.join(
            output_dir,
            f"{chapter_idx + 1:02d}_{title}.{OUTPUT_FORMAT}"
        )

        console.print(f"   Combining chunks...")
        combined = None

        for idx, segment_file in enumerate(audio_segments):
            if os.path.exists(segment_file):
                segment = AudioSegment.from_wav(segment_file)

                # Apply speed adjustment if needed
                if speed != 1.0:
                    segment = adjust_audio_speed(segment, speed)

                if combined is None:
                    # First chunk
                    combined = segment
                else:
                    # Use crossfade for smooth transition (if enabled)
                    # Chunks overlap instead of having a pause - much more natural effect
                    if crossfade_duration > 0:
                        combined = combined.append(segment, crossfade=crossfade_duration)
                    else:
                        # No crossfade - direct concatenation
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
        help=f"Maximum chunk size in characters (default: {CHUNK_SIZE}, ~30s audio)"
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

    # Estimated time (larger chunks = longer generation time per chunk)
    chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else CHUNK_SIZE
    estimated_minutes = (total_chars / chunk_size) * 20 / 60  # ~20s per chunk for larger chunks
    console.print(f"   Estimated time: ~{estimated_minutes:.0f} minutes")

    # Load TTS model
    console.print(f"\n[bold yellow]🤖 Loading TTS model...[/bold yellow]")
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
            speed=args.speed
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
