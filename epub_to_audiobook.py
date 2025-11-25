#!/usr/bin/env python3
"""
EPUB to Audiobook Converter using XTTS v2

Konwertuje plik EPUB na audiobooka z podziałem na rozdziały.
Automatycznie pomija okładki, wstępy, przedmowy, spisy treści i numery stron.
Obsługuje checkpoint do wznowienia po przerwaniu.
Wykorzystuje większe fragmenty tekstu (3000 znaków) i crossfade dla płynności mowy.

Użycie:
    python epub_to_audiobook.py book.epub
    python epub_to_audiobook.py book.epub --speaker voice.wav
    python epub_to_audiobook.py book.epub --resume  # wznowienie
    python epub_to_audiobook.py book.epub --chunk-size 5000  # większe fragmenty
    python epub_to_audiobook.py book.epub --crossfade 150  # dłuższe crossfade
    python epub_to_audiobook.py book.epub --crossfade 0  # bez crossfade
    python epub_to_audiobook.py book.epub --verbose  # pokaż pominięte elementy
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from TTS.api import TTS

warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

# Konfiguracja
# XTTS v2 oficjalnie ma limit ~224 znaki, ale model obsługuje dłuższe teksty przez wewnętrzny streaming
# Jeśli otrzymujesz ostrzeżenie o limicie - sprawdź czy audio jest kompletne
# Jeśli audio jest OK - możesz zignorować ostrzeżenie
CHUNK_SIZE = 3000  # Maksymalna liczba znaków na fragment (~30s audio dla polskiego)
MIN_CHUNK_SIZE = 200  # Minimalna liczba znaków
OUTPUT_FORMAT = "mp3"  # Format wyjściowy (mp3 lub wav)
CROSSFADE_DURATION = 100  # Czas nakładania się fragmentów w ms (crossfade dla płynności)
# Crossfade daje znacznie bardziej naturalny efekt niż pauza


def should_skip_chapter(title: str, content: str, filename: str) -> bool:
    """
    Sprawdza czy rozdział powinien być pominięty.
    Pomija okładki, wstępy, przedmowy, spis treści, itp.
    """
    title_lower = title.lower()
    filename_lower = filename.lower()

    # Słowa kluczowe do pominięcia
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

    # Sprawdź tytuł i nazwę pliku
    for keyword in skip_keywords:
        if keyword in title_lower or keyword in filename_lower:
            return True

    # Pomiń bardzo krótkie rozdziały (prawdopodobnie metadata)
    if len(content) < MIN_CHUNK_SIZE * 2:  # Minimum 400 znaków
        return True

    # Pomiń jeśli zawiera głównie numery stron (więcej niż 30% to liczby)
    digit_count = sum(c.isdigit() for c in content)
    if len(content) > 0 and digit_count / len(content) > 0.3:
        return True

    return False


def is_likely_chapter(title: str, content: str) -> bool:
    """
    Sprawdza czy to prawdopodobnie właściwy rozdział książki.
    """
    title_lower = title.lower()

    # Pozytywne wskaźniki rozdziału
    chapter_indicators = [
        'rozdział', 'rozdzial',
        'chapter',
        'część', 'czesc',
        'part',
    ]

    for indicator in chapter_indicators:
        if indicator in title_lower:
            return True

    # Sprawdź czy tytuł zawiera numer rozdziału (np. "1.", "Chapter 1", "Rozdział I")
    if re.match(r'^(rozdział|chapter|część|czesc|part)?\s*[0-9ivxIVX]+\.?\s*', title_lower):
        return True

    # Jeśli treść jest wystarczająco długa (co najmniej 1000 znaków), prawdopodobnie to rozdział
    if len(content) > 1000:
        return True

    return False


def extract_chapters_from_epub(epub_path: str, verbose: bool = False) -> tuple:
    """
    Wyciąga rozdziały z pliku EPUB.
    Pomija okładki, wstępy, przedmowy i inne elementy przed właściwą treścią.

    Returns:
        Tuple: (lista rozdziałów, lista pominiętych elementów)
    """
    book = epub.read_epub(epub_path)
    all_items = []

    # Zbierz wszystkie dokumenty
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')

            # Usuń elementy, które nie są treścią
            for element in soup.find_all(['script', 'style', 'nav']):
                element.decompose()

            # Wyciągnij tytuł rozdziału
            title = None
            for tag in ['h1', 'h2', 'h3', 'title']:
                title_tag = soup.find(tag)
                if title_tag:
                    title = title_tag.get_text().strip()
                    break

            if not title:
                title = item.get_name().replace('.xhtml', '').replace('.html', '')

            # Wyciągnij tekst
            text = soup.get_text(separator=' ')
            text = clean_text(text)

            # Usuń numery stron (np. "12", "Strona 12", "Page 12")
            text = re.sub(r'\b(strona|page)\s+\d+\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Samotne numery w liniach
            text = clean_text(text)  # Ponownie wyczyść po usunięciu numerów

            all_items.append({
                'title': title,
                'content': text,
                'filename': item.get_name()
            })

    # Filtruj rozdziały
    chapters = []
    skipped = []
    found_first_chapter = False

    for item in all_items:
        # Pomiń niepożądane elementy
        if should_skip_chapter(item['title'], item['content'], item['filename']):
            skipped.append({
                'title': item['title'],
                'reason': 'Pominięto (okładka, wstęp, metadata lub zbyt krótkie)'
            })
            continue

        # Sprawdź czy to prawdopodobnie rozdział
        if is_likely_chapter(item['title'], item['content']):
            found_first_chapter = True

        # Jeśli jeszcze nie znaleźliśmy pierwszego rozdziału
        if not found_first_chapter:
            skipped.append({
                'title': item['title'],
                'reason': 'Przed pierwszym rozdziałem'
            })
            continue

        # Dodawaj wszystkie rozdziały po znalezieniu pierwszego
        if len(item['content']) > MIN_CHUNK_SIZE:
            chapters.append({
                'title': sanitize_filename(item['title']),
                'content': item['content']
            })

    return chapters, skipped


def clean_text(text: str) -> str:
    """Czyści tekst z niepotrzebnych znaków i formatowania."""
    # Usuń wielokrotne spacje i nowe linie
    text = re.sub(r'\s+', ' ', text)
    # Usuń znaki specjalne które mogą przeszkadzać w TTS
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Zamień cudzysłowy na standardowe
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    return text.strip()


def sanitize_filename(name: str) -> str:
    """Zamienia tytuł na bezpieczną nazwę pliku."""
    # Usuń znaki niedozwolone w nazwach plików
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Zamień spacje na podkreślenia
    name = re.sub(r'\s+', '_', name)
    # Ogranicz długość
    return name[:50]


def split_into_chunks(text: str, max_size: int = CHUNK_SIZE) -> List[str]:
    """
    Dzieli tekst na większe fragmenty dla płynności mowy.
    Stara się dzielić na akapitach, a jeśli to niemożliwe, na zdaniach.
    Podobnie jak chunking w Whisper - używamy nakładających się granic dla płynności.
    """
    # Najpierw spróbuj podzielić na akapity
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Jeśli akapit mieści się w bieżącym fragmencie
        if len(current_chunk) + len(para) + 2 <= max_size:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            # Zapisz bieżący fragment jeśli istnieje
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Jeśli akapit jest za długi, podziel go na zdania
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

                        # Jeśli pojedyncze zdanie jest za długie, podziel je
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
    """Wczytuje checkpoint z pliku."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'completed_chapters': [], 'current_chapter': 0, 'current_chunk': 0}


def save_checkpoint(checkpoint_path: str, data: dict):
    """Zapisuje checkpoint do pliku."""
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f)


def generate_chapter_audio(
    tts: TTS,
    chapter: dict,
    chapter_idx: int,
    output_dir: str,
    speaker_wav: str,
    checkpoint_path: str,
    checkpoint: dict,
    chunk_size: int = CHUNK_SIZE,
    crossfade_duration: int = CROSSFADE_DURATION
) -> Optional[str]:
    """
    Generuje audio dla jednego rozdziału.

    Returns:
        Ścieżka do wygenerowanego pliku audio lub None w przypadku błędu.
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

    # Sprawdź checkpoint dla tego rozdziału
    if checkpoint['current_chapter'] == chapter_idx:
        start_chunk = checkpoint['current_chunk']

    console.print(f"\n[bold cyan]📖 Rozdział {chapter_idx + 1}: {title}[/bold cyan]")
    console.print(f"   Fragmentów: {len(chunks)}, znaków: {len(content)}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Generowanie...", total=len(chunks))
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

                # Aktualizuj checkpoint
                checkpoint['current_chapter'] = chapter_idx
                checkpoint['current_chunk'] = i + 1
                save_checkpoint(checkpoint_path, checkpoint)

            except Exception as e:
                console.print(f"[red]Błąd przy fragmencie {i}: {e}[/red]")
                continue

            progress.update(task, advance=1)

    # Połącz wszystkie fragmenty w jeden plik
    if audio_segments:
        output_file = os.path.join(
            output_dir,
            f"{chapter_idx + 1:02d}_{title}.{OUTPUT_FORMAT}"
        )

        console.print(f"   Łączenie fragmentów...")
        combined = None

        for idx, segment_file in enumerate(audio_segments):
            if os.path.exists(segment_file):
                segment = AudioSegment.from_wav(segment_file)

                if combined is None:
                    # Pierwszy fragment
                    combined = segment
                else:
                    # Użyj crossfade dla płynnego przejścia (jeśli włączone)
                    # Fragmenty nakładają się zamiast mieć pauzę - dużo bardziej naturalny efekt
                    if crossfade_duration > 0:
                        combined = combined.append(segment, crossfade=crossfade_duration)
                    else:
                        # Bez crossfade - bezpośrednie połączenie
                        combined += segment

        # Eksportuj
        if OUTPUT_FORMAT == "mp3":
            combined.export(output_file, format="mp3", bitrate="192k")
        else:
            combined.export(output_file, format="wav")

        # Wyczyść pliki tymczasowe
        for segment_file in audio_segments:
            if os.path.exists(segment_file):
                os.remove(segment_file)

        console.print(f"   [green]✅ Zapisano: {output_file}[/green]")
        return output_file

    return None


def extract_metadata(epub_path: str) -> dict:
    """
    Wyciąga metadane z pliku EPUB (tytuł, autor).
    """
    try:
        book = epub.read_epub(epub_path)
        metadata = {
            'title': 'Nieznany tytuł',
            'author': 'Nieznany autor'
        }

        # Wyciągnij tytuł
        if book.get_metadata('DC', 'title'):
            metadata['title'] = book.get_metadata('DC', 'title')[0][0]

        # Wyciągnij autora
        if book.get_metadata('DC', 'creator'):
            metadata['author'] = book.get_metadata('DC', 'creator')[0][0]

        return metadata
    except Exception as e:
        console.print(f"[yellow]Ostrzeżenie: Nie udało się wyciągnąć metadanych: {e}[/yellow]")
        return {'title': 'Nieznany tytuł', 'author': 'Nieznany autor'}


def main():
    parser = argparse.ArgumentParser(
        description="Konwertuje EPUB na audiobooka używając XTTS v2"
    )
    parser.add_argument("epub_file", help="Ścieżka do pliku EPUB")
    parser.add_argument(
        "--speaker",
        default=None,
        help="Plik WAV z próbką głosu (domyślnie: sample-agent.wav)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Katalog wyjściowy (domyślnie: nazwa_książki_audio)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Wznów od ostatniego checkpointu"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Maksymalna wielkość fragmentu w znakach (domyślnie: {CHUNK_SIZE}, ~30s audio)"
    )
    parser.add_argument(
        "--crossfade",
        type=int,
        default=CROSSFADE_DURATION,
        help=f"Czas crossfade między fragmentami w ms (domyślnie: {CROSSFADE_DURATION}). Ustaw 0 aby wyłączyć"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Wyświetl szczegółowe informacje o pominiętych rozdziałach"
    )

    args = parser.parse_args()

    # Sprawdź plik EPUB
    if not os.path.exists(args.epub_file):
        console.print(f"[red]Błąd: Plik nie istnieje: {args.epub_file}[/red]")
        sys.exit(1)

    # Ustaw katalog wyjściowy
    if args.output:
        output_dir = args.output
    else:
        book_name = Path(args.epub_file).stem
        output_dir = f"{book_name}_audio"

    os.makedirs(output_dir, exist_ok=True)

    # Ustaw plik głosu
    speaker_wav = args.speaker
    if not speaker_wav:
        # Szukaj domyślnego pliku
        default_speakers = ["sample-agent.wav", "speaker.wav", "voice.wav"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for name in default_speakers:
            path = os.path.join(script_dir, name)
            if os.path.exists(path):
                speaker_wav = path
                break

    if not speaker_wav or not os.path.exists(speaker_wav):
        console.print("[red]Błąd: Nie znaleziono pliku głosu. Użyj --speaker[/red]")
        sys.exit(1)

    # Checkpoint
    checkpoint_path = os.path.join(output_dir, ".checkpoint.json")

    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        console.print(f"[yellow]Wznawiam od rozdziału {checkpoint['current_chapter'] + 1}[/yellow]")
    else:
        checkpoint = {'completed_chapters': [], 'current_chapter': 0, 'current_chunk': 0}

    # Wyciągnij metadane
    console.print(f"\n[bold yellow]📚 Wczytuję EPUB: {args.epub_file}[/bold yellow]")
    metadata = extract_metadata(args.epub_file)
    console.print(f"   [cyan]Tytuł:[/cyan] {metadata['title']}")
    console.print(f"   [cyan]Autor:[/cyan] {metadata['author']}")

    # Wyciągnij rozdziały
    console.print(f"\n[bold yellow]🔍 Analizuję rozdziały...[/bold yellow]")
    chapters, skipped = extract_chapters_from_epub(args.epub_file, verbose=args.verbose)

    if not chapters:
        console.print("[red]Błąd: Nie znaleziono rozdziałów w pliku EPUB[/red]")
        console.print("[yellow]Sprawdź czy plik zawiera właściwe rozdziały książki.[/yellow]")
        sys.exit(1)

    console.print(f"   [green]✅ Znaleziono rozdziałów do przetworzenia: {len(chapters)}[/green]")

    if skipped:
        console.print(f"   [dim]Pominięto elementów: {len(skipped)}[/dim]")
        if args.verbose:
            console.print(f"\n[bold yellow]Pominięte elementy:[/bold yellow]")
            for item in skipped:
                console.print(f"   [dim]- {item['title']}: {item['reason']}[/dim]")

    total_chars = sum(len(ch['content']) for ch in chapters)
    console.print(f"   Łączna liczba znaków: {total_chars:,}")

    # Szacowany czas (większe fragmenty = dłuższy czas generowania na fragment)
    chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else CHUNK_SIZE
    estimated_minutes = (total_chars / chunk_size) * 20 / 60  # ~20s per chunk dla większych fragmentów
    console.print(f"   Szacowany czas: ~{estimated_minutes:.0f} minut")

    # Załaduj model TTS
    console.print(f"\n[bold yellow]🤖 Ładowanie modelu TTS...[/bold yellow]")
    try:
        import torch
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        # Użyj GPU jeśli dostępne (10x szybciej)
        if torch.cuda.is_available():
            tts = tts.to("cuda")
            console.print("[green]✅ Model załadowany na GPU[/green]")
        else:
            tts = tts.to("cpu")
            console.print("[green]✅ Model załadowany na CPU[/green]")
            console.print("[yellow]   Tip: Użyj GPU dla 10x szybszego przetwarzania[/yellow]")
    except Exception as e:
        console.print(f"[red]Błąd ładowania modelu: {e}[/red]")
        sys.exit(1)

    console.print(f"   Używam głosu: {speaker_wav}")
    console.print(f"   Katalog wyjściowy: {output_dir}")
    if args.crossfade > 0:
        console.print(f"   Crossfade między fragmentami: {args.crossfade}ms")
    else:
        console.print(f"   Crossfade wyłączony - bezpośrednie łączenie fragmentów")

    # Generuj audio dla każdego rozdziału
    start_chapter = checkpoint['current_chapter']

    for i, chapter in enumerate(chapters):
        if i < start_chapter:
            continue

        if chapter['title'] in checkpoint['completed_chapters']:
            console.print(f"[dim]Pomijam rozdział {i + 1} (już ukończony)[/dim]")
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
            crossfade_duration=args.crossfade
        )

        if result:
            checkpoint['completed_chapters'].append(chapter['title'])
            checkpoint['current_chapter'] = i + 1
            checkpoint['current_chunk'] = 0
            save_checkpoint(checkpoint_path, checkpoint)

    # Wyczyść temp
    temp_dir = os.path.join(output_dir, 'temp')
    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
        except:
            pass

    # Usuń checkpoint po zakończeniu
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    console.print(f"\n[bold green]🎉 Zakończono! Pliki audio w: {output_dir}[/bold green]")


if __name__ == "__main__":
    main()
