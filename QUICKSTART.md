# Quick Start Guide

## 🚀 5-minutowy start

### 1. Sklonuj repo i przejdź do folderu

```bash
git clone <your-repo-url>
cd EtA
```

### 2. Wymagania wstępne

- **Python 3.9-3.11** (nie kompatybilne z Python 3.12+)
- **FFmpeg** - wymagany do eksportu MP3:
  ```bash
  # macOS
  brew install ffmpeg
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```

### 3. Setup środowiska

```bash
# Utwórz wirtualne środowisko
python3 -m venv .venv

# Aktywuj środowisko
source .venv/bin/activate  # Linux/Mac
# lub
.venv\Scripts\activate     # Windows

# Zaktualizuj pip (ważne - stary pip może mieć problemy z pobieraniem dużych pakietów)
pip install --upgrade pip

# Zainstaluj zależności
pip install -r requirements.txt
```

### 4. Przygotuj głos (opcjonalnie)

Jeśli chcesz użyć własnego głosu:
- Nagraj 10-30 sekund czystego nagrania głosu
- Zapisz jako `speaker.wav` w katalogu projektu

Lub użyj domyślnego głosu `sample-agent.wav`

### 5. Konwertuj książkę!

```bash
python epub_to_audiobook.py twoja_książka.epub
```

To wszystko! 🎉

## 📦 Co otrzymasz?

```
twoja_książka_audio/
├── 01_Rozdzial_1.mp3
├── 02_Rozdzial_2.mp3
├── 03_Rozdzial_3.mp3
└── ...
```

## 🎛️ Podstawowe opcje

```bash
# Z własnym głosem
python epub_to_audiobook.py książka.epub --speaker moj_glos.wav

# Większe fragmenty = płynniejsza mowa
python epub_to_audiobook.py książka.epub --chunk-size 400

# Zobacz co zostało pominięte
python epub_to_audiobook.py książka.epub --verbose
```

## ❓ Problemy?

### "Model załadowany na CPU" - wolno działa
- ✅ To normalne, przetwarzanie na CPU trwa dłużej
- 💡 Użyj GPU z CUDA dla 10x szybszego przetwarzania

### Ostrzeżenie o limicie 224 znaków
- ✅ Możesz zignorować - model radzi sobie z dłuższymi tekstami
- 🔍 Sprawdź czy wygenerowane audio jest kompletne
- 📝 Jeśli ucięte, użyj: `--chunk-size 200`

### Przerwałem proces
- ✅ Użyj `--resume` aby wznowić od ostatniego checkpointu
```bash
python epub_to_audiobook.py książka.epub --resume
```

## 📖 Generowanie audiobooka od zera (z promptów AI)

Możesz wygenerować całą książkę — tekst + audio — używając `generatebook.sh` i katalogu z promptami.

### Struktura promptów

```
book_prompts/
├── book.md       # Specyfikacja książki (tytuł, bohaterowie, streszczenie)
├── outline.md    # Zarys rozdziałów (## Rozdział 1: Tytuł + opis)
├── rules.md      # Reguły dla AI (styl, format, ograniczenia)
└── style.md      # (opcjonalnie) Próbka stylu / dodatkowe wytyczne
```

### Generowanie

```bash
sh generatebook.sh --name "moja-ksiazka" \
  --speaker speaker.wav --optimize auto --postprocess --chunk-size 230
```

Skrypt robi trzy rzeczy automatycznie:
1. Generuje tekst rozdziałów przez Claude (z review i streszczeniami)
2. Konwertuje `book.md` → `chapters.json`
3. Odpala pipeline TTS → pliki MP3

### Przydatne flagi

```bash
# Użyj innego katalogu z promptami (np. dla tomu 2)
sh generatebook.sh --name "rod-debickich-t2" --prompts-dir ./book_prompts_t2/

# Pomiń generowanie tekstu (masz już book.md)
sh generatebook.sh --name "moja-ksiazka" --skip-generate \
  --speaker speaker.wav --optimize auto --postprocess

# Pomiń konwersję (masz już chapters.json)
sh generatebook.sh --name "moja-ksiazka" --skip-convert \
  --speaker speaker.wav --optimize auto --postprocess

# Ręcznie podaj numer tomu i serię (jeśli auto-detekcja nie działa)
sh generatebook.sh --name "moja-ksiazka" --tom 3 --series "Moja Seria" \
  --speaker speaker.wav --optimize auto --postprocess
```

> **Auto-detekcja tomu i serii:** Jeśli tytuł w `book.md` ma format `Seria — Tom V: Tytuł`,
> skrypt automatycznie wyciągnie numer tomu (Roman i arabski), nazwę serii i podtytuł.
> Flagi `--tom` i `--series` nadpisują auto-detekcję.

### Multi-tom

Jeśli masz wiele tomów, stwórz osobny katalog promptów per tom:

```bash
mkdir -p book_prompts_t2
cp book_prompts/rules.md book_prompts/style.md book_prompts_t2/
cp book_prompts/book-t2.md book_prompts_t2/book.md
cp book_prompts/outline-t2.md book_prompts_t2/outline.md

sh generatebook.sh --name "rod-debickich-t2" --prompts-dir ./book_prompts_t2/ \
  --speaker speaker.wav --optimize auto --postprocess --chunk-size 230
```

### Wynik

```
generated_books/moja-ksiazka_2026-03-16/
├── chapter_1.md          # Tekst rozdziałów
├── chapter_2.md
├── ...
├── summary_1.txt         # Streszczenia (dla ciągłości)
├── review_1.txt          # Wyniki review AI
├── book.md               # Cała książka
├── chapters.json         # JSON dla TTS
└── audio/
    ├── 01_Rozdzial_1.mp3
    ├── 02_Rozdzial_2.mp3
    └── ...
```

## 📚 Więcej info

- Pełna dokumentacja: [README.md](README.md)
- Zgłoś problem: [Issues](../../issues)
