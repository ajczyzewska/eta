# EtA - EPUB to Audiobook

Konwerter książek EPUB na audiobooki z użyciem XTTS v2 (Coqui TTS).

## Funkcje

✅ **Automatyczne filtrowanie treści:**
- Pomija okładki, wstępy, przedmowy, spisy treści
- Usuwa numery stron
- Wyciąga tylko właściwe rozdziały książki

✅ **Wysokiej jakości synteza mowy:**
- Voice cloning z własnego sampla głosu
- Obsługa języka polskiego
- Duże fragmenty tekstu (3000 znaków) dla płynności
- Crossfade między fragmentami

✅ **Checkpoint system:**
- Wznowienie po przerwaniu
- Bezpieczne przerywanie procesu

## Instalacja

### 1. Utwórz środowisko wirtualne

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# lub
.venv\Scripts\activate  # Windows
```

### 2. Zainstaluj zależności

```bash
pip install -r requirements.txt
```

### 3. Dodaj sample głosu (opcjonalnie)

Umieść plik WAV z próbką głosu (10-30 sekund) w katalogu projektu jako `speaker.wav` lub `sample-agent.wav`.

## Użycie

### Podstawowe

```bash
python epub_to_audiobook.py twoja_książka.epub
```

### Z własnym głosem

```bash
python epub_to_audiobook.py książka.epub --speaker moj_glos.wav
```

### Zaawansowane opcje

```bash
# Większe fragmenty (płynniejsza mowa)
python epub_to_audiobook.py książka.epub --chunk-size 5000

# Dłuższe crossfade (bardziej płynne przejścia)
python epub_to_audiobook.py książka.epub --crossfade 150

# Bez crossfade
python epub_to_audiobook.py książka.epub --crossfade 0

# Pokaż pominięte elementy
python epub_to_audiobook.py książka.epub --verbose

# Wznów od ostatniego checkpointu
python epub_to_audiobook.py książka.epub --resume
```

## Parametry

| Parametr | Opis | Domyślnie |
|----------|------|-----------|
| `epub_file` | Ścieżka do pliku EPUB | (wymagane) |
| `--speaker` | Plik WAV z próbką głosu | sample-agent.wav |
| `--output` | Katalog wyjściowy | nazwa_książki_audio |
| `--chunk-size` | Rozmiar fragmentu (znaki) | 3000 |
| `--crossfade` | Crossfade w ms (0=wyłącz) | 100 |
| `--resume` | Wznów od checkpointu | false |
| `--verbose` | Pokaż szczegóły | false |

## Wymagania systemowe

### CPU
- Działa na CPU (wolniej)
- Szacowany czas: ~20s na fragment 3000 znaków

### GPU (zalecane)
- NVIDIA GPU z CUDA
- 10x szybsze przetwarzanie
- Automatyczne wykrywanie i użycie GPU

## Struktura wyjściowa

```
nazwa_książki_audio/
├── 01_Rozdzial_1.mp3
├── 02_Rozdzial_2.mp3
├── 03_Rozdzial_3.mp3
└── ...
```

## Rozwiązywanie problemów

### Ostrzeżenie: "The text length exceeds the character limit of 224"

To normalne ostrzeżenie - model obsługuje dłuższe teksty przez wewnętrzny streaming. Sprawdź czy wygenerowane audio jest kompletne. Jeśli tak - możesz zignorować ostrzeżenie.

Jeśli audio jest ucięte:
```bash
python epub_to_audiobook.py książka.epub --chunk-size 200
```

### Brak GPU

Jeśli widzisz "Model załadowany na CPU" ale masz GPU NVIDIA:
1. Sprawdź czy masz zainstalowane CUDA
2. Zainstaluj PyTorch z obsługą CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Model TTS

Projekt używa **XTTS v2** (Coqui TTS):
- Model: `tts_models/multilingual/multi-dataset/xtts_v2`
- Wielojęzyczny (obsługuje polski)
- Voice cloning z własnej próbki głosu
- Wysokiej jakości synteza mowy

## Licencja

Ten projekt używa modelu XTTS v2 z Coqui TTS, który jest dostępny na licencji Mozilla Public License 2.0.

## Autorzy

Projekt stworzony z pomocą Claude (Anthropic).
